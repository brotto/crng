"""
Build `models/btc_v1.yaml` — first formal CRNG model under SPECS.md discipline.

A model (≠ preset) has:
  - train_window (out of the frozen snapshot)
  - validation_window (out of the frozen snapshot, non-overlapping with train)
  - target_fingerprint : statistics of the real training returns
  - achieved_fingerprint : CRNG (calibrated on train only) at same n, same seed
  - validation_fingerprint : statistics of the real validation returns AND
                             statistics of the CRNG output at validation n
                             (the CRNG hyperparameters are FROZEN at training time;
                             the validation is genuinely out-of-sample)
  - hyperparameters : what ``ContingencyRNG`` was instantiated with
  - version / created / author / source

This script is the only way to create or refresh a formal model. It reads
from the frozen `benchmarks/snapshot_2026-04/prices.csv` — never from
yfinance or any live source. Per SPECS.md P2, the snapshot is immutable.

Run:
    cd crng-package && PYTHONPATH=. python3 models/build_btc_v1.py

Writes: `models/btc_v1.yaml`
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crng import ContingencyRNG, from_data  # noqa: E402


SNAPSHOT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks", "snapshot_2026-04", "prices.csv",
)
SNAPSHOT_SHA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks", "snapshot_2026-04", "prices.sha256",
)
OUTPUT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_v1.yaml")

ASSET = "BTC"
TRAIN_END_ISO = "2024-04-10"   # exclusive boundary used as `<`
VAL_START_ISO = "2024-04-11"

SEED = 42


def verify_snapshot() -> str:
    with open(SNAPSHOT_CSV, "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    with open(SNAPSHOT_SHA_FILE) as fh:
        expected = fh.read().strip()
    if actual != expected:
        raise RuntimeError(
            f"Snapshot integrity failure.\n  expected: {expected}\n  actual:   {actual}"
        )
    return actual


def log_returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    r = np.diff(np.log(p))
    return r[np.isfinite(r)]


def fingerprint(series: np.ndarray) -> dict:
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = int(len(s))
    if n < 10:
        return {
            "n": n, "mean": 0.0, "std": 0.0, "kurtosis": 3.0, "skewness": 0.0,
            "vol_acf_lag1": 0.0, "vol_acf_lag5": 0.0,
            "tail_3sigma_pct": 0.0, "tail_4sigma_pct": 0.0,
        }
    m = float(np.mean(s))
    sd = float(np.std(s, ddof=1))
    if sd == 0:
        return {
            "n": n, "mean": m, "std": 0.0, "kurtosis": 3.0, "skewness": 0.0,
            "vol_acf_lag1": 0.0, "vol_acf_lag5": 0.0,
            "tail_3sigma_pct": 0.0, "tail_4sigma_pct": 0.0,
        }
    z = (s - m) / sd
    k = float(np.mean(z ** 4))
    sk = float(np.mean(z ** 3))
    az = np.abs(z)
    t3 = float(np.mean(az > 3.0) * 100.0)
    t4 = float(np.mean(az > 4.0) * 100.0)
    abs_v = np.abs(s - m)
    am = float(np.mean(abs_v))
    av = float(np.var(abs_v))
    if av > 0 and n > 6:
        acf1 = float(np.sum((abs_v[:-1] - am) * (abs_v[1:] - am)) / (n * av))
        acf5 = float(np.sum((abs_v[:-5] - am) * (abs_v[5:] - am)) / (n * av))
    else:
        acf1 = 0.0
        acf5 = 0.0
    return {
        "n": n, "mean": m, "std": sd, "kurtosis": k, "skewness": sk,
        "vol_acf_lag1": acf1, "vol_acf_lag5": acf5,
        "tail_3sigma_pct": t3, "tail_4sigma_pct": t4,
    }


def dump_yaml(data: dict) -> str:
    """Minimal hand-rolled YAML dumper (no external pyyaml dependency).

    Only handles the nested dict/scalar structure we emit here.
    """
    def _fmt(v, indent):
        pad = "  " * indent
        if isinstance(v, dict):
            lines = []
            for kk, vv in v.items():
                if isinstance(vv, (dict,)):
                    lines.append(f"{pad}{kk}:")
                    lines.append(_fmt(vv, indent + 1))
                elif isinstance(vv, (list, tuple)):
                    lines.append(f"{pad}{kk}:")
                    for item in vv:
                        lines.append(f"{pad}  - {item}")
                else:
                    lines.append(f"{pad}{kk}: {_scalar(vv)}")
            return "\n".join(lines)
        else:
            return f"{pad}{_scalar(v)}"

    def _scalar(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return ".nan" if np.isnan(v) else ".inf"
            return f"{v:.10g}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        s = str(v)
        if any(c in s for c in ":#\n\"'"):
            return '"' + s.replace('"', '\\"') + '"'
        return s

    return _fmt(data, 0) + "\n"


def main():
    sha = verify_snapshot()
    print(f"Snapshot verified: {sha}")

    prices = pd.read_csv(SNAPSHOT_CSV, index_col=0, parse_dates=True)
    if ASSET not in prices.columns:
        raise KeyError(f"Asset {ASSET} not present in snapshot columns {list(prices.columns)}")

    col = prices[ASSET].dropna()
    train_prices = col[col.index <= pd.Timestamp(TRAIN_END_ISO)]
    val_prices = col[col.index >= pd.Timestamp(VAL_START_ISO)]

    if len(train_prices) < 200:
        raise ValueError(f"Training window too short ({len(train_prices)} daily closes).")
    if len(val_prices) < 100:
        raise ValueError(f"Validation window too short ({len(val_prices)} daily closes).")

    print(f"Train: {train_prices.index.min().date()} → {train_prices.index.max().date()}"
          f"  ({len(train_prices)} closes)")
    print(f"Val  : {val_prices.index.min().date()} → {val_prices.index.max().date()}"
          f"  ({len(val_prices)} closes)")

    train_returns = log_returns(train_prices.values)
    val_returns = log_returns(val_prices.values)
    target_fp = fingerprint(train_returns)
    real_val_fp = fingerprint(val_returns)

    # CRNG calibrated on TRAIN ONLY.
    rng = from_data(train_returns, seed=SEED)
    hp = {
        "target_kurtosis": float(rng.target_kurtosis),
        "vol_clustering":  float(rng.vol_clustering),
        "n_oscillators":   int(rng.n_osc),
        "cascade_threshold": float(rng.cascade_threshold),
        "cascade_memory":  int(rng.cascade_memory),
        "amplification":   float(rng.amplification),
    }

    # Achieved (in-sample match): same n as train_returns, fresh instance at seed.
    rng_is = ContingencyRNG(
        seed=SEED,
        target_kurtosis=hp["target_kurtosis"],
        vol_clustering=hp["vol_clustering"],
        n_oscillators=hp["n_oscillators"],
        cascade_threshold=hp["cascade_threshold"],
        cascade_memory=hp["cascade_memory"],
    )
    achieved_fp = fingerprint(rng_is.generate(len(train_returns)))

    # Validation: CRNG using the FROZEN training hyperparameters, fresh seed=SEED,
    # generating same n as the real validation series. This output has never seen
    # validation data — it is genuinely out-of-sample.
    rng_oos = ContingencyRNG(
        seed=SEED,
        target_kurtosis=hp["target_kurtosis"],
        vol_clustering=hp["vol_clustering"],
        n_oscillators=hp["n_oscillators"],
        cascade_threshold=hp["cascade_threshold"],
        cascade_memory=hp["cascade_memory"],
    )
    crng_val_fp = fingerprint(rng_oos.generate(len(val_returns)))

    print(f"\nTrain real K     = {target_fp['kurtosis']:.3f}")
    print(f"Train CRNG K     = {achieved_fp['kurtosis']:.3f}")
    print(f"Val   real K     = {real_val_fp['kurtosis']:.3f}")
    print(f"Val   CRNG K     = {crng_val_fp['kurtosis']:.3f}  (frozen hyperparams)")

    model = {
        "name": "btc",
        "version": 1,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "author": "ale-brotto",
        "source": {
            "snapshot_csv": "benchmarks/snapshot_2026-04/prices.csv",
            "snapshot_sha256": sha,
            "asset_column": ASSET,
        },
        "train_window": {
            "start": str(train_prices.index.min().date()),
            "end":   str(train_prices.index.max().date()),
            "n_prices":  int(len(train_prices)),
            "n_returns": int(len(train_returns)),
        },
        "validation_window": {
            "start": str(val_prices.index.min().date()),
            "end":   str(val_prices.index.max().date()),
            "n_prices":  int(len(val_prices)),
            "n_returns": int(len(val_returns)),
            "disjoint_from_train": True,
        },
        "target_fingerprint": target_fp,
        "achieved_fingerprint": {
            **achieved_fp,
            "seed": SEED,
            "n_samples": int(len(train_returns)),
            "note": "CRNG output at train n — in-sample fit check.",
        },
        "validation_fingerprint": {
            "real": real_val_fp,
            "crng": {
                **crng_val_fp,
                "seed": SEED,
                "n_samples": int(len(val_returns)),
                "note": (
                    "CRNG hyperparameters FROZEN from train_window. This output "
                    "has not been tuned against validation data; it is OOS."
                ),
            },
        },
        "hyperparameters": hp,
        "calibration_algorithm": "crng.from_data (iterative, 5 rounds, ACF*3 clamp)",
        "notes": (
            "First formal model under SPECS.md. Uses the frozen snapshot 2026-04 "
            "split temporally: 3 years train + ~2 years OOS validation. BTC chosen "
            "because the legacy preset `btc()` overclaimed kurtosis=219 based on a "
            "different (buggy) calibration regime; this model replaces that claim "
            "with a real, reproducible, out-of-sample check."
        ),
    }

    with open(OUTPUT_YAML, "w") as fh:
        fh.write(dump_yaml(model))
    print(f"\nWrote: {OUTPUT_YAML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
