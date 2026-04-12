"""
Frozen descriptive benchmark — CRNG vs real markets vs iid Gaussian.

Reads the IMMUTABLE snapshot at `benchmarks/snapshot_2026-04/prices.csv`
(frozen 2026-04-10, SHA256 recorded in `prices.sha256`) and answers one
question per SPECS.md principle P1 (descriptive, not predictive):

    "Does CRNG, auto-calibrated on the full history of each real asset,
     reproduce the statistical fingerprint (kurtosis, vol clustering,
     tail weights) of that asset — more faithfully than an iid Gaussian
     PRNG does?"

A priori rules (SPECS.md P3 — fixed BEFORE running, no cherry-picking):
  - For each asset, CRNG is ALWAYS built via ``from_data(prices)`` on the
    FULL window. We do NOT try multiple presets and keep the best one.
  - The baseline is ``iid_gaussian()``. Not ``gaussian()``.
  - We report target (= fingerprint of REAL returns) and achieved
    (= fingerprint of CRNG output at same n) side by side. Also iid.
  - Random seeds for the simulators are fixed and listed.
  - n (samples drawn from CRNG and iid) = len(real returns) for each asset.

This script writes `benchmarks/snapshot_2026-04/frozen_benchmark_report.json`,
which is what any public claim about "CRNG reproduces real-market signatures"
must cite. The rolling (live) benchmark is NOT valid evidence.

Run:
    cd crng-package && PYTHONPATH=. python3 benchmarks/frozen_benchmark.py

See SPECS.md P1, P2, P3, P4, P5, P6.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crng import ContingencyRNG, from_data, iid_gaussian  # noqa: E402


SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot_2026-04")
CSV_PATH = os.path.join(SNAPSHOT_DIR, "prices.csv")
SHA_PATH = os.path.join(SNAPSHOT_DIR, "prices.sha256")
REPORT_PATH = os.path.join(SNAPSHOT_DIR, "frozen_benchmark_report.json")

CRNG_SEED = 42
IID_SEED = 42


def verify_snapshot_integrity() -> str:
    """Read the snapshot CSV, verify its SHA256 matches prices.sha256."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Snapshot missing: {CSV_PATH}. "
            f"Run `python3 benchmarks/freeze_snapshot_2026-04.py` first."
        )
    if not os.path.exists(SHA_PATH):
        raise FileNotFoundError(f"Missing integrity file: {SHA_PATH}")

    with open(CSV_PATH, "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    with open(SHA_PATH) as fh:
        expected = fh.read().strip()

    if actual != expected:
        raise RuntimeError(
            f"SNAPSHOT INTEGRITY FAILURE.\n"
            f"Expected (from prices.sha256): {expected}\n"
            f"Actual   (recomputed):          {actual}\n"
            f"The frozen snapshot has been modified. Per SPECS.md P2, this "
            f"is a P0 protocol violation. Do not continue until investigated."
        )
    return actual


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns, discarding non-finite entries."""
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < 3:
        return np.array([])
    r = np.diff(np.log(prices))
    return r[np.isfinite(r)]


def fingerprint(series: np.ndarray) -> dict:
    """Compute the canonical fingerprint on a *return* series.

    Matches the vocabulary used in ContingencyRNG.stats and
    benchmarks/measure_preset_fingerprints.py — so all three sources
    of truth speak the same language per SPECS.md P5.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 10:
        return {
            "n": int(len(series)),
            "mean": 0.0, "std": 0.0, "kurtosis": 3.0, "skewness": 0.0,
            "vol_acf_lag1": 0.0, "vol_acf_lag5": 0.0,
            "tail_3sigma_pct": 0.0, "tail_4sigma_pct": 0.0,
        }

    m = float(np.mean(series))
    s = float(np.std(series, ddof=1))
    if s == 0:
        return {
            "n": int(len(series)),
            "mean": m, "std": 0.0, "kurtosis": 3.0, "skewness": 0.0,
            "vol_acf_lag1": 0.0, "vol_acf_lag5": 0.0,
            "tail_3sigma_pct": 0.0, "tail_4sigma_pct": 0.0,
        }

    z = (series - m) / s
    kurt = float(np.mean(z ** 4))
    skew = float(np.mean(z ** 3))

    abs_z = np.abs(z)
    tail3 = float(np.mean(abs_z > 3.0) * 100.0)
    tail4 = float(np.mean(abs_z > 4.0) * 100.0)

    abs_v = np.abs(series - m)
    am = float(np.mean(abs_v))
    av = float(np.var(abs_v))
    n = len(abs_v)
    if av > 0 and n > 6:
        acf1 = float(np.sum((abs_v[:-1] - am) * (abs_v[1:] - am)) / (n * av))
        acf5 = float(np.sum((abs_v[:-5] - am) * (abs_v[5:] - am)) / (n * av))
    else:
        acf1 = 0.0
        acf5 = 0.0

    return {
        "n": int(n if n == len(series) else len(series)),
        "mean": m,
        "std": s,
        "kurtosis": kurt,
        "skewness": skew,
        "vol_acf_lag1": acf1,
        "vol_acf_lag5": acf5,
        "tail_3sigma_pct": tail3,
        "tail_4sigma_pct": tail4,
    }


def score_distance(target: dict, achieved: dict) -> dict:
    """Scalar-ish distances between target and achieved fingerprints.

    We do NOT combine these into a single number — per SPECS.md P4, the
    reader must see each metric's gap separately. These distances are
    only auxiliary diagnostics.
    """
    def rel(a, b):
        if abs(b) < 1e-12:
            return float("inf") if abs(a - b) > 1e-12 else 0.0
        return abs(a - b) / abs(b)

    return {
        "abs_kurtosis_gap": abs(target["kurtosis"] - achieved["kurtosis"]),
        "rel_kurtosis_gap": rel(achieved["kurtosis"], target["kurtosis"]),
        "abs_vol_acf_gap": abs(target["vol_acf_lag1"] - achieved["vol_acf_lag1"]),
        "abs_tail3_gap_pp": abs(target["tail_3sigma_pct"] - achieved["tail_3sigma_pct"]),
        "abs_tail4_gap_pp": abs(target["tail_4sigma_pct"] - achieved["tail_4sigma_pct"]),
    }


def run_asset(name: str, prices: np.ndarray) -> dict:
    """Compute target (real), CRNG-from_data, and iid_gaussian fingerprints.

    The a priori rule (SPECS.md P3): for EVERY asset we call ``from_data``
    on the full window. No conditional logic based on what we measured.
    """
    target_returns = log_returns(prices)
    n = len(target_returns)
    target = fingerprint(target_returns)

    if n < 30:
        return {
            "asset": name,
            "skipped_reason": f"insufficient real returns ({n} < 30)",
            "target": target,
        }

    crng = from_data(target_returns, seed=CRNG_SEED)
    crng_values = crng.generate(n)
    achieved_crng = fingerprint(crng_values)

    baseline = iid_gaussian(seed=IID_SEED)
    iid_values = baseline.generate(n)
    achieved_iid = fingerprint(iid_values)

    return {
        "asset": name,
        "n_real_returns": n,
        "seed_crng": CRNG_SEED,
        "seed_iid": IID_SEED,
        "crng_config": {
            "target_kurtosis": crng.target_kurtosis,
            "vol_clustering": crng.vol_clustering,
            "amplification": crng.amplification,
            "n_oscillators": crng.n_osc,
            "cascade_threshold": crng.cascade_threshold,
            "cascade_memory": crng.cascade_memory,
        },
        "target": target,
        "achieved_crng": achieved_crng,
        "achieved_iid": achieved_iid,
        "distance_crng_vs_target": score_distance(target, achieved_crng),
        "distance_iid_vs_target":  score_distance(target, achieved_iid),
    }


def print_row(label: str, f: dict):
    print(
        f"  {label:<20} "
        f"K={f['kurtosis']:>7.2f}  "
        f"acf1={f['vol_acf_lag1']:>+6.3f}  "
        f"acf5={f['vol_acf_lag5']:>+6.3f}  "
        f"3σ={f['tail_3sigma_pct']:>5.2f}%  "
        f"4σ={f['tail_4sigma_pct']:>5.2f}%"
    )


def main():
    print("=" * 88)
    print("Frozen descriptive benchmark — CRNG vs real markets vs iid Gaussian")
    print("=" * 88)

    sha = verify_snapshot_integrity()
    print(f"Snapshot CSV     : {CSV_PATH}")
    print(f"Snapshot SHA256  : {sha}")

    prices = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    print(f"Loaded           : {prices.shape[0]} rows × {prices.shape[1]} assets")
    print(f"Date range       : {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"CRNG seed        : {CRNG_SEED}")
    print(f"iid seed         : {IID_SEED}")
    print()

    results = []
    for asset in prices.columns:
        series = prices[asset].dropna().values
        print(f"[{asset}]  ({len(series)} daily prices)")
        res = run_asset(asset, series)
        if "skipped_reason" in res:
            print(f"  SKIPPED: {res['skipped_reason']}")
            results.append(res)
            print()
            continue
        print_row("target (real)",       res["target"])
        print_row("achieved (CRNG)",      res["achieved_crng"])
        print_row("achieved (iid Gauss)", res["achieved_iid"])
        d_crng = res["distance_crng_vs_target"]
        d_iid  = res["distance_iid_vs_target"]
        better_k    = "CRNG" if d_crng["abs_kurtosis_gap"]  < d_iid["abs_kurtosis_gap"]  else "iid"
        better_acf1 = "CRNG" if d_crng["abs_vol_acf_gap"]   < d_iid["abs_vol_acf_gap"]   else "iid"
        better_t3   = "CRNG" if d_crng["abs_tail3_gap_pp"]  < d_iid["abs_tail3_gap_pp"]  else "iid"
        print(
            f"  closer to target : "
            f"kurtosis→{better_k}  acf1→{better_acf1}  tail3→{better_t3}"
        )
        results.append(res)
        print()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": CSV_PATH,
        "snapshot_sha256": sha,
        "selection_rule": (
            "For every asset, CRNG is built via from_data(prices, seed=42) "
            "on the full window. No preset selection, no post-hoc choice. "
            "Baseline: iid_gaussian(seed=42). n = len(real_returns) per asset."
        ),
        "specs_principles_enforced": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "assets": results,
    }

    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 88)
    print(f"Report JSON : {REPORT_PATH}")
    print()
    print("Summary (kurtosis closeness only, see JSON for full detail):")
    crng_wins_k = 0
    iid_wins_k = 0
    for r in results:
        if "skipped_reason" in r:
            continue
        if r["distance_crng_vs_target"]["abs_kurtosis_gap"] < \
           r["distance_iid_vs_target"]["abs_kurtosis_gap"]:
            crng_wins_k += 1
        else:
            iid_wins_k += 1
        print(
            f"  {r['asset']:<8} target K = {r['target']['kurtosis']:>7.2f}  "
            f"CRNG = {r['achieved_crng']['kurtosis']:>7.2f}  "
            f"iid = {r['achieved_iid']['kurtosis']:>5.2f}"
        )
    print()
    print(f"Kurtosis closer to target : CRNG={crng_wins_k}  iid={iid_wins_k}")
    print()
    print("This is a descriptive result. It does NOT claim CRNG predicts")
    print("future prices. See SPECS.md P1 for the descriptive/predictive")
    print("separation. Cite this report by snapshot SHA256 when public.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
