"""
Measure achieved fingerprints for all CRNG presets.

Generates a reproducible JSON artifact in `benchmarks/preset_fingerprints.json`
with target vs achieved metrics for each preset, at n=100_000 and multiple seeds.

This script is the source of truth for the README preset table.
Re-run it to regenerate the JSON; the README must then be manually updated
to match (or read from the JSON via a build step).

Run:
    cd crng-package && PYTHONPATH=. python benchmarks/measure_preset_fingerprints.py

See SPECS.md P4 — target ≠ achieved, always reported side by side.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crng import (  # noqa: E402
    ContingencyRNG,
    IIDGaussian,
    gaussian,
    gold,
    eurusd,
    eth,
    btc,
    iid_gaussian,
)


N_SAMPLES = 100_000
SEEDS = [42, 123, 256, 314, 555, 777, 1001, 1337, 2025, 9999]


PRESETS = [
    ("iid_gaussian", "SPECS P6 baseline (numpy default_rng)", 3.0, 0.0, iid_gaussian),
    ("gaussian",     "internal reference (CRNG imitating iid)", 3.0, 0.0, gaussian),
    ("gold",         "Gold / XAU-USD",                          9.26, 0.3, gold),
    ("eurusd",       "EUR/USD forex",                           10.5, 0.25, eurusd),
    ("eth",          "Ethereum",                                22.85, 0.4, eth),
    ("btc",          "Bitcoin (extreme)",                       219.0, 0.5, btc),
]


def compute_stats(values: np.ndarray) -> dict:
    """Match the fingerprint vocabulary used across the project."""
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1))
    if s == 0:
        return {
            "kurtosis": 3.0,
            "skewness": 0.0,
            "vol_acf_lag1": 0.0,
            "vol_acf_lag5": 0.0,
            "tail_3sigma_pct": 0.0,
            "tail_4sigma_pct": 0.0,
        }

    z = (values - m) / s
    kurt = float(np.mean(z ** 4))
    skew = float(np.mean(z ** 3))

    abs_z = np.abs(z)
    tail3 = float(np.mean(abs_z > 3.0) * 100)
    tail4 = float(np.mean(abs_z > 4.0) * 100)

    # Vol clustering = autocorrelation of |returns| at lag k.
    abs_v = np.abs(values - m)
    abs_m = float(np.mean(abs_v))
    abs_var = float(np.var(abs_v))
    if abs_var == 0:
        acf1 = acf5 = 0.0
    else:
        n = len(abs_v)
        acf1 = float(np.sum((abs_v[:-1] - abs_m) * (abs_v[1:] - abs_m)) / (n * abs_var))
        if n > 6:
            acf5 = float(np.sum((abs_v[:-5] - abs_m) * (abs_v[5:] - abs_m)) / (n * abs_var))
        else:
            acf5 = 0.0

    return {
        "kurtosis": kurt,
        "skewness": skew,
        "vol_acf_lag1": acf1,
        "vol_acf_lag5": acf5,
        "tail_3sigma_pct": tail3,
        "tail_4sigma_pct": tail4,
    }


def measure_preset(name: str, ctor, seeds=SEEDS, n=N_SAMPLES) -> dict:
    """Measure one preset across multiple seeds and return mean ± std per metric."""
    keys = [
        "kurtosis",
        "skewness",
        "vol_acf_lag1",
        "vol_acf_lag5",
        "tail_3sigma_pct",
        "tail_4sigma_pct",
    ]
    runs = []
    for seed in seeds:
        instance = ctor(seed=seed)
        values = instance.generate(n)
        runs.append(compute_stats(values))

    summary = {"n_samples_per_seed": n, "seeds": list(seeds)}
    for k in keys:
        vals = [r[k] for r in runs]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def main():
    print(f"Measuring {len(PRESETS)} presets at n={N_SAMPLES}, {len(SEEDS)} seeds each.")
    print(f"{'Preset':<14} {'Target K':>10} {'Achieved K (μ±σ)':>24} "
          f"{'Target ACF1':>12} {'Achieved ACF1 (μ±σ)':>24}")
    print("-" * 92)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples_per_seed": N_SAMPLES,
        "seeds": SEEDS,
        "presets": {},
    }

    for name, desc, target_k, target_acf1, ctor in PRESETS:
        summary = measure_preset(name, ctor)
        k = summary["kurtosis"]
        a = summary["vol_acf_lag1"]
        report["presets"][name] = {
            "description": desc,
            "target": {"kurtosis": target_k, "vol_acf_lag1": target_acf1},
            "achieved": summary,
        }
        print(
            f"{name:<14} {target_k:>10.2f} {k['mean']:>14.3f}±{k['std']:.3f}     "
            f"{target_acf1:>12.3f} {a['mean']:>14.4f}±{a['std']:.4f}"
        )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preset_fingerprints.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
