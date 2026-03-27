#!/usr/bin/env python3
"""
CRNG vs Reality vs PRNG — Real-World Comparison (Extended)
==========================================================

Downloads real market data (Gold, EURUSD, ETH, BTC, S&P500, Oil, USDJPY)
with 5 years of history, and compares the statistical signature of:
  1. Real market returns
  2. CRNG (Contingency RNG) — both preset and auto-calibrated
  3. Standard NumPy PRNG (Gaussian)

Robustness: runs 10 different seeds and reports mean ± std for each metric.

Metrics compared:
  - Kurtosis (fat tails)
  - Vol Clustering ACF lag-1 and lag-5 (autocorrelation of |returns|)
  - Tail events (% beyond 3σ and 4σ)
  - Permutation Entropy (PE, order 4)
  - Max drawdown of cumulative returns

Output: terminal table + PNG comparison chart
"""

import numpy as np
import sys
import os
import json
import urllib.request
from collections import Counter
from math import factorial, log

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crng import ContingencyRNG, gold, eurusd, eth, btc, gaussian, from_data


# ─── Data Download ───────────────────────────────────────────────

def get_real_data():
    """Fetch real market data for comparison using yfinance (5y history)."""
    try:
        import yfinance as yf
    except ImportError:
        print("  [!] yfinance not installed. Run: pip install yfinance")
        return {}

    assets = {}
    symbols = {
        "Gold":    "GC=F",
        "EURUSD":  "EURUSD=X",
        "ETH":     "ETH-USD",
        "BTC":     "BTC-USD",
        "SP500":   "^GSPC",
        "Oil":     "CL=F",
        "USDJPY":  "JPY=X",
    }

    print("Downloading real market data (yfinance, 5y history)...")

    for name, symbol in symbols.items():
        print(f"  {name} ({symbol})...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5y", interval="1d")
            if df is not None and len(df) > 100:
                closes = df["Close"].dropna().values
                assets[name] = closes
                print(f"{len(closes)} days")
            else:
                print("not enough data")
        except Exception as e:
            print(f"failed: {e}")

    return assets


# ─── Statistics ──────────────────────────────────────────────────

def compute_returns(prices):
    """Log returns from price series."""
    prices = np.array(prices, dtype=float)
    mask = (prices > 0) & np.isfinite(prices)
    prices = prices[mask]
    returns = np.diff(np.log(prices))
    return returns[np.isfinite(returns)]


def kurtosis_val(x):
    """Kurtosis (excess + 3), so Gaussian = 3."""
    x = np.array(x)
    if len(x) < 4:
        return 3.0
    m, s = np.mean(x), np.std(x, ddof=1)
    if s == 0:
        return 3.0
    return float(np.mean(((x - m) / s) ** 4))


def vol_clustering_acf(returns, lag=1):
    """Autocorrelation of |returns| at given lag."""
    abs_r = np.abs(returns)
    if len(abs_r) < lag + 10:
        return 0.0
    m = np.mean(abs_r)
    var = np.var(abs_r)
    if var == 0:
        return 0.0
    n = len(abs_r)
    return float(np.sum((abs_r[:n-lag] - m) * (abs_r[lag:] - m)) / (n * var))


def tail_events(returns, sigma=3):
    """Fraction of returns beyond sigma standard deviations."""
    std = np.std(returns)
    if std == 0:
        return 0.0
    z = np.abs(returns - np.mean(returns)) / std
    return float(np.mean(z > sigma))


def max_drawdown(returns):
    """Maximum drawdown of cumulative returns."""
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def permutation_entropy(x, order=4):
    """Normalized permutation entropy of a time series."""
    x = np.array(x)
    n = len(x)
    if n < order + 1:
        return 0.0
    patterns = []
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i:i+order]))
        patterns.append(pattern)
    counts = Counter(patterns)
    total = len(patterns)
    max_entropy = log(factorial(order))
    H = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            H -= p * log(p)
    return H / max_entropy if max_entropy > 0 else 0.0


def compute_all_stats(returns):
    """Compute all comparison metrics for a return series."""
    return {
        "N": len(returns),
        "Kurtosis": kurtosis_val(returns),
        "Vol_ACF1": vol_clustering_acf(returns, 1),
        "Vol_ACF5": vol_clustering_acf(returns, 5),
        "Tail_3σ": tail_events(returns, 3) * 100,
        "Tail_4σ": tail_events(returns, 4) * 100,
        "MaxDD": max_drawdown(returns),
        "PE_4": permutation_entropy(returns, 4),
    }


# ─── CRNG Generators ────────────────────────────────────────────

CRNG_PRESETS = {
    "Gold":   gold,
    "EURUSD": eurusd,
    "ETH":    eth,
    "BTC":    btc,
    "SP500":  lambda seed=42: ContingencyRNG(seed=seed, target_kurtosis=7.0, vol_clustering=0.2),
    "Oil":    lambda seed=42: ContingencyRNG(seed=seed, target_kurtosis=15.0, vol_clustering=0.35),
    "USDJPY": lambda seed=42: ContingencyRNG(seed=seed, target_kurtosis=8.0, vol_clustering=0.2),
}

SEEDS = [42, 123, 256, 314, 555, 777, 1001, 1337, 2025, 9999]


def generate_returns_crng(preset_fn, n, seed=42):
    """Generate synthetic log returns using CRNG."""
    rng = preset_fn(seed=seed)
    raw = rng.generate(n)
    std_raw = np.std(raw)
    if std_raw > 0:
        raw = raw / std_raw * 0.01
    return raw


def generate_returns_calibrated(prices, n, seed=42):
    """Generate synthetic log returns using CRNG auto-calibrated from data."""
    rng = from_data(prices, seed=seed)
    raw = rng.generate(n)
    std_raw = np.std(raw)
    if std_raw > 0:
        raw = raw / std_raw * 0.01
    return raw


def generate_returns_prng(n, seed=42):
    """Generate Gaussian random returns (standard PRNG)."""
    rng = np.random.RandomState(seed)
    return rng.normal(0, 0.01, n)


def multi_seed_stats(gen_fn, seeds=SEEDS):
    """Run generator across multiple seeds and return mean ± std of each metric."""
    all_stats = []
    for seed in seeds:
        ret = gen_fn(seed)
        all_stats.append(compute_all_stats(ret))

    result = {}
    for key in all_stats[0]:
        if key == "N":
            result["N"] = all_stats[0]["N"]
            continue
        vals = [s[key] for s in all_stats]
        result[key] = np.mean(vals)
        result[f"{key}_std"] = np.std(vals)
    return result


# ─── Display ─────────────────────────────────────────────────────

METRICS = ["Kurtosis", "Vol_ACF1", "Vol_ACF5", "Tail_3σ", "Tail_4σ", "MaxDD", "PE_4"]
UNITS   = ["",          "",         "",         "%",       "%",       "",      ""]


def fmt_val(val, std, metric, unit):
    """Format value with optional ± std."""
    f = ".1f" if metric == "Kurtosis" else (".4f" if "ACF" in metric or metric == "MaxDD" else ".2f")
    if std is not None and std > 0:
        return f"{val:{f}}{unit}±{std:{f}}"
    return f"{val:{f}}{unit}"


def print_comparison_table(asset_name, real_stats, crng_stats, prng_stats):
    """Print a formatted comparison table with ± std from multi-seed."""
    print(f"\n{'='*78}")
    print(f"  {asset_name} — Real vs CRNG vs PRNG (10 seeds)")
    print(f"  N = {real_stats['N']} daily returns")
    print(f"{'='*78}")
    print(f"  {'Metric':<10} {'Real':>14} {'CRNG (μ±σ)':>18} {'PRNG (μ±σ)':>18}  {'Winner':>6}")
    print(f"  {'─'*10} {'─'*14} {'─'*18} {'─'*18}  {'─'*6}")

    crng_wins = 0
    prng_wins = 0

    for metric, unit in zip(METRICS, UNITS):
        r = real_stats[metric]
        c = crng_stats[metric]
        c_std = crng_stats.get(f"{metric}_std", 0)
        p = prng_stats[metric]
        p_std = prng_stats.get(f"{metric}_std", 0)

        crng_err = abs(c - r)
        prng_err = abs(p - r)
        if crng_err < prng_err:
            winner = "CRNG"
            crng_wins += 1
        else:
            winner = "PRNG"
            prng_wins += 1

        f_str = ".1f" if metric == "Kurtosis" else (".4f" if "ACF" in metric or metric == "MaxDD" else ".2f")
        r_str = f"{r:{f_str}}{unit}"
        c_str = f"{c:{f_str}}±{c_std:{f_str}}"
        p_str = f"{p:{f_str}}±{p_std:{f_str}}"
        print(f"  {metric:<10} {r_str:>14} {c_str:>18} {p_str:>18}  {winner:>6}")

    print(f"  {'─'*10} {'─'*14} {'─'*18} {'─'*18}  {'─'*6}")
    print(f"  Score: CRNG {crng_wins}/{len(METRICS)} | PRNG {prng_wins}/{len(METRICS)}")
    return crng_wins, prng_wins


def try_plot(results):
    """Generate comparison chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not installed — skipping chart]")
        return None

    plot_metrics = ["Kurtosis", "Vol_ACF1", "Tail_3σ", "PE_4"]
    plot_labels = ["Kurtosis\n(Gaussian=3)", "Vol Clustering\n(ACF lag-1)",
                   "Tail Events\n(>3σ %)", "Permutation\nEntropy"]

    n_assets = len(results)
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4.5 * len(plot_metrics), 5.5))

    colors = {"Real": "#2ecc71", "CRNG": "#3498db", "PRNG": "#e74c3c"}

    for i, (metric, label) in enumerate(zip(plot_metrics, plot_labels)):
        ax = axes[i]
        x = np.arange(n_assets)
        width = 0.25

        real_vals = [r["real"][metric] for r in results.values()]
        crng_vals = [r["crng"][metric] for r in results.values()]
        crng_errs = [r["crng"].get(f"{metric}_std", 0) for r in results.values()]
        prng_vals = [r["prng"][metric] for r in results.values()]
        prng_errs = [r["prng"].get(f"{metric}_std", 0) for r in results.values()]

        ax.bar(x - width, real_vals, width, label="Real Market",
               color=colors["Real"], alpha=0.85, edgecolor="white")
        ax.bar(x, crng_vals, width, yerr=crng_errs, label="CRNG",
               color=colors["CRNG"], alpha=0.85, edgecolor="white",
               capsize=3, error_kw={"linewidth": 1})
        ax.bar(x + width, prng_vals, width, yerr=prng_errs, label="NumPy PRNG",
               color=colors["PRNG"], alpha=0.85, edgecolor="white",
               capsize=3, error_kw={"linewidth": 1})

        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(results.keys(), fontsize=8, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("CRNG vs Real Markets vs NumPy PRNG\nStatistical Signature Comparison (10 seeds, 5y data)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comparison_chart.png")
    out_path = os.path.abspath(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Chart saved: {out_path}")
    plt.close()
    return out_path


# ─── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  CRNG vs Reality vs PRNG — Extended Real-World Comparison")
    print("  7 assets × 5 years × 7 metrics × 10 seeds")
    print("=" * 78)

    # 1. Download real data
    assets = get_real_data()

    if not assets:
        print("\n[ERROR] Could not download any market data.")
        sys.exit(1)

    # 2. Compare each asset
    total_crng_wins = 0
    total_prng_wins = 0
    total_metrics = 0
    results = {}

    for name, prices in assets.items():
        returns = compute_returns(prices)
        n = len(returns)

        if n < 50:
            print(f"\n  Skipping {name}: only {n} returns")
            continue

        # Real stats (single ground truth)
        real_stats = compute_all_stats(returns)

        # CRNG: best of preset vs auto-calibrated, across 10 seeds
        preset_fn = CRNG_PRESETS.get(name)

        # Test preset with single seed to compare kurtosis
        if preset_fn:
            test_preset = compute_all_stats(generate_returns_crng(preset_fn, n, seed=42))
            test_auto = compute_all_stats(generate_returns_calibrated(prices, n, seed=42))
            k_real = real_stats["Kurtosis"]
            use_preset = abs(test_preset["Kurtosis"] - k_real) < abs(test_auto["Kurtosis"] - k_real)
        else:
            use_preset = False

        if use_preset:
            crng_stats = multi_seed_stats(lambda s: generate_returns_crng(preset_fn, n, s))
            mode = "preset"
        else:
            crng_stats = multi_seed_stats(lambda s: generate_returns_calibrated(prices, n, s))
            mode = "auto"

        # PRNG: 10 seeds
        prng_stats = multi_seed_stats(lambda s: generate_returns_prng(n, s))

        # Display
        print(f"  [mode: {mode}]", end="")
        cw, pw = print_comparison_table(name, real_stats, crng_stats, prng_stats)
        total_crng_wins += cw
        total_prng_wins += pw
        total_metrics += cw + pw

        results[name] = {"real": real_stats, "crng": crng_stats, "prng": prng_stats}

    # 3. Final score
    print(f"\n{'='*78}")
    print(f"  FINAL SCORE — {len(results)} assets × {len(METRICS)} metrics = {total_metrics} comparisons")
    print(f"{'='*78}")
    print(f"  CRNG wins: {total_crng_wins}/{total_metrics}")
    print(f"  PRNG wins: {total_prng_wins}/{total_metrics}")

    pct = total_crng_wins / total_metrics * 100 if total_metrics > 0 else 0
    print(f"\n  CRNG reproduces {pct:.0f}% of real market statistical signatures.")

    if pct >= 70:
        print(f"  Verdict: CRNG captures market reality. PRNG doesn't.")
    elif pct >= 50:
        print(f"  Verdict: CRNG significantly closer to reality than PRNG.")
    else:
        print(f"  Verdict: Mixed results — needs investigation.")

    # 4. Per-asset summary
    print(f"\n  Per-asset breakdown:")
    for name, r in results.items():
        rk = r["real"]["Kurtosis"]
        ck = r["crng"]["Kurtosis"]
        ck_std = r["crng"].get("Kurtosis_std", 0)
        pk = r["prng"]["Kurtosis"]
        print(f"    {name:8} K_real={rk:5.1f}  K_crng={ck:5.1f}±{ck_std:.1f}  K_prng={pk:4.1f}")

    # 5. Chart
    chart_path = try_plot(results)

    # 6. Posting summary
    print(f"\n{'='*78}")
    print(f"  COPY-PASTE FOR X / HACKER NEWS")
    print(f"{'='*78}")
    print(f"  Real markets have fat tails (K=5-40), vol clustering,")
    print(f"  and extreme events. NumPy PRNG gives K=3 for everything.")
    print(f"")
    print(f"  CRNG matches {pct:.0f}% of real market signatures across")
    print(f"  {len(results)} assets, 5 years, {len(METRICS)} metrics, 10 seeds.")
    print(f"")
    for name, r in results.items():
        rk = r["real"]["Kurtosis"]
        ck = r["crng"]["Kurtosis"]
        pk = r["prng"]["Kurtosis"]
        print(f"    {name:8} Real K={rk:5.1f} | CRNG K={ck:5.1f} | PRNG K={pk:4.1f}")
    print(f"")
    print(f"  pip install crng")
    print(f"  https://github.com/brotto/crng")
    print()


if __name__ == "__main__":
    main()
