#!/usr/bin/env python3
"""
CRNG-Fragility Monitor — Analyzer
====================================
Computes CRNG metrics on collected data: rolling kurtosis, Z-scores,
fat-tailed confidence intervals, concurrence detection, and alert levels.

Usage:
  python analyzer.py                   # Analyze all symbols
  python analyzer.py --symbol BRENT    # Analyze specific symbol
  python analyzer.py --report          # Full analysis report
  python analyzer.py --backtest        # Run on historical crisis periods
"""

import argparse
import sys
import os
import math
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from db import (init_db, get_connection, get_raw_series, insert_metrics,
                insert_concurrence, get_latest_metrics, get_latest_concurrence)
from collector import SYMBOLS

# ══════════════════════════════════════════════════════════════════
# Statistical Functions
# ══════════════════════════════════════════════════════════════════

def rolling_mean(values, window):
    """Compute rolling mean."""
    if len(values) < window:
        return [None] * len(values)
    result = [None] * (window - 1)
    for i in range(window - 1, len(values)):
        result.append(sum(values[i - window + 1:i + 1]) / window)
    return result


def rolling_std(values, window):
    """Compute rolling standard deviation."""
    if len(values) < window:
        return [None] * len(values)
    result = [None] * (window - 1)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1:i + 1]
        mu = sum(chunk) / window
        var = sum((x - mu) ** 2 for x in chunk) / (window - 1)
        result.append(math.sqrt(var) if var > 0 else 0.001)
    return result


def rolling_kurtosis(values, window):
    """Compute rolling excess kurtosis (Fisher's definition, excess = k - 3)."""
    if len(values) < window:
        return [None] * len(values)
    result = [None] * (window - 1)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1:i + 1]
        n = len(chunk)
        mu = sum(chunk) / n
        m2 = sum((x - mu) ** 2 for x in chunk) / n
        m4 = sum((x - mu) ** 4 for x in chunk) / n

        if m2 < 1e-10:
            result.append(0.0)
        else:
            # Excess kurtosis (normal = 0)
            kurt = (m4 / (m2 ** 2)) - 3.0
            result.append(kurt)
    return result


def rolling_returns(values):
    """Compute daily log returns."""
    returns = [None]
    for i in range(1, len(values)):
        if values[i] > 0 and values[i - 1] > 0:
            returns.append(math.log(values[i] / values[i - 1]))
        else:
            returns.append(0.0)
    return returns


# ══════════════════════════════════════════════════════════════════
# CRNG Metrics Computation
# ══════════════════════════════════════════════════════════════════

def compute_metrics(symbol_key, ma_window=20, kurt_window=60):
    """Compute all CRNG metrics for a symbol."""
    series = get_raw_series(symbol_key)
    if not series or len(series) < kurt_window + 10:
        print(f"    {symbol_key}: insufficient data ({len(series) if series else 0} points, need {kurt_window + 10})")
        return []

    dates = [s[0] for s in series]
    values = [s[1] for s in series]

    # Compute returns for kurtosis
    returns = rolling_returns(values)

    # Rolling metrics on prices
    ma = rolling_mean(values, ma_window)
    std = rolling_std(values, ma_window)

    # Rolling kurtosis on returns (not prices)
    returns_clean = [r for r in returns if r is not None]
    kurt = rolling_kurtosis(returns_clean, kurt_window)
    # Pad kurt to align with dates
    kurt = [None] * (len(dates) - len(kurt)) + kurt

    results = []
    for i in range(len(dates)):
        if ma[i] is None or std[i] is None:
            continue

        z = (values[i] - ma[i]) / std[i] if std[i] > 0 else 0

        # Kurtosis (excess, so normal = 0, fat-tailed > 0)
        k = kurt[i] if i < len(kurt) and kurt[i] is not None else 0.0
        k_full = k + 3.0  # Convert back to regular kurtosis for CI calc

        # Fat-tailed CI multiplier (same formula as weather monitor)
        ci_mult = 1.645 * (1 + max(0, k) / 10.0)

        # CI on price
        ci_lo = ma[i] - ci_mult * std[i]
        ci_hi = ma[i] + ci_mult * std[i]

        # Regime classification
        abs_z = abs(z)
        if abs_z > 3 or k > 3:
            regime = "critical"
        elif abs_z > 2 or k > 1.5:
            regime = "stressed"
        else:
            regime = "normal"

        metrics = {
            'ma20': ma[i],
            'std20': std[i],
            'z_score': z,
            'kurtosis_60': k,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'ci_mult': ci_mult,
            'regime': regime
        }

        results.append((dates[i], metrics))

    return results


def analyze_all(store=True):
    """Compute metrics for all symbols and store in DB."""
    print(f"\n[ANALYZER] Computing CRNG metrics")
    print(f"{'='*60}")

    all_results = {}

    for sym_key in SYMBOLS:
        results = compute_metrics(sym_key)
        if not results:
            continue

        all_results[sym_key] = results
        latest = results[-1]
        regime_icon = {"normal": "🟢", "stressed": "🟡", "critical": "🔴"}.get(latest[1]['regime'], "⚪")

        print(f"  {sym_key:<15} Z={latest[1]['z_score']:>+6.2f}  K={latest[1]['kurtosis_60']:>6.2f}  {regime_icon} {latest[1]['regime']}")

        if store:
            for date, metrics in results:
                insert_metrics(date, sym_key, metrics)

    print(f"{'='*60}")

    # Compute concurrence for the latest date
    if all_results:
        compute_concurrence(all_results, store=store)

    return all_results


# ══════════════════════════════════════════════════════════════════
# Concurrence Detection
# ══════════════════════════════════════════════════════════════════

def compute_concurrence(all_results, store=True):
    """Detect concurrent stress across Tier 1 symbols."""
    # Get latest date with data
    latest_dates = set()
    for sym_key, results in all_results.items():
        if results:
            latest_dates.add(results[-1][0])

    if not latest_dates:
        return

    # For each date that has data, check concurrence
    # We focus on the most recent common date
    target_date = max(latest_dates)

    tier1_stressed = []
    max_k = 0
    max_k_sym = ""

    for sym_key, results in all_results.items():
        sym = SYMBOLS.get(sym_key, {})
        if sym.get('tier') != 1:
            continue

        # Find this date's metrics
        for date, metrics in reversed(results):
            if date <= target_date:
                if abs(metrics['z_score']) > 2:
                    tier1_stressed.append(sym_key)
                k = metrics['kurtosis_60']
                if k > max_k:
                    max_k = k
                    max_k_sym = sym_key
                break

    n_stressed = len(tier1_stressed)

    # Alert level
    if n_stressed >= 3 and max_k > 3:
        alert = "red"
    elif n_stressed >= 2:
        alert = "orange"
    elif n_stressed >= 1:
        alert = "yellow"
    else:
        alert = "green"

    alert_icons = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}

    print(f"\n  CONCURRENCE ({target_date}):")
    print(f"  Tier 1 stressed: {n_stressed}/4 ({', '.join(tier1_stressed) or 'none'})")
    print(f"  Max kurtosis: {max_k:.2f} ({max_k_sym})")
    print(f"  Alert: {alert_icons.get(alert, '?')} {alert.upper()}")

    if store:
        insert_concurrence(target_date, {
            'tier1_stressed': n_stressed,
            'tier1_symbols': ','.join(tier1_stressed) or None,
            'max_kurtosis': max_k,
            'max_kurtosis_symbol': max_k_sym,
            'alert_level': alert,
        })


# ══════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════

def print_report():
    """Print full analysis report."""
    print(f"\n{'='*75}")
    print(f"  CRNG-FRAGILITY MONITOR — ANALYSIS REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*75}")

    metrics = get_latest_metrics()
    if not metrics:
        print("  No metrics computed yet. Run: python analyzer.py")
        return

    # Group by tier
    tier_names = {1: "CHOKE POINT (Hormuz)", 2: "FINANCIAL STRESS",
                  3: "PHYSICAL ECONOMY", 4: "AI BUBBLE"}
    current_tier = 0

    for m in sorted(metrics, key=lambda x: (SYMBOLS.get(x['symbol'], {}).get('tier', 99), x['symbol'])):
        sym = SYMBOLS.get(m['symbol'], {})
        tier = sym.get('tier', 99)

        if tier != current_tier:
            current_tier = tier
            print(f"\n  ── Tier {tier}: {tier_names.get(tier, '?')} ──")
            print(f"  {'Symbol':<15} {'Value':>10} {'MA20':>10} {'Z':>7} {'Kurt':>7} {'CI_lo':>10} {'CI_hi':>10} {'Regime':<10}")
            print(f"  {'-'*80}")

        # Get latest price
        series = get_raw_series(m['symbol'])
        price = series[-1][1] if series else 0

        regime_icon = {"normal": "🟢", "stressed": "🟡", "critical": "🔴"}.get(m['regime'], "⚪")

        print(f"  {m['symbol']:<15} {price:>10.2f} {m['ma20']:>10.2f} {m['z_score']:>+7.2f} {m['kurtosis_60']:>7.2f} {m['ci_lo']:>10.2f} {m['ci_hi']:>10.2f} {regime_icon} {m['regime']}")

    # Concurrence
    conc = get_latest_concurrence()
    if conc:
        alert_icons = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}
        print(f"\n  ── CONCURRENCE ({conc['date']}) ──")
        print(f"  Alert Level: {alert_icons.get(conc['alert_level'], '?')} {conc['alert_level'].upper()}")
        print(f"  Tier 1 Stressed: {conc['tier1_stressed']}/4")
        if conc['tier1_symbols']:
            print(f"  Stressed Symbols: {conc['tier1_symbols']}")
        print(f"  Max Kurtosis: {conc['max_kurtosis']:.2f} ({conc['max_kurtosis_symbol']})")

    # Yield curve (2Y-10Y spread)
    y2 = get_raw_series("YIELD_2Y")
    y10 = get_raw_series("YIELD_10Y")
    if y2 and y10:
        spread = y10[-1][1] - y2[-1][1]
        inv = "⚠️ INVERTED" if spread < 0 else "Normal"
        print(f"\n  Yield Curve (10Y-2Y): {spread:+.2f}% — {inv}")

    print(f"\n{'='*75}")
    print(f"  Keen's key thresholds:")
    print(f"    Energy lockstep: -10% energy ≈ -10% GDP")
    print(f"    India fertilizer: 2-3 months without supply → famine")
    print(f"    Helium: 2-3 months → semiconductor halt")
    print(f"    Australia: 30 days oil reserve")
    print(f"{'='*75}\n")


# ══════════════════════════════════════════════════════════════════
# Historical Crisis Backtest
# ══════════════════════════════════════════════════════════════════

def backtest_crises():
    """Analyze kurtosis behavior during historical crises."""
    print(f"\n{'='*75}")
    print(f"  BACKTEST: Historical Crisis Kurtosis Analysis")
    print(f"{'='*75}")

    crises = [
        ("2008 GFC", "2008-06-01", "2009-03-31"),
        ("2020 COVID", "2020-01-15", "2020-06-30"),
        ("2022 Ukraine", "2022-01-15", "2022-06-30"),
        ("2024 Iran Tensions", "2024-01-01", "2024-06-30"),
    ]

    for name, start, end in crises:
        print(f"\n  ── {name} ({start} → {end}) ──")

        for sym_key in ["BRENT", "WTI", "VIX", "GOLD"]:
            series = get_raw_series(sym_key, start, end)
            if not series or len(series) < 30:
                continue

            values = [s[1] for s in series]
            returns = [r for r in rolling_returns(values) if r is not None]

            if len(returns) < 20:
                continue

            # Compute kurtosis of returns during crisis
            n = len(returns)
            mu = sum(returns) / n
            m2 = sum((x - mu) ** 2 for x in returns) / n
            m4 = sum((x - mu) ** 4 for x in returns) / n

            if m2 < 1e-10:
                k = 0
            else:
                k = (m4 / (m2 ** 2)) - 3.0

            vol = math.sqrt(sum(r ** 2 for r in returns) / n) * math.sqrt(252) * 100
            max_dd = min(returns) * 100

            print(f"    {sym_key:<10} K={k:>+7.2f}  Vol={vol:>6.1f}%  MaxDD={max_dd:>+6.1f}%  N={n}")

    print(f"\n{'='*75}")
    print(f"  Interpretation:")
    print(f"    K > 3: Extreme fat tails (Seneca Cliff territory)")
    print(f"    K > 1: Moderate fat tails (elevated stress)")
    print(f"    K ≈ 0: Normal distribution (gaussian regime)")
    print(f"    K < 0: Thin tails (unusually calm, possible calm before storm)")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    init_db()

    parser = argparse.ArgumentParser(description="CRNG-Fragility Monitor: Analyzer")
    parser.add_argument("--symbol", help="Analyze specific symbol")
    parser.add_argument("--report", action="store_true", help="Full analysis report")
    parser.add_argument("--backtest", action="store_true", help="Historical crisis backtest")
    args = parser.parse_args()

    if args.report:
        print_report()
    elif args.backtest:
        backtest_crises()
    elif args.symbol:
        results = compute_metrics(args.symbol.upper())
        if results:
            latest = results[-1]
            print(f"{args.symbol.upper()}: Z={latest[1]['z_score']:+.2f}, K={latest[1]['kurtosis_60']:.2f}, regime={latest[1]['regime']}")
    else:
        analyze_all()
