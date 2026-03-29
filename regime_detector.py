#!/usr/bin/env python3
"""
CRNG Regime Detector — Real-time market regime detection via CRNG calibration
==============================================================================

Uses sliding-window CRNG from_data() calibration to detect market regime changes.
When the calibrated parameters (kurtosis, vol clustering, amplification) shift
significantly, it signals a regime change.

Regimes:
  CALM       — Low kurtosis (K < 5), low vol clustering
  NORMAL     — Moderate kurtosis (5 ≤ K < 12), moderate vol clustering
  STRESSED   — High kurtosis (12 ≤ K < 30), high vol clustering
  CRISIS     — Extreme kurtosis (K ≥ 30), extreme vol clustering

Usage:
  python regime_detector.py                    # Default: SPY, Gold, BTC, ETH
  python regime_detector.py AAPL TSLA MSFT     # Custom symbols
  python regime_detector.py --window 60        # 60-day sliding window
  python regime_detector.py --live             # Continuous polling (every 5min)
"""

import numpy as np
import sys
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from crng import ContingencyRNG, from_data


# ─── Regime Classification ──────────────────────────────────────

REGIMES = {
    'CALM':     {'k_range': (0, 5),    'color': '\033[92m', 'icon': '●'},  # green
    'NORMAL':   {'k_range': (5, 12),   'color': '\033[93m', 'icon': '●'},  # yellow
    'STRESSED': {'k_range': (12, 30),  'color': '\033[91m', 'icon': '●'},  # red
    'CRISIS':   {'k_range': (30, 999), 'color': '\033[95m', 'icon': '◆'},  # magenta
}

RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'
CYAN = '\033[96m'
WHITE = '\033[97m'


def classify_regime(kurtosis):
    for name, props in REGIMES.items():
        lo, hi = props['k_range']
        if lo <= kurtosis < hi:
            return name
    return 'CRISIS'


def regime_color(regime):
    return REGIMES.get(regime, REGIMES['CRISIS'])['color']


def regime_icon(regime):
    return REGIMES.get(regime, REGIMES['CRISIS'])['icon']


# ─── Data Fetching ──────────────────────────────────────────────

def fetch_prices(symbol, period='1y', interval='1d'):
    """Fetch historical prices via yfinance."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist.empty:
        return None, None
    prices = hist['Close'].values
    dates = hist.index
    return prices, dates


def compute_returns(prices):
    """Compute log returns from prices."""
    prices = np.array(prices, dtype=float)
    prices = prices[prices > 0]
    if len(prices) < 10:
        return np.array([])
    return np.diff(np.log(prices))


# ─── CRNG Calibration & Regime Detection ────────────────────────

def calibrate_window(returns, seed=42):
    """Calibrate CRNG from a window of returns. Returns regime parameters."""
    if len(returns) < 20:
        return None

    try:
        crng = from_data(returns, seed=seed)
        stats = crng.stats(n=5000)

        # Measure actual data properties
        abs_returns = np.abs(returns)
        data_kurtosis = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4)) if np.std(returns) > 0 else 3.0

        # Vol clustering from data
        if len(abs_returns) > 1:
            mean_abs = np.mean(abs_returns)
            var_abs = np.var(abs_returns)
            if var_abs > 0:
                vol_acf = float(np.mean((abs_returns[:-1] - mean_abs) * (abs_returns[1:] - mean_abs)) / var_abs)
            else:
                vol_acf = 0.0
        else:
            vol_acf = 0.0

        # Tail events
        if np.std(returns) > 0:
            z_scores = np.abs((returns - np.mean(returns)) / np.std(returns))
            gt_2sigma = float(np.mean(z_scores > 2))
            gt_3sigma = float(np.mean(z_scores > 3))
        else:
            gt_2sigma = gt_3sigma = 0.0

        # Daily volatility (annualized)
        daily_vol = float(np.std(returns))
        annual_vol = daily_vol * np.sqrt(252) * 100  # percentage

        return {
            'kurtosis': data_kurtosis,
            'vol_clustering': vol_acf,
            'crng_kurtosis': stats['kurtosis'],
            'crng_target': crng.target_kurtosis,
            'crng_amp': crng.amplification,
            'daily_vol': daily_vol,
            'annual_vol': annual_vol,
            'gt_2sigma': gt_2sigma,
            'gt_3sigma': gt_3sigma,
            'n_returns': len(returns),
            'regime': classify_regime(data_kurtosis),
        }
    except Exception as e:
        return {'error': str(e)}


def sliding_window_analysis(returns, window_size=60, step=20):
    """Run sliding window regime analysis."""
    results = []
    n = len(returns)

    for start in range(0, n - window_size + 1, step):
        window = returns[start:start + window_size]
        params = calibrate_window(window)
        if params and 'error' not in params:
            params['window_start'] = start
            params['window_end'] = start + window_size
            results.append(params)

    return results


# ─── Regime Change Detection ────────────────────────────────────

def detect_regime_changes(history):
    """Detect points where regime changes."""
    if len(history) < 2:
        return []

    changes = []
    for i in range(1, len(history)):
        prev = history[i - 1]['regime']
        curr = history[i]['regime']
        if prev != curr:
            changes.append({
                'window': i,
                'from': prev,
                'to': curr,
                'kurtosis_from': history[i - 1]['kurtosis'],
                'kurtosis_to': history[i]['kurtosis'],
            })

    return changes


# ─── Display ────────────────────────────────────────────────────

def print_header():
    print(f"\n{BOLD}{CYAN}{'=' * 100}")
    print(f"  CRNG REGIME DETECTOR — Real-time market regime analysis via contingency calibration")
    print(f"{'=' * 100}{RESET}\n")


def print_asset_analysis(symbol, prices, dates, window_size):
    """Full analysis for one asset."""
    returns = compute_returns(prices)
    if len(returns) < window_size:
        print(f"  {symbol}: Not enough data ({len(returns)} returns, need {window_size})")
        return

    # Current regime (latest window)
    current = calibrate_window(returns[-window_size:])
    if not current or 'error' in current:
        print(f"  {symbol}: Calibration failed")
        return

    regime = current['regime']
    rc = regime_color(regime)
    ri = regime_icon(regime)

    # Full history for regime changes
    history = sliding_window_analysis(returns, window_size=window_size, step=window_size // 3)
    changes = detect_regime_changes(history)

    # Price info
    price_now = prices[-1]
    price_prev = prices[-2] if len(prices) > 1 else price_now
    price_change = (price_now / price_prev - 1) * 100
    price_sign = '+' if price_change >= 0 else ''
    price_color = '\033[92m' if price_change >= 0 else '\033[91m'

    # Print asset header
    print(f"  {BOLD}{WHITE}{symbol}{RESET}")
    print(f"  {'─' * 96}")

    # Price line
    print(f"  Price: ${price_now:,.2f}  {price_color}{price_sign}{price_change:.2f}%{RESET}   "
          f"Vol(ann): {current['annual_vol']:.1f}%   "
          f"Window: {window_size}d")

    # Regime line
    print(f"  Regime: {rc}{ri} {regime}{RESET}   "
          f"K={current['kurtosis']:.1f}   "
          f"VolACF={current['vol_clustering']:.3f}   "
          f">2σ={current['gt_2sigma']*100:.1f}%   "
          f">3σ={current['gt_3sigma']*100:.1f}%")

    # CRNG calibration
    print(f"  CRNG:  K_target={current['crng_target']:.1f}   "
          f"K_output={current['crng_kurtosis']:.1f}   "
          f"Amp={current['crng_amp']:.2f}")

    # Regime history bar
    if history:
        bar = "  History: "
        for h in history:
            r = h['regime']
            bar += f"{regime_color(r)}{regime_icon(r)}{RESET}"
        bar += f"  ({len(history)} windows)"
        print(bar)

    # Recent regime changes
    if changes:
        recent = changes[-3:]  # last 3 changes
        for c in recent:
            fc = regime_color(c['from'])
            tc = regime_color(c['to'])
            print(f"  Change: {fc}{c['from']}{RESET} → {tc}{c['to']}{RESET}  "
                  f"(K: {c['kurtosis_from']:.1f} → {c['kurtosis_to']:.1f})")
    else:
        print(f"  {DIM}No regime changes in analysis period{RESET}")

    print()

    return current


def print_regime_legend():
    print(f"  {DIM}Regimes:{RESET}", end="")
    for name, props in REGIMES.items():
        lo, hi = props['k_range']
        hi_str = f"{hi}" if hi < 999 else "∞"
        print(f"  {props['color']}{props['icon']} {name} (K={lo}-{hi_str}){RESET}", end="")
    print()


def print_comparison_table(results):
    """Print comparative table across assets."""
    if not results:
        return

    print(f"\n  {BOLD}COMPARATIVE TABLE{RESET}")
    print(f"  {'─' * 96}")
    header = f"  {'ASSET':<10} {'REGIME':<12} {'KURTOSIS':>10} {'VOL(ann)':>10} {'VOL_ACF':>10} {'CRNG_AMP':>10} {'>2σ':>8} {'>3σ':>8}"
    print(f"  {BOLD}{header}{RESET}")
    print(f"  {'─' * 96}")

    for symbol, params in sorted(results.items(), key=lambda x: x[1]['kurtosis'], reverse=True):
        r = params['regime']
        rc = regime_color(r)
        print(f"  {symbol:<10} {rc}{r:<12}{RESET} {params['kurtosis']:>10.2f} {params['annual_vol']:>9.1f}% {params['vol_clustering']:>10.3f} {params['crng_amp']:>10.2f} {params['gt_2sigma']*100:>7.1f}% {params['gt_3sigma']*100:>7.1f}%")

    print(f"  {'─' * 96}")

    # Insight
    k_values = [p['kurtosis'] for p in results.values()]
    avg_k = np.mean(k_values)
    market_regime = classify_regime(avg_k)
    mc = regime_color(market_regime)
    print(f"\n  {BOLD}Market Overview:{RESET} Avg K={avg_k:.1f} → {mc}{market_regime}{RESET}")

    # CRNG insight
    print(f"\n  {DIM}Note: CRNG Amp (amplification) indicates how far above the phase transition")
    print(f"  threshold each asset operates. Higher amp = deeper into the supercritical regime.{RESET}")


# ─── Multi-Window Analysis ──────────────────────────────────────

def multi_window_analysis(symbol, prices, windows=[20, 60, 120]):
    """Analyze same asset at multiple time scales."""
    returns = compute_returns(prices)
    if len(returns) < max(windows):
        return None

    results = {}
    for w in windows:
        params = calibrate_window(returns[-w:])
        if params and 'error' not in params:
            results[f"{w}d"] = params

    if not results:
        return None

    print(f"\n  {BOLD}MULTI-SCALE ANALYSIS: {symbol}{RESET}")
    print(f"  {'─' * 96}")
    print(f"  {'WINDOW':<10} {'REGIME':<12} {'KURTOSIS':>10} {'VOL(ann)':>10} {'VOL_ACF':>10} {'CRNG_AMP':>10}")
    print(f"  {'─' * 96}")

    for label, params in results.items():
        r = params['regime']
        rc = regime_color(r)
        print(f"  {label:<10} {rc}{r:<12}{RESET} {params['kurtosis']:>10.2f} {params['annual_vol']:>9.1f}% {params['vol_clustering']:>10.3f} {params['crng_amp']:>10.2f}")

    # Convergence insight (like the phase transition discovery)
    k_values = [results[k]['kurtosis'] for k in sorted(results.keys())]
    if len(k_values) >= 2 and k_values[0] > k_values[-1]:
        print(f"\n  {DIM}→ Kurtosis convergence: K decreases with scale ({k_values[0]:.1f} → {k_values[-1]:.1f})")
        print(f"    This matches the phase transition model — fat tails dissipate at longer scales.{RESET}")

    return results


# ─── Main ────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CRNG Regime Detector')
    parser.add_argument('symbols', nargs='*', default=['SPY', 'GLD', 'BTC-USD', 'ETH-USD'])
    parser.add_argument('--window', type=int, default=60, help='Sliding window size in days')
    parser.add_argument('--period', default='2y', help='Historical period (1y, 2y, 5y)')
    parser.add_argument('--live', action='store_true', help='Continuous polling mode')
    parser.add_argument('--interval', type=int, default=300, help='Polling interval in seconds')
    parser.add_argument('--multi', action='store_true', help='Multi-scale analysis')
    args = parser.parse_args()

    while True:
        print_header()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {DIM}Timestamp: {timestamp}   Period: {args.period}   Window: {args.window}d{RESET}")
        print_regime_legend()
        print()

        all_results = {}

        for symbol in args.symbols:
            print(f"  {DIM}Fetching {symbol}...{RESET}", end='\r')
            prices, dates = fetch_prices(symbol, period=args.period)

            if prices is None or len(prices) < args.window:
                print(f"  {symbol}: Could not fetch data or insufficient history")
                continue

            result = print_asset_analysis(symbol, prices, dates, args.window)
            if result:
                all_results[symbol] = result

            if args.multi:
                multi_window_analysis(symbol, prices, windows=[20, 60, 120, 252])

        if all_results:
            print_comparison_table(all_results)

        if not args.live:
            break

        print(f"\n  {DIM}Next update in {args.interval}s... (Ctrl+C to stop){RESET}")
        try:
            time.sleep(args.interval)
            # Clear screen for next iteration
            print('\033[2J\033[H', end='')
        except KeyboardInterrupt:
            print(f"\n  Stopped.")
            break

    print()


if __name__ == '__main__':
    main()
