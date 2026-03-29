#!/usr/bin/env python3
"""
Generate regime detector charts for X post.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from crng import from_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import yfinance as yf
from datetime import datetime


# ─── Style ──────────────────────────────────────────────────────

BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
GRID_COLOR = '#21262d'
ACCENT = '#58a6ff'

REGIME_COLORS = {
    'CALM': '#3fb950',
    'NORMAL': '#d29922',
    'STRESSED': '#f85149',
    'CRISIS': '#bc8cff',
}

def classify_regime(k):
    if k < 5: return 'CALM'
    elif k < 12: return 'NORMAL'
    elif k < 30: return 'STRESSED'
    return 'CRISIS'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.3,
    'font.family': 'monospace',
    'font.size': 11,
})


# ─── Data ───────────────────────────────────────────────────────

SYMBOLS = {
    'SPY': 'S&P 500',
    'GLD': 'Gold',
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'CL=F': 'Oil (WTI)',
    'AAPL': 'Apple',
}

def fetch_and_analyze(symbol, period='2y', window=60):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval='1d')
    if hist.empty:
        return None

    prices = hist['Close'].values
    dates = hist.index
    returns = np.diff(np.log(prices[prices > 0]))

    # Sliding window analysis
    step = window // 3
    results = []
    for start in range(0, len(returns) - window + 1, step):
        w = returns[start:start + window]
        std = np.std(w)
        if std == 0:
            continue
        k = float(np.mean(((w - np.mean(w)) / std) ** 4))
        abs_r = np.abs(w)
        mean_abs = np.mean(abs_r)
        var_abs = np.var(abs_r)
        vol_acf = float(np.mean((abs_r[:-1] - mean_abs) * (abs_r[1:] - mean_abs)) / var_abs) if var_abs > 0 else 0
        daily_vol = std * np.sqrt(252) * 100

        # Date index for this window center
        center_idx = min(start + window // 2, len(dates) - 1)

        results.append({
            'date': dates[center_idx],
            'kurtosis': k,
            'vol_acf': vol_acf,
            'annual_vol': daily_vol,
            'regime': classify_regime(k),
        })

    # Multi-scale
    multi = {}
    for w in [20, 60, 120, 252]:
        if len(returns) >= w:
            seg = returns[-w:]
            std = np.std(seg)
            if std > 0:
                multi[w] = float(np.mean(((seg - np.mean(seg)) / std) ** 4))

    return {
        'prices': prices,
        'dates': dates,
        'returns': returns,
        'windows': results,
        'multi_scale': multi,
        'current_price': prices[-1],
    }


# ─── Chart 1: Regime Timeline ──────────────────────────────────

def chart_regime_timeline(data_dict, output_path):
    n = len(data_dict)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle('CRNG REGIME DETECTOR — Market Regime Timeline',
                 fontsize=18, fontweight='bold', color=ACCENT, y=0.98)

    for idx, (symbol, label) in enumerate(SYMBOLS.items()):
        ax = axes[idx]
        info = data_dict.get(symbol)
        if not info or not info['windows']:
            ax.text(0.5, 0.5, f'{label}: No data', ha='center', va='center', transform=ax.transAxes)
            continue

        windows = info['windows']
        dates = [w['date'] for w in windows]
        kurtosis = [w['kurtosis'] for w in windows]
        regimes = [w['regime'] for w in windows]

        # Plot kurtosis line
        ax.plot(dates, kurtosis, color=ACCENT, linewidth=1.5, alpha=0.8, zorder=3)

        # Color background by regime
        for i in range(len(dates) - 1):
            color = REGIME_COLORS[regimes[i]]
            ax.axvspan(dates[i], dates[i + 1], alpha=0.15, color=color, zorder=1)

        # Scatter points colored by regime
        for regime_name, color in REGIME_COLORS.items():
            mask = [r == regime_name for r in regimes]
            d = [dates[j] for j in range(len(dates)) if mask[j]]
            k = [kurtosis[j] for j in range(len(kurtosis)) if mask[j]]
            if d:
                ax.scatter(d, k, c=color, s=25, zorder=4, alpha=0.9, label=regime_name)

        # Threshold lines
        ax.axhline(y=5, color=REGIME_COLORS['NORMAL'], linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axhline(y=12, color=REGIME_COLORS['STRESSED'], linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axhline(y=3, color='#8b949e', linestyle=':', alpha=0.3, linewidth=0.8)

        # Current regime
        curr = windows[-1]
        rc = REGIME_COLORS[curr['regime']]
        ax.annotate(f"NOW: {curr['regime']} (K={curr['kurtosis']:.1f})",
                    xy=(dates[-1], kurtosis[-1]),
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=rc,
                    arrowprops=dict(arrowstyle='->', color=rc, lw=1.5))

        ax.set_ylabel('Kurtosis', fontsize=10)
        ax.set_title(f'{label} ({symbol})  —  ${info["current_price"]:,.2f}',
                     fontsize=13, fontweight='bold', loc='left', pad=8)
        ax.set_ylim(bottom=0, top=min(max(kurtosis) * 1.3, 50))
        ax.grid(True, alpha=0.2)

    # Legend at bottom
    patches = [mpatches.Patch(color=c, label=f'{n} (K={REGIME_COLORS_RANGES[n]})', alpha=0.7)
               for n, c in REGIME_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10,
               framealpha=0.3, edgecolor=GRID_COLOR, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {output_path}")


REGIME_COLORS_RANGES = {
    'CALM': '<5',
    'NORMAL': '5-12',
    'STRESSED': '12-30',
    'CRISIS': '>30',
}


# ─── Chart 2: Multi-Scale Kurtosis Convergence ─────────────────

def chart_multi_scale(data_dict, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))

    fig.suptitle('KURTOSIS CONVERGENCE BY SCALE — Phase Transition in Action',
                 fontsize=16, fontweight='bold', color=ACCENT)

    for symbol, label in SYMBOLS.items():
        info = data_dict.get(symbol)
        if not info or not info['multi_scale']:
            continue

        ms = info['multi_scale']
        windows = sorted(ms.keys())
        k_values = [ms[w] for w in windows]

        color = {'SPY': '#58a6ff', 'GLD': '#d29922', 'BTC-USD': '#f0883e',
                 'ETH-USD': '#bc8cff', 'CL=F': '#3fb950', 'AAPL': '#f85149'}.get(symbol, ACCENT)

        ax.plot(windows, k_values, 'o-', color=color, linewidth=2.5, markersize=10,
                label=f'{label}', alpha=0.9, zorder=3)

        # Annotate endpoint
        ax.annotate(f'K={k_values[-1]:.1f}', xy=(windows[-1], k_values[-1]),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold')

    # Regime thresholds
    ax.axhline(y=3, color='#8b949e', linestyle=':', alpha=0.5, label='Gaussian (K=3)')
    ax.axhline(y=5, color=REGIME_COLORS['NORMAL'], linestyle='--', alpha=0.4)
    ax.axhline(y=12, color=REGIME_COLORS['STRESSED'], linestyle='--', alpha=0.4)

    # Phase transition annotation
    ax.fill_between([15, 260], 0, 5, alpha=0.05, color=REGIME_COLORS['CALM'])
    ax.fill_between([15, 260], 5, 12, alpha=0.05, color=REGIME_COLORS['NORMAL'])
    ax.fill_between([15, 260], 12, 50, alpha=0.05, color=REGIME_COLORS['STRESSED'])

    ax.text(25, 2, 'CALM', fontsize=9, color=REGIME_COLORS['CALM'], alpha=0.7)
    ax.text(25, 7, 'NORMAL', fontsize=9, color=REGIME_COLORS['NORMAL'], alpha=0.7)
    ax.text(25, 15, 'STRESSED', fontsize=9, color=REGIME_COLORS['STRESSED'], alpha=0.7)

    ax.set_xlabel('Window Size (trading days)', fontsize=12)
    ax.set_ylabel('Kurtosis', fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks([20, 60, 120, 252])
    ax.set_xticklabels(['20d\n(1mo)', '60d\n(3mo)', '120d\n(6mo)', '252d\n(1yr)'])
    ax.set_ylim(0, max(40, max(v for info in data_dict.values() if info for v in info.get('multi_scale', {}).values()) * 1.2))
    ax.legend(fontsize=11, loc='upper right', framealpha=0.3, edgecolor=GRID_COLOR)
    ax.grid(True, alpha=0.2)

    # Insight text
    ax.text(0.02, 0.02,
            'Fat tails dissipate at longer scales — kurtosis converges toward Gaussian.\n'
            'This is the phase transition: supercritical amplification retards convergence.',
            transform=ax.transAxes, fontsize=9, color='#8b949e',
            verticalalignment='bottom', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Chart 3: Comparative Dashboard ────────────────────────────

def chart_dashboard(data_dict, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    fig.suptitle('CRNG REGIME DETECTOR — Market Dashboard (March 2026)',
                 fontsize=16, fontweight='bold', color=ACCENT)

    symbols_with_data = [(s, l) for s, l in SYMBOLS.items() if data_dict.get(s) and data_dict[s]['windows']]

    # Panel 1: Current regime bar chart
    ax = axes[0]
    labels = [SYMBOLS[s] for s, _ in symbols_with_data]
    k_values = [data_dict[s]['windows'][-1]['kurtosis'] for s, _ in symbols_with_data]
    colors = [REGIME_COLORS[classify_regime(k)] for k in k_values]

    bars = ax.barh(labels, k_values, color=colors, alpha=0.85, edgecolor=GRID_COLOR)
    ax.axvline(x=3, color='#8b949e', linestyle=':', alpha=0.5)
    ax.axvline(x=5, color=REGIME_COLORS['NORMAL'], linestyle='--', alpha=0.4)
    ax.axvline(x=12, color=REGIME_COLORS['STRESSED'], linestyle='--', alpha=0.4)

    for bar, k in zip(bars, k_values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'K={k:.1f}', va='center', fontsize=10, fontweight='bold', color=TEXT_COLOR)

    ax.set_xlabel('Kurtosis (60d window)', fontsize=10)
    ax.set_title('Current Regime', fontsize=13, fontweight='bold', loc='left')
    ax.set_xlim(0, max(k_values) * 1.4)
    ax.grid(True, axis='x', alpha=0.2)

    # Panel 2: Vol clustering vs Kurtosis scatter
    ax = axes[1]
    for s, l in symbols_with_data:
        info = data_dict[s]
        curr = info['windows'][-1]
        color = REGIME_COLORS[curr['regime']]
        ax.scatter(curr['vol_acf'], curr['kurtosis'], s=200, c=color,
                   edgecolors='white', linewidth=1.5, zorder=3, alpha=0.9)
        ax.annotate(SYMBOLS[s], xy=(curr['vol_acf'], curr['kurtosis']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=TEXT_COLOR)

    ax.axhline(y=3, color='#8b949e', linestyle=':', alpha=0.3)
    ax.axhline(y=5, color=REGIME_COLORS['NORMAL'], linestyle='--', alpha=0.3)
    ax.set_xlabel('Vol Clustering (ACF)', fontsize=10)
    ax.set_ylabel('Kurtosis', fontsize=10)
    ax.set_title('Fat Tails vs Vol Clustering', fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.2)

    # Panel 3: Annualized Vol comparison
    ax = axes[2]
    vols = [data_dict[s]['windows'][-1]['annual_vol'] for s, _ in symbols_with_data]
    vol_colors = [REGIME_COLORS[classify_regime(data_dict[s]['windows'][-1]['kurtosis'])] for s, _ in symbols_with_data]

    bars = ax.barh(labels, vols, color=vol_colors, alpha=0.85, edgecolor=GRID_COLOR)
    for bar, v in zip(bars, vols):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{v:.0f}%', va='center', fontsize=10, fontweight='bold', color=TEXT_COLOR)

    ax.set_xlabel('Annualized Volatility (%)', fontsize=10)
    ax.set_title('Volatility (60d)', fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, axis='x', alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Main ───────────────────────────────────────────────────────

def main():
    print("Fetching market data...")
    data_dict = {}
    for symbol in SYMBOLS:
        print(f"  {symbol}...", end=' ')
        data_dict[symbol] = fetch_and_analyze(symbol, period='2y', window=60)
        if data_dict[symbol]:
            curr = data_dict[symbol]['windows'][-1] if data_dict[symbol]['windows'] else None
            if curr:
                print(f"K={curr['kurtosis']:.1f} ({curr['regime']})")
            else:
                print("no windows")
        else:
            print("FAILED")

    out_dir = '/Users/alebrotto/Deriv MCP/crng-package/charts'
    os.makedirs(out_dir, exist_ok=True)

    print("\nGenerating charts...")
    chart_regime_timeline(data_dict, f'{out_dir}/regime_timeline.png')
    chart_multi_scale(data_dict, f'{out_dir}/multi_scale.png')
    chart_dashboard(data_dict, f'{out_dir}/regime_dashboard.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
