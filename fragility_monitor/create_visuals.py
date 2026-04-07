#!/usr/bin/env python3
"""
CRNG-Fragility Monitor — Visualization Generator
===================================================
Creates charts and GIFs for blog posts and social media.
All outputs in 5:2 aspect ratio (1250x500px) for X/Twitter compatibility.
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from db import get_raw_series, get_connection
from analyzer import rolling_returns, rolling_kurtosis, rolling_mean, rolling_std

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#0d1117',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 12,
})

COLORS = {
    'crng_cyan': '#58a6ff',
    'warning_yellow': '#d29922',
    'danger_red': '#f85149',
    'success_green': '#3fb950',
    'purple': '#bc8cff',
    'orange': '#f0883e',
    'white': '#e6edf3',
    'muted': '#8b949e',
}


def fig_5x2(dpi=150):
    """Create 5:2 figure (1250x500 at 150dpi)."""
    return plt.figure(figsize=(1250/dpi, 500/dpi), dpi=dpi)


# ══════════════════════════════════════════════════════════════════
# Chart 1: Kurtosis Comparison — Crises vs 2026
# ══════════════════════════════════════════════════════════════════

def chart_kurtosis_comparison():
    """Bar chart comparing average kurtosis across crisis periods."""
    periods = {
        '2008\nGFC': {'start': '2008-06-01', 'end': '2009-03-31'},
        '2020\nCOVID': {'start': '2020-01-15', 'end': '2020-06-30'},
        '2022\nUkraine': {'start': '2022-01-15', 'end': '2022-06-30'},
        '2024\nIran': {'start': '2024-01-01', 'end': '2024-06-30'},
        '2026\nATUAL': {'start': '2025-10-01', 'end': '2026-04-07'},
    }
    symbols = ['BRENT', 'WTI', 'VIX', 'GOLD', 'NATGAS_US']

    avg_kurts = {}
    for pname, prange in periods.items():
        kurts = []
        for sym in symbols:
            series = get_raw_series(sym, prange['start'], prange['end'])
            if not series or len(series) < 30:
                continue
            values = [s[1] for s in series]
            returns = [r for r in rolling_returns(values) if r is not None]
            if len(returns) < 20:
                continue
            n = len(returns)
            mu = sum(returns) / n
            m2 = sum((x - mu)**2 for x in returns) / n
            m4 = sum((x - mu)**4 for x in returns) / n
            k = (m4 / (m2**2)) - 3.0 if m2 > 1e-10 else 0
            kurts.append(k)
        avg_kurts[pname] = sum(kurts) / len(kurts) if kurts else 0

    fig = fig_5x2()
    ax = fig.add_subplot(111)

    names = list(avg_kurts.keys())
    vals = list(avg_kurts.values())
    colors = [COLORS['muted']] * 4 + [COLORS['danger_red']]

    bars = ax.bar(names, vals, color=colors, edgecolor='none', width=0.6)

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'+{val:.1f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold',
                color=COLORS['danger_red'] if val > 4 else COLORS['white'])

    # Reference lines
    ax.axhline(y=0, color=COLORS['success_green'], linewidth=0.8, alpha=0.5)
    ax.axhline(y=3, color=COLORS['warning_yellow'], linewidth=1, linestyle='--', alpha=0.7)
    ax.text(4.6, 3.2, 'Seneca Cliff threshold', fontsize=9, color=COLORS['warning_yellow'], alpha=0.8)

    ax.set_ylabel('Average Excess Kurtosis (5 key commodities)', fontsize=11)
    ax.set_title('CRNG-Fragility Monitor: Fat-Tail Severity Across Global Crises',
                 fontsize=14, fontweight='bold', color=COLORS['crng_cyan'], pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.2)

    # Annotation
    ax.annotate('2026 is 2× COVID\nand 10× GFC', xy=(4, vals[-1]),
                xytext=(3.0, vals[-1] * 0.7),
                fontsize=10, color=COLORS['danger_red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['danger_red'], lw=1.5),
                fontweight='bold')

    plt.tight_layout()
    path = os.path.join(ASSETS, "01_kurtosis_comparison.png")
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════
# Chart 2: Rolling Kurtosis NATGAS — The Spike
# ══════════════════════════════════════════════════════════════════

def chart_natgas_kurtosis():
    """Line chart showing NATGAS rolling kurtosis over time."""
    series = get_raw_series('NATGAS_US', '2025-04-01', '2026-04-07')
    if not series:
        print("  SKIP: No NATGAS data")
        return None

    dates_str = [s[0] for s in series]
    values = [s[1] for s in series]
    returns = [r for r in rolling_returns(values) if r is not None]
    kurt = rolling_kurtosis(returns, 60)

    # Align
    kurt_aligned = [None] * (len(dates_str) - len(kurt)) + kurt

    fig = fig_5x2()
    ax = fig.add_subplot(111)

    # Plot kurtosis
    valid_dates = []
    valid_kurt = []
    for i, k in enumerate(kurt_aligned):
        if k is not None:
            valid_dates.append(i)
            valid_kurt.append(k)

    ax.fill_between(valid_dates, 0, valid_kurt, alpha=0.3, color=COLORS['danger_red'])
    ax.plot(valid_dates, valid_kurt, color=COLORS['danger_red'], linewidth=2)

    # Thresholds
    ax.axhline(y=0, color=COLORS['success_green'], linewidth=1, linestyle='-', alpha=0.5, label='Normal (K=0)')
    ax.axhline(y=3, color=COLORS['warning_yellow'], linewidth=1.5, linestyle='--', alpha=0.8, label='Fat-tail threshold (K=3)')

    # Labels
    n_ticks = 6
    tick_positions = np.linspace(0, len(dates_str)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates_str[i][:7] for i in tick_positions], fontsize=9)

    ax.set_ylabel('Rolling 60-day Excess Kurtosis', fontsize=11)
    ax.set_title('US Natural Gas: Kurtosis Explosion (K = +16.47)',
                 fontsize=14, fontweight='bold', color=COLORS['danger_red'], pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
    ax.grid(alpha=0.3)

    # Peak annotation
    peak_idx = valid_dates[valid_kurt.index(max(valid_kurt))]
    ax.annotate(f'K = {max(valid_kurt):.1f}\nUnprecedented', xy=(peak_idx, max(valid_kurt)),
                xytext=(peak_idx - 30, max(valid_kurt) * 0.6),
                fontsize=10, color=COLORS['danger_red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['danger_red'], lw=1.5),
                fontweight='bold')

    plt.tight_layout()
    path = os.path.join(ASSETS, "02_natgas_kurtosis.png")
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════
# Chart 3: Dashboard — Current Alert Status
# ══════════════════════════════════════════════════════════════════

def chart_dashboard():
    """Dashboard showing current Z-scores and regimes for all symbols."""
    conn = get_connection()

    rows = conn.execute("""
    SELECT m.symbol, m.z_score, m.kurtosis_60, m.regime, m.ma20,
           r.value, r.unit
    FROM crng_metrics m
    INNER JOIN (
        SELECT symbol, MAX(date) as max_date FROM crng_metrics GROUP BY symbol
    ) latest ON m.symbol = latest.symbol AND m.date = latest.max_date
    INNER JOIN (
        SELECT symbol, MAX(date) as max_date FROM raw_data GROUP BY symbol
    ) lr ON m.symbol = lr.symbol
    INNER JOIN raw_data r ON r.symbol = lr.symbol AND r.date = lr.max_date
    ORDER BY m.symbol
    """).fetchall()
    conn.close()

    if not rows:
        print("  SKIP: No metrics data")
        return None

    fig = fig_5x2(dpi=120)
    ax = fig.add_subplot(111)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, len(rows) - 0.5)

    regime_colors = {
        'normal': COLORS['success_green'],
        'stressed': COLORS['warning_yellow'],
        'critical': COLORS['danger_red'],
    }

    for i, row in enumerate(reversed(rows)):
        y = i
        z = row['z_score']
        k = row['kurtosis_60']
        regime = row['regime']
        color = regime_colors.get(regime, COLORS['muted'])

        # Z-score bar
        ax.barh(y, z, height=0.6, color=color, alpha=0.7, edgecolor='none')

        # Symbol label
        ax.text(-3.9, y, f"{row['symbol']}", fontsize=9, va='center',
                fontweight='bold', color=COLORS['white'])

        # Value + kurtosis label on the right
        val_str = f"{row['value']:.0f}" if row['value'] > 100 else f"{row['value']:.2f}"
        ax.text(3.9, y, f"K={k:+.1f}", fontsize=8, va='center', ha='right',
                color=color, fontweight='bold')

    # Reference lines
    ax.axvline(x=0, color=COLORS['muted'], linewidth=0.5)
    ax.axvline(x=2, color=COLORS['warning_yellow'], linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(x=-2, color=COLORS['warning_yellow'], linewidth=1, linestyle='--', alpha=0.5)

    ax.set_xlabel('Z-Score (deviations from 20-day mean)', fontsize=10)
    ax.set_title('CRNG-Fragility Dashboard — April 7, 2026',
                 fontsize=13, fontweight='bold', color=COLORS['crng_cyan'], pad=12)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.2)

    # Legend
    patches = [mpatches.Patch(color=COLORS['success_green'], label='Normal'),
               mpatches.Patch(color=COLORS['warning_yellow'], label='Stressed'),
               mpatches.Patch(color=COLORS['danger_red'], label='Critical')]
    ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(ASSETS, "03_dashboard.png")
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════
# Chart 4: Hormuz Choke Point — Supply Dependency
# ══════════════════════════════════════════════════════════════════

def chart_hormuz_dependency():
    """Visual showing % of global supply through Hormuz."""
    categories = ['Fertilizer\n(Haber-Bosch)', 'Helium\n(Semiconductors)', 'LNG\n(Energy)', 'Oil\n(Transport)', 'Sulfur\n(Chemicals)']
    percentages = [25, 30, 25, 20, 20]
    collapse_times = ['2-3 months', '2-3 months', '30 days', '30 days', '3 months']

    fig = fig_5x2()
    ax = fig.add_subplot(111)

    colors = [COLORS['danger_red'], COLORS['purple'], COLORS['orange'],
              COLORS['warning_yellow'], COLORS['crng_cyan']]

    bars = ax.barh(range(len(categories)), percentages, color=colors,
                   edgecolor='none', height=0.6, alpha=0.85)

    for i, (bar, pct, time) in enumerate(zip(bars, percentages, collapse_times)):
        ax.text(bar.get_width() + 0.5, i, f'{pct}% → collapse in {time}',
                va='center', fontsize=10, color=COLORS['white'], fontweight='bold')

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel('% of Global Supply Through Strait of Hormuz (21km)', fontsize=10)
    ax.set_title('The 21km Kill Switch: What Passes Through Hormuz',
                 fontsize=14, fontweight='bold', color=COLORS['danger_red'], pad=15)
    ax.set_xlim(0, 55)
    ax.grid(axis='x', alpha=0.2)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(ASSETS, "04_hormuz_dependency.png")
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════
# GIF: Kurtosis building up over time
# ══════════════════════════════════════════════════════════════════

def gif_kurtosis_buildup():
    """Animated GIF showing kurtosis building up from 2025 to 2026."""
    series = get_raw_series('NATGAS_US', '2025-04-01', '2026-04-07')
    if not series or len(series) < 100:
        print("  SKIP: Insufficient NATGAS data for GIF")
        return None

    dates_str = [s[0] for s in series]
    values = [s[1] for s in series]
    returns = [r for r in rolling_returns(values) if r is not None]
    kurt = rolling_kurtosis(returns, 60)
    kurt_aligned = [None] * (len(dates_str) - len(kurt)) + kurt

    # Get valid data
    valid_dates = []
    valid_kurt = []
    for i, k in enumerate(kurt_aligned):
        if k is not None:
            valid_dates.append(i)
            valid_kurt.append(k)

    if len(valid_kurt) < 20:
        print("  SKIP: Not enough valid kurtosis points")
        return None

    fig = fig_5x2(dpi=100)
    ax = fig.add_subplot(111)

    def animate(frame):
        ax.clear()
        n = min(frame * 3 + 10, len(valid_kurt))

        ax.fill_between(valid_dates[:n], 0, valid_kurt[:n], alpha=0.3, color=COLORS['danger_red'])
        ax.plot(valid_dates[:n], valid_kurt[:n], color=COLORS['danger_red'], linewidth=2)

        ax.axhline(y=0, color=COLORS['success_green'], linewidth=0.8, alpha=0.5)
        ax.axhline(y=3, color=COLORS['warning_yellow'], linewidth=1.5, linestyle='--', alpha=0.7)

        ax.set_xlim(valid_dates[0], valid_dates[-1])
        ax.set_ylim(-2, max(valid_kurt) * 1.15)

        current_k = valid_kurt[n-1]
        alert = "CRITICAL" if current_k > 3 else ("STRESSED" if current_k > 1 else "NORMAL")
        alert_color = COLORS['danger_red'] if current_k > 3 else (COLORS['warning_yellow'] if current_k > 1 else COLORS['success_green'])

        ax.set_title(f'US Natural Gas — Rolling Kurtosis: {current_k:+.1f}  [{alert}]',
                     fontsize=13, fontweight='bold', color=alert_color, pad=10)
        ax.set_ylabel('Excess Kurtosis', fontsize=10)

        n_ticks = 5
        tick_positions = np.linspace(valid_dates[0], valid_dates[-1], n_ticks, dtype=int)
        tick_positions = [t for t in tick_positions if t < len(dates_str)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([dates_str[i][:7] for i in tick_positions], fontsize=8)
        ax.grid(alpha=0.2)

    n_frames = len(valid_kurt) // 3 + 1
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=80, repeat=False)

    path = os.path.join(ASSETS, "05_kurtosis_buildup.gif")
    anim.save(path, writer='pillow', fps=12, savefig_kwargs={'facecolor': fig.get_facecolor()})
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════
# Chart 5: Oil Price with Fat-Tailed CI
# ══════════════════════════════════════════════════════════════════

def chart_oil_ci():
    """Brent oil price with CRNG fat-tailed confidence interval."""
    series = get_raw_series('BRENT', '2025-07-01', '2026-04-07')
    if not series:
        print("  SKIP: No BRENT data")
        return None

    dates_str = [s[0] for s in series]
    values = [s[1] for s in series]

    ma = rolling_mean(values, 20)
    std = rolling_std(values, 20)
    returns = [r for r in rolling_returns(values) if r is not None]
    kurt = rolling_kurtosis(returns, 60)
    kurt_aligned = [None] * (len(dates_str) - len(kurt)) + kurt

    fig = fig_5x2()
    ax = fig.add_subplot(111)

    # Price line
    ax.plot(range(len(values)), values, color=COLORS['white'], linewidth=1.5, label='Brent Price', zorder=3)

    # MA20
    valid_ma = [(i, v) for i, v in enumerate(ma) if v is not None]
    if valid_ma:
        ax.plot([x[0] for x in valid_ma], [x[1] for x in valid_ma],
                color=COLORS['crng_cyan'], linewidth=1, linestyle='--', label='MA20', alpha=0.7)

    # Fat-tailed CI
    ci_lo, ci_hi = [], []
    ci_x = []
    for i in range(len(values)):
        if ma[i] is None or std[i] is None:
            continue
        k = kurt_aligned[i] if i < len(kurt_aligned) and kurt_aligned[i] is not None else 0
        ci_mult = 1.645 * (1 + max(0, k) / 10.0)
        ci_lo.append(ma[i] - ci_mult * std[i])
        ci_hi.append(ma[i] + ci_mult * std[i])
        ci_x.append(i)

    if ci_x:
        ax.fill_between(ci_x, ci_lo, ci_hi, alpha=0.15, color=COLORS['crng_cyan'], label='CRNG Fat-tail CI')

    n_ticks = 6
    tick_positions = np.linspace(0, len(dates_str)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates_str[i][:7] for i in tick_positions], fontsize=9)

    ax.set_ylabel('USD/barrel', fontsize=11)
    ax.set_title('Brent Crude Oil — CRNG Fat-Tailed Confidence Interval',
                 fontsize=14, fontweight='bold', color=COLORS['crng_cyan'], pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(ASSETS, "06_oil_ci.png")
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from db import init_db
    init_db()

    print("\n[VISUALS] Generating charts and GIFs...")
    print("=" * 50)

    chart_kurtosis_comparison()
    chart_natgas_kurtosis()
    chart_dashboard()
    chart_hormuz_dependency()
    chart_oil_ci()
    gif_kurtosis_buildup()

    print("=" * 50)
    print("[VISUALS] Done!\n")
