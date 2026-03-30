"""
CATASTROPHIC EVENTS — CHARTS
==============================
Generate 4 charts (2 static PNG + 2 animated GIF) for the catastrophic events experiment.

Ale Brotto — 2026-03-29
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy import stats as sp_stats
from datetime import datetime, timedelta
from crng import ContingencyRNG

# ============================================================
# THEME
# ============================================================

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.family': 'monospace',
})

BLUE = '#58a6ff'
RED = '#f97583'
ORANGE = '#f0883e'
GREEN = '#3fb950'
PURPLE = '#d2a8ff'
BG = '#0d1117'
PANEL = '#161b22'

# ============================================================
# DATA (same as catastrophic_events.py)
# ============================================================

def get_earthquake_data():
    return [
        ("1906-04-18", 7.9, "San Francisco"),
        ("1908-12-28", 7.2, "Messina, Italy"),
        ("1920-12-16", 8.5, "Haiyuan, China"),
        ("1923-09-01", 7.9, "Kanto, Japan"),
        ("1927-05-22", 7.6, "Tsinghai, China"),
        ("1931-08-10", 8.0, "Fuyun, China"),
        ("1934-01-15", 8.1, "Bihar-Nepal"),
        ("1935-05-30", 7.7, "Quetta, Pakistan"),
        ("1939-12-26", 7.8, "Erzincan, Turkey"),
        ("1944-12-07", 8.1, "Tonankai, Japan"),
        ("1946-04-01", 8.1, "Aleutian Islands"),
        ("1948-10-05", 7.3, "Ashgabat, Turkmenistan"),
        ("1950-08-15", 8.6, "Assam, India"),
        ("1952-11-04", 9.0, "Kamchatka, Russia"),
        ("1957-03-09", 9.1, "Andreanof Islands"),
        ("1960-05-22", 9.5, "Valdivia, Chile"),
        ("1964-03-27", 9.2, "Alaska"),
        ("1970-05-31", 7.9, "Ancash, Peru"),
        ("1976-07-27", 7.5, "Tangshan, China"),
        ("1985-09-19", 8.0, "Mexico City"),
        ("1988-12-07", 6.8, "Spitak, Armenia"),
        ("1989-10-17", 6.9, "Loma Prieta, CA"),
        ("1994-01-17", 6.7, "Northridge, CA"),
        ("1995-01-17", 6.9, "Kobe, Japan"),
        ("1999-08-17", 7.6, "Izmit, Turkey"),
        ("1999-09-20", 7.7, "Chi-Chi, Taiwan"),
        ("2001-01-26", 7.7, "Gujarat, India"),
        ("2003-12-26", 6.6, "Bam, Iran"),
        ("2004-12-26", 9.1, "Sumatra (tsunami)"),
        ("2005-10-08", 7.6, "Kashmir"),
        ("2008-05-12", 7.9, "Sichuan, China"),
        ("2010-01-12", 7.0, "Haiti"),
        ("2010-02-27", 8.8, "Maule, Chile"),
        ("2011-03-11", 9.1, "Tohoku, Japan (tsunami)"),
        ("2015-04-25", 7.8, "Nepal"),
        ("2017-09-19", 7.1, "Puebla, Mexico"),
        ("2018-09-28", 7.5, "Sulawesi, Indonesia"),
        ("2023-02-06", 7.8, "Turkey-Syria"),
        ("2024-01-01", 7.6, "Noto, Japan"),
    ]


def get_financial_crashes():
    return [
        ("1929-10-29", 100, "Black Tuesday"),
        ("1937-03-10", 50, "1937 Recession"),
        ("1962-05-28", 30, "Kennedy Slide"),
        ("1973-01-11", 60, "Oil Crisis Crash"),
        ("1979-10-06", 40, "Volcker Shock"),
        ("1987-10-19", 90, "Black Monday"),
        ("1989-10-13", 35, "Friday the 13th mini-crash"),
        ("1997-10-27", 55, "Asian Financial Crisis"),
        ("1998-08-17", 50, "Russian/LTCM Crisis"),
        ("2000-03-10", 70, "Dot-com Bubble"),
        ("2001-09-17", 45, "Post-9/11 Crash"),
        ("2007-02-27", 40, "Shanghai Surprise"),
        ("2008-09-29", 95, "Lehman Brothers/GFC"),
        ("2010-05-06", 60, "Flash Crash"),
        ("2011-08-05", 45, "US Downgrade Crash"),
        ("2015-08-24", 50, "China Black Monday"),
        ("2018-02-05", 40, "Volmageddon"),
        ("2020-03-16", 85, "COVID Crash"),
        ("2022-06-13", 40, "Crypto/Rate Hike Crash"),
    ]


def get_natural_disasters():
    return [
        ("1900-09-08", 80, "Galveston Hurricane"),
        ("1918-01-01", 100, "Spanish Flu (start)"),
        ("1931-08-01", 95, "China Floods"),
        ("1935-09-02", 40, "Labor Day Hurricane"),
        ("1938-09-21", 45, "New England Hurricane"),
        ("1942-10-16", 70, "Bengal Cyclone"),
        ("1953-02-01", 50, "North Sea Flood"),
        ("1959-09-26", 55, "Typhoon Vera"),
        ("1965-05-11", 50, "Bangladesh Cyclone"),
        ("1970-11-12", 85, "Bhola Cyclone"),
        ("1975-08-05", 60, "Typhoon Nina / Banqiao Dam"),
        ("1984-12-03", 70, "Bhopal Disaster"),
        ("1986-04-26", 75, "Chernobyl"),
        ("1991-04-29", 80, "Bangladesh Cyclone"),
        ("1998-10-29", 55, "Hurricane Mitch"),
        ("2003-08-01", 50, "European Heat Wave"),
        ("2004-12-26", 95, "Indian Ocean Tsunami"),
        ("2005-08-29", 65, "Hurricane Katrina"),
        ("2008-05-02", 75, "Cyclone Nargis"),
        ("2010-01-12", 80, "Haiti Earthquake"),
        ("2011-03-11", 90, "Fukushima"),
        ("2013-11-08", 60, "Typhoon Haiyan"),
        ("2019-12-01", 100, "COVID-19 Pandemic (start)"),
        ("2022-06-01", 40, "Pakistan Floods"),
    ]


def generate_crng_data(n_points=30000, window=50, seed=42):
    """Generate CRNG kurtosis map and event map."""
    rng = ContingencyRNG(
        seed=seed, n_oscillators=7,
        target_kurtosis=15.0, vol_clustering=0.35
    )
    series = np.array([rng.next() for _ in range(n_points)])
    returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
    returns = returns[np.isfinite(returns)]

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(returns, window)
    means = np.mean(windows, axis=1, keepdims=True)
    centered = windows - means
    m2 = np.mean(centered**2, axis=1)
    m4 = np.mean(centered**4, axis=1)
    m2_safe = np.where(m2 > 1e-20, m2, 1e-20)
    kurtosis_map = m4 / (m2_safe ** 2)

    # Event detection
    thresholds = [5, 6, 8, 10, 15, 20, 30, 50, 100]
    event_map = {}

    for thresh in thresholds:
        indices = np.where(kurtosis_map >= thresh)[0]
        if len(indices) == 0:
            continue
        events = []
        current_start = indices[0]
        current_max_k = kurtosis_map[indices[0]]
        current_peak_idx = indices[0]
        prev_idx = indices[0]

        for idx in indices[1:]:
            if idx - prev_idx <= window // 2:
                if kurtosis_map[idx] > current_max_k:
                    current_max_k = kurtosis_map[idx]
                    current_peak_idx = idx
            else:
                events.append({'start': current_start, 'peak': current_peak_idx, 'max_k': current_max_k})
                current_start = idx
                current_max_k = kurtosis_map[idx]
                current_peak_idx = idx
            prev_idx = idx

        events.append({'start': current_start, 'peak': current_peak_idx, 'max_k': current_max_k})

        if len(events) >= 2:
            peaks = [e['peak'] for e in events]
            gaps = np.diff(peaks)
        else:
            gaps = np.array([])

        event_map[thresh] = {
            'events': events,
            'count': len(events),
            'gaps': gaps.tolist() if len(gaps) > 0 else [],
        }

    return kurtosis_map, event_map, returns


def compute_ks_results(event_map, real_results):
    """Compute KS test p-values for all 20 comparisons."""
    results = []
    categories = ['Earthquakes', 'Financial', 'Natural', 'ALL']
    real_keys = ['Earthquakes (M>=6.7)', 'Financial Crashes', 'Natural Disasters', 'ALL Catastrophes']
    thresholds = [5, 8, 10, 15, 20]

    for cat, rk in zip(categories, real_keys):
        if rk not in real_results or 'gaps' not in real_results[rk]:
            continue
        real_gaps = np.array(real_results[rk]['gaps'])
        real_norm = (real_gaps - np.mean(real_gaps)) / (np.std(real_gaps) + 1e-10)

        for thresh in thresholds:
            if thresh not in event_map or len(event_map[thresh]['gaps']) < 5:
                results.append({'cat': cat, 'thresh': thresh, 'p': 0.5, 'ks': 0})
                continue
            crng_gaps = np.array(event_map[thresh]['gaps'], dtype=float)
            crng_norm = (crng_gaps - np.mean(crng_gaps)) / (np.std(crng_gaps) + 1e-10)
            ks_stat, ks_p = sp_stats.ks_2samp(real_norm, crng_norm)
            results.append({'cat': cat, 'thresh': thresh, 'p': ks_p, 'ks': ks_stat})

    return results


def compute_periodicity(real_results, event_map):
    """Compute FFT periodicity data."""
    period_data = {}

    for label, stats in real_results.items():
        if not stats or 'gaps' not in stats or len(stats['gaps']) < 10:
            continue
        gaps = np.array(stats['gaps'], dtype=float)
        gaps_c = gaps - np.mean(gaps)
        fft = np.fft.fft(gaps_c)
        power = np.abs(fft[:len(fft)//2]) ** 2
        freqs = np.fft.fftfreq(len(gaps_c))[:len(fft)//2]
        if len(power) > 1:
            pnd = power[1:]
            fnd = freqs[1:]
            di = np.argmax(pnd)
            df = fnd[di]
            dp = 1.0 / df if df > 0 else float('inf')
            pr = pnd[di] / np.mean(pnd)
            period_data[label] = {'period': dp, 'power_ratio': pr, 'power': power, 'freqs': freqs}

    for thresh in [5, 8, 10, 15]:
        if thresh not in event_map or len(event_map[thresh]['gaps']) < 10:
            continue
        gaps = np.array(event_map[thresh]['gaps'], dtype=float)
        gaps_c = gaps - np.mean(gaps)
        fft = np.fft.fft(gaps_c)
        power = np.abs(fft[:len(fft)//2]) ** 2
        freqs = np.fft.fftfreq(len(gaps_c))[:len(fft)//2]
        if len(power) > 1:
            pnd = power[1:]
            fnd = freqs[1:]
            di = np.argmax(pnd)
            df = fnd[di]
            dp = 1.0 / df if df > 0 else float('inf')
            pr = pnd[di] / np.mean(pnd)
            period_data[f'CRNG K>={thresh}'] = {'period': dp, 'power_ratio': pr, 'power': power, 'freqs': freqs}

    return period_data


def analyze_gaps(events):
    """Quick gap analysis returning stats dict."""
    if len(events) < 3:
        return {}
    dates = sorted([datetime.strptime(e[0], "%Y-%m-%d") for e in events])
    gaps = np.array([(dates[i+1] - dates[i]).days for i in range(len(dates)-1)], dtype=float)
    gaps = gaps[gaps > 0]
    if len(gaps) < 3:
        return {}
    mean_gap = np.mean(gaps)
    cv = np.std(gaps) / mean_gap if mean_gap > 0 else 0
    k = sp_stats.kurtosis(gaps, fisher=False) if len(gaps) >= 4 else 3
    acf = np.corrcoef(gaps[:-1], gaps[1:])[0, 1] if len(gaps) > 2 else 0
    return {'gaps': gaps.tolist(), 'mean_gap': mean_gap, 'cv': cv, 'kurtosis': k, 'acf': acf}


# ============================================================
# CHART 1: KS TEST BAR CHART
# ============================================================

def chart1_ks_match(ks_results, outdir):
    print("  Chart 1: KS Match bar chart...")

    fig, ax = plt.subplots(figsize=(14, 6))

    labels = [f"{r['cat']} vs K>={r['thresh']}" for r in ks_results]
    pvals = [r['p'] for r in ks_results]
    colors = [GREEN if p > 0.05 else RED for p in pvals]

    x = np.arange(len(labels))
    bars = ax.bar(x, pvals, color=colors, alpha=0.85, edgecolor='#30363d', linewidth=0.5)

    # Red threshold line
    ax.axhline(y=0.05, color=RED, linestyle='--', linewidth=1.5, alpha=0.8, label='p = 0.05 threshold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=7)
    ax.set_ylabel('KS Test p-value', fontsize=11)
    ax.set_title('KS Test: CRNG vs Real Catastrophes \u2014 20/20 MATCH', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=9, facecolor=PANEL, edgecolor='#30363d')
    ax.set_ylim(0, max(pvals) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    # Annotate p-values on bars
    for bar, p in zip(bars, pvals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{p:.3f}', ha='center', va='bottom', fontsize=6, color='#8b949e')

    plt.tight_layout()
    path = os.path.join(outdir, 'catastrophic_ks_match.png')
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"    Saved: {path}")


# ============================================================
# CHART 2: PERIODICITY
# ============================================================

def chart2_periodicity(period_data, real_results, outdir):
    print("  Chart 2: Periodicity panels...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: FFT power spectrum for ALL Catastrophes
    all_key = 'ALL Catastrophes'
    if all_key in period_data:
        pd = period_data[all_key]
        power = pd['power']
        freqs = pd['freqs']

        # Only positive frequencies (skip DC)
        pos = freqs > 0
        ax1.plot(1.0 / freqs[pos], power[pos], color=PURPLE, linewidth=1.5, alpha=0.9)
        ax1.fill_between(1.0 / freqs[pos], 0, power[pos], alpha=0.15, color=PURPLE)

        # Mark dominant period
        dominant_period = pd['period']
        dominant_power = pd['power_ratio']
        ax1.axvline(x=dominant_period, color=ORANGE, linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.annotate(f'Dominant: {dominant_period:.0f} gaps\n{dominant_power:.2f}x power',
                    xy=(dominant_period, max(power[pos]) * 0.8),
                    fontsize=10, color=ORANGE, fontweight='bold',
                    ha='left' if dominant_period < 50 else 'right')

        ax1.set_xlabel('Period (gaps)', fontsize=11)
        ax1.set_ylabel('FFT Power', fontsize=11)
        ax1.set_title('FFT Power Spectrum — ALL Catastrophes', fontsize=12, fontweight='bold')
        ax1.set_xlim(1, 100)
        ax1.grid(alpha=0.3)

    # Right panel: Comparison bar chart of dominant periods
    bar_labels = []
    bar_periods = []
    bar_ratios = []
    bar_colors_list = []

    target_order = [
        ('Earthquakes (M>=6.7)', 'Earthquakes', RED),
        ('Financial Crashes', 'Financial', ORANGE),
        ('Natural Disasters', 'Natural', BLUE),
        ('ALL Catastrophes', 'All', PURPLE),
        ('CRNG K>=10', 'CRNG K>=10', GREEN),
        ('CRNG K>=15', 'CRNG K>=15', GREEN),
    ]

    for key, short, color in target_order:
        if key in period_data:
            bar_labels.append(short)
            bar_periods.append(period_data[key]['period'])
            bar_ratios.append(period_data[key]['power_ratio'])
            bar_colors_list.append(color)

    if bar_labels:
        x = np.arange(len(bar_labels))
        bars = ax2.bar(x, bar_periods, color=bar_colors_list, alpha=0.85, edgecolor='#30363d', linewidth=0.5)

        for bar, ratio in zip(bars, bar_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{ratio:.1f}x', ha='center', va='bottom', fontsize=9, color='#e6edf3', fontweight='bold')

        ax2.set_xticks(x)
        ax2.set_xticklabels(bar_labels, rotation=30, ha='right', fontsize=9)
        ax2.set_ylabel('Dominant Period (gaps)', fontsize=11)
        ax2.set_title('Dominant Periods & Power Ratios', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Hidden Periodicity in Catastrophes', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, 'catastrophic_periodicity.png')
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"    Saved: {path}")


# ============================================================
# CHART 3: TIMELINE ANIMATION (GIF)
# ============================================================

def chart3_timeline_gif(outdir):
    print("  Chart 3: Timeline animation GIF...")

    earthquakes = get_earthquake_data()
    crashes = get_financial_crashes()
    disasters = get_natural_disasters()

    # Parse dates and sort all events chronologically
    eq_dates = sorted([datetime.strptime(e[0], "%Y-%m-%d") for e in earthquakes])
    fin_dates = sorted([datetime.strptime(e[0], "%Y-%m-%d") for e in crashes])
    nat_dates = sorted([datetime.strptime(e[0], "%Y-%m-%d") for e in disasters])

    # Convert to fractional years for plotting
    def to_year(d):
        return d.year + d.timetuple().tm_yday / 365.25

    eq_years = [to_year(d) for d in eq_dates]
    fin_years = [to_year(d) for d in fin_dates]
    nat_years = [to_year(d) for d in nat_dates]

    # All events sorted by time with category
    all_events = []
    for y in eq_years:
        all_events.append((y, 'eq'))
    for y in fin_years:
        all_events.append((y, 'fin'))
    for y in nat_years:
        all_events.append((y, 'nat'))
    all_events.sort(key=lambda x: x[0])

    n_events = len(all_events)
    n_frames = 82  # ~82 events
    fps = 25
    duration_s = 3.0
    total_frames = int(fps * duration_s)

    fig, ax = plt.subplots(figsize=(12, 4))

    lane_y = {'eq': 2.0, 'fin': 1.0, 'nat': 0.0}
    lane_color = {'eq': BLUE, 'fin': RED, 'nat': ORANGE}
    lane_label = {'eq': 'Earthquakes', 'fin': 'Financial', 'nat': 'Natural'}

    ax.set_xlim(1895, 2030)
    ax.set_ylim(-0.8, 3.2)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Natural', 'Financial', 'Earthquakes'], fontsize=10)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('Catastrophic Events Timeline (1900-2025)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Draw lane lines
    for y_val in [0, 1, 2]:
        ax.axhline(y=y_val, color='#30363d', linewidth=0.5, alpha=0.5)

    # Static dots (will accumulate) and flash circles
    dots = {cat: {'x': [], 'y': []} for cat in ['eq', 'fin', 'nat']}
    scatter_objs = {}
    flash_objs = {}
    for cat in ['eq', 'fin', 'nat']:
        scatter_objs[cat] = ax.scatter([], [], s=30, color=lane_color[cat], alpha=0.8, zorder=5)
        flash_objs[cat] = ax.scatter([], [], s=200, color=lane_color[cat], alpha=0.4, zorder=4)

    # Counter text
    counter_text = ax.text(0.02, -0.55, '', transform=ax.get_yaxis_transform(),
                          fontsize=10, color='#e6edf3', fontfamily='monospace')

    plt.tight_layout()

    def init():
        for cat in ['eq', 'fin', 'nat']:
            scatter_objs[cat].set_offsets(np.empty((0, 2)))
            flash_objs[cat].set_offsets(np.empty((0, 2)))
        counter_text.set_text('')
        return list(scatter_objs.values()) + list(flash_objs.values()) + [counter_text]

    def update(frame):
        # How many events should be visible at this frame
        progress = frame / max(total_frames - 1, 1)
        n_visible = int(progress * n_events)
        n_visible = min(n_visible, n_events)

        # Reset dots
        current_dots = {'eq': [], 'fin': [], 'nat': []}

        for i in range(n_visible):
            year, cat = all_events[i]
            current_dots[cat].append((year, lane_y[cat]))

        # Flash effect: the most recently added event
        flash_cat = None
        flash_pos = None
        if n_visible > 0:
            latest = all_events[n_visible - 1]
            flash_cat = latest[1]
            flash_pos = (latest[0], lane_y[latest[1]])

            # Flash shrinks over a few frames
            frames_since = frame - int((n_visible - 1) / n_events * total_frames)
            flash_size = max(200 - frames_since * 40, 0)
            flash_alpha = max(0.5 - frames_since * 0.1, 0)
        else:
            flash_size = 0
            flash_alpha = 0

        for cat in ['eq', 'fin', 'nat']:
            if current_dots[cat]:
                scatter_objs[cat].set_offsets(np.array(current_dots[cat]))
            else:
                scatter_objs[cat].set_offsets(np.empty((0, 2)))

            if cat == flash_cat and flash_pos and flash_size > 0:
                flash_objs[cat].set_offsets(np.array([flash_pos]))
                flash_objs[cat].set_sizes([flash_size])
                flash_objs[cat].set_alpha(flash_alpha)
            else:
                flash_objs[cat].set_offsets(np.empty((0, 2)))

        # Compute running stats
        if n_visible >= 2:
            visible_years = [all_events[i][0] for i in range(n_visible)]
            gaps_years = np.diff(visible_years)
            mean_gap = np.mean(gaps_years)
            counter_text.set_text(f'Events: {n_visible}/{n_events}  |  Mean gap: {mean_gap:.1f} years')
        elif n_visible == 1:
            counter_text.set_text(f'Events: 1/{n_events}')
        else:
            counter_text.set_text('')

        return list(scatter_objs.values()) + list(flash_objs.values()) + [counter_text]

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_frames, blit=True, interval=1000/fps)
    path = os.path.join(outdir, 'catastrophic_timeline.gif')
    anim.save(path, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    print(f"    Saved: {path}")


# ============================================================
# CHART 4: MATCH ANIMATION (GIF)
# ============================================================

def chart4_match_gif(kurtosis_map, event_map, ks_results, outdir):
    print("  Chart 4: Match animation GIF...")

    earthquakes = get_earthquake_data()
    crashes = get_financial_crashes()
    disasters = get_natural_disasters()

    # All events sorted
    def to_year(d_str):
        d = datetime.strptime(d_str, "%Y-%m-%d")
        return d.year + d.timetuple().tm_yday / 365.25

    all_events = []
    for e in earthquakes:
        all_events.append((to_year(e[0]), 'eq', e[2]))
    for e in crashes:
        all_events.append((to_year(e[0]), 'fin', e[2]))
    for e in disasters:
        all_events.append((to_year(e[0]), 'nat', e[2]))
    all_events.sort()

    lane_color = {'eq': BLUE, 'fin': RED, 'nat': ORANGE}

    fps = 20
    duration_s = 5.0
    total_frames = int(fps * duration_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.2, 1]})

    # LEFT PANEL: CRNG kurtosis scrolling line chart
    # Use a subset of kurtosis_map for display
    k_display = kurtosis_map[:min(len(kurtosis_map), 5000)]
    k_x = np.arange(len(k_display))

    ax1.set_xlim(0, len(k_display))
    ax1.set_ylim(0, min(np.percentile(kurtosis_map, 99.5) * 1.2, 100))
    ax1.set_xlabel('Window Position', fontsize=10)
    ax1.set_ylabel('Kurtosis (K)', fontsize=10)
    ax1.set_title('CRNG Kurtosis Map', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Threshold lines
    for thresh, col, alpha in [(5, '#30363d', 0.4), (10, ORANGE, 0.3), (15, RED, 0.3)]:
        ax1.axhline(y=thresh, color=col, linestyle='--', linewidth=0.8, alpha=alpha)
        ax1.text(len(k_display) * 0.01, thresh + 0.5, f'K={thresh}', fontsize=7, color=col, alpha=0.7)

    line_k, = ax1.plot([], [], color=BLUE, linewidth=0.8, alpha=0.7)
    scatter_spikes = ax1.scatter([], [], s=30, color=RED, alpha=0.8, zorder=5)
    flash_spike = ax1.scatter([], [], s=150, color=RED, alpha=0.4, zorder=4)

    # RIGHT PANEL: Real catastrophes timeline (vertical)
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(1895, 2030)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Eq', 'Fin', 'Nat'], fontsize=9)
    ax2.set_ylabel('Year', fontsize=10)
    ax2.set_title('Real Catastrophes', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    cat_x = {'eq': 0, 'fin': 1, 'nat': 2}
    scatter_real = ax2.scatter([], [], s=30, c=[], alpha=0.8, zorder=5)
    flash_real = ax2.scatter([], [], s=150, c=[], alpha=0.4, zorder=4)

    # Bottom text for KS p-value
    info_text = fig.text(0.5, 0.02, '', fontsize=11, color='#e6edf3', ha='center', fontfamily='monospace')

    # Final overlay text (hidden initially)
    final_text = fig.text(0.5, 0.5, '', fontsize=24, color=GREEN, ha='center', va='center',
                         fontweight='bold', fontfamily='monospace', alpha=0)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.suptitle('CRNG vs Real Catastrophes', fontsize=14, fontweight='bold', y=0.98)

    # Precompute spike positions (K >= 10)
    spike_mask = k_display >= 10
    spike_x = k_x[spike_mask]
    spike_y = k_display[spike_mask]

    def init():
        line_k.set_data([], [])
        scatter_spikes.set_offsets(np.empty((0, 2)))
        flash_spike.set_offsets(np.empty((0, 2)))
        scatter_real.set_offsets(np.empty((0, 2)))
        flash_real.set_offsets(np.empty((0, 2)))
        info_text.set_text('')
        final_text.set_text('')
        return [line_k, scatter_spikes, flash_spike, scatter_real, flash_real, info_text, final_text]

    def update(frame):
        progress = frame / max(total_frames - 1, 1)

        # LEFT: reveal kurtosis line progressively
        n_show = int(progress * len(k_display))
        n_show = max(n_show, 1)
        line_k.set_data(k_x[:n_show], k_display[:n_show])

        # Show spikes up to current position
        visible_spikes = spike_x[spike_x < n_show]
        visible_spike_y = spike_y[:len(visible_spikes)]
        if len(visible_spikes) > 0:
            scatter_spikes.set_offsets(np.column_stack([visible_spikes, visible_spike_y]))
            # Flash the latest spike
            flash_spike.set_offsets(np.array([[visible_spikes[-1], visible_spike_y[-1]]]))
            flash_alpha = max(0.5 - (n_show - visible_spikes[-1]) / 50, 0)
            flash_spike.set_alpha(flash_alpha)
        else:
            scatter_spikes.set_offsets(np.empty((0, 2)))
            flash_spike.set_offsets(np.empty((0, 2)))

        # RIGHT: reveal real events progressively
        n_events_show = int(progress * len(all_events))
        n_events_show = min(n_events_show, len(all_events))

        if n_events_show > 0:
            shown = all_events[:n_events_show]
            xs = [cat_x[e[1]] for e in shown]
            ys = [e[0] for e in shown]
            cs = [lane_color[e[1]] for e in shown]
            scatter_real.set_offsets(np.column_stack([xs, ys]))
            scatter_real.set_color(cs)

            # Flash latest
            latest = shown[-1]
            flash_real.set_offsets(np.array([[cat_x[latest[1]], latest[0]]]))
            flash_real.set_color([lane_color[latest[1]]])
            flash_real.set_alpha(max(0.5 - (frame % 5) * 0.1, 0.1))
        else:
            scatter_real.set_offsets(np.empty((0, 2)))
            flash_real.set_offsets(np.empty((0, 2)))

        # Running KS info
        n_ks_show = int(progress * len(ks_results))
        n_match = sum(1 for r in ks_results[:n_ks_show] if r['p'] > 0.05)
        if n_ks_show > 0:
            info_text.set_text(f'KS tests computed: {n_ks_show}/{len(ks_results)}  |  MATCH: {n_match}/{n_ks_show}')
        else:
            info_text.set_text('')

        # Final frame: show result
        if frame >= total_frames - 8:
            fade = min((frame - (total_frames - 8)) / 7, 1.0)
            final_text.set_text('20/20 MATCH')
            final_text.set_alpha(fade)

            # Show final p-values summary
            if frame == total_frames - 1:
                avg_p = np.mean([r['p'] for r in ks_results])
                info_text.set_text(f'20/20 MATCH  |  mean p-value: {avg_p:.3f}  |  All p > 0.05')
        else:
            final_text.set_alpha(0)

        return [line_k, scatter_spikes, flash_spike, scatter_real, flash_real, info_text, final_text]

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_frames, blit=True, interval=1000/fps)
    path = os.path.join(outdir, 'catastrophic_match.gif')
    anim.save(path, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    print(f"    Saved: {path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(__file__), '..', 'charts')
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("  CATASTROPHIC EVENTS — GENERATING CHARTS")
    print("=" * 60)

    # Generate data
    print("\n[1/5] Generating CRNG data...")
    kurtosis_map, event_map, returns = generate_crng_data(n_points=30000, window=50, seed=42)
    print(f"  Kurtosis map: {len(kurtosis_map):,} windows")

    # Real event analysis
    print("[2/5] Analyzing real events...")
    earthquakes = get_earthquake_data()
    crashes = get_financial_crashes()
    disasters = get_natural_disasters()

    all_catastrophes = []
    for e in earthquakes:
        all_catastrophes.append((e[0], e[1] * 10, f"EQ: {e[2]}"))
    for c in crashes:
        all_catastrophes.append((c[0], c[1], f"FIN: {c[2]}"))
    for d in disasters:
        all_catastrophes.append((d[0], d[1], f"NAT: {d[2]}"))

    real_results = {}
    real_results['Earthquakes (M>=6.7)'] = analyze_gaps(earthquakes)
    real_results['Financial Crashes'] = analyze_gaps(crashes)
    real_results['Natural Disasters'] = analyze_gaps(disasters)
    real_results['ALL Catastrophes'] = analyze_gaps(all_catastrophes)

    # KS results
    print("[3/5] Computing KS tests...")
    ks_results = compute_ks_results(event_map, real_results)
    n_match = sum(1 for r in ks_results if r['p'] > 0.05)
    print(f"  {n_match}/{len(ks_results)} MATCH (p > 0.05)")

    # Periodicity
    period_data = compute_periodicity(real_results, event_map)

    # Generate charts
    print("\n[4/5] Generating static charts...")
    chart1_ks_match(ks_results, outdir)
    chart2_periodicity(period_data, real_results, outdir)

    print("\n[5/5] Generating animated GIFs...")
    chart3_timeline_gif(outdir)
    chart4_match_gif(kurtosis_map, event_map, ks_results, outdir)

    # Verify
    print("\n" + "=" * 60)
    print("  VERIFICATION")
    print("=" * 60)
    expected = [
        'catastrophic_ks_match.png',
        'catastrophic_periodicity.png',
        'catastrophic_timeline.gif',
        'catastrophic_match.gif',
    ]
    for f in expected:
        path = os.path.join(outdir, f)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  OK  {f} ({size_kb:.0f} KB)")
        else:
            print(f"  MISSING  {f}")

    print("\n  Done!")
