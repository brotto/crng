"""
ANFANG EXPERIMENT — CHARTS AND ANIMATIONS
============================================

Visualizations for the universal field experiment:
1. Leave-one-out results (3/3 MATCH) — hero chart
2. Pairwise gap distributions overlay — the visual proof
3. Temporal stability (1900-1970 vs 1970-2025)
4. Universal field predictor timeline (animated GIF)
5. The Anfang spiral (animated GIF) — philosophical visualization

Ale Brotto — 2026-04-02
"""

import numpy as np
from scipy import stats as sp_stats
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ============================================================
# EVENT DATA (same as novel_event_predictor.py)
# ============================================================

EARTHQUAKES = [
    "1906-04-18", "1908-12-28", "1920-12-16", "1923-09-01", "1927-05-22",
    "1931-08-10", "1934-01-15", "1935-05-30", "1939-12-26", "1944-12-07",
    "1946-04-01", "1948-10-05", "1950-08-15", "1952-11-04", "1957-03-09",
    "1960-05-22", "1964-03-27", "1970-05-31", "1976-07-27", "1985-09-19",
    "1988-12-07", "1989-10-17", "1994-01-17", "1995-01-17", "1999-08-17",
    "1999-09-20", "2001-01-26", "2003-12-26", "2004-12-26", "2005-10-08",
    "2008-05-12", "2010-01-12", "2010-02-27", "2011-03-11", "2015-04-25",
    "2017-09-19", "2018-09-28", "2023-02-06", "2024-01-01",
]

CRASHES = [
    "1929-10-29", "1937-03-10", "1962-05-28", "1973-01-11", "1979-10-06",
    "1987-10-19", "1989-10-13", "1997-10-27", "1998-08-17", "2000-03-10",
    "2001-09-17", "2007-02-27", "2008-09-29", "2010-05-06", "2011-08-05",
    "2015-08-24", "2018-02-05", "2020-03-16", "2022-06-13",
]

DISASTERS = [
    "1900-09-08", "1918-01-01", "1931-08-01", "1935-09-02", "1938-09-21",
    "1942-10-16", "1953-02-01", "1959-09-26", "1965-05-11", "1970-11-12",
    "1975-08-05", "1984-12-03", "1986-04-26", "1991-04-29", "1998-10-29",
    "2003-08-01", "2004-12-26", "2005-08-29", "2008-05-02", "2010-01-12",
    "2011-03-11", "2013-11-08", "2019-12-01", "2022-06-01",
]

COLORS = {
    'eq': '#E74C3C',      # red
    'fin': '#3498DB',     # blue
    'nat': '#2ECC71',     # green
    'all': '#F39C12',     # orange
    'bg': '#0D1117',      # dark bg
    'text': '#E6EDF3',    # light text
    'grid': '#21262D',    # subtle grid
    'accent': '#FFD700',  # gold
}


def parse_dates(date_strings):
    return sorted([datetime.strptime(d, "%Y-%m-%d") for d in date_strings])


def get_gaps(dates):
    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    return np.array([g for g in gaps if g > 0], dtype=float)


def normalize_gaps(gaps):
    return (gaps - np.mean(gaps)) / (np.std(gaps) + 1e-10)


# ============================================================
# CHART 1: LEAVE-ONE-OUT HERO CHART
# ============================================================

def chart_leave_one_out():
    """3/3 MATCH visual with p-values and confidence."""

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    categories = ['Earthquakes\nexcluded', 'Financial Crashes\nexcluded', 'Natural Disasters\nexcluded']
    trained_on = ['Crashes + Disasters', 'Earthquakes + Disasters', 'Earthquakes + Crashes']
    p_values = [0.174, 0.440, 0.901]
    colors = [COLORS['eq'], COLORS['fin'], COLORS['nat']]

    bars = ax.barh(range(3), p_values, height=0.5, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5)

    # Significance threshold
    ax.axvline(x=0.05, color=COLORS['accent'], linestyle='--', linewidth=2, alpha=0.8)
    ax.text(0.05, 2.7, 'α = 0.05', color=COLORS['accent'], fontsize=11,
            ha='center', fontweight='bold')

    # Labels
    for i, (p, cat, train) in enumerate(zip(p_values, categories, trained_on)):
        ax.text(p + 0.02, i, f'p = {p:.3f}  MATCH ✓', va='center',
                color=COLORS['text'], fontsize=14, fontweight='bold')
        ax.text(-0.01, i + 0.22, f'trained on: {train}', va='center',
                color=COLORS['text'], fontsize=9, alpha=0.6, ha='right')

    ax.set_yticks(range(3))
    ax.set_yticklabels(categories, fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_xlim(-0.01, 1.05)
    ax.set_xlabel('KS Test p-value', fontsize=12, color=COLORS['text'])

    ax.set_title('Leave-One-Category-Out: Can 2 Categories Predict the 3rd?\n',
                 fontsize=16, color=COLORS['text'], fontweight='bold')

    subtitle = ('"Der Anfang ist das, was zuletzt kommt." — Heidegger\n'
                'The beginning is what comes last.')
    ax.text(0.5, -0.18, subtitle, transform=ax.transAxes, ha='center',
            fontsize=11, color=COLORS['accent'], style='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.spines['left'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    ax.xaxis.label.set_color(COLORS['text'])

    plt.tight_layout()
    plt.savefig('charts/anfang_leave_one_out.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ anfang_leave_one_out.png")


# ============================================================
# CHART 2: GAP DISTRIBUTIONS OVERLAY
# ============================================================

def chart_gap_overlay():
    """Normalized gap distributions for all 3 categories — visual proof of universality."""

    eq_gaps = normalize_gaps(get_gaps(parse_dates(EARTHQUAKES)))
    fin_gaps = normalize_gaps(get_gaps(parse_dates(CRASHES)))
    nat_gaps = normalize_gaps(get_gaps(parse_dates(DISASTERS)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['bg'])

    # Left: Histogram overlay
    ax = axes[0]
    ax.set_facecolor(COLORS['bg'])

    bins = np.linspace(-2, 5, 30)
    ax.hist(eq_gaps, bins=bins, alpha=0.5, color=COLORS['eq'], label='Earthquakes',
            density=True, edgecolor='none')
    ax.hist(fin_gaps, bins=bins, alpha=0.5, color=COLORS['fin'], label='Financial Crashes',
            density=True, edgecolor='none')
    ax.hist(nat_gaps, bins=bins, alpha=0.5, color=COLORS['nat'], label='Natural Disasters',
            density=True, edgecolor='none')

    ax.set_xlabel('Normalized Gap (σ)', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Density', fontsize=12, color=COLORS['text'])
    ax.set_title('Gap Distributions Overlap', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.legend(fontsize=10, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])

    # Right: CDF comparison (Q-Q style)
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['bg'])

    for gaps, color, label in [(eq_gaps, COLORS['eq'], 'Earthquakes'),
                                (fin_gaps, COLORS['fin'], 'Crashes'),
                                (nat_gaps, COLORS['nat'], 'Disasters')]:
        sorted_gaps = np.sort(gaps)
        cdf = np.arange(1, len(sorted_gaps)+1) / len(sorted_gaps)
        ax2.plot(sorted_gaps, cdf, color=color, linewidth=2.5, label=label, alpha=0.85)

    ax2.set_xlabel('Normalized Gap (σ)', fontsize=12, color=COLORS['text'])
    ax2.set_ylabel('Cumulative Probability', fontsize=12, color=COLORS['text'])
    ax2.set_title('Empirical CDFs', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax2.legend(fontsize=10, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
               labelcolor=COLORS['text'])

    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])

    fig.suptitle('Three Radically Different Phenomena — One Temporal Geometry\n',
                 fontsize=16, color=COLORS['accent'], fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('charts/anfang_gap_overlay.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ anfang_gap_overlay.png")


# ============================================================
# CHART 3: TEMPORAL STABILITY
# ============================================================

def chart_temporal_stability():
    """1900-1970 vs 1970-2025: same shape, different tempo."""

    all_dates = parse_dates(EARTHQUAKES + CRASHES + DISASTERS)
    all_gaps = get_gaps(all_dates)

    cutoff = datetime(1970, 1, 1)
    early_dates = [d for d in all_dates if d < cutoff]
    late_dates = [d for d in all_dates if d >= cutoff]

    early_gaps = normalize_gaps(get_gaps(early_dates))
    late_gaps = normalize_gaps(get_gaps(late_dates))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['bg'])

    # Left: CDF comparison
    ax1.set_facecolor(COLORS['bg'])

    for gaps, color, label in [(early_gaps, '#E74C3C', '1900–1970 (28 gaps)'),
                                (late_gaps, '#3498DB', '1970–2025 (49 gaps)')]:
        sorted_g = np.sort(gaps)
        cdf = np.arange(1, len(sorted_g)+1) / len(sorted_g)
        ax1.plot(sorted_g, cdf, color=color, linewidth=3, label=label)

    ax1.set_xlabel('Normalized Gap (σ)', fontsize=12, color=COLORS['text'])
    ax1.set_ylabel('Cumulative Probability', fontsize=12, color=COLORS['text'])
    ax1.set_title('Same Shape Across 125 Years\nKS p = 0.777 → MATCH',
                  fontsize=14, color=COLORS['text'], fontweight='bold')
    ax1.legend(fontsize=11, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
               labelcolor=COLORS['text'])

    for spine in ax1.spines.values():
        spine.set_color(COLORS['grid'])
    ax1.tick_params(colors=COLORS['text'])

    # Right: Acceleration visual
    ax2.set_facecolor(COLORS['bg'])

    early_raw = get_gaps([d for d in all_dates if d < cutoff])
    late_raw = get_gaps([d for d in all_dates if d >= cutoff])

    ax2.bar(0, np.mean(early_raw), width=0.6, color='#E74C3C', alpha=0.8,
            edgecolor='white', linewidth=1.5)
    ax2.bar(1, np.mean(late_raw), width=0.6, color='#3498DB', alpha=0.8,
            edgecolor='white', linewidth=1.5)

    ax2.text(0, np.mean(early_raw) + 30, f'{np.mean(early_raw):.0f}d\n({np.mean(early_raw)/365.25:.1f}y)',
             ha='center', color=COLORS['text'], fontsize=13, fontweight='bold')
    ax2.text(1, np.mean(late_raw) + 30, f'{np.mean(late_raw):.0f}d\n({np.mean(late_raw)/365.25:.1f}y)',
             ha='center', color=COLORS['text'], fontsize=13, fontweight='bold')

    # Arrow showing 2.96x
    ax2.annotate('', xy=(1, np.mean(early_raw)*0.7), xytext=(0, np.mean(early_raw)*0.7),
                 arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2.5))
    ax2.text(0.5, np.mean(early_raw)*0.75, '2.96× faster',
             ha='center', color=COLORS['accent'], fontsize=13, fontweight='bold')

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['1900–1970', '1970–2025'], fontsize=12, color=COLORS['text'])
    ax2.set_ylabel('Mean Gap (days)', fontsize=12, color=COLORS['text'])
    ax2.set_title('Tempo Changed, Form Persists',
                  fontsize=14, color=COLORS['text'], fontweight='bold')

    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    plt.savefig('charts/anfang_temporal_stability.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ anfang_temporal_stability.png")


# ============================================================
# CHART 4: UNIVERSAL FIELD TIMELINE (ANIMATED GIF)
# ============================================================

def chart_universal_timeline_gif():
    """Animated timeline showing all 82 events converging into one field."""

    import matplotlib.animation as animation

    all_events = []
    for d in EARTHQUAKES:
        all_events.append((datetime.strptime(d, "%Y-%m-%d"), 'eq'))
    for d in CRASHES:
        all_events.append((datetime.strptime(d, "%Y-%m-%d"), 'fin'))
    for d in DISASTERS:
        all_events.append((datetime.strptime(d, "%Y-%m-%d"), 'nat'))
    all_events.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # Fixed axes
    min_year = 1898
    max_year = 2028
    ax.set_xlim(min_year, max_year)
    ax.set_ylim(-1.5, 3.5)

    # Category y-positions
    y_pos = {'eq': 2, 'fin': 1, 'nat': 0}
    cat_labels = {'eq': 'Earthquakes', 'fin': 'Financial Crashes', 'nat': 'Natural Disasters'}
    cat_colors = {'eq': COLORS['eq'], 'fin': COLORS['fin'], 'nat': COLORS['nat']}

    for cat, y in y_pos.items():
        ax.axhline(y=y, color=COLORS['grid'], linewidth=0.5, alpha=0.5)
        ax.text(min_year + 0.5, y + 0.25, cat_labels[cat], color=cat_colors[cat],
                fontsize=10, fontweight='bold', alpha=0.8)

    ax.set_yticks([])
    ax.set_xlabel('Year', fontsize=12, color=COLORS['text'])

    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])

    title = ax.set_title('', fontsize=14, color=COLORS['text'], fontweight='bold')
    counter = ax.text(0.98, 0.95, '', transform=ax.transAxes, ha='right', va='top',
                      fontsize=20, color=COLORS['accent'], fontweight='bold')

    # Bottom text for final frames
    bottom_text = ax.text(0.5, -0.2, '', transform=ax.transAxes, ha='center',
                          fontsize=13, color=COLORS['accent'], style='italic')

    dots = []

    n_events = len(all_events)
    # Pause frames at end
    total_frames = n_events + 30

    def animate(frame):
        if frame < n_events:
            date, cat = all_events[frame]
            year = date.year + date.month/12
            y = y_pos[cat]
            dot = ax.plot(year, y, 'o', color=cat_colors[cat], markersize=8,
                         alpha=0.85, markeredgecolor='white', markeredgewidth=0.5)[0]
            dots.append(dot)
            title.set_text(f'82 Catastrophic Events — One Temporal Geometry')
            counter.set_text(f'{frame+1}/82')

            if frame == n_events - 1:
                # Draw connecting "field" lines
                bottom_text.set_text('"Der Anfang ist das, was zuletzt kommt." — Heidegger')

        elif frame == n_events + 10:
            # Flash "3/3 MATCH"
            ax.text(0.5, -0.35, 'Leave-One-Out: 3/3 MATCH  |  Pairwise: 3/3 MATCH  |  7/10 Total',
                    transform=ax.transAxes, ha='center', fontsize=12,
                    color=COLORS['text'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['accent'],
                              alpha=0.15, edgecolor=COLORS['accent']))

        return dots

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=80, blit=False)
    anim.save('charts/anfang_universal_field.gif', writer='pillow', fps=12,
              savefig_kwargs={'facecolor': COLORS['bg']})
    plt.close()
    print("  ✓ anfang_universal_field.gif")


# ============================================================
# CHART 5: SCORECARD SUMMARY
# ============================================================

def chart_scorecard():
    """Final scorecard: 7/10 MATCH — clean summary visual."""

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    tests = [
        ('Leave-One-Out:\nEarthquakes', True, 0.174),
        ('Leave-One-Out:\nCrashes', True, 0.440),
        ('Leave-One-Out:\nDisasters', True, 0.901),
        ('Pairwise:\nEQ vs FIN', True, 0.314),
        ('Pairwise:\nEQ vs NAT', True, 0.986),
        ('Pairwise:\nFIN vs NAT', True, 0.531),
        ('CRNG vs\nEQ', False, 0.0001),
        ('CRNG vs\nFIN', False, 0.0000),
        ('CRNG vs\nNAT', False, 0.0014),
        ('Temporal:\n1900→2025', True, 0.777),
    ]

    n = len(tests)
    cols = 5
    rows = 2

    for i, (label, match, p) in enumerate(tests):
        row = i // cols
        col = i % cols

        x = col * 2 + 1
        y = (1 - row) * 2.5 + 0.5

        color = '#2ECC71' if match else '#E74C3C'
        symbol = '✓' if match else '✗'

        # Circle
        circle = plt.Circle((x, y), 0.7, facecolor=color, alpha=0.2,
                            edgecolor=color, linewidth=2.5)
        ax.add_patch(circle)

        # Symbol
        ax.text(x, y + 0.1, symbol, ha='center', va='center',
                fontsize=28, color=color, fontweight='bold')

        # Label below
        ax.text(x, y - 1.1, label, ha='center', va='center',
                fontsize=8, color=COLORS['text'], fontweight='bold')

        # p-value
        ax.text(x, y - 0.35, f'p={p:.3f}', ha='center', va='center',
                fontsize=7, color=COLORS['text'], alpha=0.6)

    ax.set_xlim(-0.5, cols * 2 + 0.5)
    ax.set_ylim(-1.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(cols, 4.3, 'THE FIELD IS UNIVERSAL', ha='center',
            fontsize=18, color=COLORS['accent'], fontweight='bold')
    ax.text(cols, 3.7, '7/10 MATCH — The temporal geometry is category-independent',
            ha='center', fontsize=11, color=COLORS['text'])

    plt.tight_layout()
    plt.savefig('charts/anfang_scorecard.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ anfang_scorecard.png")


# ============================================================
# CHART 6: THE ANFANG SPIRAL (ANIMATED GIF)
# ============================================================

def chart_anfang_spiral():
    """
    Philosophical visualization: events from 3 categories spiraling
    into a single convergent point — the universal field.
    """

    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=COLORS['bg'], subplot_kw={'projection': 'polar'})
    ax.set_facecolor(COLORS['bg'])

    all_events = []
    for d in EARTHQUAKES:
        dt = datetime.strptime(d, "%Y-%m-%d")
        all_events.append((dt, 'eq'))
    for d in CRASHES:
        dt = datetime.strptime(d, "%Y-%m-%d")
        all_events.append((dt, 'fin'))
    for d in DISASTERS:
        dt = datetime.strptime(d, "%Y-%m-%d")
        all_events.append((dt, 'nat'))
    all_events.sort(key=lambda x: x[0])

    cat_colors = {'eq': COLORS['eq'], 'fin': COLORS['fin'], 'nat': COLORS['nat']}

    # Map dates to spiral coordinates
    min_date = all_events[0][0]
    max_date = all_events[-1][0]
    total_days = (max_date - min_date).days

    n_events = len(all_events)
    total_frames = n_events + 40

    ax.set_ylim(0, 1.1)
    ax.grid(True, color=COLORS['grid'], alpha=0.3)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_color(COLORS['grid'])

    title = fig.suptitle('', fontsize=14, color=COLORS['text'],
                         fontweight='bold', y=0.95)
    subtitle = fig.text(0.5, 0.02, '', ha='center', fontsize=11,
                        color=COLORS['accent'], style='italic')

    dots = []

    def animate(frame):
        if frame < n_events:
            date, cat = all_events[frame]
            # Spiral: angle = time progression * 6 full turns
            progress = (date - min_date).days / total_days
            theta = progress * 6 * 2 * np.pi
            # Radius: starts outer (1.0), spirals inward to center (0.1)
            r = 1.0 - progress * 0.85

            dot = ax.plot(theta, r, 'o', color=cat_colors[cat], markersize=7,
                         alpha=0.8, markeredgecolor='white', markeredgewidth=0.3)[0]
            dots.append(dot)

            year = date.year
            title.set_text(f'The Anfang Spiral — {year}')

        elif frame == n_events:
            title.set_text('The Anfang Spiral — All Events Converge')
            subtitle.set_text('"Der Anfang ist das, was zuletzt kommt."')

            # Center point — the universal field
            ax.plot(0, 0, '*', color=COLORS['accent'], markersize=25,
                   markeredgecolor='white', markeredgewidth=1)

        elif frame == n_events + 15:
            # Add legend
            fig.text(0.15, 0.88, '● Earthquakes', color=COLORS['eq'], fontsize=10)
            fig.text(0.15, 0.85, '● Financial Crashes', color=COLORS['fin'], fontsize=10)
            fig.text(0.15, 0.82, '● Natural Disasters', color=COLORS['nat'], fontsize=10)
            fig.text(0.15, 0.79, '★ Universal Field', color=COLORS['accent'], fontsize=10)

        return dots

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=60, blit=False)
    anim.save('charts/anfang_spiral.gif', writer='pillow', fps=15,
              savefig_kwargs={'facecolor': COLORS['bg']})
    plt.close()
    print("  ✓ anfang_spiral.gif")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    os.makedirs('charts', exist_ok=True)

    print("=" * 50)
    print("  GENERATING ANFANG CHARTS")
    print("=" * 50)

    print("\n  1. Leave-One-Out hero chart...")
    chart_leave_one_out()

    print("\n  2. Gap distributions overlay...")
    chart_gap_overlay()

    print("\n  3. Temporal stability...")
    chart_temporal_stability()

    print("\n  4. Scorecard...")
    chart_scorecard()

    print("\n  5. Universal field timeline (GIF)...")
    chart_universal_timeline_gif()

    print("\n  6. Anfang spiral (GIF)...")
    chart_anfang_spiral()

    print("\n" + "=" * 50)
    print("  ALL CHARTS GENERATED")
    print("=" * 50)
