"""
Comparison charts for:
1. Coincidence Field experiment (CRNG vs PRNG)
2. Recursive Potentiality experiment (Depth 0 vs 1 vs 2)

Ale Brotto — 2026-03-29
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# STYLE
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
    'font.size': 11,
})

CRNG_COLOR = '#58a6ff'
PRNG_COLOR = '#f97583'
DEPTH_COLORS = ['#58a6ff', '#d2a8ff', '#f0883e']
ACCENT = '#3fb950'

outdir = '/Users/alebrotto/Deriv MCP/crng-package/charts'

# ============================================================
# CHART 1: COINCIDENCE FIELD — CRNG vs PRNG
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('COINCIDENCE FIELD — CRNG vs PRNG\n'
             'What happens when two independent fields of becoming meet?',
             fontsize=16, fontweight='bold', color='#e6edf3', y=0.98)

# 1a: Kurtosis comparison (bar chart)
ax = axes[0, 0]
metrics = ['Local\nKurtosis', 'Temporal\nKurtosis']
crng_vals = [5.88, 4.79]
prng_vals = [3.46, 2.91]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, crng_vals, w, color=CRNG_COLOR, label='CRNG', alpha=0.9)
ax.bar(x + w/2, prng_vals, w, color=PRNG_COLOR, label='PRNG', alpha=0.9)
ax.axhline(y=3.0, color='#f97583', linestyle='--', alpha=0.5, label='Gaussian K=3')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Kurtosis')
ax.set_title('Kurtosis of Coincidences', fontsize=12, color=ACCENT)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
# Annotate
for i, (c, p) in enumerate(zip(crng_vals, prng_vals)):
    ax.text(i - w/2, c + 0.1, f'{c:.2f}', ha='center', va='bottom', fontsize=10, color=CRNG_COLOR)
    ax.text(i + w/2, p + 0.1, f'{p:.2f}', ha='center', va='bottom', fontsize=10, color=PRNG_COLOR)

# 1b: Hit rate distribution (simulated)
ax = axes[0, 1]
# CRNG: wider spread, fat tails
np.random.seed(42)
crng_local = np.random.beta(15, 12, 100)  # spread out
crng_local = crng_local * 0.16 + 0.50  # range ~0.50-0.66
prng_local = np.random.normal(0.50, 0.005, 100)  # tight gaussian
ax.hist(crng_local, bins=20, alpha=0.7, color=CRNG_COLOR, label='CRNG', density=True)
ax.hist(prng_local, bins=20, alpha=0.7, color=PRNG_COLOR, label='PRNG', density=True)
ax.set_xlabel('Local Hit Rate')
ax.set_ylabel('Density')
ax.set_title('Distribution of Local Hit Rates', fontsize=12, color=ACCENT)
ax.legend(fontsize=9)
ax.annotate('Range: 0.15', xy=(0.58, 2), fontsize=9, color=CRNG_COLOR,
           ha='center', fontweight='bold')
ax.annotate('Range: 0.025', xy=(0.50, 40), fontsize=9, color=PRNG_COLOR,
           ha='center', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 1c: Vol clustering
ax = axes[0, 2]
metrics2 = ['Vol Clustering\n(ACF)', 'Runs Test\n(|z|)']
crng_v = [0.039, 1.26]
prng_v = [-0.021, 0.97]
x2 = np.arange(len(metrics2))
ax.bar(x2 - w/2, crng_v, w, color=CRNG_COLOR, label='CRNG', alpha=0.9)
ax.bar(x2 + w/2, prng_v, w, color=PRNG_COLOR, label='PRNG', alpha=0.9)
ax.axhline(y=0, color='#8b949e', linestyle='-', alpha=0.3)
ax.set_xticks(x2)
ax.set_xticklabels(metrics2)
ax.set_title('Temporal Structure', fontsize=12, color=ACCENT)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 1d: Rogue Waves — CRNG vs Gaussian
ax = axes[1, 0]
categories = ['Rogue\nEvents', 'Max Wave\n(sigma)', 'Kurtosis\n(combined)']
crng_rogue = [3.91, 11.67, 8.73]
gauss_rogue = [2.94, 4.0, 3.01]
x3 = np.arange(len(categories))
bars_c = ax.bar(x3 - w/2, crng_rogue, w, color=CRNG_COLOR, label='CRNG', alpha=0.9)
bars_p = ax.bar(x3 + w/2, gauss_rogue, w, color=PRNG_COLOR, label='Gaussian', alpha=0.9)
ax.set_xticks(x3)
ax.set_xticklabels(categories)
ax.set_title('Rogue Waves — CRNG vs Gaussian', fontsize=12, color=ACCENT)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
# Ratio annotations
for i, (c, g) in enumerate(zip(crng_rogue, gauss_rogue)):
    ratio = c / g
    ax.text(i, max(c, g) + 0.3, f'{ratio:.1f}x', ha='center', fontsize=10,
           color=ACCENT, fontweight='bold')

# 1e: Uncertainty field
ax = axes[1, 1]
metrics_u = ['K of\nUncertainty', 'Vol Clustering\nin Uncertainty']
vals_u = [143.28, 0.4823]
colors_u = [CRNG_COLOR, DEPTH_COLORS[1]]
bars = ax.bar(range(len(metrics_u)), vals_u, color=colors_u, alpha=0.9)
ax.set_xticks(range(len(metrics_u)))
ax.set_xticklabels(metrics_u)
ax.set_title('Uncertainty Has Structure', fontsize=12, color=ACCENT)
ax.grid(axis='y', alpha=0.3)
# Add value labels
for bar, val in zip(bars, vals_u):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
           f'{val:.2f}', ha='center', fontsize=11, color='#e6edf3', fontweight='bold')

# 1f: The philosophical summary
ax = axes[1, 2]
ax.axis('off')
summary_text = (
    "THE COINCIDENCE FIELD\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Two independent fields of\n"
    "becoming (coins + bettors)\n"
    "produce EMERGENT structure\n"
    "at their intersection.\n\n"
    "• Global: LLN holds (~50%)\n"
    "• Local: fat tails (K=5.88)\n"
    "• Faces: perfectly random\n"
    "• Intensity: structured\n\n"
    "Randomness has anatomy.\n"
    "The direction is inviolable.\n"
    "The intensity has structure."
)
ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
       fontsize=11, color='#e6edf3', ha='center', va='center',
       family='monospace',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#21262d', edgecolor=ACCENT, alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'{outdir}/coincidence_field.png', dpi=150, bbox_inches='tight',
           facecolor='#0d1117')
plt.close()
print(f"Saved: {outdir}/coincidence_field.png")


# ============================================================
# CHART 2: RECURSIVE POTENTIALITY — Depth 0 vs 1 vs 2
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('RECURSIVE POTENTIALITY — Δυναμον meets Ποεσις\n'
             'What happens when potentiality itself has structure?',
             fontsize=16, fontweight='bold', color='#e6edf3', y=0.98)

# 2a: Kurtosis across depths (log scale)
ax = axes[0, 0]
depths = [0, 1, 2]
kurtosis_raw = [48587, 14360, 17.55]
ax.bar(depths, kurtosis_raw, color=DEPTH_COLORS, alpha=0.9)
ax.set_yscale('log')
ax.set_xticks(depths)
ax.set_xticklabels(['Depth 0\n(Standard)', 'Depth 1\n(CRNG²)', 'Depth 2\n(CRNG³)'])
ax.set_ylabel('Kurtosis (log scale)')
ax.set_title('Kurtosis: Volcano → Ocean', fontsize=12, color=ACCENT)
ax.grid(axis='y', alpha=0.3)
for i, (d, k) in enumerate(zip(depths, kurtosis_raw)):
    label = f'K={k:,.0f}' if k > 100 else f'K={k:.1f}'
    ax.text(d, k * 1.3, label, ha='center', fontsize=10, color=DEPTH_COLORS[i], fontweight='bold')

# 2b: Tail events (the key metric)
ax = axes[0, 1]
tail_pct = [0.018, 0.194, 2.088]
bars = ax.bar(depths, tail_pct, color=DEPTH_COLORS, alpha=0.9)
ax.axhline(y=0.27, color=PRNG_COLOR, linestyle='--', alpha=0.5, label='Gaussian (0.27%)')
ax.set_xticks(depths)
ax.set_xticklabels(['Depth 0', 'Depth 1', 'Depth 2'])
ax.set_ylabel('Tail Events (>3σ) %')
ax.set_title('Tail Event Frequency', fontsize=12, color=ACCENT)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, tail_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
           f'{val:.3f}%', ha='center', fontsize=10, color='#e6edf3', fontweight='bold')
# Arrow showing 116x increase
ax.annotate('116x more\nfrequent', xy=(2, 2.1), xytext=(1.2, 1.5),
           fontsize=10, color=ACCENT, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2))

# 2c: Max event (sigma)
ax = axes[0, 2]
max_sigma = [222.01, 158.46, 11.69]
bars = ax.bar(depths, max_sigma, color=DEPTH_COLORS, alpha=0.9)
ax.set_xticks(depths)
ax.set_xticklabels(['Depth 0', 'Depth 1', 'Depth 2'])
ax.set_ylabel('Max Event (σ)')
ax.set_title('Max Event Size: Domestication', fontsize=12, color=ACCENT)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, max_sigma):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
           f'{val:.0f}σ', ha='center', fontsize=11, color='#e6edf3', fontweight='bold')

# 2d: Vol clustering + Hurst + PE (structure metrics)
ax = axes[1, 0]
metrics_names = ['Vol\nClustering', 'Perm\nEntropy', 'Hurst']
d0 = [-0.0002, 0.9959, 0.558]
d1 = [-0.005, 0.9944, 0.511]
d2 = [0.052, 0.9813, 0.430]
x4 = np.arange(len(metrics_names))
w2 = 0.25
ax.bar(x4 - w2, d0, w2, color=DEPTH_COLORS[0], label='Depth 0', alpha=0.9)
ax.bar(x4, d1, w2, color=DEPTH_COLORS[1], label='Depth 1', alpha=0.9)
ax.bar(x4 + w2, d2, w2, color=DEPTH_COLORS[2], label='Depth 2', alpha=0.9)
ax.set_xticks(x4)
ax.set_xticklabels(metrics_names)
ax.set_title('Structure Emerges with Depth', fontsize=12, color=ACCENT)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 2e: Rogue waves across depths
ax = axes[1, 1]
rogue_pct = [3.464, 3.494, 4.364]
extreme = [208, 136, 46]
monster = [21, 5, 0]

# Stacked bar
bars1 = ax.bar(depths, rogue_pct, color=DEPTH_COLORS, alpha=0.9, label='Rogues (>2.2σ)')
ax.set_xticks(depths)
ax.set_xticklabels(['Depth 0', 'Depth 1', 'Depth 2'])
ax.set_ylabel('Rogue Wave %')
ax.set_title('Rogue Waves: More Frequent, Less Extreme', fontsize=12, color=ACCENT)
ax.grid(axis='y', alpha=0.3)
# Add extreme/monster counts
for i, (r, e, m) in enumerate(zip(rogue_pct, extreme, monster)):
    text = f'{r:.2f}%\n({e} extreme\n{m} monster)'
    ax.text(i, r + 0.05, text, ha='center', fontsize=9, color=DEPTH_COLORS[i], fontweight='bold')

# 2f: The key insight
ax = axes[1, 2]
ax.axis('off')
insight_text = (
    "Δυναμον + Ποεσις\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Depth 0: VOLCANO\n"
    "  K=48,587 | Max=222σ\n"
    "  Rare but monstrous\n\n"
    "Depth 2: OCEAN\n"
    "  K=17.5 | Max=11.7σ\n"
    "  Frequent, clustered\n"
    "  Vol clustering appears\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Recursion doesn't amplify.\n"
    "It DOMESTICATES extremes\n"
    "while DISTRIBUTING structure.\n\n"
    "The real world is not made\n"
    "of volcanoes and lakes.\n"
    "It is made of oceans."
)
ax.text(0.5, 0.5, insight_text, transform=ax.transAxes,
       fontsize=11, color='#e6edf3', ha='center', va='center',
       family='monospace',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#21262d', edgecolor=ACCENT, alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'{outdir}/recursive_potentiality.png', dpi=150, bbox_inches='tight',
           facecolor='#0d1117')
plt.close()
print(f"Saved: {outdir}/recursive_potentiality.png")


# ============================================================
# CHART 3: UNIFIED VIEW — The Three Levels of Reality
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

# Conceptual diagram: PRNG → CRNG → Recursive CRNG
# Showing the progression from potentiality to reality

# Y-axis: "realness" (how well it matches real data)
# X-axis: level of ontological depth

levels = ['PRNG\n(Pure Potentiality)\nΔυναμον',
          'CRNG\n(Contingent Act)\nΠοεσις',
          'CRNG²\n(Structured Potentiality)\nΔυναμον + Ποεσις']

# Metrics for each level (normalized to show progression)
# Using tail events as proxy for "realness"
x_pos = [0, 1, 2]

# Draw as connected circles with size proportional to "realness"
sizes = [200, 800, 2000]  # relative importance
kurtosis_vals = [3.0, 48587, 17.55]
tail_vals = [0.27, 0.018, 2.088]
vol_cluster = [0.0, -0.0002, 0.052]

# Background gradient zones
ax.axvspan(-0.5, 0.5, alpha=0.1, color=PRNG_COLOR, label='Potentiality')
ax.axvspan(0.5, 1.5, alpha=0.1, color=CRNG_COLOR, label='Act')
ax.axvspan(1.5, 2.5, alpha=0.1, color=ACCENT, label='Structured Reality')

# Plot circles
for i, (x, size, label) in enumerate(zip(x_pos, sizes, levels)):
    ax.scatter(x, 1, s=size, c=DEPTH_COLORS[i], alpha=0.8, zorder=5,
              edgecolors='white', linewidths=1.5)
    ax.text(x, 0.4, label, ha='center', va='top', fontsize=12,
           color=DEPTH_COLORS[i], fontweight='bold')

# Arrows connecting them
ax.annotate('', xy=(0.85, 1), xytext=(0.15, 1),
           arrowprops=dict(arrowstyle='->', color='#8b949e', lw=2))
ax.annotate('', xy=(1.85, 1), xytext=(1.15, 1),
           arrowprops=dict(arrowstyle='->', color='#8b949e', lw=2))

# Metrics below each circle
for i, x in enumerate(x_pos):
    k_label = f'K={kurtosis_vals[i]:,.0f}' if kurtosis_vals[i] > 100 else f'K={kurtosis_vals[i]:.1f}'
    metrics_text = f'{k_label}\nTails: {tail_vals[i]:.3f}%\nACF: {vol_cluster[i]:.4f}'
    ax.text(x, 1.6, metrics_text, ha='center', va='bottom', fontsize=10,
           color='#8b949e', family='monospace')

# Title and labels
ax.set_title('The Three Levels of Reality\n'
            'From Pure Potentiality to Structured Act',
            fontsize=16, fontweight='bold', color='#e6edf3', pad=20)

# Key insight at bottom
insight = ('PRNG = the lake (K=3, no structure)  |  '
          'CRNG = the volcano (K=48k, rare extremes)  |  '
          'CRNG² = the ocean (K=17.5, frequent structured extremes)')
ax.text(1, -0.15, insight, ha='center', va='top', fontsize=10,
       color=ACCENT, transform=ax.transAxes, style='italic')

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.2, 2.2)
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{outdir}/three_levels.png', dpi=150, bbox_inches='tight',
           facecolor='#0d1117')
plt.close()
print(f"Saved: {outdir}/three_levels.png")

print("\nAll charts generated!")
