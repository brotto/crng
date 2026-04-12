"""
TSON Visualizations

1. Potency curve (x^x) with critical points annotated
2. Mesmitude probability ℵ(N) for different Π values
3. Arche equation Ω(N) — the reality coefficient
4. Monte Carlo distribution of first mesmitude
5. The Euler connection diagram
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from equations import potency, mesmitude_probability, arche, expected_mesmitude_instant
from simulation import model4_pure_tson, model3_potency_monte_carlo
import os

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'font.family': 'monospace',
    'font.size': 11,
})

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# Colors
GOLD = '#d4a017'
CYAN = '#58a6ff'
GREEN = '#3fb950'
RED = '#f85149'
PURPLE = '#bc8cff'
ORANGE = '#d29922'
WHITE = '#e6edf3'
MUTED = '#8b949e'

FIG_SIZE = (12.5, 5.0)  # 5:2 ratio


def chart_potency_curve():
    """Chart 1: The Potency Function Π(x) = x^x"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    x = np.linspace(0.001, 1.0, 1000)
    y = potency(x)

    # Main curve
    ax.plot(x, y, color=GOLD, linewidth=2.5, label=r'$\Pi(x) = x^x$')

    # Critical point at x = 1/e
    x_crit = 1/np.e
    y_crit = np.exp(-1/np.e)
    ax.plot(x_crit, y_crit, 'o', color=RED, markersize=10, zorder=5)
    ax.annotate(f'Minimum potency\nx = 1/e ≈ {x_crit:.3f}\nΠ = e^(-1/e) ≈ {y_crit:.4f}',
                xy=(x_crit, y_crit), xytext=(x_crit + 0.15, y_crit - 0.08),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

    # 0^0 = 1 at origin
    ax.plot(0, 1, 'o', color=CYAN, markersize=12, zorder=5)
    ax.annotate('0⁰ = 1\nPure potency\n(Nothing = max becoming)',
                xy=(0, 1), xytext=(0.08, 0.85),
                fontsize=9, color=CYAN,
                arrowprops=dict(arrowstyle='->', color=CYAN, lw=1.5))

    # 1^1 = 1 at end
    ax.plot(1, 1, 'o', color=GREEN, markersize=10, zorder=5)
    ax.annotate('1¹ = 1\nFull being\n(actuality)',
                xy=(1, 1), xytext=(0.75, 0.85),
                fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

    # Zones
    ax.axvspan(0, x_crit, alpha=0.08, color=CYAN, label='Descending: potency → non-being')
    ax.axvspan(x_crit, 1.0, alpha=0.08, color=GREEN, label='Ascending: non-being → being')

    ax.set_xlabel('x (progression from Nothing to Being)')
    ax.set_ylabel('Π(x) = x^x (potency)')
    ax.set_title('TSON: The Potency Function — x^x', fontsize=14, fontweight='bold', color=GOLD)
    ax.legend(loc='lower center', fontsize=8, framealpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, '01_potency_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def chart_mesmitude_probability():
    """Chart 2: Mesmitude Probability ℵ(N)"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    N = np.arange(0, 8, 0.01)

    # Different potency values
    for Pi, color, label in [
        (1.0, GOLD, 'Π = 1 (pure TSON: 0⁰ = 1)'),
        (0.692, ORANGE, 'Π = e^(-1/e) ≈ 0.692 (min potency)'),
        (0.5, CYAN, 'Π = 0.5'),
        (0.1, MUTED, 'Π = 0.1'),
    ]:
        y = mesmitude_probability(N, Pi)
        ax.plot(N, y, color=color, linewidth=2, label=label)

    # Key points for Π=1
    for n in [2, 3, 4]:
        p = float(mesmitude_probability(n, 1.0))
        ax.plot(n, p, 'o', color=GOLD, markersize=8, zorder=5)
        offset = (-30, 10) if n != 3 else (10, -20)
        ax.annotate(f'N={n}: {p:.3f}', xy=(n, p), xytext=offset,
                    textcoords='offset points', fontsize=9, color=GOLD,
                    arrowprops=dict(arrowstyle='->', color=GOLD, lw=1))

    # 1 - 1/e line
    euler_p = 1 - 1/np.e
    ax.axhline(euler_p, color=RED, linestyle='--', alpha=0.5, linewidth=1)
    ax.text(6.5, euler_p + 0.02, f'1 - 1/e ≈ {euler_p:.3f}', fontsize=8, color=RED)

    # Expected N*
    E_N = expected_mesmitude_instant()
    ax.axvline(E_N, color=PURPLE, linestyle=':', alpha=0.5, linewidth=1)
    ax.text(E_N + 0.05, 0.15, f'E[N*] ≈ {E_N:.2f}', fontsize=8, color=PURPLE, rotation=90)

    ax.set_xlabel('N (number of successive instants)')
    ax.set_ylabel('ℵ(N, Π) — probability of mesmitude')
    ax.set_title('TSON: Mesmitude Probability — ℵ(N, Π) = 1 - exp(-N(N-1)·Π/2)',
                 fontsize=13, fontweight='bold', color=GOLD)
    ax.legend(loc='center right', fontsize=8, framealpha=0.3)
    ax.set_xlim(-0.1, 7.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, '02_mesmitude_probability.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def chart_arche_equation():
    """Chart 3: The Arche Equation Ω(N) — Reality Coefficient"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    N_continuous = np.linspace(0.01, 7, 500)
    N_discrete = np.arange(0, 8)

    # Left: Ω(N) continuous
    omega = arche(N_continuous)
    ax1.plot(N_continuous, omega, color=GOLD, linewidth=2.5)

    # Discrete points
    omega_disc = arche(N_discrete)
    for n, o in zip(N_discrete, omega_disc):
        color = RED if n == 0 else (ORANGE if n == 1 else (GREEN if n >= 2 else MUTED))
        ax1.plot(n, o, 'o', color=color, markersize=10, zorder=5)

    # Annotations
    ax1.annotate('N=0: Nada\nΩ=0', xy=(0, 0), xytext=(0.5, 0.15),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED))
    ax1.annotate(f'N=2: Arche\nΩ={float(arche(2)):.3f}', xy=(2, float(arche(2))),
                xytext=(2.5, float(arche(2))-0.15),
                fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN))

    ax1.set_xlabel('N (instants)')
    ax1.set_ylabel('Ω(N) — reality coefficient')
    ax1.set_title('Arche Equation: Ω(N)', fontsize=12, fontweight='bold', color=GOLD)
    ax1.grid(True, alpha=0.3)

    # Right: Decomposition into Π and ℵ
    phi = N_continuous / (N_continuous + np.e)
    Pi_vals = potency(phi)
    aleph_vals = np.array([float(mesmitude_probability(n, p)) for n, p in zip(N_continuous, Pi_vals)])

    ax2.plot(N_continuous, Pi_vals, color=CYAN, linewidth=2, label='Π(φ(N)) — potency')
    ax2.plot(N_continuous, aleph_vals, color=PURPLE, linewidth=2, label='ℵ(N,Π) — mesmitude')
    ax2.plot(N_continuous, omega, color=GOLD, linewidth=2.5, label='Ω = Π · ℵ — reality')

    ax2.axhline(1 - 1/np.e, color=RED, linestyle='--', alpha=0.3, linewidth=1)
    ax2.text(5.5, 1 - 1/np.e + 0.03, '1-1/e', fontsize=7, color=RED)

    ax2.set_xlabel('N (instants)')
    ax2.set_ylabel('Value')
    ax2.set_title('Decomposition: Π × ℵ = Ω', fontsize=12, fontweight='bold', color=GOLD)
    ax2.legend(loc='center right', fontsize=8, framealpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, '03_arche_equation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def chart_monte_carlo():
    """Chart 4: Monte Carlo Distribution of First Mesmitude"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    # Pure TSON
    print("  Running Pure TSON Monte Carlo (1M trials)...")
    times4 = model4_pure_tson(n_trials=1_000_000)

    unique, counts = np.unique(times4, return_counts=True)
    pcts = counts / len(times4) * 100

    bars = ax1.bar(unique[:7], pcts[:7], color=[RED if u==1 else GOLD for u in unique[:7]],
                   edgecolor='#30363d', linewidth=0.5)
    for bar, u, p in zip(bars, unique[:7], pcts[:7]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{p:.1f}%', ha='center', fontsize=9, color=WHITE)

    ax1.set_xlabel('N* (first mesmitude instant)')
    ax1.set_ylabel('Probability (%)')
    ax1.set_title(f'Pure TSON (Π=1)\nE[N*]={np.mean(times4):.3f}',
                  fontsize=11, fontweight='bold', color=GOLD)
    ax1.grid(True, alpha=0.3, axis='y')

    # Potency-weighted
    print("  Running Potency-Weighted Monte Carlo (1M trials)...")
    times3 = model3_potency_monte_carlo(n_trials=1_000_000)

    unique3, counts3 = np.unique(times3, return_counts=True)
    pcts3 = counts3 / len(times3) * 100

    n_show = min(12, len(unique3))
    bars3 = ax2.bar(unique3[:n_show], pcts3[:n_show], color=CYAN,
                    edgecolor='#30363d', linewidth=0.5)
    for bar, u, p in zip(bars3, unique3[:n_show], pcts3[:n_show]):
        if p > 1:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{p:.1f}%', ha='center', fontsize=7, color=WHITE)

    ax2.set_xlabel('N* (first mesmitude instant)')
    ax2.set_ylabel('Probability (%)')
    ax2.set_title(f'Potency-Weighted (φ=N/(N+e))\nE[N*]={np.mean(times3):.3f}',
                  fontsize=11, fontweight='bold', color=CYAN)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, '04_monte_carlo.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def chart_euler_connection():
    """Chart 5: The Euler Connection — e governs the Nada→Ser transition"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Central concept
    ax.text(0.5, 0.92, 'e = 2.71828...', fontsize=24, fontweight='bold',
            ha='center', va='center', color=GOLD,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=GOLD, linewidth=2))

    ax.text(0.5, 0.82, 'The constant that governs the transition from Nothing to Being',
            fontsize=10, ha='center', va='center', color=MUTED, style='italic')

    # Three appearances
    connections = [
        (0.15, 0.55, f'Π(1/e) = e^(-1/e)\n≈ 0.6922',
         'Minimum potency\nof the void', CYAN, 'Point of maximum\nresistance to becoming'),
        (0.50, 0.55, f'ℵ(2,1) = 1 - 1/e\n≈ 0.6321',
         'Mesmitude probability\nat 2nd instant', RED, 'The Euler threshold:\n63.2% at N=2'),
        (0.85, 0.55, f'φ(1) = 1/(1+e)\n≈ 0.2689',
         'Normalization of\nthe first instant', GREEN, 'How "far" the first\ninstant is from Being'),
    ]

    for x, y, formula, desc, color, interp in connections:
        # Box
        ax.text(x, y, formula, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22',
                         edgecolor=color, linewidth=1.5))
        # Description
        ax.text(x, y - 0.12, desc, fontsize=8, ha='center', va='center', color=MUTED)
        # Interpretation
        ax.text(x, y - 0.25, interp, fontsize=8, ha='center', va='center',
                color=color, style='italic')

        # Arrow from e to each
        ax.annotate('', xy=(x, y + 0.08), xytext=(0.5, 0.88),
                   arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5, alpha=0.5))

    # Bottom synthesis
    ax.text(0.5, 0.12, (
        'Three manifestations of one constant. Not coincidence — necessity.\n'
        'e is the natural rate of emergence. It is ℵ-Arche:\n'
        'the mathematical signature of Being arising from Nothing.'
    ), fontsize=10, ha='center', va='center', color=WHITE,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                     edgecolor=GOLD, linewidth=1, alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('The Euler Connection — TSON', fontsize=14, fontweight='bold',
                 color=GOLD, pad=10)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, '05_euler_connection.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def chart_convergence_table():
    """Chart 6: The 0^0 convergence + mesmitude table side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    # Left: 0^0 convergence (as in TSON original)
    xs = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01])
    ys = potency(xs)

    ax1.plot(xs[::-1], ys[::-1], 'o-', color=GOLD, markersize=8, linewidth=2)

    # Mark the minimum
    x_min = 1/np.e
    y_min = np.exp(-1/np.e)
    ax1.plot(x_min, y_min, 's', color=RED, markersize=12, zorder=5)

    # Annotate each point
    for x, y in zip(xs, ys):
        ax1.annotate(f'{x}^{x}={y:.3f}', xy=(x, y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=6, color=MUTED)

    # Arrow showing convergence to 1
    ax1.annotate('→ 0⁰ = 1', xy=(0.01, potency(0.01)),
                xytext=(-0.05, 1.01), fontsize=11, color=CYAN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CYAN, lw=2))

    ax1.set_xlabel('x → 0⁺')
    ax1.set_ylabel('x^x')
    ax1.set_title('Convergence: x^x → 1 as x → 0⁺', fontsize=11, fontweight='bold', color=GOLD)
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.65, 1.05)

    # Right: The TSON fundamental equation in visual form
    N_vals = [0, 1, 2, 3, 4, 5]
    omega_vals = [float(arche(n)) for n in N_vals]
    labels = ['Ø\n(void)', 'i₀\n(1 instant)', 'i₀,i₁\n(ARCHE)', 'i₀,i₁,i₂\n(being)', 'i₀..i₃', 'i₀..i₄']
    colors = [RED, ORANGE, GREEN, GREEN, GREEN, GREEN]

    bars = ax2.bar(range(len(N_vals)), omega_vals, color=colors,
                   edgecolor='#30363d', linewidth=0.5, alpha=0.8)

    for bar, n, o, label in zip(bars, N_vals, omega_vals, labels):
        ax2.text(bar.get_x() + bar.get_width()/2, -0.06, label,
                ha='center', fontsize=7, color=MUTED)
        if o > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2, o + 0.02,
                    f'Ω={o:.3f}', ha='center', fontsize=8, color=WHITE)

    ax2.set_ylabel('Ω(N) — reality coefficient')
    ax2.set_title('From Nothing to Being: Ω(N)', fontsize=11, fontweight='bold', color=GOLD)
    ax2.set_xticks(range(len(N_vals)))
    ax2.set_xticklabels([f'N={n}' for n in N_vals])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, '06_convergence_and_arche.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == '__main__':
    print("=" * 60)
    print("TSON VISUALIZATIONS")
    print("=" * 60)

    chart_potency_curve()
    chart_mesmitude_probability()
    chart_arche_equation()
    chart_monte_carlo()
    chart_euler_connection()
    chart_convergence_table()

    print(f"\nAll charts saved to {ASSETS_DIR}/")
