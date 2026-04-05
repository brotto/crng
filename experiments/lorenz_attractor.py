"""
LORENZ ATTRACTOR: CRNG vs PRNG PERTURBATIONS
================================================

The Lorenz system is the foundational model of deterministic chaos:
  dx/dt = σ(y - x)
  dy/dt = x(ρ - z) - y
  dz/dt = xy - βz

In real applications (weather, climate), noise is added to model
unresolved processes. Standard models use Gaussian noise (K=3).
Real atmospheric perturbations have fat tails (K>>3).

Experiment:
1. Generate Lorenz attractor with NO noise (pure deterministic)
2. Generate with PRNG (Gaussian) perturbations
3. Generate with CRNG (fat-tailed, clustered) perturbations
4. Compare: trajectory statistics, attractor geometry, regime transitions

Hypothesis: CRNG perturbations produce more realistic regime switching
(the "butterfly effect" with structure).

Ale Brotto — 2026-04-05
"""

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crng import ContingencyRNG


# ============================================================
# LORENZ SYSTEM
# ============================================================

def lorenz_step(state, dt, sigma=10.0, rho=28.0, beta=8.0/3.0, noise=0.0):
    """Single Euler step of the Lorenz system with optional noise."""
    x, y, z = state
    dx = sigma * (y - x) * dt + noise
    dy = (x * (rho - z) - y) * dt + noise
    dz = (x * y - beta * z) * dt + noise
    return np.array([x + dx, y + dy, z + dz])


def generate_lorenz(n_steps=50000, dt=0.01, noise_source=None, noise_scale=0.0,
                    initial=(1.0, 1.0, 1.0), sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Generate Lorenz trajectory with configurable noise source."""

    trajectory = np.zeros((n_steps, 3))
    trajectory[0] = initial

    for i in range(1, n_steps):
        if noise_source is not None and noise_scale > 0:
            noise = noise_source() * noise_scale
        else:
            noise = 0.0

        trajectory[i] = lorenz_step(
            trajectory[i-1], dt, sigma, rho, beta, noise
        )

        # Safety: clip to prevent divergence
        trajectory[i] = np.clip(trajectory[i], -200, 200)

    return trajectory


# ============================================================
# ANALYSIS
# ============================================================

def analyze_trajectory(traj, label):
    """Analyze statistical properties of a Lorenz trajectory."""

    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

    # Returns (velocity)
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    speed = np.sqrt(dx**2 + dy**2 + dz**2)

    # Regime detection: which lobe? (x > 0 = right, x < 0 = left)
    regime = (x > 0).astype(int)
    regime_changes = np.sum(np.diff(regime) != 0)
    regime_durations = []
    current = 1
    for i in range(1, len(regime)):
        if regime[i] == regime[i-1]:
            current += 1
        else:
            regime_durations.append(current)
            current = 1
    regime_durations.append(current)
    regime_durations = np.array(regime_durations, dtype=float)

    stats = {
        'label': label,
        'x_kurtosis': sp_stats.kurtosis(x, fisher=False),
        'speed_kurtosis': sp_stats.kurtosis(speed, fisher=False),
        'speed_mean': np.mean(speed),
        'speed_std': np.std(speed),
        'speed_vol_clustering': np.corrcoef(speed[:-1], speed[1:])[0, 1],
        'regime_changes': regime_changes,
        'regime_dur_mean': np.mean(regime_durations),
        'regime_dur_kurtosis': sp_stats.kurtosis(regime_durations, fisher=False),
        'regime_dur_cv': np.std(regime_durations) / np.mean(regime_durations),
        'x_range': np.max(x) - np.min(x),
        'z_range': np.max(z) - np.min(z),
    }

    print(f"\n  {label}:")
    print(f"    X kurtosis:           {stats['x_kurtosis']:.2f}")
    print(f"    Speed kurtosis:       {stats['speed_kurtosis']:.2f}")
    print(f"    Speed vol clustering: {stats['speed_vol_clustering']:.4f}")
    print(f"    Regime changes:       {stats['regime_changes']}")
    print(f"    Regime duration mean: {stats['regime_dur_mean']:.1f} steps")
    print(f"    Regime duration CV:   {stats['regime_dur_cv']:.3f}")
    print(f"    Regime dur kurtosis:  {stats['regime_dur_kurtosis']:.2f}")

    return stats, regime_durations


# ============================================================
# CHARTS
# ============================================================

COLORS = {
    'bg': '#0D1117',
    'text': '#E6EDF3',
    'grid': '#21262D',
    'det': '#888888',
    'prng': '#3498DB',
    'crng': '#E74C3C',
    'accent': '#FFD700',
}


def chart_three_attractors(traj_det, traj_prng, traj_crng):
    """Side-by-side 3D attractors: deterministic, PRNG, CRNG."""

    fig = plt.figure(figsize=(18, 6), facecolor=COLORS['bg'])

    for i, (traj, color, title) in enumerate([
        (traj_det, COLORS['det'], 'Deterministic\n(no noise)'),
        (traj_prng, COLORS['prng'], 'PRNG Perturbations\n(Gaussian, K=3)'),
        (traj_crng, COLORS['crng'], 'CRNG Perturbations\n(fat-tailed, K>>3)'),
    ]):
        ax = fig.add_subplot(1, 3, i+1, projection='3d', facecolor=COLORS['bg'])

        # Subsample for clarity
        step = max(1, len(traj) // 10000)
        t = traj[::step]

        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=color, alpha=0.4, linewidth=0.3)
        ax.scatter(t[0, 0], t[0, 1], t[0, 2], color='white', s=30, zorder=5)

        ax.set_title(title, color=COLORS['text'], fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('X', color=COLORS['text'], fontsize=8)
        ax.set_ylabel('Y', color=COLORS['text'], fontsize=8)
        ax.set_zlabel('Z', color=COLORS['text'], fontsize=8)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(COLORS['grid'])
        ax.yaxis.pane.set_edgecolor(COLORS['grid'])
        ax.zaxis.pane.set_edgecolor(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'], labelsize=7)

    fig.suptitle('Lorenz Attractor: Three Realities',
                 color=COLORS['accent'], fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('charts/lorenz_three_attractors.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ lorenz_three_attractors.png")


def chart_regime_comparison(dur_det, dur_prng, dur_crng):
    """Compare regime duration distributions."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['bg'])

    # Left: Histogram
    ax1.set_facecolor(COLORS['bg'])
    bins = np.linspace(0, max(np.percentile(dur_det, 99),
                              np.percentile(dur_prng, 99),
                              np.percentile(dur_crng, 99)), 40)

    ax1.hist(dur_det, bins=bins, alpha=0.4, color=COLORS['det'], label='Deterministic',
             density=True, edgecolor='none')
    ax1.hist(dur_prng, bins=bins, alpha=0.5, color=COLORS['prng'], label='PRNG',
             density=True, edgecolor='none')
    ax1.hist(dur_crng, bins=bins, alpha=0.5, color=COLORS['crng'], label='CRNG',
             density=True, edgecolor='none')

    ax1.set_xlabel('Regime Duration (steps)', fontsize=12, color=COLORS['text'])
    ax1.set_ylabel('Density', fontsize=12, color=COLORS['text'])
    ax1.set_title('Regime Duration Distribution', fontsize=14,
                  color=COLORS['text'], fontweight='bold')
    ax1.legend(fontsize=10, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
               labelcolor=COLORS['text'])

    for spine in ax1.spines.values():
        spine.set_color(COLORS['grid'])
    ax1.tick_params(colors=COLORS['text'])

    # Right: CDF comparison
    ax2.set_facecolor(COLORS['bg'])

    for dur, color, label in [(dur_det, COLORS['det'], 'Deterministic'),
                               (dur_prng, COLORS['prng'], 'PRNG'),
                               (dur_crng, COLORS['crng'], 'CRNG')]:
        sorted_d = np.sort(dur)
        cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
        ax2.plot(sorted_d, cdf, color=color, linewidth=2.5, label=label)

    ax2.set_xlabel('Regime Duration (steps)', fontsize=12, color=COLORS['text'])
    ax2.set_ylabel('Cumulative Probability', fontsize=12, color=COLORS['text'])
    ax2.set_title('Regime Duration CDFs', fontsize=14,
                  color=COLORS['text'], fontweight='bold')
    ax2.legend(fontsize=10, facecolor=COLORS['bg'], edgecolor=COLORS['grid'],
               labelcolor=COLORS['text'])
    ax2.set_xlim(0, np.percentile(np.concatenate([dur_det, dur_prng, dur_crng]), 98))

    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    plt.savefig('charts/lorenz_regime_comparison.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ lorenz_regime_comparison.png")


def chart_lorenz_animated(traj_crng):
    """Animated GIF: CRNG-driven Lorenz attractor building up."""

    fig = plt.figure(figsize=(8, 8), facecolor=COLORS['bg'])
    ax = fig.add_subplot(111, projection='3d', facecolor=COLORS['bg'])

    # Subsample
    step = max(1, len(traj_crng) // 5000)
    traj = traj_crng[::step]

    ax.set_xlim(np.min(traj[:, 0])-2, np.max(traj[:, 0])+2)
    ax.set_ylim(np.min(traj[:, 1])-2, np.max(traj[:, 1])+2)
    ax.set_zlim(np.min(traj[:, 2])-2, np.max(traj[:, 2])+2)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(COLORS['grid'])
    ax.yaxis.pane.set_edgecolor(COLORS['grid'])
    ax.zaxis.pane.set_edgecolor(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'], labelsize=7)
    ax.set_xlabel('X', color=COLORS['text'], fontsize=8)
    ax.set_ylabel('Y', color=COLORS['text'], fontsize=8)
    ax.set_zlabel('Z', color=COLORS['text'], fontsize=8)

    title = fig.suptitle('', color=COLORS['accent'], fontsize=14, fontweight='bold')

    n_frames = 80
    points_per_frame = len(traj) // n_frames

    lines = []

    def animate(frame):
        end = min((frame + 1) * points_per_frame, len(traj))
        start = max(0, end - points_per_frame)

        segment = traj[start:end]
        if len(segment) > 1:
            # Color gradient: older = dimmer
            alpha = 0.3 + 0.5 * (frame / n_frames)
            line = ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                          color=COLORS['crng'], alpha=alpha, linewidth=0.5)[0]
            lines.append(line)

        # Rotate view
        ax.view_init(elev=25, azim=frame * 4.5)

        pct = min(100, int(end / len(traj) * 100))
        title.set_text(f'CRNG-Driven Lorenz Attractor — {pct}%')

        return lines

    anim = animation.FuncAnimation(fig, animate, frames=n_frames + 10,
                                   interval=100, blit=False)
    anim.save('charts/lorenz_crng_attractor.gif', writer='pillow', fps=12,
              savefig_kwargs={'facecolor': COLORS['bg']})
    plt.close()
    print("  ✓ lorenz_crng_attractor.gif")


def chart_speed_timeseries(traj_prng, traj_crng):
    """Speed (volatility) comparison: PRNG vs CRNG perturbations."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), facecolor=COLORS['bg'],
                                    sharex=True)

    for ax, traj, color, label in [
        (ax1, traj_prng, COLORS['prng'], 'PRNG Perturbations'),
        (ax2, traj_crng, COLORS['crng'], 'CRNG Perturbations'),
    ]:
        ax.set_facecolor(COLORS['bg'])
        dx = np.diff(traj[:, 0])
        dy = np.diff(traj[:, 1])
        dz = np.diff(traj[:, 2])
        speed = np.sqrt(dx**2 + dy**2 + dz**2)

        # Subsample for visibility
        step = max(1, len(speed) // 3000)
        t = np.arange(0, len(speed), step)
        s = speed[::step]

        ax.fill_between(t, 0, s, color=color, alpha=0.4)
        ax.plot(t, s, color=color, linewidth=0.3, alpha=0.6)

        k = sp_stats.kurtosis(speed, fisher=False)
        acf = np.corrcoef(speed[:-1], speed[1:])[0, 1]
        ax.set_title(f'{label}  |  K={k:.1f}  |  Vol ACF={acf:.3f}',
                     color=COLORS['text'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Speed', color=COLORS['text'], fontsize=10)

        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'])

    ax2.set_xlabel('Time Step', color=COLORS['text'], fontsize=10)

    fig.suptitle('Lorenz Trajectory Speed: Fat Tails in Motion',
                 color=COLORS['accent'], fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('charts/lorenz_speed_comparison.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ lorenz_speed_comparison.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    os.makedirs('charts', exist_ok=True)

    print("=" * 70)
    print("  LORENZ ATTRACTOR: CRNG vs PRNG PERTURBATIONS")
    print("  What does chaos look like with contingent noise?")
    print("=" * 70)

    n_steps = 50000
    dt = 0.01
    noise_scale = 0.05

    # 1. Deterministic (no noise)
    print("\n  Generating deterministic Lorenz...")
    traj_det = generate_lorenz(n_steps=n_steps, dt=dt)

    # 2. PRNG perturbations (Gaussian)
    print("  Generating PRNG-perturbed Lorenz...")
    np.random.seed(42)
    prng_gen = lambda: np.random.randn()
    traj_prng = generate_lorenz(n_steps=n_steps, dt=dt,
                                 noise_source=prng_gen, noise_scale=noise_scale)

    # 3. CRNG perturbations (fat-tailed, clustered)
    print("  Generating CRNG-perturbed Lorenz...")
    rng = ContingencyRNG(seed=42, target_kurtosis=8.0, vol_clustering=0.35,
                         n_oscillators=7)
    # Center CRNG output around 0
    crng_mean = np.mean([rng.next() for _ in range(1000)])
    rng2 = ContingencyRNG(seed=42, target_kurtosis=8.0, vol_clustering=0.35,
                          n_oscillators=7)
    crng_gen = lambda: rng2.next() - crng_mean
    traj_crng = generate_lorenz(n_steps=n_steps, dt=dt,
                                 noise_source=crng_gen, noise_scale=noise_scale)

    # Analysis
    print(f"\n{'='*70}")
    print(f"  TRAJECTORY ANALYSIS")
    print(f"{'='*70}")

    stats_det, dur_det = analyze_trajectory(traj_det, "Deterministic")
    stats_prng, dur_prng = analyze_trajectory(traj_prng, "PRNG (Gaussian)")
    stats_crng, dur_crng = analyze_trajectory(traj_crng, "CRNG (Contingent)")

    # KS test: regime durations
    print(f"\n{'='*70}")
    print(f"  KS TEST: REGIME DURATION DISTRIBUTIONS")
    print(f"{'='*70}")

    for a_label, a_dur, b_label, b_dur in [
        ("Deterministic", dur_det, "PRNG", dur_prng),
        ("Deterministic", dur_det, "CRNG", dur_crng),
        ("PRNG", dur_prng, "CRNG", dur_crng),
    ]:
        ks, p = sp_stats.ks_2samp(a_dur, b_dur)
        match = "SAME" if p > 0.05 else "DIFFERENT"
        print(f"  {a_label:15s} vs {b_label:15s}: KS={ks:.4f}, p={p:.4f} [{match}]")

    # Scorecard
    print(f"\n{'='*70}")
    print(f"  SCORECARD")
    print(f"{'='*70}")

    print(f"\n  {'Metric':30s} {'Deterministic':>14} {'PRNG':>14} {'CRNG':>14}")
    print(f"  {'-'*72}")

    metrics = [
        ('Speed Kurtosis', 'speed_kurtosis'),
        ('Speed Vol Clustering', 'speed_vol_clustering'),
        ('Regime Changes', 'regime_changes'),
        ('Regime Duration CV', 'regime_dur_cv'),
        ('Regime Duration Kurtosis', 'regime_dur_kurtosis'),
    ]

    for name, key in metrics:
        print(f"  {name:30s} {stats_det[key]:>14.3f} {stats_prng[key]:>14.3f} {stats_crng[key]:>14.3f}")

    # Charts
    print(f"\n{'='*70}")
    print(f"  GENERATING CHARTS")
    print(f"{'='*70}")

    print("\n  1. Three attractors...")
    chart_three_attractors(traj_det, traj_prng, traj_crng)

    print("\n  2. Regime comparison...")
    chart_regime_comparison(dur_det, dur_prng, dur_crng)

    print("\n  3. Speed comparison...")
    chart_speed_timeseries(traj_prng, traj_crng)

    print("\n  4. CRNG attractor animation...")
    chart_lorenz_animated(traj_crng)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
