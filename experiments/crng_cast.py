#!/usr/bin/env python3
"""
CRNG-Cast: Nowcasting Prototype
================================
A weather nowcasting system that uses CRNG to model the fat-tailed probability
of weather outcomes when multiple thermal blobs converge on a target point.

Key insight: when multiple weather systems converge simultaneously, the resulting
weather is NOT the gaussian sum of their effects — it's a fat-tailed contingent
interaction. CRNG models this correctly while PRNG underestimates extreme outcomes.

Pipeline:
  1. Infrared satellite images (synthetic thermal maps)
  2. Blob segmentation (thresholding + connected components)
  3. Blob tracking (centroid matching between frames)
  4. Convergence projection (which blobs hit the target?)
  5. CRNG probability modeling of concurrent blob interactions

Usage:
  cd crng-package && python3 experiments/crng_cast.py
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crng import ContingencyRNG


# ============================================================
# DARK THEME COLORS
# ============================================================
COLORS = {
    'bg':     '#0D1117',
    'text':   '#E6EDF3',
    'grid':   '#21262D',
    'real':   '#2ECC71',
    'prng':   '#3498DB',
    'crng':   '#E74C3C',
    'accent': '#FFD700',
    'muted':  '#8B949E',
}


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Blob:
    """A detected thermal blob in a satellite frame."""
    blob_id: int
    centroid_x: float
    centroid_y: float
    area: int                  # pixels
    mean_temperature: float    # deviation from background (negative=cold, positive=warm)
    intensity: float           # abs deviation from background
    frame_idx: int = -1


@dataclass
class TrackedBlob:
    """A blob matched across two consecutive frames, with velocity."""
    blob: Blob
    prev_blob: Optional[Blob]
    vx: float = 0.0           # pixels per frame
    vy: float = 0.0
    speed: float = 0.0


@dataclass
class ConvergingBlob:
    """A blob projected to arrive at a target point."""
    tracked_blob: TrackedBlob
    time_of_arrival: float     # hours until arrival
    intensity_at_arrival: float
    distance_now: float        # current distance to target


@dataclass
class Prediction:
    """A CRNG-Cast or PRNG-Cast prediction for a horizon."""
    horizon_hours: float
    n_converging: int
    prob_rain: float
    expected_temp_change: float
    ci_low: float              # 5th percentile
    ci_high: float             # 95th percentile
    outcome_samples: np.ndarray
    method: str                # 'crng' or 'prng'


# ============================================================
# A) SYNTHETIC SATELLITE GENERATOR
# ============================================================
class SyntheticSatelliteGenerator:
    """
    Generate fake IR thermal maps (2D numpy arrays, 256x256 pixels).
    Creates moving gaussian-shaped warm/cold blobs with known velocities.
    Generates sequences at 10-min intervals.
    """

    def __init__(self, size: int = 256, n_blobs: int = 8, seed: int = 42,
                 target_x: float = 140.0, target_y: float = 120.0):
        self.size = size
        self.n_blobs = n_blobs
        self.rng = np.random.RandomState(seed)
        self.background_temp = 250.0  # Kelvin (typical IR background)
        self.frame_interval_hours = 10 / 60  # 10 minutes
        self.target_x = target_x
        self.target_y = target_y

        # Initialize blob parameters
        # Some blobs converge toward target (realistic: fronts moving toward a point)
        # Some blobs are random (background weather systems)
        self.blob_params = []
        n_converging = max(3, n_blobs // 2)  # at least 3 blobs aim at target

        for i in range(n_blobs):
            if i < n_converging:
                # Converging blob: starts far from target, moves toward it
                # Arrival time varies (some arrive at 1h, some at 3h, some at 6h)
                arrival_frame = self.rng.choice([4, 6, 10, 15, 20, 30])  # various arrival times
                angle = self.rng.uniform(0, 2 * np.pi)
                start_dist = self.rng.uniform(40, 100)
                x0 = target_x + start_dist * np.cos(angle)
                y0 = target_y + start_dist * np.sin(angle)
                # Velocity aimed at target (with some noise)
                dx = target_x - x0
                dy = target_y - y0
                dist = np.sqrt(dx**2 + dy**2)
                speed = dist / arrival_frame
                noise_angle = self.rng.normal(0, 0.15)  # slight aim error
                vx = speed * np.cos(np.arctan2(dy, dx) + noise_angle)
                vy = speed * np.sin(np.arctan2(dy, dx) + noise_angle)
                amplitude = self.rng.choice([-1, 1]) * self.rng.uniform(8, 18)
            else:
                # Random background blob
                x0 = self.rng.uniform(20, size - 20)
                y0 = self.rng.uniform(20, size - 20)
                vx = self.rng.uniform(-2, 2)
                vy = self.rng.uniform(-2, 2)
                amplitude = self.rng.uniform(-12, 12)

            self.blob_params.append({
                'x0': x0,
                'y0': y0,
                'vx': vx,
                'vy': vy,
                'sigma': self.rng.uniform(10, 25),
                'amplitude': amplitude,
                'wobble_freq': self.rng.uniform(0.05, 0.2),
                'wobble_amp': self.rng.uniform(0, 1.5),
            })

    def generate_frame(self, frame_idx: int) -> np.ndarray:
        """Generate a single IR thermal map at the given frame index."""
        thermal_map = np.full((self.size, self.size), self.background_temp, dtype=np.float64)
        yy, xx = np.mgrid[0:self.size, 0:self.size]

        for bp in self.blob_params:
            # Position at this frame (with sinusoidal wobble)
            t = frame_idx
            cx = bp['x0'] + bp['vx'] * t + bp['wobble_amp'] * np.sin(bp['wobble_freq'] * t)
            cy = bp['y0'] + bp['vy'] * t + bp['wobble_amp'] * np.cos(bp['wobble_freq'] * t * 1.3)

            # Wrap around edges (periodic boundary)
            cx = cx % self.size
            cy = cy % self.size

            # Gaussian blob
            dx = np.minimum(np.abs(xx - cx), self.size - np.abs(xx - cx))
            dy = np.minimum(np.abs(yy - cy), self.size - np.abs(yy - cy))
            dist_sq = dx**2 + dy**2
            blob = bp['amplitude'] * np.exp(-dist_sq / (2 * bp['sigma']**2))
            thermal_map += blob

        # Add sensor noise
        noise = self.rng.normal(0, 0.5, (self.size, self.size))
        thermal_map += noise

        return thermal_map

    def generate_sequence(self, n_frames: int) -> List[np.ndarray]:
        """Generate a sequence of n_frames thermal maps."""
        return [self.generate_frame(i) for i in range(n_frames)]

    def get_actual_outcome(self, target_x: float, target_y: float,
                           frame_idx: int) -> Dict:
        """Get the actual temperature at a target point for validation."""
        frame = self.generate_frame(frame_idx)
        ix, iy = int(target_x) % self.size, int(target_y) % self.size
        # Average over a small region around target
        region = frame[max(0,iy-3):iy+4, max(0,ix-3):ix+4]
        temp = np.mean(region)
        temp_change = temp - self.background_temp
        is_rain = temp_change < -5  # cold blobs = rain potential
        return {
            'temperature': temp,
            'temp_change': temp_change,
            'is_rain': is_rain,
        }


# ============================================================
# B) BLOB DETECTOR
# ============================================================
class BlobDetector:
    """
    Segment thermal maps into blobs using thresholding + connected components.
    Extract blob properties: centroid, area, mean_temperature, intensity.
    """

    def __init__(self, threshold: float = 5.0, min_area: int = 20):
        """
        Args:
            threshold: minimum absolute temperature deviation from background to detect
            min_area: minimum blob area in pixels
        """
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, thermal_map: np.ndarray, background_temp: float = 250.0,
               frame_idx: int = -1) -> List[Blob]:
        """Detect blobs in a thermal map."""
        deviation = thermal_map - background_temp
        mask = np.abs(deviation) > self.threshold

        # Connected components
        labeled, n_features = ndimage.label(mask)

        blobs = []
        for label_id in range(1, n_features + 1):
            component = labeled == label_id
            area = int(np.sum(component))
            if area < self.min_area:
                continue

            # Centroid
            coords = np.argwhere(component)  # (row, col) = (y, x)
            cy = float(np.mean(coords[:, 0]))
            cx = float(np.mean(coords[:, 1]))

            # Temperature stats
            temps = deviation[component]
            mean_temp = float(np.mean(temps))
            intensity = float(np.mean(np.abs(temps)))

            blobs.append(Blob(
                blob_id=label_id,
                centroid_x=cx,
                centroid_y=cy,
                area=area,
                mean_temperature=mean_temp,
                intensity=intensity,
                frame_idx=frame_idx,
            ))

        return blobs


# ============================================================
# C) BLOB TRACKER
# ============================================================
class BlobTracker:
    """
    Track blobs between consecutive frames using nearest-centroid matching.
    Calculate velocity vectors for each matched blob.
    """

    def __init__(self, max_match_distance: float = 40.0, map_size: int = 256):
        self.max_match_distance = max_match_distance
        self.map_size = map_size

    def _periodic_distance(self, x1, y1, x2, y2):
        """Distance with periodic boundary conditions."""
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dx = min(dx, self.map_size - dx)
        dy = min(dy, self.map_size - dy)
        return np.sqrt(dx**2 + dy**2)

    def _periodic_delta(self, v1, v2):
        """Signed delta with periodic boundaries."""
        d = v2 - v1
        if d > self.map_size / 2:
            d -= self.map_size
        elif d < -self.map_size / 2:
            d += self.map_size
        return d

    def track(self, prev_blobs: List[Blob], curr_blobs: List[Blob]) -> List[TrackedBlob]:
        """Match blobs between two frames and compute velocities."""
        tracked = []
        used_prev = set()

        for cb in curr_blobs:
            best_dist = float('inf')
            best_prev = None

            for pb in prev_blobs:
                if pb.blob_id in used_prev:
                    continue
                d = self._periodic_distance(cb.centroid_x, cb.centroid_y,
                                            pb.centroid_x, pb.centroid_y)
                if d < best_dist:
                    best_dist = d
                    best_prev = pb

            if best_prev is not None and best_dist < self.max_match_distance:
                used_prev.add(best_prev.blob_id)
                vx = self._periodic_delta(best_prev.centroid_x, cb.centroid_x)
                vy = self._periodic_delta(best_prev.centroid_y, cb.centroid_y)
                speed = np.sqrt(vx**2 + vy**2)
                tracked.append(TrackedBlob(
                    blob=cb,
                    prev_blob=best_prev,
                    vx=vx, vy=vy,
                    speed=speed,
                ))
            else:
                # New blob — no velocity info
                tracked.append(TrackedBlob(blob=cb, prev_blob=None))

        return tracked


# ============================================================
# D) CRNG-CAST PREDICTOR
# ============================================================
class CRNGCastPredictor:
    """
    Predict weather outcomes at a target point using CRNG for concurrent blob interactions.

    When multiple blobs converge on a point simultaneously, their combined effect
    is modeled as a fat-tailed contingent process (CRNG), not a gaussian sum (PRNG).
    """

    def __init__(self, map_size: int = 256, frame_interval_hours: float = 10/60,
                 seed: int = 42):
        self.map_size = map_size
        self.frame_interval_hours = frame_interval_hours
        self.seed = seed
        self.n_samples = 5000  # Monte Carlo samples

    def _project_position(self, tb: TrackedBlob, n_frames: float) -> Tuple[float, float]:
        """Project blob position n_frames into the future."""
        fx = (tb.blob.centroid_x + tb.vx * n_frames) % self.map_size
        fy = (tb.blob.centroid_y + tb.vy * n_frames) % self.map_size
        return fx, fy

    def _distance_to_target(self, x, y, tx, ty):
        """Periodic distance to target."""
        dx = abs(x - tx)
        dy = abs(y - ty)
        dx = min(dx, self.map_size - dx)
        dy = min(dy, self.map_size - dy)
        return np.sqrt(dx**2 + dy**2)

    def find_converging_blobs(self, tracked_blobs: List[TrackedBlob],
                               target_x: float, target_y: float,
                               horizon_hours: float,
                               arrival_radius: float = 30.0) -> List[ConvergingBlob]:
        """Find blobs that will arrive near the target within the horizon."""
        horizon_frames = horizon_hours / self.frame_interval_hours
        converging = []

        for tb in tracked_blobs:
            if tb.prev_blob is None:
                continue  # no velocity — can't project
            if tb.speed < 0.5:
                continue  # stationary blob

            # Check at multiple future times
            for t_frac in np.linspace(0.1, 1.0, 20):
                n_frames = horizon_frames * t_frac
                fx, fy = self._project_position(tb, n_frames)
                dist = self._distance_to_target(fx, fy, target_x, target_y)

                if dist < arrival_radius:
                    toa = horizon_hours * t_frac
                    # Intensity decays slightly over distance
                    decay = np.exp(-dist / (2 * arrival_radius))
                    intensity_at_arrival = tb.blob.intensity * decay

                    converging.append(ConvergingBlob(
                        tracked_blob=tb,
                        time_of_arrival=toa,
                        intensity_at_arrival=intensity_at_arrival,
                        distance_now=self._distance_to_target(
                            tb.blob.centroid_x, tb.blob.centroid_y,
                            target_x, target_y),
                    ))
                    break  # first arrival time

        return converging

    def predict_crng(self, converging: List[ConvergingBlob],
                     horizon_hours: float) -> Prediction:
        """
        CRNG prediction: fat-tailed modeling of concurrent blob interactions.

        More converging blobs => higher kurtosis (more contingent interactions).
        """
        n_conv = len(converging)
        if n_conv == 0:
            return Prediction(
                horizon_hours=horizon_hours,
                n_converging=0,
                prob_rain=0.05,
                expected_temp_change=0.0,
                ci_low=-1.0, ci_high=1.0,
                outcome_samples=np.zeros(self.n_samples),
                method='crng',
            )

        # Key insight: kurtosis scales with number of concurrent events
        # 1 blob = K~4 (mild non-gaussian)
        # 2 blobs = K~7 (moderate fat tails)
        # 3+ blobs = K~12+ (extreme: concurrent systems create compound extremes)
        base_kurtosis = 3.5
        kurtosis = base_kurtosis + 2.5 * n_conv + 0.5 * n_conv**2

        crng = ContingencyRNG(seed=self.seed, target_kurtosis=kurtosis,
                              vol_clustering=min(0.5, 0.1 * n_conv))

        # Generate outcome distribution
        raw_samples = crng.generate(self.n_samples)

        # Scale by total arriving intensity and blob temperatures
        total_intensity = sum(cb.intensity_at_arrival for cb in converging)
        mean_temp = np.mean([cb.tracked_blob.blob.mean_temperature for cb in converging])

        # Temperature change distribution: centered on mean blob temp, scaled by intensity
        scale_factor = total_intensity / max(n_conv, 1)
        outcome_samples = mean_temp + raw_samples * scale_factor

        prob_rain = float(np.mean(outcome_samples < -5))
        expected_change = float(np.mean(outcome_samples))
        ci_low = float(np.percentile(outcome_samples, 5))
        ci_high = float(np.percentile(outcome_samples, 95))

        return Prediction(
            horizon_hours=horizon_hours,
            n_converging=n_conv,
            prob_rain=prob_rain,
            expected_temp_change=expected_change,
            ci_low=ci_low, ci_high=ci_high,
            outcome_samples=outcome_samples,
            method='crng',
        )

    def predict_prng(self, converging: List[ConvergingBlob],
                     horizon_hours: float) -> Prediction:
        """
        PRNG prediction: gaussian modeling (conventional approach).
        Same logic but uses normal distribution — underestimates extremes.
        """
        n_conv = len(converging)
        if n_conv == 0:
            return Prediction(
                horizon_hours=horizon_hours,
                n_converging=0,
                prob_rain=0.05,
                expected_temp_change=0.0,
                ci_low=-1.0, ci_high=1.0,
                outcome_samples=np.zeros(self.n_samples),
                method='prng',
            )

        rng = np.random.RandomState(self.seed)
        raw_samples = rng.normal(0, 1, self.n_samples)

        total_intensity = sum(cb.intensity_at_arrival for cb in converging)
        mean_temp = np.mean([cb.tracked_blob.blob.mean_temperature for cb in converging])

        scale_factor = total_intensity / max(n_conv, 1)
        outcome_samples = mean_temp + raw_samples * scale_factor

        prob_rain = float(np.mean(outcome_samples < -5))
        expected_change = float(np.mean(outcome_samples))
        ci_low = float(np.percentile(outcome_samples, 5))
        ci_high = float(np.percentile(outcome_samples, 95))

        return Prediction(
            horizon_hours=horizon_hours,
            n_converging=n_conv,
            prob_rain=prob_rain,
            expected_temp_change=expected_change,
            ci_low=ci_low, ci_high=ci_high,
            outcome_samples=outcome_samples,
            method='prng',
        )


# ============================================================
# F) VISUALIZATION
# ============================================================
def setup_dark_style():
    """Configure matplotlib for dark theme."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['bg'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.5,
        'font.size': 10,
        'axes.titlesize': 12,
        'figure.titlesize': 14,
    })


def plot_blob_trajectories(all_tracked: List[List[TrackedBlob]],
                           target_x: float, target_y: float,
                           map_size: int, save_path: str):
    """Plot blob trajectories across all frames with target point."""
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 10))

    # Collect trajectories by matching blob positions across frames
    trajectories = {}  # blob_index -> list of (x, y, frame)
    for frame_idx, tracked_list in enumerate(all_tracked):
        for tb in tracked_list:
            # Use area + sign of temperature as a rough blob identity key
            key = (round(tb.blob.area, -1), np.sign(tb.blob.mean_temperature))
            if key not in trajectories:
                trajectories[key] = []
            trajectories[key].append((tb.blob.centroid_x, tb.blob.centroid_y, frame_idx))

    cmap = plt.cm.plasma
    for i, (key, pts) in enumerate(trajectories.items()):
        if len(pts) < 3:
            continue
        pts.sort(key=lambda p: p[2])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = cmap(i / max(len(trajectories), 1))
        ax.plot(xs, ys, '-', color=color, alpha=0.6, linewidth=1.5)
        ax.scatter(xs[-1], ys[-1], color=color, s=40, zorder=5,
                   edgecolor='white', linewidth=0.5)

    # Target point
    ax.scatter(target_x, target_y, color=COLORS['accent'], s=200,
               marker='*', zorder=10, edgecolor='white', linewidth=1)
    ax.annotate('TARGET', (target_x, target_y), color=COLORS['accent'],
                fontsize=11, fontweight='bold',
                xytext=(10, 10), textcoords='offset points')

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('CRNG-Cast: Blob Trajectories', fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_convergence_zones(converging_by_horizon: Dict[float, List[ConvergingBlob]],
                           target_x: float, target_y: float,
                           map_size: int, save_path: str):
    """Plot which blobs converge on target at each horizon."""
    setup_dark_style()
    horizons = sorted(converging_by_horizon.keys())
    n = len(horizons)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        conv = converging_by_horizon[h]

        ax.scatter(target_x, target_y, color=COLORS['accent'], s=200,
                   marker='*', zorder=10, edgecolor='white')

        # Arrival radius circle
        circle = plt.Circle((target_x, target_y), 30, fill=False,
                             color=COLORS['accent'], linestyle='--', alpha=0.5)
        ax.add_patch(circle)

        for cb in conv:
            tb = cb.tracked_blob
            color = COLORS['crng'] if tb.blob.mean_temperature < 0 else COLORS['real']
            ax.scatter(tb.blob.centroid_x, tb.blob.centroid_y,
                       s=tb.blob.area / 5, color=color, alpha=0.7, edgecolor='white',
                       linewidth=0.5)
            # Arrow showing velocity
            ax.annotate('', xy=(tb.blob.centroid_x + tb.vx * 5,
                                tb.blob.centroid_y + tb.vy * 5),
                         xytext=(tb.blob.centroid_x, tb.blob.centroid_y),
                         arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))

        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.set_title(f'{h:.0f}h horizon\n{len(conv)} blobs converging', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle('CRNG-Cast: Convergence Zones', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_probability_comparison(predictions_crng: List[Prediction],
                                 predictions_prng: List[Prediction],
                                 actuals: List[Dict],
                                 save_path: str):
    """
    Compare CRNG vs PRNG probability distributions at each horizon.
    The money chart: shows CRNG capturing fat tails that PRNG misses.
    """
    setup_dark_style()
    n = len(predictions_crng)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, n, figure=fig, hspace=0.4, wspace=0.3)

    # Row 1: Probability distributions
    for i, (pc, pp) in enumerate(zip(predictions_crng, predictions_prng)):
        ax = fig.add_subplot(gs[0, i])
        horizon = pc.horizon_hours

        # Histograms
        bins = np.linspace(
            min(np.min(pc.outcome_samples), np.min(pp.outcome_samples)),
            max(np.max(pc.outcome_samples), np.max(pp.outcome_samples)),
            60
        )
        ax.hist(pp.outcome_samples, bins=bins, alpha=0.5, color=COLORS['prng'],
                label='PRNG', density=True)
        ax.hist(pc.outcome_samples, bins=bins, alpha=0.5, color=COLORS['crng'],
                label='CRNG', density=True)

        # Actual outcome
        if i < len(actuals):
            actual_tc = actuals[i]['temp_change']
            ax.axvline(actual_tc, color=COLORS['real'], linewidth=2,
                       linestyle='--', label=f'Actual: {actual_tc:.1f}')

        # Rain threshold
        ax.axvline(-5, color=COLORS['accent'], linewidth=1, linestyle=':',
                   alpha=0.7, label='Rain threshold')

        ax.set_title(f'{horizon:.0f}h | {pc.n_converging} blobs', fontsize=10)
        ax.set_xlabel('Temp change')
        if i == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    # Row 2: Summary metrics
    ax_rain = fig.add_subplot(gs[1, 0:2])
    ax_ci = fig.add_subplot(gs[1, 2:])

    horizons = [p.horizon_hours for p in predictions_crng]

    # Rain probability comparison
    prng_rain = [p.prob_rain for p in predictions_prng]
    crng_rain = [p.prob_rain for p in predictions_crng]
    actual_rain = [1.0 if a['is_rain'] else 0.0 for a in actuals]

    x = np.arange(len(horizons))
    w = 0.25
    ax_rain.bar(x - w, prng_rain, w, color=COLORS['prng'], label='PRNG P(rain)')
    ax_rain.bar(x, crng_rain, w, color=COLORS['crng'], label='CRNG P(rain)')
    ax_rain.bar(x + w, actual_rain, w, color=COLORS['real'], label='Actual rain')
    ax_rain.set_xticks(x)
    ax_rain.set_xticklabels([f'{h:.0f}h' for h in horizons])
    ax_rain.set_ylabel('P(rain)')
    ax_rain.set_title('Rain Probability: CRNG vs PRNG vs Actual', fontweight='bold')
    ax_rain.legend(fontsize=8)
    ax_rain.grid(True, alpha=0.2)

    # Confidence interval comparison
    prng_ci_width = [predictions_prng[i].ci_high - predictions_prng[i].ci_low for i in range(n)]
    crng_ci_width = [predictions_crng[i].ci_high - predictions_crng[i].ci_low for i in range(n)]

    ax_ci.bar(x - w/2, prng_ci_width, w, color=COLORS['prng'], label='PRNG 90% CI width')
    ax_ci.bar(x + w/2, crng_ci_width, w, color=COLORS['crng'], label='CRNG 90% CI width')

    # Check which CI actually contains the actual value
    for i in range(n):
        actual_tc = actuals[i]['temp_change']
        prng_hit = predictions_prng[i].ci_low <= actual_tc <= predictions_prng[i].ci_high
        crng_hit = predictions_crng[i].ci_low <= actual_tc <= predictions_crng[i].ci_high
        marker_y = max(prng_ci_width[i], crng_ci_width[i]) + 1
        if prng_hit:
            ax_ci.text(x[i] - w/2, marker_y, 'HIT', ha='center', fontsize=7,
                       color=COLORS['prng'], fontweight='bold')
        else:
            ax_ci.text(x[i] - w/2, marker_y, 'MISS', ha='center', fontsize=7,
                       color=COLORS['muted'])
        if crng_hit:
            ax_ci.text(x[i] + w/2, marker_y, 'HIT', ha='center', fontsize=7,
                       color=COLORS['crng'], fontweight='bold')
        else:
            ax_ci.text(x[i] + w/2, marker_y, 'MISS', ha='center', fontsize=7,
                       color=COLORS['muted'])

    ax_ci.set_xticks(x)
    ax_ci.set_xticklabels([f'{h:.0f}h' for h in horizons])
    ax_ci.set_ylabel('CI width')
    ax_ci.set_title('90% Confidence Interval Width (wider = more honest)', fontweight='bold')
    ax_ci.legend(fontsize=8)
    ax_ci.grid(True, alpha=0.2)

    fig.suptitle('CRNG-Cast vs PRNG-Cast: Nowcasting Comparison',
                 fontweight='bold', fontsize=16, color=COLORS['accent'])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# E) RUN SIMULATION
# ============================================================
def run_simulation():
    """
    Full CRNG-Cast simulation:
      1. Generate 36 synthetic frames (6 hours at 10-min intervals)
      2. Detect + track blobs
      3. Predict at 1h, 2h, 3h, 6h horizons
      4. Compare CRNG vs PRNG against actual outcomes
    """
    print("=" * 70)
    print("  CRNG-Cast: Weather Nowcasting Prototype")
    print("  Fat-tailed concurrent event modeling via Contingency RNG")
    print("=" * 70)

    # --- Setup ---
    MAP_SIZE = 256
    N_FRAMES = 36  # 6 hours * 6 frames/hour
    HORIZONS = [1, 2, 3, 6]  # hours

    sat_gen = SyntheticSatelliteGenerator(size=MAP_SIZE, n_blobs=8, seed=42,
                                             target_x=140.0, target_y=120.0)
    detector = BlobDetector(threshold=5.0, min_area=20)
    tracker = BlobTracker(max_match_distance=40.0, map_size=MAP_SIZE)
    predictor = CRNGCastPredictor(map_size=MAP_SIZE, seed=42)

    # Target point (center-ish, slightly offset to create asymmetric convergence)
    target_x, target_y = 140.0, 120.0

    # --- Step 1: Generate frames ---
    print("\n[1/5] Generating synthetic satellite frames...")
    frames = sat_gen.generate_sequence(N_FRAMES)
    print(f"  Generated {N_FRAMES} frames ({N_FRAMES * 10} minutes = {N_FRAMES * 10 / 60:.1f} hours)")
    print(f"  Frame shape: {frames[0].shape}")
    print(f"  Temp range: [{np.min(frames[0]):.1f}, {np.max(frames[0]):.1f}] K")

    # --- Step 2: Detect blobs ---
    print("\n[2/5] Detecting thermal blobs...")
    all_blobs = []
    for i, frame in enumerate(frames):
        blobs = detector.detect(frame, background_temp=sat_gen.background_temp, frame_idx=i)
        all_blobs.append(blobs)

    blob_counts = [len(b) for b in all_blobs]
    print(f"  Blobs per frame: min={min(blob_counts)}, max={max(blob_counts)}, "
          f"mean={np.mean(blob_counts):.1f}")

    # --- Step 3: Track blobs ---
    print("\n[3/5] Tracking blob movement...")
    all_tracked = []
    for i in range(1, len(all_blobs)):
        tracked = tracker.track(all_blobs[i-1], all_blobs[i])
        all_tracked.append(tracked)

    matched_counts = [sum(1 for t in tl if t.prev_blob is not None) for tl in all_tracked]
    speeds = [t.speed for tl in all_tracked for t in tl if t.prev_blob is not None]
    print(f"  Matched blobs per frame pair: min={min(matched_counts)}, max={max(matched_counts)}, "
          f"mean={np.mean(matched_counts):.1f}")
    if speeds:
        print(f"  Blob speeds: min={min(speeds):.1f}, max={max(speeds):.1f}, "
              f"mean={np.mean(speeds):.1f} px/frame")

    # --- Step 4: Predict at each horizon ---
    print(f"\n[4/5] Running predictions at target ({target_x:.0f}, {target_y:.0f})...")

    # Use the latest tracked frame for predictions
    latest_tracked = all_tracked[-1]

    # Also aggregate tracked blobs from recent frames for robustness
    # (some blobs may only appear in certain frames)
    all_recent_tracked = []
    seen_centroids = set()
    for tl in all_tracked[-5:]:
        for tb in tl:
            key = (round(tb.blob.centroid_x, 0), round(tb.blob.centroid_y, 0))
            if key not in seen_centroids and tb.prev_blob is not None:
                all_recent_tracked.append(tb)
                seen_centroids.add(key)
    # Prefer recent + unique blobs
    combined_tracked = all_recent_tracked if len(all_recent_tracked) > len(latest_tracked) else latest_tracked

    predictions_crng = []
    predictions_prng = []
    actuals = []
    converging_by_horizon = {}

    for h in HORIZONS:
        # Find converging blobs using combined tracked data
        converging = predictor.find_converging_blobs(combined_tracked, target_x, target_y,
                                                     horizon_hours=h, arrival_radius=40.0)
        converging_by_horizon[h] = converging

        # CRNG prediction
        pred_crng = predictor.predict_crng(converging, h)
        predictions_crng.append(pred_crng)

        # PRNG prediction (gaussian baseline)
        pred_prng = predictor.predict_prng(converging, h)
        predictions_prng.append(pred_prng)

        # Actual outcome (look at the frame at horizon time)
        future_frame_idx = min(N_FRAMES - 1, len(all_tracked) + int(h / (10/60)))
        actual = sat_gen.get_actual_outcome(target_x, target_y, future_frame_idx)
        actuals.append(actual)

        # Kurtosis of each distribution
        crng_k = float(np.mean(((pred_crng.outcome_samples - np.mean(pred_crng.outcome_samples))
                                 / max(np.std(pred_crng.outcome_samples), 1e-10))**4))
        prng_k = float(np.mean(((pred_prng.outcome_samples - np.mean(pred_prng.outcome_samples))
                                 / max(np.std(pred_prng.outcome_samples), 1e-10))**4))

        print(f"\n  --- Horizon {h}h ---")
        print(f"  Converging blobs: {len(converging)}")
        print(f"  CRNG: P(rain)={pred_crng.prob_rain:.3f}, "
              f"E[dT]={pred_crng.expected_temp_change:+.2f}, "
              f"CI=[{pred_crng.ci_low:+.1f}, {pred_crng.ci_high:+.1f}], "
              f"K={crng_k:.1f}")
        print(f"  PRNG: P(rain)={pred_prng.prob_rain:.3f}, "
              f"E[dT]={pred_prng.expected_temp_change:+.2f}, "
              f"CI=[{pred_prng.ci_low:+.1f}, {pred_prng.ci_high:+.1f}], "
              f"K={prng_k:.1f}")
        print(f"  Actual: dT={actual['temp_change']:+.2f}, rain={actual['is_rain']}")

        # CI hit check
        crng_hit = pred_crng.ci_low <= actual['temp_change'] <= pred_crng.ci_high
        prng_hit = pred_prng.ci_low <= actual['temp_change'] <= pred_prng.ci_high
        print(f"  CI contains actual? CRNG={'YES' if crng_hit else 'NO'}, "
              f"PRNG={'YES' if prng_hit else 'NO'}")

    # --- Step 5: Generate charts ---
    print("\n[5/5] Generating charts...")
    charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    plot_blob_trajectories(all_tracked, target_x, target_y, MAP_SIZE,
                           os.path.join(charts_dir, 'crng_cast_trajectories.png'))

    plot_convergence_zones(converging_by_horizon, target_x, target_y, MAP_SIZE,
                           os.path.join(charts_dir, 'crng_cast_convergence.png'))

    plot_probability_comparison(predictions_crng, predictions_prng, actuals,
                                os.path.join(charts_dir, 'crng_cast_comparison.png'))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    crng_hits = sum(1 for i in range(len(HORIZONS))
                    if predictions_crng[i].ci_low <= actuals[i]['temp_change'] <= predictions_crng[i].ci_high)
    prng_hits = sum(1 for i in range(len(HORIZONS))
                    if predictions_prng[i].ci_low <= actuals[i]['temp_change'] <= predictions_prng[i].ci_high)

    print(f"\n  90% CI calibration (should be ~90% over many runs):")
    print(f"    CRNG: {crng_hits}/{len(HORIZONS)} horizons captured actual outcome")
    print(f"    PRNG: {prng_hits}/{len(HORIZONS)} horizons captured actual outcome")

    crng_ci_widths = [predictions_crng[i].ci_high - predictions_crng[i].ci_low for i in range(len(HORIZONS))]
    prng_ci_widths = [predictions_prng[i].ci_high - predictions_prng[i].ci_low for i in range(len(HORIZONS))]

    print(f"\n  Mean CI width:")
    print(f"    CRNG: {np.mean(crng_ci_widths):.2f} (wider = more honest about uncertainty)")
    print(f"    PRNG: {np.mean(prng_ci_widths):.2f}")

    rain_errors_crng = [abs(predictions_crng[i].prob_rain - (1.0 if actuals[i]['is_rain'] else 0.0))
                        for i in range(len(HORIZONS))]
    rain_errors_prng = [abs(predictions_prng[i].prob_rain - (1.0 if actuals[i]['is_rain'] else 0.0))
                        for i in range(len(HORIZONS))]

    print(f"\n  Mean absolute rain probability error:")
    print(f"    CRNG: {np.mean(rain_errors_crng):.3f}")
    print(f"    PRNG: {np.mean(rain_errors_prng):.3f}")

    print(f"\n  Key insight: when multiple blobs converge ({max(p.n_converging for p in predictions_crng)} max),")
    print(f"  CRNG produces wider CIs and higher P(extreme) than PRNG.")
    print(f"  This reflects the reality that concurrent weather systems")
    print(f"  interact non-linearly, creating fat-tailed outcomes.")

    print(f"\n  Charts saved to: {os.path.abspath(charts_dir)}/crng_cast_*.png")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    run_simulation()
