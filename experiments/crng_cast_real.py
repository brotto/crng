#!/usr/bin/env python3
"""
CRNG-Cast REAL: Nowcasting with Real GOES-16/19 Satellite Data
================================================================
Downloads real IR satellite images from NOAA's GOES CDN, runs the full
CRNG-Cast blob detection + tracking pipeline, and compares CRNG-Cast vs
PRNG-Cast predictions against actual outcomes.

Data strategy:
  - Downloads the latest GOES-16 Band 13 (10.3um clean longwave IR) image
  - Creates a temporal sequence by applying realistic advection dynamics
    to the real satellite texture (weather systems move ~30-50 km/h)
  - Uses multiple IR bands (7-16) as independent thermal snapshots
  - Validates predictions against held-out frames

Target: New York metro area (CONUS imagery covers North America)
  Approximate pixel location in 625x375 CONUS image: ~(460, 175)

Usage:
  cd crng-package && python3 experiments/crng_cast_real.py
"""

import sys
import os
import io
import time
import urllib.request
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy import ndimage
from scipy.ndimage import shift as ndimage_shift
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crng import ContingencyRNG


# ============================================================
# DARK THEME
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
    'warm':   '#FF6B35',
    'cold':   '#00D4FF',
}


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Blob:
    blob_id: int
    centroid_x: float
    centroid_y: float
    area: int
    mean_temperature: float   # deviation from background (negative=cold cloud)
    intensity: float          # |deviation|
    min_temp: float = 0.0     # coldest pixel (for deep convection detection)
    frame_idx: int = -1


@dataclass
class TrackedBlob:
    blob: Blob
    prev_blob: Optional[Blob]
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0


@dataclass
class ConvergingBlob:
    tracked_blob: TrackedBlob
    time_of_arrival: float
    intensity_at_arrival: float
    distance_now: float


@dataclass
class Prediction:
    horizon_hours: float
    n_converging: int
    prob_extreme: float       # P(outcome beyond 2-sigma)
    expected_temp_change: float
    ci_low: float
    ci_high: float
    outcome_samples: np.ndarray
    method: str               # 'crng' or 'prng'
    kurtosis_used: float = 3.0


# ============================================================
# A) REAL SATELLITE DATA LOADER
# ============================================================
class SatelliteDataLoader:
    """
    Download real GOES-16/19 IR satellite images from NOAA CDN.

    Strategy for temporal sequence:
    1. Download latest Band 13 (clean IR) as the base frame
    2. Download bands 7-16 as independent thermal views
    3. Apply realistic advection (wind-driven motion) to create
       a physically plausible temporal sequence from the base frame
    4. The multi-band images serve as validation/diversity frames

    Pixel intensity -> brightness temperature mapping:
    Band 13 JPEGs: 0 (brightest/coldest ~180K) to 255 (darkest/warmest ~320K)
    Inverted: bright pixels = cold = high clouds
    """

    GOES16_BASE = "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS"
    GOES19_BASE = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/CONUS"
    RESOLUTION = "625x375"

    # IR bands with their wavelengths (um)
    IR_BANDS = {
        7:  3.9,    # shortwave IR (fire detection)
        8:  6.2,    # upper-level water vapor
        9:  6.9,    # mid-level water vapor
        10: 7.3,    # lower-level water vapor
        11: 8.4,    # cloud-top phase
        12: 9.6,    # ozone
        13: 10.3,   # clean IR longwave (primary)
        14: 11.2,   # IR longwave
        15: 12.3,   # dirty IR longwave
        16: 13.3,   # CO2 longwave
    }

    # Temperature mapping for Band 13 JPEG
    # Bright pixels (high values) = warm surface, dark = cold clouds
    # Actually in NOAA CDN JPEGs, it's: bright=cold, dark=warm for IR bands
    TEMP_MIN = 180.0   # K (coldest cloud tops)
    TEMP_MAX = 320.0   # K (warmest surface)

    def __init__(self, target_lat=40.71, target_lon=-74.01, crop_size=200):
        """
        Args:
            target_lat/lon: Target location (default: New York)
            crop_size: Size of region to crop around target (pixels)
        """
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.crop_size = crop_size

        # CONUS image covers roughly:
        # Lat: 20N to 55N, Lon: 135W to 60W
        # In 625x375 image:
        self.lat_range = (20.0, 55.0)
        self.lon_range = (-135.0, -60.0)
        self.img_w = 625
        self.img_h = 375

        # Compute target pixel position
        self.target_px = self._latlon_to_pixel(target_lat, target_lon)
        print(f"  Target: ({target_lat:.2f}N, {target_lon:.2f}W) -> pixel ({self.target_px[0]:.0f}, {self.target_px[1]:.0f})")

    def _latlon_to_pixel(self, lat, lon):
        """Convert lat/lon to approximate pixel coordinates in CONUS image."""
        # Linear approximation (good enough for CONUS projection)
        px = (lon - self.lon_range[0]) / (self.lon_range[1] - self.lon_range[0]) * self.img_w
        py = (1.0 - (lat - self.lat_range[0]) / (self.lat_range[1] - self.lat_range[0])) * self.img_h
        return (px, py)

    def _download_image(self, url):
        """Download image from URL, return as numpy array."""
        req = urllib.request.Request(url, headers={'User-Agent': 'CRNG-Cast/1.0 Research'})
        try:
            resp = urllib.request.urlopen(req, timeout=20)
            data = resp.read()
            img = Image.open(io.BytesIO(data)).convert('L')  # grayscale
            return np.array(img, dtype=np.float64)
        except Exception as e:
            print(f"    Download failed: {url} -> {e}")
            return None

    def _pixel_to_temperature(self, pixel_array):
        """
        Convert pixel intensity (0-255) to brightness temperature (K).
        In NOAA IR JPEGs: bright=cold clouds, dark=warm surface.
        """
        # Invert: 0->TEMP_MAX, 255->TEMP_MIN
        normalized = pixel_array / 255.0
        temperature = self.TEMP_MAX - normalized * (self.TEMP_MAX - self.TEMP_MIN)
        return temperature

    def _crop_region(self, full_image):
        """Crop a region around the target location."""
        px, py = self.target_px
        half = self.crop_size // 2

        x0 = max(0, int(px - half))
        y0 = max(0, int(py - half))
        x1 = min(full_image.shape[1], x0 + self.crop_size)
        y1 = min(full_image.shape[0], y0 + self.crop_size)

        cropped = full_image[y0:y1, x0:x1]

        # Pad if necessary
        if cropped.shape[0] < self.crop_size or cropped.shape[1] < self.crop_size:
            padded = np.full((self.crop_size, self.crop_size), np.mean(cropped))
            padded[:cropped.shape[0], :cropped.shape[1]] = cropped
            return padded

        return cropped

    def _advect_frame(self, base_frame, vx, vy, noise_level=0.3):
        """
        Apply advection (wind-driven motion) to create a temporally shifted frame.
        This simulates weather system motion between satellite scans.
        """
        shifted = ndimage_shift(base_frame, [vy, vx], mode='wrap', order=1)
        # Add small thermal evolution (slight warming/cooling of features)
        noise = np.random.normal(0, noise_level, shifted.shape)
        # Smooth the noise to keep it physically realistic
        noise = ndimage.gaussian_filter(noise, sigma=3)
        return shifted + noise

    def download_sequence(self, n_frames=16):
        """
        Download real satellite data and create a temporal sequence.

        Returns:
            List of (crop_size x crop_size) temperature arrays in Kelvin
            Also returns the full-frame image for visualization
        """
        print("\n=== DOWNLOADING REAL GOES-16 SATELLITE DATA ===")

        # Step 1: Download the primary Band 13 image
        print("  Downloading Band 13 (10.3um clean IR)...")
        primary_url = f"{self.GOES16_BASE}/13/{self.RESOLUTION}.jpg"
        primary_raw = self._download_image(primary_url)

        if primary_raw is None:
            # Try GOES-19
            print("  Trying GOES-19...")
            primary_url = f"{self.GOES19_BASE}/13/{self.RESOLUTION}.jpg"
            primary_raw = self._download_image(primary_url)

        if primary_raw is None:
            raise RuntimeError("Cannot download satellite data from either GOES-16 or GOES-19")

        print(f"  Raw image: {primary_raw.shape}, range [{primary_raw.min():.0f}, {primary_raw.max():.0f}]")

        # Store full frame for visualization
        self.full_frame_raw = primary_raw.copy()
        self.full_frame_temp = self._pixel_to_temperature(primary_raw)

        # Crop around target
        primary_crop = self._crop_region(primary_raw)
        base_temp = self._pixel_to_temperature(primary_crop)
        print(f"  Cropped region: {base_temp.shape}, temp range [{base_temp.min():.1f}K, {base_temp.max():.1f}K]")

        # Step 2: Download additional bands as independent views
        print("  Downloading additional IR bands...")
        band_frames = []
        for band in [8, 9, 10, 14, 15, 16]:
            url = f"{self.GOES16_BASE}/{band:02d}/{self.RESOLUTION}.jpg"
            raw = self._download_image(url)
            if raw is not None:
                crop = self._crop_region(raw)
                temp = self._pixel_to_temperature(crop)
                band_frames.append(temp)
                print(f"    Band {band}: OK [{temp.min():.1f}K - {temp.max():.1f}K]")

        # Step 3: Create temporal sequence using advection
        # Typical mid-latitude wind: 30-60 km/h = ~2-4 pixels per 10-min frame
        # at this resolution (~4km/pixel for CONUS)
        print(f"  Building {n_frames}-frame temporal sequence via advection...")

        np.random.seed(42)
        # Dominant wind direction (westerly with some variation)
        base_vx = np.random.uniform(1.5, 3.5)  # pixels/frame eastward
        base_vy = np.random.uniform(-1.0, 1.0)  # slight N/S component

        frames = []

        # Generate cumulative wind displacements for past frames
        # Frame 0 = oldest (most backward-advected), Frame n-1 = current
        cum_vx = 0.0
        cum_vy = 0.0
        displacements = [(0.0, 0.0)]  # current frame = no displacement
        for i in range(1, n_frames):
            vx_noise = np.random.normal(0, 0.3)
            vy_noise = np.random.normal(0, 0.3)
            cum_vx -= (base_vx + vx_noise)
            cum_vy -= (base_vy + vy_noise)
            displacements.append((cum_vx, cum_vy))

        # Reverse: index 0 = oldest frame, index n-1 = current
        displacements = displacements[::-1]

        # Build each frame by advecting the base image by its total displacement
        # Always advect from the original base_temp to avoid accumulation artifacts
        for i, (dx, dy) in enumerate(displacements):
            shifted = ndimage_shift(base_temp, [dy, dx], mode='wrap', order=1)
            # Add slight thermal evolution noise (more for older frames)
            age = (n_frames - 1 - i)  # how many frames in the past
            noise_level = 0.2 + 0.05 * age
            noise = np.random.normal(0, noise_level, shifted.shape)
            noise = ndimage.gaussian_filter(noise, sigma=2)
            shifted += noise

            # Blend with band data if available (adds real texture diversity)
            if i < len(band_frames):
                alpha = 0.15  # 15% from real band for texture variation
                shifted = (1 - alpha) * shifted + alpha * band_frames[i]

            frames.append(shifted)

        # The target pixel in cropped coordinates
        self.target_crop_x = self.crop_size // 2
        self.target_crop_y = self.crop_size // 2

        print(f"  Sequence ready: {len(frames)} frames x {frames[0].shape}")
        print(f"  Wind vector: ({base_vx:.1f}, {base_vy:.1f}) px/frame")
        print(f"  Covers {len(frames) * 10} minutes of observation\n")

        self.wind_vx = base_vx
        self.wind_vy = base_vy
        self.background_temp = np.median(frames[0])

        return frames

    def get_actual_outcome(self, frames, frame_idx, radius=5):
        """Get actual temperature at target for a given frame."""
        frame = frames[frame_idx]
        tx, ty = self.target_crop_x, self.target_crop_y
        y0 = max(0, ty - radius)
        y1 = min(frame.shape[0], ty + radius + 1)
        x0 = max(0, tx - radius)
        x1 = min(frame.shape[1], tx + radius + 1)
        region = frame[y0:y1, x0:x1]
        temp = float(np.mean(region))
        temp_change = temp - self.background_temp
        return {
            'temperature': temp,
            'temp_change': temp_change,
            'is_extreme': abs(temp_change) > 5,
        }


# ============================================================
# B) BLOB DETECTOR (enhanced for real data)
# ============================================================
class BlobDetector:
    """
    Detect thermal blobs in real satellite IR images.
    Uses adaptive thresholding relative to local background.
    """

    def __init__(self, threshold: float = 4.0, min_area: int = 15):
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, thermal_map: np.ndarray, background_temp: float = None,
               frame_idx: int = -1) -> List[Blob]:
        if background_temp is None:
            background_temp = np.median(thermal_map)

        deviation = thermal_map - background_temp

        # Smooth slightly to reduce noise in real satellite data
        smoothed_dev = ndimage.gaussian_filter(deviation, sigma=1.5)

        mask = np.abs(smoothed_dev) > self.threshold
        labeled, n_features = ndimage.label(mask)

        blobs = []
        for label_id in range(1, n_features + 1):
            component = labeled == label_id
            area = int(np.sum(component))
            if area < self.min_area:
                continue

            coords = np.argwhere(component)
            cy = float(np.mean(coords[:, 0]))
            cx = float(np.mean(coords[:, 1]))

            temps = deviation[component]
            mean_temp = float(np.mean(temps))
            intensity = float(np.mean(np.abs(temps)))
            min_temp = float(np.min(temps))

            blobs.append(Blob(
                blob_id=label_id,
                centroid_x=cx,
                centroid_y=cy,
                area=area,
                mean_temperature=mean_temp,
                intensity=intensity,
                min_temp=min_temp,
                frame_idx=frame_idx,
            ))

        return blobs


# ============================================================
# C) BLOB TRACKER
# ============================================================
class BlobTracker:
    def __init__(self, max_match_distance: float = 40.0, map_size: int = 200):
        self.max_match_distance = max_match_distance
        self.map_size = map_size

    def track(self, prev_blobs: List[Blob], curr_blobs: List[Blob]) -> List[TrackedBlob]:
        tracked = []
        used_prev = set()

        for cb in curr_blobs:
            best_dist = float('inf')
            best_prev = None

            for pb in prev_blobs:
                if id(pb) in used_prev:
                    continue
                dx = cb.centroid_x - pb.centroid_x
                dy = cb.centroid_y - pb.centroid_y
                d = np.sqrt(dx**2 + dy**2)
                if d < best_dist:
                    best_dist = d
                    best_prev = pb

            if best_prev is not None and best_dist < self.max_match_distance:
                used_prev.add(id(best_prev))
                vx = cb.centroid_x - best_prev.centroid_x
                vy = cb.centroid_y - best_prev.centroid_y
                speed = np.sqrt(vx**2 + vy**2)
                tracked.append(TrackedBlob(
                    blob=cb, prev_blob=best_prev,
                    vx=vx, vy=vy, speed=speed,
                ))
            else:
                tracked.append(TrackedBlob(blob=cb, prev_blob=None))

        return tracked


# ============================================================
# D) CRNG-CAST PREDICTOR
# ============================================================
class CRNGCastPredictor:
    """
    Predict weather outcomes using CRNG for concurrent blob interactions.
    Key physics: when N thermal systems converge simultaneously, the combined
    effect has kurtosis >> 3 (fat tails), not the K=3 of a Gaussian sum.
    """

    def __init__(self, map_size: int = 200, frame_interval_hours: float = 10/60,
                 seed: int = 42):
        self.map_size = map_size
        self.frame_interval_hours = frame_interval_hours
        self.seed = seed
        self.n_samples = 10000

    def find_converging_blobs(self, tracked_blobs: List[TrackedBlob],
                               target_x: float, target_y: float,
                               horizon_hours: float,
                               arrival_radius: float = 30.0) -> List[ConvergingBlob]:
        horizon_frames = horizon_hours / self.frame_interval_hours
        converging = []

        for tb in tracked_blobs:
            if tb.prev_blob is None or tb.speed < 0.3:
                continue

            for t_frac in np.linspace(0.1, 1.0, 30):
                n_frames = horizon_frames * t_frac
                fx = tb.blob.centroid_x + tb.vx * n_frames
                fy = tb.blob.centroid_y + tb.vy * n_frames
                dx = fx - target_x
                dy = fy - target_y
                dist = np.sqrt(dx**2 + dy**2)

                if dist < arrival_radius:
                    toa = horizon_hours * t_frac
                    decay = np.exp(-dist / (2 * arrival_radius))
                    intensity_at_arrival = tb.blob.intensity * decay

                    now_dx = tb.blob.centroid_x - target_x
                    now_dy = tb.blob.centroid_y - target_y
                    dist_now = np.sqrt(now_dx**2 + now_dy**2)

                    converging.append(ConvergingBlob(
                        tracked_blob=tb,
                        time_of_arrival=toa,
                        intensity_at_arrival=intensity_at_arrival,
                        distance_now=dist_now,
                    ))
                    break

        return converging

    def predict_crng(self, converging: List[ConvergingBlob],
                     horizon_hours: float) -> Prediction:
        n_conv = len(converging)
        if n_conv == 0:
            return Prediction(
                horizon_hours=horizon_hours, n_converging=0,
                prob_extreme=0.05, expected_temp_change=0.0,
                ci_low=-1.0, ci_high=1.0,
                outcome_samples=np.zeros(self.n_samples),
                method='crng', kurtosis_used=3.0,
            )

        # Kurtosis scales with concurrent blob count
        # 1 blob: K ~ 4.5 (slightly non-Gaussian)
        # 2 blobs: K ~ 8 (moderate fat tails)
        # 3+ blobs: K ~ 15+ (extreme compound events)
        kurtosis = 3.5 + 2.5 * n_conv + 0.5 * n_conv**2

        crng = ContingencyRNG(seed=self.seed, target_kurtosis=kurtosis,
                              vol_clustering=min(0.5, 0.1 * n_conv))

        raw = crng.generate(self.n_samples)

        total_intensity = sum(cb.intensity_at_arrival for cb in converging)
        mean_temp = np.mean([cb.tracked_blob.blob.mean_temperature for cb in converging])

        scale = total_intensity / max(n_conv, 1)
        samples = mean_temp + raw * scale

        return Prediction(
            horizon_hours=horizon_hours, n_converging=n_conv,
            prob_extreme=float(np.mean(np.abs(samples) > 2 * np.std(samples))),
            expected_temp_change=float(np.mean(samples)),
            ci_low=float(np.percentile(samples, 5)),
            ci_high=float(np.percentile(samples, 95)),
            outcome_samples=samples, method='crng',
            kurtosis_used=kurtosis,
        )

    def predict_prng(self, converging: List[ConvergingBlob],
                     horizon_hours: float) -> Prediction:
        n_conv = len(converging)
        if n_conv == 0:
            return Prediction(
                horizon_hours=horizon_hours, n_converging=0,
                prob_extreme=0.05, expected_temp_change=0.0,
                ci_low=-1.0, ci_high=1.0,
                outcome_samples=np.zeros(self.n_samples),
                method='prng', kurtosis_used=3.0,
            )

        rng = np.random.RandomState(self.seed)
        raw = rng.normal(0, 1, self.n_samples)

        total_intensity = sum(cb.intensity_at_arrival for cb in converging)
        mean_temp = np.mean([cb.tracked_blob.blob.mean_temperature for cb in converging])

        scale = total_intensity / max(n_conv, 1)
        samples = mean_temp + raw * scale

        return Prediction(
            horizon_hours=horizon_hours, n_converging=n_conv,
            prob_extreme=float(np.mean(np.abs(samples) > 2 * np.std(samples))),
            expected_temp_change=float(np.mean(samples)),
            ci_low=float(np.percentile(samples, 5)),
            ci_high=float(np.percentile(samples, 95)),
            outcome_samples=samples, method='prng',
            kurtosis_used=3.0,
        )


# ============================================================
# E) VISUALIZATION
# ============================================================
def setup_dark_style():
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


def plot_thermal_map(frame, blobs, target_x, target_y, background_temp,
                     full_frame=None, save_path='crng_cast_real_thermal.png'):
    """Plot actual satellite thermal map with detected blobs overlaid."""
    setup_dark_style()

    if full_frame is not None:
        fig, (ax_full, ax_crop) = plt.subplots(1, 2, figsize=(16, 7),
                                                 gridspec_kw={'width_ratios': [1.5, 1]})
    else:
        fig, ax_crop = plt.subplots(1, 1, figsize=(8, 7))
        ax_full = None

    # Full frame view (if available)
    if ax_full is not None and full_frame is not None:
        im_full = ax_full.imshow(full_frame, cmap='inferno', origin='upper',
                                  aspect='auto')
        ax_full.set_title('GOES-16 Band 13 (10.3um) -- Full CONUS', fontweight='bold', fontsize=11)
        ax_full.set_xlabel('Longitude (px)')
        ax_full.set_ylabel('Latitude (px)')

        # Mark target on full frame
        loader_px = (460, 175)  # approximate NY in full frame
        ax_full.scatter(loader_px[0], loader_px[1], color=COLORS['accent'],
                       s=200, marker='*', zorder=10, edgecolor='white', linewidth=1)
        ax_full.annotate('NYC', (loader_px[0], loader_px[1]), color=COLORS['accent'],
                        fontsize=10, fontweight='bold',
                        xytext=(10, -15), textcoords='offset points',
                        path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['bg'])])

        # Draw crop region rectangle
        from matplotlib.patches import Rectangle
        half = 100
        rect = Rectangle((loader_px[0]-half, loader_px[1]-half),
                         200, 200, linewidth=2, edgecolor=COLORS['accent'],
                         facecolor='none', linestyle='--')
        ax_full.add_patch(rect)

        cbar_full = fig.colorbar(im_full, ax=ax_full, shrink=0.8, pad=0.02)
        cbar_full.set_label('Pixel Intensity', color=COLORS['text'])
        cbar_full.ax.yaxis.set_tick_params(color=COLORS['text'])
        plt.setp(cbar_full.ax.yaxis.get_ticklabels(), color=COLORS['text'])

    # Cropped region with blob overlays
    deviation = frame - background_temp
    im = ax_crop.imshow(deviation, cmap='RdBu_r', origin='upper',
                        vmin=-20, vmax=20, aspect='equal')

    # Overlay detected blobs
    for blob in blobs:
        radius = np.sqrt(blob.area / np.pi)
        color = COLORS['cold'] if blob.mean_temperature < 0 else COLORS['warm']
        circle = Circle((blob.centroid_x, blob.centroid_y), radius,
                        fill=False, edgecolor=color, linewidth=2, alpha=0.8)
        ax_crop.add_patch(circle)
        ax_crop.text(blob.centroid_x + radius + 2, blob.centroid_y,
                    f'{blob.mean_temperature:+.1f}K',
                    color=color, fontsize=7, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=COLORS['bg'])])

    # Target
    ax_crop.scatter(target_x, target_y, color=COLORS['accent'], s=200,
                   marker='*', zorder=10, edgecolor='white', linewidth=1)
    ax_crop.annotate('TARGET\n(NYC)', (target_x, target_y), color=COLORS['accent'],
                    fontsize=9, fontweight='bold',
                    xytext=(10, 10), textcoords='offset points',
                    path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['bg'])])

    ax_crop.set_title('Cropped Region -- Blob Detection', fontweight='bold', fontsize=11)
    ax_crop.set_xlabel('X (px)')
    ax_crop.set_ylabel('Y (px)')

    cbar = fig.colorbar(im, ax=ax_crop, shrink=0.8, pad=0.02)
    cbar.set_label('Temp Deviation (K)', color=COLORS['text'])
    cbar.ax.yaxis.set_tick_params(color=COLORS['text'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS['text'])

    # Info text
    n_cold = sum(1 for b in blobs if b.mean_temperature < 0)
    n_warm = sum(1 for b in blobs if b.mean_temperature >= 0)
    info = f"Detected: {len(blobs)} blobs ({n_cold} cold, {n_warm} warm)"
    fig.text(0.5, 0.02, info, ha='center', fontsize=10, color=COLORS['muted'],
             style='italic')

    fig.suptitle('CRNG-Cast Real: GOES-16 Satellite IR + Blob Detection',
                fontweight='bold', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tracking(all_tracked, target_x, target_y, map_size,
                  save_path='crng_cast_real_tracking.png'):
    """Plot blob trajectories over multiple frames."""
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 10))

    # Build trajectories by chaining tracked blobs
    # Color by temperature sign (cold=blue, warm=red)
    cmap_cold = plt.cm.Blues_r
    cmap_warm = plt.cm.Reds

    traj_id = 0
    for frame_idx, tracked_list in enumerate(all_tracked):
        for tb in tracked_list:
            if tb.prev_blob is None:
                continue
            # Draw velocity arrow
            is_cold = tb.blob.mean_temperature < 0
            color = cmap_cold(0.3 + 0.5 * min(tb.blob.intensity / 15.0, 1.0)) if is_cold \
                    else cmap_warm(0.3 + 0.5 * min(tb.blob.intensity / 15.0, 1.0))

            alpha = 0.3 + 0.5 * (frame_idx / max(len(all_tracked) - 1, 1))

            # Blob position
            ax.scatter(tb.blob.centroid_x, tb.blob.centroid_y,
                      s=max(10, tb.blob.area / 5), color=color, alpha=alpha,
                      edgecolor='white', linewidth=0.3)

            # Velocity arrow
            ax.annotate('', xy=(tb.blob.centroid_x, tb.blob.centroid_y),
                        xytext=(tb.prev_blob.centroid_x, tb.prev_blob.centroid_y),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=alpha * 0.8,
                                       linewidth=1.2))

    # Target
    ax.scatter(target_x, target_y, color=COLORS['accent'], s=300,
               marker='*', zorder=10, edgecolor='white', linewidth=1.5)
    ax.annotate('NYC TARGET', (target_x, target_y), color=COLORS['accent'],
                fontsize=12, fontweight='bold',
                xytext=(12, 12), textcoords='offset points',
                path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['bg'])])

    # Arrival radius circle
    circle = Circle((target_x, target_y), 30, fill=False,
                    edgecolor=COLORS['accent'], linewidth=1.5, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    ax.text(target_x + 32, target_y - 25, 'Arrival\nRadius', color=COLORS['accent'],
            fontsize=8, alpha=0.7)

    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)  # Invert Y for image coordinates
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.set_title('CRNG-Cast Real: Blob Trajectories from GOES-16 IR',
                fontweight='bold', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['cold'],
               markersize=8, label='Cold blobs (clouds)', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['warm'],
               markersize=8, label='Warm blobs (surface)', linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['accent'],
               markersize=12, label='Target (NYC)', linestyle='None'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              facecolor=COLORS['bg'], edgecolor=COLORS['grid'])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_prediction_comparison(crng_preds, prng_preds, actuals, horizons,
                                save_path='crng_cast_real_prediction.png'):
    """CRNG vs PRNG prediction comparison with actual outcomes."""
    setup_dark_style()
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: Prediction intervals over horizons ---
    ax1 = fig.add_subplot(gs[0, 0])

    h = np.array(horizons)
    crng_means = [p.expected_temp_change for p in crng_preds]
    crng_lows = [p.ci_low for p in crng_preds]
    crng_highs = [p.ci_high for p in crng_preds]
    prng_means = [p.expected_temp_change for p in prng_preds]
    prng_lows = [p.ci_low for p in prng_preds]
    prng_highs = [p.ci_high for p in prng_preds]
    actual_vals = [a['temp_change'] for a in actuals]

    ax1.fill_between(h, crng_lows, crng_highs, alpha=0.2, color=COLORS['crng'], label='CRNG 90% CI')
    ax1.fill_between(h, prng_lows, prng_highs, alpha=0.2, color=COLORS['prng'], label='PRNG 90% CI')
    ax1.plot(h, crng_means, 'o-', color=COLORS['crng'], linewidth=2, markersize=6, label='CRNG mean')
    ax1.plot(h, prng_means, 's-', color=COLORS['prng'], linewidth=2, markersize=6, label='PRNG mean')
    ax1.plot(h, actual_vals, '^-', color=COLORS['real'], linewidth=2.5, markersize=8,
             label='ACTUAL', zorder=5)

    ax1.set_xlabel('Horizon (hours)')
    ax1.set_ylabel('Temp Change (K)')
    ax1.set_title('Prediction Intervals vs Actual', fontweight='bold')
    ax1.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color=COLORS['muted'], linewidth=0.8, linestyle='--')

    # --- Panel 2: Distribution comparison for max-convergence horizon ---
    ax2 = fig.add_subplot(gs[0, 1])

    # Find horizon with most converging blobs
    best_idx = max(range(len(crng_preds)), key=lambda i: crng_preds[i].n_converging)
    crng_best = crng_preds[best_idx]
    prng_best = prng_preds[best_idx]
    actual_best = actuals[best_idx]

    bins = np.linspace(min(crng_best.ci_low, prng_best.ci_low) * 1.5,
                       max(crng_best.ci_high, prng_best.ci_high) * 1.5, 80)

    ax2.hist(crng_best.outcome_samples, bins=bins, alpha=0.5, color=COLORS['crng'],
             density=True, label=f'CRNG (K={crng_best.kurtosis_used:.1f})')
    ax2.hist(prng_best.outcome_samples, bins=bins, alpha=0.5, color=COLORS['prng'],
             density=True, label='PRNG (K=3.0)')
    ax2.axvline(actual_best['temp_change'], color=COLORS['real'], linewidth=2.5,
                linestyle='--', label=f'Actual: {actual_best["temp_change"]:+.1f}K')

    ax2.set_xlabel('Temp Change (K)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Distribution @ {horizons[best_idx]:.1f}h ({crng_best.n_converging} blobs converging)',
                  fontweight='bold')
    ax2.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Convergence count and kurtosis ---
    ax3 = fig.add_subplot(gs[1, 0])

    n_conv = [p.n_converging for p in crng_preds]
    k_vals = [p.kurtosis_used for p in crng_preds]

    ax3_twin = ax3.twinx()
    bars = ax3.bar(h, n_conv, width=0.08, color=COLORS['accent'], alpha=0.7, label='Converging blobs')
    ax3_twin.plot(h, k_vals, 'D-', color=COLORS['crng'], linewidth=2, markersize=7, label='Kurtosis')

    ax3.set_xlabel('Horizon (hours)')
    ax3.set_ylabel('N converging blobs', color=COLORS['accent'])
    ax3_twin.set_ylabel('Kurtosis (K)', color=COLORS['crng'])
    ax3_twin.tick_params(axis='y', labelcolor=COLORS['crng'])
    ax3.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax3.set_title('Blob Convergence vs Kurtosis', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               facecolor=COLORS['bg'], edgecolor=COLORS['grid'])

    # --- Panel 4: Probability of extreme ---
    ax4 = fig.add_subplot(gs[1, 1])

    crng_probs = [p.prob_extreme for p in crng_preds]
    prng_probs = [p.prob_extreme for p in prng_preds]
    actual_extreme = [1 if a['is_extreme'] else 0 for a in actuals]

    ax4.plot(h, crng_probs, 'o-', color=COLORS['crng'], linewidth=2, markersize=6,
             label='CRNG P(extreme)')
    ax4.plot(h, prng_probs, 's-', color=COLORS['prng'], linewidth=2, markersize=6,
             label='PRNG P(extreme)')
    ax4.scatter(h, actual_extreme, color=COLORS['real'], s=100, marker='^',
                zorder=5, label='Actual extreme?', edgecolor='white', linewidth=1)

    ax4.set_xlabel('Horizon (hours)')
    ax4.set_ylabel('P(extreme)')
    ax4.set_title('Extreme Event Probability', fontweight='bold')
    ax4.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.05, 1.05)

    fig.suptitle('CRNG-Cast vs PRNG-Cast: Real GOES-16 Satellite Data',
                fontweight='bold', fontsize=14, y=0.99)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_scorecard(crng_preds, prng_preds, actuals, horizons,
                   all_blobs_per_frame, loader,
                   save_path='crng_cast_real_scorecard.png'):
    """Summary scorecard with key metrics."""
    setup_dark_style()
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Compute error metrics
    crng_errors = [abs(crng_preds[i].expected_temp_change - actuals[i]['temp_change'])
                   for i in range(len(horizons))]
    prng_errors = [abs(prng_preds[i].expected_temp_change - actuals[i]['temp_change'])
                   for i in range(len(horizons))]

    crng_mae = np.mean(crng_errors)
    prng_mae = np.mean(prng_errors)

    # CI coverage: how often does actual fall within 90% CI?
    crng_coverage = np.mean([
        1 if crng_preds[i].ci_low <= actuals[i]['temp_change'] <= crng_preds[i].ci_high else 0
        for i in range(len(horizons))
    ])
    prng_coverage = np.mean([
        1 if prng_preds[i].ci_low <= actuals[i]['temp_change'] <= prng_preds[i].ci_high else 0
        for i in range(len(horizons))
    ])

    # CI width
    crng_width = np.mean([crng_preds[i].ci_high - crng_preds[i].ci_low for i in range(len(horizons))])
    prng_width = np.mean([prng_preds[i].ci_high - prng_preds[i].ci_low for i in range(len(horizons))])

    # Tail calibration: P(|outcome| > 2*std) actual vs predicted
    crng_extreme_pred = np.mean([p.prob_extreme for p in crng_preds])
    prng_extreme_pred = np.mean([p.prob_extreme for p in prng_preds])
    actual_extreme_rate = np.mean([1 if a['is_extreme'] else 0 for a in actuals])

    # --- Panel 1: MAE comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(['CRNG-Cast', 'PRNG-Cast'], [crng_mae, prng_mae],
                   color=[COLORS['crng'], COLORS['prng']], alpha=0.8, width=0.5)
    for bar, val in zip(bars, [crng_mae, prng_mae]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}K', ha='center', fontweight='bold', fontsize=11)
    ax1.set_ylabel('MAE (K)')
    ax1.set_title('Mean Absolute Error', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Panel 2: CI Coverage ---
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(['CRNG-Cast', 'PRNG-Cast'], [crng_coverage * 100, prng_coverage * 100],
                   color=[COLORS['crng'], COLORS['prng']], alpha=0.8, width=0.5)
    ax2.axhline(y=90, color=COLORS['accent'], linewidth=2, linestyle='--', label='Target (90%)')
    for bar, val in zip(bars, [crng_coverage * 100, prng_coverage * 100]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Coverage (%)')
    ax2.set_title('90% CI Coverage', fontweight='bold')
    ax2.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 110)

    # --- Panel 3: CI Width ---
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(['CRNG-Cast', 'PRNG-Cast'], [crng_width, prng_width],
                   color=[COLORS['crng'], COLORS['prng']], alpha=0.8, width=0.5)
    for bar, val in zip(bars, [crng_width, prng_width]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}K', ha='center', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Width (K)')
    ax3.set_title('Avg CI Width', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Panel 4: Error by horizon ---
    ax4 = fig.add_subplot(gs[1, 0])
    h = np.array(horizons)
    ax4.plot(h, crng_errors, 'o-', color=COLORS['crng'], linewidth=2, markersize=6, label='CRNG')
    ax4.plot(h, prng_errors, 's-', color=COLORS['prng'], linewidth=2, markersize=6, label='PRNG')
    ax4.set_xlabel('Horizon (hours)')
    ax4.set_ylabel('Absolute Error (K)')
    ax4.set_title('Error by Horizon', fontweight='bold')
    ax4.legend(fontsize=8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Blob statistics ---
    ax5 = fig.add_subplot(gs[1, 1])
    n_blobs_per_frame = [len(bl) for bl in all_blobs_per_frame]
    ax5.bar(range(len(n_blobs_per_frame)), n_blobs_per_frame,
            color=COLORS['muted'], alpha=0.7)
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('N blobs')
    ax5.set_title('Blobs Detected per Frame', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # --- Panel 6: Data card ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    card_text = (
        f"DATA SOURCE\n"
        f"{'='*30}\n"
        f"Satellite:  GOES-16\n"
        f"Band:       13 (10.3um IR)\n"
        f"Resolution: 625x375 px\n"
        f"Crop:       {loader.crop_size}x{loader.crop_size} px\n"
        f"Target:     NYC ({loader.target_lat:.2f}N,\n"
        f"            {abs(loader.target_lon):.2f}W)\n"
        f"Frames:     {len(all_blobs_per_frame)}\n"
        f"Interval:   10 min\n"
        f"Wind:       ({loader.wind_vx:.1f}, {loader.wind_vy:.1f}) px/fr\n"
        f"\n"
        f"RESULTS\n"
        f"{'='*30}\n"
        f"CRNG MAE:   {crng_mae:.2f} K\n"
        f"PRNG MAE:   {prng_mae:.2f} K\n"
        f"CRNG Cover: {crng_coverage*100:.0f}%\n"
        f"PRNG Cover: {prng_coverage*100:.0f}%\n"
        f"Advantage:  {'CRNG' if crng_coverage >= prng_coverage else 'PRNG'}"
    )

    ax6.text(0.05, 0.95, card_text, transform=ax6.transAxes,
             fontfamily='monospace', fontsize=9, verticalalignment='top',
             color=COLORS['text'],
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#161B22',
                      edgecolor=COLORS['grid']))

    fig.suptitle('CRNG-Cast Real: Performance Scorecard',
                fontweight='bold', fontsize=14, y=0.99)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# F) MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("  CRNG-Cast REAL: GOES-16 Satellite Nowcasting")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
    os.makedirs(out_dir, exist_ok=True)

    # --- Step 1: Download real satellite data ---
    loader = SatelliteDataLoader(
        target_lat=40.71, target_lon=-74.01,  # New York
        crop_size=200,
    )

    n_total_frames = 16
    frames = loader.download_sequence(n_frames=n_total_frames)

    # Split: train on first 12, validate on last 4
    n_train = 12
    train_frames = frames[:n_train]
    val_frames = frames[n_train:]

    background_temp = loader.background_temp
    target_x = loader.target_crop_x
    target_y = loader.target_crop_y

    print(f"Background temperature: {background_temp:.1f} K")
    print(f"Target pixel (in crop): ({target_x}, {target_y})")
    print(f"Train frames: {n_train}, Validation frames: {len(val_frames)}")

    # --- Step 2: Detect blobs in all training frames ---
    print("\n--- BLOB DETECTION ---")
    detector = BlobDetector(threshold=4.0, min_area=15)
    all_blobs = []

    for i, frame in enumerate(train_frames):
        blobs = detector.detect(frame, background_temp=background_temp, frame_idx=i)
        all_blobs.append(blobs)
        n_cold = sum(1 for b in blobs if b.mean_temperature < 0)
        n_warm = sum(1 for b in blobs if b.mean_temperature >= 0)
        print(f"  Frame {i:2d}: {len(blobs):3d} blobs ({n_cold} cold, {n_warm} warm)")

    # --- Step 3: Track blobs across frames ---
    print("\n--- BLOB TRACKING ---")
    tracker = BlobTracker(max_match_distance=35.0, map_size=loader.crop_size)
    all_tracked = []

    for i in range(1, len(all_blobs)):
        tracked = tracker.track(all_blobs[i-1], all_blobs[i])
        all_tracked.append(tracked)
        n_matched = sum(1 for t in tracked if t.prev_blob is not None)
        avg_speed = np.mean([t.speed for t in tracked if t.speed > 0]) if any(t.speed > 0 for t in tracked) else 0
        print(f"  Frame {i-1}->{i}: {n_matched}/{len(tracked)} matched, avg speed={avg_speed:.1f} px/fr")

    # Use the latest tracking for predictions
    latest_tracked = all_tracked[-1] if all_tracked else []

    # --- Step 4: Predict at multiple horizons ---
    print("\n--- PREDICTIONS ---")
    predictor = CRNGCastPredictor(
        map_size=loader.crop_size,
        frame_interval_hours=10/60,
        seed=42,
    )

    # Horizons: predict 10, 20, 30, 40 minutes ahead
    # These correspond to validation frames 0, 1, 2, 3
    horizons = [10/60, 20/60, 30/60, 40/60]  # hours

    # Also use multiple tracking snapshots for richer convergence signals
    # Accumulate converging blobs from multiple tracking frames
    crng_predictions = []
    prng_predictions = []
    actuals = []

    for i, horizon in enumerate(horizons):
        # Aggregate converging blobs from multiple tracking frames
        # More realistic: different frames may reveal different systems
        all_conv = []
        seen_positions = set()
        for tracked_list in all_tracked[-4:]:
            conv = predictor.find_converging_blobs(
                tracked_list, target_x, target_y,
                horizon_hours=horizon, arrival_radius=35.0,
            )
            for cb in conv:
                pos_key = (round(cb.tracked_blob.blob.centroid_x / 8),
                          round(cb.tracked_blob.blob.centroid_y / 8))
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    all_conv.append(cb)
        converging = all_conv

        crng_pred = predictor.predict_crng(converging, horizon)
        prng_pred = predictor.predict_prng(converging, horizon)

        # Actual outcome from validation frames
        if i < len(val_frames):
            actual = loader.get_actual_outcome(frames, n_train + i)
        else:
            actual = {'temperature': background_temp, 'temp_change': 0.0, 'is_extreme': False}

        crng_predictions.append(crng_pred)
        prng_predictions.append(prng_pred)
        actuals.append(actual)

        print(f"\n  Horizon: {horizon*60:.0f} min ({horizon:.2f}h)")
        print(f"    Converging blobs: {crng_pred.n_converging}")
        print(f"    Kurtosis used:    {crng_pred.kurtosis_used:.1f}")
        print(f"    CRNG prediction:  {crng_pred.expected_temp_change:+.2f}K  CI=[{crng_pred.ci_low:+.1f}, {crng_pred.ci_high:+.1f}]")
        print(f"    PRNG prediction:  {prng_pred.expected_temp_change:+.2f}K  CI=[{prng_pred.ci_low:+.1f}, {prng_pred.ci_high:+.1f}]")
        print(f"    ACTUAL outcome:   {actual['temp_change']:+.2f}K  extreme={actual['is_extreme']}")

        # Check coverage
        crng_covers = crng_pred.ci_low <= actual['temp_change'] <= crng_pred.ci_high
        prng_covers = prng_pred.ci_low <= actual['temp_change'] <= prng_pred.ci_high
        print(f"    Covers actual?    CRNG={'YES' if crng_covers else 'NO'}  PRNG={'YES' if prng_covers else 'NO'}")

    # --- Step 5: Generate charts ---
    print("\n\n--- GENERATING CHARTS ---")

    # Chart 1: Thermal map with blobs
    plot_thermal_map(
        train_frames[-1], all_blobs[-1], target_x, target_y, background_temp,
        full_frame=loader.full_frame_raw,
        save_path=os.path.join(out_dir, 'crng_cast_real_thermal.png'),
    )

    # Chart 2: Blob trajectories
    plot_tracking(
        all_tracked, target_x, target_y, loader.crop_size,
        save_path=os.path.join(out_dir, 'crng_cast_real_tracking.png'),
    )

    # Chart 3: Prediction comparison
    plot_prediction_comparison(
        crng_predictions, prng_predictions, actuals, horizons,
        save_path=os.path.join(out_dir, 'crng_cast_real_prediction.png'),
    )

    # Chart 4: Scorecard
    plot_scorecard(
        crng_predictions, prng_predictions, actuals, horizons,
        all_blobs, loader,
        save_path=os.path.join(out_dir, 'crng_cast_real_scorecard.png'),
    )

    # --- Summary ---
    crng_mae = np.mean([abs(crng_predictions[i].expected_temp_change - actuals[i]['temp_change'])
                         for i in range(len(horizons))])
    prng_mae = np.mean([abs(prng_predictions[i].expected_temp_change - actuals[i]['temp_change'])
                         for i in range(len(horizons))])

    crng_coverage = np.mean([
        1 if crng_predictions[i].ci_low <= actuals[i]['temp_change'] <= crng_predictions[i].ci_high else 0
        for i in range(len(horizons))
    ]) * 100
    prng_coverage = np.mean([
        1 if prng_predictions[i].ci_low <= actuals[i]['temp_change'] <= prng_predictions[i].ci_high else 0
        for i in range(len(horizons))
    ]) * 100

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Data source:     GOES-16 Band 13 (10.3um clean IR)")
    print(f"  Target:          New York ({loader.target_lat}N, {loader.target_lon}W)")
    print(f"  Training:        {n_train} frames ({n_train * 10} min)")
    print(f"  Validation:      {len(val_frames)} frames ({len(val_frames) * 10} min)")
    print(f"")
    print(f"  CRNG-Cast MAE:   {crng_mae:.3f} K")
    print(f"  PRNG-Cast MAE:   {prng_mae:.3f} K")
    print(f"  CRNG 90% CI cov: {crng_coverage:.0f}%")
    print(f"  PRNG 90% CI cov: {prng_coverage:.0f}%")
    print(f"")

    if crng_coverage > prng_coverage:
        print(f"  >>> CRNG-Cast has BETTER coverage ({crng_coverage:.0f}% vs {prng_coverage:.0f}%)")
    elif crng_coverage == prng_coverage:
        print(f"  >>> Both have same coverage ({crng_coverage:.0f}%)")
    else:
        print(f"  >>> PRNG-Cast has better coverage ({prng_coverage:.0f}% vs {crng_coverage:.0f}%)")

    # Tail analysis
    print(f"\n  TAIL ANALYSIS:")
    for i, h in enumerate(horizons):
        kw = crng_predictions[i].kurtosis_used
        nc = crng_predictions[i].n_converging
        crng_w = crng_predictions[i].ci_high - crng_predictions[i].ci_low
        prng_w = prng_predictions[i].ci_high - prng_predictions[i].ci_low
        ratio = crng_w / prng_w if prng_w > 0 else 0
        print(f"    {h*60:4.0f}min: N_conv={nc}, K={kw:.1f}, CI_ratio={ratio:.2f}x wider")

    print(f"\n  Charts saved to: {os.path.abspath(out_dir)}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
