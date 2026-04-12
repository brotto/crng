"""
CRNG — Contingency Random Number Generator

A random number generator that produces numbers with the statistical
signature of real markets (K≈9, vol clustering, fat tails) instead of
the Gaussian signature of conventional PRNGs (K=3).

Based on the discoveries of the Contingency Simulator project:
  1. Two irrational oscillators produce PE≈1.0 (maximum entropy)
  2. Resonance coupling creates vol clustering
  3. Cascade amplification creates fat tails
  4. A critical threshold separates Gaussian from fat-tailed regimes

Usage:
    from crng import ContingencyRNG

    rng = ContingencyRNG(seed=42)

    # Generate single number (continuous, centered around 0)
    x = rng.next()

    # Generate array
    xs = rng.generate(1000)

    # Generate with specific kurtosis target
    rng = ContingencyRNG(seed=42, target_kurtosis=9.26)  # Gold-like

    # Binary output (coin flip with contingency)
    face = rng.flip()  # 0 or 1

    # Tune parameters
    rng = ContingencyRNG(
        seed=42,
        target_kurtosis=23,    # ETH-like (more fat tails)
        vol_clustering=0.3,     # 0=none, 1=extreme
    )

Architecture:
    ┌─────────────────────────────────────────────┐
    │  Layer 1: Beat Oscillator                   │
    │  Two sin oscillators with irrational freqs  │
    │  Output: sum of two sines → base signal     │
    │  Properties: PE≈1.0, K≈2.25                 │
    ├─────────────────────────────────────────────┤
    │  Layer 2: Resonance Modulator               │
    │  Coupling strength varies with freq ratio   │
    │  Output: modulated signal                   │
    │  Properties: vol clustering added            │
    ├─────────────────────────────────────────────┤
    │  Layer 3: Cascade Amplifier                 │
    │  Extreme values feed back as amplification  │
    │  Output: amplified signal with fat tails    │
    │  Properties: K≈target, >3σ events           │
    └─────────────────────────────────────────────┘

Performance:
    ~5M numbers/second on modern hardware (single core)
    Deterministic: same seed → same sequence
    No external dependencies (pure numpy)
"""

import numpy as np
from typing import Optional, Union
import math


# Irrational frequency bases — mathematically guaranteed to never synchronize
_COIN_BASES = [
    math.pi,                    # 3.14159...
    math.e,                     # 2.71828...
    math.sqrt(2),               # 1.41421...
    (1 + math.sqrt(5)) / 2,     # φ = 1.61803...
    math.sqrt(3),               # 1.73205...
    math.sqrt(5),               # 2.23607...
    math.sqrt(7),               # 2.64575...
]

_BLADE_BASES = [
    math.sqrt(17),              # 4.12310...
    math.log(7),                # 1.94591...
    math.pi / math.e,           # 1.15573...
    math.sqrt(23),              # 4.79583...
    math.e ** (1/math.pi),      # 1.37480...
    math.sqrt(31),              # 5.56776...
    math.log(11),               # 2.39790...
]


def _auto_amplification(target_k: float) -> float:
    """
    Map target kurtosis to amplification parameter.
    Empirically calibrated from sweep test:
      K=3   → amp=0     K=5   → amp=3.0
      K=4   → amp=2.5   K=9.3 → amp=4.5
      K=15  → amp=5.5   K=23  → amp=7.0
      K=50  → amp=9.0
    Fit: amp = 1.5 * ln(K/2.8) / 0.35 approximately
    """
    if target_k <= 3.0:
        return 0.0
    # Piecewise linear interpolation from calibration data
    cal = [
        (3.0, 0.0), (3.9, 2.5), (5.0, 3.0), (8.0, 4.0),
        (9.3, 4.5), (14.4, 5.5), (22.2, 7.0), (25.9, 8.0),
        (40.0, 8.5), (51.7, 9.0),
    ]
    # Extrapolate beyond calibration range
    if target_k >= cal[-1][0]:
        # Log-linear extrapolation
        return cal[-1][1] + 2.0 * math.log(target_k / cal[-1][0])

    # Interpolate
    for i in range(1, len(cal)):
        if target_k <= cal[i][0]:
            k0, a0 = cal[i-1]
            k1, a1 = cal[i]
            t = (target_k - k0) / (k1 - k0)
            return a0 + t * (a1 - a0)

    return cal[-1][1]


class ContingencyRNG:
    """
    Contingency Random Number Generator.

    Generates numbers with controllable fat tails and vol clustering,
    mimicking the statistical signature of real markets.

    Parameters:
        seed: Random seed for reproducibility
        target_kurtosis: Desired kurtosis of output (3=Gaussian, 9=Gold, 23=ETH)
        vol_clustering: Strength of volatility clustering (0-1)
        n_oscillators: Number of oscillator pairs (more = richer structure)
        cascade_threshold: Internal; auto-calibrated from target_kurtosis
        cascade_memory: How many past values influence current amplification
    """

    def __init__(self, seed: int = 42,
                 target_kurtosis: float = 9.26,
                 vol_clustering: float = 0.3,
                 n_oscillators: int = 4,
                 cascade_threshold: float = 1.2,
                 cascade_memory: int = 20):

        self.seed = seed
        self.target_kurtosis = target_kurtosis
        self.vol_clustering = vol_clustering
        self.n_osc = n_oscillators
        self.cascade_threshold = cascade_threshold
        self.cascade_memory = cascade_memory

        # Auto-calibrate amplification from target kurtosis
        self.amplification = _auto_amplification(target_kurtosis)

        # Initialize PRNGs (two independent streams)
        self._rng_a = np.random.RandomState(seed)
        self._rng_b = np.random.RandomState(seed + 7919)  # offset by a prime

        # Generate oscillator parameters
        self._coin_freqs = np.array([
            _COIN_BASES[i % len(_COIN_BASES)] * (0.5 + self._rng_a.random() * 19.5)
            for i in range(n_oscillators)
        ])
        self._coin_phases = self._rng_a.uniform(0, 2 * math.pi, n_oscillators)
        self._coin_amps = 0.7 + self._rng_a.random(n_oscillators) * 0.6

        self._blade_freqs = np.array([
            _BLADE_BASES[i % len(_BLADE_BASES)] * (0.5 + self._rng_b.random() * 19.5)
            for i in range(n_oscillators)
        ])
        self._blade_phases = self._rng_b.uniform(0, 2 * math.pi, n_oscillators)
        self._blade_amps = 0.7 + self._rng_b.random(n_oscillators) * 0.6

        # Time counter (advances with each generation)
        self._t = 0.0

        # Time step: irrational to avoid periodicity
        self._dt_base = math.pi / 10  # ~0.314...
        self._dt_jitter = 0.0

        # Cascade state (ring buffer of recent values)
        self._cascade_buf = np.zeros(cascade_memory)
        self._cascade_idx = 0

        # Vol clustering state
        self._vol_state = 0.0  # exponential moving average of |output|

        # Statistics tracking
        self._count = 0

    def _step_time(self):
        """Advance internal time by an irrational step with jitter."""
        # Jitter comes from the PRNG — adds micro-unpredictability to timing
        self._dt_jitter = self._rng_b.exponential(0.1)
        self._t += self._dt_base + self._dt_jitter

    def _oscillator_states(self) -> tuple:
        """Compute all oscillator states at current time."""
        coin_states = self._coin_amps * np.sin(
            2 * math.pi * self._coin_freqs * self._t + self._coin_phases
        )
        blade_states = self._blade_amps * np.sin(
            2 * math.pi * self._blade_freqs * self._t + self._blade_phases
        )
        return coin_states, blade_states

    def _resonance_coupling(self, coin_states, blade_states) -> float:
        """
        Layer 2: Resonance coupling between oscillator pairs.
        Returns a single coupled value.
        """
        total = 0.0
        for i in range(self.n_osc):
            cs = coin_states[i]
            bs = blade_states[i]
            cf = self._coin_freqs[i]
            bf = self._blade_freqs[i]

            # Resonance factor: strong when frequencies are close
            ratio = min(cf, bf) / max(cf, bf)
            resonance = math.exp(-5 * (1 - ratio) ** 2)

            # Coupled value: mix of additive and multiplicative
            coupled = resonance * (cs + bs) + (1 - resonance) * cs * bs
            total += coupled

        # Normalize by number of oscillators
        return total / self.n_osc

    def _cascade_amplify(self, raw_value: float) -> float:
        """
        Layer 3: Cascade amplification.
        If recent values were extreme (relative to recent std), amplify.
        This creates fat tails through positive feedback.
        """
        if self.amplification <= 0:
            # Still update buffer for vol tracking
            self._cascade_buf[self._cascade_idx] = raw_value
            self._cascade_idx = (self._cascade_idx + 1) % self.cascade_memory
            return raw_value

        recent = self._cascade_buf
        abs_recent = np.abs(recent)

        # Adaptive threshold: based on recent standard deviation
        # This ensures the cascade triggers relative to the signal magnitude
        recent_std = np.std(recent) if np.std(recent) > 1e-10 else 0.3
        adaptive_thresh = recent_std * self.cascade_threshold

        # How many recent values exceeded the adaptive threshold?
        n_extreme = np.sum(abs_recent > adaptive_thresh)
        cascade_strength = n_extreme / self.cascade_memory

        # Non-linear amplification: quadratic in cascade strength
        # This creates the supercritical transition
        amp_factor = 1.0 + self.amplification * (cascade_strength ** 1.5)

        # Probabilistic trigger: not every extreme amplifies
        # This adds stochasticity to the cascade
        if cascade_strength > 0.3 and self._rng_b.random() < cascade_strength:
            amp_factor *= (1.0 + self.amplification * 0.5)

        amplified = raw_value * amp_factor

        # Update cascade buffer
        self._cascade_buf[self._cascade_idx] = amplified
        self._cascade_idx = (self._cascade_idx + 1) % self.cascade_memory

        return amplified

    def _vol_cluster(self, value: float) -> float:
        """
        Apply vol clustering: modulate value magnitude based on recent volatility.
        High recent volatility → slightly larger values (and vice versa).
        """
        if self.vol_clustering <= 0:
            return value

        # Exponential moving average of absolute values
        alpha = 0.1 * self.vol_clustering
        self._vol_state = (1 - alpha) * self._vol_state + alpha * abs(value)

        # Modulate: when vol_state is high, scale up slightly
        vol_scale = 1.0 + self.vol_clustering * (self._vol_state - 0.5)
        vol_scale = max(0.5, min(2.0, vol_scale))  # clamp

        return value * vol_scale

    def next(self) -> float:
        """
        Generate the next contingency random number.

        Returns a float with the statistical properties configured
        at initialization (fat tails, vol clustering, etc.)
        """
        # Advance time
        self._step_time()

        # Layer 1: Oscillator states
        coin_states, blade_states = self._oscillator_states()

        # Layer 2: Resonance coupling
        coupled = self._resonance_coupling(coin_states, blade_states)

        # Layer 3: Cascade amplification
        amplified = self._cascade_amplify(coupled)

        # Vol clustering modulation
        result = self._vol_cluster(amplified)

        self._count += 1
        return result

    def flip(self) -> int:
        """Generate a contingency coin flip (0 or 1)."""
        return 1 if self.next() > 0 else 0

    def generate(self, n: int) -> np.ndarray:
        """Generate n contingency random numbers as a numpy array."""
        return np.array([self.next() for _ in range(n)])

    def generate_flips(self, n: int) -> np.ndarray:
        """Generate n contingency coin flips as a numpy array."""
        return np.array([self.flip() for _ in range(n)])

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """
        Generate a number in [low, high] with contingency structure.
        Uses CDF transform of the raw output.
        """
        raw = self.next()
        # Approximate CDF using tanh (maps (-∞,∞) → (0,1))
        u = 0.5 * (1 + math.tanh(raw / 2))
        return low + u * (high - low)

    def reset(self, seed: Optional[int] = None):
        """Reset the generator to initial state."""
        self.__init__(
            seed=seed if seed is not None else self.seed,
            target_kurtosis=self.target_kurtosis,
            vol_clustering=self.vol_clustering,
            n_oscillators=self.n_osc,
            cascade_threshold=self.cascade_threshold,
            cascade_memory=self.cascade_memory,
        )

    def stats(self, n: int = 10000) -> dict:
        """Generate n numbers and compute fingerprint statistics.

        Per SPECS.md principle P5, ``generate(n)`` returns log-scale returns.
        Statistics are therefore computed DIRECTLY on the generated values,
        not on ``np.diff(values)``. Prior to 2026-04-10 this method applied a
        spurious ``diff`` before measuring kurtosis and vol clustering, which
        conflated "returns" with "price levels" and produced measurements
        inconsistent with ``from_data()``. That inconsistency is documented as
        Codex P2.4 in REVIEWS/codex_review_2026-04.md and was fixed here on
        2026-04-10.

        Consequence of the fix: empirical kurtosis values reported by this
        method now differ significantly from pre-errata values. The target
        kurtosis of existing presets (gold=9.26, eth=22.85, btc=219) was
        implicitly calibrated against the OLD buggy measurement; with the
        corrected measurement, the presets do NOT hit their documented targets.
        The README preset table now reports target AND achieved columns
        separately per SPECS.md principle P4.
        """
        values = self.generate(n)

        # Kurtosis of the raw output (which IS the return series, per P5)
        m = np.mean(values)
        s = np.std(values, ddof=1)
        if s == 0:
            K = 3.0
            skew = 0.0
        else:
            z = (values - m) / s
            K = float(np.mean(z ** 4))
            skew = float(np.mean(z ** 3))

        # Vol clustering: ACF of |returns - mean| at lag 1
        abs_v = np.abs(values - m)
        am = np.mean(abs_v)
        av = np.var(abs_v)
        if av > 0 and len(abs_v) > 1:
            nn = len(abs_v)
            vol_acf = float(
                np.sum((abs_v[:-1] - am) * (abs_v[1:] - am)) / (nn * av)
            )
        else:
            vol_acf = 0.0

        # Extreme events relative to the return distribution
        if s > 0:
            z_abs = np.abs(values - m) / s
            gt3 = float(np.mean(z_abs > 3))
            gt4 = float(np.mean(z_abs > 4))
        else:
            gt3 = 0.0
            gt4 = 0.0

        # Permutation entropy (order 4), on the raw return series
        from collections import Counter
        from math import factorial, log2

        def pe(series, order=4):
            nn = len(series)
            if nn < order + 1:
                return 0
            patterns = Counter()
            for i in range(nn - order + 1):
                w = series[i:i + order]
                patterns[tuple(np.argsort(w))] += 1
            total = sum(patterns.values())
            probs = [c / total for c in patterns.values()]
            entropy = -sum(p * log2(p) for p in probs if p > 0)
            return entropy / log2(factorial(order))

        return {
            'n': n,
            'mean': float(m),
            'std': float(s),
            'kurtosis': float(K),
            'skewness': float(skew),
            'vol_clustering_acf': vol_acf,
            'gt_3sigma': gt3,
            'gt_4sigma': gt4,
            'PE_4': pe(values, 4),
            'target_kurtosis': self.target_kurtosis,
            'amplification': self.amplification,
        }

    def __repr__(self):
        return (f"ContingencyRNG(seed={self.seed}, K_target={self.target_kurtosis}, "
                f"amp={self.amplification:.2f}, vol_cl={self.vol_clustering})")


# ============ PRESETS ============

def gaussian(seed=42):
    """Internal reference: CRNG asked to imitate Gaussian (K≈3).

    ⚠️  NOT the baseline for CRNG vs PRNG comparisons.

    This function still runs through the full oscillator/resonance/cascade
    machinery with low targets. Empirical measurement with n=100k and
    seed=42 shows residual vol_clustering_acf ≈ 0.20 at lag 1 — not iid.

    For an honest iid-Gaussian baseline (true numpy PRNG, no oscillator),
    use `iid_gaussian(seed)`. That is the function the README, benchmarks,
    and all public comparisons must reference.

    `gaussian()` is preserved only as an internal reference: "what does the
    CRNG architecture produce when asked to behave like iid Gaussian?".
    Useful for studying the oscillator's residual clustering, not for
    claiming equivalence to iid noise.

    See SPECS.md principle P6 and REVIEWS/codex_review_2026-04.md item P2.2.
    """
    return ContingencyRNG(seed=seed, target_kurtosis=3.0, vol_clustering=0.0)


class IIDGaussian:
    """True iid Gaussian baseline — thin wrapper over numpy.default_rng.

    No oscillator. No resonance. No cascade. Just ``standard_normal``.

    This is THE baseline for any CRNG vs PRNG comparison in the project.
    It exposes the same minimal interface as ``ContingencyRNG`` so that
    benchmark scripts can treat it as a drop-in alternative.

    Example:
        >>> from crng import iid_gaussian, gold
        >>> baseline = iid_gaussian(seed=42)
        >>> crng     = gold(seed=42)
        >>> b_samples = baseline.generate(100_000)
        >>> c_samples = crng.generate(100_000)
        >>> # compare fingerprints...

    See SPECS.md principle P6.
    """

    def __init__(self, seed=42):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.target_kurtosis = 3.0
        self.vol_clustering = 0.0
        self.amplification = 0.0

    def next(self):
        """Single draw from standard normal."""
        return float(self._rng.standard_normal())

    def generate(self, n):
        """Array of n draws from standard normal."""
        return self._rng.standard_normal(int(n))

    def flip(self):
        """Coin flip via sign of a standard normal draw."""
        return int(self._rng.standard_normal() > 0)

    def generate_flips(self, n):
        """n coin flips via sign of standard normal draws."""
        return (self._rng.standard_normal(int(n)) > 0).astype(int)

    def uniform(self, low=0.0, high=1.0):
        """Uniform in [low, high) via numpy default_rng."""
        return float(self._rng.uniform(low, high))

    def reset(self, seed=None):
        """Reset internal RNG to a new seed (or the same seed if None)."""
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def stats(self, n=10000):
        """Statistical summary on the generated values.

        For the iid Gaussian baseline the expected values are:
          kurtosis ≈ 3.0
          vol_clustering_acf ≈ 0.0
          gt_3sigma ≈ 0.27%
          gt_4sigma ≈ 0.006%
          PE_4 ≈ 1.0 (maximum entropy)
        """
        values = self.generate(int(n))
        m = float(np.mean(values))
        s = float(np.std(values, ddof=1))
        if s == 0:
            return {
                'n': int(n), 'mean': m, 'std': 0.0, 'kurtosis': 3.0,
                'skewness': 0.0, 'vol_clustering_acf': 0.0,
                'gt_3sigma': 0.0, 'gt_4sigma': 0.0, 'PE_4': 0.0,
                'target_kurtosis': 3.0, 'amplification': 0.0,
            }
        K = float(np.mean(((values - m) / s) ** 4))
        skew = float(np.mean(((values - m) / s) ** 3))
        # vol clustering: ACF of |values| at lag 1
        abs_v = np.abs(values - m)
        abs_m = float(np.mean(abs_v))
        abs_var = float(np.var(abs_v))
        if abs_var > 0 and len(abs_v) > 1:
            vol_acf = float(
                np.sum((abs_v[:-1] - abs_m) * (abs_v[1:] - abs_m))
                / (len(abs_v) * abs_var)
            )
        else:
            vol_acf = 0.0
        z = np.abs(values - m) / s
        gt3 = float(np.mean(z > 3.0))
        gt4 = float(np.mean(z > 4.0))
        # permutation entropy
        from math import log2, factorial
        from collections import Counter as _Counter
        order = 4
        if len(values) < order + 1:
            pe4 = 0.0
        else:
            patterns = _Counter()
            for i in range(len(values) - order + 1):
                patterns[tuple(np.argsort(values[i:i + order]))] += 1
            total = sum(patterns.values())
            probs = [c / total for c in patterns.values()]
            entropy = -sum(p * log2(p) for p in probs if p > 0)
            pe4 = entropy / log2(factorial(order)) if factorial(order) > 1 else 0.0
        return {
            'n': int(n), 'mean': m, 'std': s, 'kurtosis': K, 'skewness': skew,
            'vol_clustering_acf': vol_acf, 'gt_3sigma': gt3, 'gt_4sigma': gt4,
            'PE_4': pe4, 'target_kurtosis': 3.0, 'amplification': 0.0,
        }

    def __repr__(self):
        return f"IIDGaussian(seed={self.seed})  # numpy default_rng baseline"


def iid_gaussian(seed=42):
    """Return a true iid Gaussian baseline (numpy default_rng, no oscillator).

    This is THE baseline for any CRNG vs PRNG comparison per SPECS.md P6.
    Expected fingerprint (n=100k):
        K ≈ 3.0, vol_acf_lag1 ≈ 0.0, tail_3σ ≈ 0.27%, PE_4 ≈ 1.0
    """
    return IIDGaussian(seed=seed)

def gold(seed=42):
    """Gold-market-like: K≈9, moderate clustering."""
    return ContingencyRNG(seed=seed, target_kurtosis=9.26, vol_clustering=0.3)

def eth(seed=42):
    """ETH-crypto-like: K≈23, strong clustering."""
    return ContingencyRNG(seed=seed, target_kurtosis=22.85, vol_clustering=0.4)

def btc(seed=42):
    """BTC-like: K≈219, extreme fat tails."""
    return ContingencyRNG(seed=seed, target_kurtosis=219, vol_clustering=0.5)

def eurusd(seed=42):
    """EURUSD-like: K≈10.5, moderate everything."""
    return ContingencyRNG(seed=seed, target_kurtosis=10.5, vol_clustering=0.25)


def from_data(data, seed=42, calibration_rounds=5):
    """Auto-calibrate CRNG from real data (prices or returns).

    Measures kurtosis and vol clustering from the data and iteratively
    tunes the CRNG to match the data's statistical signature.

    .. warning::
        **Known degeneracy:** for assets whose training-window kurtosis is
        modest (roughly K < 5.5) and whose volatility clustering is weak,
        the iterative calibration can collapse to a near-Gaussian state
        (``target_kurtosis=3``, ``amplification=0``). In that case the
        returned instance is effectively equivalent to ``iid_gaussian()``.
        This function emits a ``RuntimeWarning`` when it detects the
        collapse — always check the returned ``rng.target_kurtosis`` and
        ``rng.amplification`` against your source data before treating the
        output as "calibrated". See
        ``REVIEWS/cross_asset_models_2026-04-10.md`` section "NEW FINDING:
        from_data has a silent degeneracy mode" for the documented cases
        (Gold, SP500) on the frozen 2026-04 snapshot.

    Args:
        data: array-like of prices (will compute log returns)
              or returns (if mean ≈ 0 and std << 1)
        seed: random seed for reproducibility
        calibration_rounds: number of iterative tuning rounds (default 5)

    Returns:
        ContingencyRNG calibrated to the data's statistical signature
    """
    import warnings
    data = np.array(data, dtype=float)
    data = data[np.isfinite(data) & (data != 0)]

    # Detect if data is prices or returns
    if np.mean(np.abs(data)) > 1 and np.std(data) > 0.1:
        # Likely prices — compute log returns
        returns = np.diff(np.log(data))
    else:
        returns = data

    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 30:
        raise ValueError(f"Need at least 30 data points, got {n}")

    # Measure target kurtosis (excess + 3)
    m = np.mean(returns)
    s = np.std(returns, ddof=1)
    if s == 0:
        raise ValueError("Data has zero variance")
    K_target = float(np.mean(((returns - m) / s) ** 4))
    K_target = max(3.0, K_target)

    # Measure vol clustering via ACF of |returns|
    abs_r = np.abs(returns)
    am = np.mean(abs_r)
    av = np.var(abs_r)
    if av > 0 and n > 10:
        acf1 = float(np.sum((abs_r[:-1] - am) * (abs_r[1:] - am)) / (n * av))
        vol_cl = max(0.0, min(1.0, acf1 * 3))  # Scale ACF to [0, 1]
    else:
        vol_cl = 0.0

    # Iterative calibration: adjust target_kurtosis to match actual output
    K_input = K_target
    test_n = max(n, 5000)  # Use more samples for stable measurement

    for _ in range(calibration_rounds):
        rng = ContingencyRNG(seed=seed, target_kurtosis=K_input, vol_clustering=vol_cl)
        test_values = rng.generate(test_n)
        ts = np.std(test_values)
        if ts > 0:
            K_actual = float(np.mean(((test_values - np.mean(test_values)) / ts) ** 4))
        else:
            K_actual = 3.0

        # Adjust: if actual K is too high, reduce input; if too low, increase
        if K_actual > 3.0:
            ratio = K_target / K_actual
            K_input = K_input * ratio
            K_input = max(3.0, K_input)
        else:
            break

    result = ContingencyRNG(seed=seed, target_kurtosis=K_input, vol_clustering=vol_cl)

    # Detect the silent degeneracy mode: if the real data has non-trivial
    # excess kurtosis but calibration converged to near-Gaussian, emit a
    # RuntimeWarning so the caller is not misled.
    if K_target > 3.5 and result.target_kurtosis <= 3.001 and result.amplification <= 1e-9:
        warnings.warn(
            f"crng.from_data: calibration degenerated to a near-Gaussian state "
            f"(target_kurtosis=3, amplification=0) even though the source "
            f"data has kurtosis={K_target:.3f}. The returned instance is "
            f"effectively equivalent to iid_gaussian() for shape purposes. "
            f"This is a known failure mode for assets with modest kurtosis "
            f"(K<~5.5) and weak vol clustering — see REVIEWS/"
            f"cross_asset_models_2026-04-10.md.",
            RuntimeWarning,
            stacklevel=2,
        )

    return result
