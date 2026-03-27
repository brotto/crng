"""Tests for the Contingency RNG."""

import numpy as np
import pytest
from crng import ContingencyRNG, gaussian, gold, eth, btc, eurusd, from_data


# ─── Determinism ─────────────────────────────────────────────────

def test_determinism():
    """Same seed produces same sequence."""
    r1 = gold(seed=42)
    r2 = gold(seed=42)
    v1 = [r1.next() for _ in range(100)]
    v2 = [r2.next() for _ in range(100)]
    assert v1 == v2

def test_different_seeds():
    """Different seeds produce different sequences."""
    r1 = gold(seed=42)
    r2 = gold(seed=43)
    v1 = [r1.next() for _ in range(100)]
    v2 = [r2.next() for _ in range(100)]
    assert v1 != v2

def test_reset():
    """reset() restores initial state."""
    rng = gold(seed=42)
    v1 = [rng.next() for _ in range(50)]
    rng.reset()
    v2 = [rng.next() for _ in range(50)]
    assert v1 == v2


# ─── Output Types ────────────────────────────────────────────────

def test_generate_array():
    """generate() returns numpy array of correct size."""
    rng = gold(seed=42)
    xs = rng.generate(500)
    assert isinstance(xs, np.ndarray)
    assert len(xs) == 500

def test_flip():
    """flip() returns 0 or 1."""
    rng = gold(seed=42)
    for _ in range(100):
        f = rng.flip()
        assert f in (0, 1)

def test_flip_balance():
    """Flips should be approximately balanced."""
    rng = gold(seed=42)
    flips = rng.generate_flips(5000)
    p = flips.mean()
    assert 0.4 < p < 0.6, f"P(heads) = {p}, too unbalanced"

def test_uniform():
    """uniform() returns values in [low, high]."""
    rng = gold(seed=42)
    for _ in range(100):
        u = rng.uniform(0, 1)
        assert 0 <= u <= 1


# ─── Kurtosis by Preset ─────────────────────────────────────────

def test_gaussian_kurtosis():
    """Gaussian preset should produce K near 3."""
    rng = gaussian(seed=42)
    s = rng.stats(20000)
    assert 2.0 < s['kurtosis'] < 4.5

def test_gold_kurtosis():
    """Gold preset should produce K > 4."""
    rng = gold(seed=42)
    s = rng.stats(20000)
    assert s['kurtosis'] > 4.0

def test_eth_kurtosis():
    """ETH preset should produce K > 10."""
    rng = eth(seed=42)
    s = rng.stats(20000)
    assert s['kurtosis'] > 10.0

def test_btc_kurtosis():
    """BTC preset should produce very high K."""
    rng = btc(seed=42)
    s = rng.stats(20000)
    assert s['kurtosis'] > 20.0

def test_eurusd_kurtosis():
    """EURUSD preset should produce K > 4."""
    rng = eurusd(seed=42)
    s = rng.stats(20000)
    assert s['kurtosis'] > 4.0


# ─── Kurtosis Ordering ──────────────────────────────────────────

def test_kurtosis_ordering():
    """Higher target_kurtosis should produce higher actual kurtosis."""
    rng_low = ContingencyRNG(seed=42, target_kurtosis=3.0)
    rng_mid = ContingencyRNG(seed=42, target_kurtosis=9.0)
    rng_high = ContingencyRNG(seed=42, target_kurtosis=30.0)
    k_low = rng_low.stats(20000)['kurtosis']
    k_mid = rng_mid.stats(20000)['kurtosis']
    k_high = rng_high.stats(20000)['kurtosis']
    assert k_low < k_mid < k_high


# ─── Vol Clustering ──────────────────────────────────────────────

def test_vol_clustering():
    """Higher vol_clustering parameter should increase ACF of |diffs|."""
    rng_no = ContingencyRNG(seed=42, target_kurtosis=9, vol_clustering=0.0)
    rng_hi = ContingencyRNG(seed=42, target_kurtosis=9, vol_clustering=0.5)
    s_no = rng_no.stats(20000)
    s_hi = rng_hi.stats(20000)
    assert s_hi['vol_clustering_acf'] > s_no['vol_clustering_acf']


# ─── Entropy ─────────────────────────────────────────────────────

def test_high_entropy():
    """All presets should have PE > 0.99."""
    for fn in [gaussian, gold, eth, btc, eurusd]:
        rng = fn(seed=42)
        s = rng.stats(10000)
        assert s['PE_4'] > 0.99, f"{fn.__name__} PE too low: {s['PE_4']}"


# ─── Stats Dict ──────────────────────────────────────────────────

def test_stats_keys():
    """stats() should return all expected keys."""
    rng = gold(seed=42)
    s = rng.stats(5000)
    expected = {'mean', 'std', 'kurtosis', 'skewness',
                'vol_clustering_acf', 'gt_3sigma', 'gt_4sigma',
                'PE_4', 'target_kurtosis', 'amplification'}
    assert expected.issubset(set(s.keys()))


# ─── from_data() ─────────────────────────────────────────────────

def test_from_data_prices():
    """from_data() with price-like data should produce reasonable CRNG."""
    # Simulate a price series with known properties
    np.random.seed(42)
    # Use t-distributed returns for fat tails (K > 3)
    returns = np.random.standard_t(df=5, size=1000) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    rng = from_data(prices, seed=42)
    s = rng.stats(10000)

    # Should have kurtosis above Gaussian (t(5) has K=9)
    assert s['kurtosis'] > 3.5, f"K too low: {s['kurtosis']}"

def test_from_data_returns():
    """from_data() with returns should also work."""
    np.random.seed(42)
    returns = np.random.standard_t(df=5, size=1000) * 0.01

    rng = from_data(returns, seed=42)
    s = rng.stats(10000)
    assert s['kurtosis'] > 3.5

def test_from_data_gaussian():
    """from_data() with Gaussian data should produce near-Gaussian K."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 2000)

    rng = from_data(returns, seed=42)
    s = rng.stats(10000)
    # Should be near Gaussian (K=3), not wildly fat-tailed
    assert s['kurtosis'] < 6.0, f"K too high for Gaussian input: {s['kurtosis']}"

def test_from_data_minimum_samples():
    """from_data() should reject tiny datasets."""
    with pytest.raises(ValueError, match="at least 30"):
        from_data([1, 2, 3], seed=42)

def test_from_data_deterministic():
    """from_data() with same data and seed should be deterministic."""
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))

    rng1 = from_data(prices, seed=42)
    rng2 = from_data(prices, seed=42)
    v1 = [rng1.next() for _ in range(100)]
    v2 = [rng2.next() for _ in range(100)]
    assert v1 == v2


# ─── Real-World Signature Test ───────────────────────────────────

def _kurtosis(x):
    """Compute kurtosis (excess+3) of array."""
    m, s = np.mean(x), np.std(x, ddof=1)
    if s == 0:
        return 3.0
    return float(np.mean(((x - m) / s) ** 4))

def _vol_acf(x, lag=1):
    """ACF of |x| at given lag."""
    a = np.abs(x)
    m, v = np.mean(a), np.var(a)
    if v == 0:
        return 0.0
    n = len(a)
    return float(np.sum((a[:n-lag] - m) * (a[lag:] - m)) / (n * v))

def test_fat_tails_vs_prng():
    """CRNG with K>3 target should produce fatter tails than NumPy."""
    rng = ContingencyRNG(seed=42, target_kurtosis=9.0, vol_clustering=0.3)
    crng_vals = rng.generate(10000)
    prng_vals = np.random.RandomState(42).normal(0, 1, 10000)

    k_crng = _kurtosis(crng_vals)
    k_prng = _kurtosis(prng_vals)

    assert k_crng > k_prng + 1.0, (
        f"CRNG K={k_crng:.1f} should be well above PRNG K={k_prng:.1f}"
    )

def test_tail_events_vs_prng():
    """CRNG should produce more >3-sigma events than NumPy."""
    rng = ContingencyRNG(seed=42, target_kurtosis=9.0, vol_clustering=0.3)
    crng_vals = rng.generate(20000)
    prng_vals = np.random.RandomState(42).normal(0, 1, 20000)

    def tail_pct(x, sigma=3):
        s = np.std(x)
        z = np.abs(x - np.mean(x)) / s
        return np.mean(z > sigma) * 100

    assert tail_pct(crng_vals) > tail_pct(prng_vals), (
        "CRNG should have more extreme events"
    )

def test_from_data_captures_kurtosis():
    """from_data() calibrated on t(5) data should produce K closer to
    the data's K than a plain Gaussian PRNG would."""
    np.random.seed(42)
    # t(5) has theoretical K=9
    t5_returns = np.random.standard_t(df=5, size=2000) * 0.01
    k_data = _kurtosis(t5_returns)

    rng = from_data(t5_returns, seed=42)
    crng_vals = rng.generate(5000)
    k_crng = _kurtosis(crng_vals)

    prng_vals = np.random.RandomState(42).normal(0, 1, 5000)
    k_prng = _kurtosis(prng_vals)

    # CRNG should be closer to data's K than PRNG
    assert abs(k_crng - k_data) < abs(k_prng - k_data), (
        f"CRNG K={k_crng:.1f} should be closer to data K={k_data:.1f} "
        f"than PRNG K={k_prng:.1f}"
    )


# ─── Repr ─────────────────────────────────────────────────────────

def test_repr():
    """__repr__ should return a readable string."""
    rng = gold(seed=42)
    r = repr(rng)
    assert "ContingencyRNG" in r
    assert "seed=42" in r
