"""Tests for the Contingency RNG."""

import numpy as np
import pytest
from crng import ContingencyRNG, gaussian, gold, eth

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

def test_high_entropy():
    """All presets should have PE > 0.99."""
    for fn in [gaussian, gold, eth]:
        rng = fn(seed=42)
        s = rng.stats(10000)
        assert s['PE_4'] > 0.99, f"{fn.__name__} PE too low: {s['PE_4']}"

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

def test_reset():
    """reset() restores initial state."""
    rng = gold(seed=42)
    v1 = [rng.next() for _ in range(50)]
    rng.reset()
    v2 = [rng.next() for _ in range(50)]
    assert v1 == v2

def test_custom_kurtosis():
    """Custom target kurtosis should influence actual kurtosis."""
    rng_low = ContingencyRNG(seed=42, target_kurtosis=3.0)
    rng_high = ContingencyRNG(seed=42, target_kurtosis=30.0)
    s_low = rng_low.stats(20000)
    s_high = rng_high.stats(20000)
    assert s_high['kurtosis'] > s_low['kurtosis']

def test_vol_clustering():
    """Higher vol_clustering parameter should increase ACF of |diffs|."""
    rng_no = ContingencyRNG(seed=42, target_kurtosis=9, vol_clustering=0.0)
    rng_hi = ContingencyRNG(seed=42, target_kurtosis=9, vol_clustering=0.5)
    s_no = rng_no.stats(20000)
    s_hi = rng_hi.stats(20000)
    assert s_hi['vol_clustering_acf'] > s_no['vol_clustering_acf']
