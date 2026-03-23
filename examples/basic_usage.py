"""Basic usage examples for CRNG."""

from crng import gold, eth, gaussian, eurusd, ContingencyRNG

# 1. Generate gold-like numbers
print("=== Gold-like (K ~ 9) ===")
rng = gold(seed=42)
values = rng.generate(100)
print(f"First 10: {[f'{v:.3f}' for v in values[:10]]}")
stats = rng.stats(10000)
print(f"K={stats['kurtosis']:.1f}, VolACF={stats['vol_clustering_acf']:.3f}, "
      f">3sigma={stats['gt_3sigma']:.1%}")

# 2. Compare presets
print("\n=== Preset Comparison ===")
for name, rng_fn in [("Gaussian", gaussian), ("Gold", gold),
                      ("EURUSD", eurusd), ("ETH", eth)]:
    rng = rng_fn(seed=42)
    s = rng.stats(10000)
    print(f"  {name:<10} K={s['kurtosis']:>6.1f}  VolACF={s['vol_clustering_acf']:.3f}  "
          f">3sigma={s['gt_3sigma']:.1%}")

# 3. Custom kurtosis
print("\n=== Custom K=15 ===")
rng = ContingencyRNG(seed=42, target_kurtosis=15.0)
s = rng.stats(10000)
print(f"Target K=15, Actual K={s['kurtosis']:.1f}")

# 4. Coin flips
print("\n=== Contingency Coin Flips ===")
rng = gold(seed=42)
flips = rng.generate_flips(1000)
print(f"P(heads) = {flips.mean():.3f}")
print(f"First 20: {''.join('H' if f else 'T' for f in flips[:20])}")

# 5. Determinism
print("\n=== Determinism Test ===")
r1 = gold(seed=123)
r2 = gold(seed=123)
v1 = [r1.next() for _ in range(5)]
v2 = [r2.next() for _ in range(5)]
print(f"Same seed, same output: {v1 == v2}")
