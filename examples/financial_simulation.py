"""
Example: Using CRNG for financial simulation with realistic fat tails.

Compare a Monte Carlo simulation using Gaussian noise vs CRNG noise.
"""

import numpy as np
from crng import gold, gaussian

N_PATHS = 1000
N_STEPS = 252  # ~1 trading year
S0 = 100.0
MU = 0.08 / 252  # 8% annual drift
SIGMA = 0.15 / np.sqrt(252)  # 15% annual vol

print("Monte Carlo: Gaussian vs Gold-like noise")
print("=" * 50)

# Gaussian paths
rng_g = gaussian(seed=42)
gaussian_paths = np.zeros((N_PATHS, N_STEPS + 1))
gaussian_paths[:, 0] = S0
for i in range(N_PATHS):
    rng_g.reset(seed=42 + i)
    for t in range(N_STEPS):
        noise = rng_g.next() * SIGMA
        gaussian_paths[i, t + 1] = gaussian_paths[i, t] * np.exp(MU + noise)

# Gold-like paths (realistic fat tails)
rng_m = gold(seed=42)
market_paths = np.zeros((N_PATHS, N_STEPS + 1))
market_paths[:, 0] = S0
for i in range(N_PATHS):
    rng_m.reset(seed=42 + i)
    for t in range(N_STEPS):
        noise = rng_m.next() * SIGMA
        market_paths[i, t + 1] = market_paths[i, t] * np.exp(MU + noise)

# Compare
final_g = gaussian_paths[:, -1]
final_m = market_paths[:, -1]

print(f"\n{'Metric':<30} {'Gaussian':>12} {'Gold-like':>12}")
print("-" * 55)
print(f"{'Mean final price':<30} ${np.mean(final_g):>11.2f} ${np.mean(final_m):>11.2f}")
print(f"{'Median final price':<30} ${np.median(final_g):>11.2f} ${np.median(final_m):>11.2f}")
print(f"{'Std of final price':<30} ${np.std(final_g):>11.2f} ${np.std(final_m):>11.2f}")
print(f"{'5th percentile (VaR 95%)':<30} ${np.percentile(final_g, 5):>11.2f} ${np.percentile(final_m, 5):>11.2f}")
print(f"{'1st percentile (VaR 99%)':<30} ${np.percentile(final_g, 1):>11.2f} ${np.percentile(final_m, 1):>11.2f}")
print(f"{'Max drawdown (worst path)':<30} {np.min(final_g/S0 - 1):>11.1%} {np.min(final_m/S0 - 1):>11.1%}")
print(f"{'Paths ending > $200':<30} {np.mean(final_g > 200):>11.1%} {np.mean(final_m > 200):>11.1%}")
print(f"{'Paths ending < $50':<30} {np.mean(final_g < 50):>11.1%} {np.mean(final_m < 50):>11.1%}")

print("\nKey insight: Gold-like noise produces wider tails in the final")
print("price distribution, better capturing real market crash/boom risk.")
