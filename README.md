# CRNG — Contingency Random Number Generator

**Numbers that behave like real markets, not like dice.**

CRNG generates random numbers with controllable fat tails, volatility clustering, and scale convergence — statistical properties present in real financial markets but absent from conventional PRNGs.

| Property | Conventional PRNG | CRNG | Real Market (Gold) |
|:--|:--:|:--:|:--:|
| Kurtosis (K) | 3.0 | **9.8** | 9.26 |
| Vol clustering | 0.00 | **0.35** | 0.27 |
| >3-sigma events | 0.3% | **1.2%** | 1.9% |
| Scale convergence | Flat | **Natural** | Natural |
| Permutation entropy | 0.989 | **0.999** | 0.998 |

## Installation

```bash
pip install crng
```

## Quick Start

```python
from crng import gold, eth, gaussian, ContingencyRNG

# Gold-market-like numbers (K ~ 9)
rng = gold(seed=42)
x = rng.next()           # single number
xs = rng.generate(1000)   # array of 1000

# ETH-crypto-like numbers (K ~ 23)
rng = eth(seed=42)
xs = rng.generate(1000)

# Gaussian baseline (K ~ 3, like a normal PRNG)
rng = gaussian(seed=42)
xs = rng.generate(1000)

# Custom kurtosis target
rng = ContingencyRNG(seed=42, target_kurtosis=15.0, vol_clustering=0.3)
xs = rng.generate(1000)

# Binary (coin flip with contingency)
face = rng.flip()          # 0 or 1
flips = rng.generate_flips(1000)

# Validate against baselines
stats = rng.stats(10000)
print(f"K={stats['kurtosis']:.1f}, VolACF={stats['vol_clustering_acf']:.3f}")
```

## How It Works

CRNG is based on three layered mechanisms:

### Layer 1: Coupled Irrational Oscillators
Two sets of sine oscillators with frequencies based on irrational numbers (pi, e, sqrt(2), golden ratio...). Their incommensurability guarantees quasi-periodic patterns that never repeat — maximum entropy with zero periodicity.

### Layer 2: Resonance Coupling
Each oscillator pair is coupled with strength proportional to their frequency proximity. Near-resonant pairs produce strong signals; far pairs produce weak ones. This creates natural variation in amplitude — the seed of volatility clustering.

### Layer 3: Cascade Amplifier
A feedback loop where extreme values amplify subsequent values. When recent outputs exceed a threshold, the next output is scaled up — creating cascading extremes (fat tails). The system exhibits a **phase transition**: below a critical amplification, cascades dissipate (K ~ 3); above, they self-amplify (K >> 3).

## Presets

| Preset | Target K | Actual K | Modeled After |
|:--|:--:|:--:|:--|
| `gaussian()` | 3.0 | 2.8 | Pure PRNG |
| `gold()` | 9.3 | 9.8 | Gold (XAU/USD) |
| `eurusd()` | 10.5 | 9.6 | EUR/USD forex |
| `eth()` | 22.9 | 25.4 | Ethereum |
| `btc()` | 219 | 91 | Bitcoin (extreme) |

## Use Cases

- **Financial simulation**: Generate scenarios with realistic fat tails for backtesting
- **Stress testing**: Model cascade failures and extreme events
- **Monte Carlo with structure**: Replace Gaussian noise with market-like noise
- **Generative art**: Patterns with emergent structure, not white noise
- **Research**: Study the transition from Gaussian to fat-tailed behavior

## The Phase Transition

The most striking finding: the transition from Gaussian (K=3) to fat-tailed (K>>3) is a **phase transition**, not a smooth parameter change.

```
K
50+ |                        * (amp=9)
    |                   * (amp=8.5)
25  |              * (amp=7)
    |
9   |         * (amp=4.5)  <-- Gold lives here
5   |     * (amp=3)
3   |--*--*--*-----------  <-- CRITICAL THRESHOLD
    |  (amp=0..2.5)
    +------------------------
      Amplification parameter
```

Below the threshold: cascades dissipate. Above: they self-amplify exponentially. Real markets appear to operate just above this critical point.

## Paper

The research behind CRNG is documented in:

> Brotto, A. (2026). "Contingency as Mechanism: How Resonance Cascades Bridge the Gap Between Pseudo-Random and Market-Like Time Series." [arXiv preprint]

## License

MIT
