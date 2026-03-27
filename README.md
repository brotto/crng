# CRNG — Contingency Random Number Generator

**Numbers that behave like real markets, not like dice.**

CRNG generates random numbers with controllable fat tails, volatility clustering, and scale convergence — statistical properties present in real financial markets but absent from conventional PRNGs.

## Real-World Validation

Tested against **7 real assets, 5 years of daily data, 7 metrics, 10 seeds**:

| Asset | Real K | CRNG K | NumPy K | CRNG Score |
|:------|-------:|-------:|--------:|:----------:|
| Gold | 15.6 | 11.2 | 3.0 | 6/7 |
| S&P 500 | 9.6 | 8.4 | 3.0 | 6/7 |
| ETH | 8.2 | 8.5 | 3.0 | **7/7** |
| BTC | 6.9 | 7.3 | 3.0 | **7/7** |
| Oil | 6.4 | 8.5 | 3.0 | 6/7 |
| USDJPY | 6.0 | 5.1 | 3.0 | 5/7 |
| EURUSD | 4.9 | 3.1 | 3.0 | 5/7 |

**CRNG wins 42/49 metrics (86%).** NumPy PRNG produces K=3.0 for everything.

Full comparison: [`examples/real_world_comparison.py`](examples/real_world_comparison.py)

## Installation

```bash
pip install crng
```

## Quick Start

```python
from crng import gold, eth, gaussian, ContingencyRNG, from_data

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

# Auto-calibrate from real data (prices or returns)
rng = from_data(my_price_series, seed=42)
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

## API

### `ContingencyRNG(seed, target_kurtosis, vol_clustering, ...)`

| Parameter | Default | Description |
|:--|:--:|:--|
| `seed` | 42 | Reproducibility seed |
| `target_kurtosis` | 9.26 | Target kurtosis (3=Gaussian, 9=Gold, 23=ETH) |
| `vol_clustering` | 0.3 | Volatility clustering strength (0-1) |
| `n_oscillators` | 4 | Number of oscillator pairs |
| `cascade_threshold` | 1.2 | Cascade trigger threshold |
| `cascade_memory` | 20 | Cascade memory window |

### Methods

| Method | Returns | Description |
|:--|:--|:--|
| `next()` | `float` | Single random number |
| `flip()` | `int` | 0 or 1 |
| `generate(n)` | `ndarray` | Array of n numbers |
| `generate_flips(n)` | `ndarray` | Array of n flips |
| `uniform(low, high)` | `float` | Mapped to [low, high] |
| `reset(seed)` | `None` | Reset to initial state |
| `stats(n)` | `dict` | Statistical summary |

### Presets

| Preset | Target K | Modeled After |
|:--|:--:|:--|
| `gaussian()` | 3.0 | Pure PRNG |
| `gold()` | 9.3 | Gold (XAU/USD) |
| `eurusd()` | 10.5 | EUR/USD forex |
| `eth()` | 22.9 | Ethereum |
| `btc()` | 219 | Bitcoin (extreme) |

### `from_data(data, seed=42)`

Auto-calibrate from real data. Pass prices (will compute log returns) or returns directly. Uses iterative calibration to match the data's kurtosis and vol clustering.

```python
import yfinance as yf
prices = yf.Ticker("AAPL").history(period="2y")["Close"].values
rng = from_data(prices, seed=42)
synthetic = rng.generate(len(prices))
```

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

## Performance

~5M numbers/second on a single core. Pure NumPy, no external dependencies.

## Paper

> Brotto, A. (2026). "Contingency as Mechanism: How Resonance Cascades Bridge the Gap Between Pseudo-Random and Market-Like Time Series." [arXiv preprint]

## License

MIT
