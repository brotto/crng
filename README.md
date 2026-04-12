# CRNG — Contingency Random Number Generator

**A descriptive random number generator whose output carries fat tails, volatility clustering and heavier-than-Gaussian scale signatures. Not a forecaster.**

CRNG produces sequences whose *statistical fingerprint* (kurtosis, tail frequency, volatility autocorrelation) resembles real financial markets far more closely than a plain Gaussian PRNG does. It does **not** predict future prices. See [`SPECS.md`](SPECS.md) principle P1 for the descriptive/predictive separation that this project enforces.

> This README is scoped to empirical claims. Any claim that cites a number must also cite the frozen snapshot it came from. See the section [**Evidence and reproducibility**](#evidence-and-reproducibility).

---

## Real-market descriptive benchmark (snapshot 2026-04)

Frozen snapshot: `benchmarks/snapshot_2026-04/prices.csv`
SHA256: `82f8b5e5abe2f9d084769898b8d3b6ffefc5cfbd1c2757531df76d049ec9fff5`
Window: 2021-04-10 → 2026-04-10 (daily close, yfinance)
Selection rule (a priori, SPECS P3): for every asset, CRNG is built via `from_data(prices, seed=42)` on the full window. No preset picking. Baseline: `iid_gaussian(seed=42)`. `n` equals the number of real log-returns for each asset.

Re-run with `python3 benchmarks/frozen_benchmark.py`. Full numbers live in `benchmarks/snapshot_2026-04/frozen_benchmark_report.json`.

### Kurtosis (target = real returns)

| Asset   | Real K | CRNG K | iid Gauss K | Closer to real |
|:--------|------:|-------:|------------:|:--------------:|
| Gold    | 15.39 |   8.09 |        3.04 | CRNG |
| S&P 500 |  9.47 |   7.17 |        3.04 | CRNG |
| ETH     |  8.31 |   5.95 |        3.01 | CRNG |
| Oil     |  8.26 |   6.75 |        3.04 | CRNG |
| BTC     |  6.96 |   5.77 |        3.01 | CRNG |
| USDJPY  |  5.95 |   3.06 |        3.03 | CRNG |
| EURUSD  |  4.89 |   7.73 |        3.03 | iid  |

CRNG is closer to the real kurtosis on 6 of 7 assets. It overshoots on EURUSD, the most Gaussian-like asset in the set. That is an honest miss, not a failure mode we hide.

### Tail frequency — |z| > 3σ (% of observations)

| Asset   | Real  | CRNG  | iid Gauss | Closer |
|:--------|------:|------:|----------:|:------:|
| Gold    | 1.11  | 0.95  | 0.16      | CRNG |
| S&P 500 | 1.04  | 0.96  | 0.16      | CRNG |
| ETH     | 1.70  | 0.93  | 0.11      | CRNG |
| Oil     | 1.27  | 0.88  | 0.16      | CRNG |
| BTC     | 1.97  | 0.93  | 0.11      | CRNG |
| USDJPY  | 1.39  | 0.23  | 0.15      | CRNG |
| EURUSD  | 1.00  | 0.92  | 0.15      | CRNG |

CRNG is closer on 7 of 7. An iid Gaussian under-reports three-sigma events by a factor of 6.5× (S&P 500) to 18.0× (BTC), with the full per-asset ratios recoverable from the table above and from `benchmarks/snapshot_2026-04/frozen_benchmark_report.json`.

### Volatility clustering — ACF(|returns|) at lag 1

| Asset   | Real  | CRNG   | iid Gauss | Closer |
|:--------|------:|-------:|----------:|:------:|
| Gold    | +0.103 | +0.043 | +0.035 | CRNG |
| S&P 500 | +0.177 | +0.008 | +0.035 | iid  |
| ETH     | +0.168 | +0.024 | +0.006 | CRNG |
| Oil     | +0.121 | +0.017 | +0.035 | iid  |
| BTC     | +0.145 | +0.021 | +0.006 | CRNG |
| USDJPY  | +0.102 | −0.031 | +0.033 | iid  |
| EURUSD  | +0.124 | −0.044 | +0.033 | iid  |

**This is the honest weakness.** Both generators under-reproduce real volatility clustering, and on 4 of 7 assets the iid residual noise sits closer to the real ACF than CRNG does. The clustering mechanism in the current CRNG is weaker than its target nominally suggests. This is documented, not glossed.

### Summary for this snapshot

| Metric | CRNG wins | iid wins |
|:--|:--:|:--:|
| Kurtosis   | 6 | 1 |
| Tail 3σ    | 7 | 0 |
| Vol ACF(1) | 3 | 4 |
| **Total**  | **16** | **5** |

CRNG is closer to the real fingerprint on **16 of 21** comparison cells (7 assets × 3 metrics). All numbers are reproducible from `benchmarks/snapshot_2026-04/frozen_benchmark_report.json`.

> ⚠️ Prior versions of this README advertised "CRNG wins 42/49 metrics (86%)". That number came from a benchmark with two known defects: (1) the samples were re-downloaded at each run, so the evidence was not reproducible, and (2) the `stats()` routine measured kurtosis on `np.diff(values)` instead of `values`, inflating the apparent fit. Both are fixed and documented in `REVIEWS/codex_review_2026-04.md`. The table above is the replacement.

---

## Installation

```bash
pip install crng
```

## Quick Start

```python
from crng import (
    ContingencyRNG, from_data,
    iid_gaussian,         # true iid baseline (numpy default_rng)
    gaussian,             # internal reference — NOT the baseline, see note below
    gold, eurusd, eth, btc,
)

# Auto-calibrate from real data (the normal way to use CRNG)
import numpy as np
my_prices = np.array([...])          # daily closes
rng = from_data(my_prices, seed=42)  # internally takes log-returns
xs = rng.generate(len(my_prices))    # log-returns with matched fingerprint

# Preset: a frozen configuration that targets a particular kurtosis
rng = gold(seed=42)                  # target kurtosis 9.26, vol_clustering 0.3
xs = rng.generate(1000)

# The iid baseline for any CRNG-vs-PRNG comparison
baseline = iid_gaussian(seed=42)
zs = baseline.generate(1000)         # plain numpy standard_normal
```

### Interpretation

`rng.generate(n)` returns **log-scale returns**, already centred near zero. It does not return prices. To build a synthetic price path, integrate:

```python
log_prices = np.cumsum(rng.generate(n))
prices = starting_price * np.exp(log_prices)
```

This is the single semantic convention in the project (SPECS principle P5). `stats()` measures directly on the raw output because the raw output already *is* the return series.

---

## How it works (three layers)

1. **Coupled irrational oscillators.** Two banks of sine oscillators whose frequencies are products of irrationals (π, e, √2, φ, √3, √5, √7). Because the frequency ratios are irrational, the combined signal is quasi-periodic and never repeats — the entropy floor is set here.
2. **Resonance coupling.** Each oscillator pair is weighted by a Gaussian of its frequency ratio: near-resonant pairs contribute more, far pairs contribute less. This introduces slow, amplitude-varying structure — the seed of volatility clustering.
3. **Cascade amplifier.** When recent output magnitudes exceed an adaptive threshold, the next output is scaled up. Below a critical amplification value the cascade dissipates (kurtosis ≈ 3). Above it, cascades self-amplify (kurtosis ≫ 3).

The cascade's transition from dissipative to self-amplifying is sharp and is what produces the heavy-tailed regime.

---

## API

### `ContingencyRNG(seed, target_kurtosis, vol_clustering, ...)`

| Parameter | Default | Description |
|:--|:--:|:--|
| `seed` | 42 | Reproducibility seed |
| `target_kurtosis` | 9.26 | Desired kurtosis of the output (3 = Gaussian, ≫3 = fat-tailed) |
| `vol_clustering` | 0.3 | Vol-clustering strength, 0..1 |
| `n_oscillators` | 4 | Number of oscillator pairs |
| `cascade_threshold` | 1.2 | Cascade adaptive threshold multiplier |
| `cascade_memory` | 20 | Cascade memory window |

### Methods

| Method | Returns | Description |
|:--|:--|:--|
| `next()` | `float` | Single log-return |
| `generate(n)` | `ndarray` | n log-returns |
| `flip()` | `int` | 0 or 1 (sign of next()) |
| `generate_flips(n)` | `ndarray` | n coin flips |
| `uniform(low, high)` | `float` | Deterministic CDF transform of `next()` (see P5) |
| `reset(seed)` | `None` | Reset |
| `stats(n)` | `dict` | Fingerprint on the raw return series |

### Presets — target vs achieved

These are **frozen configurations** that instantiate `ContingencyRNG` with a specific `target_kurtosis` and `vol_clustering`. They are not claims about the real asset; they are convenient starting points whose achieved fingerprint you should check before relying on.

Measured at `n=100_000` over 10 seeds (`seeds=[42, 123, 256, 314, 555, 777, 1001, 1337, 2025, 9999]`). Regenerate via `python3 benchmarks/measure_preset_fingerprints.py`. Source of truth: `benchmarks/preset_fingerprints.json`.

| Preset | Target K | Achieved K (μ±σ) | Target ACF₁ | Achieved ACF₁ (μ±σ) | Achieved 3σ% |
|:--|--:|:--:|--:|:--:|--:|
| `iid_gaussian()` | 3.00 | 3.01 ± 0.01  | 0.00 | −0.001 ± 0.002 | 0.28 |
| `gaussian()`     | 3.00 | 2.83 ± 0.13  | 0.00 | −0.013 ± 0.046 | 0.18 |
| `gold()`         | 9.26 | 13.90 ± 0.88 | 0.30 | +0.021 ± 0.025 | 0.93 |
| `eurusd()`       | 10.50| 15.37 ± 1.11 | 0.25 | +0.021 ± 0.025 | 0.95 |
| `eth()`          | 22.85| 46.65 ± 2.90 | 0.40 | +0.020 ± 0.016 | 0.95 |
| `btc()`          | 219  | 172.47 ± 21.58 | 0.50 | +0.013 ± 0.007 | 0.84 |

**Read this table carefully.** The presets do **not** hit their advertised targets. `gold()` overshoots kurtosis (target 9.26, achieved 13.90 ± 0.88). `btc()` undershoots (target 219, achieved 172.47 ± 21.58). Every preset undershoots the ACF₁ target substantially: ratios of target-to-achieved ACF₁ range from 12.1× (`eurusd()`) to 37.7× (`btc()`), computed from `benchmarks/preset_fingerprints.json`. This is **reported honestly** per SPECS principle P4 and is one of the reasons we recommend `from_data()` over presets whenever you have real data.

`gaussian()` is an **internal reference**, not a baseline. It asks the oscillator/cascade machinery to imitate iid Gaussian and reports what the architecture still introduces. For any CRNG-vs-PRNG comparison, use `iid_gaussian()` instead (SPECS P6).

### `from_data(data, seed=42, calibration_rounds=5)`

Auto-calibrate from real data. Accepts prices (log-returns are computed internally) or returns. Returns a `ContingencyRNG` tuned to match the data's kurtosis and a scaled version of its `|r|` autocorrelation. This is the primary entry point for descriptive use.

```python
import yfinance as yf
prices = yf.Ticker("GC=F").history(period="5y")["Close"].values
rng = from_data(prices, seed=42)
synthetic_returns = rng.generate(len(prices))
```

---

## Evidence and reproducibility

Every numeric claim in this README is tied to a frozen artifact:

- **Real-market benchmark.** `benchmarks/snapshot_2026-04/prices.csv` + `prices.sha256` + `metadata.json` + `frozen_benchmark_report.json`. Freeze script: `benchmarks/freeze_snapshot_2026-04.py`. Consumer: `benchmarks/frozen_benchmark.py`.
- **Preset fingerprints.** `benchmarks/preset_fingerprints.json`. Generator: `benchmarks/measure_preset_fingerprints.py`.
- **Known errata and reviews.** `REVIEWS/codex_review_2026-04.md` plus each fix under `REVIEWS/errata/`.

Snapshots are immutable by protocol. To produce a new benchmark window, create `benchmarks/snapshot_2026-05/` rather than overwriting. The consumer script verifies the snapshot SHA256 before running; mismatches abort.

---

## Use cases

- **Scenario generation for stress tests.** Replace Gaussian noise with CRNG output so cascade-risk tests actually see fat tails.
- **Monte Carlo with structure.** On the 2026-04 snapshot, the real-asset 3σ event rate ranges from 1.00% (EURUSD) to 1.97% (BTC). CRNG's achieved rate ranges from 0.23% (USDJPY) to 0.96% (S&P 500) — consistently closer to real than iid (0.11%–0.16%), but not fully reproducing the real tail frequency on any of the seven assets. Full per-asset numbers in `benchmarks/snapshot_2026-04/frozen_benchmark_report.json`.
- **Teaching.** Side-by-side comparison of iid Gaussian and CRNG makes the fat-tail / clustering distinction visible from a thousand samples.

## What CRNG does **not** do

- It does **not** forecast future prices or returns. See SPECS P1.
- It does **not** capture the temporal structure of a specific asset path — only its statistical fingerprint.
- It does **not** replicate regime breaks, macro events, or conditional dependence beyond its calibration window. Descriptive ≠ causal.

## Performance

Roughly 5M samples per second per core. Pure NumPy, no external dependencies.

## Paper

> Brotto, A. (2026). *Contingency as Mechanism: Resonance Cascades as a Descriptive Bridge Between iid Noise and Market-Like Returns.* arXiv preprint, forthcoming.

## License

MIT
