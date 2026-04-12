> ⚠️ **RETRACTION 2026-04-10 — the "42/49 (86%)" headline is withdrawn.**
> Unfrozen data + a kurtosis semantic bug (`np.diff(values)` in `stats()`).
> Corrected, frozen numbers live in `posts/benchmark_errata_2026-04-10.md`
> (snapshot SHA256 `82f8b5e5…ec9fff5`; CRNG is closer to the real
> fingerprint on 16 of 21 comparison cells: 7 assets × 3 metrics).
> The body text below is preserved for audit and reflects the retracted
> pre-2026-04-10 claims — do not cite it.

# r/quant Post

## Title:
The kurtosis discriminant: every PRNG has K=3, every real market has K>=5. Zero overlap. I built a generator that fixes this.

## Body:

I've been analyzing the statistical gap between pseudo-random number generators and real market data. The finding is stark:

**Every PRNG ever built:** K = 3.0 (Mersenne Twister, PCG, xoshiro, numpy.random — all of them)

**Every real market ever measured:** K >= 5

| Asset | Kurtosis | Vol Clustering (ACF) |
|:--|:--:|:--:|
| NumPy | 3.0 | 0.00 |
| S&P 500 | 9.6 | 0.19 |
| Gold | 9.3 | 0.27 |
| EURUSD | 10.5 | 0.21 |
| Ethereum | 22.9 | 0.29 |
| Bitcoin | 218.7 | 0.31 |

Zero overlap. A binary classifier.

This means every Monte Carlo simulation, every VaR calculation, every stress test that uses `np.random.normal()` is systematically underestimating tail risk. The generator produces a Gaussian world (K=3), but markets live in a fat-tailed world (K=5-220).

I built **CRNG** (`pip install crng`) — a random number generator based on three layers:

1. Irrational-frequency oscillators (maximum entropy, incommensurable)
2. Resonance coupling (reproduces volatility clustering)
3. Cascade amplification with a critical threshold (reproduces fat tails via phase transition)

Validated against 7 assets, 5 years of daily data, 7 metrics (kurtosis, PE, Hurst, DFA, vol clustering, tail events, skewness), 10 seeds:

- **CRNG: 42/49 metrics (86%)**
- **NumPy: 7/49 (14%)**

Also built a regime detector that calibrates CRNG on sliding windows to classify market conditions. Current readings (March 2026): SPY is CALM (K=2.8) at 60d but STRESSED (K=26) at 252d — the crash signature lingers at longer scales, exactly as the model predicts (kurtosis convergence).

The theoretical foundation comes from modeling randomness as contingent encounters between independent oscillatory processes — basically, the fat tails and vol clustering emerge from resonance and supercritical cascade amplification, not from agent-based models or information asymmetry.

Paper and full methodology on GitHub: [github.com/brotto/crng](https://github.com/brotto/crng)

Interactive demo: [Google Colab](https://colab.research.google.com/github/brotto/crng/blob/main/notebooks/crng_demo.ipynb)

Curious what the quant community thinks. Is this useful for your Monte Carlo workflows? What am I missing?
