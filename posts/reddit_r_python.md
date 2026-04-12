> ⚠️ **RETRACTION 2026-04-10 — the "42/49 (86%)" headline is withdrawn.**
> Unfrozen data + a kurtosis semantic bug (`np.diff(values)` in `stats()`).
> Corrected, frozen numbers live in `posts/benchmark_errata_2026-04-10.md`
> (snapshot SHA256 `82f8b5e5…ec9fff5`; CRNG is closer to the real
> fingerprint on 16 of 21 comparison cells: 7 assets × 3 metrics).
> The body text below is preserved for audit and reflects the retracted
> pre-2026-04-10 claims — do not cite it.

# r/Python Post

## Title:
I built a RNG that reproduces 86% of real financial market statistics. NumPy matches 14%.

## Body:

Every random number generator produces Gaussian distributions with Kurtosis = 3.0. NumPy, Excel, R — all of them. Always.

Real financial markets have Kurtosis = 5 to 220. Zero overlap. This means every Monte Carlo simulation using `numpy.random.normal()` systematically underestimates extreme events.

I built **CRNG** — a Python library that produces random numbers with real market statistical signatures using three layers:

1. **Irrational oscillators** (pi, sqrt(2), e) — entropy source that never synchronizes
2. **Resonance coupling** — creates volatility clustering (big moves followed by big moves)
3. **Cascade amplification** — creates fat tails via a phase transition mechanism

Tested against 5 years of real data (Gold, S&P 500, BTC, ETH, Oil, EURUSD, USDJPY), 7 metrics, 10 seeds: **CRNG matches 42/49 metrics (86%)**. NumPy matches 7/49.

```python
pip install crng

from crng import gold, from_data

# Preset (frozen configuration, not a model)
rng = gold(seed=42)
values = rng.generate(10000)  # log-returns

# Or auto-calibrate from any real data (recommended)
rng = from_data(your_price_series)
```
<!-- errata 2026-04-10: previous snippet used `ContingencyRNG(preset='gold')`,
     which is not a valid constructor keyword. Use the preset factory
     functions (gold, eth, btc, ...) or construct with target_kurtosis and
     vol_clustering directly. Fixed per Codex review 2026-04 item P2.1. -->


**Try it live:** [Google Colab notebook](https://colab.research.google.com/github/brotto/crng/blob/main/notebooks/crng_demo.ipynb)

**GitHub:** [github.com/brotto/crng](https://github.com/brotto/crng)

Also includes a real-time **regime detector** that classifies market conditions (CALM/NORMAL/STRESSED/CRISIS) using CRNG calibration on sliding windows.

Pure Python + NumPy. No external dependencies. ~5M numbers/second.

Happy to answer questions about the theory (it involves spinning coins, Wittgenstein, and phase transitions — seriously).
