> ⚠️ **DEPRECATED — 2026-04-10.** Predictive claim built on a broken
> quantile-mapping pipeline and undocumented model selection. CRNG is a
> descriptive tool (SPECS P1), not a forecaster. See
> `REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md`. Audit only.

# r/Python Post — Catastrophic Events

## Title:
I tested my RNG against 125 years of earthquakes, financial crashes, and natural disasters. KS test: 20/20 match.

## Flair: Showcase

## Body:

**What My Project Does**

CRNG (Contingency RNG) generates random numbers with real-world statistical signatures — fat tails, volatility clustering, and kurtosis > 3. I tested whether the temporal pattern of extreme kurtosis events in CRNG matches the temporal pattern of real catastrophic events (1900-2025).

The experiment:
- Generate 30,000 CRNG points, slide a window, detect local K spikes
- Collect 82 real catastrophic events: 39 earthquakes (M≥6.7), 19 financial crashes, 24 natural disasters
- Compare gap distributions with Kolmogorov-Smirnov test

Result: **20/20 MATCH**. Not a single rejection.

```
Earthquakes vs CRNG K≥8:   p = 0.990
Natural disasters vs K≥8:  p = 0.979
All catastrophes vs K≥15:  p = 0.880
```

Bonus: FFT analysis reveals hidden quasi-periodicity in combined catastrophes (power ratio 5.46x, above 3x significance threshold). CRNG also produces significant periodicity at K≥10 (4.59x).

**Target Audience**

Researchers in risk modeling, extreme event prediction, geophysics, and anyone interested in fat-tailed distributions. The finding that standard RNGs (K=3) can't reproduce catastrophic event patterns while CRNG (K≥5) can has implications for Monte Carlo simulations in seismology, insurance, and climate science.

**Comparison**

Standard RNGs (NumPy, MT, PCG) always produce K=3 — Gaussian distributions where extreme events are vanishingly rare. Real catastrophes follow fat-tailed distributions (K=5-10 in gap sequences). CRNG matches these because it models reality as the intersection of independent oscillatory processes, not as isolated random draws.

Previous results: CRNG matches 86% of financial market metrics vs 14% for NumPy. Now extended to seismic and natural disaster domains.

```python
pip install crng

from crng import ContingencyRNG
rng = ContingencyRNG(seed=42, target_kurtosis=15.0)
```

**GitHub:** [github.com/brotto/crng](https://github.com/brotto/crng)
**Full experiment code:** `experiments/catastrophic_events.py` in the repo
