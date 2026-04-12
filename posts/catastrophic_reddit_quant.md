> ⚠️ **DEPRECATED — 2026-04-10.** Predictive claim built on a broken
> quantile-mapping pipeline and undocumented model selection. CRNG is a
> descriptive tool (SPECS P1), not a forecaster. See
> `REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md`. Audit only.

# r/quant Post — Catastrophic Events

## Title:
KS test: the gap distribution of CRNG extreme events is statistically indistinguishable from real earthquakes (p=0.990), crashes (p=0.665), and natural disasters (p=0.979)

## Body:

Following up on my earlier post about the kurtosis discriminant (K=3 for all PRNGs, K≥5 for all real markets), I tested whether CRNG's extreme event pattern matches real catastrophic events across domains.

**Method:**
- Generated 30k CRNG points, calculated sliding-window kurtosis (w=50)
- Detected "catastrophic" windows where K exceeds thresholds (K≥5, 8, 10, 15, 20, 30)
- Collected 82 real events: 39 earthquakes M≥6.7, 19 financial crashes, 24 natural disasters (1900-2025)
- Compared gap distributions (time between events) using KS test + metric matching

**Results:**

| Real vs CRNG | KS p-value | Real CV | CRNG CV |
|:--|:--:|:--:|:--:|
| Earthquakes vs K≥8 | 0.990 | 0.769 | 0.668 |
| Earthquakes vs K≥10 | 0.915 | 0.769 | 0.754 |
| Natural disasters vs K≥8 | 0.979 | 0.762 | 0.668 |
| All catastrophes vs K≥15 | 0.880 | 0.938 | 0.690 |
| Financial crashes vs K≥5 | 0.665 | 1.102 | 0.675 |

20/20 tests: MATCH (p > 0.05 in every combination).

**Hidden periodicity:**

FFT on gap sequences reveals dominant quasi-periodic components:
- Earthquakes: period = 19 gaps, power = 2.92x
- Financial crashes: period = 18 gaps, power = 2.81x
- ALL catastrophes combined: period = 78 gaps, **power = 5.46x** (significant)
- CRNG K≥10: period = 5.7 gaps, **power = 4.59x** (significant)

The combined catastrophe dataset shows periodicity well above the 3x significance threshold.

**Interpretation:**

CRNG models randomness as contingent encounters between independent oscillatory processes (irrational frequencies → resonance coupling → cascade amplification). The key insight: extreme events emerge from the same phase transition mechanism regardless of domain — seismic, financial, or meteorological. The gap distributions are domain-invariant because they all arise from the same underlying process: interference between potentialities crossing a critical amplification threshold.

Financial crashes show highest CV (1.102 — more clustered), while earthquakes and natural disasters are more regular (CV ~0.76). CRNG K≥10 best matches earthquakes (CV=0.754 vs 0.769).

**Code:** `experiments/catastrophic_events.py` at [github.com/brotto/crng](https://github.com/brotto/crng)

Curious what the quant community thinks about cross-domain extreme event universality. Is this useful for tail risk modeling?
