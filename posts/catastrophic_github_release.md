> ⚠️ **DEPRECATED — 2026-04-10.** This release note made a predictive claim
> built on a broken pipeline (quantile-mapping bug in
> `experiments/next_catastrophe.py` and undocumented model selection). Per
> SPECS.md principle P1, CRNG is a **descriptive** tool and does not justify
> predictive "next event" claims. See
> `REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md`. Text below
> preserved for audit only — do not republish.

# GitHub Release Notes — Catastrophic Events Validation

## CRNG v0.2.1 — Cross-Domain Catastrophic Event Validation

### What's New

CRNG extreme events have been validated against **125 years of real catastrophic events** across three domains: seismology, finance, and natural disasters.

### Catastrophic Events Experiment

**Data:** 82 catastrophic events (1900-2025)
- 39 earthquakes (M ≥ 6.7)
- 19 financial crashes
- 24 natural disasters

**Method:** Kolmogorov-Smirnov test comparing gap distributions between CRNG extreme K windows and real catastrophic event spacing.

**Result: 20/20 MATCH**

| Real Dataset | Best CRNG Match | KS p-value |
|:--|:--|:--:|
| Earthquakes | K ≥ 8 | 0.990 |
| Natural Disasters | K ≥ 8 | 0.979 |
| All Catastrophes | K ≥ 15 | 0.880 |
| Financial Crashes | K ≥ 5 | 0.665 |

### Hidden Periodicity

FFT analysis reveals quasi-periodic structure in catastrophic event gaps:
- All catastrophes combined: dominant period = 78 gaps, power ratio = **5.46x** (above 3x significance threshold)
- CRNG K≥10: power ratio = **4.59x**
- CRNG K≥15: power ratio = **4.81x**

### Metric Match

| Metric | Earthquakes | CRNG K≥10 |
|:--|:--:|:--:|
| CV (gap regularity) | 0.769 | 0.754 |
| ACF (momentum) | 0.127 | 0.109 |
| Hurst exponent | 0.776 | — |

### Previous Validations

- **Financial markets:** 42/49 metrics matched (86%) across 7 assets, 5 years — [v0.2.0](https://github.com/brotto/crng/releases/tag/v0.2.0)
- **Regime detection:** CALM/NORMAL/STRESSED/CRISIS classification via sliding-window calibration

### New Experiments

All code in `experiments/`:
- `coincidence_field.py` — Coincidence of two independent fields of becoming
- `recursive_potentiality.py` — Structured potentiality (CRNG feeding CRNG)
- `catastrophic_events.py` — Cross-domain catastrophic event validation

### Install

```bash
pip install crng
```

### The Philosophical Foundation

CRNG models randomness as contingent encounters between independent oscillatory processes. The key finding: extreme events across all domains — financial, seismic, meteorological — follow the same temporal distribution because they arise from the same mechanism: interference between potentialities crossing a critical amplification threshold.

The gap distributions are domain-invariant. A financial crash and an earthquake are, statistically, the same phenomenon seen from different angles.
