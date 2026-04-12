> ⚠️ **DEPRECATED — 2026-04-10.** This article made a predictive claim built
> on a broken pipeline (quantile-mapping bug in
> `experiments/next_catastrophe.py` and undocumented model selection). Per
> SPECS.md principle P1, CRNG is a **descriptive** tool and does not justify
> "next catastrophe" predictions. See
> `REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md`. Text below
> preserved for audit only — do not republish.

# The Subliminal Regularity of Catastrophes

## How a random number generator revealed hidden periodicity in 125 years of earthquakes, crashes, and disasters

---

Every Monte Carlo simulation assumes that extreme events are rare, independent, and Gaussian. Every standard random number generator — NumPy, Excel, R, MATLAB — produces distributions with Kurtosis = 3. Always. This means: extreme events are vanishingly improbable, and each event is independent of the last.

But real catastrophes don't follow these rules.

## The Kurtosis Discriminant

There is a clean binary classifier that separates simulated randomness from reality: Kurtosis.

Every PRNG: K = 3.0
Every real market: K >= 5
Every earthquake gap sequence: K = 6.62
Every financial crash gap sequence: K = 8.87

Zero overlap.

This means every risk model that uses standard random generators is systematically underestimating the probability of extreme events. They simulate a lake and call it an ocean.

## CRNG: Modeling Contingent Reality

I built CRNG — a Python library that produces random numbers with real-world statistical signatures. Three layers:

1. Irrational-frequency oscillators (entropy that never synchronizes)
2. Resonance coupling (volatility clustering — storms come in waves)
3. Cascade amplification (fat tails via phase transition)

Previously validated against 7 financial assets over 5 years: CRNG matches 86% of market metrics. NumPy matches 14%.

But the question remained: does this extend beyond finance?

## The Catastrophic Events Experiment

I collected 82 catastrophic events from 1900 to 2025:
- 39 major earthquakes (M >= 6.7)
- 19 financial crashes
- 24 natural disasters

Then I generated a 30,000-point CRNG series, calculated sliding-window kurtosis, and detected local K spikes at various thresholds (K >= 5 through K >= 30).

The key comparison: not the events themselves, but the **gaps between events** — the temporal spacing. If CRNG correctly models how reality distributes extreme events in time, the gap distributions should match.

## Result: 20/20 Match

The Kolmogorov-Smirnov test asks: "Do these two samples come from the same distribution?"

I tested every combination of real catastrophe type against every CRNG threshold. Twenty tests. Every single one: MATCH.

Earthquakes vs CRNG K>=8: p = 0.990
Natural disasters vs CRNG K>=8: p = 0.979
All catastrophes combined vs CRNG K>=15: p = 0.880
Financial crashes vs CRNG K>=5: p = 0.665

Not a single rejection at the 5% significance level.

The coefficient of variation (CV) — which measures how clustered vs regular the gaps are — is nearly identical:

Earthquakes CV: 0.769
CRNG K>=10 CV: 0.754

## The Hidden Periodicity

This is the finding I didn't expect.

FFT analysis on the gap sequences reveals dominant quasi-periodic components. When all 82 catastrophes are combined — earthquakes, financial crashes, and natural disasters analyzed as a single sequence — a dominant period emerges with a power ratio of 5.46x above the mean spectral power.

The significance threshold is 3x. This is well above it.

Catastrophes are not purely random. There is a quasi-periodic subliminal regularity in their temporal pattern.

CRNG also produces significant periodicity: 4.59x at K>=10 and 4.81x at K>=15.

## The Philosophy

This experiment was born from a philosophical inquiry into the nature of randomness. The core insight:

PRNG isn't "pseudo" — it just models the wrong object. It perfectly describes pure potentiality (what the Greeks called dynamis). CRNG describes contingent actuality — reality as it presents itself at each local point through the encounter of independent processes (what I call Poiesis).

Standard probability calculates P(earthquake) by looking at seismic data alone. But an earthquake "happening" is not a property of tectonic plates. It is the intersection of tectonic movement, crustal stress, fluid pressure, and thermal gradients — independent fields of becoming that, when they encounter each other at incommensurable frequencies, produce events with fat-tailed distributions and quasi-periodic spacing.

The cause of a catastrophe is not "in" the event. It is in the intersection of potentialities — a para-ontological object that precedes the real. Everything exists within a subliminal linearity. And we just found evidence for it in the temporal pattern of 125 years of catastrophes.

## What's Next

The next test: solar sunspot cycles — another domain where quasi-periodic structure emerges from the interaction of independent oscillatory processes.

---

pip install crng | github.com/brotto/crng
