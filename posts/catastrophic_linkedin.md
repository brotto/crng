> ⚠️ **DEPRECATED — 2026-04-10.** This post made a predictive claim built on
> a broken pipeline (quantile-mapping bug in `experiments/next_catastrophe.py`
> and undocumented model selection). Per SPECS.md principle P1, CRNG is a
> **descriptive** tool and cannot justify a predictive "next catastrophe"
> headline. See `REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md`
> and the replacement errata post. Text below preserved for audit only — do
> not republish.

# LinkedIn Post — Catastrophic Events Match

We tested our algorithm against 125 years of catastrophes. The result challenges everything we think we know about randomness.

--

THE EXPERIMENT

We built CRNG — a random number generator based on contingent encounters between oscillators with irrational frequencies. Unlike NumPy/Excel (which always produce Gaussian distributions with Kurtosis = 3), CRNG produces the fat tails and volatility clustering found in real-world data.

We asked: does the temporal pattern of extreme events in CRNG match the temporal pattern of real catastrophic events?

--

THE DATA

82 catastrophic events from 1900 to 2025:
- 39 major earthquakes (M >= 6.7)
- 19 financial crashes (Black Monday, Lehman, COVID...)
- 24 natural disasters (tsunamis, hurricanes, pandemics)

We measured the GAP distribution between events — not the events themselves, but the temporal spacing between them.

--

THE RESULT: 20/20 MATCH

We ran the Kolmogorov-Smirnov test against every combination of real catastrophe type and CRNG threshold. Result:

Earthquakes vs CRNG: p = 0.990
Natural disasters vs CRNG: p = 0.979
All catastrophes vs CRNG: p = 0.880

All 20 tests: MATCH. Not a single rejection. The gap distributions are statistically indistinguishable.

--

HIDDEN PERIODICITY

FFT analysis on the gap sequences reveals something unexpected: when ALL catastrophes are combined (earthquakes + financial crashes + natural disasters), there is a dominant quasi-periodic signal with a power ratio of 5.46x above mean — well above the 3x significance threshold.

Catastrophes are not purely random. There is a subliminal regularity in their temporal pattern.

--

WHAT THIS MEANS

The philosophical foundation: what we call "randomness" is interference between potentialities (Δυναμον). Standard random generators (PRNG) aren't wrong — they just model the wrong object. They describe pure potentiality. CRNG describes contingent actuality — reality as it presents itself at each local point through the encounter of independent processes (Ποεσις).

The coherence is remarkable: PRNG generates K=3 (Gaussian, a calm lake). Real markets, real earthquakes, and real catastrophes all show K>=5 (fat tails, an ocean). CRNG reproduces this because it models the INTERSECTION of fields of becoming, not isolated events.

The cause of a catastrophe is not "in" the event. It is in the intersection of potentialities — a para-ontological object that precedes the real. Everything exists within a subliminal linearity. We just found evidence for it.

pip install crng
github.com/brotto/crng

#DataScience #RiskManagement #Earthquakes #Python #Philosophy #PhaseTransition #OpenSource #QuantFinance
