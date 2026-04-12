> ⚠️ **RETRACTION 2026-04-10 — the "42/49 (86%)" headline in this post is
> withdrawn.** The original benchmark used unfrozen data and a `stats()`
> routine with a semantic bug (kurtosis measured on `np.diff(values)`). See
> `posts/benchmark_errata_2026-04-10.md` for the corrected frozen numbers
> (CRNG is closer to the real fingerprint on 16 of 21 comparison cells on
> snapshot 2026-04, SHA256 `82f8b5e5…ec9fff5`). The architectural narrative
> below still stands; the retracted numerical headline does not — do not
> cite it.

# Why Every Monte Carlo Simulation You've Ever Run Is Wrong

## Your risk model simulates a lake. Real markets are an ocean.

Every financial simulation in the world — every Value-at-Risk calculation, every portfolio stress test, every options pricing model — uses random numbers from `numpy.random.normal()` or equivalent.

These generators produce numbers with **Kurtosis = 3.0**. Always. Every single one. NumPy, Excel, R, MATLAB, Mersenne Twister, PCG, xoshiro — all of them. K = 3.0.

Real financial markets? **K = 5 to 220.** Zero overlap.

This means every risk model on Wall Street is systematically underestimating the probability of extreme events. It's like planning the construction of a deepwater port by studying a lake.

---

## What Is Kurtosis? (The Lake vs Ocean Analogy)

Think of a calm lake. The waves are small, predictable, all about the same size. Most waves are 30cm. Rarely one reaches 1 meter. A 5-meter wave? Never. This is **K = 3** — the Gaussian world, the bell curve from your statistics textbook.

Now think of a real ocean. Most waves are still 30cm. But occasionally, without warning, a **10-meter wave** appears. And very rarely, a **30-meter tsunami**. This is **K = 9, 20, 200**.

Kurtosis measures exactly this: **how often do unexpectedly large waves appear?**

| Asset | Kurtosis | Analogy |
|:--|:--:|:--|
| NumPy / Excel / R | 3.0 | Calm lake |
| S&P 500 | 9.6 | Open ocean |
| Gold | 9.3 | Open ocean |
| EURUSD | 10.5 | Stormy sea |
| Ethereum | 22.9 | Earthquake zone |
| Bitcoin | 218.7 | Tsunami day |

**The gap between 3.0 and 9.0 is not a rounding error.** It means extreme events happen 10x to 100x more often than your simulation predicts.

---

## The Second Problem: Storms Come in Waves

Real markets have another property that no random number generator reproduces: **volatility clustering**.

When it rains hard one day, the chance of rain the next day is higher. Storms come in blocks. Three days of heavy rain, then a week of sunshine, then two days of tempest.

Markets behave the same way. A sharp drop is rarely isolated — it's usually followed by more turbulent days. This is measured by the **autocorrelation of absolute returns** (Vol ACF):

- ACF = 0.0 → no pattern (each day independent, like a die)
- ACF = 0.27 → Gold's real pattern (storms in clusters)
- ACF = 0.29 → Ethereum's real pattern

`numpy.random.normal()` produces ACF = 0.0. Always. It assumes tomorrow has nothing to do with today. Everyone knows this is false in meteorology. In finance, it's the same thing — but we pretend otherwise.

---

## I Built a Fix: CRNG

**CRNG** (Contingency Random Number Generator) is a Python library that produces random numbers with real market statistical signatures. Three layers:

### Layer 1 — Deep Ocean Currents (Irrational Oscillators)

At the bottom of the ocean, there are currents that never repeat. Each oscillates at an irrational frequency — one at the rhythm of pi, another at sqrt(2), another at *e*. Because they're incommensurable, they **never synchronize**. This is where the base randomness comes from.

### Layer 2 — Resonance (Coupling)

When two currents pass near each other with similar rhythms, they amplify each other. Like pushing a child on a swing — if you push at the right rhythm, they go higher and higher. This creates the **storm clusters** (volatility clustering).

### Layer 3 — The Cascade (Amplifier)

When a big wave appears, it destabilizes nearby waves and creates more big waves. A domino effect. A stone falls in the river → creates waves → waves hit the bank → come back bigger. This creates the **tsunamis** (fat tails, K > 3).

**The key discovery:** there's a **critical threshold**. Below it, cascades die on their own (lake, K=3). Above it, cascades self-amplify (ocean, K=9, 23, 200). It's like the difference between water and ice — not "a bit colder," but a **phase transition**.

---

## The Results

Tested against 5 years of real data, 7 assets, 7 metrics, 10 random seeds:

| | CRNG | NumPy |
|:--|:--:|:--:|
| Metrics matched | **42/49 (86%)** | 7/49 (14%) |

| Asset | Real K | CRNG K | NumPy K |
|:--|:--:|:--:|:--:|
| Ethereum | 8.2 | 8.5 | 3.0 |
| Bitcoin | 6.9 | 7.3 | 3.0 |
| Gold | 15.6 | 11.2 | 3.0 |
| S&P 500 | 9.6 | 8.4 | 3.0 |

---

## Regime Detector: The Thermometer

I also built a real-time **regime detector** that calibrates CRNG on sliding windows of market data. Four regimes:

- **CALM** (K < 5) — Lake. Gaussian territory.
- **NORMAL** (K 5-12) — Ocean. Moderate fat tails.
- **STRESSED** (K 12-30) — Stormy. Elevated tail risk.
- **CRISIS** (K > 30) — Tsunami. Cascades dominating.

Current market status (March 2026):

| Asset | Regime | K (60d) | K (252d) |
|:--|:--|:--:|:--:|
| S&P 500 | CALM | 2.8 | **26.0** |
| Gold | NORMAL | 6.3 | 9.7 |
| Bitcoin | NORMAL | 8.2 | 10.5 |
| Ethereum | CALM | 4.9 | 5.3 |

The S&P 500 looks calm in the last 60 days (K=2.8). But zoom out to 252 days and K=26 — **STRESSED**. The crash is invisible at short scales but screaming at the yearly scale. Fat tails dissipate as you zoom in — exactly as the phase transition model predicts.

---

## Try It Yourself

```python
pip install crng
```

```python
from crng import gold, from_data

# Use a preset (frozen configuration)
rng = gold(seed=42)
values = rng.generate(10000)  # log-returns, not prices

# Or auto-calibrate from real data (recommended)
rng = from_data(your_price_series, seed=42)
```
<!-- errata 2026-04-10: previous snippet used `ContingencyRNG(preset='gold')`,
     which is not a real constructor keyword. The API is `from crng import gold`
     or `ContingencyRNG(target_kurtosis=9.26, vol_clustering=0.3)` directly.
     Fixed per Codex review 2026-04 item P2.1. -->


**Interactive notebook:** [Open in Google Colab](https://colab.research.google.com/github/brotto/crng/blob/main/notebooks/crng_demo.ipynb)

**Source code:** [github.com/brotto/crng](https://github.com/brotto/crng)

---

## The Theory Behind It

CRNG emerged from a research project asking: *what is randomness, really?*

The Law of Large Numbers is a valid theorem. But it describes **potentiality** (what *can* happen), not **actuality** (what *does* happen). PRNGs are pure mathematical objects — they embody potentiality perfectly. That's why K always equals 3.

Reality is different. Reality is local, contingent, Heraclitean. A spinning coin is neither heads nor tails until an external event intersects it. The result emerges from the **contingent encounter** between two independent processes — not from either one alone.

This is what CRNG models: contingency as a generative mechanism. Not noise added to a signal. Not a distribution with parameters. A genuine encounter between incommensurable oscillators, producing the statistical signatures of reality.

---

*Built by Ale Brotto ([@AlexandreBrotto](https://x.com/AlexandreBrotto) / [brotto.io](https://brotto.io))*
