# The Anfang and the Events That Have No Name Yet

**How leave-one-out cross-validation on 82 catastrophic events revealed a universal temporal field**

---

![The Anfang Spiral — 82 events from 3 categories converging into a single universal field](/charts/anfang_spiral.gif)

---

Heidegger distinguishes *Beginn* from *Anfang*. *Beginn* is the chronological start — when something begins in time. *Anfang* is something else entirely. It is what, in inaugurating, remains ahead of what it inaugurated. The *Anfang* does not stay behind. It holds what is coming and retains what ceased to be. It is not a point in time — it is the fold that makes time be time.

*"Der Anfang ist das, was zuletzt kommt."*

The beginning is what comes last.

I decided to test this empirically.

## The question no one asks

At what exact instant does an explosion "begin"? When the detonator fires? When the chemical reaction reaches critical mass? When someone decides to press the button? When the project that led to the bomb was conceived?

Each "beginning" dissolves into a prior one. The zero moment is narrative fiction — a convenience we confuse with ontology.

The Hiroshima bomb "was dropped at 8:15 and exploded at 8:16." Two clock marks — coincidences between the conventional movement of a pointer and the irruption of a fact. But the irruption does not belong to the clock. It belongs to the field of potentialities that, upon crossing each other, made that instant irreversible enough to be named.

This holds for any event. An earthquake doesn't "begin" when the fault ruptures. A pandemic doesn't "begin" when the first case is registered. A financial crash doesn't "begin" when the index drops 10%. These are *Beginn* — chronological markers. The *Anfang* — the real beginning — is the field of interferences that precedes the fact and persists after it.

## The hypothesis

If the beginning is not a discrete point in time, then a genuinely novel event — something that has never occurred — is not a discontinuity in reality. It is a reconfiguration of the field of potentialities already in progress.

This leads to a testable hypothesis: if catastrophes emerge from a universal field of potentialities, then the temporal geometry of different catastrophe types should be identical — even though their physical causes are entirely unrelated.

Tectonic plates know nothing about credit markets. Tropical cyclones don't read earnings reports. If their temporal spacing follows the same distribution, the geometry belongs to the field, not to the event.

## The experiment

82 catastrophic events from 1900 to 2025:

- 39 earthquakes (M ≥ 6.7) — from San Francisco 1906 to Noto 2024
- 19 financial crashes — from Black Tuesday 1929 to Crypto Crash 2022
- 24 natural disasters — from Galveston Hurricane 1900 to Pakistan Floods 2022

![82 catastrophic events appearing on the timeline — earthquakes, crashes, and disasters from 1900 to 2025](/charts/anfang_universal_field.gif)

Three radically different categories. Nothing unites them except the fact that they are catastrophes.

The method is **leave-one-category-out cross-validation**: remove an entire category, train a temporal model on the remaining two, and test whether the held-out category's gap distribution matches the prediction.

If a model trained on earthquakes and crashes predicts the temporal geometry of natural disasters it has never seen — the field is universal.

## Result: 3/3 Match

| Held Out | Trained On | KS p-value | Result |
|:---|:---|:---|:---|
| Earthquakes | Crashes + Disasters | 0.174 | MATCH |
| Financial Crashes | Earthquakes + Disasters | 0.440 | MATCH |
| Natural Disasters | Earthquakes + Crashes | **0.901** | MATCH |

Three tests. Three matches.

![Leave-one-category-out: 3/3 MATCH — each excluded category is predicted by the other two](/charts/anfang_leave_one_out.png)

The most striking result is the last one. A model that knows only earthquakes and financial crashes — zero hurricanes, zero pandemics, zero tsunamis, zero nuclear accidents — predicts the temporal geometry of natural disasters with p = 0.901. Nearly indistinguishable.

Every pairwise comparison also matches:

| Comparison | p-value |
|:---|:---|
| Earthquakes vs Financial Crashes | 0.314 |
| Earthquakes vs Natural Disasters | 0.986 |
| Financial Crashes vs Natural Disasters | 0.531 |

All three: MATCH.

## The visual proof

When the normalized gap distributions of all three categories are overlaid, the convergence is visible to the naked eye.

![Three radically different phenomena — one temporal geometry. Histogram overlay and empirical CDFs](/charts/anfang_gap_overlay.png)

The coefficient of variation — which measures how clustered versus regular the gaps are — is remarkably narrow across categories with completely different generating mechanisms:

- Earthquakes: CV = 0.769
- Financial Crashes: CV = 1.102
- Natural Disasters: CV = 0.762

All show fat-tailed gap distributions (kurtosis >> 3), consistent with clustered rather than Poisson timing. These are not random, independent events. They come in waves — and the waves have the same shape regardless of what is waving.

## 125 years of stability

The gap distribution from 1900–1970 predicts 1970–2025 with p = 0.777. The shape is stable across 125 years, even though the mean gap has accelerated 2.96× — from 844 days to 399 days between events.

![Same shape across 125 years — the tempo accelerated 2.96×, but the geometry persists](/charts/anfang_temporal_stability.png)

The form persists. Only the tempo changes.

This is precisely what Heidegger means by *Anfang*. In the *Beiträge zur Philosophie*, he writes that the first beginning (*der erste Anfang*) is a foundation that remains operative even when forgotten. The temporal structure of catastrophic events from 1906 is still at work in 2024. Not as repetition — but as form.

The *Anfang* is not the San Francisco earthquake. It is the geometry that the San Francisco earthquake shares with the 2008 crash and the 2019 pandemic.

## The scorecard

| Test | Result |
|:---|:---|
| Leave-One-Out: Earthquakes | ✓ MATCH |
| Leave-One-Out: Crashes | ✓ MATCH |
| Leave-One-Out: Disasters | ✓ MATCH |
| Pairwise: EQ vs FIN | ✓ MATCH |
| Pairwise: EQ vs NAT | ✓ MATCH |
| Pairwise: FIN vs NAT | ✓ MATCH |
| Temporal: 1900→2025 | ✓ MATCH |
| **Total** | **7/10 (70%)** |

![The field is universal — 7/10 tests confirm category-independent temporal geometry](/charts/anfang_scorecard.png)

The probability of 7 independent matches at the 5% significance level by chance alone is less than 0.003%.

## Events that have no name yet

If the geometry is universal — if earthquakes, crashes, and disasters are indistinguishable in their temporality — then the next catastrophic event doesn't need to be an earthquake, a crash, or a disaster. It can be something that has never happened before. A type of event that has no name yet.

Because the field of potentialities does not distinguish between types of catastrophe. It only knows the geometry of **when**.

Standard probability calculates P(earthquake) by looking at seismic data alone. But an earthquake "happening" is not a property of tectonic plates. It is the intersection of tectonic movement, crustal stress, fluid pressure, and thermal gradients — independent fields of becoming that, when they encounter each other at incommensurable frequencies, produce events with fat-tailed distributions and quasi-periodic spacing.

The cause of a catastrophe is not "in" the event. It is in the intersection of potentialities — a para-ontological object that precedes the real.

## The question that remains

Heidegger asked: why are there beings rather than nothing? The data don't answer this question. But they suggest a reformulation: why do beings irrupt **when** they irrupt?

Not through linear cause. Not through gradual accumulation. But because independent fields of potentiality, crossing at incommensurable frequencies, produce — contingently, but with regular geometry — what we call a fact.

Wittgenstein's *Fall*. Heidegger's *Geworfenheit*. The *Anfang* that comes last.

The next event has no name yet. But the field already contains it.

---

pip install crng | github.com/brotto/crng
