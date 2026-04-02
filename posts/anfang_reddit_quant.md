# Leave-one-out cross-validation on 82 catastrophic events reveals category-independent temporal geometry

**tl;dr**: Trained temporal gap models on 2 of 3 catastrophe categories (earthquakes, financial crashes, natural disasters), tested on the held-out category. 3/3 KS test matches. The gap distributions are statistically indistinguishable across categories despite completely different underlying mechanisms.

## Background

Previous work showed that CRNG (a fat-tailed RNG with volatility clustering) matches the gap distribution of 82 catastrophic events with 20/20 KS test concordance. This raised the question: is the temporal structure category-specific or universal?

## Method

82 events (1900-2025): 39 earthquakes M≥6.7, 19 financial crashes, 24 natural disasters.

**Leave-one-category-out**: For each category, train on the other two (merge events, sort chronologically, extract gaps), normalize, KS test against held-out category.

**Pairwise**: Direct KS comparison between each pair of categories.

**Temporal**: Split all events at 1970. Train on 1900-1970, test on 1970-2025.

## Results

### Leave-One-Out

| Held Out | Trained On | KS p-value |
|:---|:---|:---|
| Earthquakes | Crashes + Disasters | 0.174 |
| Financial Crashes | Earthquakes + Disasters | 0.440 |
| Natural Disasters | Earthquakes + Crashes | 0.901 |

### Pairwise

| Pair | KS p-value |
|:---|:---|
| EQ vs FIN | 0.314 |
| EQ vs NAT | 0.986 |
| FIN vs NAT | 0.531 |

### Temporal stability

1900-1970 vs 1970-2025: p = 0.777. Shape stable across 125 years despite 2.96x acceleration in mean gap (844d → 399d).

### Scorecard

7/10 tests MATCH at α=0.05.

## Discussion

The normalized gap distribution is invariant across:
- Categories with no shared physical mechanism
- Different historical periods
- Different event densities

This suggests the temporal geometry is not a property of earthquakes, crashes, or disasters individually, but of extreme events as a class.

Key statistics by category:
- Earthquakes: CV=0.769, K=6.62
- Financial Crashes: CV=1.102, K=8.87
- Natural Disasters: CV=0.762, K=4.62
- All Combined: CV=0.938, K=10.02

The CV range (0.762-1.102) is remarkably narrow given the completely different generating mechanisms. All show fat-tailed gap distributions (K >> 3), consistent with clustered rather than Poisson timing.

## Negative result

CRNG gap distribution matched poorly against individual categories in this configuration (0/3), though it had previously matched 20/20 against the same data with different normalization. The empirical cross-category matches are stronger evidence than the CRNG comparison.

## Code

`pip install crng` | Full experiment: [github.com/brotto/crng](https://github.com/brotto/crng) → `experiments/novel_event_predictor.py`
