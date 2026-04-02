# CRNG update: Leave-one-out cross-validation shows catastrophe temporal geometry is category-independent

**Flair: Showcase**

## What My Project Does

CRNG (Contingency RNG) is a Python library that generates random numbers with real-world statistical signatures (fat tails, volatility clustering). In a new experiment, I used leave-one-category-out cross-validation on 82 catastrophic events (earthquakes, financial crashes, natural disasters) to test whether different catastrophe types share the same temporal gap distribution. Result: 3/3 MATCH — the normalized gap distributions are statistically indistinguishable.

Key findings:
- A model trained on earthquakes + crashes predicts natural disaster timing with KS p = 0.901
- All pairwise comparisons: MATCH (p = 0.314 to 0.986)
- Temporal stability: 1900-1970 predicts 1970-2025 (p = 0.777)
- 7/10 total tests confirm universality

## Target Audience

Researchers in computational statistics, risk modeling, complex systems, and anyone interested in the temporal structure of extreme events. Also useful for Monte Carlo practitioners who want fat-tailed alternatives to standard PRNGs.

## Comparison

Unlike standard PRNGs (NumPy, random) which produce K=3 distributions, CRNG produces configurable fat-tailed distributions that match real-world data. The novelty here is not the RNG itself but the empirical finding: catastrophe timing follows a universal geometry regardless of the physical mechanism.

## Installation

```bash
pip install crng
```

## Code

The full experiment is at `experiments/novel_event_predictor.py` in the repo.

GitHub: [github.com/brotto/crng](https://github.com/brotto/crng)
