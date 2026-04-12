> ⚠️ **RETRACTION 2026-04-10 — the "42/49 (86%)" headline is withdrawn.**
> Unfrozen data + a kurtosis semantic bug (`np.diff(values)` in `stats()`).
> Corrected, frozen numbers live in `posts/benchmark_errata_2026-04-10.md`
> (snapshot SHA256 `82f8b5e5…ec9fff5`; CRNG is closer to the real
> fingerprint on 16 of 21 comparison cells: 7 assets × 3 metrics).
> The body text below is preserved for audit and reflects the retracted
> pre-2026-04-10 claims — do not cite it.

# r/algotrading Post

## Title:
I built a real-time market regime detector that classifies CALM/NORMAL/STRESSED/CRISIS using fat tail analysis

## Body:

I've been working on a regime detection tool based on kurtosis — how "fat" the tails of the return distribution are. The idea: instead of looking at moving averages or VIX, measure the actual statistical character of the market in real time.

**The regime classifications:**
- **CALM** (K < 5) — Gaussian behavior, no fat tails, smooth sailing
- **NORMAL** (K 5-12) — typical market, occasional large moves
- **STRESSED** (K 12-30) — elevated tail risk, large moves more frequent
- **CRISIS** (K > 30) — extreme regime, cascade dynamics

**Current market readings (March 2026, 60-day window):**

| Asset | Regime | K (60d) | K (252d) | Vol (ann) |
|:--|:--|:--:|:--:|:--:|
| SPY | CALM | 2.8 | 26.0 | 13% |
| GLD | NORMAL | 6.3 | 9.7 | 42% |
| BTC | NORMAL | 8.2 | 10.5 | 57% |
| ETH | CALM | 4.9 | 5.3 | 72% |
| Oil | NORMAL | 6.0 | 10.5 | 43% |
| AAPL | CALM | 4.7 | 5.2 | 34% |

**The interesting finding:** SPY looks perfectly calm at 60 days (K=2.8) but STRESSED at 252 days (K=26). The crash is invisible at short scales but screaming at the yearly scale. This multi-scale divergence is itself a signal — when short-term K is low but long-term K is high, the market has recently recovered from a stress event.

The detector uses CRNG (a custom random number generator I built, `pip install crng`) to calibrate against sliding windows. CRNG matches 86% of real market statistics — standard PRNGs match 14%.

**How to run it:**
```bash
pip install crng yfinance
python regime_detector.py --multi          # one-shot analysis
python regime_detector.py --live           # continuous monitoring
python regime_detector.py AAPL TSLA NVDA   # custom symbols
```

GitHub with all code: [github.com/brotto/crng](https://github.com/brotto/crng)

Not a trading signal — it's a regime classifier. But knowing whether you're in a K=3 or K=26 environment should probably affect your position sizing.

Would love feedback from anyone doing regime-based strategies.
