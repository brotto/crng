# Technical errata — "CRNG reproduces 42 of 49 metrics (86%)" retracted (2026-04-10)

**Status:** Formal technical retraction of a public benchmark claim.
**Author:** Alexandre Brotto
**Date:** 2026-04-10
**Supersedes:** the "42/49 = 86%" headline that shipped with the v0.2.0
release notes, the initial LinkedIn and X posts, and the v0.2.0 `README.md`.
**Public version of this errata:** `posts/benchmark_errata_2026-04-10.md`
(same content, narrative tone, intended for non-technical audiences).
**Related:** `REVIEWS/codex_review_2026-04.md` (the external review that
surfaced the defects), `SPECS.md` principles P2, P3, P5, P6.

---

## What was claimed

The v0.2.0 release notes and the accompanying public posts stated:

> "CRNG reproduces 42 of 49 market metrics (86 %) across 7 real assets
> (Gold, BTC, ETH, S&P 500, EURUSD, USDJPY, Oil) over 5 years, while a
> standard Gaussian PRNG reproduces 0 of 49."

The "42 of 49" figure was produced by a script that:

1. Re-downloaded fresh daily closes from yfinance at every invocation.
2. Computed a per-asset score using `ContingencyRNG.stats()`.
3. Counted a "win" whenever the CRNG distance to the real fingerprint was
   smaller than the iid Gaussian distance on seven metrics per asset
   (7 × 7 = 49 cells).

## Why it was retracted

**Defect 1 — The input data was not frozen.**
The benchmark re-downloaded yfinance data at every run. A claim of the form
"CRNG wins 42/49" is only reproducible if the 49 cells are defined over a
fixed sample. Two consecutive runs of the original script could produce
different scores because: (a) yfinance returns adjusted-close series whose
recent values can be restated as dividend events are published; (b) the
exact window boundaries depend on the local clock; (c) market holidays and
missing data are handled silently. This violates SPECS.md principle P2
("every public numerical claim must cite a frozen artifact") and makes the
original "42/49" figure effectively non-reproducible even by the author.

**Defect 2 — Semantic bug in `ContingencyRNG.stats()`.**
The `stats()` method was computing kurtosis and volatility-clustering
metrics on `np.diff(values)` rather than on `values` directly. The project's
semantic contract (now formalized in SPECS.md principle P5) is that
`ContingencyRNG.generate()` already returns log-scale *returns*. Applying
`np.diff` to a returns series yields the *differences between consecutive
returns*, which is a mathematically distinct quantity with different scale
and higher kurtosis than the returns themselves. The "CRNG kurtosis"
numbers that entered the 42/49 count were therefore measuring the wrong
quantity. The fix is a one-line change in `crng/__init__.py`; the
consequence is that the baseline tables for any claim based on `stats()`
must be recomputed.

**Defect 3 — No a priori model selection rule.**
The original benchmark allowed implicit per-run choices between presets
(`gold()`, `btc()`, etc.) and auto-calibration (`from_data()`) without a
pre-registered rule. This violates SPECS.md principle P3 ("seleção de modelo
antes das métricas"). In the corrected pipeline, the rule is frozen: for
every asset, CRNG is built via `from_data(prices, seed=42)` on the full
window, with no preset picking and no post-hoc choice. The baseline is
`iid_gaussian(seed=42)`, which goes through `numpy.random.default_rng` with
no oscillator machinery (SPECS P6).

## The corrected figure

After freezing the input data, fixing the `stats()` bug, and applying a
single a priori model-selection rule, the corrected figure on the
2026-04 snapshot is:

**CRNG is closer to the real fingerprint on 16 of 21 comparison cells
(7 assets × 3 metrics: kurtosis, tail ≥ 3σ frequency, ACF of |returns| at
lag 1).**

The corresponding frozen evidence:

- `benchmarks/snapshot_2026-04/prices.csv` — the frozen input, SHA-256
  `82f8b5e5abe2f9d084769898b8d3b6ffefc5cfbd1c2757531df76d049ec9fff5`
- `benchmarks/snapshot_2026-04/prices.sha256` — hash record
- `benchmarks/snapshot_2026-04/metadata.json` — yfinance provenance
  (window 2021-04-10 → 2026-04-10, daily close, 7 assets)
- `benchmarks/snapshot_2026-04/frozen_benchmark_report.json` — the 21-cell
  comparison with per-asset target, achieved-CRNG, achieved-iid, and
  signed distance to target.

Breakdown:

| Metric | CRNG wins | iid wins |
|---|---:|---:|
| Kurtosis | 6 | 1 |
| Tail 3σ% | 7 | 0 |
| Vol ACF(1) | 3 | 4 |
| **Total** | **16** | **5** |

CRNG wins on kurtosis and on tail frequency for essentially all assets.
It loses on volatility clustering for four of seven assets — this is the
honest weakness of the current clustering mechanism and is documented in
`README.md` and in `REVIEWS/cross_asset_models_2026-04-10.md`.

## What this errata does not claim

1. It does not claim that CRNG is useless. The descriptive-generative
   thesis — that CRNG reproduces fat-tail and volatility-clustering
   fingerprints more faithfully than iid Gaussian — survives, and is
   documented with frozen evidence in the 16/21 figure above.
2. It does not claim that 16/21 should itself be interpreted as a predictive
   score. The benchmark measures distributional shape matching, not
   forecasting ability. CRNG remains a descriptive-generative tool under
   SPECS.md principle P1; any future claim of predictive content would
   require a separate train/test split and a walk-forward evaluation, which
   this benchmark does not provide.
3. It does not retract the per-asset model YAML files in `models/*_v1.yaml`.
   Those files are the target/achieved contract per SPECS P4 and remain
   the authoritative record of how each preset was calibrated and what
   fingerprint it actually produces on 100,000 samples over 10 seeds.

## How this errata connects to the catastrophe retraction

This errata is one of three companion corrections filed in the
2026-04-10 / 2026-04-11 cycle:

- **This file** — retracts the 86% benchmark headline.
- **`REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md`** plus
  **`predictions/2026-04-11_next_catastrophe_RETRACTION.md`** — retract
  the sealed catastrophe prediction of 2026-03-31.
- **`REVIEWS/errata/2026-04-10_tson_expected_value.md`** — retracts the
  TSON "first mesmitude at the 2nd instant as expectation" formulation.

The three retractions share a common thread: the confusion between what
CRNG descriptively reproduces (a statistical fingerprint) and what a
downstream inference chain pretends to predict (calendar events, the
expectation of a philosophical object). `SPECS.md` principle P1 was
formalized as a direct response to the pattern.

## Version trace

- **v0.2.0 (2026-03-27)** shipped with the retracted 42/49 = 86% headline.
- **v0.2.1 (2026-04-11)** ships with the corrected 16/21 headline, the
  frozen benchmark snapshot, this technical errata, and its public
  companion `posts/benchmark_errata_2026-04-10.md`.

---

**Signed:**
Alexandre Brotto — author of the original claim and author of this
retraction.

**Compliance review:** performed with Claude Sonnet 4.5 against
SPECS.md principles P1–P7 and against the external review
`REVIEWS/codex_review_2026-04.md`.
