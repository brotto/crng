# Changelog

All notable changes to the `crng` package are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1] — 2026-04-11

### Summary

This is a **documentation and retraction release**. No behavior of the
`crng` Python API has changed. Every fix in this release is about correcting
public claims, adding the frozen-evidence infrastructure that was missing
from v0.2.0, and formally retracting two prior public artifacts whose
inferences did not survive external review.

The headline "CRNG beats NumPy on 86% of metrics (42/49)" that was shipped
with the v0.2.0 release notes, the accompanying posts, and the `README.md`
was based on a rolling benchmark that re-selected presets after seeing the
data of the day. When the same pipeline is re-run under a frozen snapshot
with an a priori model-selection rule, the corrected figure is **16 of 21
comparison cells (7 assets × 3 metrics)** on the `snapshot_2026-04/`
benchmark. The descriptive-generative thesis of CRNG — that it reproduces
fat-tail and volatility-clustering fingerprints more faithfully than an iid
Gaussian baseline — survives. The quantitative headline does not.

A second, separate retraction in this release withdraws the 2026-03-31
sealed catastrophe prediction (SHA-256 committed in `predictions/SEALED_HASH.txt`).
The method used to produce that prediction is structurally broken in four
documented ways; details and frozen measurements are in
`predictions/2026-04-11_next_catastrophe_RETRACTION.md` and
`benchmarks/retraction_2026-04-11/`.

### Added

- `SPECS.md` — the seven non-negotiable principles that now govern every
  public claim in this repository: (P1) descriptive vs predictive separation,
  (P2) frozen evidence for every numerical claim, (P3) a priori model
  selection, (P4) target vs achieved reported side by side, (P5) single
  semantic per artifact, (P6) honest iid baseline, (P7) no hedging in
  numerical results. Any future edit to README, posts, articles, or model
  cards must pass the `compliance-officer` adversarial review against these
  principles.
- `benchmarks/snapshot_2026-04/` — the frozen benchmark that supports the
  corrected 16/21 headline. Contains `prices.csv` (input data, SHA-256
  `82f8b5e5abe2f9d084769898b8d3b6ffefc5cfbd1c2757531df76d049ec9fff5`,
  recorded in `prices.sha256`), `metadata.json` (provenance: yfinance
  window 2021-04-10 → 2026-04-10, daily close, 7 assets), and
  `frozen_benchmark_report.json` (deterministic output, 21 per-asset
  target-vs-achieved comparison cells).
- `benchmarks/retraction_2026-04-11/` — frozen artifacts supporting the
  catastrophe retraction: `base_rate_2year_windows.json` (48 of 63
  non-overlapping 2-year windows in 1900–2024 contained ≥1 qualifying
  event), `retraction_day_counts.json` (seal→retraction = 11 days;
  retraction→falsification window close = 811 days),
  `fft_null_N78.json` (10⁵ Monte Carlo trials of max-peak-to-mean FFT ratio
  for iid standard-normal sequences of length 78), and
  `clipping_measurement.json` (50.355% of CRNG samples clipped to 0.001
  in the `next_catastrophe.py` quantile-mapping loop). Reproducible via
  `PYTHONPATH=. python benchmarks/retraction_2026-04-11/generate_artifacts.py`.
- `REVIEWS/codex_review_2026-04.md` — the external review that initiated
  the correction cycle.
- `posts/benchmark_errata_2026-04-10.md` — public errata documenting why
  the 42/49 = 86% figure was invalidated (the non-frozen data problem and
  the `stats()` first-differences bug).
- `REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md` — technical
  errata documenting the quantile-mapping bug in `experiments/next_catastrophe.py`.
- `REVIEWS/errata/2026-04-10_tson_expected_value.md` — technical errata
  for the TSON expected-mesmitude formula.
- `predictions/2026-04-11_next_catastrophe_RETRACTION.md` — formal
  retraction of the 2026-03-31 sealed catastrophe prediction, with the
  four structural defects enumerated and cross-referenced to the frozen
  artifacts above. The retraction is additive — the original sealed file
  and its SHA-256 hash are preserved unchanged.
- `crng.iid_gaussian(seed=...)` — the honest iid Gaussian baseline, a
  thin wrapper over `numpy.random.default_rng(seed).standard_normal`.
  This is the baseline against which any future "CRNG vs PRNG" comparison
  in this repository must be run. (`crng.gaussian()` remains available as
  an internal reference but is explicitly documented as *not* a baseline,
  because it still routes through the oscillator machinery.)
- `models/` — formal model cards in YAML with `target` and `achieved`
  fingerprints side by side, in compliance with SPECS.md principle P4.
- `compliance-officer` sub-agent — the adversarial reviewer that must be
  invoked before any public-facing change to README, posts, articles, or
  numerical claims. The agent definition lives at the workspace level
  outside the PyPI package so it does not ship with the wheel; its role
  is described in SPECS.md.

### Changed

- `README.md` — the "CRNG beats NumPy on 86% of metrics (42/49)" headline
  and all derived marketing language are removed. Replaced with the
  descriptive-only statement of what CRNG does (reproduces fat-tail and
  volatility-clustering fingerprints) and a pointer to the frozen benchmark
  snapshot with the corrected 16/21 figure. A retraction note explicitly
  marks the 86% figure as withdrawn and points at
  `REVIEWS/errata/2026-04-10_benchmark_86pct_retraction.md`.
- Public documentation throughout the repository has been tightened to
  use *descriptive* language wherever CRNG is described, and to avoid
  any claim of predictive capability. Any statement that a future market
  or event will behave in a particular way has been removed or rewritten.
- `crng/__init__.py` — the `gaussian()` docstring now explicitly states
  that it is a reference function, not an iid baseline, and directs users
  to `iid_gaussian()` for honest PRNG comparisons.
- `ContingencyRNG.generate(n)` — the docstring formalizes that the return
  value is a sequence of **log-scale returns**, not prices. This was the
  de facto semantic already, but it is now the documented contract in
  compliance with SPECS.md principle P5.
- `ContingencyRNG.stats(n)` — docstring clarifies that statistics are
  computed on the direct output of `generate()`, never on first
  differences of that output (which would apply `diff` twice).

### Deprecated

- `experiments/next_catastrophe.py` — renamed to
  `experiments/_DEPRECATED_next_catastrophe.py`. Raises `RuntimeError`
  at module import time with a pointer to the retraction document.
  Preserved for audit.
- `experiments/catastrophic_events.py` — the precursor experiment whose
  20/20 KS match and 5.46× FFT peak claims were used in the sealed
  prediction. The file is preserved for audit; a header note now
  points at `predictions/2026-04-11_next_catastrophe_RETRACTION.md`
  for the reasons why its inferences do not survive.

### Retracted (public communications, tracked separately from code)

- The v0.2.0 release notes' "CRNG beats NumPy on 86% of metrics (42/49)"
  headline. The correct frozen figure is 16 of 21 comparison cells
  (7 assets × 3 metrics) on the `snapshot_2026-04/` benchmark.
- The sealed prediction at `predictions/2026-03-31_next_catastrophe.md`,
  formally retracted in `predictions/2026-04-11_next_catastrophe_RETRACTION.md`.
  The original sealed file is preserved unchanged; its SHA-256 hash
  still validates.
- The TSON "expected first mesmitude at the 2nd instant" formulation
  where "expected" was read as "expectation." The corrected reading
  (expected value ≈ 2.421 from Monte Carlo; the 2nd instant is the
  *mode* and the *probability-of-appearance* figure, not the
  expectation) is documented in
  `REVIEWS/errata/2026-04-10_tson_expected_value.md`.

### Fixed

- `experiments/next_catastrophe.py` quantile-mapping bug: the original
  code treated `ContingencyRNG.next()` output as a uniform in `[0, 1]`
  and passed it into `np.percentile(gaps, q * 100)`. In fact, `next()`
  returns a log-scale return centered at zero, and 50.355% of samples
  with the original parameters are clipped to the floor value of 0.001.
  The resulting "CRNG-modulated distribution" was essentially a delta
  at `min_gap` with a small secondary spike at `max_gap`. The bug is
  documented in `REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md`
  and the measurement is frozen in
  `benchmarks/retraction_2026-04-11/clipping_measurement.json`.
  The file is not "fixed" in the sense of being made to work — it is
  retracted, because the intended use of `next()` as a quantile is
  categorically incorrect.

### Security

Nothing in this release is a security fix.

---

## [0.2.0] — 2026-03-27

- Initial public PyPI release of the `crng` package.
- Published with the "CRNG beats NumPy on 86% of metrics (42/49)"
  headline (now retracted; see v0.2.1 above).
- Contains `ContingencyRNG`, per-asset presets (`gold`, `eth`, `btc`,
  `eurusd`), `from_data` calibration routine, and the `gaussian` /
  `iid_gaussian` wrappers.

---

## Notes on versioning policy going forward

Starting with v0.2.1, every release in this repository must ship with:

1. A `CHANGELOG.md` entry that follows the Keep a Changelog sections
   (Added / Changed / Deprecated / Removed / Fixed / Security), plus a
   **Retracted** section for any public claim that is being formally
   withdrawn.
2. A frozen benchmark snapshot in `benchmarks/` with SHA-256 of the
   input data, so any numerical claim in the release notes is
   reproducible offline.
3. A `compliance-officer` adversarial review of the release notes and
   the README against `SPECS.md` before the tag is pushed to GitHub or
   the wheel is uploaded to PyPI.

This policy is a direct consequence of the correction cycle that
produced v0.2.1.
