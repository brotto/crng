"""
Generate frozen artifacts for the 2026-04-11 retraction of the sealed
catastrophe prediction. Four computations:

  1. Base rate of qualifying catastrophic events in non-overlapping
     2-year windows over the historical record (supports finding 1
     of compliance-officer audit).
  2. Days between retraction date (2026-04-11) and self-declared
     falsification window close (2028-06-30) — finding 2.
  3. Null distribution of max-peak-to-mean FFT ratio for iid standard-normal
     sequences of N=78 (the gap sequence length) — finding 3.
  4. Clipping measurement for ContingencyRNG(seed=42, target_kurtosis=8.0,
     vol_clustering=0.35, n_oscillators=7) over 100,000 samples — finding 7.

All outputs are written to this directory as JSON files. Deterministic
where possible (fixed seeds). Run: python generate_artifacts.py
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

# Add parent dir to path so we can import crng
HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent
sys.path.insert(0, str(PKG_ROOT))

ARTIFACTS_DIR = HERE

# ============================================================
# EVENT LIST — copied verbatim from experiments/catastrophic_events.py
# to make this script self-contained and reproducible without loading
# the deprecated experiment file.
# ============================================================

EARTHQUAKES = [
    "1906-04-18", "1908-12-28", "1920-12-16", "1923-09-01", "1927-05-22",
    "1931-08-10", "1934-01-15", "1935-05-30", "1939-12-26", "1944-12-07",
    "1946-04-01", "1948-10-05", "1950-08-15", "1952-11-04", "1957-03-09",
    "1960-05-22", "1964-03-27", "1970-05-31", "1976-07-27", "1985-09-19",
    "1988-12-07", "1989-10-17", "1994-01-17", "1995-01-17", "1999-08-17",
    "1999-09-20", "2001-01-26", "2003-12-26", "2004-12-26", "2005-10-08",
    "2008-05-12", "2010-01-12", "2010-02-27", "2011-03-11", "2015-04-25",
    "2017-09-19", "2018-09-28", "2023-02-06", "2024-01-01",
]
CRASHES = [
    "1929-10-29", "1937-03-10", "1962-05-28", "1973-01-11", "1979-10-06",
    "1987-10-19", "1989-10-13", "1997-10-27", "1998-08-17", "2000-03-10",
    "2001-09-17", "2007-02-27", "2008-09-29", "2010-05-06", "2011-08-05",
    "2015-08-24", "2018-02-05", "2020-03-16", "2022-06-13",
]
DISASTERS = [
    "1900-09-08", "1918-01-01", "1931-08-01", "1935-09-02", "1938-09-21",
    "1942-10-16", "1953-02-01", "1959-09-26", "1965-05-11", "1970-11-12",
    "1975-08-05", "1984-12-03", "1986-04-26", "1991-04-29", "1998-10-29",
    "2003-08-01", "2004-12-26", "2005-08-29", "2008-05-02", "2010-01-12",
    "2011-03-11", "2013-11-08", "2019-12-01", "2022-06-01",
]


def iso_to_date(s):
    return date.fromisoformat(s)


# ============================================================
# 1. BASE RATE OF QUALIFYING EVENTS IN NON-OVERLAPPING 2-YEAR WINDOWS
# ============================================================

def base_rate_non_overlapping_windows():
    """
    For each non-overlapping 2-year window from 1900-01-01 forward,
    count whether at least one event (any category) occurred in it.
    Return the fraction of windows that contained at least one event.
    """
    all_events_iso = sorted(set(EARTHQUAKES + CRASHES + DISASTERS))
    all_events = [iso_to_date(s) for s in all_events_iso]

    start = date(1900, 1, 1)
    # Window count up to 2024-12-31 (last complete window before seal date)
    end = date(2024, 12, 31)

    windows = []
    cur = start
    while cur < end:
        nxt_year = cur.year + 2
        try:
            nxt = date(nxt_year, cur.month, cur.day)
        except ValueError:
            # Feb 29 in non-leap year — skip to Mar 1
            nxt = date(nxt_year, cur.month + 1, 1)
        count = sum(1 for e in all_events if cur <= e < nxt)
        windows.append({
            "start": cur.isoformat(),
            "end": nxt.isoformat(),
            "n_events": count,
            "has_event": count >= 1,
        })
        cur = nxt

    n_total = len(windows)
    n_with_event = sum(1 for w in windows if w["has_event"])
    base_rate = n_with_event / n_total

    return {
        "method": "non_overlapping_2year_windows",
        "source_events": {
            "earthquakes": len(EARTHQUAKES),
            "crashes": len(CRASHES),
            "disasters": len(DISASTERS),
            "raw_total": len(EARTHQUAKES) + len(CRASHES) + len(DISASTERS),
            "unique_dates": len(all_events_iso),
        },
        "window_length_days": "~730",
        "window_start_first": windows[0]["start"],
        "window_start_last": windows[-1]["start"],
        "n_windows": n_total,
        "n_windows_with_event": n_with_event,
        "n_windows_empty": n_total - n_with_event,
        "base_rate_fraction": f"{n_with_event}/{n_total}",
        "base_rate_decimal": round(base_rate, 6),
        "windows": windows,
    }


# ============================================================
# 2. DAY COUNT: 2026-04-11 → 2028-06-30
# ============================================================

def days_until_falsification_window_close():
    retraction_date = date(2026, 4, 11)
    window_close = date(2028, 6, 30)
    delta = window_close - retraction_date

    seal_date = date(2026, 3, 31)
    seal_to_close = window_close - seal_date
    seal_to_retraction = retraction_date - seal_date

    return {
        "retraction_date": retraction_date.isoformat(),
        "seal_date": seal_date.isoformat(),
        "falsification_window_close": window_close.isoformat(),
        "days_seal_to_retraction": seal_to_retraction.days,
        "days_retraction_to_window_close": delta.days,
        "days_seal_to_window_close": seal_to_close.days,
    }


# ============================================================
# 3. FFT NULL DISTRIBUTION: max peak-to-mean ratio for iid standard-normal N=78
# ============================================================

def fft_null_distribution(n=78, n_trials=100_000, seed=1):
    """
    For iid standard-normal samples of length n, compute the FFT, take the
    magnitude of non-DC frequency bins, and record the maximum peak
    divided by the mean. This gives the null distribution against
    which the "5.46x" claim should be judged.
    """
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_trials)
    for i in range(n_trials):
        x = rng.standard_normal(n)
        # power spectrum, drop DC
        spec = np.abs(np.fft.rfft(x))[1:]
        mean_spec = spec.mean()
        max_spec = spec.max()
        ratios[i] = max_spec / mean_spec

    pct = np.percentile(ratios, [50, 75, 90, 95, 99, 99.9])

    # Fraction of trials at or above 5.46
    frac_at_or_above_546 = float(np.mean(ratios >= 5.46))
    # Fraction above 3.0 (the "significance threshold" the original claimed)
    frac_above_3 = float(np.mean(ratios >= 3.0))

    return {
        "n_samples_per_trial": n,
        "n_trials": n_trials,
        "seed": seed,
        "distribution_family": "iid_standard_normal",
        "statistic": "max(|FFT[1:]|) / mean(|FFT[1:]|)",
        "mean_ratio": float(ratios.mean()),
        "std_ratio": float(ratios.std()),
        "percentiles": {
            "50": float(pct[0]),
            "75": float(pct[1]),
            "90": float(pct[2]),
            "95": float(pct[3]),
            "99": float(pct[4]),
            "99.9": float(pct[5]),
        },
        "fraction_at_or_above_5_46": frac_at_or_above_546,
        "fraction_at_or_above_3_0": frac_above_3,
        "interpretation": (
            "The 5.46x peak-to-mean ratio reported in the original "
            "catastrophic_events.py experiment is located at approximately "
            "the {pct}-th percentile of the null distribution for iid "
            "standard normal samples of the same length. A threshold of "
            "3.0x is routinely exceeded by pure noise."
        ).format(pct=round(100.0 * float(np.mean(ratios < 5.46)), 2)),
    }


# ============================================================
# 4. CLIPPING MEASUREMENT FOR CRNG-Modulated Sampling
# ============================================================

def clipping_measurement():
    """
    Reproduce the clipping measurement from the retraction:
    - Instantiate ContingencyRNG with the same parameters used by
      Method 2 (CRNG-Modulated Sampling) of the sealed prediction.
    - Draw 100,000 samples via rng.next().
    - Count fractions below 0.001, above 0.999, and in between.
    """
    try:
        from crng import ContingencyRNG
    except ImportError as e:
        return {
            "error": f"Could not import crng package: {e}",
            "expected_result": "documented in REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md",
        }

    rng = ContingencyRNG(
        seed=42, n_oscillators=7,
        target_kurtosis=8.0, vol_clustering=0.35
    )
    n = 100_000
    samples = np.array([rng.next() for _ in range(n)])

    clipped_low = int(np.sum(samples < 0.001))
    clipped_high = int(np.sum(samples > 0.999))
    in_range = int(np.sum((samples >= 0.001) & (samples <= 0.999)))

    return {
        "parameters": {
            "seed": 42,
            "n_oscillators": 7,
            "target_kurtosis": 8.0,
            "vol_clustering": 0.35,
            "n_samples": n,
        },
        "sample_statistics": {
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "min": float(samples.min()),
            "max": float(samples.max()),
            "median": float(np.median(samples)),
        },
        "clipping_counts": {
            "below_0_001": clipped_low,
            "above_0_999": clipped_high,
            "in_range_0_001_to_0_999": in_range,
        },
        "clipping_fractions": {
            "below_0_001": round(clipped_low / n, 6),
            "above_0_999": round(clipped_high / n, 6),
            "in_range_0_001_to_0_999": round(in_range / n, 6),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Generating retraction artifacts...")
    print()

    print("1. Base rate of catastrophic events in non-overlapping 2-year windows")
    br = base_rate_non_overlapping_windows()
    print(f"   Non-overlapping 2-year windows (1900-01-01 to 2024-12-31): {br['n_windows']}")
    print(f"   Windows with >=1 event: {br['n_windows_with_event']}")
    print(f"   Windows empty: {br['n_windows_empty']}")
    print(f"   Base rate: {br['base_rate_fraction']} = {br['base_rate_decimal']}")
    with open(ARTIFACTS_DIR / "base_rate_2year_windows.json", "w") as f:
        json.dump(br, f, indent=2)
    print(f"   → base_rate_2year_windows.json")
    print()

    print("2. Day count retraction → falsification window close")
    dc = days_until_falsification_window_close()
    print(f"   Seal date: {dc['seal_date']}")
    print(f"   Retraction date: {dc['retraction_date']}")
    print(f"   Window close: {dc['falsification_window_close']}")
    print(f"   Seal → retraction: {dc['days_seal_to_retraction']} days")
    print(f"   Retraction → window close: {dc['days_retraction_to_window_close']} days")
    print(f"   Seal → window close: {dc['days_seal_to_window_close']} days")
    with open(ARTIFACTS_DIR / "retraction_day_counts.json", "w") as f:
        json.dump(dc, f, indent=2)
    print(f"   → retraction_day_counts.json")
    print()

    print("3. FFT null distribution (iid N=78, 100k trials)")
    fft = fft_null_distribution(n=78, n_trials=100_000, seed=1)
    print(f"   Mean max/mean ratio: {fft['mean_ratio']:.4f}")
    print(f"   50th percentile:  {fft['percentiles']['50']:.4f}")
    print(f"   95th percentile:  {fft['percentiles']['95']:.4f}")
    print(f"   99th percentile:  {fft['percentiles']['99']:.4f}")
    print(f"   Fraction >= 3.0:  {fft['fraction_at_or_above_3_0']:.4f}")
    print(f"   Fraction >= 5.46: {fft['fraction_at_or_above_5_46']:.4f}")
    with open(ARTIFACTS_DIR / "fft_null_N78.json", "w") as f:
        json.dump(fft, f, indent=2)
    print(f"   → fft_null_N78.json")
    print()

    print("4. Clipping measurement for CRNG-Modulated Sampling")
    clip = clipping_measurement()
    if "error" in clip:
        print(f"   SKIPPED: {clip['error']}")
    else:
        print(f"   Fraction below 0.001: {clip['clipping_fractions']['below_0_001']:.4f}")
        print(f"   Fraction above 0.999: {clip['clipping_fractions']['above_0_999']:.4f}")
        print(f"   Fraction in [0.001, 0.999]: {clip['clipping_fractions']['in_range_0_001_to_0_999']:.4f}")
    with open(ARTIFACTS_DIR / "clipping_measurement.json", "w") as f:
        json.dump(clip, f, indent=2)
    print(f"   → clipping_measurement.json")
    print()

    print("Done. Four artifacts written to:")
    print(f"   {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
