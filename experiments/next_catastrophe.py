"""
NEXT CATASTROPHE PREDICTOR (EXPERIMENTAL / PRIVATE)
=====================================================

Based on the catastrophic events experiment findings:
- 82 events from 1900-2025
- Gap distributions match CRNG with 20/20 KS tests
- Quasi-periodic structure detected (5.46x power ratio)

Method:
1. Analyze historical gap distribution (empirical)
2. Fit the dominant period from FFT
3. Monte Carlo forward simulation using CRNG gap sampling
4. Build probability density for "days until next event"
5. Report most probable window

NOT for publication. Experimental only.

Ale Brotto — 2026-03-31
"""

import numpy as np
from scipy import stats as sp_stats
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crng import ContingencyRNG


# ============================================================
# ALL 82 CATASTROPHIC EVENTS (from catastrophic_events.py)
# ============================================================

def get_all_events():
    """All 82 catastrophic events, sorted by date."""
    events = [
        # Earthquakes
        "1906-04-18", "1908-12-28", "1920-12-16", "1923-09-01", "1927-05-22",
        "1931-08-10", "1934-01-15", "1935-05-30", "1939-12-26", "1944-12-07",
        "1946-04-01", "1948-10-05", "1950-08-15", "1952-11-04", "1957-03-09",
        "1960-05-22", "1964-03-27", "1970-05-31", "1976-07-27", "1985-09-19",
        "1988-12-07", "1989-10-17", "1994-01-17", "1995-01-17", "1999-08-17",
        "1999-09-20", "2001-01-26", "2003-12-26", "2004-12-26", "2005-10-08",
        "2008-05-12", "2010-01-12", "2010-02-27", "2011-03-11", "2015-04-25",
        "2017-09-19", "2018-09-28", "2023-02-06", "2024-01-01",
        # Financial crashes
        "1929-10-29", "1937-03-10", "1962-05-28", "1973-01-11", "1979-10-06",
        "1987-10-19", "1989-10-13", "1997-10-27", "1998-08-17", "2000-03-10",
        "2001-09-17", "2007-02-27", "2008-09-29", "2010-05-06", "2011-08-05",
        "2015-08-24", "2018-02-05", "2020-03-16", "2022-06-13",
        # Natural disasters
        "1900-09-08", "1918-01-01", "1931-08-01", "1935-09-02", "1938-09-21",
        "1942-10-16", "1953-02-01", "1959-09-26", "1965-05-11", "1970-11-12",
        "1975-08-05", "1984-12-03", "1986-04-26", "1991-04-29", "1998-10-29",
        "2003-08-01", "2004-12-26", "2005-08-29", "2008-05-02", "2010-01-12",
        "2011-03-11", "2013-11-08", "2019-12-01", "2022-06-01",
    ]
    # Remove duplicates (some dates appear in both earthquake and disaster lists)
    unique = sorted(set(events))
    return [datetime.strptime(d, "%Y-%m-%d") for d in unique]


# ============================================================
# ANALYSIS
# ============================================================

def analyze_gaps(dates):
    """Full gap analysis with FFT periodicity detection."""
    gaps = np.array([(dates[i+1] - dates[i]).days for i in range(len(dates)-1)], dtype=float)
    gaps = gaps[gaps > 0]  # remove same-day events

    print(f"\n{'='*70}")
    print(f"  HISTORICAL GAP ANALYSIS ({len(dates)} events, {len(gaps)} gaps)")
    print(f"{'='*70}")

    print(f"\n  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Span: {(dates[-1] - dates[0]).days:,} days ({(dates[-1] - dates[0]).days/365.25:.1f} years)")

    print(f"\n  --- Gap Statistics ---")
    print(f"  Mean gap:    {np.mean(gaps):.1f} days ({np.mean(gaps)/30.44:.1f} months)")
    print(f"  Median gap:  {np.median(gaps):.1f} days ({np.median(gaps)/30.44:.1f} months)")
    print(f"  Std gap:     {np.std(gaps):.1f} days")
    print(f"  CV:          {np.std(gaps)/np.mean(gaps):.3f}")
    print(f"  Min gap:     {np.min(gaps):.0f} days")
    print(f"  Max gap:     {np.max(gaps):.0f} days")
    print(f"  Kurtosis:    {sp_stats.kurtosis(gaps, fisher=False):.2f}")

    # Percentiles
    print(f"\n  --- Gap Percentiles ---")
    for p in [10, 25, 50, 75, 90, 95]:
        val = np.percentile(gaps, p)
        print(f"  P{p:>2}: {val:>7.0f} days ({val/30.44:>5.1f} months)")

    # FFT for periodicity
    gaps_centered = gaps - np.mean(gaps)
    fft = np.fft.fft(gaps_centered)
    power = np.abs(fft[:len(fft)//2]) ** 2
    freqs = np.fft.fftfreq(len(gaps_centered))[:len(fft)//2]

    if len(power) > 1:
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        # Top 3 dominant frequencies
        top_indices = np.argsort(power_no_dc)[-3:][::-1]

        print(f"\n  --- FFT Dominant Periods ---")
        for i, idx in enumerate(top_indices):
            freq = freqs_no_dc[idx]
            period_gaps = 1.0 / freq if freq > 0 else float('inf')
            # Convert period in gaps to period in days
            period_days = period_gaps * np.mean(gaps)
            ratio = power_no_dc[idx] / np.mean(power_no_dc)
            print(f"  #{i+1}: Period = {period_gaps:.1f} gaps = {period_days:.0f} days "
                  f"({period_days/365.25:.1f} years) | Power ratio: {ratio:.2f}x")

    # Recent acceleration? Compare last 20 gaps vs first 20
    if len(gaps) > 40:
        early = gaps[:20]
        late = gaps[-20:]
        print(f"\n  --- Temporal Trend ---")
        print(f"  First 20 gaps mean: {np.mean(early):.1f} days")
        print(f"  Last 20 gaps mean:  {np.mean(late):.1f} days")
        print(f"  Ratio: {np.mean(early)/np.mean(late):.2f}x (>1 = accelerating)")

    return gaps


def monte_carlo_prediction(gaps, last_event_date, today, n_simulations=100000):
    """
    Monte Carlo forward prediction using CRNG-modulated gap sampling.

    Instead of sampling from a simple distribution, we:
    1. Use the empirical gap distribution as base
    2. Modulate with CRNG to preserve fat-tailed, clustered structure
    3. Account for the quasi-periodic component
    """

    print(f"\n{'='*70}")
    print(f"  MONTE CARLO PREDICTION")
    print(f"  Simulations: {n_simulations:,}")
    print(f"  Last event:  {last_event_date.strftime('%Y-%m-%d')}")
    print(f"  Today:       {today.strftime('%Y-%m-%d')}")
    print(f"  Days since:  {(today - last_event_date).days}")
    print(f"{'='*70}")

    days_since = (today - last_event_date).days

    # Method 1: Empirical bootstrap (sample from historical gaps)
    print(f"\n  --- Method 1: Empirical Bootstrap ---")
    boot_next = []
    for _ in range(n_simulations):
        sampled_gap = np.random.choice(gaps)
        boot_next.append(sampled_gap)
    boot_next = np.array(boot_next)

    # How many of these gaps have already been "survived"?
    # P(event at day X | survived to today)
    # = P(gap = X) / P(gap >= days_since)
    survived_mask = boot_next >= days_since
    p_survived = np.mean(survived_mask)
    conditional_gaps = boot_next[survived_mask] - days_since

    print(f"  P(gap >= {days_since} days): {p_survived:.4f} ({p_survived*100:.1f}%)")
    if len(conditional_gaps) > 0:
        print(f"  Conditional mean: {np.mean(conditional_gaps):.0f} days from now")
        print(f"  Conditional median: {np.median(conditional_gaps):.0f} days from now")
        for p in [25, 50, 75, 90]:
            val = np.percentile(conditional_gaps, p)
            future_date = today + timedelta(days=int(val))
            print(f"    P{p}: {val:.0f} days → {future_date.strftime('%Y-%m-%d')} ({future_date.strftime('%b %Y')})")

    # Method 2: CRNG-modulated sampling
    print(f"\n  --- Method 2: CRNG-Modulated Sampling ---")
    rng = ContingencyRNG(seed=42, target_kurtosis=8.0, vol_clustering=0.35, n_oscillators=7)

    crng_next = []
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    for _ in range(n_simulations):
        # CRNG modulates the gap: base + crng_noise * scale
        crng_val = rng.next()
        # Map CRNG output to a gap using the empirical distribution shape
        # Use CRNG to pick a quantile, then map to empirical distribution
        # This preserves clustering and fat tails
        quantile = min(max(crng_val, 0.001), 0.999)
        gap = np.percentile(gaps, quantile * 100)
        crng_next.append(gap)

    crng_next = np.array(crng_next)
    survived_crng = crng_next[crng_next >= days_since] - days_since
    p_survived_crng = np.mean(crng_next >= days_since)

    print(f"  P(gap >= {days_since} days): {p_survived_crng:.4f} ({p_survived_crng*100:.1f}%)")
    if len(survived_crng) > 0:
        print(f"  Conditional mean: {np.mean(survived_crng):.0f} days from now")
        print(f"  Conditional median: {np.median(survived_crng):.0f} days from now")
        for p in [25, 50, 75, 90]:
            val = np.percentile(survived_crng, p)
            future_date = today + timedelta(days=int(val))
            print(f"    P{p}: {val:.0f} days → {future_date.strftime('%Y-%m-%d')} ({future_date.strftime('%b %Y')})")

    # Method 3: Quasi-periodic model
    print(f"\n  --- Method 3: Quasi-Periodic Model ---")

    # Fit the dominant period from FFT
    gaps_centered = gaps - np.mean(gaps)
    fft = np.fft.fft(gaps_centered)
    power = np.abs(fft[:len(fft)//2]) ** 2
    freqs = np.fft.fftfreq(len(gaps_centered))[:len(fft)//2]

    if len(power) > 1:
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]
        dominant_idx = np.argmax(power_no_dc)
        dominant_freq = freqs_no_dc[dominant_idx]
        period_gaps = 1.0 / dominant_freq if dominant_freq > 0 else len(gaps)
        period_days = period_gaps * np.mean(gaps)
        amplitude = np.sqrt(2 * power_no_dc[dominant_idx] / len(gaps))
        phase = np.angle(fft[dominant_idx + 1])

        print(f"  Dominant period: {period_days:.0f} days ({period_days/365.25:.1f} years)")
        print(f"  Amplitude: {amplitude:.1f} days")
        print(f"  Phase: {phase:.4f} rad")

        # Current position in cycle
        # The last gap index is len(gaps)-1
        # Next gap index would be len(gaps)
        next_gap_idx = len(gaps)

        # Predicted modulation for next gap
        modulation = amplitude * np.cos(2 * np.pi * dominant_freq * next_gap_idx + phase)
        predicted_gap = np.mean(gaps) + modulation

        print(f"\n  Mean gap: {np.mean(gaps):.0f} days")
        print(f"  Periodic modulation: {modulation:+.0f} days")
        print(f"  Predicted next gap: {predicted_gap:.0f} days")

        predicted_date = last_event_date + timedelta(days=int(predicted_gap))
        print(f"  Predicted date: {predicted_date.strftime('%Y-%m-%d')} ({predicted_date.strftime('%b %Y')})")

        days_from_now = (predicted_date - today).days
        if days_from_now > 0:
            print(f"  Days from now: {days_from_now}")
        else:
            print(f"  Days AGO: {abs(days_from_now)} (we're past the periodic peak)")
            # Next cycle
            next_cycle_gap_idx = next_gap_idx + period_gaps
            modulation2 = amplitude * np.cos(2 * np.pi * dominant_freq * next_cycle_gap_idx + phase)
            next_predicted = np.mean(gaps) + modulation2
            next_date = last_event_date + timedelta(days=int(next_predicted + predicted_gap))
            print(f"  Next cycle peak: ~{next_date.strftime('%b %Y')}")

    # Method 4: Hazard rate analysis (survival function)
    print(f"\n  --- Method 4: Hazard Rate (Survival Analysis) ---")

    # Sort gaps for survival function
    sorted_gaps = np.sort(gaps)
    n = len(sorted_gaps)

    # Kaplan-Meier-like survival
    survival = np.array([(sorted_gaps >= t).sum() / n for t in range(0, int(np.max(gaps)) + 100, 30)])
    time_points = np.arange(0, int(np.max(gaps)) + 100, 30)

    # Hazard at current days_since
    s_now = (gaps >= days_since).sum() / n
    s_next_month = (gaps >= (days_since + 30)).sum() / n

    # Conditional hazard = P(event in next 30 days | survived to now)
    if s_now > 0:
        h_30 = 1 - s_next_month / s_now
        print(f"  S({days_since}): {s_now:.4f} ({s_now*100:.1f}% of gaps are this long or longer)")
        print(f"  Conditional P(event in next 30 days): {h_30:.4f} ({h_30*100:.1f}%)")
        print(f"  Conditional P(event in next 90 days): {1 - (gaps >= (days_since + 90)).sum() / n / s_now:.4f}")
        print(f"  Conditional P(event in next 180 days): {1 - (gaps >= (days_since + 180)).sum() / n / s_now:.4f}")
        print(f"  Conditional P(event in next 365 days): {1 - (gaps >= (days_since + 365)).sum() / n / s_now:.4f}")
    else:
        print(f"  We're BEYOND all historical gaps! ({days_since} days > max gap {np.max(gaps):.0f} days)")
        print(f"  By historical standards, an event is OVERDUE.")

    # ============================================================
    # SYNTHESIS
    # ============================================================

    print(f"\n{'='*70}")
    print(f"  SYNTHESIS: WHEN IS THE NEXT CATASTROPHIC EVENT?")
    print(f"{'='*70}")

    print(f"\n  Reference: today is {today.strftime('%Y-%m-%d')}")
    print(f"  Last catastrophic event: {last_event_date.strftime('%Y-%m-%d')} ({days_since} days ago)")

    # Collect all predictions
    predictions = {}

    if len(conditional_gaps) > 0:
        med_boot = np.median(conditional_gaps)
        predictions['Empirical Bootstrap'] = today + timedelta(days=int(med_boot))

    if len(survived_crng) > 0:
        med_crng = np.median(survived_crng)
        predictions['CRNG-Modulated'] = today + timedelta(days=int(med_crng))

    if 'predicted_date' in dir():
        predictions['Quasi-Periodic'] = predicted_date

    print(f"\n  --- Predictions (median/peak) ---")
    for method, date in predictions.items():
        delta = (date - today).days
        status = f"in {delta} days" if delta > 0 else f"{abs(delta)} days AGO"
        print(f"  {method:25s}: {date.strftime('%Y-%m-%d')} ({date.strftime('%b %Y')}) — {status}")

    # Confidence window (intersection of methods)
    if len(conditional_gaps) > 0:
        p25 = today + timedelta(days=int(np.percentile(conditional_gaps, 25)))
        p75 = today + timedelta(days=int(np.percentile(conditional_gaps, 75)))
        print(f"\n  50% confidence window (empirical):")
        print(f"    {p25.strftime('%b %Y')} — {p75.strftime('%b %Y')}")

    if len(survived_crng) > 0:
        p25c = today + timedelta(days=int(np.percentile(survived_crng, 25)))
        p75c = today + timedelta(days=int(np.percentile(survived_crng, 75)))
        print(f"\n  50% confidence window (CRNG):")
        print(f"    {p25c.strftime('%b %Y')} — {p75c.strftime('%b %Y')}")

    # By-category analysis
    print(f"\n  --- Per-Category Last Events ---")
    eq_last = datetime(2024, 1, 1)
    fin_last = datetime(2022, 6, 13)
    nat_last = datetime(2022, 6, 1)

    for cat, last, mean_g in [
        ("Earthquake M≥6.7", eq_last, 906),   # ~2.5 years from data
        ("Financial Crash", fin_last, 1786),    # ~4.9 years
        ("Natural Disaster", nat_last, 1886),   # ~5.2 years
    ]:
        days_ago = (today - last).days
        predicted_next = last + timedelta(days=mean_g)
        delta = (predicted_next - today).days
        status = f"in {delta} days ({predicted_next.strftime('%b %Y')})" if delta > 0 else f"OVERDUE by {abs(delta)} days"
        print(f"  {cat:25s}: last {last.strftime('%Y-%m-%d')} ({days_ago}d ago), mean gap {mean_g}d → {status}")

    return conditional_gaps, survived_crng


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  NEXT CATASTROPHE PREDICTOR")
    print("  Based on 82 events × 125 years × CRNG gap matching")
    print("  EXPERIMENTAL — NOT FOR PUBLICATION")
    print("=" * 70)

    dates = get_all_events()
    print(f"\n  Loaded {len(dates)} unique catastrophic events")
    print(f"  First: {dates[0].strftime('%Y-%m-%d')}")
    print(f"  Last:  {dates[-1].strftime('%Y-%m-%d')}")

    # Analyze historical gaps
    gaps = analyze_gaps(dates)

    # Predict
    last_event = dates[-1]
    today = datetime(2026, 3, 31)

    boot_gaps, crng_gaps = monte_carlo_prediction(gaps, last_event, today)

    print(f"\n{'='*70}")
    print(f"  DISCLAIMER: This is a philosophical experiment.")
    print(f"  Catastrophes are contingent — they emerge from the")
    print(f"  intersection of independent potentialities.")
    print(f"  No model predicts the specific event, only the")
    print(f"  temporal geometry within which events tend to cluster.")
    print(f"{'='*70}")
