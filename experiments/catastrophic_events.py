"""
CATASTROPHIC EVENTS EXPERIMENT
================================

Hypothesis: All "interferences" between potentialities occur within
an imperceptible regularity. If so, the temporal pattern of extreme
events in CRNG should match the temporal pattern of real catastrophes.

Method:
1. Generate long CRNG series (the "world running")
2. Sliding window kurtosis → detect local K spikes
3. Map frequency and spacing of K≥5, K≥6, ... K≥30
4. Fetch real catastrophic events (earthquakes, financial crashes)
5. Compare temporal patterns (gap distributions, clustering, periodicity)

Ale Brotto — 2026-03-29
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crng import ContingencyRNG
from scipy import stats as sp_stats
import json
from datetime import datetime, timedelta

# ============================================================
# PART 1: GENERATE CRNG CATASTROPHIC EVENT MAP
# ============================================================

def generate_catastrophic_map(n_points=500000, window=100, seed=42):
    """
    Generate a long CRNG series and detect extreme K windows.

    Like scanning the ocean with a kurtosis sensor:
    - Slide a window of 100 points across 500,000 points
    - At each position, calculate local kurtosis
    - Record where K exceeds catastrophic thresholds
    """

    print("=" * 70)
    print("  GENERATING CATASTROPHIC EVENT MAP")
    print(f"  Points: {n_points:,} | Window: {window}")
    print("=" * 70)

    # Use standard CRNG with high kurtosis target
    rng = ContingencyRNG(
        seed=seed, n_oscillators=7,
        target_kurtosis=15.0, vol_clustering=0.35
    )

    print("  Generating series...")
    series = np.array([rng.next() for _ in range(n_points)])

    # Calculate returns (relative changes)
    returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
    returns = returns[np.isfinite(returns)]

    print(f"  Series generated: {len(returns):,} returns")

    # Sliding window kurtosis (vectorized with stride tricks)
    print("  Calculating sliding kurtosis...")
    n = len(returns)

    # Use strided approach for speed
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(returns, window)

    # Calculate kurtosis for each window using vectorized ops
    # K = n * sum((x-mean)^4) / (sum((x-mean)^2))^2
    means = np.mean(windows, axis=1, keepdims=True)
    centered = windows - means
    m2 = np.mean(centered**2, axis=1)
    m4 = np.mean(centered**4, axis=1)
    # Avoid division by zero
    m2_safe = np.where(m2 > 1e-20, m2, 1e-20)
    kurtosis_map = m4 / (m2_safe ** 2)

    print(f"  Kurtosis map: {len(kurtosis_map):,} windows")

    # Detect catastrophic events at various thresholds
    thresholds = [5, 6, 8, 10, 15, 20, 30, 50, 100]
    event_map = {}

    print(f"\n--- CATASTROPHIC EVENT CENSUS ---")
    print(f"  {'Threshold':>10} {'Count':>8} {'Frequency':>12} {'Mean Gap':>10} {'Gap CV':>8}")
    print(f"  {'-'*58}")

    for thresh in thresholds:
        # Find windows where K exceeds threshold
        indices = np.where(kurtosis_map >= thresh)[0]

        if len(indices) == 0:
            print(f"  K≥{thresh:>4}       {0:>8} {'—':>12} {'—':>10} {'—':>8}")
            continue

        # Merge nearby events (within 1 window of each other)
        # An "event" is a contiguous region of elevated K
        events = []
        current_start = indices[0]
        current_max_k = kurtosis_map[indices[0]]
        current_peak_idx = indices[0]

        for idx in indices[1:]:
            if idx - indices[list(indices).index(idx) - 1] <= window // 2:
                # Part of same event
                if kurtosis_map[idx] > current_max_k:
                    current_max_k = kurtosis_map[idx]
                    current_peak_idx = idx
            else:
                # New event
                events.append({
                    'start': current_start,
                    'peak': current_peak_idx,
                    'max_k': current_max_k,
                })
                current_start = idx
                current_max_k = kurtosis_map[idx]
                current_peak_idx = idx

        # Don't forget last event
        events.append({
            'start': current_start,
            'peak': current_peak_idx,
            'max_k': current_max_k,
        })

        # Calculate gaps between events
        if len(events) >= 2:
            peaks = [e['peak'] for e in events]
            gaps = np.diff(peaks)
            mean_gap = np.mean(gaps)
            gap_cv = np.std(gaps) / mean_gap if mean_gap > 0 else 0
            freq = len(events) / n_points
        else:
            gaps = []
            mean_gap = 0
            gap_cv = 0
            freq = len(events) / n_points

        event_map[thresh] = {
            'events': events,
            'count': len(events),
            'frequency': freq,
            'gaps': gaps.tolist() if len(gaps) > 0 else [],
            'mean_gap': mean_gap,
            'gap_cv': gap_cv,
        }

        print(f"  K≥{thresh:>4}       {len(events):>8} {freq:>12.6f} {mean_gap:>10.1f} {gap_cv:>8.3f}")

    # Overall kurtosis statistics
    print(f"\n--- KURTOSIS MAP STATISTICS ---")
    print(f"  Mean K:    {np.mean(kurtosis_map):.2f}")
    print(f"  Median K:  {np.median(kurtosis_map):.2f}")
    print(f"  Max K:     {np.max(kurtosis_map):.2f}")
    print(f"  Std K:     {np.std(kurtosis_map):.2f}")
    print(f"  K of K:    {sp_stats.kurtosis(kurtosis_map, fisher=False):.2f}")

    return kurtosis_map, event_map, returns


# ============================================================
# PART 2: FETCH REAL CATASTROPHIC EVENTS
# ============================================================

def get_earthquake_data():
    """
    Major earthquakes (M >= 7.0) from USGS, 1900-2025.
    Using well-documented historical data.
    """

    # Major earthquakes M >= 7.5 (well-documented, from USGS catalog)
    # Format: (date, magnitude, location)
    earthquakes = [
        # 1900s
        ("1906-04-18", 7.9, "San Francisco"),
        ("1908-12-28", 7.2, "Messina, Italy"),
        ("1920-12-16", 8.5, "Haiyuan, China"),
        ("1923-09-01", 7.9, "Kanto, Japan"),
        ("1927-05-22", 7.6, "Tsinghai, China"),
        ("1931-08-10", 8.0, "Fuyun, China"),
        ("1934-01-15", 8.1, "Bihar-Nepal"),
        ("1935-05-30", 7.7, "Quetta, Pakistan"),
        ("1939-12-26", 7.8, "Erzincan, Turkey"),
        ("1944-12-07", 8.1, "Tonankai, Japan"),
        ("1946-04-01", 8.1, "Aleutian Islands"),
        ("1948-10-05", 7.3, "Ashgabat, Turkmenistan"),
        ("1950-08-15", 8.6, "Assam, India"),
        ("1952-11-04", 9.0, "Kamchatka, Russia"),
        ("1957-03-09", 9.1, "Andreanof Islands"),
        ("1960-05-22", 9.5, "Valdivia, Chile"),
        ("1964-03-27", 9.2, "Alaska"),
        ("1970-05-31", 7.9, "Ancash, Peru"),
        ("1976-07-27", 7.5, "Tangshan, China"),
        ("1985-09-19", 8.0, "Mexico City"),
        ("1988-12-07", 6.8, "Spitak, Armenia"),
        ("1989-10-17", 6.9, "Loma Prieta, CA"),
        ("1994-01-17", 6.7, "Northridge, CA"),
        ("1995-01-17", 6.9, "Kobe, Japan"),
        ("1999-08-17", 7.6, "Izmit, Turkey"),
        ("1999-09-20", 7.7, "Chi-Chi, Taiwan"),
        ("2001-01-26", 7.7, "Gujarat, India"),
        ("2003-12-26", 6.6, "Bam, Iran"),
        ("2004-12-26", 9.1, "Sumatra (tsunami)"),
        ("2005-10-08", 7.6, "Kashmir"),
        ("2008-05-12", 7.9, "Sichuan, China"),
        ("2010-01-12", 7.0, "Haiti"),
        ("2010-02-27", 8.8, "Maule, Chile"),
        ("2011-03-11", 9.1, "Tohoku, Japan (tsunami)"),
        ("2015-04-25", 7.8, "Nepal"),
        ("2017-09-19", 7.1, "Puebla, Mexico"),
        ("2018-09-28", 7.5, "Sulawesi, Indonesia"),
        ("2023-02-06", 7.8, "Turkey-Syria"),
        ("2024-01-01", 7.6, "Noto, Japan"),
    ]

    return earthquakes


def get_financial_crashes():
    """Major financial crashes/crises."""

    crashes = [
        ("1929-10-29", 100, "Black Tuesday"),
        ("1937-03-10", 50, "1937 Recession"),
        ("1962-05-28", 30, "Kennedy Slide"),
        ("1973-01-11", 60, "Oil Crisis Crash"),
        ("1979-10-06", 40, "Volcker Shock"),
        ("1987-10-19", 90, "Black Monday"),
        ("1989-10-13", 35, "Friday the 13th mini-crash"),
        ("1997-10-27", 55, "Asian Financial Crisis"),
        ("1998-08-17", 50, "Russian/LTCM Crisis"),
        ("2000-03-10", 70, "Dot-com Bubble"),
        ("2001-09-17", 45, "Post-9/11 Crash"),
        ("2007-02-27", 40, "Shanghai Surprise"),
        ("2008-09-29", 95, "Lehman Brothers/GFC"),
        ("2010-05-06", 60, "Flash Crash"),
        ("2011-08-05", 45, "US Downgrade Crash"),
        ("2015-08-24", 50, "China Black Monday"),
        ("2018-02-05", 40, "Volmageddon"),
        ("2020-03-16", 85, "COVID Crash"),
        ("2022-06-13", 40, "Crypto/Rate Hike Crash"),
    ]

    return crashes


def get_natural_disasters():
    """Major natural disasters (non-earthquake) with significant casualties."""

    disasters = [
        ("1900-09-08", 80, "Galveston Hurricane"),
        ("1918-01-01", 100, "Spanish Flu (start)"),
        ("1931-08-01", 95, "China Floods"),
        ("1935-09-02", 40, "Labor Day Hurricane"),
        ("1938-09-21", 45, "New England Hurricane"),
        ("1942-10-16", 70, "Bengal Cyclone"),
        ("1953-02-01", 50, "North Sea Flood"),
        ("1959-09-26", 55, "Typhoon Vera"),
        ("1965-05-11", 50, "Bangladesh Cyclone"),
        ("1970-11-12", 85, "Bhola Cyclone"),
        ("1975-08-05", 60, "Typhoon Nina / Banqiao Dam"),
        ("1984-12-03", 70, "Bhopal Disaster"),
        ("1986-04-26", 75, "Chernobyl"),
        ("1991-04-29", 80, "Bangladesh Cyclone"),
        ("1998-10-29", 55, "Hurricane Mitch"),
        ("2003-08-01", 50, "European Heat Wave"),
        ("2004-12-26", 95, "Indian Ocean Tsunami"),
        ("2005-08-29", 65, "Hurricane Katrina"),
        ("2008-05-02", 75, "Cyclone Nargis"),
        ("2010-01-12", 80, "Haiti Earthquake"),
        ("2011-03-11", 90, "Fukushima"),
        ("2013-11-08", 60, "Typhoon Haiyan"),
        ("2019-12-01", 100, "COVID-19 Pandemic (start)"),
        ("2022-06-01", 40, "Pakistan Floods"),
    ]

    return disasters


# ============================================================
# PART 3: COMPARE TEMPORAL PATTERNS
# ============================================================

def analyze_event_gaps(events, label):
    """Analyze the temporal structure of gaps between events."""

    if len(events) < 3:
        print(f"  {label}: insufficient events ({len(events)})")
        return {}

    dates = [datetime.strptime(e[0], "%Y-%m-%d") for e in events]
    dates.sort()

    # Gaps in days
    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    gaps = np.array(gaps, dtype=float)

    # Filter zero gaps
    gaps = gaps[gaps > 0]

    if len(gaps) < 3:
        print(f"  {label}: insufficient gaps")
        return {}

    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    cv = std_gap / mean_gap if mean_gap > 0 else 0
    k_gaps = sp_stats.kurtosis(gaps, fisher=False)
    max_gap = np.max(gaps)
    min_gap = np.min(gaps)

    # Test for periodicity: autocorrelation of gaps
    if len(gaps) > 10:
        acf = np.corrcoef(gaps[:-1], gaps[1:])[0, 1]
    else:
        acf = 0

    # Test for clustering: runs test
    above_median = (gaps > np.median(gaps)).astype(int)
    runs = 1 + np.sum(np.diff(above_median) != 0)
    n1 = np.sum(above_median)
    n0 = len(above_median) - n1
    if n1 > 0 and n0 > 0:
        expected_runs = 1 + 2 * n1 * n0 / (n1 + n0)
        var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1))
        z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    else:
        z_runs = 0

    # Permutation entropy of gaps
    def pe(x, order=3):
        from itertools import permutations
        n = len(x)
        perms = list(permutations(range(order)))
        counts = {p: 0 for p in perms}
        for i in range(n - order + 1):
            pattern = tuple(np.argsort(x[i:i + order]))
            if pattern in counts:
                counts[pattern] += 1
        total = sum(counts.values())
        if total == 0:
            return 0
        probs = [c / total for c in counts.values() if c > 0]
        return -sum(p * np.log2(p) for p in probs) / np.log2(len(perms))

    perm_ent = pe(gaps) if len(gaps) >= 6 else 0

    # Hurst exponent of gaps
    def hurst(x):
        n = len(x)
        if n < 20:
            return 0.5
        ks = range(5, min(n // 4, 30))
        rs_vals = []
        for k in ks:
            chunks = [x[i:i+k] for i in range(0, n-k, k)]
            rs = []
            for ch in chunks:
                if len(ch) < k:
                    continue
                m = np.mean(ch)
                dev = np.cumsum(ch - m)
                r = np.max(dev) - np.min(dev)
                s = np.std(ch)
                if s > 0:
                    rs.append(r / s)
            if rs:
                rs_vals.append((np.log(k), np.log(np.mean(rs))))
        if len(rs_vals) < 3:
            return 0.5
        xv, yv = zip(*rs_vals)
        slope, _, _, _, _ = sp_stats.linregress(xv, yv)
        return slope

    h = hurst(gaps) if len(gaps) >= 20 else 0.5

    print(f"\n  {label} ({len(events)} events, {len(gaps)} gaps):")
    print(f"    Mean gap:     {mean_gap:.1f} days")
    print(f"    Std gap:      {std_gap:.1f} days")
    print(f"    CV:           {cv:.3f}  (1.0=random, >1=clustered, <1=regular)")
    print(f"    Kurtosis:     {k_gaps:.2f}")
    print(f"    Min gap:      {min_gap:.0f} days")
    print(f"    Max gap:      {max_gap:.0f} days")
    print(f"    ACF(1):       {acf:.4f}  (>0=momentum, <0=mean-reversion)")
    print(f"    Runs z:       {z_runs:.4f}")
    print(f"    Perm entropy: {perm_ent:.4f}")
    print(f"    Hurst:        {h:.4f}  (0.5=random, >0.5=persistent, <0.5=mean-reverting)")

    return {
        'n_events': len(events),
        'mean_gap': mean_gap,
        'cv': cv,
        'kurtosis': k_gaps,
        'acf': acf,
        'z_runs': z_runs,
        'pe': perm_ent,
        'hurst': h,
        'gaps': gaps.tolist(),
    }


def compare_crng_to_real(crng_event_map, real_results):
    """
    Compare CRNG gap distributions to real catastrophic events.
    Find the CRNG threshold that best matches each real dataset.
    """

    print(f"\n\n{'='*70}")
    print(f"  MATCHING CRNG TO REALITY")
    print(f"  Which K threshold best matches each type of real catastrophe?")
    print(f"{'='*70}")

    for real_label, real_stats in real_results.items():
        if not real_stats or 'cv' not in real_stats:
            continue

        print(f"\n  {real_label}:")
        print(f"    Real CV={real_stats['cv']:.3f}, K={real_stats['kurtosis']:.2f}, "
              f"ACF={real_stats['acf']:.4f}, Hurst={real_stats['hurst']:.4f}")

        best_match = None
        best_score = float('inf')

        for thresh, crng_data in crng_event_map.items():
            if crng_data['count'] < 5 or len(crng_data['gaps']) < 3:
                continue

            crng_gaps = np.array(crng_data['gaps'], dtype=float)
            crng_cv = np.std(crng_gaps) / np.mean(crng_gaps) if np.mean(crng_gaps) > 0 else 0
            crng_k = sp_stats.kurtosis(crng_gaps, fisher=False) if len(crng_gaps) >= 4 else 3
            crng_acf = np.corrcoef(crng_gaps[:-1], crng_gaps[1:])[0, 1] if len(crng_gaps) > 2 else 0

            # Distance metric: weighted combination of CV, K, ACF
            score = (abs(crng_cv - real_stats['cv']) * 2 +
                    abs(np.log(crng_k + 1) - np.log(real_stats['kurtosis'] + 1)) +
                    abs(crng_acf - real_stats['acf']) * 3)

            if score < best_score:
                best_score = score
                best_match = {
                    'threshold': thresh,
                    'crng_cv': crng_cv,
                    'crng_k': crng_k,
                    'crng_acf': crng_acf,
                    'score': score,
                    'n_events': crng_data['count'],
                }

        if best_match:
            print(f"    Best CRNG match: K≥{best_match['threshold']}")
            print(f"    CRNG CV={best_match['crng_cv']:.3f}, K={best_match['crng_k']:.2f}, "
                  f"ACF={best_match['crng_acf']:.4f}")
            print(f"    Match score: {best_match['score']:.4f} (lower=better)")
            print(f"    CRNG events at this threshold: {best_match['n_events']}")

    # Also test: KS test of gap distributions
    print(f"\n\n{'='*70}")
    print(f"  KS TEST: Are CRNG gaps from the same distribution as real gaps?")
    print(f"{'='*70}")

    for real_label, real_stats in real_results.items():
        if not real_stats or 'gaps' not in real_stats:
            continue

        real_gaps = np.array(real_stats['gaps'])
        real_gaps_norm = (real_gaps - np.mean(real_gaps)) / (np.std(real_gaps) + 1e-10)

        for thresh in [5, 8, 10, 15, 20]:
            if thresh not in crng_event_map:
                continue
            crng_data = crng_event_map[thresh]
            if len(crng_data['gaps']) < 5:
                continue

            crng_gaps = np.array(crng_data['gaps'], dtype=float)
            crng_gaps_norm = (crng_gaps - np.mean(crng_gaps)) / (np.std(crng_gaps) + 1e-10)

            ks_stat, ks_p = sp_stats.ks_2samp(real_gaps_norm, crng_gaps_norm)
            match = "MATCH" if ks_p > 0.05 else "differ"
            print(f"  {real_label:30s} vs CRNG K≥{thresh:>3}: KS={ks_stat:.4f} p={ks_p:.4f} [{match}]")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  CATASTROPHIC EVENTS: CRNG vs REALITY")
    print("  Is there a subliminal regularity in catastrophes?")
    print("=" * 70)

    # Part 1: Generate CRNG catastrophic map
    kurtosis_map, event_map, returns = generate_catastrophic_map(
        n_points=30000, window=50, seed=42
    )

    # Part 2: Analyze real catastrophic events
    print(f"\n\n{'#'*70}")
    print(f"# REAL CATASTROPHIC EVENTS — TEMPORAL ANALYSIS")
    print(f"{'#'*70}")

    earthquakes = get_earthquake_data()
    crashes = get_financial_crashes()
    disasters = get_natural_disasters()

    # Combine all
    all_catastrophes = []
    for e in earthquakes:
        all_catastrophes.append((e[0], e[1] * 10, f"EQ: {e[2]}"))
    for c in crashes:
        all_catastrophes.append((c[0], c[1], f"FIN: {c[2]}"))
    for d in disasters:
        all_catastrophes.append((d[0], d[1], f"NAT: {d[2]}"))

    real_results = {}
    real_results['Earthquakes (M≥6.7)'] = analyze_event_gaps(earthquakes, 'Earthquakes (M≥6.7)')
    real_results['Financial Crashes'] = analyze_event_gaps(crashes, 'Financial Crashes')
    real_results['Natural Disasters'] = analyze_event_gaps(disasters, 'Natural Disasters')
    real_results['ALL Catastrophes'] = analyze_event_gaps(all_catastrophes, 'ALL Catastrophes Combined')

    # Part 3: Analyze CRNG gaps at each threshold
    print(f"\n\n{'#'*70}")
    print(f"# CRNG GAP ANALYSIS BY THRESHOLD")
    print(f"{'#'*70}")

    crng_gap_stats = {}
    for thresh in [5, 6, 8, 10, 15, 20, 30]:
        if thresh in event_map and event_map[thresh]['count'] >= 5:
            gaps = np.array(event_map[thresh]['gaps'], dtype=float)
            if len(gaps) < 3:
                continue
            cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
            k = sp_stats.kurtosis(gaps, fisher=False) if len(gaps) >= 4 else 3
            acf = np.corrcoef(gaps[:-1], gaps[1:])[0, 1] if len(gaps) > 2 else 0

            print(f"\n  CRNG K≥{thresh} ({event_map[thresh]['count']} events, {len(gaps)} gaps):")
            print(f"    CV:       {cv:.3f}")
            print(f"    Kurtosis: {k:.2f}")
            print(f"    ACF(1):   {acf:.4f}")

            crng_gap_stats[thresh] = {'cv': cv, 'kurtosis': k, 'acf': acf}

    # Part 4: Compare
    compare_crng_to_real(event_map, real_results)

    # Part 5: The key question — is there periodicity?
    print(f"\n\n{'#'*70}")
    print(f"# THE KEY QUESTION: IS THERE HIDDEN PERIODICITY?")
    print(f"{'#'*70}")

    # FFT on gap sequences to detect periodic components
    for label, stats in {**real_results}.items():
        if not stats or 'gaps' not in stats or len(stats['gaps']) < 10:
            continue

        gaps = np.array(stats['gaps'], dtype=float)
        gaps_centered = gaps - np.mean(gaps)

        # FFT
        fft = np.fft.fft(gaps_centered)
        power = np.abs(fft[:len(fft)//2]) ** 2
        freqs = np.fft.fftfreq(len(gaps_centered))[:len(fft)//2]

        # Find dominant frequency (excluding DC)
        if len(power) > 1:
            power_no_dc = power[1:]
            freqs_no_dc = freqs[1:]
            dominant_idx = np.argmax(power_no_dc)
            dominant_freq = freqs_no_dc[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float('inf')
            power_ratio = power_no_dc[dominant_idx] / np.mean(power_no_dc)

            print(f"\n  {label}:")
            print(f"    Dominant period: {dominant_period:.1f} gaps")
            print(f"    Power ratio:     {power_ratio:.2f}x above mean")
            print(f"    (>3x = significant periodicity)")

    # Same for CRNG
    for thresh in [5, 8, 10, 15]:
        if thresh not in event_map or len(event_map[thresh]['gaps']) < 10:
            continue

        gaps = np.array(event_map[thresh]['gaps'], dtype=float)
        gaps_centered = gaps - np.mean(gaps)

        fft = np.fft.fft(gaps_centered)
        power = np.abs(fft[:len(fft)//2]) ** 2
        freqs = np.fft.fftfreq(len(gaps_centered))[:len(fft)//2]

        if len(power) > 1:
            power_no_dc = power[1:]
            freqs_no_dc = freqs[1:]
            dominant_idx = np.argmax(power_no_dc)
            dominant_freq = freqs_no_dc[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float('inf')
            power_ratio = power_no_dc[dominant_idx] / np.mean(power_no_dc)

            print(f"\n  CRNG K≥{thresh}:")
            print(f"    Dominant period: {dominant_period:.1f} gaps")
            print(f"    Power ratio:     {power_ratio:.2f}x above mean")

    print(f"\n\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
