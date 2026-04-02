"""
SUNSPOT ANALYSIS: CRNG vs SOLAR CYCLES
=========================================

Test: Does CRNG's temporal structure match the quasi-periodic
pattern of solar activity — another domain where structure emerges
from the interaction of independent oscillatory processes?

Solar sunspot data from SILSO (1749-2026, monthly SSN).
Focus: extreme solar maxima, kurtosis of the series, gap
distributions between major solar events.

Ale Brotto — 2026-03-31
"""

import numpy as np
from scipy import stats as sp_stats
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crng import ContingencyRNG


# ============================================================
# PART 1: LOAD AND ANALYZE SUNSPOT DATA
# ============================================================

def load_sunspot_data(filepath="/tmp/sunspot_monthly.csv"):
    """Load SILSO monthly sunspot number data."""

    years = []
    months = []
    ssn = []       # sunspot number
    dates = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 4:
                y = int(parts[0].strip())
                m = int(parts[1].strip())
                val = float(parts[3].strip())
                if val >= 0:  # -1 = missing
                    years.append(y)
                    months.append(m)
                    ssn.append(val)
                    dates.append(datetime(y, m, 15))

    return np.array(years), np.array(months), np.array(ssn), dates


def analyze_sunspot_series(ssn, dates):
    """Full statistical analysis of the sunspot series."""

    print(f"\n{'='*70}")
    print(f"  SUNSPOT SERIES ANALYSIS")
    print(f"  {len(ssn)} months ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")
    print(f"{'='*70}")

    # Basic stats
    print(f"\n  --- Basic Statistics ---")
    print(f"  Mean SSN:     {np.mean(ssn):.1f}")
    print(f"  Median SSN:   {np.median(ssn):.1f}")
    print(f"  Std SSN:      {np.std(ssn):.1f}")
    print(f"  Max SSN:      {np.max(ssn):.1f}")
    print(f"  Min SSN:      {np.min(ssn):.1f}")
    print(f"  Kurtosis:     {sp_stats.kurtosis(ssn, fisher=False):.2f}")
    print(f"  Skewness:     {sp_stats.skew(ssn):.2f}")

    # Monthly returns (changes)
    returns = np.diff(ssn) / (np.abs(ssn[:-1]) + 1.0)
    returns = returns[np.isfinite(returns)]

    print(f"\n  --- Monthly Returns ---")
    print(f"  Mean return:  {np.mean(returns):.4f}")
    print(f"  Std return:   {np.std(returns):.4f}")
    print(f"  Kurtosis:     {sp_stats.kurtosis(returns, fisher=False):.2f}")
    print(f"  Vol clustering (ACF1): {np.corrcoef(np.abs(returns[:-1]), np.abs(returns[1:]))[0,1]:.4f}")

    # Detect solar maxima (peaks)
    # A maximum is a month where SSN is higher than 24 months before and after
    maxima = []
    window = 24  # 2 years
    for i in range(window, len(ssn) - window):
        local = ssn[i-window:i+window+1]
        if ssn[i] == np.max(local) and ssn[i] > 100:  # significant maxima only
            maxima.append({
                'index': i,
                'date': dates[i],
                'ssn': ssn[i],
            })

    # Merge nearby maxima (within 36 months)
    merged = []
    for m in maxima:
        if not merged or (m['date'] - merged[-1]['date']).days > 36 * 30:
            merged.append(m)
        elif m['ssn'] > merged[-1]['ssn']:
            merged[-1] = m
    maxima = merged

    print(f"\n  --- Solar Maxima (SSN > 100, 24-month window) ---")
    print(f"  Found {len(maxima)} solar maxima:")
    for m in maxima:
        print(f"    {m['date'].strftime('%Y-%m')}: SSN = {m['ssn']:.1f}")

    # Gaps between maxima
    if len(maxima) >= 3:
        gaps_days = [(maxima[i+1]['date'] - maxima[i]['date']).days
                     for i in range(len(maxima)-1)]
        gaps_months = [g / 30.44 for g in gaps_days]
        gaps_years = [g / 365.25 for g in gaps_days]
        gaps = np.array(gaps_days, dtype=float)

        print(f"\n  --- Gaps Between Solar Maxima ---")
        print(f"  N gaps:       {len(gaps)}")
        print(f"  Mean gap:     {np.mean(gaps):.0f} days ({np.mean(gaps_years):.1f} years)")
        print(f"  Median gap:   {np.median(gaps):.0f} days ({np.median(gaps)/365.25:.1f} years)")
        print(f"  Std gap:      {np.std(gaps):.0f} days")
        print(f"  CV:           {np.std(gaps)/np.mean(gaps):.3f}")
        print(f"  Kurtosis:     {sp_stats.kurtosis(gaps, fisher=False):.2f}")
        print(f"  Min gap:      {np.min(gaps):.0f} days ({np.min(gaps)/365.25:.1f} years)")
        print(f"  Max gap:      {np.max(gaps):.0f} days ({np.max(gaps)/365.25:.1f} years)")

        if len(gaps) > 3:
            acf = np.corrcoef(gaps[:-1], gaps[1:])[0,1]
            print(f"  ACF(1):       {acf:.4f}")

        # FFT for periodicity within the gap sequence
        if len(gaps) >= 6:
            gaps_c = gaps - np.mean(gaps)
            fft = np.fft.fft(gaps_c)
            power = np.abs(fft[:len(fft)//2]) ** 2
            freqs = np.fft.fftfreq(len(gaps_c))[:len(fft)//2]

            if len(power) > 1:
                power_no_dc = power[1:]
                freqs_no_dc = freqs[1:]
                dom_idx = np.argmax(power_no_dc)
                dom_freq = freqs_no_dc[dom_idx]
                period = 1.0 / dom_freq if dom_freq > 0 else float('inf')
                ratio = power_no_dc[dom_idx] / np.mean(power_no_dc)

                print(f"\n  --- FFT on Maxima Gaps ---")
                print(f"  Dominant period: {period:.1f} gaps = {period * np.mean(gaps_years):.1f} years")
                print(f"  Power ratio:     {ratio:.2f}x (>3x = significant)")

    # Detect extreme months (SSN > P95)
    p95 = np.percentile(ssn, 95)
    p99 = np.percentile(ssn, 99)

    extreme_indices = np.where(ssn >= p95)[0]

    # Merge into events (contiguous months)
    events = []
    if len(extreme_indices) > 0:
        current_start = extreme_indices[0]
        current_peak = extreme_indices[0]
        for idx in extreme_indices[1:]:
            if idx - extreme_indices[list(extreme_indices).index(idx)-1] <= 3:
                if ssn[idx] > ssn[current_peak]:
                    current_peak = idx
            else:
                events.append({'peak': current_peak, 'ssn': ssn[current_peak], 'date': dates[current_peak]})
                current_start = idx
                current_peak = idx
        events.append({'peak': current_peak, 'ssn': ssn[current_peak], 'date': dates[current_peak]})

    print(f"\n  --- Extreme Solar Events (SSN > P95 = {p95:.0f}) ---")
    print(f"  Found {len(events)} extreme events")

    if len(events) >= 3:
        event_gaps = [(events[i+1]['date'] - events[i]['date']).days
                      for i in range(len(events)-1)]
        event_gaps = np.array([g for g in event_gaps if g > 0], dtype=float)

        print(f"  Mean gap:   {np.mean(event_gaps):.0f} days ({np.mean(event_gaps)/365.25:.1f} years)")
        print(f"  CV:         {np.std(event_gaps)/np.mean(event_gaps):.3f}")
        print(f"  Kurtosis:   {sp_stats.kurtosis(event_gaps, fisher=False):.2f}")

    return {
        'ssn': ssn,
        'returns': returns,
        'maxima': maxima,
        'maxima_gaps': gaps if len(maxima) >= 3 else np.array([]),
        'extreme_events': events,
        'extreme_gaps': event_gaps if len(events) >= 3 else np.array([]),
    }


# ============================================================
# PART 2: CRNG COMPARISON
# ============================================================

def crng_sunspot_comparison(solar_data):
    """Compare CRNG gap distributions with sunspot gap distributions."""

    print(f"\n\n{'='*70}")
    print(f"  CRNG vs SUNSPOTS: GAP DISTRIBUTION COMPARISON")
    print(f"{'='*70}")

    # Generate CRNG series and detect extreme events
    rng = ContingencyRNG(
        seed=42, n_oscillators=7,
        target_kurtosis=15.0, vol_clustering=0.35
    )

    n_points = 30000
    window = 50

    print(f"\n  Generating CRNG series ({n_points:,} points, window={window})...")
    series = np.array([rng.next() for _ in range(n_points)])
    returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
    returns = returns[np.isfinite(returns)]

    # Sliding window kurtosis
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(returns, window)
    means = np.mean(windows, axis=1, keepdims=True)
    centered = windows - means
    m2 = np.mean(centered**2, axis=1)
    m4 = np.mean(centered**4, axis=1)
    m2_safe = np.where(m2 > 1e-20, m2, 1e-20)
    kurtosis_map = m4 / (m2_safe ** 2)

    print(f"  CRNG kurtosis map: mean={np.mean(kurtosis_map):.2f}, max={np.max(kurtosis_map):.2f}")

    # Extract CRNG events at various thresholds
    thresholds = [5, 8, 10, 15, 20, 30]
    crng_gaps_by_thresh = {}

    for thresh in thresholds:
        indices = np.where(kurtosis_map >= thresh)[0]
        if len(indices) < 5:
            continue

        # Merge nearby
        events = [indices[0]]
        for idx in indices[1:]:
            if idx - events[-1] > window // 2:
                events.append(idx)

        if len(events) >= 3:
            gaps = np.diff(events).astype(float)
            crng_gaps_by_thresh[thresh] = gaps

    # === COMPARISON: Solar maxima gaps vs CRNG ===

    datasets = {}
    if len(solar_data['maxima_gaps']) >= 3:
        datasets['Solar Maxima'] = solar_data['maxima_gaps']
    if len(solar_data['extreme_gaps']) >= 3:
        datasets['Solar Extreme (P95)'] = solar_data['extreme_gaps']

    # Also analyze the raw SSN series kurtosis
    print(f"\n  --- Sunspot Series Kurtosis ---")
    ssn = solar_data['ssn']
    ssn_returns = solar_data['returns']
    print(f"  Raw SSN kurtosis:     {sp_stats.kurtosis(ssn, fisher=False):.2f}")
    print(f"  SSN returns kurtosis: {sp_stats.kurtosis(ssn_returns, fisher=False):.2f}")
    print(f"  PRNG baseline:        3.00")
    print(f"  → SSN {'HAS' if sp_stats.kurtosis(ssn, fisher=False) > 4 else 'does NOT have'} fat tails (K={'>' if sp_stats.kurtosis(ssn, fisher=False) > 4 else '<'}4)")
    print(f"  → SSN returns {'HAVE' if sp_stats.kurtosis(ssn_returns, fisher=False) > 4 else 'do NOT have'} fat tails")

    # Volatility clustering in SSN
    abs_ret = np.abs(ssn_returns)
    if len(abs_ret) > 2:
        vol_acf = np.corrcoef(abs_ret[:-1], abs_ret[1:])[0,1]
        print(f"  Vol clustering ACF(1): {vol_acf:.4f} ({'YES' if vol_acf > 0.1 else 'NO'} clustering)")

    # === KS TESTS ===

    print(f"\n  --- KS TEST: CRNG gaps vs Solar gaps ---")
    print(f"  {'Dataset':35s} {'vs CRNG':15s} {'KS':>8} {'p-value':>10} {'Result':>8}")
    print(f"  {'-'*80}")

    results = []

    for label, real_gaps in datasets.items():
        real_norm = (real_gaps - np.mean(real_gaps)) / (np.std(real_gaps) + 1e-10)

        for thresh, crng_gaps in crng_gaps_by_thresh.items():
            crng_norm = (crng_gaps - np.mean(crng_gaps)) / (np.std(crng_gaps) + 1e-10)

            ks_stat, ks_p = sp_stats.ks_2samp(real_norm, crng_norm)
            match = "MATCH" if ks_p > 0.05 else "differ"

            results.append({
                'label': label,
                'threshold': thresh,
                'ks': ks_stat,
                'p': ks_p,
                'match': match,
            })

            print(f"  {label:35s} K≥{thresh:<10d} {ks_stat:>8.4f} {ks_p:>10.4f} [{match}]")

    # === CV COMPARISON ===

    print(f"\n  --- Coefficient of Variation Comparison ---")
    for label, real_gaps in datasets.items():
        real_cv = np.std(real_gaps) / np.mean(real_gaps)
        print(f"\n  {label}: CV = {real_cv:.3f}")
        for thresh, crng_gaps in crng_gaps_by_thresh.items():
            crng_cv = np.std(crng_gaps) / np.mean(crng_gaps)
            diff = abs(crng_cv - real_cv)
            marker = " ←" if diff < 0.1 else ""
            print(f"    CRNG K≥{thresh}: CV = {crng_cv:.3f} (Δ={diff:.3f}){marker}")

    # === FFT PERIODICITY ===

    print(f"\n  --- FFT Periodicity Analysis ---")

    for label, gaps in {**datasets, **{f'CRNG K≥{t}': g for t, g in crng_gaps_by_thresh.items()}}.items():
        if len(gaps) < 8:
            continue

        gaps_c = gaps - np.mean(gaps)
        fft = np.fft.fft(gaps_c)
        power = np.abs(fft[:len(fft)//2]) ** 2
        freqs = np.fft.fftfreq(len(gaps_c))[:len(fft)//2]

        if len(power) > 1:
            power_no_dc = power[1:]
            freqs_no_dc = freqs[1:]
            dom_idx = np.argmax(power_no_dc)
            dom_freq = freqs_no_dc[dom_idx]
            period = 1.0 / dom_freq if dom_freq > 0 else float('inf')
            ratio = power_no_dc[dom_idx] / np.mean(power_no_dc)
            sig = "SIGNIFICANT" if ratio > 3 else "not significant"

            print(f"  {label:35s}: period={period:.1f} gaps, power ratio={ratio:.2f}x [{sig}]")

    # === DIRECT SSN SERIES vs CRNG SERIES COMPARISON ===

    print(f"\n\n{'='*70}")
    print(f"  DIRECT SERIES COMPARISON: SSN vs CRNG")
    print(f"{'='*70}")

    # Generate CRNG series calibrated to SSN length
    ssn = solar_data['ssn']
    n_ssn = len(ssn)

    # Standard PRNG for baseline
    np.random.seed(42)
    prng_series = np.cumsum(np.random.randn(n_ssn)) + np.mean(ssn)
    prng_series = np.abs(prng_series)  # SSN is always positive

    # CRNG series
    rng2 = ContingencyRNG(seed=42, target_kurtosis=8.0, vol_clustering=0.35, n_oscillators=7)
    crng_raw = np.array([rng2.next() for _ in range(n_ssn)])
    # Scale to SSN range
    crng_series = (crng_raw - np.min(crng_raw)) / (np.max(crng_raw) - np.min(crng_raw))
    crng_series = crng_series * (np.max(ssn) - np.min(ssn)) + np.min(ssn)

    # Compare sliding window kurtosis
    w = 132  # 11 years in months

    print(f"\n  Sliding window kurtosis (window = {w} months = ~11 years):")

    for name, s in [("Real SSN", ssn), ("CRNG", crng_series), ("PRNG", prng_series)]:
        if len(s) < w + 10:
            continue
        wins = sliding_window_view(s, w)
        ks = []
        for win in wins:
            if np.std(win) > 0:
                ks.append(sp_stats.kurtosis(win, fisher=False))
        ks = np.array(ks)
        print(f"  {name:12s}: mean K={np.mean(ks):.2f}, max K={np.max(ks):.2f}, "
              f"std K={np.std(ks):.2f}, K of K={sp_stats.kurtosis(ks, fisher=False):.2f}")

    # Permutation entropy comparison
    def perm_entropy(x, order=3):
        from itertools import permutations
        n = len(x)
        perms = list(permutations(range(order)))
        counts = {p: 0 for p in perms}
        for i in range(n - order + 1):
            pattern = tuple(np.argsort(x[i:i+order]))
            if pattern in counts:
                counts[pattern] += 1
        total = sum(counts.values())
        if total == 0:
            return 0
        probs = [c / total for c in counts.values() if c > 0]
        return -sum(p * np.log2(p) for p in probs) / np.log2(len(perms))

    print(f"\n  Permutation Entropy (order=3):")
    for name, s in [("Real SSN", ssn), ("CRNG", crng_series), ("PRNG", prng_series)]:
        pe = perm_entropy(s)
        print(f"  {name:12s}: PE = {pe:.4f}")

    # Hurst exponent
    def hurst(x):
        n = len(x)
        ks = range(10, min(n // 4, 200))
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

    print(f"\n  Hurst Exponent:")
    for name, s in [("Real SSN", ssn), ("CRNG", crng_series), ("PRNG", prng_series)]:
        h = hurst(s)
        label = "persistent" if h > 0.55 else ("anti-persistent" if h < 0.45 else "random")
        print(f"  {name:12s}: H = {h:.4f} ({label})")

    # === SCORECARD ===

    print(f"\n\n{'='*70}")
    print(f"  SCORECARD: CRNG vs PRNG (who matches SSN better?)")
    print(f"{'='*70}")

    metrics = {}
    ssn_k = sp_stats.kurtosis(ssn, fisher=False)
    ssn_ret_k = sp_stats.kurtosis(solar_data['returns'], fisher=False)
    ssn_pe = perm_entropy(ssn)
    ssn_h = hurst(ssn)
    ssn_vol_acf = np.corrcoef(np.abs(solar_data['returns'][:-1]), np.abs(solar_data['returns'][1:]))[0,1]

    crng_returns = np.diff(crng_series) / (np.abs(crng_series[:-1]) + 1.0)
    crng_returns = crng_returns[np.isfinite(crng_returns)]
    prng_returns = np.diff(prng_series) / (np.abs(prng_series[:-1]) + 1.0)
    prng_returns = prng_returns[np.isfinite(prng_returns)]

    comparisons = [
        ("Series Kurtosis", ssn_k,
         sp_stats.kurtosis(crng_series, fisher=False),
         sp_stats.kurtosis(prng_series, fisher=False)),
        ("Returns Kurtosis", ssn_ret_k,
         sp_stats.kurtosis(crng_returns, fisher=False),
         sp_stats.kurtosis(prng_returns, fisher=False)),
        ("Permutation Entropy", ssn_pe,
         perm_entropy(crng_series),
         perm_entropy(prng_series)),
        ("Hurst Exponent", ssn_h,
         hurst(crng_series),
         hurst(prng_series)),
        ("Vol Clustering ACF", ssn_vol_acf,
         np.corrcoef(np.abs(crng_returns[:-1]), np.abs(crng_returns[1:]))[0,1] if len(crng_returns) > 2 else 0,
         np.corrcoef(np.abs(prng_returns[:-1]), np.abs(prng_returns[1:]))[0,1] if len(prng_returns) > 2 else 0),
    ]

    crng_wins = 0
    prng_wins = 0

    print(f"\n  {'Metric':25s} {'Real SSN':>10} {'CRNG':>10} {'PRNG':>10} {'Winner':>8}")
    print(f"  {'-'*68}")

    for name, real, crng, prng in comparisons:
        crng_dist = abs(crng - real)
        prng_dist = abs(prng - real)

        if crng_dist < prng_dist:
            winner = "CRNG"
            crng_wins += 1
        else:
            winner = "PRNG"
            prng_wins += 1

        print(f"  {name:25s} {real:>10.3f} {crng:>10.3f} {prng:>10.3f} {winner:>8}")

    total = crng_wins + prng_wins
    print(f"\n  CRNG wins: {crng_wins}/{total} ({crng_wins/total*100:.0f}%)")
    print(f"  PRNG wins: {prng_wins}/{total} ({prng_wins/total*100:.0f}%)")

    # Count KS matches
    n_matches = sum(1 for r in results if r['match'] == 'MATCH')
    n_total = len(results)
    print(f"\n  KS test matches: {n_matches}/{n_total}")

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  SUNSPOT ANALYSIS: CRNG vs SOLAR CYCLES")
    print("  Does CRNG match another quasi-periodic natural phenomenon?")
    print("=" * 70)

    # Load data
    years, months, ssn, dates = load_sunspot_data()
    print(f"\n  Loaded {len(ssn)} monthly sunspot numbers")
    print(f"  Period: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    print(f"  Span: {(dates[-1] - dates[0]).days / 365.25:.1f} years")

    # Analyze
    solar_data = analyze_sunspot_series(ssn, dates)

    # Compare with CRNG
    results = crng_sunspot_comparison(solar_data)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
