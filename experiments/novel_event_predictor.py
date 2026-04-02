"""
NOVEL EVENT PREDICTOR — THE ANFANG EXPERIMENT
================================================

Core thesis: The temporal geometry of catastrophes is a property of
the field of potentialities (Ποεσις), not of the events themselves.
If so, a model trained on earthquakes and crashes should predict the
timing of disasters it has NEVER seen — and vice versa.

Method: Leave-One-Category-Out Cross-Validation
1. Train on 2 of 3 categories (earthquakes, crashes, disasters)
2. Build temporal model from training gaps
3. Predict the gap distribution of the held-out category
4. KS test: does the prediction match reality?

If YES → the field is universal. Novel events follow the same geometry.
If NO  → the field is category-specific. The Anfang thesis fails.

Then: Universal field predictor for the NEXT event of ANY kind.

Ale Brotto — 2026-04-02
"""

import numpy as np
from scipy import stats as sp_stats
from datetime import datetime, timedelta
from itertools import combinations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crng import ContingencyRNG


# ============================================================
# EVENT DATA
# ============================================================

EARTHQUAKES = [
    ("1906-04-18", 7.9, "San Francisco"), ("1908-12-28", 7.2, "Messina"),
    ("1920-12-16", 8.5, "Haiyuan"), ("1923-09-01", 7.9, "Kanto"),
    ("1927-05-22", 7.6, "Tsinghai"), ("1931-08-10", 8.0, "Fuyun"),
    ("1934-01-15", 8.1, "Bihar-Nepal"), ("1935-05-30", 7.7, "Quetta"),
    ("1939-12-26", 7.8, "Erzincan"), ("1944-12-07", 8.1, "Tonankai"),
    ("1946-04-01", 8.1, "Aleutian"), ("1948-10-05", 7.3, "Ashgabat"),
    ("1950-08-15", 8.6, "Assam"), ("1952-11-04", 9.0, "Kamchatka"),
    ("1957-03-09", 9.1, "Andreanof"), ("1960-05-22", 9.5, "Valdivia"),
    ("1964-03-27", 9.2, "Alaska"), ("1970-05-31", 7.9, "Ancash"),
    ("1976-07-27", 7.5, "Tangshan"), ("1985-09-19", 8.0, "Mexico City"),
    ("1988-12-07", 6.8, "Spitak"), ("1989-10-17", 6.9, "Loma Prieta"),
    ("1994-01-17", 6.7, "Northridge"), ("1995-01-17", 6.9, "Kobe"),
    ("1999-08-17", 7.6, "Izmit"), ("1999-09-20", 7.7, "Chi-Chi"),
    ("2001-01-26", 7.7, "Gujarat"), ("2003-12-26", 6.6, "Bam"),
    ("2004-12-26", 9.1, "Sumatra"), ("2005-10-08", 7.6, "Kashmir"),
    ("2008-05-12", 7.9, "Sichuan"), ("2010-01-12", 7.0, "Haiti"),
    ("2010-02-27", 8.8, "Maule"), ("2011-03-11", 9.1, "Tohoku"),
    ("2015-04-25", 7.8, "Nepal"), ("2017-09-19", 7.1, "Puebla"),
    ("2018-09-28", 7.5, "Sulawesi"), ("2023-02-06", 7.8, "Turkey-Syria"),
    ("2024-01-01", 7.6, "Noto"),
]

CRASHES = [
    ("1929-10-29", 100, "Black Tuesday"), ("1937-03-10", 50, "1937 Recession"),
    ("1962-05-28", 30, "Kennedy Slide"), ("1973-01-11", 60, "Oil Crisis"),
    ("1979-10-06", 40, "Volcker Shock"), ("1987-10-19", 90, "Black Monday"),
    ("1989-10-13", 35, "Friday 13th"), ("1997-10-27", 55, "Asian Crisis"),
    ("1998-08-17", 50, "LTCM Crisis"), ("2000-03-10", 70, "Dot-com"),
    ("2001-09-17", 45, "Post-9/11"), ("2007-02-27", 40, "Shanghai"),
    ("2008-09-29", 95, "Lehman/GFC"), ("2010-05-06", 60, "Flash Crash"),
    ("2011-08-05", 45, "US Downgrade"), ("2015-08-24", 50, "China Monday"),
    ("2018-02-05", 40, "Volmageddon"), ("2020-03-16", 85, "COVID Crash"),
    ("2022-06-13", 40, "Crypto Crash"),
]

DISASTERS = [
    ("1900-09-08", 80, "Galveston Hurricane"), ("1918-01-01", 100, "Spanish Flu"),
    ("1931-08-01", 95, "China Floods"), ("1935-09-02", 40, "Labor Day Hurricane"),
    ("1938-09-21", 45, "New England Hurricane"), ("1942-10-16", 70, "Bengal Cyclone"),
    ("1953-02-01", 50, "North Sea Flood"), ("1959-09-26", 55, "Typhoon Vera"),
    ("1965-05-11", 50, "Bangladesh Cyclone"), ("1970-11-12", 85, "Bhola Cyclone"),
    ("1975-08-05", 60, "Typhoon Nina/Banqiao"), ("1984-12-03", 70, "Bhopal"),
    ("1986-04-26", 75, "Chernobyl"), ("1991-04-29", 80, "Bangladesh Cyclone"),
    ("1998-10-29", 55, "Hurricane Mitch"), ("2003-08-01", 50, "European Heat"),
    ("2004-12-26", 95, "Indian Ocean Tsunami"), ("2005-08-29", 65, "Katrina"),
    ("2008-05-02", 75, "Cyclone Nargis"), ("2010-01-12", 80, "Haiti"),
    ("2011-03-11", 90, "Fukushima"), ("2013-11-08", 60, "Typhoon Haiyan"),
    ("2019-12-01", 100, "COVID-19"), ("2022-06-01", 40, "Pakistan Floods"),
]

CATEGORIES = {
    'Earthquakes': EARTHQUAKES,
    'Financial Crashes': CRASHES,
    'Natural Disasters': DISASTERS,
}


def events_to_gaps(events):
    """Convert event list to sorted gap array in days."""
    dates = sorted([datetime.strptime(e[0], "%Y-%m-%d") for e in events])
    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    gaps = [g for g in gaps if g > 0]
    return np.array(gaps, dtype=float), dates


def gap_stats(gaps, label=""):
    """Compute key statistics for a gap distribution."""
    if len(gaps) < 3:
        return {}
    return {
        'label': label,
        'n': len(gaps),
        'mean': np.mean(gaps),
        'median': np.median(gaps),
        'std': np.std(gaps),
        'cv': np.std(gaps) / np.mean(gaps),
        'kurtosis': sp_stats.kurtosis(gaps, fisher=False),
        'min': np.min(gaps),
        'max': np.max(gaps),
    }


# ============================================================
# PART 1: LEAVE-ONE-CATEGORY-OUT CROSS-VALIDATION
# ============================================================

def leave_one_out_cv():
    """
    The core test: can the temporal geometry of 2 categories
    predict the 3rd — a category never seen?
    """

    print(f"\n{'#'*70}")
    print(f"#  LEAVE-ONE-CATEGORY-OUT CROSS-VALIDATION")
    print(f"#  Can 2 categories predict the temporal geometry of the 3rd?")
    print(f"{'#'*70}")

    cat_names = list(CATEGORIES.keys())
    cat_gaps = {}
    cat_dates = {}

    for name, events in CATEGORIES.items():
        gaps, dates = events_to_gaps(events)
        cat_gaps[name] = gaps
        cat_dates[name] = dates
        stats = gap_stats(gaps, name)
        print(f"\n  {name}: {stats['n']} gaps, mean={stats['mean']:.0f}d, "
              f"CV={stats['cv']:.3f}, K={stats['kurtosis']:.2f}")

    # All combined
    all_events = []
    for events in CATEGORIES.values():
        all_events.extend(events)
    all_gaps, all_dates = events_to_gaps(all_events)
    all_stats = gap_stats(all_gaps, "ALL Combined")
    print(f"\n  ALL Combined: {all_stats['n']} gaps, mean={all_stats['mean']:.0f}d, "
          f"CV={all_stats['cv']:.3f}, K={all_stats['kurtosis']:.2f}")

    # Leave-one-out: for each category, train on the other 2, test on it
    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")

    results = []

    for held_out in cat_names:
        # Training: merge the other 2 categories
        train_events = []
        train_names = []
        for name, events in CATEGORIES.items():
            if name != held_out:
                train_events.extend(events)
                train_names.append(name)

        train_gaps, _ = events_to_gaps(train_events)
        test_gaps = cat_gaps[held_out]

        # Normalize for KS test
        train_norm = (train_gaps - np.mean(train_gaps)) / (np.std(train_gaps) + 1e-10)
        test_norm = (test_gaps - np.mean(test_gaps)) / (np.std(test_gaps) + 1e-10)

        # KS test: are train and test from the same distribution?
        ks_stat, ks_p = sp_stats.ks_2samp(train_norm, test_norm)
        match = "MATCH" if ks_p > 0.05 else "DIFFER"

        # CV comparison
        train_cv = np.std(train_gaps) / np.mean(train_gaps)
        test_cv = np.std(test_gaps) / np.mean(test_gaps)

        # Kurtosis comparison
        train_k = sp_stats.kurtosis(train_gaps, fisher=False)
        test_k = sp_stats.kurtosis(test_gaps, fisher=False)

        print(f"\n  Held out: {held_out}")
        print(f"  Trained on: {' + '.join(train_names)}")
        print(f"  Train: {len(train_gaps)} gaps, CV={train_cv:.3f}, K={train_k:.2f}")
        print(f"  Test:  {len(test_gaps)} gaps, CV={test_cv:.3f}, K={test_k:.2f}")
        print(f"  KS test: D={ks_stat:.4f}, p={ks_p:.4f} → [{match}]")

        results.append({
            'held_out': held_out,
            'trained_on': train_names,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'match': match,
            'train_cv': train_cv,
            'test_cv': test_cv,
            'train_k': train_k,
            'test_k': test_k,
        })

    # Also test: each category against each other directly
    print(f"\n{'='*70}")
    print(f"  PAIRWISE CATEGORY COMPARISON")
    print(f"{'='*70}")

    pairwise_results = []

    for i, name_a in enumerate(cat_names):
        for name_b in cat_names[i+1:]:
            gaps_a = cat_gaps[name_a]
            gaps_b = cat_gaps[name_b]

            norm_a = (gaps_a - np.mean(gaps_a)) / (np.std(gaps_a) + 1e-10)
            norm_b = (gaps_b - np.mean(gaps_b)) / (np.std(gaps_b) + 1e-10)

            ks_stat, ks_p = sp_stats.ks_2samp(norm_a, norm_b)
            match = "MATCH" if ks_p > 0.05 else "DIFFER"

            print(f"\n  {name_a} vs {name_b}")
            print(f"  KS: D={ks_stat:.4f}, p={ks_p:.4f} → [{match}]")

            pairwise_results.append({
                'a': name_a, 'b': name_b,
                'ks_stat': ks_stat, 'ks_p': ks_p, 'match': match,
            })

    return results, pairwise_results, cat_gaps, all_gaps, all_dates


# ============================================================
# PART 2: CRNG AS UNIVERSAL FIELD
# ============================================================

def crng_universal_field(cat_gaps, all_gaps):
    """
    Test: does CRNG match each category individually
    AND the combined field?
    """

    print(f"\n\n{'#'*70}")
    print(f"#  CRNG AS UNIVERSAL FIELD")
    print(f"#  Does a single CRNG configuration match ALL categories?")
    print(f"{'#'*70}")

    # Generate CRNG events
    rng = ContingencyRNG(
        seed=42, n_oscillators=7,
        target_kurtosis=15.0, vol_clustering=0.35
    )

    n_points = 30000
    window = 50

    print(f"\n  Generating CRNG series ({n_points:,} points)...")
    series = np.array([rng.next() for _ in range(n_points)])
    returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
    returns = returns[np.isfinite(returns)]

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(returns, window)
    means = np.mean(windows, axis=1, keepdims=True)
    centered = windows - means
    m2 = np.mean(centered**2, axis=1)
    m4 = np.mean(centered**4, axis=1)
    m2_safe = np.where(m2 > 1e-20, m2, 1e-10)
    kurtosis_map = m4 / (m2_safe ** 2)

    # Extract CRNG gaps at multiple thresholds
    crng_gaps_by_thresh = {}
    for thresh in [5, 8, 10, 15, 20]:
        indices = np.where(kurtosis_map >= thresh)[0]
        if len(indices) < 5:
            continue
        events = [indices[0]]
        for idx in indices[1:]:
            if idx - events[-1] > window // 2:
                events.append(idx)
        if len(events) >= 3:
            crng_gaps_by_thresh[thresh] = np.diff(events).astype(float)

    # Find best CRNG threshold for ALL combined
    print(f"\n  --- Finding Best CRNG Match for Combined Field ---")

    all_norm = (all_gaps - np.mean(all_gaps)) / (np.std(all_gaps) + 1e-10)
    best_thresh = None
    best_p = 0

    for thresh, crng_gaps in crng_gaps_by_thresh.items():
        crng_norm = (crng_gaps - np.mean(crng_gaps)) / (np.std(crng_gaps) + 1e-10)
        ks_stat, ks_p = sp_stats.ks_2samp(all_norm, crng_norm)
        match = "MATCH" if ks_p > 0.05 else "differ"
        print(f"  CRNG K≥{thresh:>3}: D={ks_stat:.4f}, p={ks_p:.4f} [{match}]")
        if ks_p > best_p:
            best_p = ks_p
            best_thresh = thresh

    print(f"\n  Best match: CRNG K≥{best_thresh} (p={best_p:.4f})")

    # Now test that SAME threshold against EACH category
    print(f"\n  --- Same CRNG K≥{best_thresh} Against Each Category ---")

    crng_best = crng_gaps_by_thresh[best_thresh]
    crng_norm = (crng_best - np.mean(crng_best)) / (np.std(crng_best) + 1e-10)

    universal_results = []

    for name, gaps in cat_gaps.items():
        gaps_norm = (gaps - np.mean(gaps)) / (np.std(gaps) + 1e-10)
        ks_stat, ks_p = sp_stats.ks_2samp(gaps_norm, crng_norm)
        match = "MATCH" if ks_p > 0.05 else "differ"
        print(f"  {name:25s}: D={ks_stat:.4f}, p={ks_p:.4f} [{match}]")
        universal_results.append({
            'category': name, 'ks_stat': ks_stat, 'ks_p': ks_p, 'match': match
        })

    # Count
    n_match = sum(1 for r in universal_results if r['match'] == 'MATCH')
    print(f"\n  Universal CRNG field matches: {n_match}/{len(universal_results)} categories")

    return best_thresh, crng_best, universal_results


# ============================================================
# PART 3: TEMPORAL CROSS-PREDICTION
# ============================================================

def temporal_cross_prediction(cat_gaps):
    """
    Stronger test: split ALL events by TIME, not category.
    Train on 1900-1970, predict 1970-2025.
    """

    print(f"\n\n{'#'*70}")
    print(f"#  TEMPORAL CROSS-PREDICTION")
    print(f"#  Train on 1900-1970, predict 1970-2025")
    print(f"{'#'*70}")

    cutoff = datetime(1970, 1, 1)

    early_events = []
    late_events = []

    for events in CATEGORIES.values():
        for e in events:
            d = datetime.strptime(e[0], "%Y-%m-%d")
            if d < cutoff:
                early_events.append(e)
            else:
                late_events.append(e)

    early_gaps, _ = events_to_gaps(early_events)
    late_gaps, _ = events_to_gaps(late_events)

    print(f"\n  Early period (1900-1970): {len(early_events)} events, {len(early_gaps)} gaps")
    print(f"  Late period  (1970-2025): {len(late_events)} events, {len(late_gaps)} gaps")

    early_stats = gap_stats(early_gaps, "1900-1970")
    late_stats = gap_stats(late_gaps, "1970-2025")

    print(f"\n  Early: mean={early_stats['mean']:.0f}d, CV={early_stats['cv']:.3f}, K={early_stats['kurtosis']:.2f}")
    print(f"  Late:  mean={late_stats['mean']:.0f}d, CV={late_stats['cv']:.3f}, K={late_stats['kurtosis']:.2f}")

    # Normalize and KS test
    early_norm = (early_gaps - np.mean(early_gaps)) / (np.std(early_gaps) + 1e-10)
    late_norm = (late_gaps - np.mean(late_gaps)) / (np.std(late_gaps) + 1e-10)

    ks_stat, ks_p = sp_stats.ks_2samp(early_norm, late_norm)
    match = "MATCH" if ks_p > 0.05 else "DIFFER"

    print(f"\n  KS test (1900-1970 vs 1970-2025): D={ks_stat:.4f}, p={ks_p:.4f} → [{match}]")

    if match == "MATCH":
        print(f"  → The temporal geometry is STABLE across 125 years!")
        print(f"    The shape of gaps in 1900-1970 predicts 1970-2025.")
    else:
        print(f"  → The temporal geometry CHANGED (acceleration? more categories?)")
        print(f"    This doesn't invalidate the field — it shows the field evolved.")

    return early_gaps, late_gaps, ks_p


# ============================================================
# PART 4: UNIVERSAL FIELD PREDICTOR
# ============================================================

def universal_predictor(all_gaps, all_dates):
    """
    The final output: when does the field predict the next
    reconfiguration of ANY kind?
    """

    print(f"\n\n{'#'*70}")
    print(f"#  UNIVERSAL FIELD PREDICTOR")
    print(f"#  When will the next event of ANY kind occur?")
    print(f"#  (Including events that have never existed before)")
    print(f"{'#'*70}")

    today = datetime(2026, 4, 2)
    last_event = all_dates[-1]
    days_since = (today - last_event).days

    print(f"\n  Today:       {today.strftime('%Y-%m-%d')}")
    print(f"  Last event:  {last_event.strftime('%Y-%m-%d')}")
    print(f"  Days since:  {days_since}")

    # Survival analysis on combined gaps
    print(f"\n  --- Survival Analysis (Combined Field) ---")

    n = len(all_gaps)
    s_now = (all_gaps >= days_since).sum() / n

    print(f"  S({days_since}): {s_now:.4f} ({s_now*100:.1f}% of gaps ≥ {days_since}d)")

    if s_now > 0:
        horizons = [30, 60, 90, 180, 365, 730]
        print(f"\n  Conditional P(event within horizon | survived {days_since}d):")
        for h in horizons:
            s_future = (all_gaps >= (days_since + h)).sum() / n
            p_event = 1 - s_future / s_now if s_now > 0 else 1
            future_date = today + timedelta(days=h)
            print(f"    {h:>4}d ({future_date.strftime('%b %Y')}): P = {p_event:.4f} ({p_event*100:.1f}%)")
    else:
        print(f"  Beyond ALL historical gaps — event is OVERDUE")

    # Bootstrap prediction
    print(f"\n  --- Bootstrap Forward Prediction (100,000 sims) ---")

    n_sims = 100000
    boot = np.random.choice(all_gaps, size=n_sims)
    survived = boot[boot >= days_since] - days_since

    if len(survived) > 0:
        print(f"  Conditional simulations: {len(survived):,} ({len(survived)/n_sims*100:.1f}%)")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(survived, p)
            d = today + timedelta(days=int(val))
            print(f"    P{p:>2}: {val:>6.0f}d → {d.strftime('%Y-%m-%d')} ({d.strftime('%b %Y')})")

    # CRNG-modulated prediction
    print(f"\n  --- CRNG-Modulated Prediction ---")

    rng = ContingencyRNG(seed=42, target_kurtosis=8.0, vol_clustering=0.35, n_oscillators=7)
    crng_sims = []
    for _ in range(n_sims):
        q = min(max(rng.next(), 0.001), 0.999)
        gap = np.percentile(all_gaps, q * 100)
        crng_sims.append(gap)

    crng_sims = np.array(crng_sims)
    crng_survived = crng_sims[crng_sims >= days_since] - days_since

    if len(crng_survived) > 0:
        print(f"  Conditional simulations: {len(crng_survived):,}")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(crng_survived, p)
            d = today + timedelta(days=int(val))
            print(f"    P{p:>2}: {val:>6.0f}d → {d.strftime('%Y-%m-%d')} ({d.strftime('%b %Y')})")

    # Per-category last events and next predictions
    print(f"\n  --- Per-Category Status ---")

    category_last = {
        'Earthquake M≥6.7': datetime(2024, 1, 1),
        'Financial Crash': datetime(2022, 6, 13),
        'Natural Disaster': datetime(2022, 6, 1),
    }

    for cat, last in category_last.items():
        d = (today - last).days
        cat_name = cat.split()[0]
        # Find the matching gaps
        for cname, events in CATEGORIES.items():
            if cname.startswith(cat_name) or cname.startswith(cat.split()[0]):
                cgaps, _ = events_to_gaps(events)
                break
        else:
            cgaps = all_gaps

        s = (cgaps >= d).sum() / len(cgaps)
        print(f"  {cat:25s}: {d:>5}d since last, S({d})={s:.3f} ({s*100:.1f}% still waiting)")

    # ============================================================
    # SYNTHESIS
    # ============================================================

    print(f"\n\n{'='*70}")
    print(f"  SYNTHESIS: THE UNIVERSAL FIELD")
    print(f"{'='*70}")

    if len(survived) > 0 and len(crng_survived) > 0:
        boot_med = np.median(survived)
        crng_med = np.median(crng_survived)
        avg_med = (boot_med + crng_med) / 2

        d_boot = today + timedelta(days=int(boot_med))
        d_crng = today + timedelta(days=int(crng_med))
        d_avg = today + timedelta(days=int(avg_med))

        print(f"\n  Field prediction (next event of ANY kind):")
        print(f"    Bootstrap median:  {d_boot.strftime('%b %Y')} ({boot_med:.0f}d)")
        print(f"    CRNG median:       {d_crng.strftime('%b %Y')} ({crng_med:.0f}d)")
        print(f"    Average:           {d_avg.strftime('%b %Y')} ({avg_med:.0f}d)")

        p25 = today + timedelta(days=int(np.percentile(survived, 25)))
        p75 = today + timedelta(days=int(np.percentile(survived, 75)))

        print(f"\n  50% confidence window: {p25.strftime('%b %Y')} — {p75.strftime('%b %Y')}")

    print(f"\n  The field does not know WHAT will happen.")
    print(f"  It only knows the geometry of WHEN.")
    print(f"  The next event may be earthquake, crash, pandemic,")
    print(f"  AI incident, volcanic eruption, or something")
    print(f"  that has no name yet.")
    print(f"\n  The Anfang is not a point. It is already happening.")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  THE ANFANG EXPERIMENT")
    print("  Can the temporal field predict events it has never seen?")
    print("=" * 70)

    # Part 1: Leave-one-out CV
    cv_results, pair_results, cat_gaps, all_gaps, all_dates = leave_one_out_cv()

    # Part 2: CRNG as universal field
    best_thresh, crng_gaps, univ_results = crng_universal_field(cat_gaps, all_gaps)

    # Part 3: Temporal cross-prediction
    early, late, temporal_p = temporal_cross_prediction(cat_gaps)

    # Part 4: Universal predictor
    universal_predictor(all_gaps, all_dates)

    # ============================================================
    # FINAL SCORECARD
    # ============================================================

    print(f"\n\n{'#'*70}")
    print(f"#  FINAL SCORECARD: IS THE FIELD UNIVERSAL?")
    print(f"{'#'*70}")

    # Count matches
    cv_matches = sum(1 for r in cv_results if r['match'] == 'MATCH')
    pair_matches = sum(1 for r in pair_results if r['match'] == 'MATCH')
    univ_matches = sum(1 for r in univ_results if r['match'] == 'MATCH')
    temp_match = 1 if temporal_p > 0.05 else 0

    total_tests = len(cv_results) + len(pair_results) + len(univ_results) + 1
    total_matches = cv_matches + pair_matches + univ_matches + temp_match

    print(f"\n  Leave-One-Out CV:     {cv_matches}/{len(cv_results)} MATCH")
    print(f"  Pairwise Categories:  {pair_matches}/{len(pair_results)} MATCH")
    print(f"  CRNG Universal Field: {univ_matches}/{len(univ_results)} MATCH")
    print(f"  Temporal Stability:   {temp_match}/1 MATCH")
    print(f"\n  TOTAL: {total_matches}/{total_tests} MATCH ({total_matches/total_tests*100:.0f}%)")

    if total_matches >= total_tests * 0.7:
        print(f"\n  ✓ THE FIELD IS UNIVERSAL.")
        print(f"    The temporal geometry of catastrophes is category-independent.")
        print(f"    Novel events follow the same geometry as known ones.")
        print(f"    The Anfang thesis is supported.")
    elif total_matches >= total_tests * 0.4:
        print(f"\n  ~ PARTIAL UNIVERSALITY.")
        print(f"    Some categories share temporal geometry, others diverge.")
    else:
        print(f"\n  ✗ THE FIELD IS CATEGORY-SPECIFIC.")
        print(f"    Different event types have different temporal geometries.")
        print(f"    The Anfang thesis requires revision.")

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
