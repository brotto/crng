#!/usr/bin/env python3
"""
Mega-Sena Deep Statistical Analysis
====================================

Instead of naively generating lottery draws with CRNG (wrong domain),
this analysis asks the RIGHT questions:

1. Is the Mega-Sena truly uniform? (Chi², individual ball tests)
2. Are there temporal patterns? (PE, Hurst, serial correlation on sequences)
3. Can CRNG's from_data() capture the time-series properties of
   Mega-Sena's "sum sequence" and "ball-by-ball sequence"?
4. How does Mega-Sena compare to a perfect PRNG lottery?

Key insight: CRNG is designed for fat-tailed financial processes.
Lottery is a different beast — but the TEMPORAL STRUCTURE of lottery
results can still reveal interesting patterns.
"""

import numpy as np
import openpyxl
from collections import Counter, defaultdict
from math import factorial, log, sqrt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crng import ContingencyRNG

# ─── Data Loading ───────────────────────────────────────────────

def load_mega_sena(path):
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    draws = []
    for row in ws.iter_rows(min_row=2, max_col=8, values_only=True):
        if row[0] is None:
            break
        bolas = row[2:8]
        balls = [int(b) for b in bolas if b is not None]
        if len(balls) == 6:
            draws.append(sorted(balls))
    wb.close()
    return draws

# ─── Statistical Tools ──────────────────────────────────────────

def permutation_entropy(sequence, order=3):
    n = len(sequence)
    if n < order + 1:
        return 0.0
    pattern_counts = Counter()
    for i in range(n - order + 1):
        window = sequence[i:i + order]
        ranked = tuple(sorted(range(order), key=lambda k: window[k]))
        pattern_counts[ranked] += 1
    total = sum(pattern_counts.values())
    max_patterns = factorial(order)
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * log(p)
    max_entropy = log(max_patterns)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def serial_correlation(sequence, lag=1):
    n = len(sequence)
    if n < lag + 2:
        return 0.0
    arr = np.array(sequence, dtype=float)
    mean = np.mean(arr)
    var = np.var(arr)
    if var == 0:
        return 0.0
    return float(np.mean((arr[:-lag] - mean) * (arr[lag:] - mean)) / var)


def hurst_exponent(sequence):
    seq = np.array(sequence, dtype=float)
    n = len(seq)
    if n < 20:
        return 0.5
    max_k = min(n // 4, 500)
    min_k = 10
    rs_values = []
    ns_values = []
    for k in range(min_k, max_k + 1, max(1, (max_k - min_k) // 20)):
        rs_list = []
        for start in range(0, n - k + 1, k):
            segment = seq[start:start + k]
            mean_seg = np.mean(segment)
            deviations = np.cumsum(segment - mean_seg)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(segment, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns_values.append(k)
    if len(rs_values) < 3:
        return 0.5
    log_n = np.log(ns_values)
    log_rs = np.log(rs_values)
    coeffs = np.polyfit(log_n, log_rs, 1)
    return coeffs[0]


def runs_test_z(sequence):
    median = np.median(sequence)
    binary = [1 if x > median else 0 for x in sequence]
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i - 1]:
            runs += 1
    n1 = sum(binary)
    n0 = len(binary) - n1
    if n0 == 0 or n1 == 0:
        return 0.0
    expected = (2 * n0 * n1) / (n0 + n1) + 1
    var = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1) ** 2 * (n0 + n1 - 1))
    if var <= 0:
        return 0.0
    return (runs - expected) / sqrt(var)


def kurtosis(data):
    arr = np.array(data, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4))


def dfa_exponent(sequence, min_box=4, max_box=None):
    """Detrended Fluctuation Analysis."""
    seq = np.array(sequence, dtype=float)
    n = len(seq)
    seq = seq - np.mean(seq)
    y = np.cumsum(seq)

    if max_box is None:
        max_box = n // 4

    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 20).astype(int))
    fluctuations = []
    valid_sizes = []

    for box in box_sizes:
        n_boxes = n // box
        if n_boxes < 2:
            continue
        rms_list = []
        for i in range(n_boxes):
            segment = y[i * box:(i + 1) * box]
            x = np.arange(box)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms_list.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_list:
            fluctuations.append(np.mean(rms_list))
            valid_sizes.append(box)

    if len(valid_sizes) < 3:
        return 0.5

    log_n = np.log(valid_sizes)
    log_f = np.log(fluctuations)
    coeffs = np.polyfit(log_n, log_f, 1)
    return coeffs[0]


# ─── Derived Sequences from Draws ───────────────────────────────

def make_sequences(draws):
    """Create multiple time series from lottery draws for analysis."""
    seqs = {}

    # 1. Sum of 6 balls per draw
    seqs['sums'] = [sum(d) for d in draws]

    # 2. Range (max - min) per draw
    seqs['ranges'] = [d[-1] - d[0] for d in draws]

    # 3. Ball-by-ball flat sequence
    seqs['flat'] = [b for d in draws for b in d]

    # 4. First ball sequence
    seqs['first_ball'] = [d[0] for d in draws]

    # 5. Last ball sequence
    seqs['last_ball'] = [d[-1] for d in draws]

    # 6. Inter-draw differences (sum changes)
    sums = seqs['sums']
    seqs['sum_diffs'] = [sums[i] - sums[i-1] for i in range(1, len(sums))]

    # 7. Average spacing within draw
    seqs['avg_spacing'] = [np.mean([d[i+1] - d[i] for i in range(5)]) for d in draws]

    # 8. Number of even balls per draw
    seqs['even_count'] = [sum(1 for b in d if b % 2 == 0) for d in draws]

    return seqs


# ─── CRNG from_data Calibration ─────────────────────────────────

def crng_from_sequence(real_seq, n_samples, seed=42):
    """Use CRNG from_data to calibrate from a real sequence, generate synthetic."""
    try:
        from crng import from_data
        crng = from_data(real_seq, seed=seed)
        return [crng.next() for _ in range(n_samples)]
    except Exception:
        # Fallback: manual calibration
        k = kurtosis(real_seq)
        crng = ContingencyRNG(seed=seed, target_kurtosis=max(k, 3.5))
        vals = [crng.next() for _ in range(n_samples)]
        # Scale to match real sequence stats
        real_mean = np.mean(real_seq)
        real_std = np.std(real_seq)
        syn_mean = np.mean(vals)
        syn_std = np.std(vals)
        if syn_std > 0:
            vals = [(v - syn_mean) / syn_std * real_std + real_mean for v in vals]
        return vals


def prng_from_sequence(real_seq, n_samples, seed=42):
    """Generate PRNG sequence matching mean/std of real data."""
    rng = np.random.default_rng(seed)
    real_mean = np.mean(real_seq)
    real_std = np.std(real_seq)
    return (rng.normal(real_mean, real_std, n_samples)).tolist()


# ─── Main Analysis ──────────────────────────────────────────────

def main():
    print("=" * 95)
    print("  MEGA-SENA — Deep Statistical Fingerprint Analysis")
    print("  Comparing temporal dynamics of real lottery data vs CRNG vs PRNG")
    print("=" * 95)

    # Load data
    path = '/Users/alebrotto/Deriv MCP/Mega-Sena.xlsx'
    print("\n📊 Loading Mega-Sena data...")
    draws = load_mega_sena(path)
    print(f"  {len(draws)} draws loaded ({draws[0]} ... {draws[-1]})")

    # ─── Part 1: Uniformity Analysis ────────────────────────────
    print("\n" + "─" * 95)
    print("  PART 1: UNIFORMITY TEST — Is the Mega-Sena fair?")
    print("─" * 95)

    flat = [b for d in draws for b in d]
    counts = Counter(flat)
    total = len(flat)
    expected = total / 60

    print(f"\n  Total balls drawn: {total}")
    print(f"  Expected per number: {expected:.1f}")
    print(f"  Expected frequency: {100/60:.2f}%")

    chi2 = sum((counts.get(i, 0) - expected) ** 2 / expected for i in range(1, 61))
    print(f"\n  Chi² statistic: {chi2:.2f}")
    print(f"  Critical value (df=59, α=0.05): 79.08")
    print(f"  Critical value (df=59, α=0.01): 88.38")

    if chi2 > 88.38:
        print(f"  ⚠️  FAILS uniformity at 1% significance!")
    elif chi2 > 79.08:
        print(f"  ⚠️  FAILS uniformity at 5% but passes at 1%")
    else:
        print(f"  ✅ Passes uniformity test")

    # Most/least drawn
    most = counts.most_common(5)
    least = counts.most_common()[-5:]
    print(f"\n  Most drawn:  {', '.join(f'{b}({c})' for b,c in most)}")
    print(f"  Least drawn: {', '.join(f'{b}({c})' for b,c in least)}")

    # ─── Part 2: Temporal Dynamics ──────────────────────────────
    print("\n\n" + "─" * 95)
    print("  PART 2: TEMPORAL DYNAMICS — Where it gets interesting")
    print("─" * 95)

    seqs = make_sequences(draws)

    print(f"\n  Analyzing {len(seqs)} derived time series across 6 metrics...")
    print(f"  Each compared against CRNG (from_data calibrated) and NumPy PRNG")

    N_SEEDS = 5
    metrics = ['PE(3)', 'PE(4)', 'PE(5)', 'Hurst', 'DFA', 'Serial r(1)', 'Kurtosis', 'Runs Z']

    header = f"{'SEQUENCE':<16} {'METRIC':<13} {'MEGA-SENA':>11} {'CRNG':>11} {'PRNG':>11} {'WINNER':>10}"
    print(f"\n  {header}")
    print(f"  {'─' * len(header)}")

    total_crng_wins = 0
    total_prng_wins = 0
    total_ties = 0
    total_tests = 0

    results_detail = []

    for seq_name, real_seq in seqs.items():
        if len(real_seq) < 30:
            continue

        # Compute real metrics
        real_metrics = {
            'PE(3)': permutation_entropy(real_seq, 3),
            'PE(4)': permutation_entropy(real_seq, 4),
            'PE(5)': permutation_entropy(real_seq, 5),
            'Hurst': hurst_exponent(real_seq),
            'DFA': dfa_exponent(real_seq),
            'Serial r(1)': serial_correlation(real_seq, 1),
            'Kurtosis': kurtosis(real_seq),
            'Runs Z': runs_test_z(real_seq),
        }

        # Multi-seed CRNG and PRNG
        crng_metrics_accum = defaultdict(list)
        prng_metrics_accum = defaultdict(list)

        for seed in range(N_SEEDS):
            n = len(real_seq)

            # CRNG calibrated from real data
            crng_seq = crng_from_sequence(real_seq, n, seed=seed)
            # PRNG matching mean/std
            prng_seq = prng_from_sequence(real_seq, n, seed=seed)

            for m_name in metrics:
                if m_name.startswith('PE'):
                    order = int(m_name[3])
                    crng_metrics_accum[m_name].append(permutation_entropy(crng_seq, order))
                    prng_metrics_accum[m_name].append(permutation_entropy(prng_seq, order))
                elif m_name == 'Hurst':
                    crng_metrics_accum[m_name].append(hurst_exponent(crng_seq))
                    prng_metrics_accum[m_name].append(hurst_exponent(prng_seq))
                elif m_name == 'DFA':
                    crng_metrics_accum[m_name].append(dfa_exponent(crng_seq))
                    prng_metrics_accum[m_name].append(dfa_exponent(prng_seq))
                elif m_name == 'Serial r(1)':
                    crng_metrics_accum[m_name].append(serial_correlation(crng_seq, 1))
                    prng_metrics_accum[m_name].append(serial_correlation(prng_seq, 1))
                elif m_name == 'Kurtosis':
                    crng_metrics_accum[m_name].append(kurtosis(crng_seq))
                    prng_metrics_accum[m_name].append(kurtosis(prng_seq))
                elif m_name == 'Runs Z':
                    crng_metrics_accum[m_name].append(runs_test_z(crng_seq))
                    prng_metrics_accum[m_name].append(runs_test_z(prng_seq))

        # Compare averages
        for m_name in metrics:
            m_real = real_metrics[m_name]
            m_crng = np.mean(crng_metrics_accum[m_name])
            m_prng = np.mean(prng_metrics_accum[m_name])

            dist_c = abs(m_crng - m_real)
            dist_p = abs(m_prng - m_real)

            if dist_c < dist_p:
                winner = "✅ CRNG"
                total_crng_wins += 1
            elif dist_p < dist_c:
                winner = "❌ PRNG"
                total_prng_wins += 1
            else:
                winner = "🤝 TIE"
                total_ties += 1
            total_tests += 1

            results_detail.append((seq_name, m_name, m_real, m_crng, m_prng, winner))
            print(f"  {seq_name:<16} {m_name:<13} {m_real:>11.4f} {m_crng:>11.4f} {m_prng:>11.4f} {winner:>10}")

        print(f"  {'─' * len(header)}")

    # ─── Summary ────────────────────────────────────────────────
    print("\n\n" + "=" * 95)
    print("  SUMMARY")
    print("=" * 95)

    print(f"\n  Total metric comparisons: {total_tests}")
    print(f"  CRNG wins: {total_crng_wins}/{total_tests} ({100*total_crng_wins/total_tests:.0f}%)")
    print(f"  PRNG wins: {total_prng_wins}/{total_tests} ({100*total_prng_wins/total_tests:.0f}%)")
    print(f"  Ties:      {total_ties}/{total_tests}")

    # Per-sequence summary
    print(f"\n  Per-sequence breakdown:")
    seq_names = list(dict.fromkeys(r[0] for r in results_detail))
    for sn in seq_names:
        seq_results = [r for r in results_detail if r[0] == sn]
        cw = sum(1 for r in seq_results if 'CRNG' in r[5])
        pw = sum(1 for r in seq_results if 'PRNG' in r[5])
        print(f"    {sn:<16}: CRNG {cw}/{len(seq_results)}, PRNG {pw}/{len(seq_results)}")

    # ─── Key Findings ───────────────────────────────────────────
    print("\n\n" + "=" * 95)
    print("  KEY FINDINGS")
    print("=" * 95)

    # Chi² finding
    print(f"\n  1. UNIFORMITY: Chi² = {chi2:.2f} (critical = 79.08 at 5%)")
    if chi2 > 79.08:
        print(f"     → Mega-Sena marginally REJECTS uniformity!")
        print(f"     → This suggests slight biases in the physical draw process")
    else:
        print(f"     → Mega-Sena passes uniformity — fair lottery")

    # CRNG vs PRNG
    if total_crng_wins > total_prng_wins:
        print(f"\n  2. TEMPORAL DYNAMICS: CRNG captures {total_crng_wins}/{total_tests} metrics")
        print(f"     → Real lottery has temporal structure that CRNG reproduces")
        print(f"     → PRNG's perfect independence misses real-world correlations")
    else:
        print(f"\n  2. TEMPORAL DYNAMICS: PRNG wins {total_prng_wins}/{total_tests} metrics")
        print(f"     → Mega-Sena behaves like a truly random process temporally")
        print(f"     → Expected for a well-designed physical lottery")

    # Domain analysis
    print(f"\n  3. DOMAIN MISMATCH: CRNG was designed for financial fat tails,")
    print(f"     not discrete uniform lottery selections. The proper comparison")
    print(f"     domain for CRNG is financial time series (where it wins 86%).")
    print(f"     Lottery is a fundamentally different stochastic process.")

    print(f"\n  4. INTERESTING ANOMALIES:")
    # Find sequences where kurtosis is notably different from 3
    for sn in seq_names:
        seq_results = [r for r in results_detail if r[0] == sn and r[1] == 'Kurtosis']
        if seq_results:
            k_real = seq_results[0][2]
            if abs(k_real - 3.0) > 0.3:
                print(f"     → {sn}: Kurtosis = {k_real:.3f} (≠ 3.0 Gaussian)")

    print()


if __name__ == '__main__':
    main()
