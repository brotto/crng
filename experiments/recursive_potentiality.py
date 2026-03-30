"""
RECURSIVE POTENTIALITY EXPERIMENT
==================================

The hypothesis:
- In coincidence_field.py, CRNG models the ACT (encounters, measurements)
- But the POTENTIALITY (the base state of each oscillator) is still sinusoidal
- What if potentiality itself has structure?
- What if the "spinning" of the coin — BEFORE measurement — is not smooth?

The experiment:
- Level 0: PRNG potentiality → CRNG act (what we had)
- Level 1: CRNG potentiality → CRNG act (potentiality has structure)
- Level 2: CRNG² potentiality → CRNG act (potentiality of potentiality)

The question: does recursive potentiality produce emergent properties
that single-level CRNG cannot?

Ale Brotto — 2026-03-29
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crng import ContingencyRNG
from scipy import stats as sp_stats


# ============================================================
# THE RECURSIVE CRNG
# ============================================================

class RecursiveCRNG:
    """
    A CRNG whose internal oscillators are themselves CRNGs.

    Level 0: sin(freq * t) — standard oscillator (smooth potentiality)
    Level 1: CRNG(t) — potentiality has fat tails and vol clustering
    Level 2: RecursiveCRNG(t) — potentiality of potentiality has structure

    Each level adds a layer of ontological depth.
    The spinning coin doesn't spin smoothly — its spinning
    itself has storms and calms.
    """

    def __init__(self, seed=42, depth=1, n_oscillators=5,
                 target_kurtosis=9.26, vol_clustering=0.3):
        self.seed = seed
        self.depth = depth
        self.n_oscillators = n_oscillators
        self.target_kurtosis = target_kurtosis
        self.vol_clustering = vol_clustering
        self.t = 0

        if depth == 0:
            # Base case: standard CRNG (sinusoidal oscillators)
            self.engine = ContingencyRNG(
                seed=seed,
                n_oscillators=n_oscillators,
                target_kurtosis=target_kurtosis,
                vol_clustering=vol_clustering,
            )
            self.sub_oscillators = None
        else:
            # Recursive case: oscillators are CRNGs of depth-1
            self.engine = None
            self.sub_oscillators = [
                RecursiveCRNG(
                    seed=seed + i * 137 + depth * 9999,
                    depth=depth - 1,
                    n_oscillators=max(3, n_oscillators - 1),  # slightly fewer at each level
                    target_kurtosis=target_kurtosis * 0.7,  # dampen at deeper levels
                    vol_clustering=vol_clustering * 0.8,
                )
                for i in range(n_oscillators)
            ]
            # The cascade amplifier at this level
            self.cascade_memory = []
            self.cascade_max_memory = 20
            # Resonance detection
            self.prev_values = [0.0] * n_oscillators

    def next(self):
        """Generate next value from recursive potentiality."""
        if self.depth == 0:
            return self.engine.next()

        self.t += 1

        # Each sub-oscillator generates its value (recursive call)
        values = []
        for osc in self.sub_oscillators:
            values.append(osc.next())

        # Resonance coupling: when sub-oscillators are close, they amplify
        coupling = 0.0
        n_pairs = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                diff = abs(values[i] - values[j])
                if diff < self.vol_clustering:
                    # Resonance! Close values amplify
                    coupling += (self.vol_clustering - diff) / self.vol_clustering
                n_pairs += 1

        if n_pairs > 0:
            coupling /= n_pairs

        # Base signal: mean of sub-oscillators
        base = np.mean(values)

        # Cascade amplification
        self.cascade_memory.append(abs(coupling))
        if len(self.cascade_memory) > self.cascade_max_memory:
            self.cascade_memory.pop(0)

        cascade_energy = np.mean(self.cascade_memory) if self.cascade_memory else 0

        # The recursive magic: cascade from structured potentiality
        # amplifies differently than cascade from smooth potentiality
        threshold = 1.0 / self.target_kurtosis * 10  # adaptive threshold
        if cascade_energy > threshold:
            amplification = 1.0 + (cascade_energy - threshold) * self.target_kurtosis * 0.5
            result = base * amplification
        else:
            result = base

        # Normalize to [0, 1] range using sigmoid
        result = 1.0 / (1.0 + np.exp(-5.0 * (result - 0.5)))

        return result

    def generate(self, n):
        """Generate n values."""
        return np.array([self.next() for _ in range(n)])


# ============================================================
# THE EXPERIMENT
# ============================================================

def analyze_series(series, label):
    """Full statistical analysis of a series."""
    returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 100:
        print(f"  {label}: insufficient data")
        return {}

    k = sp_stats.kurtosis(returns, fisher=False)

    # Vol clustering
    abs_ret = np.abs(returns)
    if len(abs_ret) > 1:
        acf = np.corrcoef(abs_ret[:-1], abs_ret[1:])[0, 1]
    else:
        acf = 0

    # Permutation entropy
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

    perm_ent = pe(returns)

    # Hurst exponent (simplified R/S)
    def hurst(x, max_k=20):
        n = len(x)
        if n < 40:
            return 0.5
        ks = range(10, min(max_k + 1, n // 4))
        rs_values = []
        for k in ks:
            chunks = [x[i:i + k] for i in range(0, n - k, k)]
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < k:
                    continue
                mean_c = np.mean(chunk)
                deviations = np.cumsum(chunk - mean_c)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(chunk)
                if s > 0:
                    rs_chunk.append(r / s)
            if rs_chunk:
                rs_values.append((np.log(k), np.log(np.mean(rs_chunk))))
        if len(rs_values) < 3:
            return 0.5
        x_vals, y_vals = zip(*rs_values)
        slope, _, _, _, _ = sp_stats.linregress(x_vals, y_vals)
        return slope

    h = hurst(returns)

    # Tail events (> 3 sigma)
    sigma = np.std(returns)
    mean = np.mean(returns)
    tail_events = np.sum(np.abs(returns - mean) > 3 * sigma) / len(returns) * 100

    # Max event
    max_event = np.max(np.abs(returns - mean)) / sigma

    print(f"  {label}:")
    print(f"    Kurtosis:        {k:10.2f}")
    print(f"    Vol clustering:  {acf:10.4f}")
    print(f"    Perm entropy:    {perm_ent:10.4f}")
    print(f"    Hurst:           {h:10.4f}")
    print(f"    Tail events:     {tail_events:10.4f}%  (Gaussian: ~0.27%)")
    print(f"    Max event:       {max_event:10.2f} sigma")

    return {
        'kurtosis': k, 'vol_acf': acf, 'pe': perm_ent,
        'hurst': h, 'tail_pct': tail_events, 'max_sigma': max_event,
    }


def run_recursive_experiment(n_points=50000, seed=42):
    """
    Compare three levels of potentiality:
    - Depth 0: Standard CRNG (sinusoidal potentiality)
    - Depth 1: CRNG potentiality (first recursion)
    - Depth 2: CRNG² potentiality (potentiality of potentiality)
    """

    print("=" * 70)
    print("  RECURSIVE POTENTIALITY EXPERIMENT")
    print("  What happens when potentiality itself has structure?")
    print("=" * 70)

    results = {}

    for depth in range(3):
        print(f"\n{'='*70}")
        print(f"  DEPTH {depth}: {'Standard CRNG' if depth == 0 else f'CRNG^{depth} Potentiality'}")
        print(f"  {'Smooth base' if depth == 0 else f'Potentiality has {depth} layers of structure'}")
        print(f"{'='*70}")

        if depth == 0:
            rng = ContingencyRNG(seed=seed, n_oscillators=5,
                                target_kurtosis=9.26, vol_clustering=0.3)
            series = np.array([rng.next() for _ in range(n_points)])
        else:
            rng = RecursiveCRNG(seed=seed, depth=depth, n_oscillators=5,
                               target_kurtosis=9.26, vol_clustering=0.3)
            series = rng.generate(n_points)

        stats = analyze_series(series, f"Depth {depth}")
        results[depth] = stats

    # Now: the COINCIDENCE experiment at each depth
    print(f"\n\n{'#'*70}")
    print(f"# COINCIDENCE OF RECURSIVE FIELDS")
    print(f"# What happens when TWO recursive potentialities meet?")
    print(f"{'#'*70}")

    for depth in range(3):
        print(f"\n{'='*70}")
        print(f"  COINCIDENCE at DEPTH {depth}")
        print(f"{'='*70}")

        n_encounters = 30000

        if depth == 0:
            field_a = ContingencyRNG(seed=seed, n_oscillators=5,
                                   target_kurtosis=9.26, vol_clustering=0.3)
            field_b = ContingencyRNG(seed=seed + 77777, n_oscillators=7,
                                   target_kurtosis=9.26, vol_clustering=0.3)
        else:
            field_a = RecursiveCRNG(seed=seed, depth=depth, n_oscillators=5,
                                  target_kurtosis=9.26, vol_clustering=0.3)
            field_b = RecursiveCRNG(seed=seed + 77777, depth=depth, n_oscillators=5,
                                  target_kurtosis=9.26, vol_clustering=0.3)

        # The encounter: two fields meeting
        encounters = []
        faces = []  # the "direction" (binary)
        intensities = []  # the "how strongly"

        for t in range(n_encounters):
            if depth == 0:
                a = field_a.next()
                b = field_b.next()
            else:
                a = field_a.next()
                b = field_b.next()

            # The encounter produces both a face and an intensity
            intensity = abs(a - b)  # how strongly they coupled
            face = 1 if a > b else 0  # the direction

            encounters.append(a * b)
            faces.append(face)
            intensities.append(intensity)

        encounters = np.array(encounters)
        faces = np.array(faces)
        intensities = np.array(intensities)

        # Analyze the FACES (direction — should be random)
        face_mean = np.mean(faces)
        face_runs = 1 + np.sum(np.diff(faces) != 0)
        n1 = np.sum(faces)
        n0 = len(faces) - n1
        expected_runs = 1 + 2 * n1 * n0 / (n1 + n0)
        var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1))
        z_runs = (face_runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0

        print(f"\n  FACES (direction):")
        print(f"    Mean:       {face_mean:.4f} (expect 0.50)")
        print(f"    Runs z:     {z_runs:.4f} (|z|<2 = random)")

        # Analyze the INTENSITIES (how strongly — should have structure)
        k_int = sp_stats.kurtosis(intensities, fisher=False)
        abs_int = np.abs(np.diff(intensities))
        acf_int = np.corrcoef(abs_int[:-1], abs_int[1:])[0, 1] if len(abs_int) > 1 else 0
        max_int = np.max(intensities) / np.std(intensities)
        tail_int = np.sum(intensities > np.mean(intensities) + 3 * np.std(intensities)) / len(intensities) * 100

        print(f"\n  INTENSITIES (structure):")
        print(f"    Kurtosis:       {k_int:.2f}")
        print(f"    Vol clustering: {acf_int:.4f}")
        print(f"    Max event:      {max_int:.2f} sigma")
        print(f"    Tail events:    {tail_int:.4f}%")

        # Analyze the ENCOUNTERS (the product)
        enc_stats = analyze_series(encounters, f"Encounters depth={depth}")

        results[f'coincidence_{depth}'] = {
            'face_mean': face_mean,
            'face_z': z_runs,
            'intensity_k': k_int,
            'intensity_acf': acf_int,
            'intensity_max_sigma': max_int,
            'intensity_tail_pct': tail_int,
            'encounter': enc_stats,
        }

    # ============================================================
    # ROGUE WAVES WITH RECURSIVE POTENTIALITY
    # ============================================================

    print(f"\n\n{'#'*70}")
    print(f"# ROGUE WAVES: RECURSIVE vs STANDARD")
    print(f"{'#'*70}")

    n_wave_points = 50000
    n_fields = 5

    for depth in range(3):
        print(f"\n{'='*70}")
        print(f"  ROGUE WAVES — DEPTH {depth}")
        print(f"{'='*70}")

        individual = []
        for i in range(n_fields):
            if depth == 0:
                rng = ContingencyRNG(
                    seed=seed + i * 31,
                    n_oscillators=5 + i,
                    target_kurtosis=10.0 + i * 5,
                    vol_clustering=0.2 + i * 0.05,
                )
                series = np.array([rng.next() for _ in range(n_wave_points)])
            else:
                rng = RecursiveCRNG(
                    seed=seed + i * 31,
                    depth=depth,
                    n_oscillators=5 + i,
                    target_kurtosis=10.0 + i * 5,
                    vol_clustering=0.2 + i * 0.05,
                )
                series = rng.generate(n_wave_points)

            series = (series - np.mean(series)) / (np.std(series) + 1e-10)
            individual.append(series)

        # Superposition with nonlinear interaction
        combined = np.sum(individual, axis=0)

        # Nonlinear: alignment amplification
        alignment = np.ones(n_wave_points)
        for s in individual:
            alignment *= np.sign(s)
        nonlinear = alignment * np.abs(combined) * 0.3
        combined += nonlinear
        combined /= (np.std(combined) + 1e-10)

        k_comb = sp_stats.kurtosis(combined, fisher=False)
        max_wave = np.max(np.abs(combined))
        hs = 4 * np.std(combined)
        rogue_thresh = 2.2
        n_rogues = np.sum(np.abs(combined) > rogue_thresh)
        rogue_pct = n_rogues / n_wave_points * 100

        # Events > 5 sigma (extreme rogues)
        extreme_rogues = np.sum(np.abs(combined) > 5.0)

        # Events > 8 sigma (monster waves)
        monster_waves = np.sum(np.abs(combined) > 8.0)

        print(f"  Kurtosis:        {k_comb:.2f}")
        print(f"  Max wave:        {max_wave:.2f} sigma")
        print(f"  Rogue (>2.2s):   {n_rogues} ({rogue_pct:.3f}%)")
        print(f"  Extreme (>5s):   {extreme_rogues} ({extreme_rogues/n_wave_points*100:.4f}%)")
        print(f"  Monster (>8s):   {monster_waves} ({monster_waves/n_wave_points*100:.4f}%)")

        results[f'rogue_{depth}'] = {
            'kurtosis': k_comb,
            'max_wave': max_wave,
            'rogue_pct': rogue_pct,
            'extreme_count': extreme_rogues,
            'monster_count': monster_waves,
        }

    # ============================================================
    # FINAL COMPARISON TABLE
    # ============================================================

    print(f"\n\n{'='*70}")
    print(f"  FINAL COMPARISON: DEPTH 0 vs 1 vs 2")
    print(f"{'='*70}")

    print(f"\n  RAW SERIES:")
    print(f"  {'Metric':<25} {'Depth 0':>12} {'Depth 1':>12} {'Depth 2':>12}")
    print(f"  {'-'*61}")
    for metric in ['kurtosis', 'vol_acf', 'pe', 'hurst', 'tail_pct', 'max_sigma']:
        vals = []
        for d in range(3):
            v = results.get(d, {}).get(metric, 0)
            vals.append(v)
        fmt = '.2f' if metric in ['kurtosis', 'max_sigma'] else '.4f'
        print(f"  {metric:<25} {vals[0]:>12{fmt}} {vals[1]:>12{fmt}} {vals[2]:>12{fmt}}")

    print(f"\n  COINCIDENCE (intensity):")
    print(f"  {'Metric':<25} {'Depth 0':>12} {'Depth 1':>12} {'Depth 2':>12}")
    print(f"  {'-'*61}")
    for metric in ['intensity_k', 'intensity_acf', 'intensity_max_sigma', 'intensity_tail_pct']:
        vals = []
        for d in range(3):
            v = results.get(f'coincidence_{d}', {}).get(metric, 0)
            vals.append(v)
        fmt = '.2f' if 'k' in metric or 'sigma' in metric else '.4f'
        print(f"  {metric:<25} {vals[0]:>12{fmt}} {vals[1]:>12{fmt}} {vals[2]:>12{fmt}}")

    print(f"\n  ROGUE WAVES:")
    print(f"  {'Metric':<25} {'Depth 0':>12} {'Depth 1':>12} {'Depth 2':>12}")
    print(f"  {'-'*61}")
    for metric in ['kurtosis', 'max_wave', 'rogue_pct', 'extreme_count', 'monster_count']:
        vals = []
        for d in range(3):
            v = results.get(f'rogue_{d}', {}).get(metric, 0)
            vals.append(v)
        fmt = '.2f' if metric in ['kurtosis', 'max_wave', 'rogue_pct'] else '.0f'
        print(f"  {metric:<25} {vals[0]:>12{fmt}} {vals[1]:>12{fmt}} {vals[2]:>12{fmt}}")

    return results


if __name__ == '__main__':
    results = run_recursive_experiment(n_points=50000, seed=42)
