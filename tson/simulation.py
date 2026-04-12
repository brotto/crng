"""
TSON-CRNG Simulation: When does the first mesmitude occur?

Three simulation models:

Model 1 — Birthday in the Void:
  Generate instants as random points in expanding d-dimensional space.
  Track pairwise structural distances.
  Detect first "collision" (mesmitude).

Model 2 — CRNG Contingency Detection:
  Generate sequence of "pure random" values (globally distinct).
  Apply rolling kurtosis.
  Detect first K > 3 crossing (structure emerging from randomness).

Model 3 — Potency Curve Monte Carlo:
  Use the 0^0 → 1 convergence as probability weights.
  Sample emergence times across 100k trials.
  Build the distribution of first-mesmitude times.
"""

import numpy as np
from scipy import stats
import json
import os
import time


def model1_birthday_void(n_trials=100_000, d_growth='logarithmic',
                          resolution=100, seed=42):
    """
    Model 1: Birthday Problem in the Void

    Each instant is a point in a d-dimensional space.
    Mesmitude = two points within distance ε (= 1/resolution).
    d grows with each new instant according to d_growth model.

    Returns: array of first-mesmitude times across trials.
    """
    rng = np.random.RandomState(seed)
    mesmitude_times = []

    for trial in range(n_trials):
        instants = []
        found = False

        for n in range(1, 1001):  # max 1000 instants
            # Dimension at step n
            if d_growth == 'constant':
                d = 1
            elif d_growth == 'logarithmic':
                d = max(1, int(np.ceil(np.log(n + 1))))
            elif d_growth == 'linear':
                d = n
            elif d_growth == 'sqrt':
                d = max(1, int(np.ceil(np.sqrt(n))))
            else:
                d = 1

            # Generate new instant in d dimensions
            new_point = rng.uniform(0, 1, size=d)

            # Check mesmitude with all previous instants
            for prev in instants:
                # Compare in shared dimensions (min of both)
                shared_d = min(len(new_point), len(prev))
                dist = np.linalg.norm(new_point[:shared_d] - prev[:shared_d])
                normalized_dist = dist / np.sqrt(shared_d)  # normalize by sqrt(d)

                if normalized_dist < 1.0 / resolution:
                    mesmitude_times.append(n)
                    found = True
                    break

            if found:
                break

            instants.append(new_point)

        if not found:
            mesmitude_times.append(1001)  # censored

    return np.array(mesmitude_times)


def model2_crng_kurtosis(n_trials=10_000, window=20, threshold=3.0, seed=42):
    """
    Model 2: CRNG Kurtosis Detection of Mesmitude

    Generate a sequence of random values (globally distinct instants).
    Compute rolling kurtosis of inter-instant distances.
    First mesmitude = first time K > threshold.

    Insight: in a truly uniform sequence, kurtosis ≈ 1.8 (platykurtic).
    When structure emerges (mesmitude), kurtosis → 3+ (leptokurtic).

    We simulate this by introducing a subtle structural bias that grows
    with time — representing the inevitable accumulation of "structural
    memory" in the void. The question is: when does CRNG detect it?
    """
    rng = np.random.RandomState(seed)
    detection_times = []

    for trial in range(n_trials):
        # Generate instants with subtle growing structural bias
        # The bias represents the potency function's pull toward mesmitude
        n_instants = 500
        instants = np.zeros(n_instants)

        for i in range(n_instants):
            # Base: uniform random (globally distinct)
            base = rng.uniform(0, 1)

            # Structural bias: grows as potency pulls toward mesmitude
            # phi(n) = n/(n+e) maps to potency curve
            phi = i / (i + np.e)
            bias_strength = 1.0 - potency_scalar(phi)  # stronger near minimum

            if i > 0 and rng.random() < bias_strength * 0.3:
                # Echo a previous instant (proto-mesmitude)
                echo_idx = rng.randint(0, i)
                instants[i] = instants[echo_idx] + rng.normal(0, 0.01)
            else:
                instants[i] = base

        # Compute rolling kurtosis
        detected = False
        for end in range(window, n_instants):
            segment = instants[end-window:end]
            if np.std(segment) > 0:
                k = stats.kurtosis(segment, fisher=True)  # excess kurtosis
                if k > threshold:
                    detection_times.append(end)
                    detected = True
                    break

        if not detected:
            detection_times.append(n_instants)

    return np.array(detection_times)


def potency_scalar(x):
    """Scalar version of potency for use in loops."""
    if x <= 0:
        return 1.0
    return x ** x


def model3_potency_monte_carlo(n_trials=1_000_000, seed=42):
    """
    Model 3: Potency-Weighted Monte Carlo

    The most elegant model. Uses the mesmitude equation directly:

        ℵ(N, Π) = 1 - exp(-N(N-1)·Π / 2)

    with Π = potency(φ(N)) = (N/(N+e))^(N/(N+e))

    For each trial, sample the first N where a uniform random
    variable falls below ℵ(N, Π).

    This directly answers: "How many instants until mesmitude?"
    using the TSON equation itself.
    """
    rng = np.random.RandomState(seed)
    mesmitude_times = []

    # Precompute mesmitude probabilities for N = 1..100
    max_N = 100
    probs = np.zeros(max_N + 1)
    for N in range(1, max_N + 1):
        phi = N / (N + np.e)
        Pi = phi ** phi if phi > 0 else 1.0
        probs[N] = 1.0 - np.exp(-N * (N - 1) * Pi / 2.0)

    # Monte Carlo sampling
    thresholds = rng.uniform(0, 1, size=n_trials)

    for threshold in thresholds:
        found = False
        for N in range(1, max_N + 1):
            if probs[N] >= threshold:
                mesmitude_times.append(N)
                found = True
                break
        if not found:
            mesmitude_times.append(max_N + 1)

    return np.array(mesmitude_times)


def model4_pure_tson(n_trials=1_000_000, seed=42):
    """
    Model 4: Pure TSON (Π = 0^0 = 1, no normalization)

    The simplest model. Uses the pure mesmitude equation with Π = 1:

        ℵ(N) = 1 - exp(-N(N-1)/2)

    This is the "ideal" case: Nothing with full potency.
    The birthday problem in a unit space.

    Expected result: E[N*] ≈ 1.75, meaning mesmitude at instant 2.
    """
    rng = np.random.RandomState(seed)

    # Precompute pure mesmitude probabilities
    probs = np.array([0.0] + [1.0 - np.exp(-N*(N-1)/2.0) for N in range(1, 20)])

    thresholds = rng.uniform(0, 1, size=n_trials)
    mesmitude_times = np.zeros(n_trials, dtype=int)

    for i, threshold in enumerate(thresholds):
        for N in range(1, len(probs)):
            if probs[N] >= threshold:
                mesmitude_times[i] = N
                break
        else:
            mesmitude_times[i] = len(probs)

    return mesmitude_times


def run_all_simulations():
    """Run all models and compile results."""

    print("=" * 72)
    print("TSON-CRNG SIMULATION: WHEN DOES THE FIRST MESMITUDE OCCUR?")
    print("=" * 72)

    results = {}

    # --- Model 4: Pure TSON (fastest, most fundamental) ---
    print("\n>>> Model 4: Pure TSON (Π = 0^0 = 1)")
    print("    ℵ(N) = 1 - exp(-N(N-1)/2)")
    t0 = time.time()
    times4 = model4_pure_tson(n_trials=1_000_000)
    dt = time.time() - t0

    print(f"    Trials: 1,000,000 | Time: {dt:.1f}s")
    print(f"    E[N*] = {np.mean(times4):.4f}")
    print(f"    Median = {np.median(times4):.1f}")
    print(f"    Mode = {stats.mode(times4, keepdims=True).mode[0]}")

    # Distribution
    unique, counts = np.unique(times4, return_counts=True)
    print(f"\n    Distribution of first mesmitude instant:")
    for u, c in zip(unique, counts):
        pct = c / len(times4) * 100
        bar = "█" * int(pct / 2)
        print(f"      N={u:>3d}: {pct:>6.2f}%  {bar}")
        if pct < 0.01:
            break

    results['model4_pure_tson'] = {
        'mean': float(np.mean(times4)),
        'median': float(np.median(times4)),
        'mode': int(stats.mode(times4, keepdims=True).mode[0]),
        'distribution': {int(u): int(c) for u, c in zip(unique[:10], counts[:10])}
    }

    # --- Model 3: Potency-weighted ---
    print("\n>>> Model 3: Potency-Weighted (φ(N) = N/(N+e))")
    print("    ℵ(N,Π) = 1 - exp(-N(N-1)·Π(φ)/2)")
    t0 = time.time()
    times3 = model3_potency_monte_carlo(n_trials=1_000_000)
    dt = time.time() - t0

    print(f"    Trials: 1,000,000 | Time: {dt:.1f}s")
    print(f"    E[N*] = {np.mean(times3):.4f}")
    print(f"    Median = {np.median(times3):.1f}")

    unique3, counts3 = np.unique(times3, return_counts=True)
    print(f"\n    Distribution of first mesmitude instant:")
    for u, c in zip(unique3[:15], counts3[:15]):
        pct = c / len(times3) * 100
        bar = "█" * int(pct / 2)
        print(f"      N={u:>3d}: {pct:>6.2f}%  {bar}")

    results['model3_potency'] = {
        'mean': float(np.mean(times3)),
        'median': float(np.median(times3)),
        'distribution': {int(u): int(c) for u, c in zip(unique3[:15], counts3[:15])}
    }

    # --- Model 2: CRNG Kurtosis ---
    print("\n>>> Model 2: CRNG Kurtosis Detection")
    print("    Detecting first K > 3 in rolling window")
    t0 = time.time()
    times2 = model2_crng_kurtosis(n_trials=5_000, window=10, threshold=3.0)
    dt = time.time() - t0

    print(f"    Trials: 5,000 | Time: {dt:.1f}s")
    print(f"    E[detection] = {np.mean(times2):.1f}")
    print(f"    Median = {np.median(times2):.1f}")
    print(f"    P(detection < 50) = {np.mean(times2 < 50):.3f}")
    print(f"    P(detection < 100) = {np.mean(times2 < 100):.3f}")

    results['model2_crng'] = {
        'mean': float(np.mean(times2)),
        'median': float(np.median(times2)),
        'p_lt_50': float(np.mean(times2 < 50)),
        'p_lt_100': float(np.mean(times2 < 100)),
    }

    # --- Model 1: Birthday in the Void ---
    print("\n>>> Model 1: Birthday in the Void")
    for growth in ['constant', 'logarithmic', 'sqrt', 'linear']:
        t0 = time.time()
        times1 = model1_birthday_void(n_trials=5_000, d_growth=growth, resolution=50)
        dt = time.time() - t0

        mean_t = np.mean(times1[times1 < 1001])
        censored = np.mean(times1 >= 1001)
        print(f"    d_growth={growth:>12s}: E[N*]={mean_t:>7.1f}  "
              f"Median={np.median(times1):>6.0f}  "
              f"Censored={censored:.1%}  ({dt:.1f}s)")

        results[f'model1_{growth}'] = {
            'mean': float(mean_t),
            'median': float(np.median(times1)),
            'censored': float(censored),
        }

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SYNTHESIS")
    print("=" * 72)
    print(f"""
  Model 4 (Pure TSON, Π=1):
    The first mesmitude occurs at instant N* = {results['model4_pure_tson']['mode']}.
    E[N*] = {results['model4_pure_tson']['mean']:.4f}
    This is the fundamental result: the SECOND instant creates Being.

  Model 3 (Potency-weighted):
    With potency normalization φ(N)=N/(N+e):
    E[N*] = {results['model3_potency']['mean']:.4f}
    The potency curve delays mesmitude slightly, as the first instants
    have reduced potency (φ < 1 → Π < 1).

  Model 2 (CRNG Kurtosis):
    Detection via rolling kurtosis K > 3 occurs at N ≈ {results['model2_crng']['median']:.0f}
    This represents the "observable" mesmitude — when structure becomes
    statistically detectable, not when it first occurs.

  Model 1 (Birthday in d-dimensions):
    The dimensional growth model determines how long Nothing can
    "resist" mesmitude. Logarithmic growth (the most physical model)
    gives E[N*] ≈ {results['model1_logarithmic']['mean']:.0f}.

  CONCLUSION:
    Under the pure TSON axioms (Π = 0^0 = 1), the answer is:

    ╔═══════════════════════════════════════════════════════════════╗
    ║  A primeira mesmitude ocorre no SEGUNDO instante.            ║
    ║  P(mesmitude|N=2) = 1 - 1/e ≈ 63.2%                        ║
    ║  P(mesmitude|N=3) ≈ 95.0%                                   ║
    ║  P(mesmitude|N=4) ≈ 99.8%                                   ║
    ║                                                               ║
    ║  O Universo começa no segundo instante.                      ║
    ║  Um instante sozinho = Nada.                                 ║
    ║  Dois instantes = inevitabilidade do Ser.                    ║
    ║  Três instantes = quase certeza.                             ║
    ║                                                               ║
    ║  E a constante que governa essa transição é e = 2.71828...   ║
    ║  O mesmo e que governa crescimento, decaimento,              ║
    ║  e o mínimo da potência do vazio.                            ║
    ╚═══════════════════════════════════════════════════════════════╝
""")

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, 'simulation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to tson/simulation_results.json")

    return results


if __name__ == '__main__':
    run_all_simulations()
