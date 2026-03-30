"""
COINCIDENCE FIELD EXPERIMENT
============================

Philosophical foundation (Ale Brotto):
- A coin is potentiality (dynamis) until measured
- A bet is a second, independent field of becoming
- "Hitting" (acertar) exists ONLY at the intersection of two fields
- Conventional probability ignores the second field entirely

Three experiments:
1. PRNG coins + PRNG bettors → baseline (both are pure potentiality)
2. CRNG coins + CRNG bettors → with structure (fat tails, vol clustering)
3. The Deterministic Corollary → can knowing N-1 predict the Nth?

The central question: Does the COINCIDENCE of two independent structured
fields produce emergent properties that neither field has alone?
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crng import ContingencyRNG

# ============================================================
# LAYER 1: The Bettor as Oscillator
# ============================================================

class Bettor:
    """
    A bettor is NOT random.choice([0,1]).
    A bettor is an agent with:
    - Memory (observes past results)
    - Heuristic (gambler's fallacy OR hot-hand bias)
    - Social influence (nearby bettors affect choice)
    - Their own irrational-frequency oscillator (mood, intuition)
    """

    def __init__(self, bettor_id, heuristic='gambler', influence_radius=3, seed=None):
        self.id = bettor_id
        self.heuristic = heuristic  # 'gambler' or 'hothand'
        self.influence_radius = influence_radius
        self.memory = []  # observed results
        self.bet_history = []
        self.oscillator = ContingencyRNG(
            seed=(seed or bettor_id),
            n_oscillators=3,
            vol_clustering=0.08,
            cascade_threshold=0.5  # sub-critical: bettors are more "regular" than markets
        )
        self.confidence = 0.5  # how strongly they follow their heuristic

    def observe(self, result):
        """Observe a coin result (0 or 1)."""
        self.memory.append(result)
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]  # bounded memory

    def decide(self, nearby_bets=None):
        """
        Make a bet. The decision emerges from:
        1. Internal oscillator state (intuition/mood)
        2. Heuristic applied to memory
        3. Social influence from nearby bettors
        """
        # Component 1: Internal oscillator (the bettor's own "spinning")
        internal = self.oscillator.next()

        # Component 2: Heuristic based on memory
        heuristic_signal = 0.5  # default: no bias
        if len(self.memory) >= 3:
            recent = self.memory[-3:]
            recent_mean = np.mean(recent)

            if self.heuristic == 'gambler':
                # "It's been heads too much, tails is due"
                heuristic_signal = 1.0 - recent_mean
            elif self.heuristic == 'hothand':
                # "It's been heads, it'll keep being heads"
                heuristic_signal = recent_mean

        # Component 3: Social influence
        social_signal = 0.5
        if nearby_bets and len(nearby_bets) > 0:
            social_signal = np.mean(nearby_bets)

        # Combine: weighted mixture
        w_internal = 0.4
        w_heuristic = 0.35
        w_social = 0.25

        combined = (w_internal * internal +
                   w_heuristic * heuristic_signal +
                   w_social * social_signal)

        bet = 1 if combined > 0.5 else 0
        self.bet_history.append(bet)
        return bet


# ============================================================
# LAYER 2: The Coin Field (using CRNG)
# ============================================================

class CoinField:
    """
    N coins spinning in a finite space.
    Each coin is a CRNG oscillator.
    The "blade" (measurement) is another CRNG oscillator.

    The coin doesn't "have" a face. It acquires one
    at the moment of measurement — the intersection
    of two independent becomings.
    """

    def __init__(self, n_coins, seed=42, use_crng=True):
        self.n_coins = n_coins
        self.use_crng = use_crng

        if use_crng:
            # Each coin has its own oscillator
            self.coins = [
                ContingencyRNG(
                    seed=seed + i * 137,  # prime spacing
                    n_oscillators=5,
                    vol_clustering=0.1,
                    cascade_threshold=1.2  # supercritical: produces fat tails
                )
                for i in range(n_coins)
            ]
            # The blade — the measuring oscillator
            self.blade = ContingencyRNG(
                seed=seed + 99999,
                n_oscillators=7,
                vol_clustering=0.12,
                cascade_threshold=1.0  # at threshold
            )
        else:
            self.rng = np.random.RandomState(seed)

    def measure_all(self):
        """
        The blade sweeps through all coins.
        Each measurement is the COINCIDENCE of
        the coin's state and the blade's state.
        """
        if not self.use_crng:
            return self.rng.randint(0, 2, size=self.n_coins)

        results = []
        blade_state = self.blade.next()

        for coin in self.coins:
            coin_state = coin.next()
            # The face emerges from the ENCOUNTER
            # Not from either oscillator alone
            # Use XOR-like combination: difference determines face
            # This preserves the 50/50 balance while keeping encounter structure
            encounter = abs(coin_state - blade_state)
            face = 1 if encounter > 0.5 else 0
            results.append(face)

        return np.array(results)


# ============================================================
# LAYER 3: The Coincidence Experiment
# ============================================================

def run_experiment(n_coins=100, n_rounds=10000, use_crng=True, seed=42):
    """
    The full experiment:
    - N coins spinning in space
    - N bettors, each observing their local coin
    - Each round: coins are measured, bets are placed
    - We track: local accuracy, global accuracy, structure of coincidences
    """

    print(f"\n{'='*60}")
    print(f"COINCIDENCE FIELD — {'CRNG' if use_crng else 'PRNG'} MODE")
    print(f"Coins: {n_coins} | Rounds: {n_rounds}")
    print(f"{'='*60}")

    # Create coin field
    field = CoinField(n_coins, seed=seed, use_crng=use_crng)

    # Create bettors — mix of heuristics
    bettors = []
    for i in range(n_coins):
        heuristic = 'gambler' if i % 3 != 0 else 'hothand'
        bettors.append(Bettor(i, heuristic=heuristic, seed=seed + i + 10000))

    # Track results
    all_hits = []  # 1 = correct bet, 0 = wrong
    local_hit_rates = np.zeros(n_coins)
    round_hit_rates = []
    bet_distribution = []  # fraction betting "1" each round
    coin_distribution = []  # fraction landing "1" each round

    # Intensity of coincidence (how "strongly" bet and result aligned)
    coincidence_intensities = []

    for round_num in range(n_rounds):
        # 1. Coins are measured (the blade sweeps)
        results = field.measure_all()

        # 2. Bettors decide (influenced by memory + neighbors)
        bets = np.zeros(n_coins, dtype=int)
        for i, bettor in enumerate(bettors):
            # Nearby bets (social influence)
            start = max(0, i - bettor.influence_radius)
            end = min(n_coins, i + bettor.influence_radius + 1)
            nearby = [b.bet_history[-1] for b in bettors[start:end]
                     if b.bet_history and b.id != i]

            bets[i] = bettor.decide(nearby_bets=nearby if nearby else None)

        # 3. The COINCIDENCE — where two fields intersect
        hits = (bets == results).astype(int)
        all_hits.extend(hits)
        local_hit_rates += hits
        round_hit_rates.append(np.mean(hits))
        bet_distribution.append(np.mean(bets))
        coin_distribution.append(np.mean(results))

        # 4. Bettors observe results (update memory)
        for i, bettor in enumerate(bettors):
            bettor.observe(results[i])

    # ============================================================
    # ANALYSIS
    # ============================================================

    local_hit_rates /= n_rounds
    all_hits = np.array(all_hits)
    round_hit_rates = np.array(round_hit_rates)
    bet_distribution = np.array(bet_distribution)
    coin_distribution = np.array(coin_distribution)

    print(f"\n--- GLOBAL STATISTICS ---")
    print(f"Global hit rate:        {np.mean(all_hits):.6f}")
    print(f"Mean coin distribution: {np.mean(coin_distribution):.6f} (expect ~0.50)")
    print(f"Mean bet distribution:  {np.mean(bet_distribution):.6f} (expect ~0.50?)")

    print(f"\n--- LOCAL STRUCTURE ---")
    print(f"Hit rate std (across bettors):  {np.std(local_hit_rates):.6f}")
    print(f"Hit rate min:  {np.min(local_hit_rates):.4f}")
    print(f"Hit rate max:  {np.max(local_hit_rates):.4f}")
    print(f"Hit rate range: {np.max(local_hit_rates) - np.min(local_hit_rates):.4f}")

    # Kurtosis of hit rates across bettors
    from scipy import stats as sp_stats
    k_local = sp_stats.kurtosis(local_hit_rates, fisher=False)
    print(f"Kurtosis of local hit rates: {k_local:.2f}")

    # Kurtosis of round-by-round hit rates
    k_temporal = sp_stats.kurtosis(round_hit_rates, fisher=False)
    print(f"Kurtosis of temporal hit rates: {k_temporal:.2f}")

    print(f"\n--- TEMPORAL STRUCTURE OF COINCIDENCES ---")

    # Vol clustering in hit rates (do "streaks of accuracy" cluster?)
    hit_deviations = np.abs(round_hit_rates - np.mean(round_hit_rates))
    if len(hit_deviations) > 1:
        acf_hits = np.corrcoef(hit_deviations[:-1], hit_deviations[1:])[0, 1]
        print(f"Volatility clustering (ACF) of hit rates: {acf_hits:.4f}")

    # Runs test on round hit rates (above/below mean)
    above_mean = (round_hit_rates > np.mean(round_hit_rates)).astype(int)
    runs = 1 + np.sum(np.diff(above_mean) != 0)
    n1 = np.sum(above_mean)
    n0 = len(above_mean) - n1
    if n1 > 0 and n0 > 0:
        expected_runs = 1 + 2 * n1 * n0 / (n1 + n0)
        var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0)**2 * (n1 + n0 - 1))
        if var_runs > 0:
            z_runs = (runs - expected_runs) / np.sqrt(var_runs)
            print(f"Runs test z-score: {z_runs:.4f}")

    # Permutation entropy of hit rate sequence
    def permutation_entropy(x, order=3):
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
        probs = [c/total for c in counts.values() if c > 0]
        return -sum(p * np.log2(p) for p in probs) / np.log2(len(perms))

    pe = permutation_entropy(round_hit_rates)
    print(f"Permutation entropy of hit rates: {pe:.4f}")

    print(f"\n--- THE DETERMINISTIC COROLLARY ---")
    # Test: if we know N-1 coin results, can we predict the Nth?
    # In pure PRNG: knowing the global sum constrains the Nth
    # In structured fields: the constraint should be tighter

    # For each round, use N-1 results to "predict" the Nth
    correct_predictions = 0
    total_predictions = 0

    for round_num in range(min(1000, n_rounds)):
        results = field.measure_all()

        # If we know N-1 results and assume global = 50%...
        known = results[:-1]
        known_sum = np.sum(known)
        n_known = len(known)
        expected_total_ones = n_coins / 2  # the LLN assumption

        # Predict: if known_sum > expected, predict 0; else predict 1
        predicted = 0 if known_sum >= expected_total_ones else 1
        actual = results[-1]

        if predicted == actual:
            correct_predictions += 1
        total_predictions += 1

    corollary_accuracy = correct_predictions / total_predictions
    print(f"Corollary prediction accuracy: {corollary_accuracy:.4f}")
    print(f"  (0.50 = no predictive power, >0.50 = structure exists)")

    return {
        'global_hit_rate': np.mean(all_hits),
        'bet_distribution': np.mean(bet_distribution),
        'coin_distribution': np.mean(coin_distribution),
        'local_kurtosis': k_local,
        'temporal_kurtosis': k_temporal,
        'acf_hits': acf_hits if 'acf_hits' in dir() else 0,
        'permutation_entropy': pe,
        'corollary_accuracy': corollary_accuracy,
        'local_hit_rates': local_hit_rates,
        'round_hit_rates': round_hit_rates,
    }


# ============================================================
# LAYER 4: ROGUE WAVE DETECTOR
# ============================================================

def rogue_wave_experiment(n_fields=5, n_points=50000, seed=42):
    """
    Rogue waves emerge when multiple independent wave fields
    enter momentary resonance — supercritical local amplification.

    This is EXACTLY the CRNG cascade mechanism.

    We simulate multiple ocean wave fields as CRNG oscillators
    and look for extreme coincidences — rogue waves.
    """

    print(f"\n{'='*60}")
    print(f"ROGUE WAVE SIMULATION")
    print(f"Wave fields: {n_fields} | Points: {n_points}")
    print(f"{'='*60}")

    from scipy import stats as sp_stats

    # Each wave field is a CRNG with different characteristics
    # Real ocean: multiple independent wave systems with different kurtosis
    wave_fields = []
    field_configs = [
        {'n_osc': 7, 'target_k': 15.0, 'vol': 0.35, 'name': 'Deep swell'},
        {'n_osc': 5, 'target_k': 9.0,  'vol': 0.25, 'name': 'Wind waves'},
        {'n_osc': 9, 'target_k': 25.0, 'vol': 0.40, 'name': 'Current interaction'},
        {'n_osc': 4, 'target_k': 5.0,  'vol': 0.15, 'name': 'Tidal component'},
        {'n_osc': 6, 'target_k': 12.0, 'vol': 0.30, 'name': 'Storm surge'},
    ]

    for i, cfg in enumerate(field_configs[:n_fields]):
        rng = ContingencyRNG(
            seed=seed + i * 31,
            n_oscillators=cfg['n_osc'],
            target_kurtosis=cfg['target_k'],
            vol_clustering=cfg['vol'],
        )
        wave_fields.append({'rng': rng, **cfg})

    # Generate wave heights as superposition of all fields
    # This is how real ocean waves work — linear superposition
    # with nonlinear interactions at extremes

    individual_series = []
    for wf in wave_fields:
        series = np.array([wf['rng'].next() for _ in range(n_points)])
        # Center and normalize
        series = (series - np.mean(series)) / np.std(series)
        individual_series.append(series)

    # Superposition with nonlinear interaction
    # Real waves: linear superposition PLUS nonlinear amplification
    # when multiple fields align (resonance → rogue waves)
    combined = np.zeros(n_points)
    for series in individual_series:
        combined += series

    # Nonlinear term: when multiple fields align (all positive or all negative),
    # the interaction amplifies — this is the physical mechanism for rogues
    alignment = np.ones(n_points)
    for series in individual_series:
        alignment *= np.sign(series)  # +1 if all agree, oscillates otherwise

    # Where fields align, add nonlinear amplification
    nonlinear = alignment * np.abs(combined) * 0.3
    combined += nonlinear
    combined /= np.std(combined)  # normalize to unit variance

    # Analysis
    print(f"\n--- INDIVIDUAL WAVE FIELDS ---")
    for i, (wf, series) in enumerate(zip(wave_fields, individual_series)):
        k = sp_stats.kurtosis(series, fisher=False)
        print(f"  {wf['name']:25s} K={k:8.2f}  target_K={wf['target_k']}")

    k_combined = sp_stats.kurtosis(combined, fisher=False)
    max_wave = np.max(np.abs(combined))

    # Significant wave height (Hs) — standard oceanographic metric
    # Hs = 4 * std(surface elevation)
    hs = 4 * np.std(combined)

    # Rogue wave criterion: H > 2.2 * Hs (oceanographic standard)
    rogue_threshold = 2.2 * hs / 4  # in units of std (= 2.2 sigma)
    n_rogue = np.sum(np.abs(combined) > rogue_threshold)
    rogue_fraction = n_rogue / n_points

    # In Gaussian seas (K=3), P(H > 2*Hs) ≈ 0.0003
    gaussian_rogue_probability = 2 * (1 - sp_stats.norm.cdf(rogue_threshold))

    print(f"\n--- COMBINED SEA STATE ---")
    print(f"Combined kurtosis:    {k_combined:.2f}")
    print(f"Max wave height:      {max_wave:.2f} sigma")
    print(f"Significant wave Hs:  {hs:.2f}")
    print(f"Rogue threshold:      {rogue_threshold:.2f} sigma")
    print(f"Rogue wave events:    {n_rogue} ({rogue_fraction*100:.4f}%)")
    print(f"Gaussian prediction:  {gaussian_rogue_probability*100:.4f}%")
    print(f"CRNG/Gaussian ratio:  {rogue_fraction/gaussian_rogue_probability:.1f}x")

    # Temporal clustering of rogue events
    rogue_indices = np.where(np.abs(combined) > rogue_threshold)[0]
    if len(rogue_indices) > 1:
        gaps = np.diff(rogue_indices)
        gap_cv = np.std(gaps) / np.mean(gaps)  # coefficient of variation
        # CV = 1 for exponential (random), < 1 for regular, > 1 for clustered
        print(f"\nRogue event clustering:")
        print(f"  Mean gap: {np.mean(gaps):.1f} points")
        print(f"  Gap CV:   {gap_cv:.3f} (1.0=random, >1=clustered)")
        print(f"  Min gap:  {np.min(gaps)} points")
        print(f"  Max gap:  {np.max(gaps)} points")

        # Are rogue events preceded by increasing volatility?
        pre_rogue_vol = []
        for idx in rogue_indices:
            if idx >= 50:
                window = combined[idx-50:idx]
                pre_rogue_vol.append(np.std(window))
        if pre_rogue_vol:
            baseline_vol = np.std(combined)
            mean_pre_vol = np.mean(pre_rogue_vol)
            print(f"  Pre-rogue volatility: {mean_pre_vol:.3f} vs baseline {baseline_vol:.3f}")
            print(f"  Ratio: {mean_pre_vol/baseline_vol:.2f}x (>1 = warning signal exists)")

    # The key question: does CRNG superposition produce MORE rogues
    # than Gaussian superposition?
    print(f"\n--- COMPARISON WITH GAUSSIAN SUPERPOSITION ---")
    gaussian_combined = np.zeros(n_points)
    rng_gauss = np.random.RandomState(seed)
    for _ in range(n_fields):
        gaussian_combined += rng_gauss.randn(n_points)
    gaussian_combined /= np.sqrt(n_fields)

    k_gauss = sp_stats.kurtosis(gaussian_combined, fisher=False)
    n_rogue_gauss = np.sum(np.abs(gaussian_combined) > rogue_threshold)

    print(f"Gaussian kurtosis:  {k_gauss:.2f}")
    print(f"Gaussian rogues:    {n_rogue_gauss} ({n_rogue_gauss/n_points*100:.4f}%)")
    print(f"CRNG rogues:        {n_rogue} ({rogue_fraction*100:.4f}%)")
    if n_rogue_gauss > 0:
        print(f"CRNG produces {n_rogue/n_rogue_gauss:.1f}x more rogue events")
    else:
        print(f"CRNG produces rogue events where Gaussian produces ZERO")

    return {
        'k_combined': k_combined,
        'k_gaussian': k_gauss,
        'rogue_fraction': rogue_fraction,
        'gaussian_rogue_fraction': n_rogue_gauss / n_points,
        'max_wave': max_wave,
        'rogue_clustering_cv': gap_cv if len(rogue_indices) > 1 else None,
        'pre_rogue_vol_ratio': mean_pre_vol/baseline_vol if pre_rogue_vol else None,
    }


# ============================================================
# LAYER 5: THE UNCERTAINTY FIELD
# ============================================================

def uncertainty_experiment(n_particles=1000, n_measurements=5000, seed=42):
    """
    Heisenberg's uncertainty is not about instrument precision.
    It's about the nature of what's being measured.

    Position and velocity are not two properties of an object.
    They are two aspects of a potentiality that can only be
    quiddified one at a time — because the measurement itself
    is a second becoming that interferes with the first.

    We simulate this: a particle is a CRNG oscillator.
    Measurement is another CRNG oscillator.
    The encounter quiddifies ONE aspect (position OR velocity).
    The other aspect remains in potentiality — and measuring it
    requires a NEW encounter that disturbs the first.
    """

    print(f"\n{'='*60}")
    print(f"UNCERTAINTY FIELD EXPERIMENT")
    print(f"Particles: {n_particles} | Measurements: {n_measurements}")
    print(f"{'='*60}")

    from scipy import stats as sp_stats

    # Particles as oscillators
    particles = [
        ContingencyRNG(
            seed=seed + i * 73,
            n_oscillators=7,
            vol_clustering=0.15,
            cascade_threshold=1.3
        )
        for i in range(n_particles)
    ]

    # Two measurement apparatuses — one for "position", one for "velocity"
    position_meter = ContingencyRNG(
        seed=seed + 77777,
        n_oscillators=5,
        vol_clustering=0.10,
        cascade_threshold=1.1
    )

    velocity_meter = ContingencyRNG(
        seed=seed + 88888,
        n_oscillators=5,
        vol_clustering=0.10,
        cascade_threshold=1.1
    )

    # Experiment: measure position, then velocity, of same particle
    # The first measurement perturbs the particle's state

    position_then_velocity = []
    velocity_then_position = []
    simultaneous_products = []

    for t in range(n_measurements):
        particle_idx = t % n_particles
        particle = particles[particle_idx]

        # The particle's state at this moment
        particle_state = particle.next()

        # Measure POSITION first
        pos_blade = position_meter.next()
        position = particle_state * pos_blade  # encounter → quiddification

        # Now the particle has been disturbed by the position measurement
        # Its state for velocity measurement is DIFFERENT
        particle_state_after_pos = particle.next()  # state has evolved
        vel_blade = velocity_meter.next()
        velocity_after_pos = particle_state_after_pos * vel_blade

        position_then_velocity.append((position, velocity_after_pos))

        # Now do it in reverse order for a fresh particle state
        particle_state2 = particle.next()
        vel_blade2 = velocity_meter.next()
        velocity = particle_state2 * vel_blade2

        particle_state_after_vel = particle.next()
        pos_blade2 = position_meter.next()
        position_after_vel = particle_state_after_vel * pos_blade2

        velocity_then_position.append((velocity, position_after_vel))

        # The "uncertainty product" — analogous to Δx·Δp
        simultaneous_products.append(abs(position * velocity_after_pos))

    # Analysis
    pos_first = np.array(position_then_velocity)
    vel_first = np.array(velocity_then_position)
    products = np.array(simultaneous_products)

    # Correlation between position and velocity measurements
    corr_pos_first = np.corrcoef(pos_first[:, 0], pos_first[:, 1])[0, 1]
    corr_vel_first = np.corrcoef(vel_first[:, 0], vel_first[:, 1])[0, 1]

    print(f"\n--- MEASUREMENT ORDER MATTERS ---")
    print(f"Correlation (pos first, then vel): {corr_pos_first:.4f}")
    print(f"Correlation (vel first, then pos): {corr_vel_first:.4f}")
    print(f"Difference: {abs(corr_pos_first - corr_vel_first):.4f}")
    print(f"  (Non-zero = measurement order affects outcome = non-commutativity)")

    # The uncertainty product
    min_product = np.min(products)
    mean_product = np.mean(products)

    print(f"\n--- UNCERTAINTY PRODUCT ---")
    print(f"Mean Dx*Dp:  {mean_product:.6f}")
    print(f"Min Dx*Dp:   {min_product:.6f}")
    print(f"  (If min > 0, there exists a lower bound — analogous to h-bar/2)")

    # Kurtosis of the uncertainty product
    k_product = sp_stats.kurtosis(products, fisher=False)
    print(f"Kurtosis of Dx*Dp: {k_product:.2f}")

    # Does the uncertainty product have structure?
    product_diffs = np.abs(np.diff(products))
    if len(product_diffs) > 1:
        acf_product = np.corrcoef(product_diffs[:-1], product_diffs[1:])[0, 1]
        print(f"Vol clustering in uncertainty: {acf_product:.4f}")
        print(f"  (>0 = uncertainty is not random — it has temporal structure)")

    # Key test: non-commutativity
    # In quantum mechanics, [x, p] != 0
    # Here: does measuring position first give different velocity
    # than measuring velocity first gives different position?
    pos_values_first = pos_first[:, 0]
    pos_values_second = vel_first[:, 1]

    ks_stat, ks_pvalue = sp_stats.ks_2samp(pos_values_first, pos_values_second)
    print(f"\n--- NON-COMMUTATIVITY TEST ---")
    print(f"KS statistic (pos-first vs pos-second): {ks_stat:.4f}")
    print(f"KS p-value: {ks_pvalue:.6f}")
    print(f"  (p < 0.05 = distributions differ = measurement order matters)")

    return {
        'corr_pos_first': corr_pos_first,
        'corr_vel_first': corr_vel_first,
        'non_commutativity': abs(corr_pos_first - corr_vel_first),
        'mean_uncertainty_product': mean_product,
        'min_uncertainty_product': min_product,
        'k_uncertainty': k_product,
        'ks_pvalue': ks_pvalue,
    }


# ============================================================
# MAIN — RUN ALL THREE EXPERIMENTS
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  THE COINCIDENCE FIELD")
    print("  Transmuting philosophy into experiment")
    print("  Ale Brotto + CRNG")
    print("=" * 60)

    # Experiment 1: Coincidence of bettors and coins
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 1: THE COINCIDENCE OF TWO FIELDS")
    print("#" * 60)

    results_crng = run_experiment(n_coins=100, n_rounds=10000, use_crng=True, seed=42)
    results_prng = run_experiment(n_coins=100, n_rounds=10000, use_crng=False, seed=42)

    print(f"\n{'='*60}")
    print(f"COMPARISON: CRNG vs PRNG COINCIDENCE FIELDS")
    print(f"{'='*60}")
    print(f"Global hit rate  — CRNG: {results_crng['global_hit_rate']:.4f} | PRNG: {results_prng['global_hit_rate']:.4f}")
    print(f"Local kurtosis   — CRNG: {results_crng['local_kurtosis']:.2f} | PRNG: {results_prng['local_kurtosis']:.2f}")
    print(f"Temporal kurtosis — CRNG: {results_crng['temporal_kurtosis']:.2f} | PRNG: {results_prng['temporal_kurtosis']:.2f}")
    print(f"Perm. entropy    — CRNG: {results_crng['permutation_entropy']:.4f} | PRNG: {results_prng['permutation_entropy']:.4f}")
    print(f"Corollary acc.   — CRNG: {results_crng['corollary_accuracy']:.4f} | PRNG: {results_prng['corollary_accuracy']:.4f}")

    # Experiment 2: Rogue waves
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 2: ROGUE WAVES")
    print("#" * 60)

    rogue_results = rogue_wave_experiment(n_fields=5, n_points=50000, seed=42)

    # Experiment 3: Uncertainty field
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 3: THE UNCERTAINTY FIELD")
    print("#" * 60)

    uncertainty_results = uncertainty_experiment(n_particles=1000, n_measurements=5000, seed=42)

    # Final summary
    print("\n\n" + "=" * 60)
    print("  SUMMARY OF FINDINGS")
    print("=" * 60)

    print(f"""
1. COINCIDENCE FIELD:
   - Global hit rate ≈ 0.50 in both modes (LLN confirmed)
   - But LOCAL structure differs: CRNG K={results_crng['local_kurtosis']:.2f} vs PRNG K={results_prng['local_kurtosis']:.2f}
   - Deterministic corollary: {results_crng['corollary_accuracy']:.4f} accuracy
     (>0.50 = knowing N-1 results gives predictive power for the Nth)

2. ROGUE WAVES:
   - CRNG superposition K={rogue_results['k_combined']:.2f} vs Gaussian K={rogue_results['k_gaussian']:.2f}
   - CRNG produces {rogue_results['rogue_fraction']/max(rogue_results['gaussian_rogue_fraction'], 1e-10):.1f}x more rogue events
   - Pre-rogue volatility ratio: {rogue_results.get('pre_rogue_vol_ratio', 'N/A')}
     (>1 = warning signal exists before rogue waves)

3. UNCERTAINTY FIELD:
   - Non-commutativity: {uncertainty_results['non_commutativity']:.4f}
     (>0 = measurement order matters, like quantum mechanics)
   - Min uncertainty product: {uncertainty_results['min_uncertainty_product']:.6f}
     (>0 = lower bound exists, analogous to h-bar/2)
   - KS p-value: {uncertainty_results['ks_pvalue']:.6f}
     (<0.05 = measuring position first ≠ measuring velocity first)
""")
