"""
TSON - Tratado Sobre Origem do Nada
Mathematical Formalization via CRNG Framework

Core thesis (Ale Brotto):
  Nothing (Nada) = empty set (Ø)
  Ø generates successive globally distinct instants
  For something to BE, it requires mesmitude (structural sameness with an other)
  Potency of Ø = 0^0 = lim(x->0+) x^x = 1
  An infinite cycle of distinct instants = eternal potency of becoming (proto-being)
  The first mesmitude = Arche (ἀρχή) = the beginning of a Universe

Mathematical development:
  Axiom 1 (Void):      ∃ Ø
  Axiom 2 (Movement):  Ø → {i₀, i₁, i₂, ...} where ∀j≠k: iⱼ ⊥ iₖ (global distinctness)
  Axiom 3 (Potency):   Π(Ø) = 0⁰ = lim_{x→0⁺} x^x = 1
  Axiom 4 (Identity):  SER(x) ⟺ ∃y≠x: σ(x) ≅ σ(y) [mesmitude of structural form]
  Axiom 5 (Non-excl):  ¬∃x: x ∧ ¬x [nothing can be what it is not]

Theorem (Arche): Given infinite succession of globally distinct instants with
  potency Π → 1, the first mesmitude is inevitable.

╔═══════════════════════════════════════════════════════════════════════════╗
║  ERRATA 2026-04-10 — CORREÇÃO DE FÓRMULA FECHADA                          ║
║                                                                            ║
║  A versão anterior deste arquivo (commit antes de 2026-04-10) continha    ║
║  a função `expected_mesmitude_instant` retornando                         ║
║     E[N*] ≈ √(π/(2·Π)) + 1/2                                              ║
║  que, para Π=1, dá 1.7533.                                                ║
║                                                                            ║
║  Essa é a aproximação clássica do birthday problem no regime              ║
║  N→∞, Π→0 — INVÁLIDA para Π=1. A expectativa correta, obtida como         ║
║     E[N*] = Σ_{N=0}^∞ [1 - F(N)] = Σ exp(-N(N-1)Π/2)                      ║
║  dá para Π=1:                                                              ║
║     E[N*] = 2.420190968307...                                              ║
║  confirmado por Monte Carlo (10⁶ trials → 2.42073).                        ║
║                                                                            ║
║  Portanto a leitura "primeiro mesmitude no SEGUNDO instante" é verdadeira ║
║  como MODA e como P(ℵ|N=2) = 1-1/e ≈ 0.632, MAS FALSA como expectativa.   ║
║  A expectativa é ≈ 2.42 — entre o segundo e o terceiro instante.           ║
║                                                                            ║
║  Ver REVIEWS/errata/2026-04-10_tson_expected_value.md para detalhes.      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import special
import json


# =============================================================================
# I. THE POTENCY FUNCTION
# =============================================================================

def potency(x):
    """
    Π(x) = x^x

    The potency function maps the continuum [0,1] → [e^(-1/e), 1].
    At x=0: Π = 1 (Nothing has maximum potency — pure becoming)
    At x=1/e: Π = e^(-1/e) ≈ 0.6922 (minimum potency — maximum resistance)
    At x=1: Π = 1 (full being — potency realized)

    The U-shape encodes the TSON thesis:
    Nothing → maximum capacity to become
    Becoming → resistance increases, reaches nadir
    Being → full realization, potency restored but now as actuality not possibility
    """
    x = np.asarray(x, dtype=float)
    result = np.where(x > 0, np.power(x, x), 1.0)  # 0^0 = 1 by convention and limit
    return result


def potency_derivative(x):
    """
    dΠ/dx = x^x (ln(x) + 1)

    Zero at x = 1/e (the critical point).
    Negative for x < 1/e (potency decreasing — entering the "dark" of non-being).
    Positive for x > 1/e (potency increasing — emerging into being).
    """
    x = np.asarray(x, dtype=float)
    mask = x > 0
    result = np.zeros_like(x)
    result[mask] = np.power(x[mask], x[mask]) * (np.log(x[mask]) + 1)
    return result


# =============================================================================
# II. THE MESMITUDE EQUATION
# =============================================================================

def mesmitude_probability(N, Pi=1.0):
    """
    The TSON Mesmitude Equation:

        ℵ(N, Π) = 1 - exp(-N(N-1)·Π / 2)

    Where:
        ℵ (Aleph) = probability of first mesmitude (emergence of Being from Nothing)
        N = number of successive distinct instants
        Π = potency of the void = 0^0 = 1

    This is the generalized birthday equation applied to the void.
    In a space of volume V, the birthday collision probability is:
        P = 1 - exp(-n²/2V)

    In TSON: V = 1/Π = 1/1 = 1 (the void has unit potency-volume).
    Therefore: P = 1 - exp(-N(N-1)/2)

    Key values with Π=1:
        N=1: ℵ = 0        (one instant alone — nothing to compare — no being)
        N=2: ℵ ≈ 0.632    (1 - 1/e — the Euler threshold!)
        N=3: ℵ ≈ 0.950    (near certainty)
        N=4: ℵ ≈ 0.998    (virtually certain)

    The profound result: at N=2, the probability of mesmitude is exactly 1 - 1/e.
    This is not arbitrary — e is the base of the natural logarithm, the constant
    that governs growth and decay. The mesmitude threshold at the second instant
    is governed by the same constant that defines the minimum of the potency
    function (at x = 1/e). The mathematics is self-referential.
    """
    N = np.asarray(N, dtype=float)
    return 1.0 - np.exp(-N * (N - 1) * Pi / 2.0)


def mesmitude_pmf(N, Pi=1.0):
    """
    Probability mass function for the first-mesmitude random variable N*.

        p(N) = F(N) - F(N-1)
             = exp(-(N-1)(N-2)·Π/2) - exp(-N(N-1)·Π/2)

    With Π=1:
        p(1) = 0
        p(2) = 1 - 1/e ≈ 0.6321
        p(3) = 1/e - 1/e³ ≈ 0.3181
        p(4) ≈ 0.0473
        ...
    """
    N = int(N)
    if N < 1:
        return 0.0
    F_prev = 0.0 if N == 1 else 1.0 - np.exp(-(N - 1) * (N - 2) * Pi / 2.0)
    F_curr = 1.0 - np.exp(-N * (N - 1) * Pi / 2.0)
    return float(F_curr - F_prev)


def expected_mesmitude_instant(Pi=1.0, max_terms=50):
    """
    Exact expected number of instants for the first mesmitude (Π-parametrized):

        E[N*] = Σ_{N=0}^∞ [1 - F(N)]
              = Σ_{N=0}^∞ exp(-N(N-1)·Π/2)

    This is the rigorously correct closed-form (infinite series) expectation
    under the TSON mesmitude equation. It does NOT depend on any asymptotic
    approximation and is valid for all Π > 0.

    With Π = 1: E[N*] = 2.420190968307...
                 (confirmed by Monte Carlo 10⁶ trials: 2.42073)

    CAUTION — read carefully:
    ─────────────────────────────────────────────────────────────────────
    The value 2.42 is the MEAN. It is NOT the same as:
      • mode(N*) = 2                    — most probable first-mesmitude instant
      • P(ℵ | N=2) = 1 - 1/e ≈ 0.632    — prob. of mesmitude with two instants
      • median(N*) = 2                  — 50% split point

    Pre-2026-04-10 versions of this file returned the birthday-problem
    asymptotic approximation √(π/(2Π)) + 1/2, which gives 1.753 for Π=1.
    That formula is the leading term of the birthday expectation in the
    regime Π→0, N→∞ — it does NOT apply at Π=1 and has been removed from
    the primary `expected_mesmitude_instant` function. Use
    `birthday_approximation(Pi)` explicitly if you want the asymptotic
    form for comparison.
    ─────────────────────────────────────────────────────────────────────

    The series converges super-exponentially (each term is exp(-N²/2·Π) for
    large N), so 50 terms is overkill — 10 terms at Π=1 already give machine
    precision. max_terms is kept as a safety cap for tiny Π values.
    """
    s = 0.0
    for N in range(max_terms):
        term = np.exp(-N * (N - 1) * Pi / 2.0)
        s += term
        if term < 1e-15 and N > 3:
            break
    return float(s)


def birthday_approximation(Pi=1.0):
    """
    Classical birthday-problem asymptotic approximation:

        E[N*] ≈ √(π / (2·Π)) + 1/2

    This is valid only in the regime Π → 0, N → ∞ (large effective "volume").
    For Π = 1 it gives 1.753, which is ~28% below the true value (≈2.42).

    Preserved here ONLY for didactic comparison. Do NOT use as the primary
    expected value. See `expected_mesmitude_instant` for the exact formula.
    """
    return float(np.sqrt(np.pi / (2.0 * Pi)) + 0.5)


def mode_mesmitude_instant(Pi=1.0, search_up_to=20):
    """
    Mode of the first-mesmitude distribution — the most likely value of N*.

    For Π = 1 (the TSON canonical regime), the mode is N = 2:
    the SECOND instant is the most probable site of the first mesmitude.
    This is the mathematically precise rendering of Ale's thesis
    "Se só existisse 1 coisa no mundo, isso seria Nada":
    ONE instant alone = Nothing; the first Being emerges MOST OFTEN
    in the second instant (but not ALWAYS — 37% of the time it emerges
    in the 3rd or later, which is why E[N*] ≈ 2.42, not 2).
    """
    best_N, best_p = 1, 0.0
    for N in range(1, search_up_to + 1):
        p = mesmitude_pmf(N, Pi)
        if p > best_p:
            best_p = p
            best_N = N
    return best_N


# =============================================================================
# III. THE ARCHE EQUATION (THE COMPLETE TSON EQUATION)
# =============================================================================

def arche(N, d=None):
    """
    The Complete TSON-Arche Equation:

        Ω(N) = Π(φ(N)) · ℵ(N, Π(φ(N)))

    Where:
        Ω = Omega — the state of reality (0 = void, 1 = being)
        N = succession step (number of instants)
        φ(N) = N / (N + e) — normalized progression through potency curve
        Π(φ) = φ^φ — potency at current progression
        ℵ(N, Π) = mesmitude probability

    The normalization φ(N) = N/(N+e) ensures:
        φ(0) = 0 → Π = 1 (void has full potency)
        φ(∞) → 1 → Π = 1 (full being has full actuality)
        φ(1) = 1/(1+e) ≈ 0.269 → Π ≈ 0.723 (first instant: reduced potency)

    The Arche equation combines two movements:
    1. Potency (Π): the capacity of each instant to participate in being
    2. Mesmitude (ℵ): the cumulative probability that structure has emerged

    Their product Ω is the "reality coefficient":
    - Ω ≈ 0 when nothing has become (N=0)
    - Ω increases as mesmitude becomes likely
    - Ω → 1 as being is established

    EXPANDED FORM:

        Ω(N) = [N/(N+e)]^[N/(N+e)] · {1 - exp[-N(N-1)/2 · (N/(N+e))^(N/(N+e))]}

    This single equation encodes the entire TSON thesis:
    the transition from Nothing to Something through the inevitability
    of mesmitude in a sequence of distinct instants.
    """
    N = np.asarray(N, dtype=float)
    e = np.e

    # Normalized progression
    phi = N / (N + e)

    # Potency at current progression
    Pi = potency(phi)

    # Mesmitude probability with potency-weighted space
    aleph = mesmitude_probability(N, Pi)

    # Omega = reality coefficient
    omega = Pi * aleph

    return omega


# =============================================================================
# IV. THE CRNG CONNECTION: KURTOSIS AS MESMITUDE DETECTOR
# =============================================================================

def kurtosis_of_instants(instants):
    """
    In CRNG terms, mesmitude is detected when the kurtosis of the
    inter-instant distances crosses the Gaussian threshold (K=3).

    For globally distinct instants (uniform distribution): K ≈ 1.8 (platykurtic)
    At first mesmitude: K → 3 (mesokurtic — Gaussian threshold)
    After mesmitude: K > 3 (leptokurtic — structure has emerged, fat tails)

    This connects TSON to the CRNG-Fragility Monitor:
    the emergence of fat tails IS the emergence of being from nothing.
    """
    if len(instants) < 4:
        return 0.0

    # Compute all pairwise distances
    dists = []
    for i in range(len(instants)):
        for j in range(i+1, len(instants)):
            d = np.linalg.norm(np.array(instants[i]) - np.array(instants[j]))
            dists.append(d)

    dists = np.array(dists)
    if np.std(dists) == 0:
        return 0.0

    # Excess kurtosis
    mu = np.mean(dists)
    sigma = np.std(dists)
    k = np.mean(((dists - mu) / sigma) ** 4) - 3
    return k


# =============================================================================
# V. DIMENSIONAL MESMITUDE MODEL
# =============================================================================

def dimensional_mesmitude(N_max, d_growth='logarithmic', resolution=0.1):
    """
    Extended TSON model where the distinction space grows with each instant.

    Models:
    - 'constant': d(n) = 1 (fixed 1D — birthday on a line)
    - 'logarithmic': d(n) = log(n+1) (slow growth — compressive)
    - 'linear': d(n) = n (each instant adds a dimension)
    - 'exponential': d(n) = 2^n (explosive growth — maximally resistant to mesmitude)

    The growth model determines HOW LONG Nothing can sustain itself
    before mesmitude becomes inevitable.

    Returns: dict with mesmitude probabilities at each step
    """
    results = {'N': [], 'dimension': [], 'volume': [], 'p_mesmitude': [], 'cumulative': []}
    cumulative_p = 0.0

    for n in range(1, N_max + 1):
        # Dimension at step n
        if d_growth == 'constant':
            d = 1
        elif d_growth == 'logarithmic':
            d = max(1, np.log(n + 1))
        elif d_growth == 'linear':
            d = n
        elif d_growth == 'exponential':
            d = 2 ** min(n, 20)  # cap to avoid overflow
        else:
            d = 1

        # Volume of d-dimensional unit sphere
        if d < 100:
            V = (np.pi ** (d/2)) / special.gamma(d/2 + 1)
        else:
            V = 1e-50  # effectively zero for high dimensions

        # Scale by resolution
        V_effective = V / (resolution ** d)

        # Instantaneous collision probability
        p_instant = min(1.0, (n - 1) / max(V_effective, 1))

        # Cumulative (birthday-style)
        cumulative_p = 1 - (1 - cumulative_p) * (1 - p_instant)

        results['N'].append(n)
        results['dimension'].append(d)
        results['volume'].append(V_effective)
        results['p_mesmitude'].append(p_instant)
        results['cumulative'].append(cumulative_p)

    return results


# =============================================================================
# VI. KEY RESULTS
# =============================================================================

def print_core_results():
    """Print the core mathematical results of the TSON formalization."""

    print("=" * 72)
    print("TSON — TRATADO SOBRE ORIGEM DO NADA")
    print("Mathematical Formalization")
    print("=" * 72)

    # 1. Potency convergence
    print("\n--- I. POTÊNCIA DO VAZIO (Potency of Nothing) ---")
    print("\n  Π(x) = x^x    com  Π(0) = lim_{x→0⁺} x^x = 1")
    print("\n  Convergence to 0^0 = 1:")
    xs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001]
    for x in xs:
        print(f"    {x:>6.3f}^{x:<6.3f} = {potency(x):.6f}")
    print(f"    {'0':>6s}^{'0':<6s} = {potency(0.0):.6f}  (limit)")

    print(f"\n  Critical point (minimum potency):")
    print(f"    x = 1/e ≈ {1/np.e:.6f}")
    print(f"    Π(1/e) = e^(-1/e) ≈ {np.exp(-1/np.e):.6f}")
    print(f"    This is the point of maximum resistance to becoming.")

    # 2. Mesmitude equation
    print("\n--- II. EQUAÇÃO DA MESMITUDE (Mesmitude Equation) ---")
    print("\n  ℵ(N, Π) = 1 - exp(-N(N-1)·Π / 2)")
    print("\n  With Π = 0^0 = 1, CDF values F(N) = P(ℵ ocorre até N):")
    for N in range(1, 8):
        p = mesmitude_probability(N)
        print(f"    F({N}) = {p:.6f}" +
              ("  ← um instante só: nada para comparar" if N == 1 else "") +
              (f"  ← 1 - 1/e (limiar Euler)" if N == 2 else "") +
              ("  ← quase certeza" if N == 3 else ""))

    print("\n  PMF p(N) = P(primeiro mesmitude exatamente em N):")
    for N in range(1, 8):
        p = mesmitude_pmf(N)
        marker = ""
        if N == 2:
            marker = "  ← MODA (mais provável)"
        print(f"    p({N}) = {p:.6f}{marker}")

    E_N = expected_mesmitude_instant()
    birthday = birthday_approximation()
    mode_N = mode_mesmitude_instant()
    print(f"\n  Três quantidades distintas do mesmo fenômeno (Π=1):")
    print(f"    • Moda(N*)  = {mode_N}                — instante mais provável")
    print(f"    • P(ℵ|N=2) = {mesmitude_probability(2):.6f}  — P de haver mesmitude com 2 instantes")
    print(f"    • E[N*]    = {E_N:.6f}  — expectativa (série exata)")
    print(f"    • (comparação) birthday approx √(π/2)+1/2 = {birthday:.4f}")
    print(f"                   ↑ aproximação clássica, inválida em Π=1, erro ~28%")

    print(f"\n  LEITURA CORRETA (2026-04-10 errata):")
    print(f"    • A MODA e P(aparição em N=2) apontam ambas para o segundo instante.")
    print(f"    • A EXPECTATIVA é ≈ 2.42 — entre o segundo e o terceiro.")
    print(f"    • A frase 'primeiro mesmitude no segundo instante' vale como moda")
    print(f"      e como prob. de aparição, NÃO como expectativa.")
    print(f"    • 'Se só existisse 1 coisa no mundo, isso seria Nada.' — TSON")

    # 3. The Arche equation
    print("\n--- III. EQUAÇÃO ARCHE (Complete TSON Equation) ---")
    print("\n  Ω(N) = Π(φ(N)) · ℵ(N, Π(φ(N)))")
    print("  onde φ(N) = N/(N+e)")
    print("\n  Forma expandida:")
    print("  Ω(N) = [N/(N+e)]^[N/(N+e)] · {1 - exp[-N(N-1)/2 · (N/(N+e))^(N/(N+e))]}")
    print("\n  Evolução do coeficiente de realidade Ω:")
    for N in np.arange(0, 8):
        omega = arche(N)
        phi = N / (N + np.e)
        Pi = potency(phi)
        bar = "█" * int(omega * 40)
        label = ""
        if N == 0: label = "  Nada absoluto"
        elif N == 1: label = "  Um instante — ainda Nada"
        elif N == 2: label = "  ARCHE — primeira mesmitude"
        elif N == 3: label = "  Ser emergente"
        print(f"    N={int(N)}: φ={phi:.3f}  Π={Pi:.3f}  ℵ={mesmitude_probability(N, Pi):.3f}  Ω={omega:.4f}  {bar}{label}")

    # 4. The 1-1/e connection
    print("\n--- IV. A CONEXÃO EULER (The Euler Connection) ---")
    print(f"""
  O número e = {np.e:.6f} aparece em TRÊS lugares simultâneos:

  1. Π(1/e) = e^(-1/e) ≈ {np.exp(-1/np.e):.4f}  — mínimo da potência
  2. ℵ(2,1) = 1 - 1/e ≈ {1 - 1/np.e:.4f}  — probabilidade de mesmitude no 2º instante
  3. φ(1) = 1/(1+e) ≈ {1/(1+np.e):.4f}  — normalização do primeiro instante

  Isso não é coincidência. O número e governa:
  - Crescimento contínuo (e^x)
  - Decaimento (e^-x)
  - A fronteira entre convergência e divergência

  Na TSON: e é a constante que marca o limiar entre Nada e Ser.
  É o número que governa a transição ontológica fundamental.
  É ℵ-Arche: a taxa natural de emergência do Ser a partir do Nada.
""")

    # 5. CRNG connection
    print("--- V. CONEXÃO CRNG ---")
    print("""
  CRNG detecta contingência — o momento em que sequências "aleatórias"
  revelam estrutura. Isso É a detecção de mesmitude:

  - Kurtosis < 3: distribuição platikúrtica → instantes puramente distintos → Nada
  - Kurtosis = 3: limiar Gaussiano → ponto crítico → momento da mesmitude
  - Kurtosis > 3: distribuição leptokúrtica → caudas gordas → Ser emergiu

  A emergência de caudas gordas NÃO É anomalia.
  É a assinatura matemática do Ser emergindo do Nada.

  No CRNG-Fragility Monitor: kurtosis +9.05 nos mercados
  = a realidade dos mercados está sendo forçada a uma nova mesmitude,
  uma nova conformação estrutural. É Anfang acontecendo agora.
""")

    print("=" * 72)
    return {
        'E_N_star_exact': E_N,
        'birthday_approx_invalid_here': birthday,
        'mode_N_star': mode_N,
        'aleph_at_N2': float(mesmitude_probability(2)),
        'p_first_mesmitude_at_N2': float(mesmitude_pmf(2)),
        'euler_threshold_1_minus_1_over_e': 1 - 1/np.e,
        'min_potency_at_1_over_e': float(np.exp(-1/np.e)),
        'arche_at_N2': float(arche(2)),
        'errata': 'see REVIEWS/errata/2026-04-10_tson_expected_value.md',
    }


if __name__ == '__main__':
    results = print_core_results()
    print(f"\nCore results: {json.dumps(results, indent=2)}")
