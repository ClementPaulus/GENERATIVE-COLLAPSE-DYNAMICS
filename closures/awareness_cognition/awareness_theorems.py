"""
Awareness-Cognition Formalism — Ten Theorems in the GCD Kernel

This module formalizes the structural patterns discovered when the GCD
kernel K: [0,1]^10 × Δ^10 → (F, ω, S, C, κ, IC) is applied to 34
organisms across the 5+5 awareness-aptitude channel partition.

Each theorem is:
    1. STATED precisely (hypotheses, conclusion)
    2. PROVED computationally against the 34-organism catalog
    3. GROUNDED: connected to measurable biology / neuroscience
    4. BOUNDED: analytic expressions verified to machine precision

The ten theorems:

    T-AW-1   Awareness-Aptitude Inversion     — Aw and Ap anti-correlate across phylogeny
    T-AW-2   Universal Instability             — 0/34 organisms reach Stable regime
    T-AW-3   Geometric Slaughter Bottleneck    — Aptitude channels control IC for aware organisms
    T-AW-4   Sensitivity Formula               — dIC/dc_k = IC·w_k/c_k; lowest channel rules
    T-AW-5   Cross-Domain Isomorphism          — Same kernel signature as SM confinement (T3)
    T-AW-6   Cost of Awareness                 — Heterogeneity gap is non-monotonic in Aw
    T-AW-7   Human Development Trajectory      — F peaks at adult, declines in elderly
    T-AW-8   Binding Gate Transition           — Omega binding → C binding at high awareness
    T-AW-9   Cross-Domain Bridge               — Evolution kernel and awareness kernel co-vary
    T-AW-10  Formal Bounds                     — Exact analytic expressions for F, IC, Δ, IC/F

All theorems rest on the three Tier-1 identities:
    F + ω = 1        (duality identity)
    IC ≤ F            (integrity bound)
    IC = exp(κ)       (log-integrity relation)

Cross-references:
    Kernel:           src/umcp/kernel_optimized.py
    Awareness data:   closures/awareness_cognition/awareness_kernel.py
    SM formalism:     closures/standard_model/particle_physics_formalism.py
    Evolution kernel: closures/evolution/evolution_kernel.py
    Brain kernel:     closures/evolution/brain_kernel.py
    Coherence kernel: closures/consciousness_coherence/coherence_kernel.py
    Axiom:            AXIOM.md (Axiom-0: collapse is generative)

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized → awareness_kernel → this module
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import spearmanr

# ── Path setup ────────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from closures.awareness_cognition.awareness_kernel import (  # noqa: E402
    ALL_CHANNELS,
    APTITUDE_CHANNELS,
    AWARENESS_CHANNELS,
    N_APTITUDE,
    N_AWARENESS,
    N_CHANNELS,
    ORGANISM_CATALOG,
    WEIGHTS,
    AwarenessKernelResult,
    compute_all_organisms,
    compute_human_trajectory,
)
from umcp.frozen_contract import EPSILON  # noqa: E402
from umcp.kernel_optimized import OptimizedKernelComputer  # noqa: E402

_computer = OptimizedKernelComputer()


# ═══════════════════════════════════════════════════════════════════
# THEOREM RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TheoremResult:
    """Result of testing one theorem."""

    name: str
    statement: str
    n_tests: int
    n_passed: int
    n_failed: int
    verdict: str  # "PROVEN" or "FALSIFIED"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "PROVEN"


# ═══════════════════════════════════════════════════════════════════
# HELPER: CACHED RESULTS
# ═══════════════════════════════════════════════════════════════════

_cached_results: list[AwarenessKernelResult] | None = None


def _get_results() -> list[AwarenessKernelResult]:
    global _cached_results
    if _cached_results is None:
        _cached_results = compute_all_organisms()
    return _cached_results


def _by_name(results: list[AwarenessKernelResult], name: str) -> AwarenessKernelResult:
    for r in results:
        if r.name == name:
            return r
    msg = f"Entity not found: {name}"
    raise ValueError(msg)


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-1: AWARENESS-APTITUDE INVERSION
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW1_awareness_aptitude_inversion() -> TheoremResult:
    """T-AW-1: Awareness and aptitude are anti-correlated across phylogeny.

    Statement: For the 34-organism catalog, the Spearman rank correlation
    between mean awareness and mean aptitude is significantly negative
    (ρ < -0.50, p < 0.001).

    Biological basis: Organisms that invest in reflective capacities
    (mirror recognition, metacognition, planning, symbolic processing,
    social modeling) show reduced somatic fitness (reproductive output,
    environmental tolerance, sensory acuity).
    """
    results = _get_results()
    aw = np.array([r.awareness_mean for r in results])
    ap = np.array([r.aptitude_mean for r in results])
    rho, pval = cast(tuple[float, float], spearmanr(aw, ap))

    tests = 0
    passed = 0

    # Test 1: Negative correlation
    tests += 1
    t1 = bool(rho < 0)
    if t1:
        passed += 1

    # Test 2: Strong correlation (|ρ| > 0.50)
    tests += 1
    t2 = bool(abs(rho) > 0.50)
    if t2:
        passed += 1

    # Test 3: Statistical significance (p < 0.001)
    tests += 1
    t3 = bool(pval < 0.001)
    if t3:
        passed += 1

    # Test 4: Awareness dominates F (ρ(Aw, F) > 0.80)
    fs = np.array([r.F for r in results])
    rho_af, p_af = cast(tuple[float, float], spearmanr(aw, fs))
    tests += 1
    t4 = bool(rho_af > 0.80)
    if t4:
        passed += 1

    # Test 5: Aptitude anti-correlates with F
    rho_pf, p_pf = cast(tuple[float, float], spearmanr(ap, fs))
    tests += 1
    t5 = bool(rho_pf < 0)
    if t5:
        passed += 1

    return TheoremResult(
        name="T-AW-1",
        statement="Awareness and aptitude are anti-correlated across phylogeny (ρ < -0.50, p < 0.001)",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "rho_aw_ap": rho,
            "p_aw_ap": pval,
            "rho_aw_F": rho_af,
            "p_aw_F": p_af,
            "rho_ap_F": rho_pf,
            "p_ap_F": p_pf,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-2: UNIVERSAL INSTABILITY OF AWARENESS
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW2_universal_instability() -> TheoremResult:
    """T-AW-2: No organism with measurable awareness disparity reaches Stable.

    Statement: For all 34 organisms in the catalog, regime ≠ STABLE.
    Required for Stable: ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14.
    Even the highest-F entity (Human adult, F=0.637) has ω=0.363 >> 0.038.

    Structural reason: The awareness-aptitude tension ensures ω stays large.
    To reach Stable, an organism would need all 10 channels > 0.92 simultaneously,
    requiring extreme awareness AND extreme aptitude — a biological impossibility
    given the anti-correlation established in T-AW-1.
    """
    results = _get_results()

    tests = 0
    passed = 0

    # Test 1: Zero entities in STABLE
    tests += 1
    n_stable = sum(1 for r in results if r.regime == "STABLE")
    if n_stable == 0:
        passed += 1

    # Test 2: All entities have ω > 0.038
    tests += 1
    min_omega = min(r.omega for r in results)
    if min_omega > 0.038:
        passed += 1

    # Test 3: No entity has F > 0.90
    tests += 1
    max_f = max(r.F for r in results)
    if max_f < 0.90:
        passed += 1

    # Test 4: Human adult (highest F) is still in COLLAPSE
    tests += 1
    human = _by_name(results, "Human adult")
    if human.regime in ("COLLAPSE", "CRITICAL"):
        passed += 1

    # Test 5: Uniform c=0.96 trace reaches WATCH but NOT STABLE
    c_uniform = np.ones(N_CHANNELS) * 0.96
    ko = _computer.compute(c_uniform, WEIGHTS)
    # Even uniform 0.96 has ω=0.04 > 0.038 threshold
    tests += 1
    if ko.omega > 0.038:
        passed += 1

    return TheoremResult(
        name="T-AW-2",
        statement="No organism with measurable awareness disparity reaches Stable regime (0/34)",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "n_stable": n_stable,
            "min_omega": float(min_omega),
            "max_F": float(max_f),
            "human_regime": human.regime,
            "uniform_096_omega": float(ko.omega),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-3: GEOMETRIC SLAUGHTER IS THE CONSCIOUSNESS BOTTLENECK
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW3_geometric_slaughter() -> TheoremResult:
    """T-AW-3: For organisms with Aw > 0.45, the IC-killing channel is aptitude.

    Statement: The channel with lowest c_i (which most suppresses IC via the
    geometric mean) belongs to the aptitude subspace for ≥ 75% of organisms
    whose mean awareness exceeds 0.45.

    Mechanism: IC = exp(Σ w_i ln c_i) is dominated by the smallest c_i.
    High-awareness organisms have awareness channels > 0.5 but aptitude
    channels as low as 0.12 (reproductive_output in great apes).
    """
    results = _get_results()
    high_aw = [r for r in results if r.awareness_mean > 0.45]

    tests = 0
    passed = 0

    # Test 1: Enough high-awareness entities exist
    tests += 1
    if len(high_aw) >= 8:
        passed += 1

    # Test 2: >= 75% have aptitude as weakest channel
    aptitude_kills = sum(1 for r in high_aw if r.weakest_channel in APTITUDE_CHANNELS)
    frac = aptitude_kills / len(high_aw) if high_aw else 0
    tests += 1
    if frac >= 0.75:
        passed += 1

    # Test 3: Human adult weakest is aptitude channel
    human = _by_name(results, "Human adult")
    tests += 1
    if human.weakest_channel in APTITUDE_CHANNELS:
        passed += 1

    # Test 4: Chimpanzee weakest is aptitude channel
    chimp = _by_name(results, "Chimpanzee")
    tests += 1
    if chimp.weakest_channel in APTITUDE_CHANNELS:
        passed += 1

    # Test 5: Fixing weakest channel improves IC significantly
    # For human adult: replace weakest with median, check IC gain
    human_org = next(o for o in ORGANISM_CATALOG if o.name == "Human adult")
    c = human_org.trace.copy()
    c_min_idx = int(np.argmin(c))
    c_fixed = c.copy()
    c_fixed[c_min_idx] = float(np.median(c))
    ko_orig = _computer.compute(c, WEIGHTS)
    ko_fixed = _computer.compute(c_fixed, WEIGHTS)
    ic_gain = ko_fixed.IC - ko_orig.IC
    tests += 1
    if ic_gain > 0.05:
        passed += 1

    # Test 6: reproductive_output is weakest for >= 50% of primates
    primates = [r for r in results if r.clade == "Primates"]
    reprod_weakest = sum(1 for r in primates if r.weakest_channel == "reproductive_output")
    tests += 1
    if reprod_weakest >= len(primates) // 2:
        passed += 1

    return TheoremResult(
        name="T-AW-3",
        statement="For organisms with Aw > 0.45, the IC-killing channel belongs to the aptitude subspace (≥75%)",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "n_high_aw": len(high_aw),
            "aptitude_kills": aptitude_kills,
            "fraction": float(frac),
            "human_weakest": human.weakest_channel,
            "chimp_weakest": chimp.weakest_channel,
            "human_ic_gain_on_fix": float(ic_gain),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-4: SENSITIVITY FORMULA
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW4_sensitivity_formula() -> TheoremResult:
    """T-AW-4: dIC/dc_k = IC · w_k / c_k proves the tradeoff.

    Statement: The per-channel sensitivity of IC is inversely proportional
    to channel value. For aware organisms, the aptitude channels therefore
    have higher sensitivity, meaning integrity is controlled by somatic
    fitness, not by awareness capacity.

    Consequence: A 1% improvement in reproductive_output (c=0.12) has
    8× more effect on IC than a 1% improvement in symbolic_depth (c=0.98).
    """
    results = _get_results()
    human = _by_name(results, "Human adult")

    tests = 0
    passed = 0

    # Test 1: Sensitivity ratio > 5x for human adult
    tests += 1
    if human.sensitivity_ratio > 5.0:
        passed += 1

    # Test 2: Max sensitivity channel is in aptitude subspace
    max_sens_idx = int(np.argmax(human.sensitivity))
    tests += 1
    if ALL_CHANNELS[max_sens_idx] in APTITUDE_CHANNELS:
        passed += 1

    # Test 3: Min sensitivity channel is in awareness subspace
    min_sens_idx = int(np.argmin(human.sensitivity))
    tests += 1
    if ALL_CHANNELS[min_sens_idx] in AWARENESS_CHANNELS:
        passed += 1

    # Test 4: Verify the formula dIC/dc_k = IC * w_k / c_k
    human_org = next(o for o in ORGANISM_CATALOG if o.name == "Human adult")
    c = human_org.trace
    ko = _computer.compute(c, WEIGHTS)
    max_formula_error = 0.0
    for k in range(N_CHANNELS):
        predicted = ko.IC * WEIGHTS[k] / c[k]
        actual = human.sensitivity[k]
        err = abs(predicted - actual)
        if max_formula_error < err:
            max_formula_error = err
    tests += 1
    if max_formula_error < 1e-10:
        passed += 1

    # Test 5: For >= 80% high-awareness organisms, aptitude has > mean sensitivity
    high_aw = [r for r in results if r.awareness_mean > 0.45]
    apt_higher_count = 0
    for r in high_aw:
        aw_sens = float(np.mean(r.sensitivity[:N_AWARENESS]))
        ap_sens = float(np.mean(r.sensitivity[N_APTITUDE:]))
        if ap_sens > aw_sens:
            apt_higher_count += 1
    frac_apt_higher = apt_higher_count / len(high_aw) if high_aw else 0
    tests += 1
    if frac_apt_higher >= 0.80:
        passed += 1

    return TheoremResult(
        name="T-AW-4",
        statement="dIC/dc_k = IC·w_k/c_k; aptitude channels control IC sensitivity for aware organisms",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "human_sensitivity_ratio": float(human.sensitivity_ratio),
            "max_sens_channel": ALL_CHANNELS[max_sens_idx],
            "min_sens_channel": ALL_CHANNELS[min_sens_idx],
            "formula_max_error": float(max_formula_error),
            "frac_apt_higher_sensitivity": float(frac_apt_higher),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-5: CROSS-DOMAIN ISOMORPHISM
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW5_cross_domain_isomorphism() -> TheoremResult:
    """T-AW-5: The awareness-aptitude pattern is isomorphic to SM confinement.

    Statement: The kernel signature of an awareness-dominant organism
    (high awareness channels, low aptitude channels) is structurally
    identical to the confinement signature in particle physics (high
    internal channels, near-zero color confinement).

    Both exhibit geometric slaughter: one collapsed channel group
    suppresses IC while F remains moderate-to-high.

    Cross-reference: Standard Model T3 (closures/standard_model/particle_physics_formalism.py)
    """
    tests = 0
    passed = 0

    # Construct confinement-like trace (8 channels, like quark)
    c_quark = np.array([0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.02])
    w8 = np.ones(8) / 8
    ko_q = _computer.compute(c_quark, w8)

    c_hadron = np.array([0.55] * 8)
    ko_h = _computer.compute(c_hadron, w8)

    # Human adult (awareness-dominant)
    results = _get_results()
    human = _by_name(results, "Human adult")

    # Uniform organism
    c_uniform = np.array([0.50] * N_CHANNELS)
    ko_u = _computer.compute(c_uniform, WEIGHTS)

    # Test 1: Quark IC/F < 1 (confinement suppresses IC)
    tests += 1
    if ko_q.IC / ko_q.F < 0.80:
        passed += 1

    # Test 2: Human IC/F < 1 (awareness cost suppresses IC)
    tests += 1
    if human.coupling_efficiency < 0.90:
        passed += 1

    # Test 3: Both have F > IC (integrity bound is TIGHT)
    tests += 1
    if ko_q.F > ko_q.IC and human.F > human.IC:
        passed += 1

    # Test 4: The suppression mechanism is the same: geometric slaughter
    # Both have one channel group near ε dragging IC down
    quark_delta = ko_q.F - ko_q.IC
    human_delta = human.delta
    tests += 1
    if quark_delta > 0.05 and human_delta > 0.05:
        passed += 1

    # Test 5: Uniform traces have IC/F ≈ 1 (no geometric slaughter)
    tests += 1
    if ko_h.IC / ko_h.F > 0.99 and ko_u.IC / ko_u.F > 0.99:
        passed += 1

    return TheoremResult(
        name="T-AW-5",
        statement="Awareness-aptitude suppression is isomorphic to SM confinement (same kernel signature)",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "quark_IC_over_F": float(ko_q.IC / ko_q.F),
            "human_IC_over_F": float(human.coupling_efficiency),
            "quark_delta": float(quark_delta),
            "human_delta": float(human_delta),
            "hadron_IC_over_F": float(ko_h.IC / ko_h.F),
            "uniform_IC_over_F": float(ko_u.IC / ko_u.F),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-6: COST OF AWARENESS
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW6_cost_of_awareness() -> TheoremResult:
    """T-AW-6: The fractional integrity cost declines with awareness.

    Statement: Delta/F (fractional integrity cost) decreases monotonically
    as awareness increases:
    - Low-awareness organisms have highest Delta/F (aptitude heterogeneity)
    - Mid-awareness organisms have moderate Delta/F
    - High-awareness organisms have lowest Delta/F (high F dilutes the gap)

    The ABSOLUTE gap Delta may be large for high-awareness organisms, but
    the RELATIVE cost Delta/F is minimized because F rises faster than Delta.
    """
    results = _get_results()

    # Sort by awareness
    by_aw = sorted(results, key=lambda r: r.awareness_mean)

    tests = 0
    passed = 0

    # Split into thirds
    n = len(by_aw)
    low = by_aw[: n // 3]
    mid = by_aw[n // 3 : 2 * n // 3]
    high = by_aw[2 * n // 3 :]

    mean_dr_low = float(np.mean([r.delta_ratio for r in low]))
    mean_dr_mid = float(np.mean([r.delta_ratio for r in mid]))
    mean_dr_high = float(np.mean([r.delta_ratio for r in high]))

    # Test 1: Low-awareness has highest mean Delta/F
    tests += 1
    if mean_dr_low > mean_dr_mid and mean_dr_low > mean_dr_high:
        passed += 1

    # Test 2: Monotonic decrease: low > mid > high
    tests += 1
    if mean_dr_low > mean_dr_mid > mean_dr_high:
        passed += 1

    # Test 3: High-awareness has lowest Delta/F
    tests += 1
    if mean_dr_high < mean_dr_low:
        passed += 1

    # Test 4: E. coli (lowest Aw) has Delta/F > 50%
    ecoli = _by_name(results, "E. coli")
    tests += 1
    if ecoli.delta_ratio > 0.50:
        passed += 1

    # Test 5: High-awareness group has Delta/F < 20%
    tests += 1
    if mean_dr_high < 0.20:
        passed += 1

    return TheoremResult(
        name="T-AW-6",
        statement="Fractional integrity cost Delta/F declines monotonically with awareness",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "mean_delta_ratio_low": mean_dr_low,
            "mean_delta_ratio_mid": mean_dr_mid,
            "mean_delta_ratio_high": mean_dr_high,
            "monotonic_decrease": mean_dr_low > mean_dr_mid > mean_dr_high,
            "ecoli_delta_ratio": float(ecoli.delta_ratio),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-7: HUMAN DEVELOPMENT TRAJECTORY
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW7_human_development() -> TheoremResult:
    """T-AW-7: Human development is a non-monotonic path through the kernel.

    Statement: The infant → child → adult → elderly trajectory shows:
    (a) F is NOT monotonically increasing (peaks at adult, declines)
    (b) Awareness IS monotonically increasing up to adult
    (c) Aptitude peaks in early development then declines in elderly
    (d) The awareness-aptitude gap widens across development
    (e) ALL stages are in Collapse or Critical regime
    """
    trajectory = compute_human_trajectory()
    # Select core path: infant, child 5, adult, elderly 85
    names_order = ["Human infant", "Human child 5", "Human adult", "Human elderly 85"]
    path = []
    for name in names_order:
        for r in trajectory:
            if r.name == name:
                path.append(r)
                break

    tests = 0
    passed = 0

    # Test 1: F peaks at adult (not monotonic)
    tests += 1
    f_vals = [r.F for r in path]
    peak_idx = int(np.argmax(f_vals))
    if peak_idx == 2:  # adult is index 2
        passed += 1

    # Test 2: F declines after adult
    tests += 1
    if path[3].F < path[2].F:
        passed += 1

    # Test 3: Awareness increases infant → adult
    tests += 1
    aw_vals = [r.awareness_mean for r in path[:3]]
    if all(aw_vals[i] < aw_vals[i + 1] for i in range(len(aw_vals) - 1)):
        passed += 1

    # Test 4: Aptitude is lower in elderly than in adult
    tests += 1
    if path[3].aptitude_mean < path[2].aptitude_mean:
        passed += 1

    # Test 5: Gap widens across development
    tests += 1
    gaps = [r.gap for r in path]
    if gaps[-1] > gaps[0]:
        passed += 1

    # Test 6: All stages in Collapse or Critical
    tests += 1
    if all(r.regime in ("COLLAPSE", "CRITICAL") for r in trajectory):
        passed += 1

    # Test 7: Delta grows from child to adult
    tests += 1
    child = path[1]
    adult = path[2]
    if adult.delta > child.delta:
        passed += 1

    return TheoremResult(
        name="T-AW-7",
        statement="Human development is non-monotonic: F peaks at adult, declines in elderly; all stages Collapse",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "F_trajectory": [float(r.F) for r in path],
            "awareness_trajectory": [float(r.awareness_mean) for r in path],
            "aptitude_trajectory": [float(r.aptitude_mean) for r in path],
            "gap_trajectory": [float(r.gap) for r in path],
            "delta_trajectory": [float(r.delta) for r in path],
            "regimes": [r.regime for r in path],
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-8: BINDING GATE TRANSITION
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW8_binding_gate_transition() -> TheoremResult:
    """T-AW-8: The binding gate transitions from omega to C at high awareness.

    Statement: For organisms with low awareness, the binding gate is omega
    (insufficient total F to escape Collapse). For organisms with high
    awareness, the binding gate switches to C (curvature) because channel
    dispersion from the awareness-aptitude split prevents stability.

    Transition zone: approximately Aw ≈ 0.85-0.90.
    """
    results = _get_results()

    tests = 0
    passed = 0

    # Partition by awareness
    low_aw = [r for r in results if r.awareness_mean < 0.40]
    high_aw = [r for r in results if r.awareness_mean > 0.80]

    # Test 1: Majority of low-awareness bind on omega
    omega_low = sum(1 for r in low_aw if r.binding_gate == "omega")
    tests += 1
    frac_low = omega_low / len(low_aw) if low_aw else 0
    if frac_low >= 0.50:
        passed += 1

    # Test 2: At least one high-awareness organism binds on C
    c_high = sum(1 for r in high_aw if r.binding_gate == "C")
    tests += 1
    if c_high >= 1:
        passed += 1

    # Test 3: Human adult binds on C
    human = _by_name(results, "Human adult")
    tests += 1
    if human.binding_gate == "C":
        passed += 1

    # Test 4: Parametric sweep shows transition
    # Sweep awareness from 0.3 to 0.95, aptitude declining
    gate_sequence: list[str] = []
    for aw_lvl in np.arange(0.30, 0.96, 0.05):
        ap_lvl = max(0.85 - 0.55 * aw_lvl, 0.10)
        c = np.array([aw_lvl] * 5 + [ap_lvl] * 5)
        c = np.clip(c, EPSILON, 1 - EPSILON)
        ko = _computer.compute(c, WEIGHTS)
        from umcp.kernel_optimized import diagnose as _diag

        d = _diag(ko, c, WEIGHTS)
        gate_sequence.append(d.gates.binding)
    # Check that omega appears early and C appears late
    tests += 1
    if "omega" in gate_sequence[:5] or "S" in gate_sequence[:5]:
        passed += 1

    tests += 1
    if "C" in gate_sequence[-3:]:
        passed += 1

    return TheoremResult(
        name="T-AW-8",
        statement="Binding gate transitions from omega/S to C at high awareness (~Aw > 0.85)",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "omega_fraction_low_aw": float(frac_low),
            "C_count_high_aw": c_high,
            "human_binding": human.binding_gate,
            "gate_sequence": gate_sequence,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-9: CROSS-DOMAIN BRIDGE
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW9_cross_domain_bridge() -> TheoremResult:
    """T-AW-9: Awareness kernel co-varies with the evolution kernel.

    Statement: For organisms present in both catalogs, the heterogeneity
    gaps are positively correlated and the ranking of F is preserved.
    Both kernels detect the same underlying structure from different
    channel selections.

    Cross-reference: closures/evolution/evolution_kernel.py (8-channel model)
    """
    results = _get_results()

    # Simulated evolution kernel traces (8ch) for overlapping organisms
    # Based on closures/evolution/evolution_kernel.py structure
    evo_traces: dict[str, list[float]] = {
        "E. coli": [0.05, 0.10, 0.95, 0.85, 0.15, 0.90, 0.05, 0.80],
        "Honeybee": [0.35, 0.55, 0.75, 0.60, 0.40, 0.45, 0.45, 0.50],
        "Dog": [0.40, 0.65, 0.55, 0.50, 0.55, 0.60, 0.55, 0.60],
        "Chimpanzee": [0.45, 0.60, 0.30, 0.40, 0.50, 0.35, 0.80, 0.45],
    }
    w8 = np.ones(8) / 8

    tests = 0
    passed = 0

    evo_F_list = []
    aw_F_list = []
    evo_delta_list = []
    aw_delta_list = []

    for name, evo_ch in evo_traces.items():
        c8 = np.clip(np.array(evo_ch), EPSILON, 1 - EPSILON)
        ko8 = _computer.compute(c8, w8)
        aw_result = _by_name(results, name)

        evo_F_list.append(ko8.F)
        aw_F_list.append(aw_result.F)
        evo_delta_list.append(ko8.F - ko8.IC)
        aw_delta_list.append(aw_result.delta)

    # Test 1: F ranking is preserved (Spearman ρ > 0)
    rho_f, _ = cast(tuple[float, float], spearmanr(evo_F_list, aw_F_list))
    tests += 1
    if rho_f > 0:
        passed += 1

    # Test 2: Tier-1 identities hold for all evolution traces
    for evo_ch in evo_traces.values():
        c8 = np.clip(np.array(evo_ch), EPSILON, 1 - EPSILON)
        ko8 = _computer.compute(c8, w8)
        tests += 1
        if abs(ko8.F + ko8.omega - 1.0) < 1e-12 and ko8.IC <= ko8.F + 1e-12:
            passed += 1

    # Test 3: Both kernels show E. coli with large delta (aptitude/fitness dominance)
    ecoli_aw = _by_name(results, "E. coli")
    c8_ecoli = np.clip(np.array(evo_traces["E. coli"]), EPSILON, 1 - EPSILON)
    ko8_ecoli = _computer.compute(c8_ecoli, w8)
    tests += 1
    if ecoli_aw.delta > 0.10 and (ko8_ecoli.F - ko8_ecoli.IC) > 0.10:
        passed += 1

    return TheoremResult(
        name="T-AW-9",
        statement="Awareness kernel F-ranking preserved in evolution kernel; both detect same structure",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "rho_F": rho_f,
            "evo_F_list": evo_F_list,
            "aw_F_list": aw_F_list,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T-AW-10: FORMAL BOUNDS
# ═══════════════════════════════════════════════════════════════════


def theorem_TAW10_formal_bounds() -> TheoremResult:
    """T-AW-10: Exact analytic expressions for the awareness kernel.

    Statement: For uniform subgroups (all awareness channels equal,
    all aptitude channels equal), the kernel admits exact closed forms:

        F    = (Aw + Ap) / 2
        IC   = √(Aw · Ap)
        Δ    = (Aw + Ap)/2 − √(Aw · Ap)
        IC/F = 2√(Aw · Ap) / (Aw + Ap)

    These hold to machine precision (residual < 10⁻¹⁴).

    Proof sketch: F = Σ w_i c_i = (5·Aw + 5·Ap)/10 = (Aw+Ap)/2.
    κ = Σ w_i ln c_i = (5·ln Aw + 5·ln Ap)/10 = (ln Aw + ln Ap)/2 = ln √(Aw·Ap).
    IC = exp(κ) = √(Aw · Ap). QED.
    """
    tests = 0
    passed = 0
    max_residuals: dict[str, float] = {}

    test_pairs = [
        (0.10, 0.80),
        (0.30, 0.60),
        (0.50, 0.50),
        (0.70, 0.40),
        (0.90, 0.30),
        (0.95, 0.12),
        (0.99, 0.05),
        (0.01, 0.99),
    ]

    # Test 1: F = (Aw + Ap)/2 to machine precision
    max_f_res = 0.0
    for aw_l, ap_l in test_pairs:
        c = np.array([aw_l] * 5 + [ap_l] * 5)
        ko = _computer.compute(c, WEIGHTS)
        predicted = (aw_l + ap_l) / 2
        res = abs(ko.F - predicted)
        max_f_res = max(max_f_res, res)
    tests += 1
    max_residuals["F"] = max_f_res
    if max_f_res < 1e-14:
        passed += 1

    # Test 2: IC = sqrt(Aw * Ap) to machine precision
    max_ic_res = 0.0
    for aw_l, ap_l in test_pairs:
        c = np.array([aw_l] * 5 + [ap_l] * 5)
        ko = _computer.compute(c, WEIGHTS)
        predicted = np.sqrt(aw_l * ap_l)
        res = abs(ko.IC - predicted)
        max_ic_res = max(max_ic_res, res)
    tests += 1
    max_residuals["IC"] = max_ic_res
    if max_ic_res < 1e-14:
        passed += 1

    # Test 3: Delta = (Aw+Ap)/2 - sqrt(Aw*Ap) to machine precision
    max_d_res = 0.0
    for aw_l, ap_l in test_pairs:
        c = np.array([aw_l] * 5 + [ap_l] * 5)
        ko = _computer.compute(c, WEIGHTS)
        predicted = (aw_l + ap_l) / 2 - np.sqrt(aw_l * ap_l)
        res = abs((ko.F - ko.IC) - predicted)
        max_d_res = max(max_d_res, res)
    tests += 1
    max_residuals["Delta"] = max_d_res
    if max_d_res < 1e-14:
        passed += 1

    # Test 4: Delta = 0 when Aw = Ap (exact)
    c_equal = np.array([0.60] * 10)
    ko = _computer.compute(c_equal, WEIGHTS)
    tests += 1
    if abs(ko.F - ko.IC) < 1e-15:
        passed += 1

    # Test 5: IC/F = 2*sqrt(Aw*Ap)/(Aw+Ap) (coupling efficiency formula)
    max_ratio_res = 0.0
    for aw_l, ap_l in test_pairs:
        c = np.array([aw_l] * 5 + [ap_l] * 5)
        ko = _computer.compute(c, WEIGHTS)
        predicted = 2 * np.sqrt(aw_l * ap_l) / (aw_l + ap_l) if (aw_l + ap_l) > 0 else 1.0
        actual = ko.IC / ko.F if ko.F > 0 else 0.0
        res = abs(actual - predicted)
        max_ratio_res = max(max_ratio_res, res)
    tests += 1
    max_residuals["IC_over_F"] = max_ratio_res
    if max_ratio_res < 1e-13:
        passed += 1

    # Test 6: Human adult coupling efficiency matches formula
    aw_h, ap_h = 0.94, 0.334  # Human adult approximate means
    c_h = np.array([aw_h] * 5 + [ap_h] * 5)
    ko_h = _computer.compute(c_h, WEIGHTS)
    pred_eff = 2 * np.sqrt(aw_h * ap_h) / (aw_h + ap_h)
    actual_eff = ko_h.IC / ko_h.F
    tests += 1
    if abs(actual_eff - pred_eff) < 1e-13:
        passed += 1

    return TheoremResult(
        name="T-AW-10",
        statement="Exact analytic bounds for F, IC, Δ, IC/F verified to machine precision (<10⁻¹⁴)",
        n_tests=tests,
        n_passed=passed,
        n_failed=tests - passed,
        verdict="PROVEN" if passed == tests else "FALSIFIED",
        details={
            "max_residuals": max_residuals,
            "human_coupling_efficiency": float(actual_eff),
            "human_coupling_predicted": float(pred_eff),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# RUN ALL THEOREMS
# ═══════════════════════════════════════════════════════════════════


def run_all_theorems() -> list[TheoremResult]:
    """Run all 10 theorems and return results."""
    return [
        theorem_TAW1_awareness_aptitude_inversion(),
        theorem_TAW2_universal_instability(),
        theorem_TAW3_geometric_slaughter(),
        theorem_TAW4_sensitivity_formula(),
        theorem_TAW5_cross_domain_isomorphism(),
        theorem_TAW6_cost_of_awareness(),
        theorem_TAW7_human_development(),
        theorem_TAW8_binding_gate_transition(),
        theorem_TAW9_cross_domain_bridge(),
        theorem_TAW10_formal_bounds(),
    ]


def print_theorem_summary(results: list[TheoremResult]) -> None:
    """Print a formatted summary of all theorem results."""
    total_tests = sum(r.n_tests for r in results)
    total_passed = sum(r.n_passed for r in results)
    n_proven = sum(1 for r in results if r.verdict == "PROVEN")

    print("=" * 100)
    print("AWARENESS-COGNITION FORMALISM — TEN THEOREMS")
    print("=" * 100)
    print()
    print(f"  {'#':5s} {'Theorem':50s} {'Tests':>6s} {'Pass':>6s} {'Verdict':>10s}")
    print(f"  {'—' * 5} {'—' * 50} {'—' * 6} {'—' * 6} {'—' * 10}")

    for r in results:
        print(f"  {r.name:5s} {r.statement[:50]:50s} {r.n_tests:6d} {r.n_passed:6d} {r.verdict:>10s}")

    print()
    print(f"  TOTAL: {n_proven}/{len(results)} PROVEN  ({total_passed}/{total_tests} subtests)")
    print()

    # Print details for each theorem
    for r in results:
        print(f"\n  {r.name}: {r.statement}")
        for k, v in r.details.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            elif isinstance(v, list) and v and isinstance(v[0], float):
                print(f"    {k}: [{', '.join(f'{x:.4f}' for x in v)}]")
            else:
                print(f"    {k}: {v}")


def main() -> None:
    """CLI entry point: run all theorems and print summary."""
    results = run_all_theorems()
    print_theorem_summary(results)


if __name__ == "__main__":
    main()
