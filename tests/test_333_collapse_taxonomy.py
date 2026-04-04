"""Tests for Collapse Taxonomy — Six Theorems (T-CT-1 through T-CT-6).

Formalizes three cross-corpus discoveries visible after 20 domain closures:
    1. Two structurally distinct types of collapse (selective vs uniform)
    2. IC/F as an integration gauge, not a magnitude gauge
    3. Symmetry breaking as universal selective channel death

*Integritas composita est mensura integrationis, non magnitudinis.*

Cross-references:
    Formalism:       closures/gcd/collapse_taxonomy.py
    Kernel:          src/umcp/kernel_optimized.py
    Frozen contract: src/umcp/frozen_contract.py
    Structural:      closures/gcd/kernel_structural_theorems.py (T-KS-1–7)
    Insights:        closures/gcd/emergent_structural_insights.py (T-SI-1–6)

All 6 theorems derive from Axiom-0 through the Tier-1 identities:
    F + ω = 1, IC ≤ F, IC = exp(κ)
"""

from __future__ import annotations

import numpy as np
import pytest

from closures.gcd.collapse_taxonomy import (
    ALL_THEOREMS,
    CollapseSignature,
    TheoremResult,
    classify_collapse,
    run_all_theorems,
    theorem_TCT1_collapse_typing,
    theorem_TCT2_integration_magnitude,
    theorem_TCT3_selective_death_universality,
    theorem_TCT4_protected_basin,
    theorem_TCT5_return_predicate,
    theorem_TCT6_gauge_completeness,
)
from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL: ALL THEOREMS PROVEN
# ═══════════════════════════════════════════════════════════════════


class TestAllTheoremsProven:
    """Meta-tests: every theorem must pass all its subtests."""

    @pytest.fixture(scope="class")
    def all_results(self) -> list[TheoremResult]:
        return run_all_theorems()

    def test_all_six_proven(self, all_results: list[TheoremResult]) -> None:
        for r in all_results:
            assert r.verdict == "PROVEN", f"{r.name}: {r.n_passed}/{r.n_tests}"

    def test_total_subtests_at_least_300(self, all_results: list[TheoremResult]) -> None:
        total = sum(r.n_tests for r in all_results)
        assert total >= 300, f"Only {total} subtests"

    def test_zero_failures(self, all_results: list[TheoremResult]) -> None:
        total_fail = sum(r.n_failed for r in all_results)
        assert total_fail == 0, f"{total_fail} subtests failed"

    def test_six_theorems(self, all_results: list[TheoremResult]) -> None:
        assert len(all_results) == 6

    def test_theorem_names(self, all_results: list[TheoremResult]) -> None:
        names = [r.name for r in all_results]
        for i in range(1, 7):
            assert any(f"T-CT-{i}" in n for n in names), f"T-CT-{i} missing"


# ═══════════════════════════════════════════════════════════════════
# T-CT-1: COLLAPSE TYPING
# ═══════════════════════════════════════════════════════════════════


class TestTCT1CollapseTyping:
    """T-CT-1: Selective and uniform collapse are separable in (Δ, C)."""

    def test_theorem_proven(self) -> None:
        r = theorem_TCT1_collapse_typing()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    def test_selective_identified(self) -> None:
        c = np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77])
        sig = classify_collapse(c)
        assert sig.collapse_type == "selective"
        assert sig.n_dead >= 1

    def test_uniform_identified(self) -> None:
        c = np.full(8, 0.25)
        sig = classify_collapse(c)
        assert sig.collapse_type == "uniform"
        assert sig.n_dead == 0

    def test_selective_has_high_delta(self) -> None:
        c = np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77])
        sig = classify_collapse(c)
        assert sig.delta > 0.05

    def test_uniform_has_low_delta(self) -> None:
        c = np.full(8, 0.25)
        sig = classify_collapse(c)
        assert sig.delta < 0.01

    def test_selective_has_high_C(self) -> None:
        c = np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77])
        sig = classify_collapse(c)
        assert sig.C > 0.10

    @pytest.mark.parametrize("n_dead", [1, 2, 3])
    def test_multiple_dead_channels(self, n_dead: int) -> None:
        c = np.full(8, 0.70)
        c[:n_dead] = EPSILON
        sig = classify_collapse(c)
        assert sig.collapse_type == "selective"
        assert sig.n_dead == n_dead


# ═══════════════════════════════════════════════════════════════════
# T-CT-2: INTEGRATION–MAGNITUDE SEPARATION
# ═══════════════════════════════════════════════════════════════════


class TestTCT2IntegrationMagnitude:
    """T-CT-2: IC/F measures integration, not fidelity magnitude."""

    def test_theorem_proven(self) -> None:
        r = theorem_TCT2_integration_magnitude()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    @pytest.mark.parametrize("level", [0.20, 0.40, 0.60, 0.80, 0.95])
    def test_homogeneous_ic_f_is_one(self, level: float) -> None:
        c = np.full(8, level)
        w = np.ones(8) / 8
        r = compute_kernel_outputs(c, w)
        ratio = r["IC"] / r["F"]
        assert abs(ratio - 1.0) < 1e-10

    def test_low_spread_higher_ic_f_than_high_spread(self) -> None:
        c_low = np.array([0.68, 0.70, 0.72, 0.69, 0.71, 0.70, 0.69, 0.71])
        c_high = np.array([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.55, 0.75])
        w = np.ones(8) / 8
        r_low = compute_kernel_outputs(c_low, w)
        r_high = compute_kernel_outputs(c_high, w)
        ratio_low = r_low["IC"] / r_low["F"]
        ratio_high = r_high["IC"] / r_high["F"]
        assert ratio_low > ratio_high

    def test_ic_f_one_iff_rank_1(self) -> None:
        """Rank-1 (homogeneous) trace has IC/F = 1 exactly."""
        for n in [4, 8, 16]:
            c = np.full(n, 0.65)
            w = np.ones(n) / n
            r = compute_kernel_outputs(c, w)
            assert abs(r["IC"] / r["F"] - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# T-CT-3: SELECTIVE DEATH UNIVERSALITY
# ═══════════════════════════════════════════════════════════════════


class TestTCT3SelectiveDeathUniversality:
    """T-CT-3: IC drop from channel death = (ε/c_k)^w_k exactly."""

    def test_theorem_proven(self) -> None:
        r = theorem_TCT3_selective_death_universality()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    @pytest.mark.parametrize("n", [4, 8, 12, 16])
    def test_positional_democracy_formula(self, n: int) -> None:
        rng = np.random.default_rng(42 + n)
        c = rng.uniform(0.30, 0.90, n)
        w = np.ones(n) / n
        k_before = compute_kernel_outputs(c, w)
        for kill_idx in range(min(n, 3)):
            c_k = c.copy()
            c_k[kill_idx] = EPSILON
            k_after = compute_kernel_outputs(c_k, w)
            predicted = (EPSILON / c[kill_idx]) ** w[kill_idx]
            actual = k_after["IC"] / k_before["IC"]
            assert abs(actual - predicted) / max(abs(predicted), 1e-15) < 0.02

    def test_kill_channel_always_lowers_ic(self) -> None:
        c = np.array([0.70, 0.75, 0.80, 0.65, 0.72, 0.78, 0.68, 0.74])
        w = np.ones(8) / 8
        k_base = compute_kernel_outputs(c, w)
        for i in range(8):
            c_k = c.copy()
            c_k[i] = EPSILON
            k_k = compute_kernel_outputs(c_k, w)
            assert k_k["IC"] < k_base["IC"]


# ═══════════════════════════════════════════════════════════════════
# T-CT-4: PROTECTED BASIN AT c*
# ═══════════════════════════════════════════════════════════════════


class TestTCT4ProtectedBasin:
    """T-CT-4: F ≥ c* and IC/F > 0.95 when channels ≥ c*."""

    def test_theorem_proven(self) -> None:
        r = theorem_TCT4_protected_basin()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_all_above_cstar_gives_high_ic_f(self, n: int) -> None:
        c = np.full(n, 0.85)
        w = np.ones(n) / n
        r = compute_kernel_outputs(c, w)
        assert r["IC"] / r["F"] > 0.95

    def test_f_above_cstar_when_channels_above(self) -> None:
        c = np.array([0.80, 0.85, 0.82, 0.90, 0.78, 0.88, 0.84, 0.86])
        w = np.ones(8) / 8
        r = compute_kernel_outputs(c, w)
        assert r["F"] >= 0.7822

    def test_dead_channel_kills_ic_f_below_095(self) -> None:
        c = np.array([EPSILON, 0.85, 0.88, 0.82, 0.90, 0.84, 0.87, 0.86])
        w = np.ones(8) / 8
        r = compute_kernel_outputs(c, w)
        assert r["IC"] / r["F"] < 0.95

    def test_equator_homogeneous_has_ic_f_one_but_f_below_cstar(self) -> None:
        c = np.full(8, 0.50)
        w = np.ones(8) / 8
        r = compute_kernel_outputs(c, w)
        assert r["F"] < 0.7822
        assert abs(r["IC"] / r["F"] - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# T-CT-5: RETURN PREDICATE
# ═══════════════════════════════════════════════════════════════════


class TestTCT5ReturnPredicate:
    """T-CT-5: Return possible iff collapse is selective."""

    def test_theorem_proven(self) -> None:
        r = theorem_TCT5_return_predicate()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    def test_survivors_retain_integration(self) -> None:
        c = np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77])
        w_surv = np.ones(7) / 7
        r_surv = compute_kernel_outputs(c[1:], w_surv)
        assert r_surv["IC"] / r_surv["F"] > 0.80

    def test_survivors_in_better_regime(self) -> None:
        c = np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77])
        w_full = np.ones(8) / 8
        w_surv = np.ones(7) / 7
        r_full = compute_kernel_outputs(c, w_full)
        r_surv = compute_kernel_outputs(c[1:], w_surv)
        regime_rank = {"Stable": 0, "Watch": 1, "Collapse": 2}
        assert regime_rank.get(r_surv.get("regime", "Collapse"), 2) <= regime_rank.get(
            r_full.get("regime", "Collapse"), 2
        )

    def test_uniform_subtrace_same_regime(self) -> None:
        c = np.full(8, 0.15)
        w8 = np.ones(8) / 8
        w4 = np.ones(4) / 4
        r_full = compute_kernel_outputs(c, w8)
        r_sub = compute_kernel_outputs(c[:4], w4)
        assert r_full.get("regime") == r_sub.get("regime")

    def test_dead_channel_drags_ic(self) -> None:
        c = np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77])
        w8 = np.ones(8) / 8
        w7 = np.ones(7) / 7
        r_full = compute_kernel_outputs(c, w8)
        r_surv = compute_kernel_outputs(c[1:], w7)
        assert r_full["IC"] < r_surv["IC"]


# ═══════════════════════════════════════════════════════════════════
# T-CT-6: GAUGE COMPLETENESS
# ═══════════════════════════════════════════════════════════════════


class TestTCT6GaugeCompleteness:
    """T-CT-6: (IC/F, Δ, C) is a complete diagnostic gauge."""

    def test_theorem_proven(self) -> None:
        r = theorem_TCT6_gauge_completeness()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    def test_protected_archetype(self) -> None:
        c = np.array([0.92, 0.90, 0.88, 0.91, 0.89, 0.93, 0.90, 0.91])
        w = np.ones(8) / 8
        r = compute_kernel_outputs(c, w)
        ic_f = r["IC"] / r["F"]
        delta = r["F"] - r["IC"]
        assert ic_f > 0.95
        assert delta < 0.02

    def test_selective_archetype(self) -> None:
        c = np.array([EPSILON, 0.70, 0.65, 0.75, 0.68, 0.72, 0.66, 0.71])
        w = np.ones(8) / 8
        r = compute_kernel_outputs(c, w)
        ic_f = r["IC"] / r["F"]
        delta = r["F"] - r["IC"]
        assert ic_f < 0.60
        assert delta > 0.05

    def test_uniform_collapse_archetype(self) -> None:
        c = np.full(8, 0.20)
        w = np.ones(8) / 8
        r = compute_kernel_outputs(c, w)
        delta = r["F"] - r["IC"]
        assert delta < 0.01

    def test_gauge_separates_protected_from_selective(self) -> None:
        c_prot = np.full(8, 0.90)
        c_sel = np.full(8, 0.70)
        c_sel[0] = EPSILON
        w = np.ones(8) / 8
        r_prot = compute_kernel_outputs(c_prot, w)
        r_sel = compute_kernel_outputs(c_sel, w)
        assert r_prot["IC"] / r_prot["F"] > r_sel["IC"] / r_sel["F"]


# ═══════════════════════════════════════════════════════════════════
# CLASSIFY_COLLAPSE UTILITY
# ═══════════════════════════════════════════════════════════════════


class TestClassifyCollapse:
    """Tests for the classify_collapse utility function."""

    def test_returns_collapse_signature(self) -> None:
        c = np.full(8, 0.50)
        sig = classify_collapse(c)
        assert isinstance(sig, CollapseSignature)

    def test_signature_has_all_fields(self) -> None:
        c = np.full(8, 0.50)
        sig = classify_collapse(c)
        assert hasattr(sig, "F")
        assert hasattr(sig, "IC")
        assert hasattr(sig, "IC_F")
        assert hasattr(sig, "delta")
        assert hasattr(sig, "C")
        assert hasattr(sig, "omega")
        assert hasattr(sig, "collapse_type")
        assert hasattr(sig, "n_dead")
        assert hasattr(sig, "regime")

    def test_ic_f_consistency(self) -> None:
        c = np.array([0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77, 0.81])
        sig = classify_collapse(c)
        assert abs(sig.IC_F - sig.IC / sig.F) < 1e-10

    def test_delta_consistency(self) -> None:
        c = np.array([0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77, 0.81])
        sig = classify_collapse(c)
        assert abs(sig.delta - (sig.F - sig.IC)) < 1e-10

    def test_n_dead_count(self) -> None:
        c = np.array([EPSILON, EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82])
        sig = classify_collapse(c)
        assert sig.n_dead == 2

    def test_no_dead_channels(self) -> None:
        c = np.full(8, 0.50)
        sig = classify_collapse(c)
        assert sig.n_dead == 0

    def test_custom_weights(self) -> None:
        c = np.full(8, 0.70)
        w = np.array([0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1])
        sig = classify_collapse(c, w)
        assert isinstance(sig, CollapseSignature)


# ═══════════════════════════════════════════════════════════════════
# TIER-1 IDENTITIES HOLD IN ALL STATES
# ═══════════════════════════════════════════════════════════════════


class TestTier1Identities:
    """F + ω = 1, IC ≤ F, IC = exp(κ) across all collapse types."""

    @pytest.mark.parametrize(
        "trace",
        [
            np.full(8, 0.90),  # protected
            np.full(8, 0.30),  # uniform collapse
            np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77]),  # selective
            np.array([EPSILON, EPSILON, 0.60, 0.65, 0.55, 0.70, 0.58, 0.62]),  # 2 dead
            np.full(8, 0.50),  # equator
        ],
    )
    def test_duality_identity(self, trace: np.ndarray) -> None:
        w = np.ones(len(trace)) / len(trace)
        r = compute_kernel_outputs(trace, w)
        assert abs(r["F"] + r["omega"] - 1.0) < 1e-14

    @pytest.mark.parametrize(
        "trace",
        [
            np.full(8, 0.90),
            np.full(8, 0.30),
            np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77]),
            np.array([EPSILON, EPSILON, 0.60, 0.65, 0.55, 0.70, 0.58, 0.62]),
            np.full(8, 0.50),
        ],
    )
    def test_integrity_bound(self, trace: np.ndarray) -> None:
        w = np.ones(len(trace)) / len(trace)
        r = compute_kernel_outputs(trace, w)
        assert r["IC"] <= r["F"] + 1e-14

    @pytest.mark.parametrize(
        "trace",
        [
            np.full(8, 0.90),
            np.full(8, 0.30),
            np.array([EPSILON, 0.80, 0.75, 0.85, 0.70, 0.78, 0.82, 0.77]),
            np.array([EPSILON, EPSILON, 0.60, 0.65, 0.55, 0.70, 0.58, 0.62]),
            np.full(8, 0.50),
        ],
    )
    def test_log_integrity_relation(self, trace: np.ndarray) -> None:
        w = np.ones(len(trace)) / len(trace)
        r = compute_kernel_outputs(trace, w)
        import math

        assert abs(r["IC"] - math.exp(r["kappa"])) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# ALL_THEOREMS LIST
# ═══════════════════════════════════════════════════════════════════


class TestAllTheoremsList:
    """Verify the ALL_THEOREMS list is complete."""

    def test_six_theorem_functions(self) -> None:
        assert len(ALL_THEOREMS) == 6

    def test_all_callable(self) -> None:
        for fn in ALL_THEOREMS:
            assert callable(fn)

    def test_run_all_returns_six(self) -> None:
        results = run_all_theorems()
        assert len(results) == 6
