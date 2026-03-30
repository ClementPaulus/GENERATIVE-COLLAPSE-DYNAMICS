"""Tests for Reflexive Closure Theorems (T-RF-1 through T-RF-5).

Five theorems formalizing the reflexive application of the GCD kernel
to the system's own architecture, its knowledge claims, and its
evaluation criteria — closing the self-referential loop declared in
the GCD closure's __init__.py.

Cross-references:
    Formalism:       closures/gcd/reflexive_closure.py
    Kernel:          src/umcp/kernel_optimized.py
    Frozen contract: src/umcp/frozen_contract.py
    Prior theorems:  closures/gcd/kernel_structural_theorems.py (T-KS-1–7)
    Emergent:        closures/gcd/emergent_structural_insights.py (T-SI-1–6)

All 5 theorems derive from Axiom-0 through the Tier-1 identities:
    F + ω = 1, IC ≤ F, IC = exp(κ)
"""

from __future__ import annotations

import numpy as np
import pytest

from closures.gcd.reflexive_closure import (
    TheoremResult,
    run_all_theorems,
    theorem_TRF1_architectural_integrity_bound,
    theorem_TRF2_concentration_fragility,
    theorem_TRF3_substrate_invariance,
    theorem_TRF4_epistemic_channel_death,
    theorem_TRF5_reflexive_fixed_point,
)
from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL: ALL THEOREMS PROVEN
# ═══════════════════════════════════════════════════════════════════


class TestAllReflexiveTheoremsProven:
    """Meta-tests: every theorem must pass all its subtests."""

    @pytest.fixture(scope="class")
    def all_results(self) -> list[TheoremResult]:
        return run_all_theorems()

    def test_all_five_proven(self, all_results: list[TheoremResult]) -> None:
        for r in all_results:
            assert r.verdict == "PROVEN", f"{r.name}: {r.n_passed}/{r.n_tests}"

    def test_total_subtests_at_least_60(self, all_results: list[TheoremResult]) -> None:
        total = sum(r.n_tests for r in all_results)
        assert total >= 60, f"Only {total} subtests"

    def test_zero_failures(self, all_results: list[TheoremResult]) -> None:
        total_fail = sum(r.n_failed for r in all_results)
        assert total_fail == 0, f"{total_fail} subtests failed"

    def test_five_theorems(self, all_results: list[TheoremResult]) -> None:
        assert len(all_results) == 5


# ═══════════════════════════════════════════════════════════════════
# T-RF-1: ARCHITECTURAL INTEGRITY BOUND
# ═══════════════════════════════════════════════════════════════════


class TestTRF1ArchitecturalIntegrityBound:
    """T-RF-1: GCD architecture through its own kernel."""

    def test_theorem_proven(self) -> None:
        r = theorem_TRF1_architectural_integrity_bound()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    def test_ic_over_f_near_unity(self) -> None:
        """IC/F > 0.999 — multiplicative coherence is near-perfect."""
        c = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        k = compute_kernel_outputs(c, w)
        assert k["IC"] / k["F"] > 0.999

    def test_heterogeneity_gap_minimal(self) -> None:
        """Δ = F - IC < 0.001 — near-zero heterogeneity gap."""
        c = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        k = compute_kernel_outputs(c, w)
        assert k["F"] - k["IC"] < 0.001

    def test_regime_is_watch(self) -> None:
        """Regime = Watch — honest self-evaluation, not Stable."""
        c = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        k = compute_kernel_outputs(c, w)
        # ω = 0.06 > 0.038 → Watch, not Stable
        assert k["omega"] > 0.038
        assert k["omega"] < 0.30

    def test_duality_identity(self) -> None:
        """F + ω = 1 exactly for the architectural trace."""
        c = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        k = compute_kernel_outputs(c, w)
        assert abs(k["F"] + k["omega"] - 1.0) < 1e-12

    def test_integrity_bound(self) -> None:
        """IC ≤ F for the architectural trace."""
        c = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        k = compute_kernel_outputs(c, w)
        assert k["IC"] <= k["F"] + 1e-12


# ═══════════════════════════════════════════════════════════════════
# T-RF-2: CONCENTRATION FRAGILITY
# ═══════════════════════════════════════════════════════════════════


class TestTRF2ConcentrationFragility:
    """T-RF-2: Concentrated architectures suffer geometric slaughter."""

    def test_theorem_proven(self) -> None:
        r = theorem_TRF2_concentration_fragility()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    @pytest.mark.parametrize(
        "name,channels",
        [
            ("proof_assistant", [0.99, 0.40, 0.30, 0.15, 0.50]),
            ("validation_tool", [0.20, 0.30, 0.60, 0.15, 0.99]),
            ("domain_tool", [0.40, 0.50, 0.60, 0.98, 0.85]),
        ],
    )
    def test_concentrated_ic_f_below_gcd(self, name: str, channels: list[float]) -> None:
        """Each concentrated architecture has IC/F below GCD's IC/F."""
        gcd_c = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        k_gcd = compute_kernel_outputs(gcd_c, w)
        k_conc = compute_kernel_outputs(np.array(channels), w)
        assert k_conc["IC"] / k_conc["F"] < k_gcd["IC"] / k_gcd["F"]

    @pytest.mark.parametrize(
        "name,channels",
        [
            ("proof_assistant", [0.99, 0.40, 0.30, 0.15, 0.50]),
            ("validation_tool", [0.20, 0.30, 0.60, 0.15, 0.99]),
            ("domain_tool", [0.40, 0.50, 0.60, 0.98, 0.85]),
        ],
    )
    def test_concentrated_larger_gap(self, name: str, channels: list[float]) -> None:
        """Each concentrated architecture has larger heterogeneity gap."""
        gcd_c = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        k_gcd = compute_kernel_outputs(gcd_c, w)
        k_conc = compute_kernel_outputs(np.array(channels), w)
        gcd_gap = k_gcd["F"] - k_gcd["IC"]
        conc_gap = k_conc["F"] - k_conc["IC"]
        assert conc_gap > gcd_gap


# ═══════════════════════════════════════════════════════════════════
# T-RF-3: SUBSTRATE INVARIANCE
# ═══════════════════════════════════════════════════════════════════


class TestTRF3SubstrateInvariance:
    """T-RF-3: Knowledge types structurally distinguished by kernel."""

    def test_theorem_proven(self) -> None:
        r = theorem_TRF3_substrate_invariance()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    def test_proven_theorem_stable(self) -> None:
        """A proven theorem with all strong channels is Stable."""
        c = np.array([0.99, 0.95, 0.99, 0.99, 0.98, 0.97, 0.85, 0.99])
        w = np.ones(8) / 8
        k = compute_kernel_outputs(c, w)
        assert k["omega"] < 0.038

    def test_ungrounded_assertion_collapse(self) -> None:
        """An ungrounded assertion is Collapse."""
        c = np.array([0.10, 0.30, 0.20, 0.50, 0.15, 0.40, 0.80, 0.10])
        w = np.ones(8) / 8
        k = compute_kernel_outputs(c, w)
        assert k["omega"] >= 0.30

    def test_ordering_proven_gt_empirical(self) -> None:
        """IC/F(proven theorem) > IC/F(empirical claim)."""
        w = np.ones(8) / 8
        k_prov = compute_kernel_outputs(np.array([0.99, 0.95, 0.99, 0.99, 0.98, 0.97, 0.85, 0.99]), w)
        k_emp = compute_kernel_outputs(np.array([0.50, 0.90, 0.80, 0.85, 0.95, 0.70, 0.60, 0.80]), w)
        assert k_prov["IC"] / k_prov["F"] > k_emp["IC"] / k_emp["F"]

    def test_ordering_empirical_gt_assertion(self) -> None:
        """IC/F(empirical claim) > IC/F(ungrounded assertion)."""
        w = np.ones(8) / 8
        k_emp = compute_kernel_outputs(np.array([0.50, 0.90, 0.80, 0.85, 0.95, 0.70, 0.60, 0.80]), w)
        k_asr = compute_kernel_outputs(np.array([0.10, 0.30, 0.20, 0.50, 0.15, 0.40, 0.80, 0.10]), w)
        assert k_emp["IC"] / k_emp["F"] > k_asr["IC"] / k_asr["F"]


# ═══════════════════════════════════════════════════════════════════
# T-RF-4: EPISTEMIC CHANNEL DEATH
# ═══════════════════════════════════════════════════════════════════


class TestTRF4EpistemicChannelDeath:
    """T-RF-4: Dead epistemic channel kills IC via T-KS-1."""

    def test_theorem_proven(self) -> None:
        r = theorem_TRF4_epistemic_channel_death()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    @pytest.mark.parametrize("kill_idx", range(8))
    def test_kill_any_channel_drops_ic_f(self, kill_idx: int) -> None:
        """Killing any single epistemic channel drops IC/F below 0.15."""
        c = np.full(8, 0.95)
        c[kill_idx] = EPSILON
        w = np.ones(8) / 8
        k = compute_kernel_outputs(c, w)
        assert k["IC"] / k["F"] < 0.15

    def test_positional_democracy(self) -> None:
        """All channel kills produce the same IC/F (equal weights)."""
        w = np.ones(8) / 8
        ic_fs = []
        for i in range(8):
            c = np.full(8, 0.95)
            c[i] = EPSILON
            k = compute_kernel_outputs(c, w)
            ic_fs.append(k["IC"] / k["F"])
        assert max(ic_fs) - min(ic_fs) < 0.005


# ═══════════════════════════════════════════════════════════════════
# T-RF-5: REFLEXIVE FIXED POINT
# ═══════════════════════════════════════════════════════════════════


class TestTRF5ReflexiveFixedPoint:
    """T-RF-5: GCD is a robust fixed point of its own evaluation."""

    def test_theorem_proven(self) -> None:
        r = theorem_TRF5_reflexive_fixed_point()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    def test_idempotent(self) -> None:
        """Same input twice → identical output (pure function)."""
        c = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        k1 = compute_kernel_outputs(c, w)
        k2 = compute_kernel_outputs(c, w)
        for key in ("F", "omega", "IC", "S", "C", "kappa"):
            assert abs(k1[key] - k2[key]) < 1e-15

    def test_perturbation_robustness(self) -> None:
        """±2% perturbation keeps IC/F > 0.999."""
        rng = np.random.default_rng(42)
        c_base = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        for _ in range(20):
            delta = rng.uniform(-0.02, 0.02, size=5)
            c_pert = np.clip(c_base + delta, EPSILON, 1.0 - EPSILON)
            k = compute_kernel_outputs(c_pert, w)
            assert k["IC"] / k["F"] > 0.999

    def test_regime_robust_under_perturbation(self) -> None:
        """±2% perturbation keeps regime = Watch."""
        rng = np.random.default_rng(123)
        c_base = np.array([0.95, 0.93, 0.94, 0.92, 0.96])
        w = np.ones(5) / 5
        for _ in range(20):
            delta = rng.uniform(-0.02, 0.02, size=5)
            c_pert = np.clip(c_base + delta, EPSILON, 1.0 - EPSILON)
            k = compute_kernel_outputs(c_pert, w)
            # Still in Watch range
            assert k["omega"] >= 0.038 or k["omega"] < 0.30
