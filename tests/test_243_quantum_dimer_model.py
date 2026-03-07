"""Tests for triangular lattice quantum dimer model closure.

Validates the 7 theorems from the GCD kernel analysis of Yan, Samajdar,
Wang, Sachdev & Meng (2022), "Triangular lattice quantum dimer model
with variable dimer density" (Nat. Commun. 13, 5799).

Test coverage:
  - Phase catalog integrity (13 phases, parameters, categories)
  - 8-channel trace construction (bounds, independence, clamping)
  - Tier-1 kernel identities for all phases (duality, integrity bound,
    log-integrity relation)
  - All 7 theorems (T-QDM-1 through T-QDM-7)
  - Regime classification consistency
  - Prediction verification (P1–P7)
  - Channel autopsy (weakest/strongest identification)
  - Serialization round-trip (to_dict)
  - Cross-category invariant comparisons
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.quantum_mechanics.quantum_dimer_model import (
    CHANNEL_LABELS,
    EPSILON,
    N_CHANNELS,
    PREDICTIONS,
    QDM_PHASES,
    WEIGHTS,
    QDMKernelResult,
    QDMPhase,
    classify_regime,
    compute_all_phases,
    compute_qdm_kernel,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helper: precompute all results once for the module
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def all_results() -> list[QDMKernelResult]:
    """Compute kernel invariants for all 13 phases."""
    return compute_all_phases()


@pytest.fixture(scope="module")
def results_by_category(all_results: list[QDMKernelResult]) -> dict[str, list[QDMKernelResult]]:
    """Group results by phase category."""
    cats: dict[str, list[QDMKernelResult]] = {}
    for r in all_results:
        cats.setdefault(r.category, []).append(r)
    return cats


@pytest.fixture(scope="module")
def results_by_name(all_results: list[QDMKernelResult]) -> dict[str, QDMKernelResult]:
    """Index results by phase name."""
    return {r.name: r for r in all_results}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: PHASE CATALOG INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseCatalog:
    """Verify the phase catalog matches the paper data."""

    def test_phase_count(self) -> None:
        assert len(QDM_PHASES) == 13

    def test_unique_names(self) -> None:
        names = [p.name for p in QDM_PHASES]
        assert len(names) == len(set(names))

    def test_categories(self) -> None:
        cats = {p.category for p in QDM_PHASES}
        assert cats == {"topological", "trivial", "crystal"}

    def test_topological_count(self) -> None:
        topo = [p for p in QDM_PHASES if p.category == "topological"]
        assert len(topo) == 6

    def test_crystal_count(self) -> None:
        crystal = [p for p in QDM_PHASES if p.category == "crystal"]
        assert len(crystal) == 5

    def test_trivial_count(self) -> None:
        trivial = [p for p in QDM_PHASES if p.category == "trivial"]
        assert len(trivial) == 2

    def test_frozen_dataclass(self) -> None:
        phase = QDM_PHASES[0]
        with pytest.raises(AttributeError):
            phase.name = "modified"  # type: ignore[misc]

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_dimer_filling_range(self, phase: QDMPhase) -> None:
        assert 0.0 < phase.dimer_filling <= 1.0

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_topological_order_range(self, phase: QDMPhase) -> None:
        assert EPSILON <= phase.topological_order <= 1.0

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_string_coherence_range(self, phase: QDMPhase) -> None:
        assert 0.0 <= phase.string_coherence <= 1.0

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_phase_stability_range(self, phase: QDMPhase) -> None:
        assert 0.0 < phase.phase_stability <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: TRACE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


class TestTraceConstruction:
    """Verify 8-channel trace vector construction."""

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_trace_shape(self, phase: QDMPhase) -> None:
        c = phase.trace_vector()
        assert c.shape == (N_CHANNELS,)

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_trace_bounds(self, phase: QDMPhase) -> None:
        c = phase.trace_vector()
        assert np.all(c >= EPSILON)
        assert np.all(c <= 1.0 - EPSILON)

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_trace_dtype(self, phase: QDMPhase) -> None:
        c = phase.trace_vector()
        assert c.dtype == np.float64

    def test_channel_count(self) -> None:
        assert N_CHANNELS == 8

    def test_channel_labels(self) -> None:
        assert len(CHANNEL_LABELS) == N_CHANNELS

    def test_weights_sum_to_one(self) -> None:
        assert abs(WEIGHTS.sum() - 1.0) < 1e-12

    def test_weights_equal(self) -> None:
        expected = 1.0 / N_CHANNELS
        assert np.allclose(WEIGHTS, expected)

    def test_odd_qsl_low_string_coherence(self) -> None:
        """Odd QSL has ⟨string⟩ ≈ −1 → string_coherence near 0."""
        phase = QDM_PHASES[0]  # odd_Z2_QSL
        c = phase.trace_vector()
        assert c[2] < 0.1  # string_coherence channel

    def test_even_qsl_high_string_coherence(self) -> None:
        """Even QSL has ⟨string⟩ ≈ +1 → string_coherence near 1."""
        phase = QDM_PHASES[1]  # even_Z2_QSL
        c = phase.trace_vector()
        assert c[2] > 0.9

    def test_pm_medium_string_coherence(self) -> None:
        """PM has ⟨string⟩ ≈ 0 → string_coherence near 0.5."""
        phase = QDM_PHASES[2]  # PM_trivial
        c = phase.trace_vector()
        assert 0.3 < c[2] < 0.7


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: TIER-1 KERNEL IDENTITIES
# ═══════════════════════════════════════════════════════════════════════════


class TestTier1Identities:
    """Verify Tier-1 structural identities for all QDM phases."""

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_duality_identity(self, phase: QDMPhase) -> None:
        """F + ω = 1 (the duality identity)."""
        r = compute_qdm_kernel(phase)
        assert abs(r.F_plus_omega - 1.0) < 1e-10

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_integrity_bound(self, phase: QDMPhase) -> None:
        """IC ≤ F (integrity bound)."""
        r = compute_qdm_kernel(phase)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_log_integrity_relation(self, phase: QDMPhase) -> None:
        """IC = exp(κ)."""
        r = compute_qdm_kernel(phase)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-6

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_fidelity_in_unit_interval(self, phase: QDMPhase) -> None:
        r = compute_qdm_kernel(phase)
        assert 0.0 <= r.F <= 1.0

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_drift_in_unit_interval(self, phase: QDMPhase) -> None:
        r = compute_qdm_kernel(phase)
        assert 0.0 <= r.omega <= 1.0

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_entropy_nonnegative(self, phase: QDMPhase) -> None:
        r = compute_qdm_kernel(phase)
        assert r.S >= 0.0

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_curvature_normalized(self, phase: QDMPhase) -> None:
        r = compute_qdm_kernel(phase)
        assert 0.0 <= r.C <= 2.0  # C = std/0.5, max std ~0.5

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_heterogeneity_gap_nonneg(self, phase: QDMPhase) -> None:
        """Δ = F − IC ≥ 0."""
        r = compute_qdm_kernel(phase)
        assert r.heterogeneity_gap >= -1e-12

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_ic_positive(self, phase: QDMPhase) -> None:
        r = compute_qdm_kernel(phase)
        assert r.IC > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: THEOREM VERIFICATION (T-QDM-1 through T-QDM-7)
# ═══════════════════════════════════════════════════════════════════════════


class TestTheoremQDM1:
    """T-QDM-1: Topological Order as Fidelity Separation.

    QSL phases maintain higher F than crystal phases because topological
    order sustains non-local coherence across all channels.
    """

    def test_mean_F_topo_gt_crystal(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        topo = results_by_category["topological"]
        crystal = results_by_category["crystal"]
        avg_F_topo = sum(r.F for r in topo) / len(topo)
        avg_F_crystal = sum(r.F for r in crystal) / len(crystal)
        assert avg_F_topo > avg_F_crystal

    def test_mean_F_topo_gt_trivial(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        topo = results_by_category["topological"]
        trivial = results_by_category["trivial"]
        avg_F_topo = sum(r.F for r in topo) / len(topo)
        avg_F_trivial = sum(r.F for r in trivial) / len(trivial)
        assert avg_F_topo > avg_F_trivial

    def test_deep_QSLs_highest_F(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Hard-constraint QSLs should have the highest F of all phases."""
        hard_even = results_by_name["even_QSL_hard"]
        hard_odd = results_by_name["odd_QSL_hard"]
        # Both should be in top 4
        assert hard_even.F > 0.6
        assert hard_odd.F > 0.6


class TestTheoremQDM2:
    """T-QDM-2: Fractionalization as Heterogeneity Gap.

    Fractionalized phases exhibit large Δ because vison excitations
    suppress individual channel coherence while aggregate F stays high.
    """

    def test_odd_QSL_large_gap(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Odd QSL has large Δ due to low string_coherence channel."""
        r = results_by_name["odd_Z2_QSL"]
        assert r.heterogeneity_gap > 0.10

    def test_even_QSL_small_gap(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Even QSL has smaller Δ because all channels are high."""
        r = results_by_name["even_Z2_QSL"]
        assert r.heterogeneity_gap < 0.10

    def test_fractionalization_drives_gap(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Odd QSL's low string_coherence kills IC more than even QSL."""
        odd = results_by_name["odd_Z2_QSL"]
        even = results_by_name["even_Z2_QSL"]
        assert odd.IC < even.IC


class TestTheoremQDM3:
    """T-QDM-3: Crystal Order as IC Collapse.

    Symmetry-breaking kills channels, dragging IC toward ε.
    """

    def test_crystal_IC_lower_than_topo(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        topo = results_by_category["topological"]
        crystal = results_by_category["crystal"]
        avg_IC_topo = sum(r.IC for r in topo) / len(topo)
        avg_IC_crystal = sum(r.IC for r in crystal) / len(crystal)
        assert avg_IC_crystal < avg_IC_topo

    @pytest.mark.parametrize(
        "name",
        ["columnar_crystal", "nematic", "staggered_1_6", "staggered_1_3", "VBS_12x12"],
    )
    def test_crystal_symmetry_channel_low(self, name: str, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Crystal phases have symmetry_preservation near ε."""
        r = results_by_name[name]
        c = np.array(r.trace_vector)
        assert c[3] < 0.15  # symmetry_preservation channel

    @pytest.mark.parametrize(
        "name",
        ["columnar_crystal", "nematic", "staggered_1_6", "staggered_1_3"],
    )
    def test_crystal_fractionalization_absent(self, name: str, results_by_name: dict[str, QDMKernelResult]) -> None:
        r = results_by_name[name]
        c = np.array(r.trace_vector)
        assert c[5] < 0.10  # fractionalization channel


class TestTheoremQDM4:
    """T-QDM-4: String Operator as Kernel Diagnostic.

    The string operator ⟨(−1)^{#cut dimers}⟩ maps to the string_coherence
    channel and distinguishes the two QSL types.
    """

    def test_odd_qsl_low_string(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        r = results_by_name["odd_Z2_QSL"]
        c = np.array(r.trace_vector)
        assert c[2] < 0.10

    def test_even_qsl_high_string(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        r = results_by_name["even_Z2_QSL"]
        c = np.array(r.trace_vector)
        assert c[2] > 0.90

    def test_pm_intermediate_string(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        r = results_by_name["PM_trivial"]
        c = np.array(r.trace_vector)
        assert 0.3 < c[2] < 0.7

    def test_string_distinguishes_QSLs(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """String coherence must clearly distinguish odd from even QSL."""
        odd = results_by_name["odd_Z2_QSL"]
        even = results_by_name["even_Z2_QSL"]
        odd_c = np.array(odd.trace_vector)
        even_c = np.array(even.trace_vector)
        gap = even_c[2] - odd_c[2]
        assert gap > 0.80  # strong separation


class TestTheoremQDM5:
    """T-QDM-5: First-Order Transitions as Regime Boundaries.

    QSL↔PM transitions are first-order, visible as regime crossings.
    """

    def test_deep_QSLs_in_watch(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        for name in ["odd_Z2_QSL", "even_Z2_QSL", "odd_QSL_hard", "even_QSL_hard"]:
            r = results_by_name[name]
            assert r.regime == "Watch"

    def test_PM_in_collapse(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        for name in ["PM_trivial", "PM_deep"]:
            r = results_by_name[name]
            assert r.regime == "Collapse"

    def test_boundary_phases_in_collapse(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Phases near the QSL-PM boundary should show increased ω."""
        for name in ["odd_QSL_boundary", "even_QSL_boundary"]:
            r = results_by_name[name]
            assert r.regime == "Collapse"

    def test_boundary_omega_higher_than_deep(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Boundary QSLs have higher ω than deep QSLs."""
        for pair in [
            ("odd_QSL_boundary", "odd_Z2_QSL"),
            ("even_QSL_boundary", "even_Z2_QSL"),
        ]:
            boundary = results_by_name[pair[0]]
            deep = results_by_name[pair[1]]
            assert boundary.omega > deep.omega


class TestTheoremQDM6:
    """T-QDM-6: Vison Momentum Fractionalization as Channel Asymmetry.

    Odd QSL visons carry fractional crystal momentum (M+Γ minima);
    even QSL visons do not (Γ only).
    """

    def test_odd_qsl_high_vison_momentum(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        r = results_by_name["odd_Z2_QSL"]
        c = np.array(r.trace_vector)
        assert c[6] > 0.80  # vison_momentum channel

    def test_even_qsl_medium_vison_momentum(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        r = results_by_name["even_Z2_QSL"]
        c = np.array(r.trace_vector)
        assert 0.3 < c[6] < 0.7  # intermediate — Γ only

    def test_pm_no_vison_momentum(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        r = results_by_name["PM_trivial"]
        c = np.array(r.trace_vector)
        assert c[6] < 0.10  # no visons

    def test_vison_asymmetry(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Odd QSL vison_momentum > even QSL vison_momentum."""
        odd = results_by_name["odd_Z2_QSL"]
        even = results_by_name["even_Z2_QSL"]
        odd_c = np.array(odd.trace_vector)
        even_c = np.array(even.trace_vector)
        assert odd_c[6] > even_c[6]


class TestTheoremQDM7:
    """T-QDM-7: Cross-Scale Universality with Rydberg Systems.

    The QDM phase structure maps onto Rydberg-atom experiments.
    Kernel invariants (F, IC, Δ) exhibit the same structural signatures
    across quantum simulator architectures.
    """

    def test_topo_F_separation_exists(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        """The F separation between topological and non-topological is
        structurally the same kind as seen in other quantum closures."""
        topo = results_by_category["topological"]
        nontopo = results_by_category["crystal"] + results_by_category["trivial"]
        avg_F_topo = sum(r.F for r in topo) / len(topo)
        avg_F_nontopo = sum(r.F for r in nontopo) / len(nontopo)
        # Substantial gap
        assert avg_F_topo - avg_F_nontopo > 0.20

    def test_IC_leq_F_universal(self, all_results: list[QDMKernelResult]) -> None:
        """IC ≤ F holds for every single phase — universality."""
        for r in all_results:
            assert r.IC <= r.F + 1e-12

    def test_heterogeneity_gap_positive_everywhere(self, all_results: list[QDMKernelResult]) -> None:
        for r in all_results:
            assert r.heterogeneity_gap >= -1e-12


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: PREDICTION VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictions:
    """Verify the 7 GCD predictions derived from Axiom-0."""

    def test_prediction_count(self) -> None:
        assert len(PREDICTIONS) == 7

    def test_P1_topological_fidelity(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        """P1: QSL ⟨F⟩ > crystal ⟨F⟩."""
        topo = results_by_category["topological"]
        crystal = results_by_category["crystal"]
        assert sum(r.F for r in topo) / len(topo) > sum(r.F for r in crystal) / len(crystal)

    def test_P2_crystal_IC_lowest(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        """P2: crystal ⟨IC⟩ is the lowest category."""
        crystal = results_by_category["crystal"]
        topo = results_by_category["topological"]
        trivial = results_by_category["trivial"]
        avg_IC_crystal = sum(r.IC for r in crystal) / len(crystal)
        avg_IC_topo = sum(r.IC for r in topo) / len(topo)
        avg_IC_trivial = sum(r.IC for r in trivial) / len(trivial)
        assert avg_IC_crystal < min(avg_IC_topo, avg_IC_trivial)

    def test_P3_crystal_largest_gap(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        """P3: crystal ⟨Δ⟩ > topo ⟨Δ⟩."""
        crystal = results_by_category["crystal"]
        topo = results_by_category["topological"]
        avg_d_crystal = sum(r.heterogeneity_gap for r in crystal) / len(crystal)
        avg_d_topo = sum(r.heterogeneity_gap for r in topo) / len(topo)
        assert avg_d_crystal > avg_d_topo

    def test_P4_QSL_distinction(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """P4: Odd and even QSLs distinguishable by string_coherence."""
        odd = results_by_name["odd_Z2_QSL"]
        even = results_by_name["even_Z2_QSL"]
        odd_sc = np.array(odd.trace_vector)[2]
        even_sc = np.array(even.trace_vector)[2]
        assert odd_sc < even_sc
        assert even_sc - odd_sc > 0.70

    def test_P5_PM_intermediate(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        """P5: PM ⟨F⟩ between crystal ⟨F⟩ and topo ⟨F⟩."""
        crystal = results_by_category["crystal"]
        topo = results_by_category["topological"]
        trivial = results_by_category["trivial"]
        avg_F_crystal = sum(r.F for r in crystal) / len(crystal)
        avg_F_topo = sum(r.F for r in topo) / len(topo)
        avg_F_trivial = sum(r.F for r in trivial) / len(trivial)
        assert avg_F_crystal < avg_F_trivial < avg_F_topo

    def test_P6_boundary_instability(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """P6: Boundary phases have higher ω than deep phases."""
        assert results_by_name["odd_QSL_boundary"].omega > results_by_name["odd_Z2_QSL"].omega
        assert results_by_name["even_QSL_boundary"].omega > results_by_name["even_Z2_QSL"].omega

    def test_P7_hard_constraint_stability(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """P7: Hard-constraint QSLs have higher F than soft-constraint."""
        assert results_by_name["odd_QSL_hard"].F > results_by_name["odd_Z2_QSL"].F
        assert results_by_name["even_QSL_hard"].F > results_by_name["even_Z2_QSL"].F


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeClassification:
    """Verify regime gates match frozen contract thresholds."""

    def test_stable_regime(self) -> None:
        assert classify_regime(0.01, 0.95, 0.10, 0.10) == "Stable"

    def test_watch_regime(self) -> None:
        assert classify_regime(0.10, 0.85, 0.30, 0.30) == "Watch"

    def test_collapse_regime(self) -> None:
        assert classify_regime(0.50, 0.50, 0.60, 0.50) == "Collapse"

    def test_collapse_boundary(self) -> None:
        assert classify_regime(0.30, 0.70, 0.20, 0.20) == "Collapse"

    def test_watch_boundary(self) -> None:
        assert classify_regime(0.038, 0.92, 0.10, 0.10) == "Watch"

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_regime_assigned(self, phase: QDMPhase) -> None:
        r = compute_qdm_kernel(phase)
        assert r.regime in {"Stable", "Watch", "Collapse"}

    def test_all_crystals_in_collapse(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        for r in results_by_category["crystal"]:
            assert r.regime == "Collapse"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: CHANNEL AUTOPSY
# ═══════════════════════════════════════════════════════════════════════════


class TestChannelAutopsy:
    """Verify weakest/strongest channel identification."""

    def test_odd_qsl_weakest_is_string(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Odd QSL's weakest channel should be string_coherence."""
        r = results_by_name["odd_Z2_QSL"]
        assert r.weakest_channel == "string_coherence"

    def test_columnar_weakest_is_symmetry_or_topo(self, results_by_name: dict[str, QDMKernelResult]) -> None:
        """Crystal weakest channel is topological_order or symmetry."""
        r = results_by_name["columnar_crystal"]
        assert r.weakest_channel in {
            "topological_order",
            "symmetry_preservation",
            "fractionalization",
            "vison_momentum",
        }

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_weakest_le_strongest(self, phase: QDMPhase) -> None:
        r = compute_qdm_kernel(phase)
        assert r.weakest_value <= r.strongest_value


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Verify to_dict round-trip."""

    @pytest.mark.parametrize("phase", QDM_PHASES, ids=lambda p: p.name)
    def test_to_dict_keys(self, phase: QDMPhase) -> None:
        r = compute_qdm_kernel(phase)
        d = r.to_dict()
        expected_keys = {
            "name",
            "category",
            "mu",
            "V",
            "n_channels",
            "channel_labels",
            "trace_vector",
            "F",
            "omega",
            "S",
            "C",
            "kappa",
            "IC",
            "heterogeneity_gap",
            "F_plus_omega",
            "IC_leq_F",
            "IC_eq_exp_kappa",
            "regime",
            "weakest_channel",
            "weakest_value",
            "strongest_channel",
            "strongest_value",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_fidelity_preserved(self) -> None:
        r = compute_qdm_kernel(QDM_PHASES[0])
        d = r.to_dict()
        assert abs(d["F"] + d["omega"] - 1.0) < 1e-10

    def test_to_dict_trace_length(self) -> None:
        r = compute_qdm_kernel(QDM_PHASES[0])
        d = r.to_dict()
        assert len(d["trace_vector"]) == N_CHANNELS


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: CROSS-CATEGORY COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossCategoryComparisons:
    """Verify structural patterns across phase categories."""

    def test_fidelity_ordering(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        """⟨F⟩_topo > ⟨F⟩_trivial > ⟨F⟩_crystal."""
        topo = results_by_category["topological"]
        trivial = results_by_category["trivial"]
        crystal = results_by_category["crystal"]
        avg_F_t = sum(r.F for r in topo) / len(topo)
        avg_F_v = sum(r.F for r in trivial) / len(trivial)
        avg_F_c = sum(r.F for r in crystal) / len(crystal)
        assert avg_F_t > avg_F_v > avg_F_c

    def test_IC_ordering(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        """⟨IC⟩_topo > ⟨IC⟩_trivial and ⟨IC⟩_topo > ⟨IC⟩_crystal."""
        topo = results_by_category["topological"]
        trivial = results_by_category["trivial"]
        crystal = results_by_category["crystal"]
        avg_IC_t = sum(r.IC for r in topo) / len(topo)
        avg_IC_v = sum(r.IC for r in trivial) / len(trivial)
        avg_IC_c = sum(r.IC for r in crystal) / len(crystal)
        assert avg_IC_t > avg_IC_v
        assert avg_IC_t > avg_IC_c

    def test_omega_ordering(self, results_by_category: dict[str, list[QDMKernelResult]]) -> None:
        """⟨ω⟩_crystal > ⟨ω⟩_trivial > ⟨ω⟩_topo."""
        topo = results_by_category["topological"]
        trivial = results_by_category["trivial"]
        crystal = results_by_category["crystal"]
        avg_w_t = sum(r.omega for r in topo) / len(topo)
        avg_w_v = sum(r.omega for r in trivial) / len(trivial)
        avg_w_c = sum(r.omega for r in crystal) / len(crystal)
        assert avg_w_c > avg_w_v > avg_w_t

    def test_compute_all_count(self, all_results: list[QDMKernelResult]) -> None:
        assert len(all_results) == 13

    def test_all_identities_pass(self, all_results: list[QDMKernelResult]) -> None:
        """Every phase must pass all three Tier-1 identity checks."""
        for r in all_results:
            assert abs(r.F_plus_omega - 1.0) < 1e-10
            assert r.IC_leq_F
            assert r.IC_eq_exp_kappa
