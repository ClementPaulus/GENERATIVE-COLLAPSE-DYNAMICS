"""Tests for FQHE bilayer graphene closure.

Validates the 7 theorems from the GCD kernel analysis of Kim, Dev, Shaer,
Kumar, Ilin, Haug, Iskoz, Watanabe, Taniguchi, Mross, Stern & Ronen (2026),
"Aharonov-Bohm interference in even-denominator fractional quantum Hall
states" (Nature 649, 323-329).  DOI: 10.1038/s41586-025-09891-2

Test coverage:
  - State catalog integrity (11 states, parameters, categories)
  - 8-channel trace construction (bounds, independence, clamping)
  - Tier-1 kernel identities for all states (duality, integrity bound,
    log-integrity relation)
  - All 7 theorems (T-FQHE-1 through T-FQHE-7)
  - Regime classification consistency
  - Prediction verification (P1–P7)
  - Channel autopsy (weakest/strongest identification)
  - Serialization round-trip (to_dict)
  - Cross-category invariant comparisons
  - Experimental data verification (charges, flux periodicities)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.quantum_mechanics.fqhe_bilayer_graphene import (
    CHANNEL_LABELS,
    EPSILON,
    FQHE_STATES,
    N_CHANNELS,
    PREDICTIONS,
    WEIGHTS,
    FQHEKernelResult,
    FQHState,
    classify_regime,
    compute_all_states,
    compute_fqhe_kernel,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helper: precompute all results once for the module
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def all_results() -> list[FQHEKernelResult]:
    """Compute kernel invariants for all 11 FQH states."""
    return compute_all_states()


@pytest.fixture(scope="module")
def results_by_name(all_results: list[FQHEKernelResult]) -> dict[str, FQHEKernelResult]:
    """Index results by state name."""
    return {r.name: r for r in all_results}


@pytest.fixture(scope="module")
def even_results(all_results: list[FQHEKernelResult]) -> list[FQHEKernelResult]:
    """Even-denominator states only."""
    return [r for r in all_results if r.is_even_denom]


@pytest.fixture(scope="module")
def odd_results(all_results: list[FQHEKernelResult]) -> list[FQHEKernelResult]:
    """Odd-denominator states only."""
    return [r for r in all_results if not r.is_even_denom]


@pytest.fixture(scope="module")
def measured_results(all_results: list[FQHEKernelResult]) -> list[FQHEKernelResult]:
    """States with direct AB interference measurements (6 states)."""
    measured_names = {"nu_neg_1_2", "nu_3_2", "nu_neg_1_3", "nu_4_3", "nu_neg_2_3", "nu_5_3"}
    return [r for r in all_results if r.name in measured_names]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: STATE CATALOG INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════


class TestStateCatalog:
    """Verify the state catalog matches Kim et al. 2026 data."""

    def test_state_count(self) -> None:
        assert len(FQHE_STATES) == 11

    def test_unique_names(self) -> None:
        names = [s.name for s in FQHE_STATES]
        assert len(names) == len(set(names))

    def test_even_denominator_count(self) -> None:
        even = [s for s in FQHE_STATES if s.is_even_denom]
        assert len(even) == 7

    def test_odd_denominator_count(self) -> None:
        odd = [s for s in FQHE_STATES if not s.is_even_denom]
        assert len(odd) == 4

    def test_measured_states_count(self) -> None:
        """Six states have direct AB measurements in the paper."""
        measured = {"nu_neg_1_2", "nu_3_2", "nu_neg_1_3", "nu_4_3", "nu_neg_2_3", "nu_5_3"}
        catalog_names = {s.name for s in FQHE_STATES}
        assert measured.issubset(catalog_names)

    def test_frozen_dataclass(self) -> None:
        state = FQHE_STATES[0]
        with pytest.raises(AttributeError):
            state.name = "modified"  # type: ignore[misc]

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_filling_abs_positive(self, state: FQHState) -> None:
        assert state.filling_abs > 0

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_charge_in_unit_interval(self, state: FQHState) -> None:
        assert 0.0 < state.charge_e_star <= 1.0

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_flux_period_positive(self, state: FQHState) -> None:
        assert state.flux_period_phi0 > 0

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_edge_modes_positive(self, state: FQHState) -> None:
        assert state.n_edge_modes >= 1

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_edge_modes_max(self, state: FQHState) -> None:
        assert state.n_edge_modes <= state.n_edge_modes_max

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_landau_level_filling(self, state: FQHState) -> None:
        """ν_LL must be the fractional part of |ν|."""
        expected = abs(state.filling_factor) - math.floor(abs(state.filling_factor))
        assert abs(state.landau_level_filling - expected) < 1e-10


class TestExperimentalData:
    """Verify specific measured values from Kim et al. 2026."""

    def test_nu_neg_1_2_flux_period(self) -> None:
        """ν = −1/2: ΔΦ = (1.89 ± 0.26)Φ₀"""
        s = next(s for s in FQHE_STATES if s.name == "nu_neg_1_2")
        assert abs(s.flux_period_phi0 - 1.89) < 1e-6
        assert abs(s.flux_period_err - 0.26) < 1e-6

    def test_nu_neg_1_2_visibility(self) -> None:
        """ν = −1/2: visibility ~1.9%"""
        s = next(s for s in FQHE_STATES if s.name == "nu_neg_1_2")
        assert abs(s.visibility_pct - 1.9) < 1e-6

    def test_nu_neg_1_2_charge(self) -> None:
        """ν = −1/2: e* = (1/2)e"""
        s = next(s for s in FQHE_STATES if s.name == "nu_neg_1_2")
        assert abs(s.charge_e_star - 0.5) < 1e-10

    def test_nu_3_2_flux_period(self) -> None:
        """ν = 3/2: ΔΦ = (2.35 ± 0.78)Φ₀"""
        s = next(s for s in FQHE_STATES if s.name == "nu_3_2")
        assert abs(s.flux_period_phi0 - 2.35) < 1e-6
        assert abs(s.flux_period_err - 0.78) < 1e-6

    def test_nu_3_2_visibility(self) -> None:
        """ν = 3/2: visibility ~5.6%"""
        s = next(s for s in FQHE_STATES if s.name == "nu_3_2")
        assert abs(s.visibility_pct - 5.6) < 1e-6

    def test_nu_neg_1_3_charge(self) -> None:
        """ν = −1/3: e* = (1/3)e (Laughlin fundamental)."""
        s = next(s for s in FQHE_STATES if s.name == "nu_neg_1_3")
        assert abs(s.charge_e_star - 1.0 / 3.0) < 1e-10

    def test_nu_neg_2_3_charge_non_fundamental(self) -> None:
        """ν = −2/3: e* = (2/3)e, NOT the fundamental (1/3)e."""
        s = next(s for s in FQHE_STATES if s.name == "nu_neg_2_3")
        assert abs(s.charge_e_star - 2.0 / 3.0) < 1e-10
        assert not s.is_fundamental_charge

    def test_nu_5_3_charge_non_fundamental(self) -> None:
        """ν = 5/3: e* = (2/3)e, NOT the fundamental (1/3)e."""
        s = next(s for s in FQHE_STATES if s.name == "nu_5_3")
        assert abs(s.charge_e_star - 2.0 / 3.0) < 1e-10
        assert not s.is_fundamental_charge

    def test_nu_neg_2_3_flux_period(self) -> None:
        """ν = −2/3: ΔΦ = (3/2)Φ₀."""
        s = next(s for s in FQHE_STATES if s.name == "nu_neg_2_3")
        assert abs(s.flux_period_phi0 - 1.5) < 1e-10

    def test_nu_neg_1_3_flux_period(self) -> None:
        """ν = −1/3: ΔΦ = 3Φ₀."""
        s = next(s for s in FQHE_STATES if s.name == "nu_neg_1_3")
        assert abs(s.flux_period_phi0 - 3.0) < 1e-10

    def test_e_star_equals_nu_LL(self) -> None:
        """Universal pattern: e* = ν_LL × e for all 6 measured states."""
        measured = ["nu_neg_1_2", "nu_3_2", "nu_neg_1_3", "nu_4_3", "nu_neg_2_3", "nu_5_3"]
        for name in measured:
            s = next(s for s in FQHE_STATES if s.name == name)
            assert abs(s.charge_e_star - s.landau_level_filling) < 1e-10, (
                f"{name}: e*={s.charge_e_star} ≠ ν_LL={s.landau_level_filling}"
            )

    def test_even_denom_statistics_ambiguous(self) -> None:
        """Even-denominator states have ambiguous statistics."""
        for s in FQHE_STATES:
            if s.is_even_denom:
                assert s.statistics == "ambiguous", f"{s.name} should be ambiguous"

    def test_odd_denom_statistics_abelian(self) -> None:
        """Odd-denominator Laughlin states have Abelian statistics."""
        for s in FQHE_STATES:
            if not s.is_even_denom:
                assert s.statistics == "Abelian", f"{s.name} should be Abelian"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: TRACE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


class TestTraceConstruction:
    """Verify 8-channel trace vector construction."""

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_trace_shape(self, state: FQHState) -> None:
        c = state.trace_vector()
        assert c.shape == (N_CHANNELS,)

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_trace_bounds(self, state: FQHState) -> None:
        c = state.trace_vector()
        assert np.all(c >= EPSILON)
        assert np.all(c <= 1.0 - EPSILON)

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_trace_dtype(self, state: FQHState) -> None:
        c = state.trace_vector()
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

    def test_even_denom_topological_order_high(self) -> None:
        """Even-denominator states have topological_order = 1."""
        for s in FQHE_STATES:
            if s.is_even_denom:
                c = s.trace_vector()
                assert c[5] > 0.9, f"{s.name}: topological_order should be ~1"

    def test_odd_denom_topological_order_medium(self) -> None:
        """Odd-denominator states have topological_order = 0.5."""
        for s in FQHE_STATES:
            if not s.is_even_denom:
                c = s.trace_vector()
                assert abs(c[5] - 0.5) < 0.01, f"{s.name}: topological_order should be ~0.5"

    def test_hole_conjugate_charge_fundamental_epsilon(self) -> None:
        """Hole-conjugate states have charge_fundamental near ε."""
        for s in FQHE_STATES:
            if not s.is_fundamental_charge:
                c = s.trace_vector()
                assert c[2] < 0.001, f"{s.name}: charge_fundamental should be ~ε"

    def test_particle_like_charge_fundamental_high(self) -> None:
        """Particle-like states with fundamental charge have c[2] near 1."""
        for s in FQHE_STATES:
            if s.is_fundamental_charge and not s.is_even_denom:
                c = s.trace_vector()
                assert c[2] > 0.9, f"{s.name}: charge_fundamental should be ~1"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: TIER-1 KERNEL IDENTITIES
# ═══════════════════════════════════════════════════════════════════════════


class TestTier1Identities:
    """Verify Tier-1 structural identities for all FQH states."""

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_duality_identity(self, state: FQHState) -> None:
        """F + ω = 1 (the duality identity)."""
        r = compute_fqhe_kernel(state)
        assert abs(r.F_plus_omega - 1.0) < 1e-10

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_integrity_bound(self, state: FQHState) -> None:
        """IC ≤ F (integrity bound)."""
        r = compute_fqhe_kernel(state)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_log_integrity_relation(self, state: FQHState) -> None:
        """IC = exp(κ)."""
        r = compute_fqhe_kernel(state)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-6

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_fidelity_in_unit_interval(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert 0.0 <= r.F <= 1.0

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_drift_in_unit_interval(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert 0.0 <= r.omega <= 1.0

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_entropy_nonnegative(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert r.S >= 0.0

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_curvature_normalized(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert 0.0 <= r.C <= 2.0

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_heterogeneity_gap_nonneg(self, state: FQHState) -> None:
        """Δ = F − IC ≥ 0."""
        r = compute_fqhe_kernel(state)
        assert r.heterogeneity_gap >= -1e-12

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_ic_positive(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert r.IC > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: THEOREM VERIFICATION (T-FQHE-1 through T-FQHE-7)
# ═══════════════════════════════════════════════════════════════════════════


class TestTheoremFQHE1:
    """T-FQHE-1: Topological Order as Fidelity Separation.

    Even-denominator states (candidate non-Abelian) maintain higher F than
    odd-denominator states because their richer topological structure
    sustains coherence across more channels simultaneously.

    Citation: Kim et al. Nature 649, 323-329 (2026), Fig. 2
    """

    def test_mean_F_even_gt_odd(
        self, even_results: list[FQHEKernelResult], odd_results: list[FQHEKernelResult]
    ) -> None:
        avg_F_even = sum(r.F for r in even_results) / len(even_results)
        avg_F_odd = sum(r.F for r in odd_results) / len(odd_results)
        assert avg_F_even > avg_F_odd

    def test_measured_even_denom_F_above_half(self, results_by_name: dict[str, FQHEKernelResult]) -> None:
        """Both measured even-denom states have F > 0.5."""
        for name in ["nu_neg_1_2", "nu_3_2"]:
            assert results_by_name[name].F > 0.5

    def test_even_denom_topological_channel(self, even_results: list[FQHEKernelResult]) -> None:
        """Even-denominators have topological_order channel at ~1.0."""
        for r in even_results:
            c = np.array(r.trace_vector)
            assert c[5] > 0.9  # topological_order channel


class TestTheoremFQHE2:
    """T-FQHE-2: Charge Fractionalization as Heterogeneity Gap.

    Fractional charge e* < e creates channel heterogeneity.
    The heterogeneity gap Δ = F − IC scales with the degree of charge
    fractionalization.

    Citation: Kim et al. Nature 649, 323-329 (2026), Extended Data Fig. 4
    """

    def test_hole_conjugate_larger_gap(self, all_results: list[FQHEKernelResult]) -> None:
        """Hole-conjugate states have larger Δ than particle-like odd-denom."""
        hole = [r for r in all_results if not r.is_fundamental_charge]
        particle_odd = [r for r in all_results if r.is_fundamental_charge and not r.is_even_denom]
        avg_delta_hc = sum(r.heterogeneity_gap for r in hole) / len(hole)
        avg_delta_p = sum(r.heterogeneity_gap for r in particle_odd) / len(particle_odd)
        assert avg_delta_hc > avg_delta_p

    def test_charge_fundamental_epsilon_kills_IC(self, all_results: list[FQHEKernelResult]) -> None:
        """States with charge_fundamental = ε have lower IC than fundamental states."""
        non_fund = [r for r in all_results if not r.is_fundamental_charge]
        fund_odd = [r for r in all_results if r.is_fundamental_charge and not r.is_even_denom]
        avg_IC_nf = sum(r.IC for r in non_fund) / len(non_fund)
        avg_IC_f = sum(r.IC for r in fund_odd) / len(fund_odd)
        assert avg_IC_nf < avg_IC_f


class TestTheoremFQHE3:
    """T-FQHE-3: Non-Abelian Ambiguity as IC Sensitivity.

    The 2Φ₀ vs 4Φ₀ ambiguity manifests as sensitivity of IC to the
    statistics_type channel.

    Citation: Kim et al. Nature 649, 323-329 (2026), Discussion section
    """

    def test_even_denom_statistics_ambiguous_channel(self, even_results: list[FQHEKernelResult]) -> None:
        """Even-denom states have statistics_type at 0.5 (ambiguous)."""
        for r in even_results:
            c = np.array(r.trace_vector)
            assert abs(c[7] - 0.5) < 0.01

    def test_odd_denom_statistics_known_channel(self, odd_results: list[FQHEKernelResult]) -> None:
        """Odd-denom states have statistics_type at 1.0 (Abelian confirmed)."""
        for r in odd_results:
            c = np.array(r.trace_vector)
            assert c[7] > 0.9


class TestTheoremFQHE4:
    """T-FQHE-4: Hole-Conjugate Anomaly as Channel Inversion.

    Hole-conjugate states (ν = −2/3, 5/3) interfere with e* = (2/3)e
    (NOT the fundamental (1/3)e), which maps to inversion of the
    charge_fundamental channel.

    Citation: Kim et al. Nature 649, 323-329 (2026), Fig. 3
    """

    def test_hole_conjugate_IC_lowest_odd(self, odd_results: list[FQHEKernelResult]) -> None:
        """Hole-conjugate states have the lowest IC among odd-denom."""
        hole_conj = [r for r in odd_results if not r.is_fundamental_charge]
        particle = [r for r in odd_results if r.is_fundamental_charge]
        avg_IC_hc = sum(r.IC for r in hole_conj) / len(hole_conj)
        avg_IC_p = sum(r.IC for r in particle) / len(particle)
        assert avg_IC_hc < avg_IC_p

    def test_charge_fundamental_is_weakest_for_hole_conjugate(
        self, results_by_name: dict[str, FQHEKernelResult]
    ) -> None:
        """For hole-conjugate states, charge_fundamental = ε is the weakest channel."""
        for name in ["nu_neg_2_3", "nu_5_3"]:
            r = results_by_name[name]
            assert r.weakest_channel == "charge_fundamental"

    def test_particle_charge_fundamental_not_weakest(self, results_by_name: dict[str, FQHEKernelResult]) -> None:
        """For particle-like Laughlin states, charge_fundamental is NOT the weakest."""
        for name in ["nu_neg_1_3", "nu_4_3"]:
            r = results_by_name[name]
            assert r.weakest_channel != "charge_fundamental"

    def test_nu_neg_2_3_charge_non_fundamental(self, results_by_name: dict[str, FQHEKernelResult]) -> None:
        """ν = −2/3 interferes with e* = (2/3)e, not (1/3)e."""
        r = results_by_name["nu_neg_2_3"]
        assert abs(r.charge_e_star - 2.0 / 3.0) < 1e-10
        assert not r.is_fundamental_charge


class TestTheoremFQHE5:
    """T-FQHE-5: Visibility as Coherence Proxy.

    Higher AB visibility correlates with IC through the visibility channel.

    Citation: Kim et al. Nature 649, 323-329 (2026), Fig. 2
    """

    def test_nu_3_2_higher_vis_than_neg_1_2(self, results_by_name: dict[str, FQHEKernelResult]) -> None:
        """ν = 3/2 (vis 5.6%) has higher IC than ν = −1/2 (vis 1.9%)."""
        r_32 = results_by_name["nu_3_2"]
        r_neg12 = results_by_name["nu_neg_1_2"]
        assert r_32.IC > r_neg12.IC

    def test_visibility_channel_tracks_measurement(self, results_by_name: dict[str, FQHEKernelResult]) -> None:
        """Visibility channel correctly maps: 5.6% > 1.9%."""
        r_32 = results_by_name["nu_3_2"]
        r_neg12 = results_by_name["nu_neg_1_2"]
        c_32 = np.array(r_32.trace_vector)
        c_neg12 = np.array(r_neg12.trace_vector)
        assert c_32[4] > c_neg12[4]  # visibility channel


class TestTheoremFQHE6:
    """T-FQHE-6: e* = ν_LL Universality as Structural Constraint.

    The universal charge relation e* = ν_LL × e across all six states
    is a structural constraint from Axiom-0.

    Citation: Kim et al. Nature 649, 323-329 (2026), entire paper
    """

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_charge_equals_nu_LL(self, state: FQHState) -> None:
        """e* = ν_LL × e for every state."""
        assert abs(state.charge_e_star - state.landau_level_filling) < 1e-10

    def test_nu_LL_groups_have_same_charge(self, all_results: list[FQHEKernelResult]) -> None:
        """States with the same ν_LL have the same e*."""
        groups: dict[float, set[float]] = {}
        for r in all_results:
            ll = round(r.landau_level_filling, 6)
            groups.setdefault(ll, set()).add(round(r.charge_e_star, 6))
        for ll, charges in groups.items():
            assert len(charges) == 1, f"ν_LL={ll}: charges={charges}"

    def test_three_distinct_nu_LL_values(self, all_results: list[FQHEKernelResult]) -> None:
        """Three distinct ν_LL values: 1/3, 1/2, 2/3."""
        ll_values = {round(r.landau_level_filling, 6) for r in all_results}
        assert len(ll_values) == 3


class TestTheoremFQHE7:
    """T-FQHE-7: Cross-Scale Bridge (μeV → meV → GeV).

    FQHE exhibits the same confinement-as-IC-collapse pattern seen
    in QDM and the Standard Model.

    Citation: Kim et al. Nature 649, 323-329 (2026), comparison with
    Yan et al. Nat. Commun. 13, 5799 (2022) and PDG 2024.
    """

    def test_even_denom_higher_IC_F_than_hole_conjugate(
        self, even_results: list[FQHEKernelResult], all_results: list[FQHEKernelResult]
    ) -> None:
        """Even-denom (topological) have higher IC/F than hole-conjugate (ordered)."""
        hole_conj = [r for r in all_results if not r.is_fundamental_charge]
        avg_icf_even = sum(r.IC / r.F for r in even_results) / len(even_results)
        avg_icf_hc = sum(r.IC / r.F for r in hole_conj) / len(hole_conj)
        assert avg_icf_even > avg_icf_hc

    def test_IC_F_separation(self, all_results: list[FQHEKernelResult]) -> None:
        """IC/F separates even-denom from hole-conjugate by > 10×."""
        even = [r for r in all_results if r.is_even_denom]
        hole = [r for r in all_results if not r.is_fundamental_charge]
        avg_icf_even = sum(r.IC / r.F for r in even) / len(even)
        avg_icf_hole = sum(r.IC / r.F for r in hole) / len(hole)
        assert avg_icf_even / avg_icf_hole > 10.0

    def test_heterogeneity_gap_universal_diagnostic(self, all_results: list[FQHEKernelResult]) -> None:
        """Δ/F discriminates topological from ordered phases."""
        even = [r for r in all_results if r.is_even_denom]
        hole = [r for r in all_results if not r.is_fundamental_charge]
        avg_df_even = sum(r.heterogeneity_gap / r.F for r in even) / len(even)
        avg_df_hole = sum(r.heterogeneity_gap / r.F for r in hole) / len(hole)
        # Hole-conjugate should have higher Δ/F (more heterogeneous)
        assert avg_df_hole > avg_df_even


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeClassification:
    """Verify regime classification consistency."""

    def test_classify_stable(self) -> None:
        assert classify_regime(0.02, 0.95, 0.10, 0.10) == "Stable"

    def test_classify_watch(self) -> None:
        assert classify_regime(0.10, 0.85, 0.30, 0.20) == "Watch"

    def test_classify_collapse(self) -> None:
        assert classify_regime(0.35, 0.65, 0.50, 0.40) == "Collapse"

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_regime_valid(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert r.regime in ("Stable", "Watch", "Collapse")

    def test_nu_3_2_in_watch(self, results_by_name: dict[str, FQHEKernelResult]) -> None:
        """ν = 3/2 (highest F, measured even-denom) should be Watch."""
        assert results_by_name["nu_3_2"].regime == "Watch"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictions:
    """Verify all 7 GCD predictions."""

    def test_predictions_count(self) -> None:
        assert len(PREDICTIONS) == 7

    def test_predictions_keys(self) -> None:
        expected_keys = {
            "P1_even_denom_higher_F",
            "P2_charge_fractionalization_gap",
            "P3_hole_conjugate_IC_collapse",
            "P4_visibility_IC_correlation",
            "P5_nu_LL_universality",
            "P6_even_denom_ambiguity",
            "P7_cross_scale_confinement",
        }
        assert set(PREDICTIONS.keys()) == expected_keys

    @pytest.mark.parametrize("key", list(PREDICTIONS.keys()))
    def test_prediction_nonempty(self, key: str) -> None:
        assert len(PREDICTIONS[key]) > 20


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: CHANNEL AUTOPSY
# ═══════════════════════════════════════════════════════════════════════════


class TestChannelAutopsy:
    """Verify channel extrema identification."""

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_weakest_channel_in_labels(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert r.weakest_channel in CHANNEL_LABELS

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_strongest_channel_in_labels(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert r.strongest_channel in CHANNEL_LABELS

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_weakest_leq_strongest(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        assert r.weakest_value <= r.strongest_value

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_weakest_value_matches_trace(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        c = np.array(r.trace_vector)
        assert abs(r.weakest_value - np.min(c)) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Verify to_dict round-trip."""

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_to_dict_keys(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        d = r.to_dict()
        required = {
            "name",
            "F",
            "omega",
            "IC",
            "kappa",
            "S",
            "C",
            "regime",
            "heterogeneity_gap",
            "trace_vector",
            "channel_labels",
            "filling_factor",
            "charge_e_star",
            "weakest_channel",
            "strongest_channel",
        }
        assert required.issubset(d.keys())

    @pytest.mark.parametrize("state", FQHE_STATES, ids=lambda s: s.name)
    def test_to_dict_F_matches(self, state: FQHState) -> None:
        r = compute_fqhe_kernel(state)
        d = r.to_dict()
        assert abs(d["F"] - r.F) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: CROSS-CATEGORY COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossCategory:
    """Cross-category invariant comparisons."""

    def test_highest_F_is_even_denom(self, all_results: list[FQHEKernelResult]) -> None:
        """The state with highest F should be an even-denominator state."""
        best = max(all_results, key=lambda r: r.F)
        assert best.is_even_denom

    def test_lowest_IC_is_hole_conjugate(self, all_results: list[FQHEKernelResult]) -> None:
        """The state with lowest IC should be a hole-conjugate state."""
        worst = min(all_results, key=lambda r: r.IC)
        assert not worst.is_fundamental_charge

    def test_all_identities_pass(self, all_results: list[FQHEKernelResult]) -> None:
        """All 11 states pass all three Tier-1 identities."""
        for r in all_results:
            assert abs(r.F_plus_omega - 1.0) < 1e-10
            assert r.IC_leq_F
            assert r.IC_eq_exp_kappa

    def test_compute_all_states_length(self) -> None:
        """compute_all_states returns 11 results."""
        results = compute_all_states()
        assert len(results) == 11

    def test_batch_equals_individual(self) -> None:
        """Batch and individual computation produce identical results."""
        batch = compute_all_states()
        for i, state in enumerate(FQHE_STATES):
            individual = compute_fqhe_kernel(state)
            assert abs(batch[i].F - individual.F) < 1e-12
            assert abs(batch[i].IC - individual.IC) < 1e-12
