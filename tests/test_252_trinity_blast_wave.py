"""Tests for Trinity Blast Wave closure — trinity_blast_wave.py.

Validates 16 theorems (T-TB-1 through T-TB-16), 29 entity constructions
across 3 categories (fireball, device, reference), Tier-1 identity
universality, trace vector construction, fission-fusion bridge, and
narrative generation.

Test count target: ~200 tests covering:
    - Frozen physics constants (Taylor-Sedov params, Pu-239, iron peak)
    - Mack photograph data table (24 time-radius pairs)
    - Device entity construction (3 entities)
    - Reference entity construction (2 entities)
    - Trace vector channel normalization
    - 16 theorem proofs with subtests
    - Tier-1 identity universality (F+ω=1, IC≤F, IC=exp(κ))
    - Regime classification
    - Fission-fusion bridge data
    - Narrative generation
    - Full analysis integration
"""

from __future__ import annotations

import math

import pytest

from closures.nuclear_physics.trinity_blast_wave import (
    A_FIT,
    A_PEAK,
    BE_PEAK_REF,
    BINDING_DEUTERIUM,
    BINDING_PU239,
    BINDING_U238,
    C_AIR,
    CHANNEL_NAMES,
    DENSITY_JUMP_LIMIT,
    DEVICE_DATA,
    E_EXTRACTED_J,
    FISSION_EFFICIENCY,
    GAMMA_AIR,
    KT_TO_J,
    MACH_REF,
    MACK_DATA,
    P_ATM,
    P_RATIO_LOG_SCALE,
    P_RATIO_MAX,
    PU239_A,
    PU239_BE_PER_A,
    PU239_FISSILITY,
    PU239_MASS_KG,
    PU239_Z,
    Q_FISSION_MEV,
    REFERENCE_DATA,
    RHO_AIR,
    TOWER_HEIGHT_M,
    YIELD_J,
    YIELD_KT,
    YIELD_SELBY_KT,
    FissionFusionBridge,
    TrinityAnalysisResult,
    TrinityObservables,
    build_all_entities,
    build_device_entities,
    build_fireball_entities,
    build_reference_entities,
    build_trace,
    run_full_analysis,
)
from umcp.frozen_contract import EPSILON

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def full_analysis():
    """Run full analysis once for the entire test module."""
    return run_full_analysis()


@pytest.fixture(scope="module")
def fireball_entities():
    """Build fireball entities once."""
    return build_fireball_entities()


@pytest.fixture(scope="module")
def device_entities():
    """Build device entities once."""
    return build_device_entities()


@pytest.fixture(scope="module")
def reference_entities():
    """Build reference entities once."""
    return build_reference_entities()


@pytest.fixture(scope="module")
def all_entities():
    """Build all entities once."""
    return build_all_entities()


# ═══════════════════════════════════════════════════════════════
# FROZEN CONSTANTS
# ═══════════════════════════════════════════════════════════════


class TestFrozenConstants:
    """Verify frozen physics constants."""

    def test_gamma_air(self):
        assert GAMMA_AIR == 1.4

    def test_rho_air(self):
        assert RHO_AIR == 1.25

    def test_c_air(self):
        assert C_AIR == 343.0

    def test_p_atm(self):
        assert P_ATM == 101325.0

    def test_density_jump_limit(self):
        assert abs(DENSITY_JUMP_LIMIT - 6.0) < 1e-12

    def test_a_fit(self):
        assert A_FIT == 583.6

    def test_e_extracted_consistency(self):
        """E_extracted = ρ₀ × A_FIT⁵."""
        assert abs(E_EXTRACTED_J - RHO_AIR * A_FIT**5) < 1.0

    def test_yield_kt(self):
        assert YIELD_KT == 21.0

    def test_yield_selby_kt(self):
        assert YIELD_SELBY_KT == 24.8

    def test_yield_j(self):
        assert abs(YIELD_J - YIELD_KT * KT_TO_J) < 1.0

    def test_kt_to_j(self):
        assert KT_TO_J == 4.184e12

    def test_mach_ref(self):
        assert MACH_REF == 10.0

    def test_p_ratio_max(self):
        assert P_RATIO_MAX == 20000.0

    def test_p_ratio_log_scale(self):
        assert abs(P_RATIO_LOG_SCALE - math.log10(P_RATIO_MAX + 1.0)) < 1e-12

    def test_pu239_z(self):
        assert PU239_Z == 94

    def test_pu239_a(self):
        assert PU239_A == 239

    def test_pu239_mass(self):
        assert PU239_MASS_KG == 6.2

    def test_fission_efficiency(self):
        assert abs(FISSION_EFFICIENCY - 1.0 / 6.2) < 1e-6

    def test_pu239_be_per_a(self):
        assert PU239_BE_PER_A == 7.560

    def test_pu239_fissility(self):
        assert abs(PU239_FISSILITY - 94**2 / 239) < 0.01

    def test_q_fission(self):
        assert Q_FISSION_MEV == 200.0

    def test_be_peak_ref(self):
        assert BE_PEAK_REF == 8.7945

    def test_a_peak(self):
        assert A_PEAK == 62

    def test_binding_pu239(self):
        assert abs(BINDING_PU239 - PU239_BE_PER_A / BE_PEAK_REF) < 1e-6

    def test_binding_u238(self):
        assert abs(BINDING_U238 - 7.570 / BE_PEAK_REF) < 1e-6

    def test_binding_deuterium(self):
        assert abs(BINDING_DEUTERIUM - 1.112 / BE_PEAK_REF) < 1e-6

    def test_tower_height(self):
        assert TOWER_HEIGHT_M == 30.5

    def test_channel_count(self):
        assert len(CHANNEL_NAMES) == 8

    def test_extracted_yield_near_official(self):
        """E_extracted should be within 25% of official yield."""
        E_kt = E_EXTRACTED_J / KT_TO_J
        assert abs(E_kt - YIELD_KT) / YIELD_KT < 0.25


# ═══════════════════════════════════════════════════════════════
# DATA TABLE INTEGRITY
# ═══════════════════════════════════════════════════════════════


class TestDataTables:
    """Verify data table structure and consistency."""

    def test_mack_count(self):
        assert len(MACK_DATA) == 24

    def test_device_count(self):
        assert len(DEVICE_DATA) == 3

    def test_reference_count(self):
        assert len(REFERENCE_DATA) == 2

    @pytest.mark.parametrize("idx", range(24))
    def test_mack_time_positive(self, idx):
        assert MACK_DATA[idx]["t_ms"] > 0

    @pytest.mark.parametrize("idx", range(24))
    def test_mack_radius_positive(self, idx):
        assert MACK_DATA[idx]["R_m"] > 0

    @pytest.mark.parametrize("idx", range(23))
    def test_mack_time_ordering(self, idx):
        assert MACK_DATA[idx]["t_ms"] < MACK_DATA[idx + 1]["t_ms"]

    @pytest.mark.parametrize("idx", range(23))
    def test_mack_radius_ordering(self, idx):
        """Radius should be monotonically increasing."""
        assert MACK_DATA[idx]["R_m"] < MACK_DATA[idx + 1]["R_m"]

    def test_mack_first_radius(self):
        """First measurement at 0.10 ms should be ~11 m."""
        assert abs(MACK_DATA[0]["R_m"] - 11.1) < 0.1

    def test_mack_last_radius(self):
        """Last measurement at 62 ms should be ~185 m."""
        assert abs(MACK_DATA[-1]["R_m"] - 185.0) < 0.1

    def test_device_names(self):
        names = [d["name"] for d in DEVICE_DATA]
        assert "Pu-239 Core" in names
        assert "U-238 Tamper" in names
        assert "Explosive Lens Array" in names

    def test_reference_names(self):
        names = [d["name"] for d in REFERENCE_DATA]
        assert any("Conventional" in n for n in names)
        assert any("D-T" in n for n in names)


# ═══════════════════════════════════════════════════════════════
# TRACE VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════


class TestTraceVector:
    """Test trace vector construction and normalization."""

    def test_trace_shape(self):
        obs = TrinityObservables(1e-3, 40.0, 16000.0, 46.6, 1e7, 8e13, 5.9, 0.86, 0.4)
        c, w = build_trace(obs)
        assert c.shape == (8,)
        assert w.shape == (8,)

    def test_weights_sum(self):
        obs = TrinityObservables(1e-3, 40.0, 16000.0, 46.6, 1e7, 8e13, 5.9, 0.86, 0.4)
        _, w = build_trace(obs)
        assert abs(w.sum() - 1.0) < 1e-12

    def test_trace_bounds(self):
        obs = TrinityObservables(1e-3, 40.0, 16000.0, 46.6, 1e7, 8e13, 5.9, 0.86, 0.4)
        c, _ = build_trace(obs)
        for val in c:
            assert EPSILON <= val <= 1.0 - EPSILON

    def test_self_similarity_channel_near_one(self):
        """For a mid-range fireball entity, ξ ≈ 1 → c₁ ≈ 1."""
        t = 1.93e-3
        R = 48.7
        v = 0.4 * R / t
        M = v / C_AIR
        obs = TrinityObservables(t, R, v, M, 1e6, 8e13, 5.8, 0.86, 0.4)
        c, _ = build_trace(obs)
        assert c[0] > 0.95  # self_similarity should be near 1.0

    def test_device_self_similarity_near_epsilon(self):
        """For a device entity, ξ << 1 → c₁ ≈ ε."""
        t = 1e-6
        R = 0.042
        v = 5000.0
        M = v / C_AIR
        obs = TrinityObservables(t, R, v, M, 1e6, 1e5, 2.0, 0.86, 0.0)
        c, _ = build_trace(obs)
        assert c[0] < 0.05  # self_similarity near ε

    def test_mach_channel_saturates(self):
        """High Mach → c₃ → 1."""
        M = 100.0
        obs = TrinityObservables(1e-4, 11.0, 44000.0, M, 1e8, 8e13, 5.99, 0.86, 0.4)
        c, _ = build_trace(obs)
        assert c[2] > 0.90  # mach_fidelity saturates for large M

    def test_power_law_perfect(self):
        """α = 0.4 → c₄ = 1 − ε."""
        obs = TrinityObservables(1e-3, 40.0, 16000.0, 46.6, 1e7, 8e13, 5.9, 0.86, 0.4)
        c, _ = build_trace(obs)
        assert c[3] > 0.99  # power_law_quality near 1

    def test_power_law_zero(self):
        """α = 0 → c₄ near ε."""
        obs = TrinityObservables(1e-3, 40.0, 16000.0, 46.6, 1e7, 8e13, 5.9, 0.86, 0.0)
        c, _ = build_trace(obs)
        assert c[3] < 0.05  # power_law_quality near 0

    def test_channel_names(self):
        assert CHANNEL_NAMES[0] == "self_similarity"
        assert CHANNEL_NAMES[7] == "binding_fidelity"


# ═══════════════════════════════════════════════════════════════
# ENTITY BUILDERS
# ═══════════════════════════════════════════════════════════════


class TestFireballEntities:
    """Test fireball entity construction."""

    def test_count(self, fireball_entities):
        assert len(fireball_entities) == 24

    def test_category(self, fireball_entities):
        for e in fireball_entities:
            assert e.category == "fireball"

    def test_channels_length(self, fireball_entities):
        for e in fireball_entities:
            assert len(e.channels) == 8

    def test_trace_bounds(self, fireball_entities):
        for e in fireball_entities:
            assert e.trace.shape == (8,)
            for v in e.trace:
                assert EPSILON <= v <= 1.0 - EPSILON

    def test_binding_fidelity_constant(self, fireball_entities):
        """All fireball entities use Pu-239 binding fidelity."""
        for e in fireball_entities:
            assert abs(e.observables.binding_fidelity - BINDING_PU239) < 1e-10

    def test_velocity_decreasing_overall(self, fireball_entities):
        v_first = fireball_entities[0].observables.velocity_m_s
        v_last = fireball_entities[-1].observables.velocity_m_s
        assert v_first > v_last

    def test_mach_decreasing_overall(self, fireball_entities):
        M_first = fireball_entities[0].observables.mach_number
        M_last = fireball_entities[-1].observables.mach_number
        assert M_first > M_last

    @pytest.mark.parametrize("idx", range(24))
    def test_mach_positive(self, fireball_entities, idx):
        assert fireball_entities[idx].observables.mach_number > 1.0


class TestDeviceEntities:
    """Test device entity construction."""

    def test_count(self, device_entities):
        assert len(device_entities) == 3

    def test_category(self, device_entities):
        for e in device_entities:
            assert e.category == "device"

    def test_pu239_binding(self, device_entities):
        pu = next(e for e in device_entities if "Pu-239" in e.name)
        assert abs(pu.observables.binding_fidelity - BINDING_PU239) < 1e-6

    def test_he_lens_binding_near_epsilon(self, device_entities):
        he = next(e for e in device_entities if "Explosive" in e.name)
        assert he.observables.binding_fidelity < 0.01


class TestReferenceEntities:
    """Test reference entity construction."""

    def test_count(self, reference_entities):
        assert len(reference_entities) == 2

    def test_category(self, reference_entities):
        for e in reference_entities:
            assert e.category == "reference"

    def test_he_binding_near_epsilon(self, reference_entities):
        he = next(e for e in reference_entities if "Conventional" in e.name)
        assert he.observables.binding_fidelity < 0.01

    def test_dt_binding(self, reference_entities):
        dt = next(e for e in reference_entities if "D-T" in e.name)
        assert abs(dt.observables.binding_fidelity - BINDING_DEUTERIUM) < 1e-6


class TestAllEntities:
    """Test build_all_entities."""

    def test_total_count(self, all_entities):
        assert len(all_entities) == 29

    def test_category_distribution(self, all_entities):
        cats = [e.category for e in all_entities]
        assert cats.count("fireball") == 24
        assert cats.count("device") == 3
        assert cats.count("reference") == 2


# ═══════════════════════════════════════════════════════════════
# TIER-1 IDENTITY UNIVERSALITY
# ═══════════════════════════════════════════════════════════════


class TestTier1:
    """Verify Tier-1 identities for all 29 entities."""

    @pytest.mark.parametrize("idx", range(29))
    def test_duality_identity(self, all_entities, idx):
        """F + ω = 1 exactly."""
        e = all_entities[idx]
        assert abs(e.F + e.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("idx", range(29))
    def test_integrity_bound(self, all_entities, idx):
        """IC ≤ F (integrity cannot exceed fidelity)."""
        e = all_entities[idx]
        assert e.IC <= e.F + 1e-12

    @pytest.mark.parametrize("idx", range(29))
    def test_ic_exp_kappa(self, all_entities, idx):
        """IC = exp(κ) (log-integrity relation)."""
        e = all_entities[idx]
        assert abs(e.IC - math.exp(e.kappa)) < 1e-10

    @pytest.mark.parametrize("idx", range(29))
    def test_gap_nonnegative(self, all_entities, idx):
        """Δ = F − IC ≥ 0."""
        e = all_entities[idx]
        assert e.gap >= -1e-12


# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestRegimes:
    """Test regime classification distribution."""

    def test_valid_regimes(self, all_entities):
        for e in all_entities:
            assert e.regime in ("Stable", "Watch", "Collapse")

    def test_device_entities_mostly_collapse(self, device_entities):
        """Device entities should be Watch or Collapse (not self-similar)."""
        for e in device_entities:
            assert e.regime in ("Watch", "Collapse")

    def test_mid_fireball_not_collapse(self, fireball_entities):
        """Mid-phase fireball entities should not be Collapse."""
        mid = [e for e in fireball_entities if 5e-4 < e.observables.time_s < 5e-3]
        for e in mid:
            assert e.regime != "Collapse"


# ═══════════════════════════════════════════════════════════════
# THEOREM PROOFS
# ═══════════════════════════════════════════════════════════════


class TestTheorems:
    """Verify all 8 Trinity blast wave theorems."""

    def test_T_TB_1_proven(self, full_analysis):
        """T-TB-1: Self-Similar Conformance."""
        t = full_analysis.theorem_results["T-TB-1"]
        assert t["proven"], f"T-TB-1 failed: median_dev={t.get('median_deviation')}"

    def test_T_TB_1_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-1"]
        assert t["passed"] == t["tests"]

    def test_T_TB_2_proven(self, full_analysis):
        """T-TB-2: Yield Self-Consistency."""
        t = full_analysis.theorem_results["T-TB-2"]
        assert t["proven"], f"T-TB-2 failed: cv={t.get('cv')}"

    def test_T_TB_2_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-2"]
        assert t["passed"] == t["tests"]

    def test_T_TB_3_proven(self, full_analysis):
        """T-TB-3: Shock Weakening Transition."""
        t = full_analysis.theorem_results["T-TB-3"]
        assert t["proven"], f"T-TB-3 failed: mid_IC={t.get('mid_IC')}, late_IC={t.get('late_IC')}"

    def test_T_TB_3_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-3"]
        assert t["passed"] == t["tests"]

    def test_T_TB_4_proven(self, full_analysis):
        """T-TB-4: Phase Boundary F-Split."""
        t = full_analysis.theorem_results["T-TB-4"]
        assert t["proven"], f"T-TB-4 failed: early_F={t.get('early_F')}, mid_F={t.get('mid_F')}"

    def test_T_TB_4_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-4"]
        assert t["passed"] == t["tests"]

    def test_T_TB_5_proven(self, full_analysis):
        """T-TB-5: Power Law Quality."""
        t = full_analysis.theorem_results["T-TB-5"]
        assert t["proven"], f"T-TB-5 failed: frac={t.get('fraction_within_15pct')}"

    def test_T_TB_5_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-5"]
        assert t["passed"] == t["tests"]

    def test_T_TB_6_proven(self, full_analysis):
        """T-TB-6: Taylor Yield Extraction."""
        t = full_analysis.theorem_results["T-TB-6"]
        assert t["proven"], f"T-TB-6 failed: yield={t.get('extracted_kt')} kt"

    def test_T_TB_6_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-6"]
        assert t["passed"] == t["tests"]

    def test_T_TB_7_proven(self, full_analysis):
        """T-TB-7: Velocity Monotonicity."""
        t = full_analysis.theorem_results["T-TB-7"]
        assert t["proven"], f"T-TB-7 failed: frac={t.get('monotone_fraction')}"

    def test_T_TB_7_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-7"]
        assert t["passed"] == t["tests"]

    def test_T_TB_8_proven(self, full_analysis):
        """T-TB-8: Fission-Fusion Bridge."""
        t = full_analysis.theorem_results["T-TB-8"]
        assert t["proven"], f"T-TB-8 failed: binding_gap={t.get('binding_gap')}"

    def test_T_TB_8_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-8"]
        assert t["passed"] == t["tests"]

    def test_all_theorems_proven(self, full_analysis):
        """All 13 theorems must be proven."""
        for tid, result in full_analysis.theorem_results.items():
            assert result["proven"], f"{tid} not proven"


# ═══════════════════════════════════════════════════════════════
# NEW THEOREMS T-TB-9 THROUGH T-TB-13
# ═══════════════════════════════════════════════════════════════


class TestCoordinatedDecay:
    """T-TB-9: Coordinated Decay — rank-preserving geometric slaughter."""

    def test_T_TB_9_proven(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-9"]
        assert t["proven"], f"T-TB-9 failed: min_ch={t.get('min_channel_value')}"

    def test_T_TB_9_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-9"]
        assert t["passed"] == t["tests"]

    def test_min_channel_well_above_epsilon(self, full_analysis):
        """No fireball channel reaches ε — unlike QGP confinement."""
        t = full_analysis.theorem_results["T-TB-9"]
        assert t["min_channel_value"] > 0.10

    def test_multiple_significant_channels(self, full_analysis):
        """At least 3 channels contribute > 5% of κ-loss."""
        t = full_analysis.theorem_results["T-TB-9"]
        assert t["significant_channels"] >= 3

    def test_channel_kappa_fractions_sum(self, full_analysis):
        """Channel κ fractions should sum to ~1.0."""
        t = full_analysis.theorem_results["T-TB-9"]
        total = sum(t["channel_kappa_fractions"].values())
        assert abs(total - 1.0) < 0.01

    def test_no_single_channel_dominates(self, full_analysis):
        """No single channel should have > 60% of κ-loss."""
        t = full_analysis.theorem_results["T-TB-9"]
        assert t["top_contributor_fraction"] < 0.60

    def test_insight_contains_rank_preserving(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-9"]
        assert "Rank-preserving" in t["insight"]


class TestDecoherenceField:
    """T-TB-10: Decoherence Field Expansion — gap grows with R(t)."""

    def test_T_TB_10_proven(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-10"]
        assert t["proven"], f"T-TB-10 failed: growth={t.get('growth_factor')}"

    def test_T_TB_10_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-10"]
        assert t["passed"] == t["tests"]

    def test_early_gap_small(self, full_analysis):
        """Early gap should be < 0.05 (relatively coherent)."""
        t = full_analysis.theorem_results["T-TB-10"]
        assert t["early_gap"] < 0.05

    def test_late_gap_larger(self, full_analysis):
        """Late gap should be > early gap."""
        t = full_analysis.theorem_results["T-TB-10"]
        assert t["late_gap"] > t["early_gap"]

    def test_growth_factor_substantial(self, full_analysis):
        """Gap growth factor > 3×."""
        t = full_analysis.theorem_results["T-TB-10"]
        assert t["growth_factor"] > 3.0

    def test_decoherence_has_physical_radius(self, full_analysis):
        """The decoherence field has a measurable spatial extent."""
        t = full_analysis.theorem_results["T-TB-10"]
        assert t["R_late_m"] > 100  # > 100 m at late times

    def test_decoherence_area_positive(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-10"]
        assert t["decoherence_area_km2"] > 0


class TestPredictionAmplification:
    """T-TB-11: Prediction Amplification — ln(c) asymmetry."""

    def test_T_TB_11_proven(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-11"]
        assert t["proven"], f"T-TB-11 failed: amp={t.get('amplification_factor')}"

    def test_T_TB_11_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-11"]
        assert t["passed"] == t["tests"]

    def test_ic_shifts_more_than_f(self, full_analysis):
        """ΔIC/IC > ΔF/F for single-channel perturbation."""
        t = full_analysis.theorem_results["T-TB-11"]
        assert t["delta_IC_rel_pct"] > t["delta_F_rel_pct"]

    def test_amplification_exceeds_unity(self, full_analysis):
        """Amplification factor > 1 (IC more sensitive than F)."""
        t = full_analysis.theorem_results["T-TB-11"]
        assert t["amplification_factor"] > 1.0

    def test_penalty_ratio_superlinear(self, full_analysis):
        """Halving c more than doubles the κ penalty (ratio > 1.5)."""
        t = full_analysis.theorem_results["T-TB-11"]
        assert t["penalty_ratio"] > 1.5

    def test_perturbed_channel_is_blast(self, full_analysis):
        """The perturbed channel should be a blast channel, not binding."""
        t = full_analysis.theorem_results["T-TB-11"]
        assert t["perturbed_channel"] != "binding_fidelity"


class TestNuclearIrreversibility:
    """T-TB-12: Nuclear Irreversibility — two-zone decoherence."""

    def test_T_TB_12_proven(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-12"]
        assert t["proven"], f"T-TB-12 failed: std={t.get('binding_std')}"

    def test_T_TB_12_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-12"]
        assert t["passed"] == t["tests"]

    def test_binding_exactly_constant(self, full_analysis):
        """Binding fidelity std ≈ 0 across all fireball entities."""
        t = full_analysis.theorem_results["T-TB-12"]
        assert t["binding_std"] < 1e-10

    def test_binding_matches_pu239(self, full_analysis):
        """Binding fidelity = Pu-239 BE/peak ratio."""
        t = full_analysis.theorem_results["T-TB-12"]
        assert abs(t["binding_mean"] - BINDING_PU239) < 1e-6

    def test_mach_changes_significantly(self, full_analysis):
        """Mach fidelity channel decays from first to last entity."""
        t = full_analysis.theorem_results["T-TB-12"]
        assert t["mach_delta"] > 0.10

    def test_two_zone_structure(self, full_analysis):
        """Blast decays while binding stays → two distinct zones."""
        t = full_analysis.theorem_results["T-TB-12"]
        # Binding std ≈ 0 AND mach changes → two-zone proven
        assert t["binding_std"] < 1e-10
        assert t["mach_first"] > t["mach_last"]

    def test_insight_contains_two_zone(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-12"]
        assert "Two-zone" in t["insight"]


class TestSensitivityDivergence:
    """T-TB-13: Sensitivity Divergence — weaker → more sensitive."""

    def test_T_TB_13_proven(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-13"]
        assert t["proven"], f"T-TB-13 failed: ratio={t.get('sensitivity_ratio')}"

    def test_T_TB_13_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-13"]
        assert t["passed"] == t["tests"]

    def test_channel_weakens(self, full_analysis):
        """The diagnostic channel gets weaker over time."""
        t = full_analysis.theorem_results["T-TB-13"]
        assert t["c_late"] < t["c_early"]

    def test_sensitivity_increases(self, full_analysis):
        """Sensitivity grows as channel weakens (ratio > 1.5)."""
        t = full_analysis.theorem_results["T-TB-13"]
        assert t["sensitivity_ratio"] > 1.5

    def test_positive_feedback_mechanism(self, full_analysis):
        """Late sensitivity > early sensitivity (positive feedback loop)."""
        t = full_analysis.theorem_results["T-TB-13"]
        assert t["sensitivity_late"] > t["sensitivity_early"]

    def test_diagnostic_channel_is_mach(self, full_analysis):
        """Uses mach_fidelity as the diagnostic channel."""
        t = full_analysis.theorem_results["T-TB-13"]
        assert t["channel"] == "mach_fidelity"


# ═══════════════════════════════════════════════════════════════
# T-TB-14: RADIATION COUPLING
# ═══════════════════════════════════════════════════════════════


class TestRadiationCoupling:
    """T-TB-14: Radiation Coupling — τ_rad ≈ 192 μs governs early E_eff/E."""

    def test_T_TB_14_proven(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-14"]
        assert t["proven"], f"T-TB-14 failed: {t.get('passed')}/{t.get('tests')}"

    def test_T_TB_14_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-14"]
        assert t["passed"] >= 3

    def test_tau_rad_value(self, full_analysis):
        """τ_rad ≈ 192 μs."""
        t = full_analysis.theorem_results["T-TB-14"]
        assert 150 < t["tau_rad_us"] < 250

    def test_xi_departure_at_earliest(self, full_analysis):
        """ξ < 1 at the earliest data point (radiation not yet coupled)."""
        t = full_analysis.theorem_results["T-TB-14"]
        assert t["xi_earliest"] < 1.0

    def test_self_sim_depressed_early(self, full_analysis):
        """Self-similarity channel < 0.85 at earliest point."""
        t = full_analysis.theorem_results["T-TB-14"]
        assert t["c_self_sim_earliest"] < 0.85

    def test_energy_fraction_low(self, full_analysis):
        """E_eff/E < 0.70 at earliest point (energy trapped in radiation)."""
        t = full_analysis.theorem_results["T-TB-14"]
        assert t["e_fraction_earliest"] < 0.70

    def test_insight_mentions_radiation(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-14"]
        assert "Radiation" in t["insight"] or "radiation" in t["insight"]


# ═══════════════════════════════════════════════════════════════
# T-TB-15: MACH CLIFF
# ═══════════════════════════════════════════════════════════════


class TestMachCliff:
    """T-TB-15: Mach Cliff — logarithmic shock death drives gap explosion."""

    def test_T_TB_15_proven(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-15"]
        assert t["proven"], f"T-TB-15 failed: {t.get('passed')}/{t.get('tests')}"

    def test_T_TB_15_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-15"]
        assert t["passed"] >= 3

    def test_mach_decays(self, full_analysis):
        """Latest Mach number much smaller than initial."""
        t = full_analysis.theorem_results["T-TB-15"]
        assert t["M_latest"] < 10.0

    def test_kappa_penalty(self, full_analysis):
        """κ_mach contribution is meaningfully negative at latest point."""
        t = full_analysis.theorem_results["T-TB-15"]
        assert t["kappa_mach"] < -0.02

    def test_gap_explosion(self, full_analysis):
        """Late gap exceeds 5× the minimum gap."""
        t = full_analysis.theorem_results["T-TB-15"]
        assert t["gap_ratio"] > 5.0

    def test_gap_latest_positive(self, full_analysis):
        """Latest gap is positive and substantial."""
        t = full_analysis.theorem_results["T-TB-15"]
        assert t["gap_latest"] > 0.03

    def test_insight_mentions_cliff(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-15"]
        assert "cliff" in t["insight"] or "death" in t["insight"]


# ═══════════════════════════════════════════════════════════════
# T-TB-16: THREE-REGIME STRUCTURE
# ═══════════════════════════════════════════════════════════════


class TestThreeRegimeStructure:
    """T-TB-16: Three-Regime Structure — radiation → self-similar → decay."""

    def test_T_TB_16_proven(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-16"]
        assert t["proven"], f"T-TB-16 failed: {t.get('passed')}/{t.get('tests')}"

    def test_T_TB_16_subtests(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-16"]
        assert t["passed"] >= 4

    def test_all_regimes_populated(self, full_analysis):
        """All three regimes have at least one entity."""
        t = full_analysis.theorem_results["T-TB-16"]
        assert t["n_radiation"] > 0
        assert t["n_self_similar"] > 0
        assert t["n_decay"] > 0

    def test_self_similar_dominates(self, full_analysis):
        """Most entities fall in the self-similar regime."""
        t = full_analysis.theorem_results["T-TB-16"]
        assert t["n_self_similar"] > t["n_radiation"]
        assert t["n_self_similar"] > t["n_decay"]

    def test_self_similar_best_F(self, full_analysis):
        """Self-similar phase has higher mean F than radiation phase."""
        t = full_analysis.theorem_results["T-TB-16"]
        assert t["mean_F_self_similar"] > t["mean_F_radiation"]

    def test_self_similar_min_gap(self, full_analysis):
        """Self-similar phase has minimum gap (best coherence)."""
        t = full_analysis.theorem_results["T-TB-16"]
        assert t["min_gap_self_similar"] <= t["min_gap_radiation"]
        assert t["min_gap_self_similar"] <= t["min_gap_decay"]

    def test_u_shaped_trajectory(self, full_analysis):
        """Gap minimum is in the interior (not at endpoints)."""
        t = full_analysis.theorem_results["T-TB-16"]
        assert 0 < t["gap_min_index"] < 23  # Not first or last

    def test_entity_counts_sum(self, full_analysis):
        """All 24 fireball entities accounted for."""
        t = full_analysis.theorem_results["T-TB-16"]
        total = t["n_radiation"] + t["n_self_similar"] + t["n_decay"]
        assert total == 24

    def test_insight_mentions_three(self, full_analysis):
        t = full_analysis.theorem_results["T-TB-16"]
        assert "Three" in t["insight"] or "three" in t["insight"]


# ═══════════════════════════════════════════════════════════════
# FISSION-FUSION BRIDGE
# ═══════════════════════════════════════════════════════════════


class TestFissionFusionBridge:
    """Test the fission-fusion bridge data structure."""

    def test_bridge_construction(self, full_analysis):
        b = full_analysis.bridge
        assert isinstance(b, FissionFusionBridge)

    def test_fission_fuel(self, full_analysis):
        b = full_analysis.bridge
        assert b.fission_fuel_Z == 94
        assert b.fission_fuel_A == 239

    def test_fusion_fuel(self, full_analysis):
        b = full_analysis.bridge
        assert b.fusion_fuel_Z == 1
        assert b.fusion_fuel_A == 2

    def test_fusion_product(self, full_analysis):
        b = full_analysis.bridge
        assert b.fusion_product_Z == 2
        assert b.fusion_product_A == 4

    def test_fusion_q_value(self, full_analysis):
        b = full_analysis.bridge
        assert abs(b.fusion_Q_MeV - 17.6) < 0.1

    def test_energy_ratio(self, full_analysis):
        """Fusion releases ~4× more energy per nucleon than fission."""
        b = full_analysis.bridge
        assert 3.0 < b.energy_ratio < 6.0

    def test_deficit_ratio(self, full_analysis):
        """Fusion fuel is ~6× further from iron peak."""
        b = full_analysis.bridge
        assert 4.0 < b.deficit_ratio < 8.0

    def test_fission_direction(self, full_analysis):
        b = full_analysis.bridge
        assert "←Fe" in b.fission_direction

    def test_fusion_direction(self, full_analysis):
        b = full_analysis.bridge
        assert "→Fe" in b.fusion_direction

    def test_peak_reference(self, full_analysis):
        b = full_analysis.bridge
        assert b.peak_A == 62
        assert abs(b.peak_BE_per_A - 8.7945) < 1e-4

    def test_fission_binding_deficit(self, full_analysis):
        """Pu-239 deficit ≈ 0.14 (close to iron peak)."""
        b = full_analysis.bridge
        assert 0.10 < b.fission_binding_deficit < 0.20

    def test_fusion_binding_deficit(self, full_analysis):
        """Deuterium deficit ≈ 0.87 (far from iron peak)."""
        b = full_analysis.bridge
        assert 0.80 < b.fusion_binding_deficit < 0.95

    def test_lawson_criterion(self, full_analysis):
        b = full_analysis.bridge
        assert b.fusion_lawson_nTauE > 1e20


# ═══════════════════════════════════════════════════════════════
# NARRATIVE
# ═══════════════════════════════════════════════════════════════


class TestNarrative:
    """Test narrative generation."""

    def test_narrative_not_empty(self, full_analysis):
        assert len(full_analysis.narrative) > 100

    def test_narrative_five_words(self, full_analysis):
        n = full_analysis.narrative
        assert "DRIFT" in n
        assert "FIDELITY" in n
        assert "ROUGHNESS" in n
        assert "RETURN" in n
        assert "INTEGRITY" in n

    def test_narrative_bridge(self, full_analysis):
        assert "FISSION" in full_analysis.narrative
        assert "FUSION" in full_analysis.narrative

    def test_narrative_coordinated_decay(self, full_analysis):
        assert "COORDINATED DECAY" in full_analysis.narrative

    def test_narrative_decoherence_field(self, full_analysis):
        assert "DECOHERENCE FIELD" in full_analysis.narrative

    def test_narrative_prediction_amplification(self, full_analysis):
        assert "PREDICTION AMPLIFICATION" in full_analysis.narrative

    def test_narrative_radiation_coupling(self, full_analysis):
        assert "RADIATION COUPLING" in full_analysis.narrative

    def test_narrative_mach_cliff(self, full_analysis):
        assert "MACH CLIFF" in full_analysis.narrative

    def test_narrative_three_regime(self, full_analysis):
        assert "THREE-REGIME STRUCTURE" in full_analysis.narrative

    def test_narrative_yield(self, full_analysis):
        assert "20.2 kt" in full_analysis.narrative


# ═══════════════════════════════════════════════════════════════
# FULL ANALYSIS INTEGRATION
# ═══════════════════════════════════════════════════════════════


class TestFullAnalysis:
    """Integration tests for the complete analysis."""

    def test_result_type(self, full_analysis):
        assert isinstance(full_analysis, TrinityAnalysisResult)

    def test_entity_counts(self, full_analysis):
        assert full_analysis.summary["n_entities"] == 29
        assert full_analysis.summary["n_fireball"] == 24
        assert full_analysis.summary["n_device"] == 3
        assert full_analysis.summary["n_reference"] == 2

    def test_zero_tier1_violations(self, full_analysis):
        assert full_analysis.tier1_violations == 0

    def test_theorems_count(self, full_analysis):
        assert full_analysis.summary["n_theorems_total"] == 16

    def test_all_theorems_proven_summary(self, full_analysis):
        assert full_analysis.summary["n_theorems_proven"] == 16

    def test_yield_in_summary(self, full_analysis):
        """Extracted yield should be within 25% of official."""
        E_kt = full_analysis.summary["yield_extracted_kt"]
        assert abs(E_kt - YIELD_KT) / YIELD_KT < 0.25

    def test_mean_F_range(self, full_analysis):
        """Mean F should be moderate (entities span Stable to Collapse)."""
        assert 0.5 < full_analysis.summary["mean_F"] < 0.95

    def test_mean_IC_less_than_F(self, full_analysis):
        """Mean IC ≤ mean F (integrity bound at aggregate level)."""
        assert full_analysis.summary["mean_IC"] <= full_analysis.summary["mean_F"] + 1e-10

    def test_gap_positive(self, full_analysis):
        assert full_analysis.summary["mean_gap"] > 0

    def test_bridge_energy_ratio(self, full_analysis):
        assert full_analysis.summary["bridge_energy_ratio"] > 3.0

    def test_bridge_deficit_ratio(self, full_analysis):
        assert full_analysis.summary["bridge_deficit_ratio"] > 4.0
