"""Tests for continuity_theory domain — Butzbach Embedding

Comprehensive test coverage for the Butzbach embedding closure:
  - butzbach_embedding.py: 20 systems × 8 channels (Tier-1 identity sweep)
  - Degenerate limit proofs (scalar c = IC at n=1)
  - Cascade = log-integrity accumulation proof
  - Geometric slaughter demonstration (channel blindness)
  - Blind spot detection (Butzbach says healthy, GCD says fragile)
  - Protocol-to-Contract mapping
  - Rosetta vocabulary verification
  - Tier placement validation

Every test verifies structural predictions derivable from Axiom-0:
  F + ω = 1 (duality identity — complementum perfectum)
  IC ≤ F (integrity bound — limbus integritatis)
  IC = exp(κ) (log-integritas)

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized → butzbach_embedding
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.continuity_theory.butzbach_embedding import (
    N_BUTZBACH_CHANNELS,
    N_SYSTEMS,
    ROSETTA,
    SYSTEMS,
    TIER_PLACEMENT,
    ContinuityTrial,
    EmbeddingResult,
    PersistenceFunctional,
    ProtocolDeclaration,
    analyze_all_systems,
    butzbach_cascade,
    demonstrate_geometric_slaughter,
    prove_cascade_is_log_integrity,
    prove_scalar_limit,
    validate_embedding,
)

# ── Tolerances (same as frozen contract) ──────────────────────────
TOL_DUALITY = 1e-12  # F + ω = 1 exact to machine precision
TOL_EXP = 1e-9  # IC = exp(κ)
TOL_BOUND = 1e-12  # IC ≤ F (with guard)
EPS = 1e-6  # Closure-level ε


# ═══════════════════════════════════════════════════════════════════
# 1. Catalog Integrity
# ═══════════════════════════════════════════════════════════════════


class TestSystemCatalog:
    """Verify the system catalog data integrity."""

    def test_system_count(self) -> None:
        """20 systems in the catalog."""
        assert len(SYSTEMS) == 20
        assert N_SYSTEMS == 20

    def test_butzbach_channel_count(self) -> None:
        """Butzbach uses 1 channel (scalar c) — the degenerate limit."""
        assert N_BUTZBACH_CHANNELS == 1

    def test_all_systems_have_channels(self) -> None:
        """Every system has at least 2 channels for GCD analysis."""
        for sys in SYSTEMS:
            assert len(sys.channels) >= 2, f"{sys.name}: needs ≥2 channels for GCD"

    def test_all_systems_have_matching_labels(self) -> None:
        """Channel count matches label count for every system."""
        for sys in SYSTEMS:
            assert len(sys.channels) == len(sys.channel_labels), (
                f"{sys.name}: {len(sys.channels)} channels but {len(sys.channel_labels)} labels"
            )

    def test_channel_values_in_unit_interval(self) -> None:
        """All channel values must be in [0, 1]."""
        for sys in SYSTEMS:
            for i, val in enumerate(sys.channels):
                assert 0.0 <= val <= 1.0, f"{sys.name} channel {i} ({sys.channel_labels[i]}) = {val} out of [0,1]"

    def test_categories_are_valid(self) -> None:
        """Categories are from the allowed set."""
        valid = {"biological", "dormant", "engineered", "inert", "dissipative"}
        for sys in SYSTEMS:
            assert sys.category in valid, f"{sys.name}: unknown category '{sys.category}'"

    def test_unique_names(self) -> None:
        """All system names are unique."""
        names = [s.name for s in SYSTEMS]
        assert len(names) == len(set(names)), "Duplicate system names found"

    def test_e_met_non_negative(self) -> None:
        """Maintenance power is non-negative."""
        for sys in SYSTEMS:
            assert sys.e_met >= 0.0, f"{sys.name}: negative e_met = {sys.e_met}"

    def test_p_info_non_negative(self) -> None:
        """Informational power is non-negative."""
        for sys in SYSTEMS:
            assert sys.p_info >= 0.0, f"{sys.name}: negative p_info = {sys.p_info}"


# ═══════════════════════════════════════════════════════════════════
# 2. Butzbach Framework Implementation Tests
# ═══════════════════════════════════════════════════════════════════


class TestButzbachFramework:
    """Test Butzbach's own math is implemented faithfully."""

    def test_protocol_declaration(self) -> None:
        """Protocol declaration stores (B, V, Π, τ) correctly."""
        p = ProtocolDeclaration(
            boundary="Cell membrane",
            viability_set="Metabolically active",
            perturbation="Osmotic shock",
            recovery_window=3600.0,
        )
        assert p.boundary == "Cell membrane"
        assert p.recovery_window == 3600.0

    def test_protocol_to_contract(self) -> None:
        """Protocol maps to a GCD Contract declaration."""
        p = ProtocolDeclaration(
            boundary="Membrane",
            viability_set="Active",
            perturbation="Shock",
            recovery_window=60.0,
        )
        c = p.to_contract()
        assert c["declared_before_evidence"] is True
        assert c["recovery_window_s"] == 60.0
        assert "system_boundary" in c

    def test_continuity_trial_c(self) -> None:
        """c = n_recovered / n_trials."""
        t = ContinuityTrial(
            system_name="test",
            protocol=ProtocolDeclaration("B", "V", "Π", 1.0),
            n_trials=100,
            n_recovered=85,
        )
        assert abs(t.c - 0.85) < 1e-12

    def test_continuity_trial_zero_trials(self) -> None:
        """c = 0 when n_trials = 0 (avoid division by zero)."""
        t = ContinuityTrial(
            system_name="test",
            protocol=ProtocolDeclaration("B", "V", "Π", 1.0),
            n_trials=0,
            n_recovered=0,
        )
        assert t.c == 0.0

    def test_persistence_functional(self) -> None:
        """P_Ω = (E_met + P_info) · C."""
        pf = PersistenceFunctional(e_met=1.0, p_info=0.5, continuity=0.8)
        assert abs(pf.p_omega - 1.2) < 1e-12

    def test_persistence_functional_zero_continuity(self) -> None:
        """P_Ω = 0 when continuity is zero (no matter what power is available)."""
        pf = PersistenceFunctional(e_met=100.0, p_info=50.0, continuity=0.0)
        assert pf.p_omega == 0.0

    def test_landauer_floor_positive(self) -> None:
        """Landauer bound is a positive finite number."""
        pf = PersistenceFunctional(e_met=0.0, p_info=0.0, continuity=1.0)
        lb = pf.landauer_floor
        assert lb > 0.0
        assert math.isfinite(lb)

    def test_cascade_monotone(self) -> None:
        """Cascade C(t) is monotonically non-increasing when all c(t) ≤ 1."""
        c_vals = [0.95, 0.90, 0.88, 0.92, 0.85]
        cascade = butzbach_cascade(c_vals)
        for i in range(1, len(cascade)):
            assert cascade[i] <= cascade[i - 1] + 1e-15

    def test_cascade_product(self) -> None:
        """C(t) = product of c(0) through c(t-1)."""
        c_vals = [0.9, 0.8, 0.7]
        cascade = butzbach_cascade(c_vals)
        assert abs(cascade[0] - 0.9) < 1e-12
        assert abs(cascade[1] - 0.9 * 0.8) < 1e-12
        assert abs(cascade[2] - 0.9 * 0.8 * 0.7) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# 3. Degenerate Limit Proof — c = IC at n=1
# ═══════════════════════════════════════════════════════════════════


class TestScalarLimit:
    """Prove that Butzbach's scalar c equals GCD's IC when n=1."""

    @pytest.mark.parametrize(
        "c_val",
        [0.01, 0.1, 0.25, 0.5, 0.682, 0.75, 0.9, 0.95, 0.99],
    )
    def test_c_equals_IC_at_n1(self, c_val: float) -> None:
        """c = IC = F when n=1 (the degenerate limit proof)."""
        result = prove_scalar_limit(c_val)
        assert result["residual_c_vs_IC"] < TOL_DUALITY
        assert result["residual_c_vs_F"] < TOL_DUALITY
        assert result["is_degenerate_limit"] is True
        assert result["n_channels"] == 1

    @pytest.mark.parametrize(
        "c_val",
        [0.01, 0.1, 0.25, 0.5, 0.682, 0.75, 0.9, 0.95, 0.99],
    )
    def test_zero_gap_at_n1(self, c_val: float) -> None:
        """Heterogeneity gap Δ = 0 at n=1 (no heterogeneity with one channel)."""
        result = prove_scalar_limit(c_val)
        assert result["residual_gap"] < TOL_DUALITY

    @pytest.mark.parametrize(
        "c_val",
        [0.01, 0.1, 0.25, 0.5, 0.682, 0.75, 0.9, 0.95, 0.99],
    )
    def test_duality_at_n1(self, c_val: float) -> None:
        """F + ω = 1 (duality identity) holds at n=1."""
        result = prove_scalar_limit(c_val)
        assert result["duality_residual"] < TOL_DUALITY

    def test_scalar_limit_bulk(self) -> None:
        """Bulk proof: c = IC for 100 values across [0.01, 0.99]."""
        for c_val in np.linspace(0.01, 0.99, 100):
            result = prove_scalar_limit(float(c_val))
            assert result["residual_c_vs_IC"] < TOL_DUALITY, (
                f"Scalar limit failed at c={c_val}: residual={result['residual_c_vs_IC']}"
            )

    def test_no_channel_decomposition(self) -> None:
        """At n=1, channel decomposition is unavailable — this is the limitation."""
        result = prove_scalar_limit(0.5)
        assert result["channel_decomposition_available"] is False


# ═══════════════════════════════════════════════════════════════════
# 4. Cascade = Log-Integrity Accumulation
# ═══════════════════════════════════════════════════════════════════


class TestCascadeIdentity:
    """Prove that Butzbach's cascade C(t) = Π c(k) is exp(Σ ln c(k))."""

    def test_short_sequence(self) -> None:
        """Cascade identity verified for a short sequence."""
        result = prove_cascade_is_log_integrity([0.9, 0.8, 0.7])
        assert result["identity_verified"] is True
        assert result["max_residual"] < 1e-12

    def test_long_sequence(self) -> None:
        """Cascade identity verified for 50-step sequence."""
        c_seq = np.random.default_rng(42).uniform(0.5, 0.99, size=50).tolist()
        result = prove_cascade_is_log_integrity(c_seq)
        assert result["identity_verified"] is True
        assert result["max_residual"] < 1e-12

    def test_near_zero_values(self) -> None:
        """Cascade identity holds even near the guard band."""
        result = prove_cascade_is_log_integrity([0.01, 0.05, 0.1])
        assert result["identity_verified"] is True

    def test_high_fidelity_values(self) -> None:
        """Cascade identity holds for near-perfect continuity."""
        result = prove_cascade_is_log_integrity([0.99, 0.995, 0.999])
        assert result["identity_verified"] is True

    def test_cascade_equals_gcd_ic(self) -> None:
        """Butzbach cascade values equal exp(κ_accumulated) at every step."""
        seq = [0.85, 0.90, 0.75, 0.92, 0.88]
        result = prove_cascade_is_log_integrity(seq)
        for b_val, g_val in zip(result["butzbach_cascade"], result["gcd_ic_from_kappa"], strict=True):
            assert abs(b_val - g_val) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# 5. Geometric Slaughter — What Butzbach Cannot See
# ═══════════════════════════════════════════════════════════════════


class TestGeometricSlaughter:
    """Demonstrate that Butzbach's scalar c misses channel failure."""

    def test_butzbach_blind_to_single_channel_death(self) -> None:
        """When one channel dies, Butzbach still sees healthy c; GCD sees IC collapse."""
        result = demonstrate_geometric_slaughter(n_channels=8, healthy_value=0.95)
        summary = result["summary"]
        # Butzbach's c drops modestly (one channel out of 8)
        assert summary["butzbach_c_drop_pct"] < 15.0  # Less than 15% drop
        # GCD's IC drops dramatically (geometric slaughter)
        assert summary["gcd_IC_drop_pct"] > 85.0  # More than 85% drop
        # Butzbach is blind: c > 0.5 but IC < 0.1
        assert summary["butzbach_blind"] is True

    def test_gap_grows_as_channel_dies(self) -> None:
        """Heterogeneity gap Δ = F − IC grows as the killed channel value falls."""
        result = demonstrate_geometric_slaughter()
        traj = result["trajectory"]
        # Gap at start should be small (all channels healthy)
        assert traj[0]["heterogeneity_gap"] < 0.01
        # Gap at end should be large (one channel near ε)
        assert traj[-1]["heterogeneity_gap"] > 0.5

    def test_f_remains_high_while_ic_collapses(self) -> None:
        """F stays above 0.8 while IC drops below 0.15 — the invisible failure."""
        result = demonstrate_geometric_slaughter()
        last = result["trajectory"][-1]
        assert last["gcd_F"] > 0.80
        assert last["gcd_IC"] < 0.15

    def test_regime_transitions_during_slaughter(self) -> None:
        """Track regime transitions as single channel degrades."""
        result = demonstrate_geometric_slaughter()
        regimes = [r["regime"] for r in result["trajectory"]]
        # Should start in one regime and potentially change
        assert len(regimes) > 0
        # All regimes must be valid four-gate labels
        valid = {"Stable", "Watch", "Collapse"}
        for r in regimes:
            assert r in valid, f"Unexpected regime: {r}"


# ═══════════════════════════════════════════════════════════════════
# 6. Full System Analysis — Tier-1 Identity Sweep
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def all_analyses() -> list[EmbeddingResult]:
    """Analyze all 20 systems once for the test module."""
    return analyze_all_systems()


class TestTier1Identities:
    """Verify Tier-1 identities hold for all 20 systems."""

    def test_duality_all_systems(self, all_analyses: list[EmbeddingResult]) -> None:
        """F + ω = 1 for every system."""
        for a in all_analyses:
            residual = abs(a.gcd_F + a.gcd_omega - 1.0)
            assert residual < TOL_DUALITY, f"{a.name}: F+ω = {a.gcd_F + a.gcd_omega}, residual = {residual}"

    def test_integrity_bound_all_systems(self, all_analyses: list[EmbeddingResult]) -> None:
        """IC ≤ F for every system (integrity bound)."""
        for a in all_analyses:
            assert a.gcd_IC <= a.gcd_F + TOL_BOUND, f"{a.name}: IC={a.gcd_IC} > F={a.gcd_F}"

    def test_log_integrity_all_systems(self, all_analyses: list[EmbeddingResult]) -> None:
        """IC = exp(κ) for every system."""
        for a in all_analyses:
            ic_from_kappa = math.exp(a.gcd_kappa)
            residual = abs(a.gcd_IC - ic_from_kappa)
            assert residual < TOL_EXP, f"{a.name}: IC={a.gcd_IC}, exp(κ)={ic_from_kappa}, residual={residual}"

    def test_omega_in_unit_interval(self, all_analyses: list[EmbeddingResult]) -> None:
        """ω ∈ [0, 1] for every system."""
        for a in all_analyses:
            assert 0.0 <= a.gcd_omega <= 1.0, f"{a.name}: ω = {a.gcd_omega}"

    def test_fidelity_in_unit_interval(self, all_analyses: list[EmbeddingResult]) -> None:
        """F ∈ [0, 1] for every system."""
        for a in all_analyses:
            assert 0.0 <= a.gcd_F <= 1.0, f"{a.name}: F = {a.gcd_F}"

    def test_entropy_non_negative(self, all_analyses: list[EmbeddingResult]) -> None:
        """S ≥ 0 for every system."""
        for a in all_analyses:
            assert a.gcd_S >= -TOL_DUALITY, f"{a.name}: S = {a.gcd_S}"


class TestRegimeClassification:
    """Verify regime classification for all systems."""

    def test_all_systems_have_regime(self, all_analyses: list[EmbeddingResult]) -> None:
        """Every system is classified into a regime."""
        for a in all_analyses:
            assert a.gcd_regime in {"Stable", "Watch", "Collapse"}, f"{a.name}: unknown regime '{a.gcd_regime}'"

    def test_gcd_label_includes_critical_when_appropriate(self, all_analyses: list[EmbeddingResult]) -> None:
        """Critical overlay appears when IC < 0.30."""
        for a in all_analyses:
            if a.gcd_IC < 0.30:
                assert "Critical" in a.gcd_label, f"{a.name}: IC={a.gcd_IC} < 0.30 but no Critical overlay"
            else:
                assert "Critical" not in a.gcd_label, f"{a.name}: IC={a.gcd_IC} ≥ 0.30 but has Critical overlay"

    def test_inert_systems_have_regimes(self, all_analyses: list[EmbeddingResult]) -> None:
        """Inert systems (diamond, granite) still get regime classification."""
        inert = [a for a in all_analyses if a.category == "inert"]
        assert len(inert) >= 2
        for a in inert:
            assert a.gcd_regime in {"Stable", "Watch", "Collapse"}


# ═══════════════════════════════════════════════════════════════════
# 7. Blind Spot Detection
# ═══════════════════════════════════════════════════════════════════


class TestBlindSpots:
    """Detect systems where Butzbach sees healthy but GCD sees fragile."""

    def test_blind_spots_exist(self, all_analyses: list[EmbeddingResult]) -> None:
        """At least one system has a blind spot (the whole point of the embedding)."""
        blind = [a for a in all_analyses if a.blind_spot]
        assert len(blind) >= 1, "No blind spots found — embedding isn't demonstrating value"

    def test_blind_spot_definition(self, all_analyses: list[EmbeddingResult]) -> None:
        """Blind spot = Butzbach healthy (c>0.5) AND GCD fragile (Δ>0.1)."""
        for a in all_analyses:
            if a.blind_spot:
                assert a.butzbach_sees_healthy
                assert a.gcd_sees_fragile

    def test_channel_diagnostics_available(self, all_analyses: list[EmbeddingResult]) -> None:
        """Blind-spotted systems have channel diagnostics (unavailable to Butzbach)."""
        blind = [a for a in all_analyses if a.blind_spot]
        for a in blind:
            assert a.weakest_channel is not None
            assert a.strongest_channel is not None
            assert a.weakest_value < a.strongest_value

    def test_heterogeneity_gap_large_for_blind_spots(self, all_analyses: list[EmbeddingResult]) -> None:
        """Blind-spotted systems have heterogeneity gap > 0.1 by definition."""
        blind = [a for a in all_analyses if a.blind_spot]
        for a in blind:
            assert a.heterogeneity_gap > 0.1, f"{a.name}: gap = {a.heterogeneity_gap} but classified as blind spot"


# ═══════════════════════════════════════════════════════════════════
# 8. Rosetta and Tier Placement
# ═══════════════════════════════════════════════════════════════════


class TestRosetta:
    """Verify the cross-domain vocabulary mapping."""

    def test_rosetta_completeness(self) -> None:
        """All 8 Butzbach concepts are mapped."""
        assert len(ROSETTA) == 8

    def test_rosetta_has_required_fields(self) -> None:
        """Each Rosetta entry has butzbach, gcd, relation, what_gcd_adds."""
        for concept, mapping in ROSETTA.items():
            assert "butzbach" in mapping, f"Missing 'butzbach' in {concept}"
            assert "gcd" in mapping, f"Missing 'gcd' in {concept}"
            assert "relation" in mapping, f"Missing 'relation' in {concept}"
            assert "what_gcd_adds" in mapping, f"Missing 'what_gcd_adds' in {concept}"


class TestTierPlacement:
    """Verify that Butzbach's framework is correctly placed in Tier-2."""

    def test_tier_is_2(self) -> None:
        """Butzbach's framework is Tier-2 (Expansion Space)."""
        assert "Tier-2" in TIER_PLACEMENT["tier"]

    def test_has_justification(self) -> None:
        """Tier placement includes justification."""
        assert len(TIER_PLACEMENT["justification"]) > 50

    def test_has_priority_statement(self) -> None:
        """Tier placement includes priority/precedence documentation."""
        assert "priority" in TIER_PLACEMENT
        assert "Paulus" in TIER_PLACEMENT["priority"]

    def test_has_contributions(self) -> None:
        """Acknowledges what Butzbach contributes (strongman)."""
        assert "what_it_contributes" in TIER_PLACEMENT
        assert len(TIER_PLACEMENT["what_it_contributes"]) > 20

    def test_has_limitations(self) -> None:
        """Documents what Butzbach lacks."""
        assert "what_it_lacks" in TIER_PLACEMENT
        assert "Channel decomposition" in TIER_PLACEMENT["what_it_lacks"]


# ═══════════════════════════════════════════════════════════════════
# 9. Serialization / Output
# ═══════════════════════════════════════════════════════════════════


class TestSerialization:
    """Verify EmbeddingResult serialization."""

    def test_to_dict_structure(self, all_analyses: list[EmbeddingResult]) -> None:
        """to_dict() returns a properly structured dict."""
        d = all_analyses[0].to_dict()
        assert "name" in d
        assert "butzbach" in d
        assert "gcd" in d
        assert "channel_diagnostics" in d
        assert "blind_spot" in d

    def test_to_dict_butzbach_section(self, all_analyses: list[EmbeddingResult]) -> None:
        """Butzbach section has c, p_omega, sees_healthy, label."""
        d = all_analyses[0].to_dict()
        b = d["butzbach"]
        assert "c" in b
        assert "p_omega" in b
        assert "sees_healthy" in b
        assert "label" in b

    def test_to_dict_gcd_section(self, all_analyses: list[EmbeddingResult]) -> None:
        """GCD section has all kernel invariants."""
        d = all_analyses[0].to_dict()
        g = d["gcd"]
        for key in ["F", "omega", "IC", "kappa", "S", "C", "regime", "heterogeneity_gap"]:
            assert key in g, f"Missing '{key}' in GCD section"


# ═══════════════════════════════════════════════════════════════════
# 10. Full Validation Suite
# ═══════════════════════════════════════════════════════════════════


class TestValidationSuite:
    """Run the full embedded validation and check results."""

    @pytest.fixture(scope="class")
    def validation_results(self) -> dict:
        """Run validation once for the class."""
        return validate_embedding()

    def test_scalar_limit_passed(self, validation_results: dict) -> None:
        """Scalar limit proof passes across 100 values."""
        assert validation_results["scalar_limit"]["passed"]

    def test_cascade_proof_passed(self, validation_results: dict) -> None:
        """Cascade = log-integrity proof passes across 20 sequences."""
        assert validation_results["cascade_proof"]["passed"]

    def test_butzbach_blind_confirmed(self, validation_results: dict) -> None:
        """Geometric slaughter confirms Butzbach blindness."""
        assert validation_results["geometric_slaughter"]["butzbach_blind"]

    def test_tier1_identities_passed(self, validation_results: dict) -> None:
        """All Tier-1 identities hold for all 20 systems."""
        assert validation_results["tier1_identities"]["passed"]

    def test_overall_passed(self, validation_results: dict) -> None:
        """Overall validation passes."""
        assert validation_results["overall_passed"]

    def test_system_count_in_validation(self, validation_results: dict) -> None:
        """Validation covers all 20 systems."""
        assert validation_results["system_analysis"]["n_systems"] == 20

    def test_ic_drop_dramatic(self, validation_results: dict) -> None:
        """GCD IC drops more than 85% during geometric slaughter."""
        assert validation_results["geometric_slaughter"]["gcd_IC_drop_pct"] > 85.0

    def test_butzbach_c_drop_modest(self, validation_results: dict) -> None:
        """Butzbach c drops less than 15% during geometric slaughter."""
        assert validation_results["geometric_slaughter"]["butzbach_c_drop_pct"] < 15.0
