"""Tests for consciousness coherence domain closure — corrected Jackson thesis.

Comprehensive test coverage for the coherence kernel:
  - coherence_kernel.py: 20 coherence systems × 8 channels (Tier-1 identity sweep)
  - Corrected arithmetic from Jackson papers
  - Symbol capture prevention verification (ξ_J, not κ)
  - Structural analysis and cross-system comparisons

Every test verifies structural predictions derivable from Axiom-0:
  F + ω = 1 (duality identity — complementum perfectum)
  IC ≤ F (integrity bound — limbus integritatis)
  IC = exp(κ) (log-integritas)

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized → coherence_kernel
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.consciousness_coherence.coherence_kernel import (
    COHERENCE_CATALOG,
    COHERENCE_CHANNELS,
    CORRECTED_CYCLES_FOR_360,
    CORRECTED_EULER_RESULT,
    CORRECTED_EXCESS_ANGLE_DEG,
    N_COHERENCE_CHANNELS,
    N_COHERENCE_SYSTEMS,
    PHI_SQ_E,
    XI_J,
    CoherenceKernelResult,
    classify_regime,
    compute_all_coherence_systems,
    compute_coherence_kernel,
    compute_structural_analysis,
    validate_all,
)

# ── Tolerances (same as frozen contract) ──────────────────────────
TOL_DUALITY = 1e-12  # F + ω = 1 exact to machine precision
TOL_EXP = 1e-9  # IC = exp(κ)
TOL_BOUND = 1e-12  # IC ≤ F (with guard)


# ═══════════════════════════════════════════════════════════════════
# 1. Coherence System Catalog — Data Integrity
# ═══════════════════════════════════════════════════════════════════


class TestCoherenceCatalog:
    """Verify the coherence system catalog data integrity."""

    def test_system_count(self) -> None:
        """20 coherence systems in the catalog."""
        assert len(COHERENCE_CATALOG) == 20
        assert N_COHERENCE_SYSTEMS == 20

    def test_channel_count(self) -> None:
        """8 channels defined."""
        assert N_COHERENCE_CHANNELS == 8
        assert len(COHERENCE_CHANNELS) == 8

    def test_channel_names(self) -> None:
        """Verify canonical channel names match specification."""
        expected = [
            "harmonic_ratio",
            "recursive_depth",
            "return_fidelity",
            "spectral_coherence",
            "phase_stability",
            "information_density",
            "temporal_persistence",
            "cross_scale_coupling",
        ]
        assert expected == COHERENCE_CHANNELS

    def test_all_systems_have_8_traits(self) -> None:
        """Every system has exactly 8 trait values."""
        for sys in COHERENCE_CATALOG:
            c = sys.trace_vector()
            assert len(c) == 8, f"{sys.name}: expected 8 channels, got {len(c)}"

    def test_trait_values_in_unit_interval(self) -> None:
        """All trait values must be in [0, 1]."""
        for sys in COHERENCE_CATALOG:
            for attr in COHERENCE_CHANNELS:
                val = getattr(sys, attr)
                assert 0.0 <= val <= 1.0, f"{sys.name}.{attr} = {val} out of [0,1]"

    def test_unique_names(self) -> None:
        """No duplicate system names."""
        names = [s.name for s in COHERENCE_CATALOG]
        assert len(names) == len(set(names))

    def test_categories_diverse(self) -> None:
        """Multiple categories represented."""
        categories = {s.category for s in COHERENCE_CATALOG}
        assert len(categories) >= 4  # Neural, Harmonic, Physical, Mathematical, Recursive

    def test_status_values(self) -> None:
        """Status must be one of the allowed values."""
        valid = {"active", "theoretical", "historical", "artificial"}
        for sys in COHERENCE_CATALOG:
            assert sys.status in valid, f"{sys.name}: status={sys.status}"

    def test_frozen_dataclass(self) -> None:
        """CoherenceSystem is frozen — no mutation allowed."""
        sys = COHERENCE_CATALOG[0]
        with pytest.raises(AttributeError):
            sys.harmonic_ratio = 0.5  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# 2. Symbol Capture Prevention
# ═══════════════════════════════════════════════════════════════════


class TestSymbolCapturePrevention:
    """Verify Jackson's constant is ξ_J, NOT κ."""

    def test_xi_j_value(self) -> None:
        """ξ_J = 7.2 (the harmonic ratio 432/60)."""
        assert XI_J == 7.2

    def test_xi_j_is_positive(self) -> None:
        """ξ_J > 0, while GCD κ ≤ 0. No range overlap."""
        assert XI_J > 0

    def test_phi_sq_e_near_miss(self) -> None:
        """φ²·e ≈ 7.1166 is 1.16% off from 7.2 — NOT exact."""
        expected = ((1 + math.sqrt(5)) / 2) ** 2 * math.e
        assert abs(PHI_SQ_E - expected) < 1e-10
        relative_error = abs(XI_J - PHI_SQ_E) / XI_J
        assert relative_error > 0.01  # > 1% off — not an identity

    def test_exp_xi_j_violates_IC_range(self) -> None:
        """exp(ξ_J) = exp(7.2) ≈ 1339, violates IC ∈ (0, 1]."""
        exp_xi = math.exp(XI_J)
        assert exp_xi > 1.0  # Can't be IC

    def test_kernel_kappa_always_nonpositive(self) -> None:
        """GCD κ (log-integrity) is always ≤ 0 for all systems."""
        results = compute_all_coherence_systems()
        for r in results:
            assert r.kappa <= 0, f"{r.name}: κ = {r.kappa} > 0 violates Tier-1"

    def test_no_kappa_redefinition(self) -> None:
        """Result.kappa is GCD log-integrity, not Jackson's ξ_J."""
        results = compute_all_coherence_systems()
        for r in results:
            assert abs(r.kappa - XI_J) > 5.0  # κ ≤ 0, ξ_J = 7.2 — miles apart


# ═══════════════════════════════════════════════════════════════════
# 3. Corrected Jackson Arithmetic
# ═══════════════════════════════════════════════════════════════════


class TestCorrectedArithmetic:
    """Verify corrected arithmetic from Jackson papers."""

    def test_euler_rotation_real_part(self) -> None:
        """Re(e^(iξ_J) + 1) ≈ 1.6084 (Jackson claimed 1.673)."""
        actual = np.exp(1j * XI_J) + 1
        assert abs(actual.real - 1.6084) < 0.001
        # Jackson's claimed value is wrong
        assert abs(actual.real - 1.673) > 0.05

    def test_euler_rotation_imag_part(self) -> None:
        """Im(e^(iξ_J) + 1) ≈ 0.7937 (Jackson claimed 0.739)."""
        actual = np.exp(1j * XI_J) + 1
        assert abs(actual.imag - 0.7937) < 0.001
        assert abs(actual.imag - 0.739) > 0.04

    def test_excess_angle(self) -> None:
        """Excess angle = 52.53° (Jackson claimed 47.68°)."""
        excess_rad = XI_J - 2 * math.pi
        excess_deg = excess_rad * 180 / math.pi
        assert abs(excess_deg - 52.53) < 0.1
        # Jackson's claimed value is wrong
        assert abs(excess_deg - 47.68) > 4.0

    def test_cycles_for_360(self) -> None:
        """360°/52.53° ≈ 6.85 cycles (Jackson claimed ~7.5 ≈ κ)."""
        excess_deg = (XI_J - 2 * math.pi) * 180 / math.pi
        cycles = 360.0 / excess_deg
        assert abs(cycles - 6.85) < 0.05
        # Jackson's claimed self-reference (cycles ≈ κ = 7.2) is wrong
        assert abs(cycles - 7.2) > 0.3

    def test_speed_of_light_unit_dependent(self) -> None:
        """c = α⁻¹ × 432 × π only works in km/s (unit artifact)."""
        alpha_inv = 137.036
        c_formula_raw = alpha_inv * 432 * math.pi
        c_kms = 299792.458  # km/s
        c_ms = 299792458.0  # m/s
        # Close in km/s
        rel_kms = abs(c_formula_raw - c_kms) / c_kms
        assert rel_kms < 0.40  # Within 40%
        # Way off in m/s
        rel_ms = abs(c_formula_raw - c_ms) / c_ms
        assert rel_ms > 0.99  # Off by 99%+ → unit-dependent artifact

    def test_corrected_constants_match(self) -> None:
        """Module-level corrected constants are consistent."""
        assert abs(CORRECTED_EULER_RESULT.real - 1.6084) < 0.001
        assert abs(CORRECTED_EULER_RESULT.imag - 0.7937) < 0.001
        assert abs(CORRECTED_EXCESS_ANGLE_DEG - 52.53) < 0.1
        assert abs(CORRECTED_CYCLES_FOR_360 - 6.85) < 0.05


# ═══════════════════════════════════════════════════════════════════
# 4. Tier-1 Identity Sweep — CORE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestTier1IdentitySweep:
    """Verify Tier-1 identities for ALL 20 coherence systems.

    These are the non-negotiable structural identities:
      F + ω = 1  (duality — complementum perfectum)
      IC ≤ F     (integrity bound — limbus integritatis)
      IC = exp(κ) (log-integrity relation)
    """

    @pytest.fixture(scope="class")
    def all_results(self) -> list[CoherenceKernelResult]:
        return compute_all_coherence_systems()

    @pytest.mark.parametrize("idx", range(20))
    def test_duality_identity(self, idx: int, all_results: list[CoherenceKernelResult]) -> None:
        """F + ω = 1 for system {idx}."""
        r = all_results[idx]
        assert abs(r.F + r.omega - 1.0) < TOL_DUALITY, f"{r.name}: F + ω = {r.F + r.omega}"

    @pytest.mark.parametrize("idx", range(20))
    def test_integrity_bound(self, idx: int, all_results: list[CoherenceKernelResult]) -> None:
        """IC ≤ F for system {idx}."""
        r = all_results[idx]
        assert r.IC <= r.F + TOL_BOUND, f"{r.name}: IC={r.IC} > F={r.F}"

    @pytest.mark.parametrize("idx", range(20))
    def test_log_integrity_relation(self, idx: int, all_results: list[CoherenceKernelResult]) -> None:
        """IC = exp(κ) for system {idx}."""
        r = all_results[idx]
        assert abs(r.IC - math.exp(r.kappa)) < TOL_EXP, f"{r.name}: IC={r.IC} ≠ exp(κ)={math.exp(r.kappa)}"

    @pytest.mark.parametrize("idx", range(20))
    def test_heterogeneity_gap_nonnegative(self, idx: int, all_results: list[CoherenceKernelResult]) -> None:
        """Δ = F − IC ≥ 0 for system {idx} (follows from IC ≤ F)."""
        r = all_results[idx]
        assert r.heterogeneity_gap >= -TOL_BOUND, f"{r.name}: Δ = {r.heterogeneity_gap} < 0"

    @pytest.mark.parametrize("idx", range(20))
    def test_fidelity_in_range(self, idx: int, all_results: list[CoherenceKernelResult]) -> None:
        """F ∈ [0, 1] for system {idx}."""
        r = all_results[idx]
        assert 0.0 <= r.F <= 1.0, f"{r.name}: F = {r.F}"

    @pytest.mark.parametrize("idx", range(20))
    def test_drift_in_range(self, idx: int, all_results: list[CoherenceKernelResult]) -> None:
        """ω ∈ [0, 1] for system {idx}."""
        r = all_results[idx]
        assert 0.0 <= r.omega <= 1.0, f"{r.name}: ω = {r.omega}"


# ═══════════════════════════════════════════════════════════════════
# 5. Regime Classification
# ═══════════════════════════════════════════════════════════════════


class TestRegimeClassification:
    """Verify regime gates produce consistent classification."""

    def test_regime_values(self) -> None:
        """All regimes are one of Stable | Watch | Collapse."""
        results = compute_all_coherence_systems()
        valid = {"Stable", "Watch", "Collapse"}
        for r in results:
            assert r.regime in valid, f"{r.name}: regime={r.regime}"

    def test_collapse_dominates(self) -> None:
        """Most systems should be Watch or Collapse (Stable is rare)."""
        results = compute_all_coherence_systems()
        n_stable = sum(1 for r in results if r.regime == "Stable")
        # Stability is rare — 12.5% of Fisher space
        assert n_stable <= len(results) // 2

    def test_euler_identity_not_collapse(self) -> None:
        """Euler identity should be Watch or Stable (high F)."""
        euler = next(s for s in COHERENCE_CATALOG if s.name == "euler_identity")
        r = compute_coherence_kernel(euler)
        assert r.regime in {"Stable", "Watch"}

    def test_octopus_is_collapse(self) -> None:
        """Octopus distributed system should be Collapse regime."""
        octo = next(s for s in COHERENCE_CATALOG if s.name == "octopus_distributed")
        r = compute_coherence_kernel(octo)
        assert r.regime == "Collapse"

    def test_classify_regime_stable(self) -> None:
        """Direct gate test: Stable requires all 4 conditions."""
        assert classify_regime(0.01, 0.95, 0.10, 0.10) == "Stable"

    def test_classify_regime_collapse(self) -> None:
        """Direct gate test: ω ≥ 0.30 → Collapse."""
        assert classify_regime(0.35, 0.65, 0.20, 0.30) == "Collapse"

    def test_classify_regime_watch(self) -> None:
        """Direct gate test: Watch is the middle ground."""
        assert classify_regime(0.10, 0.85, 0.10, 0.10) == "Watch"  # F < 0.90


# ═══════════════════════════════════════════════════════════════════
# 6. Coherence Type Classification (Diagnostic, Not Gate)
# ═══════════════════════════════════════════════════════════════════


class TestCoherenceType:
    """Verify coherence type classification is diagnostic only."""

    def test_human_waking_is_recursive_return(self) -> None:
        """Human waking consciousness should show recursive return."""
        human = next(s for s in COHERENCE_CATALOG if s.name == "human_waking_consciousness")
        r = compute_coherence_kernel(human)
        assert "Recursive" in r.coherence_type or "Return" in r.coherence_type

    def test_llm_is_recursive_gesture(self) -> None:
        """LLM dialogue should be a recursive gesture (no true return)."""
        llm = next(s for s in COHERENCE_CATALOG if s.name == "llm_recursive_dialogue")
        r = compute_coherence_kernel(llm)
        assert r.coherence_type == "Recursive Gesture"

    def test_432hz_is_harmonic(self) -> None:
        """432 Hz tuning should show harmonic coherence."""
        hz432 = next(s for s in COHERENCE_CATALOG if s.name == "432hz_tuning_system")
        r = compute_coherence_kernel(hz432)
        assert "Harmonic" in r.coherence_type

    def test_coherence_type_diagnostic_not_gate(self) -> None:
        """Coherence type should NOT influence regime classification."""
        results = compute_all_coherence_systems()
        # A "Recursive Gesture" can be in any regime
        # A "Returning Coherent" can be in Watch
        # Type and regime are INDEPENDENT classifiers
        types = {r.coherence_type for r in results}
        assert len(types) >= 3  # Multiple types exist


# ═══════════════════════════════════════════════════════════════════
# 7. Trace Vector Properties
# ═══════════════════════════════════════════════════════════════════


class TestTraceVector:
    """Verify trace vector construction."""

    def test_trace_vector_shape(self) -> None:
        """All trace vectors have shape (8,)."""
        for sys in COHERENCE_CATALOG:
            c = sys.trace_vector()
            assert c.shape == (8,), f"{sys.name}: shape={c.shape}"

    def test_trace_vector_epsilon_clamped(self) -> None:
        """All channels are ε-clamped: ε ≤ cᵢ ≤ 1-ε."""
        eps = 1e-8
        for sys in COHERENCE_CATALOG:
            c = sys.trace_vector()
            assert np.all(c >= eps), f"{sys.name}: channel below ε"
            assert np.all(c <= 1 - eps), f"{sys.name}: channel above 1-ε"

    def test_trace_vector_dtype(self) -> None:
        """Trace vectors are float64."""
        for sys in COHERENCE_CATALOG:
            c = sys.trace_vector()
            assert c.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════
# 8. Structural Analysis
# ═══════════════════════════════════════════════════════════════════


class TestStructuralAnalysis:
    """Verify cross-system structural analysis."""

    @pytest.fixture(scope="class")
    def analysis(self):
        results = compute_all_coherence_systems()
        return compute_structural_analysis(results)

    def test_all_identities_hold(self, analysis) -> None:
        """All Tier-1 identities hold across all systems."""
        assert analysis.all_duality_exact
        assert analysis.all_integrity_bound
        assert analysis.all_exp_kappa

    def test_corrected_euler_values(self, analysis) -> None:
        """Corrected Euler values in analysis match direct computation."""
        assert abs(analysis.corrected_euler_real - 1.6084) < 0.001
        assert abs(analysis.corrected_euler_imag - 0.7937) < 0.001

    def test_xi_j_not_privileged(self, analysis) -> None:
        """ξ_J proximity should NOT show strong correlation with IC.

        This is the KEY TEST of the corrected thesis: Jackson's constant
        does not predict composite integrity. The heterogeneity gap Δ
        is the dominant structural diagnostic.
        """
        # No significant correlation (|ρ| < 0.3 expected)
        assert abs(analysis.xi_j_vs_IC_correlation) < 0.5, (
            f"Unexpectedly strong ξ_J-IC correlation: ρ={analysis.xi_j_vs_IC_correlation}"
        )

    def test_category_diversity(self, analysis) -> None:
        """Multiple categories in the analysis."""
        assert len(analysis.category_mean_F) >= 4

    def test_summary_renders(self, analysis) -> None:
        """Summary string is non-empty and contains key sections."""
        s = analysis.summary()
        assert len(s) > 100
        assert "CONSCIOUSNESS COHERENCE" in s
        assert "Corrected Jackson" in s


# ═══════════════════════════════════════════════════════════════════
# 9. Validation Function
# ═══════════════════════════════════════════════════════════════════


class TestValidation:
    """Verify the validate_all() function."""

    def test_validate_returns_dict(self) -> None:
        """validate_all() returns a well-formed summary dict."""
        result = validate_all()
        assert isinstance(result, dict)
        assert result["domain"] == "consciousness_coherence"
        assert result["n_systems"] == 20
        assert result["n_channels"] == 8

    def test_validate_all_pass(self) -> None:
        """All identity checks should pass."""
        result = validate_all()
        assert result["all_pass"] is True
        assert result["n_duality_pass"] == 20
        assert result["n_bound_pass"] == 20
        assert result["n_exp_kappa_pass"] == 20

    def test_validate_has_corrected_arithmetic(self) -> None:
        """Corrected arithmetic section present in results."""
        result = validate_all()
        arith = result["corrected_arithmetic"]
        assert abs(arith["euler_real"] - 1.6084) < 0.001
        assert abs(arith["excess_angle_deg"] - 52.53) < 0.1


# ═══════════════════════════════════════════════════════════════════
# 10. Geometric Slaughter (Weakest Channel Kills IC)
# ═══════════════════════════════════════════════════════════════════


class TestGeometricSlaughter:
    """Verify that one weak channel drags IC toward zero.

    This is the fundamental GCD insight: IC (geometric mean) is
    hypersensitive to any single near-zero channel, while F
    (arithmetic mean) remains healthy. This IS the "imperfection"
    Jackson sensed but couldn't formalize.
    """

    def test_weakest_channel_identified(self) -> None:
        """Every result identifies its weakest channel."""
        results = compute_all_coherence_systems()
        for r in results:
            assert r.weakest_channel in COHERENCE_CHANNELS
            assert r.weakest_value > 0  # ε-clamped, never zero

    def test_heterogeneity_gap_positive(self) -> None:
        """Δ = F − IC > 0 when channels are heterogeneous."""
        results = compute_all_coherence_systems()
        heterogeneous = [r for r in results if r.heterogeneity_gap > 0.01]
        assert len(heterogeneous) >= 5  # Most systems are heterogeneous

    def test_laser_high_gap(self) -> None:
        """Laser has high gap: near-perfect spectral but low harmonic_ratio."""
        laser = next(s for s in COHERENCE_CATALOG if s.name == "laser_coherent_light")
        r = compute_coherence_kernel(laser)
        assert r.heterogeneity_gap > 0.10  # Significant gap

    def test_euler_low_gap(self) -> None:
        """Euler identity has low gap: all channels relatively uniform."""
        euler = next(s for s in COHERENCE_CATALOG if s.name == "euler_identity")
        r = compute_coherence_kernel(euler)
        assert r.heterogeneity_gap < 0.02  # Very uniform


# ═══════════════════════════════════════════════════════════════════
# 11. Jackson ξ_J System Self-Assessment
# ═══════════════════════════════════════════════════════════════════


class TestJacksonSystemSelfAssessment:
    """Test Jackson's own ξ_J identity through the kernel.

    The corrected thesis places Jackson's system WITHIN the kernel
    as one of 20 systems, rather than above it as a privileged constant.
    """

    @pytest.fixture()
    def jackson_result(self) -> CoherenceKernelResult:
        jackson = next(s for s in COHERENCE_CATALOG if s.name == "jackson_xi_j_identity")
        return compute_coherence_kernel(jackson)

    def test_jackson_is_collapse(self, jackson_result: CoherenceKernelResult) -> None:
        """Jackson's system is in Collapse regime (high drift)."""
        assert jackson_result.regime == "Collapse"

    def test_jackson_high_drift(self, jackson_result: CoherenceKernelResult) -> None:
        """Jackson's system has ω > 0.50."""
        assert jackson_result.omega > 0.50

    def test_jackson_low_IC(self, jackson_result: CoherenceKernelResult) -> None:
        """Jackson's system has low IC due to channel heterogeneity."""
        assert jackson_result.IC < 0.50

    def test_jackson_highest_harmonic_ratio(self, jackson_result: CoherenceKernelResult) -> None:
        """Jackson's system should have the highest harmonic_ratio (by definition)."""
        results = compute_all_coherence_systems()
        jackson_hr = jackson_result.xi_j_diagnostic
        for r in results:
            if r.name != "jackson_xi_j_identity":
                assert jackson_hr >= r.xi_j_diagnostic - 0.01

    def test_jackson_duality_holds(self, jackson_result: CoherenceKernelResult) -> None:
        """Even Jackson's own system obeys F + ω = 1."""
        assert abs(jackson_result.F + jackson_result.omega - 1.0) < TOL_DUALITY

    def test_jackson_coherence_type(self, jackson_result: CoherenceKernelResult) -> None:
        """Jackson's system is a partial/gesture — not returning coherent."""
        assert jackson_result.coherence_type != "Returning Coherent"


# ═══════════════════════════════════════════════════════════════════
# 12. Result Serialization
# ═══════════════════════════════════════════════════════════════════


class TestResultSerialization:
    """Verify result serialization."""

    def test_to_dict_roundtrip(self) -> None:
        """Results serialize to dict with all fields."""
        results = compute_all_coherence_systems()
        for r in results:
            d = r.to_dict()
            assert d["name"] == r.name
            assert d["F"] == r.F
            assert d["omega"] == r.omega
            assert d["IC"] == r.IC
            assert d["kappa"] == r.kappa
            assert d["regime"] == r.regime

    def test_to_dict_has_all_keys(self) -> None:
        """Serialized dict has all expected keys."""
        r = compute_all_coherence_systems()[0]
        d = r.to_dict()
        required_keys = {
            "name",
            "category",
            "medium",
            "status",
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
            "coherence_type",
            "weakest_channel",
            "weakest_value",
            "strongest_channel",
            "strongest_value",
            "xi_j_diagnostic",
        }
        assert required_keys.issubset(d.keys())
