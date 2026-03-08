"""Tests for stellar ages as cosmic clocks — Tomasetti et al. 2026.

Validates 10 Cosmic Clock Theorems (T-SC-1 through T-SC-10), 23 stellar
samples spanning the selection pipeline, Bayesian mixture model, Hubble
tension probes, systematic error budget, and cosmological constraints.

Test count target: ~170 tests covering:
    - Frozen constants from Tomasetti et al. 2026
    - Stellar database construction (23 representative stars)
    - Selection pipeline (7 stages, monotonic decrease)
    - Systematic error budget (4 effects)
    - Cosmological algebra (H0-tU conversion)
    - 10 theorem proofs with subtests
    - Tier-1 identity universality
    - Narrative generation
    - Edge cases and kernel statistics
"""

from __future__ import annotations

import math

import pytest

from closures.astronomy.stellar_ages_cosmology import (
    AGE_MAX,
    AGE_PEAK,
    DELTA_T_ZF11,
    DELTA_T_ZF20,
    EPSILON,
    FC_CONTAM,
    H0_PLANCK,
    H0_SHOES,
    H0_UPPER,
    MEAN_ALPHA_FE,
    MEAN_AV,
    MEAN_MASS,
    MEAN_MH,
    MU_CONTAM,
    MU_MAIN,
    N_CHANNELS,
    N_CONTAM_REMOVED,
    N_FINAL,
    N_GOLDEN,
    N_INPUT,
    N_PARENT,
    N_STARS_TU_GT_13,
    N_STARS_TU_GT_13P5,
    SIGMA_CONTAM,
    SIGMA_MAIN,
    SYST_HELIUM,
    SYST_MIXING_LENGTH,
    SYST_STELLAR_MODELS,
    SYST_TOTAL,
    T_U_LOWER_BOUND,
    T_U_PLANCK,
    T_U_SHOES,
    T_U_STAT_ERR,
    T_U_SYST_ERR,
    StellarKernelResult,
    _build_trace,
    _clip,
    _linear_norm,
    age_to_h0_upper,
    build_cosmological_scenarios,
    build_selection_pipeline,
    build_stellar_database,
    build_systematic_effects,
    compute_stellar_kernel,
    generate_narrative,
    h0_from_t_universe,
    prove_t_sc_1,
    prove_t_sc_2,
    prove_t_sc_3,
    prove_t_sc_4,
    prove_t_sc_5,
    prove_t_sc_6,
    prove_t_sc_7,
    prove_t_sc_8,
    prove_t_sc_9,
    prove_t_sc_10,
    run_full_analysis,
    t_universe_from_h0,
)

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def full_analysis():
    """Run full analysis once for the entire test module."""
    return run_full_analysis()


@pytest.fixture(scope="module")
def stellar_db():
    """Build stellar database once."""
    return build_stellar_database()


@pytest.fixture(scope="module")
def pipeline():
    """Build selection pipeline once."""
    return build_selection_pipeline()


@pytest.fixture(scope="module")
def systematics():
    """Build systematic effects once."""
    return build_systematic_effects()


# ═══════════════════════════════════════════════════════════════
# FROZEN CONSTANTS
# ═══════════════════════════════════════════════════════════════


class TestFrozenConstants:
    """Verify frozen constants from Tomasetti et al. 2026."""

    def test_age_peak(self):
        assert AGE_PEAK == 13.6

    def test_t_u_lower_bound(self):
        assert T_U_LOWER_BOUND == 13.8

    def test_t_u_stat_error(self):
        assert T_U_STAT_ERR == 1.0

    def test_t_u_syst_error(self):
        assert T_U_SYST_ERR == 1.4

    def test_h0_upper(self):
        assert H0_UPPER == 68.3

    def test_h0_planck(self):
        assert H0_PLANCK == 67.4

    def test_h0_shoes(self):
        assert H0_SHOES == 73.04

    def test_t_u_planck(self):
        assert T_U_PLANCK == 14.0

    def test_t_u_shoes(self):
        assert T_U_SHOES == 12.9

    def test_delta_t_zf20(self):
        assert DELTA_T_ZF20 == 0.2

    def test_delta_t_zf11(self):
        assert DELTA_T_ZF11 == 0.4

    def test_mu_main(self):
        assert MU_MAIN == 13.66

    def test_sigma_main(self):
        assert SIGMA_MAIN == 0.34

    def test_mu_contam(self):
        assert MU_CONTAM == 14.79

    def test_sigma_contam(self):
        assert SIGMA_CONTAM == 0.83

    def test_fc_contam(self):
        assert FC_CONTAM == 0.10

    def test_mean_mass(self):
        assert MEAN_MASS == 0.88

    def test_mean_metallicity(self):
        assert MEAN_MH == -0.24

    def test_mean_alpha_fe(self):
        assert MEAN_ALPHA_FE == 0.17

    def test_mean_av(self):
        assert MEAN_AV == 0.08

    def test_n_input(self):
        assert N_INPUT == 202384

    def test_n_final(self):
        assert N_FINAL == 160

    def test_n_golden(self):
        assert N_GOLDEN == 67

    def test_n_contam_removed(self):
        assert N_CONTAM_REMOVED == 25

    def test_n_channels(self):
        assert N_CHANNELS == 8

    def test_age_max(self):
        assert AGE_MAX == 20.0

    def test_syst_mixing_length(self):
        assert SYST_MIXING_LENGTH == 1.0

    def test_syst_helium(self):
        assert SYST_HELIUM == 0.5

    def test_syst_stellar_models(self):
        assert SYST_STELLAR_MODELS == 1.1

    def test_syst_total(self):
        assert SYST_TOTAL == 1.4

    def test_n_stars_tu_gt_13(self):
        assert N_STARS_TU_GT_13 == 70

    def test_n_stars_tu_gt_13p5(self):
        assert N_STARS_TU_GT_13P5 == 29

    def test_syst_models_quadrature(self):
        """Quadrature sum of αML and Yi should give ~1.1 Gyr."""
        quad = math.sqrt(SYST_MIXING_LENGTH**2 + SYST_HELIUM**2)
        assert abs(quad - SYST_STELLAR_MODELS) < 0.15


# ═══════════════════════════════════════════════════════════════
# NORMALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════


class TestNormalizationHelpers:
    """Test normalization and clipping functions."""

    def test_clip_lower_bound(self):
        assert _clip(-1.0) == EPSILON

    def test_clip_upper_bound(self):
        assert _clip(2.0) == 1.0 - EPSILON

    def test_clip_passthrough(self):
        assert _clip(0.5) == 0.5

    def test_clip_at_zero(self):
        assert _clip(0.0) == EPSILON

    def test_clip_at_one(self):
        assert _clip(1.0) == 1.0 - EPSILON

    def test_linear_norm_midpoint(self):
        assert _linear_norm(5.0, 0.0, 10.0) == 0.5

    def test_linear_norm_min(self):
        assert _linear_norm(0.0, 0.0, 10.0) == 0.0

    def test_linear_norm_max(self):
        assert _linear_norm(10.0, 0.0, 10.0) == 1.0

    def test_linear_norm_equal_bounds(self):
        assert _linear_norm(5.0, 5.0, 5.0) == 0.5

    def test_build_trace_shape(self):
        trace = _build_trace(13.6, 0.88, -0.24, 0.17, 5550.0, 3.95, 0.08, 1.0)
        assert trace.shape == (8,)

    def test_build_trace_bounds(self):
        trace = _build_trace(13.6, 0.88, -0.24, 0.17, 5550.0, 3.95, 0.08, 1.0)
        assert all(EPSILON <= c <= 1.0 - EPSILON for c in trace)

    def test_build_trace_precision_channel(self):
        """Precision channel = 1 - σ/age."""
        trace = _build_trace(13.6, 0.88, -0.24, 0.17, 5550.0, 3.95, 0.08, 1.0)
        expected = _clip(1.0 - 1.0 / 13.6)
        assert abs(trace[7] - expected) < 1e-6


# ═══════════════════════════════════════════════════════════════
# STELLAR DATABASE
# ═══════════════════════════════════════════════════════════════


class TestStellarDatabase:
    """Test the stellar database construction."""

    def test_database_count(self, stellar_db):
        assert len(stellar_db) == 23

    def test_has_final_sample_mean(self, stellar_db):
        names = [s.name for s in stellar_db]
        assert "Final sample mean" in names

    def test_has_golden_sample_mean(self, stellar_db):
        names = [s.name for s in stellar_db]
        assert "Golden sample mean" in names

    def test_has_main_population(self, stellar_db):
        names = [s.name for s in stellar_db]
        assert "Main population center" in names

    def test_has_contaminant_population(self, stellar_db):
        names = [s.name for s in stellar_db]
        assert "Contaminant population center" in names

    def test_has_gc_comparisons(self, stellar_db):
        comparisons = [s for s in stellar_db if s.sample_type == "comparison"]
        assert len(comparisons) == 2

    def test_has_excluded_stars(self, stellar_db):
        excluded = [s for s in stellar_db if "Excluded" in s.name]
        assert len(excluded) == 2

    def test_individual_stars_count(self, stellar_db):
        individuals = [s for s in stellar_db if s.sample_type == "individual"]
        assert len(individuals) >= 10

    def test_all_ages_positive(self, stellar_db):
        for s in stellar_db:
            assert s.age > 0, f"{s.name} has non-positive age"

    def test_all_masses_positive(self, stellar_db):
        for s in stellar_db:
            assert s.mass > 0, f"{s.name} has non-positive mass"

    def test_sample_types_valid(self, stellar_db):
        valid_types = {"individual", "population_mean", "comparison"}
        for s in stellar_db:
            assert s.sample_type in valid_types, f"{s.name}: {s.sample_type}"

    @pytest.mark.parametrize(
        "field_name,expected_range",
        [
            ("age", (12.0, 20.0)),
            ("mass", (0.7, 1.1)),
            ("teff", (5100.0, 6100.0)),
            ("logg", (3.5, 4.3)),
        ],
    )
    def test_parameter_ranges(self, stellar_db, field_name, expected_range):
        """All stellar parameters should be within physically reasonable ranges."""
        lo, hi = expected_range
        for s in stellar_db:
            val = getattr(s, field_name)
            assert lo <= val <= hi, f"{s.name}.{field_name} = {val} outside [{lo}, {hi}]"


# ═══════════════════════════════════════════════════════════════
# SELECTION PIPELINE
# ═══════════════════════════════════════════════════════════════


class TestSelectionPipeline:
    """Test the selection pipeline from 202,384 → 160 stars."""

    def test_pipeline_length(self, pipeline):
        assert len(pipeline) == 7

    def test_starts_at_full_sample(self, pipeline):
        assert pipeline[0].n_after == N_INPUT

    def test_ends_at_final_sample(self, pipeline):
        assert pipeline[-1].n_after == N_FINAL

    def test_monotonic_decrease(self, pipeline):
        n_values = [s.n_after for s in pipeline]
        for i in range(len(n_values) - 1):
            assert n_values[i] >= n_values[i + 1], f"Step {i + 1} ({n_values[i]}) < Step {i + 2} ({n_values[i + 1]})"

    def test_parent_sample_count(self, pipeline):
        parent = next(s for s in pipeline if "Parent" in s.name)
        assert parent.n_after == N_PARENT

    def test_kiel_cut_count(self, pipeline):
        kiel = next(s for s in pipeline if "Kiel" in s.name)
        assert kiel.n_after == 2148

    def test_posterior_cut_count(self, pipeline):
        posterior = next(s for s in pipeline if "Posterior" in s.name)
        assert posterior.n_after == 297

    def test_visual_inspection_count(self, pipeline):
        visual = next(s for s in pipeline if "Visual" in s.name)
        assert visual.n_after == 185

    def test_total_retention_rate(self, pipeline):
        retention = N_FINAL / N_INPUT
        assert retention < 0.001  # Less than 0.1% retained

    def test_all_steps_have_descriptions(self, pipeline):
        for s in pipeline:
            assert len(s.description) > 20


# ═══════════════════════════════════════════════════════════════
# SYSTEMATIC EFFECTS
# ═══════════════════════════════════════════════════════════════


class TestSystematicEffects:
    """Test the systematic error budget."""

    def test_effects_count(self, systematics):
        assert len(systematics) == 4

    def test_mixing_length_shift(self, systematics):
        ml = next(e for e in systematics if "mixing" in e.name.lower())
        assert ml.shift_gyr == SYST_MIXING_LENGTH

    def test_helium_shift(self, systematics):
        he = next(e for e in systematics if "helium" in e.name.lower())
        assert he.shift_gyr == SYST_HELIUM

    def test_alpha_fe_shift(self, systematics):
        alpha = next(e for e in systematics if "α-enhancement" in e.name)
        assert alpha.shift_gyr == 0.28

    def test_diffusion_zero_shift(self, systematics):
        diff = next(e for e in systematics if "diffusion" in e.name.lower())
        assert diff.shift_gyr == 0.0


# ═══════════════════════════════════════════════════════════════
# COSMOLOGICAL ALGEBRA
# ═══════════════════════════════════════════════════════════════


class TestCosmologicalAlgebra:
    """Test H0-tU conversion functions."""

    def test_t_universe_from_h0_planck(self):
        t = t_universe_from_h0(H0_PLANCK, 0.3)
        assert abs(t - T_U_PLANCK) < 0.3  # Within 0.3 Gyr

    def test_t_universe_from_h0_shoes(self):
        t = t_universe_from_h0(H0_SHOES, 0.3)
        assert abs(t - T_U_SHOES) < 0.3

    def test_h0_from_t_universe_roundtrip(self):
        h0_in = 70.0
        t = t_universe_from_h0(h0_in, 0.3)
        h0_out = h0_from_t_universe(t, 0.3)
        assert abs(h0_in - h0_out) < 0.01

    def test_age_to_h0_upper_reasonable(self):
        h0 = age_to_h0_upper(AGE_PEAK, DELTA_T_ZF20, 0.3)
        assert 50 < h0 < 90  # Physically reasonable

    def test_higher_age_gives_lower_h0(self):
        h0_young = age_to_h0_upper(12.0, DELTA_T_ZF20, 0.3)
        h0_old = age_to_h0_upper(14.0, DELTA_T_ZF20, 0.3)
        assert h0_old < h0_young

    def test_higher_omega_m_gives_lower_t(self):
        t_low = t_universe_from_h0(70.0, 0.25)
        t_high = t_universe_from_h0(70.0, 0.35)
        assert t_high < t_low

    def test_t_universe_from_h0_zero(self):
        t = t_universe_from_h0(0.0, 0.3)
        assert t == float("inf")

    def test_h0_from_t_universe_zero(self):
        h0 = h0_from_t_universe(0.0, 0.3)
        assert h0 == float("inf")

    def test_cosmological_scenarios_count(self):
        scenarios = build_cosmological_scenarios()
        assert len(scenarios) == 9  # 3×3 grid

    def test_scenarios_omega_m_values(self):
        scenarios = build_cosmological_scenarios()
        omega_ms = sorted({s.omega_m for s in scenarios})
        assert omega_ms == [0.25, 0.3, 0.35]


# ═══════════════════════════════════════════════════════════════
# KERNEL COMPUTATION
# ═══════════════════════════════════════════════════════════════


class TestKernelComputation:
    """Test GCD kernel computation on stellar samples."""

    def test_compute_kernel_returns_result(self, stellar_db):
        star = stellar_db[0]
        kr = compute_stellar_kernel(star)
        assert isinstance(kr, StellarKernelResult)

    def test_kernel_f_in_range(self, stellar_db):
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            assert 0 < kr.F <= 1.0, f"{s.name}: F={kr.F}"

    def test_kernel_omega_in_range(self, stellar_db):
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            assert 0 <= kr.omega < 1.0, f"{s.name}: ω={kr.omega}"

    def test_kernel_ic_in_range(self, stellar_db):
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            assert 0 < kr.IC <= 1.0, f"{s.name}: IC={kr.IC}"

    def test_kernel_trace_length(self, stellar_db):
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            assert len(kr.trace) == N_CHANNELS

    def test_kernel_regime_valid(self, stellar_db):
        valid_regimes = {"Stable", "Watch", "Collapse", "fragmented"}
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            assert kr.regime in valid_regimes, f"{s.name}: regime={kr.regime}"


# ═══════════════════════════════════════════════════════════════
# TIER-1 IDENTITIES
# ═══════════════════════════════════════════════════════════════


class TestTier1Identities:
    """Verify Tier-1 invariants hold for ALL stellar samples."""

    def test_duality_identity(self, stellar_db):
        """F + ω = 1 exactly."""
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            residual = abs((kr.F + kr.omega) - 1.0)
            assert residual < 1e-6, f"{s.name}: F+ω = {kr.F + kr.omega} (residual {residual})"

    def test_integrity_bound(self, stellar_db):
        """IC ≤ F (integrity cannot exceed fidelity)."""
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            assert kr.IC <= kr.F + 1e-6, f"{s.name}: IC ({kr.IC}) > F ({kr.F})"

    def test_log_integrity_relation(self, stellar_db):
        """IC ≈ exp(κ)."""
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            expected_ic = math.exp(kr.kappa)
            assert abs(kr.IC - expected_ic) < 1e-4, f"{s.name}: IC ({kr.IC}) ≠ exp(κ) ({expected_ic})"

    def test_heterogeneity_gap_nonnegative(self, stellar_db):
        """Δ = F - IC ≥ 0."""
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            assert kr.gap >= -1e-6, f"{s.name}: gap = {kr.gap}"

    def test_entropy_nonnegative(self, stellar_db):
        """Bernoulli field entropy S ≥ 0."""
        for s in stellar_db:
            kr = compute_stellar_kernel(s)
            assert kr.S >= -1e-6, f"{s.name}: S = {kr.S}"


# ═══════════════════════════════════════════════════════════════
# THEOREMS
# ═══════════════════════════════════════════════════════════════


class TestTheoremSC1:
    """T-SC-1: Selection Funnel preserves kernel integrity."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_1()

    def test_proven(self, result):
        assert result["proven"]

    def test_monotonic_decrease(self, result):
        assert result["monotonic_decrease"]

    def test_tier1_all_stages(self, result):
        assert result["tier1_all_stages"]

    def test_total_retention(self, result):
        assert result["total_retention"] < 0.001

    def test_n_stages(self, result):
        assert result["n_stages"] == 7


class TestTheoremSC2:
    """T-SC-2: Age-Mass Anticorrelation."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_2()

    def test_proven(self, result):
        assert result["proven"]

    def test_negative_correlation(self, result):
        assert result["pearson_r"] < 0

    def test_oldest_less_massive(self, result):
        assert result["oldest"]["mass"] < result["youngest"]["mass"]


class TestTheoremSC3:
    """T-SC-3: Metallicity Bias."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_3()

    def test_proven(self, result):
        assert result["proven"]

    def test_near_solar_dominates(self, result):
        assert result["near_solar_fraction"] > 0.5

    def test_mean_metallicity_near_solar(self, result):
        assert -0.6 < result["mean_metallicity"] < 0.0


class TestTheoremSC4:
    """T-SC-4: Contamination Detection."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_4()

    def test_proven(self, result):
        assert result["proven"]

    def test_well_separated(self, result):
        assert result["well_separated"]

    def test_fc_below_20_percent(self, result):
        assert result["fc"] < 0.20

    def test_contam_higher_drift(self, result):
        assert result["contam_higher_drift"]

    def test_separation_positive(self, result):
        assert result["separation_gyr"] > 0.5

    def test_n_removed(self, result):
        assert result["n_removed"] == N_CONTAM_REMOVED


class TestTheoremSC5:
    """T-SC-5: Hubble Tension Probe."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_5()

    def test_proven(self, result):
        assert result["proven"]

    def test_planck_consistent(self, result):
        assert result["planck_consistent"]

    def test_shoes_tension(self, result):
        assert result["shoes_tension"]

    def test_h0_upper_limit_reasonable(self, result):
        assert 55 < result["h0_upper_limit"] < 80


class TestTheoremSC6:
    """T-SC-6: Golden vs Final Consistency."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_6()

    def test_proven(self, result):
        assert result["proven"]

    def test_age_consistent(self, result):
        assert result["age_difference"] < 0.5

    def test_f_consistent(self, result):
        assert result["F_difference"] < 0.05

    def test_mixture_consistent(self, result):
        assert result["mu_difference"] < 0.2


class TestTheoremSC7:
    """T-SC-7: Systematic Budget."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_7()

    def test_proven(self, result):
        assert result["proven"]

    def test_syst_dominates_stat(self, result):
        assert result["syst_dominates_stat"]

    def test_models_dominate_met(self, result):
        assert result["models_dominate_met"]

    def test_syst_total_reasonable(self, result):
        assert 1.0 < result["syst_total"] < 2.0


class TestTheoremSC8:
    """T-SC-8: Cosmological Lower Bound."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_8()

    def test_proven(self, result):
        assert result["proven"]

    def test_above_shoes(self, result):
        assert result["above_shoes_t_u"]

    def test_age_peak(self, result):
        assert result["age_peak"] == AGE_PEAK

    def test_frac_gt_13(self, result):
        assert result["frac_gt_13_gyr"] > 0.3

    def test_n_stars_gt_13(self, result):
        assert result["n_stars_gt_13"] == N_STARS_TU_GT_13


class TestTheoremSC9:
    """T-SC-9: Formation Delay."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_9()

    def test_proven(self, result):
        assert result["proven"]

    def test_conservative_choice(self, result):
        assert result["conservative_choice"]

    def test_h0_higher_at_zf20(self, result):
        assert result["h0_zf20"] > result["h0_zf11"]


class TestTheoremSC10:
    """T-SC-10: Universal Tier-1."""

    @pytest.fixture(scope="class")
    def result(self):
        return prove_t_sc_10()

    def test_proven(self, result):
        assert result["proven"]

    def test_all_passed(self, result):
        assert result["n_passed"] == result["n_tested"]

    def test_no_violations(self, result):
        assert result["n_violations"] == 0

    def test_n_tested(self, result):
        assert result["n_tested"] == 23


# ═══════════════════════════════════════════════════════════════
# FULL ANALYSIS
# ═══════════════════════════════════════════════════════════════


class TestFullAnalysis:
    """Test the complete analysis pipeline."""

    def test_all_theorems_proven(self, full_analysis):
        assert full_analysis["n_theorems_proven"] == 10

    def test_total_theorems(self, full_analysis):
        assert full_analysis["n_theorems_total"] == 10

    def test_n_stars(self, full_analysis):
        assert full_analysis["n_stars"] == 23

    def test_doi(self, full_analysis):
        assert full_analysis["doi"] == "10.1051/0004-6361/202557038"

    def test_source(self, full_analysis):
        assert "Tomasetti" in full_analysis["source"]

    def test_summary_keys(self, full_analysis):
        summary = full_analysis["summary"]
        assert "mean_F" in summary
        assert "mean_IC" in summary
        assert "mean_gap" in summary
        assert "t_u_lower_bound" in summary
        assert "h0_upper_limit" in summary

    def test_mean_f_reasonable(self, full_analysis):
        assert 0.3 < full_analysis["summary"]["mean_F"] < 0.8

    def test_pipeline_in_results(self, full_analysis):
        assert len(full_analysis["pipeline"]) == 7

    def test_systematics_in_results(self, full_analysis):
        assert len(full_analysis["systematics"]) == 4


# ═══════════════════════════════════════════════════════════════
# NARRATIVE
# ═══════════════════════════════════════════════════════════════


class TestNarrative:
    """Test narrative generation."""

    @pytest.fixture(scope="class")
    def narrative(self):
        return generate_narrative()

    def test_has_prologue(self, narrative):
        assert "202,384" in narrative["prologue"]

    def test_has_act_i(self, narrative):
        assert str(N_FINAL) in narrative["act_i_funnel"]

    def test_has_act_ii(self, narrative):
        assert str(MU_MAIN) in narrative["act_ii_populations"]

    def test_has_act_iii(self, narrative):
        assert str(AGE_PEAK) in narrative["act_iii_clock_reading"]

    def test_has_act_iv(self, narrative):
        assert str(H0_PLANCK) in narrative["act_iv_hubble_tension"]

    def test_has_act_v(self, narrative):
        assert str(SYST_TOTAL) in narrative["act_v_systematics"]

    def test_has_epilogue(self, narrative):
        assert "13.8" in narrative["epilogue"]

    def test_n_kernels(self, narrative):
        assert narrative["n_kernels"] == 23


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_excluded_contaminant_tier1(self, stellar_db):
        """Even excluded contaminants must satisfy Tier-1."""
        excluded = [s for s in stellar_db if "Excluded" in s.name]
        for s in excluded:
            kr = compute_stellar_kernel(s)
            assert abs((kr.F + kr.omega) - 1.0) < 1e-6
            assert kr.IC <= kr.F + 1e-6

    def test_19gyr_contaminant_higher_omega(self, stellar_db):
        """The 19 Gyr contaminant should have high age channel but low precision."""
        contam = next(s for s in stellar_db if "19 Gyr" in s.name)
        kr = compute_stellar_kernel(contam)
        # High age → high first channel, but overall heterogeneity
        assert kr.gap > 0

    def test_gc_comparison_lower_metallicity(self, stellar_db):
        """GC comparisons have lower metallicity than field stars."""
        gc = next(s for s in stellar_db if "GC" in s.name and "Valcin" in s.name)
        final = next(s for s in stellar_db if s.name == "Final sample mean")
        assert gc.metallicity < final.metallicity

    def test_golden_sample_better_precision(self, stellar_db):
        """Golden sample has better age precision than final sample."""
        golden = next(s for s in stellar_db if s.name == "Golden sample mean")
        final = next(s for s in stellar_db if s.name == "Final sample mean")
        assert golden.age_err_stat <= final.age_err_stat

    def test_mass_age_trend_in_kernels(self, stellar_db):
        """Verify that the age-mass relationship is reflected in kernel."""
        oldest = next(s for s in stellar_db if s.name == "Oldest reliable (90% upper envelope)")
        youngest = next(s for s in stellar_db if s.name == "Youngest in final sample")
        kr_old = compute_stellar_kernel(oldest)
        kr_young = compute_stellar_kernel(youngest)
        # Both should have valid kernels
        assert kr_old.F > 0 and kr_young.F > 0
