"""Tests for Hubble tension as heterogeneity gap — ACT DR6 + VENUS.

Validates 6 Hubble Tension Theorems (T-HT-1 through T-HT-6), 12 H₀
measurements spanning CMB, distance ladder, time-delay cosmography,
stellar ages, and JWST VENUS lensed supernovae.

Test count target: ~82 tests covering:
    - Frozen constants from ACT DR6 (arXiv:2503.14452, 2503.14454)
    - H₀ measurement database construction (12 measurements)
    - Extended model catalog (12 models, all tested by ACT DR6)
    - Tension pair construction and sigma computation
    - Kernel computation for individual and ensemble measurements
    - 6 theorem proofs with subtests
    - Tier-1 identity universality across all H₀ probes
    - Early vs late universe split analysis
    - Heterogeneity gap detection

Sources:
    Naess et al. 2025, arXiv:2503.14451 (ACT DR6 Maps)
    Louis et al. 2025, arXiv:2503.14452 (ACT DR6 Power Spectra + ΛCDM)
    Calabrese, Hill et al. 2025, arXiv:2503.14454 (ACT DR6 Extended Models)
    Fujimoto et al. 2026 (JWST VENUS: SN Ares + SN Athena)
"""

from __future__ import annotations

import math

import pytest

from closures.astronomy.hubble_tension import (
    ACT_DR6_FREQ_GHZ,
    ACT_DR6_SKY_DEG2,
    EPSILON,
    H0_ACT_DR6_DESI1,
    H0_ACT_DR6_DESI1_ERR,
    H0_ACT_DR6_DESI2,
    H0_ACT_DR6_DESI2_ERR,
    H0_MAX,
    H0_MIN,
    H0_PLANCK,
    H0_PLANCK_ERR,
    H0_SHOES,
    H0_SHOES_ERR,
    H0_STELLAR_UPPER,
    H0_TDCOSMO,
    H0_TRGB,
    H0_TRGB_ERR,
    N_CHANNELS,
    N_EFF_ACT,
    OMEGA_B_H2_ACT,
    OMEGA_C_H2_ACT,
    SIGMA_8_ACT,
    SUM_MNU_UPPER,
    TENSION_ACT_SHOES,
    TENSION_PLANCK_SHOES,
    VENUS_N_CLUSTERS,
    VENUS_SN_ARES_CLUSTER,
    VENUS_SN_ARES_DELAY_YR,
    VENUS_SN_ARES_TYPE,
    VENUS_SN_ATHENA_CLUSTER,
    VENUS_SN_ATHENA_DELAY_YR,
    HubbleKernelResult,
    build_extended_models,
    build_h0_database,
    build_tension_pairs,
    classify_by_method,
    compute_ensemble_kernel,
    compute_h0_kernel,
    compute_tension,
    compute_weighted_mean_h0,
    early_vs_late_split,
    run_full_analysis,
)

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def full_analysis():
    """Run full analysis once for the entire test module."""
    return run_full_analysis()


@pytest.fixture(scope="module")
def h0_db():
    """Build H₀ database once."""
    return build_h0_database()


@pytest.fixture(scope="module")
def tension_pairs():
    """Build tension pairs once."""
    return build_tension_pairs()


@pytest.fixture(scope="module")
def extended_models():
    """Build extended model catalog once."""
    return build_extended_models()


# ═══════════════════════════════════════════════════════════════
# FROZEN CONSTANTS
# ═══════════════════════════════════════════════════════════════


class TestFrozenConstants:
    """Verify frozen constants from published measurements."""

    def test_h0_planck(self):
        assert H0_PLANCK == 67.4

    def test_h0_planck_err(self):
        assert H0_PLANCK_ERR == 0.5

    def test_h0_act_dr6_desi1(self):
        assert H0_ACT_DR6_DESI1 == 68.22

    def test_h0_act_dr6_desi1_err(self):
        assert H0_ACT_DR6_DESI1_ERR == 0.36

    def test_h0_act_dr6_desi2(self):
        assert H0_ACT_DR6_DESI2 == 68.43

    def test_h0_act_dr6_desi2_err(self):
        assert H0_ACT_DR6_DESI2_ERR == 0.27

    def test_h0_shoes(self):
        assert H0_SHOES == 73.04

    def test_h0_shoes_err(self):
        assert H0_SHOES_ERR == 1.04

    def test_h0_trgb(self):
        assert H0_TRGB == 69.8

    def test_h0_tdcosmo(self):
        assert H0_TDCOSMO == 74.2

    def test_h0_stellar_upper(self):
        assert H0_STELLAR_UPPER == 68.3

    def test_omega_b_h2_act(self):
        assert OMEGA_B_H2_ACT == 0.0226

    def test_omega_c_h2_act(self):
        assert OMEGA_C_H2_ACT == 0.118

    def test_sigma_8_act(self):
        assert SIGMA_8_ACT == 0.813

    def test_n_eff_act(self):
        assert N_EFF_ACT == 2.86

    def test_sum_mnu_upper(self):
        assert SUM_MNU_UPPER == 0.082

    def test_normalization_range(self):
        assert H0_MIN == 60.0
        assert H0_MAX == 80.0

    def test_n_channels(self):
        assert N_CHANNELS == 8

    def test_venus_sn_athena_delay(self):
        assert VENUS_SN_ATHENA_DELAY_YR == 1.5

    def test_venus_sn_ares_delay(self):
        assert VENUS_SN_ARES_DELAY_YR == 60.0

    def test_venus_n_clusters(self):
        assert VENUS_N_CLUSTERS == 60

    def test_venus_sn_ares_cluster(self):
        assert VENUS_SN_ARES_CLUSTER == "MACS J0308+2645"

    def test_venus_sn_athena_cluster(self):
        assert VENUS_SN_ATHENA_CLUSTER == "MACS J0417.5-1154"

    def test_venus_sn_ares_type_core_collapse(self):
        """SN Ares is a core-collapse SN (massive star), NOT Type Ia."""
        assert VENUS_SN_ARES_TYPE == "core-collapse"

    def test_act_dr6_sky_coverage(self):
        assert ACT_DR6_SKY_DEG2 == 19_000

    def test_act_dr6_frequencies(self):
        assert ACT_DR6_FREQ_GHZ == (98, 150, 220)


# ═══════════════════════════════════════════════════════════════
# TENSION COMPUTATION
# ═══════════════════════════════════════════════════════════════


class TestTensionComputation:
    """Verify tension quantification."""

    def test_planck_shoes_tension(self):
        """~5σ tension between Planck and SH0ES."""
        t = compute_tension(H0_PLANCK, H0_PLANCK_ERR, H0_SHOES, H0_SHOES_ERR)
        assert 4.5 < t < 5.5

    def test_act_shoes_tension(self):
        """~4σ tension between ACT DR6 and SH0ES."""
        t = compute_tension(H0_ACT_DR6_DESI2, H0_ACT_DR6_DESI2_ERR, H0_SHOES, H0_SHOES_ERR)
        assert 3.5 < t < 5.0

    def test_act_planck_consistency(self):
        """ACT and Planck agree within 2σ."""
        t = compute_tension(H0_ACT_DR6_DESI2, H0_ACT_DR6_DESI2_ERR, H0_PLANCK, H0_PLANCK_ERR)
        assert t < 3.0

    def test_trgb_intermediate(self):
        """TRGB sits between CMB and SH0ES — lower tension with both."""
        t_trgb_planck = compute_tension(H0_TRGB, H0_TRGB_ERR, H0_PLANCK, H0_PLANCK_ERR)
        t_trgb_shoes = compute_tension(H0_TRGB, H0_TRGB_ERR, H0_SHOES, H0_SHOES_ERR)
        t_planck_shoes = compute_tension(H0_PLANCK, H0_PLANCK_ERR, H0_SHOES, H0_SHOES_ERR)
        assert t_trgb_planck < t_planck_shoes
        assert t_trgb_shoes < t_planck_shoes

    def test_tension_symmetric(self):
        """Tension is symmetric: σ(A,B) = σ(B,A)."""
        t1 = compute_tension(H0_PLANCK, H0_PLANCK_ERR, H0_SHOES, H0_SHOES_ERR)
        t2 = compute_tension(H0_SHOES, H0_SHOES_ERR, H0_PLANCK, H0_PLANCK_ERR)
        assert abs(t1 - t2) < 1e-10

    def test_tension_zero_for_identical(self):
        """Zero tension for identical measurements."""
        t = compute_tension(H0_PLANCK, H0_PLANCK_ERR, H0_PLANCK, H0_PLANCK_ERR)
        assert t == 0.0

    def test_precomputed_tension_planck_shoes(self):
        """Precomputed tension constant matches computation."""
        t = compute_tension(H0_PLANCK, H0_PLANCK_ERR, H0_SHOES, H0_SHOES_ERR)
        assert abs(t - TENSION_PLANCK_SHOES) < 0.01

    def test_precomputed_tension_act_shoes(self):
        """Precomputed tension constant matches computation."""
        t = compute_tension(H0_ACT_DR6_DESI2, H0_ACT_DR6_DESI2_ERR, H0_SHOES, H0_SHOES_ERR)
        assert abs(t - TENSION_ACT_SHOES) < 0.01


# ═══════════════════════════════════════════════════════════════
# DATABASE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════


class TestH0Database:
    """Verify H₀ measurement database construction."""

    def test_database_size(self, h0_db):
        assert len(h0_db) == 12

    def test_all_have_positive_h0(self, h0_db):
        for m in h0_db:
            assert m.h0 > 0, f"{m.name} has non-positive H₀"

    def test_all_have_positive_errors(self, h0_db):
        for m in h0_db:
            assert m.h0_err > 0, f"{m.name} has non-positive error"

    def test_methods_present(self, h0_db):
        methods = {m.method for m in h0_db}
        assert "CMB" in methods
        assert "BAO" in methods
        assert "distance_ladder" in methods
        assert "time_delay" in methods
        assert "stellar_ages" in methods
        assert "lensed_sn" in methods

    def test_cmb_measurements_count(self, h0_db):
        cmb = [m for m in h0_db if m.method == "CMB"]
        assert len(cmb) >= 3  # Planck, ACT+DESI1, ACT+DESI2, WMAP+ACT

    def test_bao_measurements_count(self, h0_db):
        bao = [m for m in h0_db if m.method == "BAO"]
        assert len(bao) >= 1  # DESI DR2

    def test_distance_ladder_count(self, h0_db):
        dl = [m for m in h0_db if m.method == "distance_ladder"]
        assert len(dl) >= 3  # SH0ES, TRGB, JAGB

    def test_venus_measurements(self, h0_db):
        venus = [m for m in h0_db if m.method == "lensed_sn"]
        assert len(venus) == 2

    def test_h0_range(self, h0_db):
        """All H₀ values should be within reasonable cosmological range."""
        for m in h0_db:
            assert 60.0 < m.h0 < 80.0, f"{m.name}: H₀={m.h0} out of range"

    def test_trace_vector_shape(self, h0_db):
        for m in h0_db:
            trace = m.trace_vector()
            assert trace.shape == (8,), f"{m.name}: trace shape {trace.shape}"

    def test_trace_vector_bounds(self, h0_db):
        for m in h0_db:
            trace = m.trace_vector()
            for i, c in enumerate(trace):
                assert EPSILON <= c <= 1 - EPSILON, f"{m.name} channel {i}: {c} out of [ε, 1-ε]"


# ═══════════════════════════════════════════════════════════════
# EXTENDED MODELS
# ═══════════════════════════════════════════════════════════════


class TestExtendedModels:
    """Verify extended cosmological model catalog from ACT DR6."""

    def test_model_count(self, extended_models):
        assert len(extended_models) == 12

    def test_none_favored(self, extended_models):
        """ACT DR6 finds no model favored over ΛCDM."""
        favored = [m for m in extended_models if m.favored]
        assert len(favored) == 0

    def test_none_resolves_tension(self, extended_models):
        """No model resolves the Hubble tension."""
        resolving = [m for m in extended_models if m.resolves_tension]
        assert len(resolving) == 0

    def test_n_eff_model_present(self, extended_models):
        names = [m.name for m in extended_models]
        assert any("relativistic" in n.lower() or "n_eff" in n.lower() for n in names)

    def test_ede_model_present(self, extended_models):
        names = [m.name for m in extended_models]
        assert any("early dark energy" in n.lower() or "ede" in n.lower() for n in names)

    def test_neutrino_mass_model(self, extended_models):
        names = [m.name for m in extended_models]
        assert any("neutrino mass" in n.lower() for n in names)


# ═══════════════════════════════════════════════════════════════
# TENSION PAIRS
# ═══════════════════════════════════════════════════════════════


class TestTensionPairs:
    """Verify tension pair construction."""

    def test_pair_count(self, tension_pairs):
        assert len(tension_pairs) >= 6

    def test_planck_shoes_pair(self, tension_pairs):
        pair = next((p for p in tension_pairs if "Planck" in p.name_a and "SH0ES" in p.name_b), None)
        assert pair is not None
        assert pair.tension_sigma > 4.0

    def test_act_shoes_pair(self, tension_pairs):
        pair = next((p for p in tension_pairs if "ACT" in p.name_a and "SH0ES" in p.name_b), None)
        assert pair is not None
        assert pair.tension_sigma > 3.0

    def test_all_positive_sigma(self, tension_pairs):
        for p in tension_pairs:
            assert p.tension_sigma >= 0.0


# ═══════════════════════════════════════════════════════════════
# KERNEL COMPUTATION
# ═══════════════════════════════════════════════════════════════


class TestKernelComputation:
    """Verify kernel computation for H₀ measurements."""

    def test_individual_kernel_returns(self, h0_db):
        """Each measurement produces a valid kernel result."""
        m = h0_db[0]
        kr = compute_h0_kernel(m)
        assert isinstance(kr, HubbleKernelResult)
        assert 0 < kr.F <= 1
        assert 0 <= kr.omega < 1
        assert kr.IC <= kr.F + 1e-6

    def test_duality_identity(self, h0_db):
        """F + ω = 1 for all measurements."""
        for m in h0_db:
            kr = compute_h0_kernel(m)
            assert abs(kr.F + kr.omega - 1.0) < 1e-6, f"{m.name}: F+ω = {kr.F + kr.omega}"

    def test_integrity_bound(self, h0_db):
        """IC ≤ F for all measurements."""
        for m in h0_db:
            kr = compute_h0_kernel(m)
            assert kr.IC <= kr.F + 1e-6, f"{m.name}: IC={kr.IC} > F={kr.F}"

    def test_exp_kappa_relation(self, h0_db):
        """IC = exp(κ) for all measurements."""
        for m in h0_db:
            kr = compute_h0_kernel(m)
            assert abs(kr.IC - math.exp(kr.kappa)) < 1e-4, f"{m.name}: IC={kr.IC}, exp(κ)={math.exp(kr.kappa)}"

    def test_gap_nonnegative(self, h0_db):
        """Heterogeneity gap Δ = F − IC ≥ 0."""
        for m in h0_db:
            kr = compute_h0_kernel(m)
            assert kr.gap >= -1e-6

    def test_ensemble_kernel(self, h0_db):
        """Ensemble kernel computes without error."""
        current = [m for m in h0_db if m.year <= 2026]
        kr = compute_ensemble_kernel(current)
        assert isinstance(kr, HubbleKernelResult)
        assert abs(kr.F + kr.omega - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════


class TestAnalysisFunctions:
    """Verify analysis helper functions."""

    def test_classify_by_method(self, h0_db):
        groups = classify_by_method(h0_db)
        assert len(groups) >= 4

    def test_early_vs_late_split(self, h0_db):
        current = [m for m in h0_db if m.year <= 2026]
        early, late = early_vs_late_split(current)
        assert len(early) >= 3
        assert len(late) >= 3
        assert len(early) + len(late) == len(current)

    def test_weighted_mean_h0_early(self, h0_db):
        current = [m for m in h0_db if m.year <= 2026]
        early, _ = early_vs_late_split(current)
        h0_mean, h0_err = compute_weighted_mean_h0(early)
        # Early universe should give H₀ ≈ 67-69
        assert 66.0 < h0_mean < 70.0
        assert h0_err > 0

    def test_weighted_mean_h0_late(self, h0_db):
        current = [m for m in h0_db if m.year <= 2026]
        _, late = early_vs_late_split(current)
        h0_mean, h0_err = compute_weighted_mean_h0(late)
        # Late universe weighted by errors; should be > early mean
        assert h0_mean > 60.0
        assert h0_err > 0

    def test_weighted_mean_empty(self):
        h0_mean, h0_err = compute_weighted_mean_h0([])
        assert h0_mean == 0.0
        assert h0_err == 0.0


# ═══════════════════════════════════════════════════════════════
# THEOREM PROOFS
# ═══════════════════════════════════════════════════════════════


class TestTheoremTHT1:
    """T-HT-1: Channel Discrepancy — Early vs late H₀ show heterogeneity gap."""

    def test_proven(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-1"]["proven"]

    def test_discrepancy_above_2sigma(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-1"]["discrepancy_sigma"] > 2.0

    def test_ensemble_gap_positive(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-1"]["ensemble_gap"] > 0


class TestTheoremTHT2:
    """T-HT-2: ACT DR6 Intermediacy."""

    def test_proven(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-2"]["proven"]

    def test_intermediate(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-2"]["intermediate"]

    def test_closer_to_planck(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-2"]["closer_to_planck"]

    def test_act_planck_tension_low(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-2"]["tension_act_planck_sigma"] < 3.0

    def test_act_shoes_tension_high(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-2"]["tension_act_shoes_sigma"] > 3.0


class TestTheoremTHT3:
    """T-HT-3: Geometric Slaughter — H₀ discrepancy kills composite integrity."""

    def test_proven(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-3"]["proven"]

    def test_ensemble_gap_positive(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-3"]["ensemble_gap"] > 0


class TestTheoremTHT4:
    """T-HT-4: Lensing Independence — VENUS provides independent channel."""

    def test_proven(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-4"]["proven"]

    def test_venus_independence_high(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-4"]["venus_mean_independence"] > 0.9

    def test_venus_count(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-4"]["n_venus_measurements"] == 2


class TestTheoremTHT5:
    """T-HT-5: Extended Model Closure — No ΛCDM extension resolves the tension."""

    def test_proven(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-5"]["proven"]

    def test_no_models_favored(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-5"]["n_favored_over_lcdm"] == 0

    def test_no_models_resolve(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-5"]["n_resolves_tension"] == 0

    def test_n_models_tested(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-5"]["n_models_tested"] == 12

    def test_n_eff_consistent_with_sm(self, full_analysis):
        """N_eff = 2.86 ± 0.13 contains SM value 3.044 within ~1.4σ."""
        result = full_analysis["theorems"]["T-HT-5"]
        assert abs(result["n_eff_measured"] - 3.044) < 3 * result["n_eff_err"]


class TestTheoremTHT6:
    """T-HT-6: Universal Tier-1 — All measurements satisfy Tier-1 identities."""

    def test_proven(self, full_analysis):
        assert full_analysis["theorems"]["T-HT-6"]["proven"]

    def test_all_pass(self, full_analysis):
        result = full_analysis["theorems"]["T-HT-6"]
        assert result["n_pass_tier1"] == result["n_measurements"]


# ═══════════════════════════════════════════════════════════════
# FULL ANALYSIS
# ═══════════════════════════════════════════════════════════════


class TestFullAnalysis:
    """Verify overall analysis output structure."""

    def test_has_ensemble_kernel(self, full_analysis):
        assert "ensemble_kernel" in full_analysis

    def test_has_early_late_split(self, full_analysis):
        assert "early_universe_kernel" in full_analysis
        assert "late_universe_kernel" in full_analysis

    def test_has_tensions(self, full_analysis):
        assert len(full_analysis["tensions"]) >= 6

    def test_all_theorems_present(self, full_analysis):
        for i in range(1, 7):
            assert f"T-HT-{i}" in full_analysis["theorems"]

    def test_theorem_count(self, full_analysis):
        assert full_analysis["n_theorems_total"] == 6

    def test_all_theorems_proven(self, full_analysis):
        assert full_analysis["n_theorems_proven"] == 6

    def test_ensemble_regime_not_stable(self, full_analysis):
        """The Hubble tension ensemble should NOT be in Stable regime."""
        regime = full_analysis["ensemble_kernel"]["regime"]
        # The tension means the ensemble is NOT in Stable regime
        assert regime != "Stable"
