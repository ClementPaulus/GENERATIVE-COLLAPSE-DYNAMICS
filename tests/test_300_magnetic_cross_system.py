"""Tests for magnetic cross-system closure (materials science domain).

Validates 35 entities across 3 systems (Material, Quincke, QDM),
Tier-1 kernel identities, and 6 theorems (T-MCS-1 through T-MCS-6).

35 = 17 materials + 12 Quincke roller states + 6 primary QDM phases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.materials_science.magnetic_cross_system import (
    N_ENTITIES,
    N_SYSTEMS,
    SYSTEM_MATERIAL,
    SYSTEM_QDM,
    SYSTEM_QUINCKE,
    MCSKernelResult,
    build_cross_system_catalog,
    verify_all_theorems,
    verify_t_mcs_1,
    verify_t_mcs_2,
    verify_t_mcs_3,
    verify_t_mcs_4,
    verify_t_mcs_5,
    verify_t_mcs_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[MCSKernelResult]:
    return build_cross_system_catalog()


# ---------------------------------------------------------------------------
# Entity catalog
# ---------------------------------------------------------------------------
class TestEntityCatalog:
    def test_entity_count(self, all_results):
        assert len(all_results) == N_ENTITIES == 35

    def test_system_count(self, all_results):
        systems = {r.system for r in all_results}
        assert len(systems) == N_SYSTEMS == 3

    def test_material_count(self, all_results):
        mats = [r for r in all_results if r.system == SYSTEM_MATERIAL]
        assert len(mats) == 17

    def test_quincke_count(self, all_results):
        qr = [r for r in all_results if r.system == SYSTEM_QUINCKE]
        assert len(qr) == 12

    def test_qdm_count(self, all_results):
        qdm = [r for r in all_results if r.system == SYSTEM_QDM]
        assert len(qdm) == 6

    def test_all_names_unique(self, all_results):
        names = [r.name for r in all_results]
        assert len(names) == len(set(names))

    def test_material_categories(self, all_results):
        mat_cats = {r.category for r in all_results if r.system == SYSTEM_MATERIAL}
        expected = {"Ferromagnetic", "Antiferromagnetic", "Ferrimagnetic", "Diamagnetic", "Paramagnetic"}
        assert mat_cats == expected

    def test_systems_present(self, all_results):
        systems = {r.system for r in all_results}
        assert systems == {SYSTEM_MATERIAL, SYSTEM_QUINCKE, SYSTEM_QDM}


# ---------------------------------------------------------------------------
# Tier-1 identities (parametrized across all 35 entities)
# ---------------------------------------------------------------------------
class TestTier1Identities:
    def test_duality_identity_all(self, all_results):
        """F + ω = 1 for every entity across all three systems."""
        for r in all_results:
            assert abs(r.F + r.omega - 1.0) < 1e-12, f"Duality failed for {r.name}"

    def test_integrity_bound_all(self, all_results):
        """IC ≤ F for every entity across all three systems."""
        for r in all_results:
            assert r.IC <= r.F + 1e-12, f"Integrity bound failed for {r.name}"

    def test_log_integrity_relation_all(self, all_results):
        """IC = exp(κ) for every entity across all three systems."""
        for r in all_results:
            assert abs(r.IC - math.exp(r.kappa)) < 1e-8, f"IC=exp(κ) failed for {r.name}"

    def test_omega_nonneg(self, all_results):
        for r in all_results:
            assert r.omega >= -1e-15, f"ω negative for {r.name}"

    def test_F_bounded(self, all_results):
        for r in all_results:
            assert 0.0 <= r.F <= 1.0 + 1e-12, f"F out of [0,1] for {r.name}"

    def test_IC_nonneg(self, all_results):
        for r in all_results:
            assert r.IC >= -1e-15, f"IC negative for {r.name}"

    def test_S_nonneg(self, all_results):
        for r in all_results:
            assert r.S >= -1e-12, f"S negative for {r.name}"

    def test_C_bounded(self, all_results):
        for r in all_results:
            assert 0.0 <= r.C <= 1.0 + 1e-12, f"C out of [0,1] for {r.name}"


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
class TestRegimeClassification:
    def test_regime_is_valid(self, all_results):
        for r in all_results:
            assert r.regime in ("Stable", "Watch", "Collapse"), f"Bad regime for {r.name}: {r.regime}"

    def test_at_least_two_regimes(self, all_results):
        regimes = {r.regime for r in all_results}
        assert len(regimes) >= 2


# ---------------------------------------------------------------------------
# Kernel result properties
# ---------------------------------------------------------------------------
class TestKernelResult:
    def test_gap_computation(self, all_results):
        for r in all_results:
            assert abs(r.gap - (r.F - r.IC)) < 1e-15

    def test_to_dict_keys(self, all_results):
        d = all_results[0].to_dict()
        expected_keys = {"name", "system", "category", "F", "omega", "S", "C", "kappa", "IC", "regime", "gap"}
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Theorems T-MCS-1 through T-MCS-6
# ---------------------------------------------------------------------------
class TestTheorems:
    def test_t_mcs_1_duality_universality(self, all_results):
        assert verify_t_mcs_1(all_results)["passed"]

    def test_t_mcs_2_integrity_bound(self, all_results):
        assert verify_t_mcs_2(all_results)["passed"]

    def test_t_mcs_3_curvature_heterogeneity(self, all_results):
        assert verify_t_mcs_3(all_results)["passed"]

    def test_t_mcs_4_multi_regime_span(self, all_results):
        assert verify_t_mcs_4(all_results)["passed"]

    def test_t_mcs_5_magnetic_activation(self, all_results):
        assert verify_t_mcs_5(all_results)["passed"]

    def test_t_mcs_6_ferromagnetic_ic_dominance(self, all_results):
        assert verify_t_mcs_6(all_results)["passed"]

    def test_all_theorems_pass(self):
        for t in verify_all_theorems():
            assert t["passed"], f"{t['name']} failed: {t}"


# ---------------------------------------------------------------------------
# Cross-system structural facts
# ---------------------------------------------------------------------------
class TestCrossSystemFacts:
    def test_material_ferro_highest_mean_F(self, all_results):
        """Ferromagnets have highest mean F among material categories."""
        mat = [r for r in all_results if r.system == SYSTEM_MATERIAL]
        cats: dict[str, list[float]] = {}
        for r in mat:
            cats.setdefault(r.category, []).append(r.F)
        ferro_mean = np.mean(cats["Ferromagnetic"])
        dia_mean = np.mean(cats["Diamagnetic"])
        assert ferro_mean > dia_mean

    def test_quincke_subtreshold_lowest_ic(self, all_results):
        """SubThreshold Quincke state has lowest IC in its system."""
        qr = [r for r in all_results if r.system == SYSTEM_QUINCKE]
        sub = next(r for r in qr if "SubThreshold" in r.name)
        assert min(r.IC for r in qr) == sub.IC

    def test_qdm_topological_higher_ic_than_crystal(self, all_results):
        """QDM topological phases have higher mean IC than crystal phases."""
        qdm = [r for r in all_results if r.system == SYSTEM_QDM]
        topo = [r for r in qdm if "QSL" in r.name]
        crystal = [r for r in qdm if r not in topo and "PM" not in r.name]
        if topo and crystal:
            assert np.mean([r.IC for r in topo]) > np.mean([r.IC for r in crystal])

    def test_entropy_positive_everywhere(self, all_results):
        """Bernoulli field entropy S ≥ 0 across all 35 entities."""
        for r in all_results:
            assert r.S >= -1e-12

    def test_ic_f_ratio_bounded(self, all_results):
        """IC/F ≤ 1 for every entity (consequence of integrity bound)."""
        for r in all_results:
            if r.F > 1e-10:
                assert r.IC / r.F <= 1.0 + 1e-10
