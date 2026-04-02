"""Tests for unified materials science theorems (T-MS-1 through T-MS-10).

Validates 10 theorems bridging crystal_morphology (17 crystals),
bioactive_compounds (12 compounds), photonic_materials (14 devices),
and particle_detector (8 scintillators) — 51 total entities, 193 subtests.
"""

from __future__ import annotations

import pytest

from closures.materials_science.materials_theorems import (
    ALL_THEOREMS,
    TheoremResult,
    run_all_theorems,
    theorem_MS1_kernel_identities,
    theorem_MS2_universal_collapse,
    theorem_MS3_fidelity_stratification,
    theorem_MS4_crystal_heterogeneity,
    theorem_MS5_gap_ordering,
    theorem_MS6_scintillator_spread,
    theorem_MS7_photonic_diversity,
    theorem_MS8_duality_exactness,
    theorem_MS9_integrity_bound,
    theorem_MS10_total_coverage,
)


@pytest.fixture(scope="module")
def all_theorem_results() -> list[TheoremResult]:
    return run_all_theorems()


class TestTheoremRunner:
    def test_all_theorems_count(self):
        assert len(ALL_THEOREMS) == 10

    def test_run_all_returns_10(self, all_theorem_results):
        assert len(all_theorem_results) == 10

    def test_all_verdicts_proven(self, all_theorem_results):
        for r in all_theorem_results:
            assert r.verdict == "PROVEN", f"{r.name} verdict={r.verdict}"

    def test_no_failures(self, all_theorem_results):
        for r in all_theorem_results:
            assert r.n_failed == 0, f"{r.name} had {r.n_failed} failures"


class TestIndividualTheorems:
    def test_t_ms_1_kernel_identities(self):
        r = theorem_MS1_kernel_identities()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_2_universal_collapse(self):
        r = theorem_MS2_universal_collapse()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_3_fidelity_stratification(self):
        r = theorem_MS3_fidelity_stratification()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_4_crystal_heterogeneity(self):
        r = theorem_MS4_crystal_heterogeneity()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_5_gap_ordering(self):
        r = theorem_MS5_gap_ordering()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_6_scintillator_spread(self):
        r = theorem_MS6_scintillator_spread()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_7_photonic_diversity(self):
        r = theorem_MS7_photonic_diversity()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_8_duality_exactness(self):
        r = theorem_MS8_duality_exactness()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_9_integrity_bound(self):
        r = theorem_MS9_integrity_bound()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_ms_10_total_coverage(self):
        r = theorem_MS10_total_coverage()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests


class TestTheoremResultStructure:
    @pytest.mark.parametrize(
        "thm_fn",
        ALL_THEOREMS,
        ids=[f"T-MS-{i + 1}" for i in range(len(ALL_THEOREMS))],
    )
    def test_result_fields(self, thm_fn):
        r = thm_fn()
        assert isinstance(r, TheoremResult)
        assert r.name
        assert r.statement
        assert r.n_tests > 0
        assert r.n_passed >= 0
        assert r.n_failed >= 0
        assert r.n_passed + r.n_failed == r.n_tests
        assert r.verdict in ("PROVEN", "FALSIFIED")
        assert isinstance(r.details, dict)
