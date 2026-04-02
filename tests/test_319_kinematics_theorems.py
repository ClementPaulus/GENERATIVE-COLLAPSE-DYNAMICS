"""Tests for unified kinematics theorems (T-KN-1 through T-KN-10).

Validates 10 theorems covering elastic conservation, energy additivity,
work-energy, oscillation classification, stability ordering, motion regimes,
phase space geometry, and cross-module consistency — 87 subtests.
"""

from __future__ import annotations

import pytest

from closures.kinematics.kinematics_theorems import (
    ALL_THEOREMS,
    TheoremResult,
    run_all_theorems,
    theorem_KN1_elastic_conservation,
    theorem_KN2_energy_additivity,
    theorem_KN3_work_energy_theorem,
    theorem_KN4_energy_conservation_detection,
    theorem_KN5_oscillation_classification,
    theorem_KN6_stability_ordering,
    theorem_KN7_stability_trend,
    theorem_KN8_motion_regime_classification,
    theorem_KN9_phase_space_geometry,
    theorem_KN10_cross_module_consistency,
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
    def test_t_kn_1_elastic_conservation(self):
        r = theorem_KN1_elastic_conservation()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_2_energy_additivity(self):
        r = theorem_KN2_energy_additivity()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_3_work_energy_theorem(self):
        r = theorem_KN3_work_energy_theorem()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_4_energy_conservation_detection(self):
        r = theorem_KN4_energy_conservation_detection()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_5_oscillation_classification(self):
        r = theorem_KN5_oscillation_classification()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_6_stability_ordering(self):
        r = theorem_KN6_stability_ordering()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_7_stability_trend(self):
        r = theorem_KN7_stability_trend()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_8_motion_regime_classification(self):
        r = theorem_KN8_motion_regime_classification()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_9_phase_space_geometry(self):
        r = theorem_KN9_phase_space_geometry()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests

    def test_t_kn_10_cross_module_consistency(self):
        r = theorem_KN10_cross_module_consistency()
        assert r.verdict == "PROVEN"
        assert r.n_passed == r.n_tests


class TestTheoremResultStructure:
    @pytest.mark.parametrize(
        "thm_fn",
        ALL_THEOREMS,
        ids=[f"T-KN-{i + 1}" for i in range(len(ALL_THEOREMS))],
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
