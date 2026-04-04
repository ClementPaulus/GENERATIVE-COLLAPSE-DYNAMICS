"""Tests for intelligence coherence closure (T-IC-1 through T-IC-6).

Validates that the structural relationship between intelligence and
coherence holds across 58 entities (40 organisms + 12 fungi + 6 mycorrhizal).

Key insight: intelligence is floor-dominated, not peak-dominated.
IC/F tracks the minimum channel, not the maximum.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_WORKSPACE = Path(__file__).resolve().parents[1]
for _p in [str(_WORKSPACE / "src"), str(_WORKSPACE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closures.evolution.evolution_kernel import ORGANISMS
from closures.evolution.fungi_kingdom import FK_ENTITIES, MS_ENTITIES
from closures.evolution.intelligence_coherence import (
    CoherenceProfile,
    coherence_landscape,
    compute_all_profiles,
    compute_evolution_profiles,
    compute_fungi_profile,
    compute_organism_profile,
    floor_recovery_curve,
    verify_all_theorems,
    verify_t_ic_1,
    verify_t_ic_2,
    verify_t_ic_3,
    verify_t_ic_4,
    verify_t_ic_5,
    verify_t_ic_6,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def all_profiles() -> list[CoherenceProfile]:
    return compute_all_profiles()


@pytest.fixture(scope="module")
def evo_profiles() -> list[CoherenceProfile]:
    return compute_evolution_profiles()


@pytest.fixture(scope="module")
def theorem_results(all_profiles: list[CoherenceProfile]) -> list[dict]:
    return [
        verify_t_ic_1(all_profiles),
        verify_t_ic_2(all_profiles),
        verify_t_ic_3(all_profiles),
        verify_t_ic_4(all_profiles),
        verify_t_ic_5(all_profiles),
        verify_t_ic_6(all_profiles),
    ]


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: ENTITY COUNT AND STRUCTURE
# ══════════════════════════════════════════════════════════════════════


class TestEntityCounts:
    """Verify correct entity counts and sources."""

    def test_total_profiles(self, all_profiles: list[CoherenceProfile]) -> None:
        assert len(all_profiles) == 58  # 40 + 12 + 6

    def test_evolution_profiles(self, evo_profiles: list[CoherenceProfile]) -> None:
        assert len(evo_profiles) == 40

    def test_source_distribution(self, all_profiles: list[CoherenceProfile]) -> None:
        sources = {p.source for p in all_profiles}
        assert "evolution" in sources
        assert "fungi" in sources
        assert "mycorrhizal" in sources

    def test_evolution_source_count(self, all_profiles: list[CoherenceProfile]) -> None:
        n = sum(1 for p in all_profiles if p.source == "evolution")
        assert n == 40

    def test_fungi_source_count(self, all_profiles: list[CoherenceProfile]) -> None:
        n = sum(1 for p in all_profiles if p.source == "fungi")
        assert n == 12

    def test_mycorrhizal_source_count(self, all_profiles: list[CoherenceProfile]) -> None:
        n = sum(1 for p in all_profiles if p.source == "mycorrhizal")
        assert n == 6


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: TIER-1 IDENTITY CHECKS
# ══════════════════════════════════════════════════════════════════════


class TestTier1Identities:
    """Verify Tier-1 identities hold for all profiles."""

    def test_duality_identity(self, all_profiles: list[CoherenceProfile]) -> None:
        for p in all_profiles:
            assert abs(p.F + p.omega - 1.0) < 1e-12, f"{p.name}: F+ω={p.F + p.omega}"

    def test_integrity_bound(self, all_profiles: list[CoherenceProfile]) -> None:
        for p in all_profiles:
            assert p.IC <= p.F + 1e-12, f"{p.name}: IC={p.IC} > F={p.F}"

    def test_log_integrity_relation(self, all_profiles: list[CoherenceProfile]) -> None:
        for p in all_profiles:
            assert abs(p.IC - math.exp(p.kappa)) < 1e-9, f"{p.name}: IC≠exp(κ)"

    def test_ic_f_ratio_range(self, all_profiles: list[CoherenceProfile]) -> None:
        for p in all_profiles:
            assert 0.0 <= p.IC_F <= 1.0 + 1e-12, f"{p.name}: IC/F={p.IC_F}"

    def test_spread_non_negative(self, all_profiles: list[CoherenceProfile]) -> None:
        for p in all_profiles:
            assert p.spread >= 0.0, f"{p.name}: negative spread"

    def test_floor_leq_peak(self, all_profiles: list[CoherenceProfile]) -> None:
        for p in all_profiles:
            assert p.floor <= p.peak + 1e-12, f"{p.name}: floor > peak"


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: COHERENCE PROFILE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════


class TestProfileConstruction:
    """Verify profiles are constructed correctly."""

    def test_organism_profile_fields(self) -> None:
        org = ORGANISMS[0]
        p = compute_organism_profile(org)
        assert p.name == org.name
        assert p.source == "evolution"
        assert p.F > 0
        assert p.IC > 0

    def test_fungi_profile_fields(self) -> None:
        ent = FK_ENTITIES[0]
        p = compute_fungi_profile(ent)
        assert p.name == ent.name
        assert p.source == "fungi"
        assert p.F > 0
        assert p.IC > 0

    def test_mycorrhizal_profile_source(self) -> None:
        ent = MS_ENTITIES[0]
        p = compute_fungi_profile(ent)
        assert p.source == "mycorrhizal"

    def test_profile_to_dict(self) -> None:
        org = ORGANISMS[0]
        p = compute_organism_profile(org)
        d = p.to_dict()
        assert "name" in d
        assert "F" in d
        assert "IC_F" in d
        assert "floor" in d
        assert "peak" in d
        assert "spread" in d

    def test_regime_values(self, all_profiles: list[CoherenceProfile]) -> None:
        valid_regimes = {"Stable", "Watch", "Collapse"}
        for p in all_profiles:
            assert p.regime in valid_regimes, f"{p.name}: regime={p.regime}"


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: THEOREM VERIFICATION
# ══════════════════════════════════════════════════════════════════════


class TestTheorems:
    """Verify all 6 T-IC theorems."""

    def test_t_ic_1_floor_dominance(self, theorem_results: list[dict]) -> None:
        t = theorem_results[0]
        assert t["name"] == "T-IC-1"
        assert t["passed"], f"T-IC-1 failed: corr={t['correlation']:.3f}"
        assert t["correlation"] > 0.70

    def test_t_ic_1_n_entities(self, theorem_results: list[dict]) -> None:
        t = theorem_results[0]
        assert t["n_entities"] == 58

    def test_t_ic_2_peak_irrelevance(self, theorem_results: list[dict]) -> None:
        t = theorem_results[1]
        assert t["name"] == "T-IC-2"
        assert t["passed"], f"T-IC-2 failed: |corr|={t['abs_correlation']:.3f}"
        assert t["abs_correlation"] < 0.20

    def test_t_ic_3_asymmetric_damage(self, theorem_results: list[dict]) -> None:
        t = theorem_results[2]
        assert t["name"] == "T-IC-3"
        assert t["passed"], f"T-IC-3 failed: ratio={t['mean_damage_ratio']:.2f}, n={t['n_tested']}"
        assert t["mean_damage_ratio"] > 1.0
        assert t["n_tested"] >= 10

    def test_t_ic_4_convergence_attractor(self, theorem_results: list[dict]) -> None:
        t = theorem_results[3]
        assert t["name"] == "T-IC-4"
        assert t["passed"], f"T-IC-4 failed: σ={t.get('std_IC_F')}, min={t.get('min_IC_F')}"
        assert t["std_IC_F"] < 0.04
        assert t["min_IC_F"] > 0.85

    def test_t_ic_4_multiple_clades(self, theorem_results: list[dict]) -> None:
        t = theorem_results[3]
        assert t["n_clades"] >= 5, f"Only {t['n_clades']} clades"

    def test_t_ic_5_peak_coherence_tradeoff(self, theorem_results: list[dict]) -> None:
        t = theorem_results[4]
        assert t["name"] == "T-IC-5"
        assert t["passed"], f"T-IC-5 failed: corr={t['correlation']:.3f}"
        assert t["correlation"] < -0.30

    def test_t_ic_5_sufficient_sample(self, theorem_results: list[dict]) -> None:
        t = theorem_results[4]
        assert t["n_in_band"] >= 5

    def test_t_ic_6_spread_coherence_inversion(self, theorem_results: list[dict]) -> None:
        t = theorem_results[5]
        assert t["name"] == "T-IC-6"
        assert t["passed"], f"T-IC-6 failed: corr={t['correlation']:.3f}"
        assert t["correlation"] < -0.40

    def test_all_theorems_pass(self, theorem_results: list[dict]) -> None:
        for t in theorem_results:
            assert t["passed"], f"{t['name']} ({t.get('title', '')}) FAILED"

    def test_all_theorems_count(self, theorem_results: list[dict]) -> None:
        assert len(theorem_results) == 6


class TestVerifyAllTheorems:
    """Test the verify_all_theorems convenience function."""

    def test_returns_six(self) -> None:
        results = verify_all_theorems()
        assert len(results) == 6

    def test_all_pass(self) -> None:
        results = verify_all_theorems()
        for r in results:
            assert r["passed"], f"{r['name']} failed"

    def test_names_sequential(self) -> None:
        results = verify_all_theorems()
        for i, r in enumerate(results, 1):
            assert r["name"] == f"T-IC-{i}"


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════════════


class TestFloorRecovery:
    """Test the floor recovery curve utility."""

    def test_human_recovery_curve(self) -> None:
        human = next(o for o in ORGANISMS if o.name == "Homo sapiens")
        curve = floor_recovery_curve(human)
        assert len(curve) == 20
        # IC/F should increase with floor value
        assert curve[-1]["IC_F"] > curve[0]["IC_F"]

    def test_curve_has_required_fields(self) -> None:
        org = ORGANISMS[0]
        curve = floor_recovery_curve(org)
        for pt in curve:
            assert "floor_value" in pt
            assert "F" in pt
            assert "IC" in pt
            assert "IC_F" in pt

    def test_custom_floor_values(self) -> None:
        org = ORGANISMS[0]
        fvals = np.array([0.01, 0.05, 0.10, 0.20, 0.40])
        curve = floor_recovery_curve(org, floor_values=fvals)
        assert len(curve) == 5

    def test_recovery_monotonic(self) -> None:
        human = next(o for o in ORGANISMS if o.name == "Homo sapiens")
        curve = floor_recovery_curve(human, floor_values=np.linspace(0.01, 0.50, 10))
        ic_f_values = [pt["IC_F"] for pt in curve]
        for i in range(1, len(ic_f_values)):
            assert ic_f_values[i] >= ic_f_values[i - 1] - 1e-9


class TestCoherenceLandscape:
    """Test the coherence landscape summary."""

    def test_landscape_fields(self) -> None:
        ls = coherence_landscape()
        assert "n_total" in ls
        assert "source_stats" in ls
        assert "global_mean_IC_F" in ls
        assert "floor_IC_F_corr" in ls
        assert "peak_IC_F_corr" in ls
        assert "spread_IC_F_corr" in ls

    def test_landscape_total(self) -> None:
        ls = coherence_landscape()
        assert ls["n_total"] == 58

    def test_landscape_correlations_sign(self) -> None:
        ls = coherence_landscape()
        assert ls["floor_IC_F_corr"] > 0  # positive
        assert ls["spread_IC_F_corr"] < 0  # negative

    def test_source_stats_keys(self) -> None:
        ls = coherence_landscape()
        for src in ["evolution", "fungi", "mycorrhizal"]:
            assert src in ls["source_stats"]
            stats = ls["source_stats"][src]
            assert "n" in stats
            assert "mean_IC_F" in stats


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: PARAMETRIZED ORGANISM TESTS
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("org", ORGANISMS, ids=[o.name for o in ORGANISMS])
class TestOrganismProfiles:
    """Test each organism individually."""

    def test_duality(self, org: object) -> None:
        p = compute_organism_profile(org)  # type: ignore[arg-type]
        assert abs(p.F + p.omega - 1.0) < 1e-12

    def test_integrity_bound(self, org: object) -> None:
        p = compute_organism_profile(org)  # type: ignore[arg-type]
        assert p.IC <= p.F + 1e-12

    def test_ic_exp_kappa(self, org: object) -> None:
        p = compute_organism_profile(org)  # type: ignore[arg-type]
        assert abs(p.IC - math.exp(p.kappa)) < 1e-9


@pytest.mark.parametrize("ent", FK_ENTITIES, ids=[e.name for e in FK_ENTITIES])
class TestFungiProfiles:
    """Test each fungus individually."""

    def test_duality(self, ent: object) -> None:
        p = compute_fungi_profile(ent)  # type: ignore[arg-type]
        assert abs(p.F + p.omega - 1.0) < 1e-12

    def test_integrity_bound(self, ent: object) -> None:
        p = compute_fungi_profile(ent)  # type: ignore[arg-type]
        assert p.IC <= p.F + 1e-12


@pytest.mark.parametrize("ent", MS_ENTITIES, ids=[e.name for e in MS_ENTITIES])
class TestMycorrhizalProfiles:
    """Test each mycorrhizal stress entity."""

    def test_duality(self, ent: object) -> None:
        p = compute_fungi_profile(ent)  # type: ignore[arg-type]
        assert abs(p.F + p.omega - 1.0) < 1e-12

    def test_integrity_bound(self, ent: object) -> None:
        p = compute_fungi_profile(ent)  # type: ignore[arg-type]
        assert p.IC <= p.F + 1e-12
