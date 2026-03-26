"""Tests for population fragmentation closure (evolution domain).

Validates 12 entities, 8-channel Population-scale trace construction,
Tier-1 kernel identities, 6 theorems (T-EV-11 through T-EV-16),
and cross-domain bridge function.

Based on Devadhasan & Carja (2025), PNAS DOI:10.1073/pnas.2513857122.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.evolution.population_fragmentation import (
    MIGRATION_IDX,
    N_PF_CHANNELS,
    PF_CHANNELS,
    PF_ENTITIES,
    PFKernelResult,
    compute_all_entities,
    compute_pf_kernel,
    cross_domain_bridge,
    verify_all_theorems,
    verify_t_ev_11,
    verify_t_ev_12,
    verify_t_ev_13,
    verify_t_ev_14,
    verify_t_ev_15,
    verify_t_ev_16,
)


@pytest.fixture(scope="module")
def all_results() -> list[PFKernelResult]:
    return compute_all_entities()


# ═════════════════════════════════════════════════════════════════════
# Section 1: Entity Catalog
# ═════════════════════════════════════════════════════════════════════


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(PF_ENTITIES) == 12

    def test_channel_count(self):
        assert N_PF_CHANNELS == 8
        assert len(PF_CHANNELS) == 8

    def test_migration_index(self):
        assert PF_CHANNELS[MIGRATION_IDX] == "migration_connectivity"

    def test_all_categories_present(self):
        cats = {e.regime_category for e in PF_ENTITIES}
        assert cats == {"well_mixed", "partial", "severe", "restoration"}

    def test_three_per_category(self):
        from collections import Counter

        counts = Counter(e.regime_category for e in PF_ENTITIES)
        assert counts["well_mixed"] == 3
        assert counts["partial"] == 3
        assert counts["severe"] == 3
        assert counts["restoration"] == 3

    @pytest.mark.parametrize("entity", PF_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        assert entity.trace_vector().shape == (8,)

    @pytest.mark.parametrize("entity", PF_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)

    @pytest.mark.parametrize("entity", PF_ENTITIES, ids=lambda e: e.name)
    def test_unique_names(self, entity):
        names = [e.name for e in PF_ENTITIES]
        assert names.count(entity.name) == 1


# ═════════════════════════════════════════════════════════════════════
# Section 2: Tier-1 Kernel Identities
# ═════════════════════════════════════════════════════════════════════


class TestTier1Identities:
    @pytest.mark.parametrize("entity", PF_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        r = compute_pf_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", PF_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        r = compute_pf_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", PF_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        r = compute_pf_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10

    @pytest.mark.parametrize("entity", PF_ENTITIES, ids=lambda e: e.name)
    def test_heterogeneity_gap_nonnegative(self, entity):
        r = compute_pf_kernel(entity)
        assert r.heterogeneity_gap >= -1e-12


# ═════════════════════════════════════════════════════════════════════
# Section 3: Theorems T-EV-11 through T-EV-16
# ═════════════════════════════════════════════════════════════════════


class TestTheorems:
    def test_t_ev_11(self, all_results):
        assert verify_t_ev_11(all_results)["passed"]

    def test_t_ev_12(self, all_results):
        assert verify_t_ev_12(all_results)["passed"]

    def test_t_ev_13(self, all_results):
        assert verify_t_ev_13(all_results)["passed"]

    def test_t_ev_14(self, all_results):
        assert verify_t_ev_14(all_results)["passed"]

    def test_t_ev_15(self, all_results):
        assert verify_t_ev_15(all_results)["passed"]

    def test_t_ev_16(self, all_results):
        assert verify_t_ev_16(all_results)["passed"]

    def test_all_theorems_pass(self):
        for t in verify_all_theorems():
            assert t["passed"], f"{t['name']} failed: {t}"


# ═════════════════════════════════════════════════════════════════════
# Section 4: Regime Classification
# ═════════════════════════════════════════════════════════════════════


class TestRegimeClassification:
    @pytest.mark.parametrize("entity", PF_ENTITIES, ids=lambda e: e.name)
    def test_regime_is_valid(self, entity):
        r = compute_pf_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")


# ═════════════════════════════════════════════════════════════════════
# Section 5: Cross-Domain Bridge
# ═════════════════════════════════════════════════════════════════════


class TestCrossDomainBridge:
    def test_bridge_returns_dict(self, all_results):
        bridge = cross_domain_bridge(all_results)
        assert isinstance(bridge, dict)

    def test_bridge_keys(self, all_results):
        bridge = cross_domain_bridge(all_results)
        expected = {
            "eco_severe_mean_IC_F",
            "tumor_mean_IC_F",
            "both_low_migration",
            "geometry_shared",
            "desirable_direction_inverted",
            "interpretation",
        }
        assert expected <= set(bridge.keys())

    def test_desirable_direction_inverted(self, all_results):
        bridge = cross_domain_bridge(all_results)
        assert bridge["desirable_direction_inverted"] is True
