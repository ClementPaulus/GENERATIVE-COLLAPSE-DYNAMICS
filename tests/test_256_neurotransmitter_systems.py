"""Tests for neurotransmitter systems closure (clinical neuroscience domain).

Validates 15 neurotransmitter entities, 8-channel trace construction,
Tier-1 kernel identities, and 6 theorems (T-NT-1 through T-NT-6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.clinical_neuroscience.neurotransmitter_systems import (
    N_NT_CHANNELS,
    NT_CHANNELS,
    NT_ENTITIES,
    NTKernelResult,
    compute_all_entities,
    compute_nt_kernel,
    verify_all_theorems,
    verify_t_nt_1,
    verify_t_nt_2,
    verify_t_nt_3,
    verify_t_nt_4,
    verify_t_nt_5,
    verify_t_nt_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[NTKernelResult]:
    return compute_all_entities()


# ── Entity Catalog ──


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(NT_ENTITIES) == 15

    def test_channel_count(self):
        assert N_NT_CHANNELS == 8
        assert len(NT_CHANNELS) == 8

    def test_all_categories_present(self):
        cats = {e.category for e in NT_ENTITIES}
        assert cats == {"monoamine", "amino_acid", "neuropeptide", "other"}

    def test_unique_names(self):
        names = [e.name for e in NT_ENTITIES]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("entity", NT_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        c = entity.trace_vector()
        assert c.shape == (8,)

    @pytest.mark.parametrize("entity", NT_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)


# ── Tier-1 Kernel Identities ──


class TestTier1Identities:
    @pytest.mark.parametrize("entity", NT_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        r = compute_nt_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", NT_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        r = compute_nt_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", NT_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        r = compute_nt_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10


# ── Theorems ──


class TestTheorems:
    def test_t_nt_1(self, all_results):
        t = verify_t_nt_1(all_results)
        assert t["passed"], t

    def test_t_nt_2(self, all_results):
        t = verify_t_nt_2(all_results)
        assert t["passed"], t

    def test_t_nt_3(self, all_results):
        t = verify_t_nt_3(all_results)
        assert t["passed"], t

    def test_t_nt_4(self, all_results):
        t = verify_t_nt_4(all_results)
        assert t["passed"], t

    def test_t_nt_5(self, all_results):
        t = verify_t_nt_5(all_results)
        assert t["passed"], t

    def test_t_nt_6(self, all_results):
        t = verify_t_nt_6(all_results)
        assert t["passed"], t

    def test_all_theorems_pass(self, all_results):
        results_list = verify_all_theorems()
        for t in results_list:
            assert t["passed"], f"{t['name']} failed: {t}"


# ── Regime Classification ──


class TestRegimeClassification:
    @pytest.mark.parametrize("entity", NT_ENTITIES, ids=lambda e: e.name)
    def test_regime_is_valid(self, entity):
        r = compute_nt_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")

    def test_regime_consistency_with_omega(self, all_results):
        for r in all_results:
            if r.omega >= 0.30:
                assert r.regime == "Collapse", f"{r.name}: ω={r.omega} but regime={r.regime}"


# ── Serialization ──


class TestSerialization:
    def test_to_dict_keys(self, all_results):
        d = all_results[0].to_dict()
        expected = {"name", "category", "F", "omega", "S", "C", "kappa", "IC", "regime"}
        assert set(d.keys()) == expected
