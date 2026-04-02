"""Tests for momentum dynamics theorems closure (kinematics domain).

Validates 12 entities, 8-channel trace construction,
Tier-1 kernel identities, and 6 theorems (T-MD-1 through T-MD-6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.kinematics.momentum_theorems import (
    MD_CHANNELS,
    MD_ENTITIES,
    N_MD_CHANNELS,
    MDKernelResult,
    compute_all_entities,
    compute_md_kernel,
    verify_all_theorems,
    verify_t_md_1,
    verify_t_md_2,
    verify_t_md_3,
    verify_t_md_4,
    verify_t_md_5,
    verify_t_md_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[MDKernelResult]:
    return compute_all_entities()


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(MD_ENTITIES) == 12

    def test_channel_count(self):
        assert N_MD_CHANNELS == 8
        assert len(MD_CHANNELS) == 8

    def test_all_categories_present(self):
        cats = {e.category for e in MD_ENTITIES}
        assert cats == {"elastic", "inelastic", "explosive", "constrained"}

    @pytest.mark.parametrize("entity", MD_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        assert entity.trace_vector().shape == (8,)

    @pytest.mark.parametrize("entity", MD_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)


class TestTier1Identities:
    @pytest.mark.parametrize("entity", MD_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        r = compute_md_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", MD_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        r = compute_md_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", MD_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        r = compute_md_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10


class TestTheorems:
    def test_t_md_1(self, all_results):
        assert verify_t_md_1(all_results)["passed"]

    def test_t_md_2(self, all_results):
        assert verify_t_md_2(all_results)["passed"]

    def test_t_md_3(self, all_results):
        assert verify_t_md_3(all_results)["passed"]

    def test_t_md_4(self, all_results):
        assert verify_t_md_4(all_results)["passed"]

    def test_t_md_5(self, all_results):
        assert verify_t_md_5(all_results)["passed"]

    def test_t_md_6(self, all_results):
        assert verify_t_md_6(all_results)["passed"]

    def test_all_theorems(self, all_results):
        results = verify_all_theorems()
        assert all(r["passed"] for r in results)


class TestRegimeClassification:
    @pytest.mark.parametrize("entity", MD_ENTITIES, ids=lambda e: e.name)
    def test_valid_regime(self, entity):
        r = compute_md_kernel(entity)
        assert r.regime in {"Stable", "Watch", "Collapse"}
