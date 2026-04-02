"""Tests for polymer theorems closure (materials science domain).

Validates 12 entities, 8-channel trace construction,
Tier-1 kernel identities, and 6 theorems (T-PO-1 through T-PO-6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.materials_science.polymer_theorems import (
    N_PO_CHANNELS,
    PO_CHANNELS,
    PO_ENTITIES,
    POKernelResult,
    compute_all_entities,
    compute_po_kernel,
    verify_all_theorems,
    verify_t_po_1,
    verify_t_po_2,
    verify_t_po_3,
    verify_t_po_4,
    verify_t_po_5,
    verify_t_po_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[POKernelResult]:
    return compute_all_entities()


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(PO_ENTITIES) == 12

    def test_channel_count(self):
        assert N_PO_CHANNELS == 8
        assert len(PO_CHANNELS) == 8

    def test_all_categories_present(self):
        cats = {e.category for e in PO_ENTITIES}
        assert cats == {"thermoplastic", "engineering", "elastomer", "thermoset"}

    @pytest.mark.parametrize("entity", PO_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        assert entity.trace_vector().shape == (8,)

    @pytest.mark.parametrize("entity", PO_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)


class TestTier1Identities:
    @pytest.mark.parametrize("entity", PO_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        r = compute_po_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", PO_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        r = compute_po_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", PO_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        r = compute_po_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10


class TestTheorems:
    def test_t_po_1(self, all_results):
        assert verify_t_po_1(all_results)["passed"]

    def test_t_po_2(self, all_results):
        assert verify_t_po_2(all_results)["passed"]

    def test_t_po_3(self, all_results):
        assert verify_t_po_3(all_results)["passed"]

    def test_t_po_4(self, all_results):
        assert verify_t_po_4(all_results)["passed"]

    def test_t_po_5(self, all_results):
        assert verify_t_po_5(all_results)["passed"]

    def test_t_po_6(self, all_results):
        assert verify_t_po_6(all_results)["passed"]

    def test_all_theorems(self, all_results):
        results = verify_all_theorems()
        assert all(r["passed"] for r in results)


class TestRegimeClassification:
    @pytest.mark.parametrize("entity", PO_ENTITIES, ids=lambda e: e.name)
    def test_valid_regime(self, entity):
        r = compute_po_kernel(entity)
        assert r.regime in {"Stable", "Watch", "Collapse"}
