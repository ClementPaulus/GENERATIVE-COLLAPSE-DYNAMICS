"""Cross-domain bridge: coherence universality (semiotics ↔ finance ↔ topology).

Tests Tier-1 kernel identity universality across three maximally disparate
abstract domains that share NO ontological substrate:

  Dynamic Semiotics:          30 sign systems, 8 channels (sign theory)
  Market Microstructure:      12 market venues, 8 channels (financial)
  Topological Persistence:    12 topological spaces, 8 channels (mathematics)

These domains span language → economics → pure mathematics.  The bridge
verifies that the kernel's algebraic structure is substrate-agnostic:
the same duality, integrity bound, and log-integrity hold whether the
channels measure sign repertoire, bid-ask spread, or Betti numbers.

Additionally tests:
 - IC/F ratio distribution across abstract domains
 - Regime partition comparison (signs vs markets vs spaces)
 - Heterogeneity gap as a universal coherence diagnostic
 - Fidelity range: which abstract domain achieves highest F?
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.continuity_theory.topological_persistence import (
    TP_ENTITIES,
    compute_tp_kernel,
)
from closures.dynamic_semiotics.semiotic_kernel import (
    SIGN_SYSTEMS,
    compute_all_sign_systems,
    compute_semiotic_kernel,
)
from closures.finance.market_microstructure import (
    MM_ENTITIES,
    compute_mm_kernel,
)


@pytest.fixture(scope="module")
def sem_results():
    return compute_all_sign_systems()


@pytest.fixture(scope="module")
def mm_results():
    return [compute_mm_kernel(e) for e in MM_ENTITIES]


@pytest.fixture(scope="module")
def tp_results():
    return [compute_tp_kernel(e) for e in TP_ENTITIES]


# ── Tier-1 Identity Universality Across Abstract Domains ──


class TestDualityAcrossDomains:
    """F + ω = 1 in signs, markets, and topological spaces."""

    @pytest.mark.parametrize("ss", SIGN_SYSTEMS, ids=lambda s: s.name)
    def test_semiotics(self, ss):
        r = compute_semiotic_kernel(ss)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", MM_ENTITIES, ids=lambda e: e.name)
    def test_finance(self, entity):
        r = compute_mm_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", TP_ENTITIES, ids=lambda e: e.name)
    def test_topology(self, entity):
        r = compute_tp_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12


class TestIntegrityBoundAcrossDomains:
    """IC ≤ F in signs, markets, and topological spaces."""

    @pytest.mark.parametrize("ss", SIGN_SYSTEMS, ids=lambda s: s.name)
    def test_semiotics(self, ss):
        r = compute_semiotic_kernel(ss)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", MM_ENTITIES, ids=lambda e: e.name)
    def test_finance(self, entity):
        r = compute_mm_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", TP_ENTITIES, ids=lambda e: e.name)
    def test_topology(self, entity):
        r = compute_tp_kernel(entity)
        assert r.IC <= r.F + 1e-12


class TestLogIntegrityAcrossDomains:
    """IC = exp(κ) in signs, markets, and topological spaces."""

    @pytest.mark.parametrize("ss", SIGN_SYSTEMS, ids=lambda s: s.name)
    def test_semiotics(self, ss):
        r = compute_semiotic_kernel(ss)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10

    @pytest.mark.parametrize("entity", MM_ENTITIES, ids=lambda e: e.name)
    def test_finance(self, entity):
        r = compute_mm_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10

    @pytest.mark.parametrize("entity", TP_ENTITIES, ids=lambda e: e.name)
    def test_topology(self, entity):
        r = compute_tp_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10


# ── Cross-Domain Structural Phenomena ──


class TestHeterogeneityGapUniversal:
    """Δ = F − IC ≥ 0 across all three abstract domains."""

    def test_all_gaps_nonneg(self, sem_results, mm_results, tp_results):
        for r in sem_results + mm_results + tp_results:
            assert r.F - r.IC >= -1e-12, f"{r.name}: Δ = {r.F - r.IC}"


class TestICFRatioDistribution:
    """IC/F ratio reveals coherence profile per domain.

    Topological spaces (pure math, closed-form) should have highest IC/F.
    Markets (high-frequency noise, heterogeneous) may have lower IC/F.
    Signs (varying from dead to living languages) span a wide IC/F range.
    """

    def test_each_domain_has_nonzero_ic_f(self, sem_results, mm_results, tp_results):
        for label, results in [
            ("semiotics", sem_results),
            ("finance", mm_results),
            ("topology", tp_results),
        ]:
            ratios = [r.IC / r.F for r in results if r.F > 1e-8]
            assert np.mean(ratios) > 0.0, f"{label} IC/F mean = {np.mean(ratios)}"

    def test_three_domains_have_distinct_mean_ic_f(self, sem_results, mm_results, tp_results):
        means = []
        for results in [sem_results, mm_results, tp_results]:
            ratios = [r.IC / r.F for r in results if r.F > 1e-8]
            means.append(np.mean(ratios))
        # At least two of three should differ by > 0.01
        diffs = [abs(means[i] - means[j]) for i in range(3) for j in range(i + 1, 3)]
        assert max(diffs) > 0.01, f"IC/F means too similar: {means}"


class TestRegimePartition:
    """Regime distribution differs across abstract domains."""

    def test_all_regimes_valid(self, sem_results, mm_results, tp_results):
        for r in sem_results + mm_results + tp_results:
            assert r.regime in ("Stable", "Watch", "Collapse")

    def test_combined_regime_diversity(self, sem_results, mm_results, tp_results):
        regimes = {r.regime for r in sem_results + mm_results + tp_results}
        assert len(regimes) >= 2, f"Only regime(s): {regimes}"


class TestFidelityRange:
    """Different abstract domains achieve different fidelity ranges.

    Semiotics has 30 entities spanning dead→living languages → wide F range.
    Finance has 12 venues from dark pools to exchanges → moderate F range.
    Topology has 12 spaces from sphere to Klein bottle → depends on invariants.
    """

    def test_semiotics_wide_F_range(self, sem_results):
        Fs = [r.F for r in sem_results]
        assert max(Fs) - min(Fs) > 0.1, f"Sem F range: {min(Fs):.3f}–{max(Fs):.3f}"

    def test_combined_spans_manifold(self, sem_results, mm_results, tp_results):
        all_F = [r.F for r in sem_results + mm_results + tp_results]
        f_range = max(all_F) - min(all_F)
        assert f_range > 0.15, f"Combined F range = {f_range:.3f}"


class TestCurvatureComparison:
    """Curvature C reflects channel heterogeneity differently per domain."""

    def test_each_domain_has_bounded_C(self, sem_results, mm_results, tp_results):
        for label, results in [
            ("semiotics", sem_results),
            ("finance", mm_results),
            ("topology", tp_results),
        ]:
            for r in results:
                assert 0.0 <= r.C <= 2.0, f"{label}/{r.name}: C = {r.C}"
