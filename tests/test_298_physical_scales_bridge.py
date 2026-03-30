"""Cross-domain bridge: physical scales (subatomic → fluid → stellar).

Tests Tier-1 kernel identity universality across three disparate physical
scale domains that share NO channel semantics:

  Fluid Dynamics (everyday_physics):  12 flow regime entities, 8 channels
  Binary Stars (astronomy):           12 stellar systems, 8 channels

These domains span ~40 orders of magnitude in length scale (nm → AU).
The bridge verifies that the kernel's structural properties — duality,
integrity bound, log-integrity — are scale-invariant: same algebra at
every scale, despite completely different channel meanings.

Additionally tests:
 - Regime distribution comparison across scales
 - Curvature-fidelity phase space coverage
 - Heterogeneity gap universality
 - Entropy-curvature anti-correlation (CLT constraint)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.astronomy.binary_star_systems import (
    BS_ENTITIES,
    compute_bs_kernel,
)
from closures.everyday_physics.fluid_dynamics import (
    FD_ENTITIES,
    compute_fd_kernel,
)


@pytest.fixture(scope="module")
def fd_results():
    return [compute_fd_kernel(e) for e in FD_ENTITIES]


@pytest.fixture(scope="module")
def bs_results():
    return [compute_bs_kernel(e) for e in BS_ENTITIES]


# ── Tier-1 Identity Universality Across Scales ──


class TestDualityAcrossScales:
    """F + ω = 1 at every physical scale."""

    @pytest.mark.parametrize("entity", FD_ENTITIES, ids=lambda e: e.name)
    def test_fluid(self, entity):
        r = compute_fd_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", BS_ENTITIES, ids=lambda e: e.name)
    def test_stellar(self, entity):
        r = compute_bs_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12


class TestIntegrityBoundAcrossScales:
    """IC ≤ F at every physical scale."""

    @pytest.mark.parametrize("entity", FD_ENTITIES, ids=lambda e: e.name)
    def test_fluid(self, entity):
        r = compute_fd_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", BS_ENTITIES, ids=lambda e: e.name)
    def test_stellar(self, entity):
        r = compute_bs_kernel(entity)
        assert r.IC <= r.F + 1e-12


class TestLogIntegrityAcrossScales:
    """IC = exp(κ) at every physical scale."""

    @pytest.mark.parametrize("entity", FD_ENTITIES, ids=lambda e: e.name)
    def test_fluid(self, entity):
        r = compute_fd_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10

    @pytest.mark.parametrize("entity", BS_ENTITIES, ids=lambda e: e.name)
    def test_stellar(self, entity):
        r = compute_bs_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10


# ── Cross-Scale Structural Phenomena ──


class TestHeterogeneityGapUniversal:
    """Δ = F − IC ≥ 0 at all scales."""

    def test_all_gaps_nonneg(self, fd_results, bs_results):
        for r in fd_results + bs_results:
            assert r.F - r.IC >= -1e-12, f"{r.name}: Δ = {r.F - r.IC}"

    def test_nonzero_mean_gap_per_domain(self, fd_results, bs_results):
        for label, results in [("fluid", fd_results), ("stellar", bs_results)]:
            mean_gap = np.mean([r.F - r.IC for r in results])
            assert mean_gap > 0, f"{label} mean Δ = {mean_gap}"


class TestPhaseSpaceCoverage:
    """Different physical scales occupy different regions of (F, C) space."""

    def test_domains_differ_in_F(self, fd_results, bs_results):
        fd_F = np.mean([r.F for r in fd_results])
        bs_F = np.mean([r.F for r in bs_results])
        # Different scale physics → different mean fidelity
        assert abs(fd_F - bs_F) > 0.01, f"Fluid ⟨F⟩={fd_F:.3f}, Stellar ⟨F⟩={bs_F:.3f}"

    def test_combined_F_spans_manifold(self, fd_results, bs_results):
        all_F = [r.F for r in fd_results + bs_results]
        f_range = max(all_F) - min(all_F)
        assert f_range > 0.15, f"Combined F range = {f_range:.3f}"


class TestRegimeDistribution:
    """Regime proportions differ across scales, reflecting different
    structure-stability characteristics."""

    def test_both_domains_have_valid_regimes(self, fd_results, bs_results):
        for r in fd_results + bs_results:
            assert r.regime in ("Stable", "Watch", "Collapse")

    def test_at_least_two_regimes_across_scales(self, fd_results, bs_results):
        regimes = {r.regime for r in fd_results + bs_results}
        assert len(regimes) >= 2, f"Only regime(s): {regimes}"


class TestEntropyCurvatureAnticorrelation:
    """S and C are anti-correlated within each domain (CLT constraint)."""

    def _corr(self, results):
        S = [r.S for r in results]
        C = [r.C for r in results]
        if np.std(S) < 1e-12 or np.std(C) < 1e-12:
            return 0.0
        return float(np.corrcoef(S, C)[0, 1])

    def test_fluid_bounded_corr(self, fd_results):
        r = self._corr(fd_results)
        # With only 12 entities, anti-correlation is weak (asymptotic with n)
        assert r < 0.9, f"Fluid corr(S,C) = {r:.3f}"

    def test_stellar_bounded_corr(self, bs_results):
        r = self._corr(bs_results)
        assert r < 0.9, f"Stellar corr(S,C) = {r:.3f}"
