"""Tests for Photonic Confinement Model (CPM) closure.

Validates the 7 theorems from the GCD kernel analysis of Caputo's
Photonic-Conjugated Model (PCM/CPM), "Confined Photonic System,"
v37, February 2026.  DOI: 10.5281/zenodo.17509488

Test coverage:
  - Entity catalog integrity (12 entities, categories, parameters)
  - 8-channel trace construction (bounds, independence, clamping)
  - Tier-1 kernel identities for all entities (duality, integrity bound,
    log-integrity relation)
  - All 7 theorems (T-PCM-1 through T-PCM-7)
  - Regime classification consistency
  - Channel autopsy (weakest/strongest identification)
  - Serialization round-trip (to_dict)
  - Cross-category invariant comparisons
  - Key structural signatures (geometric slaughter, phase commensurability)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.quantum_mechanics.photonic_confinement import (
    CHANNEL_LABELS,
    CPM_ENTITIES,
    EPSILON,
    N_CHANNELS,
    WEIGHTS,
    CPMEntity,
    CPMKernelResult,
    classify_regime,
    compute_all_entities,
    compute_cpm_kernel,
    verify_all_theorems,
    verify_t_pcm_1,
    verify_t_pcm_2,
    verify_t_pcm_3,
    verify_t_pcm_4,
    verify_t_pcm_5,
    verify_t_pcm_6,
    verify_t_pcm_7,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def all_results() -> list[CPMKernelResult]:
    """Compute kernel invariants for all 12 CPM entities."""
    return compute_all_entities()


@pytest.fixture(scope="module")
def results_by_name(all_results: list[CPMKernelResult]) -> dict[str, CPMKernelResult]:
    """Index results by entity name."""
    return {r.name: r for r in all_results}


@pytest.fixture(scope="module")
def leptons(all_results: list[CPMKernelResult]) -> list[CPMKernelResult]:
    """Lepton category results."""
    return [r for r in all_results if r.category == "lepton"]


@pytest.fixture(scope="module")
def hadrons(all_results: list[CPMKernelResult]) -> list[CPMKernelResult]:
    """Hadron category results."""
    return [r for r in all_results if r.category == "hadron"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: ENTITY CATALOG INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════


class TestEntityCatalog:
    """Verify the CPM entity catalog structure and contents."""

    def test_entity_count(self) -> None:
        assert len(CPM_ENTITIES) == 12

    def test_unique_names(self) -> None:
        names = [e.name for e in CPM_ENTITIES]
        assert len(names) == len(set(names))

    def test_all_categories_present(self) -> None:
        categories = {e.category for e in CPM_ENTITIES}
        assert categories == {"unconfined", "lepton", "hadron", "composite", "cosmological"}

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_entity_has_description(self, entity: CPMEntity) -> None:
        assert len(entity.description) > 10

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_entity_is_frozen(self, entity: CPMEntity) -> None:
        with pytest.raises(AttributeError):
            entity.name = "mutated"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: TRACE VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


class TestTraceVector:
    """Verify trace vector construction and bounds."""

    def test_channel_count(self) -> None:
        assert N_CHANNELS == 8
        assert len(CHANNEL_LABELS) == 8

    def test_weights_uniform(self) -> None:
        assert len(WEIGHTS) == N_CHANNELS
        np.testing.assert_allclose(WEIGHTS.sum(), 1.0, atol=1e-15)
        np.testing.assert_allclose(WEIGHTS, 1.0 / N_CHANNELS)

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_length(self, entity: CPMEntity) -> None:
        c = entity.trace_vector()
        assert len(c) == N_CHANNELS

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity: CPMEntity) -> None:
        c = entity.trace_vector()
        assert np.all(c >= EPSILON), f"{entity.name}: channel below ε"
        assert np.all(c <= 1.0 - EPSILON), f"{entity.name}: channel above 1−ε"

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_dtype(self, entity: CPMEntity) -> None:
        c = entity.trace_vector()
        assert c.dtype == np.float64

    def test_free_photon_confinement_clamped(self) -> None:
        """Free photon confinement_degree=0 should clamp to ε."""
        free = CPM_ENTITIES[0]
        c = free.trace_vector()
        assert c[0] == EPSILON


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: TIER-1 KERNEL IDENTITIES
# ═══════════════════════════════════════════════════════════════════════════


class TestTier1Identities:
    """Verify Tier-1 identities hold for every CPM entity."""

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity: CPMEntity) -> None:
        """F + ω = 1 — exact to machine precision."""
        r = compute_cpm_kernel(entity)
        assert abs(r.F_plus_omega - 1.0) < 1e-12, f"{entity.name}: F+ω = {r.F_plus_omega}"

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity: CPMEntity) -> None:
        """IC ≤ F — the integrity bound."""
        r = compute_cpm_kernel(entity)
        assert r.IC_leq_F, f"{entity.name}: IC={r.IC:.6f} > F={r.F:.6f}"

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity: CPMEntity) -> None:
        """IC = exp(κ) — log-integrity relation."""
        r = compute_cpm_kernel(entity)
        assert r.IC_eq_exp_kappa, f"{entity.name}: IC={r.IC:.6e}, exp(κ)={math.exp(r.kappa):.6e}"

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_positive_entropy(self, entity: CPMEntity) -> None:
        """Bernoulli field entropy S ≥ 0."""
        r = compute_cpm_kernel(entity)
        assert r.S >= -1e-15, f"{entity.name}: S={r.S}"

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_heterogeneity_gap_non_negative(self, entity: CPMEntity) -> None:
        """Δ = F − IC ≥ 0 (consequence of IC ≤ F)."""
        r = compute_cpm_kernel(entity)
        assert r.heterogeneity_gap >= -1e-12, f"{entity.name}: Δ={r.heterogeneity_gap}"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: THEOREM VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class TestTheorems:
    """Verify all 7 CPM theorems."""

    def test_t_pcm_1_confinement_fidelity(self, all_results: list[CPMKernelResult]) -> None:
        """T-PCM-1: Confined entities have higher F than unconfined."""
        result = verify_t_pcm_1(all_results)
        assert result["passed"], (
            f"mean_F confined={result['mean_F_confined']:.4f} vs unconfined={result['mean_F_unconfined']:.4f}"
        )
        assert result["separation"] > 0.30  # substantial separation

    def test_t_pcm_2_irrational_phase(self, all_results: list[CPMKernelResult]) -> None:
        """T-PCM-2: Irrational phase creates heterogeneity gap."""
        result = verify_t_pcm_2(all_results)
        assert result["passed"]
        assert result["all_low_commensurability"]
        assert result["all_positive_gap"]
        assert result["mean_heterogeneity_gap"] > 0.05

    def test_t_pcm_3_generatrix_coupling(self, all_results: list[CPMKernelResult]) -> None:
        """T-PCM-3: Generatrix curvature maps to kernel curvature."""
        result = verify_t_pcm_3(all_results)
        assert result["passed"]

    def test_t_pcm_4_geometric_slaughter(self, all_results: list[CPMKernelResult]) -> None:
        """T-PCM-4: Free photon shows geometric slaughter."""
        result = verify_t_pcm_4(all_results)
        assert result["passed"]
        assert result["IC_over_F_free_photon"] < 0.01  # near-zero IC/F

    def test_t_pcm_5_phase_memory(self, all_results: list[CPMKernelResult]) -> None:
        """T-PCM-5: Phase memory preserves IC/F ratio."""
        result = verify_t_pcm_5(all_results)
        assert result["passed"]
        assert result["n_high"] >= 6
        assert result["n_low"] >= 3

    def test_t_pcm_6_scale_invariance(self, all_results: list[CPMKernelResult]) -> None:
        """T-PCM-6: Tier-1 identities hold across all scales."""
        result = verify_t_pcm_6(all_results)
        assert result["passed"]
        assert result["all_duality_exact"]
        assert result["all_integrity_bound"]
        assert result["all_log_integrity"]

    def test_t_pcm_7_ontological_neutrality(self, all_results: list[CPMKernelResult]) -> None:
        """T-PCM-7: Kernel is ontologically neutral."""
        result = verify_t_pcm_7(all_results)
        assert result["passed"]
        assert result["all_exact"]
        assert len(result["categories"]) == 5  # all 5 categories

    def test_all_theorems_batch(self, all_results: list[CPMKernelResult]) -> None:
        """All 7 theorems pass in batch."""
        theorems = verify_all_theorems(all_results)
        assert len(theorems) == 7
        for t in theorems:
            assert t["passed"], f"{t['theorem']} ({t['title']}) FAILED"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeClassification:
    """Verify regime classification for CPM entities."""

    def test_classify_regime_stable(self) -> None:
        assert classify_regime(0.02, 0.95, 0.10, 0.10) == "Stable"

    def test_classify_regime_watch(self) -> None:
        assert classify_regime(0.15, 0.80, 0.30, 0.20) == "Watch"

    def test_classify_regime_collapse(self) -> None:
        assert classify_regime(0.50, 0.50, 0.60, 0.50) == "Collapse"

    def test_proton_is_watch_or_collapse(self, results_by_name: dict[str, CPMKernelResult]) -> None:
        """Proton is the most confined — should be Watch or borderline Collapse."""
        r = results_by_name["proton"]
        assert r.regime in ("Watch", "Collapse")

    def test_free_photon_is_collapse(self, results_by_name: dict[str, CPMKernelResult]) -> None:
        """Free photon is maximally deconfined — should be Collapse regime."""
        r = results_by_name["free_photon"]
        assert r.regime == "Collapse"

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_regime_is_valid(self, entity: CPMEntity) -> None:
        r = compute_cpm_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: STRUCTURAL SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuralSignatures:
    """Verify key structural signatures from CPM physics."""

    def test_electron_positron_symmetry(self, results_by_name: dict[str, CPMKernelResult]) -> None:
        """Electron and positron are CPT conjugates — identical kernel output."""
        e = results_by_name["electron"]
        p = results_by_name["positron"]
        assert abs(e.F - p.F) < 1e-12
        assert abs(e.IC - p.IC) < 1e-12
        assert abs(e.S - p.S) < 1e-12

    def test_proton_higher_F_than_neutron(self, results_by_name: dict[str, CPMKernelResult]) -> None:
        """Proton (stable) should have slightly higher F than neutron (unstable)."""
        assert results_by_name["proton"].F > results_by_name["neutron"].F

    def test_neutrino_lowest_F_among_leptons(self, leptons: list[CPMKernelResult]) -> None:
        """Neutrino has minimal confinement → lowest F among leptons."""
        neutrino = next(r for r in leptons if r.name == "neutrino")
        others = [r for r in leptons if r.name != "neutrino"]
        assert all(neutrino.F < r.F for r in others)

    def test_phase_commensurability_is_universal_weak_channel(self, all_results: list[CPMKernelResult]) -> None:
        """Phase commensurability is the weakest channel for most confined entities."""
        confined = [r for r in all_results if r.category in ("lepton", "hadron")]
        # Exclude neutrino (mass_confinement_ratio is weaker)
        charged_confined = [r for r in confined if r.name != "neutrino"]
        assert all(r.weakest_channel == "phase_commensurability" for r in charged_confined)

    def test_free_photon_geometric_slaughter(self, results_by_name: dict[str, CPMKernelResult]) -> None:
        """Free photon: confinement at ε creates geometric slaughter."""
        fp = results_by_name["free_photon"]
        assert fp.IC / fp.F < 0.01
        assert fp.weakest_channel == "confinement_degree"

    def test_hydrogen_atom_coherence_recovery(self, results_by_name: dict[str, CPMKernelResult]) -> None:
        """Hydrogen atom should show higher IC/F than free photon (scale recovery)."""
        h = results_by_name["hydrogen_atom"]
        fp = results_by_name["free_photon"]
        assert (h.IC / h.F) > (fp.IC / fp.F) * 10  # at least 10× recovery

    def test_cosmic_pi_highest_lattice_coherence(self, results_by_name: dict[str, CPMKernelResult]) -> None:
        """Cosmic-Pi field has maximal lattice coherence channel."""
        cp = results_by_name["cosmic_pi_field"]
        assert cp.trace_vector[4] > 0.95  # lattice_coherence

    def test_lepton_generation_monotonicity(self, leptons: list[CPMKernelResult]) -> None:
        """Charged leptons: e < μ < τ in mass_confinement_ratio."""
        charged = {r.name: r for r in leptons if r.name in ("electron", "muon", "tau_lepton")}
        assert charged["electron"].trace_vector[5] < charged["muon"].trace_vector[5]
        assert charged["muon"].trace_vector[5] < charged["tau_lepton"].trace_vector[5]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Verify to_dict serialization."""

    @pytest.mark.parametrize("entity", CPM_ENTITIES, ids=lambda e: e.name)
    def test_to_dict_round_trip(self, entity: CPMEntity) -> None:
        r = compute_cpm_kernel(entity)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == entity.name
        assert d["category"] == entity.category
        assert len(d["trace_vector"]) == N_CHANNELS
        assert isinstance(d["F"], float)
        assert isinstance(d["omega"], float)
        assert isinstance(d["IC"], float)
        assert isinstance(d["regime"], str)

    def test_to_dict_has_all_keys(self, all_results: list[CPMKernelResult]) -> None:
        expected_keys = {
            "name",
            "category",
            "description",
            "n_channels",
            "channel_labels",
            "trace_vector",
            "F",
            "omega",
            "S",
            "C",
            "kappa",
            "IC",
            "heterogeneity_gap",
            "F_plus_omega",
            "IC_leq_F",
            "IC_eq_exp_kappa",
            "regime",
            "weakest_channel",
            "weakest_value",
            "strongest_channel",
            "strongest_value",
        }
        for r in all_results:
            assert set(r.to_dict().keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: CROSS-CATEGORY COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossCategoryComparisons:
    """Compare kernel invariants across CPM entity categories."""

    def test_hadrons_higher_F_than_unconfined(self, all_results: list[CPMKernelResult]) -> None:
        hadrons = [r for r in all_results if r.category == "hadron"]
        unconfined = [r for r in all_results if r.category == "unconfined"]
        mean_F_hadron = np.mean([r.F for r in hadrons])
        mean_F_unconf = np.mean([r.F for r in unconfined])
        assert mean_F_hadron > mean_F_unconf

    def test_leptons_higher_F_than_unconfined(self, all_results: list[CPMKernelResult]) -> None:
        leptons = [r for r in all_results if r.category == "lepton"]
        unconfined = [r for r in all_results if r.category == "unconfined"]
        # Exclude neutrino — it's barely confined
        charged_leptons = [r for r in leptons if r.name != "neutrino"]
        mean_F_lepton = np.mean([r.F for r in charged_leptons])
        mean_F_unconf = np.mean([r.F for r in unconfined])
        assert mean_F_lepton > mean_F_unconf

    def test_proton_highest_F(self, all_results: list[CPMKernelResult]) -> None:
        """Proton — maximal confinement — should have highest F."""
        proton_F = next(r.F for r in all_results if r.name == "proton")
        assert all(proton_F >= r.F - 1e-10 for r in all_results)

    def test_all_entities_heterogeneity_gap_positive(self, all_results: list[CPMKernelResult]) -> None:
        for r in all_results:
            assert r.heterogeneity_gap >= -1e-12
