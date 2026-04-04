import pytest

from closures.ecology.ecology_kernel import ECOLOGICAL_STATES, analyze_ecology_state


def test_eco_t1_trophic_cascade():
    """T-ECO-1: Trophic cascades mirror QGP confinement."""
    out_kl = analyze_ecology_state("KEYSTONE_LOSS")
    out_tc = analyze_ecology_state("TROPHIC_CASCADE")
    assert out_tc["IC"] < out_kl["IC"]


def test_eco_t2_climax_coherence():
    """T-ECO-2: Climax forests maximize composite integrity."""
    out = analyze_ecology_state("CLIMAX_FOREST")
    assert out["IC"] > 0.8


def test_eco_t3_extinction_slaughter():
    """T-ECO-3: Mass extinction zeroes IC but preserves math identity."""
    out = analyze_ecology_state("MASS_EXTINCTION")
    assert out["IC"] < 0.01


def test_eco_t4_agriculture_gap():
    """T-ECO-4: Monoculture agriculture has massive heterogeneity gap."""
    out = analyze_ecology_state("AGRICULTURE")
    gap = out["F"] - out["IC"]
    assert gap > 0.1


def test_eco_t5_algal_bloom_collapse():
    """T-ECO-5: Algal blooms show high biomass but zero resilience."""
    out = analyze_ecology_state("ALGAL_BLOOM")
    assert out["F"] > 0.1
    assert out["IC"] < 0.05


@pytest.mark.parametrize("state", list(ECOLOGICAL_STATES.keys()))
def test_eco_t6_integrity_bound(state):
    """T-ECO-6: IC <= F bound holds universally across biomes."""
    out = analyze_ecology_state(state)
    assert out["IC"] <= out["F"] + 1e-10
