import pytest

from closures.dynamic_semiotics.linguistic_evolution_kernel import LINGUISTIC_STATES, analyze_language_state


def test_ling_t1_pidgin_gap():
    """T-LING-1: Pidgins preserve semantics but drop syntax, widening gap."""
    out_pidgin = analyze_language_state("PIDGIN")
    gap = out_pidgin["F"] - out_pidgin["IC"]
    assert gap > 0.1


def test_ling_t2_creolization():
    """T-LING-2: Creolization restores syntax and IC compared to pidgins."""
    out_p = analyze_language_state("PIDGIN")
    out_c = analyze_language_state("CREOLE")
    assert out_c["IC"] > out_p["IC"]


def test_ling_t3_bleaching():
    """T-LING-3: Semantic bleaching lowers fidelity but preserves transmission."""
    out = analyze_language_state("SEMANTIC_BLEACHING")
    assert out["F"] < 0.9


def test_ling_t4_liturgical_death():
    """T-LING-4: Liturgical languages maintain syntax but no transmission."""
    out = analyze_language_state("LITURGICAL_DEAD")
    assert out["IC"] < 0.05


def test_ling_t5_jargon():
    """T-LING-5: Jargon lacks syntax and transmission."""
    out = analyze_language_state("JARGON")
    assert out["IC"] < out["F"]


@pytest.mark.parametrize("state", list(LINGUISTIC_STATES.keys()))
def test_ling_t6_log_integrity(state):
    """T-LING-6: Log-integrity theorem IC = exp(kappa) holds."""
    import numpy as np

    out = analyze_language_state(state)
    assert abs(out["IC"] - np.exp(out["kappa"])) < 1e-12
