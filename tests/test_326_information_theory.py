import pytest

from closures.information_theory.information_theory_kernel import COMPLEXITY_CLASSES, analyze_complexity_class


def test_it_t1_p_vs_np():
    """T-IT-1: P != NP maps to integrity collapse."""
    out_p = analyze_complexity_class("P")
    out_np = analyze_complexity_class("NP")
    gap_p = out_p["F"] - out_p["IC"]
    gap_np = out_np["F"] - out_np["IC"]
    assert gap_np > gap_p


def test_it_t2_halting_slaughter():
    """T-IT-2: RE geometrically slaughters IC while F holds."""
    out = analyze_complexity_class("RE")
    assert out["IC"] < 0.05
    assert out["F"] > 0.001


def test_it_t3_bpp_bqp_proximity():
    """T-IT-3: Quantum class BQP introduces minor drift over BPP."""
    out_bpp = analyze_complexity_class("BPP")
    out_bqp = analyze_complexity_class("BQP")
    assert out_bqp["omega"] > out_bpp["omega"]


def test_it_t4_exptime_confinement():
    """T-IT-4: Exponential time behaves like confinement."""
    out = analyze_complexity_class("EXPTIME")
    assert out["IC"] < 0.1


def test_it_t5_constant_time_fidelity():
    """T-IT-5: O(1) approaches ideal fidelity."""
    out = analyze_complexity_class("O1")
    assert out["F"] > 0.9


@pytest.mark.parametrize("cls_name", list(COMPLEXITY_CLASSES.keys()))
def test_it_t6_duality_identity(cls_name):
    """T-IT-6: Exact duality holds across all complexity classes."""
    out = analyze_complexity_class(cls_name)
    assert abs(out["F"] + out["omega"] - 1.0) < 1e-12
