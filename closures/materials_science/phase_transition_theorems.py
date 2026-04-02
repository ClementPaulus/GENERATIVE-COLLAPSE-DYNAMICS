"""Phase Transition Theorems Closure — Materials Science Domain.

Tier-2 closure mapping 12 canonical phase transitions through the GCD kernel.
Each transition is characterized by 8 channels from thermodynamic data.

Channels (8, equal weights w_i = 1/8):
  0  order_parameter    — |ψ|/|ψ_max|, strength of order parameter (1 = fully ordered)
  1  correlation_length — ξ/ξ_max, divergence near criticality normalized
  2  specific_heat_norm — C_p/(C_p_max), normalized heat capacity anomaly
  3  latent_heat_low    — 1/(1+L/L_max), low latent heat → 1.0 (continuous → 1)
  4  symmetry_breaking  — fraction of symmetry broken (1 = full breaking)
  5  hysteresis_low     — 1 - |ΔT_hyst|/T_c, low hysteresis → 1.0
  6  universality_class — mapped to [0,1]: mean-field=0.9, Ising=0.7, XY=0.5, Heisenberg=0.3
  7  critical_exponent  — β/β_mf normalized, deviation from mean-field

12 entities across 4 categories:
  First_order (3):     water_boiling, steel_melting, BaTiO3_ferroelectric
  Second_order (3):    Ising_ferromagnet, He4_lambda, CO2_critical_point
  Topological (3):     BKT_transition, quantum_hall_plateau, skyrmion_lattice
  Quantum (3):         Mott_insulator, superconductor_BCS, Bose_Einstein_condensate

6 theorems (T-PT-1 through T-PT-6).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[2]
for _p in [str(_WORKSPACE / "src"), str(_WORKSPACE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from umcp.frozen_contract import EPSILON  # noqa: E402
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

PT_CHANNELS = [
    "order_parameter",
    "correlation_length",
    "specific_heat_norm",
    "latent_heat_low",
    "symmetry_breaking",
    "hysteresis_low",
    "universality_class",
    "critical_exponent",
]
N_PT_CHANNELS = len(PT_CHANNELS)


@dataclass(frozen=True, slots=True)
class PhaseTransitionEntity:
    """A phase transition with 8 measurable channels."""

    name: str
    category: str
    order_parameter: float
    correlation_length: float
    specific_heat_norm: float
    latent_heat_low: float
    symmetry_breaking: float
    hysteresis_low: float
    universality_class: float
    critical_exponent: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.order_parameter,
                self.correlation_length,
                self.specific_heat_norm,
                self.latent_heat_low,
                self.symmetry_breaking,
                self.hysteresis_low,
                self.universality_class,
                self.critical_exponent,
            ]
        )


PT_ENTITIES: tuple[PhaseTransitionEntity, ...] = (
    # First-order — discontinuous, latent heat, hysteresis
    PhaseTransitionEntity("water_boiling", "first_order", 0.90, 0.10, 0.30, 0.15, 0.95, 0.30, 0.90, 0.95),
    PhaseTransitionEntity("steel_melting", "first_order", 0.85, 0.05, 0.25, 0.10, 0.90, 0.20, 0.90, 0.90),
    PhaseTransitionEntity("BaTiO3_ferroelectric", "first_order", 0.80, 0.15, 0.40, 0.20, 0.85, 0.40, 0.85, 0.88),
    # Second-order — continuous, divergent ξ, universal exponents
    PhaseTransitionEntity("Ising_ferromagnet", "second_order", 0.60, 0.90, 0.85, 0.95, 0.70, 0.90, 0.70, 0.50),
    PhaseTransitionEntity("He4_lambda", "second_order", 0.55, 0.95, 0.90, 0.98, 0.65, 0.95, 0.50, 0.45),
    PhaseTransitionEntity("CO2_critical_point", "second_order", 0.50, 0.85, 0.80, 0.92, 0.60, 0.88, 0.70, 0.55),
    # Topological — non-Landau, no local order parameter
    PhaseTransitionEntity("BKT_transition", "topological", 0.30, 0.70, 0.50, 0.90, 0.40, 0.85, 0.30, 0.30),
    PhaseTransitionEntity("quantum_hall_plateau", "topological", 0.25, 0.60, 0.45, 0.88, 0.35, 0.80, 0.25, 0.25),
    PhaseTransitionEntity("skyrmion_lattice", "topological", 0.35, 0.65, 0.55, 0.85, 0.45, 0.82, 0.35, 0.35),
    # Quantum — zero-temperature, driven by quantum fluctuations
    PhaseTransitionEntity("Mott_insulator", "quantum", 0.70, 0.75, 0.70, 0.80, 0.75, 0.70, 0.60, 0.60),
    PhaseTransitionEntity("superconductor_BCS", "quantum", 0.65, 0.80, 0.75, 0.85, 0.80, 0.75, 0.90, 0.70),
    PhaseTransitionEntity("Bose_Einstein_condensate", "quantum", 0.75, 0.85, 0.80, 0.90, 0.85, 0.80, 0.50, 0.55),
)


@dataclass(frozen=True, slots=True)
class PTKernelResult:
    """Kernel output for a phase transition entity."""

    name: str
    category: str
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    regime: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "F": self.F,
            "omega": self.omega,
            "S": self.S,
            "C": self.C,
            "kappa": self.kappa,
            "IC": self.IC,
            "regime": self.regime,
        }


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def compute_pt_kernel(entity: PhaseTransitionEntity) -> PTKernelResult:
    """Compute kernel invariants for a phase transition entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_PT_CHANNELS) / N_PT_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return PTKernelResult(
        name=entity.name,
        category=entity.category,
        F=F,
        omega=omega,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_entities() -> list[PTKernelResult]:
    """Compute kernel for all phase transition entities."""
    return [compute_pt_kernel(e) for e in PT_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-PT-1 through T-PT-6
# ---------------------------------------------------------------------------


def verify_t_pt_1(results: list[PTKernelResult]) -> dict:
    """T-PT-1: Quantum transitions have highest mean IC.

    Quantum transitions (Mott, BCS, BEC) have the most homogeneous
    channel profiles — their order parameters, correlation lengths,
    and gap structures are all well-characterized, producing the
    highest multiplicative coherence.
    """
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.IC)
    q_ic = np.mean(cats["quantum"])
    other_ic = [np.mean(v) for k, v in cats.items() if k != "quantum"]
    passed = q_ic > max(other_ic)
    return {
        "name": "T-PT-1",
        "passed": bool(passed),
        "quantum_mean_IC": float(q_ic),
        "other_max_IC": float(max(other_ic)),
    }


def verify_t_pt_2(results: list[PTKernelResult]) -> dict:
    """T-PT-2: First-order transitions have highest mean curvature (channel spread)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.C)
    fo_c = np.mean(cats["first_order"])
    other_c = [np.mean(v) for k, v in cats.items() if k != "first_order"]
    passed = fo_c > max(other_c)
    return {
        "name": "T-PT-2",
        "passed": bool(passed),
        "first_order_mean_C": float(fo_c),
        "other_max_C": float(max(other_c)),
    }


def verify_t_pt_3(results: list[PTKernelResult]) -> dict:
    """T-PT-3: At least 2 distinct regimes present across all transitions."""
    regimes = {r.regime for r in results}
    passed = len(regimes) >= 2
    return {
        "name": "T-PT-3",
        "passed": bool(passed),
        "regimes_present": sorted(regimes),
        "count": len(regimes),
    }


def verify_t_pt_4(results: list[PTKernelResult]) -> dict:
    """T-PT-4: First-order transitions have highest mean heterogeneity gap Δ = F − IC."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F - r.IC)
    fo_delta = np.mean(cats["first_order"])
    other_delta = [np.mean(v) for k, v in cats.items() if k != "first_order"]
    passed = fo_delta > max(other_delta)
    return {
        "name": "T-PT-4",
        "passed": bool(passed),
        "first_order_mean_delta": float(fo_delta),
        "other_max_delta": float(max(other_delta)),
    }


def verify_t_pt_5(results: list[PTKernelResult]) -> dict:
    """T-PT-5: Topological transitions have lowest mean F (weakest order parameter)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F)
    topo_f = np.mean(cats["topological"])
    other_f = [np.mean(v) for k, v in cats.items() if k != "topological"]
    passed = topo_f < min(other_f)
    return {
        "name": "T-PT-5",
        "passed": bool(passed),
        "topological_mean_F": float(topo_f),
        "other_min_F": float(min(other_f)),
    }


def verify_t_pt_6(results: list[PTKernelResult]) -> dict:
    """T-PT-6: Ising ferromagnet has highest F among second-order transitions.

    The Ising ferromagnet is the most well-characterized second-order
    transition — its critical exponents, universality class, and order
    parameter are all precisely known, yielding the highest fidelity.
    """
    so = [r for r in results if r.category == "second_order"]
    ising = next(r for r in so if r.name == "Ising_ferromagnet")
    max_so_f = max(r.F for r in so)
    passed = abs(max_so_f - ising.F) < 1e-12
    return {
        "name": "T-PT-6",
        "passed": bool(passed),
        "Ising_ferromagnet_F": ising.F,
        "max_second_order_F": max_so_f,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-PT theorems."""
    results = compute_all_entities()
    return [
        verify_t_pt_1(results),
        verify_t_pt_2(results),
        verify_t_pt_3(results),
        verify_t_pt_4(results),
        verify_t_pt_5(results),
        verify_t_pt_6(results),
    ]


if __name__ == "__main__":
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")
