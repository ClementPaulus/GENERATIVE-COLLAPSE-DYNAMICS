"""Momentum Dynamics Theorems — Kinematics Domain.

Tier-2 closure mapping 12 collision/momentum scenarios through the GCD kernel.
Each entity is characterized by 8 momentum dynamics channels.

Channels (8, equal weights w_i = 1/8):
  0  momentum_conservation — p_final / p_initial (1 = perfectly conserved)
  1  energy_retention      — KE_final / KE_initial (1 = elastic)
  2  restitution_coeff     — coefficient of restitution (1 = elastic, 0 = inelastic)
  3  mass_symmetry         — min(m1,m2)/max(m1,m2) (1 = equal masses)
  4  velocity_transfer     — Δv_target / v_initial (1 = full transfer)
  5  impulse_regularity    — J_measured / J_predicted (1 = ideal)
  6  contact_duration_norm — 1 − t_contact / t_ref (1 = instantaneous)
  7  deformation_low       — 1 − permanent deformation fraction (1 = no damage)

12 entities across 4 categories:
  Elastic (3):     Newton_cradle, Billiard_break, Atomic_scattering
  Inelastic (3):   Car_crash, Clay_ball, Bullet_block
  Explosive (3):   Firecracker, Rocket_staging, Supernova_ejecta
  Constrained (3): Pendulum_swing, Rail_car_coupling, Ice_hockey_check

6 theorems (T-MD-1 through T-MD-6).
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

MD_CHANNELS = [
    "momentum_conservation",
    "energy_retention",
    "restitution_coeff",
    "mass_symmetry",
    "velocity_transfer",
    "impulse_regularity",
    "contact_duration_norm",
    "deformation_low",
]
N_MD_CHANNELS = len(MD_CHANNELS)


@dataclass(frozen=True, slots=True)
class MomentumEntity:
    """A collision/momentum scenario with 8 dynamics channels."""

    name: str
    category: str
    momentum_conservation: float
    energy_retention: float
    restitution_coeff: float
    mass_symmetry: float
    velocity_transfer: float
    impulse_regularity: float
    contact_duration_norm: float
    deformation_low: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.momentum_conservation,
                self.energy_retention,
                self.restitution_coeff,
                self.mass_symmetry,
                self.velocity_transfer,
                self.impulse_regularity,
                self.contact_duration_norm,
                self.deformation_low,
            ]
        )


MD_ENTITIES: tuple[MomentumEntity, ...] = (
    # Elastic collisions — momentum AND energy conserved
    MomentumEntity("Newton_cradle", "elastic", 0.99, 0.97, 0.98, 1.00, 0.95, 0.96, 0.90, 0.99),
    MomentumEntity("Billiard_break", "elastic", 0.98, 0.95, 0.96, 0.90, 0.85, 0.93, 0.88, 0.98),
    MomentumEntity("Atomic_scattering", "elastic", 1.00, 0.99, 1.00, 0.30, 0.70, 0.99, 0.99, 1.00),
    # Inelastic collisions — momentum conserved, energy lost
    MomentumEntity("Car_crash", "inelastic", 0.95, 0.10, 0.15, 0.70, 0.40, 0.80, 0.20, 0.05),
    MomentumEntity("Clay_ball", "inelastic", 0.98, 0.05, 0.02, 0.95, 0.50, 0.90, 0.15, 0.01),
    MomentumEntity("Bullet_block", "inelastic", 0.97, 0.08, 0.05, 0.02, 0.95, 0.85, 0.95, 0.10),
    # Explosive separations — internal energy adds KE
    MomentumEntity("Firecracker", "explosive", 0.90, 0.20, 0.10, 0.50, 0.30, 0.60, 0.95, 0.10),
    MomentumEntity("Rocket_staging", "explosive", 0.95, 0.30, 0.05, 0.20, 0.80, 0.85, 0.70, 0.60),
    MomentumEntity("Supernova_ejecta", "explosive", 0.80, 0.15, 0.01, 0.10, 0.90, 0.50, 0.99, 0.01),
    # Constrained — partial momentum transfer with constraints
    MomentumEntity("Pendulum_swing", "constrained", 0.95, 0.92, 0.90, 1.00, 0.88, 0.93, 0.85, 0.90),
    MomentumEntity("Rail_car_coupling", "constrained", 0.98, 0.40, 0.10, 0.80, 0.50, 0.95, 0.30, 0.70),
    MomentumEntity("Ice_hockey_check", "constrained", 0.90, 0.55, 0.50, 0.75, 0.60, 0.80, 0.70, 0.60),
)


@dataclass(frozen=True, slots=True)
class MDKernelResult:
    """Kernel output for a momentum entity."""

    name: str
    category: str
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    regime: str


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def compute_md_kernel(entity: MomentumEntity) -> MDKernelResult:
    """Compute kernel invariants for a momentum entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_MD_CHANNELS) / N_MD_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return MDKernelResult(
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


def compute_all_entities() -> list[MDKernelResult]:
    """Compute kernel for all momentum entities."""
    return [compute_md_kernel(e) for e in MD_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-MD-1 through T-MD-6
# ---------------------------------------------------------------------------


def verify_t_md_1(results: list[MDKernelResult]) -> dict:
    """T-MD-1: Elastic collisions have highest mean F (all channels preserved)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F)
    elastic_f = float(np.mean(cats["elastic"]))
    other_max = max(float(np.mean(v)) for k, v in cats.items() if k != "elastic")
    passed = elastic_f > other_max
    return {
        "name": "T-MD-1",
        "passed": bool(passed),
        "elastic_mean_F": elastic_f,
        "other_max_F": other_max,
    }


def verify_t_md_2(results: list[MDKernelResult]) -> dict:
    """T-MD-2: Inelastic collisions have highest curvature (channel heterogeneity)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.C)
    inelastic_c = float(np.mean(cats["inelastic"]))
    other_max = max(float(np.mean(v)) for k, v in cats.items() if k != "inelastic")
    passed = inelastic_c > other_max
    return {
        "name": "T-MD-2",
        "passed": bool(passed),
        "inelastic_mean_C": inelastic_c,
        "other_max_C": other_max,
    }


def verify_t_md_3(results: list[MDKernelResult]) -> dict:
    """T-MD-3: At least 2 distinct regimes present across all entities."""
    regimes = {r.regime for r in results}
    passed = len(regimes) >= 2
    return {
        "name": "T-MD-3",
        "passed": bool(passed),
        "regimes": sorted(regimes),
        "count": len(regimes),
    }


def verify_t_md_4(results: list[MDKernelResult]) -> dict:
    """T-MD-4: Newton_cradle has highest IC/F among all entities (most uniform channels)."""
    icf = [(r.name, r.IC / r.F if r.F > EPSILON else 0.0) for r in results]
    cradle_icf = next(v for n, v in icf if n == "Newton_cradle")
    max_icf = max(v for _, v in icf)
    passed = cradle_icf >= max_icf - 1e-12
    return {
        "name": "T-MD-4",
        "passed": bool(passed),
        "Newton_cradle_IC_F": float(cradle_icf),
        "max_IC_F": float(max_icf),
    }


def verify_t_md_5(results: list[MDKernelResult]) -> dict:
    """T-MD-5: Clay_ball has highest Δ (perfectly inelastic → extreme heterogeneity gap)."""
    deltas = [(r.name, r.F - r.IC) for r in results]
    clay_delta = next(d for n, d in deltas if n == "Clay_ball")
    max_delta = max(d for _, d in deltas)
    passed = clay_delta >= max_delta - 1e-12
    return {
        "name": "T-MD-5",
        "passed": bool(passed),
        "Clay_ball_delta": float(clay_delta),
        "max_delta": float(max_delta),
    }


def verify_t_md_6(results: list[MDKernelResult]) -> dict:
    """T-MD-6: Elastic mean IC > inelastic mean IC (elastic preserves multiplicative coherence)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.IC)
    elastic_ic = float(np.mean(cats["elastic"]))
    inelastic_ic = float(np.mean(cats["inelastic"]))
    passed = elastic_ic > inelastic_ic
    return {
        "name": "T-MD-6",
        "passed": bool(passed),
        "elastic_mean_IC": elastic_ic,
        "inelastic_mean_IC": inelastic_ic,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-MD theorems."""
    results = compute_all_entities()
    return [
        verify_t_md_1(results),
        verify_t_md_2(results),
        verify_t_md_3(results),
        verify_t_md_4(results),
        verify_t_md_5(results),
        verify_t_md_6(results),
    ]


if __name__ == "__main__":
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")
