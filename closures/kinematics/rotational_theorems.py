"""Rotational Kinematics Theorems — Kinematics Domain.

Tier-2 closure mapping 12 rotational systems through the GCD kernel.
Each entity is characterized by 8 rotational kinematics channels.

Channels (8, equal weights w_i = 1/8):
  0  angular_velocity_stability — 1 − σ(ω)/⟨ω⟩ (1 = constant angular velocity)
  1  torque_efficiency          — τ_applied / τ_total (1 = no parasitic torques)
  2  moment_of_inertia_uniformity — I_min / I_max over principal axes (1 = spherical)
  3  angular_acceleration_smoothness — 1 − jerk_angular / ref (1 = smooth)
  4  rotational_energy_fraction — T_rot / T_total (1 = purely rotational)
  5  centripetal_balance        — F_centripetal / F_net (1 = pure circular)
  6  bearing_friction_low       — 1 − friction_coeff / ref (1 = frictionless)
  7  angular_momentum_conservation — L_final / L_initial (1 = perfectly conserved)

12 entities across 4 categories:
  Uniform (3):      Pottery_wheel, Record_player, Ceiling_fan
  Accelerating (3): Drill_press, Wind_turbine_startup, Satellite_maneuver
  Decelerating (3): Brake_drum, Friction_wheel, Orbital_decay
  Coupled (3):      Gear_train, Belt_drive, Differential_axle

6 theorems (T-RT-1 through T-RT-6).
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

RT_CHANNELS = [
    "angular_velocity_stability",
    "torque_efficiency",
    "moment_of_inertia_uniformity",
    "angular_acceleration_smoothness",
    "rotational_energy_fraction",
    "centripetal_balance",
    "bearing_friction_low",
    "angular_momentum_conservation",
]
N_RT_CHANNELS = len(RT_CHANNELS)


@dataclass(frozen=True, slots=True)
class RotationalEntity:
    """A rotational system with 8 kinematics channels."""

    name: str
    category: str
    angular_velocity_stability: float
    torque_efficiency: float
    moment_of_inertia_uniformity: float
    angular_acceleration_smoothness: float
    rotational_energy_fraction: float
    centripetal_balance: float
    bearing_friction_low: float
    angular_momentum_conservation: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.angular_velocity_stability,
                self.torque_efficiency,
                self.moment_of_inertia_uniformity,
                self.angular_acceleration_smoothness,
                self.rotational_energy_fraction,
                self.centripetal_balance,
                self.bearing_friction_low,
                self.angular_momentum_conservation,
            ]
        )


RT_ENTITIES: tuple[RotationalEntity, ...] = (
    # Uniform rotation — steady state, high fidelity
    RotationalEntity("Pottery_wheel", "uniform", 0.95, 0.90, 0.85, 0.92, 0.95, 0.93, 0.80, 0.98),
    RotationalEntity("Record_player", "uniform", 0.99, 0.95, 0.90, 0.98, 0.90, 0.97, 0.92, 0.99),
    RotationalEntity("Ceiling_fan", "uniform", 0.93, 0.85, 0.80, 0.90, 0.88, 0.90, 0.75, 0.96),
    # Accelerating — torque-driven angular velocity increase
    RotationalEntity("Drill_press", "accelerating", 0.60, 0.80, 0.75, 0.50, 0.85, 0.70, 0.70, 0.90),
    RotationalEntity("Wind_turbine_startup", "accelerating", 0.40, 0.60, 0.70, 0.35, 0.75, 0.55, 0.80, 0.85),
    RotationalEntity("Satellite_maneuver", "accelerating", 0.50, 0.90, 0.65, 0.45, 0.70, 0.60, 0.95, 0.92),
    # Decelerating — friction/drag sapping angular momentum
    RotationalEntity("Brake_drum", "decelerating", 0.30, 0.40, 0.80, 0.25, 0.50, 0.75, 0.15, 0.40),
    RotationalEntity("Friction_wheel", "decelerating", 0.45, 0.50, 0.85, 0.35, 0.55, 0.80, 0.20, 0.50),
    RotationalEntity("Orbital_decay", "decelerating", 0.20, 0.30, 0.60, 0.15, 0.40, 0.50, 0.90, 0.30),
    # Coupled — multiple rotating components interacting
    RotationalEntity("Gear_train", "coupled", 0.85, 0.88, 0.70, 0.80, 0.80, 0.85, 0.65, 0.95),
    RotationalEntity("Belt_drive", "coupled", 0.75, 0.70, 0.65, 0.70, 0.75, 0.75, 0.55, 0.88),
    RotationalEntity("Differential_axle", "coupled", 0.80, 0.82, 0.55, 0.75, 0.78, 0.80, 0.60, 0.92),
)


@dataclass(frozen=True, slots=True)
class RTKernelResult:
    """Kernel output for a rotational entity."""

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


def compute_rt_kernel(entity: RotationalEntity) -> RTKernelResult:
    """Compute kernel invariants for a rotational entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_RT_CHANNELS) / N_RT_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return RTKernelResult(
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


def compute_all_entities() -> list[RTKernelResult]:
    """Compute kernel for all rotational entities."""
    return [compute_rt_kernel(e) for e in RT_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-RT-1 through T-RT-6
# ---------------------------------------------------------------------------


def verify_t_rt_1(results: list[RTKernelResult]) -> dict:
    """T-RT-1: Uniform rotation has highest mean F (all channels near maximum)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F)
    uniform_f = float(np.mean(cats["uniform"]))
    other_max = max(float(np.mean(v)) for k, v in cats.items() if k != "uniform")
    passed = uniform_f > other_max
    return {
        "name": "T-RT-1",
        "passed": bool(passed),
        "uniform_mean_F": uniform_f,
        "other_max_F": other_max,
    }


def verify_t_rt_2(results: list[RTKernelResult]) -> dict:
    """T-RT-2: Decelerating systems have highest mean curvature (friction channels low, others high)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.C)
    decel_c = float(np.mean(cats["decelerating"]))
    other_max = max(float(np.mean(v)) for k, v in cats.items() if k != "decelerating")
    passed = decel_c > other_max
    return {
        "name": "T-RT-2",
        "passed": bool(passed),
        "decelerating_mean_C": decel_c,
        "other_max_C": other_max,
    }


def verify_t_rt_3(results: list[RTKernelResult]) -> dict:
    """T-RT-3: At least 2 distinct regimes present."""
    regimes = {r.regime for r in results}
    passed = len(regimes) >= 2
    return {
        "name": "T-RT-3",
        "passed": bool(passed),
        "regimes": sorted(regimes),
        "count": len(regimes),
    }


def verify_t_rt_4(results: list[RTKernelResult]) -> dict:
    """T-RT-4: Record_player has highest IC/F among all entities (most uniform channels)."""
    icf = [(r.name, r.IC / r.F if r.F > EPSILON else 0.0) for r in results]
    rp_icf = next(v for n, v in icf if n == "Record_player")
    max_icf = max(v for _, v in icf)
    passed = rp_icf >= max_icf - 1e-12
    return {
        "name": "T-RT-4",
        "passed": bool(passed),
        "Record_player_IC_F": float(rp_icf),
        "max_IC_F": float(max_icf),
    }


def verify_t_rt_5(results: list[RTKernelResult]) -> dict:
    """T-RT-5: Orbital_decay has highest Δ (extreme channel spread)."""
    deltas = [(r.name, r.F - r.IC) for r in results]
    od_delta = next(d for n, d in deltas if n == "Orbital_decay")
    max_delta = max(d for _, d in deltas)
    passed = od_delta >= max_delta - 1e-12
    return {
        "name": "T-RT-5",
        "passed": bool(passed),
        "Orbital_decay_delta": float(od_delta),
        "max_delta": float(max_delta),
    }


def verify_t_rt_6(results: list[RTKernelResult]) -> dict:
    """T-RT-6: Uniform mean IC > decelerating mean IC (steady rotation preserves multiplicative coherence)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.IC)
    uniform_ic = float(np.mean(cats["uniform"]))
    decel_ic = float(np.mean(cats["decelerating"]))
    passed = uniform_ic > decel_ic
    return {
        "name": "T-RT-6",
        "passed": bool(passed),
        "uniform_mean_IC": uniform_ic,
        "decelerating_mean_IC": decel_ic,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-RT theorems."""
    results = compute_all_entities()
    return [
        verify_t_rt_1(results),
        verify_t_rt_2(results),
        verify_t_rt_3(results),
        verify_t_rt_4(results),
        verify_t_rt_5(results),
        verify_t_rt_6(results),
    ]


if __name__ == "__main__":
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")
