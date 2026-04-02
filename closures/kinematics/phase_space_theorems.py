"""Phase Space Return Theorems — Kinematics Domain.

Tier-2 closure mapping 12 dynamical systems through the GCD kernel.
Each entity is characterized by 8 phase-space trajectory channels.

Channels (8, equal weights w_i = 1/8):
  0  recurrence_rate      — fraction of trajectory revisiting initial neighborhood (1 = perfectly periodic)
  1  lyapunov_stability    — 1 − normalized max Lyapunov exponent (1 = no divergence)
  2  phase_volume_preserved — Liouville compliance (1 = Hamiltonian, volume exact)
  3  orbit_closure         — fraction of orbit segments that close (1 = all periodic)
  4  energy_surface_confinement — fraction of time on correct E surface (1 = exact)
  5  ergodic_coverage      — fraction of accessible phase space explored (1 = fully ergodic)
  6  attractor_dimension_low — 1 − D_attractor/D_phase_space (1 = point attractor)
  7  predictability_horizon — normalized horizon / reference period (1 = infinite predictability)

12 entities across 4 categories:
  Periodic (3):     Simple_pendulum, Kepler_orbit, Harmonic_lattice
  Quasiperiodic (3): Lissajous_3_4, Kolmogorov_torus, Asteroid_resonance
  Chaotic (3):      Double_pendulum, Lorenz_attractor, Three_body
  Dissipative (3):  Damped_oscillator, Van_der_Pol, Turbulent_wake

6 theorems (T-PS-1 through T-PS-6).
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

PS_CHANNELS = [
    "recurrence_rate",
    "lyapunov_stability",
    "phase_volume_preserved",
    "orbit_closure",
    "energy_surface_confinement",
    "ergodic_coverage",
    "attractor_dimension_low",
    "predictability_horizon",
]
N_PS_CHANNELS = len(PS_CHANNELS)


@dataclass(frozen=True, slots=True)
class PhaseSpaceEntity:
    """A dynamical system with 8 phase-space channels."""

    name: str
    category: str
    recurrence_rate: float
    lyapunov_stability: float
    phase_volume_preserved: float
    orbit_closure: float
    energy_surface_confinement: float
    ergodic_coverage: float
    attractor_dimension_low: float
    predictability_horizon: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.recurrence_rate,
                self.lyapunov_stability,
                self.phase_volume_preserved,
                self.orbit_closure,
                self.energy_surface_confinement,
                self.ergodic_coverage,
                self.attractor_dimension_low,
                self.predictability_horizon,
            ]
        )


PS_ENTITIES: tuple[PhaseSpaceEntity, ...] = (
    # Periodic — exactly recurrent
    PhaseSpaceEntity("Simple_pendulum", "periodic", 0.98, 0.99, 0.99, 0.98, 0.99, 0.30, 0.99, 0.98),
    PhaseSpaceEntity("Kepler_orbit", "periodic", 0.97, 0.98, 1.00, 0.97, 1.00, 0.25, 0.98, 0.97),
    PhaseSpaceEntity("Harmonic_lattice", "periodic", 0.95, 0.97, 0.99, 0.95, 0.98, 0.40, 0.97, 0.95),
    # Quasiperiodic — recurrent but never exactly closing
    PhaseSpaceEntity("Lissajous_3_4", "quasiperiodic", 0.85, 0.95, 0.98, 0.60, 0.97, 0.70, 0.90, 0.90),
    PhaseSpaceEntity("Kolmogorov_torus", "quasiperiodic", 0.80, 0.93, 0.99, 0.50, 0.98, 0.80, 0.85, 0.88),
    PhaseSpaceEntity("Asteroid_resonance", "quasiperiodic", 0.75, 0.90, 0.97, 0.55, 0.95, 0.65, 0.88, 0.82),
    # Chaotic — sensitive dependence, positive Lyapunov
    PhaseSpaceEntity("Double_pendulum", "chaotic", 0.20, 0.15, 0.95, 0.05, 0.90, 0.85, 0.40, 0.10),
    PhaseSpaceEntity("Lorenz_attractor", "chaotic", 0.30, 0.10, 0.90, 0.10, 0.85, 0.90, 0.30, 0.05),
    PhaseSpaceEntity("Three_body", "chaotic", 0.15, 0.05, 0.98, 0.03, 0.95, 0.80, 0.35, 0.03),
    # Dissipative — phase volume contracts
    PhaseSpaceEntity("Damped_oscillator", "dissipative", 0.70, 0.80, 0.30, 0.60, 0.70, 0.20, 0.95, 0.75),
    PhaseSpaceEntity("Van_der_Pol", "dissipative", 0.80, 0.75, 0.25, 0.70, 0.65, 0.30, 0.90, 0.70),
    PhaseSpaceEntity("Turbulent_wake", "dissipative", 0.10, 0.20, 0.10, 0.05, 0.30, 0.90, 0.15, 0.08),
)


@dataclass(frozen=True, slots=True)
class PSKernelResult:
    """Kernel output for a phase-space entity."""

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


def compute_ps_kernel(entity: PhaseSpaceEntity) -> PSKernelResult:
    """Compute kernel invariants for a phase-space entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_PS_CHANNELS) / N_PS_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return PSKernelResult(
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


def compute_all_entities() -> list[PSKernelResult]:
    """Compute kernel for all phase-space entities."""
    return [compute_ps_kernel(e) for e in PS_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-PS-1 through T-PS-6
# ---------------------------------------------------------------------------


def verify_t_ps_1(results: list[PSKernelResult]) -> dict:
    """T-PS-1: Periodic systems have highest mean F (most channels near 1)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F)
    periodic_f = float(np.mean(cats["periodic"]))
    other_max = max(float(np.mean(v)) for k, v in cats.items() if k != "periodic")
    passed = periodic_f > other_max
    return {
        "name": "T-PS-1",
        "passed": bool(passed),
        "periodic_mean_F": periodic_f,
        "other_max_F": other_max,
    }


def verify_t_ps_2(results: list[PSKernelResult]) -> dict:
    """T-PS-2: Chaotic systems have highest mean curvature (extreme channel heterogeneity)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.C)
    chaotic_c = float(np.mean(cats["chaotic"]))
    other_max = max(float(np.mean(v)) for k, v in cats.items() if k != "chaotic")
    passed = chaotic_c > other_max
    return {
        "name": "T-PS-2",
        "passed": bool(passed),
        "chaotic_mean_C": chaotic_c,
        "other_max_C": other_max,
    }


def verify_t_ps_3(results: list[PSKernelResult]) -> dict:
    """T-PS-3: At least 2 distinct regimes present."""
    regimes = {r.regime for r in results}
    passed = len(regimes) >= 2
    return {
        "name": "T-PS-3",
        "passed": bool(passed),
        "regimes": sorted(regimes),
        "count": len(regimes),
    }


def verify_t_ps_4(results: list[PSKernelResult]) -> dict:
    """T-PS-4: Turbulent_wake has lowest IC across all entities (worst multiplicative coherence)."""
    ic_vals = [(r.name, r.IC) for r in results]
    wake_ic = next(v for n, v in ic_vals if n == "Turbulent_wake")
    min_ic = min(v for _, v in ic_vals)
    passed = wake_ic <= min_ic + 1e-12
    return {
        "name": "T-PS-4",
        "passed": bool(passed),
        "Turbulent_wake_IC": float(wake_ic),
        "min_IC": float(min_ic),
    }


def verify_t_ps_5(results: list[PSKernelResult]) -> dict:
    """T-PS-5: Harmonic_lattice has highest IC/F (most uniform channels among periodic)."""
    periodic = [r for r in results if r.category == "periodic"]
    icf = [(r.name, r.IC / r.F if r.F > EPSILON else 0.0) for r in periodic]
    lattice_icf = next(v for n, v in icf if n == "Harmonic_lattice")
    max_icf = max(v for _, v in icf)
    passed = lattice_icf >= max_icf - 1e-12
    return {
        "name": "T-PS-5",
        "passed": bool(passed),
        "Harmonic_lattice_IC_F": float(lattice_icf),
        "max_periodic_IC_F": float(max_icf),
    }


def verify_t_ps_6(results: list[PSKernelResult]) -> dict:
    """T-PS-6: Periodic mean IC > chaotic mean IC (periodicity preserves multiplicative coherence)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.IC)
    periodic_ic = float(np.mean(cats["periodic"]))
    chaotic_ic = float(np.mean(cats["chaotic"]))
    passed = periodic_ic > chaotic_ic
    return {
        "name": "T-PS-6",
        "passed": bool(passed),
        "periodic_mean_IC": periodic_ic,
        "chaotic_mean_IC": chaotic_ic,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-PS theorems."""
    results = compute_all_entities()
    return [
        verify_t_ps_1(results),
        verify_t_ps_2(results),
        verify_t_ps_3(results),
        verify_t_ps_4(results),
        verify_t_ps_5(results),
        verify_t_ps_6(results),
    ]


if __name__ == "__main__":
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")
