"""Fluid Dynamics Closure — Everyday Physics Domain.

Tier-2 closure mapping 12 canonical flow regimes through the GCD kernel.
Each regime is characterized by 8 channels drawn from fluid mechanics.

Channels (8, equal weights w_i = 1/8):
  0  reynolds_stability     — 1/(1+Re/Re_crit), fully stable → 1.0
  1  mach_subsonic          — 1 - M for sub/transonic, ε for supersonic
  2  viscous_dominance      — viscous/inertial ratio (1 = viscous-dominated)
  3  pressure_regularity    — smoothness of pressure field (1 = uniform)
  4  turbulence_suppression — 1 - Tu intensity (1 = laminar)
  5  energy_conservation    — fraction not dissipated to heat (1 = dissipation-free)
  6  mixing_coherence       — mixing uniformity (1 = no mixing or perfect mixing)
  7  boundary_integrity     — no-slip boundary layer coherence (1 = attached)

12 entities across 4 categories:
  Laminar (3):  pipe_laminar, Couette_flow, Stokes_creeping
  Transitional (3): pipe_transition, boundary_layer_transition, Rayleigh_Benard
  Turbulent (3): pipe_turbulent, jet_turbulent, wake_turbulent
  Compressible (3): subsonic_nozzle, transonic_airfoil, supersonic_cone

6 theorems (T-FD-1 through T-FD-6).
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

FD_CHANNELS = [
    "reynolds_stability",
    "mach_subsonic",
    "viscous_dominance",
    "pressure_regularity",
    "turbulence_suppression",
    "energy_conservation",
    "mixing_coherence",
    "boundary_integrity",
]
N_FD_CHANNELS = len(FD_CHANNELS)


@dataclass(frozen=True, slots=True)
class FlowRegimeEntity:
    """A canonical flow regime with 8 measurable channels."""

    name: str
    category: str
    reynolds_stability: float
    mach_subsonic: float
    viscous_dominance: float
    pressure_regularity: float
    turbulence_suppression: float
    energy_conservation: float
    mixing_coherence: float
    boundary_integrity: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.reynolds_stability,
                self.mach_subsonic,
                self.viscous_dominance,
                self.pressure_regularity,
                self.turbulence_suppression,
                self.energy_conservation,
                self.mixing_coherence,
                self.boundary_integrity,
            ]
        )


FD_ENTITIES: tuple[FlowRegimeEntity, ...] = (
    # Laminar — high fidelity, low drift
    FlowRegimeEntity("pipe_laminar", "laminar", 0.95, 0.99, 0.90, 0.95, 0.98, 0.92, 0.95, 0.98),
    FlowRegimeEntity("Couette_flow", "laminar", 0.92, 0.99, 0.88, 0.90, 0.96, 0.90, 0.90, 0.95),
    FlowRegimeEntity("Stokes_creeping", "laminar", 0.99, 0.99, 0.99, 0.98, 0.99, 0.95, 0.98, 0.99),
    # Transitional — intermediate region
    FlowRegimeEntity("pipe_transition", "transitional", 0.50, 0.98, 0.40, 0.55, 0.50, 0.70, 0.45, 0.65),
    FlowRegimeEntity("boundary_layer_transition", "transitional", 0.45, 0.95, 0.35, 0.50, 0.45, 0.65, 0.40, 0.55),
    FlowRegimeEntity("Rayleigh_Benard", "transitional", 0.55, 0.99, 0.50, 0.60, 0.55, 0.75, 0.50, 0.70),
    # Turbulent — low fidelity, high drift
    FlowRegimeEntity("pipe_turbulent", "turbulent", 0.10, 0.98, 0.08, 0.25, 0.10, 0.40, 0.30, 0.45),
    FlowRegimeEntity("jet_turbulent", "turbulent", 0.05, 0.95, 0.05, 0.15, 0.05, 0.30, 0.20, 0.25),
    FlowRegimeEntity("wake_turbulent", "turbulent", 0.08, 0.96, 0.07, 0.20, 0.08, 0.35, 0.15, 0.10),
    # Compressible
    FlowRegimeEntity("subsonic_nozzle", "compressible", 0.70, 0.80, 0.25, 0.75, 0.65, 0.80, 0.70, 0.85),
    FlowRegimeEntity("transonic_airfoil", "compressible", 0.30, 0.15, 0.15, 0.25, 0.30, 0.50, 0.35, 0.55),
    FlowRegimeEntity("supersonic_cone", "compressible", 0.20, 0.05, 0.10, 0.40, 0.25, 0.45, 0.50, 0.60),
)


@dataclass(frozen=True, slots=True)
class FDKernelResult:
    """Kernel output for a flow regime entity."""

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


def compute_fd_kernel(entity: FlowRegimeEntity) -> FDKernelResult:
    """Compute GCD kernel for a flow regime entity."""
    c = entity.trace_vector()
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(N_FD_CHANNELS) / N_FD_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C_val = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    if omega >= 0.30:
        regime = "Collapse"
    elif omega < 0.038 and F > 0.90 and S < 0.15 and C_val < 0.14:
        regime = "Stable"
    else:
        regime = "Watch"
    return FDKernelResult(
        name=entity.name,
        category=entity.category,
        F=F,
        omega=omega,
        S=S,
        C=C_val,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_entities() -> list[FDKernelResult]:
    """Compute kernel outputs for all flow regime entities."""
    return [compute_fd_kernel(e) for e in FD_ENTITIES]


# ── Theorems ──────────────────────────────────────────────────────────


def verify_t_fd_1(results: list[FDKernelResult]) -> dict:
    """T-FD-1: Stokes creeping flow has highest F — viscous dominance
    maximizes all coherence channels simultaneously.
    """
    stokes = next(r for r in results if r.name == "Stokes_creeping")
    max_F = max(r.F for r in results)
    passed = abs(stokes.F - max_F) < 0.01
    return {"name": "T-FD-1", "passed": bool(passed), "stokes_F": stokes.F, "max_F": float(max_F)}


def verify_t_fd_2(results: list[FDKernelResult]) -> dict:
    """T-FD-2: No laminar flow reaches Collapse; all turbulent flows
    are in Collapse regime. Laminar viscous dominance prevents ω ≥ 0.30.
    """
    lam = [r for r in results if r.category == "laminar"]
    turb = [r for r in results if r.category == "turbulent"]
    no_lam_collapse = all(r.regime != "Collapse" for r in lam)
    all_turb_collapse = all(r.regime == "Collapse" for r in turb)
    passed = no_lam_collapse and all_turb_collapse
    return {
        "name": "T-FD-2",
        "passed": bool(passed),
        "laminar_regimes": [r.regime for r in lam],
        "turbulent_regimes": [r.regime for r in turb],
    }


def verify_t_fd_3(results: list[FDKernelResult]) -> dict:
    """T-FD-3: Turbulent jet has lowest IC/F — geometric slaughter
    from multiple near-zero channels (Re, Tu, viscous, boundary).
    """
    jet = next(r for r in results if r.name == "jet_turbulent")
    min_icf = min(r.IC / r.F for r in results if r.F > EPSILON)
    jet_icf = jet.IC / jet.F if jet.F > EPSILON else 0.0
    passed = abs(jet_icf - min_icf) < 0.05
    return {"name": "T-FD-3", "passed": bool(passed), "jet_IC_F": float(jet_icf), "min_IC_F": float(min_icf)}


def verify_t_fd_4(results: list[FDKernelResult]) -> dict:
    """T-FD-4: Laminar mean F > transitional mean F > turbulent mean F.

    Reynolds number increase monotonically degrades kernel fidelity.
    """
    lam_f = np.mean([r.F for r in results if r.category == "laminar"])
    trans_f = np.mean([r.F for r in results if r.category == "transitional"])
    turb_f = np.mean([r.F for r in results if r.category == "turbulent"])
    passed = lam_f > trans_f > turb_f
    return {
        "name": "T-FD-4",
        "passed": bool(passed),
        "laminar_F": float(lam_f),
        "transitional_F": float(trans_f),
        "turbulent_F": float(turb_f),
    }


def verify_t_fd_5(results: list[FDKernelResult]) -> dict:
    """T-FD-5: Transonic airfoil has lower F than subsonic nozzle —
    shock-induced compressibility kills Mach subsonic channel.
    """
    transonic = next(r for r in results if r.name == "transonic_airfoil")
    subsonic = next(r for r in results if r.name == "subsonic_nozzle")
    passed = transonic.F < subsonic.F
    return {"name": "T-FD-5", "passed": bool(passed), "transonic_F": transonic.F, "subsonic_F": subsonic.F}


def verify_t_fd_6(results: list[FDKernelResult]) -> dict:
    """T-FD-6: Compressible flows have larger heterogeneity gap (Δ)
    than laminar flows — Mach effects create channel imbalance.
    """
    comp_gap = np.mean([r.F - r.IC for r in results if r.category == "compressible"])
    lam_gap = np.mean([r.F - r.IC for r in results if r.category == "laminar"])
    passed = comp_gap > lam_gap
    return {
        "name": "T-FD-6",
        "passed": bool(passed),
        "compressible_gap": float(comp_gap),
        "laminar_gap": float(lam_gap),
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-FD theorems."""
    results = compute_all_entities()
    return [
        verify_t_fd_1(results),
        verify_t_fd_2(results),
        verify_t_fd_3(results),
        verify_t_fd_4(results),
        verify_t_fd_5(results),
        verify_t_fd_6(results),
    ]


def main() -> None:
    """Entry point."""
    results = compute_all_entities()
    print("=" * 78)
    print("FLUID DYNAMICS — GCD KERNEL ANALYSIS")
    print("=" * 78)
    print(f"{'Entity':<32} {'Cat':<14} {'F':>6} {'ω':>6} {'IC':>6} {'Δ':>6} {'Regime'}")
    print("-" * 78)
    for r in results:
        gap = r.F - r.IC
        print(f"{r.name:<32} {r.category:<14} {r.F:6.3f} {r.omega:6.3f} {r.IC:6.3f} {gap:6.3f} {r.regime}")

    print("\n── Theorems ──")
    for t in verify_all_theorems():
        print(f"  {t['name']}: {'PROVEN' if t['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
