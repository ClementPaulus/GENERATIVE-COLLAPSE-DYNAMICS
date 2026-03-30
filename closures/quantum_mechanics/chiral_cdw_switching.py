"""Chiral CDW switching closure for quantum_mechanics domain.

Motivated by Qiu et al. 2026 (arXiv:2603.22921): Synergistic chemical
and optical switching of chiral symmetry breaking in 1T-TaS₂. GCD
translation: pristine locked-chirality states → Watch (high F, IC near F);
Ti-substituted coexisting domains → progressive Collapse (geometric slaughter
via domain_wall_suppression channel); peak optical transient → deep Collapse
(IC near Critical); post-pump new-chirality state → Watch (finite τ_R,
return from Collapse demonstrated).

Channels (8):
  chiral_order_fidelity      — CDW chiral order parameter magnitude
  domain_purity              — fraction in dominant-chirality domain
  phonon_mode_coherence      — coherence of the 2 THz CDW phonon mode
  optical_switching_fidelity — fidelity of chirality-switching operation
  domain_wall_suppression    — 1 − domain-wall density (high = few walls)
  lattice_fidelity           — 1 − Ti-substitution damage (high = pristine)
  pump_selectivity           — selectivity of optical chirality writing
  achiral_suppression        — fraction NOT in achiral configuration

Entities (12):
  pristine (4)     — locked L/R chirality at 100 K and 200 K (Watch)
  ti_doped (4)     — Ti-substituted x = 0.02, 0.05, 0.10, 0.20 (Watch→Collapse)
  optical_pump (4) — pre-pump, peak transient, 100 ps return, locked new

Theorems (7): T-CCS-1 through T-CCS-7
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ws = str(Path(__file__).resolve().parents[2])
if _ws not in sys.path:
    sys.path.insert(0, _ws)

from umcp.frozen_contract import EPSILON  # noqa: E402
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

# ---------------------------------------------------------------------------
# Channel specification
# ---------------------------------------------------------------------------

N_CCS_CHANNELS: int = 8
CCS_CHANNELS: list[str] = [
    "chiral_order_fidelity",
    "domain_purity",
    "phonon_mode_coherence",
    "optical_switching_fidelity",
    "domain_wall_suppression",
    "lattice_fidelity",
    "pump_selectivity",
    "achiral_suppression",
]


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CCSEntity:
    """Single entity in the chiral CDW switching catalog."""

    name: str
    material: str
    category: str  # pristine | ti_doped | optical_pump
    doping_level: float  # Ti substitution fraction x
    temperature_K: float

    # 8 channels (all in [0, 1] before ε-clipping)
    chiral_order_fidelity: float
    domain_purity: float
    phonon_mode_coherence: float
    optical_switching_fidelity: float
    domain_wall_suppression: float
    lattice_fidelity: float
    pump_selectivity: float
    achiral_suppression: float

    def trace_vector(self) -> np.ndarray:
        """Return ε-clipped 8-channel trace vector."""
        c = np.array(
            [
                self.chiral_order_fidelity,
                self.domain_purity,
                self.phonon_mode_coherence,
                self.optical_switching_fidelity,
                self.domain_wall_suppression,
                self.lattice_fidelity,
                self.pump_selectivity,
                self.achiral_suppression,
            ],
            dtype=float,
        )
        return np.clip(c, EPSILON, 1.0 - EPSILON)


# ---------------------------------------------------------------------------
# Entity catalog (12 entities × 3 categories)
# ---------------------------------------------------------------------------

CCS_ENTITIES: list[CCSEntity] = [
    # ------------------------------------------------------------------
    # Pristine 1T-TaS₂: locked chirality at 100 K (deep in CDW phase)
    # ------------------------------------------------------------------
    CCSEntity(
        name="Pristine_100K_L",
        material="TaS2",
        category="pristine",
        doping_level=0.0,
        temperature_K=100.0,
        chiral_order_fidelity=0.95,
        domain_purity=0.97,
        phonon_mode_coherence=0.93,
        optical_switching_fidelity=0.92,
        domain_wall_suppression=0.98,
        lattice_fidelity=0.99,
        pump_selectivity=0.90,
        achiral_suppression=0.97,
    ),
    CCSEntity(
        name="Pristine_100K_R",
        material="TaS2",
        category="pristine",
        doping_level=0.0,
        temperature_K=100.0,
        chiral_order_fidelity=0.94,
        domain_purity=0.97,
        phonon_mode_coherence=0.93,
        optical_switching_fidelity=0.91,
        domain_wall_suppression=0.98,
        lattice_fidelity=0.99,
        pump_selectivity=0.91,
        achiral_suppression=0.96,
    ),
    # ------------------------------------------------------------------
    # Pristine 1T-TaS₂: near-CCDW boundary at 200 K (phonon softening)
    # ------------------------------------------------------------------
    CCSEntity(
        name="Pristine_200K_L",
        material="TaS2",
        category="pristine",
        doping_level=0.0,
        temperature_K=200.0,
        chiral_order_fidelity=0.82,
        domain_purity=0.86,
        phonon_mode_coherence=0.74,
        optical_switching_fidelity=0.78,
        domain_wall_suppression=0.89,
        lattice_fidelity=0.99,
        pump_selectivity=0.84,
        achiral_suppression=0.83,
    ),
    CCSEntity(
        name="Pristine_200K_R",
        material="TaS2",
        category="pristine",
        doping_level=0.0,
        temperature_K=200.0,
        chiral_order_fidelity=0.81,
        domain_purity=0.85,
        phonon_mode_coherence=0.73,
        optical_switching_fidelity=0.77,
        domain_wall_suppression=0.88,
        lattice_fidelity=0.99,
        pump_selectivity=0.83,
        achiral_suppression=0.82,
    ),
    # ------------------------------------------------------------------
    # Ti-substituted TixTa₁₋ₓS₂: coexisting chiral domains
    # ------------------------------------------------------------------
    CCSEntity(
        name="TiTaS2_x002",
        material="TiTaS2",
        category="ti_doped",
        doping_level=0.02,
        temperature_K=100.0,
        chiral_order_fidelity=0.88,
        domain_purity=0.82,
        phonon_mode_coherence=0.85,
        optical_switching_fidelity=0.80,
        domain_wall_suppression=0.75,
        lattice_fidelity=0.90,
        pump_selectivity=0.86,
        achiral_suppression=0.87,
    ),
    CCSEntity(
        name="TiTaS2_x005",
        material="TiTaS2",
        category="ti_doped",
        doping_level=0.05,
        temperature_K=100.0,
        chiral_order_fidelity=0.72,
        domain_purity=0.65,
        phonon_mode_coherence=0.70,
        optical_switching_fidelity=0.68,
        domain_wall_suppression=0.52,
        lattice_fidelity=0.75,
        pump_selectivity=0.72,
        achiral_suppression=0.68,
    ),
    CCSEntity(
        name="TiTaS2_x010",
        material="TiTaS2",
        category="ti_doped",
        doping_level=0.10,
        temperature_K=100.0,
        chiral_order_fidelity=0.55,
        domain_purity=0.50,
        phonon_mode_coherence=0.55,
        optical_switching_fidelity=0.52,
        domain_wall_suppression=0.35,
        lattice_fidelity=0.50,
        pump_selectivity=0.55,
        achiral_suppression=0.52,
    ),
    CCSEntity(
        name="TiTaS2_x020",
        material="TiTaS2",
        category="ti_doped",
        doping_level=0.20,
        temperature_K=100.0,
        chiral_order_fidelity=0.35,
        domain_purity=0.30,
        phonon_mode_coherence=0.40,
        optical_switching_fidelity=0.38,
        domain_wall_suppression=0.22,
        lattice_fidelity=0.01,
        pump_selectivity=0.38,
        achiral_suppression=0.28,
    ),
    # ------------------------------------------------------------------
    # Optical pump dynamics: TiTaS₂ x=0.05, femtosecond excitation
    # ------------------------------------------------------------------
    CCSEntity(
        name="Pump_pre_x005",
        material="TiTaS2",
        category="optical_pump",
        doping_level=0.05,
        temperature_K=100.0,
        chiral_order_fidelity=0.75,
        domain_purity=0.68,
        phonon_mode_coherence=0.73,
        optical_switching_fidelity=0.70,
        domain_wall_suppression=0.55,
        lattice_fidelity=0.75,
        pump_selectivity=0.85,
        achiral_suppression=0.70,
    ),
    CCSEntity(
        name="Pump_peak_transient",
        material="TiTaS2",
        category="optical_pump",
        doping_level=0.05,
        temperature_K=100.0,
        # Peak ~0.5 ps: 2 THz phonon active (phonon_mode_coherence high),
        # domain walls at maximum (domain_wall_suppression near zero →
        # geometric slaughter of IC despite other channels being moderate)
        chiral_order_fidelity=0.22,
        domain_purity=0.28,
        phonon_mode_coherence=0.88,
        optical_switching_fidelity=0.35,
        domain_wall_suppression=0.03,
        lattice_fidelity=0.75,
        pump_selectivity=0.30,
        achiral_suppression=0.40,
    ),
    CCSEntity(
        name="Pump_100ps_return",
        material="TiTaS2",
        category="optical_pump",
        doping_level=0.05,
        temperature_K=100.0,
        chiral_order_fidelity=0.65,
        domain_purity=0.70,
        phonon_mode_coherence=0.62,
        optical_switching_fidelity=0.72,
        domain_wall_suppression=0.68,
        lattice_fidelity=0.75,
        pump_selectivity=0.88,
        achiral_suppression=0.70,
    ),
    CCSEntity(
        name="Pump_locked_new",
        material="TiTaS2",
        category="optical_pump",
        doping_level=0.05,
        temperature_K=100.0,
        chiral_order_fidelity=0.80,
        domain_purity=0.82,
        phonon_mode_coherence=0.78,
        optical_switching_fidelity=0.85,
        domain_wall_suppression=0.82,
        lattice_fidelity=0.75,
        pump_selectivity=0.90,
        achiral_suppression=0.83,
    ),
]


# ---------------------------------------------------------------------------
# Kernel result dataclass and computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CCSKernelResult:
    """Kernel outputs for one chiral CDW entity."""

    name: str
    material: str
    category: str
    doping_level: float
    temperature_K: float
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    regime: str


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Apply frozen four-gate criterion (consistent across the seam)."""
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def compute_ccs_kernel(entity: CCSEntity) -> CCSKernelResult:
    """Compute GCD kernel invariants for one chiral CDW entity."""
    c = entity.trace_vector()
    w = np.full(N_CCS_CHANNELS, 1.0 / N_CCS_CHANNELS, dtype=np.float64)
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return CCSKernelResult(
        name=entity.name,
        material=entity.material,
        category=entity.category,
        doping_level=entity.doping_level,
        temperature_K=entity.temperature_K,
        F=F,
        omega=omega,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_entities() -> list[CCSKernelResult]:
    """Compute kernel results for all 12 entities."""
    return [compute_ccs_kernel(e) for e in CCS_ENTITIES]


# ---------------------------------------------------------------------------
# Theorem verification functions (T-CCS-1 through T-CCS-7)
# ---------------------------------------------------------------------------


def verify_t_ccs_1(results: list[CCSKernelResult]) -> dict:
    """T-CCS-1: Tier-1 universality — duality, integrity bound, log-integrity."""
    errors: list[str] = []
    for r in results:
        if abs(r.F + r.omega - 1.0) >= 1e-12:
            errors.append(f"{r.name}: |F+ω-1|={abs(r.F + r.omega - 1.0):.2e}")
        if r.IC > r.F + 1e-12:
            errors.append(f"{r.name}: IC={r.IC:.6f} > F={r.F:.6f}")
        if abs(r.IC - math.exp(r.kappa)) >= 1e-10:
            errors.append(f"{r.name}: |IC-exp(κ)|={abs(r.IC - math.exp(r.kappa)):.2e}")
    return {
        "name": "T-CCS-1",
        "description": "Tier-1 universality: duality + integrity bound + log-integrity",
        "passed": len(errors) == 0,
        "errors": errors,
    }


def verify_t_ccs_2(results: list[CCSKernelResult]) -> dict:
    """T-CCS-2: Locked-chirality fidelity — pristine entities have F ≥ 0.80."""
    pristine = [r for r in results if r.category == "pristine"]
    failing = [r for r in pristine if r.F < 0.80]
    return {
        "name": "T-CCS-2",
        "description": "Locked-chirality fidelity: pristine entities have F ≥ 0.80",
        "passed": len(failing) == 0,
        "pristine_F_values": {r.name: round(r.F, 4) for r in pristine},
        "failing": [r.name for r in failing],
    }


def verify_t_ccs_3(results: list[CCSKernelResult]) -> dict:
    """T-CCS-3: Doping-induced drift — mean ω(ti_doped) > mean ω(pristine)."""
    pristine = [r for r in results if r.category == "pristine"]
    ti_doped = [r for r in results if r.category == "ti_doped"]
    mean_omega_pristine = sum(r.omega for r in pristine) / len(pristine)
    mean_omega_doped = sum(r.omega for r in ti_doped) / len(ti_doped)
    passed = mean_omega_doped > mean_omega_pristine
    return {
        "name": "T-CCS-3",
        "description": "Doping-induced drift: mean ω(ti_doped) > mean ω(pristine)",
        "passed": passed,
        "mean_omega_pristine": round(mean_omega_pristine, 4),
        "mean_omega_doped": round(mean_omega_doped, 4),
    }


def verify_t_ccs_4(results: list[CCSKernelResult]) -> dict:
    """T-CCS-4: Geometric slaughter — peak transient has lowest IC in optical_pump."""
    pump = [r for r in results if r.category == "optical_pump"]
    transient = next((r for r in pump if r.name == "Pump_peak_transient"), None)
    if transient is None:
        return {"name": "T-CCS-4", "passed": False, "error": "Pump_peak_transient not found"}
    min_ic = min(r.IC for r in pump)
    passed = abs(transient.IC - min_ic) < 1e-12
    return {
        "name": "T-CCS-4",
        "description": "Geometric slaughter: Pump_peak_transient has lowest IC in optical_pump",
        "passed": passed,
        "transient_IC": round(transient.IC, 6),
        "pump_IC_values": {r.name: round(r.IC, 6) for r in pump},
    }


def verify_t_ccs_5(results: list[CCSKernelResult]) -> dict:
    """T-CCS-5: Doping-monotone drift — ω increases monotonically with Ti doping."""
    doped = sorted(
        [r for r in results if r.category == "ti_doped"],
        key=lambda r: r.doping_level,
    )
    monotone = all(doped[i].omega < doped[i + 1].omega for i in range(len(doped) - 1))
    return {
        "name": "T-CCS-5",
        "description": "Doping-monotone drift: ω increases monotonically with Ti doping",
        "passed": monotone,
        "doping_omega_pairs": [(round(r.doping_level, 3), round(r.omega, 4)) for r in doped],
    }


def verify_t_ccs_6(results: list[CCSKernelResult]) -> dict:
    """T-CCS-6: Return from collapse — post-pump omega < peak-transient omega."""
    pump = {r.name: r for r in results if r.category == "optical_pump"}
    transient = pump.get("Pump_peak_transient")
    post = pump.get("Pump_locked_new")
    if transient is None or post is None:
        return {"name": "T-CCS-6", "passed": False, "error": "required entities not found"}
    passed = post.omega < transient.omega
    return {
        "name": "T-CCS-6",
        "description": "Return from collapse: ω(Pump_locked_new) < ω(Pump_peak_transient)",
        "passed": passed,
        "omega_transient": round(transient.omega, 4),
        "omega_post_locked": round(post.omega, 4),
    }


def verify_t_ccs_7(results: list[CCSKernelResult]) -> dict:
    """T-CCS-7: Chirality-IC ordering — IC decreases monotonically with Ti doping."""
    doped = sorted(
        [r for r in results if r.category == "ti_doped"],
        key=lambda r: r.doping_level,
    )
    monotone = all(doped[i].IC > doped[i + 1].IC for i in range(len(doped) - 1))
    return {
        "name": "T-CCS-7",
        "description": "Chirality-IC ordering: IC decreases monotonically with Ti doping",
        "passed": monotone,
        "doping_IC_pairs": [(round(r.doping_level, 3), round(r.IC, 4)) for r in doped],
    }


def verify_all_theorems() -> list[dict]:
    """Verify all 7 theorems and return a list of result dicts."""
    results = compute_all_entities()
    return [
        verify_t_ccs_1(results),
        verify_t_ccs_2(results),
        verify_t_ccs_3(results),
        verify_t_ccs_4(results),
        verify_t_ccs_5(results),
        verify_t_ccs_6(results),
        verify_t_ccs_7(results),
    ]


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Chiral CDW Switching — Qiu et al. 2026 (arXiv:2603.22921) ===\n")
    results = compute_all_entities()
    print(f"{'Entity':<28} {'Category':<14} {'F':>6} {'ω':>6} {'IC':>6} {'Regime'}")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<28} {r.category:<14} {r.F:.4f} {r.omega:.4f} {r.IC:.4f} {r.regime}")
    print("\n=== Theorems ===")
    for t in verify_all_theorems():
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  {t['name']}: {status} — {t['description']}")
