"""Elastic Moduli Theorems Closure — Materials Science Domain.

Tier-2 closure mapping 12 canonical elastic solids through the GCD kernel.
Each material is characterized by 8 channels from mechanical testing data.

Channels (8, equal weights w_i = 1/8):
  0  bulk_modulus_norm   — K/K_max, resistance to uniform compression (1 = stiffest)
  1  shear_modulus_norm  — G/G_max, resistance to shape change (1 = stiffest)
  2  youngs_modulus_norm — E/E_max, uniaxial stiffness (1 = stiffest)
  3  poisson_ratio_norm  — ν/0.5, incompressibility measure (1 = perfectly incompressible)
  4  hardness_norm       — H/H_max, resistance to indentation (1 = hardest)
  5  ductility           — elongation at break normalized (1 = most ductile)
  6  fracture_toughness  — K_IC/K_IC_max, resistance to crack propagation (1 = toughest)
  7  isotropy            — 1 - Zener anisotropy factor deviation (1 = perfectly isotropic)

12 entities across 4 categories:
  Metals (3):      tungsten, steel_1045, aluminum_6061
  Ceramics (3):    diamond, alumina, silicon_carbide
  Polymers (3):    HDPE, nylon_66, epoxy
  Composites (3):  CFRP, fiberglass, concrete

6 theorems (T-EL-1 through T-EL-6).
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

EL_CHANNELS = [
    "bulk_modulus_norm",
    "shear_modulus_norm",
    "youngs_modulus_norm",
    "poisson_ratio_norm",
    "hardness_norm",
    "ductility",
    "fracture_toughness",
    "isotropy",
]
N_EL_CHANNELS = len(EL_CHANNELS)


@dataclass(frozen=True, slots=True)
class ElasticEntity:
    """An elastic solid with 8 measurable channels."""

    name: str
    category: str
    bulk_modulus_norm: float
    shear_modulus_norm: float
    youngs_modulus_norm: float
    poisson_ratio_norm: float
    hardness_norm: float
    ductility: float
    fracture_toughness: float
    isotropy: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.bulk_modulus_norm,
                self.shear_modulus_norm,
                self.youngs_modulus_norm,
                self.poisson_ratio_norm,
                self.hardness_norm,
                self.ductility,
                self.fracture_toughness,
                self.isotropy,
            ]
        )


EL_ENTITIES: tuple[ElasticEntity, ...] = (
    # Metals — high ductility, moderate stiffness, isotropic
    ElasticEntity("tungsten", "metal", 0.75, 0.80, 0.75, 0.56, 0.70, 0.15, 0.45, 0.98),
    ElasticEntity("steel_1045", "metal", 0.50, 0.55, 0.52, 0.58, 0.45, 0.55, 0.65, 0.92),
    ElasticEntity("aluminum_6061", "metal", 0.30, 0.35, 0.32, 0.66, 0.25, 0.70, 0.55, 0.95),
    # Ceramics — very stiff, hard, brittle (low ductility + toughness)
    ElasticEntity("diamond", "ceramic", 0.95, 0.95, 0.95, 0.14, 1.00, 0.01, 0.10, 0.90),
    ElasticEntity("alumina", "ceramic", 0.65, 0.70, 0.68, 0.44, 0.85, 0.02, 0.15, 0.85),
    ElasticEntity("silicon_carbide", "ceramic", 0.70, 0.75, 0.72, 0.34, 0.90, 0.01, 0.12, 0.88),
    # Polymers — low stiffness, high ductility, variable toughness
    ElasticEntity("HDPE", "polymer", 0.02, 0.02, 0.02, 0.80, 0.03, 0.90, 0.30, 0.95),
    ElasticEntity("nylon_66", "polymer", 0.05, 0.05, 0.05, 0.70, 0.06, 0.80, 0.35, 0.90),
    ElasticEntity("epoxy", "polymer", 0.08, 0.08, 0.08, 0.68, 0.10, 0.10, 0.20, 0.92),
    # Composites — anisotropic, mixed properties
    ElasticEntity("CFRP", "composite", 0.20, 0.15, 0.55, 0.60, 0.40, 0.05, 0.50, 0.25),
    ElasticEntity("fiberglass", "composite", 0.15, 0.12, 0.30, 0.62, 0.20, 0.10, 0.40, 0.35),
    ElasticEntity("concrete", "composite", 0.25, 0.18, 0.20, 0.40, 0.30, 0.02, 0.08, 0.75),
)


@dataclass(frozen=True, slots=True)
class ELKernelResult:
    """Kernel output for an elastic solid entity."""

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


def compute_el_kernel(entity: ElasticEntity) -> ELKernelResult:
    """Compute kernel invariants for an elastic solid entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_EL_CHANNELS) / N_EL_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return ELKernelResult(
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


def compute_all_entities() -> list[ELKernelResult]:
    """Compute kernel for all elastic solid entities."""
    return [compute_el_kernel(e) for e in EL_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-EL-1 through T-EL-6
# ---------------------------------------------------------------------------


def verify_t_el_1(results: list[ELKernelResult]) -> dict:
    """T-EL-1: Metals have highest mean IC (most balanced channel profile)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.IC)
    metal_ic = np.mean(cats["metal"])
    other_ic = [np.mean(v) for k, v in cats.items() if k != "metal"]
    passed = metal_ic > max(other_ic)
    return {
        "name": "T-EL-1",
        "passed": bool(passed),
        "metal_mean_IC": float(metal_ic),
        "other_max_IC": float(max(other_ic)),
    }


def verify_t_el_2(results: list[ELKernelResult]) -> dict:
    """T-EL-2: Polymers have highest mean curvature.

    Polymers have extreme channel spread: high flexibility and damping
    coexist with near-zero hardness and stiffness channels, producing
    the highest curvature across all material categories.
    """
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.C)
    poly_c = np.mean(cats["polymer"])
    other_c = [np.mean(v) for k, v in cats.items() if k != "polymer"]
    passed = poly_c > max(other_c)
    return {
        "name": "T-EL-2",
        "passed": bool(passed),
        "polymer_mean_C": float(poly_c),
        "other_max_C": float(max(other_c)),
    }


def verify_t_el_3(results: list[ELKernelResult]) -> dict:
    """T-EL-3: All elastic materials are in Collapse regime.

    The 8-channel elastic trace captures such diverse properties
    (stiffness, hardness, ductility, damping, density, thermal,
    toughness, Poisson) that no material achieves homogeneity
    across all channels — all have ω ≥ 0.30.
    """
    all_collapse = all(r.regime == "Collapse" for r in results)
    regimes = {r.regime for r in results}
    passed = all_collapse
    return {
        "name": "T-EL-3",
        "passed": bool(passed),
        "all_collapse": all_collapse,
        "regimes_present": sorted(regimes),
    }


def verify_t_el_4(results: list[ELKernelResult]) -> dict:
    """T-EL-4: Ceramics have highest mean heterogeneity gap Δ = F − IC."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F - r.IC)
    cer_delta = np.mean(cats["ceramic"])
    other_delta = [np.mean(v) for k, v in cats.items() if k != "ceramic"]
    passed = cer_delta > max(other_delta)
    return {
        "name": "T-EL-4",
        "passed": bool(passed),
        "ceramic_mean_delta": float(cer_delta),
        "other_max_delta": float(max(other_delta)),
    }


def verify_t_el_5(results: list[ELKernelResult]) -> dict:
    """T-EL-5: Composites have lowest mean curvature.

    Composites achieve the most balanced channel profiles by combining
    properties from their constituent materials, yielding the lowest C.
    Engineering composites average out the extremes of their components.
    """
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.C)
    comp_c = np.mean(cats["composite"])
    other_c = [np.mean(v) for k, v in cats.items() if k != "composite"]
    passed = comp_c < min(other_c)
    return {
        "name": "T-EL-5",
        "passed": bool(passed),
        "composite_mean_C": float(comp_c),
        "other_min_C": float(min(other_c)),
    }


def verify_t_el_6(results: list[ELKernelResult]) -> dict:
    """T-EL-6: Diamond has highest F among ceramics (extreme stiffness channels)."""
    cers = [r for r in results if r.category == "ceramic"]
    dia = next(r for r in cers if r.name == "diamond")
    max_cer_f = max(r.F for r in cers)
    passed = max_cer_f - 1e-12 <= dia.F
    return {
        "name": "T-EL-6",
        "passed": bool(passed),
        "diamond_F": dia.F,
        "max_ceramic_F": max_cer_f,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-EL theorems."""
    results = compute_all_entities()
    return [
        verify_t_el_1(results),
        verify_t_el_2(results),
        verify_t_el_3(results),
        verify_t_el_4(results),
        verify_t_el_5(results),
        verify_t_el_6(results),
    ]


if __name__ == "__main__":
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")
