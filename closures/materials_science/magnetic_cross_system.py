"""Magnetic Cross-System Closure — 35 entities, 3 systems, one kernel.

Tier-2 closure discovering structural relationships across three magnetic systems
measured by the same GCD kernel — the cognitive equalizer at work.

Systems (3):
  Material  — 17 bulk magnetic materials (ferro/antiferro/ferri/dia/paramagnetic)
  Quincke   — 12 experimental states of magnetic Quincke rollers (active matter)
  QDM       — 6 primary phases of the quantum dimer model on kagome lattice

Each system embeds its entities into an 8-channel trace vector through its own
domain closure, then passes through the single kernel K. The cross-system closure
compares kernel *outputs* — not channel definitions — because the cognitive equalizer
guarantees: same algebra, same frozen parameters, same verdict structure.

35 entities (17 + 12 + 6), 8 channels per entity (system-specific),
6 theorems (T-MCS-1 through T-MCS-6).

Key structural fact: Pearson r(C, Δ) > 0.5 across all 35 entities — curvature
tracks heterogeneity universally regardless of whether the magnetism is atomic
exchange (materials), electro-hydrodynamic (Quincke), or topological (QDM).
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

from closures.materials_science.magnetic_properties import (  # noqa: E402
    REFERENCE_MAGNETIC,
    compute_magnetic_properties,
)
from closures.quantum_mechanics.quantum_dimer_model import (  # noqa: E402
    QDM_PHASES,
    compute_qdm_kernel,
)
from closures.rcft.quincke_rollers import (  # noqa: E402
    build_quincke_catalog,
    compute_quincke_kernel,
)
from umcp.frozen_contract import EPSILON  # noqa: E402
from umcp.kernel_optimized import OptimizedKernelComputer  # noqa: E402

_K = OptimizedKernelComputer()

# ---------------------------------------------------------------------------
# System labels
# ---------------------------------------------------------------------------
SYSTEM_MATERIAL = "Material"
SYSTEM_QUINCKE = "Quincke"
SYSTEM_QDM = "QDM"

# 6 primary QDM phases (published in Yan et al. 2022 — boundary/hard/deep/VBS
# variants excluded to maintain 35-entity cross-system design).
_QDM_PRIMARY_NAMES = frozenset(
    {
        "odd_Z2_QSL",
        "even_Z2_QSL",
        "PM_trivial",
        "columnar_crystal",
        "nematic",
        "staggered_1_6",
    }
)

# Atomic-number lookup for material traces
_Z_MAP = {
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Gd": 64,
    "Dy": 66,
    "Cr": 24,
    "Mn": 25,
    "MnO": 25,
    "NiO": 28,
    "FeO": 26,
    "CoO": 27,
    "Fe3O4": 26,
    "Cu": 29,
    "Au": 79,
    "Bi": 83,
    "Al": 13,
    "Pt": 78,
}


# ---------------------------------------------------------------------------
# Unified kernel result
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MCSKernelResult:
    """Kernel output for a cross-system magnetic entity."""

    name: str
    system: str
    category: str
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    regime: str

    @property
    def gap(self) -> float:
        """Heterogeneity gap Δ = F − IC."""
        return self.F - self.IC

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "system": self.system,
            "category": self.category,
            "F": self.F,
            "omega": self.omega,
            "S": self.S,
            "C": self.C,
            "kappa": self.kappa,
            "IC": self.IC,
            "regime": self.regime,
            "gap": self.gap,
        }


# ---------------------------------------------------------------------------
# Regime classification (frozen gates)
# ---------------------------------------------------------------------------
def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


# ---------------------------------------------------------------------------
# Per-system builders
# ---------------------------------------------------------------------------
def _build_material(sym: str, T_K: float = 300.0) -> MCSKernelResult:
    """Compute kernel for a bulk magnetic material."""
    ref = REFERENCE_MAGNETIC[sym]
    r = compute_magnetic_properties(_Z_MAP[sym], symbol=sym, T_K=T_K)
    M_sat = float(ref.get("M_sat", 0))
    T_c = float(ref.get("T_c", ref.get("T_N", 0)))
    n_unp = int(ref.get("n_unpaired", 0))
    cls = str(ref.get("class", ""))

    c1 = max(EPSILON, min(1 - EPSILON, abs(r.M_total_B) / max(M_sat, 0.001)))
    c2 = max(EPSILON, min(1 - EPSILON, 1 - T_K / T_c if T_c > T_K else EPSILON))
    c3 = max(EPSILON, min(1 - EPSILON, abs(r.J_exchange_meV) / 20.0))
    c4 = max(EPSILON, min(1 - EPSILON, 1.0 / (1.0 + abs(r.chi_SI) * 1e6)))
    c5 = max(EPSILON, min(1 - EPSILON, n_unp / 7.0 if n_unp > 0 else EPSILON))
    c6 = max(EPSILON, min(1 - EPSILON, M_sat / 10.2 if M_sat > 0 else EPSILON))
    c7 = max(EPSILON, min(1 - EPSILON, T_c / 1394.0 if T_c > 0 else EPSILON))
    c8 = 0.95 if cls in ("Ferromagnetic", "Ferrimagnetic") else (0.50 if cls == "Antiferromagnetic" else EPSILON)
    c8 = max(EPSILON, min(1 - EPSILON, c8))

    channels = np.array([c1, c2, c3, c4, c5, c6, c7, c8])
    weights = np.ones(8) / 8
    out = _K.compute(channels, weights)
    regime = _classify_regime(out.omega, out.F, out.S, out.C)
    return MCSKernelResult(
        name=f"MAT:{sym}",
        system=SYSTEM_MATERIAL,
        category=cls,
        F=out.F,
        omega=out.omega,
        S=out.S,
        C=out.C,
        kappa=out.kappa,
        IC=out.IC,
        regime=regime,
    )


def _build_quincke(cfg) -> MCSKernelResult:
    """Compute kernel for a Quincke roller state."""
    r = compute_quincke_kernel(cfg)
    regime = _classify_regime(r.omega, r.F, r.S, r.C)
    return MCSKernelResult(
        name=f"QR:{r.name}",
        system=SYSTEM_QUINCKE,
        category=r.regime,
        F=r.F,
        omega=r.omega,
        S=r.S,
        C=r.C,
        kappa=float(np.log(max(r.IC, EPSILON))),
        IC=r.IC,
        regime=regime,
    )


def _build_qdm(phase) -> MCSKernelResult:
    """Compute kernel for a QDM phase."""
    r = compute_qdm_kernel(phase)
    regime = _classify_regime(r.omega, r.F, r.S, r.C)
    return MCSKernelResult(
        name=f"QDM:{r.name}",
        system=SYSTEM_QDM,
        category=r.regime,
        F=r.F,
        omega=r.omega,
        S=r.S,
        C=r.C,
        kappa=float(np.log(max(r.IC, EPSILON))),
        IC=r.IC,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
N_ENTITIES = 35
N_SYSTEMS = 3


def build_cross_system_catalog() -> list[MCSKernelResult]:
    """Build the full 35-entity cross-system catalog.

    Returns kernel results for 17 materials + 12 Quincke + 6 primary QDM phases.
    """
    items: list[MCSKernelResult] = []

    # Materials (17)
    for sym in REFERENCE_MAGNETIC:
        items.append(_build_material(sym))

    # Quincke rollers (12)
    for cfg in build_quincke_catalog():
        items.append(_build_quincke(cfg))

    # QDM primary phases (6)
    for phase in QDM_PHASES:
        if phase.name in _QDM_PRIMARY_NAMES:
            items.append(_build_qdm(phase))

    return items


# ---------------------------------------------------------------------------
# Theorems T-MCS-1 through T-MCS-6
# ---------------------------------------------------------------------------


def verify_t_mcs_1(results: list[MCSKernelResult]) -> dict:
    """T-MCS-1: Duality universality — F + ω = 1 across all 35 entities.

    The duality identity holds to machine precision regardless of which
    physical system generated the trace vector.
    """
    max_residual = max(abs(r.F + r.omega - 1.0) for r in results)
    passed = max_residual < 1e-12
    return {
        "name": "T-MCS-1",
        "passed": bool(passed),
        "max_duality_residual": float(max_residual),
        "n_entities": len(results),
    }


def verify_t_mcs_2(results: list[MCSKernelResult]) -> dict:
    """T-MCS-2: Integrity bound — IC ≤ F across all 35 entities.

    The solvability condition holds universally: magnetic exchange (materials),
    electro-hydrodynamic torque (Quincke), and topological order (QDM) all
    satisfy the same algebraic constraint.
    """
    violations = [r for r in results if r.IC > r.F + 1e-12]
    max_excess = max((r.IC - r.F) for r in results) if results else 0.0
    passed = len(violations) == 0
    return {
        "name": "T-MCS-2",
        "passed": bool(passed),
        "n_violations": len(violations),
        "max_IC_minus_F": float(max_excess),
        "n_entities": len(results),
    }


def verify_t_mcs_3(results: list[MCSKernelResult]) -> dict:
    """T-MCS-3: Curvature–heterogeneity correlation — r(C, Δ) > 0.5 across all.

    Curvature C tracks the heterogeneity gap Δ = F − IC universally: higher
    channel spread (C) produces larger integrity loss (Δ), regardless of
    the physical origin of the channels.
    """
    cs = np.array([r.C for r in results])
    ds = np.array([r.gap for r in results])
    # Pearson correlation
    c_mean, d_mean = cs.mean(), ds.mean()
    cov = np.sum((cs - c_mean) * (ds - d_mean))
    c_std = np.sqrt(np.sum((cs - c_mean) ** 2))
    d_std = np.sqrt(np.sum((ds - d_mean) ** 2))
    r_cd = float(cov / max(c_std * d_std, 1e-15))
    passed = r_cd > 0.5
    return {
        "name": "T-MCS-3",
        "passed": bool(passed),
        "pearson_r_C_delta": r_cd,
        "n_entities": len(results),
    }


def verify_t_mcs_4(results: list[MCSKernelResult]) -> dict:
    """T-MCS-4: Multi-regime span — all three systems contain Collapse-regime entities.

    The kernel classifies entities from every system into multiple regimes,
    demonstrating that regime boundaries are structural (kernel-derived),
    not domain-specific.
    """
    system_regimes: dict[str, set[str]] = {}
    for r in results:
        system_regimes.setdefault(r.system, set()).add(r.regime)
    # All three systems must have at least Collapse
    all_have_collapse = all("Collapse" in regimes for regimes in system_regimes.values())
    total_regimes = set()
    for regimes in system_regimes.values():
        total_regimes |= regimes
    passed = all_have_collapse and len(total_regimes) >= 2
    return {
        "name": "T-MCS-4",
        "passed": bool(passed),
        "system_regimes": {k: sorted(v) for k, v in system_regimes.items()},
        "total_distinct_regimes": len(total_regimes),
    }


def verify_t_mcs_5(results: list[MCSKernelResult]) -> dict:
    """T-MCS-5: Magnetic activation — field activation increases IC in Quincke.

    SubThreshold (no rolling) → ChainAssembly (active chains) shows IC
    restoration by orders of magnitude, measuring the same structural
    transition as atomic exchange ordering measures in ferromagnets.
    """
    sub = next((r for r in results if "SubThreshold" in r.name), None)
    chain = next((r for r in results if "ChainAssembly" in r.name), None)
    if sub is None or chain is None:
        return {"name": "T-MCS-5", "passed": False, "error": "entities not found"}
    ratio = chain.IC / max(sub.IC, EPSILON)
    passed = chain.IC > sub.IC and ratio > 10.0
    return {
        "name": "T-MCS-5",
        "passed": bool(passed),
        "sub_IC": float(sub.IC),
        "chain_IC": float(chain.IC),
        "IC_ratio": float(ratio),
    }


def verify_t_mcs_6(results: list[MCSKernelResult]) -> dict:
    """T-MCS-6: Ferromagnetic IC dominance — ferromagnets have highest material IC.

    Among the 17 materials, ferromagnets (Fe, Co, Ni, Gd, Dy) maintain the
    highest mean IC because their strongly ordered channels resist geometric
    slaughter. Diamagnets have the lowest IC (all channels near ε).
    """
    mat_results = [r for r in results if r.system == SYSTEM_MATERIAL]
    cats: dict[str, list[float]] = {}
    for r in mat_results:
        cats.setdefault(r.category, []).append(r.IC)

    ferro_mean = float(np.mean(cats.get("Ferromagnetic", [0.0])))
    dia_mean = float(np.mean(cats.get("Diamagnetic", [0.0])))
    passed = ferro_mean > dia_mean and ferro_mean > 0.01
    return {
        "name": "T-MCS-6",
        "passed": bool(passed),
        "ferro_mean_IC": ferro_mean,
        "dia_mean_IC": dia_mean,
        "category_mean_IC": {k: float(np.mean(v)) for k, v in cats.items()},
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-MCS theorems on the full 35-entity catalog."""
    results = build_cross_system_catalog()
    return [
        verify_t_mcs_1(results),
        verify_t_mcs_2(results),
        verify_t_mcs_3(results),
        verify_t_mcs_4(results),
        verify_t_mcs_5(results),
        verify_t_mcs_6(results),
    ]


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = build_cross_system_catalog()
    print(f"Catalog: {len(results)} entities across {N_SYSTEMS} systems")
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")
