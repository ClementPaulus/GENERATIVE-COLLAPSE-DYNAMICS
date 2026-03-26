"""Population Fragmentation Closure — Evolution Domain.

Tier-2 closure mapping 12 fragmented population configurations through the
GCD kernel.  Based on Devadhasan & Carja (2025), "Local negative frequency
dependence can decrease global coexistence in fragmented populations," PNAS
DOI:10.1073/pnas.2513857122.

Key finding encoded:  Negative frequency-dependent (NFD) selection — the
mechanism that gives rare species an advantage — breaks down when habitat
is fragmented and migration is disrupted.  In the GCD kernel this appears
as **geometric slaughter via the migration_connectivity channel**: when
that channel drops toward ε, IC collapses even when mean fidelity F stays
moderate, exactly as §3 of the orientation predicts.

Channels (8, equal weights w_i = 1/8) — Population-scale channels
from recursive_evolution.py:
  0  effective_population_size   — N_e relative to viable threshold
  1  genetic_variation           — allelic diversity maintained
  2  migration_connectivity      — inter-patch dispersal rate  [KEY CHANNEL]
  3  selection_efficiency        — strength of NFD selection
  4  drift_resistance            — resistance to stochastic loss
  5  mutation_supply             — new allele input rate
  6  recombination_rate          — genetic mixing within patches
  7  demographic_stability       — population growth consistency

12 entities across 4 fragmentation regimes:
  Well-mixed baseline (3):   Panmictic_large, Panmictic_medium, Panmictic_small
  Partial fragmentation (3): Stepping_stone, Island_model, Corridor_linked
  Severe fragmentation (3):  Archipelago_isolated, Habitat_island, Relict_patch
  Restoration / cross-domain (3): Corridor_restored, Tumor_heterogeneous, Tumor_treated

6 theorems (T-EV-11 through T-EV-16):
  T-EV-11  Fragmentation Slaughter — migration channel kills IC
  T-EV-12  NFD Reversal — selection_efficiency fails to rescue in isolation
  T-EV-13  Corridor Recovery (Scale Inversion) — restoring migration restores IC
  T-EV-14  Tier-1 Universality — all 12 entities satisfy F+ω=1, IC≤F, IC=exp(κ)
  T-EV-15  Regime Stratification — well-mixed Watch, fragmented Collapse
  T-EV-16  Cross-Domain Bridge — tumor fragmentation maps to same kernel geometry

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized →
                   recursive_evolution (Population scale) → this module
"""

from __future__ import annotations

import math
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

# ── Channel definitions (Population-scale from recursive_evolution) ──
PF_CHANNELS = [
    "effective_population_size",
    "genetic_variation",
    "migration_connectivity",
    "selection_efficiency",
    "drift_resistance",
    "mutation_supply",
    "recombination_rate",
    "demographic_stability",
]
N_PF_CHANNELS = len(PF_CHANNELS)
MIGRATION_IDX = 2  # migration_connectivity is the critical channel


@dataclass(frozen=True, slots=True)
class FragmentationEntity:
    """A population configuration with 8 Population-scale channels."""

    name: str
    regime_category: str  # well_mixed | partial | severe | restoration
    effective_population_size: float
    genetic_variation: float
    migration_connectivity: float
    selection_efficiency: float
    drift_resistance: float
    mutation_supply: float
    recombination_rate: float
    demographic_stability: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.effective_population_size,
                self.genetic_variation,
                self.migration_connectivity,
                self.selection_efficiency,
                self.drift_resistance,
                self.mutation_supply,
                self.recombination_rate,
                self.demographic_stability,
            ]
        )


# ═════════════════════════════════════════════════════════════════════
# ENTITY CATALOG — 12 population fragmentation configurations
# ═════════════════════════════════════════════════════════════════════
#
# Channel values derived from ecological population genetics:
# - Well-mixed: high migration, strong NFD, large N_e
# - Partial: reduced migration, NFD weakening
# - Severe: near-zero migration, NFD reversed (Devadhasan & Carja 2025)
# - Restoration: corridors re-open migration; tumor cross-domain bridge

PF_ENTITIES: tuple[FragmentationEntity, ...] = (
    # ── Well-mixed baseline ──────────────────────────────────────
    # Large panmictic population: all channels healthy, NFD works
    FragmentationEntity(
        "Panmictic_large",
        "well_mixed",
        0.90,
        0.85,
        0.95,
        0.80,
        0.85,
        0.75,
        0.80,
        0.82,
    ),
    # Medium population: still well-connected, slightly lower N_e
    FragmentationEntity(
        "Panmictic_medium",
        "well_mixed",
        0.78,
        0.80,
        0.90,
        0.76,
        0.75,
        0.72,
        0.78,
        0.75,
    ),
    # Small but panmictic: drift starting to matter, migration intact
    FragmentationEntity(
        "Panmictic_small",
        "well_mixed",
        0.72,
        0.74,
        0.85,
        0.72,
        0.68,
        0.70,
        0.72,
        0.70,
    ),
    # ── Partial fragmentation ────────────────────────────────────
    # Stepping-stone: migration via neighbor patches only
    FragmentationEntity(
        "Stepping_stone",
        "partial",
        0.65,
        0.68,
        0.45,
        0.62,
        0.60,
        0.64,
        0.65,
        0.58,
    ),
    # Island model: discrete patches, moderate migration
    FragmentationEntity(
        "Island_model",
        "partial",
        0.60,
        0.62,
        0.35,
        0.58,
        0.55,
        0.60,
        0.58,
        0.52,
    ),
    # Corridor-linked: few narrow corridors maintain some flow
    FragmentationEntity(
        "Corridor_linked",
        "partial",
        0.58,
        0.60,
        0.30,
        0.55,
        0.52,
        0.58,
        0.55,
        0.50,
    ),
    # ── Severe fragmentation ─────────────────────────────────────
    # Archipelago: islands with near-zero migration — KEY FINDING:
    # Other channels remain moderate (populations viable per-patch)
    # but migration → ε kills IC via geometric slaughter (§3).
    # This is Devadhasan & Carja's NFD reversal in the kernel.
    FragmentationEntity(
        "Archipelago_isolated",
        "severe",
        0.50,
        0.60,
        0.05,
        0.58,
        0.48,
        0.55,
        0.58,
        0.48,
    ),
    # Habitat island: landlocked fragment, no corridors
    FragmentationEntity(
        "Habitat_island",
        "severe",
        0.45,
        0.55,
        0.03,
        0.52,
        0.42,
        0.50,
        0.52,
        0.42,
    ),
    # Relict patch: last remnant population, minimal connectivity
    FragmentationEntity(
        "Relict_patch",
        "severe",
        0.38,
        0.45,
        0.02,
        0.42,
        0.35,
        0.42,
        0.45,
        0.35,
    ),
    # ── Restoration / cross-domain ───────────────────────────────
    # Corridor restored: fragment + wildlife corridor reopened
    # Scale inversion: IC recovers when migration channel restored
    FragmentationEntity(
        "Corridor_restored",
        "restoration",
        0.52,
        0.58,
        0.72,
        0.55,
        0.50,
        0.55,
        0.58,
        0.52,
    ),
    # Tumor heterogeneous: cancer subclones as isolated demes
    # Cross-domain bridge: same kernel geometry, different field
    FragmentationEntity(
        "Tumor_heterogeneous",
        "restoration",
        0.55,
        0.72,
        0.08,
        0.62,
        0.50,
        0.78,
        0.35,
        0.42,
    ),
    # Tumor treated: therapy creates migration barriers between clones
    FragmentationEntity(
        "Tumor_treated",
        "restoration",
        0.40,
        0.50,
        0.04,
        0.38,
        0.32,
        0.55,
        0.25,
        0.30,
    ),
)


# ═════════════════════════════════════════════════════════════════════
# KERNEL COMPUTATION
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PFKernelResult:
    """Kernel output for a population fragmentation entity."""

    name: str
    regime_category: str
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    regime: str
    migration_value: float
    heterogeneity_gap: float


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def compute_pf_kernel(entity: FragmentationEntity) -> PFKernelResult:
    """Compute kernel invariants for a population fragmentation entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_PF_CHANNELS) / N_PF_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return PFKernelResult(
        name=entity.name,
        regime_category=entity.regime_category,
        F=F,
        omega=omega,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        regime=regime,
        migration_value=entity.migration_connectivity,
        heterogeneity_gap=F - IC,
    )


def compute_all_entities() -> list[PFKernelResult]:
    """Compute kernel for all 12 fragmentation entities."""
    return [compute_pf_kernel(e) for e in PF_ENTITIES]


# ═════════════════════════════════════════════════════════════════════
# CROSS-DOMAIN BRIDGE — Ecology ↔ Oncology
# ═════════════════════════════════════════════════════════════════════


def cross_domain_bridge(results: list[PFKernelResult]) -> dict:
    """Bridge population fragmentation to tumor heterogeneity.

    Same kernel, same geometric slaughter mechanism:
    - Ecology:  patches with cut migration → rare species loss
    - Oncology: tumor subclones with therapy-induced barriers → clonal escape

    In ecology, we WANT high IC (coexistence preserved).
    In oncology, we WANT low IC (clonal diversity suppressed).
    The kernel geometry is identical; the *desirable direction* is inverted.
    """
    eco_severe = [r for r in results if r.regime_category == "severe"]
    tumors = [r for r in results if r.name.startswith("Tumor")]

    eco_mean_icf = float(np.mean([r.IC / r.F for r in eco_severe])) if eco_severe else 0.0
    tumor_mean_icf = float(np.mean([r.IC / r.F for r in tumors])) if tumors else 0.0

    eco_mean_mig = float(np.mean([r.migration_value for r in eco_severe]))
    tumor_mean_mig = float(np.mean([r.migration_value for r in tumors]))

    return {
        "eco_severe_mean_IC_F": eco_mean_icf,
        "tumor_mean_IC_F": tumor_mean_icf,
        "both_low_migration": eco_mean_mig < 0.10 and tumor_mean_mig < 0.10,
        "geometry_shared": abs(eco_mean_icf - tumor_mean_icf) < 0.25,
        "desirable_direction_inverted": True,
        "interpretation": (
            "Same geometric slaughter mechanism — dead migration channel "
            "kills IC in both ecology (bad: coexistence lost) and oncology "
            "(good: clonal diversity suppressed). The kernel is domain-blind; "
            "the contract determines which direction is desirable."
        ),
    }


# ═════════════════════════════════════════════════════════════════════
# THEOREMS T-EV-11 through T-EV-16
# ═════════════════════════════════════════════════════════════════════


def verify_t_ev_11(results: list[PFKernelResult]) -> dict:
    """T-EV-11: Fragmentation Slaughter — migration channel kills IC.

    Severe fragmentation entities have IC/F < 0.90 because the
    migration_connectivity channel (~0.02-0.05) drags the geometric
    mean down even though other channels remain moderate.  Well-mixed
    entities maintain IC/F > 0.95.  The heterogeneity gap Δ in severe
    entities exceeds well-mixed Δ by at least 10×.  This is §3
    geometric slaughter at the population scale.
    """
    severe = [r for r in results if r.regime_category == "severe"]
    well_mixed = [r for r in results if r.regime_category == "well_mixed"]

    severe_icf = [r.IC / r.F for r in severe]
    wm_icf = [r.IC / r.F for r in well_mixed]
    severe_delta = [r.heterogeneity_gap for r in severe]
    wm_delta = [r.heterogeneity_gap for r in well_mixed]

    # Severe entities: IC/F < 0.90 (geometric slaughter from dead migration)
    slaughter = all(icf < 0.90 for icf in severe_icf)
    # Well-mixed entities: IC/F > 0.95 (healthy coherence)
    healthy = all(icf > 0.95 for icf in wm_icf)
    # Heterogeneity gap amplification: severe Δ_mean > 10× well-mixed Δ_mean
    mean_severe_delta = float(np.mean(severe_delta))
    mean_wm_delta = float(np.mean(wm_delta))
    gap_amplified = mean_severe_delta > 10.0 * mean_wm_delta if mean_wm_delta > 0 else False

    return {
        "name": "T-EV-11",
        "passed": slaughter and healthy and gap_amplified,
        "severe_IC_F": [round(x, 4) for x in severe_icf],
        "well_mixed_IC_F": [round(x, 4) for x in wm_icf],
        "slaughter_confirmed": slaughter,
        "healthy_confirmed": healthy,
        "gap_amplification": round(mean_severe_delta / mean_wm_delta, 1) if mean_wm_delta > 0 else 0,
        "gap_amplified": gap_amplified,
    }


def verify_t_ev_12(results: list[PFKernelResult]) -> dict:
    """T-EV-12: NFD Reversal — selection efficiency cannot rescue IC in isolation.

    Devadhasan & Carja (2025) show NFD selection fails in fragmented
    habitats.  Here: severe entities have selection_efficiency ≥ 0.30
    but IC/F < 0.90 — the selection channel alone cannot compensate
    for the dead migration channel (geometric mean constraint).
    """
    severe = [r for r in results if r.regime_category == "severe"]

    selection_present = all(
        PF_ENTITIES[i].selection_efficiency >= 0.30 for i, e in enumerate(PF_ENTITIES) if e.regime_category == "severe"
    )
    ic_still_low = all(r.IC / r.F < 0.90 for r in severe)

    return {
        "name": "T-EV-12",
        "passed": selection_present and ic_still_low,
        "selection_present": selection_present,
        "ic_still_low": ic_still_low,
        "interpretation": (
            "NFD selection (selection_efficiency >= 0.30) cannot rescue IC/F above 0.90 "
            "when migration_connectivity is near epsilon — geometric mean "
            "constraint overrides arithmetic advantage (Devadhasan & Carja 2025)."
        ),
    }


def verify_t_ev_13(results: list[PFKernelResult]) -> dict:
    """T-EV-13: Corridor Recovery (Scale Inversion).

    Restoring migration connectivity restores IC.  The Corridor_restored
    entity has migration = 0.70 vs severe mean < 0.10, and IC/F should
    exceed the midpoint between severe and well-mixed IC/F.  This is
    scale inversion: new degrees of freedom (corridor) restore coherence.
    """
    corridor = next((r for r in results if r.name == "Corridor_restored"), None)
    severe = [r for r in results if r.regime_category == "severe"]

    if corridor is None or not severe:
        return {"name": "T-EV-13", "passed": False, "error": "missing entities"}

    corridor_icf = corridor.IC / corridor.F
    severe_mean_icf = float(np.mean([r.IC / r.F for r in severe]))
    wm = [r for r in results if r.regime_category == "well_mixed"]
    wm_mean_icf = float(np.mean([r.IC / r.F for r in wm])) if wm else 1.0

    # Corridor IC/F should be closer to well-mixed than to severe
    recovery = corridor_icf > (severe_mean_icf + wm_mean_icf) / 2.0
    migration_restored = corridor.migration_value > 0.50

    return {
        "name": "T-EV-13",
        "passed": recovery and migration_restored,
        "corridor_IC_F": round(corridor_icf, 4),
        "severe_mean_IC_F": round(severe_mean_icf, 4),
        "ratio": round(corridor_icf / severe_mean_icf, 2) if severe_mean_icf > 0 else 0,
        "migration_restored": migration_restored,
    }


def verify_t_ev_14(results: list[PFKernelResult]) -> dict:
    """T-EV-14: Tier-1 Universality — all 12 entities satisfy kernel identities.

    F + ω = 1, IC ≤ F, IC = exp(κ) must hold for every entity.
    """
    n_tests = 0
    n_passed = 0

    for r in results:
        # Duality
        n_tests += 1
        n_passed += int(abs(r.F + r.omega - 1.0) < 1e-12)

        # Integrity bound
        n_tests += 1
        n_passed += int(r.IC <= r.F + 1e-12)

        # Log-integrity relation
        n_tests += 1
        n_passed += int(abs(r.IC - math.exp(r.kappa)) < 1e-10)

    return {
        "name": "T-EV-14",
        "passed": n_passed == n_tests,
        "n_tests": n_tests,
        "n_passed": n_passed,
    }


def verify_t_ev_15(results: list[PFKernelResult]) -> dict:
    """T-EV-15: Regime Stratification.

    Well-mixed entities should be Watch or better (ω < 0.30).
    Severe fragmentation entities should be Collapse (ω ≥ 0.30).
    """
    well_mixed = [r for r in results if r.regime_category == "well_mixed"]
    severe = [r for r in results if r.regime_category == "severe"]

    wm_not_collapse = all(r.regime != "Collapse" for r in well_mixed)
    severe_collapse = all(r.regime == "Collapse" for r in severe)

    return {
        "name": "T-EV-15",
        "passed": wm_not_collapse and severe_collapse,
        "well_mixed_regimes": [r.regime for r in well_mixed],
        "severe_regimes": [r.regime for r in severe],
    }


def verify_t_ev_16(results: list[PFKernelResult]) -> dict:
    """T-EV-16: Cross-Domain Bridge — tumor fragmentation maps to same geometry.

    Tumor entities share the geometric slaughter pattern with severe
    ecological fragmentation: low migration → dead channel → IC collapse.
    """
    bridge = cross_domain_bridge(results)

    return {
        "name": "T-EV-16",
        "passed": bridge["both_low_migration"] and bridge["geometry_shared"],
        "eco_severe_IC_F": bridge["eco_severe_mean_IC_F"],
        "tumor_IC_F": bridge["tumor_mean_IC_F"],
        "both_low_migration": bridge["both_low_migration"],
        "geometry_shared": bridge["geometry_shared"],
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-EV-11 through T-EV-16 theorems."""
    results = compute_all_entities()
    return [
        verify_t_ev_11(results),
        verify_t_ev_12(results),
        verify_t_ev_13(results),
        verify_t_ev_14(results),
        verify_t_ev_15(results),
        verify_t_ev_16(results),
    ]


if __name__ == "__main__":
    print("=" * 80)
    print("  POPULATION FRAGMENTATION — Devadhasan & Carja (2025) in the GCD Kernel")
    print("=" * 80)
    results = compute_all_entities()
    print(f"\n  Entities: {len(results)}")
    for r in results:
        print(
            f"    {r.name:25s}  F={r.F:.4f}  IC={r.IC:.4f}  "
            f"IC/F={r.IC / r.F:.4f}  mig={r.migration_value:.2f}  "
            f"Δ={r.heterogeneity_gap:.4f}  {r.regime}"
        )
    print("\n  Theorems:")
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"    {t['name']}: {status}")
    print("\n  Cross-Domain Bridge:")
    bridge = cross_domain_bridge(results)
    for k, v in bridge.items():
        print(f"    {k}: {v}")
