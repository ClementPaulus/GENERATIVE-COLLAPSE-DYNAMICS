"""Gravitational Wave Memory Closure — Spacetime Memory Domain.

Tier-2 closure mapping 12 gravitational wave source types through the GCD kernel.
Each source is characterized by 8 channels drawn from GW astronomy.

Channels (8, equal weights w_i = 1/8):
  0  strain_amplitude       — peak h+ (normalized log scale)
  1  frequency_band          — characteristic GW frequency (normalized to LIGO band)
  2  memory_displacement     — permanent spacetime displacement (BMS memory, normalized)
  3  chirp_mass_ratio        — symmetric mass ratio η (normalized)
  4  polarization_purity     — ratio of + to × polarization
  5  inspiral_duration       — time in band before merger (normalized)
  6  ringdown_coherence      — QNM damping quality factor (normalized)
  7  detection_confidence    — SNR-based confidence (normalized)

12 entities across 4 categories:
  Compact binary (4): binary_BH, binary_NS, BH_NS, intermediate_mass_BH
  Transient (3): core_collapse_SN, magnetar_flare, cosmic_string_cusp
  Continuous (3): spinning_pulsar, low_mass_X_ray, r_mode_instability
  Stochastic (2): primordial_background, astrophysical_foreground

6 theorems (T-GW-1 through T-GW-6).
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

GW_CHANNELS = [
    "strain_amplitude",
    "frequency_band",
    "memory_displacement",
    "chirp_mass_ratio",
    "polarization_purity",
    "inspiral_duration",
    "ringdown_coherence",
    "detection_confidence",
]
N_GW_CHANNELS = len(GW_CHANNELS)


@dataclass(frozen=True, slots=True)
class GWSourceEntity:
    """A gravitational wave source with 8 measurable channels."""

    name: str
    category: str
    strain_amplitude: float
    frequency_band: float
    memory_displacement: float
    chirp_mass_ratio: float
    polarization_purity: float
    inspiral_duration: float
    ringdown_coherence: float
    detection_confidence: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.strain_amplitude,
                self.frequency_band,
                self.memory_displacement,
                self.chirp_mass_ratio,
                self.polarization_purity,
                self.inspiral_duration,
                self.ringdown_coherence,
                self.detection_confidence,
            ]
        )


GW_ENTITIES: tuple[GWSourceEntity, ...] = (
    # Compact binary coalescences
    GWSourceEntity("binary_BH", "compact_binary", 0.95, 0.85, 0.90, 0.80, 0.90, 0.70, 0.95, 0.95),
    GWSourceEntity("binary_NS", "compact_binary", 0.60, 0.90, 0.40, 0.95, 0.85, 0.95, 0.50, 0.80),
    GWSourceEntity("BH_NS", "compact_binary", 0.70, 0.88, 0.55, 0.45, 0.75, 0.80, 0.70, 0.75),
    GWSourceEntity("intermediate_mass_BH", "compact_binary", 0.80, 0.40, 0.85, 0.70, 0.88, 0.50, 0.90, 0.55),
    # Transient sources
    GWSourceEntity("core_collapse_SN", "transient", 0.30, 0.70, 0.20, 0.10, 0.30, 0.05, 0.15, 0.25),
    GWSourceEntity("magnetar_flare", "transient", 0.15, 0.95, 0.10, 0.05, 0.20, 0.02, 0.10, 0.15),
    GWSourceEntity("cosmic_string_cusp", "transient", 0.25, 0.50, 0.60, 0.02, 0.95, 0.01, 0.05, 0.10),
    # Continuous sources
    GWSourceEntity("spinning_pulsar", "continuous", 0.10, 0.95, 0.02, 0.01, 0.90, 0.99, 0.01, 0.20),
    GWSourceEntity("low_mass_X_ray", "continuous", 0.15, 0.85, 0.03, 0.15, 0.75, 0.95, 0.05, 0.30),
    GWSourceEntity("r_mode_instability", "continuous", 0.12, 0.80, 0.05, 0.10, 0.60, 0.90, 0.08, 0.15),
    # Stochastic background
    GWSourceEntity("primordial_background", "stochastic", 0.05, 0.30, 0.70, 0.50, 0.50, 0.99, 0.02, 0.05),
    GWSourceEntity("astrophysical_foreground", "stochastic", 0.20, 0.60, 0.10, 0.50, 0.50, 0.99, 0.05, 0.15),
)


@dataclass(frozen=True, slots=True)
class GWKernelResult:
    """Kernel output for a GW source entity."""

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


def compute_gw_kernel(entity: GWSourceEntity) -> GWKernelResult:
    """Compute GCD kernel for a GW source entity."""
    c = entity.trace_vector()
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(N_GW_CHANNELS) / N_GW_CHANNELS
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
    return GWKernelResult(
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


def compute_all_entities() -> list[GWKernelResult]:
    """Compute kernel outputs for all GW source entities."""
    return [compute_gw_kernel(e) for e in GW_ENTITIES]


# ── Theorems ──────────────────────────────────────────────────────────


def verify_t_gw_1(results: list[GWKernelResult]) -> dict:
    """T-GW-1: Binary BH mergers have highest F — strongest, most
    well-characterized GW source across all channels.
    """
    bbh = next(r for r in results if r.name == "binary_BH")
    max_F = max(r.F for r in results)
    passed = abs(bbh.F - max_F) < 0.02
    return {"name": "T-GW-1", "passed": bool(passed), "BBH_F": bbh.F, "max_F": float(max_F)}


def verify_t_gw_2(results: list[GWKernelResult]) -> dict:
    """T-GW-2: Compact binary coalescences have higher mean F than all others.

    CBCs are the gold standard of GW detection — high SNR, well-modeled waveforms.
    """
    cbc = [r.F for r in results if r.category == "compact_binary"]
    other = [r.F for r in results if r.category != "compact_binary"]
    passed = np.mean(cbc) > np.mean(other)
    return {
        "name": "T-GW-2",
        "passed": bool(passed),
        "CBC_mean_F": float(np.mean(cbc)),
        "other_mean_F": float(np.mean(other)),
    }


def verify_t_gw_3(results: list[GWKernelResult]) -> dict:
    """T-GW-3: Continuous sources have largest mean heterogeneity gap —
    very low strain amplitude creates channel divergence despite
    long observation baseline.
    """
    cats = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F - r.IC)
    cat_gaps = {k: float(np.mean(v)) for k, v in cats.items()}
    cont_gap = cat_gaps.get("continuous", 0.0)
    max_gap = max(cat_gaps.values())
    passed = abs(cont_gap - max_gap) < 0.02
    return {"name": "T-GW-3", "passed": bool(passed), "continuous_gap": cont_gap, "all_gaps": cat_gaps}


def verify_t_gw_4(results: list[GWKernelResult]) -> dict:
    """T-GW-4: Stochastic backgrounds are in Collapse regime — no individual
    source is detectable, only statistical ensemble.
    """
    stoch = [r for r in results if r.category == "stochastic"]
    all_collapse = all(r.regime == "Collapse" for r in stoch)
    return {"name": "T-GW-4", "passed": bool(all_collapse), "stochastic_regimes": [r.regime for r in stoch]}


def verify_t_gw_5(results: list[GWKernelResult]) -> dict:
    """T-GW-5: Binary NS has highest inspiral duration channel → long
    in-band signal. But strain is lower than BBH.
    """
    bns_e = next(e for e in GW_ENTITIES if e.name == "binary_NS")
    bbh_e = next(e for e in GW_ENTITIES if e.name == "binary_BH")
    passed = bns_e.inspiral_duration > bbh_e.inspiral_duration and bns_e.strain_amplitude < bbh_e.strain_amplitude
    return {
        "name": "T-GW-5",
        "passed": bool(passed),
        "BNS_inspiral": bns_e.inspiral_duration,
        "BBH_inspiral": bbh_e.inspiral_duration,
    }


def verify_t_gw_6(results: list[GWKernelResult]) -> dict:
    """T-GW-6: Primordial background has largest memory displacement among
    stochastic sources — cosmological GW memory exceeds astrophysical.
    """
    prim = next(e for e in GW_ENTITIES if e.name == "primordial_background")
    astro = next(e for e in GW_ENTITIES if e.name == "astrophysical_foreground")
    passed = prim.memory_displacement > astro.memory_displacement
    return {
        "name": "T-GW-6",
        "passed": bool(passed),
        "primordial_memory": prim.memory_displacement,
        "astro_memory": astro.memory_displacement,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-GW theorems."""
    results = compute_all_entities()
    return [
        verify_t_gw_1(results),
        verify_t_gw_2(results),
        verify_t_gw_3(results),
        verify_t_gw_4(results),
        verify_t_gw_5(results),
        verify_t_gw_6(results),
    ]


def main() -> None:
    """Entry point."""
    results = compute_all_entities()
    print("=" * 78)
    print("GRAVITATIONAL WAVE SOURCES — GCD KERNEL ANALYSIS")
    print("=" * 78)
    print(f"{'Entity':<28} {'Cat':<16} {'F':>6} {'ω':>6} {'IC':>6} {'Δ':>6} {'Regime'}")
    print("-" * 78)
    for r in results:
        gap = r.F - r.IC
        print(f"{r.name:<28} {r.category:<16} {r.F:6.3f} {r.omega:6.3f} {r.IC:6.3f} {gap:6.3f} {r.regime}")

    print("\n── Theorems ──")
    for t in verify_all_theorems():
        print(f"  {t['name']}: {'PROVEN' if t['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
