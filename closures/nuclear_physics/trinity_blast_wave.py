"""Trinity Blast Wave Closure — NUC.INTSTACK.v1

═══════════════════════════════════════════════════════════════════
TRINITY NUCLEAR TEST — TAYLOR-SEDOV SELF-SIMILAR ANALYSIS
═══════════════════════════════════════════════════════════════════

Maps the Trinity nuclear test fireball expansion (05:29:45 MWT,
July 16, 1945, Jornada del Muerto, New Mexico) through the GCD
kernel as a self-similar return structure.

The Taylor-Sedov solution:

    R(t) = A · t^(2/5)    where A = (E / ρ₀)^(1/5) · ξ₀(γ)

IS a return structure: the self-similar variable ξ = R / (A·t^0.4)
collapses all of space-time to a single function.  At every scale,
the same physics returns.  This is Axiom-0 in blast wave form:
"Only what returns is real" — and R ∝ t^(2/5) returns identically
at every decade of time and space.

Entity structure:
    24 fireball entities — Mack photograph time-radius measurements
     3 device entities   — Pu-239 core, U-238 tamper, HE lens array
     2 reference entities — conventional HE blast, D-T fusion target
    ──────────────────────
    29 total entities

8-channel trace vector (equal weights w_i = 1/8):
    c₁  self_similarity      ξ = R/(A_FIT·t^0.4) — Taylor-Sedov conformance
    c₂  energy_consistency   E_local/E_extracted — self-consistent yield
    c₃  mach_fidelity        M/(M+M_REF) — normalized shock Mach number
    c₄  power_law_quality    1−|α−0.4|/0.4 — self-similar exponent quality
    c₅  strong_shock         M²/(M²+1) — strong shock validity
    c₆  density_jump         (ρ₁/ρ₀)/6 — Rankine-Hugoniot conformance (γ=1.4)
    c₇  overpressure_norm    log₁₀(Δp/p₀+1)/log₁₀(P_MAX+1) — log overpressure
    c₈  binding_fidelity     BE_fuel/A / BE_peak — iron peak bridge

Theorems:
    T-TB-1  Self-Similar Conformance — ξ ≈ 1 across strong shock phase
    T-TB-2  Yield Self-Consistency — E_local variance < 20% in strong shock
    T-TB-3  Shock Weakening — IC drops as Mach → 1 at late times
    T-TB-4  Phase Boundary — F(strong shock) > F(weak shock)
    T-TB-5  Power Law Quality — local α within 15% of 0.4 for mid-range
    T-TB-6  Taylor Yield — extracted yield matches official 21 kt ± 25%
    T-TB-7  Velocity Monotonicity — v_shock monotonically decreasing
    T-TB-8  Fission-Fusion Bridge — fission and fusion discriminated by IC

Source data:
    Taylor, G.I. (1950) Proc. Roy. Soc. A 201(1065), 159-186
    Sedov, L.I. (1959) Similarity and Dimensional Methods in Mechanics
    Selby et al. (2021) Nuclear Technology 207(sup1), 321-325
    Mack, J.E. (1946) Semi-Popular Motion Picture Record (LANL)

Cross-references:
    closures/nuclear_physics/nuclide_binding.py     — Pu-239 binding energy
    closures/nuclear_physics/fissility.py           — Pu-239 fissility Z²/A
    closures/nuclear_physics/double_sided_collapse.py — iron peak convergence
    closures/nuclear_physics/qgp_rhic.py            — confinement analogy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

# ═══════════════════════════════════════════════════════════════════
# SECTION 0 — FROZEN PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# Taylor-Sedov blast wave parameters
GAMMA_AIR: float = 1.4  # Ratio of specific heats (diatomic air)
RHO_AIR: float = 1.25  # Air density at test site (kg/m³), Taylor (1950)
C_AIR: float = 343.0  # Speed of sound in air at ~20 °C (m/s)
P_ATM: float = 101325.0  # Atmospheric pressure at test site (Pa)

# Rankine-Hugoniot strong shock limit for γ = 1.4
DENSITY_JUMP_LIMIT: float = (GAMMA_AIR + 1) / (GAMMA_AIR - 1)  # 6.0

# Self-similar fit parameter — frozen from median of R/t^0.4 across
# 24 Mack photograph data points.  Yields E_extracted ≈ 20.2 kt,
# consistent with the official declassified value of 21 kt.
A_FIT: float = 583.6  # R = A_FIT · t^0.4 (SI units: m, s)
E_EXTRACTED_J: float = RHO_AIR * A_FIT**5  # ≈ 8.46e13 J ≈ 20.2 kt

# Yield values
KT_TO_J: float = 4.184e12  # 1 kiloton TNT in Joules
YIELD_KT: float = 21.0  # Official declassified yield (kt TNT)
YIELD_SELBY_KT: float = 24.8  # Selby et al. (2021) re-analysis
YIELD_J: float = YIELD_KT * KT_TO_J  # Official yield in Joules

# Normalization scales
MACH_REF: float = 10.0  # Reference Mach for c₃ = M/(M+M_REF)
P_RATIO_MAX: float = 20000.0  # Max pressure ratio for log normalization
P_RATIO_LOG_SCALE: float = math.log10(P_RATIO_MAX + 1.0)

# Pu-239 fission parameters (fission-fusion bridge)
PU239_Z: int = 94
PU239_A: int = 239
PU239_MASS_KG: float = 6.2  # Core mass (kg)
PU239_CONSUMED_KG: float = 1.0  # Estimated mass converted
FISSION_EFFICIENCY: float = PU239_CONSUMED_KG / PU239_MASS_KG  # ≈ 0.161
PU239_BE_PER_A: float = 7.560  # Binding energy per nucleon (MeV)
PU239_FISSILITY: float = PU239_Z**2 / PU239_A  # Z²/A ≈ 36.97
Q_FISSION_MEV: float = 200.0  # Average energy per fission event (MeV)

# Iron peak reference (from double_sided_collapse.py)
BE_PEAK_REF: float = 8.7945  # Ni-62 BE/A peak (MeV/nucleon)
A_PEAK: int = 62

# Binding fidelity for key fuels: BE/A / BE_peak
BINDING_PU239: float = PU239_BE_PER_A / BE_PEAK_REF  # 0.860
BINDING_U238: float = 7.570 / BE_PEAK_REF  # 0.861
BINDING_DEUTERIUM: float = 1.112 / BE_PEAK_REF  # 0.126

# Detonation metadata
TOWER_HEIGHT_M: float = 30.5  # 100 feet
DETONATION_TIME: str = "05:29:45 MWT July 16 1945"
DETONATION_LAT: float = 33.6775
DETONATION_LON: float = -106.4756


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════


class TrinityObservables(NamedTuple):
    """Raw blast wave observables for one measurement point."""

    time_s: float  # Time since detonation (seconds)
    radius_m: float  # Shock/fireball radius (meters)
    velocity_m_s: float  # Shock velocity (m/s)
    mach_number: float  # Mach number M = v/c₀
    overpressure_Pa: float  # Peak overpressure behind shock (Pa)
    energy_local_J: float  # Locally extracted yield (J)
    density_ratio: float  # Post-shock density ratio ρ₁/ρ₀
    binding_fidelity: float  # BE_fuel/A / BE_peak — iron peak bridge
    power_law_exponent: float  # Local α = d(ln R)/d(ln t)


@dataclass
class TrinityEntity:
    """One entity in the Trinity blast wave analysis."""

    name: str
    category: str  # "fireball", "device", "reference"
    observables: TrinityObservables
    trace: np.ndarray  # 8-channel trace vector
    weights: np.ndarray  # Equal weights (1/8)
    channels: list[str]
    # Kernel outputs
    F: float = 0.0
    omega: float = 0.0
    S: float = 0.0
    C: float = 0.0
    kappa: float = 0.0
    IC: float = 0.0
    gap: float = 0.0
    regime: str = ""
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — DATA TABLES
# ═══════════════════════════════════════════════════════════════════

# Mack photograph time-radius measurements from Taylor (1950) Table 2.
# These are the declassified fireball expansion measurements that
# Taylor used to extract the yield of the Trinity test.
# t_ms: time since detonation (milliseconds)
# R_m:  fireball/shock radius (meters)
MACK_DATA: list[dict[str, float]] = [
    {"t_ms": 0.10, "R_m": 11.1},
    {"t_ms": 0.24, "R_m": 19.9},
    {"t_ms": 0.38, "R_m": 25.4},
    {"t_ms": 0.52, "R_m": 28.8},
    {"t_ms": 0.66, "R_m": 31.9},
    {"t_ms": 0.80, "R_m": 34.2},
    {"t_ms": 0.94, "R_m": 36.3},
    {"t_ms": 1.08, "R_m": 38.9},
    {"t_ms": 1.22, "R_m": 41.0},
    {"t_ms": 1.36, "R_m": 42.8},
    {"t_ms": 1.50, "R_m": 44.4},
    {"t_ms": 1.65, "R_m": 46.0},
    {"t_ms": 1.93, "R_m": 48.7},
    {"t_ms": 3.26, "R_m": 59.0},
    {"t_ms": 3.53, "R_m": 61.1},
    {"t_ms": 3.80, "R_m": 62.9},
    {"t_ms": 4.07, "R_m": 64.3},
    {"t_ms": 4.34, "R_m": 65.6},
    {"t_ms": 4.61, "R_m": 67.3},
    {"t_ms": 15.0, "R_m": 106.5},
    {"t_ms": 25.0, "R_m": 130.0},
    {"t_ms": 34.0, "R_m": 145.0},
    {"t_ms": 53.0, "R_m": 175.0},
    {"t_ms": 62.0, "R_m": 185.0},
]

# Device entities — nuclear device components at characteristic timescales.
# These connect the blast wave phenomenology back to the nuclear physics
# closures.  Their self-similarity channels are near ε because they
# do NOT follow the Taylor-Sedov power law — this is structurally correct.
DEVICE_DATA: list[dict[str, Any]] = [
    {
        "name": "Pu-239 Core",
        "t_s": 1e-6,  # Chain reaction timescale (~80 generations × 10 ns)
        "R_m": 0.042,  # Pit radius (~4.2 cm for 6.2 kg at 19.8 g/cm³)
        "v_m_s": 5000.0,  # Implosion velocity (m/s)
        "binding_fidelity": BINDING_PU239,
        "power_law_exponent": 0.0,
        "description": "Pu-239 fissile pit — 6.2 kg, 16.1% mass consumption",
    },
    {
        "name": "U-238 Tamper",
        "t_s": 2e-6,  # Inertial confinement timescale
        "R_m": 0.12,  # Tamper shell outer radius
        "v_m_s": 3000.0,  # Tamper driven inward by HE
        "binding_fidelity": BINDING_U238,
        "power_law_exponent": 0.0,
        "description": "Natural uranium tamper — inertial confinement + neutron reflection",
    },
    {
        "name": "Explosive Lens Array",
        "t_s": 10e-6,  # Detonation wave transit time
        "R_m": 0.70,  # Outer explosive assembly radius (~1.4 m diameter)
        "v_m_s": 7900.0,  # Detonation velocity of Composition B
        "binding_fidelity": EPSILON,  # Chemical, not nuclear
        "power_law_exponent": 0.0,
        "description": "32 explosive lenses — shaped charge implosion driver",
    },
]

# Reference entities — baselines for cross-comparison.
# Conventional HE provides a non-nuclear Taylor-Sedov baseline.
# D-T fusion provides the fission→fusion bridge.
REFERENCE_DATA: list[dict[str, Any]] = [
    {
        "name": "Conventional HE (1 ton TNT)",
        "t_s": 5e-3,  # 5 ms characteristic time
        "R_m": 9.6,  # Blast radius at 5 ms for 1 ton TNT
        "v_m_s": 770.0,  # Shock velocity at 5 ms
        "binding_fidelity": EPSILON,  # Chemical energy, not nuclear binding
        "power_law_exponent": 0.4,  # Follows Taylor-Sedov
        "description": "Non-nuclear blast — follows same Taylor-Sedov scaling",
    },
    {
        "name": "D-T Fusion Target (ICF)",
        "t_s": 1e-8,  # ICF implosion timescale ~10 ns
        "R_m": 5e-5,  # Hot spot radius ~50 μm
        "v_m_s": 350000.0,  # Implosion velocity 350 km/s
        "binding_fidelity": BINDING_DEUTERIUM,  # Deuterium BE/A / BE_peak
        "power_law_exponent": 0.0,  # Different physics
        "description": "D-T inertial confinement fusion — fission→fusion bridge entity",
    },
]


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — TRACE VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

CHANNEL_NAMES: list[str] = [
    "self_similarity",
    "energy_consistency",
    "mach_fidelity",
    "power_law_quality",
    "strong_shock",
    "density_jump",
    "overpressure_norm",
    "binding_fidelity",
]


def _clip(x: float) -> float:
    """Clip to [ε, 1−ε]."""
    return max(EPSILON, min(1.0 - EPSILON, x))


def build_trace(obs: TrinityObservables) -> tuple[np.ndarray, np.ndarray]:
    """Construct 8-channel trace vector and weights from observables.

    Returns (trace, weights) where trace ∈ [ε, 1−ε]⁸.

    Channel mappings:
        c₁ = min(ξ, 1/ξ) where ξ = R/(A_FIT·t^0.4)
        c₂ = min(E_local/E_extracted, E_extracted/E_local)
        c₃ = M / (M + M_REF)
        c₄ = max(ε, 1 − |α−0.4|/0.4)
        c₅ = M² / (M² + 1)
        c₆ = (ρ₁/ρ₀) / 6.0
        c₇ = log₁₀(Δp/p₀ + 1) / log₁₀(P_MAX + 1)
        c₈ = BE_fuel/A / BE_peak (direct from observables)
    """
    t = obs.time_s
    r = obs.radius_m
    M = obs.mach_number

    # c₁: self-similarity — conformance to Taylor-Sedov power law
    r_pred = A_FIT * (t**0.4) if t > 0 else A_FIT * EPSILON
    xi = r / r_pred if r_pred > 0 else EPSILON
    c1 = min(xi, 1.0 / xi) if xi > 0 else EPSILON

    # c₂: energy consistency — local yield vs extracted yield
    a_local = r / (t**0.4) if t > 0 else EPSILON
    e_local = RHO_AIR * a_local**5
    e_ratio = e_local / E_EXTRACTED_J if E_EXTRACTED_J > 0 else EPSILON
    c2 = min(e_ratio, 1.0 / e_ratio) if e_ratio > 0 else EPSILON

    # c₃: Mach fidelity — M/(M+M_REF) ∈ [0, 1)
    c3 = M / (M + MACH_REF) if M >= 0 else EPSILON

    # c₄: power law quality — 1 − |α−0.4|/0.4
    alpha = obs.power_law_exponent
    c4 = max(0.0, 1.0 - abs(alpha - 0.4) / 0.4)

    # c₅: strong shock validity — M²/(M²+1) ∈ [0, 1)
    c5 = M**2 / (M**2 + 1.0) if M >= 0 else EPSILON

    # c₆: density jump — ρ₁/ρ₀ normalized to strong shock limit (6.0)
    c6 = obs.density_ratio / DENSITY_JUMP_LIMIT

    # c₇: overpressure — log-normalized
    p_ratio = obs.overpressure_Pa / P_ATM if P_ATM > 0 else 0.0
    c7 = math.log10(p_ratio + 1.0) / P_RATIO_LOG_SCALE if p_ratio >= 0 else EPSILON

    # c₈: binding fidelity — direct from observables
    c8 = obs.binding_fidelity

    c = np.array(
        [_clip(c1), _clip(c2), _clip(c3), _clip(c4), _clip(c5), _clip(c6), _clip(c7), _clip(c8)],
        dtype=np.float64,
    )
    w = np.full(8, 1.0 / 8.0, dtype=np.float64)
    return c, w


def _compute_kernel(c: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    """Run the GCD kernel on a trace vector."""
    return compute_kernel_outputs(c, w, epsilon=EPSILON)


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Classify regime from kernel invariants."""
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — ENTITY BUILDERS
# ═══════════════════════════════════════════════════════════════════


def _compute_mach(v: float) -> float:
    """Compute Mach number from velocity."""
    return v / C_AIR


def _compute_overpressure(M: float) -> float:
    """Compute overpressure (Pa) from Mach number using Rankine-Hugoniot.

    Δp = p₀ × 2γ(M²−1)/(γ+1) = p₀ × 7(M²−1)/6  for γ=1.4
    """
    if M <= 1.0:
        return 0.0
    return P_ATM * 2.0 * GAMMA_AIR * (M**2 - 1.0) / (GAMMA_AIR + 1.0)


def _compute_density_ratio(M: float) -> float:
    """Compute post-shock density ratio from Mach number.

    ρ₁/ρ₀ = (γ+1)M² / ((γ−1)M² + 2) → 6.0 for M → ∞ (γ=1.4)
    """
    if M <= 0:
        return 1.0
    return (GAMMA_AIR + 1.0) * M**2 / ((GAMMA_AIR - 1.0) * M**2 + 2.0)


def _make_entity(
    name: str,
    category: str,
    obs: TrinityObservables,
    metadata: dict[str, Any] | None = None,
) -> TrinityEntity:
    """Build a single Trinity entity with full kernel analysis."""
    c, w = build_trace(obs)
    k = _compute_kernel(c, w)
    regime = _classify_regime(k["omega"], k["F"], k["S"], k["C"])

    return TrinityEntity(
        name=name,
        category=category,
        observables=obs,
        trace=c,
        weights=w,
        channels=list(CHANNEL_NAMES),
        F=k["F"],
        omega=k["omega"],
        S=k["S"],
        C=k["C"],
        kappa=k["kappa"],
        IC=k["IC"],
        gap=k["F"] - k["IC"],
        regime=regime,
        metadata=metadata or {},
    )


def build_fireball_entities() -> list[TrinityEntity]:
    """Build 24 entities from Mack photograph time-radius data.

    For each (t, R) pair, computes shock velocity, Mach number,
    overpressure, density ratio, energy extraction, and local
    power-law exponent from consecutive data points.
    """
    n = len(MACK_DATA)
    times_s = [d["t_ms"] * 1e-3 for d in MACK_DATA]
    radii_m = [d["R_m"] for d in MACK_DATA]

    # Shock velocity from self-similar solution: v = (2/5) × R/t
    velocities = [0.4 * R / t for t, R in zip(times_s, radii_m, strict=True)]
    machs = [_compute_mach(v) for v in velocities]

    # Local power-law exponents from consecutive data points
    exponents: list[float] = []
    for i in range(n):
        if i == 0:
            alpha = math.log(radii_m[1] / radii_m[0]) / math.log(times_s[1] / times_s[0])
        elif i == n - 1:
            alpha = math.log(radii_m[-1] / radii_m[-2]) / math.log(times_s[-1] / times_s[-2])
        else:
            alpha = math.log(radii_m[i + 1] / radii_m[i - 1]) / math.log(times_s[i + 1] / times_s[i - 1])
        exponents.append(alpha)

    entities: list[TrinityEntity] = []
    for i in range(n):
        t_s = times_s[i]
        R_m = radii_m[i]
        M = machs[i]

        obs = TrinityObservables(
            time_s=t_s,
            radius_m=R_m,
            velocity_m_s=velocities[i],
            mach_number=M,
            overpressure_Pa=_compute_overpressure(M),
            energy_local_J=RHO_AIR * (R_m / t_s**0.4) ** 5,
            density_ratio=_compute_density_ratio(M),
            binding_fidelity=BINDING_PU239,
            power_law_exponent=exponents[i],
        )

        entity = _make_entity(
            name=f"t={MACK_DATA[i]['t_ms']:.2f}ms R={MACK_DATA[i]['R_m']:.1f}m",
            category="fireball",
            obs=obs,
            metadata={
                "t_ms": MACK_DATA[i]["t_ms"],
                "R_m": MACK_DATA[i]["R_m"],
                "v_shock_m_s": velocities[i],
                "mach": M,
                "xi": R_m / (A_FIT * t_s**0.4),
            },
        )
        entities.append(entity)

    return entities


def build_device_entities() -> list[TrinityEntity]:
    """Build 3 device physics entities.

    These represent the nuclear device components at their
    characteristic timescales.  Their self-similarity channels
    are near ε because they do not follow the Taylor-Sedov
    power law — this is structurally correct and creates
    heterogeneity that the kernel detects.
    """
    entities: list[TrinityEntity] = []
    for d in DEVICE_DATA:
        t_s = d["t_s"]
        R_m = d["R_m"]
        v = d["v_m_s"]
        M = _compute_mach(v)

        obs = TrinityObservables(
            time_s=t_s,
            radius_m=R_m,
            velocity_m_s=v,
            mach_number=M,
            overpressure_Pa=_compute_overpressure(M),
            energy_local_J=RHO_AIR * (R_m / t_s**0.4) ** 5,
            density_ratio=_compute_density_ratio(M),
            binding_fidelity=d["binding_fidelity"],
            power_law_exponent=d["power_law_exponent"],
        )

        entity = _make_entity(
            name=d["name"],
            category="device",
            obs=obs,
            metadata={"description": d["description"]},
        )
        entities.append(entity)

    return entities


def build_reference_entities() -> list[TrinityEntity]:
    """Build 2 reference entities (conventional HE and D-T fusion).

    Conventional HE provides a non-nuclear baseline that also
    follows Taylor-Sedov.  D-T fusion provides the explicit
    bridge from fission to fusion physics.
    """
    entities: list[TrinityEntity] = []
    for d in REFERENCE_DATA:
        t_s = d["t_s"]
        R_m = d["R_m"]
        v = d["v_m_s"]
        M = _compute_mach(v)

        obs = TrinityObservables(
            time_s=t_s,
            radius_m=R_m,
            velocity_m_s=v,
            mach_number=M,
            overpressure_Pa=_compute_overpressure(M),
            energy_local_J=RHO_AIR * (R_m / t_s**0.4) ** 5,
            density_ratio=_compute_density_ratio(M),
            binding_fidelity=d["binding_fidelity"],
            power_law_exponent=d["power_law_exponent"],
        )

        entity = _make_entity(
            name=d["name"],
            category="reference",
            obs=obs,
            metadata={"description": d["description"]},
        )
        entities.append(entity)

    return entities


def build_all_entities() -> list[TrinityEntity]:
    """Build all 29 Trinity entities."""
    return build_fireball_entities() + build_device_entities() + build_reference_entities()


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — THEOREM PROOFS
# ═══════════════════════════════════════════════════════════════════


def _rank(values: list[float]) -> list[float]:
    """Compute ranks for Spearman correlation."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    for rank_val, (orig_idx, _) in enumerate(indexed):
        ranks[orig_idx] = float(rank_val)
    return ranks


def _prove_T_TB_1(fireball: list[TrinityEntity]) -> dict[str, Any]:
    """T-TB-1: Self-Similar Conformance.

    The self-similarity parameter ξ = R/(A_FIT·t^0.4) should be
    close to 1.0 for entities in the strong shock phase (t > 0.3 ms).
    The median |ξ − 1| across these entities should be < 0.05.
    """
    strong_shock = [e for e in fireball if e.observables.time_s > 3e-4]
    xi_vals = [e.metadata.get("xi", 0.0) for e in strong_shock]
    deviations = [abs(xi - 1.0) for xi in xi_vals]

    median_dev = float(np.median(deviations))
    max_dev = max(deviations) if deviations else 1.0
    n_conformant = sum(1 for d in deviations if d < 0.05)

    conformant = median_dev < 0.05
    majority = n_conformant > len(strong_shock) // 2

    return {
        "id": "T-TB-1",
        "name": "Self-Similar Conformance",
        "proven": conformant and majority,
        "tests": 2,
        "passed": int(conformant) + int(majority),
        "median_deviation": median_dev,
        "max_deviation": max_dev,
        "n_conformant": n_conformant,
        "n_strong_shock": len(strong_shock),
    }


def _prove_T_TB_2(fireball: list[TrinityEntity]) -> dict[str, Any]:
    """T-TB-2: Yield Self-Consistency.

    The locally extracted yield E_local = ρ₀·(R/t^0.4)⁵ should
    vary by less than 20% (coefficient of variation) across
    the strong shock phase (t > 0.3 ms).
    """
    strong_shock = [e for e in fireball if e.observables.time_s > 3e-4]
    E_locals = [e.observables.energy_local_J for e in strong_shock]

    mean_E = float(np.mean(E_locals))
    std_E = float(np.std(E_locals))
    cv = std_E / mean_E if mean_E > 0 else 1.0
    E_kt = mean_E / KT_TO_J

    low_variance = cv < 0.20
    yield_close = abs(E_kt - YIELD_KT) / YIELD_KT < 0.25

    return {
        "id": "T-TB-2",
        "name": "Yield Self-Consistency",
        "proven": low_variance and yield_close,
        "tests": 2,
        "passed": int(low_variance) + int(yield_close),
        "cv": cv,
        "mean_E_kt": E_kt,
        "official_kt": YIELD_KT,
        "deviation_pct": abs(E_kt - YIELD_KT) / YIELD_KT * 100,
    }


def _prove_T_TB_3(fireball: list[TrinityEntity]) -> dict[str, Any]:
    """T-TB-3: Shock Weakening Transition.

    IC should be lower for late-time entities (t > 10 ms) than
    mid-phase entities (0.5 ms < t < 5 ms) because the weakening
    shock drives the strong_shock and density_jump channels toward
    lower values, increasing heterogeneity.
    """
    mid = [e for e in fireball if 5e-4 < e.observables.time_s < 5e-3]
    late = [e for e in fireball if e.observables.time_s > 1e-2]

    if not mid or not late:
        return {
            "id": "T-TB-3",
            "name": "Shock Weakening Transition",
            "proven": False,
            "tests": 1,
            "passed": 0,
            "reason": "Insufficient data in mid or late phase",
        }

    mid_ic = float(np.mean([e.IC for e in mid]))
    late_ic = float(np.mean([e.IC for e in late]))

    mid_gap = float(np.mean([e.gap for e in mid]))
    late_gap = float(np.mean([e.gap for e in late]))

    ic_drops = mid_ic > late_ic
    gap_increases = late_gap > mid_gap

    return {
        "id": "T-TB-3",
        "name": "Shock Weakening Transition",
        "proven": ic_drops and gap_increases,
        "tests": 2,
        "passed": int(ic_drops) + int(gap_increases),
        "mid_IC": mid_ic,
        "late_IC": late_ic,
        "mid_gap": mid_gap,
        "late_gap": late_gap,
    }


def _prove_T_TB_4(fireball: list[TrinityEntity]) -> dict[str, Any]:
    """T-TB-4: Phase Boundary F-Split.

    F should be higher in the strong shock phase (0.5 < t < 5 ms)
    than in both the early radiative phase (t < 0.3 ms) and the
    late weakening phase (t > 10 ms).
    """
    early = [e for e in fireball if e.observables.time_s < 3e-4]
    mid = [e for e in fireball if 5e-4 < e.observables.time_s < 5e-3]
    late = [e for e in fireball if e.observables.time_s > 1e-2]

    if not early or not mid or not late:
        return {
            "id": "T-TB-4",
            "name": "Phase Boundary F-Split",
            "proven": False,
            "tests": 1,
            "passed": 0,
            "reason": "Insufficient data in one or more phases",
        }

    early_F = float(np.mean([e.F for e in early]))
    mid_F = float(np.mean([e.F for e in mid]))
    late_F = float(np.mean([e.F for e in late]))

    mid_higher_than_early = mid_F > early_F
    mid_higher_than_late = mid_F > late_F

    return {
        "id": "T-TB-4",
        "name": "Phase Boundary F-Split",
        "proven": mid_higher_than_early and mid_higher_than_late,
        "tests": 2,
        "passed": int(mid_higher_than_early) + int(mid_higher_than_late),
        "early_F": early_F,
        "mid_F": mid_F,
        "late_F": late_F,
    }


def _prove_T_TB_5(fireball: list[TrinityEntity]) -> dict[str, Any]:
    """T-TB-5: Power Law Quality.

    The local exponent α = d(ln R)/d(ln t) should be within 15%
    of the theoretical 2/5 = 0.4 for at least 60% of mid-range
    entities (0.5 ms < t < 10 ms).
    """
    mid = [e for e in fireball if 5e-4 < e.observables.time_s < 1e-2]
    alphas = [e.observables.power_law_exponent for e in mid]

    within_15pct = sum(1 for a in alphas if abs(a - 0.4) / 0.4 < 0.15)
    frac = within_15pct / len(alphas) if alphas else 0.0

    median_alpha = float(np.median(alphas)) if alphas else 0.0

    quality_met = frac >= 0.60
    median_close = abs(median_alpha - 0.4) / 0.4 < 0.10

    return {
        "id": "T-TB-5",
        "name": "Power Law Quality",
        "proven": quality_met and median_close,
        "tests": 2,
        "passed": int(quality_met) + int(median_close),
        "fraction_within_15pct": frac,
        "median_alpha": median_alpha,
        "n_mid_range": len(mid),
    }


def _prove_T_TB_6(fireball: list[TrinityEntity]) -> dict[str, Any]:
    """T-TB-6: Taylor Yield Extraction.

    The extracted yield (from median A_local across fireball entities)
    should match the official declassified value of 21 kt within 25%.
    """
    A_locals = [e.observables.radius_m / (e.observables.time_s**0.4) for e in fireball if e.observables.time_s > 0]
    A_med = float(np.median(A_locals))
    E_med = RHO_AIR * A_med**5
    yield_kt = E_med / KT_TO_J
    deviation = abs(yield_kt - YIELD_KT) / YIELD_KT

    within_bounds = deviation < 0.25
    within_selby = abs(yield_kt - YIELD_SELBY_KT) / YIELD_SELBY_KT < 0.30

    return {
        "id": "T-TB-6",
        "name": "Taylor Yield Extraction",
        "proven": within_bounds and within_selby,
        "tests": 2,
        "passed": int(within_bounds) + int(within_selby),
        "extracted_kt": yield_kt,
        "official_kt": YIELD_KT,
        "selby_kt": YIELD_SELBY_KT,
        "deviation_from_official_pct": deviation * 100,
        "A_median": A_med,
    }


def _prove_T_TB_7(fireball: list[TrinityEntity]) -> dict[str, Any]:
    """T-TB-7: Velocity Monotonicity.

    Shock velocity v = (2/5)R/t should be monotonically decreasing
    with time.  At least 90% of consecutive pairs should be ordered.
    """
    v_vals = [e.observables.velocity_m_s for e in fireball]
    n = len(v_vals)

    n_monotone = sum(1 for i in range(n - 1) if v_vals[i] >= v_vals[i + 1] - 1e-6)
    frac = n_monotone / (n - 1) if n > 1 else 0.0

    mostly_monotone = frac >= 0.90
    first_faster = v_vals[0] > v_vals[-1]

    return {
        "id": "T-TB-7",
        "name": "Velocity Monotonicity",
        "proven": mostly_monotone and first_faster,
        "tests": 2,
        "passed": int(mostly_monotone) + int(first_faster),
        "monotone_fraction": frac,
        "v_first": v_vals[0],
        "v_last": v_vals[-1],
        "v_ratio": v_vals[0] / v_vals[-1] if v_vals[-1] > 0 else float("inf"),
    }


def _prove_T_TB_8(
    fireball: list[TrinityEntity],
    references: list[TrinityEntity],
) -> dict[str, Any]:
    """T-TB-8: Fission-Fusion Bridge.

    The kernel discriminates fission (Trinity) from fusion (D-T)
    through composite integrity.  The D-T reference has very different
    binding_fidelity from fission → creating a distinctive IC pattern.
    The conventional HE reference has binding_fidelity ≈ ε → even
    lower IC due to geometric slaughter.
    """
    dt_refs = [e for e in references if "D-T" in e.name]
    he_refs = [e for e in references if "Conventional" in e.name]

    if not dt_refs or not he_refs:
        return {
            "id": "T-TB-8",
            "name": "Fission-Fusion Bridge",
            "proven": False,
            "tests": 1,
            "passed": 0,
            "reason": "Missing D-T or HE reference entity",
        }

    mid_fireball = [e for e in fireball if 5e-4 < e.observables.time_s < 5e-3]
    if not mid_fireball:
        return {
            "id": "T-TB-8",
            "name": "Fission-Fusion Bridge",
            "proven": False,
            "tests": 1,
            "passed": 0,
            "reason": "No mid-phase fireball entities",
        }

    fission_F = float(np.mean([e.F for e in mid_fireball]))
    dt_F = dt_refs[0].F
    he_F = he_refs[0].F

    # Fission entities should be discriminated from both references
    fission_vs_dt = abs(fission_F - dt_F) > 0.01
    fission_vs_he = abs(fission_F - he_F) > 0.01

    # Binding fidelity channel discriminates the physics
    fission_binding = fireball[0].observables.binding_fidelity
    dt_binding = dt_refs[0].observables.binding_fidelity
    binding_gap = fission_binding - dt_binding

    binding_discriminated = binding_gap > 0.5  # Pu-239 is much closer to peak than D

    return {
        "id": "T-TB-8",
        "name": "Fission-Fusion Bridge",
        "proven": fission_vs_dt and fission_vs_he and binding_discriminated,
        "tests": 3,
        "passed": int(fission_vs_dt) + int(fission_vs_he) + int(binding_discriminated),
        "fission_F": fission_F,
        "dt_F": dt_F,
        "he_F": he_F,
        "fission_binding": fission_binding,
        "dt_binding": dt_binding,
        "binding_gap": binding_gap,
        "insight": (
            f"Pu-239 binding fidelity = {fission_binding:.3f} "
            f"(close to iron peak), "
            f"D binding fidelity = {dt_binding:.3f} "
            f"(far from peak → larger energetic opportunity for fusion)"
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — NARRATIVE GENERATION
# ═══════════════════════════════════════════════════════════════════


def generate_narrative(
    fireball: list[TrinityEntity],
    devices: list[TrinityEntity],
    references: list[TrinityEntity],
    theorems: dict[str, dict[str, Any]],
) -> str:
    """Generate a five-word Canon narrative for the Trinity analysis.

    Uses: Drift · Fidelity · Roughness · Return · Integrity
    """
    all_entities = fireball + devices + references
    all_F = [e.F for e in all_entities]
    all_IC = [e.IC for e in all_entities]

    mid = [e for e in fireball if 5e-4 < e.observables.time_s < 5e-3]
    mid_F = float(np.mean([e.F for e in mid])) if mid else 0.0

    n_proven = sum(1 for t in theorems.values() if t.get("proven"))

    lines = [
        "═" * 65,
        "TRINITY BLAST WAVE — CANON NARRATIVE",
        "═" * 65,
        "",
        "DRIFT: The fireball expands as R ∝ t^(2/5) across four decades",
        f"  of time (0.1 ms → 62 ms).  ω ranges from {min(e.omega for e in fireball):.3f}",
        f"  to {max(e.omega for e in fireball):.3f} across the 24 Mack photograph entities.",
        "  Device entities (Pu-239 core, tamper) show ω > 0.3 — they are",
        "  structurally in the Collapse regime because they do not conform",
        "  to the Taylor-Sedov self-similar solution.",
        "",
        "FIDELITY: The self-similar power law IS the fidelity structure.",
        f"  Mid-phase entities (0.5–5 ms) achieve F ≈ {mid_F:.3f}.",
        f"  The Taylor-Sedov solution yields E ≈ {E_EXTRACTED_J / KT_TO_J:.1f} kt,",
        f"  consistent with the official {YIELD_KT} kt (deviation < 5%).",
        "",
        "ROUGHNESS: Three sources of roughness detected:",
        "  (1) Early phase (t < 0.3 ms): radiation-dominated → ξ < 1",
        "  (2) Late phase (t > 15 ms): shock weakening → M → 1",
        "  (3) Device entities: non-blast-wave physics → ξ ≈ ε",
        "",
        "RETURN: The self-similar solution is the RETURN structure.",
        "  At every scale, the same physics returns: R ∝ t^(2/5).",
        "  This is Axiom-0 in blast wave form — only what returns is real,",
        "  and the power law returns identically at every time decade.",
        "",
        "INTEGRITY: Multiplicative coherence reveals phase structure.",
        f"  Mean IC = {float(np.mean(all_IC)):.4f}, Mean F = {float(np.mean(all_F)):.4f}",
        f"  Mean gap Δ = {float(np.mean(all_F)) - float(np.mean(all_IC)):.4f}",
        f"  Theorems proven: {n_proven}/8",
        "",
        "FISSION → FUSION BRIDGE:",
        f"  Pu-239 binding fidelity = {BINDING_PU239:.3f} (close to iron peak)",
        f"  D (fusion fuel) binding fidelity = {BINDING_DEUTERIUM:.3f} (far from peak)",
        f"  The gap {BINDING_PU239 - BINDING_DEUTERIUM:.3f} measures the structural",
        "  distance between fission and fusion in kernel space.",
        "  Both converge on the iron peak (Ni-62, BE/A = 8.7945 MeV)",
        "  — fission from above (A=239 → peak), fusion from below (A=2 → peak).",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — FISSION-FUSION BRIDGE
# ═══════════════════════════════════════════════════════════════════


@dataclass
class FissionFusionBridge:
    """Data structure connecting fission (Trinity) to fusion (D-T).

    The iron peak (Ni-62, BE/A = 8.7945 MeV/nucleon) is the fixed point
    of nuclear stability.  Fission approaches from above (A > 62), fusion
    from below (A < 62).  Both are return structures — they return toward
    the peak of binding energy per nucleon.

    The Lawson criterion for fusion (nτ_E T ≥ 1.5×10²¹ m⁻³ keV s) is
    the GCD return condition: if the triple product is met, τ_R is finite
    and the fusion plasma returns to burning.  If not, τ_R = ∞_rec.

    Cross-references:
        closures/nuclear_physics/double_sided_collapse.py — iron peak convergence
        closures/nuclear_physics/nuclide_binding.py — binding energy mapping
        closures/nuclear_physics/fissility.py — fissility parameter
    """

    # Fission side (Trinity / Pu-239)
    fission_fuel_Z: int = PU239_Z
    fission_fuel_A: int = PU239_A
    fission_fuel_BE_per_A: float = PU239_BE_PER_A  # 7.560
    fission_binding_deficit: float = 1.0 - PU239_BE_PER_A / BE_PEAK_REF  # 0.140
    fission_direction: str = "←Fe (heavy → peak, A=239 → 62)"
    fission_yield_kt: float = YIELD_KT
    fission_efficiency: float = FISSION_EFFICIENCY
    fission_Q_per_nucleon_MeV: float = Q_FISSION_MEV / PU239_A  # ~0.84

    # Fusion side (D-T reaction)
    fusion_fuel_Z: int = 1
    fusion_fuel_A: int = 2  # Deuterium
    fusion_fuel_BE_per_A: float = 1.112  # MeV/nucleon
    fusion_binding_deficit: float = 1.0 - 1.112 / BE_PEAK_REF  # 0.874
    fusion_product_Z: int = 2  # He-4
    fusion_product_A: int = 4
    fusion_product_BE_per_A: float = 7.074  # MeV/nucleon
    fusion_Q_MeV: float = 17.6  # D + T → He-4 + n + 17.6 MeV
    fusion_Q_per_nucleon_MeV: float = 17.6 / 5  # 3.52 MeV per input nucleon
    fusion_direction: str = "→Fe (light → peak, A=2 → 62)"
    fusion_lawson_nTauE: float = 1.5e21  # m⁻³ keV s

    # Iron peak (convergence point)
    peak_Z: int = 28
    peak_A: int = A_PEAK
    peak_BE_per_A: float = BE_PEAK_REF

    @property
    def energy_ratio(self) -> float:
        """Ratio of fusion to fission energy per nucleon.

        Fusion releases ~4× more energy per nucleon than fission.
        """
        return self.fusion_Q_per_nucleon_MeV / self.fission_Q_per_nucleon_MeV

    @property
    def deficit_ratio(self) -> float:
        """Ratio of fusion to fission binding deficit.

        Fusion fuel is ~6× further from the iron peak than fission fuel.
        """
        return self.fusion_binding_deficit / self.fission_binding_deficit


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TrinityAnalysisResult:
    """Complete result of the Trinity blast wave analysis."""

    fireball_entities: list[TrinityEntity]
    device_entities: list[TrinityEntity]
    reference_entities: list[TrinityEntity]
    all_entities: list[TrinityEntity]
    theorem_results: dict[str, dict[str, Any]]
    tier1_violations: int
    narrative: str
    bridge: FissionFusionBridge
    summary: dict[str, Any]


def run_full_analysis() -> TrinityAnalysisResult:
    """Run the complete Trinity blast wave kernel analysis.

    Returns a TrinityAnalysisResult with all entities, theorems,
    narrative, and fission-fusion bridge.
    """
    fireball = build_fireball_entities()
    devices = build_device_entities()
    references = build_reference_entities()
    all_entities = fireball + devices + references

    # Verify Tier-1 identities
    tier1_violations = 0
    for e in all_entities:
        if abs(e.F + e.omega - 1.0) > 1e-12:
            tier1_violations += 1
        if e.IC > e.F + 1e-12:
            tier1_violations += 1

    # Prove theorems
    theorems = {
        "T-TB-1": _prove_T_TB_1(fireball),
        "T-TB-2": _prove_T_TB_2(fireball),
        "T-TB-3": _prove_T_TB_3(fireball),
        "T-TB-4": _prove_T_TB_4(fireball),
        "T-TB-5": _prove_T_TB_5(fireball),
        "T-TB-6": _prove_T_TB_6(fireball),
        "T-TB-7": _prove_T_TB_7(fireball),
        "T-TB-8": _prove_T_TB_8(fireball, references),
    }

    n_proven = sum(1 for t in theorems.values() if t.get("proven"))

    # Generate narrative
    narrative = generate_narrative(fireball, devices, references, theorems)

    # Build bridge
    bridge = FissionFusionBridge()

    # Summary statistics
    all_F = [e.F for e in all_entities]
    all_IC = [e.IC for e in all_entities]
    summary = {
        "n_entities": len(all_entities),
        "n_fireball": len(fireball),
        "n_device": len(devices),
        "n_reference": len(references),
        "n_theorems_proven": n_proven,
        "n_theorems_total": len(theorems),
        "mean_F": float(np.mean(all_F)),
        "mean_IC": float(np.mean(all_IC)),
        "mean_gap": float(np.mean(all_F)) - float(np.mean(all_IC)),
        "F_range": (float(min(all_F)), float(max(all_F))),
        "IC_range": (float(min(all_IC)), float(max(all_IC))),
        "tier1_violations": tier1_violations,
        "yield_extracted_kt": E_EXTRACTED_J / KT_TO_J,
        "yield_official_kt": YIELD_KT,
        "A_FIT": A_FIT,
        "bridge_energy_ratio": bridge.energy_ratio,
        "bridge_deficit_ratio": bridge.deficit_ratio,
    }

    return TrinityAnalysisResult(
        fireball_entities=fireball,
        device_entities=devices,
        reference_entities=references,
        all_entities=all_entities,
        theorem_results=theorems,
        tier1_violations=tier1_violations,
        narrative=narrative,
        bridge=bridge,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — SELF-TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_full_analysis()
    print(result.narrative)
    print()
    print(f"Entities: {result.summary['n_entities']}")
    print(f"Theorems: {result.summary['n_theorems_proven']}/{result.summary['n_theorems_total']}")
    print(f"Tier-1 violations: {result.tier1_violations}")
    print(f"Yield extracted: {result.summary['yield_extracted_kt']:.1f} kt")
    print(f"Fusion/fission energy ratio: {result.bridge.energy_ratio:.1f}×")
    print(f"Fusion/fission deficit ratio: {result.bridge.deficit_ratio:.1f}×")
    print()
    for tid, tres in result.theorem_results.items():
        status = "PROVEN" if tres.get("proven") else "NOT PROVEN"
        print(f"  {tid}: {tres['name']} — {status} ({tres['passed']}/{tres['tests']})")
