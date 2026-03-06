"""
Butzbach Embedding — The Continuity Theory as a Degenerate Limit of GCD

This module formally implements Butzbach's (2026) "Continuity Theory of Life
and Memory" and demonstrates, through computation, that it is the n=1 channel
degenerate limit of the GCD kernel.

The strategy is strongman: implement his framework with full fidelity,
show WHY it works (because it inherits GCD structure), show WHAT it cannot
detect (channel-specific failure), and locate it precisely in the tier system.

═══════════════════════════════════════════════════════════════════════
BUTZBACH'S FRAMEWORK (implemented faithfully)
═══════════════════════════════════════════════════════════════════════

Definitions (Butzbach 2026):
    M  = Memory: persistent physical configuration (distinguishable
         internal structure that constrains future dynamics)
    C  = Continuity: operational return-to-viability statistic
    B  = System boundary (declared)
    V  = Viability/identity set (declared)
    Π  = Perturbation protocol (declared)
    τ  = Recovery window (declared)

    c(t; B, V, Π, τ) = Pr[X(t+τ) ∈ V | X(t) ∈ V, Π at t, internal-only]
    C(t+1) = C(t) · c(t)    (multiplicative cascade)
    P_Ω = (E_met + P_info) · C    (persistence functional)

═══════════════════════════════════════════════════════════════════════
GCD EMBEDDING (what makes it work)
═══════════════════════════════════════════════════════════════════════

The embedding proceeds by showing five exact correspondences:

    1. SCALAR LIMIT: c(t) ≡ IC when n=1, w₁=1
       Butzbach's scalar c is the integrity composite with one channel.
       Proof: IC = exp(Σ wᵢ ln cᵢ) = exp(1·ln c) = c.

    2. CASCADE = LOG-INTEGRITY ACCUMULATION:
       C(t) = Π c(k) for k=0..t-1
       ln C(t) = Σ ln c(k) = κ_accumulated
       This is exactly how κ compounds across seam steps.

    3. CONTRACT = PROTOCOL DECLARATION:
       Butzbach's (B, V, Π, τ) maps to GCD's Contract stop:
       B → system boundary (what is measured)
       V → viability predicate (regime gate)
       Π → perturbation protocol (test specification)
       τ → recovery window (τ_R)

    4. CLIFF = GEOMETRIC SLAUGHTER (n=1 restriction):
       Butzbach predicts a "low-M continuity cliff" but cannot locate it.
       GCD locates it: when ANY single channel cᵢ → ε, IC → 0 regardless
       of F. The scalar c cannot detect which channel failed.

    5. P_Ω = SEAM BUDGET (restricted):
       P_Ω = (E_met + P_info) · C
       maps to: Δκ = R·τ_R − (D_ω + D_C) with
       E_met + P_info → credit R·τ_R (resources available for return)
       C → continuity factor (what fraction survives)

═══════════════════════════════════════════════════════════════════════
WHAT BUTZBACH CANNOT DO (and GCD can)
═══════════════════════════════════════════════════════════════════════

    A. CHANNEL DECOMPOSITION: Butzbach's c is a scalar. He cannot
       determine WHY continuity failed — only THAT it failed. GCD's
       per-channel cᵢ reveals which channel killed IC.

    B. GEOMETRIC SLAUGHTER DETECTION: With n=1, there is no distinction
       between F and IC. But F ≠ IC when channels are heterogeneous.
       The heterogeneity gap Δ = F − IC is invisible to scalar c.

    C. REGIME CLASSIFICATION: Butzbach uses "living/dormant/non-living"
       as informal labels. GCD derives Stable/Watch/Collapse from
       frozen four-gate criterion on (ω, F, S, C).

    D. HISTORY CONDITIONING: C(t+1) = C(t)·c(t) assumes independent
       perturbations. GCD's κ tracks channel-wise history, allowing
       c(t | H_t) — adaptive memory.

    E. THREE-VALUED VERDICTS: Butzbach has binary (viable/not-viable).
       GCD has CONFORMANT/NONCONFORMANT/NON_EVALUABLE. The third state
       matters: some systems are not assessable, not just failed.

    F. CROSS-DOMAIN ROSETTA: Butzbach claims cross-domain
       comparability but provides no formal translation mechanism.
       GCD's Rosetta maps five words across lenses with auditable
       ledger backing.

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized → this module
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ────────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from umcp.frozen_contract import EPSILON  # noqa: E402
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

# ── Guard band ────────────────────────────────────────────────────
EPS = 1e-6  # Closure-level epsilon (above frozen ε = 1e-8)


# ═════════════════════════════════════════════════════════════════════
# SECTION 1: BUTZBACH'S FRAMEWORK (faithful implementation)
# ═════════════════════════════════════════════════════════════════════

BUTZBACH_CHANNELS: list[str] = ["continuity"]  # Scalar — n=1

N_BUTZBACH_CHANNELS = 1  # The degenerate case


@dataclass(frozen=True, slots=True)
class ProtocolDeclaration:
    """Butzbach's (B, V, Π, τ) protocol declaration.

    Maps to GCD Contract: freeze before evidence, declare all parameters.
    This is the Contract stop of the Spine, restricted to a single domain.
    """

    boundary: str  # B — system boundary description
    viability_set: str  # V — what counts as viable
    perturbation: str  # Π — standardized perturbation
    recovery_window: float  # τ — recovery time window (seconds)

    def to_contract(self) -> dict[str, Any]:
        """Express as a GCD Contract declaration."""
        return {
            "system_boundary": self.boundary,
            "viability_predicate": self.viability_set,
            "perturbation_protocol": self.perturbation,
            "recovery_window_s": self.recovery_window,
            "declared_before_evidence": True,  # Contract requires this
        }


@dataclass(frozen=True, slots=True)
class ContinuityTrial:
    """A single perturbation-recovery trial in Butzbach's framework."""

    system_name: str
    protocol: ProtocolDeclaration
    n_trials: int  # Number of perturbation-recovery attempts
    n_recovered: int  # Number that returned to V within τ

    @property
    def c(self) -> float:
        """Per-perturbation continuity capacity c(t; B, V, Π, τ).

        Butzbach: c = Pr[X(t+τ) ∈ V | X(t) ∈ V, Π at t, internal-only within B]
        Estimated as: n_recovered / n_trials
        """
        if self.n_trials == 0:
            return 0.0
        return self.n_recovered / self.n_trials


@dataclass(frozen=True, slots=True)
class PersistenceFunctional:
    """Butzbach's thermodynamic persistence functional P_Ω.

    P_Ω = (E_met + P_info) · C

    where:
        E_met  = maintenance/metabolic power (watts)
        P_info = informational maintenance power (watts, ≥ Landauer bound)
        C      = accumulated continuity (dimensionless)
    """

    e_met: float  # Maintenance power (W)
    p_info: float  # Informational maintenance power (W)
    continuity: float  # Accumulated C (dimensionless, [0, 1])

    @property
    def p_omega(self) -> float:
        """Persistence functional P_Ω (watts)."""
        return (self.e_met + self.p_info) * self.continuity

    @property
    def landauer_floor(self) -> float:
        """Landauer lower bound for P_info at 300K: kT ln 2 per bit/s."""
        kT_300K = 4.114e-21  # Boltzmann constant × 300K (joules)
        return kT_300K * math.log(2)  # Per bit erasure


def butzbach_cascade(c_values: list[float] | np.ndarray) -> np.ndarray:
    """Multiplicative cascade: C(t+1) = C(t)·c(t).

    Butzbach's core composition rule. Returns the accumulated continuity
    at each step.

    Parameters
    ----------
    c_values : sequence of per-step continuity values c(t) ∈ [0, 1]

    Returns
    -------
    np.ndarray : accumulated continuity C(t) at each step, starting from C(0) = 1.
    """
    c_arr = np.asarray(c_values, dtype=np.float64)
    # C(0) = 1, C(t) = product of c(0) through c(t-1)
    accumulated = np.cumprod(c_arr)
    return accumulated


# ═════════════════════════════════════════════════════════════════════
# SECTION 2: DEGENERATE LIMIT PROOF — c IS IC WHEN n=1
# ═════════════════════════════════════════════════════════════════════


def prove_scalar_limit(c_value: float) -> dict[str, Any]:
    """Prove that Butzbach's scalar c equals GCD's IC when n=1.

    Proof by computation:
        IC = exp(κ) where κ = Σ wᵢ ln(cᵢ,ε)
        When n=1, w₁=1: κ = ln(c), IC = exp(ln(c)) = c

    Also shows that F = c, ω = 1 - c, and the heterogeneity gap Δ = 0
    (because with one channel, there is no heterogeneity).
    """
    c_safe = max(c_value, EPSILON)
    c_arr = np.array([c_safe])
    w_arr = np.array([1.0])

    kernel = compute_kernel_outputs(c_arr, w_arr, epsilon=EPSILON)

    # The proof: these must be equal
    ic_value = kernel["IC"]
    f_value = kernel["F"]
    omega_value = kernel["omega"]
    gap = kernel["heterogeneity_gap"]

    return {
        # Butzbach's values
        "butzbach_c": c_value,
        "butzbach_omega": 1.0 - c_value,
        # GCD kernel values (n=1)
        "gcd_F": f_value,
        "gcd_omega": omega_value,
        "gcd_IC": ic_value,
        "gcd_kappa": kernel["kappa"],
        "gcd_S": kernel["S"],
        "gcd_C": kernel["C"],
        "gcd_regime": kernel["regime"],
        # The proof residuals
        "residual_c_vs_IC": abs(c_safe - ic_value),
        "residual_c_vs_F": abs(c_safe - f_value),
        "residual_gap": abs(gap),  # Must be 0 at n=1
        "duality_residual": abs(f_value + omega_value - 1.0),
        # Diagnosis
        "is_degenerate_limit": True,
        "n_channels": 1,
        "channel_decomposition_available": False,
    }


def prove_cascade_is_log_integrity(c_sequence: list[float]) -> dict[str, Any]:
    """Prove that Butzbach's cascade C(t) = Π c(k) equals accumulated κ.

    ln C(t) = Σ ln c(k) = κ_accumulated

    This is exactly how log-integrity compounds across seam steps.
    The multiplicative cascade in Butzbach's framework is the exponential
    of the additive κ accumulation in GCD.
    """
    c_arr = np.asarray(c_sequence, dtype=np.float64)
    c_safe = np.clip(c_arr, EPSILON, 1.0 - EPSILON)

    # Butzbach's cascade
    cascade = butzbach_cascade(c_safe)

    # GCD's κ accumulation
    kappa_steps = np.log(c_safe)
    kappa_accumulated = np.cumsum(kappa_steps)
    ic_from_kappa = np.exp(kappa_accumulated)

    # These must be identical
    residuals = np.abs(cascade - ic_from_kappa)

    return {
        "c_sequence": c_sequence,
        "butzbach_cascade": cascade.tolist(),
        "gcd_kappa_accumulated": kappa_accumulated.tolist(),
        "gcd_ic_from_kappa": ic_from_kappa.tolist(),
        "max_residual": float(np.max(residuals)),
        "mean_residual": float(np.mean(residuals)),
        "identity_verified": bool(np.max(residuals) < 1e-12),
    }


# ═════════════════════════════════════════════════════════════════════
# SECTION 3: WHAT BUTZBACH CANNOT SEE — CHANNEL DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SystemProfile:
    """A system analyzed through both Butzbach's scalar and GCD's channels.

    Butzbach sees only the aggregate c. GCD decomposes into n channels,
    revealing WHERE fidelity concentrates and WHERE it breaks down.
    """

    name: str
    category: str  # biological, engineered, inert, dormant
    substrate: str  # Description of physical substrate

    # Multi-channel profile (GCD view)
    channels: tuple[float, ...]  # n-channel coherence values
    channel_labels: tuple[str, ...]  # Names for each channel
    weights: tuple[float, ...] | None  # Channel weights (None → equal)

    # Butzbach metadata
    protocol: ProtocolDeclaration | None  # Declared protocol
    e_met: float  # Maintenance power (W)
    p_info: float  # Informational maintenance power (W)

    def aggregate_c(self) -> float:
        """Butzbach's scalar c — the arithmetic mean of channels.

        This is what Butzbach measures: a single return-to-viability
        probability. It corresponds to F (fidelity) in GCD, NOT to IC.
        """
        w = self._weights()
        return float(np.dot(w, self.channels))

    def _weights(self) -> np.ndarray:
        if self.weights is not None:
            return np.asarray(self.weights, dtype=np.float64)
        n = len(self.channels)
        return np.ones(n, dtype=np.float64) / n


# ── System catalog ────────────────────────────────────────────────
# 20 systems spanning Butzbach's claimed scope:
# biological, engineered, dormant, inert

_CHANNEL_LABELS_BIO = (
    "membrane_integrity",
    "metabolic_activity",
    "genetic_fidelity",
    "repair_capacity",
    "energy_homeostasis",
    "signal_transduction",
    "reproductive_competence",
    "environmental_sensing",
)

_CHANNEL_LABELS_ENG = (
    "data_retention",
    "error_correction",
    "power_stability",
    "thermal_management",
    "signal_integrity",
    "write_endurance",
    "read_fidelity",
    "interface_coherence",
)

_CHANNEL_LABELS_PHYS = (
    "structural_persistence",
    "defect_tolerance",
    "thermal_stability",
    "chemical_inertness",
    "crystallographic_order",
    "surface_coherence",
    "bulk_homogeneity",
    "isotopic_stability",
)

# Default protocol for biological systems
_BIO_PROTOCOL = ProtocolDeclaration(
    boundary="Cell membrane",
    viability_set="Metabolically active, membrane intact",
    perturbation="Standardized osmotic/thermal stress",
    recovery_window=3600.0,  # 1 hour
)

# Default protocol for engineered systems
_ENG_PROTOCOL = ProtocolDeclaration(
    boundary="Device package boundary",
    viability_set="Data retrievable within spec",
    perturbation="Power cycle + thermal excursion",
    recovery_window=60.0,  # 1 minute
)

# Default protocol for inert/dormant systems
_INERT_PROTOCOL = ProtocolDeclaration(
    boundary="Crystal/material surface",
    viability_set="Configuration distinguishable from background",
    perturbation="Environmental exposure (weathering, radiation)",
    recovery_window=86400.0,  # 1 day
)

SYSTEMS: tuple[SystemProfile, ...] = (
    # ── Biological: active ──
    SystemProfile(
        name="E. coli (log phase)",
        category="biological",
        substrate="Gram-negative bacterium",
        channels=(0.95, 0.98, 0.92, 0.88, 0.94, 0.85, 0.96, 0.80),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=1e-12,  # ~1 pW metabolic
        p_info=1e-15,
    ),
    SystemProfile(
        name="Human neuron",
        category="biological",
        substrate="Mammalian neural cell",
        channels=(0.90, 0.70, 0.85, 0.95, 0.80, 0.92, 0.01, 0.88),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=1e-9,  # ~1 nW
        p_info=1e-12,
    ),
    SystemProfile(
        name="Hepatocyte",
        category="biological",
        substrate="Mammalian liver cell",
        channels=(0.92, 0.95, 0.88, 0.90, 0.93, 0.87, 0.85, 0.75),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=5e-10,
        p_info=5e-13,
    ),
    SystemProfile(
        name="T cell (activated)",
        category="biological",
        substrate="Immune lymphocyte",
        channels=(0.88, 0.92, 0.80, 0.93, 0.85, 0.95, 0.90, 0.82),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=2e-11,
        p_info=2e-14,
    ),
    SystemProfile(
        name="Cancer cell (HeLa)",
        category="biological",
        substrate="Immortal carcinoma line",
        channels=(0.85, 0.95, 0.40, 0.60, 0.90, 0.70, 0.99, 0.50),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=3e-11,
        p_info=3e-14,
    ),
    # ── Biological: dormant ──
    SystemProfile(
        name="Lotus seed (1000 yr)",
        category="dormant",
        substrate="Dried embryo in seed coat",
        channels=(0.10, 0.01, 0.98, 0.95, 0.05, 0.02, 0.70, 0.01),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=1e-18,  # Near zero metabolic
        p_info=1e-20,
    ),
    SystemProfile(
        name="Tardigrade (tun state)",
        category="dormant",
        substrate="Cryptobiotic ecdysozoan",
        channels=(0.70, 0.01, 0.90, 0.85, 0.02, 0.01, 0.60, 0.01),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=1e-16,
        p_info=1e-18,
    ),
    SystemProfile(
        name="Bacterial endospore",
        category="dormant",
        substrate="Bacillus spore",
        channels=(0.50, 0.01, 0.99, 0.92, 0.01, 0.01, 0.80, 0.01),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=1e-19,
        p_info=1e-21,
    ),
    # ── Engineered: memory devices ──
    SystemProfile(
        name="Flash NAND cell",
        category="engineered",
        substrate="Floating-gate transistor",
        channels=(0.999, 0.95, 0.99, 0.90, 0.98, 0.90, 0.999, 0.95),
        channel_labels=_CHANNEL_LABELS_ENG,
        weights=None,
        protocol=_ENG_PROTOCOL,
        e_met=1e-6,  # ~1 μW per cell active
        p_info=1e-9,
    ),
    SystemProfile(
        name="Ferroelectric capacitor (HfO₂)",
        category="engineered",
        substrate="Hafnia-based ferroelectric",
        channels=(0.98, 0.85, 0.92, 0.88, 0.95, 0.70, 0.96, 0.90),
        channel_labels=_CHANNEL_LABELS_ENG,
        weights=None,
        protocol=_ENG_PROTOCOL,
        e_met=1e-7,
        p_info=1e-10,
    ),
    SystemProfile(
        name="Cryogenic quantum memory (4K)",
        category="engineered",
        substrate="Superconducting transmon + cavity",
        channels=(0.95, 0.99, 0.70, 0.60, 0.92, 0.80, 0.98, 0.85),
        channel_labels=_CHANNEL_LABELS_ENG,
        weights=None,
        protocol=_ENG_PROTOCOL,
        e_met=10.0,  # ~10W refrigeration
        p_info=1e-6,
    ),
    SystemProfile(
        name="Biodegradable synapse",
        category="engineered",
        substrate="Organic electrochemical device",
        channels=(0.80, 0.50, 0.65, 0.70, 0.75, 0.30, 0.82, 0.60),
        channel_labels=_CHANNEL_LABELS_ENG,
        weights=None,
        protocol=_ENG_PROTOCOL,
        e_met=1e-9,
        p_info=1e-12,
    ),
    # ── Inert: high persistence, no return ──
    SystemProfile(
        name="Diamond crystal",
        category="inert",
        substrate="sp³ carbon lattice",
        channels=(0.9999, 0.999, 0.9999, 0.9999, 0.9999, 0.999, 0.9999, 0.9999),
        channel_labels=_CHANNEL_LABELS_PHYS,
        weights=None,
        protocol=_INERT_PROTOCOL,
        e_met=0.0,
        p_info=0.0,
    ),
    SystemProfile(
        name="Granite block",
        category="inert",
        substrate="Felsic ignite rock",
        channels=(0.98, 0.95, 0.97, 0.96, 0.94, 0.92, 0.95, 0.99),
        channel_labels=_CHANNEL_LABELS_PHYS,
        weights=None,
        protocol=_INERT_PROTOCOL,
        e_met=0.0,
        p_info=0.0,
    ),
    SystemProfile(
        name="Scratched rock tile",
        category="inert",
        substrate="Basalt with incised pattern",
        channels=(0.90, 0.70, 0.85, 0.60, 0.80, 0.95, 0.88, 0.92),
        channel_labels=_CHANNEL_LABELS_PHYS,
        weights=None,
        protocol=_INERT_PROTOCOL,
        e_met=0.0,
        p_info=0.0,
    ),
    # ── Edge cases: high F but low IC (geometric slaughter) ──
    SystemProfile(
        name="Flame (candle)",
        category="dissipative",
        substrate="Combustion reaction zone",
        channels=(0.01, 0.95, 0.01, 0.01, 0.80, 0.01, 0.01, 0.01),
        channel_labels=_CHANNEL_LABELS_PHYS,
        weights=None,
        protocol=_INERT_PROTOCOL,
        e_met=80.0,  # ~80W
        p_info=0.0,
    ),
    SystemProfile(
        name="Virus (free particle)",
        category="dormant",
        substrate="Protein capsid + nucleic acid",
        channels=(0.01, 0.01, 0.95, 0.90, 0.01, 0.01, 0.01, 0.01),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=0.0,
        p_info=0.0,
    ),
    SystemProfile(
        name="Prion",
        category="biological",
        substrate="Misfolded protein aggregate",
        channels=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.95, 0.01),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=_BIO_PROTOCOL,
        e_met=0.0,
        p_info=0.0,
    ),
    # ── Butzbach's specific examples ──
    SystemProfile(
        name="Stressed bacterium (Butzbach protocol)",
        category="biological",
        substrate="E. coli under osmotic stress cycles",
        channels=(0.80, 0.60, 0.85, 0.75, 0.70, 0.65, 0.55, 0.50),
        channel_labels=_CHANNEL_LABELS_BIO,
        weights=None,
        protocol=ProtocolDeclaration(
            boundary="Cell membrane + 1μm halo",
            viability_set="Colony-forming on LB agar within 48h",
            perturbation="5-min 0.5M NaCl osmotic shock",
            recovery_window=3600.0,
        ),
        e_met=5e-13,
        p_info=5e-16,
    ),
    SystemProfile(
        name="Photonic quantum memory (Butzbach protocol)",
        category="engineered",
        substrate="Rare-earth ion crystal at 4K",
        channels=(0.92, 0.98, 0.60, 0.55, 0.90, 0.85, 0.95, 0.80),
        channel_labels=_CHANNEL_LABELS_ENG,
        weights=None,
        protocol=ProtocolDeclaration(
            boundary="Cryostat + optical cavity",
            viability_set="Photon echo fidelity ≥ 0.90",
            perturbation="50-ms dark interval + thermal noise",
            recovery_window=0.1,
        ),
        e_met=15.0,
        p_info=1e-6,
    ),
)

N_SYSTEMS = len(SYSTEMS)  # 20


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: ANALYSIS ENGINE — BUTZBACH VS GCD
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Result of embedding a system in both Butzbach's and GCD's framework.

    Shows exactly what Butzbach sees (scalar c, cascade, P_Ω) and what
    GCD reveals (per-channel, F vs IC, heterogeneity gap, regime).
    """

    # Identity
    name: str
    category: str
    n_channels: int

    # Butzbach's view (scalar)
    butzbach_c: float  # Aggregate scalar continuity
    butzbach_p_omega: float  # Persistence functional

    # GCD's view (decomposed)
    gcd_F: float
    gcd_omega: float
    gcd_IC: float
    gcd_kappa: float
    gcd_S: float
    gcd_C: float
    gcd_regime: str
    heterogeneity_gap: float  # Δ = F − IC — invisible to Butzbach

    # Channel diagnostics (unavailable to Butzbach)
    weakest_channel: str
    weakest_value: float
    strongest_channel: str
    strongest_value: float
    channel_range: float  # max - min

    # Embedding diagnostics
    butzbach_sees_healthy: bool  # c > 0.5
    gcd_sees_fragile: bool  # Δ > 0.1
    blind_spot: bool  # Butzbach says healthy, GCD says fragile

    # Classification
    butzbach_label: str  # "living", "dormant", "memory", "inert"
    gcd_label: str  # Stable/Watch/Collapse + Critical overlay

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "name": self.name,
            "category": self.category,
            "n_channels": self.n_channels,
            "butzbach": {
                "c": self.butzbach_c,
                "p_omega": self.butzbach_p_omega,
                "sees_healthy": self.butzbach_sees_healthy,
                "label": self.butzbach_label,
            },
            "gcd": {
                "F": self.gcd_F,
                "omega": self.gcd_omega,
                "IC": self.gcd_IC,
                "kappa": self.gcd_kappa,
                "S": self.gcd_S,
                "C": self.gcd_C,
                "regime": self.gcd_regime,
                "heterogeneity_gap": self.heterogeneity_gap,
                "label": self.gcd_label,
            },
            "channel_diagnostics": {
                "weakest": self.weakest_channel,
                "weakest_value": self.weakest_value,
                "strongest": self.strongest_channel,
                "strongest_value": self.strongest_value,
                "range": self.channel_range,
            },
            "blind_spot": self.blind_spot,
        }


def _classify_butzbach(system: SystemProfile) -> str:
    """Butzbach's informal classification (no formal gates)."""
    c = system.aggregate_c()
    if system.category == "inert":
        return "memory-bearing inert"
    if system.category == "dormant":
        return "compressed carrier (boot kit)" if c > 0.3 else "degraded dormant"
    if system.category == "dissipative":
        return "dissipative (no memory)"
    if system.e_met > 0 and c > 0.5:
        return "living (defended configuration)"
    if system.e_met > 0 and c <= 0.5:
        return "failing (continuity below threshold)"
    return "ambiguous (no regime gates)"


def _classify_gcd_regime(omega: float, f: float, s: float, c: float) -> str:
    """GCD formal four-gate regime classification.

    Stable: ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14
    Watch:  0.038 ≤ ω < 0.30 (or Stable gates not all met)
    Collapse: ω ≥ 0.30
    """
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and f > 0.90 and s < 0.15 and c < 0.14:
        return "Stable"
    return "Watch"


def _classify_gcd_label(regime: str, ic: float) -> str:
    """GCD formal classification with Critical overlay."""
    critical = " + Critical" if ic < 0.30 else ""
    return f"{regime}{critical}"


def analyze_system(system: SystemProfile) -> EmbeddingResult:
    """Analyze a system through both Butzbach's and GCD's frameworks.

    This is the core comparison engine. For each system, it computes:
    1. Butzbach's scalar c and P_Ω
    2. GCD's full kernel (F, ω, IC, κ, S, C, regime)
    3. Channel-level diagnostics (invisible to Butzbach)
    4. Whether there is a blind spot (Butzbach healthy, GCD fragile)
    """
    channels = np.asarray(system.channels, dtype=np.float64)
    channels = np.clip(channels, EPSILON, 1.0 - EPSILON)
    n = len(channels)
    w = system._weights()

    # Butzbach's view: scalar aggregate
    butzbach_c = system.aggregate_c()
    butzbach_p = PersistenceFunctional(
        e_met=system.e_met,
        p_info=system.p_info,
        continuity=butzbach_c,
    )

    # GCD's view: full channel decomposition
    kernel = compute_kernel_outputs(channels, w, epsilon=EPSILON)

    # Four-gate regime classification (Tier-0)
    gcd_regime = _classify_gcd_regime(kernel["omega"], kernel["F"], kernel["S"], kernel["C"])

    # Channel diagnostics (what Butzbach cannot see)
    labels = system.channel_labels
    min_idx = int(np.argmin(channels))
    max_idx = int(np.argmax(channels))

    # Blind spot detection
    butzbach_healthy = butzbach_c > 0.5
    gcd_fragile = kernel["heterogeneity_gap"] > 0.1
    blind_spot = bool(butzbach_healthy and gcd_fragile)

    return EmbeddingResult(
        name=system.name,
        category=system.category,
        n_channels=n,
        butzbach_c=butzbach_c,
        butzbach_p_omega=butzbach_p.p_omega,
        gcd_F=float(kernel["F"]),
        gcd_omega=float(kernel["omega"]),
        gcd_IC=float(kernel["IC"]),
        gcd_kappa=float(kernel["kappa"]),
        gcd_S=float(kernel["S"]),
        gcd_C=float(kernel["C"]),
        gcd_regime=gcd_regime,
        heterogeneity_gap=float(kernel["heterogeneity_gap"]),
        weakest_channel=labels[min_idx],
        weakest_value=float(channels[min_idx]),
        strongest_channel=labels[max_idx],
        strongest_value=float(channels[max_idx]),
        channel_range=float(channels[max_idx] - channels[min_idx]),
        butzbach_sees_healthy=bool(butzbach_healthy),
        gcd_sees_fragile=bool(gcd_fragile),
        blind_spot=blind_spot,
        butzbach_label=_classify_butzbach(system),
        gcd_label=_classify_gcd_label(gcd_regime, float(kernel["IC"])),
    )


def analyze_all_systems() -> list[EmbeddingResult]:
    """Run the embedding analysis on all 20 systems."""
    return [analyze_system(s) for s in SYSTEMS]


# ═════════════════════════════════════════════════════════════════════
# SECTION 5: GEOMETRIC SLAUGHTER DEMONSTRATION
# ═════════════════════════════════════════════════════════════════════


def demonstrate_geometric_slaughter(
    n_channels: int = 8,
    healthy_value: float = 0.95,
) -> dict[str, Any]:
    """Demonstrate that Butzbach's scalar c misses geometric slaughter.

    Start with n healthy channels. Kill one channel progressively.
    Track Butzbach's c (arithmetic mean) vs GCD's IC (geometric mean).

    This is the central proof that channel decomposition matters:
    F stays healthy while IC collapses — and Butzbach sees only F.
    """
    kill_values = np.logspace(0, -8, 50)  # 1.0 → 1e-8 (to ε)
    kill_values = np.clip(kill_values, EPSILON, 1.0)

    results = []
    for kill in kill_values:
        channels = np.full(n_channels, healthy_value)
        channels[0] = kill  # Kill channel 0
        channels = np.clip(channels, EPSILON, 1.0 - EPSILON)
        w = np.ones(n_channels) / n_channels

        kernel = compute_kernel_outputs(channels, w, epsilon=EPSILON)

        # Butzbach's view: arithmetic mean
        butzbach_c = float(np.dot(w, channels))

        # Four-gate regime
        regime = _classify_gcd_regime(kernel["omega"], kernel["F"], kernel["S"], kernel["C"])

        results.append(
            {
                "killed_channel_value": float(kill),
                "butzbach_c": butzbach_c,
                "gcd_F": float(kernel["F"]),
                "gcd_IC": float(kernel["IC"]),
                "heterogeneity_gap": float(kernel["heterogeneity_gap"]),
                "regime": regime,
                "ic_over_f": float(kernel["IC"]) / float(kernel["F"]) if kernel["F"] > 0 else 0.0,
            }
        )

    # Summary statistics
    start = results[0]
    end = results[-1]

    return {
        "n_channels": n_channels,
        "healthy_value": healthy_value,
        "trajectory": results,
        "summary": {
            "butzbach_c_start": start["butzbach_c"],
            "butzbach_c_end": end["butzbach_c"],
            "butzbach_c_drop_pct": (1 - end["butzbach_c"] / start["butzbach_c"]) * 100,
            "gcd_IC_start": start["gcd_IC"],
            "gcd_IC_end": end["gcd_IC"],
            "gcd_IC_drop_pct": (1 - end["gcd_IC"] / start["gcd_IC"]) * 100,
            "max_gap": max(r["heterogeneity_gap"] for r in results),
            "butzbach_blind": bool(end["butzbach_c"] > 0.5 and end["gcd_IC"] < 0.1),
        },
    }


# ═════════════════════════════════════════════════════════════════════
# SECTION 6: ROSETTA — BUTZBACH'S VOCABULARY IN GCD TERMS
# ═════════════════════════════════════════════════════════════════════

ROSETTA: dict[str, dict[str, str]] = {
    "Memory (M)": {
        "butzbach": "Persistent physical configuration constraining future dynamics",
        "gcd": "Channel coherence vector c = (c₁, ..., cₙ) with cᵢ ∈ [ε, 1-ε]",
        "relation": "M is the undecomposed aggregate of the trace vector",
        "what_gcd_adds": "Per-channel decomposition reveals WHERE memory resides",
    },
    "Continuity (C)": {
        "butzbach": "Return-to-viability probability c(t; B, V, Π, τ)",
        "gcd": "IC = exp(κ) = exp(Σ wᵢ ln cᵢ) — integrity composite",
        "relation": "c = IC when n=1; c ≈ F when measured as aggregate",
        "what_gcd_adds": "Separate F (arithmetic) from IC (geometric) to detect fragility",
    },
    "Cascade C(t)": {
        "butzbach": "C(t+1) = C(t)·c(t) — multiplicative accumulation",
        "gcd": "κ_acc = Σ ln c(k) — additive log-integrity accumulation",
        "relation": "Identical: C(t) = exp(κ_acc(t)). Same math, different notation",
        "what_gcd_adds": "Per-channel κᵢ tracking allows history-conditioned c(t|H_t)",
    },
    "Protocol (B,V,Π,τ)": {
        "butzbach": "Declared system boundary, viability set, perturbation, recovery window",
        "gcd": "Contract stop: freeze sources, normalization, policy, thresholds before evidence",
        "relation": "Identical methodological principle: declare before evidence",
        "what_gcd_adds": "Contract is the first stop of a five-stop Spine, not a standalone",
    },
    "P_Ω functional": {
        "butzbach": "P_Ω = (E_met + P_info)·C — persistence power weighted by continuity",
        "gcd": "Seam budget: Δκ = R·τ_R − (D_ω + D_C) — debit/credit reconciliation",
        "relation": "Both are resource-continuity products; P_Ω is single-channel",
        "what_gcd_adds": "Seam budget decomposes into drift cost D_ω and curvature cost D_C",
    },
    "Low-M cliff": {
        "butzbach": "Continuity collapses when repair-relevant memory becomes scarce",
        "gcd": "Geometric slaughter: one cᵢ → ε kills IC while F stays healthy",
        "relation": "Same phenomenon; GCD locates it, Butzbach predicts it",
        "what_gcd_adds": "Channel identity of the failing component; cliff location in (F,IC) space",
    },
    "Living regime": {
        "butzbach": "Informal: 'life is recoverable configuration under flow'",
        "gcd": "Formal: Stable (ω<0.038 ∧ F>0.90 ∧ S<0.15 ∧ C<0.14), frozen four-gate",
        "relation": "Butzbach's 'living' ⊂ GCD's 'Stable' when boundary conditions are met",
        "what_gcd_adds": "Three regimes (Stable/Watch/Collapse) + Critical overlay, all derived from gates",
    },
    "Boot kit (dormancy)": {
        "butzbach": "Compressed carrier of recoverable configuration (seeds, spores)",
        "gcd": "System where E_met → 0 but M is high and τ_R is finite upon reactivation",
        "relation": "Same conceptual treatment of dormancy",
        "what_gcd_adds": "Channel profile shows WHICH aspects are compressed vs preserved",
    },
}


# ═════════════════════════════════════════════════════════════════════
# SECTION 7: TIER PLACEMENT
# ═════════════════════════════════════════════════════════════════════

TIER_PLACEMENT = {
    "tier": "Tier-2 (Expansion Space)",
    "justification": (
        "Butzbach's Continuity Theory is a substrate-specific (biological/physical) "
        "restriction of the GCD kernel to n=1 channels. It inherits Tier-1 identities "
        "(F+ω=1, IC≤F when decomposed) but does not add new structural identities. "
        "It is validated through Tier-0 protocol against Tier-1 invariants. "
        "No feedback from this closure to Tier-1 or Tier-0."
    ),
    "what_it_contributes": (
        "Explicit worked examples in biological and engineered substrates. "
        "Landauer bound coupling for P_info (GCD does not invoke this directly). "
        "Falsification protocol for low-M cliff (experimentally actionable). "
        "Dormancy-as-boot-kit framing (elegant special case of zero-metabolic persistence)."
    ),
    "what_it_lacks": (
        "Channel decomposition. Heterogeneity gap. Formal regime gates. "
        "Three-valued verdicts. History conditioning. Cross-domain Rosetta. "
        "Derivation from a single axiom. Frozen parameter justification. "
        "Computational verification suite."
    ),
    "priority": (
        "GCD (Paulus 2025) predates Butzbach (2026) by Zenodo DOIs, "
        "SHA-256 integrity chain, and append-only git ledger. "
        "The multiplicative cascade, return-to-viability measure, "
        "boundary declaration requirement, and cliff prediction all appear "
        "in GCD's published artifacts before Butzbach's first manuscript."
    ),
}


# ═════════════════════════════════════════════════════════════════════
# SECTION 8: VALIDATION
# ═════════════════════════════════════════════════════════════════════


def validate_embedding() -> dict[str, Any]:
    """Run all embedding validations and return summary.

    Checks:
    1. Scalar limit proof (c = IC at n=1) across 100 values
    2. Cascade = log-integrity proof across random sequences
    3. Geometric slaughter demonstration
    4. Blind spot detection across all 20 systems
    5. Tier-1 identity verification for all systems
    """
    results: dict[str, Any] = {}

    # 1. Scalar limit proof
    scalar_proofs = []
    for c_val in np.linspace(0.01, 0.99, 100):
        proof = prove_scalar_limit(c_val)
        scalar_proofs.append(proof)
    max_residual = max(p["residual_c_vs_IC"] for p in scalar_proofs)
    results["scalar_limit"] = {
        "n_tested": 100,
        "max_residual_c_vs_IC": max_residual,
        "passed": bool(max_residual < 1e-12),
    }

    # 2. Cascade proof
    rng = np.random.default_rng(42)
    cascade_proofs = []
    for _ in range(20):
        seq = rng.uniform(0.5, 0.99, size=rng.integers(5, 30)).tolist()
        proof = prove_cascade_is_log_integrity(seq)
        cascade_proofs.append(proof)
    max_casc_residual = max(p["max_residual"] for p in cascade_proofs)
    results["cascade_proof"] = {
        "n_sequences": 20,
        "max_residual": max_casc_residual,
        "passed": bool(max_casc_residual < 1e-12),
    }

    # 3. Geometric slaughter
    slaughter = demonstrate_geometric_slaughter()
    results["geometric_slaughter"] = {
        "butzbach_c_drop_pct": slaughter["summary"]["butzbach_c_drop_pct"],
        "gcd_IC_drop_pct": slaughter["summary"]["gcd_IC_drop_pct"],
        "butzbach_blind": slaughter["summary"]["butzbach_blind"],
        "max_gap": slaughter["summary"]["max_gap"],
    }

    # 4. Full system analysis
    analyses = analyze_all_systems()
    blind_spots = [a for a in analyses if a.blind_spot]
    results["system_analysis"] = {
        "n_systems": len(analyses),
        "n_blind_spots": len(blind_spots),
        "blind_spot_names": [a.name for a in blind_spots],
        "regime_distribution": {},
    }
    for a in analyses:
        regime = a.gcd_regime
        results["system_analysis"]["regime_distribution"][regime] = (
            results["system_analysis"]["regime_distribution"].get(regime, 0) + 1
        )

    # 5. Tier-1 identity check
    tier1_failures = 0
    for a in analyses:
        if abs(a.gcd_F + a.gcd_omega - 1.0) > 1e-12:
            tier1_failures += 1
        if a.gcd_IC > a.gcd_F + 1e-12:
            tier1_failures += 1
    results["tier1_identities"] = {
        "n_checked": len(analyses) * 2,
        "n_failures": tier1_failures,
        "passed": bool(tier1_failures == 0),
    }

    results["overall_passed"] = all(
        [
            results["scalar_limit"]["passed"],
            results["cascade_proof"]["passed"],
            results["geometric_slaughter"]["butzbach_blind"],
            results["tier1_identities"]["passed"],
        ]
    )

    return results
