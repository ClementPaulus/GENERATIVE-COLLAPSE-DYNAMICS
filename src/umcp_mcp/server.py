"""UMCP MCP Server — GCD-native tool surface.

Every tool enforces frozen parameters internally. The model receives
structured outputs with Tier-1 invariants as typed fields — never as
raw strings it could misinterpret or redefine.

Tools:
    compute_kernel      Compute (F, ω, S, C, κ, IC) from trace vector + weights
    classify_regime     Classify regime from kernel invariants
    validate_seam       Compute seam budget and check residual ≤ tol
    check_identities    Verify algebraic identities (F+ω=1, IC≤F, IC=exp(κ))
    frozen_parameters   Read-only access to all frozen constants
    orientation_receipt  Return the 10 computational ground truth receipts
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # optional dependency
    FastMCP = None  # type: ignore[assignment, misc]

from umcp.frozen_contract import (
    ALPHA,
    DEFAULT_THRESHOLDS,
    EPSILON,
    P_EXPONENT,
    TOL_SEAM,
    classify_regime,
    compute_budget_delta_kappa,
    compute_seam_residual,
    cost_curvature,
    gamma_omega,
)
from umcp.kernel_optimized import KernelOutputs, OptimizedKernelComputer

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

if FastMCP is None:
    msg = "The 'mcp' package is required: pip install 'mcp[cli]>=1.0.0'"
    raise ImportError(msg)

mcp = FastMCP(
    "umcp",
    instructions=(
        "UMCP/GCD kernel server. All frozen parameters are enforced internally. "
        "Tier-1 symbols (F, ω, S, C, κ, IC) are return values — never redefine them. "
        "Verdicts are three-valued: CONFORMANT / NONCONFORMANT / NON_EVALUABLE. "
        "Classical results (AM-GM, Shannon) are degenerate limits, not sources."
    ),
)

# Singleton kernel computer — frozen epsilon
_kernel = OptimizedKernelComputer(epsilon=EPSILON)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kernel_to_dict(k: KernelOutputs) -> dict[str, Any]:
    """Convert KernelOutputs to a clean dict for MCP response."""
    return {
        "F": k.F,
        "omega": k.omega,
        "S": k.S,
        "C": k.C,
        "kappa": k.kappa,
        "IC": k.IC,
        "heterogeneity_gap": k.heterogeneity_gap,
        "regime": k.regime,
        "is_homogeneous": k.is_homogeneous,
        "computation_mode": k.computation_mode,
    }


def _parse_vector(raw: list[float], name: str) -> np.ndarray:
    """Parse and validate a numeric list into a numpy array."""
    if not raw:
        msg = f"{name} must be a non-empty list of floats"
        raise ValueError(msg)
    return np.asarray(raw, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def compute_kernel(
    trace_vector: list[float],
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Compute Tier-1 kernel invariants from a trace vector.

    The kernel K: [0,1]^n × Δ^n → (F, ω, S, C, κ, IC) is computed
    with frozen ε = 1e-8 guard band. Coordinates are pre-clipped to
    [ε, 1-ε]. Weights default to uniform if not provided.

    Args:
        trace_vector: Channel values in [0, 1]. Each entry is a
            measurable property mapped to the unit interval.
        weights: Simplex weights (must sum to 1.0). If omitted,
            uniform weights 1/n are used.

    Returns:
        Dict with F, omega, S, C, kappa, IC, heterogeneity_gap,
        regime label, and computation mode.
    """
    c = _parse_vector(trace_vector, "trace_vector")
    n = len(c)

    if weights is not None:
        w = _parse_vector(weights, "weights")
        if len(w) != n:
            msg = f"weights length ({len(w)}) must match trace_vector length ({n})"
            raise ValueError(msg)
    else:
        w = np.full(n, 1.0 / n)

    result = _kernel.compute(c, w)
    return _kernel_to_dict(result)


@mcp.tool()
def classify_regime_tool(
    omega: float,
    F: float,
    S: float,
    C: float,
    IC: float,
) -> dict[str, Any]:
    """Classify the regime from kernel invariants using frozen gates.

    Four-gate criterion with frozen thresholds:
        Stable:   ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14
        Watch:    0.038 ≤ ω < 0.30
        Collapse: ω ≥ 0.30
        Critical: IC < 0.30 (severity overlay)

    Args:
        omega: Drift (ω = 1 - F).
        F: Fidelity.
        S: Bernoulli field entropy.
        C: Curvature.
        IC: Integrity composite.

    Returns:
        Dict with regime label, is_critical flag, and gate margins.
    """
    thresholds = DEFAULT_THRESHOLDS
    regime = classify_regime(omega, F, S, C, IC, thresholds)
    is_critical = thresholds.I_critical_max > IC

    return {
        "regime": regime.value,
        "is_critical": is_critical,
        "gates": {
            "omega": omega,
            "F": F,
            "S": S,
            "C": C,
            "IC": IC,
        },
        "thresholds": {
            "omega_stable_max": thresholds.omega_stable_max,
            "F_stable_min": thresholds.F_stable_min,
            "S_stable_max": thresholds.S_stable_max,
            "C_stable_max": thresholds.C_stable_max,
            "omega_collapse_min": thresholds.omega_collapse_min,
            "IC_critical_max": thresholds.I_critical_max,
        },
    }


@mcp.tool()
def validate_seam(
    kappa_pre: float,
    kappa_post: float,
    tau_R: float,
    R: float = 0.01,
    drift_cost: float = 0.0,
    curvature_cost: float = 0.0,
) -> dict[str, Any]:
    """Validate a seam crossing (collapse-return boundary).

    Computes:
        Δκ_ledger  = κ_post - κ_pre       (observed change)
        Δκ_budget  = R·τ_R - (D_ω + D_C)  (expected change)
        residual   = Δκ_budget - Δκ_ledger
        PASS if |residual| ≤ tol_seam (0.005)

    Args:
        kappa_pre: Log-integrity before the seam.
        kappa_post: Log-integrity after the seam.
        tau_R: Return time (re-entry delay). Use float('inf') for no-return.
        R: Return credit rate.
        drift_cost: D_ω — drift penalty.
        curvature_cost: D_C — curvature penalty.

    Returns:
        Dict with delta_kappa, budget, residual, pass status, and reasons.
    """
    if tau_R == float("inf"):
        return {
            "delta_kappa_ledger": kappa_post - kappa_pre,
            "delta_kappa_budget": 0.0,
            "residual": -(kappa_post - kappa_pre),
            "pass": False,
            "reasons": ["tau_R = INF_REC: no return → no credit"],
            "tol_seam": TOL_SEAM,
        }

    delta_kappa_ledger = kappa_post - kappa_pre
    delta_kappa_budget = compute_budget_delta_kappa(R, tau_R, drift_cost, curvature_cost)
    residual = compute_seam_residual(delta_kappa_budget, delta_kappa_ledger)

    reasons: list[str] = []
    passed = abs(residual) <= TOL_SEAM
    if not passed:
        reasons.append(f"|residual| = {abs(residual):.6f} > tol_seam = {TOL_SEAM}")

    return {
        "delta_kappa_ledger": delta_kappa_ledger,
        "delta_kappa_budget": delta_kappa_budget,
        "residual": residual,
        "pass": passed,
        "reasons": reasons,
        "tol_seam": TOL_SEAM,
    }


@mcp.tool()
def compute_costs(
    omega: float,
    C: float,
) -> dict[str, Any]:
    """Compute drift and curvature costs from kernel invariants.

    Uses frozen parameters:
        Γ(ω) = ω^p / (1 - ω + ε),  p = 3, ε = 1e-8
        D_C = α·C,                   α = 1.0

    Args:
        omega: Drift value.
        C: Curvature value.

    Returns:
        Dict with gamma (drift cost), D_C (curvature cost), total cost,
        and the frozen parameters used.
    """
    gamma = gamma_omega(omega, p=P_EXPONENT, epsilon=EPSILON)
    d_c = cost_curvature(C, alpha=ALPHA)

    return {
        "gamma_omega": gamma,
        "D_C": d_c,
        "total_cost": gamma + d_c,
        "frozen_params": {
            "p": P_EXPONENT,
            "alpha": ALPHA,
            "epsilon": EPSILON,
        },
    }


@mcp.tool()
def check_identities(
    trace_vector: list[float],
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Verify the three algebraic identities on a trace vector.

    Checks:
        1. F + ω = 1  (duality identity — exact by construction)
        2. IC ≤ F     (integrity bound — solvability condition)
        3. IC = exp(κ) (log-integrity relation)

    Args:
        trace_vector: Channel values in [0, 1].
        weights: Simplex weights. Uniform if omitted.

    Returns:
        Dict with each identity's residual, pass status, and the
        three-valued verdict (CONFORMANT/NONCONFORMANT/NON_EVALUABLE).
    """
    c = _parse_vector(trace_vector, "trace_vector")
    n = len(c)
    w = _parse_vector(weights, "weights") if weights is not None else np.full(n, 1.0 / n)

    k = _kernel.compute(c, w)

    duality_residual = abs(k.F + k.omega - 1.0)
    integrity_holds = k.IC <= k.F + 1e-12  # numerical tolerance
    log_residual = abs(k.IC - np.exp(k.kappa))

    all_pass = duality_residual < 1e-12 and integrity_holds and log_residual < 1e-10

    return {
        "duality": {
            "identity": "F + ω = 1",
            "residual": duality_residual,
            "pass": duality_residual < 1e-12,
        },
        "integrity_bound": {
            "identity": "IC ≤ F",
            "IC": k.IC,
            "F": k.F,
            "gap": k.F - k.IC,
            "pass": integrity_holds,
        },
        "log_integrity": {
            "identity": "IC = exp(κ)",
            "IC": k.IC,
            "exp_kappa": float(np.exp(k.kappa)),
            "residual": log_residual,
            "pass": log_residual < 1e-10,
        },
        "verdict": "CONFORMANT" if all_pass else "NONCONFORMANT",
    }


@mcp.tool()
def frozen_parameters() -> dict[str, Any]:
    """Return all frozen parameters (read-only).

    These are seam-derived constants — the unique values where seams
    close consistently across all 20 domains. They are discovered by
    the mathematics, not chosen by convention.

    Returns:
        Dict with all frozen parameter names, values, and roles.
    """
    return {
        "epsilon": {"value": EPSILON, "role": "Guard band / ε-clamp"},
        "p_exponent": {"value": P_EXPONENT, "role": "Drift cost exponent in Γ(ω) = ω^p/(1−ω+ε)"},
        "alpha": {"value": ALPHA, "role": "Curvature cost coefficient in D_C = α·C"},
        "tol_seam": {"value": TOL_SEAM, "role": "Seam residual tolerance: |s| ≤ tol for PASS"},
        "regime_thresholds": {
            "omega_stable_max": DEFAULT_THRESHOLDS.omega_stable_max,
            "F_stable_min": DEFAULT_THRESHOLDS.F_stable_min,
            "S_stable_max": DEFAULT_THRESHOLDS.S_stable_max,
            "C_stable_max": DEFAULT_THRESHOLDS.C_stable_max,
            "omega_collapse_min": DEFAULT_THRESHOLDS.omega_collapse_min,
            "IC_critical_max": DEFAULT_THRESHOLDS.I_critical_max,
        },
        "note": "Frozen — consistent across the seam. Seam-derived, not tuned.",
    }


@mcp.tool()
def orientation_receipts() -> dict[str, Any]:
    """Return the 10 computational ground truth receipts.

    These numbers are compressed derivation chains. A model with these
    receipts loaded cannot misclassify GCD structures because the
    numbers constrain what can be said.

    Returns:
        Dict with all 10 orientation section receipts and their
        operational constraints.
    """
    return {
        "receipts": [
            {
                "section": 1,
                "name": "Duality",
                "receipt": "max|F + ω - 1| = 0.0e+00",
                "constraint": "F + ω = 1 is exact — not approximate, EXACTLY zero",
            },
            {
                "section": 2,
                "name": "Integrity Bound",
                "receipt": "Δ for (0.95, 0.001) = 0.4447",
                "constraint": "One dead channel creates massive heterogeneity gap",
            },
            {
                "section": 3,
                "name": "Geometric Slaughter",
                "receipt": "IC/F with 1 dead channel (8ch) = 0.1143",
                "constraint": "7 perfect channels cannot save IC from 1 dead channel",
            },
            {
                "section": 4,
                "name": "First Weld",
                "receipt": "Γ(0.682) = 0.9975",
                "constraint": "First weld threshold at c ≈ 0.318",
            },
            {
                "section": 5,
                "name": "Confinement Cliff",
                "receipt": "Neutron IC/F = 0.0089, Proton IC/F = 0.0371",
                "constraint": "Confinement drops IC/F 100× — dead color channel kills IC",
            },
            {
                "section": 6,
                "name": "Scale Inversion",
                "receipt": "Nickel IC/F = 0.9573",
                "constraint": "Atoms RESTORE coherence with new degrees of freedom",
            },
            {
                "section": 7,
                "name": "Full Spine",
                "receipt": "Stable regime = 12.5% of Fisher space",
                "constraint": "87.5% of the manifold is NOT stable — stability is rare",
            },
            {
                "section": 8,
                "name": "Equator Convergence",
                "receipt": "S + κ at c=1/2 = 0.0",
                "constraint": "Perfect cancellation at equator — four-way convergence",
            },
            {
                "section": 9,
                "name": "Super-Exponential",
                "receipt": "Gap shrinks 28.5×",
                "constraint": "IC convergence faster than exponential",
            },
            {
                "section": 10,
                "name": "Seam Composition",
                "receipt": "Associativity error = 5.55e-17",
                "constraint": "Seam composition is an exact monoid",
            },
        ],
        "key_constraints": [
            "IC ≤ F is the SOLVABILITY CONDITION (not AM-GM): c₁,₂ = F ± √(F²−IC²) requires IC ≤ F for real solutions",
            "S is BERNOULLI FIELD ENTROPY (not Shannon): Shannon is the degenerate limit when cᵢ ∈ {0,1}",
            "F + ω = 1 is the DUALITY IDENTITY (not unitarity): structural identity of collapse",
            "Frozen parameters are SEAM-DERIVED (not hyperparameters): discovered, not chosen",
        ],
    }


@mcp.tool()
def compute_kernel_batch(
    trace_vectors: list[list[float]],
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Compute kernel invariants for multiple entities at once.

    All entities share the same weight vector (same channel structure).
    Each trace vector maps to a different entity measured on those channels.

    Args:
        trace_vectors: List of trace vectors, each in [0, 1]^n.
        weights: Shared simplex weights. Uniform if omitted.

    Returns:
        Dict with list of kernel results and summary statistics.
    """
    if not trace_vectors:
        msg = "trace_vectors must be a non-empty list"
        raise ValueError(msg)

    n = len(trace_vectors[0])
    w = _parse_vector(weights, "weights") if weights is not None else np.full(n, 1.0 / n)

    results = []
    for i, tv in enumerate(trace_vectors):
        c = _parse_vector(tv, f"trace_vectors[{i}]")
        if len(c) != n:
            msg = f"trace_vectors[{i}] length ({len(c)}) != expected ({n})"
            raise ValueError(msg)
        k = _kernel.compute(c, w)
        results.append(_kernel_to_dict(k))

    # Summary statistics
    f_values = [r["F"] for r in results]
    ic_values = [r["IC"] for r in results]

    return {
        "results": results,
        "count": len(results),
        "summary": {
            "F_mean": float(np.mean(f_values)),
            "F_std": float(np.std(f_values)),
            "IC_mean": float(np.mean(ic_values)),
            "IC_std": float(np.std(ic_values)),
            "gap_mean": float(np.mean([r["heterogeneity_gap"] for r in results])),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the UMCP MCP server."""
    import sys

    mcp.run(transport="sse" if "--sse" in sys.argv else "stdio")
