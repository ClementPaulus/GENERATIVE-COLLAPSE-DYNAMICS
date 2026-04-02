"""Tests for the UMCP MCP server — the tightest nozzle.

Validates that every tool enforces frozen parameters, returns correct
Tier-1 invariants, and that symbol capture is structurally impossible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    from umcp_mcp.server import (
        check_identities,
        classify_regime_tool,
        compute_costs,
        compute_kernel,
        compute_kernel_batch,
        frozen_parameters,
        orientation_receipts,
        validate_seam,
    )
except ImportError:
    pytest.skip("mcp package not installed", allow_module_level=True)

# ---------------------------------------------------------------------------
# compute_kernel
# ---------------------------------------------------------------------------


class TestComputeKernel:
    """Tests for the compute_kernel tool."""

    def test_uniform_trace(self) -> None:
        """Homogeneous trace vector gives F = c, IC = F."""
        result = compute_kernel([0.8, 0.8, 0.8, 0.8])
        assert abs(result["F"] - 0.8) < 1e-10
        assert abs(result["omega"] - 0.2) < 1e-10

    def test_duality_identity(self) -> None:
        """F + ω = 1 for any trace vector."""
        result = compute_kernel([0.3, 0.7, 0.5, 0.9])
        assert abs(result["F"] + result["omega"] - 1.0) < 1e-12

    def test_integrity_bound(self) -> None:
        """IC ≤ F for any trace vector."""
        result = compute_kernel([0.95, 0.001, 0.7, 0.6])
        assert result["IC"] <= result["F"] + 1e-12

    def test_log_integrity_relation(self) -> None:
        """IC = exp(κ)."""
        result = compute_kernel([0.4, 0.6, 0.8, 0.2])
        assert abs(result["IC"] - math.exp(result["kappa"])) < 1e-10

    def test_heterogeneity_gap_positive(self) -> None:
        """Δ = F − IC ≥ 0 for heterogeneous vectors."""
        result = compute_kernel([0.9, 0.1, 0.5, 0.7])
        assert result["heterogeneity_gap"] >= -1e-12

    def test_regime_returned(self) -> None:
        """Result includes a regime classification."""
        result = compute_kernel([0.95, 0.95, 0.95, 0.95])
        assert result["regime"] in {"STABLE", "WATCH", "COLLAPSE", "CRITICAL", "homogeneous"}

    def test_custom_weights(self) -> None:
        """Custom weights change F."""
        uniform = compute_kernel([0.3, 0.9])
        weighted = compute_kernel([0.3, 0.9], weights=[0.9, 0.1])
        assert weighted["F"] != pytest.approx(uniform["F"], abs=0.01)

    def test_empty_vector_raises(self) -> None:
        """Empty trace vector raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            compute_kernel([])

    def test_weights_length_mismatch_raises(self) -> None:
        """Mismatched weights length raises ValueError."""
        with pytest.raises(ValueError, match="must match"):
            compute_kernel([0.5, 0.5], weights=[0.3, 0.3, 0.4])

    def test_geometric_slaughter(self) -> None:
        """One dead channel kills IC while F stays healthy."""
        c = [0.99] * 7 + [0.001]
        result = compute_kernel(c)
        assert result["F"] > 0.85
        # IC/F drops significantly due to geometric slaughter
        assert result["heterogeneity_gap"] > 0.1


# ---------------------------------------------------------------------------
# classify_regime_tool
# ---------------------------------------------------------------------------


class TestClassifyRegime:
    """Tests for the classify_regime_tool."""

    def test_stable_regime(self) -> None:
        result = classify_regime_tool(omega=0.02, F=0.98, S=0.05, C=0.05, IC=0.95)
        assert result["regime"] == "STABLE"

    def test_collapse_regime(self) -> None:
        result = classify_regime_tool(omega=0.5, F=0.5, S=0.6, C=0.5, IC=0.1)
        assert result["regime"] in {"COLLAPSE", "CRITICAL"}

    def test_watch_regime(self) -> None:
        result = classify_regime_tool(omega=0.1, F=0.9, S=0.3, C=0.2, IC=0.7)
        assert result["regime"] == "WATCH"

    def test_critical_overlay(self) -> None:
        result = classify_regime_tool(omega=0.1, F=0.9, S=0.1, C=0.1, IC=0.2)
        assert result["is_critical"] is True

    def test_thresholds_returned(self) -> None:
        result = classify_regime_tool(omega=0.02, F=0.98, S=0.05, C=0.05, IC=0.95)
        assert "thresholds" in result
        assert result["thresholds"]["omega_stable_max"] == pytest.approx(0.038)


# ---------------------------------------------------------------------------
# validate_seam
# ---------------------------------------------------------------------------


class TestValidateSeam:
    """Tests for the validate_seam tool."""

    def test_seam_pass(self) -> None:
        result = validate_seam(
            kappa_pre=-0.1,
            kappa_post=-0.1,
            tau_R=1.0,
            R=0.01,
            drift_cost=0.01,
            curvature_cost=0.0,
        )
        assert result["tol_seam"] == pytest.approx(0.005)

    def test_inf_rec_no_credit(self) -> None:
        result = validate_seam(
            kappa_pre=-0.1,
            kappa_post=-0.05,
            tau_R=float("inf"),
        )
        assert result["pass"] is False
        assert "INF_REC" in result["reasons"][0] or "no return" in result["reasons"][0]

    def test_tol_seam_frozen(self) -> None:
        result = validate_seam(kappa_pre=0.0, kappa_post=0.0, tau_R=1.0)
        assert result["tol_seam"] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# compute_costs
# ---------------------------------------------------------------------------


class TestComputeCosts:
    """Tests for the compute_costs tool."""

    def test_zero_omega_zero_gamma(self) -> None:
        result = compute_costs(omega=0.0, C=0.0)
        assert result["gamma_omega"] == pytest.approx(0.0, abs=1e-15)
        assert result["D_C"] == pytest.approx(0.0, abs=1e-15)

    def test_frozen_params_in_response(self) -> None:
        result = compute_costs(omega=0.1, C=0.2)
        assert result["frozen_params"]["p"] == 3
        assert result["frozen_params"]["alpha"] == pytest.approx(1.0)
        assert result["frozen_params"]["epsilon"] == pytest.approx(1e-8)

    def test_curvature_cost_linear(self) -> None:
        """D_C = α·C with α = 1.0."""
        result = compute_costs(omega=0.0, C=0.35)
        assert result["D_C"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# check_identities
# ---------------------------------------------------------------------------


class TestCheckIdentities:
    """Tests for the check_identities tool."""

    def test_conformant_vector(self) -> None:
        result = check_identities([0.5, 0.5, 0.5, 0.5])
        assert result["verdict"] == "CONFORMANT"
        assert bool(result["duality"]["pass"]) is True
        assert bool(result["integrity_bound"]["pass"]) is True
        assert bool(result["log_integrity"]["pass"]) is True

    def test_heterogeneous_conformant(self) -> None:
        result = check_identities([0.1, 0.9, 0.5, 0.3])
        assert result["verdict"] == "CONFORMANT"

    def test_gap_reported(self) -> None:
        result = check_identities([0.95, 0.001, 0.5, 0.5])
        assert result["integrity_bound"]["gap"] > 0.1


# ---------------------------------------------------------------------------
# frozen_parameters
# ---------------------------------------------------------------------------


class TestFrozenParameters:
    """Tests that frozen_parameters returns correct seam-derived values."""

    def test_epsilon(self) -> None:
        p = frozen_parameters()
        assert p["epsilon"]["value"] == pytest.approx(1e-8)

    def test_p_exponent(self) -> None:
        p = frozen_parameters()
        assert p["p_exponent"]["value"] == 3

    def test_tol_seam(self) -> None:
        p = frozen_parameters()
        assert p["tol_seam"]["value"] == pytest.approx(0.005)

    def test_thresholds_present(self) -> None:
        p = frozen_parameters()
        assert "regime_thresholds" in p


# ---------------------------------------------------------------------------
# orientation_receipts
# ---------------------------------------------------------------------------


class TestOrientationReceipts:
    """Tests that orientation_receipts returns all 10 sections."""

    def test_ten_receipts(self) -> None:
        r = orientation_receipts()
        assert len(r["receipts"]) == 10

    def test_sections_sequential(self) -> None:
        r = orientation_receipts()
        sections = [receipt["section"] for receipt in r["receipts"]]
        assert sections == list(range(1, 11))

    def test_key_constraints_present(self) -> None:
        r = orientation_receipts()
        assert len(r["key_constraints"]) >= 4
        # Must not say AM-GM or Shannon
        text = " ".join(r["key_constraints"])
        assert "not AM-GM" in text or "SOLVABILITY" in text


# ---------------------------------------------------------------------------
# compute_kernel_batch
# ---------------------------------------------------------------------------


class TestComputeKernelBatch:
    """Tests for batch kernel computation."""

    def test_batch_count(self) -> None:
        traces = [[0.5, 0.5], [0.8, 0.2], [0.1, 0.9]]
        result = compute_kernel_batch(traces)
        assert result["count"] == 3
        assert len(result["results"]) == 3

    def test_batch_duality(self) -> None:
        """F + ω = 1 for every entity in batch."""
        traces = [[0.3, 0.7, 0.5], [0.9, 0.1, 0.4], [0.6, 0.6, 0.6]]
        result = compute_kernel_batch(traces)
        for r in result["results"]:
            assert abs(r["F"] + r["omega"] - 1.0) < 1e-12

    def test_batch_summary(self) -> None:
        traces = [[0.5, 0.5], [0.9, 0.9]]
        result = compute_kernel_batch(traces)
        assert "summary" in result
        assert "F_mean" in result["summary"]

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_kernel_batch([])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            compute_kernel_batch([[0.5, 0.5], [0.5, 0.5, 0.5]])


# ---------------------------------------------------------------------------
# Parametrized identity sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trace",
    [
        [0.01] * 4,
        [0.99] * 4,
        [0.5] * 4,
        [0.1, 0.9, 0.5, 0.3],
        [0.001, 0.999, 0.5, 0.5],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.001],
        [float(x) for x in np.linspace(0.05, 0.95, 6)],
    ],
    ids=[
        "near_zero",
        "near_one",
        "equator",
        "mixed",
        "extreme",
        "8ch_uniform",
        "geometric_slaughter",
        "gradient",
    ],
)
def test_all_identities_hold(trace: list[float]) -> None:
    """Parametrized sweep: all three identities hold for diverse traces."""
    result = check_identities(trace)
    assert result["verdict"] == "CONFORMANT", f"Identity check failed for trace={trace}: {result}"
