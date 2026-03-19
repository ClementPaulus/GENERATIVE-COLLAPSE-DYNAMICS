"""Market Microstructure Closure — Finance Domain.

Tier-2 closure mapping 12 market venue types through the GCD kernel.
Each venue is characterized by 8 channels drawn from market microstructure theory.

Channels (8, equal weights w_i = 1/8):
  0  bid_ask_spread          — tightness of the spread (normalized, 1 = tight)
  1  order_book_depth        — available liquidity at best quotes (normalized)
  2  tick_frequency          — trade arrival rate (normalized)
  3  price_impact            — resilience to large orders (1 = low impact)
  4  maker_taker_ratio       — proportion of passive vs aggressive flow
  5  latency_sensitivity     — structural advantage of speed (1 = low sensitivity)
  6  information_asymmetry   — adverse selection cost (1 = low asymmetry)
  7  regulatory_transparency — reporting and oversight quality (normalized)

12 entities across 4 categories:
  Equity (3): NYSE_listed, NASDAQ_electronic, dark_pool
  Fixed income (3): US_treasury, corporate_bond, municipal_bond
  Derivatives (3): options_exchange, futures_CME, OTC_swap
  Alternative (3): forex_spot, crypto_spot, prediction_market

6 theorems (T-MM-1 through T-MM-6).
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

MM_CHANNELS = [
    "bid_ask_spread",
    "order_book_depth",
    "tick_frequency",
    "price_impact",
    "maker_taker_ratio",
    "latency_sensitivity",
    "information_asymmetry",
    "regulatory_transparency",
]
N_MM_CHANNELS = len(MM_CHANNELS)


@dataclass(frozen=True, slots=True)
class MarketVenueEntity:
    """A market venue type with 8 measurable channels."""

    name: str
    category: str
    bid_ask_spread: float
    order_book_depth: float
    tick_frequency: float
    price_impact: float
    maker_taker_ratio: float
    latency_sensitivity: float
    information_asymmetry: float
    regulatory_transparency: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.bid_ask_spread,
                self.order_book_depth,
                self.tick_frequency,
                self.price_impact,
                self.maker_taker_ratio,
                self.latency_sensitivity,
                self.information_asymmetry,
                self.regulatory_transparency,
            ]
        )


MM_ENTITIES: tuple[MarketVenueEntity, ...] = (
    # Equity venues
    MarketVenueEntity("NYSE_listed", "equity", 0.90, 0.92, 0.85, 0.80, 0.70, 0.50, 0.75, 0.95),
    MarketVenueEntity("NASDAQ_electronic", "equity", 0.92, 0.88, 0.95, 0.78, 0.65, 0.35, 0.72, 0.90),
    MarketVenueEntity("dark_pool", "equity", 0.85, 0.40, 0.30, 0.90, 0.90, 0.70, 0.40, 0.50),
    # Fixed income
    MarketVenueEntity("US_treasury", "fixed_income", 0.95, 0.95, 0.70, 0.90, 0.75, 0.60, 0.85, 0.98),
    MarketVenueEntity("corporate_bond", "fixed_income", 0.50, 0.45, 0.30, 0.40, 0.55, 0.80, 0.50, 0.80),
    MarketVenueEntity("municipal_bond", "fixed_income", 0.40, 0.35, 0.20, 0.35, 0.60, 0.85, 0.45, 0.75),
    # Derivatives
    MarketVenueEntity("options_exchange", "derivatives", 0.70, 0.65, 0.80, 0.55, 0.60, 0.40, 0.50, 0.85),
    MarketVenueEntity("futures_CME", "derivatives", 0.88, 0.90, 0.90, 0.75, 0.65, 0.30, 0.70, 0.92),
    MarketVenueEntity("OTC_swap", "derivatives", 0.30, 0.25, 0.15, 0.30, 0.50, 0.90, 0.35, 0.60),
    # Alternative
    MarketVenueEntity("forex_spot", "alternative", 0.98, 0.95, 0.98, 0.85, 0.55, 0.25, 0.65, 0.70),
    MarketVenueEntity("crypto_spot", "alternative", 0.60, 0.50, 0.90, 0.40, 0.45, 0.55, 0.30, 0.20),
    MarketVenueEntity("prediction_market", "alternative", 0.35, 0.20, 0.25, 0.30, 0.70, 0.80, 0.25, 0.40),
)


@dataclass(frozen=True, slots=True)
class MMKernelResult:
    """Kernel output for a market venue entity."""

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


def compute_mm_kernel(entity: MarketVenueEntity) -> MMKernelResult:
    """Compute GCD kernel for a market venue entity."""
    c = entity.trace_vector()
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(N_MM_CHANNELS) / N_MM_CHANNELS
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
    return MMKernelResult(
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


def compute_all_entities() -> list[MMKernelResult]:
    """Compute kernel outputs for all market venue entities."""
    return [compute_mm_kernel(e) for e in MM_ENTITIES]


# ── Theorems ──────────────────────────────────────────────────────────


def verify_t_mm_1(results: list[MMKernelResult]) -> dict:
    """T-MM-1: US Treasuries have highest F — deepest, most liquid,
    most transparent market in the world.
    """
    treas = next(r for r in results if r.name == "US_treasury")
    max_F = max(r.F for r in results)
    passed = abs(treas.F - max_F) < 0.05
    return {"name": "T-MM-1", "passed": bool(passed), "treasury_F": treas.F, "max_F": float(max_F)}


def verify_t_mm_2(results: list[MMKernelResult]) -> dict:
    """T-MM-2: OTC swaps and prediction markets are in Collapse regime.

    Low transparency, thin liquidity, high information asymmetry.
    """
    otc = next(r for r in results if r.name == "OTC_swap")
    pred = next(r for r in results if r.name == "prediction_market")
    passed = otc.regime == "Collapse" and pred.regime == "Collapse"
    return {"name": "T-MM-2", "passed": bool(passed), "OTC_regime": otc.regime, "prediction_regime": pred.regime}


def verify_t_mm_3(results: list[MMKernelResult]) -> dict:
    """T-MM-3: Forex spot has largest heterogeneity gap among alternative
    venues — 24-hour trading with high latency sensitivity and
    information asymmetry creates channel divergence.
    """
    alt = [r for r in results if r.category == "alternative"]
    forex = next(r for r in alt if r.name == "forex_spot")
    forex_gap = forex.F - forex.IC
    max_alt_gap = max(r.F - r.IC for r in alt)
    passed = abs(forex_gap - max_alt_gap) < 0.01
    return {"name": "T-MM-3", "passed": bool(passed), "forex_gap": float(forex_gap), "max_alt_gap": float(max_alt_gap)}


def verify_t_mm_4(results: list[MMKernelResult]) -> dict:
    """T-MM-4: Forex spot has highest F among alternative venues.

    24-hour, $7T daily volume, extremely tight spreads.
    """
    alt = [r for r in results if r.category == "alternative"]
    forex = next(r for r in alt if r.name == "forex_spot")
    max_alt_F = max(r.F for r in alt)
    passed = abs(forex.F - max_alt_F) < 0.01
    return {"name": "T-MM-4", "passed": bool(passed), "forex_F": forex.F, "max_alt_F": float(max_alt_F)}


def verify_t_mm_5(results: list[MMKernelResult]) -> dict:
    """T-MM-5: Dark pools have largest heterogeneity gap among equity venues.

    Opaque order flow creates channel imbalance: tight spreads but
    low tick transparency and high information asymmetry.
    """
    eq = [r for r in results if r.category == "equity"]
    dp = next(r for r in eq if r.name == "dark_pool")
    dp_gap = dp.F - dp.IC
    max_eq_gap = max(r.F - r.IC for r in eq)
    passed = abs(dp_gap - max_eq_gap) < 0.01
    return {
        "name": "T-MM-5",
        "passed": bool(passed),
        "dark_pool_gap": float(dp_gap),
        "max_equity_gap": float(max_eq_gap),
    }


def verify_t_mm_6(results: list[MMKernelResult]) -> dict:
    """T-MM-6: Exchange-traded venues have higher mean F than OTC venues.

    Centralized clearing, standardized contracts, regulatory oversight.
    """
    exchange = [
        r.F
        for r in results
        if r.name in ("NYSE_listed", "NASDAQ_electronic", "options_exchange", "futures_CME", "forex_spot")
    ]
    otc = [
        r.F
        for r in results
        if r.name in ("dark_pool", "OTC_swap", "prediction_market", "corporate_bond", "municipal_bond")
    ]
    passed = np.mean(exchange) > np.mean(otc)
    return {
        "name": "T-MM-6",
        "passed": bool(passed),
        "exchange_mean_F": float(np.mean(exchange)),
        "otc_mean_F": float(np.mean(otc)),
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-MM theorems."""
    results = compute_all_entities()
    return [
        verify_t_mm_1(results),
        verify_t_mm_2(results),
        verify_t_mm_3(results),
        verify_t_mm_4(results),
        verify_t_mm_5(results),
        verify_t_mm_6(results),
    ]


def main() -> None:
    """Entry point."""
    results = compute_all_entities()
    print("=" * 78)
    print("MARKET MICROSTRUCTURE — GCD KERNEL ANALYSIS")
    print("=" * 78)
    print(f"{'Entity':<24} {'Cat':<14} {'F':>6} {'ω':>6} {'IC':>6} {'Δ':>6} {'Regime'}")
    print("-" * 78)
    for r in results:
        gap = r.F - r.IC
        print(f"{r.name:<24} {r.category:<14} {r.F:6.3f} {r.omega:6.3f} {r.IC:6.3f} {gap:6.3f} {r.regime}")

    print("\n── Theorems ──")
    for t in verify_all_theorems():
        print(f"  {t['name']}: {'PROVEN' if t['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
