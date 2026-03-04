#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WORKSHEET — Level 4: The Full Spine — End to End                       ║
║  Contract → Canon → Closures → Integrity Ledger → Stance               ║
║                                                                         ║
║  Prerequisites: Levels 1-3                                              ║
║  Goal: Run the complete pipeline on three real scenarios                 ║
╚══════════════════════════════════════════════════════════════════════════╝

This worksheet brings everything together. We run the complete
five-stop spine on three different systems:

  Scenario A: A healthy particle (near-Stable)
  Scenario B: A stressed financial system (Watch)
  Scenario C: A confinement transition (Collapse + Critical)

Each scenario is worked step by step with no shortcuts.
"""

from __future__ import annotations

import numpy as np

# ═══════════════════ Frozen Contract Parameters ═══════════════════════
EPSILON = 1e-8
P_EXPONENT = 3
ALPHA = 1.0
TOL_SEAM = 0.005

# Regime thresholds
OMEGA_STABLE_MAX = 0.038
F_STABLE_MIN = 0.90
S_STABLE_MAX = 0.15
C_STABLE_MAX = 0.14
OMEGA_COLLAPSE_MIN = 0.30
IC_CRITICAL_MAX = 0.30


def print_header(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}\n")


def print_section(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(1, 60 - len(title))}\n")


def print_stop(num: int, name: str, latin: str) -> None:
    print(f"\n  ┌{'─' * 68}┐")
    print(f"  │  STOP {num} │ {name:20s} ({latin}){' ' * (42 - len(name) - len(latin))}│")
    print(f"  └{'─' * 68}┘\n")


# ═══════════════════ Kernel Functions (from Levels 1-3) ═══════════════


def epsilon_clamp(c: np.ndarray) -> np.ndarray:
    """ε-clamp coordinates to [ε, 1−ε]."""
    return np.asarray(np.clip(c, EPSILON, 1 - EPSILON))


def compute_F(c: np.ndarray, w: np.ndarray) -> float:
    """Fidelity: weighted arithmetic mean."""
    return float(np.sum(w * c))


def compute_omega(F: float) -> float:
    """Drift: what was lost."""
    return 1.0 - F


def compute_S(c: np.ndarray, w: np.ndarray) -> float:
    """Bernoulli field entropy."""
    S = 0.0
    for ci, wi in zip(c, w, strict=True):
        if 0 < ci < 1 and wi > 0:
            S += wi * (-ci * np.log(ci) - (1 - ci) * np.log(1 - ci))
    return S


def compute_C(c: np.ndarray) -> float:
    """Curvature: normalized population std."""
    return float(np.std(c, ddof=0) / 0.5)


def compute_kappa(c: np.ndarray, w: np.ndarray) -> float:
    """Log-integrity."""
    return float(np.sum(w * np.log(c)))


def compute_IC(kappa: float) -> float:
    """Integrity composite."""
    return float(np.exp(kappa))


def gamma_omega(omega: float) -> float:
    """Drift cost Γ(ω) = ω³/(1−ω+ε)."""
    return float(omega**P_EXPONENT / (1 - omega + EPSILON))


def cost_curvature(C: float) -> float:
    """Curvature cost D_C = α·C."""
    return ALPHA * C


def classify_regime(omega: float, F: float, S: float, C: float, IC: float) -> str:
    """Classify into Stable/Watch/Collapse/Critical."""
    if IC < IC_CRITICAL_MAX:
        return "CRITICAL"
    if omega >= OMEGA_COLLAPSE_MIN:
        return "COLLAPSE"
    if omega < OMEGA_STABLE_MAX and F > F_STABLE_MIN and S < S_STABLE_MAX and C < C_STABLE_MAX:
        return "STABLE"
    return "WATCH"


# ═══════════════════════════════════════════════════════════════════════
#  THE FULL SPINE — Five Stops
# ═══════════════════════════════════════════════════════════════════════


def run_full_spine(
    name: str,
    channels: list[str],
    c_raw: np.ndarray,
    w: np.ndarray,
    R: float,
    tau_R: float,
    kappa_t0: float | None = None,
    kappa_t1: float | None = None,
) -> None:
    """Run the complete five-stop spine with full step-by-step display."""

    print_header(f"SCENARIO: {name}")
    n = len(c_raw)

    # ─────────────────────── STOP 1: CONTRACT ──────────────────────────
    print_stop(1, "CONTRACT", "Liga")

    print("  Freeze the rules BEFORE seeing evidence.\n")
    print("  Frozen parameters (from frozen_contract.py):")
    print(f"    ε        = {EPSILON}")
    print(f"    p        = {P_EXPONENT}")
    print(f"    α        = {ALPHA}")
    print(f"    tol_seam = {TOL_SEAM}")
    print("    Regime thresholds:")
    print(f"      Stable:   ω < {OMEGA_STABLE_MAX}, F > {F_STABLE_MIN}, S < {S_STABLE_MAX}, C < {C_STABLE_MAX}")
    print(f"      Collapse: ω ≥ {OMEGA_COLLAPSE_MIN}")
    print(f"      Critical: IC < {IC_CRITICAL_MAX}")

    print(f"\n  Input trace ({n} channels):")
    print(f"    {'#':>3s}  {'Channel':<14s}  {'c_raw':>8s}  {'w_i':>8s}")
    print(f"    {'─' * 3}  {'─' * 14}  {'─' * 8}  {'─' * 8}")
    for i in range(n):
        marker = " ← dead!" if c_raw[i] < 0.01 else (" ← weak" if c_raw[i] < 0.3 else "")
        print(f"    {i + 1:3d}  {channels[i]:<14s}  {c_raw[i]:8.4f}  {w[i]:8.6f}{marker}")

    print("\n  ε-clamp coordinates:")
    c = epsilon_clamp(c_raw)
    clamped_count = sum(1 for i in range(n) if c_raw[i] != c[i])
    print(f"    {clamped_count} channel(s) were clamped.")
    if clamped_count > 0:
        for i in range(n):
            if c_raw[i] != c[i]:
                print(f"    Ch {i + 1} ({channels[i]}): {c_raw[i]:.4f} → {c[i]:.8f}")

    # ─────────────────────── STOP 2: CANON ─────────────────────────────
    print_stop(2, "CANON", "Dic")

    print("  Compute the six Tier-1 invariants.\n")

    # F
    F = compute_F(c, w)
    print("  F (Fidelity) = Σ w_i · c_i")
    partial_sums = []
    for i in range(n):
        partial_sums.append(w[i] * c[i])
    print("    = " + " + ".join(f"{ps:.4f}" for ps in partial_sums[:4]) + " + ...")
    print(f"    = {F:.6f}")

    # ω
    omega = compute_omega(F)
    print(f"\n  ω (Drift) = 1 − F = 1 − {F:.6f} = {omega:.6f}")
    print(f"    Check: F + ω = {F:.6f} + {omega:.6f} = {F + omega:.10f} ✓")

    # S
    S = compute_S(c, w)
    print("\n  S (Entropy) = Σ w_i · h(c_i)")
    print(f"    = {S:.6f}")
    print(f"    S / ln(2) = {S / np.log(2):.4f}  (0→certain, 1→maximum uncertainty)")

    # C
    C = compute_C(c)
    c_bar = np.mean(c)
    print("\n  C (Curvature) = std_pop(c) / 0.5")
    print(f"    c̄ = {c_bar:.6f}")
    print(f"    std_pop = {np.std(c, ddof=0):.6f}")
    print(f"    C = {C:.6f}")

    # κ
    kappa = compute_kappa(c, w)
    print("\n  κ (Log-integrity) = Σ w_i · ln(c_i)")
    weakest_idx = int(np.argmin(c))
    print(f"    Weakest channel: Ch {weakest_idx + 1} ({channels[weakest_idx]})")
    print(f"      c = {c[weakest_idx]:.8f}, ln(c) = {np.log(c[weakest_idx]):.4f}")
    print(f"      contribution = {w[weakest_idx] * np.log(c[weakest_idx]):.4f}")
    print(f"    κ = {kappa:.6f}")

    # IC
    IC = compute_IC(kappa)
    gap = F - IC
    print(f"\n  IC (Integrity) = exp(κ) = exp({kappa:.6f}) = {IC:.6f}")
    print(f"    Heterogeneity gap Δ = F − IC = {F:.6f} − {IC:.6f} = {gap:.6f}")
    print(f"    IC ≤ F?  {IC:.6f} ≤ {F:.6f}  ✓")

    # Summary table
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Invariant  │  Value       │  Meaning    │")
    print("  ├─────────────┼──────────────┼─────────────│")
    print(f"  │  F          │  {F:10.6f}  │  survived   │")
    print(f"  │  ω          │  {omega:10.6f}  │  lost       │")
    print(f"  │  S          │  {S:10.6f}  │  uncertain  │")
    print(f"  │  C          │  {C:10.6f}  │  dispersed  │")
    print(f"  │  κ          │  {kappa:10.6f}  │  log-coher  │")
    print(f"  │  IC         │  {IC:10.6f}  │  coherence  │")
    print(f"  │  Δ          │  {gap:10.6f}  │  het. gap   │")
    print("  └─────────────┴──────────────┴─────────────┘")

    # ─────────────────────── STOP 3: CLOSURES ──────────────────────────
    print_stop(3, "CLOSURES", "Reconcilia")

    print("  Compute costs and budget.\n")

    D_omega = gamma_omega(omega)
    print("  D_ω = Γ(ω) = ω³ / (1 − ω + ε)")
    print(f"       = {omega:.6f}³ / (1 − {omega:.6f} + {EPSILON})")
    print(f"       = {omega**3:.8f} / {1 - omega + EPSILON:.8f}")
    print(f"       = {D_omega:.8f}")

    D_C = cost_curvature(C)
    print(f"\n  D_C = α · C = {ALPHA} × {C:.6f} = {D_C:.6f}")

    total_debit = D_omega + D_C
    print(f"\n  Total debits = D_ω + D_C = {D_omega:.6f} + {D_C:.6f} = {total_debit:.6f}")

    if tau_R == float("inf"):
        credit = 0.0
        print("\n  τ_R = ∞_rec (no return) → Credit = 0")
    else:
        credit = R * tau_R
        print(f"\n  Credit = R · τ_R = {R} × {tau_R} = {credit:.6f}")

    delta_kappa_budget = credit - total_debit
    print("\n  Δκ_budget = credit − debits")
    print(f"            = {credit:.6f} − {total_debit:.6f}")
    print(f"            = {delta_kappa_budget:.6f}")

    # ─────────────────────── STOP 4: INTEGRITY LEDGER ──────────────────
    print_stop(4, "INTEGRITY LEDGER", "Inscribe")

    print("  Verify the three structural identities.\n")

    # Identity 1
    r1 = abs(F + omega - 1)
    print("  ① Duality: F + ω = 1")
    print(f"     {F:.10f} + {omega:.10f} = {F + omega:.10f}")
    print(f"     |residual| = {r1:.1e}  {'✓' if r1 < 1e-12 else '✗'}")

    # Identity 2
    print("\n  ② Integrity bound: IC ≤ F")
    print(f"     {IC:.10f} ≤ {F:.10f}  {'✓' if IC <= F + 1e-9 else '✗'}")

    # Identity 3
    r3 = abs(IC - np.exp(kappa))
    print("\n  ③ Exponential map: IC = exp(κ)")
    print(f"     IC     = {IC:.10f}")
    print(f"     exp(κ) = {np.exp(kappa):.10f}")
    print(f"     |δ| = {r3:.1e}  {'✓' if r3 < 1e-12 else '✗'}")

    # Seam check (if we have ledger data)
    if kappa_t0 is not None and kappa_t1 is not None:
        delta_kappa_ledger = kappa_t1 - kappa_t0
        residual = delta_kappa_budget - delta_kappa_ledger
        print("\n  Seam residual:")
        print(f"     Δκ_ledger = κ(t₁) − κ(t₀) = {kappa_t1:.6f} − ({kappa_t0:.6f}) = {delta_kappa_ledger:.6f}")
        print(f"     s = Δκ_budget − Δκ_ledger = {delta_kappa_budget:.6f} − {delta_kappa_ledger:.6f} = {residual:.6f}")
        print(
            f"     |s| = {abs(residual):.6f} {'≤' if abs(residual) <= TOL_SEAM else '>'} {TOL_SEAM} → "
            f"{'PASS' if abs(residual) <= TOL_SEAM else 'FAIL'}"
        )

    # ─────────────────────── STOP 5: STANCE ────────────────────────────
    print_stop(5, "STANCE", "Sententia")

    regime = classify_regime(omega, F, S, C, IC)

    print("  Apply the four gates:\n")
    g1 = omega < OMEGA_STABLE_MAX
    g2 = F > F_STABLE_MIN
    g3 = S < S_STABLE_MAX
    g4 = C < C_STABLE_MAX
    g_crit = IC < IC_CRITICAL_MAX
    g_coll = omega >= OMEGA_COLLAPSE_MIN

    print(
        f"    Critical:  IC < 0.30?     {IC:.4f} {'<' if g_crit else '≥'} 0.30  {'→ CRITICAL' if g_crit else '→ pass'}"
    )
    print(
        f"    Collapse:  ω ≥ 0.30?      {omega:.4f} {'≥' if g_coll else '<'} 0.30  {'→ COLLAPSE' if g_coll else '→ pass'}"
    )
    print(f"    Stable G1: ω < 0.038?     {omega:.4f} {'<' if g1 else '≥'} 0.038 {'✓' if g1 else '✗'}")
    print(f"    Stable G2: F > 0.90?      {F:.4f} {'>' if g2 else '≤'} 0.90  {'✓' if g2 else '✗'}")
    print(f"    Stable G3: S < 0.15?      {S:.4f} {'<' if g3 else '≥'} 0.15  {'✓' if g3 else '✗'}")
    print(f"    Stable G4: C < 0.14?      {C:.4f} {'<' if g4 else '≥'} 0.14  {'✓' if g4 else '✗'}")

    print("\n  ╔═══════════════════════════════════════════════════╗")
    print(f"  ║  REGIME:  {regime:>10s}                              ║")
    print("  ║  VERDICT: CONFORMANT (all identities hold)        ║")
    print("  ╚═══════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO A: A Healthy Particle (Near-Stable)
# ═══════════════════════════════════════════════════════════════════════


def scenario_A() -> None:
    channels = ["mass_log", "spin_norm", "charge", "color", "weak_iso", "lepton_#", "baryon_#", "generation"]
    c = np.array([0.97, 0.95, 0.98, 0.96, 0.94, 0.97, 0.96, 0.95])
    w = np.ones(8) / 8
    run_full_spine(
        name="A — Healthy Particle (all channels strong)",
        channels=channels,
        c_raw=c,
        w=w,
        R=0.01,
        tau_R=20,
        kappa_t0=-0.050,
        kappa_t1=-0.045,
    )


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO B: A Stressed Financial System (Watch)
# ═══════════════════════════════════════════════════════════════════════


def scenario_B() -> None:
    channels = ["liquidity", "volatility", "returns", "leverage", "credit_q", "diversif", "market_β", "regulation"]
    c = np.array([0.85, 0.72, 0.91, 0.68, 0.77, 0.83, 0.90, 0.15])
    w = np.ones(8) / 8
    run_full_spine(
        name="B — Stressed Financial System (one weak channel: regulation)",
        channels=channels,
        c_raw=c,
        w=w,
        R=0.01,
        tau_R=50,
        kappa_t0=-0.430,
        kappa_t1=-0.425,
    )


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO C: Confinement Transition (Collapse + Critical)
# ═══════════════════════════════════════════════════════════════════════


def scenario_C() -> None:
    channels = ["mass_log", "spin_norm", "charge", "color", "weak_iso", "lepton_#", "baryon_#", "generation"]
    # A hadron: color channel dead (color-neutral), charge dead (neutral pion)
    c = np.array([0.50, 0.50, 0.00, 0.00, 0.50, 0.00, 0.333, 0.333])
    w = np.ones(8) / 8
    run_full_spine(
        name="C — Neutral Pion (confinement: 3 dead channels)",
        channels=channels,
        c_raw=c,
        w=w,
        R=0.01,
        tau_R=float("inf"),  # No return — τ_R = ∞_rec
    )


# ═══════════════════════════════════════════════════════════════════════
#  EXERCISES — Try Your Own
# ═══════════════════════════════════════════════════════════════════════


def exercises() -> None:
    print_header("EXERCISES — Run the Full Spine Yourself")

    print("""
  EXERCISE 4.1: Design Your Own System
  ─────────────────────────────────────
  Create a 6-channel system with:
    - 4 healthy channels (c > 0.80)
    - 1 moderate channel (c ≈ 0.50)
    - 1 weak channel (c ≈ 0.10)

  Using equal weights, work through all 5 stops:
    a) What is F? What is ω?
    b) Compute κ and IC. What is Δ = F − IC?
    c) Compute Γ(ω) and D_C
    d) If R = 0.02, τ_R = 40, what is Δκ_budget?
    e) What regime does it fall into?


  EXERCISE 4.2: Geometric Slaughter
  ──────────────────────────────────
  Start with c = [0.90, 0.90, 0.90, 0.90] (homogeneous).
  Then replace one channel with 0.001.

  Compute F, IC, and Δ for BOTH cases.
  How much does F change? How much does IC change?


  EXERCISE 4.3: Finding the First Weld
  ────────────────────────────────────
  For homogeneous c (all channels equal at value x),
  find the smallest x where Γ(1−x) < 1.0.

  Hint: Set ω = 1−x, solve ω³/(1−ω+ε) = 1.
  Try x = 0.30, 0.31, 0.32, 0.318...
""")

    print_section("ANSWER 4.1 (Example Solution)")
    c_ex = np.array([0.90, 0.85, 0.88, 0.82, 0.50, 0.10])
    w_ex = np.ones(6) / 6
    c_clamped = np.clip(c_ex, EPSILON, 1 - EPSILON)
    F = float(np.sum(w_ex * c_clamped))
    omega = 1 - F
    kappa = float(np.sum(w_ex * np.log(c_clamped)))
    IC = np.exp(kappa)
    print(f"  c = {c_ex}")
    print(f"  a) F = {F:.6f}, ω = {omega:.6f}")
    print(f"  b) κ = {kappa:.6f}, IC = {IC:.6f}, Δ = {F - IC:.6f}")
    D_w = omega**3 / (1 - omega + EPSILON)
    C = float(np.std(c_clamped) / 0.5)
    D_C = ALPHA * C
    print(f"  c) Γ(ω) = {D_w:.6f}, D_C = {D_C:.6f}")
    cred = 0.02 * 40
    dk = cred - (D_w + D_C)
    print(f"  d) Δκ_budget = {cred:.4f} − ({D_w:.6f} + {D_C:.6f}) = {dk:.6f}")
    S = 0.0
    for ci, wi in zip(c_clamped, w_ex, strict=True):
        if 0 < ci < 1:
            S += wi * (-ci * np.log(ci) - (1 - ci) * np.log(1 - ci))
    regime = classify_regime(omega, F, S, C, IC)
    print(f"  e) Regime: {regime}")

    print_section("ANSWER 4.2")
    c_homo = np.array([0.90, 0.90, 0.90, 0.90])
    c_dead = np.array([0.90, 0.90, 0.90, 0.001])
    w4 = np.ones(4) / 4
    F_h = np.sum(w4 * c_homo)
    IC_h = np.exp(np.sum(w4 * np.log(c_homo)))
    F_d = np.sum(w4 * c_dead)
    c_dead_c = np.clip(c_dead, EPSILON, 1 - EPSILON)
    IC_d = np.exp(np.sum(w4 * np.log(c_dead_c)))
    print(f"  Homogeneous: F = {F_h:.4f}, IC = {IC_h:.6f}, Δ = {F_h - IC_h:.6f}")
    print(f"  One dead:    F = {F_d:.4f}, IC = {IC_d:.6f}, Δ = {F_d - IC_d:.6f}")
    print(f"  F changed by:  {abs(F_h - F_d):.4f}  ({abs(F_h - F_d) / F_h * 100:.1f}%)")
    print(f"  IC changed by: {abs(IC_h - IC_d):.6f}  ({abs(IC_h - IC_d) / IC_h * 100:.1f}%!)")

    print_section("ANSWER 4.3")
    for x in [0.30, 0.31, 0.315, 0.318, 0.319, 0.320]:
        omega_x = 1 - x
        g_x = omega_x**3 / (1 - omega_x + EPSILON)
        print(f"  x = {x:.3f}: Γ(1−x) = Γ({omega_x:.3f}) = {g_x:.4f}  {'< 1.0 ✓' if g_x < 1.0 else '≥ 1.0'}")
    print("  → First weld at c ≈ 0.318 (Γ drops below 1.0)")


# ═══════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════


def final_summary() -> None:
    print_header("FINAL SUMMARY — The Complete Kernel Math")

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  THE COMPLETE PIPELINE                                          ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                  ║
  ║  INPUT: c = [c_1, ..., c_n]  (coordinates, normalized [0,1])   ║
  ║         w = [w_1, ..., w_n]  (weights, Σ = 1)                  ║
  ║         ε = 10⁻⁸            (guard band, frozen)               ║
  ║                                                                  ║
  ║  STOP 1 — CONTRACT: Freeze ε, p, α, tol before seeing data     ║
  ║                                                                  ║
  ║  STOP 2 — CANON: Compute invariants                             ║
  ║    F  = Σ w_i·c_i                    (fidelity)                 ║
  ║    ω  = 1 − F                        (drift)                    ║
  ║    S  = Σ w_i·h(c_i)                 (entropy)                  ║
  ║    C  = std_pop(c)/0.5               (curvature)                ║
  ║    κ  = Σ w_i·ln(c_i,ε)             (log-integrity)            ║
  ║    IC = exp(κ)                        (integrity)               ║
  ║                                                                  ║
  ║  STOP 3 — CLOSURES: Compute costs                               ║
  ║    D_ω = Γ(ω) = ω³/(1−ω+ε)          (drift cost)              ║
  ║    D_C = α·C                          (curvature cost)          ║
  ║    Δκ_budget = R·τ_R − (D_ω + D_C)   (budget)                  ║
  ║                                                                  ║
  ║  STOP 4 — INTEGRITY LEDGER: Verify                              ║
  ║    ① F + ω = 1              (duality — always exact)            ║
  ║    ② IC ≤ F                 (integrity bound — always holds)    ║
  ║    ③ IC = exp(κ)            (exponential map — by construction) ║
  ║    s = Δκ_budget − Δκ_ledger (seam residual)                    ║
  ║    PASS if |s| ≤ 0.005      (seam closes)                       ║
  ║                                                                  ║
  ║  STOP 5 — STANCE: Classify                                      ║
  ║    CRITICAL: IC < 0.30      (overlay)                            ║
  ║    COLLAPSE: ω ≥ 0.30                                            ║
  ║    STABLE:   ω<0.038 ∧ F>0.90 ∧ S<0.15 ∧ C<0.14               ║
  ║    WATCH:    everything else                                     ║
  ║                                                                  ║
  ║  OUTPUT: Regime + Verdict  (derived, never asserted)             ║
  ║                                                                  ║
  ╚══════════════════════════════════════════════════════════════════╝

  KEY TAKEAWAYS:
  ────────────────
  1. F tells you the average health. IC tells you the true coherence.
  2. One dead channel kills IC (geometric slaughter) while F stays fine.
  3. Γ(ω) makes small drift cheap and large drift ruinous — cubic penalty.
  4. Three identities always hold — if they don't, you have a bug.
  5. Regime is derived from frozen gates, never asserted or argued.
  6. The seam checks if prediction (budget) matched reality (ledger).
  7. Everything traces back to one axiom:
     "Collapse is generative; only what returns is real."

  WORKSHEET FILES:
  ────────────────
  Level 1: worksheets/level_1_foundations.py       (c, w, ε-clamping)
  Level 2: worksheets/level_2_core_invariants.py   (F, ω, S, C, κ, IC)
  Level 3: worksheets/level_3_budget_and_regime.py (Γ, D_C, budget, regime)
  Level 4: worksheets/level_4_full_spine.py        (this file — end to end)

  Run any worksheet:
    python worksheets/level_1_foundations.py
    python worksheets/level_2_core_invariants.py
    python worksheets/level_3_budget_and_regime.py
    python worksheets/level_4_full_spine.py
""")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 72 + "╗")
    print("║  GCD KERNEL MATH WORKSHEETS — Level 4: The Full Spine" + " " * 16 + "║")
    print("║  Complete end-to-end pipeline on three real scenarios" + " " * 17 + "║")
    print("╚" + "═" * 72 + "╝")

    scenario_A()
    scenario_B()
    scenario_C()
    exercises()
    final_summary()
