#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WORKSHEET — Level 3: Cost Functions, Budget, and Regime Classification ║
║  How the kernel's invariants become decisions                           ║
║                                                                         ║
║  Prerequisites: Level 2 (F, ω, S, C, κ, IC)                            ║
║  Goal: Compute Γ(ω), D_C, the budget identity Δκ, and regime gates     ║
╚══════════════════════════════════════════════════════════════════════════╝

This worksheet teaches the PROTOCOL layer (Tier-0):
  1. Drift cost Γ(ω) — how expensive is it to have drifted?
  2. Curvature cost D_C — how expensive is channel dispersion?
  3. Budget identity Δκ — does the return credit cover the costs?
  4. Seam residual s — did the books balance?
  5. Regime gates — Stable / Watch / Collapse / Critical
"""

from __future__ import annotations

import numpy as np

# Frozen parameters — sourced from frozen_contract.py
EPSILON = 1e-8
P_EXPONENT = 3
ALPHA = 1.0
TOL_SEAM = 0.005


def print_header(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}\n")


def print_section(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(1, 60 - len(title))}\n")


# ════════════════════════════════════════════════════════════════════════
#  §3.1  DRIFT COST Γ(ω) — The Price of Losing Fidelity
# ════════════════════════════════════════════════════════════════════════


def section_3_1_drift_cost() -> None:
    # Lemma refs: L19 (Γ monotonicity), L18 (Γ pole at ω=1)
    print_header("§3.1  DRIFT COST Γ(ω) — The Price of Losing Fidelity")

    print(f"""  CONCEPT:
  ────────
  The drift cost function penalizes systems that have drifted far
  from perfect fidelity:

    Γ(ω) = ω^p / (1 − ω + ε)

  where:
    ω  = drift (1 − F)
    p  = {P_EXPONENT}      (frozen exponent — cubic penalty)
    ε  = {EPSILON}  (frozen guard band)

  KEY PROPERTIES:
  • Numerator ω^p: Cubic growth → small drift costs almost nothing,
    large drift costs enormously. This is why p = 3 (prime).
  • Denominator (1 − ω + ε): As ω → 1 (total loss), denominator → ε,
    creating a pole. Near-total loss costs ~infinity.
  • ε in denominator: Prevents division by zero at ω = 1.

  INTUITION: Γ(ω) is the "tax" on drift. Small drift = cheap.
  Large drift = ruinous. Near-total drift = impossible.
""")

    print_section("WORKED EXAMPLE: Compute Γ(ω) for Several Values")

    print(f"  Formula: Γ(ω) = ω³ / (1 − ω + {EPSILON})\n")
    print(f"    {'ω':>8s}  │  {'ω³':>12s}  │  {'1−ω+ε':>12s}  │  {'Γ(ω)':>12s}  │  {'Regime':>10s}")
    print(f"    {'─' * 8}  │  {'─' * 12}  │  {'─' * 12}  │  {'─' * 12}  │  {'─' * 10}")

    test_omegas = [0.01, 0.038, 0.05, 0.10, 0.20, 0.274, 0.30, 0.50, 0.70, 0.90, 0.99]
    for omega in test_omegas:
        numerator = omega**P_EXPONENT
        denominator = 1 - omega + EPSILON
        gamma = numerator / denominator
        regime = "Stable" if omega < 0.038 else ("Watch" if omega < 0.30 else "Collapse")
        print(f"    {omega:8.3f}  │  {numerator:12.6f}  │  {denominator:12.6f}  │  {gamma:12.6f}  │  {regime:>10s}")

    print_section("STEP-BY-STEP: Γ(0.274) — Our Running Example")

    omega = 0.273750
    num = omega**3
    den = 1 - omega + EPSILON

    print(f"  ω = {omega}")
    print(f"\n  Step 1: ω³ = {omega}³")
    print(f"         = {omega} × {omega} × {omega}")
    print(f"         = {omega**2:.6f} × {omega}")
    print(f"         = {num:.6f}")
    print(f"\n  Step 2: 1 − ω + ε = 1 − {omega} + {EPSILON}")
    print(f"         = {den:.10f}")
    print(f"\n  Step 3: Γ(ω) = {num:.6f} / {den:.6f}")
    gamma_val = num / den
    print(f"         = {gamma_val:.6f}")

    print_section("KEY INSIGHT: The Cubic Makes Low Drift Almost Free")
    print(f"    Γ(0.01) = {0.01**3 / (1 - 0.01 + EPSILON):.8f}  ← negligible!")
    print(f"    Γ(0.10) = {0.10**3 / (1 - 0.10 + EPSILON):.8f}  ← still small")
    print(f"    Γ(0.30) = {0.30**3 / (1 - 0.30 + EPSILON):.8f}  ← starting to bite")
    print(f"    Γ(0.50) = {0.50**3 / (1 - 0.50 + EPSILON):.8f}  ← expensive")
    print(f"    Γ(0.90) = {0.90**3 / (1 - 0.90 + EPSILON):.8f}  ← ruinous")
    print(f"    Γ(0.99) = {0.99**3 / (1 - 0.99 + EPSILON):.8f}  ← near pole!")

    print_section("EXERCISE 3.1")
    print("""  Compute Γ(ω) for ω = 0.15:
    a) ω³ = ?
    b) 1 − ω + ε = ?
    c) Γ(0.15) = ?
    d) What regime is this?
""")
    print_section("ANSWER 3.1")
    o = 0.15
    print(f"    a) ω³ = 0.15³ = {o**3:.6f}")
    print(f"    b) 1 − ω + ε = 1 − 0.15 + 10⁻⁸ = {1 - o + EPSILON:.10f}")
    g = o**3 / (1 - o + EPSILON)
    print(f"    c) Γ(0.15) = {o**3:.6f} / {1 - o + EPSILON:.6f} = {g:.6f}")
    print("    d) ω = 0.15, 0.038 ≤ 0.15 < 0.30 → Watch regime")


# ════════════════════════════════════════════════════════════════════════
#  §3.2  CURVATURE COST D_C — The Price of Channel Dispersion
# ════════════════════════════════════════════════════════════════════════


def section_3_2_curvature_cost() -> None:
    print_header("§3.2  CURVATURE COST D_C — The Price of Channel Dispersion")

    print(f"""  CONCEPT:
  ────────
  The curvature cost is simply:

    D_C = α · C

  where:
    C  = curvature (std_pop(c) / 0.5, from Level 2 §2.4)
    α  = {ALPHA}  (frozen coefficient)

  Since α = 1.0, D_C = C exactly. The coefficient could be different
  in other contracts, but the seam discovered α = 1.0.

  INTUITION: If your channels are spread out (high C), there is a cost.
  Heterogeneity = friction = roughness in the collapse field.
""")

    print_section("WORKED EXAMPLE: D_C for Our Running Example")

    C = 0.461621  # From Level 2
    D_C = ALPHA * C

    print(f"  From Level 2: C = {C:.6f}")
    print("  D_C = α × C")
    print(f"      = {ALPHA} × {C:.6f}")
    print(f"      = {D_C:.6f}")

    print_section("EXERCISE 3.2")
    print("""  For c = [0.9, 0.8, 0.7, 0.6], C = 0.223607 (from Level 2 Exercise):
  Compute D_C.
""")
    print_section("ANSWER 3.2")
    print("    D_C = 1.0 × 0.223607 = 0.223607")


# ════════════════════════════════════════════════════════════════════════
#  §3.3  THE BUDGET IDENTITY — Credits vs. Debits
# ════════════════════════════════════════════════════════════════════════


def section_3_3_budget() -> None:
    # Lemma refs: L20 (Δκ composition), L21 (budget identity)
    print_header("§3.3  THE BUDGET IDENTITY — Does Return Cover the Costs?")

    print("""  CONCEPT:
  ────────
  The budget identity is an accounting equation:

    Δκ_budget = R · τ_R − (D_ω + D_C)
                ─────────   ───────────
                 CREDIT       DEBITS

  where:
    R    = return credit rate (how much credit per unit return time)
    τ_R  = return time (how long until the system re-enters)
    D_ω  = Γ(ω) = drift cost (debit)
    D_C  = α·C  = curvature cost (debit)

  ACCOUNTING METAPHOR:
    Credit: R · τ_R  = "You returned, here's your reward"
    Debit:  D_ω      = "You drifted, here's the penalty"
    Debit:  D_C      = "Your channels were rough, here's the friction cost"
    Net:    Δκ        = credit − debits

  If Δκ > 0: system gained integrity (return succeeded)
  If Δκ < 0: system lost integrity (costs exceeded return)
  If Δκ ≈ 0: break-even
""")

    print_section("WORKED EXAMPLE: Full Budget Calculation")

    # Values from our running example
    omega = 0.273750
    C = 0.461621

    # Compute costs
    D_omega = omega**P_EXPONENT / (1 - omega + EPSILON)
    D_C = ALPHA * C

    # Budget parameters
    R = 0.01  # Typical return rate
    tau_R = 50.0  # 50 timesteps to return

    print("  INPUT VALUES:")
    print(f"    ω    = {omega:.6f}")
    print(f"    C    = {C:.6f}")
    print(f"    R    = {R}")
    print(f"    τ_R  = {tau_R}")

    print("\n  STEP 1: Compute D_ω = Γ(ω)")
    print("    D_ω = ω³ / (1 − ω + ε)")
    print(f"        = {omega:.6f}³ / (1 − {omega:.6f} + {EPSILON})")
    print(f"        = {omega**3:.6f} / {1 - omega + EPSILON:.6f}")
    print(f"        = {D_omega:.6f}")

    print("\n  STEP 2: Compute D_C = α · C")
    print(f"    D_C = {ALPHA} × {C:.6f}")
    print(f"        = {D_C:.6f}")

    print("\n  STEP 3: Total debits")
    total_debit = D_omega + D_C
    print(f"    D_ω + D_C = {D_omega:.6f} + {D_C:.6f} = {total_debit:.6f}")

    print("\n  STEP 4: Compute credit = R · τ_R")
    credit = R * tau_R
    print(f"    R · τ_R = {R} × {tau_R} = {credit:.6f}")

    print("\n  STEP 5: Budget Δκ = credit − debits")
    delta_kappa = credit - total_debit
    print(f"    Δκ = {credit:.6f} − {total_debit:.6f}")
    print(f"       = {delta_kappa:.6f}")

    if delta_kappa > 0:
        print("\n  RESULT: Δκ > 0 → System GAINED integrity (surplus!)")
    elif delta_kappa < 0:
        print("\n  RESULT: Δκ < 0 → System LOST integrity (deficit)")
    else:
        print("\n  RESULT: Δκ ≈ 0 → Break even")

    print_section("SPECIAL CASE: τ_R = ∞ (No Return)")
    print("""  If the system never returns (τ_R = ∞_rec = INF_REC):
    Credit = R · ∞ → In practice: credit = 0 (no return → no reward)
    Budget = 0 − (D_ω + D_C) < 0 → always negative
    Verdict: GESTURE (not a weld — the system didn't return)

  If τ_R = INF_REC, nulla fides datur. (No credit is given.)
""")

    print_section("EXERCISE 3.3")
    print("""  A system has:
    ω = 0.10, C = 0.20, R = 0.02, τ_R = 30

  Compute:
    a) D_ω = Γ(0.10)
    b) D_C = α × 0.20
    c) Total debits
    d) Credit = R × τ_R
    e) Δκ_budget = credit − debits
    f) Did the system gain or lose integrity?
""")
    print_section("ANSWER 3.3")
    o_ex = 0.10
    C_ex = 0.20
    D_w = o_ex**3 / (1 - o_ex + EPSILON)
    D_c = ALPHA * C_ex
    cred = 0.02 * 30
    dk = cred - (D_w + D_c)
    print(f"    a) D_ω = 0.10³ / (1 − 0.10 + ε) = {o_ex**3:.6f} / {1 - o_ex + EPSILON:.6f} = {D_w:.6f}")
    print(f"    b) D_C = 1.0 × 0.20 = {D_c:.6f}")
    print(f"    c) Total = {D_w:.6f} + {D_c:.6f} = {D_w + D_c:.6f}")
    print(f"    d) Credit = 0.02 × 30 = {cred:.6f}")
    print(f"    e) Δκ = {cred:.6f} − {D_w + D_c:.6f} = {dk:.6f}")
    print(f"    f) Δκ {'> 0 → gained integrity (surplus)' if dk > 0 else '< 0 → lost integrity (deficit)'}")


# ════════════════════════════════════════════════════════════════════════
#  §3.4  SEAM RESIDUAL — Did the Books Balance?
# ════════════════════════════════════════════════════════════════════════


def section_3_4_seam_residual() -> None:
    # Lemma refs: L27 (residual accumulation bound), L45 (seam associativity)
    print_header("§3.4  SEAM RESIDUAL — Did the Books Balance?")

    print(f"""  CONCEPT:
  ────────
  The seam residual compares two things:
    1. What the BUDGET predicted (Δκ_budget)
    2. What actually HAPPENED in the ledger (Δκ_ledger)

    s = Δκ_budget − Δκ_ledger

  PASS condition: |s| ≤ tol_seam = {TOL_SEAM}

  If |s| ≤ {TOL_SEAM}: The budget model matched reality → PASS (weld)
  If |s| > {TOL_SEAM}: The model failed to predict reality → FAIL (gesture)

  ANALOGY: You predicted your monthly expenses would be $1000.
  You actually spent $1003. Residual = $3. If tolerance is $5, you PASS.
""")

    print_section("WORKED EXAMPLE: Seam Residual Check")

    # Budget prediction
    delta_kappa_budget = 0.0100  # From budget model

    # What actually happened (from κ measurements)
    kappa_t0 = -0.3500  # κ at start
    kappa_t1 = -0.3410  # κ at end
    delta_kappa_ledger = kappa_t1 - kappa_t0

    print("  Step 1: Δκ_ledger = κ(t₁) − κ(t₀)")
    print(f"         = {kappa_t1:.4f} − ({kappa_t0:.4f})")
    print(f"         = {delta_kappa_ledger:.4f}")

    print(f"\n  Step 2: Δκ_budget (from §3.3) = {delta_kappa_budget:.4f}")

    residual = delta_kappa_budget - delta_kappa_ledger
    print("\n  Step 3: s = Δκ_budget − Δκ_ledger")
    print(f"         = {delta_kappa_budget:.4f} − {delta_kappa_ledger:.4f}")
    print(f"         = {residual:.4f}")

    print("\n  Step 4: |s| ≤ tol_seam?")
    print(f"         |{residual:.4f}| = {abs(residual):.4f}")
    print(
        f"         {abs(residual):.4f} ≤ {TOL_SEAM}?  {'✓ PASS (weld)' if abs(residual) <= TOL_SEAM else '✗ FAIL (gesture)'}"
    )

    print_section("THREE PASS CONDITIONS (All Must Hold)")
    print(f"""  For a seam to PASS — for a weld to form — ALL three must hold:

    ① |s| ≤ {TOL_SEAM}              Budget matched reality
    ② τ_R is finite (not ∞_rec)    Something actually returned
    ③ |IC_post/IC_pre − exp(Δκ)| < tol   Exponential identity held

  If ANY fails, the emission is a GESTURE — it exists, but didn't weld.
""")

    print_section("EXERCISE 3.4")
    print(f"""  A seam has:
    κ(t₀) = −0.500, κ(t₁) = −0.488
    Δκ_budget = 0.010

  a) Compute Δκ_ledger
  b) Compute s
  c) Does the seam PASS? (tol_seam = {TOL_SEAM})
""")
    print_section("ANSWER 3.4")
    dkl = -0.488 - (-0.500)
    s_ex = 0.010 - dkl
    print(f"    a) Δκ_ledger = −0.488 − (−0.500) = {dkl:.4f}")
    print(f"    b) s = 0.010 − {dkl:.4f} = {s_ex:.4f}")
    print(
        f"    c) |{s_ex:.4f}| = {abs(s_ex):.4f}; "
        f"{abs(s_ex):.4f} {'≤' if abs(s_ex) <= TOL_SEAM else '>'} 0.005 → "
        f"{'PASS' if abs(s_ex) <= TOL_SEAM else 'FAIL'}"
    )


# ════════════════════════════════════════════════════════════════════════
#  §3.5  REGIME GATES — Classifying the System
# ════════════════════════════════════════════════════════════════════════


def section_3_5_regime_gates() -> None:
    # Lemma refs: L22 (gate monotonicity), L40 (absorbing stable)
    print_header("§3.5  REGIME CLASSIFICATION — Stable / Watch / Collapse")

    print("""  CONCEPT:
  ────────
  The four invariants (ω, F, S, C) feed into FOUR GATES that produce
  a regime label. The gates are frozen thresholds:

  ┌────────────────────────────────────────────────────────────────┐
  │  STABLE requires ALL FOUR:                                    │
  │    ω < 0.038   AND   F > 0.90   AND   S < 0.15   AND C < 0.14│
  │                                                                │
  │  WATCH:                                                        │
  │    0.038 ≤ ω < 0.30  (or Stable gates not all satisfied)      │
  │                                                                │
  │  COLLAPSE:                                                     │
  │    ω ≥ 0.30                                                    │
  │                                                                │
  │  CRITICAL overlay:                                             │
  │    IC < 0.30  (accompanies any regime — severity flag)         │
  └────────────────────────────────────────────────────────────────┘

  Stable is CONJUNCTIVE — ALL four must be satisfied simultaneously.
  If even one gate fails, the system drops to Watch.
  If ω ≥ 0.30, it's Collapse regardless of other gates.
  Critical is an overlay, not a separate regime.
""")

    print_section("WORKED EXAMPLE: Classify Our Running Example")

    omega = 0.273750
    F = 0.726250
    S = 0.461003
    C = 0.461621
    IC = 0.652042

    print("  Input values:")
    print(f"    ω  = {omega:.6f}")
    print(f"    F  = {F:.6f}")
    print(f"    S  = {S:.6f}")
    print(f"    C  = {C:.6f}")
    print(f"    IC = {IC:.6f}")

    print("\n  Step 1: Check Critical overlay")
    print(f"    IC < 0.30?  {IC:.6f} < 0.30?  {'YES → CRITICAL' if IC < 0.30 else 'NO → continue'}")

    print("\n  Step 2: Check Collapse")
    print(f"    ω ≥ 0.30?   {omega:.6f} ≥ 0.30?  {'YES → COLLAPSE' if omega >= 0.30 else 'NO → continue'}")

    print("\n  Step 3: Check Stable (ALL four must hold)")
    g1 = omega < 0.038
    g2 = F > 0.90
    g3 = S < 0.15
    g4 = C < 0.14
    print(f"    Gate 1: ω < 0.038?   {omega:.4f} < 0.038?  {'✓' if g1 else '✗'}")
    print(f"    Gate 2: F > 0.90?    {F:.4f} > 0.90?   {'✓' if g2 else '✗'}")
    print(f"    Gate 3: S < 0.15?    {S:.4f} < 0.15?   {'✓' if g3 else '✗'}")
    print(f"    Gate 4: C < 0.14?    {C:.4f} < 0.14?   {'✓' if g4 else '✗'}")
    all_stable = g1 and g2 and g3 and g4
    print(f"    All four? {'YES → STABLE' if all_stable else 'NO → default to WATCH'}")

    regime = "CRITICAL" if IC < 0.30 else ("COLLAPSE" if omega >= 0.30 else ("STABLE" if all_stable else "WATCH"))
    print("\n  ╔═══════════════════════════════════╗")
    print(f"  ║  REGIME: {regime:>10s}                ║")
    print("  ╚═══════════════════════════════════╝")

    print_section("WORKED EXAMPLE: A Stable System")

    print("  c = [0.97, 0.96, 0.98, 0.95, 0.97, 0.96, 0.98, 0.97]")
    c_stable = np.array([0.97, 0.96, 0.98, 0.95, 0.97, 0.96, 0.98, 0.97])
    w = np.ones(8) / 8
    F_s = np.sum(w * c_stable)
    omega_s = 1 - F_s
    S_s = np.sum(w * np.array([-ci * np.log(ci) - (1 - ci) * np.log(1 - ci) for ci in c_stable]))
    C_s = float(np.std(c_stable) / 0.5)
    kappa_s = np.sum(w * np.log(c_stable))
    IC_s = np.exp(kappa_s)

    print(f"    F  = {F_s:.6f}    (> 0.90? {'✓' if F_s > 0.90 else '✗'})")
    print(f"    ω  = {omega_s:.6f}    (< 0.038? {'✓' if omega_s < 0.038 else '✗'})")
    print(f"    S  = {S_s:.6f}    (< 0.15? {'✓' if S_s < 0.15 else '✗'})")
    print(f"    C  = {C_s:.6f}    (< 0.14? {'✓' if C_s < 0.14 else '✗'})")
    print(f"    IC = {IC_s:.6f}    (≥ 0.30? {'✓' if IC_s >= 0.30 else '✗'})")

    g = [omega_s < 0.038, F_s > 0.90, S_s < 0.15, C_s < 0.14]
    print(f"    All gates: {'✓ ✓ ✓ ✓ → STABLE' if all(g) else 'Not all → WATCH'}")

    print_section("WORKED EXAMPLE: A Collapse System")

    c_collapse = np.array([0.30, 0.25, 0.40, 0.10, 0.20, 0.35, 0.15, 0.05])
    F_c = np.sum(w * c_collapse)
    omega_c = 1 - F_c
    IC_c = np.exp(np.sum(w * np.log(np.clip(c_collapse, 1e-8, 1 - 1e-8))))

    print(f"  c = {c_collapse}")
    print(f"    F  = {F_c:.6f}")
    print(f"    ω  = {omega_c:.6f}    (≥ 0.30? {'YES → COLLAPSE' if omega_c >= 0.30 else 'NO'})")
    print(f"    IC = {IC_c:.6f}    (< 0.30? {'YES → +CRITICAL overlay' if IC_c < 0.30 else 'NO'})")

    print_section("EXERCISE 3.5")
    print("""  Classify each system:

  System A: ω = 0.02, F = 0.98, S = 0.10, C = 0.05, IC = 0.97
  System B: ω = 0.15, F = 0.85, S = 0.30, C = 0.20, IC = 0.60
  System C: ω = 0.50, F = 0.50, S = 0.40, C = 0.45, IC = 0.10
  System D: ω = 0.05, F = 0.95, S = 0.20, C = 0.10, IC = 0.25

  For each: apply the gates in order (Critical → Collapse → Stable → Watch).
""")
    print_section("ANSWER 3.5")
    print("    A: All four gates pass → STABLE")
    print("    B: ω < 0.30 but ω ≥ 0.038 → WATCH (S and C also fail Stable)")
    print("    C: ω ≥ 0.30 → COLLAPSE, IC < 0.30 → +CRITICAL overlay")
    print("    D: IC < 0.30 → CRITICAL (overrides everything)")


# ════════════════════════════════════════════════════════════════════════
#  §3.6  SUMMARY — Level 3 Cheat Sheet
# ════════════════════════════════════════════════════════════════════════


def section_3_6_summary() -> None:
    print_header("§3.6  SUMMARY — Level 3 Cheat Sheet")

    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │ COST FUNCTIONS                                                     │
  ├─────────────┬──────────────────────────────┬───────────────────────│
  │ D_ω = Γ(ω)  │ ω^{P_EXPONENT} / (1 − ω + ε)            │ Drift penalty          │
  │ D_C         │ α · C                        │ Curvature penalty      │
  │ α           │ {ALPHA}                          │ Frozen coefficient     │
  │ p           │ {P_EXPONENT}                            │ Frozen exponent        │
  │ ε           │ {EPSILON}                      │ Frozen guard band      │
  └─────────────┴──────────────────────────────┴───────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │ BUDGET AND SEAM                                                    │
  ├──────────────┬─────────────────────────────────────────────────────│
  │ Δκ_budget    │ R · τ_R − (D_ω + D_C)        credit − debits      │
  │ Δκ_ledger    │ κ(t₁) − κ(t₀)                what actually changed │
  │ Residual s   │ Δκ_budget − Δκ_ledger         prediction error     │
  │ PASS         │ |s| ≤ {TOL_SEAM}, τ_R finite, exp identity holds   │
  └──────────────┴─────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │ REGIME GATES                                                       │
  ├─────────────┬──────────────────────────────────────────────────────│
  │ STABLE      │ ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14  │
  │ WATCH       │ 0.038 ≤ ω < 0.30 (or Stable gates fail)           │
  │ COLLAPSE    │ ω ≥ 0.30                                            │
  │ CRITICAL    │ IC < 0.30 (overlay on any regime)                   │
  └─────────────┴──────────────────────────────────────────────────────┘

  KEY IDEAS:
  • Γ(ω) is cubic → small drift is nearly free, large drift is ruinous
  • The budget is an accounting equation: credits vs. debits
  • The seam checks if budget prediction matched reality
  • Regime gates are FROZEN thresholds — not tunable
  • Stable is conjunctive (ALL gates must pass)

  NEXT: Level 4 runs the FULL SPINE end-to-end on real data.
""")


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 72 + "╗")
    print("║  GCD KERNEL MATH WORKSHEETS — Level 3: Budget & Regime" + " " * 16 + "║")
    print("║  Cost functions, budget identity, regime classification" + " " * 16 + "║")
    print("╚" + "═" * 72 + "╝")

    section_3_1_drift_cost()
    section_3_2_curvature_cost()
    section_3_3_budget()
    section_3_4_seam_residual()
    section_3_5_regime_gates()
    section_3_6_summary()
