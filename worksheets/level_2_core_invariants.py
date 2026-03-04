#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WORKSHEET — Level 2: Core Invariants (F, ω, S, C, κ, IC)              ║
║  The six numbers that describe any collapsed system                     ║
║                                                                         ║
║  Prerequisites: Level 1 (coordinates, weights, ε-clamping)              ║
║  Goal: Compute and understand each Tier-1 invariant step by step        ║
╚══════════════════════════════════════════════════════════════════════════╝

Tier-1 invariants are IMMUTABLE — they are structural identities of
collapse, discovered across 146 experiments. Never redefine them.

The six invariants:
  F  — Fidelity       (what survived)
  ω  — Drift          (what was lost)
  S  — Entropy        (uncertainty in the collapse field)
  C  — Curvature      (channel dispersion)
  κ  — Log-integrity  (logarithmic coherence)
  IC — Integrity      (multiplicative coherence)
"""

from __future__ import annotations

import numpy as np


def print_header(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}\n")


def print_section(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(1, 60 - len(title))}\n")


# ════════════════════════════════════════════════════════════════════════
#  RUNNING EXAMPLE — Used Throughout This Worksheet
# ════════════════════════════════════════════════════════════════════════


def setup_example() -> tuple[np.ndarray, np.ndarray, float]:
    """Set up the running example used across all sections."""
    print_header("RUNNING EXAMPLE — Used Throughout All Sections")

    c = np.array([0.85, 0.72, 0.91, 0.68, 0.77, 0.83, 0.90, 0.15])
    eps = 1e-8
    c_eps = np.clip(c, eps, 1 - eps)  # All already in range
    n = len(c)
    w = np.ones(n) / n

    print("  An 8-channel system (e.g., a particle measured on 8 properties):\n")
    print(f"    {'Ch':>3s}  {'c_i':>8s}  {'w_i':>8s}   Description")
    print(f"    {'─' * 3}  {'─' * 8}  {'─' * 8}   {'─' * 30}")
    labels = ["mass_log", "spin", "charge", "color", "weak_iso", "lepton#", "baryon#", "generat'n"]
    for i in range(n):
        marker = " ← weak!" if c[i] < 0.3 else ""
        print(f"    {i + 1:3d}  {c[i]:8.4f}  {w[i]:8.6f}   {labels[i]}{marker}")

    print(f"\n    ε = {eps}")
    print(f"    n = {n} channels")
    print(f"    Σ w_i = {w.sum():.1f} ✓")
    print("\n    Note: Channel 8 (c = 0.15) is the weak link. Watch what it does.\n")

    return c_eps, w, eps


# ════════════════════════════════════════════════════════════════════════
#  §2.1  FIDELITY (F) — The Weighted Arithmetic Mean
#  Lemma refs: L4 (F bounds), L9 (F convexity)
# ════════════════════════════════════════════════════════════════════════


def section_2_1_fidelity(c: np.ndarray, w: np.ndarray) -> float:
    print_header("§2.1  FIDELITY (F) — What Survived Collapse")

    print("""  CONCEPT:
  ────────
  Fidelity F is the weighted average of all coordinates:

    F = Σ w_i · c_i  =  w_1·c_1 + w_2·c_2 + ... + w_n·c_n

  It answers: "On average, how much of this system survived collapse?"

  F ∈ [0, 1]
    F = 1  → perfect preservation (all channels at 1.0)
    F = 0  → total loss (all channels at 0.0 / ε)
    F = 0.5 → half survived on average
""")

    print_section("WORKED EXAMPLE: Compute F Step by Step")

    n = len(c)
    print("  F = Σ w_i · c_i\n")
    print(f"    = (1/{n}) × (c_1 + c_2 + ... + c_{n})\n")

    running_sum = 0.0
    for i in range(n):
        product = w[i] * c[i]
        running_sum += product
        print(
            f"    + w_{i + 1} × c_{i + 1} = {w[i]:.6f} × {c[i]:.4f} = {product:.6f}   (running sum: {running_sum:.6f})"
        )

    F: float = float(np.sum(w * c))
    print(f"\n    F = {F:.6f}")
    print(f"\n  Interpretation: {F * 100:.1f}% of the system survived on average.")

    # Show that the weak channel barely affects F
    print_section("KEY INSIGHT: F is Tolerant of Weak Channels")
    c_no_weak = c.copy()
    c_no_weak[-1] = 0.85  # Replace weak channel with average
    F_no_weak = np.sum(w * c_no_weak)
    print(f"    With weak channel (c_8 = 0.15): F = {F:.6f}")
    print(f"    If c_8 were 0.85:               F = {F_no_weak:.6f}")
    print(f"    Difference:                      {abs(F - F_no_weak):.6f}")
    print("\n    F barely changes! The arithmetic mean absorbs one bad channel.")
    print("    This is why F alone is not enough — we need IC (§2.5).")

    print_section("EXERCISE 2.1")
    print("""  Compute F for: c = [0.9, 0.8, 0.7, 0.6], w = [0.25, 0.25, 0.25, 0.25]

  Show each step:  F = w_1·c_1 + w_2·c_2 + w_3·c_3 + w_4·c_4
""")
    print_section("ANSWER 2.1")
    print("    F = 0.25×0.9 + 0.25×0.8 + 0.25×0.7 + 0.25×0.6")
    print("      = 0.225   + 0.200   + 0.175   + 0.150")
    print("      = 0.7500")

    return F


# ════════════════════════════════════════════════════════════════════════
#  §2.2  DRIFT (ω) — What Was Lost
# ════════════════════════════════════════════════════════════════════════


def section_2_2_drift(F: float) -> float:
    # Lemma refs: L4 (ω = 1 − F), L22 (collapse gate monotonicity)
    print_header("§2.2  DRIFT (ω) — What Was Lost to Collapse")

    print("""  CONCEPT:
  ────────
  Drift is simply:

    ω = 1 − F

  That's it. One subtraction. But this simplicity is profound.

  THE DUALITY IDENTITY (Complementum Perfectum):

    F + ω = 1    exactly    (residual = 0.0, not ≈ 0, EXACTLY 0)

  This means: every channel contributes EITHER to fidelity (survived)
  or to drift (lost). There is no third bucket. No unaccounted residual.
  The books always balance. Tertia via nulla — no third way.
""")

    print_section("WORKED EXAMPLE: Compute ω")

    omega = 1 - F
    print(f"  From §2.1, F = {F:.6f}")
    print("  ω = 1 − F")
    print(f"    = 1 − {F:.6f}")
    print(f"    = {omega:.6f}")
    print(f"\n  Verification: F + ω = {F:.6f} + {omega:.6f} = {F + omega:.10f}")
    print(f"  Residual: |F + ω − 1| = {abs(F + omega - 1):.1e}")
    print(f"\n  Interpretation: {omega * 100:.1f}% drifted away from fidelity.")

    print_section("WHY ω MATTERS: It's the Cost Input")
    print("""  ω feeds into the drift cost function Γ(ω) = ω³ / (1 − ω + ε).
  Small ω → small Γ → cheap to return.
  Large ω → large Γ → expensive or impossible to return.

  The regime gates use ω directly:
    Stable:   ω < 0.038
    Watch:    0.038 ≤ ω < 0.30
    Collapse: ω ≥ 0.30
""")
    print(f"  Our ω = {omega:.6f} → regime = {'Stable' if omega < 0.038 else 'Watch' if omega < 0.30 else 'Collapse'}")

    print_section("EXERCISE 2.2")
    print("""  For F = 0.7500 (from Exercise 2.1):
    a) Compute ω
    b) Verify F + ω = 1
    c) Which regime does this fall into?
""")
    print_section("ANSWER 2.2")
    print("    a) ω = 1 − 0.7500 = 0.2500")
    print("    b) F + ω = 0.7500 + 0.2500 = 1.0000 ✓")
    print("    c) ω = 0.25 → 0.038 ≤ 0.25 < 0.30 → Watch regime")

    return omega


# ════════════════════════════════════════════════════════════════════════
#  §2.3  ENTROPY (S) — Uncertainty of the Collapse Field
# ════════════════════════════════════════════════════════════════════════


def section_2_3_entropy(c: np.ndarray, w: np.ndarray) -> float:
    # Lemma refs: L5 (S = ln 2 iff c = 1/2), L6 (S bounds), L41 (S + κ ≤ ln 2)
    print_header("§2.3  ENTROPY (S) — Bernoulli Field Entropy")

    print("""  CONCEPT:
  ────────
  Each channel c_i can be thought of as a coin with probability c_i of
  "heads" (surviving). The uncertainty of that coin is:

    h(c_i) = −c_i · ln(c_i) − (1 − c_i) · ln(1 − c_i)

  This is the BERNOULLI entropy of each channel.

  The system entropy S is the weighted sum:

    S = Σ w_i · h(c_i)

  S measures: "How uncertain is the collapse field?"
    S = 0     → every channel is at 0 or 1 (perfectly decided)
    S = ln(2) → every channel is at 0.5 (maximum uncertainty)
    S ≈ 0.693 → ln(2) — ceiling of uncertainty per channel

  NOTE: This is Bernoulli field entropy — the full collapse-field form.
  The classical binary form is the degenerate limit when the collapse
  field is removed (c_i ∈ {{0,1}} only).
""")

    print_section("WORKED EXAMPLE: Compute h(c_i) for Each Channel")

    n = len(c)
    print("  For each channel, compute h(c_i) = −c_i·ln(c_i) − (1−c_i)·ln(1−c_i):\n")
    print(f"    {'Ch':>3s}  {'c_i':>8s}  {'−c·ln(c)':>10s}  {'−(1−c)ln(1−c)':>14s}  {'h(c_i)':>8s}")
    print(f"    {'─' * 3}  {'─' * 8}  {'─' * 10}  {'─' * 14}  {'─' * 8}")

    h_values = []
    for i in range(n):
        ci = c[i]
        if ci <= 0 or ci >= 1:
            hi = 0.0
        else:
            term1 = -ci * np.log(ci)
            term2 = -(1 - ci) * np.log(1 - ci)
            hi = term1 + term2

        h_values.append(hi)
        t1 = -ci * np.log(ci) if ci > 0 else 0
        t2 = -(1 - ci) * np.log(1 - ci) if ci < 1 else 0
        print(f"    {i + 1:3d}  {ci:8.4f}  {t1:10.6f}  {t2:14.6f}  {hi:8.6f}")

    print("\n  Now compute S = Σ w_i · h(c_i):\n")

    S = 0.0
    for i in range(n):
        contrib = w[i] * h_values[i]
        S += contrib
        print(f"    + w_{i + 1}·h_{i + 1} = {w[i]:.6f} × {h_values[i]:.6f} = {contrib:.6f}   (running: {S:.6f})")

    print(f"\n    S = {S:.6f}")
    print(f"    S / ln(2) = {S / np.log(2):.4f}  (0 = no uncertainty, 1 = maximum)")

    print_section("KEY INSIGHT: Extreme Channels Have Low Entropy")
    print(f"    h(0.50) = {-0.5 * np.log(0.5) - 0.5 * np.log(0.5):.6f} = ln(2) (max!)")
    print(f"    h(0.90) = {-0.9 * np.log(0.9) - 0.1 * np.log(0.1):.6f} (moderately certain)")
    print(f"    h(0.15) = {-0.15 * np.log(0.15) - 0.85 * np.log(0.85):.6f} (fairly certain it's low)")
    print(f"    h(0.01) = {-0.01 * np.log(0.01) - 0.99 * np.log(0.99):.6f} (very certain it's dead)")

    print_section("EXERCISE 2.3")
    print("""  Compute h(c_i) for c_i = 0.3:
    h(0.3) = −0.3·ln(0.3) − 0.7·ln(0.7) = ?

  Then for c = [0.9, 0.8, 0.7, 0.6] with equal weights, compute S.
""")
    print_section("ANSWER 2.3")
    h03 = -0.3 * np.log(0.3) - 0.7 * np.log(0.7)
    print("    h(0.3) = −0.3×(−1.2040) − 0.7×(−0.3567)")
    print(f"           = 0.3612 + 0.2497 = {h03:.6f}")
    vals = [0.9, 0.8, 0.7, 0.6]
    h_vals = [-v * np.log(v) - (1 - v) * np.log(1 - v) for v in vals]
    S_ex = sum(h / 4 for h in h_vals)
    print("\n    For c = [0.9, 0.8, 0.7, 0.6]:")
    for i, v in enumerate(vals):
        print(f"      h({v}) = {h_vals[i]:.6f}")
    print(f"    S = (1/4) × ({' + '.join(f'{h:.4f}' for h in h_vals)})")
    print(f"      = {S_ex:.6f}")

    return S


# ════════════════════════════════════════════════════════════════════════
#  §2.4  CURVATURE (C) — Channel Dispersion
# ════════════════════════════════════════════════════════════════════════


def section_2_4_curvature(c: np.ndarray) -> float:
    print_header("§2.4  CURVATURE (C) — Channel Dispersion")

    print("""  CONCEPT:
  ────────
  Curvature measures how SPREAD OUT the channels are:

    C = std_pop(c) / 0.5

  where std_pop is the POPULATION standard deviation (not sample):

    std_pop(c) = sqrt( (1/n) · Σ (c_i − c̄)² )

  Dividing by 0.5 normalizes C to [0, 1]:
    C = 0 → all channels identical (homogeneous)
    C = 1 → maximum spread (e.g., half at 0, half at 1)

  WHY 0.5? The maximum population std of values in [0,1] is 0.5
  (when half the values are 0 and half are 1).
""")

    print_section("WORKED EXAMPLE: Compute C Step by Step")

    n = len(c)
    c_bar = np.mean(c)

    print("  Step 1: Mean c̄ = (1/n) · Σ c_i")
    print(f"         c̄ = (1/{n}) × ({' + '.join(f'{ci:.4f}' for ci in c)})")
    print(f"         c̄ = {c.sum():.4f} / {n}")
    print(f"         c̄ = {c_bar:.6f}\n")

    print("  Step 2: Squared deviations (c_i − c̄)²\n")
    diffs_sq = []
    for i in range(n):
        diff = c[i] - c_bar
        diff_sq = diff**2
        diffs_sq.append(diff_sq)
        print(f"    (c_{i + 1} − c̄)² = ({c[i]:.4f} − {c_bar:.4f})² = ({diff:+.4f})² = {diff_sq:.6f}")

    variance = np.mean(diffs_sq)
    std_pop = np.sqrt(variance)

    print(f"\n  Step 3: Population variance = (1/{n}) × Σ (c_i − c̄)²")
    print(f"         = (1/{n}) × {sum(diffs_sq):.6f}")
    print(f"         = {variance:.6f}")

    print("\n  Step 4: Population std = √(variance)")
    print(f"         = √({variance:.6f})")
    print(f"         = {std_pop:.6f}")

    C: float = float(std_pop / 0.5)
    print("\n  Step 5: Curvature C = std_pop / 0.5")
    print(f"         = {std_pop:.6f} / 0.5")
    print(f"         = {C:.6f}")

    print_section("KEY INSIGHT: The Weak Channel Creates Curvature")
    c_no_weak = c.copy()
    c_no_weak[-1] = c_bar  # Replace with mean
    C_no_weak = float(np.std(c_no_weak) / 0.5)
    print(f"    With weak channel (c_8 = 0.15):  C = {C:.6f}")
    print(f"    If c_8 = c̄ = {c_bar:.4f}:           C = {C_no_weak:.6f}")
    print("    The weak channel accounts for most of the dispersion.")

    print_section("EXERCISE 2.4")
    print("""  Compute C for c = [0.9, 0.8, 0.7, 0.6]:
    a) Mean c̄ = ?
    b) Each (c_i − c̄)² = ?
    c) Variance = ?
    d) std_pop = ?
    e) C = std_pop / 0.5 = ?
""")
    print_section("ANSWER 2.4")
    c_ex = np.array([0.9, 0.8, 0.7, 0.6])
    cb = np.mean(c_ex)
    var_ex = np.mean((c_ex - cb) ** 2)
    std_ex = np.sqrt(var_ex)
    C_ex = std_ex / 0.5
    print(f"    a) c̄ = (0.9+0.8+0.7+0.6)/4 = {cb:.4f}")
    for i, ci in enumerate(c_ex):
        print(f"    b) (c_{i + 1}−c̄)² = ({ci}−{cb})² = {(ci - cb) ** 2:.6f}")
    print(f"    c) Var = {var_ex:.6f}")
    print(f"    d) std = {std_ex:.6f}")
    print(f"    e) C = {std_ex:.6f}/0.5 = {C_ex:.6f}")

    return C


# ════════════════════════════════════════════════════════════════════════
#  §2.5  LOG-INTEGRITY (κ) AND INTEGRITY COMPOSITE (IC)
# ════════════════════════════════════════════════════════════════════════


def section_2_5_integrity(c: np.ndarray, w: np.ndarray, F: float) -> tuple[float, float]:
    # Lemma refs: L7 (κ bounds), L34 (heterogeneity gap Δ = F − IC)
    print_header("§2.5  LOG-INTEGRITY (κ) AND INTEGRITY COMPOSITE (IC)")

    print("""  CONCEPT:
  ────────
  F (fidelity) is the ARITHMETIC mean. It can hide a dead channel.
  IC (integrity) is the GEOMETRIC mean. It CANNOT hide a dead channel.

  The kernel computes IC in log space for numerical stability:

    κ = Σ w_i · ln(c_i,ε)     (log-integrity, always ≤ 0)
    IC = exp(κ)                (integrity composite, in (0, 1])

  WHY GEOMETRIC MEAN?
  The geometric mean of [1.0, 0.001] = sqrt(1.0 × 0.001) = 0.032
  The arithmetic mean of [1.0, 0.001] = (1.0 + 0.001)/2 = 0.501

  The geometric mean punishes dead channels ruthlessly.
  This is the principle of GEOMETRIC SLAUGHTER (trucidatio geometrica).

  THE INTEGRITY BOUND (Limbus Integritatis):

    IC ≤ F    always

  The geometric mean NEVER exceeds the arithmetic mean.
  The gap Δ = F − IC is the HETEROGENEITY GAP.
""")

    print_section("WORKED EXAMPLE: Compute κ and IC Step by Step")

    n = len(c)
    print("  Step 1: Compute ln(c_i,ε) for each channel\n")
    print(f"    {'Ch':>3s}  {'c_i':>8s}  {'ln(c_i)':>10s}  {'w_i':>8s}  {'w_i·ln(c_i)':>12s}")
    print(f"    {'─' * 3}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 12}")

    kappa = 0.0
    for i in range(n):
        log_ci = np.log(c[i])
        contrib = w[i] * log_ci
        kappa += contrib
        marker = " ← LARGE negative!" if log_ci < -1.0 else ""
        print(f"    {i + 1:3d}  {c[i]:8.4f}  {log_ci:10.6f}  {w[i]:8.6f}  {contrib:12.6f}{marker}")

    print(f"\n  Step 2: Sum → κ = {kappa:.6f}")

    IC = np.exp(kappa)
    print(f"\n  Step 3: IC = exp(κ) = exp({kappa:.6f}) = {IC:.6f}")
    print("\n  Step 4: Verify integrity bound")
    gap = F - IC
    print(f"    F  = {F:.6f}  (arithmetic mean)")
    print(f"    IC = {IC:.6f}  (geometric mean)")
    print(f"    F − IC = {gap:.6f}  (heterogeneity gap Δ)")
    print(f"    IC ≤ F?  {IC:.6f} ≤ {F:.6f}  {'✓' if IC <= F + 1e-9 else '✗'}")

    print_section("THE KILLER: Channel 8 Destroys IC")
    print("    Channel 8: c_8 = 0.15")
    print(f"    ln(0.15) = {np.log(0.15):.4f}")
    print(f"    w_8 · ln(0.15) = {w[-1]:.4f} × {np.log(0.15):.4f} = {w[-1] * np.log(0.15):.4f}")
    print(f"\n    This single channel contributes {abs(w[-1] * np.log(0.15)):.4f} to κ")
    print(f"    while each strong channel contributes only ~ {abs(w[0] * np.log(c[0])):.4f}.")
    print(f"    The weak channel has ~{abs(w[-1] * np.log(c[-1])) / abs(w[0] * np.log(c[0])):.0f}× the influence!")

    print_section("GEOMETRIC SLAUGHTER IN ACTION")
    print("\n    Watch IC collapse as one channel dies (other 7 perfect at 0.90):\n")
    print(f"      {'c_dead':>10s}  {'F':>8s}  {'IC':>10s}  {'Δ=F−IC':>8s}  {'IC/F':>6s}")
    print(f"      {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 6}")
    for c_dead in [0.90, 0.50, 0.10, 0.01, 0.001, 1e-4, 1e-6, 1e-8]:
        c_test = np.array([0.90] * 7 + [c_dead])
        w_test = np.ones(8) / 8
        F_t = np.sum(w_test * c_test)
        kappa_t = np.sum(w_test * np.log(np.clip(c_test, 1e-8, 1 - 1e-8)))
        IC_t = np.exp(kappa_t)
        print(f"      {c_dead:10.1e}  {F_t:8.4f}  {IC_t:10.6f}  {F_t - IC_t:8.4f}  {IC_t / F_t:6.4f}")

    print_section("EXERCISE 2.5")
    print("""  For c = [0.9, 0.8, 0.7, 0.6] with equal weights (w_i = 0.25):
    a) Compute ln(c_i) for each channel
    b) Compute κ = Σ w_i · ln(c_i)
    c) Compute IC = exp(κ)
    d) Compute Δ = F − IC  (use F = 0.75 from Exercise 2.1)
    e) Verify IC ≤ F
""")
    print_section("ANSWER 2.5")
    c_ex = np.array([0.9, 0.8, 0.7, 0.6])
    w_ex = np.ones(4) / 4
    for _i, ci in enumerate(c_ex):
        print(f"    a) ln({ci}) = {np.log(ci):.6f}")
    kappa_ex = np.sum(w_ex * np.log(c_ex))
    IC_ex = np.exp(kappa_ex)
    print(
        f"    b) κ = 0.25×{np.log(0.9):.4f} + 0.25×{np.log(0.8):.4f} + 0.25×{np.log(0.7):.4f} + 0.25×{np.log(0.6):.4f}"
    )
    print(f"       κ = {kappa_ex:.6f}")
    print(f"    c) IC = exp({kappa_ex:.6f}) = {IC_ex:.6f}")
    print(f"    d) Δ = 0.75 − {IC_ex:.6f} = {0.75 - IC_ex:.6f}")
    print(f"    e) {IC_ex:.6f} ≤ 0.75 → ✓")

    return kappa, IC


# ════════════════════════════════════════════════════════════════════════
#  §2.6  THE THREE STRUCTURAL IDENTITIES — Always True
# ════════════════════════════════════════════════════════════════════════


def section_2_6_identities(F: float, omega: float, IC: float, kappa: float) -> None:
    # Lemma refs: L4 (F + ω = 1), L7 (IC ≤ F), L34 (Δ ≥ 0)
    print_header("§2.6  THE THREE STRUCTURAL IDENTITIES — Always True")

    print("""  CONCEPT:
  ────────
  No matter what data you put in, three things ALWAYS hold:

  IDENTITY 1 — Duality:          F + ω = 1         (exactly)
  IDENTITY 2 — Integrity bound:  IC ≤ F            (always)
  IDENTITY 3 — Exponential map:  IC = exp(κ)       (by construction)

  These are not approximations. They are structural consequences of
  the definitions. They hold to machine precision.

  If your code ever produces results violating these, there is a BUG.
""")

    print_section("VERIFICATION on Our Running Example")

    # Identity 1
    print("  Identity 1: F + ω = 1")
    print(f"    F = {F:.10f}")
    print(f"    ω = {omega:.10f}")
    print(f"    F + ω = {F + omega:.10f}")
    print(f"    |F + ω − 1| = {abs(F + omega - 1):.1e}  {'✓ EXACT' if abs(F + omega - 1) < 1e-15 else '✗ BUG!'}")

    # Identity 2
    print("\n  Identity 2: IC ≤ F")
    print(f"    IC = {IC:.10f}")
    print(f"    F  = {F:.10f}")
    print(f"    IC ≤ F?  {'✓' if IC <= F + 1e-9 else '✗ BUG!'}")
    print(f"    Δ  = F − IC = {F - IC:.10f}  (heterogeneity gap)")

    # Identity 3
    print("\n  Identity 3: IC = exp(κ)")
    print(f"    κ      = {kappa:.10f}")
    print(f"    exp(κ) = {np.exp(kappa):.10f}")
    print(f"    IC     = {IC:.10f}")
    print(
        f"    |IC − exp(κ)| = {abs(IC - np.exp(kappa)):.1e}  "
        f"{'✓ EXACT' if abs(IC - np.exp(kappa)) < 1e-15 else '✗ BUG!'}"
    )

    print_section("WHEN IDENTITIES BECOME EQUALITIES")
    print("""
    Identity 2 becomes an EQUALITY (IC = F) when:
      All channels are identical: c_1 = c_2 = ... = c_n
      (Geometric mean = Arithmetic mean when all values equal)

    This is the "homogeneous" case. Any heterogeneity makes IC < F.
    The bigger the heterogeneity, the bigger the gap Δ.
""")


# ════════════════════════════════════════════════════════════════════════
#  §2.7  SUMMARY — Level 2 Cheat Sheet
# ════════════════════════════════════════════════════════════════════════


def section_2_7_summary(F: float, omega: float, S: float, C: float, kappa: float, IC: float) -> None:
    print_header("§2.7  SUMMARY — Level 2 Cheat Sheet")

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │ Symbol │ Name       │ Formula                    │ Our Value     │
  ├────────┼────────────┼────────────────────────────┼───────────────│
  │   F    │ Fidelity   │ Σ w_i · c_i               │ {F:12.6f}    │
  │   ω    │ Drift      │ 1 − F                      │ {omega:12.6f}    │
  │   S    │ Entropy    │ Σ w_i·h(c_i)               │ {S:12.6f}    │
  │   C    │ Curvature  │ std_pop(c) / 0.5           │ {C:12.6f}    │
  │   κ    │ Log-integ  │ Σ w_i · ln(c_i)            │ {kappa:12.6f}    │
  │  IC    │ Integrity  │ exp(κ)                      │ {IC:12.6f}    │
  │   Δ    │ Het. gap   │ F − IC                      │ {F - IC:12.6f}    │
  └───────────────────────────────────────────────────────────────────┘

  THREE IDENTITIES (always hold):
    ① F + ω = 1          (duality)
    ② IC ≤ F             (integrity bound)
    ③ IC = exp(κ)        (exponential map)

  KEY INSIGHT: F and IC tell different stories.
    F = "How healthy is the average?"  (tolerant of weak channels)
    IC = "How coherent is the whole?"  (destroyed by weak channels)
    Δ = "How much are the channels disagreeing?"

  NEXT: Level 3 computes costs, budget, and regime classification.
""")


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 72 + "╗")
    print("║  GCD KERNEL MATH WORKSHEETS — Level 2: Core Invariants" + " " * 16 + "║")
    print("║  Computing F, ω, S, C, κ, IC step by step" + " " * 28 + "║")
    print("╚" + "═" * 72 + "╝")

    c, w, eps = setup_example()
    F = section_2_1_fidelity(c, w)
    omega = section_2_2_drift(F)
    S = section_2_3_entropy(c, w)
    C = section_2_4_curvature(c)
    kappa, IC = section_2_5_integrity(c, w, F)
    section_2_6_identities(F, omega, IC, kappa)
    section_2_7_summary(F, omega, S, C, kappa, IC)
