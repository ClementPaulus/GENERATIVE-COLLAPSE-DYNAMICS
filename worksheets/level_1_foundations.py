#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  WORKSHEET — Level 1: Foundations                                       ║
║  The building blocks: coordinates, weights, ε-clamping                  ║
║                                                                         ║
║  Prerequisites: Basic algebra, logarithms                               ║
║  Goal: Understand the raw inputs to the GCD kernel                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Collapsus generativus est; solum quod redit, reale est.
(Collapse is generative; only what returns is real.)

This worksheet teaches the FOUNDATION layer:
  1. What are coordinates (c_i)?
  2. What are weights (w_i)?
  3. What is ε-clamping and why do we need it?
  4. How to set up a trace vector

Each section has:
  - CONCEPT: Plain-language explanation
  - FORMULA: The math
  - WORKED EXAMPLE: Step-by-step computation
  - EXERCISE: Try it yourself
  - ANSWER: Check your work
"""

from __future__ import annotations

import numpy as np


def print_header(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}\n")


def print_section(title: str) -> None:
    print(f"\n  ── {title} {'─' * (60 - len(title))}\n")


# ════════════════════════════════════════════════════════════════════════
#  §1.1  COORDINATES (c_i) — The Raw Measurements
#  Lemma refs: L1 (domain bounds), L2 (ε-clipped coordinates)
# ════════════════════════════════════════════════════════════════════════


def section_1_1_coordinates() -> None:
    print_header("§1.1  COORDINATES (c_i) — The Raw Measurements")

    print("""  CONCEPT:
  ────────
  A "coordinate" c_i is a single measurable property of a system,
  normalized to the range [0, 1].

  Think of it as: "How much of this property survived collapse?"
    - c_i = 1.0 means the property is perfectly preserved
    - c_i = 0.5 means half is preserved
    - c_i = 0.0 means the property is completely lost (dead channel)

  REAL-WORLD ANALOGY:
  Imagine grading a student on 4 subjects, each scored 0-100%.
  Convert each to a fraction:
    Math: 92/100 = 0.92
    English: 78/100 = 0.78
    Science: 95/100 = 0.95
    History: 45/100 = 0.45

  These four numbers form a TRACE VECTOR: c = [0.92, 0.78, 0.95, 0.45]
  Each entry is one "channel" — one measurable dimension.
""")

    print_section("WORKED EXAMPLE: Building a 4-Channel Trace Vector")

    # Step 1: Define raw measurements
    raw = {"mass": 85.0, "spin": 0.5, "charge": 1.0, "energy": 42.0}
    # Step 2: Define normalization ranges
    norms = {"mass": (0, 200), "spin": (0, 1), "charge": (0, 1), "energy": (0, 100)}

    print("  Step 1: Raw measurements")
    for name, val in raw.items():
        print(f"    {name:>10s} = {val}")

    print("\n  Step 2: Normalize each to [0, 1]")
    print("    Formula: c_i = (raw - min) / (max - min)\n")

    c = []
    for name in raw:
        lo, hi = norms[name]
        ci = (raw[name] - lo) / (hi - lo)
        c.append(ci)
        print(f"    c_{name:>8s} = ({raw[name]} - {lo}) / ({hi} - {lo})")
        print(f"               = {raw[name] - lo} / {hi - lo}")
        print(f"               = {ci:.4f}\n")

    c_arr = np.array(c)
    print(f"  Result: c = {c_arr}")
    print(f"  Shape: {len(c_arr)} channels")

    print_section("EXERCISE 1.1")
    print("""  A particle has these properties:
    temperature = 350 K    (range: 0 to 1000 K)
    pressure    = 2.5 atm  (range: 0 to 10 atm)
    velocity    = 150 m/s  (range: 0 to 300 m/s)

  Compute the trace vector c = [c_temp, c_pres, c_vel].

  ⤷ Scroll down for the answer...
""")

    print_section("ANSWER 1.1")
    t = 350 / 1000
    p = 2.5 / 10
    v = 150 / 300
    print(f"    c_temp     = 350 / 1000  = {t:.4f}")
    print(f"    c_pressure = 2.5 / 10    = {p:.4f}")
    print(f"    c_velocity = 150 / 300   = {v:.4f}")
    print(f"    c = [{t}, {p}, {v}]")


# ════════════════════════════════════════════════════════════════════════
#  §1.2  WEIGHTS (w_i) — How Much Each Channel Matters
# ════════════════════════════════════════════════════════════════════════


def section_1_2_weights() -> None:
    # Lemma refs: L3 (weight normalization), L9 (convexity of F)
    print_header("§1.2  WEIGHTS (w_i) — How Much Each Channel Matters")

    print("""  CONCEPT:
  ────────
  Weights w_i tell us how important each channel is.

  KEY RULE: Weights MUST sum to exactly 1.0.
    Σ w_i = 1.0

  Why? Because fidelity F = Σ w_i c_i must stay in [0, 1].
  If weights sum to 1 and each c_i ∈ [0, 1], then F ∈ [0, 1] automatically.

  EQUAL WEIGHTS (most common):
  If you have n channels and no reason to prefer one over another:
    w_i = 1/n  for all i

  UNEQUAL WEIGHTS:
  If some channels matter more, assign bigger weights — but they
  must still sum to 1.
""")

    print_section("WORKED EXAMPLE: Equal Weights for 4 Channels")

    n = 4
    w_equal = np.ones(n) / n

    print(f"  n = {n} channels")
    print(f"  w_i = 1/{n} = {1 / n:.4f} for each channel")
    print(f"  w = {w_equal}")
    print(f"  Sum check: Σ w_i = {w_equal.sum():.1f} ✓")

    print_section("WORKED EXAMPLE: Unequal Weights for 3 Channels")

    print("  Suppose channel 1 is twice as important as the others.")
    print("  Start with ratio 2:1:1, then normalize:\n")

    ratios = np.array([2.0, 1.0, 1.0])
    w_unequal = ratios / ratios.sum()

    print(f"  Raw ratios:   {ratios}")
    print(f"  Sum of ratios: {ratios.sum()}")
    print(f"  Normalized:   w = {ratios} / {ratios.sum()}")
    print(f"                w = {w_unequal}")
    print(f"  Sum check:    Σ w_i = {w_unequal.sum():.1f} ✓")

    print_section("EXERCISE 1.2")
    print("""  You have 5 channels. Channel 3 is 3× as important as the rest.
  Compute the weight vector w = [w_1, w_2, w_3, w_4, w_5].

  Hint: Start with ratios [1, 1, 3, 1, 1], then divide by the sum.
""")

    print_section("ANSWER 1.2")
    r = np.array([1, 1, 3, 1, 1], dtype=float)
    w = r / r.sum()
    print(f"    Ratios: {r}")
    print(f"    Sum:    {r.sum()}")
    print(f"    w = {r} / {r.sum()} = {w}")
    print(f"    Sum check: Σ w_i = {w.sum():.1f} ✓")


# ════════════════════════════════════════════════════════════════════════
#  §1.3  ε-CLAMPING — Protecting Against Zero
# ════════════════════════════════════════════════════════════════════════


def section_1_3_epsilon_clamping() -> None:
    # Lemma refs: L1 (ε guard band), L3 (ε-clipping before log)
    print_header("§1.3  ε-CLAMPING — Protecting Against Zero")

    print("""  CONCEPT:
  ────────
  The kernel needs to compute ln(c_i). But ln(0) = −∞!

  To avoid this, we CLAMP every coordinate to the range [ε, 1−ε]
  where ε = 10⁻⁸ (the frozen guard band).

  ε-clamping rule:
    c_i,ε = max(ε, min(c_i, 1 − ε))

  This means:
    - If c_i = 0.0 → c_i,ε = 10⁻⁸    (not zero — tiny but finite)
    - If c_i = 1.0 → c_i,ε = 0.99999999  (not exactly 1)
    - If c_i = 0.5 → c_i,ε = 0.5      (unchanged — already in range)

  WHY? Two reasons:
    1. ln(0) is undefined → ε-clamping makes ln(c_i) finite
    2. ln(1) = 0  → clamping to 1−ε preserves slight information

  ε = 10⁻⁸ is FROZEN — discovered by the seam, not chosen by convention.
""")

    print_section("WORKED EXAMPLE: ε-Clamp Each Coordinate")

    eps = 1e-8
    raw_coords = [0.0, 0.001, 0.5, 0.999, 1.0]

    print(f"  ε = {eps}")
    print(f"  1 − ε = {1 - eps}\n")
    print(f"  {'c_i (raw)':>12s}  →  {'c_i,ε (clamped)':>16s}   ln(c_i,ε)")
    print(f"  {'─' * 12}     {'─' * 16}   {'─' * 12}")

    for ci in raw_coords:
        clamped = max(eps, min(ci, 1 - eps))
        log_val = np.log(clamped)
        flag = " ← clamped!" if ci != clamped else ""
        print(f"  {ci:12.8f}  →  {clamped:16.8f}   {log_val:12.6f}{flag}")

    print_section("KEY INSIGHT: Why ln(ε) Matters")
    print(f"  ln(ε) = ln(10⁻⁸) = {np.log(eps):.4f}")
    print("  This is a large negative number!")
    print(f"  A dead channel (c_i ≈ 0) contributes ≈ {np.log(eps):.2f} to κ.")
    print("  This is the mechanism behind 'geometric slaughter' (§3 of orientation).")

    print_section("EXERCISE 1.3")
    print("""  ε-clamp the following coordinates (ε = 10⁻⁸):
    a) c = −0.05  (out of range!)
    b) c = 0.0
    c) c = 0.73
    d) c = 1.0
    e) c = 1.2   (out of range!)

  Compute ln(c_i,ε) for each.
""")

    print_section("ANSWER 1.3")
    test_vals = [("a", -0.05), ("b", 0.0), ("c", 0.73), ("d", 1.0), ("e", 1.2)]
    for label, ci in test_vals:
        clamped = max(eps, min(ci, 1 - eps))
        print(f"    {label}) c = {ci:6.2f} → c_ε = {clamped:.8f}, ln(c_ε) = {np.log(clamped):.6f}")


# ════════════════════════════════════════════════════════════════════════
#  §1.4  PUTTING IT TOGETHER — A Complete Trace Setup
# ════════════════════════════════════════════════════════════════════════


def section_1_4_complete_trace() -> None:
    print_header("§1.4  PUTTING IT TOGETHER — A Complete Trace Setup")

    print("""  CONCEPT:
  ────────
  Before running the kernel, you need:
    1. A trace vector c = [c_1, c_2, ..., c_n]     (n channels)
    2. A weight vector w = [w_1, w_2, ..., w_n]     (Σ w_i = 1)
    3. ε-clamped coordinates c_ε = clamp(c, ε, 1−ε)

  Then EVERY kernel computation uses c_ε and w.
  Let's do a complete setup from scratch.
""")

    print_section("WORKED EXAMPLE: 8-Channel Particle Trace")

    channels = [
        "mass_log ",
        "spin_norm",
        "charge   ",
        "color    ",
        "weak_iso ",
        "lepton_# ",
        "baryon_# ",
        "generat'n",
    ]
    c_raw = np.array([0.65, 0.50, 1.00, 0.00, 0.50, 0.00, 0.333, 0.333])
    n = len(c_raw)
    eps = 1e-8

    print("  Step 1: Raw coordinates and channels\n")
    for i, (name, ci) in enumerate(zip(channels, c_raw, strict=True)):
        print(f"    Channel {i + 1} ({name}): c_{i + 1} = {ci:.3f}")

    print(f"\n  Step 2: Equal weights (n = {n})\n")
    w = np.ones(n) / n
    print(f"    w_i = 1/{n} = {w[0]:.6f}")
    print(f"    w = {w}")
    print(f"    Σ w_i = {w.sum():.1f} ✓")

    print(f"\n  Step 3: ε-clamp (ε = {eps})\n")
    c_clamped = np.clip(c_raw, eps, 1 - eps)
    print(f"    {'Channel':<12s}  {'c_raw':>8s}  →  {'c_ε':>12s}  {'Changed?':>10s}")
    print(f"    {'─' * 12}  {'─' * 8}     {'─' * 12}  {'─' * 10}")
    for i in range(n):
        changed = "YES ← clamp" if c_raw[i] != c_clamped[i] else "no"
        print(f"    {channels[i]:<12s}  {c_raw[i]:8.3f}  →  {c_clamped[i]:12.8f}  {changed:>10s}")

    print("\n  Ready for kernel computation!")
    print(f"    c_ε = {c_clamped}")
    print(f"    w   = {w}")

    print_section("EXERCISE 1.4")
    print("""  Set up a 5-channel trace for a financial system:
    Raw scores: [0.85, 0.0, 0.72, 1.0, 0.93]
    Use equal weights.

  Questions:
    a) What is w_i for each channel?
    b) Which channels get ε-clamped?
    c) Write out the final c_ε vector (ε = 10⁻⁸).
""")

    print_section("ANSWER 1.4")
    c_fin = np.array([0.85, 0.0, 0.72, 1.0, 0.93])
    w_fin = np.ones(5) / 5
    c_fin_clamped = np.clip(c_fin, eps, 1 - eps)
    print(f"    a) w_i = 1/5 = {w_fin[0]:.4f}")
    print("    b) Channels 2 (0.0 → ε) and 4 (1.0 → 1−ε) get clamped")
    print(f"    c) c_ε = {c_fin_clamped}")


# ════════════════════════════════════════════════════════════════════════
#  §1.5  SUMMARY — Level 1 Cheat Sheet
# ════════════════════════════════════════════════════════════════════════


def section_1_5_summary() -> None:
    print_header("§1.5  SUMMARY — Level 1 Cheat Sheet")

    print("""
  ┌────────────────────────────────────────────────────────────────┐
  │ CONCEPT         │ FORMULA / RULE                              │
  ├─────────────────┼────────────────────────────────────────────  │
  │ Coordinate c_i  │ Normalized measurement, c_i ∈ [0, 1]       │
  │ Weight w_i      │ Channel importance, Σ w_i = 1               │
  │ Equal weights   │ w_i = 1/n                                   │
  │ ε-clamping      │ c_i,ε = max(ε, min(c_i, 1−ε))             │
  │ ε value         │ 10⁻⁸ (frozen, seam-derived)                │
  │ Trace vector    │ c = [c_1, c_2, ..., c_n]                   │
  │ n channels      │ Measurable dimensions of the system         │
  └────────────────────────────────────────────────────────────────┘

  KEY IDEAS:
  • Each channel measures ONE property of the system
  • Dead channel: c_i = 0 (clamped to ε) — crucial for later (§3)
  • Weights distribute importance — they MUST sum to 1
  • ε-clamping keeps logarithms finite — protective, not distorting

  NEXT: Level 2 computes the invariants F, ω, S, C, κ, IC from these inputs.
""")


# ════════════════════════════════════════════════════════════════════════
#  MAIN — Run All Sections
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 72 + "╗")
    print("║  GCD KERNEL MATH WORKSHEETS — Level 1: Foundations" + " " * 20 + "║")
    print("║  Building blocks: coordinates, weights, ε-clamping" + " " * 19 + "║")
    print("╚" + "═" * 72 + "╝")

    section_1_1_coordinates()
    section_1_2_weights()
    section_1_3_epsilon_clamping()
    section_1_4_complete_trace()
    section_1_5_summary()
