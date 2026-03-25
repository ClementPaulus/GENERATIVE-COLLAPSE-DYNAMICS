#!/usr/bin/env python3
"""Magnetic kernel exploration — derive new structural facts.

NOT a permanent closure. Exploratory computation to identify
new magnetic kernel patterns across all 17 materials, temperature
sweeps, frustration analysis, and cross-system comparison.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from closures.materials_science.magnetic_properties import (
    REFERENCE_MAGNETIC,
    compute_magnetic_properties,
)
from umcp.kernel_optimized import OptimizedKernelComputer

_KERNEL = OptimizedKernelComputer()


def compute_kernel(channels: list[float], weights: list[float]):
    """Compute kernel from plain lists."""
    return _KERNEL.compute(np.array(channels), np.array(weights))


EPS = 1e-8

Z_MAP = {
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Gd": 64,
    "Dy": 66,
    "Cr": 24,
    "Mn": 25,
    "MnO": 25,
    "NiO": 28,
    "FeO": 26,
    "CoO": 27,
    "Fe3O4": 26,
    "Cu": 29,
    "Au": 79,
    "Bi": 83,
    "Al": 13,
    "Pt": 78,
}


def build_trace(sym: str, T_K: float = 300.0) -> tuple[list[float], list[float], str]:
    """Build 8-channel trace vector for a magnetic material at temperature T."""
    ref = REFERENCE_MAGNETIC[sym]
    r = compute_magnetic_properties(Z_MAP[sym], symbol=sym, T_K=T_K)

    M_sat = float(ref.get("M_sat", 0))
    T_c = float(ref.get("T_c", ref.get("T_N", 0)))
    n_unp = int(ref.get("n_unpaired", 0))
    cls = str(ref.get("class", ""))

    c1 = max(EPS, min(1 - EPS, abs(r.M_total_B) / max(M_sat, 0.001)))
    c2 = max(EPS, min(1 - EPS, 1 - T_K / T_c if T_c > T_K else EPS))
    c3 = max(EPS, min(1 - EPS, abs(r.J_exchange_meV) / 20.0))
    c4 = max(EPS, min(1 - EPS, 1.0 / (1.0 + abs(r.chi_SI) * 1e6)))
    c5 = max(EPS, min(1 - EPS, n_unp / 7.0 if n_unp > 0 else EPS))
    c6 = max(EPS, min(1 - EPS, M_sat / 10.2 if M_sat > 0 else EPS))
    c7 = max(EPS, min(1 - EPS, T_c / 1394.0 if T_c > 0 else EPS))

    if cls in ("Ferromagnetic", "Ferrimagnetic"):
        c8 = 0.95
    elif cls == "Antiferromagnetic":
        c8 = 0.50
    else:
        c8 = EPS
    c8 = max(EPS, min(1 - EPS, c8))

    channels = [c1, c2, c3, c4, c5, c6, c7, c8]
    weights = [1 / 8] * 8
    return channels, weights, cls


def main() -> None:
    # ═══════════════════════════════════════════════════════════════
    # PART 1: Kernel invariants for all 17 materials at T=300K
    # ═══════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PART 1: KERNEL INVARIANTS FOR ALL 17 MAGNETIC MATERIALS (T=300K)")
    print("=" * 80)

    results: dict[str, dict] = {}
    for sym in REFERENCE_MAGNETIC:
        channels, weights, cls = build_trace(sym, 300.0)
        k = compute_kernel(channels, weights)
        results[sym] = {"channels": channels, "kernel": k, "class": cls}
        print(
            f"  {sym:6s} [{cls:18s}] F={k.F:.4f} IC={k.IC:.6f} "
            f"Δ={k.F - k.IC:.4f} ω={k.omega:.4f} S={k.S:.4f} C={k.C:.4f}"
        )

    print()
    print("--- Sorted by IC (ascending = weakest integrity first) ---")
    for sym in sorted(results, key=lambda s: results[s]["kernel"].IC):
        k = results[sym]["kernel"]
        c = results[sym]["class"]
        ratio = k.IC / max(k.F, 1e-12)
        print(f"  {sym:6s} [{c:18s}] IC={k.IC:.6f}  Δ={k.F - k.IC:.4f}  IC/F={ratio:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PART 2: Class averages — FM vs AF vs Dia vs Para
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("PART 2: CLASS AVERAGES (Kernel Mean ± Spread)")
    print("=" * 80)

    from collections import defaultdict

    class_data: dict[str, list[dict]] = defaultdict(list)
    for _sym, res in results.items():
        class_data[res["class"]].append(res)

    for cls in ["Ferromagnetic", "Antiferromagnetic", "Ferrimagnetic", "Diamagnetic", "Paramagnetic"]:
        if cls not in class_data:
            continue
        items = class_data[cls]
        n = len(items)
        F_vals = [d["kernel"].F for d in items]
        IC_vals = [d["kernel"].IC for d in items]
        D_vals = [d["kernel"].F - d["kernel"].IC for d in items]
        S_vals = [d["kernel"].S for d in items]
        C_vals = [d["kernel"].C for d in items]

        F_avg = sum(F_vals) / n
        IC_avg = sum(IC_vals) / n
        D_avg = sum(D_vals) / n
        S_avg = sum(S_vals) / n
        C_avg = sum(C_vals) / n

        F_std = (sum((x - F_avg) ** 2 for x in F_vals) / max(n - 1, 1)) ** 0.5
        IC_std = (sum((x - IC_avg) ** 2 for x in IC_vals) / max(n - 1, 1)) ** 0.5

        print(
            f"  {cls:20s} (n={n}): "
            f"<F>={F_avg:.4f}±{F_std:.4f}  "
            f"<IC>={IC_avg:.6f}±{IC_std:.6f}  "
            f"<Δ>={D_avg:.4f}  <S>={S_avg:.4f}  <C>={C_avg:.4f}"
        )

    # ═══════════════════════════════════════════════════════════════
    # PART 3: Temperature sweep — Fe through its Curie point
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("PART 3: TEMPERATURE SWEEP — Fe (T_c=1043K)")
    print("=" * 80)

    T_c_Fe = 1043.0
    temps = [50, 100, 200, 300, 500, 700, 900, 1000, 1020, 1040, 1043, 1050, 1100, 1200, 1500]

    for T in temps:
        channels, weights, _ = build_trace("Fe", float(T))
        k = compute_kernel(channels, weights)
        T_ratio = T / T_c_Fe
        marker = " ← T_c" if T == 1043 else ""
        print(
            f"  T={T:5d}K  T/T_c={T_ratio:.3f}  "
            f"F={k.F:.4f}  IC={k.IC:.6f}  Δ={k.F - k.IC:.4f}  "
            f"ω={k.omega:.4f}  S={k.S:.4f}  C={k.C:.4f}{marker}"
        )

    # ═══════════════════════════════════════════════════════════════
    # PART 4: Temperature sweep — Gd (near room temperature T_c=292K)
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("PART 4: TEMPERATURE SWEEP — Gd (T_c=292K, near room temp)")
    print("=" * 80)

    T_c_Gd = 292.0
    temps_gd = [50, 100, 150, 200, 250, 280, 290, 291, 292, 293, 295, 300, 350, 400, 500]

    for T in temps_gd:
        channels, weights, _ = build_trace("Gd", float(T))
        k = compute_kernel(channels, weights)
        T_ratio = T / T_c_Gd
        marker = " ← T_c" if T == 292 else ""
        print(
            f"  T={T:5d}K  T/T_c={T_ratio:.3f}  "
            f"F={k.F:.4f}  IC={k.IC:.6f}  Δ={k.F - k.IC:.4f}  "
            f"ω={k.omega:.4f}  S={k.S:.4f}  C={k.C:.4f}{marker}"
        )

    # ═══════════════════════════════════════════════════════════════
    # PART 5: Frustration analysis — competing J_FM and J_AF
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("PART 5: FRUSTRATION PARAMETER SWEEP — η = J_FM/(J_FM + J_AF)")
    print("=" * 80)
    print("  (Modeled as 8-channel system where frustration kills c3,c4,c7)")

    for eta_pct in range(0, 105, 5):
        eta = eta_pct / 100.0
        # At eta=0 (pure AF) or eta=1 (pure FM): one ordering wins, channels alive
        # At eta=0.5: maximum frustration, channels die
        frustration = 1.0 - 4.0 * (eta - 0.5) ** 2  # peaks at η=0.5
        frustration = max(0, min(1, frustration))

        # Channels: c1=intralayer(alive), c2=interlayer(alive),
        # c3=frustration_low(dies at peak), c4=hysteresis_low(dies at peak),
        # c5=dissipation_low, c6=velocity_coherence,
        # c7=config_stability(dies at peak), c8=sliding_fidelity
        c1 = 0.7
        c2 = max(EPS, min(1 - EPS, eta * 0.9 + 0.05))  # FM ordering
        c3 = max(EPS, min(1 - EPS, 1.0 - 0.95 * frustration))  # frustration kills this
        c4 = max(EPS, min(1 - EPS, 1.0 - 0.90 * frustration))  # hysteresis
        c5 = max(EPS, min(1 - EPS, 1.0 - 0.70 * frustration))  # dissipation
        c6 = max(EPS, min(1 - EPS, 0.85 - 0.55 * frustration))  # velocity coherence
        c7 = max(EPS, min(1 - EPS, 1.0 - 0.95 * frustration))  # config stability
        c8 = max(EPS, min(1 - EPS, 0.90 - 0.70 * frustration))  # sliding fidelity

        channels = [c1, c2, c3, c4, c5, c6, c7, c8]
        weights = [1 / 8] * 8
        k = compute_kernel(channels, weights)

        marker = " ← PEAK FRUSTRATION" if eta_pct == 50 else ""
        print(
            f"  η={eta:.2f}  frust={frustration:.3f}  "
            f"F={k.F:.4f}  IC={k.IC:.6f}  Δ={k.F - k.IC:.4f}  "
            f"IC/F={k.IC / max(k.F, 1e-12):.4f}  "
            f"ω={k.omega:.4f}  S={k.S:.4f}{marker}"
        )

    # ═══════════════════════════════════════════════════════════════
    # PART 6: Key structural facts discovered
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("PART 6: DERIVED STRUCTURAL FACTS")
    print("=" * 80)

    # Fact 1: FM vs AF integrity ratio
    fm_ics = [results[s]["kernel"].IC for s in results if results[s]["class"] == "Ferromagnetic"]
    af_ics = [results[s]["kernel"].IC for s in results if results[s]["class"] == "Antiferromagnetic"]
    dia_ics = [results[s]["kernel"].IC for s in results if results[s]["class"] == "Diamagnetic"]

    fm_avg = sum(fm_ics) / len(fm_ics)
    af_avg = sum(af_ics) / len(af_ics)
    dia_avg = sum(dia_ics) / len(dia_ics)

    print("\n  FACT 1: FM/AF Integrity Ratio")
    print(f"    <IC>_FM  = {fm_avg:.6f}")
    print(f"    <IC>_AF  = {af_avg:.6f}")
    print(f"    <IC>_Dia = {dia_avg:.6f}")
    print(f"    FM/AF ratio = {fm_avg / max(af_avg, 1e-12):.2f}×")
    print(f"    FM/Dia ratio = {fm_avg / max(dia_avg, 1e-12):.2f}×")

    # Fact 2: Heterogeneity gap by class
    fm_gaps = [
        results[s]["kernel"].F - results[s]["kernel"].IC for s in results if results[s]["class"] == "Ferromagnetic"
    ]
    af_gaps = [
        results[s]["kernel"].F - results[s]["kernel"].IC for s in results if results[s]["class"] == "Antiferromagnetic"
    ]
    dia_gaps = [
        results[s]["kernel"].F - results[s]["kernel"].IC for s in results if results[s]["class"] == "Diamagnetic"
    ]

    print("\n  FACT 2: Heterogeneity Gap by Class")
    print(f"    <Δ>_FM  = {sum(fm_gaps) / len(fm_gaps):.4f}")
    print(f"    <Δ>_AF  = {sum(af_gaps) / len(af_gaps):.4f}")
    print(f"    <Δ>_Dia = {sum(dia_gaps) / len(dia_gaps):.4f}")

    # Fact 3: Which channel is the IC killer for each class?
    print("\n  FACT 3: IC-Killing Channels (min channel per material)")
    for sym in results:
        ch = results[sym]["channels"]
        min_ch = min(ch)
        min_idx = ch.index(min_ch)
        ch_names = [
            "moment_ratio",
            "ordering_str",
            "exchange_str",
            "suscept_coh",
            "unpaired_frac",
            "satur_ratio",
            "thermal_stab",
            "class_coh",
        ]
        k = results[sym]["kernel"]
        if min_ch < 0.01:
            print(f"    {sym:6s}: min channel = c{min_idx + 1} ({ch_names[min_idx]}) = {min_ch:.2e} → IC={k.IC:.6f}")

    # Fact 4: Duality check
    print("\n  FACT 4: Duality Identity F + ω = 1 (verification)")
    max_residual = 0.0
    for sym in results:
        k = results[sym]["kernel"]
        residual = abs(k.F + k.omega - 1.0)
        max_residual = max(max_residual, residual)
    print(f"    max|F + ω - 1| = {max_residual:.2e} (must be 0.0)")

    # Fact 5: Integrity bound
    print("\n  FACT 5: Integrity Bound IC ≤ F (verification)")
    all_pass = True
    for sym in results:
        k = results[sym]["kernel"]
        if k.IC > k.F + 1e-10:
            print(f"    VIOLATION: {sym} IC={k.IC} > F={k.F}")
            all_pass = False
    print(f"    IC ≤ F: {'PASSED' if all_pass else 'FAILED'} for all 17 materials")

    # Fact 6: Correlation between T_c and IC
    print("\n  FACT 6: Correlation T_c vs IC (ordered materials only)")
    ordered = []
    for sym in results:
        ref = REFERENCE_MAGNETIC[sym]
        T_c = float(ref.get("T_c", ref.get("T_N", 0)))
        if T_c > 0:
            ordered.append((sym, T_c, results[sym]["kernel"].IC, results[sym]["class"]))

    ordered.sort(key=lambda x: x[1])
    for sym, tc, ic, cls in ordered:
        print(f"    {sym:6s} T_c={tc:7.1f}K  IC={ic:.6f}  [{cls}]")

    # Compute Spearman rank correlation
    t_ranks = list(range(len(ordered)))
    ic_sorted = sorted(range(len(ordered)), key=lambda i: ordered[i][2])
    ic_ranks = [0] * len(ordered)
    for rank, idx in enumerate(ic_sorted):
        ic_ranks[idx] = rank

    n = len(ordered)
    d_sq = sum((t_ranks[i] - ic_ranks[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d_sq / (n * (n**2 - 1))
    print(f"\n    Spearman ρ(T_c, IC) = {rho:.4f}")
    if abs(rho) > 0.6:
        direction = "positive" if rho > 0 else "negative"
        print(f"    → Strong {direction} correlation: higher T_c → {'higher' if rho > 0 else 'lower'} IC at 300K")
    else:
        print("    → Weak or no monotonic correlation")

    print()
    print("=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
