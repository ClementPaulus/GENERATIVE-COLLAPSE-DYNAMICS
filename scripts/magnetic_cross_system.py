#!/usr/bin/env python3
"""Cross-system magnetic comparison: materials × Quincke × QDM.

Discovers structural relationships across three magnetic systems measured
by the same kernel — the cognitive equalizer at work.

Uses the formal closure: closures/materials_science/magnetic_cross_system.py
35 entities = 17 materials + 12 Quincke + 6 primary QDM phases.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from closures.materials_science.magnetic_cross_system import (
    build_cross_system_catalog,
    verify_all_theorems,
)


def main():
    # Build the 35-entity catalog from the formal closure
    all_results = build_cross_system_catalog()

    # Convert to dicts for display
    all_items = [r.to_dict() for r in all_results]

    n_total = len(all_items)
    print("=" * 100)
    print(f"CROSS-SYSTEM MAGNETIC KERNEL COMPARISON ({n_total} entities across 3 systems)")
    print("=" * 100)

    # Sort all by IC
    all_items.sort(key=lambda x: x["IC"])

    print(f"\n{'Name':30s} {'System':10s} {'F':>7s} {'IC':>10s} {'Δ':>7s} {'ω':>7s} {'S':>7s} {'C':>7s} {'IC/F':>7s}")
    print("-" * 100)
    for item in all_items:
        icf = item["IC"] / max(item["F"], 1e-12)
        print(
            f"  {item['name']:28s} {item['system']:10s} {item['F']:7.4f} {item['IC']:10.6f} "
            f"{item['gap']:7.4f} {item['omega']:7.4f} {item['S']:7.4f} {item['C']:7.4f} {icf:7.4f}"
        )

    # System-level statistics
    for system in ["Material", "Quincke", "QDM"]:
        items = [x for x in all_items if x["system"] == system]
        n = len(items)
        F_vals = [x["F"] for x in items]
        IC_vals = [x["IC"] for x in items]
        gap_vals = [x["gap"] for x in items]

        print(
            f"\n  {system:10s} (n={n}): "
            f"<F>={sum(F_vals) / n:.4f}  "
            f"<IC>={sum(IC_vals) / n:.6f}  "
            f"<Δ>={sum(gap_vals) / n:.4f}  "
            f"IC_range=[{min(IC_vals):.6f}, {max(IC_vals):.6f}]  "
            f"F_range=[{min(F_vals):.4f}, {max(F_vals):.4f}]"
        )

    # Regime distribution
    print("\n" + "=" * 100)
    print("REGIME DISTRIBUTION")
    print("=" * 100)
    collapse_items = [x for x in all_items if x["omega"] >= 0.30]
    watch_items = [x for x in all_items if 0.038 <= x["omega"] < 0.30]
    stable_items = [x for x in all_items if x["omega"] < 0.038]

    for label, items in [("COLLAPSE", collapse_items), ("WATCH", watch_items), ("STABLE", stable_items)]:
        systems_present = {x["system"] for x in items}
        counts = ", ".join(f"{s}: {sum(1 for x in items if x['system'] == s)}" for s in sorted(systems_present))
        print(f"\n  {label} (n={len(items)}): {counts}")
        for item in sorted(items, key=lambda x: x["IC"]):
            print(f"    {item['name']:28s} F={item['F']:.4f}  IC={item['IC']:.6f}  ω={item['omega']:.4f}")

    # Theorems
    print("\n" + "=" * 100)
    print("THEOREM VERIFICATION (T-MCS-1 through T-MCS-6)")
    print("=" * 100)
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")

    print("\n" + "=" * 100)
    print(f"DONE — {n_total} entities, 3 systems, one kernel, 6 theorems")
    print("=" * 100)


if __name__ == "__main__":
    main()
