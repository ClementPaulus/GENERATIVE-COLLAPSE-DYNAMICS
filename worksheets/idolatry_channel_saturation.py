"""Idolatry as channel saturation — structural verification."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import OptimizedKernelComputer

kernel = OptimizedKernelComputer(epsilon=EPSILON)
CHANNELS = [
    "material_wealth",
    "community",
    "inner_peace",
    "integrity",
    "compassion",
    "forgiveness",
    "humility",
    "purpose",
]
w = np.ones(8) / 8

# Balanced baseline (composite of prescriptive teachings)
balanced = np.array([0.35, 0.86, 0.80, 0.89, 0.85, 0.79, 0.89, 0.88])

print("=" * 70)
print("  IDOLATRY = CHANNEL SATURATION = GEOMETRIC SLAUGHTER")
print("=" * 70)
print()
print('  "You shall have no other gods before me" (Exodus 20:3)')
print('  "You shall not make for yourself an idol" (Exodus 20:4)')
print()
print("  Structural translation: Do not saturate ANY single channel")
print("  to the point where it starves the others. The commandment")
print("  is domain-general -- it applies to ANYTHING that becomes")
print("  the dominant channel.")
print()

header = f"  {'Idolized Channel':<20} {'F':>6} {'IC':>6} {'D':>6} {'IC/F':>6} {'C':>6} {'Regime':<15}"
sep = f"  {'---' * 5:<20} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':<15}"
print(header)
print(sep)

out = kernel.compute(balanced, w)
print(
    f"  {'NO IDOLATRY':<20} {out.F:6.3f} {out.IC:6.3f} "
    f"{out.heterogeneity_gap:6.3f} {out.IC / out.F:6.3f} {out.C:6.3f} {out.regime:<15}"
)
print()

for i, ch in enumerate(CHANNELS):
    idol = np.full(8, 0.15)
    idol[i] = 0.98
    out = kernel.compute(idol, w)
    print(
        f"  {ch:<20} {out.F:6.3f} {out.IC:6.3f} "
        f"{out.heterogeneity_gap:6.3f} {out.IC / out.F:6.3f} {out.C:6.3f} {out.regime:<15}"
    )

print("""
  --- THE STRUCTURAL INSIGHT ---

  It does not matter WHICH channel you idolize.
  Idolize wealth    --> fragmented.
  Idolize purpose   --> fragmented.
  Idolize compassion--> fragmented.
  Idolize humility  --> fragmented.

  EVERY form of idolatry produces the SAME structural result:
  one saturated channel, seven starving channels, IC crashes.

  The commandment against idolatry is not about which object
  you worship. It is about the STRUCTURAL PATTERN of saturating
  any single channel at the expense of the others.

  Even idolizing a "good" thing (compassion, humility, purpose)
  produces geometric slaughter. The kernel is indifferent to
  moral valence -- it measures channel heterogeneity.
""")

print("=" * 70)
print("  HOW MUCH IDOLATRY BEFORE COLLAPSE?")
print("=" * 70)
print()
print("  Gradually saturate one channel while depleting others:")
print()
print(f"  {'Idol level':>10} {'Others':>7} {'F':>6} {'IC':>6} {'IC/F':>6} {'C':>6} {'Regime':<15}")
print(
    f"  {'----------':>10} {'-------':>7} {'------':>6} {'------':>6} {'------':>6} {'------':>6} {'---------------':<15}"
)

for idol_level in np.arange(0.50, 1.00, 0.05):
    others_level = max(0.10, 0.80 - (idol_level - 0.50) * 1.2)
    c = np.full(8, others_level)
    c[0] = idol_level
    out = kernel.compute(c, w)
    marker = " <-- coherence threshold" if 0.93 < out.IC / out.F < 0.97 else ""
    print(
        f"  {idol_level:10.2f} {others_level:7.2f} {out.F:6.3f} {out.IC:6.3f} "
        f"{out.IC / out.F:6.3f} {out.C:6.3f} {out.regime:<15}{marker}"
    )

print("""
  The system transitions from coherent to fragmented as the
  idolized channel pulls away from the others. There is a
  THRESHOLD where IC/F drops below ~0.95 and coherence is lost.

  The prohibition against idolatry is: do not cross this threshold.
  It is not about the object. It is about the PATTERN.

  This also explains why the commandment is FIRST -- before
  "do not murder," "do not steal," etc. Channel saturation
  is the ROOT structural failure. All other moral failures
  are DOWNSTREAM consequences of a fragmented trace vector.

  If IC/F is near 1.0 (no idolatry, balanced channels), the
  system is coherent and the other commandments follow naturally.
  If IC/F crashes (one idol dominates), the system fragments
  and all channels degrade -- including the ones that prevent
  murder, theft, and false witness.

  The ordering of the commandments IS the ordering of
  structural priority. Anti-idolatry is first because
  geometric slaughter is the root cause.

  Structura mensurat, non agens.
""")
