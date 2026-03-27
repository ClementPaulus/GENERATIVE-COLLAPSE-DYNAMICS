/**
 * Spacetime Physics — GCD Budget Surface and Gravitational Analogs
 *
 * Ported from closures/spacetime_memory/spacetime_kernel.py
 * Maps GCD kernel invariants to gravitational phenomena:
 *   Event horizon  = ω → 1 pole (Γ → ∞)
 *   Gravity        = dΓ/dω (gradient of drift cost)
 *   Mass           = accumulated |κ| (well depth)
 *   Tidal force    = d²Γ/dω² (curvature of cost surface)
 *   Lensing        = deflection from heterogeneity gap Δ
 *   Time dilation  = descent/ascent cost asymmetry
 *
 * Tier-0 Protocol: no Tier-1 symbol is redefined.
 */

import { EPSILON, P_EXPONENT, ALPHA } from './constants';

/* ─── Budget Surface ────────────────────────────────────────────── */

/**
 * Budget surface height: Γ(ω) + α·C
 * The 2D cost landscape over (ω, C) parameter space.
 */
export function budgetSurfaceHeight(
  omega: number,
  C: number,
  p: number = P_EXPONENT,
  alpha: number = ALPHA,
  epsilon: number = EPSILON,
): number {
  let num = 1.0;
  for (let i = 0; i < p; i++) num *= omega;
  const gamma = num / (1.0 - omega + epsilon);
  return gamma + alpha * C;
}

/* ─── Derivatives of Γ(ω) — Gravitational Analogs ───────────────── */

/**
 * First derivative of Γ(ω) = ω^p / (1 - ω + ε)
 * dΓ/dω = [p·ω^(p-1)·(1-ω+ε) + ω^p] / (1-ω+ε)²
 * Gravitational analog: "gravitational field strength"
 */
export function dGamma(
  omega: number,
  p: number = P_EXPONENT,
  epsilon: number = EPSILON,
): number {
  const denom = 1.0 - omega + epsilon;
  let omP = 1.0;
  for (let i = 0; i < p; i++) omP *= omega;
  let omPm1 = 1.0;
  for (let i = 0; i < p - 1; i++) omPm1 *= omega;
  return (p * omPm1 * denom + omP) / (denom * denom);
}

/**
 * Second derivative of Γ(ω).
 * Tidal force analog: measures how rapidly the gravitational field changes.
 * Computed numerically for robustness.
 */
export function d2Gamma(
  omega: number,
  p: number = P_EXPONENT,
  epsilon: number = EPSILON,
): number {
  const h = 1e-6;
  const left = omega - h > 0 ? dGamma(omega - h, p, epsilon) : dGamma(0, p, epsilon);
  const right = omega + h < 1 ? dGamma(omega + h, p, epsilon) : dGamma(1.0 - epsilon, p, epsilon);
  return (right - left) / (2 * h);
}

/* ─── Well Depth (Mass Analog) ──────────────────────────────────── */

/**
 * Well depth = |κ|. Accumulated log-integrity measures how deep
 * the gravitational potential well is. Larger |κ| → more massive object.
 */
export function wellDepth(kappa: number): number {
  return Math.abs(kappa);
}

/* ─── Gravitational Lensing ─────────────────────────────────────── */

export type LensingMorphology =
  | 'perfect_ring'   // Δ < 0.01 — Einstein ring
  | 'thick_arc'      // Δ < 0.10 — strong lensing arc
  | 'thin_arc'       // Δ < 0.30 — distorted arc
  | 'distorted'      // Δ ≥ 0.30 — weak lensing distortion
  ;

/**
 * Deflection angle from heterogeneity gap and well depth.
 * θ_defl = 4·wellDepth / (Δ + ε)
 * Analogous to Einstein's deflection formula θ = 4GM/(c²b).
 */
export function deflectionAngle(
  delta: number,
  wd: number,
  epsilon: number = EPSILON,
): number {
  return 4.0 * wd / (delta + epsilon);
}

/**
 * Classify lensing morphology from heterogeneity gap.
 */
export function classifyLensing(delta: number): LensingMorphology {
  if (delta < 0.01) return 'perfect_ring';
  if (delta < 0.10) return 'thick_arc';
  if (delta < 0.30) return 'thin_arc';
  return 'distorted';
}

/* ─── Arrow of Time (Descent/Ascent Asymmetry) ──────────────────── */

/**
 * Cost of descending from omStart to omEnd (falling in).
 * Integral of Γ(ω) from omStart to omEnd via trapezoidal rule.
 */
export function descentCost(
  omStart: number,
  omEnd: number,
  steps: number = 200,
): number {
  if (omStart >= omEnd) return 0;
  const h = (omEnd - omStart) / steps;
  let sum = 0;
  for (let i = 0; i <= steps; i++) {
    const om = omStart + i * h;
    const g = budgetSurfaceHeight(om, 0);
    sum += (i === 0 || i === steps) ? g * 0.5 : g;
  }
  return sum * h;
}

/**
 * Cost of ascending from omEnd back to omStart (escaping).
 * Asymmetric: ascent integrates against the Γ gradient.
 * ascentCost = descentCost × (1 + Γ(omEnd)/Γ(omStart+ε))
 */
export function ascentCost(
  omStart: number,
  omEnd: number,
): number {
  const dc = descentCost(omStart, omEnd);
  const gStart = budgetSurfaceHeight(Math.max(omStart, 0.001), 0);
  const gEnd = budgetSurfaceHeight(omEnd, 0);
  return dc * (1 + gEnd / (gStart + EPSILON));
}

/**
 * Arrow asymmetry: ratio of ascent cost to descent cost.
 * > 1 means it's harder to escape than to fall in.
 * At the event horizon this diverges — no return.
 */
export function arrowAsymmetry(
  omStart: number,
  omEnd: number,
): number {
  const dc = descentCost(omStart, omEnd);
  if (dc < EPSILON) return 1.0;
  return ascentCost(omStart, omEnd) / dc;
}

/* ─── Black Hole Entities ───────────────────────────────────────── */

export interface SpacetimeEntity {
  name: string;
  symbol: string;
  c: number[];
  w: number[];
  description: string;
  grAnalog: string;
}

/**
 * Curated black hole entities for the simulation.
 * Trace vectors from closures/spacetime_memory/*.py
 */
export const BLACK_HOLE_ENTITIES: SpacetimeEntity[] = [
  {
    name: 'Stellar Black Hole',
    symbol: 'BH',
    c: [0.99, 0.50, 0.99, 0.99, 0.60, 0.10, 0.95, 0.40],
    w: Array(8).fill(0.125),
    description: '~3–20 M☉. Endpoint of massive star collapse. One channel (c[5]=0.10) near floor — the information channel collapses at the horizon.',
    grAnalog: 'Schwarzschild solution (non-rotating, uncharged)',
  },
  {
    name: 'Event Horizon',
    symbol: 'EH',
    c: [0.10, 0.02, 0.08, 0.02, 0.05, 0.05, 0.45, 0.65],
    w: Array(8).fill(0.125),
    description: 'The surface of no return. Almost all channels near floor — maximum drift, minimum fidelity. The pole Γ(ω→1)→∞ lives here.',
    grAnalog: 'r = 2GM/c² (Schwarzschild radius)',
  },
  {
    name: 'Photon Sphere',
    symbol: 'PS',
    c: [0.30, 0.15, 0.25, 0.12, 0.25, 0.20, 0.55, 0.70],
    w: Array(8).fill(0.125),
    description: 'r = 3GM/c². Light orbits but is unstable. Intermediate collapse — all channels depressed but not at floor.',
    grAnalog: 'r = 1.5 × Schwarzschild radius',
  },
  {
    name: 'Binary BH Merger',
    symbol: 'BBH',
    c: [0.95, 0.85, 0.90, 0.80, 0.90, 0.70, 0.95, 0.95],
    w: Array(8).fill(0.125),
    description: 'Pre-merger inspiral. High coherence — the gravitational wave signal carries nearly intact information. IC high.',
    grAnalog: 'LIGO/Virgo inspiral-merger-ringdown',
  },
  {
    name: 'Accretion Disk',
    symbol: 'AD',
    c: [0.75, 0.60, 0.70, 0.50, 0.65, 0.40, 0.80, 0.55],
    w: Array(8).fill(0.125),
    description: 'Matter spiraling inward. Intermediate fidelity — energy extracted but structure partially preserved. Watch regime typical.',
    grAnalog: 'Shakura-Sunyaev thin disk / ADAF',
  },
  {
    name: 'Near-Horizon Loop',
    symbol: 'NHL',
    c: [0.20, 0.30, 0.10, 0.25, 0.15, 0.12, 0.90, 0.15],
    w: Array(8).fill(0.125),
    description: 'Causal loop at the stretched horizon. Most channels near floor, but c[6]=0.90 — the circulation channel persists even at the boundary.',
    grAnalog: 'Stretched horizon / membrane paradigm',
  },
];
