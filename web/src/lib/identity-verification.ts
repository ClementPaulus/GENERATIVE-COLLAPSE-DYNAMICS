/**
 * Identity Verification State Model
 *
 * Each structural identity has a verification state that honestly describes
 * what the web layer can demonstrate. This prevents overclaiming —
 * no identity is labelled "verified" unless the page actually runs a
 * computation that checks it.
 *
 * Four states, ordered by strength of evidence:
 *
 *   exact      — Algebraically forced; the live-verify button runs a sweep
 *                and confirms the residual to machine precision.
 *   sampled    — Numerically checked by sampling random configurations;
 *                the live-verify button runs a Monte Carlo sweep.
 *   narrative  — Structural / analytical result whose proof is given in
 *                prose. No live computation is offered in-browser.
 *   pending    — Derivation is incomplete, or the canonical source has
 *                not yet been linked. Honest about the gap.
 */

export type VerificationState = 'exact' | 'sampled' | 'narrative' | 'pending';

export interface VerificationBadge {
  state: VerificationState;
  label: string;
  color: string;
  bgColor: string;
  tooltip: string;
}

export const VERIFICATION_BADGES: Record<VerificationState, VerificationBadge> = {
  exact: {
    state: 'exact',
    label: 'Exact',
    color: 'text-green-400',
    bgColor: 'bg-green-900/30 border-green-700/50',
    tooltip: 'Algebraically forced — verified to machine precision by live computation',
  },
  sampled: {
    state: 'sampled',
    label: 'Sampled',
    color: 'text-blue-400',
    bgColor: 'bg-blue-900/30 border-blue-700/50',
    tooltip: 'Numerically checked by Monte Carlo sweep in-browser',
  },
  narrative: {
    state: 'narrative',
    label: 'Narrative',
    color: 'text-kernel-400',
    bgColor: 'bg-kernel-800/50 border-kernel-700/50',
    tooltip: 'Analytical result — proof given in prose; no live computation in-browser',
  },
  pending: {
    state: 'pending',
    label: 'Pending',
    color: 'text-amber-400',
    bgColor: 'bg-amber-900/30 border-amber-700/50',
    tooltip: 'Derivation incomplete or canonical source not yet linked',
  },
};

/**
 * Render an HTML badge for a verification state.
 */
export function renderBadge(state: VerificationState): string {
  const badge = VERIFICATION_BADGES[state];
  return `<span class="text-[10px] font-medium px-2 py-0.5 rounded border ${badge.bgColor} ${badge.color}" title="${badge.tooltip}">${badge.label}</span>`;
}

/**
 * Map from identity ID to its verification state.
 *
 * Classification criteria:
 *   exact    — definition-level algebraic identities; verification code
 *              computes an exact residual (e.g. |F+ω−1| = 0.0e+00)
 *   sampled  — inequalities or statistical properties checked by sampling
 *              (e.g. IC ≤ F across 500 random traces)
 *   narrative — proofs rely on calculus, series, or analytic arguments
 *              that are not reproduced by the in-browser kernel
 */
export const IDENTITY_VERIFICATION: Record<string, VerificationState> = {
  // E series — Equator / Exact
  E1: 'exact',     // F + ω = 1 — algebraic tautology, residual = 0.0
  E2: 'exact',     // IC = exp(κ) — definitional, residual < 10⁻¹⁵
  E3: 'sampled',   // IC ≤ F — inequality checked by sampling
  E4: 'exact',     // S(½) + κ(½) = 0 — exact evaluation at a point
  E5: 'narrative',  // g_F(θ) = 1 — Fisher metric calculation (not in TS kernel)
  E6: 'narrative',  // f(θ) = 2cos²θ·ln(tan θ) — Fisher coordinate identity
  E7: 'narrative',  // ∫g_F·S dc = π²/3 — spectral integral (quadrature)
  E8: 'narrative',  // Seam associativity — proved via composition rules

  // B series — Bound / Composition
  B1: 'sampled',   // S ≤ h(F) — inequality checked by sampling
  B2: 'narrative',  // IC geometric composition — proved algebraically
  B3: 'narrative',  // F arithmetic composition — proved by linearity
  B4: 'narrative',  // Gap composition law — proved from B2 + B3
  B5: 'sampled',   // Perturbation chain — near-homogeneous approximation checked
  B6: 'exact',     // Cardano root — x³+x−1 evaluated at x=0.6823
  B7: 'exact',     // Equator quintuple point — exact evaluation at c=½
  B8: 'sampled',   // c* — verified via frozen_contract.py bisection equation
  B9: 'narrative',  // Reflection formula — antisymmetry argument
  B10: 'sampled',  // Fisher space partition — Monte Carlo sampling
  B11: 'narrative', // Super-exponential convergence — CLT argument
  B12: 'narrative', // Low-rank closures — PCA argument

  // D series — Duality / Deep
  D1: 'exact',     // Solvability equation — algebraic construction
  D2: 'narrative',  // Rank classification — structural argument
  D3: 'exact',     // Homogeneous fast path — exact evaluation
  D4: 'sampled',   // Confinement cliff — computed from specific traces
  D5: 'sampled',   // Scale inversion — computed from specific traces
  D6: 'narrative',  // Het. gap invariant — follows from B4
  D7: 'narrative',  // Entropy-curvature correlation — CLT argument
  D8: 'sampled',   // Budget conservation — computed from specific traces
};

/**
 * Returns true if the identity has a live verification path in-browser.
 */
export function hasLiveVerification(id: string): boolean {
  const state = IDENTITY_VERIFICATION[id];
  return state === 'exact' || state === 'sampled';
}
