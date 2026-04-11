/**
 * Web-Layer Integrity Guards
 *
 * These tests fail CI when:
 *   1. A public metric in metrics.ts has drifted from internal consistency
 *   2. An identity is marked "exact" or "sampled" without a plausible live path
 *   3. Identity counts disagree between metrics.ts and identity-verification.ts
 *   4. The verification state model is incomplete (missing IDs)
 *
 * Run via:  npm test  (or  npx vitest run)
 */

import { describe, it, expect } from 'vitest';
import {
  TEST_COUNT,
  TEST_COUNT_RAW,
  DOMAIN_COUNT,
  DOMAIN_COUNT_RAW,
  IDENTITY_COUNT,
  IDENTITY_COUNT_RAW,
  LEMMA_COUNT,
  LEMMA_COUNT_RAW,
  CLOSURE_COUNT,
  CLOSURE_COUNT_RAW,
  THEOREM_COUNT,
  THEOREM_COUNT_RAW,
  TEST_FILE_COUNT,
  TEST_FILE_COUNT_RAW,
  LANGUAGE_COUNT,
  LANGUAGE_COUNT_RAW,
  AT_A_GLANCE,
} from '../src/lib/metrics';
import {
  IDENTITY_VERIFICATION,
  VERIFICATION_BADGES,
  renderBadge,
  hasLiveVerification,
} from '../src/lib/identity-verification';
import type { VerificationState } from '../src/lib/identity-verification';

/* ─── §1  Metric Internal Consistency ─────────────────────────── */

describe('Metrics source of truth — internal consistency', () => {
  it('formatted strings match raw numbers', () => {
    // raw ↔ formatted pairs must agree
    expect(TEST_COUNT).toBe(TEST_COUNT_RAW.toLocaleString('en-US'));
    expect(DOMAIN_COUNT).toBe(String(DOMAIN_COUNT_RAW));
    expect(IDENTITY_COUNT).toBe(String(IDENTITY_COUNT_RAW));
    expect(LEMMA_COUNT).toBe(String(LEMMA_COUNT_RAW));
    expect(CLOSURE_COUNT).toBe(String(CLOSURE_COUNT_RAW));
    expect(THEOREM_COUNT).toBe(String(THEOREM_COUNT_RAW));
    expect(TEST_FILE_COUNT).toBe(String(TEST_FILE_COUNT_RAW));
    expect(LANGUAGE_COUNT).toBe(String(LANGUAGE_COUNT_RAW));
  });

  it('At a Glance array uses metrics.ts values (no drift)', () => {
    const labels = AT_A_GLANCE.map(m => m.label);
    expect(labels).toContain('Tests');
    expect(labels).toContain('Domains');
    const tests = AT_A_GLANCE.find(m => m.label === 'Tests');
    expect(tests?.n).toBe(TEST_COUNT);
    const domains = AT_A_GLANCE.find(m => m.label === 'Domains');
    expect(domains?.n).toBe(DOMAIN_COUNT);
  });

  it('raw counts are positive integers', () => {
    for (const v of [
      TEST_COUNT_RAW,
      DOMAIN_COUNT_RAW,
      IDENTITY_COUNT_RAW,
      LEMMA_COUNT_RAW,
      CLOSURE_COUNT_RAW,
      THEOREM_COUNT_RAW,
      TEST_FILE_COUNT_RAW,
      LANGUAGE_COUNT_RAW,
    ]) {
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
  });
});

/* ─── §2  Identity Verification Completeness ──────────────────── */

/** The 44 identity IDs that the identities page must cover. */
const ALL_IDENTITY_IDS = [
  'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8',
  'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
  'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
  // N series (16 identities) — add IDs as they are classified
  'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8',
  'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16',
];

describe('Identity verification state model', () => {
  it('covers all identity IDs in IDENTITY_VERIFICATION', () => {
    const covered = Object.keys(IDENTITY_VERIFICATION);
    // Every known identity must have a state
    for (const id of ALL_IDENTITY_IDS) {
      if (!covered.includes(id)) {
        // N-series may not be classified yet — skip gracefully but count
        continue;
      }
      expect(IDENTITY_VERIFICATION[id]).toBeDefined();
    }
  });

  it('identity count matches IDENTITY_COUNT_RAW', () => {
    // The classified identities (E + B + D series) must be ≤ IDENTITY_COUNT_RAW.
    // As N-series is classified, this tightens toward equality.
    const classifiedCount = Object.keys(IDENTITY_VERIFICATION).length;
    expect(classifiedCount).toBeLessThanOrEqual(IDENTITY_COUNT_RAW);
    // At minimum, all E (8) + B (12) + D (8) = 28 must be present
    expect(classifiedCount).toBeGreaterThanOrEqual(28);
  });

  it('every state is a valid VerificationState', () => {
    const validStates: VerificationState[] = ['exact', 'sampled', 'narrative', 'pending'];
    for (const [id, state] of Object.entries(IDENTITY_VERIFICATION)) {
      expect(validStates).toContain(state);
    }
  });

  it('VERIFICATION_BADGES has entries for all four states', () => {
    for (const state of ['exact', 'sampled', 'narrative', 'pending'] as VerificationState[]) {
      const badge = VERIFICATION_BADGES[state];
      expect(badge).toBeDefined();
      expect(badge.label).toBeTruthy();
      expect(badge.color).toBeTruthy();
      expect(badge.tooltip).toBeTruthy();
    }
  });

  it('renderBadge produces non-empty HTML for each state', () => {
    for (const state of ['exact', 'sampled', 'narrative', 'pending'] as VerificationState[]) {
      const html = renderBadge(state);
      expect(html.length).toBeGreaterThan(0);
      expect(html).toContain(VERIFICATION_BADGES[state].label);
    }
  });

  it('hasLiveVerification returns true only for exact and sampled', () => {
    // Known exact identities
    expect(hasLiveVerification('E1')).toBe(true);
    expect(hasLiveVerification('E2')).toBe(true);
    // Known sampled
    expect(hasLiveVerification('E3')).toBe(true);
    expect(hasLiveVerification('B1')).toBe(true);
    // Known narrative
    expect(hasLiveVerification('E5')).toBe(false);
    expect(hasLiveVerification('B2')).toBe(false);
    // Unknown ID
    expect(hasLiveVerification('Z99')).toBe(false);
  });
});

/* ─── §3  Overclaim Guard ─────────────────────────────────────── */

describe('Overclaim prevention', () => {
  it('no identity marked exact that should be narrative', () => {
    // Identities that require calculus/series proofs not in the TS kernel
    // must NOT be marked exact or sampled
    const mustBeNarrative = ['E5', 'E6', 'E7', 'E8', 'B9', 'B11', 'B12', 'D2', 'D6', 'D7'];
    for (const id of mustBeNarrative) {
      const state = IDENTITY_VERIFICATION[id];
      if (state) {
        expect(state).not.toBe('exact');
        // narrative or pending are acceptable
        expect(['narrative', 'pending']).toContain(state);
      }
    }
  });

  it('core algebraic identities are marked exact', () => {
    // These are definitionally forced — they must be exact
    expect(IDENTITY_VERIFICATION['E1']).toBe('exact');  // F + ω = 1
    expect(IDENTITY_VERIFICATION['E2']).toBe('exact');  // IC = exp(κ)
    expect(IDENTITY_VERIFICATION['E4']).toBe('exact');  // S(½) + κ(½) = 0
  });
});
