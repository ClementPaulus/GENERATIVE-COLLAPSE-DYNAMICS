/**
 * @file regime.h
 * @brief Regime Classification — C formalization (Tier-0 Protocol)
 *
 * Regime labels are derived from frozen gates on the Tier-1 invariants.
 * They are never asserted — always computed.
 *
 * The classification uses kernel outputs (F, ω, S, C, IC) and the
 * frozen thresholds from the contract.
 *
 * Collapsus generativus est; solum quod redit, reale est.
 */

#ifndef UMCP_C_REGIME_H
#define UMCP_C_REGIME_H

#include "types.h"
#include "contract.h"
#include "kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Regime Classification ─────────────────────────────────────── */

/**
 * Classify a kernel result into a regime using contract thresholds.
 *
 * Gate logic (conjunctive for Stable):
 *   CRITICAL: IC < 0.30 (overlay — checked first)
 *   COLLAPSE: ω ≥ 0.30
 *   WATCH:    0.038 ≤ ω < 0.30 (or Stable gates not all met)
 *   STABLE:   ω < 0.038 ∧ F > 0.90 ∧ S < 0.15 ∧ C < 0.14
 *
 * @param result     Kernel output (F, ω, S, C, IC)
 * @param thresholds Frozen regime thresholds
 * @return Regime label
 */
umcp_regime_t umcp_classify_regime(
    const umcp_kernel_result_t  *result,
    const umcp_regime_thresholds_t *thresholds);

/**
 * Classify using default thresholds from the contract.
 */
umcp_regime_t umcp_classify_regime_default(const umcp_kernel_result_t *result);

/**
 * Check if a regime is in a critical state (IC < critical threshold).
 * Critical is an overlay — it can accompany any regime.
 */
int umcp_is_critical(const umcp_kernel_result_t *result,
                     const umcp_regime_thresholds_t *thresholds);

/**
 * Compute the Fisher-space partition percentages for diagnostic display.
 * Uniform sampling of [0,1]^n yields approximately:
 *   Stable:   12.5%
 *   Watch:    24.4%
 *   Collapse: 63.1%
 * This demonstrates that stability is rare.
 *
 * @param n_channels Number of channels
 * @param n_samples  Number of random samples
 * @param thresholds Frozen thresholds
 * @param pct_stable  Output: percentage classified Stable
 * @param pct_watch   Output: percentage classified Watch
 * @param pct_collapse Output: percentage classified Collapse
 * @param pct_critical Output: percentage classified Critical
 */
void umcp_regime_partition(
    size_t n_channels, size_t n_samples,
    const umcp_regime_thresholds_t *thresholds,
    double *pct_stable, double *pct_watch,
    double *pct_collapse, double *pct_critical);

#ifdef __cplusplus
}
#endif

#endif /* UMCP_C_REGIME_H */
