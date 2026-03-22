/**
 * @file seam.h
 * @brief Seam Chain Accumulation — Pure C Core (Tier-0 Protocol)
 *
 * Stable C ABI for incremental seam chain accounting:
 *   - Lemma 20: Δκ composes additively across seam chains (O(1) query)
 *   - Lemma 27: Sublinear residual growth → returning dynamics
 *
 * The seam accumulator is the ledger's arithmetic backbone.
 * C gives us a zero-allocation, embeddable accumulator with
 * a stable ABI for cross-language use.
 *
 * No Tier-1 symbol is redefined.
 */

#ifndef UMCP_C_SEAM_H
#define UMCP_C_SEAM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Seam record (single seam) ─────────────────────────────────── */

typedef struct {
    int    t0;                  /**< Start timestep                   */
    int    t1;                  /**< End timestep                     */
    double kappa_t0;            /**< Log-integrity at t0              */
    double kappa_t1;            /**< Log-integrity at t1              */
    double tau_R;               /**< Return time                      */
    double delta_kappa_ledger;  /**< Observed: κ(t1) − κ(t0)         */
    double delta_kappa_budget;  /**< Expected: R·τ_R − (D_ω + D_C)   */
    double residual;            /**< budget − ledger                  */
    double cumulative_residual; /**< Running Σ|sₖ|                    */
} umcp_seam_record_t;

/* ─── Seam chain accumulator ────────────────────────────────────── */

/**
 * Fixed-capacity seam chain accumulator.
 * No heap allocation — uses a caller-provided buffer.
 */
typedef struct {
    umcp_seam_record_t *buffer;  /**< Caller-allocated ring buffer     */
    size_t capacity;             /**< Max seams in buffer               */
    size_t count;                /**< Number of seams added so far      */
    double total_delta_kappa;    /**< O(1) accumulated Δκ               */
    double cumulative_abs_res;   /**< O(1) accumulated |residual|       */
    double max_residual;         /**< Largest |residual| seen           */
} umcp_seam_chain_t;

/* ─── Chain lifecycle ───────────────────────────────────────────── */

/**
 * Initialize a seam chain with caller-provided storage.
 *
 * @param chain    Chain struct to initialize
 * @param buffer   Pre-allocated buffer for seam records
 * @param capacity Number of elements in buffer
 */
void umcp_seam_init(umcp_seam_chain_t *chain,
                    umcp_seam_record_t *buffer, size_t capacity);

/**
 * Add a seam to the chain with O(1) update (OPT-10, Lemma 20).
 *
 * @param chain    The accumulator
 * @param t0, t1   Seam endpoints
 * @param kappa_t0, kappa_t1  Log-integrity values
 * @param tau_R    Return time
 * @param R        Budget rate (return reward)
 * @param D_omega  Drift penalty
 * @param D_C      Curvature penalty
 * @param record   Output: filled seam record (may be NULL if not needed)
 * @return 0 on success, -1 if chain is full
 */
int umcp_seam_add(umcp_seam_chain_t *chain,
                  int t0, int t1,
                  double kappa_t0, double kappa_t1,
                  double tau_R,
                  double R, double D_omega, double D_C,
                  umcp_seam_record_t *record);

/* ─── Chain metrics (O(1) queries) ──────────────────────────────── */

/**
 * Get the total Δκ accumulated across all seams.
 */
double umcp_seam_total_delta_kappa(const umcp_seam_chain_t *chain);

/**
 * Get the cumulative absolute residual.
 */
double umcp_seam_cumulative_residual(const umcp_seam_chain_t *chain);

/**
 * Get the max single-seam residual.
 */
double umcp_seam_max_residual(const umcp_seam_chain_t *chain);

/**
 * Get the number of seams in the chain.
 */
size_t umcp_seam_count(const umcp_seam_chain_t *chain);

/**
 * Check whether the chain exhibits returning dynamics (Lemma 27).
 * Returns 1 if sublinear residual growth detected, 0 otherwise.
 *
 * @param chain The accumulator
 * @param tol   Tolerance for seam PASS (frozen: 0.005)
 */
int umcp_seam_is_returning(const umcp_seam_chain_t *chain, double tol);

#ifdef __cplusplus
}
#endif

#endif /* UMCP_C_SEAM_H */
