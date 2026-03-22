/**
 * @file trace.h
 * @brief Trace Vector Management — C formalization (Tier-0 Protocol)
 *
 * The trace vector c ∈ [0,1]^n is the fundamental input to the kernel.
 * This module handles:
 *   - Allocation and lifecycle of trace vectors
 *   - Embedding raw measurements into [0,1]^n (normalization)
 *   - Clipping policy enforcement (pre_clip: FACE_POLICY)
 *   - Weight vector validation (w ∈ Δ^n, simplex)
 *   - Tier-1 identity verification on computed outputs
 *
 * The trace is the bridge between Tier-2 (domain closures that choose
 * which quantities become channels) and Tier-1 (the kernel that
 * computes invariants from those channels).
 *
 * Collapsus generativus est; solum quod redit, reale est.
 */

#ifndef UMCP_C_TRACE_H
#define UMCP_C_TRACE_H

#include "types.h"
#include "kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Trace Vector ──────────────────────────────────────────────── */

/**
 * Managed trace vector with metadata.
 *
 * Holds the coordinate array, weight array, and dimensional info.
 * The trace owns its memory — callers must use init/free.
 */
typedef struct {
    double  *c;         /**< Coordinate array c ∈ [ε, 1-ε]^n    */
    double  *w;         /**< Weight array w ∈ Δ^n, sum(w) = 1    */
    size_t   n;         /**< Number of channels                   */
    double   epsilon;   /**< Guard band (from contract)           */
    int      clipped;   /**< 1 if pre_clip has been applied       */
} umcp_trace_t;

/**
 * Initialize a trace vector with given dimensions.
 * Allocates c and w arrays on the heap.
 *
 * @param trace   Output trace (must not be NULL)
 * @param n       Number of channels (must be > 0)
 * @param epsilon Guard band (frozen: 1e-8)
 * @return UMCP_OK on success, UMCP_ERR_ZERO_DIM if n=0
 */
int umcp_trace_init(umcp_trace_t *trace, size_t n, double epsilon);

/**
 * Free a trace vector's internal buffers.
 * Safe to call on a zero-initialized trace.
 */
void umcp_trace_free(umcp_trace_t *trace);

/**
 * Set uniform weights w_i = 1/n for all channels.
 */
void umcp_trace_uniform_weights(umcp_trace_t *trace);

/**
 * Set custom weights and validate they form a probability simplex.
 *
 * @param trace  Target trace
 * @param w      Weight array (n elements, must sum to ~1)
 * @return UMCP_OK if valid, UMCP_ERR_WEIGHT_SUM if |sum(w)-1| > 1e-6
 */
int umcp_trace_set_weights(umcp_trace_t *trace, const double *w);

/**
 * Set channel values from raw data.
 * Applies pre_clip: clamp each value to [epsilon, 1-epsilon].
 *
 * @param trace Target trace
 * @param raw   Raw coordinate values (n elements)
 * @return UMCP_OK on success
 */
int umcp_trace_set_channels(umcp_trace_t *trace, const double *raw);

/* ─── Embedding (Raw Data → [0,1]^n) ───────────────────────────── */

/**
 * Linear normalization: maps [raw_min, raw_max] → [ε, 1-ε].
 *
 * This is the standard Tier-0 embedding for scalar measurements.
 * For log-scale data, use umcp_trace_embed_log().
 *
 * @param trace   Target trace
 * @param raw     Raw measurement array (n elements)
 * @param raw_min Domain minimum for normalization
 * @param raw_max Domain maximum for normalization
 * @return UMCP_OK on success, UMCP_ERR_RANGE if raw_min >= raw_max
 */
int umcp_trace_embed_linear(umcp_trace_t *trace, const double *raw,
                            double raw_min, double raw_max);

/**
 * Logarithmic normalization: maps [log(raw_min), log(raw_max)] → [ε, 1-ε].
 *
 * Used for quantities spanning orders of magnitude (particle masses, etc.).
 *
 * @param trace   Target trace
 * @param raw     Raw measurement array (n elements, must be > 0)
 * @param raw_min Domain minimum (must be > 0)
 * @param raw_max Domain maximum (must be > 0)
 * @return UMCP_OK on success, UMCP_ERR_RANGE if invalid bounds
 */
int umcp_trace_embed_log(umcp_trace_t *trace, const double *raw,
                         double raw_min, double raw_max);

/* ─── Validation ────────────────────────────────────────────────── */

/**
 * Validates the Tier-1 identities on a kernel result:
 *   1. |F + ω - 1| < tol           (duality identity)
 *   2. IC ≤ F + tol                 (integrity bound)
 *   3. |IC - exp(κ)| < tol          (log-integrity relation)
 *
 * @param result    Kernel output to validate
 * @param tol       Tolerance for identity checks (suggested: 1e-9)
 * @return UMCP_CONFORMANT if all identities hold,
 *         UMCP_NONCONFORMANT if any identity is violated
 */
umcp_verdict_t umcp_validate_identities(
    const umcp_kernel_result_t *result, double tol);

#ifdef __cplusplus
}
#endif

#endif /* UMCP_C_TRACE_H */
