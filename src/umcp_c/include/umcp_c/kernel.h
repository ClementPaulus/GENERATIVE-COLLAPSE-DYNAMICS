/**
 * @file kernel.h
 * @brief GCD Kernel — Pure C Core (Tier-0 Protocol)
 *
 * Stable C ABI for the six Tier-1 kernel invariants:
 *   F  = Σ wᵢcᵢ                        (Fidelity)
 *   ω  = 1 − F                          (Drift)
 *   S  = −Σ wᵢ[cᵢ ln cᵢ + (1−cᵢ)ln(1−cᵢ)]  (Bernoulli field entropy)
 *   C  = σ_pop(c) / 0.5                 (Curvature proxy)
 *   κ  = Σ wᵢ ln(cᵢ)                   (Log-integrity)
 *   IC = exp(κ)                          (Integrity composite)
 *
 * This is the lowest layer of the three-layer sandwich:
 *   C (raw math) → C++ (types, validation, bindings) → Python (orchestration)
 *
 * Design principles:
 *   - Pure C99, no C++ dependencies
 *   - No heap allocation in the hot path
 *   - extern "C" ABI — callable from any language via FFI
 *   - Same formulas, same frozen parameters as kernel_optimized.py
 *   - No Tier-1 symbol is redefined — this is Tier-0 implementation
 *
 * Collapsus generativus est; solum quod redit, reale est.
 */

#ifndef UMCP_C_KERNEL_H
#define UMCP_C_KERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Frozen parameter defaults (seam-derived, not chosen) ──────── */
#define UMCP_EPSILON_DEFAULT    1e-8
#define UMCP_HOMOG_TOL_DEFAULT  1e-15

/* ─── Return codes ──────────────────────────────────────────────── */
#define UMCP_OK                 0
#define UMCP_ERR_NULL_PTR      -1
#define UMCP_ERR_ZERO_DIM      -2
#define UMCP_ERR_WEIGHT_SUM    -3
#define UMCP_ERR_RANGE         -4

/* ─── Kernel output structure ───────────────────────────────────── */

/**
 * Result container for kernel computation.
 * All fields are plain doubles — no hidden state, no pointers.
 * Trivially copyable across any ABI boundary.
 */
typedef struct {
    double F;           /**< Fidelity (weighted arithmetic mean)         */
    double omega;       /**< Drift = 1 − F                               */
    double S;           /**< Bernoulli field entropy                      */
    double C;           /**< Curvature proxy (normalized population std)  */
    double kappa;       /**< Log-integrity: κ = Σ wᵢ ln(cᵢ)             */
    double IC;          /**< Integrity composite = exp(κ), IC ≤ F        */
    double delta;       /**< Heterogeneity gap = F − IC (Δ ≥ 0)          */
    int    is_homogeneous; /**< 1 if all channels equal, 0 otherwise     */
} umcp_kernel_result_t;

/* ─── Core computation ──────────────────────────────────────────── */

/**
 * Compute all six kernel invariants from a trace vector.
 *
 * This is the innermost hot path — zero allocation, single pass.
 * Implements OPT-1 (homogeneity detection) and OPT-4 (log-space κ).
 *
 * @param c       Coordinate array, c ∈ [ε, 1−ε]^n (must not be NULL)
 * @param w       Weight array, w ∈ Δ^n, sum(w) ≈ 1 (must not be NULL)
 * @param n       Number of channels (must be > 0)
 * @param epsilon Guard band for log stability (frozen: 1e-8)
 * @param out     Output structure (must not be NULL)
 * @return UMCP_OK on success, negative error code on failure
 */
int umcp_kernel_compute(const double *c, const double *w, size_t n,
                        double epsilon, umcp_kernel_result_t *out);

/**
 * Batch compute kernel invariants over T trace rows.
 *
 * Processes a contiguous T×n row-major matrix of coordinates
 * with shared weights.
 *
 * @param trace   Row-major T×n coordinate matrix
 * @param w       Weight array (n elements), shared across rows
 * @param T       Number of rows (timesteps)
 * @param n       Number of channels per row
 * @param epsilon Guard band
 * @param out     Output array (T elements, caller-allocated)
 * @return UMCP_OK on success, negative error code on failure
 */
int umcp_kernel_batch(const double *trace, const double *w,
                      size_t T, size_t n, double epsilon,
                      umcp_kernel_result_t *out);

/* ─── Scalar utilities (inline candidates) ──────────────────────── */

/**
 * Bernoulli entropy: h(c) = −c ln(c) − (1−c) ln(1−c).
 * The unique entropy of the collapse field.
 * (Shannon entropy is the degenerate limit when c ∈ {0,1}.)
 */
double umcp_bernoulli_entropy(double c);

/**
 * Clamp a value to [epsilon, 1 - epsilon].
 * Guard band prevents log(0) in κ computation.
 */
double umcp_clamp(double v, double epsilon);

#ifdef __cplusplus
}
#endif

#endif /* UMCP_C_KERNEL_H */
