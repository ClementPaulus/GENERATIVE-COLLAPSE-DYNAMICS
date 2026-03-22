/**
 * @file kernel.c
 * @brief GCD Kernel — Pure C Implementation
 *
 * The innermost computation layer. Every operation is a weighted
 * reduction over a double array — perfect for SIMD and cache-local
 * processing. No allocation, no branches in the hot path (aside
 * from the homogeneity fast-path OPT-1).
 *
 * Implements the four primitive equations:
 *   F = Σ wᵢcᵢ            (Definition 4)
 *   κ = Σ wᵢ ln(cᵢ)       (Lemma 2)
 *   S = −Σ wᵢ h(cᵢ)       (Definition 6)
 *   C = σ_pop(c) / 0.5     (Definition 7)
 *
 * And two derived values:
 *   ω  = 1 − F             (Definition 5)
 *   IC = exp(κ)            (Lemma 4)
 *
 * Tier-0 Protocol: no Tier-1 symbol is redefined.
 */

#include "umcp_c/kernel.h"

#include <math.h>
#include <string.h>

/* ─── Scalar utilities ──────────────────────────────────────────── */

double umcp_bernoulli_entropy(double c)
{
    if (c <= 0.0 || c >= 1.0) return 0.0;
    return -(c * log(c) + (1.0 - c) * log(1.0 - c));
}

double umcp_clamp(double v, double epsilon)
{
    if (v < epsilon) return epsilon;
    if (v > 1.0 - epsilon) return 1.0 - epsilon;
    return v;
}

/* ─── Homogeneity detection (OPT-1, Lemma 10) ──────────────────── */

static int is_homogeneous(const double *c, size_t n)
{
    double c0 = c[0];
    for (size_t i = 1; i < n; ++i) {
        double diff = c[i] - c0;
        if (diff > UMCP_HOMOG_TOL_DEFAULT || diff < -UMCP_HOMOG_TOL_DEFAULT) {
            return 0;
        }
    }
    return 1;
}

/* ─── Homogeneous fast path ─────────────────────────────────────── */

static void compute_homogeneous(double c_val, umcp_kernel_result_t *out)
{
    out->F     = c_val;
    out->omega = 1.0 - c_val;
    out->kappa = log(c_val);
    out->IC    = c_val;     /* Geometric mean = arithmetic mean (Lemma 4 equality) */
    out->C     = 0.0;       /* No dispersion (Lemma 10) */
    out->S     = umcp_bernoulli_entropy(c_val);
    out->delta = 0.0;       /* No heterogeneity gap */
    out->is_homogeneous = 1;
}

/* ─── Full heterogeneous computation (single pass) ──────────────── */

static void compute_heterogeneous(const double *c, const double *w,
                                  size_t n, umcp_kernel_result_t *out)
{
    double F     = 0.0;
    double kappa = 0.0;
    double S     = 0.0;
    double sum_c = 0.0;
    double sum_c2 = 0.0;

    /*
     * Single pass: compute F, κ, S, and accumulate for C.
     * This loop is the critical hot path — four reductions fused
     * into one pass for cache locality.
     */
    for (size_t i = 0; i < n; ++i) {
        double ci = c[i];
        double wi = w[i];

        /* Fidelity: F = Σ wᵢcᵢ (Definition 4) */
        F += wi * ci;

        /* Log-integrity: κ = Σ wᵢ ln(cᵢ) (OPT-4, Lemma 2) */
        kappa += wi * log(ci);

        /* Bernoulli field entropy (Definition 6) */
        if (wi > 0.0) {
            S += wi * umcp_bernoulli_entropy(ci);
        }

        /* Accumulate for population std (Definition 7) */
        sum_c  += ci;
        sum_c2 += ci * ci;
    }

    double n_d = (double)n;
    double mean_c = sum_c / n_d;
    double var_c  = sum_c2 / n_d - mean_c * mean_c;

    /* Guard against floating-point rounding producing tiny negatives */
    if (var_c < 0.0) var_c = 0.0;

    out->F     = F;
    out->omega = 1.0 - F;
    out->kappa = kappa;
    out->IC    = exp(kappa);
    out->C     = sqrt(var_c) / 0.5;   /* Normalized std (Definition 7) */
    out->S     = S;
    out->delta = F - out->IC;          /* Heterogeneity gap (Lemma 34) */
    out->is_homogeneous = 0;
}

/* ─── Public API ────────────────────────────────────────────────── */

int umcp_kernel_compute(const double *c, const double *w, size_t n,
                        double epsilon, umcp_kernel_result_t *out)
{
    /* Input validation */
    if (!c || !w || !out) return UMCP_ERR_NULL_PTR;
    if (n == 0) return UMCP_ERR_ZERO_DIM;

    (void)epsilon;  /* Used by caller for clamping before this function */

    /* OPT-1: Homogeneity detection (Lemma 10) */
    if (is_homogeneous(c, n)) {
        compute_homogeneous(c[0], out);
        return UMCP_OK;
    }

    compute_heterogeneous(c, w, n, out);
    return UMCP_OK;
}

int umcp_kernel_batch(const double *trace, const double *w,
                      size_t T, size_t n, double epsilon,
                      umcp_kernel_result_t *out)
{
    if (!trace || !w || !out) return UMCP_ERR_NULL_PTR;
    if (T == 0 || n == 0) return UMCP_ERR_ZERO_DIM;

    for (size_t t = 0; t < T; ++t) {
        int rc = umcp_kernel_compute(trace + t * n, w, n, epsilon, &out[t]);
        if (rc != UMCP_OK) return rc;
    }

    return UMCP_OK;
}
