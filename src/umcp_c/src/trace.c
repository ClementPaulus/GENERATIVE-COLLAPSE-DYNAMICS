/**
 * @file trace.c
 * @brief Trace Vector Management implementation
 *
 * The trace vector is the bridge between Tier-2 domain closures
 * (which choose what to measure) and Tier-1 kernel computation
 * (which computes the invariants).
 */

#include "umcp_c/trace.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── Lifecycle ─────────────────────────────────────────────────── */

int umcp_trace_init(umcp_trace_t *trace, size_t n, double epsilon)
{
    if (!trace) return UMCP_ERR_NULL_PTR;
    if (n == 0) return UMCP_ERR_ZERO_DIM;

    trace->c = (double *)calloc(n, sizeof(double));
    trace->w = (double *)calloc(n, sizeof(double));
    if (!trace->c || !trace->w) {
        free(trace->c);
        free(trace->w);
        trace->c = NULL;
        trace->w = NULL;
        return UMCP_ERR_NULL_PTR;
    }

    trace->n       = n;
    trace->epsilon = epsilon;
    trace->clipped = 0;

    /* Default to uniform weights */
    umcp_trace_uniform_weights(trace);

    return UMCP_OK;
}

void umcp_trace_free(umcp_trace_t *trace)
{
    if (!trace) return;
    free(trace->c);
    free(trace->w);
    trace->c = NULL;
    trace->w = NULL;
    trace->n = 0;
}

/* ─── Weight Management ─────────────────────────────────────────── */

void umcp_trace_uniform_weights(umcp_trace_t *trace)
{
    if (!trace || !trace->w || trace->n == 0) return;
    double val = 1.0 / (double)trace->n;
    for (size_t i = 0; i < trace->n; ++i) {
        trace->w[i] = val;
    }
}

int umcp_trace_set_weights(umcp_trace_t *trace, const double *w)
{
    if (!trace || !w) return UMCP_ERR_NULL_PTR;

    double sum = 0.0;
    for (size_t i = 0; i < trace->n; ++i) {
        if (w[i] < 0.0) return UMCP_ERR_RANGE;
        sum += w[i];
    }

    if (fabs(sum - 1.0) > 1e-6) return UMCP_ERR_WEIGHT_SUM;

    memcpy(trace->w, w, trace->n * sizeof(double));
    return UMCP_OK;
}

/* ─── Channel Setting ───────────────────────────────────────────── */

int umcp_trace_set_channels(umcp_trace_t *trace, const double *raw)
{
    if (!trace || !raw) return UMCP_ERR_NULL_PTR;

    double eps = trace->epsilon;
    for (size_t i = 0; i < trace->n; ++i) {
        double v = raw[i];
        /* Pre-clip: clamp to [ε, 1-ε] */
        if (v < eps) v = eps;
        if (v > 1.0 - eps) v = 1.0 - eps;
        trace->c[i] = v;
    }
    trace->clipped = 1;

    return UMCP_OK;
}

/* ─── Embedding: Linear Normalization ───────────────────────────── */

int umcp_trace_embed_linear(umcp_trace_t *trace, const double *raw,
                            double raw_min, double raw_max)
{
    if (!trace || !raw) return UMCP_ERR_NULL_PTR;

    double range = raw_max - raw_min;
    if (range <= 0.0) return UMCP_ERR_RANGE;

    double eps = trace->epsilon;
    double target_min = eps;
    double target_max = 1.0 - eps;
    double target_range = target_max - target_min;

    for (size_t i = 0; i < trace->n; ++i) {
        /* Linear map: [raw_min, raw_max] → [ε, 1-ε] */
        double normalized = target_min + (raw[i] - raw_min) / range * target_range;

        /* Clamp to [ε, 1-ε] */
        if (normalized < eps) normalized = eps;
        if (normalized > 1.0 - eps) normalized = 1.0 - eps;

        trace->c[i] = normalized;
    }
    trace->clipped = 1;

    return UMCP_OK;
}

/* ─── Embedding: Logarithmic Normalization ──────────────────────── */

int umcp_trace_embed_log(umcp_trace_t *trace, const double *raw,
                         double raw_min, double raw_max)
{
    if (!trace || !raw) return UMCP_ERR_NULL_PTR;
    if (raw_min <= 0.0 || raw_max <= 0.0) return UMCP_ERR_RANGE;
    if (raw_min >= raw_max) return UMCP_ERR_RANGE;

    double log_min = log(raw_min);
    double log_max = log(raw_max);
    double log_range = log_max - log_min;

    if (log_range <= 0.0) return UMCP_ERR_RANGE;

    double eps = trace->epsilon;
    double target_min = eps;
    double target_max = 1.0 - eps;
    double target_range = target_max - target_min;

    for (size_t i = 0; i < trace->n; ++i) {
        double val = raw[i];
        if (val <= 0.0) val = raw_min;  /* Floor to domain min */

        double log_val = log(val);
        double normalized = target_min + (log_val - log_min) / log_range * target_range;

        /* Clamp to [ε, 1-ε] */
        if (normalized < eps) normalized = eps;
        if (normalized > 1.0 - eps) normalized = 1.0 - eps;

        trace->c[i] = normalized;
    }
    trace->clipped = 1;

    return UMCP_OK;
}

/* ─── Identity Validation ───────────────────────────────────────── */

umcp_verdict_t umcp_validate_identities(
    const umcp_kernel_result_t *result, double tol)
{
    if (!result) return UMCP_NON_EVALUABLE;

    /* Identity 1: F + ω = 1 (duality) */
    double duality_err = fabs(result->F + result->omega - 1.0);
    if (duality_err > tol) return UMCP_NONCONFORMANT;

    /* Identity 2: IC ≤ F (integrity bound) */
    if (result->IC > result->F + tol) return UMCP_NONCONFORMANT;

    /* Identity 3: IC = exp(κ) (log-integrity relation) */
    double expected_IC = exp(result->kappa);
    double logint_err = fabs(result->IC - expected_IC);
    if (logint_err > 1e-6) return UMCP_NONCONFORMANT;

    return UMCP_CONFORMANT;
}
