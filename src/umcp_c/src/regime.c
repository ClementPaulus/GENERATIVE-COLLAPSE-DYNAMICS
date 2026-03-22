/**
 * @file regime.c
 * @brief Regime Classification implementation
 *
 * Regime labels are derived from frozen gates on Tier-1 invariants.
 * Diagnostica informant, portae decernunt.
 * (Diagnostics inform; gates decide.)
 */

#include "umcp_c/regime.h"
#include <stdlib.h>
#include <math.h>

/* ─── Regime Classification ─────────────────────────────────────── */

umcp_regime_t umcp_classify_regime(
    const umcp_kernel_result_t *result,
    const umcp_regime_thresholds_t *thresholds)
{
    if (!result || !thresholds) return UMCP_REGIME_WATCH;

    double omega = result->omega;
    double F     = result->F;
    double S     = result->S;
    double C     = result->C;
    double IC    = result->IC;

    /* Critical overlay takes precedence */
    if (IC < thresholds->IC_critical_max) {
        return UMCP_REGIME_CRITICAL;
    }

    /* Collapse: ω ≥ 0.30 */
    if (omega >= thresholds->omega_collapse_min) {
        return UMCP_REGIME_COLLAPSE;
    }

    /* Watch: 0.038 ≤ ω < 0.30 */
    if (omega >= thresholds->omega_watch_min) {
        return UMCP_REGIME_WATCH;
    }

    /* Stable requires ALL conditions (conjunctive) */
    if (omega < thresholds->omega_stable_max &&
        F     > thresholds->F_stable_min &&
        S     < thresholds->S_stable_max &&
        C     < thresholds->C_stable_max) {
        return UMCP_REGIME_STABLE;
    }

    /* Default to Watch if not clearly stable */
    return UMCP_REGIME_WATCH;
}

umcp_regime_t umcp_classify_regime_default(const umcp_kernel_result_t *result)
{
    umcp_regime_thresholds_t defaults = {
        .omega_stable_max   = 0.038,
        .F_stable_min       = 0.90,
        .S_stable_max       = 0.15,
        .C_stable_max       = 0.14,
        .omega_watch_min    = 0.038,
        .omega_watch_max    = 0.30,
        .omega_collapse_min = 0.30,
        .IC_critical_max    = 0.30
    };
    return umcp_classify_regime(result, &defaults);
}

int umcp_is_critical(const umcp_kernel_result_t *result,
                     const umcp_regime_thresholds_t *thresholds)
{
    if (!result || !thresholds) return 0;
    return result->IC < thresholds->IC_critical_max ? 1 : 0;
}

/* ─── Fisher-Space Partition ────────────────────────────────────── */

/*
 * Simple LCG for deterministic pseudo-random numbers.
 * Not cryptographic — used only for regime partition estimation.
 */
static uint64_t lcg_state = 0;

static void lcg_seed(uint64_t seed) { lcg_state = seed; }

static double lcg_uniform(void)
{
    /* Numerical Recipes LCG: period 2^64 */
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(lcg_state >> 11) / (double)(1ULL << 53);
}

void umcp_regime_partition(
    size_t n_channels, size_t n_samples,
    const umcp_regime_thresholds_t *thresholds,
    double *pct_stable, double *pct_watch,
    double *pct_collapse, double *pct_critical)
{
    if (!thresholds || !pct_stable || !pct_watch ||
        !pct_collapse || !pct_critical || n_channels == 0 || n_samples == 0) {
        return;
    }

    lcg_seed(42);  /* Deterministic for reproducibility */

    size_t counts[4] = {0, 0, 0, 0};
    double *c = (double *)malloc(n_channels * sizeof(double));
    double *w = (double *)malloc(n_channels * sizeof(double));
    if (!c || !w) {
        free(c); free(w);
        return;
    }

    double wval = 1.0 / (double)n_channels;
    for (size_t j = 0; j < n_channels; ++j) w[j] = wval;

    umcp_kernel_result_t result;
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_channels; ++j) {
            c[j] = lcg_uniform();
            /* Clamp to [ε, 1-ε] */
            if (c[j] < 1e-8) c[j] = 1e-8;
            if (c[j] > 1.0 - 1e-8) c[j] = 1.0 - 1e-8;
        }

        if (umcp_kernel_compute(c, w, n_channels, 1e-8, &result) == UMCP_OK) {
            umcp_regime_t regime = umcp_classify_regime(&result, thresholds);
            counts[regime]++;
        }
    }

    double total = (double)n_samples;
    *pct_stable   = 100.0 * (double)counts[UMCP_REGIME_STABLE]   / total;
    *pct_watch    = 100.0 * (double)counts[UMCP_REGIME_WATCH]    / total;
    *pct_collapse = 100.0 * (double)counts[UMCP_REGIME_COLLAPSE] / total;
    *pct_critical = 100.0 * (double)counts[UMCP_REGIME_CRITICAL] / total;

    free(c);
    free(w);
}
