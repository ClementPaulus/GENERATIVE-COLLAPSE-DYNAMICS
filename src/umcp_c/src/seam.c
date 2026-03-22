/**
 * @file seam.c
 * @brief Seam Chain Accumulation — Pure C Implementation
 *
 * Implements the seam chain accumulator with O(1) incremental updates
 * (OPT-10, Lemma 20) and returning-dynamics detection (OPT-11, Lemma 27).
 *
 * The budget model:
 *   Δκ_budget = R · τ_R − (D_ω + D_C)
 *   Δκ_ledger = κ(t1) − κ(t0)
 *   residual  = budget − ledger
 *
 * Seam chain metrics (total_delta_kappa, cumulative_abs_residual)
 * accumulate in O(1) per seam addition — no re-scanning of history.
 *
 * Tier-0 Protocol: no Tier-1 symbol is redefined.
 */

#include "umcp_c/seam.h"

#include <math.h>
#include <string.h>

/* ─── Lifecycle ─────────────────────────────────────────────────── */

void umcp_seam_init(umcp_seam_chain_t *chain,
                    umcp_seam_record_t *buffer, size_t capacity)
{
    if (!chain) return;
    chain->buffer             = buffer;
    chain->capacity           = capacity;
    chain->count              = 0;
    chain->total_delta_kappa  = 0.0;
    chain->cumulative_abs_res = 0.0;
    chain->max_residual       = 0.0;
}

/* ─── Add seam (O(1) update) ────────────────────────────────────── */

int umcp_seam_add(umcp_seam_chain_t *chain,
                  int t0, int t1,
                  double kappa_t0, double kappa_t1,
                  double tau_R,
                  double R, double D_omega, double D_C,
                  umcp_seam_record_t *record)
{
    if (!chain) return -1;
    if (chain->buffer && chain->count >= chain->capacity) return -1;

    /* Lemma 20: ledger change composes additively */
    double dk_ledger = kappa_t1 - kappa_t0;

    /* Budget model */
    double dk_budget = R * tau_R - (D_omega + D_C);

    /* Residual */
    double residual = dk_budget - dk_ledger;
    double abs_res  = fabs(residual);

    /* O(1) incremental accumulation (OPT-10) */
    chain->total_delta_kappa  += dk_ledger;
    chain->cumulative_abs_res += abs_res;
    if (abs_res > chain->max_residual) {
        chain->max_residual = abs_res;
    }

    /* Store in buffer if available */
    umcp_seam_record_t rec;
    rec.t0                  = t0;
    rec.t1                  = t1;
    rec.kappa_t0            = kappa_t0;
    rec.kappa_t1            = kappa_t1;
    rec.tau_R               = tau_R;
    rec.delta_kappa_ledger  = dk_ledger;
    rec.delta_kappa_budget  = dk_budget;
    rec.residual            = residual;
    rec.cumulative_residual = chain->cumulative_abs_res;

    if (chain->buffer) {
        chain->buffer[chain->count] = rec;
    }
    chain->count++;

    /* Copy record out if requested */
    if (record) {
        *record = rec;
    }

    return 0;
}

/* ─── O(1) queries ──────────────────────────────────────────────── */

double umcp_seam_total_delta_kappa(const umcp_seam_chain_t *chain)
{
    return chain ? chain->total_delta_kappa : 0.0;
}

double umcp_seam_cumulative_residual(const umcp_seam_chain_t *chain)
{
    return chain ? chain->cumulative_abs_res : 0.0;
}

double umcp_seam_max_residual(const umcp_seam_chain_t *chain)
{
    return chain ? chain->max_residual : 0.0;
}

size_t umcp_seam_count(const umcp_seam_chain_t *chain)
{
    return chain ? chain->count : 0;
}

int umcp_seam_is_returning(const umcp_seam_chain_t *chain, double tol)
{
    if (!chain || chain->count == 0) return 0;

    /*
     * Lemma 27: Sublinear residual growth indicates returning dynamics.
     * The mean |residual| should decrease (or stay bounded) as K grows.
     * If mean_res ≤ tol, the chain is returning.
     */
    double mean_res = chain->cumulative_abs_res / (double)chain->count;
    return mean_res <= tol ? 1 : 0;
}
