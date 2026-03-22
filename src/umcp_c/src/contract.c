/**
 * @file contract.c
 * @brief Frozen Contract implementation
 *
 * Trans suturam congelatum — frozen across the seam.
 * Same rules on both sides of every collapse-return boundary.
 */

#include "umcp_c/contract.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ─── Default Contract ──────────────────────────────────────────── */

void umcp_contract_default(umcp_contract_t *contract)
{
    if (!contract) return;

    contract->epsilon     = UMCP_EPSILON;
    contract->p_exponent  = UMCP_P_EXPONENT;
    contract->alpha       = UMCP_ALPHA;
    contract->lambda      = UMCP_LAMBDA;
    contract->tol_seam    = UMCP_TOL_SEAM;
    contract->domain_min  = UMCP_DOMAIN_MIN;
    contract->domain_max  = UMCP_DOMAIN_MAX;

    /* Default regime thresholds from The Episteme of Return */
    contract->thresholds.omega_stable_max   = 0.038;
    contract->thresholds.F_stable_min       = 0.90;
    contract->thresholds.S_stable_max       = 0.15;
    contract->thresholds.C_stable_max       = 0.14;
    contract->thresholds.omega_watch_min    = 0.038;
    contract->thresholds.omega_watch_max    = 0.30;
    contract->thresholds.omega_collapse_min = 0.30;
    contract->thresholds.IC_critical_max    = 0.30;
}

/* ─── Contract Validation ───────────────────────────────────────── */

int umcp_contract_validate(const umcp_contract_t *contract)
{
    if (!contract) return UMCP_ERR_NULL_PTR;

    if (contract->epsilon <= 0.0)     return UMCP_ERR_RANGE;
    if (contract->p_exponent < 1)     return UMCP_ERR_RANGE;
    if (contract->alpha < 0.0)        return UMCP_ERR_RANGE;
    if (contract->tol_seam <= 0.0)    return UMCP_ERR_RANGE;
    if (contract->domain_min >= contract->domain_max) return UMCP_ERR_RANGE;

    /* Threshold ordering */
    const umcp_regime_thresholds_t *t = &contract->thresholds;
    if (t->omega_stable_max < 0.0)    return UMCP_ERR_RANGE;
    if (t->omega_watch_min < t->omega_stable_max - 1e-12) return UMCP_ERR_RANGE;
    if (t->omega_collapse_min < t->omega_watch_min - 1e-12) return UMCP_ERR_RANGE;

    return UMCP_OK;
}

/* ─── Contract Equality ─────────────────────────────────────────── */

int umcp_contract_equal(const umcp_contract_t *a, const umcp_contract_t *b)
{
    if (!a || !b) return 0;

    const double tol = 1e-15;

    if (fabs(a->epsilon - b->epsilon) > tol) return 0;
    if (a->p_exponent != b->p_exponent)      return 0;
    if (fabs(a->alpha - b->alpha) > tol)     return 0;
    if (fabs(a->lambda - b->lambda) > tol)   return 0;
    if (fabs(a->tol_seam - b->tol_seam) > tol) return 0;
    if (fabs(a->domain_min - b->domain_min) > tol) return 0;
    if (fabs(a->domain_max - b->domain_max) > tol) return 0;

    /* Compare thresholds */
    const umcp_regime_thresholds_t *ta = &a->thresholds;
    const umcp_regime_thresholds_t *tb = &b->thresholds;

    if (fabs(ta->omega_stable_max   - tb->omega_stable_max)   > tol) return 0;
    if (fabs(ta->F_stable_min       - tb->F_stable_min)       > tol) return 0;
    if (fabs(ta->S_stable_max       - tb->S_stable_max)       > tol) return 0;
    if (fabs(ta->C_stable_max       - tb->C_stable_max)       > tol) return 0;
    if (fabs(ta->omega_collapse_min - tb->omega_collapse_min)  > tol) return 0;
    if (fabs(ta->IC_critical_max    - tb->IC_critical_max)     > tol) return 0;

    return 1;
}

/* ─── Cost Closures ─────────────────────────────────────────────── */

double umcp_gamma_omega(double omega, int p, double epsilon)
{
    /* Γ(ω) = ω^p / (1 − ω + ε)  — drift cost closure */
    double num = 1.0;
    for (int i = 0; i < p; ++i) num *= omega;
    return num / (1.0 - omega + epsilon);
}

double umcp_cost_curvature(double C, double alpha)
{
    return alpha * C;
}

double umcp_budget_delta_kappa(double R, double tau_R,
                                double D_omega, double D_C)
{
    /* Δκ_budget = R·τ_R − (D_ω + D_C) */
    if (umcp_is_inf_rec(tau_R)) return 0.0;  /* No return → no credit */
    return R * tau_R - (D_omega + D_C);
}

double umcp_seam_residual(double delta_kappa_budget,
                           double delta_kappa_ledger)
{
    return delta_kappa_budget - delta_kappa_ledger;
}

/* ─── Seam PASS Check ───────────────────────────────────────────── */

umcp_seam_status_t umcp_check_seam_pass(
    double residual, double tau_R, double I_ratio,
    double delta_kappa, double tol_seam, double tol_exp,
    char *fail_reason, size_t reason_len)
{
    int pass = 1;

    /* Condition 1: |s| ≤ tol_seam */
    if (fabs(residual) > tol_seam) {
        pass = 0;
        if (fail_reason && reason_len > 0) {
            snprintf(fail_reason, reason_len,
                     "|s|=%.6f > tol=%.4f", fabs(residual), tol_seam);
        }
        return UMCP_SEAM_FAIL;
    }

    /* Condition 2: τ_R is finite (not INF_REC) */
    if (umcp_is_inf_rec(tau_R)) {
        pass = 0;
        if (fail_reason && reason_len > 0) {
            snprintf(fail_reason, reason_len, "tau_R=INF_REC (no return)");
        }
        return UMCP_SEAM_FAIL;
    }

    /* Condition 3: |I_ratio − exp(Δκ)| < tol_exp */
    double exp_dk = exp(delta_kappa);
    double identity_err = fabs(I_ratio - exp_dk);
    if (identity_err >= tol_exp) {
        pass = 0;
        if (fail_reason && reason_len > 0) {
            snprintf(fail_reason, reason_len,
                     "|I_ratio-exp(dk)|=%.2e >= %.2e", identity_err, tol_exp);
        }
        return UMCP_SEAM_FAIL;
    }

    (void)pass;  /* Used through early returns above */
    if (fail_reason && reason_len > 0) fail_reason[0] = '\0';
    return UMCP_SEAM_PASS;
}
