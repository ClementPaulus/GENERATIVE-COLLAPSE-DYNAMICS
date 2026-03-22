/**
 * @file contract.h
 * @brief Frozen Contract — C formalization (Tier-0 Protocol)
 *
 * The frozen contract defines the measurement constitution:
 * parameters that are consistent across the seam — same rules
 * on both sides of every collapse-return boundary.
 *
 * These values are seam-derived, not chosen by convention.
 * Trans suturam congelatum — frozen across the seam.
 *
 * Collapsus generativus est; solum quod redit, reale est.
 */

#ifndef UMCP_C_CONTRACT_H
#define UMCP_C_CONTRACT_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Frozen Parameters (seam-derived, not prescribed) ──────────── */

/** Guard band — pole at ω=1 does not affect measurements to machine precision */
#define UMCP_EPSILON        1e-8

/** Drift cost exponent — unique integer where ω_trap is a Cardano root of x³+x−1=0 */
#define UMCP_P_EXPONENT     3

/** Curvature cost coefficient (unit coupling) */
#define UMCP_ALPHA          1.0

/** Auxiliary coefficient */
#define UMCP_LAMBDA         0.2

/** Seam residual tolerance — width where IC ≤ F holds at 100% across all 20 domains */
#define UMCP_TOL_SEAM       0.005

/** Normalization domain bounds */
#define UMCP_DOMAIN_MIN     0.0
#define UMCP_DOMAIN_MAX     1.0

/* ─── Derived Structural Constants ──────────────────────────────── */

/** c* ≈ 0.7822 — logistic self-dual fixed point (maximizes S + κ per channel) */
#define UMCP_C_STAR         0.7822

/** ω_trap ≈ 0.6823 — trapping threshold where Γ(ω_trap) = α */
#define UMCP_OMEGA_TRAP     0.6823

/** c_trap ≈ 0.3177 — channel-space trapping threshold */
#define UMCP_C_TRAP         0.3177

/* ─── Regime Thresholds ─────────────────────────────────────────── */

/**
 * Regime gate thresholds — frozen per run.
 * These translate continuous Tier-1 invariants into discrete regime labels.
 *
 * Stable is conjunctive: stability requires ALL invariants clean simultaneously.
 * Critical is an overlay, not a regime.
 */
typedef struct {
    double omega_stable_max;     /**< ω < 0.038 for Stable      */
    double F_stable_min;         /**< F > 0.90 for Stable        */
    double S_stable_max;         /**< S < 0.15 for Stable        */
    double C_stable_max;         /**< C < 0.14 for Stable        */
    double omega_watch_min;      /**< ω ≥ 0.038 for Watch        */
    double omega_watch_max;      /**< ω < 0.30 for Watch         */
    double omega_collapse_min;   /**< ω ≥ 0.30 for Collapse      */
    double IC_critical_max;      /**< IC < 0.30 for Critical      */
} umcp_regime_thresholds_t;

/**
 * The complete frozen contract — all parameters for one run.
 *
 * If any parameter changes, a new contract variant must be declared
 * because comparability has changed.
 */
typedef struct {
    double                   epsilon;
    int                      p_exponent;
    double                   alpha;
    double                   lambda;
    double                   tol_seam;
    double                   domain_min;
    double                   domain_max;
    umcp_regime_thresholds_t thresholds;
} umcp_contract_t;

/* ─── Contract Operations ───────────────────────────────────────── */

/**
 * Initialize a contract with default frozen parameters.
 * All values match frozen_contract.py exactly.
 */
void umcp_contract_default(umcp_contract_t *contract);

/**
 * Validate that a contract has internally consistent parameters.
 *
 * Checks:
 *   - epsilon > 0
 *   - p_exponent >= 1
 *   - alpha >= 0
 *   - tol_seam > 0
 *   - domain_min < domain_max
 *   - threshold ordering is consistent
 *
 * @return UMCP_OK if valid, UMCP_ERR_RANGE if any parameter is invalid
 */
int umcp_contract_validate(const umcp_contract_t *contract);

/**
 * Compare two contracts for equality (same frozen parameters).
 * Used to verify trans suturam congelatum — same rules both sides.
 *
 * @return 1 if contracts are equal, 0 if they differ
 */
int umcp_contract_equal(const umcp_contract_t *a, const umcp_contract_t *b);

/* ─── Cost Closures (Tier-0 protocol) ───────────────────────────── */

/**
 * Drift cost Γ(ω) = ω^p / (1 − ω + ε)
 */
double umcp_gamma_omega(double omega, int p, double epsilon);

/**
 * Curvature cost D_C = α · C
 */
double umcp_cost_curvature(double C, double alpha);

/**
 * Budget identity Δκ_budget = R·τ_R − (D_ω + D_C)
 */
double umcp_budget_delta_kappa(double R, double tau_R,
                                double D_omega, double D_C);

/**
 * Seam residual s = Δκ_budget − Δκ_ledger
 */
double umcp_seam_residual(double delta_kappa_budget,
                           double delta_kappa_ledger);

/**
 * Check all three PASS conditions for a seam weld.
 *
 * PASS requires ALL of:
 *   1. |s| ≤ tol_seam (budget closed)
 *   2. τ_R is finite (something returned)
 *   3. |I_ratio − exp(Δκ)| < tol_exp (exponential identity held)
 *
 * @param residual    Seam residual s
 * @param tau_R       Return time
 * @param I_ratio     Integrity ratio I_post/I_pre
 * @param delta_kappa Ledger Δκ = κ(t1) − κ(t0)
 * @param tol_seam    Seam tolerance (frozen: 0.005)
 * @param tol_exp     Exponential identity tolerance (1e-6)
 * @param fail_reason Buffer for failure message (can be NULL)
 * @param reason_len  Size of fail_reason buffer
 * @return UMCP_SEAM_PASS or UMCP_SEAM_FAIL
 */
umcp_seam_status_t umcp_check_seam_pass(
    double residual, double tau_R, double I_ratio,
    double delta_kappa, double tol_seam, double tol_exp,
    char *fail_reason, size_t reason_len);

#ifdef __cplusplus
}
#endif

#endif /* UMCP_C_CONTRACT_H */
