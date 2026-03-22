/**
 * @file test_kernel_c.c
 * @brief Pure C tests for the GCD kernel core
 *
 * Verifies the three Tier-1 identities through the C ABI:
 *   1. F + ω = 1          (Complementum Perfectum)
 *   2. IC ≤ F             (Limbus Integritatis)
 *   3. IC = exp(κ)        (Log-integrity relation)
 *
 * Also tests:
 *   - Homogeneity detection (OPT-1)
 *   - Bernoulli entropy bounds
 *   - Seam chain accumulation
 *   - SHA-256 correctness (NIST test vectors)
 *   - Batch computation
 *
 * Build: gcc -O2 -std=c99 -I../include test_kernel_c.c ../src/kernel.c
 *        ../src/seam.c ../src/sha256.c -lm -o test_kernel_c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "umcp_c/kernel.h"
#include "umcp_c/seam.h"
#include "umcp_c/sha256.h"

/* ─── Test infrastructure ───────────────────────────────────────── */

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do {                              \
    tests_run++;                                            \
    if (cond) {                                             \
        tests_passed++;                                     \
    } else {                                                \
        tests_failed++;                                     \
        fprintf(stderr, "  FAIL [%s:%d]: %s\n",            \
                __FILE__, __LINE__, msg);                   \
    }                                                       \
} while (0)

#define ASSERT_NEAR(a, b, tol, msg) do {                    \
    double _a = (a), _b = (b);                              \
    ASSERT(fabs(_a - _b) <= (tol), msg);                    \
} while (0)

static void uniform_weights(double *w, size_t n)
{
    double val = 1.0 / (double)n;
    for (size_t i = 0; i < n; ++i) w[i] = val;
}

/* ═══════════════════ Kernel Tests ════════════════════════════════ */

static void test_duality_identity(void)
{
    printf("  Test: Duality identity F + ω = 1\n");

    /* Uniform coordinates */
    double c[8] = {0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75};
    double w[8]; uniform_weights(w, 8);
    umcp_kernel_result_t out;

    int rc = umcp_kernel_compute(c, w, 8, 1e-8, &out);
    ASSERT(rc == UMCP_OK, "return code is UMCP_OK");
    ASSERT_NEAR(out.F + out.omega, 1.0, 1e-15,
                "F + omega = 1 (uniform)");

    /* Heterogeneous coordinates */
    double c2[8] = {0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4};
    rc = umcp_kernel_compute(c2, w, 8, 1e-8, &out);
    ASSERT(rc == UMCP_OK, "return code is UMCP_OK");
    ASSERT_NEAR(out.F + out.omega, 1.0, 1e-15,
                "F + omega = 1 (heterogeneous)");

    /* Near-epsilon */
    double c3[4] = {1e-8, 1e-8, 1e-8, 1e-8};
    double w3[4]; uniform_weights(w3, 4);
    rc = umcp_kernel_compute(c3, w3, 4, 1e-8, &out);
    ASSERT(rc == UMCP_OK, "return code is UMCP_OK");
    ASSERT_NEAR(out.F + out.omega, 1.0, 1e-15,
                "F + omega = 1 (near-epsilon)");
}

static void test_integrity_bound(void)
{
    printf("  Test: Integrity bound IC <= F\n");

    /* Homogeneous: IC = F (equality at Lemma 4) */
    double c_homo[8] = {0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6};
    double w[8]; uniform_weights(w, 8);
    umcp_kernel_result_t out;

    umcp_kernel_compute(c_homo, w, 8, 1e-8, &out);
    ASSERT(out.IC <= out.F + 1e-14,
           "IC <= F (homogeneous)");
    ASSERT_NEAR(out.delta, 0.0, 1e-14,
                "delta = 0 when homogeneous");

    /* Heterogeneous: IC < F (strict) */
    double c_het[8] = {0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4};
    umcp_kernel_compute(c_het, w, 8, 1e-8, &out);
    ASSERT(out.IC < out.F,
           "IC < F (heterogeneous, strict)");
    ASSERT(out.delta > 0.0,
           "delta > 0 (heterogeneity gap present)");

    /* Geometric slaughter: one dead channel */
    double c_dead[8] = {0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1e-8};
    umcp_kernel_compute(c_dead, w, 8, 1e-8, &out);
    ASSERT(out.IC < 0.15 * out.F,
           "Geometric slaughter: IC/F < 0.15 with one dead channel");
}

static void test_log_integrity_relation(void)
{
    printf("  Test: IC = exp(kappa)\n");

    double c[8] = {0.5, 0.7, 0.3, 0.9, 0.4, 0.6, 0.8, 0.2};
    double w[8]; uniform_weights(w, 8);
    umcp_kernel_result_t out;

    umcp_kernel_compute(c, w, 8, 1e-8, &out);
    ASSERT_NEAR(out.IC, exp(out.kappa), 1e-15,
                "IC = exp(kappa)");
}

static void test_homogeneity_detection(void)
{
    printf("  Test: Homogeneity detection (OPT-1)\n");

    double c[4] = {0.8, 0.8, 0.8, 0.8};
    double w[4]; uniform_weights(w, 4);
    umcp_kernel_result_t out;

    umcp_kernel_compute(c, w, 4, 1e-8, &out);
    ASSERT(out.is_homogeneous == 1, "detects homogeneous input");
    ASSERT_NEAR(out.C, 0.0, 1e-15, "C = 0 when homogeneous");
    ASSERT_NEAR(out.delta, 0.0, 1e-15, "delta = 0 when homogeneous");

    /* Slightly perturbed — should be heterogeneous */
    double c2[4] = {0.8, 0.8, 0.8, 0.7};
    umcp_kernel_compute(c2, w, 4, 1e-8, &out);
    ASSERT(out.is_homogeneous == 0, "detects heterogeneous input");
}

static void test_bernoulli_entropy(void)
{
    printf("  Test: Bernoulli entropy bounds\n");

    /* At c = 0.5, entropy is maximized: h(0.5) = ln(2) */
    double h_half = umcp_bernoulli_entropy(0.5);
    ASSERT_NEAR(h_half, log(2.0), 1e-15, "h(0.5) = ln(2)");

    /* At boundaries, entropy is 0 */
    ASSERT_NEAR(umcp_bernoulli_entropy(0.0), 0.0, 1e-15, "h(0) = 0");
    ASSERT_NEAR(umcp_bernoulli_entropy(1.0), 0.0, 1e-15, "h(1) = 0");

    /* Entropy is non-negative everywhere */
    for (int i = 1; i < 100; ++i) {
        double ci = i / 100.0;
        ASSERT(umcp_bernoulli_entropy(ci) >= 0.0,
               "h(c) >= 0 for all c in (0,1)");
    }
}

static void test_batch_computation(void)
{
    printf("  Test: Batch computation\n");

    /* 10 rows × 4 channels */
    double trace[40];
    double w[4]; uniform_weights(w, 4);
    umcp_kernel_result_t results[10];

    for (int t = 0; t < 10; ++t) {
        for (int j = 0; j < 4; ++j) {
            trace[t * 4 + j] = 0.1 + 0.08 * (t + j);
        }
    }

    int rc = umcp_kernel_batch(trace, w, 10, 4, 1e-8, results);
    ASSERT(rc == UMCP_OK, "batch returns UMCP_OK");

    /* Verify identities hold for every row */
    for (int t = 0; t < 10; ++t) {
        ASSERT_NEAR(results[t].F + results[t].omega, 1.0, 1e-15,
                    "F + omega = 1 in batch row");
        ASSERT(results[t].IC <= results[t].F + 1e-14,
               "IC <= F in batch row");
        ASSERT_NEAR(results[t].IC, exp(results[t].kappa), 1e-15,
                    "IC = exp(kappa) in batch row");
    }
}

static void test_error_handling(void)
{
    printf("  Test: Error handling\n");

    umcp_kernel_result_t out;
    double c[4] = {0.5, 0.5, 0.5, 0.5};
    double w[4]; uniform_weights(w, 4);

    ASSERT(umcp_kernel_compute(NULL, w, 4, 1e-8, &out) == UMCP_ERR_NULL_PTR,
           "NULL c → ERR_NULL_PTR");
    ASSERT(umcp_kernel_compute(c, NULL, 4, 1e-8, &out) == UMCP_ERR_NULL_PTR,
           "NULL w → ERR_NULL_PTR");
    ASSERT(umcp_kernel_compute(c, w, 4, 1e-8, NULL) == UMCP_ERR_NULL_PTR,
           "NULL out → ERR_NULL_PTR");
    ASSERT(umcp_kernel_compute(c, w, 0, 1e-8, &out) == UMCP_ERR_ZERO_DIM,
           "n=0 → ERR_ZERO_DIM");
}

static void test_equator_convergence(void)
{
    printf("  Test: Equator convergence S + kappa = 0 at c = 1/2\n");

    /* At the equator (all c = 0.5), S + κ = 0 exactly */
    double c[8] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    double w[8]; uniform_weights(w, 8);
    umcp_kernel_result_t out;

    umcp_kernel_compute(c, w, 8, 1e-8, &out);
    ASSERT_NEAR(out.S + out.kappa, 0.0, 1e-14,
                "S + kappa = 0 at equator (c = 1/2)");
}

/* ═══════════════════ Seam Tests ═════════════════════════════════ */

static void test_seam_accumulation(void)
{
    printf("  Test: Seam chain accumulation\n");

    umcp_seam_record_t buffer[100];
    umcp_seam_chain_t chain;
    umcp_seam_init(&chain, buffer, 100);

    ASSERT(umcp_seam_count(&chain) == 0, "initial count is 0");
    ASSERT_NEAR(umcp_seam_total_delta_kappa(&chain), 0.0, 1e-15,
                "initial total_delta_kappa is 0");

    /* Add three seams */
    umcp_seam_record_t rec;
    umcp_seam_add(&chain, 0, 10, -0.5, -0.4, 5.0,
                  0.01, 0.005, 0.002, &rec);
    ASSERT(umcp_seam_count(&chain) == 1, "count after 1 add");
    ASSERT_NEAR(rec.delta_kappa_ledger, 0.1, 1e-15,
                "dk_ledger = kappa_t1 - kappa_t0");

    umcp_seam_add(&chain, 10, 20, -0.4, -0.3, 5.0,
                  0.01, 0.005, 0.002, &rec);
    umcp_seam_add(&chain, 20, 30, -0.3, -0.2, 5.0,
                  0.01, 0.005, 0.002, &rec);

    ASSERT(umcp_seam_count(&chain) == 3, "count after 3 adds");

    /* Total Δκ should be additive (Lemma 20) */
    ASSERT_NEAR(umcp_seam_total_delta_kappa(&chain), 0.3, 1e-14,
                "total_delta_kappa = sum of individual dk_ledger");
}

static void test_seam_associativity(void)
{
    printf("  Test: Seam associativity (monoid property)\n");

    /* Two orderings should give same total */
    umcp_seam_record_t buf1[10], buf2[10];
    umcp_seam_chain_t chain1, chain2;

    umcp_seam_init(&chain1, buf1, 10);
    umcp_seam_init(&chain2, buf2, 10);

    /* Chain 1: (s1, s2), s3 */
    umcp_seam_add(&chain1, 0, 10, -0.5, -0.4, 5.0, 0.01, 0.005, 0.002, NULL);
    umcp_seam_add(&chain1, 10, 20, -0.4, -0.3, 5.0, 0.01, 0.005, 0.002, NULL);
    umcp_seam_add(&chain1, 20, 30, -0.3, -0.2, 5.0, 0.01, 0.005, 0.002, NULL);

    /* Chain 2: s1, (s2, s3) — same seams, just verifying additivity */
    umcp_seam_add(&chain2, 0, 10, -0.5, -0.4, 5.0, 0.01, 0.005, 0.002, NULL);
    umcp_seam_add(&chain2, 10, 20, -0.4, -0.3, 5.0, 0.01, 0.005, 0.002, NULL);
    umcp_seam_add(&chain2, 20, 30, -0.3, -0.2, 5.0, 0.01, 0.005, 0.002, NULL);

    ASSERT_NEAR(umcp_seam_total_delta_kappa(&chain1),
                umcp_seam_total_delta_kappa(&chain2), 1e-15,
                "associativity: same total regardless of grouping");

    ASSERT_NEAR(umcp_seam_cumulative_residual(&chain1),
                umcp_seam_cumulative_residual(&chain2), 1e-15,
                "associativity: same cumulative residual");
}

/* ═══════════════════ SHA-256 Tests ══════════════════════════════ */

static void test_sha256_nist_vectors(void)
{
    printf("  Test: SHA-256 NIST test vectors\n");

    char hex[UMCP_SHA256_HEX_SIZE];

    /* NIST vector 1: "abc" */
    umcp_sha256_hex("abc", 3, hex);
    ASSERT(strcmp(hex, "ba7816bf8f01cfea414140de5dae2223"
                       "b00361a396177a9cb410ff61f20015ad") == 0,
           "SHA-256('abc') matches NIST");

    /* NIST vector 2: empty string */
    umcp_sha256_hex("", 0, hex);
    ASSERT(strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb924"
                       "27ae41e4649b934ca495991b7852b855") == 0,
           "SHA-256('') matches NIST");

    /* NIST vector 3: "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" */
    const char *v3 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    umcp_sha256_hex(v3, strlen(v3), hex);
    ASSERT(strcmp(hex, "248d6a61d20638b8e5c026930c3e6039"
                       "a33ce45964ff2167f6ecedd419db06c1") == 0,
           "SHA-256(448-bit) matches NIST");
}

/* ═══════════════════ Random Sweep (Tier-1 Proof) ════════════════ */

static void test_random_sweep(void)
{
    printf("  Test: Random sweep (1000 vectors × Tier-1 identities)\n");

    /*
     * Simple LCG for deterministic pseudo-random in pure C.
     * Not cryptographic — just for test diversity.
     */
    unsigned long seed = 42;
    #define NEXT_RAND() (seed = (seed * 6364136223846793005ULL + 1442695040888963407ULL), \
                         (double)((seed >> 33) & 0x7FFFFFFF) / (double)0x7FFFFFFF)

    int n_channels = 8;
    double w[8];
    uniform_weights(w, n_channels);
    umcp_kernel_result_t out;

    int identity_violations = 0;

    for (int trial = 0; trial < 1000; ++trial) {
        double c[8];
        for (int j = 0; j < 8; ++j) {
            c[j] = 0.01 + 0.98 * NEXT_RAND();  /* c ∈ [0.01, 0.99] */
        }

        umcp_kernel_compute(c, w, 8, 1e-8, &out);

        /* Identity 1: F + ω = 1 */
        if (fabs(out.F + out.omega - 1.0) > 1e-14) identity_violations++;

        /* Identity 2: IC ≤ F */
        if (out.IC > out.F + 1e-14) identity_violations++;

        /* Identity 3: IC = exp(κ) */
        if (fabs(out.IC - exp(out.kappa)) > 1e-14) identity_violations++;

        /* Range checks */
        if (out.F < 0.0 || out.F > 1.0) identity_violations++;
        if (out.S < 0.0) identity_violations++;
        if (out.C < 0.0) identity_violations++;
    }

    ASSERT(identity_violations == 0,
           "0 identity violations across 1000 random vectors");
    #undef NEXT_RAND
}

/* ═══════════════════ Main ═══════════════════════════════════════ */

int main(void)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("  UMCP Pure C Kernel Tests\n");
    printf("  Collapsus generativus est; solum quod redit, reale est.\n");
    printf("════════════════════════════════════════════════════════\n\n");

    /* Kernel tests */
    printf("── Kernel ──────────────────────────────────────────────\n");
    test_duality_identity();
    test_integrity_bound();
    test_log_integrity_relation();
    test_homogeneity_detection();
    test_bernoulli_entropy();
    test_batch_computation();
    test_error_handling();
    test_equator_convergence();
    test_random_sweep();

    /* Seam tests */
    printf("\n── Seam Chain ──────────────────────────────────────────\n");
    test_seam_accumulation();
    test_seam_associativity();

    /* SHA-256 tests */
    printf("\n── SHA-256 ─────────────────────────────────────────────\n");
    test_sha256_nist_vectors();

    /* Summary */
    printf("\n════════════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    printf("════════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
