/**
 * @file kernel.cpp
 * @brief Kernel computation — compilation unit.
 *
 * The C++ kernel API (kernel.hpp) provides types, validation, and advanced
 * features. For the innermost arithmetic, it can delegate to the pure C
 * core (umcp_c) which provides:
 *   - Stable C ABI for cross-language FFI
 *   - Zero-allocation hot path
 *   - SIMD-friendly loop structure
 *
 * The three-layer sandwich: C (raw math) → C++ (types, validation) → Python
 */

#include "umcp/kernel.hpp"

// The pure C kernel core is available via the C ABI
extern "C" {
#include "umcp_c/kernel.h"
}

namespace umcp {

/**
 * Batch kernel computation delegating to the C core for the hot loop.
 *
 * The C core's umcp_kernel_batch does zero allocation and processes
 * rows in a cache-friendly single pass. The C++ layer converts the
 * C result structs to C++ KernelOutputs and adds regime classification.
 */
std::vector<KernelOutputs> batch_via_c_core(
        const double* trace, const double* w,
        std::size_t T, std::size_t n, double epsilon) {

    // Allocate C result buffer
    std::vector<umcp_kernel_result_t> c_results(T);

    int rc = umcp_kernel_batch(trace, w, T, n, epsilon, c_results.data());
    if (rc != UMCP_OK) {
        throw std::runtime_error("C kernel batch failed with code " +
                                 std::to_string(rc));
    }

    // Convert C structs → C++ KernelOutputs
    std::vector<KernelOutputs> results;
    results.reserve(T);

    for (std::size_t t = 0; t < T; ++t) {
        const auto& cr = c_results[t];
        KernelOutputs out{};
        out.F = cr.F;
        out.omega = cr.omega;
        out.S = cr.S;
        out.C = cr.C;
        out.kappa = cr.kappa;
        out.IC = cr.IC;
        out.delta = cr.delta;
        out.is_homogeneous = (cr.is_homogeneous != 0);
        out.computation_mode = cr.is_homogeneous
            ? "fast_homogeneous" : "full_heterogeneous";

        // Classify heterogeneity regime (C++ enrichment)
        if (out.delta < 1e-6) {
            out.regime = "homogeneous";
        } else if (out.delta < 0.01) {
            out.regime = "coherent";
        } else if (out.delta < 0.05) {
            out.regime = "heterogeneous";
        } else {
            out.regime = "fragmented";
        }

        results.push_back(std::move(out));
    }

    return results;
}

}  // namespace umcp
