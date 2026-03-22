# UMCP C99 Orchestration Core

> *Collapsus generativus est; solum quod redit, reale est.*

The entire Tier-0 protocol formalized in portable C99 (~1,900 lines).
Provides the foundational computation AND orchestration layer in the three-layer sandwich:

```
C (raw math + orchestration, stable ABI) → C++ (types, pybind11) → Python (domain closures)
```

## Why C?

1. **Stable ABI** — `extern "C"` functions callable from any language (C++, Python, Rust, Julia, Go, WASM) without name mangling or ABI fragility.
2. **Zero-allocation hot path** — kernel, regime classification, and ledger operations require no heap allocation. All results are returned by value or written to caller-provided buffers.
3. **Reduced mechanical overhead** — ~1,900 lines of C formalize what takes ~5,000 lines in Python. Once the protocol is fully synthesized, C maps mathematical structure directly to computation with minimal abstraction overhead.
4. **Embeddable** — no dependencies beyond `<math.h>` and `<string.h>`. Runs on microcontrollers, compiles to WebAssembly.
5. **SIMD-ready** — the fused single-pass loop structure is designed for explicit SIMD intrinsics (AVX2/NEON) as a future extension.

## Components

### Computational Kernel (166 test assertions)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `include/umcp_c/kernel.h` | Kernel computation API | `umcp_kernel_compute()`, `umcp_kernel_batch()` |
| `include/umcp_c/seam.h` | Seam chain accumulation | `umcp_seam_init()`, `umcp_seam_add()` |
| `include/umcp_c/sha256.h` | Portable SHA-256 (FIPS 180-4) | `umcp_sha256_init/update/final()`, `umcp_sha256_file()` |
| `src/kernel.c` | Kernel implementation — single-pass fused loop | F, ω, S, C, κ, IC in one traversal |
| `src/seam.c` | O(1) incremental Δκ accumulation (Lemma 20) | Budget model with returning dynamics |
| `src/sha256.c` | Full FIPS 180-4 SHA-256 | 64-round transform, streaming API |
| `tests/test_kernel_c.c` | 166 test assertions | Tier-1 identities, NIST vectors, random sweep |

### Orchestration Backbone (160 test assertions)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `include/umcp_c/types.h` | Foundation types: regime, verdict, seam status, error codes | `umcp_regime_t`, `umcp_verdict_t`, `umcp_seam_status_t` |
| `include/umcp_c/contract.h` | Frozen contract: all seam-derived parameters + cost closures | `umcp_contract_init()`, `umcp_contract_gamma()`, `umcp_contract_delta_kappa()` |
| `include/umcp_c/regime.h` | Regime classification: four-gate criterion + Fisher partition | `umcp_regime_classify()` |
| `include/umcp_c/trace.h` | Trace vector lifecycle: allocation, embedding, identity validation | `umcp_trace_embed()`, `umcp_trace_validate()` |
| `include/umcp_c/ledger.h` | Integrity ledger: append-only with O(1) running statistics | `umcp_ledger_append()`, `umcp_ledger_verdict()` |
| `include/umcp_c/pipeline.h` | The spine in C: Contract → Canon → Closures → Ledger → Stance | `umcp_pipeline_init()`, `umcp_pipeline_step()` |
| `src/contract.c` | Frozen contract init, validation, Γ(ω), D_C, Δκ budget | Cost closures matching Python exactly |
| `src/regime.c` | Gate precedence: CRITICAL→COLLAPSE→WATCH→STABLE | 4-gate conjunctive Stable criterion |
| `src/trace.c` | pre_clip embedding, simplex weight validation | Clipping + ε-guard + normalization check |
| `src/ledger.c` | Append + build_entry + running stats + verdict aggregation | O(1) Welford running mean/variance |
| `src/pipeline.c` | Five-stop spine orchestrator (145 lines) | The complete validation pipeline |
| `tests/test_orchestration.c` | 160 test assertions | Types, contract, regime, trace, ledger, pipeline |

### Module Dependency Chain

```
types.h ← kernel.h / contract.h / seam.h / sha256.h
                ↓            ↓
           regime.h      trace.h
                ↓            ↓
              ledger.h ← ───┘
                ↓
            pipeline.h  (imports all of the above)
```

## Tier-1 Invariants (Computed)

The kernel computes six invariants from a trace vector `c[n]` with weights `w[n]`:

| Symbol | Name | Formula |
|--------|------|---------|
| **F** | Fidelity | F = Σ wᵢcᵢ |
| **ω** | Drift | ω = 1 − F |
| **S** | Bernoulli field entropy | S = −Σ wᵢ[cᵢ ln(cᵢ) + (1−cᵢ)ln(1−cᵢ)] |
| **C** | Curvature | C = σ(c)/0.5 |
| **κ** | Log-integrity | κ = Σ wᵢ ln(clamp(cᵢ, ε, 1−ε)) |
| **IC** | Integrity composite | IC = exp(κ) |

Three algebraic identities hold by construction: **F + ω = 1**, **IC ≤ F**, **IC = exp(κ)**.

## Build (Standalone)

```bash
cd src/umcp_c
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./test_umcp_c              # 166 kernel tests
./test_umcp_orchestration  # 160 orchestration tests
```

## Build (Integrated with C++)

The C core is automatically included when building the C++ accelerator:

```bash
cd src/umcp_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./umcp_c/test_umcp_c              # C kernel tests (166)
./umcp_c/test_umcp_orchestration  # C orchestration tests (160)
./test_umcp_kernel                # C++ tests (434 assertions)
# Total: 760 assertions across the C/C++ stack
```

## Frozen Parameters

All frozen parameters match `src/umcp/frozen_contract.py`:
- **ε** = 1e-8 (guard band, passed as argument — not hardcoded)
- **p** = 3 (drift cost exponent in Γ(ω) = ω^p/(1−ω+ε))
- **α** = 1.0 (curvature cost coefficient in D_C = α·C)
- **λ** = 0.2 (auxiliary coefficient)
- **tol_seam** = 0.005 (seam residual tolerance)
- **Homogeneity tolerance** = 1e-15 (internal constant for OPT-1 fast path)

## Test Coverage

### Kernel Tests (166 assertions — `test_kernel_c.c`)
- Duality identity F + ω = 1 (exact to machine precision)
- Integrity bound IC ≤ F (all test vectors)
- Log-integrity relation IC = exp(κ) (relative error < 1e-12)
- Homogeneity detection (OPT-1 fast path)
- Bernoulli entropy bounds [0, ln2] across 99 channel values
- Batch computation (10 rows × 4 channels)
- Error handling (NULL pointers, zero dimensions, invalid weights, out-of-range channels)
- Equator convergence S + κ = 0 at c = 1/2
- Random sweep (1000 vectors × 3 Tier-1 identities)
- Seam chain accumulation and associativity (monoid property, error < 1e-15)
- SHA-256 NIST test vectors (empty, "abc", 448-bit)

### Orchestration Tests (160 assertions — `test_orchestration.c`)
- Foundation types: all enum values, string conversions, comparison operators
- Frozen contract: parameter init, Γ(ω) boundary conditions, D_C computation, Δκ budget
- Regime classification: four-gate conjunctive Stable, Watch thresholds, Collapse boundary, Critical overlay
- Trace management: pre_clip embedding, ε-guard validation, simplex check, error handling
- Integrity ledger: append, running statistics, verdict aggregation, capacity limits
- Full pipeline: Contract → Canon → Closures → Ledger → Stance, end-to-end verdict

### Total: 326 C assertions (+ 434 C++ Catch2 = 760 across the stack)

## API Examples

### Kernel Computation

```c
#include "umcp_c/kernel.h"

double channels[] = {0.95, 0.85, 0.70, 0.60};
double weights[]  = {0.25, 0.25, 0.25, 0.25};
umcp_kernel_result_t result;

int rc = umcp_kernel_compute(channels, weights, 4, 1e-8, &result);
if (rc == UMCP_OK) {
    printf("F=%.4f  ω=%.4f  IC=%.4f\n", result.F, result.omega, result.IC);
    // F + ω = 1.0 exactly
    // IC ≤ F    always
}
```

### Full Validation Pipeline (The Spine in C)

```c
#include "umcp_c/pipeline.h"

umcp_pipeline_t pipeline;
umcp_pipeline_init(&pipeline);  // Loads frozen contract defaults

double channels[] = {0.95, 0.85, 0.70, 0.60};
double weights[]  = {0.25, 0.25, 0.25, 0.25};

umcp_pipeline_result_t result;
int rc = umcp_pipeline_step(&pipeline, channels, weights, 4, &result);
if (rc == UMCP_OK) {
    // result.kernel   — F, ω, S, C, κ, IC
    // result.regime   — STABLE / WATCH / COLLAPSE (+ CRITICAL overlay)
    // result.verdict  — CONFORMANT / NONCONFORMANT / NON_EVALUABLE
    // result.budget   — Γ(ω), D_C, Δκ
    printf("Regime: %s  Verdict: %s\n",
           umcp_regime_str(result.regime),
           umcp_verdict_str(result.verdict));
}
```
