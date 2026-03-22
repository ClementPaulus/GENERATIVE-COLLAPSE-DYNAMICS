cmake_minimum_required(VERSION 3.16)
project(umcp_c VERSION 1.0.0 LANGUAGES C)

# ─── C99 required (portable, no C++ dependency) ──────────────────
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Compiler optimizations ──────────────────────────────────────
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -Wpedantic -Wstrict-prototypes)
    # Enable architecture-specific SIMD for auto-vectorization
    if(NOT DEFINED UMCP_PORTABLE)
        add_compile_options(-march=native)
    endif()
    # Enable fast-math for kernel hot path (F, κ, S are tolerant)
    # Not enabled by default — uncomment for benchmarks:
    # add_compile_options(-ffast-math)
elseif(MSVC)
    add_compile_options(/O2 /W4)
endif()

# ─── Static library: the pure C kernel core ──────────────────────
add_library(umcp_c_core STATIC
    src/kernel.c
    src/seam.c
    src/sha256.c
    src/contract.c
    src/regime.c
    src/trace.c
    src/ledger.c
    src/pipeline.c
)
target_include_directories(umcp_c_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Link math library (required on Linux for log, exp, sqrt)
target_link_libraries(umcp_c_core PUBLIC m)

# ─── Pure C test executables ─────────────────────────────────────
option(UMCP_C_BUILD_TESTS "Build pure C tests" ON)
if(UMCP_C_BUILD_TESTS)
    enable_testing()

    # Original kernel/seam/SHA-256 tests
    add_executable(test_umcp_c tests/test_kernel_c.c)
    target_link_libraries(test_umcp_c PRIVATE umcp_c_core)
    add_test(NAME umcp_c_tests COMMAND test_umcp_c)

    # Orchestration layer tests (types, contract, regime, trace, ledger, pipeline)
    add_executable(test_umcp_orchestration tests/test_orchestration.c)
    target_link_libraries(test_umcp_orchestration PRIVATE umcp_c_core)
    add_test(NAME umcp_orchestration_tests COMMAND test_umcp_orchestration)
endif()
