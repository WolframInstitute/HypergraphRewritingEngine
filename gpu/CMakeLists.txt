set(HG_GPU_ARCHS "75;80;86;89;90" CACHE STRING
    "CUDA architectures for the hg_gpu library (semicolon-separated CC numbers)")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_library(hg_gpu STATIC
    src/evolve.cu
    src/initial_upload.cu
    src/ir_canon.cu
    src/match.cu
    src/event_identity.cu
    src/persistent.cu
    src/rewrite.cu
    src/content_hash.cu
)

set_property(TARGET hg_gpu PROPERTY CUDA_ARCHITECTURES ${HG_GPU_ARCHS})

# hg_gpu is built with separable compilation (relocatable device code), so its device symbols
# need a device-link (nvcc -dlink) step before it can be linked into a plain C++ target such as
# hg_evolve_gpu. Resolve them inside the static library so any non-CUDA consumer links cleanly.
# The Makefile/Ninja generators device-link at the consumer automatically; the Visual Studio
# generator does not, leaving __cudaRegisterLinkedBinary unresolved (LNK2019) — this makes the
# device link explicit and generator-independent, so the Windows MSVC build links too.
set_property(TARGET hg_gpu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Static-link the CUDA runtime so consumers (hg_evolve_gpu) carry no cudart64_*.dll /
# libcudart.so dependency. The only unavoidable runtime dependency of a GPU binary is the
# NVIDIA driver (nvcuda.dll / libcuda.so): it is the installed driver, cannot be statically
# linked, and is present on every machine that has a usable NVIDIA GPU.
set_property(TARGET hg_gpu PROPERTY CUDA_RUNTIME_LIBRARY Static)

target_include_directories(hg_gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(hg_gpu PUBLIC cxx_std_17)


target_compile_options(hg_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror=cross-execution-space-call>
    # Compile the per-architecture code-generation passes in parallel (one nvcc invocation emits
    # SASS for every arch in CUDA_ARCHITECTURES; without this it does them serially). 0 = use all
    # CPUs. This is the dominant cost of the broad (Turing..Hopper) shippable build.
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=0>
)

# Parallelise the device optimisation of a SINGLE translation unit.
#
# --threads above parallelises across architectures; it does nothing for a single-architecture
# build, where the cost is one TU's device phases. persistent.cu is that TU: three kernels each
# inlining the whole pipeline (match, rewrite, canonicalization, dedup, quotient), and measured
# with nvcc --time on sm_89 alone it spends 207.6 s in cicc and 217.1 s in ptxas out of 427 s
# total -- preprocessing is 0.35 s, so this is optimisation of the kernels, not header volume.
#
# --split-compile lets both phases optimise partitions of the module concurrently. 0 = use all
# CPUs. Measured on the same TU and the same host: 427 s -> 37.9 s, with cicc 207.6 -> 18.7 and
# ptxas 217.1 -> 17.6.
#
# Requires nvcc 11.6. Older toolchains keep the serial build rather than failing to configure.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.6)
    target_compile_options(hg_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>
    )
endif()

if(BUILD_TESTING OR CMAKE_TESTING_ENABLED OR TARGET gtest)
    add_executable(hg_gpu_tests
        tests/test_smoke.cu
        tests/test_atomic_pool.cu
        tests/test_capacity_overflow.cu
        tests/test_device_map_sentinels.cu
        tests/test_hash_table.cu
        tests/test_ring_buffer.cu
        tests/test_lock_free_list.cu
        tests/test_termination.cu
        tests/test_edge_signature.cu
        tests/test_quotient_expansion.cu
        tests/test_indices.cu
        tests/test_engine_state.cu
        tests/test_match.cu
        tests/test_rewrite.cu
        tests/test_ir_canon.cu
        tests/test_exploration_probability.cu
    )
    set_property(TARGET hg_gpu_tests PROPERTY CUDA_ARCHITECTURES ${HG_GPU_ARCHS})
    target_link_libraries(hg_gpu_tests PRIVATE hg_gpu gtest gtest_main)

    add_executable(gpu_differential_tests
        tests/test_gpu_vs_cpu_differential.cpp
    )
    target_link_libraries(gpu_differential_tests PRIVATE
        hg_gpu hypergraph gtest gtest_main)
    # This file states its 26 workloads with C++20 DESIGNATED initializers, which name the
    # members they set and leave the rest at their declared defaults on purpose. GCC reports
    # each omission under -Wmissing-field-initializers; clang does not warn on designated
    # initializers at all. The defect that flag is an error for tree-wide is a POSITIONAL
    # initializer that runs out of members and zero-fills the tail silently -- which is what
    # happened to the determinism test's drop counters -- and a designated initializer cannot
    # have it, because every member it sets is named. Measured: this is the only C++ TU of the
    # 162 in build_gpu that reports the warning.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set_source_files_properties(tests/test_gpu_vs_cpu_differential.cpp
            PROPERTIES COMPILE_OPTIONS -Wno-missing-field-initializers)
    endif()

    add_executable(bench_cpu_vs_gpu
        tests/bench_cpu_vs_gpu.cpp
    )
    target_link_libraries(bench_cpu_vs_gpu PRIVATE hg_gpu hypergraph)

    add_executable(bench_gpu_evolve
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/bench_gpu_evolve.cpp
    )
    target_link_libraries(bench_gpu_evolve PRIVATE hg_gpu hypergraph)

    # Pins whether the persistent scheduler's causal-edge count is schedule-dependent, and
    # whether the quotient expansion race is the mechanism (constant with pruning off, varying
    # with it on). Companion to Rewrite.PersistentSchedulerThroughEngineRunMatchesTheStepLoop.
    add_executable(quotient_causal_probe_gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/quotient_causal_probe_gpu.cpp
    )
    target_link_libraries(quotient_causal_probe_gpu PRIVATE hg_gpu hypergraph)

    # Separates a presentation-order cause from a rank-assignment one when the device's
    # canonical edge ranks move between block counts.
    add_executable(gpu_rank_order_probe
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gpu_rank_order_probe.cu
    )
    target_link_libraries(gpu_rank_order_probe PRIVATE hg_gpu)
endif()
