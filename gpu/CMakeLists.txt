# CMAKE_CUDA_ARCHITECTURES IS THE KNOB. It is CMake's own since 3.18, every CUDA target
# honours it without being told, it accepts `native` (3.24+) to mean "the card in this
# machine", and it is what a user, a CI lane or a packager will reach for. HG_GPU_ARCHS is
# only the SHIPPING DEFAULT -- the set a release artifact carries SASS for -- and it applies
# solely when the caller has not said otherwise.
#
# It used to be the other way round: HG_GPU_ARCHS was forced onto every target's
# CUDA_ARCHITECTURES property, so -DCMAKE_CUDA_ARCHITECTURES=89 was accepted and silently
# ignored, and a measurement box with ONE card compiled SASS for five architectures. Setting
# the standard variable and letting the targets inherit it is what makes the standard knob
# work; overriding a per-target property is what broke it.
set(HG_GPU_ARCHS "75;80;86;89;90" CACHE STRING
    "Default CUDA architectures for shipped artifacts; CMAKE_CUDA_ARCHITECTURES overrides")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${HG_GPU_ARCHS})
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Device code is linked by nvcc's own device-link step, which IPO does not participate in, so
# every CUDA target opts out of the global Release setting rather than inheriting it.
add_library(hg_gpu STATIC
    src/engine_state.cu
    src/errors.cu
    src/quotient.cu
    src/evolve.cu
    src/initial_upload.cu
    src/ir_canon.cu
    src/match.cu
    src/event_identity.cu
    src/persistent.cu
    src/rewrite.cu
    src/content_hash.cu
)


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
    target_link_libraries(hg_gpu_tests PRIVATE hg_gpu gtest gtest_main)

# NO DEVICE LTO, AND THE PER-CONFIG PROPERTY IS THE ONE THAT DECIDES. The root sets
# CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE, which initialises
# INTERPROCEDURAL_OPTIMIZATION_RELEASE on every target created after it, and that beats a
# config-less INTERPROCEDURAL_OPTIMIZATION FALSE. Setting only the config-less one left every
# CUDA target compiling to `code=[compute_89,lto_89]` -- whole-program device LTO -- so nvlink
# re-optimised the entire device program at link time and took 14.6 GB of resident memory to do
# it on a single architecture.
#
# Device code is linked by nvcc's own device-link step. Every CUDA target opts out for all
# configurations rather than inheriting the host setting.
foreach(hg_cuda_target hg_gpu hg_gpu_tests)
    set_target_properties(${hg_cuda_target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION FALSE
        INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
        INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE
        INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL FALSE)

    # CHECKED, NOT ASSUMED. The opt-out above is easy to write and easy to get wrong -- the
    # first version set only the config-less property and did nothing, and nothing said so.
    # What the mistake cost is why this check exists rather than a comment: device LTO
    # flattened the replay's recursive cycle into one 61,616-byte frame, which forced a
    # per-thread stack that collapsed occupancy, and the GPU test suite ran 23.6 s instead of
    # 7.5 s while one test faulted outright.
    foreach(hg_cfg RELEASE RELWITHDEBINFO MINSIZEREL DEBUG)
        get_target_property(hg_ipo ${hg_cuda_target} INTERPROCEDURAL_OPTIMIZATION_${hg_cfg})
        if(hg_ipo)
            message(FATAL_ERROR
                "${hg_cuda_target} has INTERPROCEDURAL_OPTIMIZATION_${hg_cfg} on. Device LTO "
                "flattens the reconstruction replay's recursion into a single huge stack frame; "
                "see the note above this check.")
        endif()
    endforeach()
endforeach()

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

# DEVICE LTO IS OFF FOR EVERY CUDA TARGET, NOT JUST TWO OF THEM.
#
# The opt-out above names hg_gpu and hg_gpu_tests and sits inside the test-only block, so
# gpu_differential_tests, bench_gpu_evolve and the probes inherited the root's
# CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=TRUE. Linking those LTO device objects against
# hg_gpu's non-LTO ones is what produced, on the rented box:
#
#   nvlink fatal : elfLink linker library load error
#   gpu/CMakeFiles/gpu_differential_tests.dir/cmake_device_link.o
#
# and with BUILD_TESTS=OFF -- the shipping GPU configuration -- the block does not run at all,
# so hg_gpu itself gets device LTO: the case whose measured cost is recorded above (a 61,616
# byte frame, occupancy collapsed, the suite at 23.6 s against 7.5 with one test faulting).
#
# Applied here, at file scope, after every target exists, so a CUDA target added later cannot
# miss it by being declared outside one particular if().
foreach(hg_cuda_tgt hg_gpu hg_gpu_tests gpu_differential_tests bench_cpu_vs_gpu
                    bench_gpu_evolve quotient_causal_probe_gpu gpu_rank_order_probe)
    if(TARGET ${hg_cuda_tgt})
        set_target_properties(${hg_cuda_tgt} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION FALSE
            INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE
            INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL FALSE
            INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)
        get_target_property(hg_ipo_chk ${hg_cuda_tgt} INTERPROCEDURAL_OPTIMIZATION_RELEASE)
        if(hg_ipo_chk)
            message(FATAL_ERROR "device LTO still on for ${hg_cuda_tgt}")
        endif()
    endif()
endforeach()
