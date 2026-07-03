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
    src/rewrite.cu
    src/wl_hash.cu
)

set_property(TARGET hg_gpu PROPERTY CUDA_ARCHITECTURES ${HG_GPU_ARCHS})

target_include_directories(hg_gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(hg_gpu PUBLIC cxx_std_17)

target_compile_options(hg_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror=cross-execution-space-call>
)

if(BUILD_TESTING OR CMAKE_TESTING_ENABLED OR TARGET gtest)
    add_executable(hg_gpu_tests
        tests/test_smoke.cu
        tests/test_atomic_pool.cu
        tests/test_hash_table.cu
        tests/test_ring_buffer.cu
        tests/test_lock_free_list.cu
        tests/test_termination.cu
        tests/test_warp_ops.cu
        tests/test_edge_signature.cu
        tests/test_indices.cu
        tests/test_partial_match.cu
        tests/test_engine_state.cu
        tests/test_match.cu
        tests/test_rewrite.cu
        tests/test_wl_hash.cu
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

    add_executable(bench_cpu_vs_gpu
        tests/bench_cpu_vs_gpu.cpp
    )
    target_link_libraries(bench_cpu_vs_gpu PRIVATE hg_gpu hypergraph)
endif()
