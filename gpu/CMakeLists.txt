# GPU Hypergraph Rewriting Engine
# Requires CUDA Toolkit 11.0+ for compute capability 8.6 (Ampere)

cmake_minimum_required(VERSION 3.18)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    message(WARNING "CUDA compiler not found. GPU targets will not be built.")
    return()
endif()

# Set CUDA architecture (RTX 3070 Ti = SM86)
set(CMAKE_CUDA_ARCHITECTURES 86)

# CUDA compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/hypergraph/include
)

# =============================================================================
# GPU Library
# =============================================================================

set(GPU_SOURCES
    wl_hash.cu
    pattern_match.cu
    rewrite.cu
    causal.cu
    evolution.cu
    signature_index.cu
    gpu_evolution_host.cu
)

set(GPU_HEADERS
    types.cuh
    hash_table.cuh
    memory_pool.cuh
    work_queue.cuh
    wl_hash.cuh
    pattern_match.cuh
    rewrite.cuh
    causal.cuh
    evolution.cuh
    signature_index.cuh
)

# Only build if source files exist
set(GPU_SOURCES_EXIST TRUE)
foreach(src ${GPU_SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        set(GPU_SOURCES_EXIST FALSE)
        message(STATUS "GPU source not yet created: ${src}")
    endif()
endforeach()

if(GPU_SOURCES_EXIST)
    add_library(gpu_hypergraph STATIC ${GPU_SOURCES})

    target_compile_features(gpu_hypergraph PUBLIC cxx_std_17)
    set_target_properties(gpu_hypergraph PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Link against hypergraph for shared types
    target_link_libraries(gpu_hypergraph PUBLIC hypergraph)
endif()

# =============================================================================
# GPU Tests
# =============================================================================

set(GPU_TEST_SOURCES
    tests/cuda_warmup.cu
    tests/test_gpu_basic.cu
    tests/test_gpu_components.cu
    tests/test_gpu_wl_hash.cu
    tests/test_gpu_pattern_match.cu
    tests/test_gpu_rewrite.cu
    tests/test_gpu_evolution.cu
)

# Check if test sources exist
set(GPU_TESTS_EXIST TRUE)
foreach(src ${GPU_TEST_SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        set(GPU_TESTS_EXIST FALSE)
        message(STATUS "GPU test not yet created: ${src}")
    endif()
endforeach()

if(GPU_SOURCES_EXIST AND GPU_TESTS_EXIST)
    add_executable(gpu_tests ${GPU_TEST_SOURCES})

    target_link_libraries(gpu_tests
        gpu_hypergraph
        hypergraph
        gtest
        gtest_main
    )

    set_target_properties(gpu_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Register with CTest
    add_test(NAME gpu_tests COMMAND gpu_tests)
endif()

# =============================================================================
# GPU vs CPU Unified Comparison Test (C++20, uses host interface)
# =============================================================================

if(GPU_SOURCES_EXIST AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_cpu_unified.cpp")
    add_executable(gpu_cpu_comparison_tests tests/test_gpu_cpu_unified.cpp)

    # Use C++20 for unified headers
    target_compile_features(gpu_cpu_comparison_tests PUBLIC cxx_std_20)

    # Link against GPU (via host interface), hypergraph (unified), and gtest
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(gpu_cpu_comparison_tests
        gpu_hypergraph
        hypergraph
        job_system
        gtest
        gtest_main
        Threads::Threads
        CUDA::cudart
        CUDA::cuda_driver
    )

    add_test(NAME gpu_cpu_comparison_tests COMMAND gpu_cpu_comparison_tests)
endif()

# =============================================================================
# Benchmark executable (optional)
# =============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_gpu.cu" AND GPU_SOURCES_EXIST)
    add_executable(benchmark_gpu benchmark_gpu.cu)
    target_link_libraries(benchmark_gpu gpu_hypergraph hypergraph)
endif()
