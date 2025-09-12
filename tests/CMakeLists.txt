cmake_minimum_required(VERSION 3.14)

# Unified Test Suite
project(UnifiedTestSuite CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Find and configure GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Create unified test executable
add_executable(unified_tests
    main.cpp
    test_helpers.hpp
    test_unified_simple.cpp
    test_core_hypergraph.cpp
    test_edge_signature.cpp
    test_pattern_matching_comprehensive.cpp  
    test_wolfram_states.cpp  
    test_parallel_tasks.cpp
    test_wolfram_evolution.cpp
    test_determinism_fuzzing.cpp
)

# Link against all required libraries
target_link_libraries(unified_tests
    PRIVATE
    # Core libraries
    hypergraph
    job_system  
    lockfree_deque
    
    # GoogleTest
    GTest::gtest_main
    GTest::gtest
    
    # System libraries
    Threads::Threads
)

# Include directories
target_include_directories(unified_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


# Enable GoogleTest discovery
include(GoogleTest)
gtest_discover_tests(unified_tests)

# Add test target for easy execution
add_custom_target(run_tests
    COMMAND unified_tests --gtest_color=yes
    DEPENDS unified_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unified test suite..."
)

# Performance test target (runs only performance-related tests)
add_custom_target(run_perf_tests
    COMMAND unified_tests --gtest_filter="*Stress*:*Performance*:*Concurrency*" --gtest_color=yes
    DEPENDS unified_tests  
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running performance tests..."
)

# Determinism fuzzing test target (runs determinism tests)
add_custom_target(run_determinism_tests
    COMMAND unified_tests --gtest_filter="*DeterminismFuzzing*" --gtest_color=yes
    DEPENDS unified_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running determinism fuzzing tests..."
)

# Integration test target (runs only integration tests)
add_custom_target(run_integration_tests
    COMMAND unified_tests --gtest_filter="*Integration*" --gtest_color=yes
    DEPENDS unified_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running integration tests..."
)

# Unit test target (runs core unit tests only)
add_custom_target(run_unit_tests
    COMMAND unified_tests --gtest_filter="*Unit*" --gtest_color=yes
    DEPENDS unified_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
    COMMENT "Running unit tests..."
)