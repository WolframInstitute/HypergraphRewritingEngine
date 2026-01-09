cmake_minimum_required(VERSION 3.14)

# Test Suite Configuration

# Find required packages
find_package(Threads REQUIRED)

# Try to find WolframScript for round-trip WXF testing
# Check for WSL2 environment first - search for any version
set(WOLFRAMSCRIPT_FOUND FALSE)
if(EXISTS "/mnt/c/Program Files/Wolfram Research/Wolfram")
    file(GLOB WOLFRAM_VERSIONS "/mnt/c/Program Files/Wolfram Research/Wolfram/*")
    foreach(VERSION_DIR ${WOLFRAM_VERSIONS})
        if(EXISTS "${VERSION_DIR}/wolframscript.exe")
            set(WOLFRAMSCRIPT_EXECUTABLE "${VERSION_DIR}/wolframscript.exe")
            set(WOLFRAMSCRIPT_FOUND TRUE)
            get_filename_component(VERSION_NAME ${VERSION_DIR} NAME)
            message(STATUS "Found WolframScript ${VERSION_NAME} in WSL2: ${WOLFRAMSCRIPT_EXECUTABLE}")
            break()
        endif()
    endforeach()
endif()

# Also check for Mathematica installations
if(NOT WOLFRAMSCRIPT_FOUND AND EXISTS "/mnt/c/Program Files/Wolfram Research/Mathematica")
    file(GLOB MATHEMATICA_VERSIONS "/mnt/c/Program Files/Wolfram Research/Mathematica/*")
    foreach(VERSION_DIR ${MATHEMATICA_VERSIONS})
        if(EXISTS "${VERSION_DIR}/wolframscript.exe")
            set(WOLFRAMSCRIPT_EXECUTABLE "${VERSION_DIR}/wolframscript.exe")
            set(WOLFRAMSCRIPT_FOUND TRUE)
            get_filename_component(VERSION_NAME ${VERSION_DIR} NAME)
            message(STATUS "Found WolframScript in Mathematica ${VERSION_NAME}: ${WOLFRAMSCRIPT_EXECUTABLE}")
            break()
        endif()
    endforeach()
endif()

# Try native Linux locations if not in WSL2
if(NOT WOLFRAMSCRIPT_FOUND)
    # Try to find it in standard locations
    find_program(WOLFRAMSCRIPT_EXECUTABLE
        NAMES wolframscript WolframScript
        PATHS
            "/usr/local/bin"
            "/usr/bin"
            "/opt/Wolfram/WolframEngine/*/bin"
            "/opt/Wolfram/Mathematica/*/bin"
            "$ENV{HOME}/.WolframEngine/*/bin"
            "$ENV{HOME}/.Mathematica/*/bin"
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS"
            "/Applications/Mathematica.app/Contents/MacOS"
        DOC "Path to WolframScript executable"
    )

    if(WOLFRAMSCRIPT_EXECUTABLE)
        set(WOLFRAMSCRIPT_FOUND TRUE)
        message(STATUS "Found WolframScript: ${WOLFRAMSCRIPT_EXECUTABLE}")
    endif()
endif()

# Allow user to override with environment variable
if(DEFINED ENV{WOLFRAMSCRIPT_PATH})
    set(WOLFRAMSCRIPT_EXECUTABLE "$ENV{WOLFRAMSCRIPT_PATH}")
    set(WOLFRAMSCRIPT_FOUND TRUE)
    message(STATUS "Using WolframScript from environment: ${WOLFRAMSCRIPT_EXECUTABLE}")
endif()

if(NOT WOLFRAMSCRIPT_FOUND)
    message(STATUS "WolframScript not found - WXF round-trip tests will be skipped")
    message(STATUS "You can set WOLFRAMSCRIPT_PATH environment variable to specify the location")
endif()

# Make WolframScript path available to tests
if(WOLFRAMSCRIPT_FOUND)
    # Escape the path properly for use in C++ string literal
    string(REPLACE "\\" "\\\\" ESCAPED_WOLFRAMSCRIPT_PATH "${WOLFRAMSCRIPT_EXECUTABLE}")
    add_compile_definitions(WOLFRAMSCRIPT_EXECUTABLE="${ESCAPED_WOLFRAMSCRIPT_PATH}")
    add_compile_definitions(WOLFRAMSCRIPT_AVAILABLE=1)
else()
    add_compile_definitions(WOLFRAMSCRIPT_AVAILABLE=0)
endif()

# Make paclet build flag available to tests
if(BUILD_MATHEMATICA_PACLET)
    add_compile_definitions(BUILD_MATHEMATICA_PACLET=1)
else()
    add_compile_definitions(BUILD_MATHEMATICA_PACLET=0)
endif()

# Detect WSL environment for paclet path handling
if(DEFINED ENV{WSL_DISTRO_NAME})
    add_compile_definitions(WSL_ENVIRONMENT=1)
    message(STATUS "WSL environment detected for paclet tests")
else()
    add_compile_definitions(WSL_ENVIRONMENT=0)
endif()

# Find and configure GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# =============================================================================
# Common test libraries
# =============================================================================
set(TEST_COMMON_LIBS
    hypergraph
    job_system
    lockfree_deque
    wxf
    GTest::gtest_main
    GTest::gtest
    Threads::Threads
)

# Helper function to create test executable
function(create_test_executable TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE ${TEST_COMMON_LIBS})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if(DEBUG_OUTPUT)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DEBUG_OUTPUT)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
        target_link_options(${TARGET_NAME} PRIVATE -static-libgcc -static-libstdc++)
    endif()

    if(BUILD_MATHEMATICA_PACLET)
        add_dependencies(${TARGET_NAME} paclet)
    endif()

    include(GoogleTest)
    gtest_discover_tests(${TARGET_NAME})
endfunction()

# =============================================================================
# all_tests - Complete test suite (everything)
# =============================================================================
create_test_executable(all_tests
    main.cpp
    test_helpers.hpp
    # Core tests
    test_simple.cpp
    test_canonicalization.cpp
    test_wxf.cpp
    test_paclet.cpp
    test_grid_performance.cpp
    test_repeated_invocation.cpp
    # Evolution and pattern matching
    ../hypergraph/tests/test_pattern_matching.cpp
    ../hypergraph/tests/test_parallel_evolution.cpp
    ../hypergraph/tests/test_multiple_initial_states.cpp
    ../hypergraph/tests/test_evolution_limits.cpp
    # Causal/branchial graph tests
    ../hypergraph/tests/test_causal_branchial.cpp
    ../hypergraph/tests/test_event_canonicalization.cpp
    # Determinism and stress tests
    ../hypergraph/tests/test_determinism_fuzzing.cpp
    ../hypergraph/tests/test_ut_vs_utinc_performance.cpp
    ../hypergraph/tests/test_blackhole_idempotent.cpp
)

# =============================================================================
# core_tests - Fast core functionality tests
# =============================================================================
create_test_executable(core_tests
    main.cpp
    test_helpers.hpp
    test_simple.cpp
    test_canonicalization.cpp
    test_wxf.cpp
)

# =============================================================================
# evolution_tests - Evolution and pattern matching tests
# =============================================================================
create_test_executable(evolution_tests
    main.cpp
    test_helpers.hpp
    ../hypergraph/tests/test_pattern_matching.cpp
    ../hypergraph/tests/test_parallel_evolution.cpp
    ../hypergraph/tests/test_multiple_initial_states.cpp
    ../hypergraph/tests/test_evolution_limits.cpp
)

# =============================================================================
# causal_tests - Causal and branchial graph tests
# =============================================================================
create_test_executable(causal_tests
    main.cpp
    test_helpers.hpp
    ../hypergraph/tests/test_causal_branchial.cpp
    ../hypergraph/tests/test_event_canonicalization.cpp
)

# =============================================================================
# stress_tests - Determinism fuzzing and performance tests
# =============================================================================
create_test_executable(stress_tests
    main.cpp
    test_helpers.hpp
    ../hypergraph/tests/test_determinism_fuzzing.cpp
    ../hypergraph/tests/test_ut_vs_utinc_performance.cpp
    test_grid_performance.cpp
    test_repeated_invocation.cpp
)

# =============================================================================
# integration_tests - Integration tests (paclet, blackhole)
# =============================================================================
create_test_executable(integration_tests
    main.cpp
    test_helpers.hpp
    test_paclet.cpp
    ../hypergraph/tests/test_blackhole_idempotent.cpp
)

# =============================================================================
# Run targets for convenience
# =============================================================================

# Run all tests
add_custom_target(run_tests
    COMMAND all_tests --gtest_color=yes
    DEPENDS all_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests..."
)

# Run core tests (fast)
add_custom_target(run_core_tests
    COMMAND core_tests --gtest_color=yes
    DEPENDS core_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running core tests..."
)

# Run evolution tests
add_custom_target(run_evolution_tests
    COMMAND evolution_tests --gtest_color=yes
    DEPENDS evolution_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running evolution tests..."
)

# Run causal tests
add_custom_target(run_causal_tests
    COMMAND causal_tests --gtest_color=yes
    DEPENDS causal_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running causal/branchial tests..."
)

# Run stress tests
add_custom_target(run_stress_tests
    COMMAND stress_tests --gtest_color=yes
    DEPENDS stress_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running stress/determinism tests..."
)

# Run integration tests
add_custom_target(run_integration_tests
    COMMAND integration_tests --gtest_color=yes
    DEPENDS integration_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running integration tests..."
)

# =============================================================================
# Quantum Analysis Tests (geodesic tracing + particle detection)
# =============================================================================
# Standalone test for geodesic/particle analysis (doesn't need GoogleTest)
add_executable(test_quantum_analysis
    test_quantum_analysis.cpp
)

target_link_libraries(test_quantum_analysis
    PRIVATE
    viz_blackhole
    hypergraph
    Threads::Threads
)

target_include_directories(test_quantum_analysis PRIVATE
    ${CMAKE_SOURCE_DIR}/visualisation/blackhole/include
    ${CMAKE_SOURCE_DIR}/hypergraph/include
)

# Static linking for MinGW
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    target_link_options(test_quantum_analysis PRIVATE -static-libgcc -static-libstdc++)
endif()

# Test target
add_custom_target(run_quantum_tests
    COMMAND test_quantum_analysis
    DEPENDS test_quantum_analysis
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running quantum analysis tests (geodesics + particles)..."
)
