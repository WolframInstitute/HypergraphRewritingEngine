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

# Also check for Wolfram Language installations under the legacy install directory
if(NOT WOLFRAMSCRIPT_FOUND AND EXISTS "/mnt/c/Program Files/Wolfram Research/Mathematica")
    file(GLOB MATHEMATICA_VERSIONS "/mnt/c/Program Files/Wolfram Research/Mathematica/*")
    foreach(VERSION_DIR ${MATHEMATICA_VERSIONS})
        if(EXISTS "${VERSION_DIR}/wolframscript.exe")
            set(WOLFRAMSCRIPT_EXECUTABLE "${VERSION_DIR}/wolframscript.exe")
            set(WOLFRAMSCRIPT_FOUND TRUE)
            get_filename_component(VERSION_NAME ${VERSION_DIR} NAME)
            message(STATUS "Found WolframScript in Wolfram Language ${VERSION_NAME}: ${WOLFRAMSCRIPT_EXECUTABLE}")
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
if(BUILD_WOLFRAM_LANGUAGE_PACLET)
    add_compile_definitions(BUILD_WOLFRAM_LANGUAGE_PACLET=1)
else()
    add_compile_definitions(BUILD_WOLFRAM_LANGUAGE_PACLET=0)
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
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

    # Where the repository is, baked in at configure time. A test that needs a file from the
    # source tree -- the paclet, a corpus, a golden -- resolves it from here. A relative path
    # would resolve against the working directory, which is whatever the caller happened to be
    # in, so the same binary passes from one directory and fails from another.
    target_compile_definitions(${TARGET_NAME} PRIVATE HG_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

    if(DEBUG_OUTPUT)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DEBUG_OUTPUT)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
        target_link_options(${TARGET_NAME} PRIVATE -static-libgcc -static-libstdc++)
    endif()

    if(BUILD_WOLFRAM_LANGUAGE_PACLET)
        add_dependencies(${TARGET_NAME} paclet)
    endif()

    # ONLY all_tests IS A CTEST CITIZEN. gtest_discover_tests registers a `<target>_NOT_BUILT`
    # FAILING test for a binary that does not exist, and it does not build it -- so registering
    # the subset targets made ctest report five permanent failures on every CI run, because CI
    # builds all_tests and deliberately not the subsets. They are strict subsets of all_tests:
    # discovering them would also run every one of their cases a second time.
    #
    # The subsets exist for a developer who wants one slice, and the run_<name> custom targets
    # below are how that is done. Nothing is lost from the gate by leaving them out of it.
    if(TARGET_NAME STREQUAL "all_tests")
        include(GoogleTest)
        gtest_discover_tests(${TARGET_NAME})
    endif()
endfunction()

# =============================================================================
# all_tests - Complete test suite (everything)
# =============================================================================
create_test_executable(all_tests
    main.cpp
    test_helpers.hpp
    test_helpers.cpp
    # Hypergraph tests
    ../hypergraph/tests/test_join_core.cpp
    ../hypergraph/tests/test_slot_core.cpp
    ../hypergraph/tests/test_concurrent_map.cpp
    ../hypergraph/tests/test_concurrent_key_set.cpp
    ../hypergraph/tests/test_concurrent_map_fuzz.cpp
    ../hypergraph/tests/test_quotient_completeness.cpp
    ../hypergraph/tests/test_causal_tr_exactness.cpp
    ../hypergraph/tests/test_causal_determinism.cpp
    ../hypergraph/tests/test_ir_canonicalization.cpp
    ../hypergraph/tests/test_pattern_matching.cpp
    ../hypergraph/tests/test_match_completeness.cpp
    ../hypergraph/tests/test_match_dedup_exactness.cpp
    ../hypergraph/tests/test_parallel_evolution.cpp
    ../hypergraph/tests/test_multiple_initial_states.cpp
    ../hypergraph/tests/test_evolution_limits.cpp
    ../hypergraph/tests/test_sampling_reproducibility.cpp
    ../hypergraph/tests/test_causal_branchial.cpp
    ../hypergraph/tests/test_event_canonicalization.cpp
    ../hypergraph/tests/test_determinism_fuzzing.cpp
    ../hypergraph/tests/test_segmented_array.cpp
    ../hypergraph/tests/test_blackhole_idempotent.cpp
    ../hypergraph/tests/test_grid_performance.cpp
    ../hypergraph/tests/test_repeated_invocation.cpp
    ../hypergraph/tests/test_reference_oracle.cpp
    ../hypergraph/tests/test_oracle_corpus.cpp
    ../hypergraph/tests/test_golden_matrix.cpp
    ../hypergraph/tests/test_event_identity_authority.cpp
    ../hypergraph/tests/test_sampled_soundness.cpp
    ../hypergraph/tests/test_feature_matrix.cpp
    # WXF tests
    ../wxf/tests/test_wxf.cpp
    # Paclet tests
    ../paclet_source/tests/test_paclet.cpp
    # FFI WXF serialization pin test: exercises run_rewriting_core end to end.
    # Pulls in the FFI marshaling TU (compiled in standalone mode, no Wolfram SDK).
    ../paclet_source/tests/test_wxf_serialization.cpp
    # The option surface is written down in four places (declared, sent, parsed, documented)
    # and nothing links them; this reconciles them by reading the sources.
    ../paclet_source/tests/test_option_surface.cpp
    # The session handle space and lifetime rule, asserted while the slot is still small enough
    # to read in one sitting -- these are the contract the verbs get written against.
    ../paclet_source/tests/test_session.cpp
    ../paclet_source/hypergraph_ffi.cpp
    ../paclet_source/paclet_support.cpp
)
# run_rewriting_core is the standalone-binary marshaling path (WXF over the FFI):
# no Wolfram SDK, process-kill abort. HG_STANDALONE_BINARY only guards code in the
# two paclet_source TUs, so defining it target-wide is inert for the other tests.
target_compile_definitions(all_tests PRIVATE HG_STANDALONE_BINARY)
target_include_directories(all_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/paclet_source
)


# =============================================================================
# core_tests - Fast core functionality tests
# =============================================================================
create_test_executable(core_tests
    main.cpp
    test_helpers.hpp
    test_helpers.cpp
    ../hypergraph/tests/test_ir_canonicalization.cpp
    ../wxf/tests/test_wxf.cpp
)

# =============================================================================
# evolution_tests - Evolution and pattern matching tests
# =============================================================================
create_test_executable(evolution_tests
    main.cpp
    test_helpers.hpp
    test_helpers.cpp
    ../hypergraph/tests/test_pattern_matching.cpp
    ../hypergraph/tests/test_match_completeness.cpp
    ../hypergraph/tests/test_match_dedup_exactness.cpp
    ../hypergraph/tests/test_parallel_evolution.cpp
    ../hypergraph/tests/test_multiple_initial_states.cpp
    ../hypergraph/tests/test_evolution_limits.cpp
    ../hypergraph/tests/test_sampling_reproducibility.cpp
)

# =============================================================================
# causal_tests - Causal and branchial graph tests
# =============================================================================
create_test_executable(causal_tests
    main.cpp
    test_helpers.hpp
    test_helpers.cpp
    ../hypergraph/tests/test_causal_branchial.cpp
    ../hypergraph/tests/test_event_canonicalization.cpp
)

# =============================================================================
# stress_tests - Determinism fuzzing and performance tests
# =============================================================================
create_test_executable(stress_tests
    main.cpp
    test_helpers.hpp
    test_helpers.cpp
    ../hypergraph/tests/test_determinism_fuzzing.cpp
    ../hypergraph/tests/test_segmented_array.cpp
    ../hypergraph/tests/test_grid_performance.cpp
    ../hypergraph/tests/test_repeated_invocation.cpp
)

# =============================================================================
# integration_tests - Integration tests (paclet, blackhole)
# =============================================================================
create_test_executable(integration_tests
    main.cpp
    test_helpers.hpp
    test_helpers.cpp
    ../paclet_source/tests/test_paclet.cpp
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
