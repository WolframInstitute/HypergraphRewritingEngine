cmake_minimum_required(VERSION 3.14)

# Unified Test Suite
project(UnifiedTestSuite CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    test_debug_relationships.cpp
    test_canonicalization.cpp
    test_pattern_matching_isolation.cpp
    test_isolated_duplicate_bug.cpp
    test_debug_edge_selection.cpp
    test_wxf.cpp
    test_paclet.cpp
)

# Link against all required libraries
target_link_libraries(unified_tests
    PRIVATE
    # Core libraries
    hypergraph
    job_system
    lockfree_deque
    wxf  # WXF serialization library

    # GoogleTest
    GTest::gtest_main
    GTest::gtest

    # System libraries
    Threads::Threads
)

# Add paclet dependency when paclet building is enabled
if(BUILD_MATHEMATICA_PACLET)
    add_dependencies(unified_tests paclet)
endif()

# Static linking for MinGW to avoid DLL dependencies during test discovery
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    target_link_options(unified_tests PRIVATE -static-libgcc -static-libstdc++)
endif()

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

# WXF test target (runs WXF serialization tests)
add_custom_target(run_wxf_tests
    COMMAND unified_tests --gtest_filter="*WXF*" --gtest_color=yes
    DEPENDS unified_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running WXF serialization tests..."
)