cmake_minimum_required(VERSION 3.14)

find_package(Threads REQUIRED)

# Try to find WolframScript for Wolfram integration benchmarks
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

if(NOT WOLFRAMSCRIPT_FOUND)
    message(STATUS "WolframScript not found - Wolfram integration benchmarks will be disabled")
endif()

# Detect WSL environment for proper path handling
if(DEFINED ENV{WSL_DISTRO_NAME})
    add_compile_definitions(WSL_ENVIRONMENT=1)
    message(STATUS "WSL environment detected for benchmarks")
else()
    add_compile_definitions(WSL_ENVIRONMENT=0)
endif()

# Create benchmark_results directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmark_results)

# ============================================================================
# Benchmark Suite
# ============================================================================

add_executable(benchmark_suite
    canonicalization_benchmarks.cpp
    uniqueness_tree_benchmarks.cpp
    pattern_matching_benchmarks.cpp
    state_management_benchmarks.cpp
    event_relationship_benchmarks.cpp
    evolution_benchmarks.cpp
    job_system_benchmarks.cpp
    wxf_benchmarks.cpp
    wolfram_integration_benchmark.cpp
    v1_v2_3_comparison_benchmarks.cpp
)

target_link_libraries(benchmark_suite PRIVATE benchmark_framework wxf)

# Define WOLFRAMSCRIPT_AVAILABLE for conditional compilation
# Only enable if both WolframScript AND paclet are available
if(WOLFRAMSCRIPT_FOUND AND BUILD_MATHEMATICA_PACLET)
    string(REPLACE "\\" "\\\\" ESCAPED_WOLFRAMSCRIPT_PATH "${WOLFRAMSCRIPT_EXECUTABLE}")
    target_compile_definitions(benchmark_suite PRIVATE
        WOLFRAMSCRIPT_EXECUTABLE="${ESCAPED_WOLFRAMSCRIPT_PATH}"
        WOLFRAMSCRIPT_AVAILABLE=1
    )
else()
    target_compile_definitions(benchmark_suite PRIVATE WOLFRAMSCRIPT_AVAILABLE=0)
endif()

# Define source directory for git commands
string(REPLACE "\\" "\\\\" ESCAPED_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
target_compile_definitions(benchmark_suite PRIVATE
    BENCHMARK_SOURCE_DIR="${ESCAPED_SOURCE_DIR}"
)

if(MINGW)
    target_link_options(benchmark_suite PRIVATE -static-libgcc -static-libstdc++)
endif()

if(BUILD_MATHEMATICA_PACLET)
    add_dependencies(benchmark_suite paclet)
endif()

# ============================================================================
# Run Targets
# ============================================================================

# Primary target: run benchmarks and save to benchmark_results/
# Usage: cmake .. -DBENCHMARK_FILTER=benchmark_name && make run_benchmarks
set(BENCHMARK_FILTER "" CACHE STRING "Benchmark name filter")
add_custom_target(run_benchmarks
    COMMAND $<TARGET_FILE:benchmark_suite> ${CMAKE_SOURCE_DIR}/benchmark_results ${BENCHMARK_FILTER}
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND_EXPAND_LISTS
    COMMENT "Running benchmark suite..."
)