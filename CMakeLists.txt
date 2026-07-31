cmake_minimum_required(VERSION 3.14)
# CMP0091 (CMake 3.15+) makes CMAKE_MSVC_RUNTIME_LIBRARY take effect, so a Windows/MSVC build
# can select the static C runtime (/MT). Must be set before project(). Without NEW the property
# is silently ignored and the binary keeps a VCRUNTIME140 / MSVCP140 (VC++ redist) dependency.
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
project(HypergraphEngine VERSION 0.0.1 LANGUAGES CXX)

# Shared CPU/GPU definitions (hgcommon)
include_directories(${CMAKE_SOURCE_DIR}/common/include)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "Build type not specified, defaulting to Release")
endif()

# Detect WSL environment
# WSL (Windows Subsystem for Linux) requires special handling for:
# - File locking (Java/Lucene operations fail on WSL filesystem)
# - Path conversions between Linux and Windows
# Note: Use CMAKE_HOST_SYSTEM_NAME to detect the build host, not the target
set(IS_WSL FALSE)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    if(EXISTS "/proc/sys/fs/binfmt_misc/WSLInterop" OR DEFINED ENV{WSL_DISTRO_NAME})
        set(IS_WSL TRUE)
        message(STATUS "WSL environment detected")
    endif()
endif()

# Export compile commands for clangd and other tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create symlink to compile_commands.json in project root for clangd
# This runs at configure time, not build time
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
endforeach()

# Benchmarking option for event relationship processing
option(SINGLE_THREADED_EVENT_RELATION_PROCESSING "Use single-threaded event relation processing for benchmarking" OFF)
if(SINGLE_THREADED_EVENT_RELATION_PROCESSING)
    add_compile_definitions(SINGLE_THREADED_EVENT_RELATION_PROCESSING)
endif()

# Fuzzing tests option (disabled by default for faster builds)
option(ENABLE_FUZZING_TESTS "Enable determinism fuzzing tests (slow but comprehensive)" OFF)
if(ENABLE_FUZZING_TESTS)
    add_compile_definitions(ENABLE_FUZZING_TESTS)
endif()

# Sanitizer options for detecting memory errors and race conditions
option(ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer for race condition detection" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# Sanitizer configuration (only one can be enabled at a time for ASAN/TSAN)
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "Cannot enable both AddressSanitizer and ThreadSanitizer simultaneously")
endif()

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=thread)
endif()

if(ENABLE_UBSAN)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=undefined)
endif()

add_subdirectory(lockfree_deque)
add_subdirectory(job_system)

# When visualization is enabled, create viz_events library BEFORE hypergraph
# so hypergraph can link to it for the VizEventSink static variables
if(BUILD_VISUALIZATION)
    add_library(viz_events STATIC
        visualisation/events/src/viz_event_sink.cpp
    )
    target_include_directories(viz_events PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/visualisation/events/include
    )
endif()

add_subdirectory(hypergraph)
add_subdirectory(wxf)

# Unified test suite
option(BUILD_TESTS "Build unified test suite" ON)
if(BUILD_TESTS)
    add_subdirectory(testing)
endif()

# Examples - API documentation and usage examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Regenerates reference/golden_matrix.txt, which the identity-matrix tests read. A build target
# rather than an ad-hoc compile: the file it writes is the expected output of 216 runs, so
# reproducing it has to be one command with the project's own include paths and flags.
# The exactness-and-cost harness. It proves output EXACT against the brute-force oracle while
# reporting arena bytes_allocated(), which is durable and noise-free where RSS is not -- so a
# change can be shown to preserve correctness and to move memory in a reproducible direction.
add_executable(event_key_decomposition_probe tools/event_key_decomposition_probe.cpp)
target_include_directories(event_key_decomposition_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(event_key_decomposition_probe PRIVATE hypergraph)

add_executable(thinning_mode_probe tools/thinning_mode_probe.cpp)
target_include_directories(thinning_mode_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(thinning_mode_probe PRIVATE hypergraph)

add_executable(cost_matrix tools/cost_matrix.cpp)
target_include_directories(cost_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cost_matrix PRIVATE hypergraph)

add_executable(generate_golden_matrix tools/generate_golden_matrix.cpp)
target_include_directories(generate_golden_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(generate_golden_matrix PRIVATE hypergraph)

# Repeat-count instrument for event identity stability. The gate compares one worker count
# against one other; this asks the same question many times over a spread of worker counts, which
# is what makes a "stable" reading evidence rather than a single agreement.
add_executable(event_signature_stability_probe tools/event_signature_stability_probe.cpp)
target_include_directories(event_signature_stability_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(event_signature_stability_probe PRIVATE hypergraph)

# Host counterpart of gpu_rank_order_probe: separates a stable presentation order from an
# order-independent rank rule as the reason the CPU's event identities do not move.
add_executable(cpu_rank_order_probe tools/cpu_rank_order_probe.cpp)
target_include_directories(cpu_rank_order_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpu_rank_order_probe PRIVATE hypergraph)

# Benchmarks - Performance measurement tools
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarking)
    add_subdirectory(benchmarks)
endif()

# Optional: Build GPU acceleration (requires CUDA). Added before the paclet so
# the paclet's optional GPU binary (hg_evolve_gpu) can link the hg_gpu target.
option(BUILD_GPU "Build GPU-accelerated evolution engine" OFF)
if(BUILD_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
        add_subdirectory(gpu)
    else()
        message(WARNING "CUDA not found - GPU support disabled")
    endif()
endif()

# Optional: Build Wolfram Language paclet if the Wolfram Language is found
option(BUILD_WOLFRAM_LANGUAGE_PACLET "Build Wolfram Language paclet library" ON)
if(BUILD_WOLFRAM_LANGUAGE_PACLET)
    add_subdirectory(paclet_source)
endif()

# Optional: Build interactive visualization
option(BUILD_VISUALIZATION "Build interactive hypergraph visualization" OFF)
if(BUILD_VISUALIZATION)
    add_subdirectory(visualisation)
endif()

# Every host tool gets a build target, automatically.
#
# 51 of 57 files under tools/ had none. `make <tool>` answered "No rule to make target", so each
# was compiled by hand once, produced its number, and then rotted -- including
# forwarding_validator_probe, the sole caller of the match-completeness validator, and cost_matrix,
# which describes itself as the harness the paper's ablation uses. A measurement whose instrument
# cannot be rebuilt cannot be repeated, and a number that cannot be repeated is not evidence.
#
# EXCLUDE_FROM_ALL so a probe that has fallen behind the API it measures does not break the
# default build or CI, while `make <tool>` still works and the rot is discoverable rather than
# invisible. `make host_tools` builds them all, which is what turns "this probe no longer
# compiles" into something a person finds on purpose instead of by needing it.
#
# Tools declared explicitly above keep their own targets; this skips those names.
file(GLOB HG_TOOL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp)
add_custom_target(host_tools)
foreach(tool_src ${HG_TOOL_SOURCES})
    get_filename_component(tool_name ${tool_src} NAME_WE)
    # A tool that includes the CUDA headers belongs to the GPU build, not this one, and would
    # fail here for a reason that says nothing about the tool.
    file(READ ${tool_src} tool_text)
    string(FIND "${tool_text}" "hg_gpu/" tool_needs_gpu)
    # ir_incremental_probe is the ONE tool that has genuinely rotted: it reads
    # IRCanonicalizer::last_stats, which no longer exists. It measured incremental IR, and
    # incremental WL hashing was built bit-identical and then dropped, so what it probes may no
    # longer be a thing to probe. Named here rather than deleted, because whether the probe or the
    # API should come back is a decision, not a build detail -- and because a silent skip would
    # put it straight back in the state this whole target exists to end.
    set(HG_ROTTED_TOOLS ir_incremental_probe)
    list(FIND HG_ROTTED_TOOLS ${tool_name} tool_is_rotted)
    if(NOT TARGET ${tool_name} AND tool_needs_gpu EQUAL -1 AND tool_is_rotted EQUAL -1)
        add_executable(${tool_name} EXCLUDE_FROM_ALL ${tool_src})
        target_include_directories(${tool_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${tool_name} PRIVATE hypergraph)
        add_dependencies(host_tools ${tool_name})
    endif()
endforeach()

