cmake_minimum_required(VERSION 3.14)
project(HypergraphEngine VERSION 0.0.1 LANGUAGES CXX)

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
add_subdirectory(hypergraph)
add_subdirectory(wxf)

# Unified test suite
option(BUILD_TESTS "Build unified test suite" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Examples - API documentation and usage examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Benchmarks - Performance measurement tools
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarking)
    add_subdirectory(benchmarks)
endif()

# Optional: Build Mathematica paclet if Mathematica is found
option(BUILD_MATHEMATICA_PACLET "Build Mathematica paclet library" ON)
if(BUILD_MATHEMATICA_PACLET)
    add_subdirectory(paclet_source)
endif()

# Optional: Build GPU acceleration (requires CUDA)
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

# Optional: Build interactive visualization
option(BUILD_VISUALIZATION "Build interactive hypergraph visualization" OFF)
if(BUILD_VISUALIZATION)
    add_subdirectory(visualisation)
endif()