cmake_minimum_required(VERSION 3.14)
project(HypergraphEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    add_subdirectory(paclet)
endif()