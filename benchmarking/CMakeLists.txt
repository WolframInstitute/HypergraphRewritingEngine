cmake_minimum_required(VERSION 3.14)

# Benchmarking framework - provides benchmark_main and headers
add_library(benchmark_framework STATIC benchmark_main.cpp)
target_include_directories(benchmark_framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_framework PUBLIC hypergraph wxf)

# Define source directory for git commands
string(REPLACE "\\" "\\\\" ESCAPED_SOURCE_DIR "${CMAKE_SOURCE_DIR}")

# Get compiler info
set(COMPILER_INFO "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
string(REPLACE "\\" "\\\\" ESCAPED_COMPILER_INFO "${COMPILER_INFO}")

target_compile_definitions(benchmark_framework PUBLIC
    BENCHMARK_SOURCE_DIR="${ESCAPED_SOURCE_DIR}"
    BENCHMARK_COMPILER_INFO="${ESCAPED_COMPILER_INFO}"
)
