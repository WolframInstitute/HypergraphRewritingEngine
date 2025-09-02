cmake_minimum_required(VERSION 3.14)
project(Hypergraph VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create hypergraph library with canonicalization, pattern matching, rewriting, parallel rewriting and Wolfram Physics
add_library(hypergraph 
    src/canonicalization.cpp
    src/pattern_matching.cpp
    src/pattern_matching_tasks.cpp
    src/rewriting.cpp
    src/wolfram_evolution.cpp
    src/thread_local_pool.cpp
)

# Include directories
target_include_directories(hypergraph PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enable position independent code for shared library linking
set_target_properties(hypergraph PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Link dependencies
find_package(Threads REQUIRED)
target_link_libraries(hypergraph PUBLIC 
    Threads::Threads
    job_system
    lockfree_deque
)

# Optional debug definitions (can be enabled via cmake -DDEBUG_OUTPUT=ON)
option(DEBUG_OUTPUT "Enable debug output" OFF)
if(DEBUG_OUTPUT)
    target_compile_definitions(hypergraph PRIVATE DEBUG)
endif()

# Note: Examples, benchmarks, and tests are now built separately in their own directories:
# - examples/     - API usage examples (make examples)  
# - benchmarks/   - Performance tests (make benchmarks)
# - tests/        - Unit tests (make unified_tests)