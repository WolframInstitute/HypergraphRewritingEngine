cmake_minimum_required(VERSION 3.14)

# Create hypergraph library with canonicalization, pattern matching, rewriting, parallel rewriting and Wolfram Physics
add_library(hypergraph
    src/canonicalization.cpp
    src/pattern_matching.cpp
    src/pattern_matching_tasks.cpp
    src/rewriting.cpp
    src/wolfram_evolution.cpp
    src/wolfram_states.cpp
    src/thread_local_pool.cpp
    src/uniqueness_tree.cpp
    # unified architecture is header-only (no .cpp files needed)
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
    target_compile_definitions(hypergraph PRIVATE ENABLE_DEBUG_OUTPUT)
endif()

# Optional brute force isomorphism check (can be enabled via cmake -DENABLE_BRUTE_FORCE_ISOMORPHISM_CHECK=ON)
option(ENABLE_BRUTE_FORCE_ISOMORPHISM_CHECK "Enable brute force isomorphism verification" OFF)
if(ENABLE_BRUTE_FORCE_ISOMORPHISM_CHECK)
    target_compile_definitions(hypergraph PRIVATE ENABLE_BRUTE_FORCE_ISOMORPHISM_CHECK)
endif()

# Note: Examples, benchmarks, and tests are now built separately in their own directories:
# - examples/     - API usage examples (make examples)  
# - benchmarks/   - Performance tests (make benchmarks)
# - tests/        - Unit tests (make unified_tests)