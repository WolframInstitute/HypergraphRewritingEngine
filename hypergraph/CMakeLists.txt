cmake_minimum_required(VERSION 3.14)

# Create hypergraph library
add_library(hypergraph
    src/canonicalization.cpp
    src/ir_canonicalization.cpp
    src/rewriter.cpp
    src/causal_graph.cpp
    src/parallel_evolution.cpp
    src/hypergraph.cpp
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

# Visualization event streaming (conditional on viz_events target existence)
# When HYPERGRAPH_ENABLE_VISUALIZATION is defined, hypergraph uses VizEventSink
# which requires linking to viz_events library for the static variable definitions
if(TARGET viz_events)
    target_link_libraries(hypergraph PUBLIC viz_events)
    # Enable the VIZ_EMIT_* macros so hypergraph actually emits events
    target_compile_definitions(hypergraph PUBLIC HYPERGRAPH_ENABLE_VISUALIZATION=1)
endif()

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