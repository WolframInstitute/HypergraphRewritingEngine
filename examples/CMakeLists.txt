cmake_minimum_required(VERSION 3.14)

# Examples
project(HypergraphExamples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# List of example source files (without .cpp extension)
set(EXAMPLE_SOURCES
    hypergraph_basics
    pattern_matching
    job_system_usage
    basic_evolution
    multiway_graph_usage
)

# List of example executables (with example_ prefix)
set(EXAMPLE_PROGRAMS)

# Create individual executables for each example
foreach(EXAMPLE_SRC ${EXAMPLE_SOURCES})
    set(EXAMPLE_TARGET "example_${EXAMPLE_SRC}")
    list(APPEND EXAMPLE_PROGRAMS ${EXAMPLE_TARGET})
    
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SRC}.cpp)
    
    # Link against all required libraries
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
        # Core libraries
        hypergraph
        job_system  
        lockfree_deque
        
        # System libraries
        Threads::Threads
    )
    
endforeach()

# Custom target to build all examples
add_custom_target(examples
    DEPENDS ${EXAMPLE_PROGRAMS}
    COMMENT "Building all examples..."
)

# Custom target to run all examples
add_custom_target(run_examples
    COMMENT "Running all examples..."
)

# Add run commands for each example
foreach(EXAMPLE ${EXAMPLE_PROGRAMS})
    add_custom_target(run_${EXAMPLE}
        COMMAND ${CMAKE_BINARY_DIR}/${EXAMPLE}
        DEPENDS ${EXAMPLE}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running ${EXAMPLE} example..."
    )
    
    # Make run_examples depend on individual run targets
    add_dependencies(run_examples run_${EXAMPLE})
endforeach()

# Print available examples
add_custom_target(list_examples
    COMMAND ${CMAKE_COMMAND} -E echo "Available examples:"
    COMMAND ${CMAKE_COMMAND} -E echo "  example_hypergraph_basics     - Basic hypergraph operations"  
    COMMAND ${CMAKE_COMMAND} -E echo "  example_pattern_matching      - Pattern matching with variables"
    COMMAND ${CMAKE_COMMAND} -E echo "  example_job_system_usage      - Task-based parallelism"
    COMMAND ${CMAKE_COMMAND} -E echo "  example_basic_evolution       - Simple Wolfram evolution"
    COMMAND ${CMAKE_COMMAND} -E echo "  example_multiway_graph_usage  - Multiway graph operations"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Usage:"
    COMMAND ${CMAKE_COMMAND} -E echo "  make examples                          - Build all examples"
    COMMAND ${CMAKE_COMMAND} -E echo "  make run_example_<name>               - Run specific example"
    COMMAND ${CMAKE_COMMAND} -E echo "  make run_examples                     - Run all examples"
    COMMENT "Listing available examples"
)