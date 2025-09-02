cmake_minimum_required(VERSION 3.14)

# Benchmarks
project(HypergraphBenchmarks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# List of benchmark executables  
set(BENCHMARK_PROGRAMS
    performance_benchmark
)

# Create individual executables for each benchmark
foreach(BENCHMARK ${BENCHMARK_PROGRAMS})
    add_executable(${BENCHMARK} ${BENCHMARK}.cpp)
    
    # Link against all required libraries
    target_link_libraries(${BENCHMARK}
        PRIVATE
        # Core libraries
        hypergraph
        job_system  
        lockfree_deque
        
        # System libraries
        Threads::Threads
    )
    
    # Add optimization flags for benchmarks
    target_compile_options(${BENCHMARK} PRIVATE -O3 -DNDEBUG)
endforeach()

# Custom target to build all benchmarks
add_custom_target(benchmarks
    DEPENDS ${BENCHMARK_PROGRAMS}
    COMMENT "Building all benchmarks..."
)

# Custom target to run all benchmarks
add_custom_target(run_benchmarks
    COMMENT "Running all benchmarks..."
)

# Add run commands for each benchmark
foreach(BENCHMARK ${BENCHMARK_PROGRAMS})
    add_custom_target(run_${BENCHMARK}
        COMMAND ${CMAKE_BINARY_DIR}/benchmarks/${BENCHMARK}
        DEPENDS ${BENCHMARK}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running ${BENCHMARK} benchmark..."
    )
    
    # Individual benchmarks are not automatically run with run_benchmarks
    # (since they may take a long time)
endforeach()

# Custom target for comprehensive performance suite
add_custom_target(performance_suite
    DEPENDS run_performance_benchmark
    COMMENT "Running comprehensive performance benchmark suite..."
)

# Print available benchmarks
add_custom_target(list_benchmarks
    COMMAND ${CMAKE_COMMAND} -E echo "Available benchmarks:"
    COMMAND ${CMAKE_COMMAND} -E echo "  performance_benchmark - Comprehensive performance testing"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Benchmark suites:"
    COMMAND ${CMAKE_COMMAND} -E echo "  performance_suite     - Complete performance benchmark suite"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Usage:"
    COMMAND ${CMAKE_COMMAND} -E echo "  make benchmarks        - Build all benchmarks"
    COMMAND ${CMAKE_COMMAND} -E echo "  make run_<benchmark>   - Run specific benchmark"
    COMMAND ${CMAKE_COMMAND} -E echo "  make <suite>           - Run benchmark suite"
    COMMENT "Listing available benchmarks"
)