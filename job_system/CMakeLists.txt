cmake_minimum_required(VERSION 3.14)
project(JobSystem VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(job_system INTERFACE)
target_include_directories(job_system INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

find_package(Threads REQUIRED)
target_link_libraries(job_system INTERFACE 
    Threads::Threads
    lockfree_deque
)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmark executable
add_executable(job_system_benchmark benchmark/job_system_benchmark.cpp)
target_link_libraries(job_system_benchmark
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)


# Minimal benchmark for quick testing
add_executable(minimal_benchmark benchmark/minimal_benchmark.cpp)
target_link_libraries(minimal_benchmark
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)


# Debug benchmark
add_executable(debug_benchmark benchmark/debug_benchmark.cpp)
target_link_libraries(debug_benchmark
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)


# Debug multithread
add_executable(debug_multithread benchmark/debug_multithread.cpp)
target_link_libraries(debug_multithread
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)


# Test multithread
add_executable(test_multithread benchmark/test_multithread.cpp)
target_link_libraries(test_multithread
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)


# Debug test
add_executable(debug_test benchmark/debug_test.cpp)
target_link_libraries(debug_test
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)
target_compile_definitions(debug_test PRIVATE JOB_SYSTEM_DEBUG DEQUE_DEBUG)


# Fast test
add_executable(fast_test benchmark/fast_test.cpp)
target_link_libraries(fast_test
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)


# Profile benchmark
add_executable(profile_benchmark benchmark/profile_benchmark.cpp)
target_link_libraries(profile_benchmark
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)


# Simple profile
add_executable(simple_profile benchmark/simple_profile.cpp)
target_link_libraries(simple_profile
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)


# Callgrind benchmark
add_executable(callgrind_benchmark benchmark/callgrind_benchmark.cpp)
target_link_libraries(callgrind_benchmark
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)
target_compile_options(callgrind_benchmark PRIVATE -g -O2)


# Scaling benchmark
add_executable(scaling_benchmark benchmark/scaling_benchmark.cpp)
target_link_libraries(scaling_benchmark
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)
target_compile_options(scaling_benchmark PRIVATE -O2)



# Final scaling benchmark
add_executable(final_scaling benchmark/final_scaling.cpp)
target_link_libraries(final_scaling
    PRIVATE
    job_system
    lockfree_deque
    Threads::Threads
)
target_compile_options(final_scaling PRIVATE -O3)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(job_system INTERFACE -g -O0 -fsanitize=thread)
    target_link_options(job_system INTERFACE -fsanitize=thread)
endif()