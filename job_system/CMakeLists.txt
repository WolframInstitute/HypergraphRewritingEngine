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


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(job_system INTERFACE -g -O0 -fsanitize=thread)
    target_link_options(job_system INTERFACE -fsanitize=thread)
endif()