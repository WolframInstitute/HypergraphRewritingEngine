cmake_minimum_required(VERSION 3.14)

add_library(job_system STATIC
    src/job_pool.cpp
    src/affinity.cpp
    src/park.cpp
    src/capacity.cpp
)
target_include_directories(job_system PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# The paclet is a shared library and links this one in, so its objects must be
# position independent -- as hypergraph's and wxf's are, for the same reason.
set_target_properties(job_system PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

find_package(Threads REQUIRED)
target_link_libraries(job_system PUBLIC
    Threads::Threads
    lockfree_deque
)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


# Sanitizers are now controlled at the top level via ENABLE_TSAN/ENABLE_ASAN options
# See main CMakeLists.txt for configuration