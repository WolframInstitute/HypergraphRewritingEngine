cmake_minimum_required(VERSION 3.14)

add_library(lockfree_deque INTERFACE)
target_include_directories(lockfree_deque INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

find_package(Threads REQUIRED)
target_link_libraries(lockfree_deque INTERFACE Threads::Threads)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Sanitizers are now controlled at the top level via ENABLE_TSAN/ENABLE_ASAN options
# See main CMakeLists.txt for configuration