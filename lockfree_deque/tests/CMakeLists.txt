include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

FetchContent_MakeAvailable(googletest)

add_executable(test_lockfree_deque test_deque.cpp)

target_link_libraries(test_lockfree_deque
    PRIVATE
    lockfree_deque
    GTest::gtest_main
    GTest::gtest
    Threads::Threads
)

# libatomic is a separate library only in the GNU toolchain; Apple's clang and
# MSVC provide atomics in the compiler runtime and have no 'atomic' to link.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(test_lockfree_deque PRIVATE atomic)
endif()

# Static linking for MinGW to avoid DLL dependencies during test discovery
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    target_link_options(test_lockfree_deque PRIVATE -static-libgcc -static-libstdc++)
endif()

include(GoogleTest)
gtest_discover_tests(test_lockfree_deque)