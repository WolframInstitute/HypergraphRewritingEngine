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
    atomic
)


include(GoogleTest)
gtest_discover_tests(test_lockfree_deque)