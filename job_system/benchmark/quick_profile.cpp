#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>

enum class TestJobType { COMPUTE };

__attribute__((noinline))
void work() {
    volatile int x = 42;
    x *= 2;
}

int main() {
    using namespace job_system;
    
    std::atomic<int> counter{0};
    JobSystem<TestJobType> js(1);
    js.start();
    
    // Just 10 jobs - fast test
    for (int i = 0; i < 10; ++i) {
        auto job = make_job([&counter]() {
            work();
            counter.fetch_add(1);
        }, TestJobType::COMPUTE);
        js.submit(std::move(job));
    }
    
    js.wait_for_completion();
    js.shutdown();
    
    printf("Done: %d jobs\n", counter.load());
    return 0;
}