#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>

enum class TestJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    printf("=== Debug Test with 2 threads, 3 jobs ===\n");
    
    std::atomic<int> counter{0};
    
    JobSystem<TestJobType> js(2);
    js.start();
    
    printf("\nSubmitting 3 jobs...\n");
    for (int i = 0; i < 3; ++i) {
        auto job = make_job([&counter, i]() {
            printf("[Job %d] Executing\n", i);
            counter.fetch_add(1);
            printf("[Job %d] Done\n", i);
        }, TestJobType::COMPUTE);
        js.submit(std::move(job));
        printf("Submitted job %d\n", i);
    }
    
    printf("\nCalling wait_for_completion()...\n");
    js.wait_for_completion();
    
    printf("\nAll done! Counter = %d\n", counter.load());
    
    js.shutdown();
    return 0;
}