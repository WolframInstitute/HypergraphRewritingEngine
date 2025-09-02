#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>

enum class TestJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    printf("Starting minimal test...\n");
    
    std::atomic<int> counter{0};
    JobSystem<TestJobType> js(1);
    js.start();
    
    printf("Submitting single job...\n");
    auto job = make_job([&counter]() {
        printf("Job executing...\n");
        counter.fetch_add(1);
        printf("Job done.\n");
    }, TestJobType::COMPUTE);
    js.submit(std::move(job));
    
    printf("Waiting for completion...\n");
    js.wait_for_completion();
    
    printf("Counter: %d\n", counter.load());
    js.shutdown();
    printf("Test complete.\n");
    return 0;
}