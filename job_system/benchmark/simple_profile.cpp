#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>
#include <chrono>

enum class SimpleJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    printf("=== Simple Profile Test ===\n");
    
    const size_t num_jobs = 100;
    const size_t num_threads = 1;  // Start with single thread
    
    std::atomic<int> counter{0};
    
    JobSystem<SimpleJobType> js(num_threads);
    js.start();
    
    printf("Submitting %zu jobs...\n", num_jobs);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_jobs; ++i) {
        auto job = make_job([&counter, i]() {
            printf("Job %zu executing\n", i);
            counter.fetch_add(1);
        }, SimpleJobType::COMPUTE);
        js.submit(std::move(job));
    }
    
    printf("Waiting for completion...\n");
    js.wait_for_completion();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Completed %d jobs in %ld milliseconds\n", counter.load(), duration.count());
    
    js.shutdown();
    return 0;
}