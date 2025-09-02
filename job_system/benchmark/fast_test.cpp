#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>
#include <chrono>

enum class TestJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    printf("=== Fast Test with Multiple Thread Counts ===\n");
    
    std::vector<size_t> thread_counts = {1, 2, 4, 8};
    const size_t num_jobs = 100;
    
    for (size_t num_threads : thread_counts) {
        std::atomic<int> counter{0};
        
        JobSystem<TestJobType> js(num_threads);
        js.start();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_jobs; ++i) {
            auto job = make_job([&counter]() {
                counter.fetch_add(1);
            }, TestJobType::COMPUTE);
            js.submit(std::move(job));
        }
        
        js.wait_for_completion();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("Threads: %zu, Jobs: %d, Time: %ldms, Counter: %d\n", 
               num_threads, num_jobs, duration.count(), counter.load());
        
        js.shutdown();
    }
    
    return 0;
}