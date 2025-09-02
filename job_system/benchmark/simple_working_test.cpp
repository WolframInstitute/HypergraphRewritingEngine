#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <vector>

enum class TestJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    printf("=== Thread Scaling Report ===\n");
    
    std::vector<size_t> thread_counts = {1, 2, 4, 8};
    const size_t num_jobs = 100;
    
    printf("Jobs per test: %zu\n\n", num_jobs);
    printf("Threads  Time(ms)  Jobs/sec  Speedup\n");
    printf("-------  --------  --------  -------\n");
    
    double baseline_rate = 0;
    
    for (size_t threads : thread_counts) {
        std::atomic<int> counter{0};
        
        JobSystem<TestJobType> js(threads);
        js.start();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Submit all jobs
        for (size_t i = 0; i < num_jobs; ++i) {
            auto job = make_job([&counter]() {
                // Minimal work
                volatile int x = 42;
                x *= 2;
                counter.fetch_add(1);
            }, TestJobType::COMPUTE);
            js.submit(std::move(job));
        }
        
        // Wait for completion
        while (counter.load() < num_jobs) {
            std::this_thread::yield();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double jobs_per_sec = (num_jobs * 1000.0) / duration.count();
        if (threads == 1) baseline_rate = jobs_per_sec;
        double speedup = jobs_per_sec / baseline_rate;
        
        printf("%7zu  %8ld  %8.0f  %7.2fx\n", 
               threads, duration.count(), jobs_per_sec, speedup);
        
        js.shutdown();
    }
    
    return 0;
}