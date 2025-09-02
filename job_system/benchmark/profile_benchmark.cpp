#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>
#include <chrono>

enum class ProfileJobType { COMPUTE };

// Simple CPU work to measure overhead against
void actual_work() {
    volatile int result = 0;
    for (int i = 0; i < 1000; ++i) {
        result += i * i;
    }
}

int main() {
    using namespace job_system;
    
    printf("=== Job System Profiling Benchmark ===\n");
    
    const size_t num_jobs = 10000;
    const size_t num_threads = 4;
    
    std::atomic<int> counter{0};
    
    JobSystem<ProfileJobType> js(num_threads);
    js.start();
    
    printf("Submitting %zu jobs with %zu threads...\n", num_jobs, num_threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_jobs; ++i) {
        auto job = make_job([&counter]() {
            actual_work();  // The actual work we want to measure
            counter.fetch_add(1);
        }, ProfileJobType::COMPUTE);
        js.submit(std::move(job));
    }
    
    js.wait_for_completion();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Completed %d jobs in %ld microseconds\n", counter.load(), duration.count());
    printf("Average time per job: %.2f microseconds\n", 
           static_cast<double>(duration.count()) / num_jobs);
    
    js.shutdown();
    
    // Also benchmark the raw work function for comparison
    printf("\nBenchmarking raw work function...\n");
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_jobs; ++i) {
        actual_work();
    }
    end = std::chrono::high_resolution_clock::now();
    auto raw_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Raw work completed in %ld microseconds\n", raw_duration.count());
    printf("Average time per raw work: %.2f microseconds\n", 
           static_cast<double>(raw_duration.count()) / num_jobs);
    
    double overhead = static_cast<double>(duration.count()) / raw_duration.count();
    printf("Overhead factor: %.2fx\n", overhead);
    
    return 0;
}