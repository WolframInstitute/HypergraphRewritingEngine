#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <vector>
#include <thread>

enum class BenchJobType { COMPUTE };

__attribute__((noinline))
void cpu_work() {
    volatile int result = 0;
    for (int i = 0; i < 1000; ++i) {
        result += i * i;
    }
}

struct BenchResult {
    size_t threads;
    size_t jobs_per_sec;
    double speedup;
    double efficiency;
};

BenchResult benchmark_threads(size_t num_threads, size_t num_jobs) {
    using namespace job_system;
    
    std::atomic<int> counter{0};
    JobSystem<BenchJobType> js(num_threads);
    js.start();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_jobs; ++i) {
        auto job = make_job([&counter]() {
            cpu_work();
            counter.fetch_add(1);
        }, BenchJobType::COMPUTE);
        js.submit(std::move(job));
    }
    
    js.wait_for_completion();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double jobs_per_sec = (num_jobs * 1000000.0) / duration.count();
    
    js.shutdown();
    
    return {num_threads, static_cast<size_t>(jobs_per_sec), 0.0, 0.0};
}

int main() {
    printf("=== Job System Thread Scaling Benchmark ===\n");
    
    const size_t num_jobs = 2000;  // Moderate count for good measurement
    const size_t max_threads = std::thread::hardware_concurrency();
    
    printf("Hardware concurrency: %zu threads\n", max_threads);
    printf("Jobs per test: %zu\n\n", num_jobs);
    
    std::vector<size_t> thread_counts = {1};  // Start with single thread only
    
    std::vector<BenchResult> results;
    
    for (size_t threads : thread_counts) {
        printf("Testing %zu threads... ", threads);
        fflush(stdout);
        
        auto result = benchmark_threads(threads, num_jobs);
        results.push_back(result);
        
        printf("%zu jobs/sec\n", result.jobs_per_sec);
    }
    
    // Calculate speedup and efficiency
    if (!results.empty()) {
        double baseline = results[0].jobs_per_sec;
        for (auto& result : results) {
            result.speedup = result.jobs_per_sec / baseline;
            result.efficiency = result.speedup / result.threads;
        }
    }
    
    printf("\n=== Scaling Analysis ===\n");
    printf("Threads  Jobs/Sec   Speedup   Efficiency\n");
    printf("-------  ---------  --------  ----------\n");
    
    for (const auto& result : results) {
        printf("%7zu  %9zu  %8.2fx  %9.1f%%\n", 
               result.threads, result.jobs_per_sec, 
               result.speedup, result.efficiency * 100);
    }
    
    return 0;
}