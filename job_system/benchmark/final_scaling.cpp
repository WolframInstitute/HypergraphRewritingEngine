#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <vector>
#include <thread>

enum class TestJobType { COMPUTE };

struct BenchResult {
    size_t threads;
    double jobs_per_sec;
    double speedup;
    double efficiency;
};

BenchResult benchmark_scaling(size_t threads, size_t num_jobs, bool use_sleep) {
    using namespace job_system;
    
    std::atomic<int> counter{0};
    JobSystem<TestJobType> js(threads);
    js.start();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_jobs; ++i) {
        auto job = make_job([&counter, use_sleep]() {
            if (use_sleep) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } else {
                // Light CPU work
                volatile int result = 0;
                for (int j = 0; j < 1000; ++j) {
                    result += j * j;
                }
            }
            counter.fetch_add(1);
        }, TestJobType::COMPUTE);
        js.submit(std::move(job));
    }
    
    while (counter.load() < static_cast<int>(num_jobs)) {
        std::this_thread::yield();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double jobs_per_sec = (num_jobs * 1000000.0) / duration_us;
    
    js.shutdown();
    
    return {threads, jobs_per_sec, 0.0, 0.0};
}

int main() {
    printf("=== JOB SYSTEM THREAD SCALING ANALYSIS ===\n\n");
    
    const size_t max_threads = std::thread::hardware_concurrency();
    std::vector<size_t> thread_counts = {1, 2, 4, 8};
    if (max_threads >= 16) thread_counts.push_back(16);
    if (max_threads > 16) thread_counts.push_back(max_threads);
    
    printf("Hardware threads: %zu\n\n", max_threads);
    
    // Test 1: CPU-bound
    printf("Test 1: CPU-bound jobs (100,000 jobs with computation)\n");
    printf("-------------------------------------------------------\n");
    printf("Threads  Jobs/sec      Speedup  Efficiency\n");
    
    std::vector<BenchResult> cpu_results;
    for (size_t threads : thread_counts) {
        auto result = benchmark_scaling(threads, 100000, false);
        cpu_results.push_back(result);
    }
    
    // Calculate speedup and efficiency
    double baseline_cpu = cpu_results[0].jobs_per_sec;
    for (auto& r : cpu_results) {
        r.speedup = r.jobs_per_sec / baseline_cpu;
        r.efficiency = (r.speedup / r.threads) * 100;
        printf("%7zu  %12.0f  %7.2fx  %9.1f%%\n", 
               r.threads, r.jobs_per_sec, r.speedup, r.efficiency);
    }
    
    // Test 2: I/O-bound
    printf("\nTest 2: I/O-bound jobs (10,000 jobs with 100μs sleep)\n");
    printf("------------------------------------------------------\n");
    printf("Threads  Jobs/sec      Speedup  Efficiency\n");
    
    std::vector<BenchResult> io_results;
    for (size_t threads : thread_counts) {
        auto result = benchmark_scaling(threads, 10000, true);
        io_results.push_back(result);
    }
    
    // Calculate speedup and efficiency
    double baseline_io = io_results[0].jobs_per_sec;
    for (auto& r : io_results) {
        r.speedup = r.jobs_per_sec / baseline_io;
        r.efficiency = (r.speedup / r.threads) * 100;
        printf("%7zu  %12.0f  %7.2fx  %9.1f%%\n", 
               r.threads, r.jobs_per_sec, r.speedup, r.efficiency);
    }
    
    // Summary
    printf("\n=== SCALING SUMMARY ===\n");
    printf("CPU-bound scaling at %zu threads: %.2fx speedup (%.1f%% efficiency)\n",
           cpu_results.back().threads, cpu_results.back().speedup, cpu_results.back().efficiency);
    printf("I/O-bound scaling at %zu threads: %.2fx speedup (%.1f%% efficiency)\n",
           io_results.back().threads, io_results.back().speedup, io_results.back().efficiency);
    
    return 0;
}