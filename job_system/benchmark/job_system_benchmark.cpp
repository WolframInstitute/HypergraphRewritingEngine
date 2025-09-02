#include <job_system/job_system.hpp>
#include <iostream>
#include <atomic>
#include <chrono>
#include <vector>
#include <iomanip>
#include <thread>

enum class BenchmarkJobType {
    COMPUTE
};

struct BenchmarkResult {
    size_t num_threads;
    size_t num_jobs;
    double duration_ms;
    double jobs_per_second;
    double speedup;
    double efficiency;
    size_t jobs_stolen;
};

class JobSystemBenchmark {
public:
    static void light_cpu_work() {
        // Very light CPU work for fast testing
        volatile int result = 0;
        for (int i = 0; i < 10; ++i) {
            result += i;
        }
    }
    
    static BenchmarkResult run_benchmark(size_t num_threads, size_t num_jobs) {
        using namespace job_system;
        
        std::atomic<size_t> jobs_completed{0};
        
        JobSystem<BenchmarkJobType> job_system(num_threads);
        job_system.start();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_jobs; ++i) {
            auto job = make_job([&jobs_completed]() {
                light_cpu_work();
                jobs_completed.fetch_add(1, std::memory_order_relaxed);
            }, BenchmarkJobType::COMPUTE);
            
            job_system.submit(std::move(job));
        }
        
        // Use the original wait_for_completion which works correctly
        job_system.wait_for_completion();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        auto stats = job_system.get_statistics();
        job_system.shutdown();
        
        double duration_ms = duration.count() / 1000.0;
        double jobs_per_second = (num_jobs * 1000.0) / duration_ms;
        
        return BenchmarkResult{
            num_threads,
            num_jobs,
            duration_ms,
            jobs_per_second,
            0.0, // speedup calculated later
            0.0, // efficiency calculated later
            stats.total_jobs_stolen
        };
    }
    
    static void run_scaling_benchmark() {
        const size_t num_jobs = 200;  // Small count for fast results
        const size_t max_threads = std::thread::hardware_concurrency();
        
        std::cout << "=== Job System Thread Scaling Benchmark ===\n";
        std::cout << "Hardware concurrency: " << max_threads << " threads\n";
        std::cout << "Jobs per test: " << num_jobs << "\n";
        std::cout << "Job work: Light CPU computation\n\n";
        
        // Test different thread counts
        std::vector<size_t> thread_counts = {1, 2, 4, 8};
        if (max_threads > 8) {
            thread_counts.push_back(max_threads);
        }
        
        std::vector<BenchmarkResult> results;
        
        std::cout << "Running benchmarks...\n";
        for (size_t thread_count : thread_counts) {
            std::cout << "Testing " << thread_count << " threads... ";
            std::cout.flush();
            
            try {
                auto result = run_benchmark(thread_count, num_jobs);
                results.push_back(result);
                
                std::cout << std::fixed << std::setprecision(0) 
                          << result.jobs_per_second << " jobs/sec\n";
            } catch (const std::exception& e) {
                std::cout << "FAILED: " << e.what() << "\n";
            }
        }
        
        // Calculate speedups and efficiency
        if (!results.empty()) {
            double baseline_performance = results[0].jobs_per_second;
            for (auto& result : results) {
                result.speedup = result.jobs_per_second / baseline_performance;
                result.efficiency = result.speedup / result.num_threads;
            }
        }
        
        // Print results table
        std::cout << "\n=== Results Summary ===\n";
        std::cout << std::left 
                  << std::setw(8) << "Threads"
                  << std::setw(12) << "Duration(ms)"
                  << std::setw(15) << "Jobs/Second"
                  << std::setw(10) << "Speedup"
                  << std::setw(12) << "Efficiency"
                  << std::setw(12) << "Jobs Stolen"
                  << "\n";
        std::cout << std::string(75, '-') << "\n";
        
        for (const auto& result : results) {
            std::cout << std::left 
                      << std::setw(8) << result.num_threads
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.duration_ms
                      << std::setw(15) << std::setprecision(0) << result.jobs_per_second
                      << std::setw(10) << std::setprecision(2) << result.speedup << "x"
                      << std::setw(12) << std::setprecision(1) << (result.efficiency * 100) << "%"
                      << std::setw(12) << result.jobs_stolen
                      << "\n";
        }
        
        // Analysis
        std::cout << "\n=== Analysis ===\n";
        
        auto best_result = *std::max_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.jobs_per_second < b.jobs_per_second;
            });
        
        std::cout << "Best performance: " << std::fixed << std::setprecision(0)
                  << best_result.jobs_per_second << " jobs/sec with " 
                  << best_result.num_threads << " threads\n";
        
        auto last_result = results.back();
        std::cout << "Max threads (" << last_result.num_threads << "): " 
                  << std::setprecision(2) << last_result.speedup << "x speedup, "
                  << std::setprecision(1) << (last_result.efficiency * 100)
                  << "% efficiency\n";
        
        // Check scaling quality
        if (last_result.efficiency > 0.7) {
            std::cout << "Scaling quality: Excellent (>70% efficiency)\n";
        } else if (last_result.efficiency > 0.5) {
            std::cout << "Scaling quality: Good (>50% efficiency)\n";
        } else {
            std::cout << "Scaling quality: Poor (<50% efficiency)\n";
        }
        
        std::cout << "Work stealing events: " << last_result.jobs_stolen 
                  << " (indicates load balancing activity)\n";
    }
};

int main() {
    try {
        JobSystemBenchmark::run_scaling_benchmark();
        std::cout << "\nBenchmark completed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}