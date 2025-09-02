#include <job_system/job_system.hpp>
#include <iostream>
#include <atomic>
#include <chrono>

enum class TestJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    std::cout << "=== Minimal Job System Benchmark ===\n";
    std::cout << "Hardware: " << std::thread::hardware_concurrency() << " cores\n\n";
    
    const size_t num_jobs = 1000;
    
    for (size_t threads : {1u, 2u, 4u, std::thread::hardware_concurrency()}) {
        std::atomic<int> counter{0};
        
        JobSystem<TestJobType> js(threads);
        js.start();
        
        auto start = std::chrono::steady_clock::now();
        
        for (size_t i = 0; i < num_jobs; ++i) {
            auto job = make_job([&counter]() {
                // Minimal work
                volatile int x = 42;
                counter.fetch_add(1);
            }, TestJobType::COMPUTE);
            js.submit(std::move(job));
        }
        
        js.wait_for_completion();
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        auto stats = js.get_statistics();
        js.shutdown();
        
        if (duration.count() > 0) {
            double jobs_per_sec = (num_jobs * 1000.0) / duration.count();
            std::cout << threads << " threads: " << duration.count() << "ms, "
                      << (int)jobs_per_sec << " jobs/sec, "
                      << stats.total_jobs_stolen << " stolen\n";
        } else {
            std::cout << threads << " threads: <1ms, very fast, "
                      << stats.total_jobs_stolen << " stolen\n";
        }
    }
    
    return 0;
}