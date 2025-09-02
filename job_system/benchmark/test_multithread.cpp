#include <job_system/job_system.hpp>
#include <iostream>
#include <atomic>
#include <chrono>

enum class TestJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    std::cout << "=== Testing Multi-Thread JobSystem ===\n";
    
    const size_t num_jobs = 100;
    const size_t num_threads = 2;
    
    std::atomic<int> counter{0};
    
    JobSystem<TestJobType> js(num_threads);
    js.start();
    
    std::cout << "Submitting " << num_jobs << " jobs to " << num_threads << " threads...\n";
    
    auto start = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < num_jobs; ++i) {
        auto job = make_job([&counter]() {
            counter.fetch_add(1);
        }, TestJobType::COMPUTE);
        js.submit(std::move(job));
    }
    
    std::cout << "Jobs submitted. Using original wait_for_completion()...\n";
    
    // Use the original wait_for_completion method
    js.wait_for_completion();
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Completed! Counter: " << counter.load() << "/" << num_jobs << std::endl;
    std::cout << "Duration: " << duration.count() << "ms\n";
    
    auto stats = js.get_statistics();
    std::cout << "Total executed: " << stats.total_jobs_executed << std::endl;
    std::cout << "Total stolen: " << stats.total_jobs_stolen << std::endl;
    
    js.shutdown();
    
    return counter.load() == num_jobs ? 0 : 1;
}