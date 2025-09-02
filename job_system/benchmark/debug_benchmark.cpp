#include <job_system/job_system.hpp>
#include <iostream>
#include <atomic>
#include <chrono>
#include <thread>

enum class TestJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    std::cout << "=== Debugging Job System Benchmark ===\n";
    
    const size_t num_jobs = 10;
    const size_t num_threads = 1;
    
    std::atomic<int> counter{0};
    std::atomic<int> jobs_started{0};
    
    std::cout << "Creating JobSystem with " << num_threads << " threads...\n";
    JobSystem<TestJobType> js(num_threads);
    
    std::cout << "Starting JobSystem...\n";
    js.start();
    
    std::cout << "Submitting " << num_jobs << " jobs...\n";
    for (size_t i = 0; i < num_jobs; ++i) {
        auto job = make_job([&counter, &jobs_started, i]() {
            jobs_started.fetch_add(1);
            std::cout << "Job " << i << " executing on thread " 
                     << std::this_thread::get_id() << std::endl;
            
            // Light work
            volatile int x = 0;
            for (int j = 0; j < 100; ++j) {
                x += j;
            }
            
            counter.fetch_add(1);
            std::cout << "Job " << i << " completed. Total: " << counter.load() << std::endl;
        }, TestJobType::COMPUTE);
        
        js.submit(std::move(job));
        std::cout << "Submitted job " << i << std::endl;
    }
    
    std::cout << "All jobs submitted. Jobs started: " << jobs_started.load() 
              << ", completed: " << counter.load() << std::endl;
    
    std::cout << "Waiting for completion...\n";
    
    // Debug the wait_for_completion by implementing our own with logging
    int wait_iterations = 0;
    while (true) {
        auto stats = js.get_statistics();
        
        std::cout << "Wait iteration " << wait_iterations++ << ": ";
        std::cout << "executed=" << stats.total_jobs_executed
                  << ", counter=" << counter.load();
        
        // Worker stats not available in current implementation
        bool has_work = (counter.load() < num_jobs);
        
        // Note: active_jobs_by_type not available in standard JobSystem
        
        std::cout << " has_work=" << has_work << std::endl;
        
        if (!has_work && counter.load() == num_jobs) {
            std::cout << "All work completed!\n";
            break;
        }
        
        if (wait_iterations > 100) {
            std::cout << "ERROR: Wait timeout after 100 iterations!\n";
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "Final stats:\n";
    auto final_stats = js.get_statistics();
    std::cout << "  Total executed: " << final_stats.total_jobs_executed << std::endl;
    std::cout << "  Counter value: " << counter.load() << std::endl;
    std::cout << "  Jobs started: " << jobs_started.load() << std::endl;
    
    std::cout << "Shutting down...\n";
    js.shutdown();
    
    std::cout << "Debug completed.\n";
    return 0;
}