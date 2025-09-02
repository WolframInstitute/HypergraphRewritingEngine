#include <job_system/job_system.hpp>
#include <iostream>
#include <atomic>
#include <chrono>
#include <thread>

enum class TestJobType { COMPUTE };

int main() {
    using namespace job_system;
    
    std::cout << "=== Debugging Multi-Thread Issue ===\n";
    
    const size_t num_jobs = 5;
    const size_t num_threads = 2;  // Test with just 2 threads
    
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
            std::cout << "Job " << i << " starting on thread " 
                     << std::this_thread::get_id() << std::endl;
            
            // Very light work
            volatile int x = 42;
            
            counter.fetch_add(1);
            std::cout << "Job " << i << " completed. Total: " << counter.load() << std::endl;
        }, TestJobType::COMPUTE);
        
        js.submit(std::move(job));
        std::cout << "Submitted job " << i << std::endl;
    }
    
    std::cout << "\nAll jobs submitted. Checking worker status...\n";
    
    // Check each worker's status
    for (int iter = 0; iter < 10; ++iter) {
        std::cout << "\nIteration " << iter << ":\n";
        
        auto stats = js.get_statistics();
        std::cout << "  Total executed: " << stats.total_jobs_executed << std::endl;
        std::cout << "  Counter: " << counter.load() << std::endl;
        std::cout << "  Jobs started: " << jobs_started.load() << std::endl;
        
        // Worker-specific stats not available in current implementation
        
        // Note: active_jobs_by_type not available in standard JobSystem
        
        if (counter.load() >= num_jobs) {
            std::cout << "\nAll jobs completed!\n";
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\nAttempting shutdown...\n";
    js.shutdown();
    
    std::cout << "Debug completed.\n";
    return 0;
}