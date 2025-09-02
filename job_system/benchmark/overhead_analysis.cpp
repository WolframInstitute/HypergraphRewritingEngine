#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>
#include <chrono>

enum class TestJobType { COMPUTE };

// Measurable work function 
__attribute__((noinline))
void actual_work() {
    volatile int result = 0;
    for (int i = 0; i < 100; ++i) {
        result += i * i;
    }
}

int main() {
    using namespace job_system;
    
    const size_t num_jobs = 1000;
    
    // Test with job system
    {
        std::atomic<int> counter{0};
        JobSystem<TestJobType> js(1);  // Single thread to avoid synchronization overhead
        js.start();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_jobs; ++i) {
            auto job = make_job([&counter]() {
                actual_work();
                counter.fetch_add(1);
            }, TestJobType::COMPUTE);
            js.submit(std::move(job));
        }
        
        js.wait_for_completion();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printf("Job system: %d jobs in %ld microseconds (%.2f us/job)\n", 
               counter.load(), duration.count(), 
               static_cast<double>(duration.count()) / num_jobs);
        
        js.shutdown();
    }
    
    // Test raw function calls
    {
        std::atomic<int> counter{0};
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_jobs; ++i) {
            actual_work();
            counter.fetch_add(1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printf("Raw calls: %d jobs in %ld microseconds (%.2f us/job)\n", 
               counter.load(), duration.count(), 
               static_cast<double>(duration.count()) / num_jobs);
    }
    
    return 0;
}