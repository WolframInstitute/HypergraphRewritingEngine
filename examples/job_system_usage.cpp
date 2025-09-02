/**
 * Job System Usage Example
 * 
 * Demonstrates task-based parallelism:
 * - Creating and submitting jobs
 * - Work distribution across threads
 * - Synchronization and statistics
 */

#include <job_system/job_system.hpp>
#include <lockfree_deque/deque.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>

enum class TaskType {
    TASK1,
    TASK2,
    TASK3,
    TASK4
};

int main() {
    std::cout << "=== Job System Usage Example ===\n\n";
    
    // Create job system with worker threads
    const std::size_t num_workers = std::min(4u, std::thread::hardware_concurrency());
    std::cout << "Creating job system with " << num_workers << " worker threads\n\n";
    
    job_system::JobSystem<TaskType> job_system(num_workers);
    job_system.start();
    
    // Example 1: Basic job submission
    std::cout << "=== Example 1: Basic Job Submission ===\n";
    std::atomic<int> counter1{0};
    
    // Submit several jobs
    for (int i = 0; i < 10; ++i) {
        job_system.submit_function([&counter1, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::cout << "Job " << i << " executing on thread " 
                      << std::this_thread::get_id() << "\n";
            counter1.fetch_add(1);
        }, TaskType::TASK1);
    }
    
    job_system.wait_for_completion();
    std::cout << "Completed " << counter1.load() << " jobs\n\n";
    
    // Example 2: Different task types
    std::cout << "=== Example 2: Mixed Task Types ===\n";
    std::atomic<int> task1_count{0};
    std::atomic<int> task2_count{0};
    
    for (int i = 0; i < 5; ++i) {
        job_system.submit_function([&task1_count]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            task1_count.fetch_add(1);
        }, TaskType::TASK1);
        
        job_system.submit_function([&task2_count]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            task2_count.fetch_add(1);
        }, TaskType::TASK2);
    }
    
    job_system.wait_for_completion();
    std::cout << "TASK1 completed: " << task1_count.load() << "\n";
    std::cout << "TASK2 completed: " << task2_count.load() << "\n\n";
    
    // Example 3: Job system with shared data structures
    std::cout << "=== Example 3: Shared Lock-Free Data Structure ===\n";
    lockfree::Deque<int> shared_deque;
    std::atomic<int> producer_count{0};
    std::atomic<int> consumer_count{0};
    
    // Submit producer jobs
    for (int i = 0; i < 20; ++i) {
        job_system.submit_function([&shared_deque, &producer_count, i]() {
            shared_deque.push_back(i * 10);
            producer_count.fetch_add(1);
        }, TaskType::TASK1);
    }
    
    // Submit consumer jobs
    for (int i = 0; i < 20; ++i) {
        job_system.submit_function([&shared_deque, &consumer_count]() {
            while (true) {
                auto item = shared_deque.try_pop_front();
                if (item.has_value()) {
                    consumer_count.fetch_add(1);
                    break;
                }
                std::this_thread::yield();
            }
        }, TaskType::TASK2);
    }
    
    job_system.wait_for_completion();
    std::cout << "Items produced: " << producer_count.load() << "\n";
    std::cout << "Items consumed: " << consumer_count.load() << "\n";
    std::cout << "Items remaining in deque: " << (shared_deque.empty() ? "0" : "some") << "\n\n";
    
    // Example 4: Performance timing
    std::cout << "=== Example 4: Performance Measurement ===\n";
    const int heavy_work_jobs = 50;
    std::atomic<int> work_completed{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < heavy_work_jobs; ++i) {
        job_system.submit_function([&work_completed]() {
            // Simulate some CPU-intensive work
            volatile int sum = 0;
            for (int j = 0; j < 100000; ++j) {
                sum += j;
            }
            work_completed.fetch_add(1);
        }, TaskType::TASK3);
    }
    
    job_system.wait_for_completion();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Completed " << work_completed.load() << " heavy jobs in " 
              << duration.count() << "ms\n";
    
    // Get job system statistics
    auto stats = job_system.get_statistics();
    std::cout << "\nJob System Statistics:\n";
    std::cout << "  Total jobs executed: " << stats.total_jobs_executed << "\n";
    std::cout << "  Total jobs stolen: " << stats.total_jobs_stolen << "\n";
    std::cout << "  Total jobs deferred: " << stats.total_jobs_deferred << "\n";
    
    std::cout << "\n=== Example completed successfully ===\n";
    return 0;
}