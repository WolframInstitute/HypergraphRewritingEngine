#include <job_system/job_system.hpp>
#include <iostream>
#include <atomic>
#include <vector>
#include <chrono>

enum class GameJobType {
    GRAPHICS,
    PHYSICS,
    AI,
    AUDIO,
    NETWORKING
};

int main() {
    using namespace job_system;
    
    std::cout << "Job System Basic Usage Example\n";
    std::cout << "==============================\n\n";
    
    // Create job system with default number of threads (CPU cores)
    JobSystem<GameJobType> job_system;
    
    // Set up incompatible job types (e.g., graphics and physics can't run together)
    job_system.register_incompatibility(GameJobType::GRAPHICS, GameJobType::PHYSICS);
    job_system.register_incompatibility(GameJobType::AUDIO, GameJobType::NETWORKING);
    
    std::cout << "Starting job system with " << job_system.get_num_workers() << " worker threads\n";
    job_system.start();
    
    // Example 1: Basic job submission
    std::cout << "\n1. Basic Job Execution:\n";
    {
        std::atomic<int> counter{0};
        
        for (int i = 0; i < 10; ++i) {
            auto job = make_job([&counter, i]() {
                std::cout << "  Job " << i << " executing on thread " 
                         << std::this_thread::get_id() << "\n";
                counter.fetch_add(1);
            }, GameJobType::GRAPHICS);
            
            job_system.submit(std::move(job));
        }
        
        job_system.wait_for_completion();
        std::cout << "  Completed " << counter.load() << " jobs\n";
    }
    
    // Example 2: LIFO vs FIFO scheduling
    std::cout << "\n2. LIFO Scheduling (cache-friendly for related tasks):\n";
    {
        std::atomic<int> task_counter{0};
        
        for (int i = 0; i < 5; ++i) {
            auto job = make_job([&task_counter, i]() {
                std::cout << "  LIFO Task " << i << " processed\n";
                task_counter.fetch_add(1);
            }, GameJobType::AI);
            
            // Submit to worker 0 with LIFO scheduling
            job_system.submit_to_worker(0, std::move(job), ScheduleMode::LIFO);
        }
        
        job_system.wait_for_completion();
    }
    
    std::cout << "\n3. FIFO Scheduling (fair processing):\n";
    {
        std::atomic<int> task_counter{0};
        
        for (int i = 0; i < 5; ++i) {
            auto job = make_job([&task_counter, i]() {
                std::cout << "  FIFO Task " << i << " processed\n";
                task_counter.fetch_add(1);
            }, GameJobType::AI);
            
            // Submit to worker 0 with FIFO scheduling
            job_system.submit_to_worker(0, std::move(job), ScheduleMode::FIFO);
        }
        
        job_system.wait_for_completion();
    }
    
    // Example 3: Job type compatibility
    std::cout << "\n4. Job Type Compatibility (graphics and physics are incompatible):\n";
    {
        std::atomic<bool> graphics_running{false};
        std::atomic<bool> physics_running{false};
        
        auto graphics_job = make_job([&]() {
            graphics_running.store(true);
            std::cout << "  Graphics job started\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            // Verify physics is not running concurrently
            if (physics_running.load()) {
                std::cout << "  ERROR: Physics running concurrently with graphics!\n";
            } else {
                std::cout << "  Graphics job completed safely\n";
            }
            graphics_running.store(false);
        }, GameJobType::GRAPHICS);
        
        auto physics_job = make_job([&]() {
            physics_running.store(true);
            std::cout << "  Physics job started\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            // Verify graphics is not running concurrently
            if (graphics_running.load()) {
                std::cout << "  ERROR: Graphics running concurrently with physics!\n";
            } else {
                std::cout << "  Physics job completed safely\n";
            }
            physics_running.store(false);
        }, GameJobType::PHYSICS);
        
        // Submit both jobs - they should execute sequentially
        job_system.submit(std::move(graphics_job));
        job_system.submit(std::move(physics_job));
        
        job_system.wait_for_completion();
    }
    
    // Example 4: Jobs with return values using futures
    std::cout << "\n5. Jobs with Future Return Values:\n";
    {
        auto future1 = job_system.submit_with_future([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return 42;
        }, GameJobType::AI);
        
        auto future2 = job_system.submit_with_future([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            return std::string("Hello from job system!");
        }, GameJobType::NETWORKING);
        
        std::cout << "  Future 1 result: " << future1.get() << "\n";
        std::cout << "  Future 2 result: " << future2.get() << "\n";
    }
    
    // Example 5: Performance statistics
    std::cout << "\n6. Performance Statistics:\n";
    {
        const int num_jobs = 1000;
        std::atomic<int> counter{0};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_jobs; ++i) {
            job_system.submit_function([&counter]() {
                counter.fetch_add(1);
            }, GameJobType::GRAPHICS);
        }
        
        job_system.wait_for_completion();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        auto stats = job_system.get_statistics();
        
        std::cout << "  Executed " << num_jobs << " jobs in " << duration.count() << "ms\n";
        std::cout << "  Total jobs executed: " << stats.total_jobs_executed << "\n";
        std::cout << "  Total jobs stolen: " << stats.total_jobs_stolen << "\n";
        std::cout << "  Total jobs deferred: " << stats.total_jobs_deferred << "\n";
        
        double throughput = (num_jobs * 1000.0) / duration.count();
        std::cout << "  Throughput: " << throughput << " jobs/second\n";
    }
    
    std::cout << "\nShutting down job system...\n";
    job_system.shutdown();
    
    std::cout << "Example completed successfully!\n";
    return 0;
}