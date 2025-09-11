#ifndef JOB_SYSTEM_JOB_SYSTEM_HPP
#define JOB_SYSTEM_JOB_SYSTEM_HPP

#include <job_system/job.hpp>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>

namespace job_system {

template<typename JobType>
class JobSystem {
private:
    struct WorkerData {
        std::queue<JobPtr<JobType>> queue;
        std::mutex mutex;
        std::condition_variable cv;
        std::thread thread;
        std::atomic<bool> stop{false};
        std::atomic<size_t> jobs_executed{0};
        std::atomic<size_t> jobs_executing{0};
    };
    
    std::vector<std::unique_ptr<WorkerData>> workers_;
    std::atomic<size_t> round_robin_{0};
    std::atomic<bool> is_running_{false};
    size_t num_threads_;
    
    // Global work tracking for robust completion detection
    std::atomic<size_t> total_submitted_{0};
    std::atomic<size_t> total_completed_{0};
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;
    
    void worker_loop(WorkerData* data) {
        while (true) {
            JobPtr<JobType> job;
            
            {
                std::unique_lock<std::mutex> lock(data->mutex);
                data->cv.wait(lock, [data] { 
                    return data->stop.load() || !data->queue.empty(); 
                });
                
                if (data->stop.load() && data->queue.empty()) {
                    break;
                }
                
                if (!data->queue.empty()) {
                    job = std::move(data->queue.front());
                    data->queue.pop();
                }
            }
            
            if (job) {
                data->jobs_executing.fetch_add(1);
                job->execute();
                data->jobs_executing.fetch_sub(1);
                data->jobs_executed.fetch_add(1);
                
                // Notify completion for robust tracking
                total_completed_.fetch_add(1);
                completion_cv_.notify_all();
            }
        }
    }
    
public:
    explicit JobSystem(size_t num_threads = 0, size_t queue_capacity = 1024)
        : num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {
        (void)queue_capacity; // Unused - kept for API compatibility
        
        if (num_threads_ == 0) num_threads_ = 1;
        
        workers_.reserve(num_threads_);
        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back(std::make_unique<WorkerData>());
        }
    }
    
    ~JobSystem() {
        shutdown();
    }
    
    void start() {
        if (is_running_.load()) return;
        
        // Reset counters for new session
        total_submitted_.store(0);
        total_completed_.store(0);
        
        for (size_t i = 0; i < workers_.size(); ++i) {
            auto* worker = workers_[i].get();
            worker->thread = std::thread([this, worker] {
                worker_loop(worker);
            });
        }
        
        is_running_.store(true);
    }
    
    void shutdown() {
        if (!is_running_.load()) return;
        
        for (auto& worker : workers_) {
            {
                std::lock_guard<std::mutex> lock(worker->mutex);
                worker->stop.store(true);
            }
            worker->cv.notify_all();
        }
        
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }
        
        is_running_.store(false);
    }
    
    void submit(JobPtr<JobType> job, ScheduleMode mode = ScheduleMode::LIFO) {
        if (!is_running_.load()) {
            throw std::runtime_error("FastJobSystem is not running");
        }
        
        // Track submission for robust completion detection
        total_submitted_.fetch_add(1);
        
        size_t worker_idx = round_robin_.fetch_add(1) % workers_.size();
        auto* worker = workers_[worker_idx].get();
        
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->queue.push(std::move(job));
        }
        worker->cv.notify_one();
        
        // Ensure the job is visible to other threads
        std::atomic_thread_fence(std::memory_order_release);
    }
    
    void submit_to_worker(size_t worker_id, JobPtr<JobType> job, ScheduleMode mode = ScheduleMode::LIFO) {
        if (!is_running_.load()) {
            throw std::runtime_error("FastJobSystem is not running");
        }
        
        if (worker_id >= workers_.size()) {
            throw std::out_of_range("Invalid worker ID");
        }
        
        // Track submission for robust completion detection
        total_submitted_.fetch_add(1);
        
        auto* worker = workers_[worker_id].get();
        
        {
            std::lock_guard<std::mutex> lock(worker->mutex);
            worker->queue.push(std::move(job));
        }
        worker->cv.notify_one();
    }
    
    bool try_submit(JobPtr<JobType> job, ScheduleMode mode = ScheduleMode::LIFO) {
        if (!is_running_.load()) return false;
        
        submit(std::move(job), mode);
        return true;
    }
    
    template<typename F>
    void submit_function(F&& func, JobType job_type, int priority = 0, ScheduleMode mode = ScheduleMode::LIFO) {
        auto job = make_job(std::forward<F>(func), job_type, priority);
        submit(std::move(job), mode);
    }
    
    // No futures/promises - keep it simple
    
    void wait_for_completion() {
#ifdef JOBSYSTEM_DEBUG
        int debug_count = 0;
#endif
        
        while (true) {
            // Use condition variable to wait efficiently
            std::unique_lock<std::mutex> lock(completion_mutex_);
            
            // Wait until all submitted jobs are completed OR timeout for debug
            completion_cv_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                return total_submitted_.load() == total_completed_.load();
            });
            
            size_t submitted = total_submitted_.load();
            size_t completed = total_completed_.load();
            
            // Debug output every 1000 iterations
#ifdef JOBSYSTEM_DEBUG
            if (++debug_count % 1000 == 0) {
                printf("[JOB_SYSTEM DEBUG] wait_for_completion loop %d: submitted=%zu, completed=%zu, pending=%zu\n", 
                       debug_count, submitted, completed, submitted - completed);
                fflush(stdout);
            }
#endif
            
            // Check if all work is done
            if (submitted == completed) {
                // Double-check that no jobs are executing or queued
                bool truly_done = true;
                for (const auto& worker : workers_) {
                    std::lock_guard<std::mutex> worker_lock(worker->mutex);
                    if (!worker->queue.empty() || worker->jobs_executing.load() > 0) {
                        truly_done = false;
                        break;
                    }
                }
                
                if (truly_done) {
                    // Use memory fence to ensure we see all pending submissions
                    std::atomic_thread_fence(std::memory_order_acquire);
                    
                    // Re-check with proper memory ordering
                    submitted = total_submitted_.load();
                    completed = total_completed_.load();
                    
                    if (submitted == completed) {
                        // One final check - any executing jobs might be about to submit
                        bool any_executing = false;
                        for (const auto& worker : workers_) {
                            if (worker->jobs_executing.load() > 0) {
                                any_executing = true;
                                break;
                            }
                        }
                        
                        if (!any_executing) {
                            // Really done - no jobs executing, all submitted == completed
                            break;
                        }
                    }
                }
            }
        }
        
#ifdef JOBSYSTEM_DEBUG
        printf("[JOB_SYSTEM DEBUG] wait_for_completion FINISHED after %d iterations (submitted=%zu, completed=%zu)\n", 
               debug_count, total_submitted_.load(), total_completed_.load());
        fflush(stdout);
#endif
    }
    
    size_t get_num_workers() const {
        return workers_.size();
    }
    
    bool is_running() const {
        return is_running_.load();
    }
    
    
    struct SystemStatistics {
        size_t total_jobs_executed;
        size_t total_jobs_stolen;
        size_t total_jobs_deferred;
    };
    
    SystemStatistics get_statistics() const {
        size_t total_executed = 0;
        for (const auto& worker : workers_) {
            total_executed += worker->jobs_executed.load();
        }
        
        return SystemStatistics{
            total_executed,
            0,  // No work stealing in this implementation
            0   // No deferred jobs
        };
    }
    
    // Compatibility stubs (not used in fast implementation)
    void register_incompatibility(JobType type1, JobType type2) {}
    void register_compatibility_function(std::function<bool(JobType, JobType)> func) {}
    void clear_compatibility_rules() {}
};

// FastJobSystem is a separate implementation
// Use FastJobSystem<JobType> directly

} // namespace job_system

#endif // JOB_SYSTEM_JOB_SYSTEM_HPP