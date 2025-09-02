#ifndef HYPERGRAPH_WORK_STEALING_SCHEDULER_HPP
#define HYPERGRAPH_WORK_STEALING_SCHEDULER_HPP

#include <hypergraph/pattern_matching_tasks.hpp>
#include <lockfree_deque/deque.hpp>
#include <job_system/job_system.hpp>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <chrono>

namespace hypergraph {

/**
 * Enhanced worker data with work stealing capability.
 * Following HGMatch architecture: all tasks use LIFO for memory efficiency.
 */
struct WorkStealingWorkerData {
    // Single LIFO queue for all tasks - HGMatch architecture
    lockfree::Deque<job_system::JobPtr<PatternMatchingTaskType>> lifo_queue;
    
    std::thread thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> seeking_work{false};
    std::atomic<std::size_t> jobs_executed{0};
    std::atomic<std::size_t> jobs_stolen{0};
    std::atomic<std::size_t> steal_attempts{0};
    
    // Worker ID for stealing decisions
    std::size_t worker_id;
    
    WorkStealingWorkerData(std::size_t id) : worker_id(id) {}
    
    /**
     * Try to get work from local LIFO queue (HGMatch architecture).
     */
    std::optional<job_system::JobPtr<PatternMatchingTaskType>> try_get_local_work() {
        // LIFO: most recent first for memory efficiency
        return lifo_queue.try_pop_back();
    }
    
    /**
     * Try to steal work from another worker using "steal half" policy.
     */
    std::vector<job_system::JobPtr<PatternMatchingTaskType>> try_steal_from(WorkStealingWorkerData& victim) {
        std::vector<job_system::JobPtr<PatternMatchingTaskType>> stolen_work;
        steal_attempts.fetch_add(1);
        
        // HGMatch "steal half" from victim's LIFO queue
        std::vector<job_system::JobPtr<PatternMatchingTaskType>> temp_stolen;
        
        // Estimate queue size by attempting multiple pops
        constexpr std::size_t max_steals = 32;  // Reasonable upper bound
        for (std::size_t i = 0; i < max_steals; ++i) {
            auto job = victim.lifo_queue.try_pop_front();  // Steal from front (oldest work)
            if (!job) break;
            temp_stolen.push_back(std::move(*job));
        }
        
        if (temp_stolen.size() > 1) {
            // "Steal half" policy: take half, return the rest
            std::size_t keep_count = temp_stolen.size() / 2;
            
            // Return the second half to victim's queue
            for (std::size_t i = keep_count; i < temp_stolen.size(); ++i) {
                victim.lifo_queue.try_push_front(std::move(temp_stolen[i]));
            }
            
            // Keep the first half
            temp_stolen.resize(keep_count);
            jobs_stolen.fetch_add(temp_stolen.size());
            return temp_stolen;
        } else if (temp_stolen.size() == 1) {
            // If only one job, return it and don't steal
            victim.lifo_queue.try_push_front(std::move(temp_stolen[0]));
        }
        
        return std::vector<job_system::JobPtr<PatternMatchingTaskType>>{};
    }
    
    /**
     * Add work to LIFO queue (HGMatch architecture - all tasks are LIFO).
     */
    bool add_work(job_system::JobPtr<PatternMatchingTaskType> job) {
        return lifo_queue.try_push_back(std::move(job));
    }
};

/**
 * Work-stealing scheduler with differentiated policies for pattern matching tasks.
 */
class WorkStealingScheduler {
private:
    std::vector<std::unique_ptr<WorkStealingWorkerData>> workers_;
    std::atomic<std::size_t> round_robin_counter_{0};
    std::atomic<bool> is_running_{false};
    std::size_t num_workers_;
    
    // Work stealing parameters
    std::atomic<std::size_t> steal_attempts_since_success_{0};
    static constexpr std::size_t max_steal_attempts_before_yield_ = 100;
    
    /**
     * Main worker loop with work stealing.
     */
    void worker_loop(WorkStealingWorkerData* worker) {
        std::mt19937 rng(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_int_distribution<std::size_t> worker_dist(0, workers_.size() - 1);
        
        while (!worker->stop.load(std::memory_order_acquire)) {
            job_system::JobPtr<PatternMatchingTaskType> job;
            
            // Try to get local work first
            auto local_work = worker->try_get_local_work();
            if (local_work) {
                job = std::move(*local_work);
            } else {
                // No local work, try to steal
                worker->seeking_work.store(true, std::memory_order_release);
                
                bool found_work = false;
                std::size_t steal_attempts = 0;
                constexpr std::size_t max_steal_attempts = 4;  // Try a few workers
                
                while (steal_attempts < max_steal_attempts && !found_work) {
                    // Pick a random victim worker
                    std::size_t victim_id = worker_dist(rng);
                    if (victim_id == worker->worker_id) {
                        steal_attempts++;
                        continue;  // Don't steal from self
                    }
                    
                    auto& victim = *workers_[victim_id];
                    if (victim.seeking_work.load(std::memory_order_acquire)) {
                        steal_attempts++;
                        continue;  // Victim is also seeking work
                    }
                    
                    auto stolen_jobs = worker->try_steal_from(victim);
                    if (!stolen_jobs.empty()) {
                        // Execute first job immediately, queue the rest
                        job = std::move(stolen_jobs[0]);
                        
                        for (std::size_t i = 1; i < stolen_jobs.size(); ++i) {
                            worker->add_work(std::move(stolen_jobs[i]));
                        }
                        
                        found_work = true;
                    }
                    
                    steal_attempts++;
                }
                
                worker->seeking_work.store(false, std::memory_order_release);
                
                if (!found_work) {
                    // No work found, brief yield to avoid spinning
                    std::this_thread::yield();
                    continue;
                }
            }
            
            // Execute the job
            if (job) {
                job->execute();
                worker->jobs_executed.fetch_add(1);
            }
        }
    }
    
public:
    explicit WorkStealingScheduler(std::size_t num_workers = 0) 
        : num_workers_(num_workers == 0 ? std::thread::hardware_concurrency() : num_workers) {
        
        if (num_workers_ == 0) num_workers_ = 1;
        
        workers_.reserve(num_workers_);
        for (std::size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(std::make_unique<WorkStealingWorkerData>(i));
        }
    }
    
    ~WorkStealingScheduler() {
        shutdown();
    }
    
    void start() {
        if (is_running_.load()) return;
        
        for (auto& worker : workers_) {
            worker->thread = std::thread([this, worker = worker.get()] {
                worker_loop(worker);
            });
        }
        
        is_running_.store(true);
    }
    
    void shutdown() {
        if (!is_running_.load()) return;
        
        // Signal all workers to stop
        for (auto& worker : workers_) {
            worker->stop.store(true);
        }
        
        // Wait for all workers to finish
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }
        
        is_running_.store(false);
    }
    
    /**
     * Submit a job with automatic queue selection based on task type.
     */
    bool submit(job_system::JobPtr<PatternMatchingTaskType> job) {
        if (!is_running_.load()) return false;
        
        // Use round-robin for initial work distribution
        std::size_t worker_idx = round_robin_counter_.fetch_add(1) % workers_.size();
        
        return workers_[worker_idx]->add_work(std::move(job));
    }
    
    /**
     * Submit job to specific worker (for locality optimization).
     */
    bool submit_to_worker(std::size_t worker_id, job_system::JobPtr<PatternMatchingTaskType> job) {
        if (!is_running_.load() || worker_id >= workers_.size()) return false;
        
        return workers_[worker_id]->add_work(std::move(job));
    }
    
    /**
     * Template function to submit job with automatic creation.
     */
    template<typename F>
    bool submit_function(F&& func, PatternMatchingTaskType task_type, int priority = 0) {
        auto job = job_system::make_job(std::forward<F>(func), task_type, priority);
        return submit(std::move(job));
    }
    
    /**
     * Wait for all work to complete.
     */
    void wait_for_completion() {
        while (true) {
            bool has_work = false;
            
            for (const auto& worker : workers_) {
                if (!worker->lifo_queue.empty()) {
                    has_work = true;
                    break;
                }
            }
            
            if (!has_work) break;
            
            std::this_thread::yield();
        }
    }
    
    std::size_t get_num_workers() const {
        return workers_.size();
    }
    
    bool is_running() const {
        return is_running_.load();
    }
    
    /**
     * Get comprehensive statistics about work stealing and execution.
     */
    struct SchedulerStatistics {
        std::size_t total_jobs_executed;
        std::size_t total_jobs_stolen;
        std::size_t total_steal_attempts;
        std::size_t workers_seeking_work;
        double steal_success_rate;
        
        struct PerWorkerStats {
            std::size_t worker_id;
            std::size_t jobs_executed;
            std::size_t jobs_stolen;
            std::size_t steal_attempts;
            bool seeking_work;
        };
        
        std::vector<PerWorkerStats> per_worker_stats;
    };
    
    SchedulerStatistics get_statistics() const {
        SchedulerStatistics stats;
        stats.total_jobs_executed = 0;
        stats.total_jobs_stolen = 0;
        stats.total_steal_attempts = 0;
        stats.workers_seeking_work = 0;
        
        for (const auto& worker : workers_) {
            auto jobs_executed = worker->jobs_executed.load();
            auto jobs_stolen = worker->jobs_stolen.load();
            auto steal_attempts = worker->steal_attempts.load();
            bool seeking_work = worker->seeking_work.load();
            
            stats.total_jobs_executed += jobs_executed;
            stats.total_jobs_stolen += jobs_stolen;
            stats.total_steal_attempts += steal_attempts;
            if (seeking_work) stats.workers_seeking_work++;
            
            stats.per_worker_stats.push_back({
                worker->worker_id, jobs_executed, jobs_stolen, steal_attempts, seeking_work
            });
        }
        
        stats.steal_success_rate = (stats.total_steal_attempts > 0) ? 
            static_cast<double>(stats.total_jobs_stolen) / stats.total_steal_attempts : 0.0;
        
        return stats;
    }
    
    /**
     * Get current queue sizes for monitoring load balance.
     */
    struct QueueSizes {
        std::size_t worker_id;
        bool lifo_empty;
        bool seeking_work;
    };
    
    std::vector<QueueSizes> get_queue_sizes() const {
        std::vector<QueueSizes> sizes;
        
        for (const auto& worker : workers_) {
            sizes.push_back({
                worker->worker_id,
                worker->lifo_queue.empty(),
                worker->seeking_work.load()
            });
        }
        
        return sizes;
    }
};

} // namespace hypergraph

#endif // HYPERGRAPH_WORK_STEALING_SCHEDULER_HPP