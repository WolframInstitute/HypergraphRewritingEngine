#pragma once

#include <job_system/job.hpp>
#include <job_system/work_stealing_deque.hpp>
#include <lockfree_deque/deque.hpp>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <random>
#include <string>

namespace job_system {

// Error types that can occur during job execution
enum class ErrorType {
    None = 0,
    OutOfMemory,   // std::bad_alloc caught
    Aborted,       // AbortedException caught (user requested abort)
    Exception,     // std::exception caught
    Unhandled      // Non-std::exception type caught
};

// Lock-free work-stealing scheduler.
//
// Each worker owns a Chase-Lev deque: it pushes/pops its own bottom (so nested jobs
// submitted from a running job stay node-local and lock-free), and idle workers steal
// the top of others' deques. External submissions (from non-worker threads) and local
// overflow go to a shared lock-free injector. Idle workers park on a condition variable
// (the only lock, off the hot path); all queue operations are lock-free. The design is
// architecture-neutral (single-word atomics, no double-width CAS, no arch-specific code).
template<typename JobType>
class JobSystem {
private:
    using JobRaw = Job<JobType>*;

    struct WorkerData {
        WorkStealingDeque<JobRaw> deque;
        std::thread thread;
        std::atomic<bool> stop{false};
        std::atomic<size_t> jobs_executed{0};
        std::atomic<size_t> jobs_executing{0};
        std::atomic<size_t> jobs_stolen{0};
        explicit WorkerData(size_t cap) : deque(cap) {}
    };

    std::vector<std::unique_ptr<WorkerData>> workers_;
    lockfree::Deque<JobRaw> injector_;
    size_t num_threads_;
    size_t queue_capacity_;
    std::atomic<bool> is_running_{false};

    std::atomic<size_t> total_submitted_{0};
    std::atomic<size_t> total_completed_{0};
    std::atomic<int> idle_workers_{0};

    std::mutex park_mutex_;                  // idle parking only (off the hot path)
    std::condition_variable park_cv_;
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;

    std::atomic<ErrorType> error_type_{ErrorType::None};

    // Optional hook run on the worker thread after EACH job's execute() — used to
    // recycle the per-worker scratch arena between tasks (allocation architecture).
    std::function<void()> on_job_complete_;

    // Identify the worker (if any) running on the current thread for THIS system, so a
    // nested submit can go straight to that worker's own deque.
    static inline thread_local JobSystem* t_sys_ = nullptr;
    static inline thread_local WorkerData* t_worker_ = nullptr;

    // Latch an error: stop every worker and wake all waiters so no wait can hang on a
    // job orphaned in an exited worker's queue.
    void stop_all_workers() {
        for (auto& w : workers_) w->stop.store(true, std::memory_order_release);
        park_cv_.notify_all();
        completion_cv_.notify_all();
    }

    void wake_one_worker() {
        if (idle_workers_.load(std::memory_order_acquire) > 0) park_cv_.notify_one();
    }

    JobRaw find_work(WorkerData* data, std::mt19937& rng) {
        if (JobRaw j = data->deque.pop()) return j;            // own work (LIFO)
        size_t n = workers_.size();
        if (n > 1) {                                            // steal a victim's top
            for (size_t attempt = 0; attempt < n; ++attempt) {
                WorkerData* victim = workers_[rng() % n].get();
                if (victim == data) continue;
                if (JobRaw j = victim->deque.steal()) {
                    data->jobs_stolen.fetch_add(1, std::memory_order_relaxed);
                    return j;
                }
            }
        }
        if (auto opt = injector_.try_pop_front()) return *opt;  // external work
        return nullptr;
    }

    void run_job(WorkerData* data, JobRaw job) {
        data->jobs_executing.fetch_add(1);
        try {
            job->execute();
        } catch (const std::bad_alloc&) {
            error_type_.store(ErrorType::OutOfMemory, std::memory_order_release);
            stop_all_workers();
        } catch (const std::exception& e) {
            error_type_.store(std::string(e.what()) == "Operation aborted"
                                  ? ErrorType::Aborted : ErrorType::Exception,
                              std::memory_order_release);
            stop_all_workers();
        } catch (...) {
            error_type_.store(ErrorType::Unhandled, std::memory_order_release);
            stop_all_workers();
        }
        delete job;
        if (on_job_complete_) on_job_complete_();   // recycle per-worker scratch
        data->jobs_executing.fetch_sub(1);
        data->jobs_executed.fetch_add(1, std::memory_order_relaxed);

        // Notify the completion waiter only at quiescence (this job brings completed up
        // to submitted), not on every job. The waiter also polls on a timeout, so a
        // missed wakeup from a racing submit only adds latency, never a hang.
        size_t done = total_completed_.fetch_add(1) + 1;
        if (done == total_submitted_.load(std::memory_order_acquire)) {
            completion_cv_.notify_all();
        }
    }

    void worker_loop(WorkerData* data, size_t index) {
        t_sys_ = this;
        t_worker_ = data;
        std::mt19937 rng(static_cast<uint32_t>(index) * 2654435761u + 1u);

        while (true) {
            if (JobRaw job = find_work(data, rng)) {
                run_job(data, job);
                continue;
            }
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) break;
            if (data->stop.load(std::memory_order_acquire)) break;  // shutdown, drained

            idle_workers_.fetch_add(1, std::memory_order_release);
            {
                std::unique_lock<std::mutex> lock(park_mutex_);
                park_cv_.wait_for(lock, std::chrono::microseconds(200), [this, data] {
                    return data->stop.load(std::memory_order_acquire)
                        || error_type_.load(std::memory_order_acquire) != ErrorType::None
                        || !injector_.empty();
                });
            }
            idle_workers_.fetch_sub(1, std::memory_order_release);
        }

        t_sys_ = nullptr;
        t_worker_ = nullptr;
    }

    void drain_and_delete() {
        for (auto& w : workers_) {
            while (JobRaw j = w->deque.pop()) delete j;
        }
        while (auto opt = injector_.try_pop_front()) delete *opt;
    }

    // Route a job to the current worker's own deque (nested submit) or the injector.
    void enqueue(JobRaw raw) {
        if (t_sys_ == this && t_worker_ != nullptr) {
            if (!t_worker_->deque.push(raw)) injector_.push_back(raw);  // local overflow
        } else {
            injector_.push_back(raw);
        }
        wake_one_worker();
    }

public:
    explicit JobSystem(size_t num_threads = 0, size_t queue_capacity = 4096)
        : injector_(32768),
          num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads),
          queue_capacity_(queue_capacity == 0 ? 4096 : queue_capacity) {
        if (num_threads_ == 0) num_threads_ = 1;
        workers_.reserve(num_threads_);
        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back(std::make_unique<WorkerData>(queue_capacity_));
        }
    }

    ~JobSystem() {
        shutdown();
    }

    // Register a callback run on the worker thread after EACH job completes (after
    // execute(), even on error). Used to reset per-worker scratch between tasks.
    void set_on_job_complete(std::function<void()> cb) { on_job_complete_ = std::move(cb); }

    void start() {
        if (is_running_.load()) return;
        total_submitted_.store(0);
        total_completed_.store(0);
        error_type_.store(ErrorType::None, std::memory_order_relaxed);

        for (size_t i = 0; i < workers_.size(); ++i) {
            workers_[i]->stop.store(false, std::memory_order_relaxed);
        }
        for (size_t i = 0; i < workers_.size(); ++i) {
            auto* worker = workers_[i].get();
            worker->thread = std::thread([this, worker, i] { worker_loop(worker, i); });
        }
        is_running_.store(true);
    }

    void shutdown() {
        if (!is_running_.load()) return;
        for (auto& worker : workers_) worker->stop.store(true, std::memory_order_release);
        park_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) worker->thread.join();
        }
        drain_and_delete();  // free any jobs abandoned without wait_for_completion
        is_running_.store(false);
    }

    // ScheduleMode is accepted for API compatibility; work-stealing serves a worker's
    // own deque LIFO and the injector FIFO.
    void submit(JobPtr<JobType> job, ScheduleMode mode = ScheduleMode::LIFO) {
        (void)mode;
        if (!is_running_.load()) throw std::runtime_error("JobSystem is not running");
        total_submitted_.fetch_add(1);
        enqueue(job.release());
    }

    void submit_to_worker(size_t worker_id, JobPtr<JobType> job, ScheduleMode mode = ScheduleMode::LIFO) {
        (void)mode;
        if (!is_running_.load()) throw std::runtime_error("JobSystem is not running");
        if (worker_id >= workers_.size()) throw std::out_of_range("Invalid worker ID");
        total_submitted_.fetch_add(1);
        enqueue(job.release());  // worker_id is an affinity hint only under work-stealing
    }

    bool try_submit(JobPtr<JobType> job, ScheduleMode mode = ScheduleMode::LIFO) {
        if (!is_running_.load()) return false;
        submit(std::move(job), mode);
        return true;
    }

    template<typename F>
    void submit_function(F&& func, JobType job_type, int priority = 0, ScheduleMode mode = ScheduleMode::LIFO) {
        submit(make_job(std::forward<F>(func), job_type, priority), mode);
    }

    // Wait for completion with an abort callback. Returns true if aborted (by the
    // callback or a worker error), false if completed normally.
    template<typename AbortCheck>
    bool wait_for_completion_with_abort(AbortCheck&& abort_check) {
        while (true) {
            if (abort_check()) return true;
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) return true;

            std::unique_lock<std::mutex> lock(completion_mutex_);
            completion_cv_.wait_for(lock, std::chrono::milliseconds(50), [this] {
                return total_submitted_.load() == total_completed_.load();
            });

            if (is_quiescent()) return false;
        }
    }

    void wait_for_completion() {
        while (true) {
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) return;

            std::unique_lock<std::mutex> lock(completion_mutex_);
            completion_cv_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                return total_submitted_.load() == total_completed_.load();
            });

            if (is_quiescent()) return;
        }
    }

private:
    // True when all submitted work has completed and nothing is queued or executing.
    bool is_quiescent() {
        if (total_submitted_.load() != total_completed_.load()) return false;
        if (!injector_.empty()) return false;
        for (const auto& worker : workers_) {
            if (!worker->deque.empty() || worker->jobs_executing.load() > 0) return false;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        return total_submitted_.load() == total_completed_.load();
    }

public:
    size_t get_num_workers() const { return workers_.size(); }

    size_t get_pending_count() const {
        size_t s = total_submitted_.load(std::memory_order_relaxed);
        size_t c = total_completed_.load(std::memory_order_relaxed);
        return s > c ? s - c : 0;
    }

    size_t get_executing_count() const {
        size_t count = 0;
        for (const auto& worker : workers_) count += worker->jobs_executing.load(std::memory_order_relaxed);
        return count;
    }

    bool is_running() const { return is_running_.load(); }

    ErrorType get_error_type() const { return error_type_.load(std::memory_order_acquire); }
    bool has_error() const { return get_error_type() != ErrorType::None; }

    const char* get_error_description() const {
        switch (get_error_type()) {
            case ErrorType::None: return "No error";
            case ErrorType::OutOfMemory: return "Out of memory";
            case ErrorType::Aborted: return "Aborted";
            case ErrorType::Exception: return "Exception thrown";
            case ErrorType::Unhandled: return "Unhandled exception type";
        }
        return "Unknown error";
    }

    struct SystemStatistics {
        size_t total_jobs_executed;
        size_t total_jobs_stolen;
        size_t total_jobs_deferred;
    };

    SystemStatistics get_statistics() const {
        size_t total_executed = 0, total_stolen = 0;
        for (const auto& worker : workers_) {
            total_executed += worker->jobs_executed.load();
            total_stolen += worker->jobs_stolen.load();
        }
        return SystemStatistics{total_executed, total_stolen, 0};
    }

    // Compatibility stubs (no per-type incompatibility model in this scheduler).
    void register_incompatibility(JobType, JobType) {}
    void register_compatibility_function(std::function<bool(JobType, JobType)>) {}
    void clear_compatibility_rules() {}
};

} // namespace job_system

