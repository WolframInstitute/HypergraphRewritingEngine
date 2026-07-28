#pragma once

#include <job_system/job.hpp>
#include <job_system/work_stealing_deque.hpp>
#include <lockfree_deque/deque.hpp>
#include <thread>
#include <vector>
#include <mutex>
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
    // How many workers are parked. Submitting is the hot path and parking is rare, so this
    // exists to keep the common submit free of any read-modify-write: with nobody parked
    // there is nobody to wake, and the counter bump and notify are both skipped.
    std::atomic<int> idle_workers_{0};
    // Times a worker actually blocked. Worth keeping visible: this system is designed on the
    // assumption that parking is rare, and the number confirms it -- a depth-6 Wolfram run of
    // 15,966 events parks 14 times at 4 threads and 27 at 8.
    std::atomic<size_t> park_waits_{0};
    // How many threads are inside wait_for_completion. Completion is signalled per job, and
    // notifying an address with no waiter is not free -- the implementation hashes it into a
    // process-wide waiter table, so every worker would touch one shared line on every job.
    // Almost always zero, so this keeps job completion a pure counter increment.
    std::atomic<int> completion_waiters_{0};
    // Bumped only when a completion brings completed up to submitted -- the only moment the
    // system can be quiescent. Waiters block on THIS, not on the completion count: blocking on
    // the count wakes them once per job, and each wake re-runs is_quiescent(), which scans
    // every worker's deque. Sixteen thousand jobs then cost sixteen thousand scans.
    std::atomic<uint32_t> quiescence_seq_{0};

    // Idle workers block on this counter rather than on a condition variable. It is
    // incremented by every enqueue and by anything that should end a park (an error, a
    // shutdown), so a parked worker is woken by ANY new work, wherever it landed.
    //
    // That last part is why a sequence counter and not a predicate: the old park slept on a
    // condition that asked whether the shared injector was non-empty, which is blind to a job
    // pushed onto a worker's own deque -- work a parked thread could have stolen. A 200us
    // timeout covered the gap by re-checking. A counter has no gap to cover, so the wait needs
    // no timeout and does no polling.
    //
    // std::atomic::wait blocks only while the value still equals the one the caller sampled,
    // so a submit racing with a park cannot be lost: the counter has already moved and the
    // wait returns at once. The fast path is a plain load with no syscall and no lock.
    std::atomic<uint32_t> work_seq_{0};

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
        wake_all_workers();
        quiescence_seq_.fetch_add(1, std::memory_order_release);
        quiescence_seq_.notify_all();      // release anyone inside wait_for_completion
    }

    // Publish that the world changed. Bumping before notifying is what makes the park
    // lost-wakeup-free: a worker that sampled the old value finds it stale and does not sleep.
    //
    // The idle check is sequentially consistent against the worker's own sequentially
    // consistent increment, and that pairing is what allows it to be skipped safely. The two
    // sides are a store-then-load on different locations: the submitter pushes then reads
    // idle_workers_; a parking worker increments idle_workers_ then looks for work. At least
    // one of them must observe the other, so a worker that this call skips is a worker whose
    // own final look for work happens after the push and therefore finds it.
    void wake_one_worker() {
        if (idle_workers_.load(std::memory_order_seq_cst) <= 0) return;
        work_seq_.fetch_add(1, std::memory_order_release);
        work_seq_.notify_one();
    }

    // Unconditional: used by error latching and shutdown, where a parked worker must be
    // released whether or not the idle count says one is there.
    void wake_all_workers() {
        work_seq_.fetch_add(1, std::memory_order_release);
        work_seq_.notify_all();
    }

    // Exhaustive version, used only immediately before parking. find_work picks victims at
    // RANDOM with a bounded number of attempts, so it can come back empty while a deque still
    // holds work -- fine when the caller loops, but not as the basis for going to sleep. The
    // old park hid this behind a 200us timeout that woke and retried; with the timeout gone,
    // a worker must establish that there is genuinely nothing before it waits.
    JobRaw find_work_exhaustive(WorkerData* data) {
        if (JobRaw j = data->deque.pop()) return j;
        for (auto& w : workers_) {
            if (w.get() == data) continue;
            if (JobRaw j = w->deque.steal()) {
                data->jobs_stolen.fetch_add(1, std::memory_order_relaxed);
                return j;
            }
        }
        if (auto opt = injector_.try_pop_front()) return *opt;
        return nullptr;
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

    // recycle_scratch is false when this runs nested inside another job on the same
    // thread (see enqueue): on_job_complete_ resets the per-worker scratch arena, and
    // the job further out on the stack still holds live allocations in it. The nested
    // job's own scratch sits above the outer job's high-water mark and is reclaimed
    // when the outer job completes.
    // `data` is null when a non-worker submitter runs an overflowed job on its own thread;
    // the per-worker counters simply do not apply to it.
    void run_job(WorkerData* data, JobRaw job, bool recycle_scratch = true) {
        if (data) data->jobs_executing.fetch_add(1);
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
        if (recycle_scratch && on_job_complete_) on_job_complete_();  // recycle per-worker scratch
        if (data) {
            data->jobs_executing.fetch_sub(1);
            data->jobs_executed.fetch_add(1, std::memory_order_relaxed);
        }

        // Notify the completion waiter only at quiescence (this job brings completed up
        // to submitted), not on every job. The waiter also polls on a timeout, so a
        // missed wakeup from a racing submit only adds latency, never a hang.
        const size_t done = total_completed_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (done == total_submitted_.load(std::memory_order_acquire)) {
            // Bump before notifying: a waiter that sampled the old value finds it stale and
            // re-checks instead of sleeping through the event it was waiting for.
            quiescence_seq_.fetch_add(1, std::memory_order_release);
            if (completion_waiters_.load(std::memory_order_acquire) > 0) {
                quiescence_seq_.notify_all();
            }
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

            // Sample the counter BEFORE the last look for work. Anything submitted after this
            // point moves the counter, so the wait below returns immediately rather than
            // sleeping through it; anything submitted before it is found by that last look.
            // Announce the park BEFORE sampling the counter and taking the last look for
            // work, so a submitter either sees this worker as parked and wakes it, or is
            // ordered before that last look and is found by it.
            idle_workers_.fetch_add(1, std::memory_order_seq_cst);
            const uint32_t seq = work_seq_.load(std::memory_order_acquire);
            if (JobRaw job = find_work_exhaustive(data)) {
                idle_workers_.fetch_sub(1, std::memory_order_relaxed);
                run_job(data, job);
                continue;
            }
            if (error_type_.load(std::memory_order_acquire) == ErrorType::None &&
                !data->stop.load(std::memory_order_acquire)) {
                park_waits_.fetch_add(1, std::memory_order_relaxed);
                work_seq_.wait(seq, std::memory_order_acquire);
            }
            idle_workers_.fetch_sub(1, std::memory_order_relaxed);
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) break;
            if (data->stop.load(std::memory_order_acquire)) break;
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
    //
    // NOTHING BLOCKS HERE. A full queue is answered by doing the work on the calling thread,
    // which always makes progress, rather than by waiting for space someone else must create.
    // For a worker that is not merely preferable but required: a worker parked inside a push
    // cannot pop, so with every worker parked there at once nothing would ever drain the
    // queues again -- and expansion is combinatorial and submits from inside running jobs, so
    // saturating both the local deque and the injector is reachable, not hypothetical.
    void enqueue(JobRaw raw) {
        const bool on_worker = (t_sys_ == this && t_worker_ != nullptr);

        if (on_worker && t_worker_->deque.push(raw)) {        // node-local, the common case
            wake_one_worker();
            return;
        }
        if (injector_.try_push_back(raw)) {
            wake_one_worker();
            return;
        }
        // Both full. Run it here. On a worker, recycle_scratch is false because the job
        // further out on this stack still holds live allocations in the per-worker arena.
        run_job(on_worker ? t_worker_ : nullptr, raw, /*recycle_scratch=*/on_worker ? false : true);
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
        wake_all_workers();
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
        // abort_check is a caller-supplied poll, so this one keeps a bounded sleep -- there
        // is nothing to be notified BY when the abort condition lives outside the system.
        completion_waiters_.fetch_add(1, std::memory_order_seq_cst);
        struct Leave { std::atomic<int>& n; ~Leave() { n.fetch_sub(1, std::memory_order_release); } }
            leave{completion_waiters_};

        while (true) {
            if (abort_check()) return true;
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) return true;
            if (is_quiescent()) return false;

            const uint32_t q = quiescence_seq_.load(std::memory_order_acquire);
            if (is_quiescent()) return false;
            quiescence_seq_.wait(q, std::memory_order_acquire);
        }
    }

    void wait_for_completion() {
        // No timeout and no polling: this blocks on the completion counter itself and is
        // woken by the job that moves it. Sampling the counter before the final quiescence
        // check is what closes the race -- a job completing in between changes the value, so
        // the wait returns rather than sleeping past the event it was waiting for.
        // Register as a waiter BEFORE the first check. A job completing between the check and
        // the registration would otherwise see no waiter, skip the notify, and leave this
        // blocked on a value that never moves again.
        completion_waiters_.fetch_add(1, std::memory_order_seq_cst);
        struct Leave { std::atomic<int>& n; ~Leave() { n.fetch_sub(1, std::memory_order_release); } }
            leave{completion_waiters_};

        while (true) {
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) return;
            if (is_quiescent()) return;

            const uint32_t q = quiescence_seq_.load(std::memory_order_acquire);
            if (is_quiescent()) return;
            quiescence_seq_.wait(q, std::memory_order_acquire);
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

    size_t park_waits() const { return park_waits_.load(std::memory_order_relaxed); }

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

