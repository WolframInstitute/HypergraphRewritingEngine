#pragma once
#include "hgcommon/namespace.hpp"

#include <job_system/job.hpp>
#include <job_system/work_stealing_deque.hpp>
#include <lockfree_deque/deque.hpp>
#include <hgcommon/park.hpp>
#include <hgcommon/affinity.hpp>
#include <hgcommon/core.hpp>  // splitmix64 -- the steal victim draw
#include <hgcommon/phase_timing.hpp>  // the idle bucket, entered below
#include <hgcommon/capacity.hpp>  // the one error kind that is not a defect
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace HG_NAMESPACE {
namespace jobs {

// Error types that can occur during job execution
enum class ErrorType {
    None = 0,
    OutOfMemory,   // std::bad_alloc caught
    Aborted,       // AbortedException caught (user requested abort)
    // A CONFIGURED LIMIT was reached (hgcommon::CapacityExhausted). Distinct from every other
    // kind because it is not a defect: the work done so far is valid and the caller wants it.
    // Whoever owns the run decides what to do -- the engine serves the truncated graph with a
    // warning -- but the classification has to happen HERE, at the catch, while the type is
    // still known.
    CapacityExhausted,
    Exception,     // std::exception caught
    Unhandled      // Non-std::exception type caught
};

// Lock-free work-stealing scheduler.
//
// Each worker owns a Chase-Lev deque: it pushes/pops its own bottom (so nested jobs
// submitted from a running job stay node-local and lock-free), and idle workers steal
// the top of others' deques. External submissions (from non-worker threads) and local
// overflow go to a shared lock-free injector. Idle workers park on a 32-bit sequence
// counter through std::atomic::wait, so the system holds no mutex and no condition
// variable anywhere; all queue operations are lock-free. The design is
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
        // Of those, the ones taken from a worker sharing this one's last-level cache. The
        // ratio to jobs_stolen is how a run reports whether the locality preference found
        // anything to prefer, rather than it being assumed from the topology.
        std::atomic<size_t> jobs_stolen_near{0};
        explicit WorkerData(size_t cap) : deque(cap) {}
    };

    std::vector<std::unique_ptr<WorkerData>> workers_;
    lockfree::Deque<JobRaw> injector_;
    size_t num_threads_;
    // Serial execution: no workers; wait_for_completion() drains the injector inline
    // on the calling thread (see the constructor comment).
    bool serial_ = false;
    // Logical CPUs the workers bind to, empty meaning the operating system places them.
    std::vector<unsigned> worker_cpus_;
    // WHICH WORKERS SHARE A LAST-LEVEL CACHE, as a CSR over worker indices: the peers of worker
    // i are peers_[peer_begin_[i]) .. peers_[peer_begin_[i+1]), i itself excluded. Written once
    // by start() before any worker thread exists and read-only thereafter, so the steal path
    // reads it without synchronisation.
    //
    // BOTH EMPTY means there is no grouping worth preferring -- an unreadable topology, a
    // machine whose cores all share one cache, or a set where no two workers share one -- and
    // find_work then draws from the whole pool, which is what it does on such a machine anyway.
    std::vector<unsigned> peer_begin_;
    std::vector<unsigned> peers_;
    // A caller-stated domain per worker, which start() uses in place of asking the platform.
    // Empty means ask. See set_worker_cache_domains.
    std::vector<unsigned> worker_cache_domains_;
    std::atomic<size_t> pin_failures_{0};
    // Workers that have passed their binding attempt this start; start() waits for all of
    // them, which is what makes pin_failures() settled rather than racing worker startup.
    std::atomic<size_t> workers_entered_{0};
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
    static_assert(sizeof(uint32_t) == 4, "the park backends compare a 32-bit word");

    // Idle workers block on this counter. It is incremented by every enqueue and by anything
    // that should end a park (an error, a shutdown), so a parked worker is woken by ANY new
    // work, wherever it landed.
    //
    // That last part is why the wait is on a sequence counter and not on a predicate over the
    // queues. A predicate can only name places it knows to look, and a job pushed onto a
    // worker's own deque -- which a parked thread could have stolen -- is not the injector
    // being non-empty. Any predicate narrower than "something happened" leaves a gap that only
    // a timed re-check can cover. A counter is that predicate exactly, so the wait needs no
    // timeout and does no polling.
    //
    // std::atomic::wait blocks only while the value still equals the one the caller sampled,
    // so a submit racing with a park cannot be lost: the counter has already moved and the
    // wait returns at once. The fast path is a plain load with no syscall and no lock.
    std::atomic<uint32_t> work_seq_{0};

    std::atomic<ErrorType> error_type_{ErrorType::None};

    // The first worker exception's what(), kept so the failure names its cause and not just
    // its category. Fixed buffer rather than a std::string because this is written from a
    // catch block on a worker thread, where allocating is the last thing to attempt --
    // OutOfMemory is one of the states being reported.
    //
    // Three states, so a reader never observes a partially written buffer: 0 empty,
    // 1 claimed by the thread that won the exchange, 2 published. Only 2 may be read.
    static constexpr size_t kErrorMessageCap = 512;
    char error_message_[kErrorMessageCap]{};
    std::atomic<uint8_t> error_message_state_{0};

    // First failure wins, matching error_type_: later ones are consequences of the stop.
    void record_error_message(const char* what) noexcept {
        uint8_t expected = 0;
        if (!error_message_state_.compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel, std::memory_order_relaxed))
            return;
        const size_t n = what ? std::strlen(what) : 0;
        const size_t k = n < kErrorMessageCap - 1 ? n : kErrorMessageCap - 1;
        if (k) std::memcpy(error_message_, what, k);
        error_message_[k] = '\0';
        error_message_state_.store(2, std::memory_order_release);
    }

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
        hgcommon::unpark_all(quiescence_seq_);   // release anyone inside wait_for_completion
    }

    // Publish that the world changed. Bumping before notifying is what makes the park
    // lost-wakeup-free: a worker that sampled the old value finds it stale and does not sleep.
    //
    // Skipping the wake when nobody is idle is safe only because of the fence below.
    //
    // The two sides are a store-then-load on DIFFERENT locations: the submitter pushes the job
    // then reads idle_workers_, while a parking worker increments idle_workers_ then takes its
    // last look for work. For "at least one observes the other" to hold, both sides need a
    // sequentially consistent fence between their write and their read. Sequential consistency on
    // idle_workers_ alone does not supply it -- the job becomes reachable through the deque's
    // acquire/release compare-exchange, so without the fences both threads may read stale and
    // both conclude there is nothing to do: the submitter skips the wake, the worker parks, and
    // the job sits in a deque with nobody awake to take it. Verified in
    // verification/genmc/job_system_no_lost_wakeup.cpp, which reports a non-terminating spinloop
    // when either fence is removed.
    //
    // The caller pushes immediately before calling this, so the fence sits between the push and
    // the read. The worker's matching fence is in the park path.
    void wake_one_worker() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (idle_workers_.load(std::memory_order_seq_cst) <= 0) return;
        work_seq_.fetch_add(1, std::memory_order_release);
        hgcommon::unpark_one(work_seq_);
    }

    // Unconditional: used by error latching and shutdown, where a parked worker must be
    // released whether or not the idle count says one is there.
    void wake_all_workers() {
        work_seq_.fetch_add(1, std::memory_order_release);
        hgcommon::unpark_all(work_seq_);
    }

    // Group the workers by the cache their bound CPU sits behind, filling peer_begin_/peers_.
    // Called by start() before the first worker thread exists, which is what lets the steal
    // path read the result without synchronisation.
    //
    // It answers "leave it off" in every case where a preference would decide nothing: fewer
    // than three workers (a thief has one possible victim), no pinning (the operating system
    // moves the thread, so a binding-derived grouping describes where it was, not where it is),
    // an unreadable topology, and a machine whose workers all share one cache.
    void build_cache_peers() {
        peer_begin_.clear();
        peers_.clear();
        const size_t n = workers_.size();
        if (n < 3) return;
        std::vector<unsigned> domain = worker_cache_domains_;
        if (domain.empty()) {
            if (worker_cpus_.empty()) return;
            std::vector<unsigned> cpus;
            cpus.reserve(n);
            for (size_t i = 0; i < n; ++i) cpus.push_back(worker_cpus_[i % worker_cpus_.size()]);
            domain = hgcommon::cache_domains_of(cpus);
        }
        if (domain.size() != n) return;

        peer_begin_.assign(n + 1, 0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j)
                if (j != i && domain[j] == domain[i]) peers_.push_back(static_cast<unsigned>(j));
            peer_begin_[i + 1] = static_cast<unsigned>(peers_.size());
        }
        // Nobody shares, or everybody does: either way the near set is not a subset worth
        // drawing from first, and the empty pair restores the undivided draw.
        if (peers_.empty() || peers_.size() == n * (n - 1)) {
            peer_begin_.clear();
            peers_.clear();
        }
    }

    // Exhaustive version, used only immediately before parking. find_work picks victims at
    // RANDOM with a bounded number of attempts, so it can come back empty while a deque still
    // holds work -- fine when the caller loops, but not as the basis for going to sleep. The
    // park has no timeout to wake it for a retry, so a worker must establish that there is
    // genuinely nothing anywhere before it waits.
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

    // The steal victim is drawn by advancing a per-worker splitmix64 state, which is one
    // multiply-shift chain over a single uint64. std::mt19937 was 2.5 KB of state per worker,
    // seeded once, to produce `rng() % n` -- and it put <random> in this header, one of the two
    // standard headers whose joint removal from the engine's closure is 196 ms of a 1198 ms
    // translation unit. splitmix64 is hgcommon's, so there is one mixer here rather than a
    // second one written for this call site.
    //
    // The draw does not have to be reproducible: which worker a job is stolen from does not
    // change what the run computes, only who computes it. The gates that assert results are
    // independent of thread count already cover that.
    //
    // A VICTIM THAT SHARES THIS WORKER'S CACHE IS TRIED FIRST. A stolen job goes on to touch
    // the data the victim was working, so where that data already sits decides what the steal
    // costs. Cores that share a last-level cache pass it between them at cache speed; cores on
    // separate caches move it over an off-die fabric. Measured on an EPYC 9174F, which splits
    // 16 cores across EIGHT L3 instances: the same two-thread run takes 1519 ms when the pair
    // shares an L3 and 1852 ms when it does not, 21% for placement alone. A part with one
    // shared cache has nothing to prefer and build_cache_peers leaves the preference off.
    //
    // It stays a PREFERENCE. Near victims are tried, then the draw widens to the whole pool: an
    // idle core costs more than a distant line, so a thief never declines a far steal it could
    // have taken.
    JobRaw find_work(WorkerData* data, uint64_t& rng, size_t self) {
        if (JobRaw j = data->deque.pop()) return j;            // own work (LIFO)
        size_t n = workers_.size();
        if (!peer_begin_.empty()) {                             // near victims first
            const unsigned lo = peer_begin_[self];
            const unsigned span = peer_begin_[self + 1] - lo;
            for (unsigned attempt = 0; attempt < span; ++attempt) {
                WorkerData* victim = workers_[peers_[lo + hgcommon::splitmix64(++rng) % span]].get();
                if (JobRaw j = victim->deque.steal_if_nonempty()) {
                    data->jobs_stolen.fetch_add(1, std::memory_order_relaxed);
                    data->jobs_stolen_near.fetch_add(1, std::memory_order_relaxed);
                    return j;
                }
            }
        }
        if (n > 1) {                                            // steal a victim's top
            for (size_t attempt = 0; attempt < n; ++attempt) {
                WorkerData* victim = workers_[hgcommon::splitmix64(++rng) % n].get();
                if (victim == data) continue;
                if (JobRaw j = victim->deque.steal_if_nonempty()) {
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
        } catch (const hgcommon::CapacityExhausted& e) {
            record_error_message(e.what());
            error_type_.store(ErrorType::CapacityExhausted, std::memory_order_release);
            stop_all_workers();
        } catch (const std::bad_alloc& e) {
            record_error_message(e.what());
            error_type_.store(ErrorType::OutOfMemory, std::memory_order_release);
            stop_all_workers();
        } catch (const std::exception& e) {
            const bool aborted = std::strcmp(e.what(), "Operation aborted") == 0;
            if (!aborted) record_error_message(e.what());
            error_type_.store(aborted ? ErrorType::Aborted : ErrorType::Exception,
                              std::memory_order_release);
            stop_all_workers();
        } catch (...) {
            record_error_message("non-std exception");
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
                hgcommon::unpark_all(quiescence_seq_);
            }
        }
    }

    void worker_loop(WorkerData* data, size_t index) {
        t_sys_ = this;
        t_worker_ = data;
        // Bind this worker if the caller named a core set. Worker i takes the i-th CPU in that
        // set; more workers than CPUs means they share, which is the caller's choice to make
        // rather than this loop's to refuse. Empty set: placement stays the operating system's.
        if (!worker_cpus_.empty()) {
            const unsigned cpu = worker_cpus_[index % worker_cpus_.size()];
            if (!hgcommon::pin_this_thread_to_cpu(cpu))
                pin_failures_.fetch_add(1, std::memory_order_relaxed);
        }
        // Past the binding attempt: start() waits on this, which is what makes pin_failures()
        // a settled count once start() returns. Release pairs with start()'s acquire so the
        // failure increment above is visible when the count is.
        workers_entered_.fetch_add(1, std::memory_order_release);
        uint64_t rng = static_cast<uint64_t>(index) * 2654435761ull + 1ull;

        while (true) {
            if (JobRaw job = find_work(data, rng, index)) {
                run_job(data, job);
                continue;
            }
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) break;
            if (data->stop.load(std::memory_order_acquire)) break;  // shutdown, drained

            // FROM HERE TO THE TOP OF THE LOOP THIS WORKER HAS NO JOB. The announcement, the
            // last exhaustive look and the park all belong to one bucket, because the question
            // the instrument exists to answer is whether adding a worker bought work or bought
            // waiting -- and a report whose buckets cover only the guarded work says nothing
            // about the workers that found none.
            hgcommon::PhaseTimer _idle(hgcommon::Phase::Idle);

            // Sample the counter BEFORE the last look for work. Anything submitted after this
            // point moves the counter, so the wait below returns immediately rather than
            // sleeping through it; anything submitted before it is found by that last look.
            // Announce the park BEFORE sampling the counter and taking the last look for
            // work, so a submitter either sees this worker as parked and wakes it, or is
            // ordered before that last look and is found by it.
            //
            // The fence is what makes that hold. It pairs with the one in wake_one_worker: the
            // two threads write different locations and then read the other's, and only a
            // sequentially consistent fence on BOTH sides forbids them from both reading stale.
            // Without it the submitter can read this worker as not-yet-idle while this worker
            // reads the deque as still empty, and the job is left queued with the worker parked.
            idle_workers_.fetch_add(1, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const uint32_t seq = work_seq_.load(std::memory_order_acquire);
            if (JobRaw job = find_work_exhaustive(data)) {
                idle_workers_.fetch_sub(1, std::memory_order_relaxed);
                run_job(data, job);
                continue;
            }
            if (error_type_.load(std::memory_order_acquire) == ErrorType::None &&
                !data->stop.load(std::memory_order_acquire)) {
                park_waits_.fetch_add(1, std::memory_order_relaxed);
                hgcommon::park_if_equal(work_seq_, seq);
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
        // Serial: everything goes through the injector FIFO; the draining thread is the
        // only executor, so there is nobody to wake. A full injector runs the job right
        // here -- possibly inside the submitting job, so the scratch arena is not
        // recycled (the job further out on this stack may hold live allocations in it);
        // the next drain-loop job recycles as usual.
        if (serial_) {
            if (injector_.try_push_back(raw)) return;
            run_job(nullptr, raw, /*recycle_scratch=*/false);
            return;
        }
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
    // SERIAL MODE (serial = true): no worker threads exist and none are ever spawned --
    // start() only arms the counters, submit() routes to the injector, and
    // wait_for_completion() drains it inline on the calling thread in FIFO order.
    // Jobs submitted by a running job land behind it and run in submission order, so a
    // run is deterministic by construction. This is the single-threaded execution mode
    // (and the WebAssembly path, where spawning threads is not available): everything
    // the workers would do happens on the thread that waits.
    explicit JobSystem(size_t num_threads = 0, size_t queue_capacity = 4096,
                       bool serial = false)
        : injector_(32768),
          num_threads_(serial ? 0
                              : (num_threads == 0 ? std::thread::hardware_concurrency()
                                                  : num_threads)),
          queue_capacity_(queue_capacity == 0 ? 4096 : queue_capacity),
          serial_(serial) {
        if (!serial_ && num_threads_ == 0) num_threads_ = 1;
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

    // Bind each worker to a logical CPU, before start(). Worker i takes cpus[i % cpus.size()].
    //
    // THIS IS FOR MEASUREMENT AND FOR HETEROGENEOUS PARTS, and it is off unless asked for. A
    // speedup curve taken across cores of different speeds has no honest denominator -- one
    // E-core of a 14900K does in 30.370 ms what a P-core does in 18.042 ms -- so a caller that
    // wants a curve that means something names a homogeneous set. An empty vector restores the
    // default, which is that the operating system places every worker.
    //
    // Whether a binding took effect is NOT assumed: pin_failures() counts workers that asked and
    // were refused, which is every worker on macOS (no such API) and any CPU index the platform
    // cannot express. A caller reporting a pinned measurement checks it is zero. Settled once
    // start() returns -- start() waits for every worker to pass its binding attempt.
    void set_worker_cpus(std::vector<unsigned> cpus) { worker_cpus_ = std::move(cpus); }
    const std::vector<unsigned>& worker_cpus() const { return worker_cpus_; }

    // State which workers share a cache, one id per worker, instead of having start() ask the
    // platform. Equal ids mean "these two workers are cheap to pass work between"; the steal
    // path prefers a victim carrying the thief's id.
    //
    // The platform answer covers the ordinary case and needs pinning to mean anything, since an
    // unpinned thread migrates away from the CPU its grouping was derived from. A caller that
    // places its threads by some other route -- a container CPU set, a NUMA policy, a machine
    // whose topology the kernel does not publish -- knows the grouping the platform cannot be
    // asked for, and this is where it says so. An empty vector, the default, means ask.
    //
    // Read by start(); changing it while running does nothing until the next start().
    void set_worker_cache_domains(std::vector<unsigned> domains) {
        worker_cache_domains_ = std::move(domains);
    }
    size_t pin_failures() const { return pin_failures_.load(std::memory_order_relaxed); }

    void start() {
        if (is_running_.load()) return;
        total_submitted_.store(0);
        total_completed_.store(0);
        error_type_.store(ErrorType::None, std::memory_order_relaxed);
        error_message_state_.store(0, std::memory_order_relaxed);

        for (size_t i = 0; i < workers_.size(); ++i) {
            workers_[i]->stop.store(false, std::memory_order_relaxed);
        }
        workers_entered_.store(0, std::memory_order_relaxed);
        build_cache_peers();                      // before any worker exists: see peer_begin_
        for (size_t i = 0; i < workers_.size(); ++i) {
            auto* worker = workers_[i].get();
            worker->thread = std::thread([this, worker, i] { worker_loop(worker, i); });
        }
        // Wait until every worker has passed its binding attempt, so pin_failures() is a
        // settled count when start() returns rather than a snapshot racing worker startup --
        // on a small machine a spawned thread can lag past an entire short workload before it
        // first runs. One yield loop, once per start; the workers park themselves if no work
        // arrives, so this costs thread-spawn latency and nothing else.
        while (workers_entered_.load(std::memory_order_acquire) < workers_.size())
            std::this_thread::yield();
        is_running_.store(true);
    }

private:
    // Serial-mode executor: run injector jobs on this thread until nothing remains.
    // A job's nested submits land behind it in the injector, so the order is the
    // submission order. Stops early on a latched error, exactly as the workers do.
    void drain_serial() {
        while (error_type_.load(std::memory_order_acquire) == ErrorType::None) {
            auto opt = injector_.try_pop_front();
            if (!opt) break;
            run_job(nullptr, *opt, /*recycle_scratch=*/true);
        }
    }

public:

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
        if (serial_) {
            // Serial: this thread IS the executor; the abort poll runs between jobs,
            // a strictly finer granularity than the parallel path's bounded sleep.
            while (error_type_.load(std::memory_order_acquire) == ErrorType::None) {
                if (abort_check()) return true;
                auto opt = injector_.try_pop_front();
                if (!opt) return false;   // drained: serial has no other queue to wait on
                run_job(nullptr, *opt, /*recycle_scratch=*/true);
            }
            return true;
        }
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
            hgcommon::park_if_equal(quiescence_seq_, q);
        }
    }

    void wait_for_completion() {
        if (serial_) { drain_serial(); return; }
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
            hgcommon::park_if_equal(quiescence_seq_, q);
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

    // The first worker exception's what(), or "" if none was recorded. Valid until reset().
    const char* get_error_message() const {
        return error_message_state_.load(std::memory_order_acquire) == 2 ? error_message_ : "";
    }

    const char* get_error_description() const {
        switch (get_error_type()) {
            case ErrorType::None: return "No error";
            case ErrorType::OutOfMemory: return "Out of memory";
            case ErrorType::Aborted: return "Aborted";
            // Named, because this is the one error type that is not a defect: the work done so
            // far is valid and the caller wants it. Falling through to "Unknown error" told a
            // reader the opposite of what the classification exists to say.
            case ErrorType::CapacityExhausted: return "Configured capacity limit reached";
            case ErrorType::Exception: return "Exception thrown";
            case ErrorType::Unhandled: return "Unhandled exception type";
        }
        return "Unknown error";
    }

    struct SystemStatistics {
        size_t total_jobs_executed;
        size_t total_jobs_stolen;
        size_t total_jobs_deferred;
        // Of total_jobs_stolen, those taken from a victim sharing the thief's last-level cache.
        // Zero with a non-zero steal count means the locality preference is off on this machine
        // or found nothing near, which is a fact about the run and not a failure.
        size_t total_jobs_stolen_near;
    };

    SystemStatistics get_statistics() const {
        size_t total_executed = 0, total_stolen = 0, total_near = 0;
        for (const auto& worker : workers_) {
            total_executed += worker->jobs_executed.load();
            total_stolen += worker->jobs_stolen.load();
            total_near += worker->jobs_stolen_near.load();
        }
        return SystemStatistics{total_executed, total_stolen, 0, total_near};
    }

    // Whether the steal path is preferring cache-local victims on this machine, and how many
    // workers the average thief has near it. Zero means the undivided draw is in use.
    size_t cache_peer_groups() const {
        if (peer_begin_.empty()) return 0;
        size_t domains = 0;
        for (size_t i = 0; i + 1 < peer_begin_.size(); ++i)
            if (peer_begin_[i + 1] > peer_begin_[i]) ++domains;
        return domains;
    }

    // Compatibility stubs (no per-type incompatibility model in this scheduler).
    void register_incompatibility(JobType, JobType) {}
    void register_compatibility_function(std::function<bool(JobType, JobType)>) {}
    void clear_compatibility_rules() {}
};

}  // namespace jobs
}  // namespace HG_NAMESPACE