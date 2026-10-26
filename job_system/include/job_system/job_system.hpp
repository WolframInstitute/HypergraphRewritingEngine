#pragma once
#include "hgcommon/namespace.hpp"
#include "hgcommon/cache_line.hpp"

#include <job_system/job.hpp>
#include <job_system/work_stealing_deque.hpp>
#include <lockfree_deque/deque.hpp>

// Queue sizes at construction. Each is a buffer filled slot by slot when the system is built,
// so under a model checker they set the loop bound construction needs; the harnesses define
// them small (verification/genmc/engine_*.cpp), the shipped defaults are below.
#ifndef HG_JOB_QUEUE_CAPACITY
#define HG_JOB_QUEUE_CAPACITY 4096
#endif
#ifndef HG_JOB_INJECTOR_CAPACITY
#define HG_JOB_INJECTOR_CAPACITY 32768
#endif
#ifndef HG_JOB_ERROR_MESSAGE_CAP
#define HG_JOB_ERROR_MESSAGE_CAP 512
#endif
#if defined(HG_VERIFICATION)
#include <genmc.h>  // __VERIFIER_thread_create_symmetric
#endif
#include <hgcommon/park.hpp>
#include <hgcommon/rendezvous.hpp>
#include <hgcommon/park_gate.hpp>
#include <hgcommon/affinity.hpp>
#include <hgcommon/diag.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hgcommon/core.hpp>  // splitmix64 -- the steal victim draw
#include <hgcommon/portable_intrinsics.hpp>  // ctz64 -- the awake mask walk
#include <hgcommon/phase_timing.hpp>  // the idle bucket, entered below
#include <hgcommon/capacity.hpp>  // the one error kind that is not a defect
#include <thread>
#if defined(HG_VERIFICATION)
#include <pthread.h>
#endif
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

    struct WorkerData;
#if defined(HG_VERIFICATION)
    // The checker's symmetry reduction treats threads as interchangeable only when they run
    // the same function with the same argument and nothing touches memory between the spawn
    // calls. Every worker is therefore started on run_worker with the JobSystem as its only
    // argument and claims its index from spawn_claim_ on entry; the thread ids stay in
    // registers until the last worker is running and are stored afterwards.
    struct WorkerThread {
        pthread_t handle{};
        bool live = false;
        void join() { if (live) { pthread_join(handle, nullptr); live = false; } }
    };
    static void* run_worker(void* p) {
        auto* sys = static_cast<JobSystem*>(p);
        const size_t index = sys->spawn_claim_.fetch_add(1, std::memory_order_relaxed);
        sys->worker_loop(sys->workers_[index].get(), index);
        return nullptr;
    }
    std::atomic<size_t> spawn_claim_{0};
    static constexpr size_t kMaxVerifiedWorkers = 4;
    void spawn_workers() {
        const size_t n = workers_.size();
        if (n > kMaxVerifiedWorkers) std::abort();
        // Serial mode has no workers: nothing to spawn, and a spawned thread would index an
        // empty worker vector.
        if (n == 0) return;
        spawn_claim_.store(0, std::memory_order_relaxed);
        const pthread_t t0 = __VERIFIER_thread_create(nullptr, &run_worker, this);
        const pthread_t t1 = n > 1 ? __VERIFIER_thread_create_symmetric(nullptr, &run_worker, this, t0) : t0;
        const pthread_t t2 = n > 2 ? __VERIFIER_thread_create_symmetric(nullptr, &run_worker, this, t1) : t1;
        const pthread_t t3 = n > 3 ? __VERIFIER_thread_create_symmetric(nullptr, &run_worker, this, t2) : t2;
        const pthread_t ids[kMaxVerifiedWorkers] = {t0, t1, t2, t3};
        for (size_t i = 0; i < n; ++i) {
            workers_[i]->thread.handle = ids[i];
            workers_[i]->thread.live = true;
        }
    }
#else
    struct WorkerThread {
        std::thread handle;
        void spawn(JobSystem* sys, WorkerData* worker, size_t index) {
            handle = std::thread([sys, worker, index] { sys->worker_loop(worker, index); });
        }
        void join() { if (handle.joinable()) handle.join(); }
    };
#endif

    struct WorkerData {
        WorkStealingDeque<JobRaw> deque;
        // The worker's OS thread. Under HG_VERIFICATION it is a pthread_t spawned with
        // pthread_create, which is the primitive the model checker models
        // (__VERIFIER_thread_create); std::thread wraps the same call in a heap-allocated
        // polymorphic state whose typeinfo derives from libstdc++'s, an external constant the
        // interpreter cannot materialise and dies on before the first thread runs. The body,
        // its arguments and the join are identical either way, and the substitution touches
        // no memory any other thread reads.
        WorkerThread thread;
        std::atomic<bool> stop{false};
        std::atomic<size_t> jobs_executed{0};
        std::atomic<size_t> jobs_executing{0};
        std::atomic<size_t> jobs_stolen{0};
        // Of those, the ones taken from a worker sharing this one's last-level cache. The
        // ratio to jobs_stolen is how a run reports whether the locality preference found
        // anything to prefer, rather than it being assumed from the topology.
        std::atomic<size_t> jobs_stolen_near{0};
#if HG_ENGINE_STATS
        // Scheduler counters (SchedulerCounters), written by this worker only.
        std::atomic<size_t> wakes_issued{0};      // parked workers this worker's submits signalled
        std::atomic<size_t> wakes_empty{0};       // returns from a park followed by another park, no job between
        std::atomic<size_t> steal_attempts{0};    // victim deque reads
        std::atomic<size_t> steal_failures{0};    // of those, the ones that took nothing
        bool woke = false;                        // returned from a park and has not run a job since
#endif
        // The park domain this worker parks on, and the domain its submits wake first (the
        // first park domain of its cache). Written once by build_domain_map before any worker
        // thread exists, read-only thereafter.
        unsigned domain = 0;
        unsigned wake_home = 0;
        // The worker this one last stole from, or the wake hint after a park; the worker count
        // when none. Owner-only.
        unsigned last_victim = ~0u;
        unsigned self = 0;   // this worker's index; written by build_domain_map
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
    // steal_work then draws from the whole pool, which is what it does on such a machine anyway.
    std::vector<unsigned> peer_begin_;
    std::vector<unsigned> peers_;
    // A caller-stated domain per worker, which start() uses in place of asking the platform.
    // Empty means ask. See set_worker_cache_domains.
    std::vector<unsigned> worker_cache_domains_;
    std::atomic<size_t> pin_failures_{0};
    // Workers that have passed their binding attempt this start; start() waits for all of
    // them, which is what makes pin_failures() settled rather than racing worker startup.
    std::atomic<size_t> workers_entered_{0};
    // Workers that have left worker_loop this start. After an error a worker leaves once it finds
    // no job, and wait_for_completion returns only when every worker has, so no job is still
    // writing when the caller reads the results.
    std::atomic<size_t> workers_exited_{0};
    // Submits made from a worker that is not inside a job. The quiescence predicate's soundness
    // rests on this being zero; see enqueue().
    std::atomic<size_t> late_submits_{0};
    // TWO HALVES OF ONE FAULT, counted separately. A job entered by run_job a second time was
    // handed out twice; a job found still queued once the system reads as quiescent was never
    // handed out at all. The counters balance when the two pair up, so neither shows in
    // submitted == completed, and the run comes back one job short with no error.
    std::atomic<size_t> double_executions_{0};
    std::atomic<size_t> abandoned_at_quiescence_{0};
    std::atomic<size_t> abandoned_already_run_{0};
#if HG_ENGINE_STATS
    // Scheduler counters that are not per worker: wakes signalled by submits from threads that
    // are not workers, and jobs run on the submitting thread because both queues were full. The
    // searcher high-water mark is ParkGate::searching_max().
    std::atomic<size_t> wakes_issued_off_worker_{0};
    std::atomic<size_t> inline_runs_{0};
#endif
    size_t queue_capacity_;
    std::atomic<bool> is_running_{false};

    std::atomic<size_t> total_submitted_{0};
    std::atomic<size_t> total_completed_{0};
    // How many workers are parked. Submitting is the hot path and parking is rare, so this
    // exists to keep the common submit free of any read-modify-write: with nobody parked
    // there is nobody to wake, and the counter bump and notify are both skipped.

    // Times a worker actually blocked. Worth keeping visible: this system is designed on the
    // assumption that parking is rare, and the number confirms it -- a depth-6 Wolfram run of
    // 15,966 events parks 14 times at 4 threads and 27 at 8.

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
    // ONE PARK WORD PER CACHE DOMAIN, not one for the machine.
    //
    // A parked worker waits on a futex word and a submitter wakes ONE waiter on it. With a
    // single word that waiter is arbitrary, so a submitter on one cache domain routinely wakes a
    // worker on another, which then steals the job across the fabric. The steal path is already
    // topology-aware (see build_cache_peers) and the wake path was not, so the placement it
    // achieves is undone at the moment work is handed out.
    //
    // MEASURED. On a part whose cores share one last-level cache the difference is nil, and the
    // engine scales monotonically -- 14900K, eight workers: wolfram24 6.50x, wpp 5.67x,
    // disc2x2 4.35x, multirule 2.21x, cycle4 2.14x. On an EPYC 9174F, which is sixteen cores
    // over EIGHT L3 instances of two cores each, the same workloads dip at exactly four workers
    // -- the first count that must cross a domain -- and the curve stops being monotonic.
    //
    // Waking within the submitter's own domain first keeps a job and the worker that takes it in
    // one cache. Falling back to the other domains when no local worker is idle is what keeps
    // this from being a lost wakeup; see wake_one_worker.
    //
    // Padded because these are written by every submit: two domains' words on one line would
    // reintroduce, between domains, precisely the sharing this exists to remove.
    // The park/wake handshake is hgcommon::ParkGate; this owns only the storage it runs over.
    // Keeping the protocol there is what makes it checkable --
    // verification/genmc/job_system_no_lost_wakeup.cpp drives the same header this does, where it
    // used to drive a transcription of it.
    std::unique_ptr<hgcommon::ParkGate::Domain[]> domains_;
    hgcommon::ParkGate park_gate_;
    size_t num_domains_ = 1;
    // THE AWAKE MASK: bit i is clear while worker i is between its idle announcement and the
    // return of its park, the span in which its deque is empty (only its owner pushes to it, and
    // the owner found it empty before announcing). steal_work reads only the deques of workers
    // whose bit is set. Written twice per park, read once per search; pools of more than 64
    // workers do not use it. A stale bit costs one read of an empty deque or one missed victim,
    // and the last look before a park reads every deque regardless.
    alignas(hgcommon::kDestructiveLine) std::atomic<uint64_t> awake_mask_{0};
    // THE WAKE HINT: the index of the worker whose submit last signalled a parked worker, or of
    // the deque a chaining searcher found work in (the worker count for the injector). A woken
    // worker's search reads that deque first. Written once per signal, read once per wake.
    alignas(hgcommon::kDestructiveLine) std::atomic<unsigned> wake_hint_{0};

    std::atomic<ErrorType> error_type_{ErrorType::None};

    // The first worker exception's what(), kept so the failure names its cause and not just
    // its category. Fixed buffer rather than a std::string because this is written from a
    // catch block on a worker thread, where allocating is the last thing to attempt --
    // OutOfMemory is one of the states being reported.
    //
    // Three states, so a reader never observes a partially written buffer: 0 empty,
    // 1 claimed by the thread that won the exchange, 2 published. Only 2 may be read.
    static constexpr size_t kErrorMessageCap = HG_JOB_ERROR_MESSAGE_CAP;
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
    // Called before and after a job run on the thread that submitted it (run_inline_).
    std::function<void()> on_inline_begin_;
    std::function<void()> on_inline_end_;

    // Identify the worker (if any) running on the current thread for THIS system, so a
    // nested submit can go straight to that worker's own deque.
    static inline thread_local JobSystem* t_sys_ = nullptr;
    static inline thread_local WorkerData* t_worker_ = nullptr;
    // The system whose outermost run_inline_ is on this thread's stack, and the jobs that
    // overflowed under it (run_inline_). Null outside run_inline_.
    static inline thread_local JobSystem* t_inline_sys_ = nullptr;
    static inline thread_local std::vector<JobRaw>* t_inline_pending_ = nullptr;

    // What a throwing job does (run_job's catch clauses): latch the kind, stop every worker.
    void latch_error(ErrorType kind) {
        error_type_.store(kind, std::memory_order_release);
        stop_all_workers();
    }

public:
#if defined(HG_VERIFICATION)
    // A job's way to fail under a model checker, which interprets no C++ exception: the latch a
    // CapacityExhausted throw reaches through run_job's catch.
    void fail_current_job_for_verification() { latch_error(ErrorType::CapacityExhausted); }
#endif

private:
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
    // A worker's submits wake from its cache's first park domain, so a job pushed here is offered
    // first to a worker sharing this cache; a thread that is not a worker starts at domain 0. The
    // submitter's backlog, for the throttle's burst test, is its own deque's length; a thread that
    // is not a worker has none.
    void wake_one_worker() {
        const bool on_worker = t_sys_ == this && t_worker_ != nullptr;
        const bool woke = on_worker
            ? park_gate_.wake_one(t_worker_->wake_home, [w = t_worker_] { return w->deque.size(); })
            : park_gate_.wake_one(0u);
        if (woke) wake_hint_.store(on_worker ? t_worker_->self : static_cast<unsigned>(workers_.size()),
                                   std::memory_order_relaxed);
        HG_STAT(if (woke) {
            if (on_worker)
                t_worker_->wakes_issued.fetch_add(1, std::memory_order_relaxed);
            else
                wakes_issued_off_worker_.fetch_add(1, std::memory_order_relaxed);
        });
    }

    void wake_all_workers() { park_gate_.wake_all(); }

    // Group the workers by the cache their bound CPU sits behind, filling peer_begin_/peers_.
    // Called by start() before the first worker thread exists, which is what lets the steal
    // path read the result without synchronisation.
    //
    // It answers "leave it off" in every case where a preference would decide nothing: fewer
    // than three workers (a thief has one possible victim), no pinning (the operating system
    // moves the thread, so a binding-derived grouping describes where it was, not where it is),
    // an unreadable topology, and a machine whose workers all share one cache.
    // THE DOMAIN MAP, and the ONLY place the topology is consulted for the wake path.
    //
    // ONE DOMAIN IS ALWAYS A CORRECT ANSWER and is what every uncertain case returns: an
    // unreadable topology, a machine whose cores all share one cache, workers whose placement
    // the operating system chose, or a platform hgcommon::cache_domains_of does not know. In
    // that shape every path below is exactly what it was before domains existed -- one park
    // word, one idle count, one wake -- so a part this cannot describe is never worse off, it
    // simply gains nothing. That matters most where the answer is least certain: an ARM part may
    // report no last-level cache index at all, or clusters that share L2 rather than L3.
    //
    // The engine's correctness never depends on the map being RIGHT, only on it being
    // CONSISTENT -- a worker parks on the word its own index selects and a submitter that finds
    // no idle worker there scans the rest, so a wrong grouping costs locality and not progress.
    // DEFAULT PLACEMENT: FILL CACHE DOMAINS IN ORDER.
    //
    // Without this the topology machinery below decides nothing. build_domain_map derives its
    // grouping from the pinned CPUs, so a run that pins nothing reports ONE domain, and then the
    // per-domain parking and the cache-peer steal order are exactly the single-domain fallback --
    // present, correct, and inert. The engine knew the topology and did not use it.
    //
    // The cost of not using it is measured. On a part with eight two-core L3 instances, two
    // workers left to the scheduler often land on two different instances, and two workers on
    // different instances cost 2.7x against two that share one. Four corpus workloads come in
    // BELOW ONE at two workers there -- 0.73 to 0.97 -- and the same four reach 1.43x to 1.78x on
    // a part whose cores share a single L3.
    //
    // Ordering the CPUs domain-major and handing worker i the i-th of them is what makes the
    // second worker share the first's cache instead of racing it. performance_cpus() supplies the
    // set, so a heterogeneous part contributes only cores of one kind -- a speedup column across
    // cores of different speeds divides by a quantity that is not compute.
    //
    // AN EXPLICIT SET ALWAYS WINS: a caller that named CPUs gets exactly those, and this does
    // nothing. It also does nothing when the topology is unreadable or there is one core to give,
    // which is the same "leave it off" the grouping below applies to every uncertain case.
    void ensure_default_cpu_order() {
        // The placement trace is a diagnostic of stats builds; a release build reads no
        // environment here, and under the model checker getenv is a call it cannot execute.
        bool dbg = false;
        HG_STAT(dbg = std::getenv("HG_DBG_PLACEMENT") != nullptr);
        if (!worker_cpus_.empty()) return;        // a caller that named CPUs gets exactly those
        // The rule is hgcommon::default_worker_cpus(): the performance cores (every core on a
        // homogeneous part) this process is allowed, domain-major, or empty when one cache covers
        // them all. A worker count beyond that set shares it through the modulo in worker_loop.
        std::vector<unsigned> packed = hgcommon::default_worker_cpus();
        if (packed.empty()) return;
#if !defined(HG_VERIFICATION)
        if (dbg) {
            const std::vector<unsigned> dom = hgcommon::cache_domains_of(packed);
            char line[256];
            int at = std::snprintf(line, sizeof line, "[placement] worker cpus, domain-major:");
            for (size_t i = 0; i < packed.size() && i < 8 && at > 0 && at < int(sizeof line); ++i)
                at += std::snprintf(line + at, sizeof line - at, " %u(d%u)", packed[i], dom[i]);
            hgcommon::diag_printf("%s%s (%zu cpus)\n", line, packed.size() > 8 ? " ..." : "",
                                  packed.size());
        }
#endif
        worker_cpus_ = std::move(packed);
    }

    // PARK DOMAINS ARE (CACHE, TIER) PAIRS. Within one last-level cache, workers bound to CPUs are
    // split by tier: the first hardware thread of a core of the fastest class is tier 0, its SMT
    // sibling tier 1, the next class tiers 2 and 3, and so on (hgcommon::host_topology). The
    // domains are numbered cache-major, tier-minor, and a worker's submits wake from the first
    // domain of its cache, so ParkGate::wake_one tries the cache's tiers in order before the other
    // caches: a woken worker is a performance core's first thread while one is parked, and a
    // sibling only when every first thread of the cache is awake. Unbound workers have no CPU and
    // are all tier 0, which is the cache-only map.
    void build_domain_map() {
        const size_t n = workers_.size();
        std::vector<unsigned> worker_domain(n, 0u);
        std::vector<unsigned> wake_home(n, 0u);
        num_domains_ = 1;

        std::vector<unsigned> cache = worker_cache_domains_;
        if (cache.empty() && !worker_cpus_.empty()) {
            std::vector<unsigned> cpus;
            cpus.reserve(n);
            for (size_t i = 0; i < n; ++i) cpus.push_back(worker_cpus_[i % worker_cpus_.size()]);
            cache = hgcommon::cache_domains_of(cpus);
        }
        if (cache.size() != n) cache.assign(n, 0u);
        std::vector<unsigned> tier(n, 0u);
        if (!worker_cpus_.empty()) {
            const hgcommon::HostTopology& topo = hgcommon::host_topology();
            for (size_t i = 0; i < n; ++i) {
                const int k = topo.index_of(worker_cpus_[i % worker_cpus_.size()]);
                if (k < 0) continue;
                const hgcommon::HostCpu& c = topo.cpus[static_cast<size_t>(k)];
                tier[i] = c.core_class * 2u + (c.first_thread ? 0u : 1u);
            }
        }
        // Dense domain ids in (cache, tier) order.
        std::vector<std::pair<unsigned, unsigned>> keys;
        for (size_t i = 0; i < n; ++i) keys.emplace_back(cache[i], tier[i]);
        std::vector<std::pair<unsigned, unsigned>> order = keys;
        std::sort(order.begin(), order.end());
        order.erase(std::unique(order.begin(), order.end()), order.end());
        if (order.size() > 1) {
            num_domains_ = order.size();
            for (size_t i = 0; i < n; ++i) {
                worker_domain[i] = static_cast<unsigned>(
                    std::lower_bound(order.begin(), order.end(), keys[i]) - order.begin());
                wake_home[i] = static_cast<unsigned>(
                    std::lower_bound(order.begin(), order.end(), std::make_pair(cache[i], 0u)) -
                    order.begin());
            }
        }
        domains_ = std::unique_ptr<hgcommon::ParkGate::Domain[]>(
            new hgcommon::ParkGate::Domain[num_domains_]);
        park_gate_.seat(domains_.get(), static_cast<uint32_t>(num_domains_),
                        static_cast<uint32_t>(n), wake_policy_throttled());
        for (size_t i = 0; i < n; ++i) {
            workers_[i]->domain = worker_domain[i];
            workers_[i]->self = static_cast<unsigned>(i);
            workers_[i]->wake_home = wake_home[i];
        }
    }

    // HG_TUNE_WAKE_POLICY=per-submit signals a parked worker on every submit while one is idle;
    // anything else, or unset, is the searching-worker throttle (hgcommon/park_gate.hpp). The
    // per-submit policy stays until the wall-clock A/B in docs/TUNING.md 2.4 decides between them.
    static bool wake_policy_throttled() {
#if defined(HG_VERIFICATION)
        return true;   // the checker cannot execute getenv
#else
        const char* p = std::getenv("HG_TUNE_WAKE_POLICY");
        return !(p && std::strcmp(p, "per-submit") == 0);
#endif
    }

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

    // Exhaustive version, used only immediately before parking. steal_work picks victims at
    // RANDOM with a bounded number of attempts, so it can come back empty while a deque still
    // holds work -- fine when the caller loops, but not as the basis for going to sleep. The
    // park has no timeout to wake it for a retry, so a worker must establish that there is
    // genuinely nothing anywhere before it waits.
    JobRaw find_work_exhaustive(WorkerData* data) {
        if (JobRaw j = data->deque.pop()) return j;
        [[maybe_unused]] size_t attempts = 0;
        JobRaw found = nullptr;
        for (auto& w : workers_) {
            if (w.get() == data) continue;
            HG_STAT(++attempts);
            if (JobRaw j = w->deque.steal()) {
                HG_STAT(data->jobs_stolen.fetch_add(1, std::memory_order_relaxed));
                found = j;
                break;
            }
        }
        HG_STAT(count_steals(data, attempts, found != nullptr));
        if (found) return found;
        if (auto opt = injector_.try_pop_front()) return *opt;
        return nullptr;
    }

#if HG_ENGINE_STATS
    // One update per search rather than one per victim: the attempts are counted in a register.
    static void count_steals(WorkerData* data, size_t attempts, bool took) {
        if (attempts == 0) return;
        data->steal_attempts.fetch_add(attempts, std::memory_order_relaxed);
        data->steal_failures.fetch_add(attempts - (took ? 1 : 0), std::memory_order_relaxed);
    }
#endif

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
    // The search after the own deque came back empty: near victims, random victims, the injector.
    // `more` is set when the queue the job came from still holds work after the take -- two jobs
    // for a worker's deque, whose owner takes the next one itself -- which is the throttle's
    // condition for waking the next worker (ParkGate::search_found).
    JobRaw steal_work(WorkerData* data, uint64_t& rng, size_t self, bool& more) {
        size_t n = workers_.size();
        [[maybe_unused]] size_t attempts = 0;
        if (!peer_begin_.empty()) {                             // near victims first
            const unsigned lo = peer_begin_[self];
            const unsigned span = peer_begin_[self + 1] - lo;
            for (unsigned attempt = 0; attempt < span; ++attempt) {
                WorkerData* victim = workers_[peers_[lo + hgcommon::splitmix64(++rng) % span]].get();
                HG_STAT(++attempts);
                if (JobRaw j = victim->deque.steal_if_nonempty()) {
                    HG_STAT(data->jobs_stolen.fetch_add(1, std::memory_order_relaxed));
                    HG_STAT(data->jobs_stolen_near.fetch_add(1, std::memory_order_relaxed));
                    HG_STAT(count_steals(data, attempts, true));
                    more = victim->deque.size() >= 2;
                    return j;
                }
            }
        }
        if (n > 1 && n <= 64) {                                 // awake victims, each once
            // Upward from a starting worker, wrapping: every awake victim is read once, and a
            // parked worker, whose deque is empty, is not read at all. The start is the victim of
            // this worker's last successful steal when there was one (work that piles on one deque
            // is found at the first read), and a random worker otherwise.
            const uint64_t m = awake_mask_.load(std::memory_order_relaxed) & ~(uint64_t{1} << self);
            const unsigned from = data->last_victim < n
                ? data->last_victim
                : static_cast<unsigned>(hgcommon::splitmix64(++rng) % n);
            const uint64_t upper = from == 0 ? m : m & ~((uint64_t{1} << from) - 1);
            for (uint64_t part : {upper, m & ~upper}) {
                for (; part; part &= part - 1) {
                    const unsigned v = static_cast<unsigned>(hgcommon::ctz64(part));
                    WorkerData* victim = workers_[v].get();
                    HG_STAT(++attempts);
                    if (JobRaw j = victim->deque.steal_if_nonempty()) {
                        HG_STAT(data->jobs_stolen.fetch_add(1, std::memory_order_relaxed));
                        HG_STAT(count_steals(data, attempts, true));
                        more = victim->deque.size() >= 2;
                        data->last_victim = v;
                        return j;
                    }
                }
            }
            data->last_victim = static_cast<unsigned>(n);
        } else if (n > 64) {                                    // steal a victim's top
            for (size_t attempt = 0; attempt < n; ++attempt) {
                WorkerData* victim = workers_[hgcommon::splitmix64(++rng) % n].get();
                if (victim == data) continue;
                HG_STAT(++attempts);
                if (JobRaw j = victim->deque.steal_if_nonempty()) {
                    HG_STAT(data->jobs_stolen.fetch_add(1, std::memory_order_relaxed));
                    HG_STAT(count_steals(data, attempts, true));
                    more = victim->deque.size() >= 2;
                    return j;
                }
            }
        }
        HG_STAT(count_steals(data, attempts, false));
        if (auto opt = injector_.try_pop_front()) {             // external work
            more = !injector_.empty();
            return *opt;
        }
        return nullptr;
    }

    // recycle_scratch is false when this runs on the thread that submitted the job because
    // every queue was full (see enqueue): on_job_complete_ resets that thread's scratch arena,
    // and the submitter -- a job further out on the stack, or code seeding work from a thread
    // that is not a worker -- still holds live allocations in it. The job's own scratch sits
    // above the submitter's high-water mark, and run_inline_ gives it back when the job
    // returns.
    // `data` is null when a non-worker submitter runs an overflowed job on its own thread;
    // the per-worker counters simply do not apply to it.
    void run_job(WorkerData* data, JobRaw job, bool recycle_scratch = true) {
        HG_STAT(if (job->runs.fetch_add(1, std::memory_order_acq_rel) != 0)
                    double_executions_.fetch_add(1, std::memory_order_relaxed));
        if (data) data->jobs_executing.fetch_add(1);
        try {
            // After an error the job is discarded unrun, so the queues drain without running the
            // jobs they hold or the children those would submit.
            if (error_type_.load(std::memory_order_acquire) == ErrorType::None) job->execute();
        } catch (const hgcommon::CapacityExhausted& e) {
            record_error_message(e.what());
            latch_error(ErrorType::CapacityExhausted);
        } catch (const std::bad_alloc& e) {
            record_error_message(e.what());
            latch_error(ErrorType::OutOfMemory);
        } catch (const std::exception& e) {
            const bool aborted = std::strcmp(e.what(), "Operation aborted") == 0;
            if (!aborted) record_error_message(e.what());
            latch_error(aborted ? ErrorType::Aborted : ErrorType::Exception);
        } catch (...) {
            record_error_message("non-std exception");
            latch_error(ErrorType::Unhandled);
        }
        delete job;
        if (recycle_scratch && on_job_complete_) on_job_complete_();  // recycle per-worker scratch
        if (data) {
            data->jobs_executing.fetch_sub(1);
            HG_STAT(data->jobs_executed.fetch_add(1, std::memory_order_relaxed));
        }

        // Notify the completion waiter only at quiescence (this job brings completed up to
        // submitted), not on every job.
        //
        // THE ORDER IS THE MECHANISM, and the wake is only an optimisation. Bumping the sequence
        // BEFORE reading the waiter count is what makes a stale read of that count harmless: a
        // waiter that already sampled the old sequence finds it stale and re-checks rather than
        // parking, and one that has not yet sampled reads the new value. The waiter parks on this
        // very word under a value compare, so the write this might miss is the one it compares
        // against.
        //
        // No StoreLoad barrier is needed between the two, and that is checked rather than
        // assumed: with the barrier removed a model checker cannot tell the two apart. Inverting
        // the order -- decide the wake, then publish -- is what would lose the wakeup, and is why
        // it is written this way round.
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
        hgcommon::exempt_this_thread_from_power_throttling();
        // Bind this worker if the caller named a core set. Worker i takes the i-th CPU in that
        // set; more workers than CPUs means they share, which is the caller's choice to make
        // rather than this loop's to refuse. Empty set: placement stays the operating system's.
        if (!worker_cpus_.empty()) {
            const unsigned cpu = worker_cpus_[index % worker_cpus_.size()];
            if (!hgcommon::pin_this_thread_to_cpu(cpu))
                HG_STAT(pin_failures_.fetch_add(1, std::memory_order_relaxed));
        }
        // Past the binding attempt: start() waits on this, which is what makes pin_failures()
        // a settled count once start() returns. Release pairs with start()'s acquire so the
        // failure increment above is visible when the count is.
        workers_entered_.fetch_add(1, std::memory_order_release);
        uint64_t rng = static_cast<uint64_t>(index) * 2654435761ull + 1ull;
        const uint64_t awake_bit = index < 64 ? uint64_t{1} << index : 0;
        bool woken = false;   // the last park returned and no job has run since

        while (true) {
            if (JobRaw job = data->deque.pop()) {               // own work (LIFO)
                HG_STAT(data->woke = false);
                woken = false;
                run_job(data, job);
                continue;
            }
            // THE SEARCH. Under the throttle this worker counts itself as a searcher (submits skip
            // their wake while one searches) or, past max(1, busy / 2) searchers and not just
            // woken, declines and goes straight to the last look. A searcher that took a job
            // leaves the count and, if it was the last one and the queue it took from still holds
            // work, wakes the next worker (hgcommon/park_gate.hpp).
            using Search = hgcommon::ParkGate::Search;
            const Search search = park_gate_.search_begin(woken);
            const bool counted = search == Search::Counted;
            bool more = false;
            JobRaw stolen = search != Search::Declined ? steal_work(data, rng, index, more) : nullptr;
            if (stolen) {
                if (counted) {
                    const bool woke = park_gate_.search_found(data->wake_home, more);
                    if (woke) wake_hint_.store(data->last_victim, std::memory_order_relaxed);
                    HG_STAT(if (woke) data->wakes_issued.fetch_add(1, std::memory_order_relaxed));
                }
                HG_STAT(data->woke = false);
                woken = false;
                run_job(data, stolen);
                continue;
            }
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None ||
                data->stop.load(std::memory_order_acquire)) {   // error, or shutdown and drained
                if (counted) park_gate_.search_cancel();
                break;
            }

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
            // ANNOUNCING, THE LAST LOOK AND THE PARK ARE ONE STEP, and the gate owns their order:
            // a counted searcher leaves the searching count, both idle counts are published, and
            // only then is the sequence sampled and the last look taken, which is what the
            // barrier pairing in wake_one needs. The two callables are the caller's -- where work
            // is found, and whether this worker must leave -- and are no part of the protocol.
            //
            // The awake mask loses this worker's bit for the duration: its deque is empty from
            // here until it runs a job again, so steal_work skips it (see awake_mask_).
            [[maybe_unused]] bool parked = false;
            if (awake_bit) awake_mask_.fetch_and(~awake_bit, std::memory_order_relaxed);
            JobRaw found = park_gate_.park_unless(
                data->domain, counted,
                [&] { return find_work_exhaustive(data); },
                [&] {
                    const bool keep = error_type_.load(std::memory_order_acquire) == ErrorType::None &&
                                      !data->stop.load(std::memory_order_acquire);
                    HG_STAT(if (keep) {
                        if (data->woke) data->wakes_empty.fetch_add(1, std::memory_order_relaxed);
                        parked = true;
                    });
                    return keep;
                });
            if (awake_bit) awake_mask_.fetch_or(awake_bit, std::memory_order_relaxed);
            if (found) {
                HG_STAT(data->woke = false);
                woken = false;
                run_job(data, found);
                continue;
            }
            HG_STAT(data->woke = parked);
            woken = true;
            // The worker that signalled last, or the queue it found work in: the first victim
            // this worker's search reads (a stale value costs one read).
            data->last_victim = wake_hint_.load(std::memory_order_relaxed);
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) break;
            if (data->stop.load(std::memory_order_acquire)) break;
        }

        t_sys_ = nullptr;
        t_worker_ = nullptr;
        workers_exited_.fetch_add(1, std::memory_order_release);
        quiescence_seq_.fetch_add(1, std::memory_order_release);
        hgcommon::unpark_all(quiescence_seq_);
    }

    void drain_and_delete() {
        for (auto& w : workers_) {
            while (JobRaw j = w->deque.pop()) delete j;
        }
        while (auto opt = injector_.try_pop_front()) delete *opt;
    }

    // A job run on the thread that submitted it, because there was no room to queue it (see
    // enqueue). It must not recycle that thread's scratch arena: whatever submitted it still
    // holds live allocations there -- on a worker the job further out on this stack, on any
    // other thread the code seeding work (evolve_more builds its resume lists there). So it
    // gives back exactly the scratch it took: on_inline_begin_ marks the arena before it runs,
    // and on_inline_end_ releases back to that mark after.
    //
    // AT MOST ONE INLINE JOB OF THIS SYSTEM IS ON A THREAD'S STACK. A job that overflows while
    // the thread already runs one goes on t_inline_pending_, and the outermost run_inline_ runs
    // those jobs one after another, last in first out, before it returns. Run nested, every
    // job of a chain that keeps the queues full (rewrite, scan, expand, rewrite, ...) adds
    // frames, and a 7219-edge state at 2000 steps overflows an 8 MB worker stack. A deferred
    // job was submitted and is not yet complete, so quiescence cannot be reached before it
    // runs, and the job further out does not return before it runs. If the list cannot grow
    // (bad_alloc), the job runs nested.
    void run_inline_(WorkerData* data, JobRaw raw) {
        if (t_inline_sys_ == this) {
            try {
                t_inline_pending_->push_back(raw);
                return;
            } catch (const std::bad_alloc&) {
                run_inline_one_(data, raw);
                return;
            }
        }
        std::vector<JobRaw> pending;
        JobSystem* const outer_sys = t_inline_sys_;
        std::vector<JobRaw>* const outer_pending = t_inline_pending_;
        t_inline_sys_ = this;
        t_inline_pending_ = &pending;
        run_inline_one_(data, raw);
        while (!pending.empty()) {
            const JobRaw next = pending.back();
            pending.pop_back();
            run_inline_one_(data, next);
        }
        t_inline_sys_ = outer_sys;
        t_inline_pending_ = outer_pending;
    }

    void run_inline_one_(WorkerData* data, JobRaw raw) {
        if (on_inline_begin_) on_inline_begin_();
        run_job(data, raw, /*recycle_scratch=*/false);
        if (on_inline_end_) on_inline_end_();
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
        // here, possibly inside the submitting job (run_inline_); the next drain-loop job
        // recycles as usual.
        if (serial_) {
            if (injector_.try_push_back(raw)) return;
            HG_STAT(inline_runs_.fetch_add(1, std::memory_order_relaxed));
            run_inline_(nullptr, raw);
            return;
        }
        const bool on_worker = (t_sys_ == this && t_worker_ != nullptr);

        // COMPLETE-THEN-SUBMIT IS THE ONE THING QUIESCENCE CANNOT DEFEND AGAINST, so it is
        // counted rather than assumed. The predicate is sound only while a job submits its
        // children BEFORE it returns: a submit made after the submitting job has been booked
        // complete opens a window in which the counters agree and every queue is empty while a
        // child is still owed, and NO ordering of the reads closes it, because at that instant
        // there is nothing to see. verification/tla/Quiescence.tla is that argument, and
        // MCQuiescenceLateSubmit is the violation -- so this is a precondition the engine owes
        // the protocol, and this is where it is checked.
        //
        // jobs_executing is decremented after the job body returns, so a worker enqueuing with a
        // zero count is enqueuing from outside any job. Off-worker submits are the caller seeding
        // work and are not this.
        //
        // One relaxed load of a line this worker already owns -- run_job writes jobs_executing on
        // every job -- and a branch that predicts perfectly, on a path that then pushes to a
        // deque.
        if (on_worker && t_worker_->jobs_executing.load(std::memory_order_relaxed) == 0)
            HG_STAT(late_submits_.fetch_add(1, std::memory_order_relaxed));

        if (on_worker && t_worker_->deque.push(raw)) {        // node-local, the common case
            wake_one_worker();
            return;
        }
        if (injector_.try_push_back(raw)) {
            wake_one_worker();
            return;
        }
        // Both full. Run it here (run_inline_).
        HG_STAT(inline_runs_.fetch_add(1, std::memory_order_relaxed));
        run_inline_(on_worker ? t_worker_ : nullptr, raw);
    }

public:
    // SERIAL MODE (serial = true): no worker threads exist and none are ever spawned --
    // start() only arms the counters, submit() routes to the injector, and
    // wait_for_completion() drains it inline on the calling thread in FIFO order.
    // Jobs submitted by a running job land behind it and run in submission order, so a
    // run is deterministic by construction. This is the single-threaded execution mode
    // (and the WebAssembly path, where spawning threads is not available): everything
    // the workers would do happens on the thread that waits.
    explicit JobSystem(size_t num_threads = 0, size_t queue_capacity = HG_JOB_QUEUE_CAPACITY,
                       bool serial = false)
        : injector_(HG_JOB_INJECTOR_CAPACITY),
          num_threads_(serial ? 0
                              : (num_threads == 0 ? hgcommon::default_worker_count()
                                                  : num_threads)),
          queue_capacity_(queue_capacity == 0 ? HG_JOB_QUEUE_CAPACITY : queue_capacity),
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
    // Register the calls made around a job run on the thread that submitted it because every
    // queue was full (run_inline_): `begin` before it runs and `end` after, on that thread.
    // Used to give back exactly the scratch the job
    // took, where on_job_complete_ would recycle what its submitter still holds.
    void set_on_inline_job(std::function<void()> begin, std::function<void()> end) {
        on_inline_begin_ = std::move(begin);
        on_inline_end_ = std::move(end);
    }

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

    // Zero, or the quiescence predicate is unsound for this run. See enqueue().
    size_t late_submits() const { return late_submits_.load(std::memory_order_relaxed); }
    // Zero on every sound run: see the counters' comment.
    size_t double_executions() const { return double_executions_.load(std::memory_order_relaxed); }
    size_t abandoned_at_quiescence() const { return abandoned_at_quiescence_.load(std::memory_order_relaxed); }
    size_t abandoned_already_run() const { return abandoned_already_run_.load(std::memory_order_relaxed); }

    void start() {
        if (is_running_.load()) return;
        total_submitted_.store(0);
        total_completed_.store(0);
        late_submits_.store(0, std::memory_order_relaxed);
        double_executions_.store(0, std::memory_order_relaxed);
        abandoned_at_quiescence_.store(0, std::memory_order_relaxed);
        abandoned_already_run_.store(0, std::memory_order_relaxed);
        error_type_.store(ErrorType::None, std::memory_order_relaxed);
        error_message_state_.store(0, std::memory_order_relaxed);

        for (size_t i = 0; i < workers_.size(); ++i) {
            workers_[i]->stop.store(false, std::memory_order_relaxed);
        }
#if HG_ENGINE_STATS
        for (auto& w : workers_) {
            w->wakes_issued.store(0, std::memory_order_relaxed);
            w->wakes_empty.store(0, std::memory_order_relaxed);
            w->steal_attempts.store(0, std::memory_order_relaxed);
            w->steal_failures.store(0, std::memory_order_relaxed);
            w->woke = false;
        }
        wakes_issued_off_worker_.store(0, std::memory_order_relaxed);
        inline_runs_.store(0, std::memory_order_relaxed);
#endif
        workers_entered_.store(0, std::memory_order_relaxed);
        awake_mask_.store(workers_.size() >= 64 ? ~uint64_t{0}
                                                : (uint64_t{1} << workers_.size()) - 1,
                          std::memory_order_relaxed);
        workers_exited_.store(0, std::memory_order_relaxed);
        ensure_default_cpu_order();               // before the map: the map is derived from it
        build_domain_map();                       // before any worker exists: parks index it
        build_cache_peers();                      // before any worker exists: see peer_begin_
#if defined(HG_VERIFICATION)
        spawn_workers();
#else
        // A spawn that fails (std::system_error when the thread or its stack cannot be created)
        // stops and joins the workers already running before the error leaves start(): a
        // joinable std::thread destroyed with this object calls std::terminate.
        size_t spawned = 0;
        try {
            for (; spawned < workers_.size(); ++spawned)
                workers_[spawned]->thread.spawn(this, workers_[spawned].get(), spawned);
        } catch (...) {
            for (auto& worker : workers_) worker->stop.store(true, std::memory_order_release);
            wake_all_workers();
            for (size_t i = 0; i < spawned; ++i) workers_[i]->thread.join();
            throw;
        }
#endif
        // Wait until every worker has passed its binding attempt, so pin_failures() is a
        // settled count when start() returns rather than a snapshot racing worker startup --
        // on a small machine a spawned thread can lag past an entire short workload before it
        // first runs. One yield loop, once per start; the workers park themselves if no work
        // arrives, so this costs thread-spawn latency and nothing else.
        // Not in the verification build: it has no pinning, so the count settles nothing, and the
        // checker explored no execution in which this loop read the workers' increments (0
        // complete, 24 blocked on job_system_error_wait).
#if !defined(HG_VERIFICATION)
        while (workers_entered_.load(std::memory_order_acquire) < workers_.size())
            std::this_thread::yield();
#endif
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
            worker->thread.join();
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
            // After an error the workers run what their queues hold and leave; the wait ends
            // when the last one has left. Each leaving worker bumps quiescence_seq_.
            const uint32_t q = quiescence_seq_.load(std::memory_order_acquire);
            if (error_type_.load(std::memory_order_acquire) != ErrorType::None) {
#if defined(HG_ERROR_WAIT_EARLY)
                return;   // calibration only (verification/genmc/job_system_error_wait.cpp)
#endif
                if (workers_exited_.load(std::memory_order_acquire) == workers_.size()) return;
            } else if (is_quiescent()) {
                HG_STAT(count_abandoned());
                return;
            }
            hgcommon::park_if_equal(quiescence_seq_, q);
        }
    }

private:
    // Stats builds only. The system reads as quiescent, so every worker is parked or between
    // jobs and nothing is being handed out: any job the queues still hold is one quiescence
    // should have waited for. Taken through steal(), the entry any thread may use, and freed;
    // a run that counts one here has already lost it.
    void count_abandoned() {
        for (auto& w : workers_) {
            while (JobRaw j = w->deque.steal()) {
                abandoned_at_quiescence_.fetch_add(1, std::memory_order_relaxed);
                if (j->runs.load(std::memory_order_acquire) != 0)
                    abandoned_already_run_.fetch_add(1, std::memory_order_relaxed);
                delete j;
            }
        }
        while (auto opt = injector_.try_pop_front()) {
            abandoned_at_quiescence_.fetch_add(1, std::memory_order_relaxed);
            if ((*opt)->runs.load(std::memory_order_acquire) != 0)
                abandoned_already_run_.fetch_add(1, std::memory_order_relaxed);
            delete *opt;
        }
    }
public:

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

    size_t park_waits() const { return park_gate_.park_waits(); }

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

#if HG_ENGINE_STATS
    // The scheduler's idle-side traffic over one start(): docs/TUNING.md 2.4.
    struct SchedulerCounters {
        size_t jobs_executed = 0;
        size_t jobs_stolen = 0;
        size_t wakes_issued = 0;     // parked workers signalled by submits
        size_t wakes_empty = 0;      // a worker returned from a park and parked again with no job between
        size_t steal_attempts = 0;   // victim deque reads
        size_t steal_failures = 0;   // of those, the ones that took nothing
        size_t inline_runs = 0;      // jobs run on the submitter because both queues were full
        size_t parks = 0;            // park calls
        int searching_max = 0;       // most workers searching at once
        size_t busiest_worker_jobs = 0;
        size_t idlest_worker_jobs = 0;
        size_t workers_with_jobs = 0;
    };

    SchedulerCounters scheduler_counters() const {
        SchedulerCounters c;
        c.idlest_worker_jobs = workers_.empty() ? 0 : SIZE_MAX;
        for (const auto& w : workers_) {
            const size_t ran = w->jobs_executed.load(std::memory_order_relaxed);
            c.jobs_executed += ran;
            c.jobs_stolen += w->jobs_stolen.load(std::memory_order_relaxed);
            c.wakes_issued += w->wakes_issued.load(std::memory_order_relaxed);
            c.wakes_empty += w->wakes_empty.load(std::memory_order_relaxed);
            c.steal_attempts += w->steal_attempts.load(std::memory_order_relaxed);
            c.steal_failures += w->steal_failures.load(std::memory_order_relaxed);
            c.busiest_worker_jobs = std::max(c.busiest_worker_jobs, ran);
            c.idlest_worker_jobs = std::min(c.idlest_worker_jobs, ran);
            c.workers_with_jobs += ran != 0 ? 1u : 0u;
        }
        c.wakes_issued += wakes_issued_off_worker_.load(std::memory_order_relaxed);
        c.inline_runs = inline_runs_.load(std::memory_order_relaxed);
        c.parks = park_gate_.park_waits();
        c.searching_max = park_gate_.searching_max();
        return c;
    }
#endif

    // The park domain map (build_domain_map): how many, which one worker i parks on, and which
    // one its submits wake first. Settled once start() returns.
    size_t park_domains() const { return num_domains_; }
    unsigned worker_park_domain(size_t i) const { return workers_[i]->domain; }
    unsigned worker_wake_home(size_t i) const { return workers_[i]->wake_home; }
    bool wake_throttled() const { return park_gate_.throttled(); }

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