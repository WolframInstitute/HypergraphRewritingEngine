#pragma once
#include "hgcommon/core.hpp"
#include "hgcommon/namespace.hpp"
//
// THE PARK/WAKE HANDSHAKE: a worker with nothing to do sleeps, and a submitter wakes one.
//
// THE FAILURE IT EXISTS TO PREVENT is a job sitting in a queue with nobody awake to take it. That
// is not a slow run, it is a run that never finishes, and it cannot be found by inspecting output.
//
// THE SHAPE. Each side writes one location and then reads the other's:
//
//   the worker     announces itself idle, then takes one last look for work
//   the submitter  publishes the job, then reads whether anyone is idle
//
// Under acquire/release both may read the value from before the other's write -- the submitter
// sees nobody idle and skips the wake, the worker sees no work and parks -- and the job is left
// queued with the worker asleep. A StoreLoad barrier on EACH side forbids that outcome; see
// hgcommon/rendezvous.hpp for the class and for the shapes that resemble it and are safe.
//
// ANNOUNCE BEFORE THE LAST LOOK, and sample the sequence before it too. Anything submitted after
// the sample moves the sequence, so the park returns immediately rather than sleeping through it;
// anything submitted before the sample is found by the last look. The park compares the sequence
// against what was sampled, which is what makes the wake an optimisation rather than the
// mechanism.
//
// TWO COUNTS, not one. The global count lets a submit skip the wake machinery entirely; the
// per-domain count tells a submitter whether a worker sharing its cache is available, and a job
// handed to a worker in the submitter's own domain stays in that cache. Both are announced before
// the last look, and both are released on every path out.
//
// THE SEARCHING COUNT (the throttle). A worker whose own deque is empty counts itself into
// `searching_` while it steals (search_begin). A submit skips its wake while a worker searches,
// unless the submitter's own backlog exceeds twice the searchers. That skip is the same StoreLoad
// pair as the idle count: the submitter publishes the job, passes the barrier and reads
// `searching_`; a searcher that found nothing leaves the count, announces itself idle, passes the
// barrier and takes the last look. A submitter that read the searcher in the count is therefore
// ordered before that look, which finds the job. A searcher that found a job leaves the count in
// search_found and, if it was the last searcher, the queue it took from still holds work and a
// worker is idle, wakes one more: wakes chain one at a time while there is work to find. At most max(1, busy / 2) workers search at once; a
// worker over that bound goes straight to the last look and the park, unless it was just woken.
//
// SEPARATE FROM THE JOB SYSTEM because that is what makes it checkable. It touches nothing but its
// own atomics and the caller's "is there work" predicate, so a model checker can be handed the
// protocol rather than a running JobSystem -- which is not reachable: constructing one prunes to
// 2,659 lines and verifies, starting one prunes to 8,952 and segfaults GenMC v0.17.0.
// verification/genmc/job_system_no_lost_wakeup{,_domains}.cpp run this header, and
// verification/tla/Quiescence.tla models the protocol.

#include "hgcommon/park.hpp"
#include "hgcommon/rendezvous.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

// HG_PARK_GATE_WEAK_ORDERS drops both sides of the handshake to release/acquire, which is what
// the property rests on and therefore what a harness must be able to remove. It is set only by
// verification/genmc/run.sh's HG_HARNESS_DEFINES, so the calibration is a command anyone can
// repeat rather than an assertion in a comment. The defect is reinstated HERE, in the real path,
// rather than in a copy of it that could drift from what ships.
//
// Two more calibrations act on the searching count only:
//   HG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER  a searcher leaves the count after the barrier, so
//                                            the decrement and the submitter's read of the count
//                                            are no longer ordered by it (fails under RC11 only)
//   HG_PARK_GATE_SEARCH_LEAVE_LATE           a searcher leaves the count after its last look
//                                            (fails under sequential consistency too)
#if defined(HG_PARK_GATE_WEAK_ORDERS)
#  define HG_PG_ANNOUNCE std::memory_order_release
#  define HG_PG_OBSERVE  std::memory_order_acquire
#  define HG_PG_BARRIER() ((void)0)
#else
#  define HG_PG_ANNOUNCE std::memory_order_seq_cst
#  define HG_PG_OBSERVE  std::memory_order_seq_cst
#  define HG_PG_BARRIER() rendezvous_barrier<rv::WorkerParkWake>()
#endif

namespace HG_NAMESPACE {
namespace common {

class ParkGate {
public:
    // ONE CACHE LINE PER DOMAIN. Every submit reads these and every park writes them, so two
    // domains sharing a line would reintroduce between domains exactly the sharing that splitting
    // by domain exists to remove.
    struct alignas(64) Domain {
        std::atomic<uint32_t> seq{0};
        std::atomic<int>      idle{0};
    };

    // `workers` bounds the searchers (max(1, busy / 2)). `throttled` false makes every submit
    // signal while a worker is idle (HG_TUNE_WAKE_POLICY=per-submit, kept for the wall-clock A/B).
    void seat(Domain* domains, uint32_t n, uint32_t workers = 1, bool throttled = true) {
        dom_ = domains;
        n_ = n;
        workers_ = static_cast<int>(workers);
        throttled_ = throttled;
        reset();
    }

    void reset() {
        for (uint32_t d = 0; d < n_; ++d) {
            // seq is NOT reset: a worker parked on a value from a previous run would be left
            // watching a word that had gone backwards. It only ever has to DIFFER from what a
            // waiter sampled, so it runs monotonically for the life of the gate.
            dom_[d].idle.store(0, std::memory_order_relaxed);
        }
        idle_.store(0, std::memory_order_relaxed);
        searching_.store(0, std::memory_order_relaxed);
        park_waits_.store(0, std::memory_order_relaxed);
        HG_STAT(searching_max_.store(0, std::memory_order_relaxed));
        std::atomic_thread_fence(std::memory_order_release);
    }

    uint32_t domains() const { return n_; }
    bool throttled() const { return throttled_; }
    int idle_workers() const { return idle_.load(HG_PG_OBSERVE); }
    size_t park_waits() const { return park_waits_.load(std::memory_order_relaxed); }
#if HG_ENGINE_STATS
    int searching_max() const { return searching_max_.load(std::memory_order_relaxed); }
#endif

    // The submitter's side. The caller pushes the job immediately before calling this, so the
    // barrier sits between the push and the reads of the counts. `backlog()` returns the number
    // of jobs the submitter holds queued (its own deque); it is called only when a worker is idle
    // and another is searching. Returns whether a parked worker was signalled.
    template <class Backlog>
    bool wake_one(unsigned home, Backlog&& backlog) {
        HG_PG_BARRIER();
        if (idle_.load(HG_PG_OBSERVE) <= 0) return false;
        if (throttled_) {
            const int s = searching_.load(HG_PG_OBSERVE);
            if (s > 0 && backlog() <= 2u * static_cast<size_t>(s)) return false;
        }
        return signal(home);
    }

    // A submitter with no queue of its own (a thread that is not a worker): backlog 0.
    bool wake_one(unsigned home) { return wake_one(home, [] { return size_t{0}; }); }

    void wake_all() {
        for (uint32_t d = 0; d < n_; ++d) {
            dom_[d].seq.fetch_add(1, std::memory_order_release);
            unpark_all(dom_[d].seq);
        }
    }

    // What search_begin decided for a worker whose own deque is empty.
    enum class Search : uint8_t {
        Counted,    // counted in the searching count; steal, then search_found or park_unless(.., true, ..)
        Uncounted,  // per-submit policy in a release build: steal, nothing counted
        Declined,   // the bound was reached: no steal; park_unless(.., false, ..) takes the last look
    };

    // `woken` is true for a worker whose last park returned: it was woken for work, so the bound
    // does not apply to it. In a stats build the per-submit policy counts searchers too, for
    // searching_max(); its submits never read the count.
    Search search_begin(bool woken) {
#if !HG_ENGINE_STATS
        if (!throttled_) return Search::Uncounted;
#endif
        const int s = searching_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (throttled_ && !woken) {
            const int busy = workers_ - idle_.load(std::memory_order_relaxed) - s;
            if (s > 1 && s > busy / 2) {
                searching_.fetch_sub(1, std::memory_order_relaxed);
                return Search::Declined;
            }
        }
        HG_STAT({
            int seen = searching_max_.load(std::memory_order_relaxed);
            while (s > seen &&
                   !searching_max_.compare_exchange_weak(seen, s, std::memory_order_relaxed)) {}
        });
        return Search::Counted;
    }

    // A counted searcher took a job; `more` says the queue it took from still holds work. Under the
    // throttle the last searcher out wakes one more worker if work remains and a worker is idle,
    // so wakes chain one at a time while there is work to find. Returns whether a parked worker
    // was signalled. No property rests on this wake: it adds a worker, and a skipped one leaves
    // the work to the submitters' wakes and the workers' own searches.
    bool search_found(unsigned home, bool more) {
        const int before = searching_.fetch_sub(1, std::memory_order_relaxed);
        if (!throttled_ || before != 1 || !more) return false;
        if (idle_.load(std::memory_order_relaxed) <= 0) return false;
        return signal(home);
    }

    // A counted searcher leaves without parking (an error was latched or shutdown was asked for).
    void search_cancel() { searching_.fetch_sub(1, std::memory_order_relaxed); }

    // The worker's side of the park. `searching` says whether the caller is counted in the
    // searching count (search_begin returned Counted and it found nothing); it leaves the count
    // before the idle announcement. `look()` is the last exhaustive search for work and returns
    // something contextually false when there is none; `keep_waiting()` is false once the worker
    // must leave (an error was latched, or shutdown was asked for).
    //
    // Returns what look() found. A non-null result means the gate did NOT park.
    template <class Look, class KeepWaiting>
    auto park_unless(unsigned home, bool searching, Look&& look, KeepWaiting&& keep_waiting)
        -> decltype(look()) {
#if !defined(HG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER) && !defined(HG_PARK_GATE_SEARCH_LEAVE_LATE)
        if (searching) searching_.fetch_sub(1, HG_PG_ANNOUNCE);
#endif
        if (n_ == 0) {
#if defined(HG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER) || defined(HG_PARK_GATE_SEARCH_LEAVE_LATE)
            if (searching) searching_.fetch_sub(1, HG_PG_ANNOUNCE);
#endif
            return look();
        }
        if (home >= n_) home = 0;

        // Announce BEFORE sampling the sequence and before the last look, which is what the
        // barrier pairing needs.
        idle_.fetch_add(1, HG_PG_ANNOUNCE);
        dom_[home].idle.fetch_add(1, HG_PG_ANNOUNCE);
        HG_PG_BARRIER();
#if defined(HG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER)
        if (searching) searching_.fetch_sub(1, std::memory_order_relaxed);   // calibration only
#endif

        const uint32_t seq = dom_[home].seq.load(std::memory_order_acquire);
        auto job = look();
#if defined(HG_PARK_GATE_SEARCH_LEAVE_LATE)
        if (searching) searching_.fetch_sub(1, std::memory_order_relaxed);   // calibration only
#endif
        if (job) {
            dom_[home].idle.fetch_sub(1, std::memory_order_relaxed);
            idle_.fetch_sub(1, std::memory_order_relaxed);
            return job;
        }
        if (keep_waiting()) {
            HG_STAT(park_waits_.fetch_add(1, std::memory_order_relaxed));
            park_if_equal(dom_[home].seq, seq);
        }
        dom_[home].idle.fetch_sub(1, std::memory_order_relaxed);
        idle_.fetch_sub(1, std::memory_order_relaxed);
        return decltype(look()){};
    }

private:
    // Move one idle domain's park word and wake one waiter on it: the home domain first, then the
    // others in order. Returns whether a domain with an idle worker was found.
    bool signal(unsigned home) {
        if (n_ == 0) return false;
        if (home >= n_) home = 0;

        // The submitter's own domain first: a job pushed here is warm here.
        if (dom_[home].idle.load(HG_PG_OBSERVE) > 0) {
            dom_[home].seq.fetch_add(1, std::memory_order_release);
            unpark_one(dom_[home].seq);
            return true;
        }
        // FALLING BACK TO THE OTHER DOMAINS IS WHAT KEEPS THIS FROM BEING A LOST WAKEUP. A job
        // whose own domain has nobody idle must still reach whoever is.
#if defined(HG_PARK_GATE_NO_REMOTE_SCAN)
        return false;   // calibration only; see the note on HG_PARK_GATE_WEAK_ORDERS above
#else
        for (uint32_t d = 0; d < n_; ++d) {
            const uint32_t k = (home + 1 + d) % n_;
            if (dom_[k].idle.load(HG_PG_OBSERVE) <= 0) continue;
            dom_[k].seq.fetch_add(1, std::memory_order_release);
            unpark_one(dom_[k].seq);
            return true;
        }
        return false;
#endif
    }

    Domain*  dom_ = nullptr;
    uint32_t n_ = 0;
    int      workers_ = 1;
    bool     throttled_ = true;
    // The GLOBAL idle count, which is what lets a submit skip the per-domain scan entirely.
    alignas(64) std::atomic<int> idle_{0};
    std::atomic<size_t> park_waits_{0};
    // Read by every submit and written by every search: its own line.
    alignas(64) std::atomic<int> searching_{0};
#if HG_ENGINE_STATS
    std::atomic<int> searching_max_{0};
#endif
};

}  // namespace common
}  // namespace HG_NAMESPACE
