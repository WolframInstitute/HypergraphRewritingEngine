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
// SEPARATE FROM THE JOB SYSTEM because that is what makes it checkable. It touches nothing but its
// own atomics and the caller's "is there work" predicate, so a model checker can be handed the
// protocol rather than a running JobSystem -- which is not reachable: constructing one prunes to
// 2,659 lines and verifies, starting one prunes to 8,952 and segfaults GenMC v0.17.0.
// verification/genmc/job_system_no_lost_wakeup.cpp runs this header.

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

    void seat(Domain* domains, uint32_t n) { dom_ = domains; n_ = n; reset(); }

    void reset() {
        for (uint32_t d = 0; d < n_; ++d) {
            // seq is NOT reset: a worker parked on a value from a previous run would be left
            // watching a word that had gone backwards. It only ever has to DIFFER from what a
            // waiter sampled, so it runs monotonically for the life of the gate.
            dom_[d].idle.store(0, std::memory_order_relaxed);
        }
        idle_.store(0, std::memory_order_relaxed);
        park_waits_.store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
    }

    uint32_t domains() const { return n_; }
    int idle_workers() const { return idle_.load(HG_PG_OBSERVE); }
    size_t park_waits() const { return park_waits_.load(std::memory_order_relaxed); }

    // The submitter's side. The caller pushes the job immediately before calling this, so the
    // barrier sits between the push and the read of the idle counts.
    void wake_one(unsigned home) {
        HG_PG_BARRIER();
        if (idle_.load(HG_PG_OBSERVE) <= 0) return;
        if (n_ == 0) return;
        if (home >= n_) home = 0;

        // The submitter's own domain first: a job pushed here is warm here.
        if (dom_[home].idle.load(HG_PG_OBSERVE) > 0) {
            dom_[home].seq.fetch_add(1, std::memory_order_release);
            unpark_one(dom_[home].seq);
            return;
        }
        // FALLING BACK TO THE OTHER DOMAINS IS WHAT KEEPS THIS FROM BEING A LOST WAKEUP. A job
        // whose own domain has nobody idle must still reach whoever is.
#if defined(HG_PARK_GATE_NO_REMOTE_SCAN)
        return;   // calibration only; see the note on HG_PARK_GATE_WEAK_ORDERS above
#else
        for (uint32_t d = 0; d < n_; ++d) {
            const uint32_t k = (home + 1 + d) % n_;
            if (dom_[k].idle.load(HG_PG_OBSERVE) <= 0) continue;
            dom_[k].seq.fetch_add(1, std::memory_order_release);
            unpark_one(dom_[k].seq);
            return;
        }
#endif
    }

    void wake_all() {
        for (uint32_t d = 0; d < n_; ++d) {
            dom_[d].seq.fetch_add(1, std::memory_order_release);
            unpark_all(dom_[d].seq);
        }
    }

    // The worker's side. `look()` is the last exhaustive search for work and returns something
    // contextually false when there is none; `keep_waiting()` is false once the worker must leave
    // (an error was latched, or shutdown was asked for).
    //
    // Returns what look() found. A non-null result means the gate did NOT park.
    template <class Look, class KeepWaiting>
    auto park_unless(unsigned home, Look&& look, KeepWaiting&& keep_waiting) -> decltype(look()) {
        if (n_ == 0) return look();
        if (home >= n_) home = 0;

        // Announce BEFORE sampling the sequence and before the last look, which is what the
        // barrier pairing needs.
        idle_.fetch_add(1, HG_PG_ANNOUNCE);
        dom_[home].idle.fetch_add(1, HG_PG_ANNOUNCE);
        HG_PG_BARRIER();

        const uint32_t seq = dom_[home].seq.load(std::memory_order_acquire);
        if (auto job = look()) {
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
    Domain*  dom_ = nullptr;
    uint32_t n_ = 0;
    // The GLOBAL idle count, which is what lets a submit skip the per-domain scan entirely.
    alignas(64) std::atomic<int> idle_{0};
    std::atomic<size_t> park_waits_{0};
};

}  // namespace common
}  // namespace HG_NAMESPACE
