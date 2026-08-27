#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdint>

// Where a run's cycles go, by phase, on the HOST.
//
// The device has had this since the persistent kernel landed (clock64 deltas per block, summed
// at exit into PersistentEvolveStats). The host had nothing comparable, and the substitutes are
// both wrong here: wall-clock timers cannot see inside a run where every worker is in some
// phase at once, and instruction counts mispredict time badly on this engine -- removing 16% of
// executed instructions once moved the wall clock by less than the machine's run-to-run spread.
// Cycles per phase, summed over workers, is the quantity that decides what to optimise.
//
// COMPILED OUT UNLESS ASKED FOR. Everything below is behind HG_PHASE_TIMING, set by the CMake
// option of the same name (default OFF). With it off there is no counter storage, no thread
// slot, no read of the cycle counter and no branch: PhaseTimer is an empty object whose
// constructor and destructor do nothing, so a release build carries no trace of it. This is
// deliberate -- the counters occupy one cache line per worker per phase, and a shipping build
// should not pay a line, a branch or an rdtsc for a measurement nobody is taking.
//
// WHAT A BUCKET MEANS. The counter runs while a thread is stalled, descheduled or waiting, so a
// bucket is ELAPSED cycles inside that phase, not issued work. Buckets are read as fractions of
// their sum, exactly as the device's are. Summed across workers they exceed wall time by
// roughly the thread count, which is the point: a phase that is 60% of worker cycles is where
// the parallel machine is spending itself, whatever the wall clock says.

// EVERYTHING BELOW STAYS IN THIS HEADER. Two reasons, and neither is that it was skipped.
// It is all inside HG_PHASE_TIMING, off by default, so a shipping build compiles none of it
// and there is nothing for a translation unit to carry. And PhaseTimer's constructor and
// destructor ARE the measurement: outlining them puts a call inside the interval whose
// cycles they count, which changes the number being read.
#ifdef HG_PHASE_TIMING
#include "hgcommon/portable_intrinsics.hpp"
#include <atomic>
#endif

namespace HG_NAMESPACE {
namespace common {

// THE BUCKETS COVER GUARDED REGIONS, NOT A WORKER'S WHOLE LIFE. A phase is entered by a
// PhaseTimer on the stack, so time outside every guard -- looking for work, stealing, parking,
// the wake that follows -- falls in NO bucket, and the fractions reported below are of the
// guarded total rather than of the run.
//
// EVERY MEMBER HERE IS ENTERED SOMEWHERE. A phase nothing enters reports a structural zero that
// reads as the measurement "no time was spent here", which is a stronger claim than this
// instrument can make; on a workload with fewer states than workers it is also the opposite of
// the truth. Adding a member without a PhaseTimer that enters it makes the report lie.
//
// Idle is entered by the job system's worker loop, around the failed search and the park it
// leads to. It is the bucket that says whether adding workers bought work or bought waiting,
// and without it the other four are fractions of a total that silently excludes the answer.
enum class Phase : uint32_t {
    Match = 0,      // candidate enumeration and the join, including delta matching
    Rewrite,        // applying a match: consuming edges, producing edges, minting the event
    Canon,          // canonicalization and the state identity it decides
    Quotient,       // quotient registration and the reconstruction's bookkeeping
    Idle,           // a worker with no job: the search that failed, the park, and the wake
    Count
};

#ifdef HG_PHASE_TIMING

// One cache line per phase per worker, so two workers never share a line and the counters
// cannot manufacture the contention they are measuring.
struct alignas(64) PhaseSlot {
    uint64_t cycles[static_cast<uint32_t>(Phase::Count)] = {};
};
// alignas(64) already makes sizeof a multiple of 64 -- a type's size is always a multiple of its
// alignment -- so the slot occupies whole lines without a padding member, and the assertion is
// what says so rather than a comment. A hand-computed `64 - (8*Count) % 64` filler cannot state
// it: at Count == 8 that expression is 64, which appends a whole dead line to every one of the
// 256 slots at the exact point the counters start needing a second line anyway.
static_assert(sizeof(PhaseSlot) % 64 == 0,
              "a PhaseSlot must occupy whole cache lines, so two threads never share one");

inline constexpr uint32_t kMaxTimedThreads = 256;
inline PhaseSlot g_phase_slots[kMaxTimedThreads];
inline std::atomic<uint32_t> g_phase_next_slot{0};

// A thread's own slot, claimed once. Threads beyond the cap share the last slot rather than
// writing out of bounds: the sum stays correct, only the per-thread split degrades, and the cap
// is far above the worker counts this engine runs.
inline uint32_t phase_slot() {
    static thread_local uint32_t slot = [] {
        const uint32_t s = g_phase_next_slot.fetch_add(1, std::memory_order_relaxed);
        return s < kMaxTimedThreads ? s : kMaxTimedThreads - 1;
    }();
    return slot;
}

// Accumulates into the caller's own slot with a plain add: no atomic, because only the owning
// thread writes it, and the read below happens with the workers drained.
//
// EXCLUSIVE, NOT INCLUSIVE. These phases nest -- a rewrite canonicalizes its child and registers
// a quotient transition, both inside the rewrite -- so a timer that simply measured its own
// span would charge a canonicalization to canon AND to rewrite, and the fractions would sum
// past 100% while naming no phase correctly. Entering a nested phase therefore banks the
// parent's elapsed time and restarts its clock on the way out, so every cycle lands in exactly
// one bucket: the innermost phase running at the time.
class PhaseTimer {
public:
    explicit PhaseTimer(Phase p)
        : bucket_(&g_phase_slots[phase_slot()].cycles[static_cast<uint32_t>(p)]),
          parent_(t_current), t0_(cycle_counter()) {
        if (parent_) parent_->bank(t0_);
        t_current = this;
    }
    ~PhaseTimer() {
        const uint64_t t1 = cycle_counter();
        *bucket_ += t1 - t0_;
        t_current = parent_;
        if (parent_) parent_->t0_ = t1;
    }

    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;

private:
    void bank(uint64_t now) { *bucket_ += now - t0_; }

    uint64_t* bucket_;
    PhaseTimer* parent_;
    uint64_t  t0_;
    static inline thread_local PhaseTimer* t_current = nullptr;
};

inline uint64_t phase_cycles(Phase p) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < kMaxTimedThreads; ++i)
        total += g_phase_slots[i].cycles[static_cast<uint32_t>(p)];
    return total;
}

inline void phase_reset() {
    for (uint32_t i = 0; i < kMaxTimedThreads; ++i)
        for (uint32_t p = 0; p < static_cast<uint32_t>(Phase::Count); ++p)
            g_phase_slots[i].cycles[p] = 0;
}

inline constexpr bool phase_timing_compiled() { return true; }

#else   // HG_PHASE_TIMING

class PhaseTimer {
public:
    explicit PhaseTimer(Phase) {}
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
};

inline uint64_t phase_cycles(Phase) { return 0; }
inline void phase_reset() {}
inline constexpr bool phase_timing_compiled() { return false; }

#endif  // HG_PHASE_TIMING

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::Match:    return "match";
        case Phase::Rewrite:  return "rewrite";
        case Phase::Canon:    return "canon";
        case Phase::Quotient: return "quotient";
        case Phase::Idle:     return "idle";
        default:              return "?";
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
