#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace HG_NAMESPACE {
namespace engine {

// Per-state match-task join. See docs/ARCHITECTURE.md, Sampling.
//
// Matching one state is a tree of MATCH/SCAN/EXPAND tasks, so no single task sees all of that
// state's matches. Anything that has to act on the state's matches AS A SET needs to know when
// that tree has drained: the sampling spine (a state whose every draw failed keeps its
// lowest-ranked own-found transition) and the drain selections (cap_at_drain).
//
// Two monotone counters and the task that equalises them is the drainer. This is a JOIN over one
// state's own tasks, not a barrier: every other state runs through untouched, and nothing global
// is consulted.
//
// The atomic steps of the join and of the spine are the member functions below, so the engine
// and verification/genmc/spine_min_rank.cpp run the same bodies.
struct MatchJoin {
    static constexpr uint64_t kNoSpine = ~0ULL;

    std::atomic<size_t> pushed{0};
    std::atomic<size_t> completed{0};
    // Matches this state has accepted, post-dedup. The drain gate needs it to show the
    // drain fired after the last one rather than merely once.
    std::atomic<size_t> matches{0};
    // Set at this state's drain, before its children list is read (rv::ChildInheritance).
    std::atomic<uint32_t> drained{0};
    // Claimed by whichever side hands this state its parent's matches, so it happens once.
    std::atomic<uint32_t> inherited{0};
    // Set when a stop cuts this state's matching; cleared when the resume is submitted. The
    // state does not drain while it is set.
    std::atomic<uint32_t> resume_pending{0};
    // Stages the state's scan and expand tasks reached, ORed (stats builds): a lost
    // claim reads back as the highest stage its tasks got to. Bits: 1 scan entered,
    // 2 scan past its gates, 4 a produced edge was in the state's set, 8 a signature
    // matched, 16 a candidate validated, 32 complete_match reached, 64 a claim won,
    // 128 a claim answered duplicate, 256 expand entered, 512 expand saw a candidate.
    std::atomic<uint32_t> trace{0};
    // Sampling spine bookkeeping (transition_rate_ < 1 only). A fixed rate is a knife-edge:
    // below 1/branching the sampled evolution goes extinct before reaching depth. The spine
    // keeps the minimum-rank OWN-FOUND transition alive when none of the state's own draws
    // passed.
    //
    // OWN-FOUND ONLY: a state's own matching completes exactly at its drain, so the minimum over
    // own ranks is a function of the state alone, while the stored list also holds forwarded
    // arrivals, which race the drain (a spine over them made WHICH transition survived depend on
    // the schedule, caught at 8 workers by SamplingReproducibility). Forwarded draws neither mark
    // nor force: the surviving set is own-passers, plus forwarded-passers, plus the own-minimum
    // when no own draw passed. A state with NO own-found matches has no spine and relies on its
    // forwarded draws; measured not to bite on the corpus.
    std::atomic<uint32_t> own_spawned{0};
    std::atomic<uint64_t> own_min_key{kNoSpine};

    // Books a task before it can be seen.
    void note_pushed() { pushed.fetch_add(1, std::memory_order_release); }

    // Books a task's completion after every effect of it, and returns the completed count it
    // made. ACQ_REL: the release publishes this task's spine fold and mark to the drainer, and
    // the acquire makes every earlier completion's effects visible to this task if it drains.
    size_t note_completed() {
#if defined(HG_CALIBRATE_MATCH_JOIN_RELAXED_COMPLETION)
        return completed.fetch_add(1, std::memory_order_relaxed) + 1;
#else
        return completed.fetch_add(1, std::memory_order_acq_rel) + 1;
#endif
    }

    // Whether the completion that returned `done` balances the counters, which makes its task
    // the drainer. Read after booking the completion: a task that will still spawn more has not
    // completed, so anything it pushes is already counted.
    bool drains_at(size_t done) const { return done == pushed.load(std::memory_order_acquire); }

    // Folds one own-found transition's rank into the running minimum. Relaxed: the drainer
    // reads the result through note_completed's acq_rel chain.
    void fold_own_rank(uint64_t ranked) {
#if defined(HG_CALIBRATE_SPINE_FOLD_BY_STORE)
        if (ranked < own_min_key.load(std::memory_order_relaxed))
            own_min_key.store(ranked, std::memory_order_relaxed);
#else
        uint64_t seen = own_min_key.load(std::memory_order_relaxed);
        while (ranked < seen &&
               !own_min_key.compare_exchange_weak(seen, ranked, std::memory_order_relaxed)) {}
#endif
    }

    // An own draw passed, or the spine forced the minimum through.
    void mark_own_spawned() { own_spawned.store(1, std::memory_order_release); }

    // Read at the drain: the rank the spine forces through, or kNoSpine when an own draw passed
    // or the state found no own matches.
    uint64_t spine_rank_at_drain() const {
        if (own_spawned.load(std::memory_order_acquire) != 0) return kNoSpine;
        return own_min_key.load(std::memory_order_acquire);
    }
};

}  // namespace engine
}  // namespace HG_NAMESPACE
