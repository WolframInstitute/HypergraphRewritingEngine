#pragma once
#include "hgcommon/namespace.hpp"
//
// CLAIMING A MATCH EXACTLY ONCE, over a set keyed by a 64-bit hash that can collide.
//
// TWO PROPERTIES, each the site of a shipped defect class:
//
//   EXACTLY-ONCE. Two threads claiming the SAME match agree on one winner. Both winning
//   double-applies a rewrite; both losing drops a match, and forwarding is inductive -- a dropped
//   match deletes its whole subtree while the run stays self-consistent.
//
//   NO DROP ON COLLISION. Two DIFFERENT matches with the SAME hash must BOTH win. Deciding on
//   hash equality alone silently loses real matches, which is what the probe walk exists to
//   prevent: a key-only set answers "present" when a DIFFERENT match occupies the slot.
//
// THE HASH ONLY SELECTS WHERE TO LOOK. Identity is the CONTENT comparison, at every point the
// walk can conclude -- on the lookup and again on the offer, because the slot can change between
// them. A probe that ends without either a content match or a free key moves to the next key
// rather than concluding anything.
//
// THE STABLE COPY IS ALLOCATED ON THE STRENGTH OF A LOOKUP THAT JUST MISSED, and that is
// deliberate: the copy has to exist before the exchange that publishes it. Another thread can
// claim the key in that window and the copy is then permanent cost, so it is made AT MOST ONCE
// per claim and only once a probe has actually missed. True duplicates are answered from the
// lookup and never reach it, which matters because they are routine -- delta matching finds a
// match on k produced edges k times, once anchored on each.
//
// RUNNING OUT OF PROBES PROCESSES THE MATCH. A redundant rewrite is recoverable; a lost one is
// not, and the two are not symmetric.
//
// THE STORAGE HALF IS THE CALLER'S -- which set, which probe-key derivation, which content
// comparison, and which counters move. That is what lets a model checker be handed this rule
// without being handed ParallelEvolutionEngine, whose header the interpreter cannot take (1130
// lines, <thread>). verification/genmc/claim_match_rendezvous.cpp drives THIS body over a real
// ConcurrentMap.
//
// Ctx supplies:
//   uint32_t   max_probes() const
//   uint64_t   probe_key(uint32_t n) const     nth key for this hash, skipping reserved sentinels
//   ProbeState probe(uint64_t key) const       lookup, classified by CONTENT
//   void       make_stable()                   materialise the copy that will be published
//   ClaimState offer(uint64_t key)             insert-if-absent, classified by CONTENT
//   void       note_collision()
//   void       note_exhausted()

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// What a probed key holds, decided by comparing CONTENTS and never by the hash alone.
enum class ProbeState : uint8_t {
    Miss,        // the key is free: this claim may take it
    Duplicate,   // an EQUAL match is already claimed here
    Collision,   // a DIFFERENT match occupies the key
};

enum class ClaimState : uint8_t {
    Won,         // this thread's offer was the one that landed
    Duplicate,   // another thread's offer landed and it is the SAME match
    Collision,   // another thread's offer landed and it is a DIFFERENT match
};

// Returns true when the caller must process the match, false when an equal one was already
// claimed.
template <class Ctx>
HG_HD bool dedup_claim(Ctx& ctx) {
    bool have_stable = false;
    for (uint32_t n = 0; n < ctx.max_probes(); ++n) {
        const uint64_t key = ctx.probe_key(n);

        const ProbeState p = ctx.probe(key);
#if defined(HG_CALIBRATE_DEDUP_HASH_ONLY)
        // THE DEFECT: any hit is treated as a duplicate, so identity is the HASH and not the
        // content. Two different matches that collide then leave only one claimed, and the other
        // is dropped along with every match its subtree would have produced. Calibration only;
        // set by verification/genmc/run.sh's HG_HARNESS_DEFINES, so it is a command anyone can
        // repeat rather than an assertion in a comment.
        if (p != ProbeState::Miss) return false;
#else
        if (p == ProbeState::Duplicate) return false;
        if (p == ProbeState::Collision) { ctx.note_collision(); continue; }
#endif

        if (!have_stable) { ctx.make_stable(); have_stable = true; }

        // The slot may have been taken since the lookup, so the offer is classified the same way
        // and by the same comparison. Concluding from the lookup alone is what lets two threads
        // both win, or one lose to a match that is not its own.
        const ClaimState c = ctx.offer(key);
        if (c == ClaimState::Won) return true;
        if (c == ClaimState::Duplicate) return false;
        ctx.note_collision();
    }
    ctx.note_exhausted();
    return true;
}

}  // namespace common
}  // namespace HG_NAMESPACE
