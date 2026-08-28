#pragma once
#include "hgcommon/namespace.hpp"
//
// INSERT-IF-ABSENT INTO AN OPEN-ADDRESSED TABLE: where to probe, and which of the threads that
// meet on a key is told it inserted.
//
// THE ELECTION IS THE VALUE EXCHANGE, not the key exchange, and that is the whole content of this
// rule. UNPUBLISHED changes to something exactly once per slot, so exactly one thread's exchange
// succeeds however many offer and whatever they offer -- and the thread it elects is by
// construction the one whose value is stored.
//
// The key exchange elects one thread too, but a DIFFERENT one. Its winner can lose the value
// exchange, and then it reports inserted while carrying a stranger's value while the value's owner
// reports not-inserted: one signature, two canonical events. Comparing the stored value against
// the caller's does not work either, because most callers offer a constant presence marker, so
// every one of them would match and every one would be told it inserted.
//
// NOTHING WAITS. A thread that finds a key claimed but unpublished OFFERS its own value rather
// than waiting for the claimant, so a claimant that is descheduled between the two exchanges
// blocks nobody.
//
// THE PROBE RUN IS EXHAUSTED, NOT BOUNDED. With linear probing a key lives anywhere in its
// contiguous run, so giving up early would miss existing keys and insert duplicates -- a silent
// double-count in place of a silent drop. Exhaustion is a third outcome the caller is told about,
// because a full table that answered "already present" is indistinguishable from a hit, and the
// dedup map is what decides whether a state has been SEEN.
//
// THE STORAGE HALF IS THE CALLER'S: how a key or value word is exchanged, and AT WHAT SCOPE. That
// is what differs between the device (cuda::atomic_ref at thread_scope_device) and a checker
// (annotated __atomic builtins), and it is why the decision is separable at all --
// verification/gpumc/hash_insert_elects_one.cpp runs this body under scoped-RC11, without a
// persistent CUDA kernel. GenMC has no notion of a scope and would check a program the device
// does not run.
//
// Ctx supplies:
//   uint32_t capacity() const
//   uint32_t initial_slot() const                 where this key's probe run starts
//   uint32_t next_slot(uint32_t s) const          linear, wrapping
//   KeyState key_state(uint32_t s) const          ACQUIRE load, classified
//   bool     claim_key(uint32_t s)                CAS empty -> our key; false means re-read
//   InsertOutcome offer_value(uint32_t s)         CAS unpublished -> ours; records what stood

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// What a probed slot holds, relative to the key being inserted. Anything else is another key's,
// and the probe advances past it.
enum class KeyState : uint8_t { Ours, Empty, Other };

enum class InsertOutcome : uint8_t {
    Inserted,     // this thread's value exchange won; it owns the entry
    Present,      // another thread's value stood; the caller takes that one
    Overflowed,   // the probe run was exhausted without finding the key or a free slot
};

template <class Ctx>
HG_HD InsertOutcome hash_insert_claim(Ctx& ctx) {
    uint32_t slot = ctx.initial_slot();
    for (uint32_t i = 0; i < ctx.capacity(); ++i) {
        // The inner loop re-reads THIS slot; the outer one advances. A lost key exchange means
        // something landed here since the read -- possibly this very key -- so the slot is
        // re-classified rather than skipped, which is what keeps a key to one entry.
        for (;;) {
            const KeyState st = ctx.key_state(slot);
            if (st == KeyState::Ours) return ctx.offer_value(slot);
            if (st == KeyState::Empty) {
                if (ctx.claim_key(slot)) return ctx.offer_value(slot);
                continue;
            }
            break;
        }
        slot = ctx.next_slot(slot);
    }
    return InsertOutcome::Overflowed;
}

}  // namespace common
}  // namespace HG_NAMESPACE
