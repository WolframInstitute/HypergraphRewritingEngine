#pragma once
#include "hgcommon/namespace.hpp"
//
// THE CLAIM RULE OF A BOUNDED MPMC RING: which position a thread takes, and when a slot is its
// turn. Every slot carries a SEQUENCE NUMBER and that number alone says whose turn it is:
//
//   seq[s] == pos          the slot is free for the producer reserving position pos
//   seq[s] == pos + 1      the slot holds the item for the consumer reserving position pos
//   seq[s] <  pos          a full lap ahead: full for a producer, empty for a consumer
//
// seq[i] starts at i, a producer publishes with seq = pos + 1 and a consumer releases with
// seq = pos + capacity, so a slot advances by exactly one capacity per lap and the three tests
// stay exact across wraps of the 64-bit cursors.
//
// PRODUCING AND CONSUMING ARE THE SAME RULE with two constants, which is why they are one body
// here. A producer waits for seq == pos and leaves seq == pos + 1; a consumer waits for
// seq == pos + 1 and leaves seq == pos + capacity. Written twice they agree until one is edited.
//
// THE RESERVATION IS A CAS, not an unconditional bump. That is what makes the queue safe when
// the same workers both produce and consume, which is how a device-resident scheduler uses it: a
// bump has nothing to undo with, and a rollback store can land on a slot another thread has
// already taken, which either loses an item or hands one out twice. An item lost from a queue
// whose producers are its consumers is not a dropped unit of work -- it is a run that never
// terminates, because the termination detector waits for a completion that can no longer happen.
//
// THE STORAGE HALF IS THE CALLER'S, and that is the whole reason this is separate. How a cursor
// or a sequence word is read and written, and AT WHAT SCOPE, is what differs between the device
// (cuda::atomic_ref at thread_scope_device) and a checker (annotated __atomic builtins). The
// decision does not differ, so a model checker can be handed the decision without being handed a
// persistent CUDA kernel. verification/gpumc/ring_exactly_once.cpp runs this body under
// scoped-RC11; GenMC has no notion of a scope and would check a program the device does not run.
//
// Ctx supplies:
//   uint32_t mask() const                                 capacity - 1, capacity a power of two
//   uint64_t cursor_load() const                          relaxed; the cursor only allocates
//   bool     cursor_cas(uint64_t& expected, uint64_t d)    relaxed, may fail spuriously
//   uint64_t seq_load(uint32_t slot) const                 ACQUIRE
//   void     seq_store(uint32_t slot, uint64_t v)          RELEASE
//   void     transfer(uint32_t slot)                       write the item in, or read it out
//
// The release-store synchronizes-with the acquire-load, which is what orders the non-atomic slot
// access inside transfer() against the other side's. The cursor CAS is relaxed deliberately: the
// sequence handshake carries the ordering and the cursor only hands out positions.

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// `want` is what seq must equal, as an offset from the claimed position: 0 for a producer
// (the slot is free) and 1 for a consumer (the slot holds an item). `leave` is what seq is set
// to on success, again as an offset: 1 for a producer (published) and capacity for a consumer
// (free again, one lap on).
//
// Returns false only when the queue was OBSERVED full (producer) or empty (consumer), and never
// mutates a slot it did not win.
template <class Ctx>
HG_HD bool ring_claim(Ctx& ctx, uint64_t want, uint64_t leave) {
    uint64_t pos = ctx.cursor_load();
    for (;;) {
        const uint32_t s = static_cast<uint32_t>(pos) & ctx.mask();
        // Signed difference, so a slot a lap behind reads as negative rather than as a huge
        // unsigned value. This is the test that stays exact across a cursor wrap.
        const int64_t dif = static_cast<int64_t>(ctx.seq_load(s))
                          - static_cast<int64_t>(pos + want);
        if (dif == 0) {
            // The slot is this thread's only if it also wins the cursor. A failed exchange has
            // refreshed pos, so the loop retries at the position it lost to rather than
            // re-reading and racing again.
            if (ctx.cursor_cas(pos, pos + 1)) {
                ctx.transfer(s);
                ctx.seq_store(s, pos + leave);
                return true;
            }
        } else if (dif < 0) {
            return false;
        } else {
            pos = ctx.cursor_load();
        }
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
