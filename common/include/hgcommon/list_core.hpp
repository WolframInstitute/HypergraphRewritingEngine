#pragma once
#include "hgcommon/namespace.hpp"
//
// PREPEND TO AN INTRUSIVE LOCK-FREE LIST, AND WALK IT: the one body both the device list
// (gpu/include/hg_gpu/lock_free_list.hpp) and a model checker drive.
//
// THE RULE. A pusher links its node in front of whatever head it read and installs it with a
// compare-exchange; a failed exchange has refreshed the head, so the loop relinks and retries.
// The exchange is ACQ_REL: release publishes this node's fields to a walker that loads the head
// with acquire, and acquire is for the pusher itself, which goes on to walk the chain below its
// own node reading values other threads published. A walk from the head visits exactly the nodes
// published before the head was loaded; a walk from BELOW a node visits every node linked
// strictly before it, which is what lets two pushers meet exactly once -- of any two, exactly
// one is older.
//
// THE STORAGE HALF IS THE CALLER'S: how the head word and a node's next field are read and
// exchanged, and AT WHAT SCOPE. That is what differs between the device (cuda::atomic_ref at
// thread_scope_device) and a checker (annotated __atomic builtins), and it is why the rule is
// separable at all -- verification/gpumc/replay_rendezvous_meets.cpp runs this body under
// scoped-RC11. Ctx supplies:
//
//   uint32_t invalid() const                       the tail marker
//   uint32_t head_load_relaxed() const
//   uint32_t head_load_acquire() const
//   bool     head_cas(uint32_t& expected, uint32_t desired)   ACQ_REL success, relaxed failure
//   void     set_next(uint32_t node, uint32_t next)             plain store into the node
//   uint32_t next_of(uint32_t node) const                       plain load

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

template <class Ctx>
HG_HD void list_push(Ctx& ctx, uint32_t node) {
    uint32_t prev = ctx.head_load_relaxed();
    for (;;) {
        ctx.set_next(node, prev);
        if (ctx.head_cas(prev, node)) return;
        // prev was refreshed by the failed exchange; relink and retry.
    }
}

// Every node from the head, most recent first.
template <class Ctx, class Fn>
HG_HD void list_for_each(const Ctx& ctx, Fn&& fn) {
    for (uint32_t idx = ctx.head_load_acquire(); idx != ctx.invalid(); idx = ctx.next_of(idx))
        fn(idx);
}

// Every node linked strictly before `mine`, most recent first.
template <class Ctx, class Fn>
HG_HD void list_for_each_before(const Ctx& ctx, uint32_t mine, Fn&& fn) {
    if (mine == ctx.invalid()) return;
    for (uint32_t idx = ctx.next_of(mine); idx != ctx.invalid(); idx = ctx.next_of(idx))
        fn(idx);
}

}  // namespace common
}  // namespace HG_NAMESPACE
