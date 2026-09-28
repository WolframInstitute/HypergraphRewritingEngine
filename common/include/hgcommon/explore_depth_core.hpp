#pragma once
#include "hgcommon/namespace.hpp"
//
// QUOTIENT EXPLORATION DEPTH, one body for the host and the device.
//
// Under quotient exploration a canonical state is expanded once, and the step budget applies to
// its SHORTEST depth from an initial state. Arrivals race, so a state can be reached first along
// a longer path. Three rules make the expanded set depend only on shortest depths:
//   - a state's depth is a monotone minimum (explore_try_lower);
//   - a child's depth is one past its parent's CURRENT depth (explore_register_child);
//   - lowering a state's depth lowers its children's depths in turn (explore_relax).
// The expansion claim is separate from the depth and is the Ctx's: a state at or past the budget
// is deferred without a claim, so a shorter path found later can still claim and expand it.
//
// THE RENDEZVOUS. The registrar pushes the child into the parent's list, fences, and reads the
// parent's depth; the relaxer lowers the parent's depth, fences, and walks the parent's list.
// Both fences are seq_cst, which excludes the execution where the walk misses the child and the
// read misses the lowered depth. verification/genmc/depth_relax_child_registration.cpp checks it
// under RC11 and verification/gpumc/depth_relax_child_registration.cpp under scoped RC11.
//
// THE WALK is depth-first over explicit frames, one per level, each holding a list cursor. A
// frame is pushed only for a state whose depth this walk lowered, and depths increase by one per
// frame, so the frame count is bounded by the deepest expanded state. A state lowered twice by
// one walk (through two paths of different length) is walked twice; the result is the same.
//
// Ctx supplies:
//   using Node = ...;                                  a list cursor
//   uint32_t depth_load(uint32_t s)                    acquire
//   bool     depth_cas(uint32_t s, uint32_t& expected, uint32_t desired)   acq_rel; on failure
//                                                      `expected` holds the current value
//   void     children_push(uint32_t parent, uint32_t child)
//   Node     children_head(uint32_t s)                 acquire
//   bool     children_end(Node)
//   uint32_t children_value(Node)
//   Node     children_next(Node)
//   void     fence()                                   seq_cst
//   void     admit(uint32_t s, uint32_t depth)         s was lowered to depth: claim and expand it
//                                                      under the budget, defer it otherwise
//   bool     frame_push(Node at, uint32_t depth)       false when the frame store is full
//   bool     frame_top(Node*& at, uint32_t& depth)     false when empty
//   void     frame_pop()

#include "hgcommon/core.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace common {

// A state no path has reached yet.
constexpr uint32_t kExploreNoDepth = 0xFFFFFFFFu;

// Lower s's depth to `depth`. True only when this call lowered it.
template <class Ctx>
HG_HD bool explore_try_lower(Ctx& c, uint32_t s, uint32_t depth) {
    uint32_t cur = c.depth_load(s);
    while (depth < cur) {
        if (c.depth_cas(s, cur, depth)) return true;
    }
    return false;
}

// Record `child` under `parent` and lower the child to one past the parent's current depth
// (`fallback` when the parent has none). Returns the child's new depth, or kExploreNoDepth when
// this arrival did not lower it.
template <class Ctx>
HG_HD uint32_t explore_register_child(Ctx& c, uint32_t parent, uint32_t child, uint32_t fallback) {
    c.children_push(parent, child);
    c.fence();
    const uint32_t pd = c.depth_load(parent);
    const uint32_t d = (pd == kExploreNoDepth) ? fallback : pd + 1u;
    return explore_try_lower(c, child, d) ? d : kExploreNoDepth;
}

// `s` was just lowered to `depth` (and admitted): lower its descendants. False when the frame
// store filled; the states whose frames were refused keep their lowered depths and are admitted,
// but their children are not walked.
template <class Ctx>
HG_HD bool explore_relax(Ctx& c, uint32_t s, uint32_t depth) {
    using Node = typename Ctx::Node;
    bool complete = true;
    c.fence();
    if (!c.frame_push(c.children_head(s), depth)) return false;
    Node* at = nullptr;
    uint32_t d = 0;
    while (c.frame_top(at, d)) {
        if (c.children_end(*at)) { c.frame_pop(); continue; }
        const uint32_t k = c.children_value(*at);
        *at = c.children_next(*at);
        if (!explore_try_lower(c, k, d + 1u)) continue;
        c.admit(k, d + 1u);
        c.fence();
        if (!c.frame_push(c.children_head(k), d + 1u)) complete = false;
    }
    return complete;
}

}  // namespace common
}  // namespace HG_NAMESPACE
