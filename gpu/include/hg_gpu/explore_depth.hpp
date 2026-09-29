#pragma once
#include "hgcommon/namespace.hpp"
//
// QUOTIENT EXPLORATION DEPTH ON THE DEVICE: the storage hgcommon/explore_depth_core.hpp runs
// over, and the expand log that turns an admitted state into match items.
//
// A canonical state is expanded once, at the shortest depth any path reaches it by. Per state:
// its depth (a monotone minimum), a claim separate from the depth, and the list of canonical
// states its expansion produced. An arrival registers the child under its parent and lowers
// the child's depth; a lowered state is admitted (claimed and appended to the expand log when
// under the budget, recorded on a session's frontier otherwise) and its descendants are lowered
// in turn. The walk runs on a block's thread 0 over that block's frame slice.
//
// THE EXPAND LOG. Expanding a state pushes one match item per rule, and a full ring falls back
// to matching the item on the pushing block with every lane; the walk runs on one lane. So an
// admitted state is appended to a work log (hg_gpu/work_log.hpp) and a block takes it at the top
// of the persistent loop. A state is claimed once, so the log holds at most one entry per state.
// Roots and a session's resumed frontier enter the same way.

#include "hgcommon/explore_depth_core.hpp"
#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/lock_free_list.hpp"
#include "hg_gpu/work_log.hpp"
#include "hg_gpu/types.hpp"

#include <cuda/atomic>
#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

// A state to expand at `depth`: the persistent loop pushes its (state, rule) items at that step.
struct ExpandEntry {
    StateId  state;
    uint32_t depth;
    uint32_t published;
};

struct ExploreView {
    uint32_t* depth   = nullptr;   // per state; hgcommon::kExploreNoDepth until reached
    uint32_t* claimed = nullptr;   // per state; 1 once claimed for expansion
    typename LockFreeList<StateId>::DeviceView children{};   // keyed by canonical state
    WorkLogView<ExpandEntry> expand{};
    // The walk's frames: `frame_levels` per block, node and depth.
    uint32_t* frame_node  = nullptr;
    uint32_t* frame_depth = nullptr;
    uint32_t  frame_levels = 0;
    uint32_t  max_states   = 0;

    __device__ bool claim(StateId s) { return atomicCAS(&claimed[s], 0u, 1u) == 0u; }
};

// Stream-ordered reset through the view: the expand log (published flags, counter, cursor, done)
// always; depths, claims and child lists as well when `full`.
void explore_reset_async(const ExploreView& v, bool full);

// Host owner. Sized from the state and event budgets; a session keeps one across calls and a
// one-shot run clears its own per launch.
class ExploreState {
public:
    ExploreState(uint32_t max_states, uint32_t max_events);
    ~ExploreState();
    ExploreState(const ExploreState&)            = delete;
    ExploreState& operator=(const ExploreState&) = delete;

    uint32_t max_states() const { return max_states_; }
    uint32_t max_events() const { return max_events_; }

    // Every state unreached and unclaimed, the lists and the log empty.
    void clear();
    ExploreView view() const;

private:
    uint32_t                max_states_;
    uint32_t                max_events_;
    uint32_t*               words_ = nullptr;   // depth, claimed, then the log's cursor and done
    LockFreeList<StateId>   children_;
    Pool<ExpandEntry>       expand_;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE
