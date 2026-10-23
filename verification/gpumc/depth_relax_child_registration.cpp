// GPUMC harness: on the device, a child never strands at a stale depth when its parent is relaxed
// concurrently, and every state lowered under the budget is claimed and logged for expansion.
//
// Runs hgcommon/explore_depth_core.hpp's explore_register_child, explore_try_lower and
// explore_relax over gpu/src/persistent.cu's DeviceExploreCtx itself -- the child lists
// (LockFreeList), the depth words, the expansion claim (ExploreView::claim) and the expand log
// (WorkLogView::append) -- with the device's atomics and fences (verification/gpumc/device).
//
//   registrar (explore_register_child)          relaxer (explore_try_lower, explore_relax)
//     push the child into the parent's list       lower the parent's depth
//     __threadfence                               __threadfence
//     read the parent's depth                     walk the parent's child list
//
// THE SHAPE. Two blocks of one thread: parent state 0 at depth 3 and its child 1, registered by
// one block while the other lowers the parent to depth 1. The budget is 10 steps.
//
// THE PROPERTY. The child ends at depth 2 (one past the parent's lowered depth), whichever side
// lowers it, and it is claimed once and logged once for expansion, at the depth it had when it
// was claimed (2, or 4 when the registrar claims it before the relaxation). No access races.
// verification/genmc/depth_relax_child_registration.cpp checks the core over the host's list
// under RC11.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_NO_THREADFENCE makes the fences no-ops; the checker must report
// the child stranded at the parent's old depth plus one.
// GPUMC-CELL: clean PASS -- --disable-function-inliner
// GPUMC-CELL: no_threadfence FAIL -DHG_GPUMC_CALIBRATE_NO_THREADFENCE -- --disable-function-inliner

#include "hg_gpumc.hpp"
#include "persistent.cu"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using Node = g::LockFreeList<g::StateId>::Node;

constexpr uint32_t kParent = 0, kChild = 1;
constexpr uint32_t kOldDepth = 3, kNewDepth = 1, kBudget = 10;

g::DeviceState* g_ds;
g::ExploreView* g_ev;
g::SessionView* g_sess;
uint32_t g_depth[2] = {kOldDepth, hgcommon::kExploreNoDepth};
uint32_t g_claimed[2] = {1, 0};   // the parent is claimed already
Node g_nodes[4];
uint32_t g_nodes_n;
uint32_t g_heads[2] = {g::INVALID_ID, g::INVALID_ID};
g::ExpandEntry g_expand[4];
uint32_t g_expand_n, g_expand_cursor, g_expand_done;
uint32_t g_frame_node[2 * 4], g_frame_depth[2 * 4];
uint32_t g_errors[g::DeviceErrors::kMaxKinds];

g::DeviceExploreCtx ctx(unsigned block) {
    return g::DeviceExploreCtx{*g_ds, *g_ev, *g_sess, kBudget, g_frame_node + 4 * block,
                               g_frame_depth + 4 * block, 4, false};
}

void* registrar(void*) {
    hg_gpumc::enter(0, 0);
    g::DeviceExploreCtx c = ctx(0);
    const uint32_t d = hgcommon::explore_register_child(c, kParent, kChild, kOldDepth + 1u);
    if (d != hgcommon::kExploreNoDepth) {
        c.admit(kChild, d);
        hgcommon::explore_relax(c, kChild, d);
    }
    return nullptr;
}

void* relaxer(void*) {
    hg_gpumc::enter(1, 0);
    g::DeviceExploreCtx c = ctx(1);
    if (hgcommon::explore_try_lower(c, kParent, kNewDepth))
        hgcommon::explore_relax(c, kParent, kNewDepth);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[2] = {1, 1};
    hg_gpumc::launch(2, lanes);
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_sess = new (malloc(sizeof(g::SessionView))) g::SessionView{};
    g_ev = new (malloc(sizeof(g::ExploreView))) g::ExploreView{};
    g_ev->depth = g_depth;
    g_ev->claimed = g_claimed;
    g_ev->children = {{g_nodes, &g_nodes_n, 4}, g_heads, 2};
    g_ev->expand.items = {g_expand, &g_expand_n, 4};
    g_ev->expand.cursor = &g_expand_cursor;
    g_ev->expand.done = &g_expand_done;
    g_ev->max_states = 2;

    pthread_t a, b;
    pthread_create(&a, nullptr, registrar, nullptr);
    pthread_create(&b, nullptr, relaxer, nullptr);
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);

    assert(g_depth[kChild] == kNewDepth + 1u &&
           "the child stranded at a depth its parent no longer has");
    assert(g_claimed[kChild] == 1u);

    for (uint32_t i = 0; i < g_expand_n; ++i) {
        assert(g_expand[i].state == kChild && g_expand[i].depth >= kNewDepth + 1u);

    }
    assert(g_expand_n == 1 && "the child was not logged for expansion exactly once");
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
