// GPUMC harness: in the quotient replay, an instance published and a match captured at once never
// miss each other.
//
// Runs gpu/include/hg_gpu/quotient_expansion.hpp's qe_publish_child (LockFreeList::push_warp, then
// qe_drive_instance), qe_seed_root_instance (qe_add_instance, then qe_drive_instance) and
// qe_drive_match themselves, with the device's lists, maps, pools, work log, fences and warp
// collectives (verification/gpumc/device).
//
// THE PROTOCOL. A replay warp links its lanes' child instances of one (class, depth) as one chain
// with one exchange (push_warp: each lane writes its link and fences, the warp synchronises, the
// highest lane exchanges the head and fences, the warp synchronises), and each lane then scans the
// class's captured matches (qe_drive_instance: fence, walk by_from). A root instance is recorded
// and pushed by one lane (qe_add_instance: record, fence, push) and scans the same way. A capture
// pushes its match to by_from and every lane of its warp scans the class's instance lists
// (qe_drive_match: lane 0 fences, the warp synchronises, the lanes walk the (depth, shard) lists).
// Each side appends a replay task (instance, match) for what it finds. A side that publishes and
// then scans with no fence between can miss a publication the other side made before its scan.
//
// THE SHAPES. Block 1 is a warp of one lane capturing match record 0 of class H; max_steps is 1,
// so the match side walks the (depth 0, shard 0..15) lists and its one lane walks shard 0, which
// the instances are in.
//   default         block 0 is a warp of two lanes, each with a child instance of H at depth 0
//                   (records 0 and 1), linked as one chain (a warp's children go in its warp's
//                   shard; warp 0 is shard 0).
//   -DSHAPE_ROOT    block 0 is one lane seeding H's root instance (record 0) from state 0; its
//                   lane-0 shard is 0.
//
// THE PROPERTY. For each instance, the task log holds a task (instance, match 0). A pair may be
// appended by both sides; the replay's claim words apply it once (qr_apply). No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_NO_THREADFENCE removes the fences; the checker must report an
// instance with no task.
// GPUMC-CELL: warp_chain PASS
// GPUMC-CELL: root PASS -DSHAPE_ROOT
// GPUMC-CELL: warp_chain_no_threadfence FAIL -DHG_GPUMC_CALIBRATE_NO_THREADFENCE
// GPUMC-CELL: root_no_threadfence FAIL -DSHAPE_ROOT -DHG_GPUMC_CALIBRATE_NO_THREADFENCE

#include "hg_gpumc.hpp"
#include "hg_gpu/quotient_expansion.hpp"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using InstNode = g::LockFreeList<g::QeInstRef>::Node;
using MatchNode = g::LockFreeList<g::QeMatchRef>::Node;

#if defined(SHAPE_ROOT)
constexpr uint32_t kInstances = 1;
#else
constexpr uint32_t kInstances = 2;
#endif
constexpr uint64_t kClass = 0x5151;
constexpr uint32_t kKeys = 4;
constexpr uint32_t kNodes = 4;
constexpr uint32_t kTasks = 8;
constexpr uint32_t kMapSlots = 4;
constexpr uint32_t kWords = 16;

g::DeviceState* g_ds;
g::QeView* g_qe;
InstNode g_inst_nodes[kNodes];
uint32_t g_inst_nodes_n;
uint32_t g_inst_heads[kKeys];
MatchNode g_match_nodes[kNodes];
uint32_t g_match_nodes_n;
uint32_t g_match_heads[kKeys];
g::QeTask g_tasks[kTasks];
uint32_t g_tasks_n, g_tasks_cursor, g_tasks_done;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
// SHAPE_ROOT: the instance record pool, the class frame and rep maps, the slot-word arena.
g::DeviceQcInstance g_instances[2];
uint32_t g_instances_n, g_inst_next_id;
g::MapSlot<uint64_t, uint64_t> g_frame_slots[kMapSlots];
g::MapSlot<uint64_t, uint32_t> g_rep_slots[kMapSlots];
uint32_t g_words[kWords];
uint32_t g_words_cursor;
uint64_t g_state_hash[1] = {kClass};
g::StateEdgeSlice g_slices[1] = {{0, 1}};

void* replay_warp(void* arg) {
    const unsigned lane = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(0, lane);
    const g::QeView& qe = *g_qe;
#if defined(SHAPE_ROOT)
    g::qe_seed_root_instance(*g_ds, qe, 0, 0);
#else
    // A child as DeviceQrCtx::claim records it: its list node claimed, its warp's shard.
    g::QeChild child;
    child.rec = lane;
    child.hash = kClass;
    child.depth = 0;
    child.bucket = g::qe_inst_bucket(qe, g::qe_inst_key(kClass, 0), 0);
    child.node = qe.by_key.claim_node(g::QeInstRef{kClass, lane, 0});
    g::qe_publish_child(*g_ds, qe, child);
#endif
    return nullptr;
}

void* capture_warp(void*) {
    hg_gpumc::enter(1, 0);
    const g::QeView& qe = *g_qe;
    // qe_capture_expansion's publication of match record 0, then its warp's scan.
    const uint64_t key = g::qe_expansion_key(*g_ds, kClass, 0);
    qe.by_from.push(g::qe_bucket(key, qe.by_from.num_keys), g::QeMatchRef{key, 0});
    g::qe_drive_match(*g_ds, qe, 0, kClass, 0);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[2] = {kInstances, 1};
    hg_gpumc::launch(2, lanes);
    for (uint32_t k = 0; k < kKeys; ++k) g_inst_heads[k] = g_match_heads[k] = g::INVALID_ID;
    for (uint32_t s = 0; s < kMapSlots; ++s) {
        g_frame_slots[s] = {0, ~uint64_t{0}};
        g_rep_slots[s] = {0, ~uint32_t{0}};
    }
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->errors.counters = g_errors;
    g_ds->state_canonical_hash = g_state_hash;
    g_ds->state_edge_slices = g_slices;
    g_qe = new (malloc(sizeof(g::QeView))) g::QeView{};
    g_qe->enabled = 1;
    g_qe->replay = 1;
    g_qe->max_steps = 1;
    g_qe->by_key = {{g_inst_nodes, &g_inst_nodes_n, kNodes}, g_inst_heads, kKeys};
    g_qe->by_from = {{g_match_nodes, &g_match_nodes_n, kNodes}, g_match_heads, kKeys};
    g_qe->tasks.items = {g_tasks, &g_tasks_n, kTasks};
    g_qe->tasks.cursor = &g_tasks_cursor;
    g_qe->tasks.done = &g_tasks_done;
    g_qe->instances = {g_instances, &g_instances_n, 2};
    g_qe->inst_next_id = &g_inst_next_id;
    g_qe->frame = {g_frame_slots, kMapSlots, kMapSlots - 1, nullptr};
    g_qe->rep = {g_rep_slots, kMapSlots, kMapSlots - 1, nullptr};
    g_qe->arr_words = g_words;
    g_qe->arr_cursor = &g_words_cursor;
    g_qe->arr_capacity = kWords;

    pthread_t t[3];
    for (long l = 0; l < static_cast<long>(kInstances); ++l)
        pthread_create(&t[l], nullptr, replay_warp, reinterpret_cast<void*>(l));
    pthread_create(&t[kInstances], nullptr, capture_warp, nullptr);
    for (uint32_t i = 0; i <= kInstances; ++i) pthread_join(t[i], nullptr);

    bool met[2] = {false, false};
    for (uint32_t i = 0; i < g_tasks_n && i < kTasks; ++i) {
        assert(g_tasks[i].published == 1u && g_tasks[i].match == 0u && g_tasks[i].depth == 0u);
        if (g_tasks[i].rec < 2) met[g_tasks[i].rec] = true;
    }
    for (uint32_t r = 0; r < kInstances; ++r)
        assert(met[r] && "an instance and the match missed each other");
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
