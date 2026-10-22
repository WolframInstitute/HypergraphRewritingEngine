// GPUMC harness: the full-capture causal rendezvous records each (producer, consumer, edge) once,
// and the online transitive reduction keeps its predecessor lists consistent.
//
// Runs gpu/src/rewrite.cu's register_as_producer and register_as_consumer (add_causal_edge:
// the pair and triple maps, the causal edge pool, the reduced predecessor list and the
// reachability search) themselves, with the device's maps, lists, pools and atomics
// (verification/gpumc/device).
//
// THE PROTOCOL. A rewrite registers itself as the producer of each edge it produced (a
// compare-exchange on edge_producer[e], then a walk of e's consumers) and as a consumer of each
// edge it consumed (a push on e's consumer list, then an acquire load of edge_producer[e]); each
// side adds the causal edge for the other it finds, and the triple map admits each (producer,
// consumer, edge) once. A consumer of e matches a state that holds e, which exists only after e's
// producer registered: the harness starts each consumer after the producer publishes its
// registration (a release store the consumers acquire), as the persistent kernel's queue does.
//
// THE SHAPE. Producer event 0 makes edge 0; consumer events 1 and 2 consume it, at once, in three
// blocks of one thread. -DSHAPE_TR turns the transitive reduction on, so each kept pair is pushed
// on the consumer's predecessor list and checked for reachability first.
//
// THE PROPERTY. The causal edge pool holds exactly (0, 1, edge 0) and (0, 2, edge 0); under
// SHAPE_TR each consumer's predecessor list holds the producer once. No access races.
//
// CALIBRATION. -DCALIBRATE_CONSUMER_FIRST starts the consumers without waiting for the producer's
// registration, which the device's queue never does: the two sides then miss each other (store
// buffering), and the checker must report a missing edge.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: tr PASS -DSHAPE_TR
// GPUMC-CELL: consumer_first FAIL -DCALIBRATE_CONSUMER_FIRST

#include "hg_gpumc.hpp"
#include "rewrite.cu"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using Map = g::ConcurrentMap<uint64_t, uint32_t>;
using ListNode = g::LockFreeList<g::EventId>::Node;

constexpr uint32_t kSlots = 8;

g::DeviceState* g_ds;
g::EventId g_producer[1] = {g::INVALID_ID};
ListNode g_consumer_nodes[4];
uint32_t g_consumer_nodes_n;
uint32_t g_consumer_heads[1];
ListNode g_pred_nodes[4];
uint32_t g_pred_nodes_n;
uint32_t g_pred_heads[3];
g::MapSlot<uint64_t, uint32_t> g_triples[kSlots], g_pairs[kSlots];
g::DeviceCausalEdge g_edges[4];
uint32_t g_edges_n;
uint32_t g_errors[g::DeviceErrors::kMaxKinds];
uint32_t g_registered;

void* producer(void*) {
    hg_gpumc::enter(0, 0);
    const g::CausalViews v(*g_ds);
    g::register_as_producer(*g_ds, v, 0, 0);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(g_registered).store(
        1u, cuda::memory_order_release);
    return nullptr;
}

void* consumer(void* arg) {
    const g::EventId ev = static_cast<g::EventId>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(ev, 0);
#if !defined(CALIBRATE_CONSUMER_FIRST)
    const cuda::atomic_ref<uint32_t, cuda::thread_scope_device> reg(g_registered);
    while (reg.load(cuda::memory_order_acquire) == 0u) {
    }
#endif
    const g::CausalViews v(*g_ds);
    g::register_as_consumer(*g_ds, v, ev, 0);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    g_consumer_heads[0] = g::INVALID_ID;
    for (uint32_t i = 0; i < 3; ++i) g_pred_heads[i] = g::INVALID_ID;
    for (uint32_t s = 0; s < kSlots; ++s) {
        g_triples[s] = {0, Map::DeviceView::UNPUBLISHED};
        g_pairs[s] = {0, Map::DeviceView::UNPUBLISHED};
    }
    // On the heap: GPUMC's static allocator places a struct of this size wrongly.
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_ds->edge_producer = g_producer;
    g_ds->edge_consumers = {{g_consumer_nodes, &g_consumer_nodes_n, 4}, g_consumer_heads, 1};
    g_ds->preds_list = {{g_pred_nodes, &g_pred_nodes_n, 4}, g_pred_heads, 3};
    g_ds->causal_triple_dedup = {g_triples, kSlots, kSlots - 1, nullptr};
    g_ds->causal_pair_dedup = {g_pairs, kSlots, kSlots - 1, nullptr};
    g_ds->causal_edge_pool = {g_edges, &g_edges_n, 4};
    g_ds->errors.counters = g_errors;
#if defined(SHAPE_TR)
    g_ds->tr_enabled = true;
#endif

    pthread_t t[3];
    pthread_create(&t[0], nullptr, producer, nullptr);
    pthread_create(&t[1], nullptr, consumer, reinterpret_cast<void*>(1L));
    pthread_create(&t[2], nullptr, consumer, reinterpret_cast<void*>(2L));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);

    bool seen[3] = {false, false, false};
    assert(g_edges_n == 2 && "the causal edges were not recorded exactly once each");
    for (uint32_t i = 0; i < 2; ++i) {
        assert(g_edges[i].from == 0 && g_edges[i].shared_edge == 0 && g_edges[i].to >= 1 &&
               g_edges[i].to <= 2);
        assert(!seen[g_edges[i].to]);
        seen[g_edges[i].to] = true;
    }
#if defined(SHAPE_TR)
    for (uint32_t c = 1; c <= 2; ++c) {
        uint32_t n = 0;
        g_ds->preds_list.for_each(c, [&](g::EventId p) {
            assert(p == 0);
            ++n;
        });
        assert(n == 1 && "a consumer's predecessor list does not hold the producer once");
    }
#endif
    for (uint32_t k = 0; k < g::DeviceErrors::kMaxKinds; ++k) assert(g_errors[k] == 0u);
    return 0;
}
