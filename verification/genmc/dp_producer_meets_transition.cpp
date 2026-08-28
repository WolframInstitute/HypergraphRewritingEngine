// GenMC harness: the quotient-causal DP's producer/transition rendezvous never drops a pair.
//
// Drives hgcommon/quotient_causal_core.hpp ITSELF -- qc_add_producer and qc_process_transition,
// the bodies both engines run -- with the shape the host (Hypergraph::QcCtx) has around them:
//
//   producer side (qc_add_producer)            transition side (Hypergraph::register_quotient_transition)
//     push the producer at (S, d, orbit)         push the transition out of S
//     fence                                      fence
//     scan the transitions out of S              (S, d) is reached: scan the producers at
//       -> emit(producer, t.canon_event)           (S, d, orbit) -> emit(producer, t.canon_event)
//
// If BOTH scans miss, the causal edge (producer -> t.canon_event) is never emitted, and the
// reconstructed causal graph is one edge short with every count that does not read edges
// untouched. The core's own comment says the fence on BOTH sides is what forbids this; here
// the checker decides it.
//
// WHAT IS BOUNDED. One state S reached at depth 0, one producer, one transition consuming the
// producer's orbit, max_steps 1. The Ctx is storage only: two list_core lists (the same body
// the engines' lock-free lists drive), one reached flag per depth, one seen flag, one emitted
// flag. Every decision -- what to publish before what, when to scan, what to emit -- is the
// core's.
//
// CALIBRATED. -DCALIBRATE_NO_FENCE makes Ctx::fence() a no-op, and the checker must report the
// edge missing.
#include "hgcommon/list_core.hpp"
#include "hgcommon/quotient_causal_core.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

constexpr uint32_t kInvalid = 0xFFFFFFFFu;
constexpr uint64_t kS = 0x5151ull;        // the state's canonical hash
constexpr uint64_t kTo = 0x7070ull;       // the transition's target hash
constexpr uint32_t kOrbit = 3;
constexpr uint32_t kProducer = 41;
constexpr uint32_t kEvent = 42;

struct Transition {
    uint64_t to_hash = kTo;
    uint32_t canon_event = kEvent;
    uint32_t num_consumed = 1, num_produced = 0, num_survivors = 0;
    uint32_t consumed(uint32_t) const { return kOrbit; }
    uint32_t produced(uint32_t) const { return 0; }
    uint32_t surv_from(uint32_t) const { return 0; }
    uint32_t surv_to(uint32_t) const { return 0; }
};

// Node 0 is the producer node, node 1 the transition node; two lists over one pool.
struct Node { uint32_t producer; uint32_t next; };
Node g_nodes[2];
std::atomic<uint32_t> g_head_prod{kInvalid};
std::atomic<uint32_t> g_head_trans{kInvalid};
Transition g_t;
std::atomic<uint32_t> g_reached[2]{{0}, {0}};
std::atomic<uint32_t> g_seen{0};
std::atomic<uint32_t> g_emitted{0};

struct ListOps {
    std::atomic<uint32_t>* head;
    uint32_t invalid() const { return kInvalid; }
    uint32_t head_load_relaxed() const { return head->load(std::memory_order_relaxed); }
    uint32_t head_load_acquire() const { return head->load(std::memory_order_acquire); }
    bool head_cas(uint32_t& expected, uint32_t desired) {
        return head->compare_exchange_weak(expected, desired, std::memory_order_acq_rel,
                                           std::memory_order_relaxed);
    }
    void set_next(uint32_t node, uint32_t next) { g_nodes[node].next = next; }
    uint32_t next_of(uint32_t node) const { return g_nodes[node].next; }
};

struct Ctx {
    using Transition = ::Transition;
    uint32_t max_steps() const { return 1; }
    bool enter(uint32_t) const { return true; }
    void defer_reach(uint64_t h, uint32_t d) { hgcommon::qc_reach(*this, h, d); }
    void defer_producer(uint64_t h, uint32_t d, uint32_t o, uint32_t p) {
        hgcommon::qc_add_producer(*this, h, d, o, p);
    }
    bool mark_reached(uint64_t, uint64_t, uint32_t depth) {
        return g_reached[depth].exchange(1, std::memory_order_acq_rel) == 0;
    }
    bool mark_producer_seen(uint64_t) { return g_seen.exchange(1, std::memory_order_acq_rel) == 0; }
    void push_producer(uint64_t, uint32_t producer) {
        g_nodes[0].producer = producer;
        ListOps ops{&g_head_prod};
        hgcommon::list_push(ops, 0);
    }
    template <class F> void for_each_producer(uint64_t, F&& f) {
        ListOps ops{&g_head_prod};
        hgcommon::list_for_each(ops, [&](uint32_t idx) { f(g_nodes[idx].producer); });
    }
    template <class F> void for_each_transition_from(uint64_t hash, F&& f) {
        if (hash != kS) return;
        ListOps ops{&g_head_trans};
        hgcommon::list_for_each(ops, [&](uint32_t) { f(g_t); });
    }
    void emit(uint32_t producer, uint32_t consumer) {
        if (producer == kProducer && consumer == kEvent) g_emitted.fetch_add(1, std::memory_order_relaxed);
    }
    void fence() {
#if !defined(CALIBRATE_NO_FENCE)
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    }
};

void* producer_side(void*) {
    Ctx c;
    hgcommon::qc_add_producer(c, kS, 0, kOrbit, kProducer);
    return nullptr;
}

void* transition_side(void*) {
    Ctx c;
    ListOps ops{&g_head_trans};
    hgcommon::list_push(ops, 1);
    c.fence();
    if (g_reached[0].load(std::memory_order_acquire))
        hgcommon::qc_process_transition(c, g_t, kS, 0);
    return nullptr;
}

}  // namespace

int main() {
    Ctx root;
    hgcommon::qc_reach(root, kS, 0);   // S is reached at depth 0 before either side runs
    pthread_t a, b;
    pthread_create(&a, nullptr, producer_side, nullptr);
    pthread_create(&b, nullptr, transition_side, nullptr);
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);
    // At least one side emits the edge; neither emitting is the dropped pair.
    assert(g_emitted.load(std::memory_order_relaxed) >= 1 && "producer and transition both missed each other");
    return 0;
}
