#pragma once
#include "hgcommon/namespace.hpp"
//
// THE SYMMETRIC RENDEZVOUS, in one place, because the engine has nine of them and one of them
// was written with the fence on a single side.
//
// THE SHAPE. Two threads must meet, and neither knows whether the other has arrived. Each
// publishes its own arrival and then scans for the other's:
//
//     thread A:  write X   then read Y      acts only if it observes Y
//     thread B:  write Y   then read X      acts only if it observes X
//
// Under acquire/release BOTH reads may return the value from before the other's write, so both
// threads decide the other is absent and the meeting never happens. It is not delayed and it is
// not retried: nothing further will re-drive it, because from each thread's view there was
// nothing to see. A StoreLoad barrier on EACH side forbids that outcome -- at least one of the
// two must observe the other -- and that is the whole content of this header.
//
// WHEN NOT TO USE IT. Only one side needs to read for the pair to be safe. The engine's
// per-state match join is asymmetric: note_match_task_pushed only increments, and
// note_match_task_done increments its own counter and then reads the other, so the ordering runs
// through the read-modify-write chain on the counter it shares and no fence is required.
// Wrapping a case like that would put an mfence on the hottest counter in the engine to buy
// nothing. The test is whether BOTH sides read the other's location to decide whether to act.
//
// A single shared location is also not this. Two threads that both push to one lock-free list
// and then walk that same list are ordered by coherence on the head: whichever pushes second
// necessarily sees the first. The engine's branchial co-consumer bucket is that case.
//
// NOR IS A READ THAT ONLY SKIPS AN OPTIMISATION. The job system's completion notify writes the
// sequence counter and then reads the waiter count, and the waiter writes the waiter count and
// then reads the sequence -- which looks exactly like the class. It is not, because the two reads
// do not gate the same thing. Missing the waiter count skips a wake; missing the sequence does
// not lose anything, because the waiter parks on that very word under a value compare and the
// write it might have missed is the value it compares against. What makes it safe is the ORDER --
// publish the sequence, THEN look for a waiter -- not a barrier between them. Checked: removing
// the barrier leaves a checker unable to tell the two apart.
//
// So the test is not "do both sides read". It is: DOES MISSING THE READ LOSE THE EVENT WITH NO
// OTHER PATH TO IT. In the depth join it did -- a depth that neither thread settles is never
// settled by anything later.
//
// COST. Both callables inline and what is emitted is publish(); mfence; scan(); -- the same
// instructions the hand-written sites had. The tag is a compile-time name and generates nothing.
//
// THE TAG NAMES THE PARTNER. Both halves of one rendezvous carry the same tag, so the pair is
// greppable and a rendezvous with only one side shows up as a tag used once. The tags are
// declared together below rather than at their use sites, so the list of the engine's handshakes
// is one thing a reader can look at.

#include <atomic>
#include <cstdint>
#if defined(HG_RENDEZVOUS_CHAOS)
#  include <sched.h>
#endif

namespace HG_NAMESPACE {
namespace common {

namespace rv {
// Every symmetric handshake in the engine, named so both halves of one can be found together.
//
// A TAG USED ONCE IS A QUESTION, not necessarily a fault: usually it means the missing half, and
// that is what this list is for. But one barrier can serve two pairings -- a thread that has just
// published complete[d] is simultaneously the partner of the thread finishing at d+1 and of the
// thread re-checking the report baton -- so a single use may be answered by a DIFFERENT tag's
// site rather than by its own. Each comment below names where the partner is.
struct EdgeProducerConsumer {};   // causal_graph: producer publishes / consumer publishes
struct QuotientInstanceMatch {};  // hypergraph: qc_add_instance / qc_capture_expansion
struct MatchStoreScan {};         // parallel_evolution: store_match_for_state / the scan
struct CanonChildDepth {};        // parallel_evolution: child registration / propagate_explore_depth
struct DepthSettleCascade {};     // depth_join: settle d / finish at d+1
struct DepthReportBaton {};       // depth_join: drop the baton; partner is the post-CAS
                                  //   DepthSettleCascade barrier, which the settler already holds
struct QuotientCoreHook {};       // quotient_causal_core's Ctx::fence, host realisation: covers
                                  //   qc_reach and qc_add_producer, and pairs with the transition
                                  //   registration in register_canonical_transition
struct WorkerParkWake {};         // job_system: wake_one_worker / a worker announcing itself idle
struct DequeTakeSteal {};         // work_stealing_deque: the owner taking / a thief stealing
}  // namespace rv

// publish() the caller's own arrival, then scan() for the other side's. See above for the
// argument, and for the two shapes that must NOT be routed through here.
// SCHEDULE CHAOS, test builds only (-DHG_RENDEZVOUS_CHAOS=1). Every handshake in the engine passes
// through here, so yielding the CPU with probability 1/8 between publish and fence, and between
// fence and scan, puts the other side's whole action into exactly the windows a missed rendezvous
// would need. Nothing else changes: the fence and the order are the shipped ones.
#if defined(HG_RENDEZVOUS_CHAOS)
inline void rendezvous_chaos_point() {
    static thread_local uint32_t s = 0x9E3779B9u ^ static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&s) >> 4);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    if ((s & 7u) == 0u) sched_yield();
}
#else
inline void rendezvous_chaos_point() {}
#endif

template <class Tag, class Publish, class Scan>
inline void rendezvous(Publish&& publish, Scan&& scan) {
    publish();
    rendezvous_chaos_point();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    rendezvous_chaos_point();
    scan();
}

// The half of a rendezvous whose two sides are not adjacent in one function -- the publish
// happens in one call and the scan in another, so they cannot be handed to rendezvous() together.
// Same barrier, same argument, and the same tag as its partner; it exists so those sites are
// counted and greppable with the rest rather than carrying a bare fence.
template <class Tag>
inline void rendezvous_barrier() {
    rendezvous_chaos_point();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    rendezvous_chaos_point();
}

}  // namespace common
}  // namespace HG_NAMESPACE
