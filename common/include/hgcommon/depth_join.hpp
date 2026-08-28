#pragma once
#include "hgcommon/namespace.hpp"
//
// THE DEPTH JOIN: deciding that a depth of a breadth-first exploration can receive no more work,
// and saying so in depth order.
//
// (Not join_core.hpp, which is the pattern-matching join. Same word, unrelated: that one binds a
// rule's edges, this one counts tasks.)
//
// A task at depth d only ever submits at depths ABOVE d. That single property is what lets a
// depth be declared finished with no barrier and no wait: once d-1 has settled, nothing can put
// work at d, so d is finished exactly when its own count reaches zero.
//
// THE PROTOCOL IS SEPARATE FROM ITS CALLER because it is checkable on its own. It reads and
// writes nothing but the atomics below -- no hypergraph, no job system, no allocation -- so a
// model checker can be handed the protocol rather than the program it is embedded in, which is
// the only form in which this is checkable at all. verification/genmc/depth_report_order.cpp
// runs this header.
//
// The caller owns the slot storage and passes it in. The engine sizes it once per run from its
// step budget; a harness puts four on the stack. Neither needs a second copy of the rule.

#include "hgcommon/rendezvous.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace HG_NAMESPACE {
namespace common {

class DepthJoin {
public:
    // ONE CACHE LINE PER DEPTH. `live` is the hottest counter in an evolution -- every task
    // increments it when submitted and decrements it when done -- and four depths would fit in a
    // 64-byte line at the natural 16-byte size. Depths run CONCURRENTLY by construction, so those
    // four counters are written by different threads at the same time and the line ping-pongs
    // between cores for no reason: nothing reads a neighbour's field.
    //
    // Sized once per run, so on a ten-step run the padding costs 768 bytes against 192. There is
    // no size at which this trade reverses.
    struct alignas(64) Slot {
        std::atomic<size_t>  live{0};       // tasks submitted at this depth, minus those done
        std::atomic<uint8_t> complete{0};
    };

    // `slots` must outlive this, and `n` is the number of depths INCLUDING depth 0.
    void seat(Slot* slots, uint32_t n) { slot_ = slots; n_ = n; reset(); }

    void reset() {
        for (uint32_t d = 0; d < n_; ++d) {
            slot_[d].live.store(0, std::memory_order_relaxed);
            slot_[d].complete.store(0, std::memory_order_relaxed);
        }
        roots_seeded_.store(false, std::memory_order_relaxed);
        late_arrivals_.store(0, std::memory_order_relaxed);
        notified_.store(0, std::memory_order_relaxed);
        reporting_.store(false, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
    }

    uint32_t depths() const { return n_; }

    // Depth 0 cannot settle before the roots are in: until then arrivals are still moving and an
    // early match would fire the signal on an empty depth.
    void mark_roots_seeded() { roots_seeded_.store(true, std::memory_order_release); }

    // Arrivals at a depth that had already settled. The protocol's whole claim is that this
    // cannot happen, so it is counted rather than assumed: a non-zero value means a depth was
    // reported complete while work could still land in it.
    size_t late_arrivals() const { return late_arrivals_.load(std::memory_order_relaxed); }

    // Every task is booked at the depth it RUNS at: pushed before it can be seen, done after
    // every effect of it is visible.
    void push(uint32_t depth) {
        if (depth >= n_) return;
        slot_[depth].live.fetch_add(1, std::memory_order_acq_rel);
        if (slot_[depth].complete.load(std::memory_order_acquire))
            late_arrivals_.fetch_add(1, std::memory_order_relaxed);
    }

    template <class Emit>
    void done(uint32_t depth, Emit&& emit) {
        if (depth >= n_) return;
        // Settle only on the transition to zero: any other decrement leaves work live here.
        if (slot_[depth].live.fetch_sub(1, std::memory_order_acq_rel) == 1)
            settle_from(depth, static_cast<Emit&&>(emit));
    }

    // Settle `depth` if it can be, then cascade: the depth above may have been waiting only on
    // this one, and may already have no live work of its own.
    template <class Emit>
    void settle_from(uint32_t depth, Emit&& emit) {
        // STORELOAD, and the protocol does not work without it. Settling is a symmetric
        // rendezvous between two threads that each write one location and then read the other:
        //
        //   the thread finishing at d+1   decrements live[d+1], then reads complete[d]
        //   the thread settling d         writes  complete[d],   then reads live[d+1]
        //
        // Under acquire/release both are permitted to read the value from before the other's
        // write, and then NEITHER settles d+1 -- it is not late, it never happens, and no
        // further event re-drives the cascade. Two fences, one on each side of the handshake,
        // forbid the outcome where both miss. This one covers the decrementing side (and the
        // seeding side, which stores roots_seeded_ and calls straight in); the one after the CAS
        // below covers the settling side.
        rendezvous_barrier<rv::DepthSettleCascade>();

        // Depth 0 runs no task -- a root's match task runs at depth 1 -- so it is complete by
        // definition once the roots are in, and the chain starts above it.
        for (uint32_t d = (depth == 0 ? 1u : depth); d < n_; ++d) {
            if (slot_[d].complete.load(std::memory_order_acquire)) continue;
            if (d == 1) {
                if (!roots_seeded_.load(std::memory_order_acquire)) break;
            } else if (!slot_[d - 1].complete.load(std::memory_order_acquire)) {
                break;
            }
            if (slot_[d].live.load(std::memory_order_acquire) != 0) break;

            uint8_t expected = 0;
            if (!slot_[d].complete.compare_exchange_strong(
                    expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                continue;   // another thread settled it; its cascade covers the depths above
            }
            // The settling side of the handshake described at the top: this thread has just
            // published complete[d] and is about to read live[d+1] on the next iteration.
            rendezvous_barrier<rv::DepthSettleCascade>();
        }
        // On EVERY exit, including the early ones: a thread that settles a depth and then stops
        // because the one above is not ready still owes that depth's report.
        report(static_cast<Emit&&>(emit));
    }

private:
    // REPORTING INHERITS THE SETTLE ORDER rather than the order the settling threads are
    // scheduled in. Settling a depth and reporting it cannot be one step, so a thread that
    // settles d can be descheduled before it reports while another walks past the now-complete d,
    // settles d+1 and reports that first -- describing a run in which d+1 drained before d,
    // which never happened.
    //
    // A cursor claimed per depth is not enough, and the harness says so: claiming and emitting
    // are themselves two steps, so the thread that claims d can be descheduled before its emit
    // while the thread its claim just released emits d+1. Whatever the unit, the release of the
    // next step must come AFTER the report of this one.
    //
    // So ONE REPORTER AT A TIME, and it drains as far as it can. A thread that cannot take the
    // baton returns immediately -- it never waits on another thread, and it does not need to,
    // because the holder re-checks after releasing and picks up whatever settled meanwhile. The
    // re-check is what closes the window where a depth settles just as the baton is dropped.
    //
    // notified_ starts at 0 because depth 0 runs no task and is never reported -- a correct
    // starting value rather than a sentinel.
    template <class Emit>
    void report(Emit&& emit) {
        if (reporting_.exchange(true, std::memory_order_acq_rel)) return;
        for (;;) {
            uint32_t n = notified_.load(std::memory_order_relaxed);
            while (n + 1 < n_ && slot_[n + 1].complete.load(std::memory_order_acquire)) {
                emit(n + 1);
                notified_.store(n + 1, std::memory_order_release);
                ++n;
            }
            reporting_.store(false, std::memory_order_release);
            // STORELOAD AGAIN, for the same reason and between the same kind of pair: dropping
            // the baton and re-checking is a write then a read, and a thread that settles a
            // depth just then writes complete[d] and reads the baton. Without the fence both
            // may read the value from before the other's write -- the settler sees the baton
            // held and leaves, the holder sees nothing new and leaves -- and the depth is never
            // reported. Its partner is the fence after the settling CAS.
            rendezvous_barrier<rv::DepthReportBaton>();
            if (!(n + 1 < n_ && slot_[n + 1].complete.load(std::memory_order_acquire))) return;
            // Something settled while the baton was being dropped. Take it back if it is free;
            // if another thread has it, that thread's own re-check covers this depth.
            if (reporting_.exchange(true, std::memory_order_acq_rel)) return;
        }
    }

    Slot*    slot_ = nullptr;
    uint32_t n_ = 0;
    std::atomic<bool>     roots_seeded_{false};
    std::atomic<size_t>   late_arrivals_{0};
    std::atomic<uint32_t> notified_{0};
    std::atomic<bool>     reporting_{false};
};

}  // namespace common
}  // namespace HG_NAMESPACE
