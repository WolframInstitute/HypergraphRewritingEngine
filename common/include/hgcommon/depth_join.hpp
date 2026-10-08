#pragma once
#include "hgcommon/core.hpp"
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
// writes nothing but the atomics below -- no hypergraph, no job system -- so a model checker can
// be handed the protocol rather than the program it is embedded in, which is the only form in
// which this is checkable at all. verification/genmc/depth_report_order.cpp runs this header.
//
// THE STORAGE IS A TEMPLATE PARAMETER; the protocol is written once. FlatDepthSlots is a
// caller-owned array (a harness puts three on the stack); GrowingDepthSlots allocates a depth's
// slot the first time the depth is pushed, so the engine's storage is proportional to the depth
// a run reaches, not to its step budget.
//
// AN EMPTY DEPTH ENDS THE CASCADE. A task at depth d submits at d or d + 1, and every seed is
// pushed at or below the floor given to mark_roots_seeded. So a depth above the floor that the
// cascade reaches with nothing ever pushed at it (no task, no hold) can receive nothing, and
// neither can any depth above it: the cascade stops there instead of walking every empty depth
// up to the budget, and those depths are not reported.

#include "hgcommon/rendezvous.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace HG_NAMESPACE {
namespace common {

// ONE CACHE LINE PER DEPTH. `live` is the hottest counter in an evolution -- every task
// increments it when submitted and decrements it when done -- and four depths would fit in a
// 64-byte line at the natural 16-byte size. Depths run CONCURRENTLY by construction, so those
// four counters are written by different threads at the same time and the line ping-pongs
// between cores for no reason: nothing reads a neighbour's field.
struct alignas(64) DepthSlot {
    std::atomic<size_t>  live{0};       // tasks and holds pushed at this depth, minus those done
    std::atomic<uint8_t> complete{0};
    std::atomic<uint8_t> arrived{0};    // a task was pushed here (push)
    std::atomic<uint8_t> held{0};       // a hold was pushed here (hold)
};

// A caller-owned array of n slots.
struct FlatDepthSlots {
    DepthSlot* slots = nullptr;
    DepthSlot* find(uint32_t d) const { return &slots[d]; }
    DepthSlot& get(uint32_t d) { return slots[d]; }
    template <class F> void for_each(uint32_t n, F&& f) {
        for (uint32_t d = 0; d < n; ++d) f(slots[d]);
    }
};

#if !defined(__CUDACC__)
// Slots allocated on first use, in segments of 64 << k entries for k = 0, 1, ...: segment k holds
// depths [64 (2^k - 1), 64 (2^(k+1) - 1)), so 27 segments cover every uint32_t depth. A segment
// is installed by compare-exchange; a thread that loses frees its own and takes the winner's.
// Segments are kept across reset and freed by the destructor. `Extra` is the caller's per-depth
// datum, on the cache line after the slot.
struct NoDepthExtra {};
template <class Extra = NoDepthExtra>
class GrowingDepthSlots {
public:
    struct alignas(64) Entry { DepthSlot slot; Extra extra; };
    static constexpr uint32_t kSegments = 27;

    GrowingDepthSlots() {
        for (auto& s : seg_) s.store(nullptr, std::memory_order_relaxed);
    }
    ~GrowingDepthSlots() {
        for (auto& s : seg_) delete[] s.load(std::memory_order_relaxed);
    }
    GrowingDepthSlots(const GrowingDepthSlots&) = delete;
    GrowingDepthSlots& operator=(const GrowingDepthSlots&) = delete;

    DepthSlot* find(uint32_t d) const {
        Entry* e = find_entry(d);
        return e ? &e->slot : nullptr;
    }
    DepthSlot& get(uint32_t d) { return get_entry(d).slot; }
    // The caller's datum at depth d, or null when nothing was ever pushed in d's segment.
    Extra* find_extra(uint32_t d) const {
        Entry* e = find_entry(d);
        return e ? &e->extra : nullptr;
    }

    // Every allocated entry. Called with no task live (reset, and the caller's own reset).
    template <class F> void for_each(uint32_t, F&& f) {
        for_each_entry([&](Entry& e) { f(e.slot); });
    }
    template <class F> void for_each_extra(F&& f) {
        for_each_entry([&](Entry& e) { f(e.extra); });
    }

    // Bytes held by the allocated segments.
    size_t bytes() const {
        size_t b = 0;
        for (uint32_t k = 0; k < kSegments; ++k)
            if (seg_[k].load(std::memory_order_acquire)) b += size_of(k) * sizeof(Entry);
        return b;
    }

private:
    static uint64_t size_of(uint32_t k) { return uint64_t(64) << k; }
    static uint32_t segment_of(uint32_t d, uint64_t& off) {
        const uint64_t q = (uint64_t(d) >> 6) + 1;
        uint32_t k = 0;
        while ((q >> (k + 1)) != 0) ++k;
        off = uint64_t(d) - ((uint64_t(1) << k) - 1) * 64;
        return k;
    }
    template <class F> void for_each_entry(F&& f) {
        for (uint32_t k = 0; k < kSegments; ++k) {
            Entry* p = seg_[k].load(std::memory_order_acquire);
            if (!p) continue;
            for (uint64_t i = 0; i < size_of(k); ++i) f(p[i]);
        }
    }
    Entry* find_entry(uint32_t d) const {
        uint64_t off;
        const uint32_t k = segment_of(d, off);
        Entry* p = seg_[k].load(std::memory_order_acquire);
        return p ? &p[off] : nullptr;
    }
    Entry& get_entry(uint32_t d) {
        uint64_t off;
        const uint32_t k = segment_of(d, off);
        Entry* p = seg_[k].load(std::memory_order_acquire);
        if (!p) {
            Entry* fresh = new Entry[size_of(k)];
            if (seg_[k].compare_exchange_strong(p, fresh, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
                p = fresh;
            } else {
                delete[] fresh;
            }
        }
        return p[off];
    }

    std::atomic<Entry*> seg_[kSegments];
};
#endif

template <class Storage>
class DepthJoinT {
public:
    using Slot = DepthSlot;

    // `n` is the number of depths INCLUDING depth 0; a push at or above it is ignored.
    void seat(uint32_t n) { n_ = n; reset(); }

    Storage& storage() { return store_; }
    const Storage& storage() const { return store_; }

    void reset() {
        store_.for_each(n_, [](Slot& s) {
            s.live.store(0, std::memory_order_relaxed);
            s.complete.store(0, std::memory_order_relaxed);
            s.arrived.store(0, std::memory_order_relaxed);
            s.held.store(0, std::memory_order_relaxed);
        });
        seed_floor_ = 0;
        roots_seeded_.store(false, std::memory_order_relaxed);
        late_arrivals_.store(0, std::memory_order_relaxed);
        notified_.store(0, std::memory_order_relaxed);
        reporting_.store(false, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
    }

    uint32_t depths() const { return n_; }

    // Depth 0 cannot settle before the roots are in: until then arrivals are still moving and an
    // early match would fire the signal on an empty depth. `floor` is the deepest depth a seed
    // was pushed at; an empty depth at or below it does not end the cascade.
    void mark_roots_seeded(uint32_t floor = 0) {
        seed_floor_ = floor;
        roots_seeded_.store(true, std::memory_order_release);
    }

    // Arrivals at a depth that had already settled. The protocol's whole claim is that this
    // cannot happen, so it is counted rather than assumed: a non-zero value means a depth was
    // reported complete while work could still land in it.
    size_t late_arrivals() const { return late_arrivals_.load(std::memory_order_relaxed); }

    // Whether a task was pushed at `depth`. Read by the caller once the depth is reported.
    bool arrived(uint32_t depth) const {
        const Slot* s = depth < n_ ? store_.find(depth) : nullptr;
        return s && s->arrived.load(std::memory_order_acquire);
    }

    // Every task is booked at the depth it RUNS at: pushed before it can be seen, done after
    // every effect of it is visible. Returns true for the first task pushed at the depth.
    bool push(uint32_t depth) {
        if (depth >= n_) return false;
        Slot& s = store_.get(depth);
        const bool first = s.arrived.load(std::memory_order_relaxed) == 0 &&
                           s.arrived.exchange(1, std::memory_order_relaxed) == 0;
        add(s);
        return first;
    }

    // Holds `depth` open without counting as a task; released by done(depth).
    void hold(uint32_t depth) {
        if (depth >= n_) return;
        Slot& s = store_.get(depth);
        s.held.store(1, std::memory_order_relaxed);
        add(s);
    }

    template <class Emit>
    void done(uint32_t depth, Emit&& emit) {
        if (depth >= n_) return;
        // Settle only on the transition to zero: any other decrement leaves work live here.
        if (store_.get(depth).live.fetch_sub(1, std::memory_order_acq_rel) == 1)
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
            Slot* s = store_.find(d);
            if (s && s->complete.load(std::memory_order_acquire)) continue;
            if (d == 1) {
                if (!roots_seeded_.load(std::memory_order_acquire)) break;
            } else if (!settled(d - 1)) {
                break;
            }
            // Every depth below d is complete, so every push at d has been made (a task at d - 1
            // pushes before its done) and is visible through that completion. Nothing pushed
            // means nothing can arrive here or above, unless a seed was pushed this deep.
            if (!s || (!s->arrived.load(std::memory_order_acquire) &&
                       !s->held.load(std::memory_order_acquire))) {
                if (d > seed_floor_) break;
                s = &store_.get(d);
            }
            if (s->live.load(std::memory_order_acquire) != 0) break;

            uint8_t expected = 0;
            if (!s->complete.compare_exchange_strong(
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
            while (n + 1 < n_ && settled(n + 1)) {
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
            if (!(n + 1 < n_ && settled(n + 1))) return;
            // Something settled while the baton was being dropped. Take it back if it is free;
            // if another thread has it, that thread's own re-check covers this depth.
            if (reporting_.exchange(true, std::memory_order_acq_rel)) return;
        }
    }

    void add(Slot& s) {
        s.live.fetch_add(1, std::memory_order_acq_rel);
        if (s.complete.load(std::memory_order_acquire))
            HG_STAT(late_arrivals_.fetch_add(1, std::memory_order_relaxed));
    }
    bool settled(uint32_t d) const {
        const Slot* s = store_.find(d);
        return s && s->complete.load(std::memory_order_acquire);
    }

    Storage  store_{};
    uint32_t n_ = 0;
    uint32_t seed_floor_ = 0;   // written before roots_seeded_'s release, read after its acquire
    std::atomic<bool>     roots_seeded_{false};
    std::atomic<size_t>   late_arrivals_{0};
    std::atomic<uint32_t> notified_{0};
    std::atomic<bool>     reporting_{false};
};

// The protocol over a caller-owned array: `slots` must outlive the join.
class DepthJoin : public DepthJoinT<FlatDepthSlots> {
public:
    void seat(Slot* slots, uint32_t n) {
        storage().slots = slots;
        DepthJoinT<FlatDepthSlots>::seat(n);
    }
};

}  // namespace common
}  // namespace HG_NAMESPACE
