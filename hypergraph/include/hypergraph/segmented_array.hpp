#pragma once
#include "hgcommon/namespace.hpp"

#include "hgcommon/capacity.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "hgcommon/portable_intrinsics.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// SegmentedArray<T>: Append-only array with O(1) random access
// =============================================================================
//
// Array of fixed-size segments. Never reallocates existing segments, so
// pointers to elements remain stable. New segments allocated on demand.
//
// Thread safety: an element is placed at an index the caller's own counter handed out
// (emplace_at), or reached by index with no construction (slot). The array records no extent. A
// reader reaches an index only after the element's writer published that index to it (a job, a
// state's edge set, a fenced rendezvous), which orders the construction before the read.
//
// Usage:
//   SegmentedArray<Edge> edges;
//   edges.emplace_at(id, arena, ...);   // id from the caller's counter
//   Edge& e = edges[id];                // O(1) access
//

template<typename T>
class SegmentedArray {
public:
    static constexpr uint32_t DEFAULT_SEGMENT_SHIFT = 10;
    static constexpr size_t DEFAULT_SEGMENT_SIZE = size_t(1) << DEFAULT_SEGMENT_SHIFT;

    // OVERRIDABLE SO A MODEL CHECKER CAN AFFORD TO CONSTRUCT ONE. The directory is an inline array
    // of this many atomics and the constructor stores nullptr into every one, so a Hypergraph --
    // which holds seven of these, before the causal graph and the evolution engine add more --
    // costs 28,672 atomic stores to build. A checker models each as an event, so the object is
    // more expensive to construct than the step under test is to run, and the exploration never
    // reaches the property.
    //
    // The value changes CAPACITY and nothing else: the index decomposition and the growth
    // schedule are unchanged. A harness that shrinks it checks the same algorithm on a smaller
    // directory.
#ifndef HG_SEGMENTED_ARRAY_MAX_SEGMENTS
#define HG_SEGMENTED_ARRAY_MAX_SEGMENTS 4096
#endif
    // The largest segment shift a constructor is given effect for. A segment is zero-filled when
    // it is allocated, one store per element, and the engine's arrays hold elements of tens of
    // bytes; a checker bounding every loop needs the first segment small. The engine harnesses
    // define this (verification/genmc/engine_*.cpp); the shipped value caps nothing.
#ifndef HG_SEGMENTED_ARRAY_MAX_SHIFT
#define HG_SEGMENTED_ARRAY_MAX_SHIFT 63
#endif
    static constexpr size_t MAX_SEGMENTS = HG_SEGMENTED_ARRAY_MAX_SEGMENTS;

    // SEGMENTS GROW, so the capacity is the INDEX TYPE's limit and not the container's. Segment k
    // holds segment_size << min(k, GROWTH_STEPS) elements: the first is exactly as small as a
    // uniform one, so a run with ten edges still allocates ten edges' worth, and 4096 segments
    // reach 2^32 elements -- which is every value a uint32_t index can name. A workload cannot
    // exhaust this array without first exhausting the ids that address it.
    //
    // The doublings stop at GROWTH_STEPS so that no single segment allocation is unbounded: past
    // it every segment is segment_size << GROWTH_STEPS, and the growth that remains is in the
    // COUNT of segments rather than their size.
    static constexpr uint32_t GROWTH_STEPS = 10;

    // THE SIZE IS GIVEN AS A SHIFT, so a segment size that is not a power of two cannot be
    // expressed. The index decomposition is a shift and a mask rather than a 64-bit divide, which
    // is only valid for a power of two, and that was previously a runtime PRECONDITION -- checked
    // in the constructor and announced by printing to stderr and aborting.
    //
    // A constructor with an error state is the wrong shape for a constraint the caller can always
    // satisfy: every call site either takes the default or derives the value, so the check was
    // guarding against a mistake no caller could make while giving every caller a way to die.
    // Taking log2 as the parameter makes the illegal value unrepresentable instead of reported,
    // which removes the check, the abort, the stderr write and the ctz64 call together.
    //
    // It also removes the reason this class could not be model-checked. GenMC v0.17.0 crashes on
    // BOTH ways that precondition could announce itself -- std::fprintf segfaults it, and so does
    // a throw inside a constructor -- which is why an earlier attempt that removed only one of
    // them still saw the crash and concluded the class itself was untakeable. With no error state
    // there is nothing to announce.
    explicit SegmentedArray(uint32_t segment_shift = DEFAULT_SEGMENT_SHIFT)
        : segment_size_(size_t(1) << std::min<uint32_t>(segment_shift, HG_SEGMENTED_ARRAY_MAX_SHIFT))
        , seg_shift_(std::min<uint32_t>(segment_shift, HG_SEGMENTED_ARRAY_MAX_SHIFT))
        , seg_mask_(segment_size_ - 1)
        , geom_end_(segment_size_ * ((size_t(1) << (GROWTH_STEPS + 1)) - 1))
        , cap_shift_(segment_shift + GROWTH_STEPS)
        , cap_mask_((segment_size_ << GROWTH_STEPS) - 1) {
        for (size_t i = 0; i < MAX_SEGMENTS; ++i) {
            segments_[i].store(nullptr, std::memory_order_relaxed);
            ahead_[i].store(0, std::memory_order_relaxed);
        }
    }

    ~SegmentedArray() {
        // Note: We don't free segments here - they're arena-allocated
        // If using heap allocation, would need to track and free
    }

    // Non-copyable, non-movable
    SegmentedArray(const SegmentedArray&) = delete;
    SegmentedArray& operator=(const SegmentedArray&) = delete;
    SegmentedArray(SegmentedArray&&) = delete;
    SegmentedArray& operator=(SegmentedArray&&) = delete;

    // Access element by index - O(1). Does not wait: the caller holds the index only after the
    // element's writer published it to that caller (a job, a state's edge set), which orders
    // the construction before this read. The engine's num_states()/num_edges()/
    // num_raw_events() report claim counters, which run ahead of what is published, so they do
    // not bound a valid index.
    const T& at_published(uint32_t idx) const {
        const Loc L = locate(idx);
        T* segment = segments_[L.seg].load(std::memory_order_acquire);
        if (!segment) {
            throw std::logic_error(
                "SegmentedArray: the segment for this index is not published; the index was "
                "reached before its writer published it.");
        }
        return segment[L.off];
    }

    T& operator[](uint32_t idx) {
        return const_cast<T&>(static_cast<const SegmentedArray*>(this)->at_published(idx));
    }

    const T& operator[](uint32_t idx) const { return at_published(idx); }

    // The element at idx, its segment created if absent. Every element of a segment is
    // value-initialised by the arena before the segment pointer is published, so the element is
    // its default until a caller writes it. Concurrent callers for one index get one object.
    // The look-ahead is emplace_at's, for the same reason: callers index these arrays by slots
    // handed out in sequence, so the boundary is crossed by many threads at once.
    template<typename Arena>
    T& slot(uint32_t idx, Arena& arena) {
        const Loc L = locate(idx);
        T* segment = get_or_create_segment(L.seg, arena);
        look_ahead(L, arena);
        return segment[L.off];
    }

    // The element at idx if its segment exists: the reader of an array written through slot().
    // An element never written reads as its default.
    const T* find(uint32_t idx) const {
        const Loc L = locate(idx);
        T* segment = segments_[L.seg].load(std::memory_order_acquire);
        return segment ? &segment[L.off] : nullptr;
    }

    // Construct the element at an index the caller's counter handed out (edge, state and event
    // ids). The construction is published by the caller's own publication of the index; the
    // array adds none.
    template<typename Arena, typename... Args>
    void emplace_at(uint32_t idx, Arena& arena, Args&&... args) {
        const Loc L = locate(idx);
        const size_t seg_idx = L.seg, offset = L.off;

        // Ensure segment exists (thread-safe)
        T* segment = get_or_create_segment(seg_idx, arena);

        look_ahead(L, arena);

        // Construct the element directly with provided arguments
        new (&segment[offset]) T(std::forward<Args>(args)...);
#if defined(HG_CALIBRATE_SEGMENTED_HIGH_WATER)
        calibration_advance(idx);
#endif
    }

    // The segment geometry, for a caller that reasons about bytes per segment (the test that
    // pins the give-back of losing segment allocations).
    size_t segment_first_index(size_t seg_idx) const {
        if (seg_idx <= GROWTH_STEPS)
            return ((size_t(1) << seg_idx) - 1) << seg_shift_;
        return geom_end_ + ((seg_idx - (GROWTH_STEPS + 1)) << cap_shift_);
    }
    size_t segment_bytes(size_t seg_idx) const { return sizeof(T) * segment_capacity(seg_idx); }

private:
    // Segment and offset for an index. Two regimes, and the branch is predictable because a run
    // spends almost all of its accesses in whichever one its size puts it in.
    struct Loc { size_t seg; size_t off; };
    Loc locate(size_t idx) const {
        if (idx < geom_end_) {
            // idx lies in segment k where segment_size * (2^k - 1) <= idx, so k is the highest
            // set bit of idx/segment_size + 1.
            const size_t q = (idx >> seg_shift_) + 1;
            const size_t k = static_cast<size_t>(hgcommon::floor_log2_64(q));
            return { k, idx - (((size_t(1) << k) - 1) << seg_shift_) };
        }
        const size_t r = idx - geom_end_;
        return { (GROWTH_STEPS + 1) + (r >> cap_shift_), r & cap_mask_ };
    }

    size_t segment_capacity(size_t seg_idx) const {
        return segment_size_ << (seg_idx <= GROWTH_STEPS ? seg_idx : GROWTH_STEPS);
    }

    // Look ahead: the first thread to place an element in the second half of a segment creates
    // the next one. Indices are handed out by shared counters, so the threads that cross a
    // segment boundary do so within microseconds of each other; a thread that reaches an absent
    // segment allocates and zero-fills it, and all but one lose the install CAS. Half a segment
    // ahead the creator publishes long before the boundary is reached. The creator is elected by
    // an exchange on the next segment's flag, so the other threads in the second half read two
    // words and do not allocate. A single trigger index (three quarters through) was measured
    // too late on the replay's arrays, whose slots are filled out of order: bigpath n128 depth 3
    // quotient at 16 threads spent 29% of its time in memset of segments that lost the CAS.
    // Skipped at the last segment: the element being placed here still fits.
    template<typename Arena>
    void look_ahead(const Loc& L, Arena& arena) {
        if (L.off < segment_capacity(L.seg) / 2 || L.seg + 1 >= MAX_SEGMENTS) return;
        const size_t next = L.seg + 1;
        if (segments_[next].load(std::memory_order_relaxed)) return;
        if (ahead_[next].load(std::memory_order_relaxed)) return;
        if (ahead_[next].exchange(1, std::memory_order_relaxed)) return;
        get_or_create_segment(next, arena);
    }

    template<typename Arena>
    T* get_or_create_segment(size_t seg_idx, Arena& arena) {
        // The segment table is a fixed inline array, so an index past it would CAS into
        // whatever member follows and corrupt it with no symptom at the point of the
        // mistake. Every write path funnels through here, so one check covers them all.
        if (seg_idx >= MAX_SEGMENTS) {
            // A CONFIGURED LIMIT, not a defect: hgcommon::CapacityExhausted is what the job
            // system classifies on and what lets the engine serve the truncated graph with a
            // warning instead of terminating the caller. See hgcommon/capacity.hpp.
            throw hgcommon::CapacityExhausted(
                "SegmentedArray: capacity exhausted. Segments grow, so this is reached only "
                "once the array holds about 2^32 elements -- the whole range of the uint32_t "
                "index that names them. Widening the index is the only thing past it.");
        }

        T* segment = segments_[seg_idx].load(std::memory_order_acquire);
        if (segment) {
            return segment;
        }

        // Need to allocate new segment
        T* new_segment = arena.template allocate_array<T>(segment_capacity(seg_idx));

        // Try to install it
        T* expected = nullptr;
        if (segments_[seg_idx].compare_exchange_strong(
                expected, new_segment,
                std::memory_order_release,
                std::memory_order_acquire)) {
            return new_segment;
        }

        // Another thread published its segment first. The allocation above is this worker's
        // most recent, so the cursor gives it back and the next request reuses the bytes.
        // Trivially destructible elements only: allocate_array registers no destructor for
        // them, so nothing refers to the bytes once they are given back. Measured before this
        // give-back (bench_cpu_evolve wpp depth 7, same output): the arena's used bytes grew
        // from 191 MB at one thread to 394 MB at eight, all of it at this line.
        if constexpr (std::is_trivially_destructible_v<T>) {
            arena.release_last(new_segment, sizeof(T) * segment_capacity(seg_idx));
        }
        return expected;
    }

    size_t segment_size_;
    uint32_t seg_shift_;   // log2(segment_size_)
    size_t seg_mask_;      // segment_size_ - 1
    size_t geom_end_;      // first index past the doubling region
    uint32_t cap_shift_;   // log2(segment_size_ << GROWTH_STEPS)
    size_t cap_mask_;      // (segment_size_ << GROWTH_STEPS) - 1
    std::atomic<T*> segments_[MAX_SEGMENTS];
    // look_ahead's election: set by the one thread that creates segment i ahead of need.
    std::atomic<uint8_t> ahead_[MAX_SEGMENTS];

#if defined(HG_CALIBRATE_SEGMENTED_HIGH_WATER)
    // MODEL-CHECKER CALIBRATION (verification/genmc/segmented_array_published_read.cpp): an
    // extent advanced by each emplace_at as a high-water mark, and a reader bounded by it. A
    // lower index whose emplace is still constructing reads as published.
    std::atomic<uint32_t> count_{0};

public:
    const T* get(uint32_t idx) const {
        if (idx >= count_.load(std::memory_order_acquire)) return nullptr;
        return find(idx);
    }
    void calibration_advance(uint32_t idx) {
        uint32_t expected = count_.load(std::memory_order_relaxed);
        while (expected <= idx &&
               !count_.compare_exchange_weak(expected, idx + 1, std::memory_order_release,
                                             std::memory_order_relaxed)) {}
    }
#endif
};

}  // namespace engine
}  // namespace HG_NAMESPACE