#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include "hgcommon/core.hpp"
#include "hgcommon/portable_intrinsics.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// SparseBitset: Sparse chunked bitset for edge membership tracking
// =============================================================================
//
// Tracks which EdgeIds are present in a state. Only allocates storage for
// non-empty chunks, making it memory-efficient for sparse states.
//
// Interface designed to be implementation-agnostic - can be swapped for
// dense bitset, sorted array, or hash set if needed.
//
// Thread safety: Safe for concurrent reads after construction.
//                Single-writer during construction only.
//
// Allocation: All memory allocated from provided arena.
//

// The counters below stay in this header, and are the one place in it that is neither a
// template nor force-inlined. They are inside HG_BITSET_STATS, which no shipping build
// defines, and the accumulator is read from inside contains() -- which is force-inlined for
// a measured reason recorded at its definition.
#ifdef HG_BITSET_STATS
// SparseBitset search-depth histogram. Answers one question: how many DEPENDENT loads does
// contains() serialise? Chain length, not instruction count, is what a latency-bound path pays.
// Registered as HG_BITSET_STATS in the top-level CMakeLists; never on in a shipping build.
struct BitsetStats {
    uint64_t calls = 0;          // contains() invocations
    uint64_t iters = 0;          // binary-search iterations (each one a dependent load)
    uint64_t entries = 0;        // sum of num_entries_ seen, for the mean chunk count
    uint64_t dense = 0;          // calls whose entries_ were a contiguous chunk_id run
    uint64_t hits = 0;           // calls that found the chunk
};
// Global accumulators. Each worker's thread_local folds itself in when the thread ends, so a
// reader after the join sees every worker without any synchronisation on the hot path.
struct BitsetStatsGlobal {
    std::atomic<uint64_t> calls{0}, iters{0}, entries{0}, dense{0}, hits{0};
};
inline BitsetStatsGlobal& bitset_stats_global() {
    static BitsetStatsGlobal g;
    return g;
}
struct BitsetStatsTLS {
    BitsetStats s;
    ~BitsetStatsTLS() {
        BitsetStatsGlobal& g = bitset_stats_global();
        g.calls.fetch_add(s.calls, std::memory_order_relaxed);
        g.iters.fetch_add(s.iters, std::memory_order_relaxed);
        g.entries.fetch_add(s.entries, std::memory_order_relaxed);
        g.dense.fetch_add(s.dense, std::memory_order_relaxed);
        g.hits.fetch_add(s.hits, std::memory_order_relaxed);
    }
};
inline BitsetStats& bitset_stats() {
    static thread_local BitsetStatsTLS t;
    return t.s;
}
// Folds the CALLING thread in as well, so a single-threaded run reports without waiting on
// destruction order. Idempotent per call site only in the sense that it zeroes what it folds.
inline void bitset_stats_report(const char* tag) {
    BitsetStats& me = bitset_stats();
    BitsetStatsGlobal& g = bitset_stats_global();
    uint64_t calls   = g.calls.load(std::memory_order_relaxed)   + me.calls;
    uint64_t iters   = g.iters.load(std::memory_order_relaxed)   + me.iters;
    uint64_t entries = g.entries.load(std::memory_order_relaxed) + me.entries;
    uint64_t dense   = g.dense.load(std::memory_order_relaxed)   + me.dense;
    uint64_t hits    = g.hits.load(std::memory_order_relaxed)    + me.hits;
    std::fprintf(stderr,
        "[bitset:%s] contains_calls=%llu mean_entries=%.2f mean_search_depth=%.2f "
        "dense_frac=%.4f hit_frac=%.4f\n",
        tag,
        (unsigned long long)calls,
        calls ? double(entries) / double(calls) : 0.0,
        calls ? double(iters) / double(calls) : 0.0,
        calls ? double(dense) / double(calls) : 0.0,
        calls ? double(hits) / double(calls) : 0.0);
}
#endif

class SparseBitset {
public:
    static constexpr size_t BITS_PER_CHUNK = 512;  // 64 bytes per chunk (cache line)
    static constexpr size_t WORDS_PER_CHUNK = BITS_PER_CHUNK / 64;
    static constexpr size_t CHUNK_SHIFT = 9;  // log2(512)
    static constexpr size_t CHUNK_MASK = BITS_PER_CHUNK - 1;

    struct Chunk {
        uint64_t words[WORDS_PER_CHUNK];

        Chunk();

        bool get(size_t bit_index) const;
        void set(size_t bit_index);
        void clear(size_t bit_index);
        bool empty() const;
        size_t popcount() const;

        template<typename F>
        void for_each(size_t base_id, F&& f) const {
            for (size_t w = 0; w < WORDS_PER_CHUNK; ++w) {
                uint64_t word = words[w];
                while (word) {
                    size_t bit = hgcommon::ctz64(word);
                    f(static_cast<uint32_t>(base_id + w * 64 + bit));
                    word &= word - 1;  // Clear lowest set bit
                }
            }
        }
    };

    // Entry in the chunk index. `owned` distinguishes a chunk this bitset allocated
    // (safe to mutate in place) from a chunk SHARED by reference from a parent state
    // (copy-on-write before any mutation). Chunks are immutable once a state is
    // published, so sharing them across states is lock-free and race-free. Field order
    // keeps the struct at 16 bytes.
    struct ChunkEntry {
        Chunk* chunk;
        uint32_t chunk_id;
        bool owned;
    };

    // Default constructor - empty bitset
    SparseBitset();

    // Move constructor - takes ownership of the other's data (single-owner
    // context: relaxed atomic access)
    SparseBitset(SparseBitset&& other) noexcept;

    // Move assignment - takes ownership of the other's data (single-owner
    // context: relaxed atomic access)
    SparseBitset& operator=(SparseBitset&& other) noexcept;

    // Delete copy constructor and assignment to prevent accidental aliasing
    SparseBitset(const SparseBitset&) = delete;
    SparseBitset& operator=(const SparseBitset&) = delete;

    // Is this edge in the set? O(log num_chunks).
    //
    // Force-inlined, with find_chunk, because it is the engine's hottest predicate -- 15% of a
    // 5-step Wolfram evolution -- and as a mere hint its inlining tracked the size of whatever
    // else lived in the same translation unit. Pointing the matcher at the shared join added
    // template instantiations to this header's users; GCC's per-unit budget tightened and
    // contains was outlined from ParallelEvolutionEngine::execute_expand_task, which cost
    // 2.15M instructions (+3.7%) while changing no arithmetic. Forcing only contains and
    // leaving find_chunk a hint is WORSE than either (+4.5%): contains becomes a wrapper around
    // a call. Measured with tools/cost_sweep.sh, wolfram-5step.
    HG_INLINE bool contains(uint32_t edge_id) const {
        if (num_entries_ == 0) return false;

        uint32_t chunk_id = edge_id >> CHUNK_SHIFT;
        size_t bit_index = edge_id & CHUNK_MASK;


#ifdef HG_BITSET_STATS
        {
            BitsetStats& st = bitset_stats();
            ++st.calls;
            st.entries += num_entries_;
            size_t n = num_entries_;
            size_t d = 0;
            while ((size_t(1) << d) < n) ++d;
            st.iters += d;
            bool contiguous = true;
            for (size_t i = 1; i < n; ++i) {
                if (entries_[i].chunk_id != entries_[i - 1].chunk_id + 1) { contiguous = false; break; }
            }
            if (contiguous) ++st.dense;
        }
#endif

        const Chunk* chunk = find_chunk(chunk_id);
#ifdef HG_BITSET_STATS
        if (chunk) ++bitset_stats().hits;
#endif
        return chunk && chunk->get(bit_index);
    }

    // Add edge to set
    // Arena must be provided for potential new chunk allocation
    template<typename Arena>
    void set(uint32_t edge_id, Arena& arena) {
        uint32_t chunk_id = edge_id >> CHUNK_SHIFT;
        size_t bit_index = edge_id & CHUNK_MASK;

        Chunk* chunk = find_or_create_owned_chunk(chunk_id, arena);
        if (!chunk->get(bit_index)) {
            chunk->set(bit_index);
            invalidate_count();
        }
    }

    // Remove edge from set. Needs the arena to copy-on-write a shared chunk before
    // mutating it (a shared chunk belongs to a parent state and must not be touched).
    template<typename Arena>
    void clear(uint32_t edge_id, Arena& arena) {
        uint32_t chunk_id = edge_id >> CHUNK_SHIFT;
        size_t bit_index = edge_id & CHUNK_MASK;

        size_t idx;
        if (!find_entry_index(chunk_id, idx)) return;
        if (!entries_[idx].chunk->get(bit_index)) return;
        Chunk* chunk = make_entry_owned(idx, arena);  // COW if shared
        chunk->clear(bit_index);
        invalidate_count();
        // Note: we don't remove empty chunks - they stay allocated
    }

    // Number of set bits. The lazy cache fill is safe under concurrent const
    // readers: a bitset is only shared between threads once its contents are
    // immutable, so racing fills compute the same total (idempotent). The cached
    // value is stored before the valid flag (release) so a reader that observes
    // valid (acquire) also observes the value.
    size_t count() const;

    // Is the set empty?
    bool empty() const;

    // Iterate over all set bits
    template<typename F>
    void for_each(F&& f) const {
        for (size_t i = 0; i < num_entries_; ++i) {
            uint32_t base_id = entries_[i].chunk_id << CHUNK_SHIFT;
            entries_[i].chunk->for_each(base_id, f);
        }
    }

    // Create a derived bitset: copy parent, clear consumed, set produced
    // This is the typical pattern for creating a child state's edge set
    template<typename Arena>
    static SparseBitset derive(
        const SparseBitset& parent,
        const uint32_t* consumed, size_t num_consumed,
        const uint32_t* produced, size_t num_produced,
        Arena& arena
    ) {
        SparseBitset result;

        // Share the parent's chunks BY REFERENCE (copy-on-write). Chunks are immutable
        // once a state is published, so the child can point straight at them; only the
        // handful of chunks a consumed/produced edge actually touches get copied — on
        // write, below. This is the difference between O(E) and O(delta) memory per
        // derived state, and it removes the per-chunk memcpy from the rewrite hot path.
        if (parent.num_entries_ > 0) {
            size_t initial_capacity = parent.num_entries_ + 4;  // Room for new chunks
            result.entries_ = arena.template allocate_array<ChunkEntry>(initial_capacity);
            result.capacity_ = initial_capacity;

            for (size_t i = 0; i < parent.num_entries_; ++i) {
                result.entries_[i].chunk    = parent.entries_[i].chunk;   // shared
                result.entries_[i].chunk_id = parent.entries_[i].chunk_id;
                result.entries_[i].owned    = false;                      // COW on first write
            }
            result.num_entries_ = parent.num_entries_;
        }

        // Clear consumed edges (copy-on-write the touched chunk).
        for (size_t i = 0; i < num_consumed; ++i) {
            result.clear(consumed[i], arena);
        }

        // Set produced edges (copy-on-write existing chunks / create owned new ones).
        for (size_t i = 0; i < num_produced; ++i) {
            result.set(produced[i], arena);
        }

        return result;
    }

    // Create from a list of edge IDs
    template<typename Arena>
    static SparseBitset from_edges(const uint32_t* edges, size_t num_edges, Arena& arena) {
        SparseBitset result;
        for (size_t i = 0; i < num_edges; ++i) {
            result.set(edges[i], arena);
        }
        return result;
    }

    // Number of chunks (for diagnostics)
    size_t num_chunks() const;

private:
    // Binary search for chunk by id (const version)
    HG_INLINE const Chunk* find_chunk(uint32_t chunk_id) const {
        if (num_entries_ == 0) return nullptr;

        size_t lo = 0, hi = num_entries_;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (entries_[mid].chunk_id < chunk_id) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < num_entries_ && entries_[lo].chunk_id == chunk_id) {
            return entries_[lo].chunk;
        }
        return nullptr;
    }

    // Binary search. Returns true and sets out_idx to the entry index when chunk_id
    // is present; otherwise returns false and sets out_idx to the insertion point.
    bool find_entry_index(uint32_t chunk_id, size_t& out_idx) const;

    // Ensure entries_[idx]'s chunk is private to this bitset. If it is shared from a
    // parent (copy-on-write), copy it once and take ownership. Returns the mutable chunk.
    template<typename Arena>
    Chunk* make_entry_owned(size_t idx, Arena& arena) {
        if (!entries_[idx].owned) {
            Chunk* copy = arena.template create<Chunk>();
            std::memcpy(copy->words, entries_[idx].chunk->words, sizeof(Chunk::words));
            entries_[idx].chunk = copy;
            entries_[idx].owned = true;
        }
        return entries_[idx].chunk;
    }

    // Find or create a chunk owned by this bitset (COW an existing shared chunk),
    // maintaining sorted order.
    template<typename Arena>
    Chunk* find_or_create_owned_chunk(uint32_t chunk_id, Arena& arena) {
        size_t idx;
        if (find_entry_index(chunk_id, idx)) {
            return make_entry_owned(idx, arena);
        }

        // Insert a fresh, owned chunk at position idx.
        ensure_capacity(arena);
        for (size_t i = num_entries_; i > idx; --i) {
            entries_[i] = entries_[i - 1];
        }
        Chunk* new_chunk = arena.template create<Chunk>();
        entries_[idx].chunk = new_chunk;
        entries_[idx].chunk_id = chunk_id;
        entries_[idx].owned = true;
        ++num_entries_;

        return new_chunk;
    }

    template<typename Arena>
    void ensure_capacity(Arena& arena) {
        if (num_entries_ < capacity_) return;

        size_t new_capacity = (capacity_ == 0) ? 4 : capacity_ * 2;
        ChunkEntry* new_entries = arena.template allocate_array<ChunkEntry>(new_capacity);

        if (entries_) {
            std::memcpy(new_entries, entries_, num_entries_ * sizeof(ChunkEntry));
        }

        entries_ = new_entries;
        capacity_ = new_capacity;
        // Old entries_ left in arena, not freed (arena semantics)
    }

    void invalidate_count();

    ChunkEntry* entries_;
    size_t num_entries_;
    size_t capacity_;
    // Atomic so concurrent const readers can fill the cache without a data race
    // (single-owner mutation paths use relaxed ops; see count()).
    mutable std::atomic<size_t> count_cached_;
    mutable std::atomic<bool> count_valid_;
};

}  // namespace engine
}  // namespace HG_NAMESPACE