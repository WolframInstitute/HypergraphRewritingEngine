#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include "hgcommon/portable_intrinsics.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace hypergraph {

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

class SparseBitset {
public:
    static constexpr size_t BITS_PER_CHUNK = 512;  // 64 bytes per chunk (cache line)
    static constexpr size_t WORDS_PER_CHUNK = BITS_PER_CHUNK / 64;
    static constexpr size_t CHUNK_SHIFT = 9;  // log2(512)
    static constexpr size_t CHUNK_MASK = BITS_PER_CHUNK - 1;

    struct Chunk {
        uint64_t words[WORDS_PER_CHUNK];

        Chunk() {
            std::memset(words, 0, sizeof(words));
        }

        bool get(size_t bit_index) const {
            size_t word_idx = bit_index / 64;
            size_t bit_idx = bit_index % 64;
            return (words[word_idx] >> bit_idx) & 1;
        }

        void set(size_t bit_index) {
            size_t word_idx = bit_index / 64;
            size_t bit_idx = bit_index % 64;
            words[word_idx] |= (1ULL << bit_idx);
        }

        void clear(size_t bit_index) {
            size_t word_idx = bit_index / 64;
            size_t bit_idx = bit_index % 64;
            words[word_idx] &= ~(1ULL << bit_idx);
        }

        bool empty() const {
            for (size_t i = 0; i < WORDS_PER_CHUNK; ++i) {
                if (words[i] != 0) return false;
            }
            return true;
        }

        size_t popcount() const {
            size_t count = 0;
            for (size_t i = 0; i < WORDS_PER_CHUNK; ++i) {
                count += hgcommon::popcount64(words[i]);
            }
            return count;
        }

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
    SparseBitset()
        : entries_(nullptr)
        , num_entries_(0)
        , capacity_(0)
        , count_cached_(0)
        , count_valid_(true)
    {}

    // Move constructor - takes ownership of the other's data (single-owner
    // context: relaxed atomic access)
    SparseBitset(SparseBitset&& other) noexcept
        : entries_(other.entries_)
        , num_entries_(other.num_entries_)
        , capacity_(other.capacity_)
        , count_cached_(other.count_cached_.load(std::memory_order_relaxed))
        , count_valid_(other.count_valid_.load(std::memory_order_relaxed))
    {
        // Clear the source to prevent aliasing
        other.entries_ = nullptr;
        other.num_entries_ = 0;
        other.capacity_ = 0;
        other.count_cached_.store(0, std::memory_order_relaxed);
        other.count_valid_.store(true, std::memory_order_relaxed);
    }

    // Move assignment - takes ownership of the other's data (single-owner
    // context: relaxed atomic access)
    SparseBitset& operator=(SparseBitset&& other) noexcept {
        if (this != &other) {
            // Take over other's data
            entries_ = other.entries_;
            num_entries_ = other.num_entries_;
            capacity_ = other.capacity_;
            count_cached_.store(other.count_cached_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            count_valid_.store(other.count_valid_.load(std::memory_order_relaxed), std::memory_order_relaxed);

            // Clear the source to prevent aliasing
            other.entries_ = nullptr;
            other.num_entries_ = 0;
            other.capacity_ = 0;
            other.count_cached_.store(0, std::memory_order_relaxed);
            other.count_valid_.store(true, std::memory_order_relaxed);
        }
        return *this;
    }

    // Delete copy constructor and assignment to prevent accidental aliasing
    SparseBitset(const SparseBitset&) = delete;
    SparseBitset& operator=(const SparseBitset&) = delete;

    // Check if edge is present - O(log num_chunks)
    bool contains(uint32_t edge_id) const {
        if (num_entries_ == 0) return false;

        uint32_t chunk_id = edge_id >> CHUNK_SHIFT;
        size_t bit_index = edge_id & CHUNK_MASK;

        const Chunk* chunk = find_chunk(chunk_id);
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
    size_t count() const {
        if (count_valid_.load(std::memory_order_acquire)) {
            return count_cached_.load(std::memory_order_relaxed);
        }
        size_t total = 0;
        for (size_t i = 0; i < num_entries_; ++i) {
            total += entries_[i].chunk->popcount();
        }
        count_cached_.store(total, std::memory_order_relaxed);
        count_valid_.store(true, std::memory_order_release);
        return total;
    }

    // Is the set empty?
    bool empty() const {
        return count() == 0;
    }

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
    size_t num_chunks() const {
        return num_entries_;
    }

private:
    // Binary search for chunk by id (const version)
    const Chunk* find_chunk(uint32_t chunk_id) const {
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
    bool find_entry_index(uint32_t chunk_id, size_t& out_idx) const {
        size_t lo = 0, hi = num_entries_;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (entries_[mid].chunk_id < chunk_id) lo = mid + 1; else hi = mid;
        }
        out_idx = lo;
        return lo < num_entries_ && entries_[lo].chunk_id == chunk_id;
    }

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

    void invalidate_count() {
        count_valid_.store(false, std::memory_order_relaxed);
    }

    ChunkEntry* entries_;
    size_t num_entries_;
    size_t capacity_;
    // Atomic so concurrent const readers can fill the cache without a data race
    // (single-owner mutation paths use relaxed ops; see count()).
    mutable std::atomic<size_t> count_cached_;
    mutable std::atomic<bool> count_valid_;
};

}  // namespace hypergraph
