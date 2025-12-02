#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace hypergraph::unified {

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
                count += __builtin_popcountll(words[i]);
            }
            return count;
        }

        template<typename F>
        void for_each(size_t base_id, F&& f) const {
            for (size_t w = 0; w < WORDS_PER_CHUNK; ++w) {
                uint64_t word = words[w];
                while (word) {
                    size_t bit = __builtin_ctzll(word);
                    f(static_cast<uint32_t>(base_id + w * 64 + bit));
                    word &= word - 1;  // Clear lowest set bit
                }
            }
        }
    };

    // Entry in the chunk index: maps chunk_id to chunk pointer
    struct ChunkEntry {
        uint32_t chunk_id;
        Chunk* chunk;
    };

    // Default constructor - empty bitset
    SparseBitset()
        : entries_(nullptr)
        , num_entries_(0)
        , capacity_(0)
        , count_cached_(0)
        , count_valid_(true)
    {}

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

        Chunk* chunk = find_or_create_chunk(chunk_id, arena);
        if (!chunk->get(bit_index)) {
            chunk->set(bit_index);
            invalidate_count();
        }
    }

    // Remove edge from set
    void clear(uint32_t edge_id) {
        uint32_t chunk_id = edge_id >> CHUNK_SHIFT;
        size_t bit_index = edge_id & CHUNK_MASK;

        Chunk* chunk = find_chunk_mutable(chunk_id);
        if (chunk && chunk->get(bit_index)) {
            chunk->clear(bit_index);
            invalidate_count();
            // Note: we don't remove empty chunks - they stay allocated
        }
    }

    // Number of set bits
    size_t count() const {
        if (count_valid_) {
            return count_cached_;
        }
        size_t total = 0;
        for (size_t i = 0; i < num_entries_; ++i) {
            total += entries_[i].chunk->popcount();
        }
        count_cached_ = total;
        count_valid_ = true;
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

        // Copy parent's entries
        if (parent.num_entries_ > 0) {
            size_t initial_capacity = parent.num_entries_ + 4;  // Room for new chunks
            result.entries_ = arena.template allocate_array<ChunkEntry>(initial_capacity);
            result.capacity_ = initial_capacity;

            for (size_t i = 0; i < parent.num_entries_; ++i) {
                // Deep copy each chunk
                Chunk* new_chunk = arena.template create<Chunk>();
                std::memcpy(new_chunk->words, parent.entries_[i].chunk->words, sizeof(Chunk::words));
                result.entries_[i].chunk_id = parent.entries_[i].chunk_id;
                result.entries_[i].chunk = new_chunk;
            }
            result.num_entries_ = parent.num_entries_;
        }

        // Clear consumed edges
        for (size_t i = 0; i < num_consumed; ++i) {
            result.clear(consumed[i]);
        }

        // Set produced edges
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

    // Binary search for chunk by id (mutable version)
    Chunk* find_chunk_mutable(uint32_t chunk_id) {
        return const_cast<Chunk*>(find_chunk(chunk_id));
    }

    // Find or create chunk, maintaining sorted order
    template<typename Arena>
    Chunk* find_or_create_chunk(uint32_t chunk_id, Arena& arena) {
        // Find insertion point
        size_t lo = 0, hi = num_entries_;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (entries_[mid].chunk_id < chunk_id) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Check if already exists
        if (lo < num_entries_ && entries_[lo].chunk_id == chunk_id) {
            return entries_[lo].chunk;
        }

        // Need to insert at position 'lo'
        ensure_capacity(arena);

        // Shift entries to make room
        for (size_t i = num_entries_; i > lo; --i) {
            entries_[i] = entries_[i - 1];
        }

        // Create new chunk
        Chunk* new_chunk = arena.template create<Chunk>();
        entries_[lo].chunk_id = chunk_id;
        entries_[lo].chunk = new_chunk;
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
        count_valid_ = false;
    }

    ChunkEntry* entries_;
    size_t num_entries_;
    size_t capacity_;
    mutable size_t count_cached_;
    mutable bool count_valid_;
};

}  // namespace hypergraph::unified
