#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "hgcommon/core.hpp"
#include "hgcommon/portable_intrinsics.hpp"

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// SparseBitset: the set of EdgeIds present in a state
// =============================================================================
//
// Edge ids are grouped in chunks of 512 bits (64 bytes); a chunk is allocated once one of its
// ids is set. A derived state shares its parent's chunks and copies the chunks it changes
// (copy-on-write). The chunks are found through one of two indexes:
//
// - FLAT, up to kFlatChunks (32) chunks: a sorted array of (chunk id, chunk) entries, searched by
//   bisection. A derived state copies the array: at most 32 x 16 bytes.
// - RADIX, beyond that: a tree of 8-slot nodes (64 bytes each) indexed by chunk id, `height_`
//   levels deep (height h covers chunk ids below 8^h). A derived state shares its parent's tree
//   and copies the path from the root to each chunk it changes (path copying), so a state of any
//   size costs O(changed chunks x height) bytes and a single history of n events O(n log n).
// An index turns radix when it passes kFlatChunks and stays radix in every state derived from it.
//
// Ownership is a tag: the `owned` field of a flat entry, the low bit of a radix slot (and of
// root_). It is set when this bitset allocated the node or chunk, which may then be mutated in
// place, and clear when it is shared with the parent, which is copied before the first write.
// A node's copy carries the parent's slots with the tag cleared. Nodes and chunks of a published
// state are never written again, so sharing them across states and threads is race-free.
//
// Thread safety: concurrent reads after construction; one writer during construction.
// Allocation: entries, nodes and chunks come from the arena passed to set, clear, derive and
// from_edges.

class SparseBitset {
public:
    static constexpr size_t BITS_PER_CHUNK = 512;  // 64 bytes per chunk (cache line)
    static constexpr size_t WORDS_PER_CHUNK = BITS_PER_CHUNK / 64;
    static constexpr size_t CHUNK_SHIFT = 9;  // log2(512)
    static constexpr size_t CHUNK_MASK = BITS_PER_CHUNK - 1;
    static constexpr uint32_t FANOUT_SHIFT = 3;
    static constexpr uint32_t FANOUT = 1u << FANOUT_SHIFT;  // 8 slots: one 64-byte node
    static constexpr uint32_t kFlatChunks = 32;
    static constexpr uint32_t kRadix = ~uint32_t{0};  // num_entries_ of a radix index

    struct Chunk {
        uint64_t words[WORDS_PER_CHUNK];

        Chunk();

        HG_INLINE bool get(size_t bit_index) const { return (words[bit_index >> 6] >> (bit_index & 63)) & 1; }
        void set(size_t bit_index) { words[bit_index >> 6] |= uint64_t{1} << (bit_index & 63); }
        void clear(size_t bit_index) { words[bit_index >> 6] &= ~(uint64_t{1} << (bit_index & 63)); }
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

    // A flat index entry. Field order keeps the struct at 16 bytes.
    struct ChunkEntry {
        Chunk* chunk;
        uint32_t chunk_id;
        bool owned;
    };

    struct Node {
        uintptr_t slot[FANOUT];  // child pointer | owned tag; 0 for an absent subtree
        Node() { std::memset(slot, 0, sizeof(slot)); }
    };

    SparseBitset();
    SparseBitset(SparseBitset&& other) noexcept;
    SparseBitset& operator=(SparseBitset&& other) noexcept;
    SparseBitset(const SparseBitset&) = delete;
    SparseBitset& operator=(const SparseBitset&) = delete;

    // Is this edge in the set?
    //
    // Force-inlined, with find_chunk, because it is the engine's hottest predicate -- 15% of a
    // 5-step Wolfram evolution -- and as a mere hint its inlining tracked the size of whatever
    // else lived in the same translation unit (measured with tools/cost_sweep.sh, wolfram-5step:
    // an outlined contains cost 2.15M instructions, +3.7%).
    HG_INLINE bool contains(uint32_t edge_id) const {
        const Chunk* chunk = find_chunk(edge_id >> CHUNK_SHIFT);
        return chunk && chunk->get(edge_id & CHUNK_MASK);
    }

    // Add edge to set. Arena must be provided for entry, node and chunk allocation.
    template<typename Arena>
    void set(uint32_t edge_id, Arena& arena) {
        const uint32_t chunk_id = edge_id >> CHUNK_SHIFT;
        const size_t bit_index = edge_id & CHUNK_MASK;
        if (num_entries_ != kRadix) {
            uint32_t lo;
            if (flat_find(chunk_id, lo)) {
                ChunkEntry& e = entries_[lo];
                if (e.chunk->get(bit_index)) return;
                if (!e.owned) own_entry(e, arena);
                e.chunk->set(bit_index);
            } else {
                insert_chunk(lo, chunk_id, arena)->set(bit_index);
            }
        } else {
            const Chunk* now = find_chunk_radix(chunk_id);
            if (now && now->get(bit_index)) return;
            owned_chunk_radix(chunk_id, arena)->set(bit_index);
        }
        adjust_count(+1);
    }

    // Remove edge from set. A shared chunk (and, in a radix index, the nodes on its path) is
    // copied first. An emptied chunk stays in the index.
    template<typename Arena>
    void clear(uint32_t edge_id, Arena& arena) {
        const uint32_t chunk_id = edge_id >> CHUNK_SHIFT;
        const size_t bit_index = edge_id & CHUNK_MASK;
        if (num_entries_ != kRadix) {
            uint32_t lo;
            if (!flat_find(chunk_id, lo)) return;
            ChunkEntry& e = entries_[lo];
            if (!e.chunk->get(bit_index)) return;
            if (!e.owned) own_entry(e, arena);
            e.chunk->clear(bit_index);
        } else {
            const Chunk* now = find_chunk_radix(chunk_id);
            if (!now || !now->get(bit_index)) return;
            owned_chunk_radix(chunk_id, arena)->clear(bit_index);
        }
        adjust_count(-1);
    }

    // Number of set bits. The lazy cache fill is safe under concurrent const
    // readers: a bitset is only shared between threads once its contents are
    // immutable, so racing fills compute the same total (idempotent). The cached
    // value is stored before the valid flag (release) so a reader that observes
    // valid (acquire) also observes the value.
    size_t count() const;

    bool empty() const;

    // Every set bit, in increasing edge id.
    template<typename F>
    void for_each(F&& f) const {
        for_each_chunk([&](uint32_t chunk_id, const Chunk& c) {
            c.for_each(size_t(chunk_id) << CHUNK_SHIFT, f);
        });
    }

    // Every chunk, in increasing chunk id, as (chunk id, chunk).
    template<typename F>
    void for_each_chunk(F&& f) const {
        if (num_entries_ != kRadix) {
            for (uint32_t i = 0; i < num_entries_; ++i) f(entries_[i].chunk_id, *entries_[i].chunk);
            return;
        }
        walk(root_ & ~uintptr_t{1}, height_, 0, f);
    }

    // A child state's edge set: the parent's, without `consumed`, with `produced`.
    template<typename Arena>
    static SparseBitset derive(
        const SparseBitset& parent,
        const uint32_t* consumed, size_t num_consumed,
        const uint32_t* produced, size_t num_produced,
        Arena& arena
    ) {
        SparseBitset result;
        if (parent.num_entries_ == kRadix) {
            result.num_entries_ = kRadix;
            result.root_ = parent.root_ & ~uintptr_t{1};
            result.height_ = parent.height_;
        } else if (parent.num_entries_ > 0) {
            const uint32_t cap = parent.num_entries_ + 4;
            result.entries_ = arena.template allocate_array<ChunkEntry>(cap);
            result.capacity_ = cap;
            for (uint32_t i = 0; i < parent.num_entries_; ++i) {
                result.entries_[i].chunk = parent.entries_[i].chunk;   // shared
                result.entries_[i].chunk_id = parent.entries_[i].chunk_id;
                result.entries_[i].owned = false;                      // COW on first write
            }
            result.num_entries_ = parent.num_entries_;
        }
        result.count_cached_.store(parent.count_cached_.load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
        result.count_valid_.store(parent.count_valid_.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
        for (size_t i = 0; i < num_consumed; ++i) result.clear(consumed[i], arena);
        for (size_t i = 0; i < num_produced; ++i) result.set(produced[i], arena);
        return result;
    }

    template<typename Arena>
    static SparseBitset from_edges(const uint32_t* edges, size_t num_edges, Arena& arena) {
        SparseBitset result;
        for (size_t i = 0; i < num_edges; ++i) result.set(edges[i], arena);
        return result;
    }

    // Number of chunks (for diagnostics).
    size_t num_chunks() const;

private:
    // The chunk of `chunk_id`, or null.
    HG_INLINE const Chunk* find_chunk(uint32_t chunk_id) const {
        if (num_entries_ == kRadix) return find_chunk_radix(chunk_id);
        uint32_t lo;
        return flat_find(chunk_id, lo) ? entries_[lo].chunk : nullptr;
    }

    // Bisection of the flat index: true with `lo` at the entry of `chunk_id`, else false with
    // `lo` at its insertion point.
    HG_INLINE bool flat_find(uint32_t chunk_id, uint32_t& lo) const {
        lo = 0;
        uint32_t hi = num_entries_;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (entries_[mid].chunk_id < chunk_id) lo = mid + 1; else hi = mid;
        }
        return lo < num_entries_ && entries_[lo].chunk_id == chunk_id;
    }

    HG_NOINLINE const Chunk* find_chunk_radix(uint32_t chunk_id) const;

    // A shared flat entry's chunk, copied and owned.
    template<typename Arena>
    static void own_entry(ChunkEntry& e, Arena& arena) {
        Chunk* copy = arena.template create<Chunk>();
        std::memcpy(copy->words, e.chunk->words, sizeof(copy->words));
        e.chunk = copy;
        e.owned = true;
    }

    // A new chunk for `chunk_id`, absent from the flat index, at insertion point `lo`; a full
    // flat index turns radix first.
    template<typename Arena>
    HG_NOINLINE Chunk* insert_chunk(uint32_t lo, uint32_t chunk_id, Arena& arena) {
        if (num_entries_ >= kFlatChunks) {
            to_radix(arena);
            return owned_chunk_radix(chunk_id, arena);
        }
        if (num_entries_ == capacity_) {
            const uint32_t cap = capacity_ ? capacity_ * 2 : 4;
            ChunkEntry* grown = arena.template allocate_array<ChunkEntry>(cap);
            if (entries_) std::memcpy(grown, entries_, num_entries_ * sizeof(ChunkEntry));
            entries_ = grown;
            capacity_ = cap;
        }
        for (uint32_t i = num_entries_; i > lo; --i) entries_[i] = entries_[i - 1];
        Chunk* fresh = arena.template create<Chunk>();
        entries_[lo] = ChunkEntry{fresh, chunk_id, true};
        ++num_entries_;
        return fresh;
    }

    // The chunk of `chunk_id` in a radix index, owned by this bitset: every shared node and chunk
    // on its path is copied, an absent one created, and the tree raised to cover chunk_id.
    template<typename Arena>
    HG_NOINLINE Chunk* owned_chunk_radix(uint32_t chunk_id, Arena& arena) {
        while ((uint64_t(chunk_id) >> (FANOUT_SHIFT * height_)) != 0) {
            if (!root_) { ++height_; continue; }
            Node* top = arena.template create<Node>();
            top->slot[0] = root_;
            root_ = reinterpret_cast<uintptr_t>(top) | 1;
            ++height_;
        }
        uintptr_t* slot = &root_;
        for (uint32_t h = height_; h > 0; --h) {
            Node* node = owned_node(*slot, arena);
            slot = &node->slot[(chunk_id >> (FANOUT_SHIFT * (h - 1))) & (FANOUT - 1)];
        }
        if (!(*slot & 1)) {
            Chunk* c = arena.template create<Chunk>();
            if (*slot) std::memcpy(c->words, reinterpret_cast<const Chunk*>(*slot)->words, sizeof(c->words));
            *slot = reinterpret_cast<uintptr_t>(c) | 1;
        }
        return reinterpret_cast<Chunk*>(*slot & ~uintptr_t{1});
    }

    // The flat entries moved into a radix tree, each slot tagged as its entry was.
    template<typename Arena>
    HG_NOINLINE void to_radix(Arena& arena) {
        ChunkEntry* flat = entries_;
        const uint32_t n = num_entries_;
        entries_ = nullptr;
        num_entries_ = kRadix;
        capacity_ = 0;
        root_ = 0;
        height_ = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t id = flat[i].chunk_id;
            while ((uint64_t(id) >> (FANOUT_SHIFT * height_)) != 0) {
                if (!root_) { ++height_; continue; }
                Node* top = arena.template create<Node>();
                top->slot[0] = root_;
                root_ = reinterpret_cast<uintptr_t>(top) | 1;
                ++height_;
            }
            uintptr_t* slot = &root_;
            for (uint32_t h = height_; h > 0; --h)
                slot = &owned_node(*slot, arena)->slot[(id >> (FANOUT_SHIFT * (h - 1))) & (FANOUT - 1)];
            *slot = reinterpret_cast<uintptr_t>(flat[i].chunk) | (flat[i].owned ? 1 : 0);
        }
    }

    // The node `slot` points at, made owned: a shared node is copied with its slots' tags
    // cleared, an absent one created.
    template<typename Arena>
    static Node* owned_node(uintptr_t& slot, Arena& arena) {
        if (!(slot & 1)) {
            Node* n = arena.template create<Node>();
            if (slot) {
                const Node* from = reinterpret_cast<const Node*>(slot);
                for (uint32_t i = 0; i < FANOUT; ++i) n->slot[i] = from->slot[i] & ~uintptr_t{1};
            }
            slot = reinterpret_cast<uintptr_t>(n) | 1;
        }
        return reinterpret_cast<Node*>(slot & ~uintptr_t{1});
    }

    template<typename F>
    static void walk(uintptr_t p, uint32_t h, uint32_t base, F& f) {
        if (!p) return;
        if (h == 0) { f(base, *reinterpret_cast<const Chunk*>(p)); return; }
        const Node* n = reinterpret_cast<const Node*>(p);
        for (uint32_t i = 0; i < FANOUT; ++i)
            walk(n->slot[i] & ~uintptr_t{1}, h - 1, (base << FANOUT_SHIFT) | i, f);
    }

    // One bit set (+1) or cleared (-1): the cached count follows while it is valid.
    void adjust_count(int delta) {
        if (count_valid_.load(std::memory_order_relaxed))
            count_cached_.store(count_cached_.load(std::memory_order_relaxed) + size_t(delta),
                                std::memory_order_relaxed);
    }

    union {
        ChunkEntry* entries_;  // flat index
        uintptr_t root_;       // radix: top node (or the chunk, at height 0) | owned tag; 0 when empty
    };
    uint32_t num_entries_;     // flat: entries in use; kRadix: the radix index is in use
    union {
        uint32_t capacity_;    // flat: entries allocated
        uint32_t height_;      // radix: node levels above the chunks
    };
    // Atomic so concurrent const readers can fill the cache without a data race
    // (single-owner mutation paths use relaxed ops; see count()).
    mutable std::atomic<size_t> count_cached_;
    mutable std::atomic<bool> count_valid_;
};

}  // namespace engine
}  // namespace HG_NAMESPACE
