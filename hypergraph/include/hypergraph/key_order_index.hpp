#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include "hgcommon/key_order_core.hpp"

namespace HG_NAMESPACE {
namespace engine {

// The host's order-statistic multiset of (key, value) entries in key order
// (hgcommon/key_order_core.hpp), each bucket a vector, so a bucket of equal keys grows without
// bound.
//
// "MatchesPerStep" keeps one per state (select_positions in parallel_evolution.cpp).
template <class V>
class KeyOrderIndex {
public:
    using Entry = hgcommon::KeyOrderEntry<V>;

    KeyOrderIndex() { reset(hgcommon::ko_min_bits); }
    KeyOrderIndex(const KeyOrderIndex& o) : s_(o.s_) { s_.tree = s_.tree_storage.data(); }
    KeyOrderIndex& operator=(const KeyOrderIndex& o) {
        s_ = o.s_;
        s_.tree = s_.tree_storage.data();
        return *this;
    }

    uint64_t size() const { return s_.size; }

    void clear() { reset(hgcommon::ko_min_bits); }

    void insert(uint64_t key, V value) {
        if (hgcommon::ko_needs_grow(s_)) grow();
        hgcommon::ko_insert(s_, key, value);
    }

    // Removes one entry with `key` for which same(value) holds; false when there is none.
    template <class Same>
    bool erase(uint64_t key, Same&& same) {
        return hgcommon::ko_erase(s_, key, std::forward<Same>(same));
    }

    // Whether an entry with `key` for which same(value) holds is present.
    template <class Same>
    bool contains(uint64_t key, Same&& same) const {
        return hgcommon::ko_contains(s_, key, std::forward<Same>(same));
    }

    // The entry at position p (0 <= p < size()) in key order, and the positions [*lo, *hi) of the
    // entries whose key equals it.
    Entry select(uint64_t p, uint64_t* lo, uint64_t* hi) const {
        return hgcommon::ko_select(s_, p, lo, hi);
    }

    // Every entry, in key order.
    template <class F>
    void for_each(F&& f) const {
        hgcommon::ko_for_each(s_, std::forward<F>(f));
    }

private:
    struct Store {
        uint32_t bits = 0;
        uint64_t* tree = nullptr;
        uint64_t size = 0;
        std::vector<std::vector<Entry>> buckets;
        std::vector<uint64_t> tree_storage;

        Entry* run(uint32_t b) { return buckets[b].data(); }
        const Entry* run(uint32_t b) const { return buckets[b].data(); }
        uint32_t count(uint32_t b) const { return static_cast<uint32_t>(buckets[b].size()); }
        void set_count(uint32_t b, uint32_t n) { buckets[b].resize(n); }
        bool reserve(uint32_t b, uint32_t n) {
            std::vector<Entry>& v = buckets[b];
            while (v.size() < n) v.push_back(Entry{0u, V{}});
            return true;
        }
        void tree_add(uint32_t i, int64_t d) {
            tree[i] = static_cast<uint64_t>(static_cast<int64_t>(tree[i]) + d);
        }
        void size_add(int64_t d) { size = static_cast<uint64_t>(static_cast<int64_t>(size) + d); }
    };

    void reset(uint32_t bits) {
        s_.bits = bits;
        s_.buckets.assign(size_t{1} << bits, {});
        s_.tree_storage.assign((size_t{1} << bits) + 1, 0);
        s_.tree = s_.tree_storage.data();
        s_.size = 0;
    }

    // Twice the buckets.
    void grow() {
        Store next;
        next.bits = s_.bits + 1;
        next.buckets.assign(size_t{1} << next.bits, {});
        next.tree_storage.assign((size_t{1} << next.bits) + 1, 0);
        next.tree = next.tree_storage.data();
        hgcommon::ko_rebuild(s_, next);
        s_ = std::move(next);
        s_.tree = s_.tree_storage.data();
    }

    Store s_;
};

}  // namespace engine
}  // namespace HG_NAMESPACE
