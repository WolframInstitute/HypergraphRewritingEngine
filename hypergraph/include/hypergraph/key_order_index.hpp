#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace HG_NAMESPACE {
namespace engine {

// An order-statistic multiset of (key, value) entries in key order, for keys spread evenly over
// 64 bits (hashes). Entries sit in 2^bits buckets by the key's top bits, each bucket sorted by
// key, and a Fenwick tree over the bucket sizes finds the bucket of a position. With about
// kPerBucket entries per bucket, insert and erase cost O(kPerBucket + bits) and select O(bits).
// Equal keys are adjacent in one bucket.
//
// "MatchesPerStep" keeps one per state (select_positions in parallel_evolution.cpp).
template <class V>
class KeyOrderIndex {
public:
    struct Entry {
        uint64_t key;
        V value;
    };

    KeyOrderIndex() { reset_buckets(kMinBits); }

    uint64_t size() const { return size_; }

    void clear() {
        reset_buckets(kMinBits);
        size_ = 0;
    }

    void insert(uint64_t key, V value) {
        if (size_ + 1 > (uint64_t{kPerBucket} << bits_)) grow();
        const uint32_t b = bucket_of(key);
        std::vector<Entry>& v = buckets_[b];
        size_t i = v.size();
        v.push_back(Entry{key, value});
        while (i > 0 && v[i - 1].key > key) { v[i] = v[i - 1]; --i; }
        v[i] = Entry{key, value};
        add(b, 1);
        ++size_;
    }

    // Removes one entry with `key` for which same(value) holds; false when there is none.
    template <class Same>
    bool erase(uint64_t key, Same&& same) {
        const uint32_t b = bucket_of(key);
        std::vector<Entry>& v = buckets_[b];
        size_t i = lower(v, key);
        for (; i < v.size() && v[i].key == key; ++i) {
            if (!same(v[i].value)) continue;
            v.erase(v.begin() + static_cast<std::ptrdiff_t>(i));
            add(b, -1);
            --size_;
            return true;
        }
        return false;
    }

    // Whether an entry with `key` for which same(value) holds is present.
    template <class Same>
    bool contains(uint64_t key, Same&& same) const {
        const std::vector<Entry>& v = buckets_[bucket_of(key)];
        for (size_t i = lower(v, key); i < v.size() && v[i].key == key; ++i)
            if (same(v[i].value)) return true;
        return false;
    }

    // The entry at position p (0 <= p < size()) in key order, and the positions [*lo, *hi) of the
    // entries whose key equals it.
    const Entry& select(uint64_t p, uint64_t* lo, uint64_t* hi) const {
        uint32_t b = 0;
        uint64_t before = 0;
        for (uint32_t step = 1u << bits_; step != 0; step >>= 1) {
            const uint32_t next = b + step;
            if (next <= (1u << bits_) && before + tree_[next] <= p) {
                b = next;
                before += tree_[next];
            }
        }
        // Bucket b (0-based) holds position p.
        const std::vector<Entry>& v = buckets_[b];
        const size_t at = static_cast<size_t>(p - before);
        size_t l = at, h = at + 1;
        while (l > 0 && v[l - 1].key == v[at].key) --l;
        while (h < v.size() && v[h].key == v[at].key) ++h;
        *lo = before + l;
        *hi = before + h;
        return v[at];
    }

    // Every entry, in key order.
    template <class F>
    void for_each(F&& f) const {
        for (const std::vector<Entry>& v : buckets_)
            for (const Entry& e : v) f(e);
    }

private:
    static constexpr uint32_t kMinBits = 4;
    static constexpr uint32_t kPerBucket = 16;

    uint32_t bucket_of(uint64_t key) const { return static_cast<uint32_t>(key >> (64 - bits_)); }

    static size_t lower(const std::vector<Entry>& v, uint64_t key) {
        size_t lo = 0, hi = v.size();
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (v[mid].key < key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    // Fenwick tree over buckets, 1-based.
    void add(uint32_t b, int64_t d) {
        for (uint32_t i = b + 1; i <= (1u << bits_); i += i & (0u - i))
            tree_[i] = static_cast<uint64_t>(static_cast<int64_t>(tree_[i]) + d);
    }

    void reset_buckets(uint32_t bits) {
        bits_ = bits;
        buckets_.assign(size_t{1} << bits, {});
        tree_.assign((size_t{1} << bits) + 1, 0);
    }

    // Twice the buckets: each bucket splits in two by the next key bit, in order.
    void grow() {
        std::vector<std::vector<Entry>> old;
        old.swap(buckets_);
        reset_buckets(bits_ + 1);
        for (std::vector<Entry>& v : old)
            for (const Entry& e : v) buckets_[bucket_of(e.key)].push_back(e);
        for (uint32_t b = 0; b < (1u << bits_); ++b)
            if (!buckets_[b].empty()) add(b, static_cast<int64_t>(buckets_[b].size()));
    }

    std::vector<std::vector<Entry>> buckets_;
    std::vector<uint64_t> tree_;
    uint32_t bits_ = kMinBits;
    uint64_t size_ = 0;
};

}  // namespace engine
}  // namespace HG_NAMESPACE
