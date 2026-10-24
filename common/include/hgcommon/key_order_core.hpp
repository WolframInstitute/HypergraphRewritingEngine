#pragma once
#include "hgcommon/namespace.hpp"
// AN ORDER-STATISTIC MULTISET OF (key, value) ENTRIES IN KEY ORDER, for keys spread evenly over
// 64 bits (hashes), one body for host and device. Entries sit in 2^bits buckets by the key's top
// bits, each bucket a sorted run, and a Fenwick tree over the bucket sizes finds the bucket of a
// position. With about ko_load entries per bucket, insert and erase cost O(ko_load + bits) and
// select O(bits + ko_load). Equal keys are adjacent in one bucket.
//
// The buckets' storage is the caller's, a Store with
//   uint32_t bits; uint64_t* tree; uint64_t size;      the bucket count 2^bits, the Fenwick tree
//                                                       [2^bits + 1] (1-based), the entries
//   KeyOrderEntry<V>* run(uint32_t b);                  bucket b's sorted run
//   uint32_t count(uint32_t b); void set_count(uint32_t b, uint32_t n);
//   bool reserve(uint32_t b, uint32_t n);               room for n entries in bucket b, or false
//   void tree_add(uint32_t i, int64_t d); void size_add(int64_t d);
//                                                       add to tree node i and to the size
// The host's KeyOrderIndex keeps each bucket in a vector, so a bucket of equal keys grows without
// bound; the device's held match sets keep kHeldBucketCap entries per bucket, and add to the tree
// and the size atomically, so the lanes of a block can edit different buckets at once. ko_needs_grow says when the caller rebuilds into twice the buckets (ko_rebuild).
//
// "MatchesPerStep" orders each state's held matches by it (HeldMatches on the host,
// DeviceHeldSet on the device).

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// Entries per bucket on average before the buckets double, and the smallest bucket count.
constexpr uint32_t ko_load = 16;
constexpr uint32_t ko_min_bits = 4;

template <class V>
struct KeyOrderEntry {
    uint64_t key;
    V value;
};

HG_HD inline uint32_t ko_buckets(uint32_t bits) { return 1u << bits; }

HG_HD inline uint32_t ko_bucket_of(uint64_t key, uint32_t bits) {
    return bits == 0u ? 0u : static_cast<uint32_t>(key >> (64u - bits));
}

// Whether inserting one more entry would pass the average load.
template <class Store>
HG_HD inline bool ko_needs_grow(const Store& s) {
    return s.size + 1u > (uint64_t{ko_load} << s.bits);
}

// Empty buckets and tree for s.bits.
template <class Store>
HG_HD inline void ko_clear(Store& s) {
    const uint32_t nb = ko_buckets(s.bits);
    for (uint32_t b = 0; b < nb; ++b) s.set_count(b, 0u);
    for (uint32_t i = 0; i <= nb; ++i) s.tree[i] = 0u;
    s.size = 0;
}

template <class Store>
HG_HD inline void ko_tree_add(Store& s, uint32_t b, int64_t d) {
    const uint32_t nb = ko_buckets(s.bits);
    for (uint32_t i = b + 1u; i <= nb; i += i & (0u - i))
        s.tree_add(i, d);
}

// The first index in run r of n entries whose key is not below `key`.
template <class E>
HG_HD inline uint32_t ko_lower(const E* r, uint32_t n, uint64_t key) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (r[mid].key < key) lo = mid + 1u;
        else hi = mid;
    }
    return lo;
}

// Inserts (key, value) after the entries with an equal key. False, inserting nothing, when the
// store refuses the bucket room.
template <class Store, class V>
HG_HD inline bool ko_insert(Store& s, uint64_t key, V value) {
    const uint32_t b = ko_bucket_of(key, s.bits);
    const uint32_t n = s.count(b);
    if (!s.reserve(b, n + 1u)) return false;
    auto* r = s.run(b);
    uint32_t i = n;
    while (i > 0u && r[i - 1u].key > key) {
        r[i] = r[i - 1u];
        --i;
    }
    r[i].key = key;
    r[i].value = value;
    s.set_count(b, n + 1u);
    ko_tree_add(s, b, 1);
    s.size_add(1);
    return true;
}

// Removes one entry with `key` for which same(value) holds; false when there is none.
template <class Store, class Same>
HG_HD inline bool ko_erase(Store& s, uint64_t key, Same&& same) {
    const uint32_t b = ko_bucket_of(key, s.bits);
    auto* r = s.run(b);
    const uint32_t n = s.count(b);
    for (uint32_t i = ko_lower(r, n, key); i < n && r[i].key == key; ++i) {
        if (!same(r[i].value)) continue;
        for (uint32_t j = i + 1u; j < n; ++j) r[j - 1u] = r[j];
        s.set_count(b, n - 1u);
        ko_tree_add(s, b, -1);
        s.size_add(-1);
        return true;
    }
    return false;
}

// Whether an entry with `key` for which same(value) holds is present.
template <class Store, class Same>
HG_HD inline bool ko_contains(const Store& s, uint64_t key, Same&& same) {
    const uint32_t b = ko_bucket_of(key, s.bits);
    const auto* r = s.run(b);
    const uint32_t n = s.count(b);
    for (uint32_t i = ko_lower(r, n, key); i < n && r[i].key == key; ++i)
        if (same(r[i].value)) return true;
    return false;
}

// The bucket holding position p (0 <= p < size), and the entries before it.
template <class Store>
HG_HD inline uint32_t ko_bucket_at(const Store& s, uint64_t p, uint64_t* before) {
    const uint32_t nb = ko_buckets(s.bits);
    uint32_t b = 0;
    uint64_t seen = 0;
    for (uint32_t step = nb; step != 0u; step >>= 1) {
        const uint32_t next = b + step;
        if (next <= nb && seen + s.tree[next] <= p) {
            b = next;
            seen += s.tree[next];
        }
    }
    *before = seen;
    return b;
}

// The entry at position p (0 <= p < size) in key order, and the positions [*lo, *hi) of the
// entries whose key equals it.
template <class Store>
HG_HD inline auto ko_select(const Store& s, uint64_t p, uint64_t* lo, uint64_t* hi) {
    uint64_t before = 0;
    const uint32_t b = ko_bucket_at(s, p, &before);
    const auto* r = s.run(b);
    const uint32_t n = s.count(b);
    const uint32_t at = static_cast<uint32_t>(p - before);
    uint32_t l = at, h = at + 1u;
    while (l > 0u && r[l - 1u].key == r[at].key) --l;
    while (h < n && r[h].key == r[at].key) ++h;
    *lo = before + l;
    *hi = before + h;
    return r[at];
}

// Every entry, in key order.
template <class Store, class F>
HG_HD inline void ko_for_each(const Store& s, F&& f) {
    const uint32_t nb = ko_buckets(s.bits);
    for (uint32_t b = 0; b < nb; ++b) {
        const auto* r = s.run(b);
        const uint32_t n = s.count(b);
        for (uint32_t i = 0; i < n; ++i) f(r[i]);
    }
}

// Every entry of `from` into `to`, whose bits are from.bits + 1 and whose buckets and tree are
// cleared here: each bucket splits in two by the next key bit, in order. False when `to` refuses
// a bucket's room.
template <class From, class To>
HG_HD inline bool ko_rebuild(const From& from, To& to) {
    ko_clear(to);
    const uint32_t nb = ko_buckets(from.bits);
    for (uint32_t b = 0; b < nb; ++b) {
        const auto* r = from.run(b);
        const uint32_t n = from.count(b);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t t = ko_bucket_of(r[i].key, to.bits);
            const uint32_t m = to.count(t);
            if (!to.reserve(t, m + 1u)) return false;
            to.run(t)[m] = r[i];
            to.set_count(t, m + 1u);
        }
    }
    const uint32_t nt = ko_buckets(to.bits);
    for (uint32_t t = 0; t < nt; ++t)
        if (to.count(t)) ko_tree_add(to, t, static_cast<int64_t>(to.count(t)));
    to.size = from.size;
    return true;
}

}  // namespace common
}  // namespace HG_NAMESPACE
