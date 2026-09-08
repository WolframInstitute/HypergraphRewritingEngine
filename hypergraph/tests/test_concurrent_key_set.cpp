#include <gtest/gtest.h>

#include "hypergraph/arena.hpp"
#include "hypergraph/concurrent_key_set.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

// The accounting of a sharded set under many workers: every winning insert is counted exactly
// once, and size() agrees with the keys an enumeration walks. The engine reads
// ShardedKeySet::size() as its applied-claim tally (Hypergraph::applied_claims), and the
// determinism tests compare that tally against the enumerated keys; a count kept per worker
// and flushed in batches is only admissible if this passes with workers that take distinct
// arena worker indices, which std::thread workers do on their first arena call.

using hg::engine::ConcurrentHeterogeneousArena;
using hg::engine::ShardedKeySet;

namespace {

// THE KEYS THIS SET IS GIVEN ARE NOT ALL HASHES, AND THE SHARD INDEX HAS TO SURVIVE THAT.
//
// The causal and branchial pair sets key on id_key(a, b) = ((a+1) << 32) | (b+1), which packs
// two dense ids without mixing them. Bits taken straight out of that word are a function of one
// id alone, so an index reading the high half sends every pair of a small run to one shard and
// every worker to one table -- the contention the sharding exists to remove. The mapping must
// spread the ids that actually occur, not the ids a hash would produce.
TEST(ShardedKeySetSharding, PackedIdPairsSpreadAcrossShards) {
    using Set = ShardedKeySet<uint64_t>;
    constexpr size_t kShards = HG_KEY_SET_SHARDS;

    // Every ordered pair over the first 24 event ids: the regime a run of a few dozen events
    // is in, and the one where an unmixed index degenerates completely.
    std::vector<size_t> hits(kShards, 0);
    size_t keys = 0;
    for (uint32_t a = 0; a < 24; ++a) {
        for (uint32_t b = 0; b < 24; ++b) {
            hits[Set::shard_of(hg::common::id_key(a, b))]++;
            ++keys;
        }
    }

    size_t occupied = 0;
    size_t worst = 0;
    for (size_t h : hits) {
        if (h) ++occupied;
        worst = std::max(worst, h);
    }
    EXPECT_EQ(occupied, kShards)
        << "only " << occupied << " of " << kShards << " shards received a key, so workers "
        << "inserting these pairs share tables";
    // A perfectly even split is keys/kShards; allow four times that before calling it skewed.
    EXPECT_LE(worst, 4 * keys / kShards + 1)
        << "one shard took " << worst << " of " << keys << " keys";

    // The mapping is a function of the key alone, which is what keeps two workers racing one
    // key inside one table and so keeps the dedup claim exact.
    for (uint32_t a = 0; a < 24; ++a) {
        EXPECT_EQ(Set::shard_of(hg::common::id_key(a, a)),
                  Set::shard_of(hg::common::id_key(a, a)));
    }
}

TEST(ShardedKeySetAccounting, SizeEqualsEnumeratedKeysUnderManyWorkers) {
    ConcurrentHeterogeneousArena arena;
    ShardedKeySet<uint64_t> set;
    set.set_arena(&arena);
    constexpr int      kThreads = 16;
    constexpr uint64_t kPerThread = 20000;

    std::atomic<int> ready{0};
    std::atomic<int> go{0};
    std::atomic<uint64_t> wins{0};
    std::vector<std::thread> t;
    for (int w = 0; w < kThreads; ++w) {
        t.emplace_back([&, w] {
            // Take a worker index the way an engine worker does: through the arena.
            (void)hg::engine::arena_worker_index();
            ready.fetch_add(1, std::memory_order_release);
            while (go.load(std::memory_order_acquire) == 0) {}
            uint64_t local = 0;
            for (uint64_t i = 0; i < kPerThread; ++i) {
                // Distinct per thread, spread over the shard bits (>> 40), clear of the
                // sentinels 0 and ~0.
                const uint64_t key = (uint64_t(w) << 40) | ((i + 1) << 8) | 1u;
                if (set.insert(key)) ++local;
                // A second insert of a key already present must not count.
                if (set.insert(key)) ++local;
            }
            wins.fetch_add(local, std::memory_order_relaxed);
        });
    }
    while (ready.load(std::memory_order_acquire) < kThreads) {}
    go.store(1, std::memory_order_release);
    for (auto& th : t) th.join();

    const uint64_t expected = uint64_t(kThreads) * kPerThread;
    EXPECT_EQ(wins.load(), expected);
    EXPECT_EQ(set.size(), expected);
    EXPECT_EQ(set.count_enumerated(), expected);
    for (int w = 0; w < kThreads; ++w)
        for (uint64_t i = 0; i < kPerThread; i += 997)
            EXPECT_TRUE(set.contains((uint64_t(w) << 40) | ((i + 1) << 8) | 1u));
}

}  // namespace
