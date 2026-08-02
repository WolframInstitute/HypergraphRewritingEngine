// ConcurrentMap fuzz: randomized concurrent get-or-create against the
// exactly-once contract. The map's failure class is not a data race (every
// access is atomic; TSAN stays quiet) but a LINEARIZABILITY break: two threads
// both told was_inserted for one key, splitting a rendezvous. That surfaces
// only through these assertions, so the fuzz drives the resize machinery as
// hard as possible -- tiny initial capacities so tables chain repeatedly,
// contended key ranges so claims race the seals -- across many seeded shapes.
//
// Deterministic: fixed seed set, so a failure is a reproducer, not an anecdote.
// The double-growth window this hammers was reached by bounded model checking,
// not by fuzzing -- random scheduling almost never lands the interleaving --
// so this gate COMPLEMENTS verification/genmc, it does not replace it.
#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "hypergraph/concurrent_map.hpp"

namespace {

struct FuzzShape {
    uint32_t seed;
    int threads;
    uint64_t keys;         // distinct key universe (1..keys)
    size_t init_capacity;  // tiny => many growths
    int ops_per_thread;
};

void run_shape(const FuzzShape& fs) {
    hypergraph::ConcurrentMap<uint64_t, uint64_t> map(fs.init_capacity);
    std::vector<std::atomic<uint32_t>> wins(fs.keys + 1);
    for (auto& w : wins) w.store(0, std::memory_order_relaxed);
    std::vector<std::atomic<uint64_t>> stored(fs.keys + 1);
    for (auto& v : stored) v.store(0, std::memory_order_relaxed);

    std::vector<std::thread> ts;
    for (int t = 0; t < fs.threads; ++t) {
        ts.emplace_back([&, t] {
            std::mt19937_64 rng(fs.seed * 1000003ull + t);
            // Zipf-ish bias: square a uniform draw so low keys stay hot and
            // contended while the tail still forces growth.
            for (int i = 0; i < fs.ops_per_thread; ++i) {
                double u = std::uniform_real_distribution<double>(0, 1)(rng);
                uint64_t k = 1 + static_cast<uint64_t>(u * u * (fs.keys - 1));
                const uint64_t v = (k << 16) | static_cast<uint64_t>(t + 1);
                auto [got, inserted] = map.insert_if_absent(k, v);
                if (inserted) {
                    wins[k].fetch_add(1, std::memory_order_relaxed);
                    stored[k].store(v, std::memory_order_relaxed);
                } else {
                    // The exchange's answer must denote a value some caller
                    // actually offered for this key.
                    ASSERT_EQ(got >> 16, k) << "seed " << fs.seed;
                }
            }
        });
    }
    for (auto& th : ts) th.join();

    size_t touched = 0;
    for (uint64_t k = 1; k <= fs.keys; ++k) {
        const uint32_t w = wins[k].load(std::memory_order_relaxed);
        auto r = map.lookup(k);
        if (w == 0) continue;  // key never won -- may still exist via a rival's win
        ++touched;
        // EXACTLY-ONCE: one winner per key, ever, across every table generation.
        ASSERT_EQ(w, 1u) << "key " << k << " claimed " << w << " times, seed " << fs.seed;
        ASSERT_TRUE(r.has_value()) << "winner's key unretrievable, seed " << fs.seed;
        ASSERT_EQ(*r, stored[k].load(std::memory_order_relaxed))
            << "lookup disagrees with the winner, seed " << fs.seed;
    }
    ASSERT_EQ(map.count_unique(), [&] {
        size_t n = 0;
        for (uint64_t k = 1; k <= fs.keys; ++k)
            if (map.lookup(k).has_value()) ++n;
        return n;
    }()) << "count_unique disagrees with lookups, seed " << fs.seed;
    ASSERT_GT(touched, 0u);
}

}  // namespace

TEST(ConcurrentMapFuzz, ExactlyOnceUnderGrowthPressure) {
    // Tiny capacities: a 2-slot table with thousands of inserts chains many
    // generations, so claims race installs and seals constantly.
    for (uint32_t seed = 0; seed < 6; ++seed) {
        run_shape({seed, 8, 4000, 2, 4000});
    }
}

TEST(ConcurrentMapFuzz, ExactlyOnceAcrossShapes) {
    const FuzzShape shapes[] = {
        {100, 2, 50, 2, 8000},      // few keys, extreme contention
        {101, 16, 20000, 4, 2000},  // wide fan, many growths
        {102, 4, 500, 64, 6000},    // moderate, mid capacity
        {103, 12, 3, 2, 5000},      // three keys, all threads collide
    };
    for (const auto& fs : shapes) run_shape(fs);
}
