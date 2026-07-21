// Direct contention tests for the CPU ConcurrentMap -- the core lock-free dedup
// structure on the hot path (canonical-state map, event dedup). It was previously
// exercised only indirectly through evolution; its GPU analog has a dedicated test
// but the CPU one had none. These assert first-writer-wins, no-loss, no-duplicate
// insertion, and correctness across resizes under many-thread contention.
#include <gtest/gtest.h>
#include <hypergraph/concurrent_map.hpp>

#include <atomic>
#include <thread>
#include <vector>

using namespace hypergraph;
using Map = ConcurrentMap<uint64_t, uint32_t>;  // keys avoid 0 (EMPTY) and ~0 (LOCKED)

static unsigned threads() { return std::max(4u, std::thread::hardware_concurrency()); }

TEST(ConcurrentMapTest, SingleThreadedBasics) {
    Map m;
    auto [v1, ins1] = m.insert_if_absent(1, 100u);
    EXPECT_TRUE(ins1);
    EXPECT_EQ(v1, 100u);
    EXPECT_EQ(m.lookup(1).value_or(0u), 100u);

    auto [v2, ins2] = m.insert_if_absent(1, 200u);  // duplicate key
    EXPECT_FALSE(ins2);
    EXPECT_EQ(v2, 100u);                            // first writer wins
    EXPECT_EQ(m.lookup(1).value_or(0u), 100u);

    EXPECT_FALSE(m.lookup(999).has_value());
    EXPECT_EQ(m.count_unique(), 1u);
}

TEST(ConcurrentMapTest, ConcurrentDistinctKeys) {
    Map m;
    const unsigned T = threads();
    const uint64_t N = 5000;
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < T; ++t)
        ts.emplace_back([&, t] {
            for (uint64_t i = 0; i < N; ++i) {
                uint64_t k = static_cast<uint64_t>(t) * N + i + 1;  // disjoint, != 0
                m.insert_if_absent_waiting(k, static_cast<uint32_t>(k));
            }
        });
    for (auto& th : ts) th.join();

    EXPECT_EQ(m.count_unique(), static_cast<size_t>(T) * N);
    for (unsigned t = 0; t < T; ++t)
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t k = static_cast<uint64_t>(t) * N + i + 1;
            auto v = m.lookup(k);
            ASSERT_TRUE(v.has_value()) << "missing key " << k;
            EXPECT_EQ(*v, static_cast<uint32_t>(k));
        }
}

TEST(ConcurrentMapTest, FirstWriterWinsNoDuplicateInserts) {
    Map m;
    const unsigned T = threads();
    const uint64_t K = 4000;
    std::atomic<uint64_t> totalInserted{0};
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < T; ++t)
        ts.emplace_back([&, t] {
            uint64_t local = 0;
            for (uint64_t k = 1; k <= K; ++k) {  // every thread races on the same keys
                auto [v, ins] = m.insert_if_absent_waiting(k, t + 1u);
                if (ins) ++local;
            }
            totalInserted += local;
        });
    for (auto& th : ts) th.join();

    // Each key is inserted exactly once across all threads: no key double-inserted.
    EXPECT_EQ(totalInserted.load(), K);
    EXPECT_EQ(m.count_unique(), static_cast<size_t>(K));
    for (uint64_t k = 1; k <= K; ++k) {
        auto v = m.lookup(k);
        ASSERT_TRUE(v.has_value()) << "missing key " << k;
        EXPECT_GE(*v, 1u);
        EXPECT_LE(*v, T);  // whichever thread won stored its own value
    }
}

TEST(ConcurrentMapTest, ResizeUnderContention) {
    Map m(16);  // tiny initial capacity forces repeated resizes under load
    const unsigned T = threads();
    const uint64_t N = 3000;
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < T; ++t)
        ts.emplace_back([&, t] {
            for (uint64_t i = 0; i < N; ++i) {
                uint64_t k = static_cast<uint64_t>(t) * N + i + 1;
                m.insert_if_absent_waiting(k, static_cast<uint32_t>(k));
            }
        });
    for (auto& th : ts) th.join();

    EXPECT_EQ(m.count_unique(), static_cast<size_t>(T) * N);
    for (unsigned t = 0; t < T; ++t)
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t k = static_cast<uint64_t>(t) * N + i + 1;
            ASSERT_TRUE(m.lookup(k).has_value()) << "missing after resize: " << k;
        }
}
