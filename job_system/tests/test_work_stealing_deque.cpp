#include <gtest/gtest.h>
#include <job_system/work_stealing_deque.hpp>
#include <atomic>
#include <thread>
#include <vector>

using job_system::WorkStealingDeque;

TEST(WorkStealingDeque, SingleThreadOwner) {
    WorkStealingDeque<int*> dq(16);
    EXPECT_TRUE(dq.empty());
    for (int i = 0; i < 8; ++i) ASSERT_TRUE(dq.push(new int(i)));
    // Owner pops LIFO.
    for (int i = 7; i >= 0; --i) {
        int* x = dq.pop();
        ASSERT_NE(x, nullptr);
        EXPECT_EQ(*x, i);
        delete x;
    }
    EXPECT_EQ(dq.pop(), nullptr);
    EXPECT_TRUE(dq.empty());
}

TEST(WorkStealingDeque, FullReturnsFalse) {
    WorkStealingDeque<int*> dq(2);  // rounds to capacity 2
    ASSERT_TRUE(dq.push(new int(1)));
    ASSERT_TRUE(dq.push(new int(2)));
    int* overflow = new int(3);
    EXPECT_FALSE(dq.push(overflow));  // full
    delete overflow;
    delete dq.pop();
    delete dq.pop();
}

// One owner pushes/pops the bottom while several thieves steal the top. Every id must
// be taken exactly once: no loss, no duplication. Run across small (heavy last-element
// race) and large capacities.
TEST(WorkStealingDeque, OwnerThievesConservation) {
    for (std::size_t cap : {std::size_t(2), std::size_t(16), std::size_t(1024)}) {
        const int N = 50000;
        const int num_thieves = 4;
        WorkStealingDeque<int*> dq(cap);
        std::vector<std::atomic<int>> seen(N);
        for (auto& s : seen) s.store(0, std::memory_order_relaxed);
        std::atomic<int> recorded{0}, duplicates{0}, out_of_range{0};

        auto record = [&](int id) {
            if (id < 0 || id >= N) { out_of_range.fetch_add(1); return; }
            if (seen[id].fetch_add(1) != 0) duplicates.fetch_add(1);
            else recorded.fetch_add(1);
        };

        std::vector<std::thread> thieves;
        for (int c = 0; c < num_thieves; ++c) {
            thieves.emplace_back([&]() {
                while (recorded.load(std::memory_order_acquire) < N) {
                    int* x = dq.steal();
                    if (x) { record(*x); delete x; }
                    else std::this_thread::yield();
                }
            });
        }

        // Owner: push every id, popping to make room when full; then help drain.
        for (int id = 0; id < N; ++id) {
            int* x = new int(id);
            while (!dq.push(x)) {
                int* y = dq.pop();
                if (y) { record(*y); delete y; }
                else std::this_thread::yield();
            }
        }
        while (recorded.load(std::memory_order_acquire) < N) {
            int* y = dq.pop();
            if (y) { record(*y); delete y; }
            else std::this_thread::yield();
        }

        for (auto& th : thieves) th.join();

        EXPECT_EQ(recorded.load(), N) << "capacity " << cap;
        EXPECT_EQ(duplicates.load(), 0) << "capacity " << cap;
        EXPECT_EQ(out_of_range.load(), 0) << "capacity " << cap;
        int missing = 0;
        for (int i = 0; i < N; ++i) if (seen[i].load() != 1) ++missing;
        EXPECT_EQ(missing, 0) << "capacity " << cap;
        EXPECT_TRUE(dq.empty());
    }
}
