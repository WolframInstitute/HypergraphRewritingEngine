#include <gtest/gtest.h>
#include <lockfree_deque/deque.hpp>
#include <thread>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <atomic>
#include <set>

class LockFreeDequeTest : public ::testing::Test {
protected:
    lockfree::Deque<int> deque;
};

TEST_F(LockFreeDequeTest, InitiallyEmpty) {
    EXPECT_TRUE(deque.empty());
}

TEST_F(LockFreeDequeTest, PushFrontPopFront) {
    deque.push_front(42);
    EXPECT_FALSE(deque.empty());
    
    auto value = deque.try_pop_front();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, 42);
    EXPECT_TRUE(deque.empty());
}

TEST_F(LockFreeDequeTest, PushBackPopBack) {
    deque.push_back(99);
    EXPECT_FALSE(deque.empty());
    
    auto value = deque.try_pop_back();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, 99);
    EXPECT_TRUE(deque.empty());
}

TEST_F(LockFreeDequeTest, PushFrontPopBack) {
    deque.push_front(123);
    EXPECT_FALSE(deque.empty());
    
    auto value = deque.try_pop_back();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, 123);
    EXPECT_TRUE(deque.empty());
}

TEST_F(LockFreeDequeTest, PushBackPopFront) {
    deque.push_back(456);
    EXPECT_FALSE(deque.empty());
    
    auto value = deque.try_pop_front();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, 456);
    EXPECT_TRUE(deque.empty());
}

TEST_F(LockFreeDequeTest, PopFromEmpty) {
    EXPECT_FALSE(deque.try_pop_front().has_value());
    EXPECT_FALSE(deque.try_pop_back().has_value());
}

TEST_F(LockFreeDequeTest, MultipleElements) {
    deque.push_front(1);
    deque.push_front(2);
    deque.push_back(3);
    deque.push_back(4);
    
    auto v1 = deque.try_pop_front();
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(*v1, 2);
    
    auto v2 = deque.try_pop_back();
    ASSERT_TRUE(v2.has_value());
    EXPECT_EQ(*v2, 4);
    
    auto v3 = deque.try_pop_front();
    ASSERT_TRUE(v3.has_value());
    EXPECT_EQ(*v3, 1);
    
    auto v4 = deque.try_pop_back();
    ASSERT_TRUE(v4.has_value());
    EXPECT_EQ(*v4, 3);
    
    EXPECT_TRUE(deque.empty());
}

TEST_F(LockFreeDequeTest, FIFOOrder) {
    const int count = 10;
    for (int i = 0; i < count; ++i) {
        deque.push_back(i);
    }
    
    for (int i = 0; i < count; ++i) {
        auto value = deque.try_pop_front();
        ASSERT_TRUE(value.has_value());
        EXPECT_EQ(*value, i);
    }
    
    EXPECT_TRUE(deque.empty());
}

TEST_F(LockFreeDequeTest, LIFOOrder) {
    const int count = 10;
    for (int i = 0; i < count; ++i) {
        deque.push_front(i);
    }
    
    for (int i = count - 1; i >= 0; --i) {
        auto value = deque.try_pop_front();
        ASSERT_TRUE(value.has_value());
        EXPECT_EQ(*value, i);
    }
    
    EXPECT_TRUE(deque.empty());
}

TEST(LockFreeDequeMultiThreaded, ConcurrentPushFront) {
    lockfree::Deque<int> deque;
    const int num_threads = 4;
    const int items_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&deque, t, items_per_thread]() {
            for (int i = 0; i < items_per_thread; ++i) {
                deque.push_front(t * items_per_thread + i);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::set<int> seen;
    while (auto value = deque.try_pop_front()) {
        seen.insert(*value);
    }
    
    EXPECT_EQ(seen.size(), num_threads * items_per_thread);
}

TEST(LockFreeDequeMultiThreaded, ConcurrentPushBack) {
    lockfree::Deque<int> deque;
    const int num_threads = 4;
    const int items_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&deque, t, items_per_thread]() {
            for (int i = 0; i < items_per_thread; ++i) {
                deque.push_back(t * items_per_thread + i);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::set<int> seen;
    while (auto value = deque.try_pop_back()) {
        seen.insert(*value);
    }
    
    EXPECT_EQ(seen.size(), num_threads * items_per_thread);
}

TEST(LockFreeDequeMultiThreaded, ProducerConsumer) {
    lockfree::Deque<int> deque;
    const int num_items = 100;
    
    std::thread producer([&deque, num_items]() {
        for (int i = 0; i < num_items; ++i) {
            if (i % 2 == 0) {
                deque.push_front(i);
            } else {
                deque.push_back(i);
            }
        }
    });
    
    std::thread consumer([&deque, num_items]() {
        int consumed = 0;
        int retries = 0;
        const int max_retries = 1000000;
        
        while (consumed < num_items && retries < max_retries) {
            if (auto value = deque.try_pop_front()) {
                consumed++;
                retries = 0;
            } else if (auto value = deque.try_pop_back()) {
                consumed++;
                retries = 0;
            } else {
                retries++;
                std::this_thread::yield();
            }
        }
        
        EXPECT_EQ(consumed, num_items);
    });
    
    producer.join();
    consumer.join();
    
    EXPECT_TRUE(deque.empty());
}

TEST(LockFreeDequeMultiThreaded, MixedOperations) {
    lockfree::Deque<int> deque(8192); // Larger capacity for stress test
    const int num_threads = 8;
    const int operations_per_thread = 50;
    std::atomic<int> total_pushed{0};
    std::atomic<int> total_popped{0};
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&deque, &total_pushed, &total_popped, t, operations_per_thread]() {
            std::mt19937 gen(t);
            std::uniform_int_distribution<> op_dist(0, 3);
            
            for (int i = 0; i < operations_per_thread; ++i) {
                int op = op_dist(gen);
                switch (op) {
                    case 0:
                        deque.push_front(i);
                        total_pushed.fetch_add(1);
                        break;
                    case 1:
                        deque.push_back(i);
                        total_pushed.fetch_add(1);
                        break;
                    case 2:
                        if (deque.try_pop_front()) {
                            total_popped.fetch_add(1);
                        }
                        break;
                    case 3:
                        if (deque.try_pop_back()) {
                            total_popped.fetch_add(1);
                        }
                        break;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int remaining = 0;
    while (deque.try_pop_front()) {
        remaining++;
    }
    
    EXPECT_EQ(total_pushed.load() - total_popped.load(), remaining);
}

// High-contention stress with a strict conservation invariant (zero tolerance):
// after draining, what remains must equal pushed minus popped, with nothing lost or
// duplicated. Run across small/wrapping and large capacities.
TEST(LockFreeDequeStress, HighContentionConserves) {
    for (std::size_t cap : {std::size_t(4), std::size_t(64), std::size_t(8192)}) {
        lockfree::Deque<int> deque(cap);
        const int num_threads = 8;
        const int duration_ms = 50;
        std::atomic<bool> stop{false};
        std::atomic<long> pushed{0};
        std::atomic<long> popped{0};

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&deque, &stop, &pushed, &popped, t]() {
                std::mt19937 gen(t + 1);
                std::uniform_int_distribution<> op_dist(0, 3);
                while (!stop.load(std::memory_order_relaxed)) {
                    switch (op_dist(gen)) {
                        case 0: if (deque.try_push_front(t)) pushed.fetch_add(1); break;
                        case 1: if (deque.try_push_back(t)) pushed.fetch_add(1); break;
                        case 2: if (deque.try_pop_front()) popped.fetch_add(1); break;
                        case 3: if (deque.try_pop_back()) popped.fetch_add(1); break;
                    }
                }
            });
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
        stop.store(true);
        for (auto& thread : threads) thread.join();

        long remaining = 0;
        while (deque.try_pop_front()) remaining++;
        EXPECT_EQ(pushed.load() - popped.load(), remaining);
        EXPECT_TRUE(deque.empty());
    }
}

// Identity-tracking MPMC fuzz: every produced id must be popped exactly once -- no
// loss, no duplication -- across small (wrapping) and large capacities. A watchdog
// guards against a hang if a regression loses items.
TEST(LockFreeDequeFuzz, MPMCNoLossNoDuplicate) {
    for (std::size_t cap : {std::size_t(4), std::size_t(16), std::size_t(1024)}) {
        const int P = 6, C = 6, K = 4000;
        const int total = P * K;
        lockfree::Deque<int> deque(cap);
        std::vector<std::atomic<int>> seen(total);
        for (auto& s : seen) s.store(0, std::memory_order_relaxed);
        std::atomic<int> pushed{0}, popped{0};
        std::atomic<int> duplicates{0}, out_of_range{0}, stuck{0};

        std::vector<std::thread> threads;
        // producers: each owns a disjoint id range, pushed randomly to either end
        for (int p = 0; p < P; ++p) {
            threads.emplace_back([&, p]() {
                std::mt19937 gen(p + 1);
                for (int i = 0; i < K; ++i) {
                    int id = p * K + i;
                    while (!((gen() & 1) ? deque.try_push_back(id) : deque.try_push_front(id))) {
                        std::this_thread::yield();
                    }
                    pushed.fetch_add(1);
                }
            });
        }
        // consumers: pop from either end until everything is drained
        for (int c = 0; c < C; ++c) {
            threads.emplace_back([&, c]() {
                std::mt19937 gen(1000 + c);
                long idle = 0;
                while (popped.load(std::memory_order_acquire) < total) {
                    auto v = (gen() & 1) ? deque.try_pop_front() : deque.try_pop_back();
                    if (v) {
                        idle = 0;
                        int id = *v;
                        if (id < 0 || id >= total) { out_of_range.fetch_add(1); continue; }
                        if (seen[id].fetch_add(1) != 0) duplicates.fetch_add(1);
                        popped.fetch_add(1);
                    } else if (++idle > 200000000L) {
                        stuck.fetch_add(1);
                        break;  // watchdog: a correct deque never starves here
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }
        for (auto& thread : threads) thread.join();

        EXPECT_EQ(stuck.load(), 0) << "consumer starved at capacity " << cap;
        EXPECT_EQ(pushed.load(), total);
        EXPECT_EQ(popped.load(), total);
        EXPECT_EQ(duplicates.load(), 0);
        EXPECT_EQ(out_of_range.load(), 0);
        EXPECT_TRUE(deque.empty());
        int missing = 0;
        for (int i = 0; i < total; ++i) if (seen[i].load() != 1) ++missing;
        EXPECT_EQ(missing, 0) << "items lost at capacity " << cap;
    }
}

class LockFreeDequeWithStrings : public ::testing::Test {
protected:
    lockfree::Deque<std::string> deque;
};

TEST_F(LockFreeDequeWithStrings, StringOperations) {
    deque.push_front("hello");
    deque.push_back("world");
    deque.push_front("foo");
    deque.push_back("bar");
    
    auto v1 = deque.try_pop_front();
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(*v1, "foo");
    
    auto v2 = deque.try_pop_back();
    ASSERT_TRUE(v2.has_value());
    EXPECT_EQ(*v2, "bar");
    
    auto v3 = deque.try_pop_front();
    ASSERT_TRUE(v3.has_value());
    EXPECT_EQ(*v3, "hello");
    
    auto v4 = deque.try_pop_back();
    ASSERT_TRUE(v4.has_value());
    EXPECT_EQ(*v4, "world");
    
    EXPECT_TRUE(deque.empty());
}

struct ComplexObject {
    int id;
    std::string name;
    std::vector<int> data;
    
    ComplexObject(int i, std::string n, std::vector<int> d)
        : id(i), name(std::move(n)), data(std::move(d)) {}
    
    bool operator==(const ComplexObject& other) const {
        return id == other.id && name == other.name && data == other.data;
    }
};

TEST(LockFreeDequeComplex, ComplexObjectHandling) {
    lockfree::Deque<ComplexObject> deque;
    
    ComplexObject obj1(1, "first", {1, 2, 3});
    ComplexObject obj2(2, "second", {4, 5, 6});
    
    deque.push_front(obj1);
    deque.push_back(obj2);
    
    auto retrieved1 = deque.try_pop_front();
    ASSERT_TRUE(retrieved1.has_value());
    EXPECT_EQ(*retrieved1, obj1);
    
    auto retrieved2 = deque.try_pop_back();
    ASSERT_TRUE(retrieved2.has_value());
    EXPECT_EQ(*retrieved2, obj2);
    
    EXPECT_TRUE(deque.empty());
}