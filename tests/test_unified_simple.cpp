#include <gtest/gtest.h>
#include <lockfree_deque/deque.hpp>
#include <job_system/job_system.hpp>
#include <hypergraph/canonicalization.hpp>
#include <thread>
#include <atomic>
#include "test_helpers.hpp"

enum class TaskType {
    TASK1,
    TASK2,
    TASK3,
    TASK4
};

// === LEVEL 1: CORE DATA STRUCTURE TESTS ===

class LockFreeDequeSimpleTest : public ::testing::Test {
protected:
    lockfree::Deque<int> deque;
};

TEST_F(LockFreeDequeSimpleTest, BasicOperations) {
    EXPECT_TRUE(deque.empty());
    
    deque.push_front(42);
    EXPECT_FALSE(deque.empty());
    
    auto value = deque.try_pop_front();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, 42);
    EXPECT_TRUE(deque.empty());
}

TEST_F(LockFreeDequeSimpleTest, ConcurrentAccess) {
    const int num_items = 1000;
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    std::atomic<bool> done{false};
    
    std::thread producer([&]() {
        for (int i = 0; i < num_items; ++i) {
            deque.push_back(i);
            produced.fetch_add(1);
        }
        done = true;
    });
    
    std::thread consumer([&]() {
        while (!done || !deque.empty()) {
            if (auto item = deque.try_pop_front()) {
                consumed.fetch_add(1);
            }
            std::this_thread::yield();
        }
    });
    
    producer.join();
    consumer.join();
    
    EXPECT_EQ(produced.load(), num_items);
    EXPECT_EQ(consumed.load(), num_items);
    EXPECT_TRUE(deque.empty());
}

// === JOB SYSTEM TESTS (Fixed) ===

class JobSystemSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        job_system = std::make_unique<job_system::JobSystem<TaskType>>(2);
        job_system->start();
    }
    
    void TearDown() override {
        // Job system doesn't have stop() method - just let destructor handle cleanup
        job_system.reset();
    }
    
    std::unique_ptr<job_system::JobSystem<TaskType>> job_system;
};

TEST_F(JobSystemSimpleTest, BasicExecution) {
    std::atomic<int> counter{0};
    
    auto job = job_system::make_job([&counter]() {
        counter.fetch_add(1);
    }, TaskType::TASK1);
    
    job_system->submit(std::move(job));
    job_system->wait_for_completion();
    
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(JobSystemSimpleTest, MultipleJobs) {
    const int num_jobs = 100;
    std::atomic<int> counter{0};
    
    for (int i = 0; i < num_jobs; ++i) {
        job_system->submit_function([&counter]() {
            counter.fetch_add(1);
        }, TaskType::TASK1);
    }
    
    job_system->wait_for_completion();
    EXPECT_EQ(counter.load(), num_jobs);
}

TEST_F(JobSystemSimpleTest, Statistics) {
    const int num_jobs = 50;
    std::atomic<int> counter{0};
    
    for (int i = 0; i < num_jobs; ++i) {
        job_system->submit_function([&counter]() {
            counter.fetch_add(1);
        }, TaskType::TASK1);
    }
    
    job_system->wait_for_completion();
    
    auto stats = job_system->get_statistics();
    EXPECT_GE(stats.total_jobs_executed, static_cast<size_t>(num_jobs));
    EXPECT_EQ(stats.total_jobs_stolen, 0);  // Current implementation doesn't steal
    EXPECT_EQ(stats.total_jobs_deferred, 0);
}

// === CANONICALIZATION TESTS (Fixed) ===

class CanonicalizationSimpleTest : public ::testing::Test {
protected:
    hypergraph::Canonicalizer canonicalizer;
};

TEST_F(CanonicalizationSimpleTest, EmptyHypergraph) {
    hypergraph::Hypergraph empty_hg;
    auto result = canonicalizer.canonicalize(empty_hg);
    
    // Check the structure exists
    EXPECT_TRUE(result.canonical_form.edges.empty());
    EXPECT_EQ(result.canonical_form.vertex_count, 0);
}

TEST_F(CanonicalizationSimpleTest, SingleEdge) {
    auto hg = test_utils::create_test_hypergraph({{1, 2}});
    auto result = canonicalizer.canonicalize(hg);
    
    EXPECT_FALSE(result.canonical_form.edges.empty());
    EXPECT_EQ(result.canonical_form.vertex_count, 2);
}

TEST_F(CanonicalizationSimpleTest, VertexRelabeling) {
    // Same structure, different vertex labels should canonicalize to same form
    auto hg1 = test_utils::create_test_hypergraph({{10, 20}, {10, 30}});
    auto hg2 = test_utils::create_test_hypergraph({{100, 200}, {100, 300}});
    
    test_utils::expect_canonical_equal(hg1, hg2);
}

TEST_F(CanonicalizationSimpleTest, DifferentStructures) {
    // Different connectivity patterns should canonicalize differently
    auto hg1 = test_utils::create_test_hypergraph({{1, 2}, {3, 4}});  // Disconnected
    auto hg2 = test_utils::create_test_hypergraph({{1, 2}, {2, 3}});  // Connected
    
    test_utils::expect_canonical_different(hg1, hg2);
}

// === INTEGRATION TEST ===

TEST(IntegrationTest, JobSystemWithLockFreeDeque) {
    // Test job system working with lock-free data structures
    lockfree::Deque<int> shared_deque;
    job_system::JobSystem<TaskType> js(2);
    js.start();
    
    const int num_jobs = 100;
    std::atomic<int> jobs_completed{0};
    
    // Submit jobs that push to shared deque
    for (int i = 0; i < num_jobs; ++i) {
        js.submit_function([&shared_deque, &jobs_completed, i]() {
            shared_deque.push_back(i);
            jobs_completed.fetch_add(1);
        }, TaskType::TASK1);
    }
    
    js.wait_for_completion();
    
    // Small delay to ensure all atomic operations complete
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    EXPECT_EQ(jobs_completed.load(), num_jobs);
    
    // Verify all items were added
    int items_found = 0;
    while (shared_deque.try_pop_front().has_value()) {
        items_found++;
    }
    EXPECT_EQ(items_found, num_jobs);
}