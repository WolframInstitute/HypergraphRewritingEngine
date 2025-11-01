// BENCHMARK_CATEGORY: Job System

#include "benchmark_framework.hpp"
#include <job_system/job_system.hpp>
#include <thread>
#include <atomic>
#include <random>

using namespace benchmark;
using namespace job_system;

BENCHMARK(job_system_2d_sweep, "2D parameter sweep of job system across thread count and batch size for parallel scalability analysis") {
    size_t max_threads = std::thread::hardware_concurrency();
    std::vector<size_t> thread_counts;
    for (size_t i = 1; i <= max_threads; ++i) {
        thread_counts.push_back(i);
    }

    std::vector<int> batch_sizes = {10, 50, 100, 500, 1000, 5000, 10000};

    for (size_t num_threads : thread_counts) {
        for (int batch_size : batch_sizes) {
            BENCHMARK_PARAM("num_threads", num_threads);
            BENCHMARK_PARAM("batch_size", batch_size);

            JobSystem<std::function<void()>> js(num_threads);
            js.start();

            BENCHMARK_CODE([&]() {
                std::atomic<int> counter{0};

                BENCHMARK_TIMING_START("submission");
                // Create batch of jobs with sleep workload
                for (int i = 0; i < batch_size; ++i) {
                    js.submit_function([&counter]() {
                        // Fixed sleep for 1ms with small random variance
                        static thread_local std::mt19937 gen(std::random_device{}());
                        static thread_local std::normal_distribution<> dist(50.0, 5.0);
                        int sleep_us = static_cast<int>(dist(gen));
                        if (sleep_us > 0) {
                            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
                        }
                        counter.fetch_add(1, std::memory_order_relaxed);
                    }, std::function<void()>());
                }
                BENCHMARK_TIMING_STOP("submission");

                BENCHMARK_TIMING_START("execution");
                // Wait for all jobs to complete
                js.wait_for_completion();
                BENCHMARK_TIMING_STOP("execution");
            });

            js.shutdown();
        }
    }
}

BENCHMARK(job_system_overhead, "Measures job system overhead with minimal workload across varying batch sizes") {
    size_t num_threads = std::thread::hardware_concurrency();

    for (int batch_size : {10, 100, 1000, 10000}) {
        BENCHMARK_PARAM("batch_size", batch_size);
        BENCHMARK_META("x_scale", "log");

        JobSystem<std::function<void()>> js(num_threads);
        js.start();

        BENCHMARK_CODE([&]() {
            std::atomic<int> counter{0};

            BENCHMARK_TIMING_START("submission");
            // Minimal workload - just increment counter
            for (int i = 0; i < batch_size; ++i) {
                js.submit_function([&counter]() {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }, std::function<void()>());
            }
            BENCHMARK_TIMING_STOP("submission");

            BENCHMARK_TIMING_START("execution");
            js.wait_for_completion();
            BENCHMARK_TIMING_STOP("execution");
        });

        js.shutdown();
    }
}

BENCHMARK(job_system_scaling_efficiency, "Evaluates parallel efficiency with fixed total work across different thread counts") {
    size_t max_threads = std::thread::hardware_concurrency();
    std::vector<size_t> thread_counts;
    for (size_t i = 1; i <= max_threads; ++i) {
        thread_counts.push_back(i);
    }

    constexpr int total_work = 10000;  // Fixed amount of work

    for (size_t num_threads : thread_counts) {
        BENCHMARK_PARAM("num_threads", num_threads);

        JobSystem<std::function<void()>> js(num_threads);
        js.start();

        BENCHMARK_CODE([&]() {
            std::atomic<int> counter{0};

            BENCHMARK_TIMING_START("submission");
            for (int i = 0; i < total_work; ++i) {
                js.submit_function([&counter]() {
                    // Fixed sleep for 1ms with small random variance
                    static thread_local std::mt19937 gen(std::random_device{}());
                    static thread_local std::normal_distribution<> dist(50.0, 5.0);
                    int sleep_us = static_cast<int>(dist(gen));
                    if (sleep_us > 0) {
                        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
                    }
                    counter.fetch_add(1, std::memory_order_relaxed);
                }, std::function<void()>());
            }
            BENCHMARK_TIMING_STOP("submission");

            BENCHMARK_TIMING_START("execution");
            js.wait_for_completion();
            BENCHMARK_TIMING_STOP("execution");
        });

        js.shutdown();
    }
}
