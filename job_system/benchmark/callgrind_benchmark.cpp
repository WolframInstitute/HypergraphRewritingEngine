#include <job_system/job_system.hpp>
#include <cstdio>
#include <atomic>

enum class ProfileJobType { COMPUTE };

// Measurable work function 
__attribute__((noinline))
void actual_work() {
    volatile int result = 0;
    for (int i = 0; i < 1000; ++i) {
        result += i * i;
    }
}

int main() {
    using namespace job_system;
    
    const size_t num_jobs = 500;  // Moderate count for detailed profiling
    const size_t num_threads = 1;  // Single thread to isolate overhead
    
    std::atomic<int> counter{0};
    
    JobSystem<ProfileJobType> js(num_threads);
    js.start();
    
    for (size_t i = 0; i < num_jobs; ++i) {
        auto job = make_job([&counter]() {
            actual_work();  // The actual work we want to measure overhead against
            counter.fetch_add(1);
        }, ProfileJobType::COMPUTE);
        js.submit(std::move(job));
    }
    
    js.wait_for_completion();
    js.shutdown();
    
    printf("Completed %d jobs\n", counter.load());
    return 0;
}