// How much headroom does the deque's 32-bit ABA tag actually have?
//
// WHY THIS EXISTS. deque.hpp packs {tag:32, head:16, tail:16} into one 64-bit word and increments
// the tag on every successful operation, so a compare-exchange commits only when nothing changed
// since the load. That defeats ABA -- but only until the tag wraps. The header stated the defence
// without stating the bound, and a bound nobody has measured is a bound nobody can check.
//
// THE HAZARD, precisely. A popper loads the packed word, then loads the item pointer out of the
// slot, then compare-exchanges the word. If between its load and its exchange the tag completes a
// full 2^32 cycle AND head/tail return to the values it read, the exchange succeeds against a
// state that is coincidentally identical, and the popper returns an item pointer whose slot has
// since been drained and refilled -- the same item handed to two consumers.
//
// So the question is not "can the tag wrap" (over a long run it certainly can) but "can 2^32
// successful operations complete inside ONE thread's load-to-exchange window". That window is a
// handful of instructions, so it takes a deschedule, and this tool measures how long a deschedule
// would have to be.
//
// WHY THE TAG CANNOT SIMPLY BE WIDENED. The three fields have to share one 64-bit word because a
// 128-bit compare-exchange is not lock-free on the targets this engine supports. The indices need
// 16 bits each: the injector queue is constructed at 32768 entries, and `tail - head` is computed
// in 16-bit arithmetic so the difference is unambiguous. 64 - 16 - 16 = 32 is what remains.
//
// Usage: deque_aba_headroom [seconds_per_config]

#include "lockfree_deque/deque.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

// Pointer payloads live inline in their slot, so this measures the index algorithm rather than
// the allocator -- which is also how the job system uses it (WorkStealingDeque requires a
// pointer T, and JobRaw is a pointer).
using Q = lockfree::Deque<int*>;

int g_items[64];

struct Result {
    unsigned threads;
    uint64_t ops;
    double seconds;
};

// Every thread pushes and pops, so the queue neither drains nor fills and the measurement is of
// contention on the packed word rather than of the empty or full early-outs.
Result run(unsigned threads, double seconds, size_t capacity) {
    Q q(capacity);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> total{0};
    std::vector<std::thread> pool;

    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            uint64_t local = 0;
            int* mine = &g_items[t % 64];
            while (!stop.load(std::memory_order_relaxed)) {
                // Each iteration that succeeds bumps the tag once per successful operation.
                if (q.try_push_back(mine)) ++local;
                if (q.try_pop_front()) ++local;
                if (q.try_push_front(mine)) ++local;
                if (q.try_pop_back()) ++local;
            }
            total.fetch_add(local, std::memory_order_relaxed);
        });
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true, std::memory_order_relaxed);
    for (auto& th : pool) th.join();
    const auto t1 = std::chrono::steady_clock::now();

    return {threads, total.load(std::memory_order_relaxed),
            std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace

int main(int argc, char** argv) {
    const double secs = argc > 1 ? std::atof(argv[1]) : 1.0;
    const double kTagPeriod = 4294967296.0;   // 2^32 successful operations

    std::printf("deque ABA headroom -- 32-bit tag, %.0f operations per wrap\n", kTagPeriod);
    std::printf("%-8s %-16s %-16s %s\n", "threads", "ops/s", "wrap (s)", "deschedule needed");

    double best_rate = 0.0;
    for (unsigned t : {1u, 2u, 4u, 8u}) {
        // 4096 is the per-worker queue capacity the job system uses by default.
        const Result r = run(t, secs, 4096);
        const double rate = r.seconds > 0 ? double(r.ops) / r.seconds : 0.0;
        if (rate > best_rate) best_rate = rate;
        const double wrap = rate > 0 ? kTagPeriod / rate : 0.0;
        std::printf("%-8u %-16.3e %-16.1f %s\n", t, rate, wrap,
                    "one thread stalled this long, continuously");
    }

    std::printf("\nA wrap alone is harmless. ABA requires the 2^32 operations to complete inside\n"
                "ONE thread's load-to-exchange window, so the number above is how long that\n"
                "thread must be descheduled while the others run flat out.\n");
    if (best_rate > 0)
        std::printf("Fastest observed: %.3e ops/s -> %.1f s of continuous deschedule.\n",
                    best_rate, kTagPeriod / best_rate);
    return 0;
}
