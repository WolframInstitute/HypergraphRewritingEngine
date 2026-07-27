// tools/arena_bump_bench.cpp
//
// Isolates the cost of the per-worker scratch arena's bump path, which backs every
// SVec/ScratchAlloc temporary on the hot path. Reports nanoseconds per allocation
// under the mark/release cycle that real callers use.
//
// Build:
//   g++ -O2 -std=c++17 -I hypergraph/include tools/arena_bump_bench.cpp \
//       -o /tmp/arena_bump_bench -pthread

#include <hypergraph/arena.hpp>
#include <chrono>
#include <cstdio>

using namespace hypergraph;

int main() {
    constexpr int OUTER = 200000;   // mark/release cycles (≈ one canonicalization each)
    constexpr int INNER = 240;      // allocations per cycle (≈ IR's measured count)

    auto& a = worker_scratch();
    volatile void* sink = nullptr;
    double best = 1e30;

    for (int rep = 0; rep < 7; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < OUTER; ++i) {
            auto mk = a.mark();
            for (int j = 0; j < INNER; ++j) sink = a.allocate_raw(24 + (j & 7) * 8, 8);
            a.release(mk);
        }
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count()
                  / (double(OUTER) * INNER);
        if (ns < best) best = ns;
    }
    (void)sink;
    std::printf("scratch arena bump: %.3f ns/alloc  (%d cycles x %d allocs, min of 7)\n",
                best, OUTER, INNER);
    return 0;
}
