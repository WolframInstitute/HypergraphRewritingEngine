// tools/arena_reset_test.cpp
//
// Validates the A1 primitive: ConcurrentHeterogeneousArena::reset() recycles blocks
// instead of freeing them, so a per-worker scratch arena touches malloc ONCE
// (during warmup) and never again across many task cycles. Counts global allocations.
//
// Build: g++ -O2 -std=c++17 -I hypergraph/include tools/arena_reset_test.cpp -o /tmp/arena_reset_test

#include <hypergraph/arena.hpp>
#include <cstdio>
#include <cstdlib>
#include <new>

static long long g_allocs = 0; static bool g_track = false;
void* operator new(std::size_t n){ if(g_track)++g_allocs; void* p=std::malloc(n?n:1); if(!p)throw std::bad_alloc(); return p; }
void operator delete(void* p)noexcept{ std::free(p); }
void operator delete(void* p,std::size_t)noexcept{ std::free(p); }

using namespace hypergraph;

int main(){
    ConcurrentHeterogeneousArena scratch(ConcurrentHeterogeneousArena::DEFAULT_BLOCK_SIZE, /*recycle=*/true);
    auto do_task = [&](int objs){               // simulate one task's scratch usage
        for (int i = 0; i < objs; ++i) (void)scratch.allocate_raw(64, 8);
    };

    // warmup one task, then measure mallocs across many reset cycles
    g_track = true;
    do_task(20000);                              // ~1.2 MB > one 1 MB block -> grows a 2nd block
    size_t peak = scratch.bytes_allocated();
    long long after_warmup = g_allocs;

    const int CYCLES = 2000;
    long long bad_reset = 0;
    for (int task = 0; task < CYCLES; ++task) {
        scratch.reset();
        if (scratch.bytes_allocated() != 0) ++bad_reset;   // reset must zero usage
        do_task(20000);
        if (scratch.bytes_allocated() != peak) { /* same high-water each task */ }
    }
    long long steady = g_allocs - after_warmup;
    g_track = false;

    std::printf("warmup mallocs (build initial blocks): %lld\n", after_warmup);
    std::printf("mallocs across %d reset+task cycles: %lld  (expect 0 -> blocks recycled)\n", CYCLES, steady);
    std::printf("bytes after reset always zero: %s\n", bad_reset==0 ? "yes" : "NO");
    std::printf("scratch high-water: %zu bytes (bounded, not growing per task)\n", peak);
    bool ok = (steady == 0) && (bad_reset == 0);
    std::printf("RESULT: %s\n", ok ? "PASS (malloc touched once, then recycled)" : "FAIL");
    return ok ? 0 : 1;
}
