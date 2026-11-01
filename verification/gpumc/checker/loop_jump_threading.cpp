// Checker test for the loop-jump-threading fix in gpumc-fixes.patch: a loop flag read after the loop.
//
// `bool ok = true; while (ok && full()) ok = grow();` starts with ok true, so the pass sends the
// entry straight into the loop body and the header runs only from the latch. The read of ok after
// the loop then has no definition on the path that leaves the body on the first iteration.
// gpu/src/persistent.cu held_select_step has this loop.
//
// gpumc_docker:latest reports "Instruction does not dominate all uses!" and aborts in
// LLVMModule.cpp:201 (transformLLVMModule); gpumc_docker:hg-fixes reports one complete execution
// and no error.
// GPUMC-CELL: patched PASS -- --disable-function-inliner
#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

uint32_t g_size = 5;
uint32_t g_cap = 2;
bool g_done;

__attribute__((noinline)) bool full() { return g_size > g_cap; }

__attribute__((noinline)) bool grow() {
    g_cap *= 2;
    return g_cap < 64;
}

void* run(void*) {
    bool ok = true;
    while (ok && full())
        ok = grow();
    if (ok) g_done = true;
    return nullptr;
}

}  // namespace

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, run, nullptr);
    pthread_join(t, nullptr);
    assert(g_done && g_cap == 8);
    return 0;
}
