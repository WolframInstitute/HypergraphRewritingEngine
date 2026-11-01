// Checker test for the loop-jump-threading fix in gpumc-fixes.patch: a PHI in the body block the
// entry is sent to.
//
// `first` is true on entry, so the pass sends the entry straight into the loop body. The block it
// jumps to merges the header's branch with the call to g(), so it has a PHI for v, and the pass
// gave that PHI no entry for the new edge from the preheader.
//
// gpumc_docker:latest reports "PHINode should have one entry for each predecessor of its parent
// basic block!" and aborts in LLVMModule.cpp (transformLLVMModule); gpumc_docker:hg-fixes reports
// one complete execution and no error.
// GPUMC-CELL: patched PASS -- --disable-function-inliner
#include <cassert>
#include <pthread.h>

namespace {

int g_n;
int g_sum;

__attribute__((noinline)) int g() { return 2; }
__attribute__((noinline)) bool done() { return ++g_n == 3; }

void* run(void*) {
    bool first = true;
    for (;;) {
        int v = 1;
        if (!first) v = g();
        g_sum += v;
        first = false;
        if (done()) break;
    }
    return nullptr;
}

}  // namespace

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, run, nullptr);
    pthread_join(t, nullptr);
    assert(g_sum == 5);
    return 0;
}
