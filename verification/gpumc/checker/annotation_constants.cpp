// Checker test for the InstAnnotator fix in gpumc-fixes.patch: an undef pointer and a
// floating-point constant in a loop header's condition.
//
// LoopJumpThreadingPass annotates the header's branch condition with the values entering from the
// preheader. In `walk`, `p` enters the header undefined (a pointer undef); in `settle`, `x` enters
// as the double 4.0.
//
// gpumc_docker:latest aborts on `walk` in llvm::cast<IntegerType> (getIntegerBitWidth of a pointer
// undef, InstAnnotator.cpp:66) and stops on `settle` with "Only integer and null constants
// currently allowed in assume() expressions"; gpumc_docker:hg-fixes reports one complete execution
// and no error.
// GPUMC-CELL: patched PASS -- --disable-function-inliner
#include <cassert>
#include <pthread.h>

namespace {

int g_nodes[3] = {1, 2, 0};
int g_steps;
double g_x;

__attribute__((noinline)) int* next(int* p) { return p == nullptr ? &g_nodes[0] : (*p ? &g_nodes[*p] : nullptr); }
__attribute__((noinline)) double halve(double x) { return x / 2.0; }

void walk() {
    int* p;
    bool first = true;
    while (first || p != nullptr) {
        p = next(first ? nullptr : p);
        first = false;
        ++g_steps;
    }
}

void settle() {
    double x = 4.0;
    bool first = true;
    while (first || x > 1.0) {
        x = halve(x);
        first = false;
    }
    g_x = x;
}

void* run(void*) {
    walk();
    settle();
    return nullptr;
}

}  // namespace

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, run, nullptr);
    pthread_join(t, nullptr);
    assert(g_steps == 4 && g_x == 1.0);
    return 0;
}
