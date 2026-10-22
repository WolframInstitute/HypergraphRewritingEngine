// Checker test for the floating-point support in gpumc-fixes.patch: fma, fabs, floor and sqrt.
//
// gpumc_docker:latest aborts with "Code generator does not support intrinsic function
// 'llvm.fma.f64'"; gpumc_docker:hg-fixes reports one complete execution and no error.
// GPUMC-CELL: patched PASS
#include <cassert>
#include <cmath>
#include <pthread.h>

namespace {

double g_a, g_b, g_c;
double g_fma, g_fabs, g_floor, g_sqrt;

void* run(void*) {
    g_fma = std::fma(g_a, 2.0, 0.5);
    g_fabs = std::fabs(g_b);
    g_floor = std::floor(g_b);
    g_sqrt = std::sqrt(g_c);
    return nullptr;
}

}  // namespace

int main() {
    g_a = 1.5;
    g_b = -2.25;
    g_c = 9.0;
    pthread_t t;
    pthread_create(&t, nullptr, run, nullptr);
    pthread_join(t, nullptr);
    assert(g_fma == 3.5 && g_fabs == 2.25 && g_floor == -3.0 && g_sqrt == 3.0);
    return 0;
}
