// A module holding llvm.fma, as hgcommon/det_math.hpp's std::fma compiles, on a path no execution
// takes. The transform stopped with "Code generator does not support intrinsic function
// 'llvm.fma.f64'" whether or not the path ran; fma is lowered to a call to the C library's fma.
#include <atomic>
#include <cassert>
#include <cmath>

std::atomic<int> g_flag{0};
volatile double g_a = 1.5, g_b = 2.0, g_c = 0.25;

int main() {
    double r = 0.0;
    if (g_flag.load(std::memory_order_relaxed) == 1) r = std::fma(g_a, g_b, g_c);
    assert(r == 0.0);
    return 0;
}
