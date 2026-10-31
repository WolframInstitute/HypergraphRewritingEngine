// A multiply-add in one expression (clang emits llvm.fmuladd under its default -ffp-contract=on)
// and fabs (llvm.fabs) are lowered by the checker, as hgcommon/det_math.hpp and
// hgcommon/state_geometry_core.hpp use them in every composed harness. Without the lowering the
// checker stops with "Code generator does not support intrinsic function".
#include <cassert>
#include <cmath>

int main() {
    volatile double a = 3.0, b = 0.5, c = -4.0;
    const double m = a * b + c;                 // llvm.fmuladd.f64
    assert(m == -2.5);
    assert(__builtin_fabs(m) == 2.5);           // llvm.fabs.f64
    volatile double z = -0.0;
    assert(!std::signbit(__builtin_fabs(z)));
    volatile float f = -1.5f;
    assert(__builtin_fabsf(f) == 1.5f);         // llvm.fabs.f32
    return 0;
}
