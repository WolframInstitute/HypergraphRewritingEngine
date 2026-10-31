// fneg, LLVM's one unary operator, which clang emits for a negated double. The interpreter had no
// visitor for it: the instruction was printed and the checker stopped on a segmentation fault.
#include <cassert>

volatile double g_x = 1.5;

int main() {
    double y = -g_x;
    assert(y == -1.5);
    return 0;
}
