// Expect: HG-ERROR-SITE load
// Env: HG_GENMC_ERROR_SITE=1
// With HG_GENMC_ERROR_SITE set, the load an error is reported on prints its instruction and the
// functions on its call stack.
#include <cstdlib>

__attribute__((noinline)) int read_one(const int* p) { return p[1]; }

int main() {
    int* p = static_cast<int*>(std::malloc(8));
    volatile int v = read_one(p);
    (void)v;
    std::free(p);
    return 0;
}
