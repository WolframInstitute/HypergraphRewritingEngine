// Expect: Attempt to read from uninitialized memory
// A read of heap memory nothing wrote is reported as an error, not stopped by an internal check.
#include <cstdlib>

int main() {
    int* p = static_cast<int*>(std::malloc(8));
    volatile int v = p[1];
    (void)v;
    std::free(p);
    return 0;
}
