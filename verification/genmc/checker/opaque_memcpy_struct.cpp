// A memcpy through untyped pointers, of a struct with fields of two widths and a length known
// only at run time, is lowered to a bounded loop; every field reads back as copied.
#include <cassert>
#include <cstring>

struct S {
    int a;
    long b;
};

__attribute__((noinline)) void copy(void* d, const void* s, unsigned long n) {
    std::memcpy(d, s, n);
}

int main() {
    S x{1, 2}, y{0, 0};
    volatile unsigned long n = sizeof(S);
    copy(&y, &x, n);
    assert(y.a == 1 && y.b == 2);
    return 0;
}
