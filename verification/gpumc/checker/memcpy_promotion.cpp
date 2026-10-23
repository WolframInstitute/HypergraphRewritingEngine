// Checker test for the memory-intrinsic promotion in gpumc-fixes.patch.
//
// 1. A struct copied into a different struct type with the same field layout (memcpy whose
//    source and destination GEP types differ) copies every field. Field widths match, since
//    GPUMC reports a mixed-size access when a load spans two stores.
// 2. A memset of 0x01 over a struct of 32-bit fields stores 0x01010101 in each field.
// 3. A memset of zero over a struct holding a pointer and a double stores null and 0.0.
// 4. A memcpy of a prefix copies the fields inside the prefix and no other.
// 5. A memcpy whose length is the source's size and not the destination's is promoted over the
//    source's type.
//
// gpumc_docker:latest stops with "BUG: Failure at PromoteMemIntrinsicPass.cpp:78/promoteMemSet()";
// gpumc_docker:hg-fixes reports one complete execution and no error.
// GPUMC-CELL: patched PASS
#include <cassert>
#include <cstdint>
#include <cstring>
#include <pthread.h>

namespace {

struct A {
    uint32_t x, y;
    uint64_t z;
};
struct B {
    uint32_t q, r;
    uint64_t s;
};
struct Ptrs {
    void* p;
    double d;
};

A g_a;
B g_b;
B g_b2;
uint32_t g_words[2];
Ptrs g_ptrs;
struct Pair {
    uint32_t a, b;
};
Pair g_pair;

void* run(void*) {
    g_a.x = 1;
    g_a.y = 2;
    g_a.z = 3;
    std::memcpy(&g_b, &g_a, sizeof(B));
    std::memset(g_words, 1, sizeof(g_words));
    g_ptrs.p = &g_a;
    g_ptrs.d = 1.5;
    std::memset(&g_ptrs, 0, sizeof(g_ptrs));
    // 4. The first 8 bytes of A into B: the two 32-bit fields, not the 64-bit one.
    g_b2.s = 9;
    std::memcpy(&g_b2, &g_a, 8);
    // 5. A 4-byte source into the first field of an 8-byte struct (the source's size).
    g_pair.b = 7;
    std::memcpy(&g_pair, &g_a.y, sizeof(uint32_t));
    return nullptr;
}

}  // namespace

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, run, nullptr);
    pthread_join(t, nullptr);
    assert(g_b.q == 1u && g_b.r == 2u && g_b.s == 3u && "the copy is not A's bytes");
    assert(g_words[0] == 0x01010101u && g_words[1] == 0x01010101u && "memset stored the fill byte");
    assert(g_ptrs.p == nullptr && g_ptrs.d == 0.0);
    assert(g_b2.q == 1u && g_b2.r == 2u && g_b2.s == 9u && "the 8-byte copy reached past 8 bytes");
    assert(g_pair.a == 2u && g_pair.b == 7u && "the 4-byte copy into an 8-byte struct");
    return 0;
}
