// Expect: Mixed-size access to static memory
// A 64-bit store to a global followed by a 32-bit load of its upper half, the shape clang gives
// a 16-byte struct returned by value into a global array and then read field by field. The load
// resolved to the initializer (0) instead of the stored half, and a wrong value went on silently;
// it is reported as a mixed-size access.
#include <atomic>
#include <cassert>
#include <cstdint>

uint64_t g_word;
std::atomic<uint64_t> g_src{(uint64_t(7) << 32) | 3};

int main() {
    *reinterpret_cast<volatile uint64_t*>(&g_word) = g_src.load(std::memory_order_relaxed);
    const uint32_t hi = reinterpret_cast<volatile uint32_t*>(&g_word)[1];
    assert(hi == 7);
    return 0;
}
