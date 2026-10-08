// GenMC harness: a SegmentedArray element is read only after its writer published its index,
// and the read sees the constructed element.
//
// THE PROPERTY. The array records no extent. emplace_at constructs the element at an index the
// caller's counter handed out, and the caller publishes the index (a job, a state's edge set, a
// fenced rendezvous). A reader that acquires that publication reads the element through
// operator[]: the segment is non-null and both fields hold the constructed values.
//
// THE RACE. Two writers place elements at indices 0 and 1, both in segment 0 (segment shift 1,
// two elements per segment), so both race to create the segment and one gives its allocation
// back. Main reads index 0 after acquiring writer 0's flag, and index 1 after acquiring writer
// 1's.
//
// CALIBRATION. -DHG_CALIBRATE_SEGMENTED_HIGH_WATER puts back an extent advanced by every
// emplace_at as a high-water mark, and main reads index 0 through get(), bounded by it, with no
// flag. Writer 1 advances the mark to 2 while writer 0 is still constructing index 0, so main
// reads an element under construction: the checker reports the non-atomic race or the
// assertion.
//
// WHAT IS BOUNDED. Two writers and main, one element each, one segment of two elements.
//
// GENMC-LINK: support
// GENMC-ARGS: --disable-estimation
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=2 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=1
// GENMC-CALIBRATE: -DHG_CALIBRATE_SEGMENTED_HIGH_WATER
//
// Build/run: verification/genmc/run.sh segmented_array_published_read

#include <pthread.h>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>

#include "hypergraph/segmented_array.hpp"

namespace {

struct Pair {
    uint32_t a = 0;
    uint32_t b = 0;
    Pair() = default;
    Pair(uint32_t x, uint32_t y) : a(x), b(y) {}
};

// Disjoint storage per call, value-initialised, as the arena's allocate_array is. A given-back
// allocation is not reused: the harness checks the array's publication, and the arena's
// cursor give-back has its own harness (arena_cursor_vs_shared_disjoint).
struct StubArena {
    static constexpr int kCap = 4;
    alignas(16) unsigned char storage[kCap * 64];
    std::atomic<int> next{0};

    template <typename T>
    T* allocate_array(size_t n) {
        const int i = next.fetch_add(1, std::memory_order_relaxed);
        assert(i < kCap && sizeof(T) * n <= 64);
        T* arr = reinterpret_cast<T*>(storage + i * 64);
        for (size_t k = 0; k < n; ++k) new (&arr[k]) T();
        return arr;
    }
    bool release_last(void*, size_t) { return false; }
};

using Array = hypergraph::SegmentedArray<Pair>;

Array*     g_array;
StubArena* g_arena;
std::atomic<uint32_t> g_published[2];

void* writer(void* arg) {
    const uint32_t id = static_cast<uint32_t>(reinterpret_cast<long>(arg));
    g_array->emplace_at(id, *g_arena, 10u + id, 20u + id);
    g_published[id].store(1, std::memory_order_release);
    return nullptr;
}

}  // namespace

int main() {
    StubArena arena;
    Array array(1);
    g_arena = &arena;
    g_array = &array;
    g_published[0].store(0, std::memory_order_relaxed);
    g_published[1].store(0, std::memory_order_relaxed);

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, writer, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, writer, reinterpret_cast<void*>(1L));

#if defined(HG_CALIBRATE_SEGMENTED_HIGH_WATER)
    if (const Pair* p = array.get(0)) assert(p->a == 10u && p->b == 20u);
#else
    for (uint32_t i = 0; i < 2; ++i) {
        if (g_published[i].load(std::memory_order_acquire) != 0) {
            const Pair& p = array[i];
            assert(p.a == 10u + i && p.b == 20u + i);
        }
    }
#endif

    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);
    return 0;
}
