// GPUMC harness: the persistent kernel's work logs (the expand entries and the replay's tasks) hand
// every appended entry to exactly one consumer, which reads it fully written.
//
// Runs gpu/include/hg_gpu/work_log.hpp's WorkLogView::append, claim, await and book themselves
// (over Pool::claim), with the device's atomics (verification/gpumc/device).
//
// THE PROTOCOL. A producer claims a slot with the pool's add, writes the entry and stores its
// `published` word with release. A consumer claims a run of consecutive entries below the readable
// count with one compare-exchange on the cursor, waits for each entry's `published` with acquire,
// reads it, and books the run in `done`.
//
// THE SHAPE. Two producer blocks append one task each; two consumer blocks each claim up to two
// entries once, read them and book them. Blocks are one thread.
//
// THE PROPERTY. The claimed runs are disjoint, each claimed entry reads as its producer wrote it,
// and `done` is the number claimed. No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the cursor's compare-exchange a load and a
// separate store; the checker must report an entry claimed twice.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/quotient_expansion.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

constexpr uint32_t kCap = 4;

g::QeTask g_items[kCap];
uint32_t g_count, g_cursor, g_done;
uint32_t g_base[2], g_n[2];
bool g_payload_ok[2] = {true, true};

g::WorkLogView<g::QeTask> log_view() {
    g::WorkLogView<g::QeTask> w;
    w.items = {g_items, &g_count, kCap};
    w.cursor = &g_cursor;
    w.done = &g_done;
    return w;
}

void* producer(void* arg) {
    const unsigned id = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(id, 0);
    g::WorkLogView<g::QeTask> w = log_view();
    const bool ok = w.append(g::QeTask{0x100u + id, id, 1u, 2u, 0u});
    assert(ok && "an append into a log with room was refused");
    return nullptr;
}

void* consumer(void* arg) {
    const unsigned c = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(2 + c, 0);
    g::WorkLogView<g::QeTask> w = log_view();
    uint32_t base = 0;
    const uint32_t n = w.claim(2, base);
    for (uint32_t i = 0; i < n; ++i) {
        const g::QeTask& t = w.await(base + i);
        if (t.hash != 0x100u + t.rec || t.depth != 1u || t.match != 2u) g_payload_ok[c] = false;
    }
    if (n) w.book(n);
    g_base[c] = base;
    g_n[c] = n;
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[4] = {1, 1, 1, 1};
    hg_gpumc::launch(4, lanes);
    pthread_t t[4];
    for (long i = 0; i < 2; ++i) pthread_create(&t[i], nullptr, producer, reinterpret_cast<void*>(i));
    for (long i = 0; i < 2; ++i)
        pthread_create(&t[2 + i], nullptr, consumer, reinterpret_cast<void*>(i));
    for (int i = 0; i < 4; ++i) pthread_join(t[i], nullptr);

    assert(g_payload_ok[0] && g_payload_ok[1] && "a claimed entry was read before it was written");
    if (g_n[0] && g_n[1])
        assert((g_base[0] + g_n[0] <= g_base[1] || g_base[1] + g_n[1] <= g_base[0]) &&
               "two consumers claimed one entry");
    assert(g_done == g_n[0] + g_n[1]);
    assert(g_cursor == g_done && "the cursor and the booked count disagree");
    return 0;
}
