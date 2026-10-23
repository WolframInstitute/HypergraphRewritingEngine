// GPUMC harness: threads inserting the same key into the device map elect exactly one inserter,
// and the stored value is that thread's; a thread's lookup after its insert finds the key.
//
// Runs gpu/include/hg_gpu/hash_table.hpp's ConcurrentMap::DeviceView::insert_if_absent (the
// hgcommon::hash_insert_claim body over the device's key and value exchanges) and lookup
// themselves, with the device's atomics (verification/gpumc/device).
//
// WHY BOTH HALVES. `inserted` is not a courtesy flag: the device marks an event CANONICAL on it
// and points every later event at the STORED value, and qe.applied gates an APPLICATION on it.
// One thread reporting inserted while another's value stands gives one signature two canonical
// events.
//
// THE BOUND. Three threads in three blocks, the same key, distinct 32-bit values, a map of two
// slots: the election, and a probe that can pass a slot another thread holds.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes every compare-exchange a load and a separate
// store; the checker must report two inserters or a stored value that is not the inserter's.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/hash_table.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using Map = g::ConcurrentMap<uint64_t, uint32_t>;

constexpr uint32_t kSlots = 2;
constexpr uint32_t kThreads = 3;
constexpr uint64_t kKey = 7;

g::MapSlot<uint64_t, uint32_t> g_slots[kSlots];
uint32_t g_saturated;
uint32_t g_offered[kThreads];
bool g_inserted[kThreads];
uint32_t g_returned[kThreads];
bool g_found[kThreads];
uint32_t g_found_value[kThreads];

Map::DeviceView map() { return {g_slots, kSlots, kSlots - 1, &g_saturated}; }

void* inserter(void* arg) {
    const unsigned id = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(id, 0);
    g_offered[id] = 10 + id;
    const Map::DeviceView m = map();
    const auto r = m.insert_if_absent(kKey, g_offered[id]);
    assert(!r.overflowed && "a two-slot map with one key overflowed");
    g_inserted[id] = r.inserted;
    g_returned[id] = r.value;
    const auto l = m.lookup(kKey);
    g_found[id] = l.found;
    g_found_value[id] = l.value;
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[kThreads] = {1, 1, 1};
    hg_gpumc::launch(kThreads, lanes);
    // ConcurrentMap::clear: every key EMPTY, every value UNPUBLISHED.
    for (uint32_t s = 0; s < kSlots; ++s) g_slots[s] = {0, Map::DeviceView::UNPUBLISHED};
    pthread_t t[kThreads];
    for (long i = 0; i < static_cast<long>(kThreads); ++i)
        pthread_create(&t[i], nullptr, inserter, reinterpret_cast<void*>(i));
    for (uint32_t i = 0; i < kThreads; ++i) pthread_join(t[i], nullptr);

    uint32_t winners = 0, winner = 0;
    for (uint32_t i = 0; i < kThreads; ++i)
        if (g_inserted[i]) { ++winners; winner = i; }
    assert(winners == 1 && "the key was inserted by no thread, or by more than one");
    const auto stood = map().lookup(kKey);
    assert(stood.found && stood.value == g_offered[winner] &&
           "the stored value is not the inserter's");
    for (uint32_t i = 0; i < kThreads; ++i) {
        assert(g_returned[i] == g_offered[winner] && "an insert returned a value that did not stand");
        assert(g_found[i] && g_found_value[i] == g_offered[winner] &&
               "a lookup after the thread's own insert missed the key");
    }
    return 0;
}
