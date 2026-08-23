// GenMC harness: ConcurrentMap::lookup sees a value that settled before the call started,
// even while a growth is carrying that entry between tables and marking the source drained.
//
// THE WINDOW IT TARGETS. Chain walks skip tables whose `drained` flag is set, and the flag is
// only trustworthy in one read order: every flag read BEFORE any table is probed. Read at the
// moment the walk reaches each table, the flag reopens the hand-off window
// ConcurrentKeySet::contains had -- the head probed before the carry lands its copy, the old
// table drained by the time the walk arrives and skipped, the entry invisible at both. The
// pre-read orders the walk after the drain's release whenever it decides to skip, so the
// copies the drain promised are visible to every probe below it.
//
// Two threads: a reader looking up a key whose entry settled before either thread started
// (absent is never a linearizable answer), and a grower whose inserts push the map through
// its growths -- capacity 2 to 4 on the first (working capacity), 4 to 8 when the count
// crosses 3 -- so the carry and the drain race the read.
//
// CALIBRATED by breaking the read order: moving the drained load into the walk (read at
// arrival, after newer tables were probed) makes this harness report the violation within
// two explored executions; the pre-read is clean across the space (6 complete executions,
// exhaustive RC11, no bound).
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass

#include <pthread.h>

#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {

using Map = hypergraph::ConcurrentMap<uint64_t, uint64_t>;

constexpr uint64_t kProbe = 7;   // settled before the threads start; must be seen
constexpr uint64_t kB = 3, kC = 5, kD = 9;

Map* g_map;
uint64_t g_seen;

void* w_read(void*) {
    auto v = g_map->lookup(kProbe);
    g_seen = v.has_value() ? *v : 0;
    return nullptr;
}

// The growth driver: three inserts walk the count from 1 to 4, warranting a growth past the
// 0.75 load factor at each table size on the way, with the carries and drains in flight while
// the reader runs.
void* w_grow(void*) {
    g_map->insert_if_absent(kB, 50);
    g_map->insert_if_absent(kC, 90);
    g_map->insert_if_absent(kD, 60);
    return nullptr;
}

}  // namespace

int main() {
    Map map(/*initial_capacity=*/2, /*arena=*/nullptr, /*working_capacity=*/4);
    g_map = &map;
    map.insert_if_absent(kProbe, 100);

    pthread_t t1, t2;
    pthread_create(&t1, nullptr, w_read, nullptr);
    pthread_create(&t2, nullptr, w_grow, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    // The entry settled before the reader started, so every linearization answers 100.
    assert(g_seen == 100);
    return 0;
}
