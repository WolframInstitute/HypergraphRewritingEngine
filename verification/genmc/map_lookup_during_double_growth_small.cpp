// GenMC harness (the smaller spelling): ConcurrentMap::lookup sees a value that settled before the call started,
// while TWO growths overlap -- the second table superseded before the first migration is done.
//
// WHAT THE 2-THREAD HARNESS SERIALISES AWAY. map_lookup_during_growth has one grower, so every
// growth completes its carry and drain before the next begins. In the engine many threads
// insert at once, and a table can be sealed while the migration INTO it is still carrying the
// previous table's entries: the carry then bounces off sealed slots and re-offers at the newer
// head, the middle table drains without ever having held the key, and the reader's snapshot of
// drained flags meets three tables in three states. That is the shape here: a reader looking up
// a key that settled before any thread started, and two growers whose inserts each warrant a
// growth, running concurrently.
//
// The engine's forwarding chain and match store are ConcurrentMaps keyed by state id, whose
// last growth lands at the last expanded level -- where every lost event of the live
// nondeterminism failures sits. Absent is never a linearizable answer for the probed key.
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
constexpr uint64_t kB = 3, kC = 5, kD = 9;   // three inserts: the settled key plus these cross 1.5 and 3, two growths
Map* g_map;
uint64_t g_seen;

void* w_read(void*) {
    auto v = g_map->lookup(kProbe);
    g_seen = v.has_value() ? *v : 0;
    return nullptr;
}
// Two growers. Each pair of inserts crosses a load-factor threshold on its own; together they
// can drive the 2->4 and 4->8 growths with both migrations in flight at once.
void* w_grow1(void*) {
    g_map->insert_if_absent(kB, 50);
    g_map->insert_if_absent(kC, 90);
    return nullptr;
}
void* w_grow2(void*) {
    g_map->insert_if_absent(kD, 60);
    return nullptr;
}
}  // namespace

int main() {
    Map map(/*initial_capacity=*/2, /*arena=*/nullptr, /*working_capacity=*/4);
    g_map = &map;
    map.insert_if_absent(kProbe, 100);
    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, w_read, nullptr);
    pthread_create(&t2, nullptr, w_grow1, nullptr);
    pthread_create(&t3, nullptr, w_grow2, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);
    // The entry settled before the reader started, so every linearization answers 100.
    assert(g_seen == 100);
    return 0;
}
