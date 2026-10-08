// GenMC harness: ConcurrentMap::insert_if_absent of a key that settled before any thread
// started answers (its value, not inserted) while a growth overtakes the claim.
//
// THE CLASS. A walk that loads the head and is then overtaken by a growth it never loaded:
// the carry of an older table's entries lands in the head as it stands at carry time, which
// can be above the walk's start, and the older table drains. map_lookup_during_double_growth
// reports the lookup half of that class; this is the claim half. A claimant whose chain scan
// misses the settled entry offers into its stale head, and if that head is not yet sealed the
// key is claimed twice -- two winners for one key, the shape of the historical branchial
// double claims (concurrent_map_double_growth_3t's comment records them disappearing when the
// map was pre-sized). Absent is never a linearizable answer for a settled key.
//
// THE BOUND. One claimant of the settled key and one grower. kPre and kB are in the map before
// the threads start (capacity 2, past its 1.5 threshold), so the grower's two inserts grow the
// map while the claimant walks it. Measured on the v0.19 fork: 322 executions in 1 s, and
// HG_CALIBRATE_MAP_NO_CHAIN_SCAN is reported after 51. Two growers of one insert each (three
// threads) gave no verdict in 3000 s, and two growers of two inserts each none in 3163 s.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
// GENMC-CALIBRATE: -DHG_CALIBRATE_MAP_NO_CHAIN_SCAN
#include <pthread.h>
#include <cassert>
#include <cstdint>
#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {
using Map = hypergraph::ConcurrentMap<uint64_t, uint64_t>;
constexpr uint64_t kPre = 7;   // settled before the threads start
constexpr uint64_t kB = 3, kC = 5, kD = 9;
Map* g_map;
uint64_t g_val;
bool g_ins;

void* w_claim(void*) {
    auto [v, ins] = g_map->insert_if_absent(kPre, 200);
    g_val = v; g_ins = ins;
    return nullptr;
}
void* w_grow1(void*) {
    g_map->insert_if_absent(kC, 90);
    g_map->insert_if_absent(kD, 60);
    return nullptr;
}
}  // namespace

int main() {
    Map map(/*initial_capacity=*/2, /*arena=*/nullptr, /*working_capacity=*/4);
    g_map = &map;
    map.insert_if_absent(kPre, 100);
    map.insert_if_absent(kB, 50);
    pthread_t t1, t2;
    pthread_create(&t1, nullptr, w_claim, nullptr);
    pthread_create(&t2, nullptr, w_grow1, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    // The key was settled with 100 before the claimant started: not inserted, value 100.
    assert(!g_ins);
    assert(g_val == 100);
    auto k = map.lookup(kPre);
    assert(k.has_value() && *k == 100);
    assert(map.lookup(kB).has_value() && map.lookup(kC).has_value());
    assert(map.lookup(kD).has_value());
    return 0;
}
