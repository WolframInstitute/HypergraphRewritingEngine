// GenMC harness: ConcurrentMap::insert_if_absent of a key that settled before any thread
// started answers (its value, not inserted) while two growths overlap.
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
// One claimant of the settled key, two growers whose inserts overlap their growths.
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
constexpr uint64_t kPre = 7;   // settled before the threads start
constexpr uint64_t kB = 3, kC = 5, kD = 9, kE = 11;
Map* g_map;
uint64_t g_val;
bool g_ins;

void* w_claim(void*) {
    auto [v, ins] = g_map->insert_if_absent(kPre, 200);
    g_val = v; g_ins = ins;
    return nullptr;
}
void* w_grow1(void*) {
    g_map->insert_if_absent(kB, 50);
    g_map->insert_if_absent(kC, 90);
    return nullptr;
}
void* w_grow2(void*) {
    g_map->insert_if_absent(kD, 60);
    g_map->insert_if_absent(kE, 70);
    return nullptr;
}
}  // namespace

int main() {
    Map map(/*initial_capacity=*/2, /*arena=*/nullptr, /*working_capacity=*/4);
    g_map = &map;
    map.insert_if_absent(kPre, 100);
    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, w_claim, nullptr);
    pthread_create(&t2, nullptr, w_grow1, nullptr);
    pthread_create(&t3, nullptr, w_grow2, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);
    // The key was settled with 100 before the claimant started: not inserted, value 100.
    assert(!g_ins);
    assert(g_val == 100);
    auto k = map.lookup(kPre);
    assert(k.has_value() && *k == 100);
    assert(map.lookup(kB).has_value() && map.lookup(kC).has_value());
    assert(map.lookup(kD).has_value() && map.lookup(kE).has_value());
    return 0;
}
