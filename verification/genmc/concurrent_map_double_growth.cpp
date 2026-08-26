// GenMC harness: exactly-one insert winner across a warranted double growth, THREE workers.
//
// Same property and same shape as concurrent_map_double_growth_2t, with W2's third insert
// carried by its own worker. The extra thread's interleavings put this at 3.3e9 executions
// (genmc --mode=estimate), which cannot be exhausted here -- so this harness runs in
// ESTIMATION mode, which samples real executions and reports a violation when it hits one.
// A clean estimate is evidence, not proof; the proof at this shape is the 2t harness, which
// is exhaustive.
//
// SCHEDULE BOUNDING DOES NOT RESCUE IT, measured 2026-08-19. The 3t harness is exhausted under
// `--sc --bound=2 --bound-type=context`; applying the same bound here times out at 580s, and so
// does `--bound=1`. The size is in the map's own resize and migration paths, not in the
// interleavings, so bounding the schedule leaves the graph as large as it was. Reducing this
// harness far enough to exhaust would mean removing the fourth key -- which is the one that
// warrants the SECOND growth, and therefore the entire property.
//
// Sampling is not a weak instrument for this defect class. The un-anchored map -- a re-drive
// claiming at the head on a verdict formed against an older table -- is refuted here in
// 0.10s, while the dedicated fuzz gate (hypergraph/tests/test_concurrent_map_fuzz.cpp,
// millions of operations at growth-saturated shapes) passes: random scheduling does not
// reach the interleaving.
//
//   capacity 2 (threshold: count > 1.5), one key pre-inserted single-threaded;
//   W1 inserts K   -- the contended key;
//   W2 inserts kB, then K  -- the second K claim, plus load toward the next threshold;
//   W3 inserts kC          -- more load: with all four keys in, count 4 > 3 = 4 * 0.75,
//                             so a SECOND growth (4 -> 8) is warranted while claims are in flight.
//
// The assertions: exactly one of the two K-claims reports was_inserted, both observe the same
// stored value, lookup agrees, and nothing else was lost while the tables churned.
//
// CALIBRATED, which matters more here than anywhere else in this directory: a sampler that
// samples nothing also reports no violation. Two defects are recorded. The un-anchored map -- a
// re-drive claiming at the head on an absence verdict formed against an older table -- is
// refuted in 0.10s. So is the losing caller keeping its own answer in publish_value, returning
// (value, true) where it returns (current, false): estimation reports a safety violation and
// exits 42 rather than completing. This harness samples, and what it samples reaches the defects.
//
// GENMC-ARGS: --mode=estimate
// GENMC-EXPECT: pass

#include <pthread.h>
#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {

using Map = hypergraph::ConcurrentMap<uint64_t, uint64_t>;

constexpr uint64_t kPre = 3, kK = 7, kB = 5, kC = 9;

Map* g_map;
uint64_t g_val[2];
bool g_ins[2];

void* w1(void*) {
    auto [v, ins] = g_map->insert_if_absent(kK, 100);
    g_val[0] = v; g_ins[0] = ins;
    return nullptr;
}
void* w2(void*) {
    g_map->insert_if_absent(kB, 50);
    auto [v, ins] = g_map->insert_if_absent(kK, 200);
    g_val[1] = v; g_ins[1] = ins;
    return nullptr;
}
void* w3(void*) {
    g_map->insert_if_absent(kC, 90);
    return nullptr;
}

}  // namespace

int main() {
    Map map(/*initial_capacity=*/2, /*arena=*/nullptr, /*working_capacity=*/4);
    g_map = &map;
    map.insert_if_absent(kPre, 30);

    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, w1, nullptr);
    pthread_create(&t2, nullptr, w2, nullptr);
    pthread_create(&t3, nullptr, w3, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);

    // Exactly one K-claim wins, and both observe the winner's value.
    assert(g_ins[0] != g_ins[1]);
    assert(g_val[0] == g_val[1]);

    auto k = map.lookup(kK);
    assert(k.has_value() && *k == g_val[0]);

    // Nothing else was lost while the tables churned.
    assert(map.lookup(kPre).has_value());
    assert(map.lookup(kB).has_value());
    assert(map.lookup(kC).has_value());
    assert(map.count_unique() == 4);
    return 0;
}
