// GenMC harness: exactly-one insert winner across a WARRANTED double growth, at a bound
// small enough to EXHAUST.
//
// The property. A key claimed while the table chain grows twice must still produce exactly
// one was_inserted, one stored value, and no lost entries. The map holds that by anchoring
// every absence verdict to the head it claims in: an insert enters through the chain, and a
// scan that finds no claim SEALS the slot a late claim would need (find_and_settle_in_chain),
// so each older table either yields its claim or is foreclosed. resize() seals whole tables
// the same way after installing the new head. Re-drives -- after a seal, an exhausted probe
// run, or a growth -- go back through that scan (drive_at_head) rather than claiming at the
// head on a verdict formed against an older table.
//
// The shape, minimal for two warranted growths:
//   capacity 2 (threshold: count > 1.5), one key pre-inserted single-threaded;
//   W1 inserts K            -- the contended key;
//   W2 inserts kB, K, kC    -- the rival K claim, plus the load that carries count to
//                              4 > 3 = 4 * 0.75, warranting the SECOND growth (4 -> 8)
//                              while W1's claim is still in flight.
//
// WHY TWO WORKERS AND NOT THREE. The three-worker spelling of the same shape prices at
// 3.3e9 executions (genmc --mode=estimate, ~40 days) -- unrunnable, and an enumeration that
// cannot finish proves nothing. Folding W3's insert into W2 preserves both growths and the
// concurrent claim pair while removing one thread's interleavings: 130,897 complete
// executions, 200s, exhaustive. Sized before running, not discovered by waiting.
//
// THAT RESULT NO LONGER REPRODUCES, measured 2026-08-19 on a quiet box (load 2.97): no verdict
// in 9m50s, and none in a second run given 55 minutes. The recorded 130,897 executions in 200s
// was real when it was written, so the state space this harness explores has grown since --
// which is a statement about ConcurrentMap, not about the harness. GenMC's estimator is no help
// deciding by how much: it reports 1,437 executions and 55s for this shape, an order of
// magnitude BELOW a run that then does not finish, so an estimate here is not evidence that a
// harness will complete.
//
// The property is therefore NOT currently verified at this shape. What is verified for this map
// is concurrent_map_agreement (2 threads, 1 key, no resize, 32 executions) and
// concurrent_map_resize (2 threads, ONE growth, 176 executions).
//
// CALIBRATED. Removing the re-anchor from drive_at_head (so a re-drive claims at the head on
// a stale verdict) makes this harness report the safety violation after 1,517 executions.
// The bound is small BECAUSE it is sufficient, not because it is convenient.
//
// The assertions: exactly one of the two K-claims reports was_inserted, both observe the same
// stored value, lookup agrees, and nothing else was lost while the tables churned.
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

constexpr uint64_t kPre = 3, kK = 7, kB = 5, kC = 9;

Map* g_map;
uint64_t g_val[2];
bool g_ins[2];

void* w1(void*) {
    auto [v, ins] = g_map->insert_if_absent(kK, 100);
    g_val[0] = v; g_ins[0] = ins;
    return nullptr;
}
void* w2b(void*) {
    g_map->insert_if_absent(kB, 50);
    auto [v, ins] = g_map->insert_if_absent(kK, 200);
    g_val[1] = v; g_ins[1] = ins;
    g_map->insert_if_absent(kC, 90);
    return nullptr;
}

}  // namespace

int main() {
    Map map(/*initial_capacity=*/2);
    g_map = &map;
    map.insert_if_absent(kPre, 30);

    pthread_t t1, t2;
    pthread_create(&t1, nullptr, w1, nullptr);
    pthread_create(&t2, nullptr, w2b, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

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
