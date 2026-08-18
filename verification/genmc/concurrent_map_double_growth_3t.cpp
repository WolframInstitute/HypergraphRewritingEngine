// Two INDEPENDENT claimants of one key while a third thread drives the growths.
//
// WHY THIS EXISTS BESIDE THE 2t HARNESS. concurrent_map_double_growth_2t verifies the same
// shape and is calibrated, but it reaches tractability by FOLDING the third worker's inserts
// into the second: one thread does B, K, C in order, so the growth drivers and the second
// K-claimant are serialised with respect to each other. Its own comment says this "removes one
// thread's interleavings". The engine runs at eight threads and double-claims anyway --
// measured on the WPP workload, 29,240 branchial edges recorded from 30,063 winning claims,
// with the failure disappearing when the map is pre-sized so it never resizes. So the folded
// interleavings are exactly the ones under suspicion, and a harness that removes them cannot
// speak to it.
//
// WHY IT REDUCES ALONG A DIFFERENT AXIS. The three-worker spelling of the 2t shape prices at
// 3.3e9 executions (~40 days) and is unrunnable, which is why the fold was chosen. This one
// keeps all three threads and bounds CONTEXT SWITCHES instead, which is the reduction the
// README names for exactly this case. The cost is stated rather than hidden: bounding requires
// --sc, so this checks SEQUENTIAL CONSISTENCY, not RC11. It can therefore find an interleaving
// bug and CANNOT find a relaxed-memory one. The 2t harness covers RC11 for the folded shape;
// this covers three-thread interleavings under SC. Neither subsumes the other.
//
// The assertions are the 2t harness's: exactly one of the two K-claims reports was_inserted,
// both observe the same stored value, lookup agrees, and nothing else was lost.
//
// GENMC-ARGS: --disable-estimation --sc --bound=2 --bound-type=context
// GENMC-EXPECT: pass

#include <pthread.h>
#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {

using Map = hypergraph::ConcurrentMap<uint64_t, uint64_t>;

constexpr uint64_t kPre = 3, kK = 7, kB = 5;

Map* g_map;
uint64_t g_val[2];
bool g_ins[2];

// The two claimants of K, each in its own thread -- this is the separation the 2t harness folds
// away.
void* w_claim1(void*) {
    auto [v, ins] = g_map->insert_if_absent(kK, 100);
    g_val[0] = v; g_ins[0] = ins;
    return nullptr;
}
void* w_claim2(void*) {
    auto [v, ins] = g_map->insert_if_absent(kK, 200);
    g_val[1] = v; g_ins[1] = ins;
    return nullptr;
}
// The growth driver, independent of both claimants, so a table can be installed while a
// claimant holds an absence verdict for a table that is no longer the head.
void* w_grow(void*) {
    g_map->insert_if_absent(kB, 50);
    return nullptr;
}

}  // namespace

int main() {
    Map map(/*initial_capacity=*/2);
    g_map = &map;
    map.insert_if_absent(kPre, 30);

    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, w_claim1, nullptr);
    pthread_create(&t2, nullptr, w_claim2, nullptr);
    pthread_create(&t3, nullptr, w_grow, nullptr);
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
    assert(map.count_unique() == 3);
    return 0;
}
