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
// NOT CURRENTLY EXHAUSTED, measured 2026-08-19. The bound below is what makes this shape
// tractable in principle, and GenMC estimates 24,565 executions and 204s for it; the actual run
// produced no verdict in 15 minutes on a quiet box. The estimator is optimistic by more than an
// order of magnitude on this map -- the 2t harness estimates 55s and does not finish in 55
// minutes -- so neither the estimate nor this file's earlier wording is evidence of a completed
// verification.
//
// THE DEFECT THIS HARNESS FOUND, and the fix it verifies.
//
// At a FOUR-context bound this reported a safety violation on the shipped code and was clean at
// three. Bisected to `assert(g_ins[0] != g_ins[1])`: two callers were told they inserted the
// same key, both handed the SAME value while both believed they created it.
//
// THE MECHANISM, read out of the counterexample. Three tables exist. One claimant's value
// exchange BEAT the retiring table's seal -- the seal exchanges ABSENT to FORWARDED and lost the
// race -- so a settled entry stayed behind in a superseded table. The other claimant migrated
// that value forward as part of a resize it was performing, and then settled its OWN value in a
// third table. Two exchanges won for one key, in two tables.
//
// WHY THE VERDICT CANNOT COME FROM ONE EXCHANGE. `was_inserted` was "my value exchange won the
// slot I reached", and a key can be reached in more than one table, so that predicate is not
// unique. Nor can the verdict be a value comparison: concurrent_map_repeated_offer states that
// two callers may offer the SAME value, and a comparison calls both of them the inserter.
//
// The fix is the CONJUNCTION -- the caller inserted iff its own exchange won AND the value the
// map now answers with is the one it offered. Same values are separated by the exchange; two
// winning exchanges in different tables are separated by the answer, because a lookup walks the
// chain newest-first and there is exactly one answer.
//
// WHY THIS HARNESS COULD NOT SEE ANY OF IT BEFORE. Its header records "130,897 complete
// executions, 200s, exhaustive" and it had stopped completing -- no verdict in 55 minutes --
// because growth was changed to jump straight to DEFAULT_INITIAL_CAPACITY, making the second
// growth's seal 1024 slots x 2 compare-exchanges. The verification regressed behind a
// performance change. The growth target is a constructor parameter now, defaulted, so the
// harness can bound the protocol without changing it.
//
// GENMC-ARGS: --disable-estimation --sc --bound=4 --bound-type=context
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
    Map map(/*initial_capacity=*/2, /*arena=*/nullptr, /*working_capacity=*/16);
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
