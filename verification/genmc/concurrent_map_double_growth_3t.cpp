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
// A DEFECT LIVES HERE. At a FOUR-context bound this harness reports a safety violation on the
// shipped code, and is clean at three. Bisected to `assert(g_ins[0] != g_ins[1])`: two callers
// are told they inserted the same key. The value assertion holds, so both are handed the SAME
// value while both believe they created it.
//
// WHAT IT COSTS THE ENGINE. Every get-or-create site reads
//     lst = ins.second ? nl : ins.first;
// so two winners means each caller keeps the container IT allocated. One key ends up with two
// lists, the map retains one, and everything pushed into the orphan is invisible to every
// reader that goes through the map. In the quotient replay that is an instance list: those
// instances never meet the matches captured for their class, so those (instance, match)
// applications never run -- fewer raw events, and every causal and branchial pair over them
// changes, while the canonical STATE set is untouched. That is the shape of the intermittent
// CausalDeterminism failures exactly: states agree across runs, event count and both relations
// do not.
//
// NOT AN ARTEFACT OF THE REDUCTION. The violation persists at working_capacity 4 and 16 -- at
// 16 the table holds 3 keys against a threshold of 12, so it is not a full table -- and it is
// found under --sc, whose behaviours are a subset of RC11, so it is real on the shipped model.
//
// TWO FIXES ATTEMPTED AND BOTH REFUTED, by harnesses already here:
//   verdict = (anchored value == our offer)  -- refuted by concurrent_map_repeated_offer, which
//       states that was_inserted comes from the publishing EXCHANGE, not a value comparison:
//       two callers offering the same value would both compare equal.
//   verdict = anchored.second                -- under-reports and gives ZERO winners, because a
//       migration or another anchor can place our value at the head first.
// Both refutations say the same thing: the verdict logic is not the defect. TWO EXCHANGES CAN
// WIN FOR ONE KEY, so the fault is in the mutual exclusion between a settle in a superseded
// table and a settle at the head. That is where the fix has to go.
//
// WHAT THE COUNTEREXAMPLE SHOWS. Reading the successful compare-exchanges out of the trace
// (the two offers are 100 and 200, so the writes name their authors): the claim-2 thread
// settles 200 in one table, and the claim-1 thread settles 100 in ANOTHER -- after that same
// thread has already MIGRATED 200 forward as part of a resize it performed. So its chain scan
// concluded the key was absent from a table where a settled entry for it existed.
//
// FOUR CANDIDATE FIXES, ALL REFUTED, so the next attempt does not repeat them:
//   1. verdict = (anchored value == our offer)
//        refuted by concurrent_map_repeated_offer: was_inserted comes from the publishing
//        EXCHANGE, and two callers offering the same value would both compare equal.
//   2. verdict = anchored.second
//        refuted here AND by repeated_offer: it under-reports to ZERO winners, because a
//        migration or a rival anchor can place our value at the head before our anchor runs.
//   3. claim only in the CURRENT head (re-drive when the table we hold is superseded)
//        still violates. The stale-head claim is not the path.
//   4. do not truncate the scan's probe run at LOCKED (continue instead of break)
//        still violates. The seal converting EMPTY to LOCKED is not hiding the entry.
//
// So the surviving hypothesis is narrower than any of those: a settle that wins in a table
// which is no longer the head must not be authoritative, and making it so without losing the
// value -- the old table stays reachable and readers find it there -- is the part that is not
// yet worked out.
//
// GENMC-ARGS: --disable-estimation --sc --bound=4 --bound-type=context
// GENMC-EXPECT: fail   (the defect above; restore to `pass` with the fix)

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
