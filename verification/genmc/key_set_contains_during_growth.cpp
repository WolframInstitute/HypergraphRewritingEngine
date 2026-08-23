// GenMC harness: ConcurrentKeySet::contains sees a key that settled before the call started,
// even while a growth is carrying that key between tables.
//
// THE DEFECT IT TARGETS. contains() walks the chain newest-first. A reader loads the head H1
// and probes it before the migration has carried the key in -- absent. It then walks to H0,
// where the migration has since carried the key out and sealed its slot MIGRATED. grow()'s
// pre-seal pass has already closed every EMPTY slot of H0, so the probe run finds no EMPTY
// stop and exhausts the table over MIGRATED entries -- absent again. The key was reachable in
// at least one table at every instant (the carry lands in H1 before the seal lands in H0), but
// this walk visited each table on the wrong side of the hand-off and reported a settled key
// absent. insert()'s chain scan has the same window and is saved by its head claim: the CAS
// probes the same deterministic run the carry uses, so it meets the carried key and reports
// kLost. contains() has no claim to back it up; the walk order is the whole answer.
//
// Walking OLDEST-FIRST closes the window: a probe that meets the seal (MIGRATED, acquire)
// synchronizes with the exchange that published it (acq_rel), and the carry into the newer
// table is ordered before that seal -- so the destination table, walked later, shows the key.
//
// WHY IT MATTERS. CausalGraph::add_causal_edge exempts an already-stored event pair from the
// redundancy check with seen_causal_event_pairs_.contains(pair_key): a stored pair's later
// raw edges are multiplicity, not candidates for dropping. A contains that misses the settled
// pair re-runs the redundancy check, and when the pair has become reachable through other kept
// edges the arriving triple is dropped -- one causal edge missing from a full-capture run,
// with states, events and branchial identical. That is the CausalDeterminism firing on the
// macOS runner (WPP, TR on, threads=16 on ~4 cores, causal 25367 against 25368).
//
// WHAT IS COVERED. Two threads: one reader of a key seeded before either thread starts (so
// every linearization must answer true), one grower whose first insert takes the seventh of
// eight slots and whose second therefore grows and migrates while the read is in flight.
// Exhaustive under RC11 with no context bound: 11 complete executions, 9 blocked (the
// blocked ones are contains()'s head-stability retry, which the checker's spin-assume
// treatment cuts after the re-walk it would repeat is explored elsewhere).
//
// CALIBRATED by the defect itself, at these same arguments: with the newest-first walk the
// violation is reported in the first explored execution (g_seen false after both joins); the
// oldest-first walk is clean across the full space.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass

#include <pthread.h>

#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_key_set.hpp"

namespace {

// Key domain avoids both reserved keys: EMPTY_KEY = 0 and MIGRATED_KEY = ~0.
using Set = hypergraph::ConcurrentKeySet<uint64_t>;

constexpr uint64_t kProbe = 11;             // seeded before the threads start; must be seen
constexpr uint64_t kFill1 = 5, kFill2 = 3;  // the grower's two inserts
constexpr uint64_t kSeed[6] = {11, 13, 17, 19, 23, 29};

Set* g_set;
bool g_seen;

void* w_read(void*) { g_seen = g_set->contains(kProbe); return nullptr; }

// The growth driver. Its first insert takes the seventh of eight slots and puts the table over
// the load factor; its second therefore grows, installing the successor and migrating the six
// seeds -- kProbe among them -- while the reader is in flight.
void* w_grow(void*) {
    g_set->insert(kFill1);
    g_set->insert(kFill2);
    return nullptr;
}

}  // namespace

int main() {
    // working_capacity 16 rather than the default 1024: the migration and the successor's
    // initialisation are per-slot, and a 1024-wide successor cannot be exhausted by a checker.
    // The protocol is identical at either width; see the constructor's note.
    Set set(/*initial_capacity=*/8, /*arena=*/nullptr, /*working_capacity=*/16);
    for (uint64_t k : kSeed) set.insert(k);
    g_set = &set;

    pthread_t t1, t2;
    pthread_create(&t1, nullptr, w_read, nullptr);
    pthread_create(&t2, nullptr, w_grow, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    // A key settled before the reader started is seen in every execution. There is no
    // concurrent insert of kProbe to race with: absent is never a linearizable answer here.
    assert(g_seen);
    return 0;
}
