// GenMC harness: a key NEVER inserted is never reported as already present, across a growth.
//
// WHAT IS BEING PROVED, and why it is the other half. key_set_exactly_once bounds ONE key: two
// threads insert it, exactly one is told it inserted. That is the at-most-once direction, and it
// says nothing about a DIFFERENT key being wrongly rejected. Both are safety properties of the
// same boolean, and the engine acts on that boolean at five call sites -- so a false "already
// present" for a key never inserted silently drops whatever the caller would have done.
//
// WHAT IT WOULD LOOK LIKE IN THE ENGINE. qc_applied_ claims (instance, match) pairs: a false
// rejection means that application never runs, so one raw event never exists, and with it every
// causal and branchial pair it belonged to. The canonical STATE set is untouched, because the
// state was still explored. Observed shape of the intermittent quotient determinism failure:
// state fingerprints agree across runs while the event count and the branchial relation do not.
//
// THE WINDOW. insert() loads the head, scans the superseded chain for the key, and only then
// claims at the head -- three steps, with a growth able to land between any two. The chain scan
// is what a false positive would come from: a key found in a retiring table that is not this
// key, or a table skipped as drained while it still holds one. Growth is on both threads' entry
// path here, so the scan and the migration overlap by construction.
//
// WHAT IS BOUNDED. Two threads, two DISTINCT keys, an initial capacity of one so the first
// insertion crosses a growth. Both must be told they inserted; neither may be rejected.
//
// CALIBRATED with the false positive the window paragraph describes: find_in_table stopping at
// any occupant rather than at the key, `cur != EMPTY_KEY` where it tests `cur == key`. That is a
// key found in a retiring table that is not this key, and this harness reports a safety violation
// for it (genmc exit 42). Restored, it is clean in 102 complete executions.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh key_set_distinct_keys_across_growth

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_key_set.hpp"

namespace {

using Set = hypergraph::ConcurrentKeySet<uint64_t>;

constexpr uint64_t kSeed = 3;    // pre-loaded, so the table is already at its threshold
constexpr uint64_t kA    = 7;    // thread 0's key -- inserted by nobody else
constexpr uint64_t kB    = 9;    // thread 1's key -- inserted by nobody else

Set* g_set;
bool g_inserted[2];

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    g_inserted[id] = g_set->insert(id == 0 ? kA : kB);
    return nullptr;
}

}  // namespace

int main() {
    // Capacity 1 with a 0.75 threshold: the pre-load puts the set over its threshold, so both
    // threads enter through the growth path rather than racing to reach it.
    Set set(1);
    set.insert(kSeed);
    g_set = &set;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(1L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // BOTH INSERTED. Neither key was ever inserted by anyone else, so a false verdict here is a
    // caller told to skip work it was the only one asked to do.
    assert(g_inserted[0]);
    assert(g_inserted[1]);

    // And both are findable afterwards: a claim that reported a win must leave a key the
    // container can hand back, which is the invariant migrate_into broke once (f694c062).
    assert(set.contains(kA));
    assert(set.contains(kB));
    assert(set.contains(kSeed));
    return 0;
}
