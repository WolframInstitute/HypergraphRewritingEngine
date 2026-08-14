// GenMC harness: ConcurrentKeySet reports insertion exactly once per key, ACROSS A GROWTH.
//
// WHAT IS BEING PROVED. Two threads insert the SAME key into a set whose table is too small to
// hold it, so at least one of them runs the growth path -- copy every key into a doubled table,
// compare-exchange it into place, then stamp the superseded slots MIGRATED. The contract every
// caller relies on is that exactly one of them is told it inserted: the five call sites in the
// engine act on that boolean (register a branchial pair, drive a quotient transition, capture a
// transition once), so two winners means the same pair is counted twice and one winner too few
// means a transition is never driven.
//
// WHY THIS HARNESS AND NOT A FUZZ. A fuzz over this structure passed 40 seeds at 16 threads with
// capacity-1 tables and 20,000 operations per thread, and the engine's determinism gate still
// found an extra branchial pair on iteration 1496. That is the standing lesson of
// verification/genmc/README.md: sampling can only ever fail to reproduce, while GenMC is
// exhaustive over the interleavings and reads-from choices RC11 permits for the bounded program.
// The defect this harness targets is a window of a few instructions between a claim landing in a
// table and that table ceasing to be the head, which random scheduling essentially never hits.
//
// THE DEFECT IT EXISTS TO EXCLUDE. Thread A reads the head H0 and begins claiming there. Thread B
// grows, installs H1, and scans H1->prev for the key -- before A's claim is published. B finds
// nothing, claims at H1, and is told it inserted. A's compare-exchange into H0 then succeeds and
// A is told the same. Both report a first insertion of one key. The set answers this by anchoring
// the verdict to the table it was decided in: a claim that lands in a table which is no longer
// the head is stale, and the caller re-drives against the current head rather than reporting a
// win. Deleting that check is how this harness is calibrated -- it must report a violation.
//
// WHAT IS BOUNDED. Two threads, one key, an initial capacity of one so the first insertion
// crosses a growth, and no third operation. State the bound with the result: this is a statement
// about every execution of THIS program under RC11, not about unbounded thread counts.
//
// Build/run: verification/genmc/run.sh key_set_exactly_once

#include <pthread.h>
#include <cassert>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_key_set.hpp"

namespace {

// Key domain avoids both reserved keys: EMPTY_KEY = 0 and MIGRATED_KEY = ~0.
using Set = hypergraph::ConcurrentKeySet<uint64_t>;

constexpr uint64_t kSeed   = 3;   // pre-loaded, so the table is one insert from its threshold
constexpr uint64_t kFiller = 5;   // thread 1 inserts this FIRST, which is what forces the growth
constexpr uint64_t kKey    = 7;   // the contested key: both threads insert exactly this

Set* g_set;
bool g_inserted[2];

// Thread 0 races straight at the contested key, so its claim can be in flight against the table
// that thread 1 is retiring.
void* racer(void*) {
    g_inserted[0] = g_set->insert(kKey);
    return nullptr;
}

// Thread 1 forces a growth and only then goes for the contested key, so its chain scan of the
// retired table runs while thread 0 may still be claiming inside it. That ordering is the whole
// point: without the scan's seal, thread 0's claim lands in a table thread 1 has already looked
// past, and both are told they inserted one key.
void* grower(void*) {
    g_set->insert(kFiller);
    g_inserted[1] = g_set->insert(kKey);
    return nullptr;
}

}  // namespace

int main() {
    // Capacity 1 with a load-factor threshold of 0.75 means the very first insertion finds the
    // table over its threshold, so growth is on the path of both threads rather than a rare
    // event they might miss.
    // PRE-LOADED SO GROWTH IS ON BOTH THREADS' ENTRY PATH.
    //
    // Growth is triggered by count exceeding the load factor, so a set that starts empty only
    // grows AFTER a claim has completed -- which serialises claim and growth and makes the window
    // this harness exists for unreachable. Seeding one key into a capacity-1 table puts both
    // threads over the threshold before either claims, so growth and claim overlap. Calibrated:
    // with the chain scan's seal removed, this harness reports the violation.
    // Capacity 2 holding one key: the next insert crosses the load factor, so the growth happens
    // on thread 1's filler insert rather than on entry, leaving thread 0 free to be mid-claim in
    // the table being retired.
    Set set(/*initial_capacity=*/2);
    set.insert(kSeed);
    g_set = &set;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, racer, nullptr);
    pthread_create(&t1, nullptr, grower, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // A1: exactly one caller is told it inserted. Two winners double-count the key at every call
    // site; zero winners means the key is present and nobody was told they added it.
    assert(g_inserted[0] != g_inserted[1]);

    // A2: the key is present afterwards, whichever thread won and whichever table it landed in.
    // A growth that carried the key forward and a claim that landed in the superseded table are
    // both covered: contains() walks the chain.
    assert(set.contains(kKey));

    // A3: the count agrees with the single reported insertion. A stale claim that was re-driven
    // must not have left a count behind it.
    assert(set.size() == 3);   // seed, filler, and the one contested key

    return 0;
}
