// GenMC harness: ConcurrentKeySet enumerates every key exactly once, ACROSS A GROWTH.
//
// WHAT IS BEING PROVED. One thread inserts a key while another grows the table under it, and the
// main thread then enumerates. The contract is that for_each emits each present key once: the
// quotient reconstruction reads the branchial pair set through for_each
// (for_each_reconstructed_branchial_as), so a key emitted twice is a branchial edge reported
// twice, and a key emitted zero times is one lost.
//
// WHY IT IS A SEPARATE HARNESS FROM key_set_exactly_once. That one bounds the INSERT verdict; this
// one bounds what the container then contains. They failed differently in practice, which is the
// argument for keeping them apart: a first attempt at this structure enumerated 79,215 keys for
// 30,063 distinct ones, because growth copied a key into the new table and left it in the old
// while both stayed reachable through the chain, and a second attempt reported one key inserted
// twice, because a claim could land in a table that was being retired. One harness would have
// blamed whichever assertion happened to be checked first.
//
// THE DEFECT IT EXISTS TO EXCLUDE. A key present in two tables of the chain, emitted once per
// table. The set answers it with the MIGRATED sentinel: growth carries a key into the new table
// and then seals the slot it came from, so a slot says which of the two it is and enumeration
// skips the sealed one. Removing the seal, or emitting sealed slots, must make this harness fire.
//
// WHAT IS BOUNDED. Two threads, two keys, an initial capacity of one so growth is on the path,
// and one enumeration after both threads join. This is a statement about every execution of THIS
// program under RC11, not about unbounded thread counts.
//
// Build/run: verification/genmc/run.sh key_set_enumeration

#include <pthread.h>
#include <cassert>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_key_set.hpp"

namespace {

using Set = hypergraph::ConcurrentKeySet<uint64_t>;

constexpr uint64_t kSeed = 3;
constexpr uint64_t kKeyA = 7;
constexpr uint64_t kKeyB = 9;

Set* g_set;
bool g_inserted[2];

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    g_inserted[id] = g_set->insert(id == 0 ? kKeyA : kKeyB);
    return nullptr;
}

}  // namespace

int main() {
    // PRE-LOADED SO GROWTH IS ON BOTH THREADS' ENTRY PATH.
    //
    // Growth is triggered by count exceeding the load factor, so a set that starts empty only
    // grows AFTER a claim has completed -- which serialises claim and growth and makes the window
    // this harness exists for unreachable. Seeding one key into a capacity-1 table puts both
    // threads over the threshold before either claims, so growth and claim overlap. Calibrated:
    // with the chain scan's seal removed, this harness reports the violation.
    Set set(/*initial_capacity=*/1);
    set.insert(kSeed);
    g_set = &set;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(1L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // Distinct keys, so both insertions are first insertions whatever the interleaving.
    assert(g_inserted[0]);
    assert(g_inserted[1]);

    unsigned seen_a = 0, seen_b = 0, seen_seed = 0, seen_other = 0;
    set.for_each([&](uint64_t k) {
        if (k == kKeyA)      ++seen_a;
        else if (k == kKeyB) ++seen_b;
        else if (k == kSeed) ++seen_seed;
        else                 ++seen_other;
    });
    assert(seen_seed == 1);   // the seeded key survives the growth exactly once

    // E1: each key appears exactly once, however many tables the chain holds.
    assert(seen_a == 1);
    assert(seen_b == 1);

    // E2: nothing else is emitted -- in particular neither reserved key escapes as a value.
    assert(seen_other == 0);

    // E3: enumeration agrees with membership.
    assert(set.contains(kKeyA));
    assert(set.contains(kKeyB));

    return 0;
}
