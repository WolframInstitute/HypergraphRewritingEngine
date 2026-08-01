// GenMC harness: a key inserted while the table is being replaced still gets ONE entry.
//
// WHAT IS BEING PROVED. Resize is a different algorithm from insert -- allocate a bigger table,
// rehash every settled entry into it, then compare-exchange it into place -- and it runs
// concurrently with the inserts that triggered it. concurrent_map.hpp names the hazard directly:
//
//     "An entry can live only in one of them -- an insert that resolved against a table after its
//      slot had been rehashed completes there -- and inserting again would give one key two
//      entries, which for the get-or-create callers means two container objects and a silently
//      split rendezvous."
//
// Two threads crossing a resize with the same key must therefore still agree on one winner, and
// the keys already in the map must survive being rehashed underneath them.
//
// This is the configuration that forces it. Capacity 2 resizes once the count exceeds 1.5, so two
// keys are inserted single-threaded to reach the threshold; both worker threads then read a count
// over it, both call resize(), and both insert the same new key. So the harness covers two
// concurrent resizes racing to install their table AND two concurrent inserts of one key crossing
// that installation -- the loser of the table compare-exchange discards its table and re-reads
// the winner's, which is the path where a stale table pointer would show up.
//
// Assertions:
//   A1  exactly one inserter of the new key reports was_inserted
//   A2  both return the same value, and it is one of the two offered
//   A3  the map agrees: lookup of the new key returns that value
//   A4  NOTHING IS LOST ACROSS THE REHASH -- both pre-existing keys still resolve to their
//       original values, whichever table won
//   A5  no key acquired a second entry: count_unique() is exactly 3
//
// A5 is the one that would catch the split rendezvous. A4 is the one that would catch a rehash
// dropping an entry, or an insert completing into a table that lost the race and was discarded.
//
// WHAT IS BOUNDED. Two threads, capacity 2 growing to 4, three keys total, one resize round.

#include <pthread.h>
#include <cassert>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {

using Map = hypergraph::ConcurrentMap<uint64_t, uint64_t>;

constexpr uint64_t kOldKeyA = 3, kOldValueA = 30;
constexpr uint64_t kOldKeyB = 5, kOldValueB = 50;
constexpr uint64_t kNewKey  = 7;
constexpr uint64_t kValueA  = 100, kValueB = 200;

Map* g_map;
uint64_t g_value[2];
bool g_inserted[2];

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    auto [got, inserted] = g_map->insert_if_absent(kNewKey, id == 0 ? kValueA : kValueB);
    g_value[id] = got;
    g_inserted[id] = inserted;
    return nullptr;
}

}  // namespace

int main() {
    // Capacity 2 with two keys already in it puts the count over the 0.75 load factor, so the
    // next insert_if_absent on either thread resizes before it does anything else.
    Map map(/*initial_capacity=*/2);
    g_map = &map;
    map.insert_if_absent(kOldKeyA, kOldValueA);
    map.insert_if_absent(kOldKeyB, kOldValueB);

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(1L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // A1, A2: the agreement contract holds across the table swap.
    assert(g_inserted[0] != g_inserted[1]);
    assert(g_value[0] == g_value[1]);
    assert(g_value[0] == kValueA || g_value[0] == kValueB);
    assert(g_value[g_inserted[0] ? 0 : 1] == (g_inserted[0] ? kValueA : kValueB));

    // A3: the map agrees with both callers.
    auto found = map.lookup(kNewKey);
    assert(found.has_value());
    assert(*found == g_value[0]);

    // A4: the rehash preserved what was already there. An entry reachable only through a table
    // that lost the install race, or dropped while being copied, fails here.
    auto a = map.lookup(kOldKeyA);
    auto b = map.lookup(kOldKeyB);
    assert(a.has_value() && *a == kOldValueA);
    assert(b.has_value() && *b == kOldValueB);

    // A5: three keys, three entries. A key carrying two entries is the split rendezvous.
    assert(map.count_unique() == 3);

    return 0;
}
