// GenMC harness: ConcurrentMap get-or-create agreement.
//
// WHAT IS BEING PROVED. Two threads call insert_if_absent(K, ...) with the SAME key and
// DIFFERENT values. The header states the contract every get-or-create caller relies on:
//
//     "Exactly one offer wins, and every caller returns the winner -- so was_inserted means
//      'the value you passed is the one now stored'."
//
// A caller that gets was_inserted == true keeps the object it built and publishes it; one that
// gets false destroys its object and adopts the returned one. If two callers could BOTH be told
// they won, two objects would be published for one key -- which is the silently split rendezvous
// the header warns about. If both were told they lost, the map would hold a value neither of
// them can name. If the two callers were handed DIFFERENT values, the engine would have two
// container objects for one key: exactly the failure that produced four separate correctness
// bugs through this class already.
//
// So the harness asserts, over EVERY interleaving GenMC enumerates:
//   A1  exactly one of the two calls reports was_inserted
//   A2  both calls return the SAME value
//   A3  that value is one of the two offered (not a torn or default read)
//   A4  a subsequent lookup returns that same value
//
// WHY GenMC RATHER THAN A STRESS TEST. A stress test samples interleavings; on this box the
// determinism gate went 150 runs without firing a defect known to exist. GenMC enumerates the
// executions of the RC11 model exhaustively for this bounded program, so a clean run is a proof
// over that bound rather than a failure to reproduce.
//
// WHAT IS BOUNDED. Two threads, one key, a table small enough that the probe run is short, and
// no resize. Resize is a separate harness -- it is a different algorithm (rehash-then-CAS-install)
// and mixing it in here would blow up the state space without sharpening either question.
//
// CALIBRATED. The election is publish_value's ABSENT-to-value compare-exchange, so the defect is
// the losing caller keeping its own answer: return (value, true) where it returns (current,
// false). That is A1 and A2 broken at once -- two callers told they inserted, each handed a
// different value -- and this harness reports a safety violation for it (genmc exit 42). Restored,
// it is clean in 32 complete executions.
//
// Build/run: verification/genmc/run.sh concurrent_map_agreement

#include <pthread.h>
#include <cassert>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {

// A map whose key domain avoids both sentinels (EMPTY_KEY = 0, LOCKED_KEY = ~0) and whose value
// domain avoids ABSENT_VALUE (0). Capacity 4 keeps the probe run -- and so the state space --
// small; the algorithm under test does not depend on capacity beyond the probe bound.
using Map = hypergraph::ConcurrentMap<uint64_t, uint64_t>;

constexpr uint64_t kKey = 7;
constexpr uint64_t kValueA = 100;
constexpr uint64_t kValueB = 200;

Map* g_map;
uint64_t g_value[2];
bool g_inserted[2];

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    const uint64_t mine = id == 0 ? kValueA : kValueB;
    auto [got, inserted] = g_map->insert_if_absent(kKey, mine);
    g_value[id] = got;
    g_inserted[id] = inserted;
    return nullptr;
}

}  // namespace

int main() {
    Map map(/*initial_capacity=*/4);
    g_map = &map;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(1L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // A1: exactly one winner. Two winners means two published objects for one key; zero winners
    // means the stored value belongs to no caller.
    assert(g_inserted[0] != g_inserted[1]);

    // A2: both callers agree on what is stored. Disagreement is the split rendezvous.
    assert(g_value[0] == g_value[1]);

    // A3: the agreed value is one that was actually offered.
    assert(g_value[0] == kValueA || g_value[0] == kValueB);

    // A2': the winner's own value is the one stored -- this is what was_inserted MEANS.
    assert(g_value[g_inserted[0] ? 0 : 1] == (g_inserted[0] ? kValueA : kValueB));

    // A4: the map agrees with both callers.
    auto found = map.lookup(kKey);
    assert(found.has_value());
    assert(*found == g_value[0]);

    return 0;
}
