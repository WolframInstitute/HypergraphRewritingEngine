// GenMC harness: is the WARRANTED double-growth residual actually reachable?
//
// 29283f7 closed the resize double-claim by removing the UNNECESSARY second installation:
// resize() re-tests the load factor against the current head, so a thread whose resize decision
// was made against a stale table does not install over a table that already has room. The
// RESIDUAL recorded with it: when growth is GENUINELY warranted twice over -- the count crosses
// the threshold again between one installation and the next -- a thread still holding the middle
// table can claim a key there while a rival, whose chain scan already passed that table, claims
// the same key at the head. Closing that needs seal-and-migrate, a rewrite of the primitive.
//
// This harness exists to decide whether the rewrite has anything to fix AT A BOUND GENMC CAN
// EXPLORE. It arranges the tightest configuration that can warrant two growths:
//
//   capacity 2 (threshold: count > 1.5), one key pre-inserted single-threaded;
//   W1 inserts K   -- the contended key;
//   W2 inserts kB, then K  -- the second K claim, plus load toward the next threshold;
//   W3 inserts kC          -- more load: with all four keys in, count 4 > 3 = 4 * 0.75,
//                             so a SECOND growth (4 -> 8) is warranted while claims are in flight.
//
// THE ASSERTION IS THE RESIDUAL'S ABSENCE: exactly one of the two K-claims reports
// was_inserted, and lookup(K) agrees with the winner. A safety violation here is not a harness
// bug -- it is the residual, reached, and the seal-and-migrate rewrite becomes justified with a
// reproducer. No violation over the full enumeration means the residual is UNREACHABLE at this
// bound, and the recorded configuration must be enlarged before any rewrite is argued for.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: violation

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
void* w2(void*) {
    g_map->insert_if_absent(kB, 50);
    auto [v, ins] = g_map->insert_if_absent(kK, 200);
    g_val[1] = v; g_ins[1] = ins;
    return nullptr;
}
void* w3(void*) {
    g_map->insert_if_absent(kC, 90);
    return nullptr;
}

}  // namespace

int main() {
    Map map(/*initial_capacity=*/2);
    g_map = &map;
    map.insert_if_absent(kPre, 30);

    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, w1, nullptr);
    pthread_create(&t2, nullptr, w2, nullptr);
    pthread_create(&t3, nullptr, w3, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);

    // The residual's signature: both K-claims told they inserted.
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
