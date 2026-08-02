// GenMC harness: was_inserted comes from the publishing EXCHANGE, not from comparing values.
//
// THE PROPERTY. Exactly one call ever sees was_inserted for a key -- across every table
// generation, and however many times a value is offered. A caller may offer the same value for
// the same key repeatedly; only the call whose compare-exchange published it won.
//
// WHY THIS IS ITS OWN HARNESS. concurrent_map_double_growth_2t holds the double-growth window
// exhaustively, and every offer in it is DISTINCT (100 from W1, 200 from W2). While offers
// differ, "the stored value equals mine" and "my exchange won" agree in every execution, so
// that harness cannot separate them -- it reported clean while ConcurrentMapFuzz found a second
// winner in about 1.5% of runs. Folding a repeated offer into it instead pushed the enumeration
// past 560s without completing, and an enumeration that cannot finish proves nothing. So the
// two properties get two harnesses, each sized to EXHAUST.
//
// THE SHAPE, minimal for the anchoring path:
//   capacity 2 (threshold: count > 1.5);
//   W1 inserts K TWICE with the SAME value -- the repeated offer;
//   W2 inserts kB, then K                  -- kB carries count past the threshold so the head
//                                             moves while W1's claim is in flight, which is
//                                             what puts a settle on the anchoring path; K is
//                                             the concurrent rival.
// One growth suffices: the verdict is re-derived only when a settle anchors at a head that is
// no longer the table it settled in, and the head moving once already reaches that.
//
// CALIBRATED. Deriving the verdict as `anchored.first == value` in settle() and
// find_and_settle_in_chain() -- which is what shipped -- makes this report the safety violation
// after 1,452 executions. Clean, it exhausts at 3,755 complete executions in a few seconds.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass

#include <pthread.h>
#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {

using Map = hypergraph::ConcurrentMap<uint64_t, uint64_t>;

constexpr uint64_t kK = 7, kB = 5;

Map* g_map;
uint64_t g_val[3];
bool g_ins[3];

void* w1(void*) {
    auto [v, ins] = g_map->insert_if_absent(kK, 100);
    g_val[0] = v; g_ins[0] = ins;
    // The SAME value again. At most one of these two calls may report was_inserted: the second
    // finds its own value stored, which a comparison cannot tell from having just won it.
    auto [v2, ins2] = g_map->insert_if_absent(kK, 100);
    g_val[2] = v2; g_ins[2] = ins2;
    return nullptr;
}

void* w2(void*) {
    g_map->insert_if_absent(kB, 50);
    auto [v, ins] = g_map->insert_if_absent(kK, 200);
    g_val[1] = v; g_ins[1] = ins;
    return nullptr;
}

}  // namespace

int main() {
    Map map(/*initial_capacity=*/2);
    g_map = &map;

    pthread_t t1, t2;
    pthread_create(&t1, nullptr, w1, nullptr);
    pthread_create(&t2, nullptr, w2, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    // EXACTLY ONE winner among the three offers for K.
    const int wins = (g_ins[0] ? 1 : 0) + (g_ins[1] ? 1 : 0) + (g_ins[2] ? 1 : 0);
    assert(wins == 1);

    // And every caller observes that winner's value.
    assert(g_val[0] == g_val[1]);
    assert(g_val[0] == g_val[2]);

    auto k = map.lookup(kK);
    assert(k.has_value() && *k == g_val[0]);

    // Nothing else was lost while the table grew.
    assert(map.lookup(kB).has_value());
    assert(map.count_unique() == 2);
    return 0;
}
