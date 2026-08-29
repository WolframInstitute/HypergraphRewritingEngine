// GenMC harness: ConcurrentKeySet::contains sees a key seeded before any thread started,
// while two growths overlap.
//
// THE CLASS. key_set_contains_during_growth has one grower, so its growths run one after the
// other. With two growers a table can be superseded before the migration INTO it is done,
// and a reader that loaded the middle table can find the seeded key's copy above its start.
// ConcurrentMap had exactly this window (map_lookup_during_double_growth); the key set is the
// sibling structure with its own migration, and it is checked in the same shape rather than
// assumed to share the map's answer. Absent is never a linearizable answer here.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
#include <pthread.h>
#include <cassert>
#include <cstdint>
#include "genmc_support.hpp"
#include "hypergraph/concurrent_key_set.hpp"

namespace {
using Set = hypergraph::ConcurrentKeySet<uint64_t>;
constexpr uint64_t kProbe = 11;             // seeded before the threads start; must be seen
constexpr uint64_t kSeed[2] = {11, 13};   // two seeds: the smallest table that still grows twice under the four inserts
Set* g_set;
bool g_seen;

void* w_read(void*) { g_seen = g_set->contains(kProbe); return nullptr; }
void* w_grow1(void*) { g_set->insert(5); return nullptr; }
void* w_grow2(void*) { g_set->insert(7); return nullptr; }
}  // namespace

int main() {
    Set set(/*initial_capacity=*/2, /*arena=*/nullptr, /*working_capacity=*/4);
    for (uint64_t k : kSeed) set.insert(k);
    g_set = &set;
    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, w_read, nullptr);
    pthread_create(&t2, nullptr, w_grow1, nullptr);
    pthread_create(&t3, nullptr, w_grow2, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);
    assert(g_seen);
    return 0;
}
