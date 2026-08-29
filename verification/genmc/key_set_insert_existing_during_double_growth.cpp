// GenMC harness: ConcurrentKeySet::insert of a key seeded before any thread started answers
// "already present" while two growths overlap.
//
// THE CLASS, claim half, on the key set: a claimant whose walk is overtaken by a growth it
// never loaded misses the settled copy above its start and reports a fresh insert for a key
// the set already holds. The engine's matched_raw_states_ and dedup sets are this structure;
// a false "inserted" there is a state matched twice or a match applied twice.
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
constexpr uint64_t kPre = 11;
constexpr uint64_t kSeed[2] = {11, 13};   // two seeds: the smallest table that still grows twice under the four inserts
Set* g_set;
bool g_inserted;

void* w_claim(void*) { g_inserted = g_set->insert(kPre); return nullptr; }
void* w_grow1(void*) { g_set->insert(5); return nullptr; }
void* w_grow2(void*) { g_set->insert(7); return nullptr; }
}  // namespace

int main() {
    Set set(/*initial_capacity=*/2, /*arena=*/nullptr, /*working_capacity=*/4);
    for (uint64_t k : kSeed) set.insert(k);
    g_set = &set;
    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, w_claim, nullptr);
    pthread_create(&t2, nullptr, w_grow1, nullptr);
    pthread_create(&t3, nullptr, w_grow2, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);
    // Seeded before the claimant started: never a fresh insert.
    assert(!g_inserted);
    assert(g_set->contains(kPre));
    return 0;
}
