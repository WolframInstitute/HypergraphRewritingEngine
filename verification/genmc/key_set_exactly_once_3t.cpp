// GenMC harness: ConcurrentKeySet reports insertion exactly once per key when the growth is
// driven by a thread that is NEITHER claimant and the retiring table still has a free slot.
//
// WHY THIS AND NOT key_set_exactly_once. That harness runs two threads on a capacity-2 table,
// and neither condition this one needs is on its path. Its grower is also a claimant, and
// grow() carries and seals every slot of the retiring table before it returns, so by the time
// that thread claims the contested key the old table is fully MIGRATED and a rival's late claim
// there meets sealed slots and re-drives. Worse, capacity 2 over the 0.75 load factor is FULL:
// there is no free slot for a late claim to land in at all. So the window is unreachable --
// 741 executions, no violation, and none of them can get near it.
//
// WHAT THE WINDOW NEEDS, and why the shape below is what it is. A claim can only land in a
// retiring table if that table is over its load-factor threshold while still holding a free
// slot: count > 0.75*capacity and count < capacity, which first happens at capacity 8 with
// seven keys. So: eight slots, six seeded, and the grower's first insert takes the seventh --
// leaving exactly one free slot and putting the table over threshold, so its SECOND insert
// grows while a claimant can still CAS into what is being retired.
//
// THE DEFECT IT TARGETS. Thread A loads the head H0, passes the count check, and starts claiming
// there. Thread B pushes the count over the threshold and grows, installing H1 and beginning the
// migration. Thread C, entering at H1, scans the chain for the key -- H0 is not drained yet, so
// it is walked -- does not find it, claims at H1 and is told it inserted. A's compare-exchange
// then lands in the one free slot of H0, which B's migration has not sealed yet, and A is told
// it inserted too. insert() reads table_ once and claim() returns kWon on a successful exchange
// without re-reading it, so nothing rejects A's verdict. The carry-then-seal exchange keeps the
// key from being LOST -- B carries it into H1, where it finds C's copy -- but losing a key and
// double-reporting a claim are different failures, and only the first is what that exchange is
// for.
//
// WHY IT MATTERS. Every caller of insert() acts on the boolean. Hypergraph::QrCtx::claim is
// qc_applied_.insert(apply_key), and it gates the whole of qr_apply: it is the only thing making
// the two paths into qc_apply -- qc_add_instance iterating a class's captured matches, and
// qc_capture_expansion replaying a new match against the instances already standing -- safe
// against each other. Two winners for one (instance, match) pair replay that pair twice, which
// is one extra raw event, one extra causal edge, its branchial pairs, and the extra instance
// record the replay descends into.
//
// WHAT IS BOUNDED. Three threads, one contested key, one growth out of an eight-slot table,
// under a ONE-context bound. This is a statement about every execution of THIS program under
// RC11 at that bound, not about unbounded thread counts. The bound is one rather than the four
// its ConcurrentMap counterpart uses because an eight-slot table is what the free-slot condition
// costs: the seal pass, the migration and a sixteen-slot successor put the four-context space
// past 580s with no verdict, and two and three do the same. One context is where it completes
// -- 19,414 complete executions.
//
// CALIBRATED, which is the only thing that makes a bound that small worth stating. Deleting the
// seal pass from grow() must make this harness report the violation, and it does: 173 executions
// at this same bound, both claimants told they inserted one key.
//
// GENMC-ARGS: --disable-estimation --sc --bound=1 --bound-type=context
// GENMC-EXPECT: pass

#include <pthread.h>
#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_key_set.hpp"

namespace {

// Key domain avoids both reserved keys: EMPTY_KEY = 0 and MIGRATED_KEY = ~0.
using Set = hypergraph::ConcurrentKeySet<uint64_t>;

constexpr uint64_t kKey = 7;              // the contested key: both claimants insert exactly this
constexpr uint64_t kFill1 = 5, kFill2 = 3;  // the grower's two inserts
constexpr uint64_t kSeed[6] = {11, 13, 17, 19, 23, 29};

Set* g_set;
bool g_ins[2];

// The two claimants of the contested key, each in its own thread -- this is the separation the
// two-thread harness folds away.
void* w_claim1(void*) { g_ins[0] = g_set->insert(kKey); return nullptr; }
void* w_claim2(void*) { g_ins[1] = g_set->insert(kKey); return nullptr; }

// The growth driver, which never touches the contested key. Its first insert takes the seventh
// of eight slots and puts the table over the load factor; its second therefore grows, installing
// the successor and migrating while both claimants are already in flight.
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

    pthread_t t1, t2, t3;
    pthread_create(&t1, nullptr, w_claim1, nullptr);
    pthread_create(&t2, nullptr, w_claim2, nullptr);
    pthread_create(&t3, nullptr, w_grow, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);
    pthread_join(t3, nullptr);

    // A1: exactly one caller is told it inserted.
    assert(g_ins[0] != g_ins[1]);

    // A2: the key is present afterwards, whichever table it landed in.
    assert(set.contains(kKey));

    // A3: the count agrees with the single reported insertion -- a stale claim that was
    // re-driven must not have left a count behind it.
    assert(set.size() == 9);   // six seeds, two fillers, and the one contested key
    return 0;
}
