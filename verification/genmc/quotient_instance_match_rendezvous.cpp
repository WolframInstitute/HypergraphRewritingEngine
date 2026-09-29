// GenMC harness: the quotient replay's (instance, match) rendezvous never drops a pair.
//
// WHAT IS BEING PROVED, and why nothing proved it. Under quotient exploration every raw event is
// reconstructed by applying a captured MATCH to an INSTANCE, and neither side exists first. Both
// sides therefore publish and then scan for the other:
//
//   instance side (Hypergraph::qc_add_instance)      match side (Hypergraph::qc_capture_expansion)
//     insert the shard entry into qc_instances_        insert the match list into qc_expansion_
//     push the instance to its worker's shard          push the match
//     seq_cst fence                                    seq_cst fence
//     look up qc_expansion_ and scan it                look up qc_instances_; for each shard
//                                                      that is not empty, scan the first here
//                                                      and hand the rest to qc_spawn_ jobs
//
// A job scans after the capture's fence in happens-before order but on another thread. The
// capture decides which shards to hand off from its own empty() reads after its fence, so a
// shard it read as empty is one whose instance scans the match itself.
//
// If BOTH scans miss, the pair is never applied: one fewer raw event, and with it every causal
// and branchial pair that event belonged to. The canonical state and event counts are untouched,
// because the state was still explored -- so the loss is invisible to every count a caller reads.
//
// claim_match_rendezvous models the MATCH-DEDUP rendezvous in parallel_evolution.hpp. This one
// is a different rendezvous in a different file over two different maps, and it had no harness.
//
// WHY THE FENCES ARE NOT OBVIOUSLY ENOUGH. The scan does not read the peer's list directly: it
// reaches it through a ConcurrentMap lookup, and that lookup answers ABSENT for an entry whose
// value has been claimed but not yet settled. So a scan can miss its peer for a reason the
// fences say nothing about, and the informal argument -- a lookup that misses implies the peer
// has not published, so the peer's own scan runs later and catches it -- chains a liveness claim
// onto an ordering one. That is the argument this harness exists to check rather than believe.
//
// WHAT IS BOUNDED. One class, one match and two instances in two shards, so a capture that sees
// both hands one to a job: three threads plus the job, which the capture starts with
// pthread_create as its happens-before edge (the job system's push and pop). Over the REAL
// ConcurrentMap and the REAL LockFreeList. A statement about every execution of THIS program
// under RC11, not about unbounded thread counts.
//
// AND THE PAIR IS APPLIED ONCE. Both sides may see each other, so each claims the pair, and the
// claim decides which one applies it. An instance claims a match of its class in its own bits
// when the match's per-class index is below the instance's claim_cap (qr_claim_bits of
// qr_claim_words of the class's match count read when the instance was created), and in the
// shared key set otherwise
// (Hypergraph::QrCtx::claim). The two sides choose the same place because they compare the same
// two values, each written before its record is published; exactly one claim must win.
//
// CALIBRATION -- the harness must be able to fail. -DCALIBRATE_NO_MATCH_FENCE removes the match
// side's seq_cst fence and -DCALIBRATE_NO_INSTANCE_FENCE the instance side's: the publishes and
// the scans then interleave so that each scan runs before the other's publish is visible. -DCALIBRATE_SPLIT_CLAIM makes the match side always claim in the
// key set, so the two sides claim in different places and both win. A harness that cannot fail
// proves nothing.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh quotient_instance_match_rendezvous

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <new>

#include "genmc_support.hpp"
#include "hypergraph/concurrent_map.hpp"
#include "hypergraph/concurrent_key_set.hpp"
#include "hypergraph/lock_free_list.hpp"
#include "hgcommon/quotient_replay_core.hpp"

namespace {

using List = hypergraph::LockFreeList<uint64_t>;
struct Shards { List list[2]; };                     // QcInstanceShards, two shards
using InstMap  = hypergraph::ConcurrentMap<uint64_t, Shards*>;
using MatchMap = hypergraph::ConcurrentMap<uint64_t, List*>;

// Exclusive by construction, as in the other list harnesses: one slot per call, no reuse. The
// arena's own disjointness is a separate property with its own harness.
struct StubArena {
    static constexpr int kCap = 8;
    alignas(16) unsigned char storage[kCap * 64];
    std::atomic<int> next{0};
    template <typename T, typename... Args>
    T* create(Args&&... args) {
        const int i = next.fetch_add(1, std::memory_order_relaxed);
        assert(i < kCap && sizeof(T) <= 64);
        return new (storage + i * 64) T(static_cast<Args&&>(args)...);
    }
};

constexpr uint64_t kClass = 0x51ull;   // the one class every side keys on
constexpr uint64_t kMatch = 22;        // instances are 0 and 1, pushed to shards 0 and 1

InstMap*  g_instances;   // qc_instances_ : class -> shards of instances
MatchMap* g_matches;     // qc_expansion_ : class -> list of matches
Shards*   g_shards;
List*     g_match_list;
StubArena* g_arena;

// The class's match count (QcExpansion::n), each instance's record and the match's record.
// Each record is written before its id is pushed, and read after the id is seen in the list.
std::atomic<uint32_t> g_class_nmatch{0};
struct InstRec { uint32_t claim_cap = 0; std::atomic<uint64_t> bits{0}; };
InstRec g_inst_rec[2];
uint32_t g_match_local = 0;
hypergraph::ConcurrentKeySet<uint64_t>* g_applied;
std::atomic<int> g_wins[2];

void fence(bool on) {
    if (on) std::atomic_thread_fence(std::memory_order_seq_cst);
}

// Hypergraph::QrCtx::claim for the pair (inst, kMatch).
void claim(uint64_t inst, bool match_side) {
    bool won;
#if defined(CALIBRATE_SPLIT_CLAIM)
    const bool bits = !match_side && g_match_local < g_inst_rec[inst].claim_cap;
#else
    (void)match_side;
    const bool bits = g_match_local < g_inst_rec[inst].claim_cap;
#endif
    if (bits) {
        const uint64_t bit = uint64_t{1} << g_match_local;
        won = (g_inst_rec[inst].bits.fetch_or(bit, std::memory_order_acq_rel) & bit) == 0;
    } else {
        won = g_applied->insert((inst << 32) | kMatch);
    }
    if (won) g_wins[inst].fetch_add(1, std::memory_order_relaxed);
}

// The instance side. Publish the shard entry, push the instance to its shard, fence, then scan
// for matches.
void instance_side(uint64_t inst) {
    g_inst_rec[inst].claim_cap = hgcommon::qr_claim_bits(
        hgcommon::qr_claim_words(g_class_nmatch.load(std::memory_order_acquire)));
    g_instances->insert_if_absent(kClass, g_shards);
    g_shards->list[inst].push(inst, *g_arena);
#if defined(CALIBRATE_NO_INSTANCE_FENCE)
    fence(false);
#else
    fence(true);
#endif
    if (auto r = g_matches->lookup(kClass)) {
        (*r)->for_each([&](uint64_t v) { if (v == kMatch) claim(inst, false); });
    }
}
void* instance0(void*) { instance_side(0); return nullptr; }
void* instance1(void*) { instance_side(1); return nullptr; }

// Hypergraph::qc_apply_list: look the shards up again and apply the match to one of them.
void apply_list(uint32_t l) {
    if (auto r = g_instances->lookup(kClass)) {
        (*r)->list[l].for_each([&](uint64_t inst) { claim(inst, true); });
    }
}
void* job(void* arg) {
    apply_list(static_cast<uint32_t>(reinterpret_cast<long>(arg)));
    return nullptr;
}

// The match side: publish, fence, then the first non-empty shard here and the rest as jobs.
void* match_side(void*) {
    g_match_local = g_class_nmatch.fetch_add(1, std::memory_order_acq_rel);
    g_matches->insert_if_absent(kClass, g_match_list);
    g_match_list->push(kMatch, *g_arena);
#if defined(CALIBRATE_NO_MATCH_FENCE)
    fence(false);
#else
    fence(true);
#endif
    pthread_t spawned;
    bool spawned_one = false;
    if (auto r = g_instances->lookup(kClass)) {
        bool ran_one = false;
        for (uint32_t l = 0; l < 2; ++l) {
            if ((*r)->list[l].empty()) continue;
            if (ran_one) {
                pthread_create(&spawned, nullptr, job, reinterpret_cast<void*>(static_cast<long>(l)));
                spawned_one = true;
                continue;
            }
            ran_one = true;
            apply_list(l);
        }
    }
    if (spawned_one) pthread_join(spawned, nullptr);
    return nullptr;
}

}  // namespace

int main() {
    StubArena arena;
    InstMap instances(8);
    MatchMap matches(8);
    Shards shards;
    List match_list;
    hypergraph::ConcurrentKeySet<uint64_t> applied(8);
    g_applied = &applied;
    g_arena = &arena;
    g_instances = &instances;
    g_matches = &matches;
    g_shards = &shards;
    g_match_list = &match_list;

    pthread_t t0, t1, t2;
    pthread_create(&t0, nullptr, instance0, nullptr);
    pthread_create(&t1, nullptr, instance1, nullptr);
    pthread_create(&t2, nullptr, match_side, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    // EACH PAIR IS APPLIED EXACTLY ONCE. A missed pair is one raw event that never happens, and
    // with it every causal and branchial pair it belonged to -- while the state and canonical
    // event counts a caller reads stay exactly as they were.
    assert(g_wins[0].load(std::memory_order_relaxed) == 1 && "instance 0 and the match: not once");
    assert(g_wins[1].load(std::memory_order_relaxed) == 1 && "instance 1 and the match: not once");
    return 0;
}
