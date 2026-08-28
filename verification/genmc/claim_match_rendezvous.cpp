// GenMC harness: the match-dedup rendezvous claims exactly once, and NEVER drops on collision.
//
// IT RUNS THE ENGINE'S OWN RULE, over the real ConcurrentMap. claim_match's loop is
// hgcommon/dedup_claim_core.hpp and this drives THAT body, so the harness cannot describe a
// rendezvous the engine has stopped running. It could not include the rule in place for a real
// reason: claim_match is a member of ParallelEvolutionEngine, whose header the interpreter cannot
// take (1130 lines, <thread>) -- which is why the decision is separable and the storage is not.
//
// The Ops below supply the set, the probe-key derivation and the content comparison. The
// arena-backed make_stable is a plain pointer here: the allocation strategy is not what is under
// test, the exchange discipline is.
//
// TWO PROPERTIES, each the site of a shipped defect class:
//
//   P1  EXACTLY-ONCE. Two threads claiming the SAME match (same hash, same content) agree on one
//       winner. Both winning double-applies a rewrite; both losing drops a match, and forwarding
//       is inductive -- a dropped match deletes its whole subtree while the run stays
//       self-consistent.
//
//   P2  NO DROP ON COLLISION (#74's root). Two DIFFERENT matches with the SAME 64-bit hash must
//       BOTH win: the original code dropped the second on hash equality alone, silently losing
//       real matches. The probe walk is the fix, and this enumerates every interleaving of two
//       colliding claims racing through it.
//
// CALIBRATED against P2's own history. HG_CALIBRATE_DEDUP_HASH_ONLY treats a probe hit as a
// duplicate without comparing contents -- deciding on hash equality alone, which is what dropped
// real matches -- and the checker must report a safety violation. The defect is reinstated in the
// SHARED body, in the real path, rather than in a copy of it that could drift from what ships.
//
// GENMC-ARGS: --disable-estimation

#include <pthread.h>
#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"
#include "hgcommon/dedup_claim_core.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {

struct Rec { uint32_t a, b; };
bool recs_equal(const Rec& x, const Rec& y) { return x.a == y.a && x.b == y.b; }

// The engine's own constants. MATCH_MAP's sentinels are the map's EMPTY/LOCKED defaults (0 and
// ~0), which the derivation dodges by construction.
constexpr uint32_t kMaxDedupProbes = 8;
uint64_t dedup_probe_key(uint64_t h, uint32_t n) {
    uint64_t k = h + static_cast<uint64_t>(n) * 0x9E3779B97F4A7C15ull;
    if (k == 0 || k == ~0ull) k += 0x9E3779B97F4A7C15ull;
    return k;
}

using Map = hypergraph::ConcurrentMap<uint64_t, const Rec*>;
Map* g_map;
Rec g_recs[2];

struct Ops {
    uint64_t    h;
    const Rec&  rec;
    const Rec*  stable_src;
    const Rec*  stable = nullptr;

    uint32_t max_probes() const { return kMaxDedupProbes; }
    uint64_t probe_key(uint32_t n) const { return dedup_probe_key(h, n); }

    hgcommon::ProbeState probe(uint64_t key) const {
        auto seen = g_map->lookup(key);
        if (!seen) return hgcommon::ProbeState::Miss;
        if (*seen && recs_equal(**seen, rec)) return hgcommon::ProbeState::Duplicate;
        return hgcommon::ProbeState::Collision;
    }

    void make_stable() { stable = stable_src; }

    hgcommon::ClaimState offer(uint64_t key) {
        auto [existing, inserted] = g_map->insert_if_absent(key, stable);
        if (inserted) return hgcommon::ClaimState::Won;
        if (existing && recs_equal(*existing, rec)) return hgcommon::ClaimState::Duplicate;
        return hgcommon::ClaimState::Collision;
    }

    void note_collision() {}
    void note_exhausted() {}
};

bool claim_match(uint64_t h, const Rec& rec, const Rec* stable_src) {
    Ops ops{h, rec, stable_src};
    return hgcommon::dedup_claim(ops);
}

bool g_won[2];
const uint64_t kHash = 0x1234;

void* same_rec_worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    g_won[id] = claim_match(kHash, g_recs[0], &g_recs[0]);
    return nullptr;
}

void* diff_rec_worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    g_won[id] = claim_match(kHash, g_recs[id], &g_recs[id]);   // same hash, different content
    return nullptr;
}

}  // namespace

int main() {
    g_recs[0] = {1, 10};
    g_recs[1] = {2, 20};

    // P1: one match, two claimants, exactly one winner.
    {
        Map map(8);
        g_map = &map;
        pthread_t t0, t1;
        pthread_create(&t0, nullptr, same_rec_worker, reinterpret_cast<void*>(0L));
        pthread_create(&t1, nullptr, same_rec_worker, reinterpret_cast<void*>(1L));
        pthread_join(t0, nullptr);
        pthread_join(t1, nullptr);
        assert(g_won[0] != g_won[1]);
    }

    // P2: two matches, one hash, both must win -- a loser here is a silently lost match.
    {
        Map map(8);
        g_map = &map;
        pthread_t t0, t1;
        pthread_create(&t0, nullptr, diff_rec_worker, reinterpret_cast<void*>(0L));
        pthread_create(&t1, nullptr, diff_rec_worker, reinterpret_cast<void*>(1L));
        pthread_join(t0, nullptr);
        pthread_join(t1, nullptr);
        assert(g_won[0] && g_won[1]);
    }
    return 0;
}
