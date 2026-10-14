#pragma once
#include "hgcommon/namespace.hpp"
// THE SAMPLING DECISIONS, one body for host and device.
//
// Which transitions a thinned run keeps is a DECISION, not a storage question, so it is spelled
// once. It was spelled once and only the host could reach it: `TransitionRate` and `RuleWeights`
// were accepted, applied on the CPU, and reported as unimplemented on the GPU -- not because the
// device lacks anything, but because the rule lived in ParallelEvolutionEngine where no kernel
// can call it. Everything below is a pure function of values both engines already hold.
//
// THE DRAW IS KEYED ON THE TRANSITION, NEVER ON THREAD STATE. Drawing from a worker RNG would
// make the surviving subgraph depend on which thread reached the transition first, so the same
// run would sample differently at a different thread count -- and on a different DEVICE -- and
// "representative sample" would have nothing stable to be representative of. Keyed this way, the
// same seed gives the same subgraph at any worker count and on either engine, which is what
// makes a CPU/GPU differential test of a SAMPLED run meaningful at all.

#include <cstdint>

#include "hgcommon/core.hpp"
#include "hgcommon/ir_core.hpp"  // ir_heapsort_idx

namespace HG_NAMESPACE {
namespace common {

// The rate a rule's transitions survive at: the run's rate scaled by that rule's weight, clamped.
// `weights` may be null (every rule weighted 1) or shorter than the rule set (rules past its end
// are weighted 1), which is what a partial override means on the WL side.
HG_HD inline double sampling_rate_for_rule(double transition_rate, const double* weights,
                                           uint32_t num_weights, uint32_t rule) {
    double w = 1.0;
    if (weights != nullptr && rule < num_weights) w = weights[rule];
    const double r = transition_rate * w;
    return r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
}

// Does this transition survive the draw?
//
// `transition_key` is the transition's isomorphism-invariant identity -- on both engines
// event_signature(EVENT_SIG_TRANSITION, ...) over the input state's canonical hash, the rule, and
// the consumed edges' canonical ranks WITHIN that state. Two runs that reach the same transition
// compute the same key, which is the whole point.
HG_HD inline bool transition_survives(uint64_t transition_key, uint64_t random_seed, double rate) {
    if (rate >= 1.0) return true;
    if (rate <= 0.0) return false;

    // splitmix64 of (seed, transition). The seed is mixed by multiplication rather than xor so
    // that seed 0 is not the identity -- a caller who leaves the seed alone still gets a draw.
    uint64_t x = transition_key ^ (random_seed * 0x9E3779B97F4A7C15ULL);
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= (x >> 31);

    // Compared in [0,1) via the top 53 bits, so the threshold means the same thing at any rate.
    const double u = static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
    return u < rate;
}

// THE ORDER A PER-(state, rule) CAP KEEPS ITS k IN. Seeded, so a different seed keeps a
// different k rather than the same skeleton every run, and derived from the transition's own
// identity, so the kept set does not depend on which worker or which DEVICE reached it first.
//
// A DIFFERENT MIX FROM transition_survives, deliberately: the same key must not produce a rank
// correlated with whether it survived a rate draw, or the two controls would compound instead
// of composing.
HG_HD inline uint64_t transition_rank(uint64_t transition_key, uint64_t random_seed) {
    uint64_t x = transition_key ^ (random_seed * 0x9E3779B97F4A7C15ULL) ^ 0xA5A5A5A5A5A5A5A5ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Whether a cap of k keeps a candidate that `below` candidates rank strictly below. Over a
// rank-sorted list these are the cap_keep_count(total, k, equal rank) first entries: the first k
// and every entry tied with the k-th. transition_rank is a bijection of the transition key, so
// equal ranks are the host's ties on (rank, key). The device's per-rule, per-state and per-step
// caps keep by this predicate, or by rank_cut_keeps, which is the same set.
HG_HD inline bool cap_keeps(uint64_t below, uint64_t k) { return below < k; }

// THE CUT OF A k-LOWEST SELECTION, ranks counted with multiplicity: the k-th smallest of
// ranks[0, n). `all` when n <= k. rank_cut_keeps keeps every entry whose rank is at or below the
// cut, the entries with fewer than k ranked strictly below them (cap_keeps).
struct RankCut {
    uint64_t rank;
    bool     all;
};

HG_HD inline bool rank_cut_keeps(RankCut c, uint64_t rank) { return c.all || rank <= c.rank; }

HG_HD inline RankCut rank_cut(const uint64_t* ranks, uint32_t n, uint32_t k) {
    if (n <= k) return RankCut{~0ULL, true};
    uint32_t below = 0;
    uint64_t floor = 0;
    bool have_floor = false;
    for (;;) {
        // The next distinct rank above the floor, and how many entries hold it. One exists:
        // `below` < k < n.
        uint64_t next = 0;
        uint32_t count = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint64_t r = ranks[i];
            if (have_floor && r <= floor) continue;
            if (count == 0 || r < next) { next = r; count = 1; }
            else if (r == next) ++count;
        }
        if (below + count >= k) return RankCut{next, false};
        below += count;
        floor = next;
        have_floor = true;
    }
}

// The ExplorationProbability coin: whether a state is expanded, drawn on an isomorphism-invariant
// key -- the class's canonical hash under quotient exploration, the creating transition's key
// under full capture -- and the seed, on a stream of its own so it does not correlate with the
// transition draw on the same key.
HG_HD inline bool explore_survives(uint64_t invariant_key, uint64_t seed, double probability) {
    if (probability >= 1.0) return true;
    if (probability <= 0.0) return false;
    constexpr uint64_t kStateStream = 0xD1B54A32D192ED03ULL;
    uint64_t x = (invariant_key ^ kStateStream) ^ (seed * 0x9E3779B97F4A7C15ULL);
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= (x >> 31);
    const double u = static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
    return u < probability;
}

// Whether a run chooses transitions by transition_rank once the matches they are chosen from are
// complete, instead of taking each as it is found: k per rule of a state (MatchesPerStateRule)
// and k per state (MaxSuccessorStatesPerParent), chosen at the state's drain; N per step
// (MaxStatesPerStep), chosen once the step's matching is complete.
HG_HD inline uint32_t drain_selects(uint32_t matches_per_state_rule,
                                    uint32_t successors_per_parent, uint32_t states_per_step) {
    return (matches_per_state_rule != 0u || successors_per_parent != 0u ||
            states_per_step != 0u) ? 1u : 0u;
}

// How many candidates a cap of k keeps from `total` in rank order: k, extended over every
// candidate after the k-th that ties it. `tied(i, j)` is whether candidates i and j have equal
// (rank, key). Tied candidates are automorphic transitions of one class, and list order decides
// which of them falls at the cut; list order is set by the schedule, and the raw causal relation
// depends on which one is applied, so a cut never falls inside a tie. Used by
// MatchesPerStateRule (per rule), MaxSuccessorStatesPerParent and MaxStatesPerStep.
template <class Tied>
HG_HD inline uint64_t cap_keep_count(uint64_t total, uint64_t k, Tied&& tied) {
    if (k >= total) return total;
    if (k == 0) return 0;
    uint64_t n = k;
    while (n < total && tied(n - 1, n)) ++n;
    return n;
}

// Whether ANY draw can fail. Testing `transition_rate < 1` alone would skip sampling entirely for
// a caller who left the rate at 1 and weighted a single rule to zero. `drain_selection` is
// drain_selects above.
HG_HD inline bool sampling_active(double transition_rate, const double* weights,
                                  uint32_t num_weights, uint32_t drain_selection) {
    if (transition_rate < 1.0) return true;
    if (drain_selection != 0) return true;
    for (uint32_t i = 0; weights != nullptr && i < num_weights; ++i)
        if (weights[i] < 1.0) return true;
    return false;
}

// Whether a run's transition key reads the matched edges' automorphism orbits, and so whether
// every state computes its orbit table. Under Full states a draw or a cap keys on orbits
// (canonical_transition_key, transition_key_device): automorphic transitions share the key, so
// the kept set is invariant under the state's automorphisms, which the quotient reconstruction
// needs to replay it onto every instance of the class.
HG_HD inline bool transition_key_reads_orbits(bool full_states, bool sampling) {
    return full_states && sampling;
}

// THE SUCCESSOR CAP. MaxSuccessorStatesPerParent (over one state's transitions) and
// MaxStatesPerStep (over one step's) keep every transition into the k lowest-ranked distinct
// successor states. Successors are told apart by the run's state identity:
//   None       every transition makes its own state, so the cap counts transitions;
//   Automatic  the successor's content, read through SuccessorCursor (content_hash_of); a
//              transition that creates a vertex makes a state no other transition makes;
//   Full       the successor's canonical hash, from IR over SuccessorCursor.
// A successor's rank is the smallest transition_rank among the transitions into it. The cut is
// rank_cut over one rank per successor: the k lowest and every successor tied with the k-th.
//
// `unique`: the transition's successor is its own; otherwise `key` is the successor's 64-bit
// identity hash, and two transitions with equal keys have one successor.
struct SuccessorIdentity {
    uint64_t key;
    uint32_t unique;
};

HG_HD inline bool same_successor(SuccessorIdentity a, SuccessorIdentity b) {
    return a.unique == 0u && b.unique == 0u && a.key == b.key;
}

// Each candidate's successor rank: out[i] is the smallest rank[j] over the candidates j with i's
// successor. lead[i] is 1 for exactly one candidate of each successor, so the ranks at the lead
// candidates are one per successor (successor_cap_keeps' input). `order` is n words of scratch.
// O(n log n): one sort of the candidates by successor.
HG_HD inline void successor_ranks(const uint64_t* rank, const SuccessorIdentity* id, uint32_t n,
                                  uint32_t* order, uint64_t* out, uint8_t* lead) {
    for (uint32_t i = 0; i < n; ++i) order[i] = i;
    // Unique successors sort by index (each its own run), shared ones by key.
    ir_heapsort_idx(order, n, [&](uint32_t a, uint32_t b) {
        const uint64_t ka = id[a].unique ? a : id[a].key, kb = id[b].unique ? b : id[b].key;
        if (id[a].unique != id[b].unique) return id[a].unique < id[b].unique ? -1 : 1;
        if (ka != kb) return ka < kb ? -1 : 1;
        return a < b ? -1 : (a > b ? 1 : 0);
    });
    uint32_t s = 0;
    while (s < n) {
        uint32_t e = s + 1;
        while (e < n && same_successor(id[order[s]], id[order[e]])) ++e;
        uint64_t m = rank[order[s]];
        uint32_t first = order[s];
        for (uint32_t j = s + 1; j < e; ++j) {
            if (rank[order[j]] < m) m = rank[order[j]];
            if (order[j] < first) first = order[j];
        }
        for (uint32_t j = s; j < e; ++j) {
            out[order[j]] = m;
            lead[order[j]] = order[j] == first ? 1u : 0u;
        }
        s = e;
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
