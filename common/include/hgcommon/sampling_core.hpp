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
//
// `step` is rank_step's value: the transition's step when every state is matched at one step,
// 0 otherwise. At step 0 the draw is the step-free draw.
HG_HD inline bool transition_survives(uint64_t transition_key, uint64_t random_seed, double rate,
                                      uint32_t step) {
    if (rate >= 1.0) return true;
    if (rate <= 0.0) return false;

    // splitmix64 of (seed, step, transition). The seed is mixed by multiplication rather than xor
    // so that seed 0 is not the identity -- a caller who leaves the seed alone still gets a draw.
    uint64_t x = transition_key ^ (random_seed * 0x9E3779B97F4A7C15ULL) ^
                 (static_cast<uint64_t>(step) * 0xD6E8FEB86659FD93ULL);
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
//
// `step` is rank_step's value: the transition's step when every state is matched at one step, 0
// otherwise. At step 0 the rank is the step-free rank.
HG_HD inline uint64_t transition_rank(uint64_t transition_key, uint64_t random_seed,
                                      uint32_t step) {
    uint64_t x = transition_key ^ (random_seed * 0x9E3779B97F4A7C15ULL) ^ 0xA5A5A5A5A5A5A5A5ULL ^
                 (static_cast<uint64_t>(step) * 0xC2B2AE3D27D4EB4FULL);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// The step a transition's rank and draw read. Under a step cap, or under None or Automatic states,
// every state is matched at one step, so the rank and the draw mix that step in: a history that
// returns to an isomorphic state draws again there. Under Full states without a step cap a state
// is its isomorphism class, and the rank and the draw read 0, so a class's transitions are drawn
// the same at every step it is reached, as quotient exploration draws them (docs/SPEC.md §5.4).
// `step` is the step of the transition's event. `reads_step` is rank_reads_step's value.
HG_HD inline uint32_t rank_step(uint32_t step, uint32_t reads_step) {
    return reads_step != 0u ? step : 0u;
}

// Whether ranks and draws read the step (rank_step).
HG_HD inline uint32_t rank_reads_step(bool full_states, uint64_t max_states_per_step) {
    return (max_states_per_step != 0u || !full_states) ? 1u : 0u;
}

// THE "MatchesPerStep" SELECTION. Without "StateCost" a step keeps n of its T transitions, drawn
// uniformly without replacement from the seed and the step. The step's transitions are ordered
// by (group, key) (step_order_before), and the kept transitions are those at the drawn
// positions, each with every transition of equal (group, key).
//
// The key is fixed when the transition is found and does not change from one step to the next,
// and the group is its source state's. Under None states the key is lineage_transition_key and
// the group the state's lineage label. Under Automatic and Full states the key is the canonical
// transition key and the group 0: two transitions with equal keys are automorphic, in one state
// or in two isomorphic states.
struct StepOrder {
    uint64_t group;
    uint64_t key;
};

HG_HD inline bool step_order_before(StepOrder a, StepOrder b) {
    return a.group != b.group ? a.group < b.group : a.key < b.key;
}

HG_HD inline bool step_order_equal(StepOrder a, StepOrder b) {
    return a.group == b.group && a.key == b.key;
}

// A transition's rank under "MatchesPerStep": the order of the drain caps and, under "StateCost",
// of successors of equal cost.
HG_HD inline uint64_t step_order_rank(StepOrder o, uint64_t random_seed, uint32_t step) {
    return transition_rank(o.key ^ o.group, random_seed, step);
}

// The j-th draw of a step's selection.
HG_HD inline uint64_t step_position_draw(uint64_t random_seed, uint32_t step, uint64_t j) {
    constexpr uint64_t kPositionStream = 0x8EBC6AF09C88C6E3ULL;
    return splitmix64(splitmix64((random_seed * 0x9E3779B97F4A7C15ULL) ^ kPositionStream ^
                                 (static_cast<uint64_t>(step) * 0xC2B2AE3D27D4EB4FULL)) ^ j);
}

// The positions in [0, total) a step keeps: min(n, total) of them in out[], ascending, drawn by
// Floyd's algorithm. O(n^2) in n, independent of `total`. Returns how many.
HG_HD inline uint32_t step_positions(uint64_t random_seed, uint32_t step, uint64_t total,
                                     uint32_t n, uint64_t* out) {
    if (total <= n) {
        for (uint64_t i = 0; i < total; ++i) out[i] = i;
        return static_cast<uint32_t>(total);
    }
    uint32_t k = 0;
    for (uint64_t j = total - n; j < total; ++j) {
        const uint64_t t = step_position_draw(random_seed, step, j) % (j + 1u);
        bool have = false;
        for (uint32_t i = 0; i < k && !have; ++i) have = out[i] == t;
        out[k++] = have ? j : t;
    }
    for (uint32_t i = 1; i < k; ++i) {
        const uint64_t v = out[i];
        uint32_t j = i;
        while (j > 0 && out[j - 1] > v) { out[j] = out[j - 1]; --j; }
        out[j] = v;
    }
    return k;
}

// THE EVENT ORDER ("Generational", "EventOrderingFunction"): SetReplace's event ordering
// functions (libSetReplace/HypergraphMatcher.cpp, MatchComparator). An order is a list of keys,
// each 4 bits of a uint32_t from the low end, 0 ending the list. A key compares one function of
// two matches: their edge ids sorted ascending ("OldestEdge" 1, reversed "LeastOldEdge" 2),
// sorted descending ("LeastRecentEdge" 3, reversed "NewestEdge" 4), in pattern order
// ("RuleOrdering" 5, reversed "ReverseRuleOrdering" 6), or their rule index ("RuleIndex" 7,
// reversed "ReverseRuleIndex" 8). Two id lists compare lexicographically, and when one is a
// prefix of the other the shorter comes first. The first key that tells the matches apart
// decides. An edge's age is its id: under both modes both engines issue edge ids in creation
// order, the initial edges in input order and then the produced edges in the order of the events
// and of each right-hand side.
enum : uint32_t {
    kEventOrderOldestEdge = 1,
    kEventOrderLeastOldEdge = 2,
    kEventOrderLeastRecentEdge = 3,
    kEventOrderNewestEdge = 4,
    kEventOrderRuleOrdering = 5,
    kEventOrderReverseRuleOrdering = 6,
    kEventOrderRuleIndex = 7,
    kEventOrderReverseRuleIndex = 8,
};

// SetReplace's default order: "LeastRecentEdge", "RuleOrdering", "RuleIndex".
constexpr uint32_t kStandardEventOrder =
    kEventOrderLeastRecentEdge | (kEventOrderRuleOrdering << 4) | (kEventOrderRuleIndex << 8);

// -1, 0 or 1 as match a comes before, ties with, or comes after match b under `order`.
HG_HD inline int event_order_compare(uint32_t order, const uint32_t* a, uint32_t na,
                                     uint32_t rule_a, const uint32_t* b, uint32_t nb,
                                     uint32_t rule_b) {
    constexpr uint32_t kMax = 16;
    const uint32_t ma = na < kMax ? na : kMax, mb = nb < kMax ? nb : kMax;
    auto sort = [](const uint32_t* in, uint32_t* v, uint32_t n, bool desc) {
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t x = in[i];
            uint32_t j = i;
            while (j > 0 && (desc ? v[j - 1] < x : v[j - 1] > x)) { v[j] = v[j - 1]; --j; }
            v[j] = x;
        }
    };
    auto lex = [](const uint32_t* x, uint32_t nx, const uint32_t* y, uint32_t ny) -> int {
        for (uint32_t i = 0; i < nx && i < ny; ++i)
            if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
        return nx == ny ? 0 : (nx < ny ? -1 : 1);
    };
    for (uint32_t k = order; k != 0; k >>= 4) {
        const uint32_t key = k & 0xFu;
        const uint32_t function = (key - 1u) >> 1;
        int c = 0;
        if (function <= 1u) {
            uint32_t sa[kMax], sb[kMax];
            sort(a, sa, ma, function == 1u);
            sort(b, sb, mb, function == 1u);
            c = lex(sa, ma, sb, mb);
        } else if (function == 2u) {
            c = lex(a, ma, b, mb);
        } else {
            c = rule_a < rule_b ? -1 : (rule_a > rule_b ? 1 : 0);
        }
        if (c != 0) return ((key - 1u) & 1u) ? -c : c;
    }
    return 0;
}

HG_HD inline bool event_order_before(uint32_t order, const uint32_t* a, uint32_t na,
                                     uint32_t rule_a, const uint32_t* b, uint32_t nb,
                                     uint32_t rule_b) {
    return event_order_compare(order, a, na, rule_a, b, nb, rule_b) < 0;
}

// The code of an "EventOrderingFunction" name, 0 for a name that is not one.
inline uint32_t event_order_code(const char* name) {
    constexpr const char* kNames[8] = {"OldestEdge",   "LeastOldEdge",        "LeastRecentEdge",
                                       "NewestEdge",   "RuleOrdering",        "ReverseRuleOrdering",
                                       "RuleIndex",    "ReverseRuleIndex"};
    for (uint32_t i = 0; i < 8; ++i) {
        const char* p = kNames[i];
        const char* q = name;
        while (*p != '\0' && *p == *q) { ++p; ++q; }
        if (*p == '\0' && *q == '\0') return i + 1u;
    }
    return 0;
}

// `order` with key `code` appended. A key whose function is already in the order compares
// matches the order has already found equal, so it is left out.
inline uint32_t event_order_append(uint32_t order, uint32_t code) {
    uint32_t n = 0;
    for (uint32_t k = order; k != 0; k >>= 4, ++n)
        if (((k & 0xFu) - 1u) >> 1 == (code - 1u) >> 1) return order;
    return order | (code << (4u * n));
}

// Whether `order` tells every two distinct matches of `num_rules` rules apart. Two matches are
// the same match when they have the same rule and the same edges in pattern order. So the order
// needs "RuleOrdering" or "ReverseRuleOrdering", and with two or more rules "RuleIndex" or
// "ReverseRuleIndex".
inline bool event_order_total(uint32_t order, uint32_t num_rules) {
    bool pattern = false, rule = num_rules <= 1u;
    for (uint32_t k = order; k != 0; k >>= 4) {
        const uint32_t function = ((k & 0xFu) - 1u) >> 1;
        pattern = pattern || function == 2u;
        rule = rule || function == 3u;
    }
    return pattern && rule;
}

// A GENERATION'S EVENTS: the matches taken in event order, each unless it shares an edge with
// one taken before it, until `limit` are taken. With no limit the set is maximal and its matches
// share no edge. `order[0..n)` lists the matches in event order; `edges_of(m, out)` writes match
// m's edge ids and returns their count; `used(e)` and `use(e)` read and mark an edge; `take(m)`
// takes match m.
template <class EdgesOf, class Used, class Use, class Take>
HG_HD inline void greedy_disjoint(const uint32_t* order, uint32_t n, uint32_t limit,
                                  EdgesOf&& edges_of, Used&& used, Use&& use, Take&& take) {
    uint32_t edges[16];
    uint32_t taken = 0;
    for (uint32_t i = 0; i < n && taken < limit; ++i) {
        const uint32_t k = edges_of(order[i], edges);
        bool free = true;
        for (uint32_t j = 0; j < k && free; ++j) free = !used(edges[j]);
        if (!free) continue;
        for (uint32_t j = 0; j < k; ++j) use(edges[j]);
        take(order[i]);
        ++taken;
    }
}

// LINEAGE LABELS: the identities "MatchesPerStep" orders transitions by under None states.
// An initial state's label comes from its canonical hash, and each of its edges' from the state's
// label and the edge's canonical rank in it. A transition's key mixes the rule and the matched
// edges' labels in pattern order. A rewrite's produced edge at right-hand-side position j has
// lineage_edge(key, j), and the state it makes has lineage_state(parent label, key). Every label
// is a function of the inputs, so both engines and every schedule compute the same ones.
HG_HD inline uint64_t lineage_initial_state(uint64_t canonical_hash) {
    return splitmix64(canonical_hash ^ 0x5851F42D4C957F2DULL);
}

HG_HD inline uint64_t lineage_edge(uint64_t origin, uint32_t index) {
    return splitmix64(origin ^ (static_cast<uint64_t>(index) + 1u) * 0x9E3779B97F4A7C15ULL);
}

HG_HD inline uint64_t lineage_transition_key(uint32_t rule, const uint64_t* labels, uint32_t n) {
    uint64_t h = splitmix64(0x2545F4914F6CDD1DULL ^ rule);
    for (uint32_t i = 0; i < n; ++i) h = splitmix64(h ^ labels[i]);
    return h;
}

HG_HD inline uint64_t lineage_state(uint64_t parent_label, uint64_t transition_key) {
    return splitmix64(parent_label * 0xBF58476D1CE4E5B9ULL ^ transition_key);
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

// rank_cut over the entries i < n with take(i), entry i's rank being rank_at(i).
template <class RankAt, class Take>
HG_HD inline RankCut rank_cut_where(uint32_t n, uint32_t k, RankAt&& rank_at, Take&& take) {
    uint32_t taken = 0;
    for (uint32_t i = 0; i < n; ++i) taken += take(i) ? 1u : 0u;
    if (taken <= k) return RankCut{~0ULL, true};
    uint32_t below = 0;
    uint64_t floor = 0;
    bool have_floor = false;
    for (;;) {
        // The next distinct rank above the floor, and how many entries hold it. One exists:
        // `below` < k < taken.
        uint64_t next = 0;
        uint32_t count = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (!take(i)) continue;
            const uint64_t r = rank_at(i);
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

HG_HD inline RankCut rank_cut(const uint64_t* ranks, uint32_t n, uint32_t k) {
    return rank_cut_where(n, k, [&](uint32_t i) { return ranks[i]; },
                          [](uint32_t) { return true; });
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

// Whether a step's selection hands each kept transition's state's matches to the child it makes,
// which updates them by the rewrite's consumed and produced edges: "MatchesPerStep" under None
// states (lineage labels) without "StateCost", with no other drain selection and no exploration
// coin. One rule for the host's HeldMatches and the device's held sets.
HG_HD inline bool held_matches_apply(bool lineage, bool state_cost, uint32_t matches_per_state_rule,
                                     uint32_t successors_per_parent,
                                     double exploration_probability) {
    return lineage && !state_cost && matches_per_state_rule == 0u && successors_per_parent == 0u &&
           exploration_probability >= 1.0;
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

// The number of rule weights a run's draws read: `num_weights`, or 0 when every rule's rate is 1
// (sampling_rate_for_rule), so no draw can fail.
HG_HD inline uint32_t effective_rule_weights(double transition_rate, const double* weights,
                                             uint32_t num_weights) {
    return sampling_active(transition_rate, weights, num_weights, 0u) ? num_weights : 0u;
}

// Whether a draw can fail, from the rate and effective_rule_weights.
HG_HD inline bool draw_can_fail(double transition_rate, uint32_t num_rule_weights) {
    return transition_rate < 1.0 || num_rule_weights != 0u;
}

// Whether "MatchesPerStep" orders transitions by lineage labels (StepOrder): under None states.
// `step_cap_counts_matches` is set by "MatchesPerStep". One rule for the host's lineage_on_ and the
// device's EngineState::set_lineage.
HG_HD inline bool lineage_applies(bool step_cap_counts_matches, uint64_t max_states_per_step,
                                  bool none_states) {
    return step_cap_counts_matches && max_states_per_step != 0u && none_states;
}

// Whether the ExplorationProbability coin reads the creating transition's key: full capture under
// None and Automatic states. Under Full states and under quotient exploration it reads the class's
// canonical hash.
HG_HD inline bool explore_coin_reads_transition_key(double exploration_probability,
                                                    bool quotient_exploration, bool full_states) {
    return exploration_probability < 1.0 && !quotient_exploration && !full_states;
}

// WHETHER A RUN READS TRANSITION KEYS (canonical_transition_key, transition_key_device), and so each
// state's canonical hash and edge ranks, or its edge orbits under Full states. A drain selection
// (drain_selects) replaces the draw and reads them, except "MatchesPerStep" under None states,
// which orders by lineage labels (lineage_applies). Without one, the draw reads them when one can
// fail (draw_can_fail; `num_rule_weights` is effective_rule_weights). The coin of
// explore_coin_reads_transition_key reads them. "Generational" reads none, and the FFI clears the
// options above under it. One rule for the host's configure and the device's.
HG_HD inline bool run_reads_transition_keys(double transition_rate, uint32_t num_rule_weights,
                                            uint32_t drain_selection, bool lineage,
                                            bool explore_coin) {
    return (drain_selection != 0u ? !lineage : draw_can_fail(transition_rate, num_rule_weights)) ||
           explore_coin;
}

// Whether a run's transition key reads the matched edges' automorphism orbits, and so whether
// every state computes its orbit table. Under Full states a draw or a cap keys on orbits
// (canonical_transition_key, transition_key_device): automorphic transitions share the key, so
// the kept set is invariant under the state's automorphisms, which the quotient reconstruction
// needs to replay it onto every instance of the class. `reads_keys` is run_reads_transition_keys.
HG_HD inline bool transition_key_reads_orbits(bool full_states, bool reads_keys) {
    return full_states && reads_keys;
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

// v[i] for every candidate i becomes v at the lead candidate of i's successor. `order`, `lead`
// and `id` are as successor_ranks leaves them (order sorted by successor).
HG_HD inline void successor_spread(const SuccessorIdentity* id, const uint32_t* order,
                                   const uint8_t* lead, uint32_t n, uint64_t* v) {
    uint32_t s = 0;
    while (s < n) {
        uint32_t e = s + 1;
        while (e < n && same_successor(id[order[s]], id[order[e]])) ++e;
        uint64_t x = v[order[s]];
        for (uint32_t j = s; j < e; ++j)
            if (lead[order[j]]) x = v[order[j]];
        for (uint32_t j = s; j < e; ++j) v[order[j]] = x;
        s = e;
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
