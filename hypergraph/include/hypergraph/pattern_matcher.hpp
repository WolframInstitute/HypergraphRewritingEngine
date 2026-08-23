#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <atomic>
#include <functional>

#include "types.hpp"
#include "signature.hpp"
#include "pattern.hpp"
#include "hgcommon/match_core.hpp"
#include "hgcommon/join_core.hpp"
#include "index.hpp"
#include "arena.hpp"
#include "bitset.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"
#include "concurrent_map.hpp"

namespace HG_NAMESPACE {
namespace engine {

// CANDIDATE-BRANCH COVERAGE. The join reaches candidates three ways and they are not
// interchangeable: an unbound seed edge with no repeated variable scans the state's edges by
// arity, an unbound seed edge WITH a repeated variable scans a signature partition, and a seed
// with bound variables intersects the inverted index. A corpus that never produces a
// repeated-variable seed leaves the middle one unexercised by every bench and every sweep, which
// is what happened until the Repeated shape was added.
//
// Registered as HG_MATCH_BRANCH_STATS in the top-level CMakeLists; never on in a shipping build,
// where HG_MATCH_BRANCH_HIT expands to nothing.
#ifdef HG_MATCH_BRANCH_STATS
inline std::atomic<uint64_t>* match_branch_counters() {
    static std::atomic<uint64_t> c[3];
    return c;
}
#define HG_MATCH_BRANCH_HIT(i) \
    (HG_NAMESPACE::engine::match_branch_counters()[(i)].fetch_add(1, std::memory_order_relaxed))
inline uint64_t match_branch_count(int i) {
    return match_branch_counters()[i].load(std::memory_order_relaxed);
}
#else
#define HG_MATCH_BRANCH_HIT(i) ((void)0)
#endif

// =============================================================================
// Pattern matching on the host
// =============================================================================
//
// The join itself -- recursion, edge-injectivity, binding and unwind, which pattern position is
// bound next -- is hgcommon/join_core.hpp, so the device runs the same one. This file supplies
// the host's half: candidate enumeration from the signature and inverted-vertex indices, and
// what to do with a completed match.
//
// Matching runs synchronously here: the whole depth-first search happens inside one task, and
// only the completed match spawns a job. ParallelEvolutionEngine::execute_scan_task and
// ::execute_expand_task are the other host scheduler, expanding by spawning a task per candidate
// rather than by recursing; they select the next pattern position with the same
// hgcommon::join_next_position this join uses.
//
// =============================================================================
// WHAT THIS JOIN COSTS, AND WHERE IT STOPS BEING OPTIMAL
// =============================================================================
//
// This is an edge-at-a-time backtracking join: the rule's match_order fixes which LHS edge is
// bound at each depth, and each step generates candidates by intersecting the inverted vertex
// index on the variables already bound. In database terms that is a binary join plan executed
// depth-first with index lookups, and compute_match_order picks the best plan of that kind --
// connected, so no step is a cartesian product.
//
// FOR ONE OR TWO LHS EDGES, WHICH IS EVERY RULE IN THE TEST CORPUS, THAT IS OPTIMAL. Every
// two-atom conjunctive query is acyclic (GYO removes each atom's exclusive variables, leaving
// one atom contained in the other), and with a connected order the work is the number of edge
// pairs sharing the bound vertex -- the output size -- plus an O(arity) validation per candidate.
// There is no third atom for a partial match to fail against, so no intermediate result is built
// that the output does not contain.
//
// FOR THREE OR MORE LHS EDGES IT IS NOT OPTIMAL, and MAX_PATTERN_EDGES is 16, so rules can reach
// that. Two separate gaps open:
//
//   Acyclic patterns: a partial match can now fail to extend, so intermediate results are built
//   that never appear in the output. Yannakakis-style semi-join reduction would restore
//   O(input + output); this does not do it.
//
//   Cyclic patterns: no binary plan is optimal at all, whatever the order. The triangle LHS
//   {{x,y},{y,z},{z,x}} over a state of N binary edges has at most N^1.5 matches (the AGM bound),
//   while this join binds N candidates for the first edge and up to N for the second before the
//   third can prune -- Omega(N^2). The gap is polynomial and grows with N. Closing it needs a
//   worst-case-optimal join, which enumerates VARIABLE by variable rather than edge by edge,
//   intersecting the edges that mention the current variable at each step (Leapfrog Triejoin and
//   its relatives). That is a different algorithm class, not a better order: the ordering work
//   already reached the best binary plan, so no further ordering effort can close this.
//
// A worst-case-optimal join would additionally need per-position sorted access -- "given the
// binding so far, seek the next vertex value at this position" -- which InvertedVertexIndex does
// not provide; it answers set containment, not ordered seek.


// =============================================================================
// Candidate Validation
// =============================================================================

// Validate candidate edge against pattern edge, extending binding
// Returns true if validation succeeds, binding is modified in place
bool validate_candidate(
    const VertexId* edge_vertices,
    uint8_t edge_arity,
    const PatternEdge& pattern_edge,
    VariableBinding& binding
);

// =============================================================================
// Pattern Matching Context
// =============================================================================
// Shared context for all tasks in a matching session.

template<typename EdgeAccessor, typename SignatureAccessor = std::function<const EdgeSignature&(EdgeId)>>
struct PatternMatchingContext {
    // Rule being matched
    const RewriteRule* rule;
    uint16_t rule_index;

    // State being matched against
    StateId state_id;
    const SparseBitset* state_edges;  // Bitset of edges in this state

    // Indices for candidate generation
    const SignatureIndex* sig_index;
    const InvertedVertexIndex* inv_index;

    // Edge accessor
    EdgeAccessor get_edge;

    // Signature accessor (cached signatures for O(1) lookup)
    SignatureAccessor get_signature;

    // Per-edge pattern signatures and compatible-signature caches are read directly
    // from `rule` (immutable after RewriteRule::compute_var_counts) — no per-session
    // copy: the context used to embed EdgeSignature[16] + CompatibleSignatureCache[16]
    // (~17 KB) and memcpy them from the rule on every state x rule matching session.

    // Coordination
    std::atomic<bool>* should_terminate;
    std::atomic<size_t>* matches_found;
    size_t max_matches;

    // Match callback: called for each complete match
    // Signature: void(rule_index, edges_in_pattern_order, num_edges, binding, state_id)
    using MatchCallback = std::function<void(
        uint16_t, const EdgeId*, uint8_t, const VariableBinding&, StateId)>;
    MatchCallback on_match;

    PatternMatchingContext(
        const RewriteRule* r,
        uint16_t ridx,
        StateId sid,
        const SparseBitset* edges,
        const SignatureIndex* sig,
        const InvertedVertexIndex* inv,
        EdgeAccessor accessor,
        SignatureAccessor sig_accessor,
        MatchCallback callback
    )
        : rule(r)
        , rule_index(ridx)
        , state_id(sid)
        , state_edges(edges)
        , sig_index(sig)
        , inv_index(inv)
        , get_edge(accessor)
        , get_signature(sig_accessor)
        , should_terminate(nullptr)
        , matches_found(nullptr)
        , max_matches(SIZE_MAX)
        , on_match(callback)
    {}
};

// =============================================================================
// Candidate Generation (HGMatch Algorithm 4)
// =============================================================================

template<typename EdgeAccessor, typename CandidateCallback>
void generate_candidates(
    const PatternEdge& pattern_edge,
    const EdgeSignature& pattern_sig,
    const CompatibleSignatureCache& sig_cache,  // Pre-computed compatible signatures
    const VertexId* bindings,                   // indexed by variable; read only where bound
    uint32_t bound_mask,
    const SparseBitset& state_edges,
    const SignatureIndex& sig_index,
    const InvertedVertexIndex& inv_index,
    const EdgeAccessor& get_edge,
    CandidateCallback&& on_candidate
) {
    // Collect bound vertices and their required positions
    VertexId bound_vertices[MAX_ARITY];
    uint8_t bound_positions[MAX_ARITY];
    uint8_t num_bound = 0;

    for (uint8_t i = 0; i < pattern_edge.arity; ++i) {
        uint8_t var = pattern_edge.var_at(i);
        if (bound_mask & (1u << var)) {
            bound_vertices[num_bound] = bindings[var];
            bound_positions[num_bound] = i;
            num_bound++;
        }
    }

    if (num_bound == 0) {
        if (pattern_sig.num_distinct() == pattern_edge.arity) {
            // All-distinct pattern edge: it imposes no vertex-repetition constraint, so
            // its compatible data signatures are every set-partition of the arity
            // (Bell(k)), whose per-signature edge-lists re-union to exactly the arity-k
            // edges present in this state. The signature index holds whole-evolution
            // history keyed by signature; drawing candidates from it walks that global
            // history filtered by the state bitset. Scan this state's own edges once and
            // keep those of matching arity — the same candidate set in one pass.
            // validate_candidate re-checks arity downstream.
            HG_MATCH_BRANCH_HIT(0);
            const uint8_t want_arity = pattern_edge.arity;
            state_edges.for_each([&](EdgeId eid) {
                const auto& edge = get_edge(eid);
                if (edge.arity == want_arity) {
                    on_candidate(eid, edge);
                }
            });
        } else {
            // Repeated-variable seed edge: the signature level genuinely prunes, so scan
            // the compatible signature partition using the pre-computed cache.
            HG_MATCH_BRANCH_HIT(1);
            sig_index.for_each_candidate_cached(sig_cache, state_edges, [&](EdgeId eid) {
                on_candidate(eid, get_edge(eid));
            });
        }
    } else {
        // Have bound variables: use inverted index intersection. The intersection has
        // already fetched each edge to test containment; it hands that edge to us.
        HG_MATCH_BRANCH_HIT(2);
        inv_index.for_each_edge_containing_all(
            bound_vertices, num_bound, state_edges, get_edge,
            [&](EdgeId eid, const auto& edge) {
                // Check bound vertices at the required positions. No signature test here:
                // validate_candidate (run by on_candidate) binds each variable on first
                // occurrence and checks equality on repeat, which enforces exactly the
                // repetition constraint signature_compatible would — at O(arity) instead
                // of O(arity^2) — and rejects every edge the signature test would.
                bool valid = true;
                for (uint8_t i = 0; i < num_bound && valid; ++i) {
                    if (edge.vertices[bound_positions[i]] != bound_vertices[i]) {
                        valid = false;
                    }
                }

                if (valid) {
                    on_candidate(eid, edge);
                }
            }
        );
    }
}

// =============================================================================
// The join, over this engine's indices
// =============================================================================
// hgcommon/join_core.hpp owns the recursion, the edge-injectivity rule, the binding and its
// unwind, and the order in which pattern positions are bound. This supplies the two things that
// are genuinely host-specific: how candidates are enumerated, and what an emitted match is.

template<typename EdgeAccessor, typename SignatureAccessor>
struct HostJoinContext {
    using JoinState = hgcommon::JoinState<MAX_PATTERN_EDGES, MAX_VARS, EdgeId, VertexId>;

    PatternMatchingContext<EdgeAccessor, SignatureAccessor>* mc;

    // Enumerating a candidate already fetched its edge, so the candidate carries it rather than
    // handing the join an id to look up again.
    //
    // The edge type comes from the accessor, not from types.hpp: the matcher is generic over it
    // (the pattern-matching tests supply their own). The accessor must return a REFERENCE --
    // pointing at a returned temporary would dangle for the whole join.
    using EdgeRef = decltype(std::declval<const EdgeAccessor&>()(EdgeId{}));
    static_assert(std::is_reference_v<EdgeRef>,
                  "edge accessor must return a reference; the join holds the edge across binding");
    using EdgeType = std::remove_cv_t<std::remove_reference_t<EdgeRef>>;

    struct Candidate {
        EdgeId          id;
        const EdgeType* edge;
    };

    uint8_t        num_lhs_edges()          const { return mc->rule->num_lhs_edges; }
    uint8_t        order_at(uint8_t k)      const { return mc->rule->match_order[k]; }
    const uint8_t* pattern_vars(uint8_t p)  const { return mc->rule->lhs[p].vars; }
    uint8_t        pattern_arity(uint8_t p) const { return mc->rule->lhs[p].arity; }

    Candidate candidate_of(EdgeId e) const { return Candidate{e, &mc->get_edge(e)}; }
    EdgeId    candidate_id(const Candidate& c)      const { return c.id; }
    const VertexId* edge_vertices(const Candidate& c) const { return c.edge->vertices; }
    uint8_t         edge_arity(const Candidate& c)    const { return c.edge->arity; }

    // Every enumeration branch in generate_candidates is already filtered by the state's edge
    // bitset, so a candidate reaching the join is in the state by construction. A SEEDED join
    // takes its anchor from the caller, not from the enumerator, so the anchor's membership is
    // checked at the call site (find_delta_matches) instead.
    bool usable(EdgeId) const { return true; }

    bool aborted() const {
        return mc->should_terminate && mc->should_terminate->load();
    }

    template<typename F>
    void for_each_candidate(uint8_t p, const JoinState& st, F&& f) const {
        generate_candidates(
            mc->rule->lhs[p], mc->rule->lhs_sig[p], mc->rule->lhs_cache[p],
            st.binding, st.bound_mask, *mc->state_edges,
            *mc->sig_index, *mc->inv_index, mc->get_edge,
            [&](EdgeId eid, const EdgeType& edge) { f(Candidate{eid, &edge}); });
    }
};

// A completed match: deduplicate, report, count. One body for the full scan and the seeded scan.
//
// `scratch` is INVALID_ID in every slot on entry and is left that way on exit. It belongs to
// the session, not the match, so a completed match costs two writes per bound variable rather
// than the 128-byte memset a fresh VariableBinding would zero-fill.
template<typename EdgeAccessor, typename SignatureAccessor, typename JoinState>
void emit_match(PatternMatchingContext<EdgeAccessor, SignatureAccessor>& mc,
                VariableBinding& scratch,
                const JoinState& st) {
    EdgeId edges_in_order[MAX_PATTERN_EDGES];
    std::memset(edges_in_order, 0xFF, sizeof(edges_in_order));
    for (uint8_t d = 0; d < st.depth; ++d) edges_in_order[st.pattern[d]] = st.matched[d];

    if (mc.on_match) {
        // Only the BOUND variables are copied across. The join's unwind restores the mask and
        // leaves behind the vertex a discarded branch wrote, while resolve_rhs_vertices
        // (hgcommon/rewrite_core.hpp) reads bindings[var] directly and takes INVALID_ID to mean
        // "not matched, allocate a fresh vertex" -- it never consults the mask. So an unbound
        // variable MUST carry INVALID_ID here, or a rule whose RHS introduces that variable
        // would build its edge from a vertex belonging to a match that was rejected.
        for (uint32_t m = st.bound_mask; m; m &= m - 1) {
            const uint8_t var = static_cast<uint8_t>(hgcommon::ctz(m));
            scratch.bind(var, st.binding[var]);
        }
        mc.on_match(mc.rule_index, edges_in_order, mc.rule->num_lhs_edges, scratch, mc.state_id);
        // Back to all-INVALID_ID for the next match. Callers copy the binding rather than
        // retaining a reference to it.
        for (uint32_t m = st.bound_mask; m; m &= m - 1) {
            scratch.unbind(static_cast<uint8_t>(hgcommon::ctz(m)));
        }
    }

    if (mc.matches_found) {
        size_t count = mc.matches_found->fetch_add(1) + 1;
        if (count >= mc.max_matches && mc.should_terminate) {
            mc.should_terminate->store(true);
        }
    }
}

// =============================================================================
// SCAN Task
// =============================================================================
// Every match of the rule in the state. Executes synchronously.

template<typename EdgeAccessor, typename SignatureAccessor>
void scan_pattern(
    PatternMatchingContext<EdgeAccessor, SignatureAccessor>& mc
) {
    if (mc.rule->num_lhs_edges == 0) return;

    HostJoinContext<EdgeAccessor, SignatureAccessor> ctx{&mc};
    typename decltype(ctx)::JoinState st;
    st.reset();
    VariableBinding scratch;
    hgcommon::join_dfs(ctx, st, [&](const auto& s) { emit_match(mc, scratch, s); });
}

// =============================================================================
// Public API
// =============================================================================

// Find all matches for a rule in a state
template<typename EdgeAccessor, typename SignatureAccessor, typename MatchCallback>
void find_matches(
    const RewriteRule& rule,
    uint16_t rule_index,
    StateId state_id,
    const SparseBitset& state_edges,
    const SignatureIndex& sig_index,
    const InvertedVertexIndex& inv_index,
    EdgeAccessor get_edge,
    SignatureAccessor get_signature,
    MatchCallback&& on_match,
    std::atomic<bool>* should_terminate = nullptr,
    std::atomic<size_t>* matches_found = nullptr,
    size_t max_matches = SIZE_MAX
) {
    PatternMatchingContext<EdgeAccessor, SignatureAccessor> ctx(
        &rule, rule_index, state_id, &state_edges,
        &sig_index, &inv_index, get_edge, get_signature,
        std::forward<MatchCallback>(on_match)
    );

    ctx.should_terminate = should_terminate;
    ctx.matches_found = matches_found;
    ctx.max_matches = max_matches;

    scan_pattern(ctx);
}

// Backward-compatible overload: computes signatures on-the-fly
// Use the version with SignatureAccessor for better performance
template<typename EdgeAccessor, typename MatchCallback>
void find_matches(
    const RewriteRule& rule,
    uint16_t rule_index,
    StateId state_id,
    const SparseBitset& state_edges,
    const SignatureIndex& sig_index,
    const InvertedVertexIndex& inv_index,
    EdgeAccessor get_edge,
    MatchCallback&& on_match,
    std::atomic<bool>* should_terminate = nullptr,
    std::atomic<size_t>* matches_found = nullptr,
    size_t max_matches = SIZE_MAX
) {
    // Create a signature accessor that computes on-the-fly
    auto compute_signature = [&get_edge](EdgeId eid) -> EdgeSignature {
        const auto& edge = get_edge(eid);
        return EdgeSignature::from_edge(edge.vertices, edge.arity);
    };

    find_matches(rule, rule_index, state_id, state_edges,
                 sig_index, inv_index, get_edge, compute_signature,
                 std::forward<MatchCallback>(on_match),
                 should_terminate, matches_found, max_matches);
}

// =============================================================================
// Delta Matching - Only find NEW matches involving produced edges
// =============================================================================
// For match forwarding optimization: new matches must include at least one
// produced edge. We start pattern matching from produced edges only, which
// dramatically reduces the search space.
//
// For a k-edge pattern, we try each produced edge at each pattern position.
// Deduplication handles overlaps when multiple produced edges are in one match.

template<typename EdgeAccessor, typename SignatureAccessor>
void scan_pattern_from_edge(
    PatternMatchingContext<EdgeAccessor, SignatureAccessor>& mc,
    EdgeId starting_edge,
    uint8_t pattern_position
) {
    if (mc.rule->num_lhs_edges == 0) return;

    // The anchor bypasses generate_candidates, so the signature test it would have applied is
    // applied here. join_seed binds the anchor, which rejects any edge this would have.
    const EdgeSignature& data_sig = mc.get_signature(starting_edge);
    if (!signature_compatible(data_sig, mc.rule->lhs_sig[pattern_position])) {
        return;
    }

    HostJoinContext<EdgeAccessor, SignatureAccessor> ctx{&mc};
    typename decltype(ctx)::JoinState st;
    VariableBinding scratch;
    hgcommon::join_seed(ctx, st, starting_edge, pattern_position,
                        [&](const auto& s) { emit_match(mc, scratch, s); });
}

// Find matches that include at least one of the produced edges
// This is used for delta matching: only search for NEW patterns
template<typename EdgeAccessor, typename SignatureAccessor, typename MatchCallback>
void find_delta_matches(
    const RewriteRule& rule,
    uint16_t rule_index,
    StateId state_id,
    const SparseBitset& state_edges,
    const SignatureIndex& sig_index,
    const InvertedVertexIndex& inv_index,
    EdgeAccessor get_edge,
    SignatureAccessor get_signature,
    MatchCallback&& on_match,
    const EdgeId* produced_edges,
    uint8_t num_produced,
    std::atomic<bool>* should_terminate = nullptr,
    std::atomic<size_t>* matches_found = nullptr,
    size_t max_matches = SIZE_MAX
) {
    if (num_produced == 0) return;

    PatternMatchingContext<EdgeAccessor, SignatureAccessor> ctx(
        &rule, rule_index, state_id, &state_edges,
        &sig_index, &inv_index, get_edge, get_signature,
        std::forward<MatchCallback>(on_match)
    );

    ctx.should_terminate = should_terminate;
    ctx.matches_found = matches_found;
    ctx.max_matches = max_matches;

    // For each produced edge, try it at each pattern position
    // This ensures we find all matches that include at least one produced edge
    for (uint8_t p = 0; p < num_produced; ++p) {
        EdgeId produced = produced_edges[p];

        // Skip if edge not in state (shouldn't happen, but safety check)
        if (!state_edges.contains(produced)) continue;

        for (uint8_t pos = 0; pos < rule.num_lhs_edges; ++pos) {
            if (should_terminate && should_terminate->load()) return;

            scan_pattern_from_edge(ctx, produced, pos);
        }
    }
}

}  // namespace engine
}  // namespace HG_NAMESPACE