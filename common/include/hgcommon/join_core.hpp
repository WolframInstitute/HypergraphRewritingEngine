#pragma once
//
// THE JOIN: one body, host and device.
//
// Matching a rule's LHS against a state is an edge-at-a-time backtracking join. It was written
// twice -- hypergraph/include/hypergraph/pattern_matcher.hpp and gpu/src/match.cu, 619 and 616
// lines -- sharing only the 20-line bind_pattern_edge. The two agreed until they did not: the
// shipping CPU-21/GPU-23 event-count divergence is pinned by
// gpu/tests CanonicalEventCount.ReconstructionGapIsStillOpen.
//
// WHAT IS ACTUALLY DIFFERENT between the two sides, and therefore what is a parameter here:
//
//   1. CANDIDATE ENUMERATION. The host intersects an inverted vertex index; the device strides
//      a CSR slice, or walks the inverted index with a bounded dedup buffer, or falls back to a
//      signature-bucket walk. This is a genuine difference -- different memory systems want
//      different access -- so it is the Ctx's job and nothing else here knows about it.
//   2. WHAT AN EMITTED MATCH IS. The host calls a callback; the device claims a pool slot and
//      publishes. Emit's job.
//
// WHAT IS NOT DIFFERENT, and is therefore stated once here: the recursion, the edge-injectivity
// rule, the binding and its unwind, and the order in which pattern edges are bound.
//
// THE ORDER IS AN EXPLICIT PARAMETER, because the two sides represent it differently and that
// difference was invisible. The host keeps the LHS in its authored order and indirects through
// RewriteRule::match_order at match time (pattern.hpp:116). The device physically reorders
// DeviceRule::lhs[] when the rule is built (match.cu:515, "we physically reorder here") and then
// binds lhs[depth]. Both express "bind this pattern edge at this depth"; only one of them can be
// read off a rule. Ctx::order_at(k) is that function, so a reader sees the choice instead of
// inferring it from which array is indexed.
//
// EDGE-INJECTIVE, VERTEX-NON-INJECTIVE. A match is a morphism that is injective on EDGES and
// unrestricted on vertices: distinct pattern variables may bind the same vertex, and two pattern
// edges may not take the same data edge. That is the convention the reference implementation
// uses (reference/MultiwayReference.wl: "all ordered injective edge assignments with a
// consistent (non-injective-allowed) vertex binding"), and it is what makes the DPO rewrite
// total -- see docs/REPRESENTATION_DESIGN.md. The injectivity check lives here, once.
//
// THE UNWIND SAVES NO VALUES. A variable bound for the first time by this pattern edge has no
// previous value to restore, so clearing exactly the bits the edge newly set restores the
// binding exactly. The device previously copied the whole binding array per candidate; the bit
// difference is both cheaper and the same operation.

#include "hgcommon/core.hpp"
#include "hgcommon/match_core.hpp"

#include <cstdint>

namespace hgcommon {

// Per-thread join state. Templated on the bounds so host and device can size it from their own
// limits without a second definition.
template <uint32_t MaxEdges, uint32_t MaxVars, typename EdgeIdT, typename VertexIdT>
struct JoinState {
    EdgeIdT   matched[MaxEdges];   // the edge bound at each DEPTH
    uint8_t   pattern[MaxEdges];   // the pattern edge bound at each depth
    VertexIdT binding[MaxVars];
    uint32_t  bound_mask = 0;
    uint8_t   depth = 0;

    // A reset frame has NO bound variables and no live values. The values matter as well as the
    // mask: an enumerator may read a variable it expects the schedule to have bound already
    // (the device pivots on one), and hgcommon::resolve_rhs_vertices reads the array directly
    // and takes INVALID_ID to mean "not matched, allocate a fresh vertex". Once per join, not
    // per candidate -- the unwind below restores the mask alone.
    HG_HD void reset() {
        bound_mask = 0;
        depth = 0;
        for (uint32_t v = 0; v < MaxVars; ++v) binding[v] = static_cast<VertexIdT>(INVALID_ID);
    }

    // Edge-injectivity: a data edge may be taken by at most one pattern edge.
    HG_HD bool already_taken(EdgeIdT e) const {
        for (uint8_t i = 0; i < depth; ++i)
            if (matched[i] == e) return true;
        return false;
    }

    // Which pattern positions are bound, as a bitmask over pattern indices.
    HG_HD uint32_t bound_pattern_mask() const {
        uint32_t m = 0;
        for (uint8_t i = 0; i < depth; ++i) m |= (1u << pattern[i]);
        return m;
    }
};

// WHICH PATTERN POSITION TO BIND NEXT: the first in the schedule that is not bound yet.
//
// Selecting by COUNT -- order[depth] -- assumes the search began at order[0]. That holds for a
// full scan and does NOT hold for a seeded one: an anchor pinned at some other position leaves
// order[0] never bound at all, so every match through it is silently missed, and because
// forwarding is inductive each miss deletes a whole subtree while the run stays self-consistent.
//
// Takes the schedule as an accessor, not an array, for two reasons: the device HAS no order
// array (it physically reorders DeviceRule::lhs[] at build time, so its schedule is the
// identity), and a caller that expands by SPAWNING A TASK per candidate instead of recursing --
// ParallelEvolutionEngine::execute_expand_task -- selects with this same function rather than
// its own copy of the loop.
//
// 0xFF means every position is bound.
template <typename OrderAt>
HG_HD HG_INLINE uint8_t join_next_position(OrderAt&& order_at, uint8_t num_lhs_edges,
                                           uint32_t bound_pattern_mask) {
    for (uint8_t k = 0; k < num_lhs_edges; ++k) {
        const uint8_t p = order_at(k);
        if (!(bound_pattern_mask & (1u << p))) return p;
    }
    return 0xFFu;
}

// Clear exactly the variables bound since `saved_mask`. No values are saved because a variable
// bound for the first time had none.
template <typename St>
HG_HD inline void join_unbind_since(St& st, uint32_t saved_mask) {
    st.bound_mask = saved_mask;
}

// The join.
//
// Ctx must provide:
//   uint8_t             num_lhs_edges() const
//   uint8_t             order_at(uint8_t k) const            -- the k'th position in the schedule
//   const uint8_t*      pattern_vars(uint8_t p) const
//   uint8_t             pattern_arity(uint8_t p) const
//   Cand                candidate_of(EdgeIdT e) const        -- a candidate from a bare edge id
//   EdgeIdT             candidate_id(const Cand& c) const
//   const VertexIdT*    edge_vertices(const Cand& c) const
//   uint8_t             edge_arity(const Cand& c) const
//   bool                usable(EdgeIdT e) const             -- e.g. "is in this state"
//   template <class F> void for_each_candidate(uint8_t p, const St& st, F&& f) const
//   bool                aborted() const
//
// A CANDIDATE IS WHATEVER THE ENUMERATOR PRODUCES, not necessarily an edge id. Enumerating a
// candidate already reads its edge, so it hands that read to the join rather than having the
// join repeat the lookup, and the Ctx says how to read an id and vertices back out of it. A
// port whose enumerator yields bare ids makes Cand the id and all three accessors identities.
//
// Emit is called with the completed state; it may inspect st.matched / st.pattern / st.binding.
//
// Depth is bounded by MaxEdges through num_lhs_edges(), which every caller validates at rule
// construction, so the recursion terminates without its own guard.
template <typename Ctx, typename St, typename Emit>
HG_HD void join_dfs(const Ctx& ctx, St& st, Emit&& emit) {
    if (ctx.aborted()) return;

    if (st.depth == ctx.num_lhs_edges()) {
        emit(st);
        return;
    }

    const uint8_t p = join_next_position([&](uint8_t k) { return ctx.order_at(k); },
                                         ctx.num_lhs_edges(), st.bound_pattern_mask());
    if (p == 0xFFu) return;   // every position bound but depth disagreed: emit nothing

    ctx.for_each_candidate(p, st, [&](const auto& cand) {
        if (ctx.aborted()) return;
        const auto id = ctx.candidate_id(cand);
        if (!ctx.usable(id)) return;
        if (st.already_taken(id)) return;            // edge-injective

        const uint32_t saved = st.bound_mask;
        if (!bind_pattern_edge(ctx.edge_vertices(cand), ctx.edge_arity(cand),
                               ctx.pattern_vars(p), ctx.pattern_arity(p),
                               st.binding, st.bound_mask)) {
            // bind_pattern_edge may have bound some variables before hitting the mismatch.
            join_unbind_since(st, saved);
            return;
        }

        st.pattern[st.depth] = p;
        st.matched[st.depth] = id;
        ++st.depth;

        join_dfs(ctx, st, emit);

        --st.depth;
        join_unbind_since(st, saved);
    });
}

// Seed the join at a GIVEN pattern position with a GIVEN edge, then run it.
//
// This is what delta matching is: the same join, anchored so that every emitted match uses the
// anchor edge at that position. It is not a second algorithm and does not get a second body.
//
// The anchor may sit at ANY position, which is why join_dfs takes the next position as the
// first UNBOUND one in the schedule rather than as order_at(depth): seeding at position 2 must
// still bind position 0, and taking order_at(1) next would leave it unbound forever.
template <typename Ctx, typename St, typename Emit, typename EdgeIdT>
HG_HD bool join_seed(const Ctx& ctx, St& st, EdgeIdT anchor, uint8_t at_pattern, Emit&& emit) {
    st.reset();
    if (!ctx.usable(anchor)) return false;

    const auto cand = ctx.candidate_of(anchor);
    const uint32_t saved = st.bound_mask;
    if (!bind_pattern_edge(ctx.edge_vertices(cand), ctx.edge_arity(cand),
                           ctx.pattern_vars(at_pattern), ctx.pattern_arity(at_pattern),
                           st.binding, st.bound_mask)) {
        join_unbind_since(st, saved);
        return false;
    }
    st.pattern[0] = at_pattern;
    st.matched[0] = anchor;
    st.depth = 1;
    join_dfs(ctx, st, emit);
    return true;
}

}  // namespace hgcommon
