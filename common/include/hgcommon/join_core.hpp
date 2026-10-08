#pragma once
#include "hgcommon/namespace.hpp"
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
//      a CSR slice, or walks the pivot vertex's incident list, or walks the compatible
//      signature buckets. This is a genuine difference -- different memory systems want
//      different access -- so it is the Ctx's job, behind a per-depth cursor, and nothing else
//      here knows about it.
//   2. WHAT AN EMITTED MATCH IS. The host calls a callback; the device claims a pool slot and
//      publishes. Emit's job.
//
// WHAT IS NOT DIFFERENT, and is therefore stated once here: the depth-first search, the
// edge-injectivity rule, the binding and its unwind, and the order in which pattern edges are
// bound.
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
// consistent (non-injective-allowed) vertex binding"), and it is what makes the rewrite
// total as a double pushout: the gluing condition holds for every match. Identification -- what
// the match identifies must be preserved by the rule -- holds because the match identifies only
// vertices (it is edge-injective) and a rule preserves every vertex of its left side; dangling
// triggers only on vertex deletion, and no vertex is ever deleted. So no match the join emits is
// ever rejected at application, which is exactly what makes non-injective vertex binding safe.
// The injectivity check lives here, once.
//
// THE UNWIND SAVES NO VALUES. A variable bound for the first time by this pattern edge has no
// previous value to restore, so clearing exactly the bits the edge newly set restores the
// binding exactly. The device previously copied the whole binding array per candidate; the bit
// difference is both cheaper and the same operation.

#include "hgcommon/core.hpp"
#include "hgcommon/match_core.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace common {

// Per-thread join state. Templated on the bounds so host and device can size it from their own
// limits without a second definition.
template <uint32_t MaxEdges, uint32_t MaxVars, typename EdgeIdT, typename VertexIdT>
struct JoinState {
    static constexpr uint32_t kMaxEdges = MaxEdges;
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
//   using Cursor = ...;                                       -- one depth's place in its candidates
//   void                cursor_open(uint8_t p, const St& st, Cursor& c) const
//   bool                cursor_next(Cursor& c, Cand& out) const   -- false when exhausted
//   bool                aborted() const
//
// A CANDIDATE IS WHATEVER THE ENUMERATOR PRODUCES, not necessarily an edge id. Enumerating a
// candidate already reads its edge, so it hands that read to the join rather than having the
// join repeat the lookup, and the Ctx says how to read an id and vertices back out of it. A
// port whose enumerator yields bare ids makes Cand the id and all three accessors identities.
//
// Emit is called with the completed state; it may inspect st.matched / st.pattern / st.binding.
//
// ITERATIVE, with one cursor per depth. A recursive join has a call cycle, and a device linker
// cannot size the stack of a kernel that contains one; this body has none. The order of the
// matches is the depth-first order of the candidates each cursor yields. Depth is bounded by
// MaxEdges through num_lhs_edges(), which every caller validates at rule construction.
template <typename Ctx, typename St, typename Emit>
HG_HD void join_dfs(const Ctx& ctx, St& st, Emit&& emit) {
    using Cursor = typename Ctx::Cursor;
    using Cand = typename Ctx::Cand;
    const uint8_t n = ctx.num_lhs_edges();
    const uint8_t base = st.depth;
    const uint32_t mask0 = st.bound_mask;
    if (ctx.aborted()) return;
    if (base == n) { emit(st); return; }

    auto next_position = [&]() {
        return join_next_position([&](uint8_t k) { return ctx.order_at(k); }, n,
                                  st.bound_pattern_mask());
    };
    Cursor cur[St::kMaxEdges];
    uint8_t at[St::kMaxEdges];         // the pattern position each depth binds
    uint32_t saved[St::kMaxEdges];     // the mask before each depth's binding
    {
        const uint8_t p = next_position();
        if (p == 0xFFu) return;        // every position bound but depth disagreed: emit nothing
        at[base] = p;
        ctx.cursor_open(p, st, cur[base]);
    }
    uint8_t d = base;
    for (;;) {
        if (ctx.aborted()) {
            st.depth = base;
            st.bound_mask = mask0;
            return;
        }
        Cand cand;
        if (!ctx.cursor_next(cur[d], cand)) {
            if (d == base) return;
            --d;
            --st.depth;
            join_unbind_since(st, saved[d]);
            continue;
        }
        const auto id = ctx.candidate_id(cand);
        if (!ctx.usable(id)) continue;
        if (st.already_taken(id)) continue;           // edge-injective

        const uint8_t p = at[d];
        saved[d] = st.bound_mask;
        if (!bind_pattern_edge(ctx.edge_vertices(cand), ctx.edge_arity(cand),
                               ctx.pattern_vars(p), ctx.pattern_arity(p),
                               st.binding, st.bound_mask)) {
            // bind_pattern_edge may have bound some variables before hitting the mismatch.
            join_unbind_since(st, saved[d]);
            continue;
        }
        st.pattern[d] = p;
        st.matched[d] = id;
        ++st.depth;

        const uint8_t next = st.depth == n ? 0xFFu : next_position();
        if (st.depth == n && !ctx.aborted()) emit(st);
        if (next == 0xFFu) {
            --st.depth;
            join_unbind_since(st, saved[d]);
            continue;
        }
        ++d;
        at[d] = next;
        ctx.cursor_open(next, st, cur[d]);
    }
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

}  // namespace common
}  // namespace HG_NAMESPACE
