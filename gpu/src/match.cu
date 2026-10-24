#include "hgcommon/namespace.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>

#include "hgcommon/join_core.hpp"   // THE JOIN -- one body, shared with the host matcher
#include "hgcommon/sampling_core.hpp"
#include "hg_gpu/exploration.hpp"
#include "hg_gpu/content_hash.hpp"   // DeviceContentCursor
#include "hg_gpu/ir_canon.hpp"       // successor_exact_hash_device
#include "hg_gpu/rewrite.hpp"        // bind_matched_vertices
#include "hgcommon/rewrite_core.hpp" // SuccessorCursor
#include "hgcommon/content_core.hpp" // content_hash_of

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

namespace {

// What match_state_rule below owns, and what it does not.
//
// THE JOIN IS NOT HERE. The recursion, edge-injectivity, binding and unwind, and which pattern
// position is bound next are hgcommon/join_core.hpp, so the host engine runs the same body.
// This file supplies the two device-specific halves: how candidates are enumerated, and the
// parallelism.
//
// PARALLELISM: one BLOCK per (state, rule) pair, or per part of one when the pass emits every
// match as found (match_state_rule_splits): part p of P anchors the p-th of P equal ranges of the
// state's slice. Threads inside the block stride the part's anchors for pattern edge 0, each
// running the whole DFS subtree below a different root, with no inter-thread coordination --
// outputs go straight to the global match pool. A thread's DFS diverges from its warp's other
// threads, so a block's time grows with the anchors it holds; the persistent scheduler sets the
// parts (match_parts_of) so that a step's few states spread over the grid.
//
//   gridDim.x  = num_state_ids * num_rules
//   blockDim.x = kMatchBlockThreads (warp-aligned; 32 by default), declared in match.hpp
//     because match_state_rule's body stripes across exactly those threads -- every scheduler
//     calling it must launch with this shape, so it is contract, not a private detail.
//
// ENUMERATION, in the Ctx's cursor. Pattern edge 0, and an edge with no bound pivot vertex, take
// their candidates from the state's own CSR slice: O(|state|), each edge exactly once, membership
// free. An edge with a bound pivot vertex in a state past DeviceState::slice_scan_max_edges walks
// the pivot's incident list (VertexInvertedIndex), which is degree-bounded, and skips the entries
// of other states; at or below the threshold the slice is cheaper. The threshold also gates lazy
// index maintenance: below it the index is never read. A pass without a per-(state, rule) or
// per-parent cap, whose ranks share the buffer (kMatchSharedWords), copies a state at or
// below the threshold into the block's shared memory (stage_state_edges) and scans the copy,
// skipping edges whose arity or pivot vertex cannot bind the pattern edge.

using MatchJoinState = hgcommon::JoinState<kMaxPatternEdges, kMaxVars, EdgeId, VertexId>;

// A state's edges copied into the block's shared memory for one pass (stage_state_edges):
// the ids in slice order, each edge's arity, and the vertices of each edge at a fixed stride of
// `stride` words. An edge whose arity exceeds the stride has no vertices copied; no pattern edge
// of the rule has that arity, so hgcommon::bind_pattern_edge rejects it before reading them.
// `count` 0 means the state is not staged.
struct MatchStage {
    EdgeId*   ids    = nullptr;
    uint8_t*  arity  = nullptr;
    VertexId* verts  = nullptr;
    uint32_t  stride = 0;
    uint32_t  count  = 0;
};

// The join's view of this device. hgcommon/join_core.hpp owns the recursion, the
// edge-injectivity rule, the binding and its unwind, and which pattern position is bound
// next; this supplies candidate enumeration and nothing else.
struct MatchJoinCtx {
    const DeviceState& ds;
    const DeviceRule& rule;
    StateId           state_id;
    // Loaded once: read through `ds` inside the join, each would be reloaded after every match
    // the join emits, since an emit's atomics may write what the reference points at.
    const Edge*       edges;
    const VertexId*   verts;
    const EdgeId*     ids;
    StateEdgeSlice    slice;
    uint32_t          scan_max;
    MatchStage        stage;

    __device__ MatchJoinCtx(const DeviceState& d, const DeviceRule& r, StateId s,
                            const MatchStage& st)
        : ds(d), rule(r), state_id(s), edges(d.edge_pool.data), verts(d.vertex_pool.data),
          ids(d.state_edge_ids),
          slice(s < d.max_states ? d.state_edge_slices[s] : StateEdgeSlice{0, 0}),
          scan_max(d.slice_scan_max_edges), stage(st) {}

    __device__ uint8_t num_lhs_edges() const { return rule.num_lhs_edges; }

    // DeviceRule::order, as the host's RewriteRule::match_order: the authored pattern edge
    // bound at depth k.
    __device__ uint8_t order_at(uint8_t k) const { return rule.order[k]; }

    __device__ const uint8_t* pattern_vars(uint8_t p)  const { return rule.lhs[p].vars; }
    __device__ uint8_t        pattern_arity(uint8_t p) const { return rule.lhs[p].arity; }

    // A candidate carries its id, its vertices and its arity, read once by the cursor.
    struct Cand {
        EdgeId          id;
        const VertexId* v;
        uint8_t         arity;
    };
    __device__ Cand from_pool(EdgeId e) const {
        const Edge& ed = edges[e];
        return Cand{e, verts + ed.vertex_offset, ed.arity};
    }
    __device__ Cand candidate_of(EdgeId e) const { return from_pool(e); }
    __device__ EdgeId candidate_id(const Cand& c) const { return c.id; }
    __device__ const VertexId* edge_vertices(const Cand& c) const { return c.v; }
    __device__ uint8_t edge_arity(const Cand& c) const { return c.arity; }

    // Every cursor yields edges of this state only: a scan reads the state's own slice, and the
    // pivot walk skips list entries outside the slice. A seeded anchor comes from the slice.
    __device__ bool usable(EdgeId) const { return true; }
    __device__ bool aborted() const { return false; }

    // The two paths are strictly EITHER/OR per pattern position: running both would enumerate a
    // candidate twice and emit a duplicate match. The pivot vertex's incident list holds each
    // incident edge once (VertexInvertedIndex).
    using List = typename LockFreeList<EdgeId>::DeviceView;
    static constexpr uint8_t kScan = 0, kPivot = 1, kStaged = 2;
    static constexpr uint32_t kEnd = Pool<typename LockFreeList<EdgeId>::Node>::kInvalid;
    struct Cursor {
        uint32_t at;     // kScan, kStaged: the next slice index; kPivot: the next list node or kEnd
        uint8_t  mode;
        uint8_t  p;      // the pattern position
        // kStaged: the pattern edge's arity, and the vertex `pv` its edges hold at position `pk`
        // (the bound pivot variable's place in the pattern edge), or pk kNoFilter.
        uint8_t  pa;
        uint8_t  pk;
        VertexId pv;
    };
    static constexpr uint8_t kNoFilter = 0xFF;

    // The first node of list[key], as LockFreeList::DeviceView::for_each starts its walk: a
    // relaxed load for an empty list, the acquire only for a list that has a node.
    __device__ static uint32_t list_first(const List& l, uint32_t key) {
        if (key >= l.num_keys) return kEnd;
        typename List::Ops ops{&l, key};
        if (ops.head_load_relaxed() == ops.invalid()) return kEnd;
        return ops.head_load_acquire();
    }

    __device__ void cursor_open(uint8_t p, const MatchJoinState& st, Cursor& c) const {
        c.p = p;
        c.at = 0;
        c.mode = stage.count ? kStaged : kScan;
        const DevicePatternEdge& pe = rule.lhs[p];
        c.pa = pe.arity;
        c.pk = kNoFilter;
        c.pv = 0;
        if (pe.pivot_var != kNoPivotVar && (st.bound_mask >> pe.pivot_var) & 1u) {
            for (uint8_t k = 0; k < pe.arity; ++k)
                if (pe.vars[k] == pe.pivot_var) { c.pk = k; break; }
            c.pv = st.binding[pe.pivot_var];
        }
        if (slice.count > scan_max && pe.pivot_var != kNoPivotVar) {
            c.mode = kPivot;
            c.at = list_first(ds.vertex_inverted_index.list, st.binding[pe.pivot_var]);
        }
    }

    __device__ bool cursor_next(Cursor& c, Cand& out) const {
        if (c.mode == kStaged) {
            // Edges of another arity, or without the pivot vertex at its place, cannot bind the
            // pattern edge (hgcommon::bind_pattern_edge) and are skipped here.
            while (c.at < stage.count) {
                const uint32_t i = c.at++;
                if (stage.arity[i] != c.pa) continue;
                const VertexId* v = stage.verts + size_t(i) * stage.stride;
                if (c.pk != kNoFilter && v[c.pk] != c.pv) continue;
                out = Cand{stage.ids[i], v, c.pa};
                return true;
            }
            return false;
        }
        if (c.mode == kScan) {
            if (c.at >= slice.count) return false;
            out = from_pool(ids[slice.offset + c.at++]);
            return true;
        }
        // The incident list holds the vertex's edges in every state: entries outside this
        // state's slice are skipped.
        while (c.at != kEnd) {
            const auto& node = ds.vertex_inverted_index.list.pool.at(c.at);
            const EdgeId e = node.value;
            c.at = node.next;
            if (slice_position(ids, slice, e) == UINT32_MAX) continue;
            out = from_pool(e);
            return true;
        }
        return false;
    }
};

// The widest left-hand edge of `rule`: the vertex stride of a stage.
__device__ __forceinline__ uint32_t rule_max_lhs_arity(const DeviceRule& rule) {
    uint32_t a = 0;
    for (uint8_t p = 0; p < rule.num_lhs_edges; ++p)
        a = rule.lhs[p].arity > a ? rule.lhs[p].arity : a;
    return a;
}

// Copies state `sid`'s edges into `smem` (`bytes` of the block's shared memory) on every thread
// of the block, when the state is matched by scanning its slice (at most slice_scan_max_edges
// edges) and its edges fit. Returns the stage, with count 0 when the state is not staged.
// Every thread of the block calls it; it ends with a barrier.
__device__ __noinline__ MatchStage stage_state_edges(const DeviceState& ds, const DeviceRule& rule,
                                                     StateId sid, uint64_t* smem, uint32_t bytes) {
    MatchStage s{};
    if (smem == nullptr || sid >= ds.max_states) return s;
    const StateEdgeSlice sl = ds.state_edge_slices[sid];
    const uint32_t stride = rule_max_lhs_arity(rule);
    // Per edge: its id, its arity byte and `stride` vertices.
    const uint32_t per_edge = 4u + 1u + 4u * stride;
    if (sl.count == 0 || sl.count > ds.slice_scan_max_edges || sl.count * per_edge > bytes)
        return s;
    unsigned char* base = reinterpret_cast<unsigned char*>(smem);
    s.ids = reinterpret_cast<EdgeId*>(base);
    s.verts = reinterpret_cast<VertexId*>(base + 4u * sl.count);
    s.arity = base + 4u * sl.count + 4u * stride * sl.count;
    s.stride = stride;
    s.count = sl.count;
    // The previous pass's threads may still read the buffer.
    __syncthreads();
    const Edge* edges = ds.edge_pool.data;
    const VertexId* verts = ds.vertex_pool.data;
    for (uint32_t i = threadIdx.x; i < sl.count; i += blockDim.x) {
        const EdgeId e = ds.state_edge_ids[sl.offset + i];
        const Edge ed = edges[e];
        s.ids[i] = e;
        s.arity[i] = ed.arity;
        if (ed.arity <= stride)
            for (uint32_t k = 0; k < ed.arity; ++k)
                s.verts[i * stride + k] = verts[ed.vertex_offset + k];
    }
    __syncthreads();
    return s;
}

}  // namespace

// One (state, rule) pair, matched by one BLOCK -- threads inside it stripe the depth-0
// candidates. EXTERNAL linkage, so a scheduler in another translation unit drives this same
// implementation rather than growing a second copy. The helpers it calls stay file-local,
// which is fine: they are defined above it in this file.
//
// It sits outside the anonymous namespace for that reason alone. The alternative -- putting
// the other scheduler in THIS file -- was measured and rejected: match.cu already costs about
// 5 GB to compile on its own, and adding one more kernel took a single nvcc to 8 GB.
// See gpu/ARCHITECTURE.md sec 2.
// Which completed matches a pass emits.
//   kCapCount   record each match's rank, emit nothing     } the per-(state, rule) cap: the k
//   kCapEmit    emit ranks at or below the cut               } smallest ranks of the pair, ties kept
//   kDraw       emit the matches that survive their draw; record the state's minimum rank
//               and whether any match survived
//   kSpineEmit  emit the matches whose rank is that minimum
//   kParentCount  record each match's rank, rule and edges, emit    } the per-state cap: the
//                 nothing                                           } transitions into the k
//   kParentEmit   emit a kept match, once                           } lowest-ranked successors
// The caps' rank is selection_rank_device, the spine's hgcommon::transition_rank of
// transition_key_device, and the draw hgcommon::transition_survives of that key: the host's
// cap_at_drain, spine_at_drain and transition_survives compute the same values.
//   kCollect      record each match's rule and edges, emit nothing ("Generational", no rank)
enum class EmitMode : uint8_t {
    kCapCount, kCapEmit, kDraw, kSpineEmit, kParentCount, kParentEmit, kCollect
};

struct EmitCtl {
    EmitMode            mode;
    uint32_t            cut_all;      // kCapEmit: the cut keeps every match (RankCut::all)
    uint64_t            threshold;    // kCapEmit: the cut rank; kSpineEmit: the minimum
    uint32_t*           s_seen;
    uint32_t*           s_overflow;
    uint64_t*           s_ranks;
    unsigned long long* s_min_rank;
    uint32_t*           s_survived;
    // kParentCount: the rule and the edges (kMaxPatternEdges, by pattern position) of each
    // recorded rank. kParentEmit: the kept transitions' ranks, rules and edges (keep_n of them),
    // their indices in rank order (s_by_rank) and a bit each, cleared by the one emission of the
    // transition.
    uint16_t*           s_rules = nullptr;
    EdgeId*             s_edges = nullptr;
    uint32_t*           s_keep_quota = nullptr;
    uint32_t            keep_n = 0;
    const uint32_t*     s_by_rank = nullptr;
    // kCapCount, kParentCount: how many entries the buffers hold.
    uint32_t            capacity = kDrainCapBuffer;
};

// Not inlined: it is called from the join's innermost completion callback, which is
// instantiated per rule shape, and inlined there ptxas ran out of memory assembling this file.
__device__ __noinline__ bool emit_admit(const DeviceState& ds, StateId state_id, RuleId rid,
                                        const uint8_t* pattern, const EdgeId* matched,
                                        uint8_t depth, uint32_t step, const EmitCtl& c) {
    EdgeId edges[kMaxPatternEdges];
    for (uint8_t i = 0; i < kMaxPatternEdges; ++i) edges[i] = INVALID_ID;
    for (uint8_t d = 0; d < depth; ++d) edges[pattern[d]] = matched[d];
    // The draw keys on the transition key and the event's step (rank_step); the caps and the
    // spine order by selection rank, which without a step cap is transition_rank of the same key
    // and step.
    const uint32_t rs = hgcommon::rank_step(step + 1u, ds.rank_reads_step);
    uint64_t key = 0, r = 0;
    if (c.mode == EmitMode::kDraw || c.mode == EmitMode::kSpineEmit) {
        key = transition_key_device(ds, state_id, rid, edges, depth);
        r = hgcommon::transition_rank(key, ds.sampling_seed, rs);
    } else if (c.mode != EmitMode::kCollect) {
        r = selection_rank_device(ds, state_id, rid, edges, depth, step + 1u);
    }

    switch (c.mode) {
    case EmitMode::kCapCount: {
        const uint32_t at = atomicAdd(c.s_seen, 1u);
        if (at < c.capacity) c.s_ranks[at] = r;
        else atomicExch(c.s_overflow, 1u);
        return false;
    }
    case EmitMode::kCapEmit:
        return hgcommon::rank_cut_keeps(hgcommon::RankCut{c.threshold, c.cut_all != 0u}, r);
    case EmitMode::kDraw: {
        atomicMin(c.s_min_rank, static_cast<unsigned long long>(r));
        const double rate = hgcommon::sampling_rate_for_rule(
            ds.transition_rate, ds.rule_weights, ds.num_rule_weights, rid);
        if (!hgcommon::transition_survives(key, ds.sampling_seed, rate, rs)) return false;
        atomicExch(c.s_survived, 1u);
        return true;
    }
    case EmitMode::kSpineEmit:
        // Every match at the minimum rank: under Full states automorphic transitions share it.
        return r == c.threshold;
    case EmitMode::kParentCount:
    case EmitMode::kCollect: {
        const uint32_t at = atomicAdd(c.s_seen, 1u);
        if (at < c.capacity) {
            if (c.s_ranks) c.s_ranks[at] = r;
            c.s_rules[at] = static_cast<uint16_t>(rid);
            for (uint8_t i = 0; i < kMaxPatternEdges; ++i)
                c.s_edges[at * kMaxPatternEdges + i] = edges[i];
        } else {
            atomicExch(c.s_overflow, 1u);
        }
        return false;
    }
    case EmitMode::kParentEmit: {
        // The kept transition with this rule and these edges, emitted once: found among the
        // candidates of its rank, by binary search over s_by_rank.
        uint32_t lo = 0, hi = c.keep_n;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2u;
            if (c.s_ranks[c.s_by_rank[mid]] < r) lo = mid + 1u;
            else hi = mid;
        }
        for (uint32_t j = lo; j < c.keep_n && c.s_ranks[c.s_by_rank[j]] == r; ++j) {
            const uint32_t i = c.s_by_rank[j];
            if (c.s_rules[i] != rid) continue;
            bool same = true;
            for (uint8_t k = 0; k < kMaxPatternEdges && same; ++k)
                same = c.s_edges[i * kMaxPatternEdges + k] == edges[k];
            if (!same) continue;
            const uint32_t bit = 1u << (i & 31u);
            return (atomicAnd(c.s_keep_quota + (i >> 5), ~bit) & bit) != 0u;
        }
        return false;
    }
    }
    return false;
}

// ONE PASS OF THE JOIN, compiled ONCE and called up to twice per rule. Not inlined for the
// same reason as emit_admit: a lambda called twice instantiated the whole DFS twice and ptxas
// ran out of memory. `ctl` null emits every match. `stage` is the state's edges in shared memory
// or an empty stage. The pass anchors part `part` of `parts` equal ranges of the state's slice
// (match_split_root_begin). A non-null `out_rank` receives each emitted record's
// step_candidate_word_device at the record's index in `out`. `costs`, when given, receives a
// CostEntry for each emission, booked on the step (step_cost_one).
__device__ __noinline__ void match_state_rule_pass(
        const DeviceState& ds, const DeviceRule* rules, StateId state_id, uint32_t rid, uint32_t step,
        typename Pool<MatchRecord>::DeviceView out, const EmitCtl* ctl,
        const MatchStage& stage = MatchStage{}, uint32_t part = 0, uint32_t parts = 1,
        uint64_t* out_rank = nullptr, WorkLogView<CostEntry>* costs = nullptr) {
    const DeviceRule& rule = rules[rid];

    if (rule.num_lhs_edges == 0) return;

    const MatchJoinCtx ctx(ds, rule, state_id, stage);

    // A completed match. matched_edges is indexed by PATTERN position, not by depth.
    auto emit = [&] (const MatchJoinState& st) {
        if (ctl && !emit_admit(ds, state_id, rid, st.pattern, st.matched, st.depth, step, *ctl)) {
            return;
        }
        const uint32_t idx = out.claim();
        if (idx == Pool<MatchRecord>::kInvalid) {
            ds.errors.record(ErrorKind::kMatchPoolFull);
            return;
        }
        EdgeId edges[kMaxPatternEdges];
        for (uint8_t i = 0; i < kMaxPatternEdges; ++i) edges[i] = INVALID_ID;
        for (uint8_t d = 0; d < st.depth; ++d) edges[st.pattern[d]] = st.matched[d];
        if (out_rank)
            out_rank[idx] = step_candidate_word_device(ds, state_id, rid, edges, st.depth,
                                                       step + 1u, stage.count ? stage.ids : nullptr,
                                                       stage.count);
        MatchRecord& m = out.at(idx);
        m.rule_id   = rid;
        m.state_id  = state_id;
        m.step      = step;
        m.num_edges = st.depth;
        for (uint8_t i = 0; i < kMaxPatternEdges; ++i) m.matched_edges[i] = edges[i];
        publish_match(m);
        // Booked before the append, inside the match item that still holds the step open; a
        // candidate whose entry finds the log full is costed by the selection.
        if (costs) {
            step_book(ds, step, 1u);
            if (!costs->append(CostEntry{idx, step, 0u}) && step < ds.step_slots)
                atomicSub(&ds.step_pending[step], 1u);
        }
    };

    // One thread, one depth-0 candidate, one whole DFS subtree: the join anchored at the
    // schedule's first pattern edge. A single-edge rule completes on the anchor itself and emits
    // through this same path.
    auto run_dfs_from_root = [&] (EdgeId root_cand) {
        MatchJoinState st;
        hgcommon::join_seed(ctx, st, root_cand, rule.order[0], emit);
    };

    // Stride pattern edge 0's candidates in this part's range of the state's own edge slice
    // across the block's threads. THIS is the device's part of matching -- the parallelism --
    // and it is why match_state_rule exists rather than a call straight into join_core.
    const StateEdgeSlice sl0 = ds.state_edge_slices[state_id];
    const uint32_t begin = match_split_root_begin(sl0.count, part, parts);
    const uint32_t end = match_split_root_begin(sl0.count, part + 1u, parts);
    for (uint32_t i = begin + threadIdx.x; i < end; i += blockDim.x)
        run_dfs_from_root(stage.count ? stage.ids[i] : ds.state_edge_ids[sl0.offset + i]);
}

// The block's spill slot holding at least `words`, for every thread of the block: claimed by
// thread 0 from the block's arena when the slot is smaller. Null when there is no slot
// (kDrainCapBufferFull recorded) or the arena cannot hold it (kIRArenaExhausted recorded, which
// grow-and-retry doubles).
__device__ __noinline__ uint32_t* claim_spill(const DeviceState& ds, const SuccessorCtx* succ,
                                              uint64_t words) {
    __shared__ uint32_t* s_spill;
    if (threadIdx.x == 0) {
        s_spill = nullptr;
        if (!succ || !succ->spill) {
            ds.errors.record(ErrorKind::kDrainCapBufferFull);
        } else if (*succ->spill_words >= words) {
            s_spill = *succ->spill;
        } else if (uint32_t* p = DeviceArena::View(succ->arena).claim(words)) {
            *succ->spill = p;
            *succ->spill_words = words;
            s_spill = p;
        } else {
            ds.errors.record(ErrorKind::kIRArenaExhausted);
        }
    }
    __syncthreads();
    uint32_t* p = s_spill;
    __syncthreads();
    return p;
}

// The candidate arrays of MaxSuccessorStatesPerParent: in shared memory and the block's
// SuccessorScratch for up to kDrainCapBuffer candidates, else carved from a spill buffer of
// exactly the state's count.
struct ParentBuffers {
    uint64_t*                    ranks;   // the candidate's own rank
    uint16_t*                    rules;
    uint8_t*                     kept;
    uint32_t*                    quota;
    EdgeId*                      edges;   // kMaxPatternEdges per candidate, by pattern position
    hgcommon::SuccessorIdentity* id;
    uint64_t*                    srank;   // the rank of the candidate's successor
    uint32_t*                    order;
    uint8_t*                     lead;
    uint32_t                     capacity;
};

// Words of a spill buffer for n candidates, and its carving. Every array starts on an even word.
__device__ __forceinline__ uint64_t even_words(uint64_t bytes) { return ((bytes + 7u) / 8u) * 2u; }
__device__ __forceinline__ uint64_t parent_spill_words(uint32_t n) {
    return even_words(8ull * n) * 2u + even_words(2ull * n) + even_words(1ull * n) * 2u +
           even_words(4ull * ((n + 31u) / 32u)) + even_words(4ull * n * kMaxPatternEdges) +
           even_words(sizeof(hgcommon::SuccessorIdentity) * uint64_t{n}) + even_words(4ull * n);
}
__device__ __forceinline__ ParentBuffers parent_spill_buffers(uint32_t* p, uint32_t n) {
    ParentBuffers b{};
    auto take = [&](uint64_t bytes) { uint32_t* at = p; p += even_words(bytes); return at; };
    b.ranks = reinterpret_cast<uint64_t*>(take(8ull * n));
    b.srank = reinterpret_cast<uint64_t*>(take(8ull * n));
    b.rules = reinterpret_cast<uint16_t*>(take(2ull * n));
    b.kept  = reinterpret_cast<uint8_t*>(take(1ull * n));
    b.lead  = reinterpret_cast<uint8_t*>(take(1ull * n));
    b.quota = take(4ull * ((n + 31u) / 32u));
    b.edges = reinterpret_cast<EdgeId*>(take(4ull * n * kMaxPatternEdges));
    b.id    = reinterpret_cast<hgcommon::SuccessorIdentity*>(
        take(sizeof(hgcommon::SuccessorIdentity) * uint64_t{n}));
    b.order = take(4ull * n);
    b.capacity = n;
    return b;
}

// One counting pass of every rule into `b`: each transition's rank, rule and edges.
__device__ __forceinline__ void parent_count_pass(
        const DeviceState& ds, const DeviceRule* rules, StateId state_id, uint32_t step,
        typename Pool<MatchRecord>::DeviceView out, uint32_t* s_seen, uint32_t* s_overflow,
        const ParentBuffers& b, EmitMode mode = EmitMode::kParentCount) {
    if (threadIdx.x == 0) { *s_seen = 0; *s_overflow = 0; }
    __syncthreads();
    EmitCtl count{mode, 0u, 0u, s_seen, s_overflow, b.ranks, nullptr, nullptr};
    count.s_rules = b.rules;
    count.s_edges = b.edges;
    count.capacity = b.capacity;
    for (uint32_t r = 0; r < ds.num_rules; ++r)
        match_state_rule_pass(ds, rules, state_id, r, step, out, &count);
    __syncthreads();
}

// MaxSuccessorStatesPerParent, in the block of the state's rule 0, over every rule: the host's
// cap_at_drain. The first pass ranks every transition of the state and records its rule and
// edges, in shared memory and the block's scratch, or when the state has more than
// kDrainCapBuffer transitions, again into a spill buffer of exactly that many. The block then
// keeps the transitions ranked at or below their rule's MatchesPerStateRule cut (hgcommon::
// rank_cut_where per rule: fewer than cap_k of the rule rank strictly below them), and of those
// the transitions into the parent_k lowest-ranked successor states (hgcommon::successor_ranks,
// then rank_cut over one rank per successor, as keep_lowest_successors on the host); the second
// pass emits each kept transition once, found by its rank, rule and edges. No draw is applied
// to the kept transitions, as on the host.
__device__ __noinline__ void match_state_parent_capped(
        const DeviceState& ds, const DeviceRule* rules, StateId state_id, uint32_t step,
        typename Pool<MatchRecord>::DeviceView out, uint32_t* s_seen, uint32_t* s_overflow,
        uint32_t* s_keep_n, uint64_t* s_buf, const SuccessorCtx* succ,
        const SuccessorScratch* scratch) {
    const uint32_t cap_k = ds.matches_per_state_rule;
    const uint32_t parent_k = ds.max_successor_states_per_parent;
    // The candidates' ranks, rules, keep bytes and quota bits, carved from the block's match
    // buffer (kMatchSharedWords).
    uint64_t* s_ranks = s_buf;
    uint16_t* s_rules = reinterpret_cast<uint16_t*>(s_buf + kDrainCapBuffer);
    uint8_t*  s_kept  = reinterpret_cast<uint8_t*>(s_buf + kDrainCapBuffer + kDrainCapBuffer / 4u);
    uint32_t* s_quota = reinterpret_cast<uint32_t*>(s_buf + kDrainCapBuffer +
                                                    kDrainCapBuffer / 4u + kDrainCapBuffer / 8u);

    if (!succ || !scratch || blockIdx.x >= scratch->blocks) {
        // No scratch to hold the candidates: the cap cannot choose, so every transition is
        // emitted and the run records it, under the engine's partial-result contract.
        if (threadIdx.x == 0) ds.errors.record(ErrorKind::kScratchOverflow);
        __syncthreads();
        for (uint32_t r = 0; r < ds.num_rules; ++r)
            match_state_rule_pass(ds, rules, state_id, r, step, out, nullptr);
        return;
    }
    const size_t base = size_t(blockIdx.x) * kDrainCapBuffer;
    ParentBuffers b{s_ranks, s_rules, s_kept, s_quota, scratch->edges + base * kMaxPatternEdges,
                    scratch->id + base, scratch->rank + base, scratch->order + base,
                    scratch->lead + base, kDrainCapBuffer};

    parent_count_pass(ds, rules, state_id, step, out, s_seen, s_overflow, b);
    if (*s_overflow) {
        const uint32_t total = *s_seen;
        __syncthreads();
        uint32_t* spill = claim_spill(ds, succ, parent_spill_words(total));
        if (spill) {
            b = parent_spill_buffers(spill, total);
            parent_count_pass(ds, rules, state_id, step, out, s_seen, s_overflow, b);
        }
        if (!spill || *s_overflow) {
            // No buffer holds the state's transitions (recorded by claim_spill): every
            // transition is emitted, under the engine's partial-result contract.
            if (threadIdx.x == 0 && spill) ds.errors.record(ErrorKind::kDrainCapBufferFull);
            __syncthreads();
            for (uint32_t r = 0; r < ds.num_rules; ++r)
                match_state_rule_pass(ds, rules, state_id, r, step, out, nullptr);
            return;
        }
    }
    const uint32_t n = *s_seen;
    // Kept by MatchesPerStateRule: ranked at or below the rule's cut, by one thread.
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < n; ++i) b.kept[i] = 1u;
        if (cap_k != 0u) {
            for (uint32_t r = 0; r < ds.num_rules; ++r) {
                const hgcommon::RankCut cut = hgcommon::rank_cut_where(
                    n, cap_k, [&](uint32_t i) { return b.ranks[i]; },
                    [&](uint32_t i) { return b.rules[i] == r; });
                for (uint32_t i = 0; i < n; ++i)
                    if (b.rules[i] == r && !hgcommon::rank_cut_keeps(cut, b.ranks[i]))
                        b.kept[i] = 0u;
            }
        }
        // The kept candidates, compacted in place: each lands at or before its slot.
        uint32_t w = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (!b.kept[i]) continue;
            b.ranks[w] = b.ranks[i];
            b.rules[w] = b.rules[i];
            for (uint8_t k = 0; k < kMaxPatternEdges; ++k)
                b.edges[w * kMaxPatternEdges + k] = b.edges[i * kMaxPatternEdges + k];
            ++w;
        }
        *s_keep_n = w;
        for (uint32_t q = 0; q < (w + 31u) / 32u; ++q) b.quota[q] = 0u;
    }
    __syncthreads();
    const uint32_t m = *s_keep_n;
    // Each candidate's successor, on the whole warp (a Full successor's hash runs on it).
    for (uint32_t i = 0; i < m; ++i) {
        hgcommon::SuccessorIdentity sid{0, 1};
        if (!successor_identity_device(ds, rules[b.rules[i]], state_id,
                                       b.edges + i * kMaxPatternEdges, *succ, sid))
            sid = hgcommon::SuccessorIdentity{0, 1};
        if (threadIdx.x == 0) b.id[i] = sid;
    }
    __syncthreads();
    // The successor cut, by one thread: rank_cut over one rank per successor, a keep bit per
    // candidate whose successor is at or below it, and the candidates in rank order for the
    // emission's lookup.
    if (threadIdx.x == 0) {
        hgcommon::successor_ranks(b.ranks, b.id, m, b.order, b.srank, b.lead);
        const hgcommon::RankCut cut = hgcommon::rank_cut_where(
            m, parent_k, [&](uint32_t i) { return b.srank[i]; },
            [&](uint32_t i) { return b.lead[i] != 0u; });
        for (uint32_t i = 0; i < m; ++i)
            if (hgcommon::rank_cut_keeps(cut, b.srank[i])) b.quota[i >> 5] |= 1u << (i & 31u);
        for (uint32_t i = 0; i < m; ++i) b.order[i] = i;
        hgcommon::ir_heapsort_idx(b.order, m, [&](uint32_t x, uint32_t y) {
            if (b.ranks[x] != b.ranks[y]) return b.ranks[x] < b.ranks[y] ? -1 : 1;
            return x < y ? -1 : (x > y ? 1 : 0);
        });
    }
    __syncthreads();
    EmitCtl keep{EmitMode::kParentEmit, 0u, 0u, nullptr, nullptr, b.ranks, nullptr, nullptr};
    keep.s_rules = b.rules;
    keep.s_edges = b.edges;
    keep.s_keep_quota = b.quota;
    keep.keep_n = m;
    keep.s_by_rank = b.order;
    for (uint32_t r = 0; r < ds.num_rules; ++r)
        match_state_rule_pass(ds, rules, state_id, r, step, out, &keep);
}

// "Generational", in the block of the state's rule 0, over every rule: the host's
// generation_at_drain. A first pass counts the state's matches; a second records each match's
// rule and edges into a spill buffer of exactly that many, with an order array and a bit per
// edge of the state's slice. Thread 0 orders the matches by hgcommon::event_order_before (the
// standard order, or under "EventOrderingFunction" ds.event_order, where it takes one match: the
// host's ordered_at_drain), takes them by hgcommon::greedy_disjoint, writes the generation's list to the arena
// (generation_list) and emits one record of kGenerationRule. A state with no match emits nothing.
__device__ __noinline__ void match_state_generation(
        const DeviceState& ds, const DeviceRule* rules, StateId state_id, uint32_t step,
        typename Pool<MatchRecord>::DeviceView out, const SuccessorCtx* succ) {
    __shared__ uint32_t s_seen;
    __shared__ uint32_t s_overflow;
    ParentBuffers none{};
    parent_count_pass(ds, rules, state_id, step, out, &s_seen, &s_overflow, none,
                      EmitMode::kCollect);
    const uint32_t n = s_seen;
    if (n == 0u) return;
    const StateEdgeSlice slice = ds.state_edge_slices[state_id];
    const uint32_t bitmap_words = (slice.count + 31u) / 32u;
    const uint64_t words = even_words(2ull * n) + even_words(4ull * n * kMaxPatternEdges) +
                           even_words(4ull * n) + even_words(4ull * bitmap_words);
    uint32_t* spill = claim_spill(ds, succ, words);
    if (!spill) return;   // claim_spill recorded why; the state is not rewritten
    ParentBuffers b{};
    {
        uint32_t* p = spill;
        auto take = [&](uint64_t bytes) { uint32_t* at = p; p += even_words(bytes); return at; };
        b.rules = reinterpret_cast<uint16_t*>(take(2ull * n));
        b.edges = reinterpret_cast<EdgeId*>(take(4ull * n * kMaxPatternEdges));
        b.order = take(4ull * n);
        b.quota = take(4ull * bitmap_words);   // the slice's used edges
        b.capacity = n;
    }
    parent_count_pass(ds, rules, state_id, step, out, &s_seen, &s_overflow, b,
                      EmitMode::kCollect);
    if (threadIdx.x == 0) {
        const uint32_t m = s_seen < n ? s_seen : n;
        const EdgeId* ids = ds.state_edge_ids;
        auto edges_at = [&](uint32_t i) { return b.edges + size_t(i) * kMaxPatternEdges; };
        auto count_at = [&](uint32_t i) { return uint32_t{rules[b.rules[i]].num_lhs_edges}; };
        for (uint32_t i = 0; i < m; ++i) b.order[i] = i;
        const uint32_t event_order =
            ds.event_order != 0u ? ds.event_order : hgcommon::kStandardEventOrder;
        hgcommon::ir_heapsort_idx(b.order, m, [&](uint32_t x, uint32_t y) {
            const int c = hgcommon::event_order_compare(event_order, edges_at(x), count_at(x),
                                                        b.rules[x], edges_at(y), count_at(y),
                                                        b.rules[y]);
            if (c != 0) return c;
            return x < y ? -1 : (x > y ? 1 : 0);
        });
        for (uint32_t w = 0; w < bitmap_words; ++w) b.quota[w] = 0u;
        // The taken matches are compacted to the front of `order`, in the order taken.
        uint32_t taken = 0, consumed = 0;
        hgcommon::greedy_disjoint(
            b.order, m, ds.event_order != 0u ? 1u : UINT32_MAX,
            [&](uint32_t i, uint32_t* o) {
                const uint32_t k = count_at(i);
                for (uint32_t j = 0; j < k; ++j) o[j] = edges_at(i)[j];
                return k;
            },
            // A bit per position in the slice; slice_position indexes `ids`.
            [&](uint32_t e) {
                const uint32_t p = slice_position(ids, slice, e);
                if (p == UINT32_MAX) return true;
                const uint32_t at = p - slice.offset;
                return ((b.quota[at >> 5] >> (at & 31u)) & 1u) != 0u;
            },
            [&](uint32_t e) {
                const uint32_t p = slice_position(ids, slice, e);
                if (p == UINT32_MAX) return;
                const uint32_t at = p - slice.offset;
                b.quota[at >> 5] |= 1u << (at & 31u);
                ++consumed;
            },
            [&](uint32_t i) { b.order[taken++] = i; });
        uint32_t* list = nullptr;
        if (taken) {
            list = DeviceArena::View(succ->arena)
                       .claim(2ull + uint64_t{taken} * kGenerationEntryWords + consumed);
            if (!list) ds.errors.record(ErrorKind::kIRArenaExhausted);
        }
        const uint32_t idx = list ? out.claim() : Pool<MatchRecord>::kInvalid;
        if (list && idx == Pool<MatchRecord>::kInvalid) ds.errors.record(ErrorKind::kMatchPoolFull);
        if (list && idx != Pool<MatchRecord>::kInvalid) {
            list[0] = taken;
            list[1] = consumed;
            for (uint32_t t = 0; t < taken; ++t) {
                const uint32_t i = b.order[t];
                uint32_t* e = list + 2 + size_t(t) * kGenerationEntryWords;
                e[0] = b.rules[i];
                for (uint32_t j = 0; j < kMaxPatternEdges; ++j) e[1 + j] = edges_at(i)[j];
            }
            // The consumed edges ascending: the slice is ascending.
            uint32_t* c = list + 2 + size_t(taken) * kGenerationEntryWords;
            uint32_t w = 0;
            for (uint32_t at = 0; at < slice.count; ++at)
                if ((b.quota[at >> 5] >> (at & 31u)) & 1u) c[w++] = ids[slice.offset + at];
            const uintptr_t addr = reinterpret_cast<uintptr_t>(list);
            MatchRecord& rec = out.at(idx);
            rec.rule_id = kGenerationRule;
            rec.state_id = state_id;
            rec.step = step;
            rec.num_edges = 0;
            for (uint32_t j = 0; j < kMaxPatternEdges; ++j) rec.matched_edges[j] = INVALID_ID;
            rec.matched_edges[0] = static_cast<EdgeId>(addr & 0xFFFFFFFFu);
            rec.matched_edges[1] = static_cast<EdgeId>(static_cast<uint64_t>(addr) >> 32);
            __threadfence();
            publish_match(rec);
        }
    }
    __syncthreads();
}

__device__ bool successor_identity_device(const DeviceState& ds, const DeviceRule& rule,
                                          StateId parent, const EdgeId* matched,
                                          const SuccessorCtx& ctx,
                                          hgcommon::SuccessorIdentity& out) {
    out = hgcommon::SuccessorIdentity{0, 1};
    if (ctx.mode == CanonicalizationMode::None) return true;
    // A fresh vertex is in no other state.
    if (ctx.mode == CanonicalizationMode::Automatic &&
        hgcommon::num_fresh_variables(rule.new_var_mask) != 0)
        return true;
    VertexId binding[kMaxVars];
    bind_matched_vertices(ds, rule, matched, binding);
    EdgeId consumed[kMaxPatternEdges];
    uint32_t nc = 0;
    for (uint8_t p = 0; p < rule.num_lhs_edges; ++p)
        if (matched[p] != INVALID_ID) consumed[nc++] = matched[p];
    if (ctx.mode == CanonicalizationMode::Automatic) {
        uint64_t key = 0;
        if ((threadIdx.x & 31u) == 0) {
            const hgcommon::SuccessorCursor<DeviceContentCursor, DeviceRhsEdge> c(
                DeviceContentCursor(ds, parent), consumed, nc, rule.rhs, rule.num_rhs_edges,
                rule.new_var_mask, binding, 0u);
            key = hgcommon::content_hash_of(c);
        }
        key = __shfl_sync(0xffffffffu, static_cast<unsigned long long>(key), 0);
        out = hgcommon::SuccessorIdentity{key, 0};
        return true;
    }
    uint64_t h = 0;
    const ExactHashStatus st = successor_exact_hash_device(
        ds, parent, consumed, nc, rule.rhs, rule.num_rhs_edges, rule.new_var_mask, binding,
        ctx.arena, *ctx.slot, *ctx.slot_words, h, nullptr, nullptr, IrWarpAll{});
    if (st != ExactHashStatus::kOk) {
        if ((threadIdx.x & 31u) == 0) ds.errors.record(error_kind_for(st));
        return false;
    }
    out = hgcommon::SuccessorIdentity{h, 0};
    return true;
}

__device__ void step_cost_one(const DeviceState& ds, const DeviceRule* rules,
                              typename Pool<MatchRecord>::DeviceView out, const SuccessorCtx& ctx,
                              uint32_t i) {
    // Under Full states counting successors, the cost's canonical form gives the identity.
    const bool by_succ = !ds.step_cap_counts_matches && ctx.mode != CanonicalizationMode::None;
    const bool full_id = by_succ && ctx.mode == CanonicalizationMode::Full;
    const MatchRecord& m = out.at(i);
    const DeviceRule& rule = rules[m.rule_id];
    hgcommon::SuccessorIdentity id{0, 1};
    uint64_t key = 0;
    const bool costed = ds.state_cost != nullptr;
    bool ok = true;
    if (costed)
        ok = successor_cost_device(ds, rule, m.state_id, m.matched_edges, ctx, key,
                                   full_id ? &id : nullptr);
    if (ok && by_succ && (!costed || !full_id))
        ok = successor_identity_device(ds, rule, m.state_id, m.matched_edges, ctx, id);
    if (ok && (threadIdx.x & 31u) == 0) {
        ctx.cand_cost[i] = key;
        ctx.cand_id[i] = id;
        ctx.cand_ready[i] = 1u;
    }
}

__device__ bool successor_cost_device(const DeviceState& ds, const DeviceRule& rule,
                                      StateId parent, const EdgeId* matched,
                                      const SuccessorCtx& ctx, uint64_t& out,
                                      hgcommon::SuccessorIdentity* full_id) {
    out = hgcommon::state_cost_key(HUGE_VAL);
    VertexId binding[kMaxVars];
    bind_matched_vertices(ds, rule, matched, binding);
    EdgeId consumed[kMaxPatternEdges];
    uint32_t nc = 0;
    for (uint8_t p = 0; p < rule.num_lhs_edges; ++p)
        if (matched[p] != INVALID_ID) consumed[nc++] = matched[p];
    uint64_t h = 0;
    uint32_t* form = nullptr;
    uint32_t words = 0;
    const ExactHashStatus st = successor_exact_hash_device(
        ds, parent, consumed, nc, rule.rhs, rule.num_rhs_edges, rule.new_var_mask, binding,
        ctx.arena, *ctx.slot, *ctx.slot_words, h, &form, &words, IrWarpAll{});
    if (st != ExactHashStatus::kOk) {
        if ((threadIdx.x & 31u) == 0) ds.errors.record(error_kind_for(st));
        return false;
    }
    if (full_id) *full_id = hgcommon::SuccessorIdentity{h, 0};
    if (!state_cost_of_form_device(ds, form, words, ctx.arena, *ctx.slot, *ctx.slot_words,
                                   *ds.state_cost, out, IrWarpAll{})) {
        if ((threadIdx.x & 31u) == 0) ds.errors.record(ErrorKind::kIRArenaExhausted);
        out = hgcommon::state_cost_key(HUGE_VAL);
        return false;
    }
    return true;
}

// The entry point: one block, one (state, rule) pair. Unsampled and uncapped, a single pass that
// emits every match.
//
// Under the per-(state, rule) cap, two passes: the first ranks every match of the pair, the
// second emits the k smallest and every match tied with the k-th (hgcommon::rank_cut). The block has found all of the pair's matches after the first,
// which is the completion point the host calls the state's drain. No draw is applied to the
// kept k, as on the host.
//
// Under sampling without the cap, the block of rule 0 matches EVERY rule of the state and the
// blocks of the other rules do nothing. The spine chooses among all of a state's transitions:
// when none survives its draw, the ones with the smallest rank are emitted. That needs the whole
// state in one block.
__device__ void match_state_rule(const DeviceState& ds,
                                 const DeviceRule* rules,
                                 StateId           state_id,
                                 uint32_t          rid,
                                 uint32_t          step,
                                 typename Pool<MatchRecord>::DeviceView out,
                                 const SuccessorCtx* succ,
                                 const SuccessorScratch* scratch,
                                 uint32_t part, uint32_t parts, uint64_t* out_rank) {
    const uint32_t cap_k = ds.matches_per_state_rule;
    const uint32_t parent_k = ds.max_successor_states_per_parent;
    // Under a step cap, each candidate this call emits goes to the cost log, whose entries idle
    // blocks run (step_cost_one) before the step's selection.
    WorkLogView<CostEntry> cost_log =
        succ != nullptr ? succ->cost_log : WorkLogView<CostEntry>{};
    WorkLogView<CostEntry>* costs =
        ds.max_states_per_step != 0u && succ != nullptr && succ->cand_ready != nullptr
            ? &cost_log
            : nullptr;
    // The block's match buffer: the state's staged edges when no cap ranks candidates, the
    // caps' ranks and candidate arrays otherwise.
    __shared__ uint64_t s_buf[kMatchSharedWords];
    if (ds.generational != 0u) {
        if (rid == 0u) match_state_generation(ds, rules, state_id, step, out, succ);
        return;
    }
    if (match_state_rule_splits(ds)) {
        const MatchStage stage = stage_state_edges(ds, rules[rid], state_id, s_buf, sizeof(s_buf));
        match_state_rule_pass(ds, rules, state_id, rid, step, out, nullptr, stage, part, parts,
                              out_rank, costs);
        return;
    }

    __shared__ uint32_t s_seen;
    __shared__ uint32_t s_overflow;
    __shared__ uint32_t s_keep_n;
    __shared__ uint32_t s_survived;
    __shared__ uint64_t s_threshold;
    __shared__ uint32_t s_cut_all;
    __shared__ unsigned long long s_min_rank;
    uint64_t* s_ranks = s_buf;

    if (parent_k != 0u) {
        if (rid == 0u)
            match_state_parent_capped(ds, rules, state_id, step, out, &s_seen, &s_overflow,
                                      &s_keep_n, s_buf, succ, scratch);
        return;
    }

    if (cap_k == 0u) {
        if (rid != 0u) return;
        if (threadIdx.x == 0) { s_survived = 0; s_min_rank = ~0ULL; }
        __syncthreads();
        const EmitCtl draw{EmitMode::kDraw, 0u, 0u, nullptr, nullptr, nullptr, &s_min_rank,
                           &s_survived};
        for (uint32_t r = 0; r < ds.num_rules; ++r)
            match_state_rule_pass(ds, rules, state_id, r, step, out, &draw,
                                  stage_state_edges(ds, rules[r], state_id, s_buf, sizeof(s_buf)));
        __syncthreads();
        if (s_survived != 0u || s_min_rank == ~0ULL) return;
        const EmitCtl spine{EmitMode::kSpineEmit, 0u, s_min_rank, nullptr, nullptr, nullptr,
                            nullptr, nullptr};
        for (uint32_t r = 0; r < ds.num_rules; ++r)
            match_state_rule_pass(ds, rules, state_id, r, step, out, &spine,
                                  stage_state_edges(ds, rules[r], state_id, s_buf, sizeof(s_buf)));
        return;
    }

    if (threadIdx.x == 0) { s_seen = 0; s_overflow = 0; s_threshold = ~0ULL; }
    __syncthreads();

    EmitCtl count{EmitMode::kCapCount, cap_k, ~0ULL, &s_seen, &s_overflow, s_ranks,
                  nullptr, nullptr};
    match_state_rule_pass(ds, rules, state_id, rid, step, out, &count);
    __syncthreads();

    // More matches than shared memory ranks: counted again into a spill buffer of exactly
    // that many (claim_spill records why when there is none).
    uint64_t* ranks = s_ranks;
    bool spilled = false;
    if (s_overflow) {
        const uint32_t total = s_seen;
        __syncthreads();
        if (uint32_t* spill = claim_spill(ds, succ, 2ull * total)) {
            ranks = reinterpret_cast<uint64_t*>(spill);
            if (threadIdx.x == 0) { s_seen = 0; s_overflow = 0; }
            __syncthreads();
            count.s_ranks = ranks;
            count.capacity = total;
            match_state_rule_pass(ds, rules, state_id, rid, step, out, &count);
            __syncthreads();
            spilled = true;
        }
    }

    if (threadIdx.x == 0) {
        if (s_overflow) {
            // No buffer holds the pair's matches: the cap is NOT applied to this pair rather than
            // applied to the wrong k, and the run is under the engine's partial-result contract
            // (claim_spill recorded why).
            if (spilled) ds.errors.record(ErrorKind::kDrainCapBufferFull);
            s_threshold = ~0ULL;
            s_cut_all = 1u;
        } else {
            // The k-th smallest rank counted with multiplicity (hgcommon::rank_cut); every match
            // ranked at or below it is kept.
            const hgcommon::RankCut cut = hgcommon::rank_cut(ranks, s_seen, cap_k);
            s_threshold = cut.rank;
            s_cut_all = cut.all ? 1u : 0u;
        }
    }
    __syncthreads();

    const EmitCtl keep{EmitMode::kCapEmit, s_cut_all, s_threshold, &s_seen, &s_overflow,
                       s_ranks, nullptr, nullptr};
    match_state_rule_pass(ds, rules, state_id, rid, step, out, &keep, MatchStage{}, 0u, 1u,
                          nullptr, costs);
}

namespace {

// Batch driver: one block per (state, rule) pair of a state set.
__global__ void k_match_batch(const __grid_constant__ DeviceState ds,
                              const DeviceRule* rules,
                              uint32_t          num_rules,
                              const StateId*    state_ids,
                              uint32_t          num_state_ids,
                              typename Pool<MatchRecord>::DeviceView out,
                              uint32_t          bid_offset,
                              uint32_t          step) {
    uint32_t bid = blockIdx.x + bid_offset;
    uint32_t total = num_rules * num_state_ids;
    if (bid >= total) return;

    uint32_t state_idx = bid / num_rules;
    uint32_t rid       = bid - state_idx * num_rules;
    match_state_rule(ds, rules, state_ids[state_idx], rid, step, out);
}

}  // namespace

namespace {

// Connectivity-ordered LHS schedule.
//
// HGMatch/MaCH adapted to Wolfram semantics: at DFS depth ≥ 1, the pattern
// edge being bound must share at least one variable with a pattern edge
// already bound at a shallower depth. This lets the match kernel look up
// candidates via `vertex_inverted_index[binding[pivot_var]]` — a degree-
// bounded list — instead of scanning the state's whole edge slice.
//
// Greedy schedule: start with the first edge of rule.lhs that has at least
// one variable (any rule with a non-empty LHS). For each subsequent slot,
// pick the unplaced LHS edge whose variable set has largest overlap with
// already-bound variables; on ties, the earliest in rule.lhs. The pivot_var emitted
// for each edge ≥ 1 is one of the variables shared with the bound set
// (pick the first one found in the source LHS positional order for
// determinism).
//
// If a rule's LHS is disconnected (no overlap between some pair of components)
// the greedy picks one edge from the second component without a pivot —
// for safety we emit pivot_var = kNoPivotVar on that edge and the match
// kernel scans the state's slice for it. In practice Wolfram rules
// have connected LHS.
struct ScheduledEdge {
    uint8_t src_index;       // original index in rule.lhs
    uint8_t pivot_var;       // kNoPivotVar on edge 0 (or disconnected seeds)
};

std::vector<ScheduledEdge> schedule_lhs_edges(const RewriteRule& rule) {
    const uint8_t n = static_cast<uint8_t>(rule.lhs.size());
    std::vector<ScheduledEdge> out;
    out.reserve(n);
    if (n == 0) return out;

    std::vector<bool> placed(n, false);
    std::vector<bool> bound_var(rule.num_lhs_vars, false);

    // Pick seed: edge 0 (matches pre-existing behaviour; any non-empty edge
    // is fine — connectivity starts here).
    out.push_back({0, kNoPivotVar});
    placed[0] = true;
    for (uint8_t v : rule.lhs[0]) bound_var[v] = true;

    while (out.size() < n) {
        int best_idx = -1;
        uint8_t best_pivot = kNoPivotVar;
        int best_overlap = -1;
        for (uint8_t e = 0; e < n; ++e) {
            if (placed[e]) continue;
            int overlap = 0;
            uint8_t first_shared = kNoPivotVar;
            for (uint8_t v : rule.lhs[e]) {
                if (v < bound_var.size() && bound_var[v]) {
                    if (first_shared == kNoPivotVar) first_shared = v;
                    ++overlap;
                }
            }
            if (overlap > best_overlap) {
                best_overlap = overlap;
                best_idx     = e;
                best_pivot   = first_shared;
            }
        }
        if (best_idx < 0) break;  // shouldn't happen given placed[] bookkeeping
        // best_pivot is kNoPivotVar only if this edge shares no var with the
        // already-bound subgraph (disconnected rule). Match kernel handles
        // that case by scanning the state's slice for this edge.
        out.push_back({static_cast<uint8_t>(best_idx), best_pivot});
        placed[best_idx] = true;
        for (uint8_t v : rule.lhs[best_idx]) bound_var[v] = true;
    }
    return out;
}

}  // namespace

DeviceRule make_device_rule(const RewriteRule& rule) {
    // Validate BEFORE anything is written. DeviceRule's lhs[] and rhs[] are fixed at
    // kMaxPatternEdges and the counts are uint8_t, so an oversized rule would truncate on the
    // cast and then be written past the end of the array -- a host-side buffer overflow reached
    // from caller data, before a single kernel launches. new_var_mask is 32 bits, so a variable
    // index at or above MAX_VARS would shift by the width of the type, which is undefined.
    //
    // These are programmer errors in the caller's rule, not capacity overflows in an evolution,
    // so they throw. The overflow contract (partial work plus a warning) covers a run that
    // outgrows its pools; it does not cover a rule that cannot be represented at all.
    if (rule.lhs.size() > kMaxPatternEdges || rule.rhs.size() > kMaxPatternEdges) {
        throw std::runtime_error(
            "make_device_rule: rule has " + std::to_string(rule.lhs.size()) + " LHS and " +
            std::to_string(rule.rhs.size()) + " RHS edges, above kMaxPatternEdges (" +
            std::to_string(kMaxPatternEdges) + ")");
    }
    if (rule.lhs.empty()) {
        throw std::runtime_error("make_device_rule: rule has an empty LHS, which matches "
                                 "everywhere and has no binding to apply");
    }
    for (const auto& e : rule.lhs) {
        if (e.size() > kMaxArity)
            throw std::runtime_error("make_device_rule: LHS edge arity " +
                                     std::to_string(e.size()) + " above kMaxArity (" +
                                     std::to_string(kMaxArity) + ")");
        for (uint8_t v : e)
            if (v >= hgcommon::MAX_VARS)
                throw std::runtime_error("make_device_rule: LHS variable index " +
                                         std::to_string(v) + " at or above MAX_VARS (" +
                                         std::to_string(hgcommon::MAX_VARS) + ")");
    }
    for (const auto& e : rule.rhs) {
        if (e.size() > kMaxArity)
            throw std::runtime_error("make_device_rule: RHS edge arity " +
                                     std::to_string(e.size()) + " above kMaxArity (" +
                                     std::to_string(kMaxArity) + ")");
        for (uint8_t v : e)
            if (v >= hgcommon::MAX_VARS)
                throw std::runtime_error("make_device_rule: RHS variable index " +
                                         std::to_string(v) + " at or above MAX_VARS (" +
                                         std::to_string(hgcommon::MAX_VARS) + ")");
    }

    DeviceRule d;
    d.num_lhs_edges = static_cast<uint8_t>(rule.lhs.size());
    d.num_lhs_vars  = rule.num_lhs_vars;
    d.num_rhs_edges = static_cast<uint8_t>(rule.rhs.size());
    d.num_rhs_vars  = rule.num_rhs_vars;

    // Which variables are new, taken from the rule rather than inferred from the counts.
    {
        uint32_t lhs_mask = 0, rhs_mask = 0;
        for (const auto& e : rule.lhs) for (uint8_t v : e) lhs_mask |= (uint32_t(1) << v);
        for (const auto& e : rule.rhs) for (uint8_t v : e) rhs_mask |= (uint32_t(1) << v);
        d.new_var_mask = rhs_mask & ~lhs_mask;
    }

    // LHS in authored order; the connectivity schedule goes in d.order and each edge's pivot on
    // the edge itself.
    auto schedule = schedule_lhs_edges(rule);
    for (uint8_t e = 0; e < d.num_lhs_edges; ++e) {
        const auto& sch = schedule[e];
        const auto& src = rule.lhs[sch.src_index];
        DevicePatternEdge& dst = d.lhs[sch.src_index];
        dst.arity = static_cast<uint8_t>(src.size());
        for (uint8_t i = 0; i < dst.arity; ++i) dst.vars[i] = src[i];
        dst.pivot_var = sch.pivot_var;
        d.order[e] = sch.src_index;
    }

    for (uint8_t e = 0; e < d.num_rhs_edges; ++e) {
        const auto& src = rule.rhs[e];
        DeviceRhsEdge& dst = d.rhs[e];
        dst.arity = static_cast<uint8_t>(src.size());
        for (uint8_t i = 0; i < dst.arity; ++i) dst.vars[i] = src[i];
    }
    return d;
}

void run_match_kernel_batch_nosync(const EngineState& engine,
                                   const DeviceRule*  d_rules,
                                   uint32_t           num_rules,
                                   const StateId*     d_state_ids,
                                   uint32_t           num_state_ids,
                                   Pool<MatchRecord>& out_matches,
                                   uint32_t           step) {
    if (num_rules == 0 || num_state_ids == 0) return;
    // One block per (state, rule); threads inside the block parallelise
    // pattern-edge-0 candidate enumeration.
    uint32_t grid  = num_rules * num_state_ids;
    int      block = static_cast<int>(kMatchBlockThreads);
    uint32_t cap   = engine.config().max_blocks_per_launch;
    if (cap == 0 || grid <= cap) {
        k_match_batch<<<grid, block>>>(engine.device(), d_rules, num_rules,
                                       d_state_ids, num_state_ids, out_matches.view(), 0u, step);
    } else {
        for (uint32_t off = 0; off < grid; off += cap) {
            uint32_t n = (grid - off < cap) ? (grid - off) : cap;
            k_match_batch<<<n, block>>>(engine.device(), d_rules, num_rules,
                                        d_state_ids, num_state_ids, out_matches.view(), off,
                                        step);
            HG_CUDA_CHECK(cudaDeviceSynchronize(), "k_match_batch chunk sync");
        }
    }
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "k_match_batch sync");
}

uint32_t run_match_kernel(const EngineState&             engine,
                          const std::vector<DeviceRule>& rules,
                          StateId                        state_id,
                          Pool<MatchRecord>&             out_matches,
                          uint32_t                       step) {
    if (rules.empty()) return 0;

    DeviceRule* d_rules = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    HG_CUDA_CHECK(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    out_matches.reset();

    // One block per rule, over match_state_rule -- the same implementation the batched driver
    // and the persistent scheduler use, so a test written against this entry point constrains
    // what ships.
    const StateId* d_state = nullptr;
    HG_CUDA_CHECK(cudaMalloc((void**)&d_state, sizeof(StateId)), "state alloc");
    HG_CUDA_CHECK(cudaMemcpy((void*)d_state, &state_id, sizeof(StateId), cudaMemcpyHostToDevice),
          "state copy");
    k_match_batch<<<(uint32_t)rules.size(), kMatchBlockThreads>>>(
        engine.device(), d_rules, (uint32_t)rules.size(), d_state, 1u,
        out_matches.view(), /*bid_offset=*/0, step);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "run_match_kernel sync");
    cudaFree((void*)d_state);
    cudaFree(d_rules);

    return out_matches.size_host();
}

size_t match_kernels_stack_bytes() {
    cudaFuncAttributes a{};
    HG_CUDA_CHECK(cudaFuncGetAttributes(&a, reinterpret_cast<const void*>(&k_match_batch)),
                  "match kernel attributes");
    return a.localSizeBytes;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE