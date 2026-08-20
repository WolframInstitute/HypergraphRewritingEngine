#include "hgcommon/namespace.hpp"
#include "hg_gpu/edge_signature.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>

#include "hgcommon/join_core.hpp"   // THE JOIN -- one body, shared with the host matcher
#include "hgcommon/sampling_core.hpp"
#include "hg_gpu/exploration.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

namespace {

// Recursively enumerate all coarsenings of the pattern partition into `out`.
// Each coarsening is an EdgeSignature whose partition is coarser-than-or-
// equal to the pattern partition (i.e. positions in the same pattern var
// class also share a data class; positions in different pattern classes may
// be merged or kept separate).
//
// merge_to[c] = data class that pattern class c collapses into. Builds up
// merge_to one pattern class at a time; at depth = num_classes emits the
// induced data signature hash.
void enumerate_coarsenings(const EdgeSignature& pattern,
                           uint8_t num_classes,
                           uint8_t depth,
                           int     max_data_class,   // -1 ⇒ no class assigned yet
                           uint8_t* merge_to,
                           std::vector<uint64_t>& out) {
    if (depth == num_classes) {
        EdgeSignature data;
        data.arity = pattern.arity;
        for (uint8_t i = 0; i < pattern.arity; ++i) {
            data.pattern[i] = merge_to[pattern.pattern[i]];
        }
        out.push_back(signature_hash(data));
        return;
    }
    int upper = max_data_class + 1;
    for (int c = 0; c <= upper; ++c) {
        merge_to[depth] = static_cast<uint8_t>(c);
        int new_max = (c > max_data_class) ? c : max_data_class;
        enumerate_coarsenings(pattern, num_classes, depth + 1, new_max,
                              merge_to, out);
    }
}

std::vector<uint64_t> compatible_signature_hashes(const DevicePatternEdge& pe) {
    EdgeSignature pattern;
    pattern.arity = pe.arity;
    uint8_t var_to_class[kMaxArity];
    for (auto& v : var_to_class) v = 0xFF;
    uint8_t num_classes = 0;
    for (uint8_t i = 0; i < pe.arity; ++i) {
        uint8_t v = pe.vars[i];
        if (var_to_class[v] == 0xFF) var_to_class[v] = num_classes++;
        pattern.pattern[i] = var_to_class[v];
    }

    std::vector<uint64_t> out;
    if (pe.arity == 0) {
        out.push_back(signature_hash(pattern));
        return out;
    }

    uint8_t merge_to[kMaxArity];
    enumerate_coarsenings(pattern, num_classes, 0, /*max_data_class=*/-1,
                          merge_to, out);

    // Coarsenings can produce duplicates when the pattern itself has
    // multiple classes containing only one var (each independent class can
    // be assigned to the same data class) — sort + unique.
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

__device__ bool state_contains(const DeviceState& ds, StateId sid, EdgeId eid) {
    // Binary search in the state's sorted CSR edge-id slice. Slices stay
    // sorted because initial state has ascending IDs and each rewrite
    // appends its (consecutive, higher) produced IDs after the surviving
    // parent tail — see rewrite.cu's commit section.
    if (sid >= ds.max_states) return false;
    StateEdgeSlice sl = ds.state_edge_slices[sid];
    uint32_t lo = sl.offset;
    uint32_t hi = sl.offset + sl.count;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        EdgeId v = ds.state_edge_ids[mid];
        if (v == eid) return true;
        if (v < eid) lo = mid + 1;
        else         hi = mid;
    }
    return false;
}

// What match_state_rule below owns, and what it does not.
//
// THE JOIN IS NOT HERE. The recursion, edge-injectivity, binding and unwind, and which pattern
// position is bound next are hgcommon/join_core.hpp, so the host engine runs the same body.
// This file supplies the two device-specific halves: how candidates are enumerated, and the
// parallelism.
//
// PARALLELISM: one BLOCK per (state, rule) pair; threads inside it stride the candidates for
// pattern edge 0, each running the whole DFS subtree below a different root, with no
// inter-thread coordination -- outputs go straight to the global match pool. That is what
// unblocked the one-thread-per-(state x rule) pathology where n=1000, frontier=1, rule=1 put
// all the work on a single thread (944 ms on n1000_s1) while the rest of the device idled.
//
//   gridDim.x  = num_state_ids * num_rules
//   blockDim.x = kMatchBlockThreads (warp-aligned; 32 by default), declared in match.hpp
//     because match_state_rule's body stripes across exactly those threads -- every scheduler
//     calling it must launch with this shape, so it is contract, not a private detail.
//
// ENUMERATION is adaptive on state size, in Ctx::for_each_candidate. Multiway states are small
// (tens of edges) while the signature and vertex-inverted indices are global across the whole
// evolution, so their buckets grow with total edge count and walking them costs O(evolution)
// per state. At or below DeviceState::slice_scan_max_edges (an EngineConfig knob) candidates
// come straight from the state's own CSR slice: O(|state|), each edge exactly once, no dedup
// buffer, membership free. Above it -- single huge states, the visualiser regime -- the global
// indices win and the pivot/signature machinery is used. The threshold also gates lazy index
// maintenance: below it the indices are never read.

using MatchJoinState = hgcommon::JoinState<kMaxPatternEdges, kMaxVars, EdgeId, VertexId>;

// The join's view of this device. hgcommon/join_core.hpp owns the recursion, the
// edge-injectivity rule, the binding and its unwind, and which pattern position is bound
// next; this supplies candidate enumeration and nothing else.
struct MatchJoinCtx {
    DeviceState       ds;
    const DeviceRule& rule;
    StateId           state_id;

    __device__ uint8_t num_lhs_edges() const { return rule.num_lhs_edges; }

    // The device's schedule is the IDENTITY: build_device_rule physically reorders
    // DeviceRule::lhs[] into join order when the rule is built, so position k of the
    // schedule is lhs[k]. The host instead keeps its LHS authored and indirects through
    // RewriteRule::match_order. Both say "bind this pattern edge at this position".
    __device__ uint8_t order_at(uint8_t k) const { return k; }

    __device__ const uint8_t* pattern_vars(uint8_t p)  const { return rule.lhs[p].vars; }
    __device__ uint8_t        pattern_arity(uint8_t p) const { return rule.lhs[p].arity; }

    // Every enumerator here walks id lists, so a candidate IS an id and there is nothing
    // already-read to carry along.
    __device__ EdgeId candidate_of(EdgeId e) const { return e; }
    __device__ EdgeId candidate_id(EdgeId e) const { return e; }

    __device__ const VertexId* edge_vertices(EdgeId e) const {
        return &ds.vertex_pool.at(ds.edge_pool.at(e).vertex_offset);
    }
    __device__ uint8_t edge_arity(EdgeId e) const { return ds.edge_pool.at(e).arity; }

    __device__ bool usable(EdgeId e) const { return state_contains(ds, state_id, e); }
    __device__ bool aborted() const { return false; }

    // Adaptive on state size, and the paths are strictly EITHER/OR: running two of them
    // would enumerate a candidate twice and emit a duplicate match. Multiway states are
    // small (tens of edges) while the signature and vertex-inverted indices span the whole
    // evolution, so their buckets cost O(evolution) per state; at or below
    // slice_scan_max_edges the state's own CSR slice gives each edge exactly once, with no
    // dedup buffer and with membership for free.
    template <typename F>
    __device__ void for_each_candidate(uint8_t p, const MatchJoinState& st, F&& f) const {
        const DevicePatternEdge& pe = rule.lhs[p];

        const StateEdgeSlice sl = ds.state_edge_slices[state_id];
        if (sl.count <= ds.slice_scan_max_edges) {
            for (uint32_t i = 0; i < sl.count; ++i) f(ds.state_edge_ids[sl.offset + i]);
            return;
        }

        if (pe.pivot_var != kNoPivotVar) {
            const VertexId pivot_vert = st.binding[pe.pivot_var];
            // Bounded dedup: a self-loop {a,a} appears twice in list[a], and concurrent
            // inserts from rewrite kernels interleave those with other edges, so a
            // last-seen check is not enough. Collect first, then hand over -- only one
            // enumerator may call f, because a candidate tried during collection would be
            // tried again by the signature walk after an overflow.
            constexpr uint32_t kMaxIncidentSeen = 256;
            EdgeId   seen[kMaxIncidentSeen];
            uint32_t n_seen = 0;
            bool     overflowed = false;
            ds.vertex_inverted_index.for_each_incident(
                pivot_vert,
                [&] (EdgeId cand) {
                    if (overflowed) return;
                    for (uint32_t i = 0; i < n_seen; ++i) {
                        if (seen[i] == cand) return;
                    }
                    if (n_seen >= kMaxIncidentSeen) { overflowed = true; return; }
                    seen[n_seen++] = cand;
                });
            if (!overflowed) {
                for (uint32_t i = 0; i < n_seen; ++i) f(seen[i]);
                return;
            }
        }

        // Union over every compatible signature bucket: Wolfram binding lets distinct vars
        // collapse onto one vertex, so a matching data edge's signature may be coarser than
        // the pattern's. Each edge appears in its own bucket exactly once.
        for (uint8_t s = 0; s < pe.num_compat_sigs; ++s) {
            ds.signature_index.list.for_each(
                static_cast<uint32_t>(pe.compat_sig_hashes[s]) & ds.signature_index.mask, f);
        }
    }
};

}  // namespace

// One (state, rule) pair, matched by one BLOCK -- threads inside it stripe the depth-0
// candidates. EXTERNAL linkage, so a scheduler in another translation unit drives this same
// implementation rather than growing a second copy. The helpers it calls stay file-local,
// which is fine: they are defined above it in this file.
//
// It sits outside the anonymous namespace for that reason alone. The alternative -- putting
// the other scheduler in THIS file -- was measured and rejected: match.cu already costs about
// 5 GB to compile on its own, and adding one more kernel took a single nvcc to 8 GB.
// See docs/GPU_PERSISTENT_DESIGN.md.
// THE PER-(state, rule) CAP'S DECISION FOR ONE COMPLETED MATCH, deliberately NOT inlined.
//
// It is called from the join's innermost completion callback, which is instantiated per rule
// shape through a template. Inlined there it carried the transition key and the selection
// arithmetic into the DFS and ptxas ran out of memory assembling this file. One call per
// completed match is not a cost worth that.
//
// `counting` is pass one: record the rank, emit nothing. Otherwise pass two: admit iff the rank
// is within the threshold AND the block has not already taken k. The counter settles ties, which
// a 64-bit rank makes vanishingly rare but which must still never admit k+1.
__device__ __noinline__ bool drain_cap_admit(const DeviceState& ds, StateId state_id, RuleId rid,
                                             const uint8_t* pattern, const EdgeId* matched,
                                             uint8_t depth, bool counting, uint32_t cap_k,
                                             uint32_t* s_seen, uint32_t* s_overflow,
                                             uint32_t* s_emitted, uint64_t s_threshold,
                                             uint64_t* s_ranks) {
    EdgeId edges[kMaxPatternEdges];
    for (uint8_t i = 0; i < kMaxPatternEdges; ++i) edges[i] = INVALID_ID;
    for (uint8_t d = 0; d < depth; ++d) edges[pattern[d]] = matched[d];
    const uint64_t r = hgcommon::transition_rank(
        transition_key_device(ds, state_id, rid, edges, depth), ds.sampling_seed);

    if (counting) {
        const uint32_t at = atomicAdd(s_seen, 1u);
        if (at < kDrainCapBuffer) s_ranks[at] = r;
        else atomicExch(s_overflow, 1u);
        return false;
    }
    if (r > s_threshold) return false;
    return atomicAdd(s_emitted, 1u) < cap_k;
}

// ONE PASS OF THE JOIN, compiled ONCE and called up to twice.
//
// __noinline__ is load-bearing, not a hint: the per-(state, rule) cap needs the join run twice --
// once to learn every match's rank, once to emit the chosen k -- and a lambda called twice
// instantiated this whole DFS twice, at which point ptxas ran out of memory assembling the file.
// A non-inlined function called twice is one body.
__device__ __noinline__ void match_state_rule_pass(
        DeviceState ds, const DeviceRule* rules, StateId state_id, uint32_t rid, uint32_t step,
        typename Pool<MatchRecord>::DeviceView out,
        bool capping, bool counting, uint32_t cap_k, uint32_t* s_seen, uint32_t* s_overflow,
        uint32_t* s_emitted, uint64_t s_threshold, uint64_t* s_ranks) {
    const DeviceRule& rule = rules[rid];

    if (rule.num_lhs_edges == 0) return;

    const MatchJoinCtx ctx{ds, rule, state_id};

    // A completed match. matched_edges is indexed by PATTERN position, not by depth.
    auto emit = [&] (const MatchJoinState& st) {
        if (capping && !drain_cap_admit(ds, state_id, rid, st.pattern, st.matched, st.depth,
                                        counting, cap_k, s_seen, s_overflow, s_emitted,
                                        s_threshold, s_ranks)) {
            return;
        }
        const uint32_t idx = out.claim();
        if (idx == Pool<MatchRecord>::kInvalid) {
            ds.errors.record(ErrorKind::kMatchPoolFull);
            return;
        }
        MatchRecord& m = out.at(idx);
        m.rule_id   = rid;
        m.state_id  = state_id;
        m.step      = step;
        m.num_edges = st.depth;
        for (uint8_t i = 0; i < kMaxPatternEdges; ++i) m.matched_edges[i] = INVALID_ID;
        for (uint8_t d = 0; d < st.depth; ++d) m.matched_edges[st.pattern[d]] = st.matched[d];
        publish_match(m);
    };

    // One thread, one depth-0 candidate, one whole DFS subtree: the join anchored at position 0.
    // A single-edge rule completes on the anchor itself and emits through this same path.
    auto run_dfs_from_root = [&] (EdgeId root_cand) {
        MatchJoinState st;
        hgcommon::join_seed(ctx, st, root_cand, 0, emit);
    };

    // Stride pattern edge 0's candidates across the block's threads. THIS is the device's part
    // of matching -- the parallelism -- and it is why match_state_rule exists rather than a
    // call straight into join_core. Small states index their slice directly; the
    // signature-bucket walk covers large states, where every thread traverses the bucket but
    // acts only on its own stripe.
    const DevicePatternEdge& pe0 = rule.lhs[0];
    StateEdgeSlice sl0 = ds.state_edge_slices[state_id];

    auto drive_join = [&] () {
        if (sl0.count <= ds.slice_scan_max_edges) {
            for (uint32_t i = threadIdx.x; i < sl0.count; i += blockDim.x) {
                run_dfs_from_root(ds.state_edge_ids[sl0.offset + i]);
            }
        } else {
            uint32_t cand_seen = 0;
            for (uint8_t s = 0; s < pe0.num_compat_sigs; ++s) {
                ds.signature_index.list.for_each(
                    static_cast<uint32_t>(pe0.compat_sig_hashes[s]) & ds.signature_index.mask,
                    [&] (EdgeId cand) {
                        if ((cand_seen % blockDim.x) == threadIdx.x) {
                            run_dfs_from_root(cand);
                        }
                        ++cand_seen;
                    });
            }
        }
    };

    drive_join();
}

// The entry point: one block, one (state, rule) pair. Without the cap this is a single pass and
// nothing about matching changes. With it, TWO -- because choosing k of M requires all M, and
// this block is where all M become known, which is the same completion point the host calls a
// state's drain. Capping as matches arrive would decide the kept set by schedule, and then the
// same seed would keep a different k on a different device, or on the same device twice.
__device__ void match_state_rule(DeviceState       ds,
                                 const DeviceRule* rules,
                                 StateId           state_id,
                                 uint32_t          rid,
                                 uint32_t          step,
                                 typename Pool<MatchRecord>::DeviceView out) {
    const uint32_t cap_k = ds.matches_per_state_rule;
    if (cap_k == 0u) {
        match_state_rule_pass(ds, rules, state_id, rid, step, out,
                              false, false, 0u, nullptr, nullptr, nullptr, ~0ULL, nullptr);
        return;
    }

    __shared__ uint32_t s_seen;
    __shared__ uint32_t s_overflow;
    __shared__ uint32_t s_emitted;
    __shared__ uint64_t s_threshold;
    __shared__ uint64_t s_ranks[kDrainCapBuffer];
    if (threadIdx.x == 0) { s_seen = 0; s_overflow = 0; s_emitted = 0; s_threshold = ~0ULL; }
    __syncthreads();

    match_state_rule_pass(ds, rules, state_id, rid, step, out, true, /*counting=*/true, cap_k,
                          &s_seen, &s_overflow, &s_emitted, ~0ULL, s_ranks);
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint32_t n = s_seen < kDrainCapBuffer ? s_seen : kDrainCapBuffer;
        if (s_overflow) {
            // More matches than the buffer can rank: the k-th smallest cannot be identified, so
            // the cap is NOT applied to this pair rather than applied to the wrong k. Recorded,
            // which puts the run under the engine's partial-result contract instead of silently
            // returning a differently-sampled answer.
            ds.errors.record(ErrorKind::kDrainCapBufferFull);
            s_threshold = ~0ULL;
        } else if (n > cap_k) {
            // The k-th smallest, k small: k passes of a min above a floor. Same shape the host's
            // drain uses, and it needs no sort and no scratch.
            uint64_t floor = 0; bool have_floor = false;
            for (uint32_t taken = 0; taken < cap_k; ++taken) {
                uint64_t best = ~0ULL; bool found = false;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint64_t r = s_ranks[i];
                    if (have_floor && r <= floor) continue;
                    if (r < best) { best = r; found = true; }
                }
                if (!found) break;
                floor = best; have_floor = true;
            }
            s_threshold = have_floor ? floor : ~0ULL;
        }
    }
    __syncthreads();

    match_state_rule_pass(ds, rules, state_id, rid, step, out, true, /*counting=*/false, cap_k,
                          &s_seen, &s_overflow, &s_emitted, s_threshold, s_ranks);
}

namespace {

// Batch driver: one block per (state, rule) pair of a state set.
__global__ void k_match_batch(DeviceState      ds,
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
// bounded list — instead of walking the global signature_index bucket.
//
// Greedy schedule: start with the first edge of rule.lhs that has at least
// one variable (any rule with a non-empty LHS). For each subsequent slot,
// pick the unplaced LHS edge whose variable set has largest overlap with
// already-bound variables. On ties, pick the edge with smallest signature
// bucket (heuristic — prefer more selective seeds). The pivot_var emitted
// for each edge ≥ 1 is one of the variables shared with the bound set
// (pick the first one found in the source LHS positional order for
// determinism).
//
// If a rule's LHS is disconnected (no overlap between some pair of components)
// the greedy picks one edge from the second component without a pivot —
// for safety we emit pivot_var = kNoPivotVar on that edge and the match
// kernel falls back to signature_index for it. In practice Wolfram rules
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
        // that case by falling back to signature_index for this edge.
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

    // Emit LHS in connectivity-scheduled order. The DFS binds edges in the
    // ORDER they appear in `d.lhs[]`, so we physically reorder here.
    auto schedule = schedule_lhs_edges(rule);
    for (uint8_t e = 0; e < d.num_lhs_edges; ++e) {
        const auto& sch = schedule[e];
        const auto& src = rule.lhs[sch.src_index];
        DevicePatternEdge& dst = d.lhs[e];
        dst.arity = static_cast<uint8_t>(src.size());
        for (uint8_t i = 0; i < dst.arity; ++i) dst.vars[i] = src[i];
        dst.pivot_var = sch.pivot_var;

        auto compats = compatible_signature_hashes(dst);
        if (compats.size() > kMaxCompatibleSigs) {
            throw std::runtime_error(
                "make_device_rule: pattern edge has more than kMaxCompatibleSigs"
                " compatible signatures (raise kMaxCompatibleSigs or reduce arity)");
        }
        dst.num_compat_sigs = static_cast<uint8_t>(compats.size());
        for (size_t k = 0; k < compats.size(); ++k) dst.compat_sig_hashes[k] = compats[k];
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

}  // namespace gpu
}  // namespace HG_NAMESPACE