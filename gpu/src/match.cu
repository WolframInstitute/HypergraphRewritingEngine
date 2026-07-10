#include "hg_gpu/edge_signature.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/partial_match.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace hg_gpu {

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

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::run_match_kernel ") + what + ": " +
                                 cudaGetErrorString(err));
    }
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

// One thread per (state, rule). Recursive DFS extending PartialMatch one
// pattern edge at a time. Per pattern edge: union over every compatible
// signature bucket (Wolfram non-distinct binding allows distinct vars to
// collapse to the same data vertex, so the data signature may be coarser
// than the pattern's signature). For each candidate: filter exact compat,
// arity, in-state, not-consumed; bind vars (saving/restoring on backtrack);
// recurse; emit on full match.
// Candidate enumeration is adaptive on state size. Multiway states are small
// (tens of edges) while the signature and vertex-inverted indices are global
// across the whole evolution, so their buckets grow with total edge count and
// walking them costs O(evolution) per state. At or below this slice length,
// candidates come straight from the state's own CSR slice: O(|state|) work,
// each edge exactly once (no dedup buffer), state membership free. Above it
// (single huge states, the visualiser regime) the global indices win and the
// pivot/signature machinery is used. The two paths are strictly either/or:
// running both would enumerate a candidate twice and emit duplicate matches.
// The threshold lives in DeviceState::slice_scan_max_edges (EngineConfig knob),
// and also gates lazy index maintenance: below it the indices are never read.

__global__ void k_match_one_state(DeviceState ds,
                                  const DeviceRule* rules,
                                  uint32_t          num_rules,
                                  StateId           state_id,
                                  typename Pool<MatchRecord>::DeviceView out) {
    uint32_t rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= num_rules) return;

    PartialMatch pm;
    pm.reset(rules[rid].num_lhs_edges, rules[rid].num_lhs_vars);

    auto& rule = rules[rid];

    auto recurse = [&] (auto& self_ref, uint8_t depth) -> void {
        if (depth == rule.num_lhs_edges) {
            uint32_t idx = out.claim();
            if (idx == Pool<MatchRecord>::kInvalid) return;
            MatchRecord& m = out.at(idx);
            m.rule_id   = rid;
            m.state_id  = state_id;
            m.num_edges = depth;
            for (uint8_t i = 0; i < depth; ++i) m.matched_edges[i] = pm.matched_edges[i];
            for (uint8_t i = depth; i < kMaxPatternEdges; ++i) m.matched_edges[i] = INVALID_ID;
            return;
        }

        const DevicePatternEdge& pe = rule.lhs[depth];

        auto try_candidate = [&] (EdgeId cand) {
            const Edge& e = ds.edge_pool.at(cand);
            if (e.arity != pe.arity)              return;
            if (!state_contains(ds, state_id, cand)) return;
            if (pm.is_consumed(cand))             return;

            uint32_t saved_bound_mask = pm.bound_mask;
            VertexId saved_bindings[kMaxVars];
            #pragma unroll
            for (uint32_t v = 0; v < kMaxVars; ++v) saved_bindings[v] = pm.var_binding[v];

            bool ok = true;
            for (uint8_t i = 0; i < pe.arity; ++i) {
                VertexId vert = ds.vertex_pool.at(e.vertex_offset + i);
                if (!pm.check_or_bind_var(pe.vars[i], vert)) { ok = false; break; }
            }

            if (ok) {
                pm.bind_pattern_edge(depth, cand);
                pm.set_consumed(cand);
                self_ref(self_ref, depth + 1);
                pm.clear_consumed(cand);
                pm.unbind_pattern_edge(depth);
            }

            uint32_t newly_bound = pm.bound_mask & ~saved_bound_mask;
            while (newly_bound) {
                int v = __ffs(newly_bound) - 1;
                pm.var_binding[v] = saved_bindings[v];
                pm.bound_mask &= ~(1u << v);
                newly_bound &= newly_bound - 1;
            }
        };

        // At depth 0 (no bindings yet): seed from signature_index. At depth
        // ≥ 1 with a bound pivot_var: seed from vertex_inverted_index with
        // per-iteration dedup (see k_match_batch comment).
        StateEdgeSlice sl_ = ds.state_edge_slices[state_id];
        if (sl_.count <= ds.slice_scan_max_edges) {
            for (uint32_t i = 0; i < sl_.count; ++i) {
                try_candidate(ds.state_edge_ids[sl_.offset + i]);
            }
        } else if (depth > 0 && pe.pivot_var != kNoPivotVar) {
            VertexId pivot_vert = pm.var_binding[pe.pivot_var];
            // Per-iteration "seen" buffer for duplicate suppression; 256
            // keeps high-degree vertices in evolved states on the fast path.
            // Candidates are collected FIRST and tried only if the buffer
            // holds them all: exactly one of the two enumerators ever calls
            // try_candidate, because a candidate tried during collection
            // would be tried again by the signature walk after an overflow,
            // emitting duplicate match records (and the bucket order under
            // concurrent inserts would make the duplicate count vary run to
            // run). The signature walk is exact on its own: each edge
            // appears in its bucket exactly once, no dedup needed.
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
            if (overflowed) {
                for (uint8_t s = 0; s < pe.num_compat_sigs; ++s) {
                    ds.signature_index.list.for_each(
                        static_cast<uint32_t>(pe.compat_sig_hashes[s]) & ds.signature_index.mask,
                        try_candidate);
                }
            } else {
                for (uint32_t i = 0; i < n_seen; ++i) try_candidate(seen[i]);
            }
        } else {
            for (uint8_t s = 0; s < pe.num_compat_sigs; ++s) {
                ds.signature_index.list.for_each(
                    static_cast<uint32_t>(pe.compat_sig_hashes[s]) & ds.signature_index.mask,
                    try_candidate);
            }
        }
    };

    recurse(recurse, 0);
}

// Batched variant: one BLOCK per (state, rule) pair. Threads within the
// block cooperate by striding over candidates for pattern edge 0 — each
// thread runs the full DFS subtree from a different root candidate
// independently. This unblocks the previous one-thread-per-(state×rule)
// pathology where for n=1000, frontier=1, rule=1, only ONE GPU thread did
// all the work (944 ms on n1000_s1) while the rest of the device sat idle.
//
// Block layout:
//   gridDim.x  = num_state_ids * num_rules
//   blockDim.x = kMatchBlockThreads (warp-aligned; 32 by default)
//
// Cooperation pattern:
//   - All threads walk the same signature_index linked list (read-only,
//     cache-friendly), but each thread acts only on candidates whose
//     0-based seen-index has (idx % blockDim.x == threadIdx.x). The
//     list-walk itself is trivial work compared to the DFS.
//   - Each thread carries its OWN PartialMatch in registers/local mem.
//     No inter-thread coordination during DFS — outputs go straight to
//     the global match pool via atomicAdd.
//
// For pattern_edges == 1 (single-LHS rules) the DFS body collapses to
// "emit one MatchRecord per candidate", which still benefits because the
// emit-to-output is the only contended atomic and threads parallelise
// the candidate filter / variable-binding work.
constexpr uint32_t kMatchBlockThreads = 32;

__global__ void k_match_batch(DeviceState      ds,
                              const DeviceRule* rules,
                              uint32_t          num_rules,
                              const StateId*    state_ids,
                              uint32_t          num_state_ids,
                              typename Pool<MatchRecord>::DeviceView out) {
    uint32_t bid = blockIdx.x;
    uint32_t total = num_rules * num_state_ids;
    if (bid >= total) return;

    uint32_t state_idx = bid / num_rules;
    uint32_t rid       = bid - state_idx * num_rules;
    StateId  state_id  = state_ids[state_idx];
    const DeviceRule& rule = rules[rid];

    if (rule.num_lhs_edges == 0) return;

    const DevicePatternEdge& pe0 = rule.lhs[0];

    // Each thread maintains its own DFS state. The DFS body for depth ≥ 1
    // is the same as the original recursive lambda; we just specialise the
    // loop at depth 0 to iterate candidates striped across the block.
    auto run_dfs_from_root = [&] (EdgeId root_cand) {
        const Edge& e0 = ds.edge_pool.at(root_cand);
        if (e0.arity != pe0.arity)                return;
        if (!state_contains(ds, state_id, root_cand)) return;

        PartialMatch pm;
        pm.reset(rule.num_lhs_edges, rule.num_lhs_vars);

        // Try to bind pattern_edge 0 to root_cand.
        for (uint8_t i = 0; i < pe0.arity; ++i) {
            VertexId vert = ds.vertex_pool.at(e0.vertex_offset + i);
            if (!pm.check_or_bind_var(pe0.vars[i], vert)) return;
        }
        pm.bind_pattern_edge(0, root_cand);
        pm.set_consumed(root_cand);

        // Single-edge rule: emit and return.
        if (rule.num_lhs_edges == 1) {
            uint32_t idx = out.claim();
            if (idx == Pool<MatchRecord>::kInvalid) {
                ds.errors.record(ErrorKind::kMatchPoolFull);
                return;
            }
            MatchRecord& m = out.at(idx);
            m.rule_id   = rid;
            m.state_id  = state_id;
            m.num_edges = 1;
            m.matched_edges[0] = root_cand;
            for (uint8_t i = 1; i < kMaxPatternEdges; ++i) m.matched_edges[i] = INVALID_ID;
            return;
        }

        // Recursive DFS for depth ≥ 1. Same body as the single-thread
        // version; per-thread PartialMatch keeps it independent.
        auto recurse = [&] (auto& self_ref, uint8_t depth) -> void {
            if (depth == rule.num_lhs_edges) {
                uint32_t idx = out.claim();
                if (idx == Pool<MatchRecord>::kInvalid) {
                    ds.errors.record(ErrorKind::kMatchPoolFull);
                    return;
                }
                MatchRecord& m = out.at(idx);
                m.rule_id   = rid;
                m.state_id  = state_id;
                m.num_edges = depth;
                for (uint8_t i = 0; i < depth; ++i) m.matched_edges[i] = pm.matched_edges[i];
                for (uint8_t i = depth; i < kMaxPatternEdges; ++i) m.matched_edges[i] = INVALID_ID;
                return;
            }

            const DevicePatternEdge& pe = rule.lhs[depth];

            auto try_candidate = [&] (EdgeId cand) {
                const Edge& ec = ds.edge_pool.at(cand);
                if (ec.arity != pe.arity)               return;
                if (!state_contains(ds, state_id, cand)) return;
                if (pm.is_consumed(cand))               return;

                uint32_t saved_bound_mask = pm.bound_mask;
                VertexId saved_bindings[kMaxVars];
                #pragma unroll
                for (uint32_t v = 0; v < kMaxVars; ++v) saved_bindings[v] = pm.var_binding[v];

                bool ok = true;
                for (uint8_t i = 0; i < pe.arity; ++i) {
                    VertexId vert = ds.vertex_pool.at(ec.vertex_offset + i);
                    if (!pm.check_or_bind_var(pe.vars[i], vert)) { ok = false; break; }
                }

                if (ok) {
                    pm.bind_pattern_edge(depth, cand);
                    pm.set_consumed(cand);
                    self_ref(self_ref, depth + 1);
                    pm.clear_consumed(cand);
                    pm.unbind_pattern_edge(depth);
                }

                uint32_t newly_bound = pm.bound_mask & ~saved_bound_mask;
                while (newly_bound) {
                    int v = __ffs(newly_bound) - 1;
                    pm.var_binding[v] = saved_bindings[v];
                    pm.bound_mask &= ~(1u << v);
                    newly_bound &= newly_bound - 1;
                }
            };

            // Adapted-HGMatch candidate seeding at depth ≥ 1:
            //   - Normal case (connectivity-scheduled LHS): pivot_var is
            //     bound; iterate vertex_inverted_index[binding[pivot_var]]
            //     for a degree-bounded candidate list, deduplicating across
            //     occurrences (a self-loop {a,a} appears twice in list[a],
            //     and concurrent inserts from rewrite kernels can interleave
            //     these with other edges — so a per-iteration "seen" set
            //     is required, not just a last-seen check). For a single
            //     pattern edge each candidate must be tried at most once or
            //     we emit duplicate match records.
            //   - Fallback (rare, disconnected LHS): fall back to the
            //     signature_index walk.
            StateEdgeSlice sl_ = ds.state_edge_slices[state_id];
            if (sl_.count <= ds.slice_scan_max_edges) {
                for (uint32_t i = 0; i < sl_.count; ++i) {
                    try_candidate(ds.state_edge_ids[sl_.offset + i]);
                }
            } else if (pe.pivot_var != kNoPivotVar) {
                VertexId pivot_vert = pm.var_binding[pe.pivot_var];
                // Bounded "seen" buffer for duplicate suppression; 256
                // covers high-degree vertices in typical evolved Wolfram
                // states. Collect first, then try: exactly one of the two
                // enumerators ever calls try_candidate, because a candidate
                // tried during collection would be tried again by the
                // signature walk after an overflow, emitting duplicate
                // match records with a count that varies with concurrent
                // bucket-insertion order. The signature walk is exact on
                // its own: each edge appears in its bucket exactly once.
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
                        if (n_seen >= kMaxIncidentSeen) {
                            overflowed = true;
                            return;
                        }
                        seen[n_seen++] = cand;
                    });
                if (overflowed) {
                    for (uint8_t s = 0; s < pe.num_compat_sigs; ++s) {
                        ds.signature_index.list.for_each(
                            static_cast<uint32_t>(pe.compat_sig_hashes[s]) & ds.signature_index.mask,
                            try_candidate);
                    }
                } else {
                    for (uint32_t i = 0; i < n_seen; ++i) try_candidate(seen[i]);
                }
            } else {
                for (uint8_t s = 0; s < pe.num_compat_sigs; ++s) {
                    ds.signature_index.list.for_each(
                        static_cast<uint32_t>(pe.compat_sig_hashes[s]) & ds.signature_index.mask,
                        try_candidate);
                }
            }
        };

        recurse(recurse, 1);
    };

    // Stride pattern_edge_0 candidates across the block's threads. Small
    // states index their slice directly (each thread takes every 32nd edge);
    // the signature-bucket walk covers large states, where every thread
    // traverses the bucket but only acts on its own stripe.
    StateEdgeSlice sl0 = ds.state_edge_slices[state_id];
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
    DeviceRule d;
    d.num_lhs_edges = static_cast<uint8_t>(rule.lhs.size());
    d.num_lhs_vars  = rule.num_lhs_vars;
    d.num_rhs_edges = static_cast<uint8_t>(rule.rhs.size());
    d.num_rhs_vars  = rule.num_rhs_vars;

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

uint32_t run_match_kernel_batch(const EngineState& engine,
                                const DeviceRule*  d_rules,
                                uint32_t           num_rules,
                                const StateId*     d_state_ids,
                                uint32_t           num_state_ids,
                                Pool<MatchRecord>& out_matches) {
    if (num_rules == 0 || num_state_ids == 0) return 0;
    run_match_kernel_batch_nosync(engine, d_rules, num_rules,
                                  d_state_ids, num_state_ids, out_matches);
    return out_matches.size_host();
}

void run_match_kernel_batch_nosync(const EngineState& engine,
                                   const DeviceRule*  d_rules,
                                   uint32_t           num_rules,
                                   const StateId*     d_state_ids,
                                   uint32_t           num_state_ids,
                                   Pool<MatchRecord>& out_matches) {
    if (num_rules == 0 || num_state_ids == 0) return;
    // One block per (state, rule); threads inside the block parallelise
    // pattern-edge-0 candidate enumeration.
    uint32_t grid  = num_rules * num_state_ids;
    int      block = static_cast<int>(kMatchBlockThreads);
    k_match_batch<<<grid, block>>>(engine.device(), d_rules, num_rules,
                                   d_state_ids, num_state_ids, out_matches.view());
    check(cudaDeviceSynchronize(), "k_match_batch sync");
}

uint32_t run_match_kernel(const EngineState&             engine,
                          const std::vector<DeviceRule>& rules,
                          StateId                        state_id,
                          Pool<MatchRecord>&             out_matches) {
    if (rules.empty()) return 0;

    DeviceRule* d_rules = nullptr;
    check(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    check(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    out_matches.reset();

    int block = 32;
    int grid  = (int)((rules.size() + block - 1) / block);
    k_match_one_state<<<grid, block>>>(engine.device(), d_rules, (uint32_t)rules.size(),
                                       state_id, out_matches.view());
    check(cudaDeviceSynchronize(), "k_match_one_state sync");
    cudaFree(d_rules);

    return out_matches.size_host();
}

}  // namespace hg_gpu
