#include "hg_gpu/edge_signature.hpp"
#include "hg_gpu/rewrite.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <stdexcept>
#include <string>

namespace hg_gpu {

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::run_rewrite_kernel ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// Event + causal + branchial device helpers
// ---------------------------------------------------------------------------

__device__ uint64_t hash_causal_triple(EventId p, EventId c, EdgeId e) {
    uint64_t h = 14695981039346656037ULL;
    h ^= p; h *= 1099511628211ULL;
    h ^= c; h *= 1099511628211ULL;
    h ^= e; h *= 1099511628211ULL;
    // Guarantee non-zero (the ConcurrentMap uses 0 as EMPTY sentinel).
    if (h == 0) h = 1;
    return h;
}

__device__ uint64_t branchial_pair_key(EventId a, EventId b) {
    uint32_t lo = a < b ? a : b;
    uint32_t hi = a < b ? b : a;
    uint64_t k = (static_cast<uint64_t>(lo) << 32) | hi;
    if (k == 0) k = 1;
    return k;
}

// Transitive-reduction reachability check. Returns true iff `c` is already
// in Desc[p] (so causal edge (p → c) would be redundant).
__device__ bool is_reachable_via_desc(DeviceState ds, EventId p, EventId c) {
    uint64_t key = (static_cast<uint64_t>(p) << 32) | c;
    auto r = ds.desc_set.lookup(key);
    return r.found;
}

// Transitive closure update on adding causal edge (p, c). Mirrors
// causal_graph.cpp::update_transitive_closure. Collects ancestors of p
// (including p) and descendants of c (including c), then cross-inserts
// every (a, d) pair into Desc[a] and Anc[d]. Each insert is first checked
// against the set; only first-writer appends to the list.
//
// Uses bounded local scratch — kTrScratch caps the per-call reachability
// set size. Workloads that exceed the cap still get correct causal edges
// (add still proceeds), but the closure may be incomplete and some
// redundant edges may not be filtered. Auto-tuner will raise the cap if
// profiling shows it matters.
constexpr uint32_t kTrScratch = 512;

__device__ void update_tr(DeviceState ds, EventId p, EventId c) {
    EventId ancestors[kTrScratch];
    uint32_t n_anc = 0;
    bool anc_truncated = false;
    ancestors[n_anc++] = p;
    ds.anc_list.for_each(p, [&](EventId a) {
        if (n_anc < kTrScratch) ancestors[n_anc++] = a;
        else                    anc_truncated = true;
    });

    EventId descendants[kTrScratch];
    uint32_t n_desc = 0;
    bool desc_truncated = false;
    descendants[n_desc++] = c;
    ds.desc_list.for_each(c, [&](EventId d) {
        if (n_desc < kTrScratch) descendants[n_desc++] = d;
        else                     desc_truncated = true;
    });
    if (anc_truncated || desc_truncated) {
        ds.errors.record(ErrorKind::kScratchOverflow);
    }

    for (uint32_t i = 0; i < n_anc; ++i) {
        EventId a = ancestors[i];
        for (uint32_t j = 0; j < n_desc; ++j) {
            EventId d = descendants[j];
            if (a == d) continue;

            uint64_t desc_key = (static_cast<uint64_t>(a) << 32) | d;
            auto r = ds.desc_set.insert_if_absent(desc_key, 1u);
            if (r.inserted) {
                if (ds.desc_list.push(a, d) == INVALID_ID) {
                    ds.errors.record(ErrorKind::kDescListNodes);
                }
            }

            uint64_t anc_key = (static_cast<uint64_t>(d) << 32) | a;
            auto r2 = ds.anc_set.insert_if_absent(anc_key, 1u);
            if (r2.inserted) {
                if (ds.anc_list.push(d, a) == INVALID_ID) {
                    ds.errors.record(ErrorKind::kAncListNodes);
                }
            }
        }
    }
}

// Try to add a causal edge (p → c via shared edge e). First-writer-wins via
// the causal_triple_dedup map. Multiplicity is preserved — distinct shared
// edges between the same (p, c) pair produce distinct triple keys and thus
// distinct CausalEdge entries. With TR enabled, skips redundant edges by
// checking reachability first, and updates the transitive closure on each
// successful add.
__device__ void try_add_causal_edge(DeviceState ds, EventId p, EventId c, EdgeId e) {
    if (p == INVALID_ID || c == INVALID_ID || p == c) return;

    // Mirror CPU causal_graph.cpp::add_causal_edge:
    // - TR enabled AND pair (p,c) NOT yet seen: reject if reachable (redundant)
    // - TR enabled AND pair (p,c) already seen: always add (multiplicity —
    //   different shared edges between the same pair are all kept)
    // - TR disabled: always add
    uint64_t pair_key = (static_cast<uint64_t>(p) << 32) | c;
    if (ds.tr_enabled) {
        auto pair_lookup = ds.causal_pair_dedup.lookup(pair_key);
        if (!pair_lookup.found && is_reachable_via_desc(ds, p, c)) return;
    }

    uint64_t key = hash_causal_triple(p, c, e);
    auto r = ds.causal_triple_dedup.insert_if_absent(key, 1u);
    if (!r.inserted) return;  // already present (dup) — silently skip
    uint32_t idx = ds.causal_edge_pool.claim();
    if (idx == Pool<DeviceCausalEdge>::kInvalid) {
        ds.errors.record(ErrorKind::kCausalPoolFull);
        return;
    }
    ds.causal_edge_pool.at(idx) = DeviceCausalEdge{p, c, e};

    if (ds.tr_enabled) {
        update_tr(ds, p, c);
        // Mark the pair as seen — subsequent edges between the same (p, c)
        // skip the reachability check.
        ds.causal_pair_dedup.insert_if_absent(pair_key, 1u);
    }
}

__device__ void try_add_branchial_edge(DeviceState ds, EventId a, EventId b, EdgeId shared) {
    if (a == INVALID_ID || b == INVALID_ID || a == b) return;
    uint64_t key = branchial_pair_key(a, b);
    auto r = ds.branchial_pair_dedup.insert_if_absent(key, 1u);
    if (!r.inserted) return;  // already added (dup)
    uint32_t idx = ds.branchial_edge_pool.claim();
    if (idx == Pool<DeviceBranchialEdge>::kInvalid) {
        ds.errors.record(ErrorKind::kBranchialPoolFull);
        return;
    }
    EventId lo = a < b ? a : b;
    EventId hi = a < b ? b : a;
    ds.branchial_edge_pool.at(idx) = DeviceBranchialEdge{lo, hi, shared};
}

// Causal rendezvous: register this event as producer of `eid` (via atomic
// CAS on edge_producer[]), then iterate existing consumers and create causal
// edges for each.
__device__ void register_as_producer(DeviceState ds, EventId my_event, EdgeId eid) {
    cuda::atomic_ref<EventId, cuda::thread_scope_device> pref(ds.edge_producer[eid]);
    EventId expected = INVALID_ID;
    bool won = pref.compare_exchange_strong(
        expected, my_event,
        cuda::memory_order_release, cuda::memory_order_acquire);
    if (!won) return;  // another event already claimed this producer slot
    // We won. Iterate consumers already registered for this edge.
    ds.edge_consumers.for_each(eid, [&](EventId consumer) {
        try_add_causal_edge(ds, my_event, consumer, eid);
    });
}

// Causal rendezvous: register this event as consumer of `eid`, then read the
// producer (acquire). If set, create the causal edge. At least one side
// (producer or consumer) always detects the other because producer writes
// the slot before iterating consumers and consumer appends to the list
// before loading the slot.
__device__ void register_as_consumer(DeviceState ds, EventId my_event, EdgeId eid) {
    if (ds.edge_consumers.push(eid, my_event) == INVALID_ID) {
        ds.errors.record(ErrorKind::kEdgeConsumerNodes);
        // Don't return — we still want the producer-side detection so the
        // causal edge isn't lost; the missed-listing only affects future
        // consumers of this edge.
    }
    // After append, reload producer with acquire.
    cuda::atomic_ref<EventId, cuda::thread_scope_device> pref(ds.edge_producer[eid]);
    EventId p = pref.load(cuda::memory_order_acquire);
    if (p != INVALID_ID) {
        try_add_causal_edge(ds, p, my_event, eid);
    }
}

// Branchial scan: register this event to its input state's event list, then
// walk prior events and create a branchial edge for any pair sharing a
// consumed edge.
__device__ void register_branchial(DeviceState ds, EventId my_event, StateId input_state,
                                   const EdgeId* my_consumed, uint8_t my_num_consumed) {
    if (ds.state_events.push(input_state, my_event) == INVALID_ID) {
        ds.errors.record(ErrorKind::kStateEventNodes);
        // Continue — siblings still scan our event via their own pushes
        // (best-effort branchial coverage).
    }
    ds.state_events.for_each(input_state, [&](EventId other) {
        if (other == my_event) return;
        // Check shared consumed edge. We need `other`'s consumed list; read
        // it from the event_pool.
        const DeviceEvent& oev = ds.event_pool.at(other);
        for (uint8_t i = 0; i < my_num_consumed; ++i) {
            EdgeId mine = my_consumed[i];
            for (uint8_t j = 0; j < oev.num_consumed; ++j) {
                if (oev.consumed_edges[j] == mine) {
                    try_add_branchial_edge(ds, my_event, other, mine);
                    return;  // one shared edge is enough per (us, other) pair
                }
            }
        }
    });
}

__global__ void k_rewrite(DeviceState              ds,
                          const DeviceRule*        rules,
                          const MatchRecord*       matches,
                          uint32_t                 num_matches,
                          uint32_t                 step) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_matches) return;

    const MatchRecord& m = matches[tid];
    const DeviceRule&  rule = rules[m.rule_id];

    // 1. Re-derive var bindings from matched_edges. volatile to defeat an
    //    observed miscompile on nvcc with this kernel's register pressure
    //    (binding[i] read inconsistently across iterations of the RHS
    //    construction loop — see M6.4 debugging session).
    volatile VertexId binding[kMaxVars];
    #pragma unroll
    for (uint32_t v = 0; v < kMaxVars; ++v) binding[v] = INVALID_ID;
    for (uint8_t p = 0; p < rule.num_lhs_edges; ++p) {
        EdgeId dedge = m.matched_edges[p];
        if (dedge == INVALID_ID) continue;
        const Edge& e = ds.edge_pool.at(dedge);
        for (uint8_t i = 0; i < rule.lhs[p].arity && i < e.arity; ++i) {
            uint8_t v = rule.lhs[p].vars[i];
            binding[v] = ds.vertex_pool.at(e.vertex_offset + i);
        }
    }

    // -------------------------------------------------------------------
    // Preflight reservation: claim every capacity-bounded resource we need
    // before doing ANY mutation. If any claim fails, record the specific
    // error and abort leaving no half-initialized state. This replaces the
    // previous piecemeal "claim, then silently early-return mid-kernel"
    // pattern which left the new state's bitset uninitialized and produced
    // spurious OOBs in the WL hash / dedup downstream.
    // -------------------------------------------------------------------
    const uint8_t num_new_vars = (rule.num_rhs_vars > rule.num_lhs_vars)
        ? (rule.num_rhs_vars - rule.num_lhs_vars) : 0;

    // Total vertex slots needed across all RHS edges.
    uint32_t vert_slots_needed = 0;
    for (uint8_t r = 0; r < rule.num_rhs_edges; ++r) {
        vert_slots_needed += rule.rhs[r].arity;
    }

    // Reserve state slot. Use CAS-loop so we never bump state_count past
    // max_states — this keeps host-side downstream indexing safe without a
    // post-hoc cap.
    uint32_t new_sid;
    {
        uint32_t cur = *ds.state_count;
        for (;;) {
            if (cur >= ds.max_states) {
                ds.errors.record(ErrorKind::kStatePoolFull);
                return;
            }
            uint32_t prev = atomicCAS(ds.state_count, cur, cur + 1u);
            if (prev == cur) { new_sid = cur; break; }
            cur = prev;
        }
    }

    // Reserve event slot.
    EventId my_event = ds.event_pool.claim();
    if (my_event == Pool<DeviceEvent>::kInvalid) {
        ds.errors.record(ErrorKind::kEventPoolFull);
        return;
    }

    // Reserve all RHS edges in one consecutive run.
    uint32_t first_eid = (rule.num_rhs_edges == 0)
        ? 0u
        : ds.edge_pool.claim_n(rule.num_rhs_edges);
    if (rule.num_rhs_edges > 0 && first_eid == Pool<Edge>::kInvalid) {
        ds.errors.record(ErrorKind::kEdgePoolFull);
        return;
    }
    // Reserve the new state's CSR edge-list slice up front. Size is
    // parent.count - n_consumed + n_produced. Failure to reserve means
    // the per-step state-edge budget is exceeded — report and abort.
    StateEdgeSlice parent_slice = ds.state_edge_slices[m.state_id];
    uint32_t new_slice_count =
        parent_slice.count + rule.num_rhs_edges
        - rule.num_lhs_edges;  // assume all matched edges are in parent (match invariant)
    uint32_t new_slice_offset =
        (new_slice_count == 0) ? 0u
        : atomicAdd(ds.state_edge_ids_counter, new_slice_count);
    if (new_slice_count > 0 &&
        new_slice_offset + new_slice_count > ds.state_edge_ids_capacity) {
        ds.errors.record(ErrorKind::kStatePoolFull);
        return;
    }

    // Reserve all vertex slots in one consecutive run.
    uint32_t first_vert_off = (vert_slots_needed == 0)
        ? 0u
        : ds.vertex_pool.claim_n(vert_slots_needed);
    if (vert_slots_needed > 0 && first_vert_off == Pool<VertexId>::kInvalid) {
        ds.errors.record(ErrorKind::kVertexPoolFull);
        return;
    }

    // Reserve fresh vertex IDs (vertex_high_water bump).
    uint32_t vid_base = 0;
    if (num_new_vars > 0) {
        vid_base = atomicAdd(ds.vertex_high_water,
                             static_cast<uint32_t>(num_new_vars));
        // vertex_inverted_index keys range over [0, num_keys).
        if (vid_base + num_new_vars > ds.vertex_inverted_index.list.num_keys) {
            ds.errors.record(ErrorKind::kVertexPoolFull);
            return;
        }
        for (uint8_t i = 0; i < num_new_vars; ++i) {
            binding[rule.num_lhs_vars + i] = vid_base + i;
        }
    }

    // -------------------------------------------------------------------
    // Commit: every reservation above succeeded, so from here on we write
    // freely into our reserved slots without further capacity checks.
    // -------------------------------------------------------------------

    // For each RHS edge: claim edge record + indices. `produced[r]` is the
    // EdgeId we assigned to RHS edge r (equals first_eid + r by claim_n).
    EdgeId produced[kMaxPatternEdges];
    for (uint8_t i = 0; i < kMaxPatternEdges; ++i) produced[i] = INVALID_ID;

    uint32_t vert_cursor = first_vert_off;
    for (uint8_t r = 0; r < rule.num_rhs_edges; ++r) {
        const DeviceRhsEdge& re = rule.rhs[r];
        uint32_t new_eid  = first_eid + r;
        uint32_t vert_off = vert_cursor;
        vert_cursor += re.arity;

        for (uint8_t i = 0; i < re.arity; ++i) {
            ds.vertex_pool.at(vert_off + i) = binding[re.vars[i]];
        }

        Edge ne{};
        ne.arity         = re.arity;
        ne.vertex_offset = vert_off;
        VertexId local_verts[kMaxArity];
        for (uint8_t i = 0; i < re.arity; ++i) local_verts[i] = binding[re.vars[i]];
        ne.signature     = signature_hash_from_vertices(local_verts, re.arity);
        ne.creator_event = my_event;
        ne.step          = step;
        ds.edge_pool.at(new_eid) = ne;

        // signature_index.insert / vertex_inverted_index.insert push into
        // LockFreeLists whose node pools may be full. Record softly — this
        // causes match-candidate misses, not memory corruption.
        if (ds.signature_index.insert(new_eid, ne.signature) == INVALID_ID) {
            ds.errors.record(ErrorKind::kSigIndexNodes);
        }
        for (uint8_t i = 0; i < re.arity; ++i) {
            VertexId v = binding[re.vars[i]];
            if (v >= ds.vertex_inverted_index.list.num_keys) continue;
            if (ds.vertex_inverted_index.insert(v, new_eid) == INVALID_ID) {
                ds.errors.record(ErrorKind::kInvIndexNodes);
            }
        }

        if (r < kMaxPatternEdges) produced[r] = new_eid;
    }

    // Build the new state's CSR edge-list slice by merge-filtering parent
    // edges (dropping consumed ones) then appending produced edges.
    // Correctness relies on:
    //   (a) parent's slice is sorted ascending by EdgeId
    //   (b) m.matched_edges[] holds consumed-edge IDs which we sort here
    //   (c) produced EdgeIds are all > any parent edge (guaranteed by
    //       edge_pool.claim_n having issued a fresh consecutive run
    //       AFTER parent's edges were created in a prior step)
    // — so the result of "parent_minus_consumed ++ produced" is sorted.
    EdgeId consumed_asc[kMaxPatternEdges];
    uint8_t n_consumed_asc = rule.num_lhs_edges;
    for (uint8_t i = 0; i < n_consumed_asc; ++i) consumed_asc[i] = m.matched_edges[i];
    // Ascending insertion sort (n ≤ 16).
    for (uint8_t i = 1; i < n_consumed_asc; ++i) {
        EdgeId key = consumed_asc[i];
        int8_t j = static_cast<int8_t>(i) - 1;
        while (j >= 0 && consumed_asc[j] > key) {
            consumed_asc[j + 1] = consumed_asc[j];
            --j;
        }
        consumed_asc[j + 1] = key;
    }

    EdgeId* new_ids     = ds.state_edge_ids + new_slice_offset;
    const EdgeId* p_ids = ds.state_edge_ids + parent_slice.offset;
    uint32_t cursor = 0;
    uint8_t  ci     = 0;  // consumed cursor
    for (uint32_t pi = 0; pi < parent_slice.count; ++pi) {
        EdgeId e = p_ids[pi];
        while (ci < n_consumed_asc && consumed_asc[ci] < e) ++ci;
        if (ci < n_consumed_asc && consumed_asc[ci] == e) { ++ci; continue; }
        new_ids[cursor++] = e;
    }
    for (uint8_t r = 0; r < rule.num_rhs_edges; ++r) {
        new_ids[cursor++] = first_eid + r;
    }
    // Publish slice. Count may be < new_slice_count if some matched edges
    // were not found in the parent — shouldn't happen under the match
    // invariant, but clamp defensively.
    StateEdgeSlice sl{new_slice_offset, cursor};
    ds.state_edge_slices[new_sid] = sl;

    // 7. Write the Event record.
    DeviceEvent& ev = ds.event_pool.at(my_event);
    ev.id             = my_event;
    ev.canonical_id   = INVALID_ID;
    ev.input_state    = m.state_id;
    ev.output_state   = new_sid;
    ev.rule           = m.rule_id;
    ev.step           = step;
    ev.num_consumed   = rule.num_lhs_edges;
    ev.num_produced   = rule.num_rhs_edges;
    for (uint8_t i = 0; i < rule.num_lhs_edges && i < kMaxPatternEdges; ++i)
        ev.consumed_edges[i] = m.matched_edges[i];
    for (uint8_t i = rule.num_lhs_edges; i < kMaxPatternEdges; ++i)
        ev.consumed_edges[i] = INVALID_ID;
    for (uint8_t i = 0; i < rule.num_rhs_edges && i < kMaxPatternEdges; ++i)
        ev.produced_edges[i] = produced[i];
    for (uint8_t i = rule.num_rhs_edges; i < kMaxPatternEdges; ++i)
        ev.produced_edges[i] = INVALID_ID;

    __threadfence();  // make the event visible before any rendezvous reads it

    // [DIAGNOSIS] disabled temporarily
    #if 1
    // 8. Causal rendezvous — producer side (our produced edges).
    for (uint8_t r = 0; r < rule.num_rhs_edges; ++r) {
        if (produced[r] != INVALID_ID) register_as_producer(ds, my_event, produced[r]);
    }
    #endif

    // 9. Causal rendezvous — consumer side (our consumed edges).
    //
    // Sort consumed edges by descending producer-EventId so that online
    // TR correctly marks the later edges in the chain as redundant when
    // their producer is already reachable via an earlier (higher-EventId)
    // producer. Mirrors rewriter.cpp:145–172 on CPU.
    EdgeId consumed_sorted[kMaxPatternEdges];
    uint8_t  n_cons = rule.num_lhs_edges;
    for (uint8_t i = 0; i < n_cons; ++i) consumed_sorted[i] = m.matched_edges[i];

    // Insertion sort, descending by producer-EventId.
    for (uint8_t i = 1; i < n_cons; ++i) {
        EdgeId  key_eid = consumed_sorted[i];
        EventId key_prod = (key_eid != INVALID_ID) ? ds.edge_producer[key_eid] : INVALID_ID;
        int8_t j = static_cast<int8_t>(i) - 1;
        while (j >= 0) {
            EdgeId  cur_eid = consumed_sorted[j];
            EventId cur_prod = (cur_eid != INVALID_ID) ? ds.edge_producer[cur_eid] : INVALID_ID;
            // Treat INVALID_ID as the smallest (sort to end). Valid
            // EventIds compare by magnitude; we want descending, so move
            // cur_eid to position j+1 when cur_prod < key_prod.
            bool swap;
            if (key_prod == INVALID_ID)       swap = false;
            else if (cur_prod == INVALID_ID)  swap = true;
            else                              swap = (cur_prod < key_prod);
            if (!swap) break;
            consumed_sorted[j + 1] = consumed_sorted[j];
            --j;
        }
        consumed_sorted[j + 1] = key_eid;
    }

    #if 1
    for (uint8_t p = 0; p < n_cons; ++p) {
        EdgeId eid = consumed_sorted[p];
        if (eid != INVALID_ID) register_as_consumer(ds, my_event, eid);
    }

    // 10. Branchial scan: our sibling events in the same input state.
    register_branchial(ds, my_event, m.state_id, ev.consumed_edges, rule.num_lhs_edges);
    #endif
}

}  // namespace

uint32_t run_rewrite_kernel(EngineState&                   engine,
                            const std::vector<DeviceRule>& rules,
                            const Pool<MatchRecord>&       matches,
                            uint32_t                       num_matches,
                            uint32_t                       step) {
    if (num_matches == 0) return 0;

    DeviceRule* d_rules = nullptr;
    check(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    check(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
                     cudaMemcpyHostToDevice), "rules copy");

    uint32_t n = run_rewrite_kernel_with(engine, d_rules, matches, num_matches, step);
    cudaFree(d_rules);
    return n;
}

uint32_t run_rewrite_kernel_with(EngineState&             engine,
                                 const DeviceRule*        d_rules,
                                 const Pool<MatchRecord>& matches,
                                 uint32_t                 num_matches,
                                 uint32_t                 step) {
    if (num_matches == 0) return 0;
    const uint32_t state_count_before = engine.num_states_host();
    run_rewrite_kernel_with_nosync(engine, d_rules, matches, num_matches, step);
    uint32_t state_count_after = engine.num_states_host();
    return state_count_after - state_count_before;
}

void run_rewrite_kernel_with_nosync(EngineState&             engine,
                                    const DeviceRule*        d_rules,
                                    const Pool<MatchRecord>& matches,
                                    uint32_t                 num_matches,
                                    uint32_t                 step) {
    if (num_matches == 0) return;
    int block = 64;
    int grid  = (int)((num_matches + block - 1) / block);
    k_rewrite<<<grid, block>>>(engine.device(), d_rules, matches.view().data,
                               num_matches, step);
    check(cudaDeviceSynchronize(), "k_rewrite sync");
}

}  // namespace hg_gpu
