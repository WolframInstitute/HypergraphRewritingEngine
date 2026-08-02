#include "hg_gpu/edge_signature.hpp"
#include "hgcommon/rewrite_core.hpp"  // shared with the host rewriter
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

// Online-TR redundancy oracle, the device twin of causal_graph.cpp::is_reachable: a candidate
// edge (p -> c) is redundant iff c is already reachable from p via kept edges, answered by a
// backward BFS from c over the reduced predecessor adjacency. No closure is stored, so keeping
// an edge costs one list push instead of an ancestors-x-descendants cross-product of map
// inserts, and memory is O(kept causal pairs).
//
// Event ids are monotone along every causal edge (a producer's event exists before its
// consumer's), so the search prunes to ids > p: anything smaller can neither be p nor have p
// as an ancestor. The search reads a settled sub-DAG: an ancestor completed its own causal
// registration before any state carrying its produced edges was enqueued, and the queue's
// release/acquire handshake orders that before c's rewrite.
//
// Scratch is a bounded local stack + open-addressed visited table. Overflow records
// kScratchOverflow and answers "not reachable", which KEEPS the candidate edge: the causal
// relation stays complete, only the reduction may retain a redundant edge.
constexpr uint32_t kReachStack   = 256;
constexpr uint32_t kReachVisited = 512;   // power of two; entries store id + 1, 0 = empty

__device__ bool is_reachable_preds(DeviceState ds, EventId p, EventId c) {
    if (p == c) return true;
    if (p >= c) return false;

    EventId  stack[kReachStack];
    uint32_t visited[kReachVisited];
    for (uint32_t i = 0; i < kReachVisited; ++i) visited[i] = 0;
    bool overflow = false;

    auto visit = [&](EventId x) -> bool {   // true iff newly inserted
        uint32_t slot = (x * 2654435761u) & (kReachVisited - 1u);
        for (uint32_t probe = 0; probe < kReachVisited; ++probe) {
            const uint32_t held = visited[slot];
            if (held == x + 1u) return false;
            if (held == 0u) { visited[slot] = x + 1u; return true; }
            slot = (slot + 1u) & (kReachVisited - 1u);
        }
        overflow = true;   // table full: treat as seen, which can only under-explore
        return false;
    };

    uint32_t sp = 0;
    stack[sp++] = c;
    visit(c);
    while (sp > 0) {
        const EventId x = stack[--sp];
        bool found = false;
        ds.preds_list.for_each(x, [&](EventId q) {
            if (found) return;
            if (q == p) { found = true; return; }
            if (q > p && visit(q)) {
                if (sp < kReachStack) stack[sp++] = q;
                else overflow = true;
            }
        });
        if (found) return true;
    }
    if (overflow) ds.errors.record(ErrorKind::kScratchOverflow);
    return false;
}

}  // namespace

// Try to add a causal edge (p → c via shared edge e). First-writer-wins via
// the causal_triple_dedup map. Multiplicity is preserved — distinct shared
// edges between the same (p, c) pair produce distinct triple keys and thus
// distinct CausalEdge entries. With TR enabled, redundancy is decided by the
// backward-reachability oracle, and a KEPT edge's only bookkeeping is one
// preds_list push per unique event pair. EXTERNAL linkage (declared in rewrite.hpp): the
// quotient-causal DP emits its canonical-event pairs through this same machinery.
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
        if (!pair_lookup.found && is_reachable_preds(ds, p, c)) return;
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
        // Record the kept edge in the reduced adjacency once per unique event pair (so
        // preds_list holds no duplicate producers), and mark the pair as seen — subsequent
        // edges between the same (p, c) skip the reachability check.
        auto pr = ds.causal_pair_dedup.insert_if_absent(pair_key, 1u);
        if (pr.inserted) {
            if (ds.preds_list.push(c, p) == INVALID_ID) {
                ds.errors.record(ErrorKind::kTrPredsNodes);
            }
        }
    }
}

namespace {

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
// Branchial edges connect sibling events of the same input state whose consumed
// edge sets overlap. Co-consumers are found through a per-(state, edge) index
// rather than a pairwise scan over all siblings, mirroring the CPU's
// state_edge_events_ design: each consumed edge is pushed then its bucket is
// walked, and push-then-scan guarantees that of any co-consuming pair at least
// one sees the other. Buckets are hashed, so an entry can belong to another
// state that consumed the same edge id (shared CSR edges) or to a colliding
// (state, edge) pair; matching the edge and then the other event's input state
// filters both, at one 4-byte read per candidate instead of scanning every
// sibling's consumed array. Pair-level dedup in try_add_branchial_edge keeps a
// pair sharing several edges single.
__device__ void register_branchial(DeviceState ds, EventId my_event, StateId input_state,
                                   const EdgeId* my_consumed, uint8_t my_num_consumed) {
    for (uint8_t i = 0; i < my_num_consumed; ++i) {
        EdgeId mine = my_consumed[i];
        if (mine == INVALID_ID) continue;
        uint64_t h = (static_cast<uint64_t>(input_state) << 32) | mine;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
        uint32_t bucket = static_cast<uint32_t>(h) & (ds.branchial_index.num_keys - 1u);
        uint64_t entry  = (static_cast<uint64_t>(my_event) << 32) | mine;
        if (ds.branchial_index.push(bucket, entry) == INVALID_ID) {
            ds.errors.record(ErrorKind::kBranchialIndexNodes);
            // Continue — co-consumers that pushed successfully still see us
            // when they walk (best-effort coverage, mirrors the old paths).
        }
        ds.branchial_index.for_each(bucket, [&](uint64_t other_entry) {
            if (static_cast<EdgeId>(other_entry) != mine) return;
            EventId other = static_cast<EventId>(other_entry >> 32);
            if (other == my_event) return;
            if (ds.event_pool.at(other).input_state != input_state) return;
            try_add_branchial_edge(ds, my_event, other, mine);
        });
    }
}

}  // namespace

// One match, applied by one THREAD: consumes the matched edges, produces the RHS edges, and
// emits the event. EXTERNAL linkage, so a scheduler in another translation unit drives this
// same implementation rather than growing a second copy; the helpers it calls stay file-local,
// which is fine since they are defined above it here.
//
// Returns the state it created AND the event it wrote, or a default-constructed AppliedMatch
// when a capacity claim failed. A scheduler that finishes the work itself needs both: the state
// to hash and re-enqueue, the event to stamp an identity onto once that hash exists.
// See docs/GPU_PERSISTENT_DESIGN.md.
__device__ AppliedMatch apply_one_match(DeviceState       ds,
                                        const DeviceRule* rules,
                                        const MatchRecord& m,
                                        uint32_t          step,
                                        unsigned long long* sub) {
    const DeviceRule&  rule = rules[m.rule_id];
    const unsigned long long t_start = clock64();


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
    const uint8_t num_new_vars = static_cast<uint8_t>(__popc(rule.new_var_mask));

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
                return AppliedMatch{};
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
        return AppliedMatch{};
    }

    // Reserve all RHS edges in one consecutive run.
    uint32_t first_eid = (rule.num_rhs_edges == 0)
        ? 0u
        : ds.edge_pool.claim_n(rule.num_rhs_edges);
    if (rule.num_rhs_edges > 0 && first_eid == Pool<Edge>::kInvalid) {
        ds.errors.record(ErrorKind::kEdgePoolFull);
        return AppliedMatch{};
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
        return AppliedMatch{};
    }

    // Reserve all vertex slots in one consecutive run.
    uint32_t first_vert_off = (vert_slots_needed == 0)
        ? 0u
        : ds.vertex_pool.claim_n(vert_slots_needed);
    if (vert_slots_needed > 0 && first_vert_off == Pool<VertexId>::kInvalid) {
        ds.errors.record(ErrorKind::kVertexPoolFull);
        return AppliedMatch{};
    }

    // Reserve fresh vertex IDs (vertex_high_water bump).
    uint32_t vid_base = 0;
    if (num_new_vars > 0) {
        vid_base = atomicAdd(ds.vertex_high_water,
                             static_cast<uint32_t>(num_new_vars));
        // vertex_inverted_index keys range over [0, num_keys).
        if (vid_base + num_new_vars > ds.vertex_inverted_index.list.num_keys) {
            ds.errors.record(ErrorKind::kVertexPoolFull);
            return AppliedMatch{};
        }
        // The fresh ids are consecutive from the high-water bump; which variable takes which
        // is the rewrite's rule and lives in hgcommon.
        VertexId merged[kMaxVars];
        #pragma unroll
        for (uint32_t v = 0; v < kMaxVars; ++v) merged[v] = binding[v];
        hgcommon::assign_fresh_consecutive(rule.new_var_mask, vid_base, merged);
        #pragma unroll
        for (uint32_t v = 0; v < kMaxVars; ++v) binding[v] = merged[v];
    }

    // -------------------------------------------------------------------
    // Commit: every reservation above succeeded, so from here on we write
    // freely into our reserved slots without further capacity checks.
    // -------------------------------------------------------------------
    const unsigned long long t_reserved = clock64();

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

        VertexId local_binding[kMaxVars];
        #pragma unroll
        for (uint32_t v = 0; v < kMaxVars; ++v) local_binding[v] = binding[v];
        VertexId local_verts[kMaxArity];
        // The device merges its fresh vertices into the binding, so the same array serves
        // as both sources.
        if (!hgcommon::resolve_rhs_vertices(re.vars, re.arity, local_binding, local_binding,
                                            local_verts)) {
            ds.errors.record(ErrorKind::kVertexPoolFull);
            return AppliedMatch{};
        }
        for (uint8_t i = 0; i < re.arity; ++i) {
            ds.vertex_pool.at(vert_off + i) = local_verts[i];
        }

        Edge ne{};
        ne.arity         = re.arity;
        ne.vertex_offset = vert_off;
        ne.signature     = signature_hash_from_vertices(local_verts, re.arity);
        ne.creator_event = my_event;
        ne.step          = step;
        ds.edge_pool.at(new_eid) = ne;

        // Indices are maintained only once some state has exceeded the slice-scan
        // threshold; below it the match kernels never read them, and skipping the
        // inserts avoids heavy CAS contention on hub-vertex and shared-signature
        // bucket heads. signature_index.insert / vertex_inverted_index.insert push
        // into LockFreeLists whose node pools may be full. Record softly — this
        // causes match-candidate misses, not memory corruption.
        if (ds.maintain_indices) {
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
        }

        if (r < kMaxPatternEdges) produced[r] = new_eid;
    }
    const unsigned long long t_emitted = clock64();

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
    // A state larger than the slice-scan threshold will be matched through the
    // indices, so raise the rebuild flag if they are not being maintained yet.
    if (!ds.maintain_indices && cursor > ds.slice_scan_max_edges) {
        atomicExch(ds.needs_indices, 1u);
    }
    const unsigned long long t_csr = clock64();

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
    const unsigned long long t_event = clock64();

    // Under the quotient-causal route the raw-edge rendezvous is replaced by the orbit-keyed
    // DP (quotient_causal.hpp), driven from the scheduler once the child is canonicalized --
    // which raw child wins the canonical slot must not decide the causal set. Mirrors the
    // rewriter.cpp gate. Branchial registration below stays on either way, as on the host.
    if (!ds.quotient_causal) {
    // 8. Causal rendezvous — producer side (our produced edges).
    for (uint8_t r = 0; r < rule.num_rhs_edges; ++r) {
        if (produced[r] != INVALID_ID) register_as_producer(ds, my_event, produced[r]);
    }

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

    for (uint8_t p = 0; p < n_cons; ++p) {
        EdgeId eid = consumed_sorted[p];
        if (eid != INVALID_ID) register_as_consumer(ds, my_event, eid);
    }
    }  // end !quotient_causal (raw-edge rendezvous)
    const unsigned long long t_causal = clock64();

    // 10. Branchial scan: our sibling events in the same input state.
    register_branchial(ds, my_event, m.state_id, ev.consumed_edges, rule.num_lhs_edges);

    if (sub) {
        atomicAdd(&sub[0], t_reserved - t_start);
        atomicAdd(&sub[1], t_emitted - t_reserved);
        atomicAdd(&sub[2], t_csr - t_emitted);
        atomicAdd(&sub[3], t_event - t_csr);
        atomicAdd(&sub[4], t_causal - t_event);
        atomicAdd(&sub[5], clock64() - t_causal);
    }

    return AppliedMatch{new_sid, my_event};
}

namespace {

// Batch driver: one thread per match in the pool.
__global__ void k_rewrite(DeviceState              ds,
                          const DeviceRule*        rules,
                          const MatchRecord*       matches,
                          uint32_t                 num_matches,
                          uint32_t                 step,
                          uint32_t                 tid_offset) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x + tid_offset;
    if (tid >= num_matches) return;
    apply_one_match(ds, rules, matches[tid], step);
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
    uint32_t grid = (num_matches + block - 1) / block;
    uint32_t cap  = engine.config().max_blocks_per_launch;
    if (cap == 0 || grid <= cap) {
        k_rewrite<<<grid, block>>>(engine.device(), d_rules, matches.view().data,
                                   num_matches, step, 0u);
    } else {
        for (uint32_t off = 0; off < grid; off += cap) {
            uint32_t n = (grid - off < cap) ? (grid - off) : cap;
            k_rewrite<<<n, block>>>(engine.device(), d_rules, matches.view().data,
                                    num_matches, step, off * (uint32_t)block);
            check(cudaDeviceSynchronize(), "k_rewrite chunk sync");
        }
    }
    check(cudaDeviceSynchronize(), "k_rewrite sync");
}

}  // namespace hg_gpu
