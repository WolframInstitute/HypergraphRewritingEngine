#include "hgcommon/namespace.hpp"
#include "hgcommon/reach_core.hpp"
#include "hg_gpu/edge_signature.hpp"
#include "hgcommon/core.hpp"          // id_key -- the packed-pair rule, shared with the host
#include "hgcommon/rewrite_core.hpp"  // shared with the host rewriter
#include "hgcommon/ir_core.hpp"       // IrSerial, the one-thread lane policy
#include "hg_gpu/rewrite.hpp"

#include "hg_gpu/exploration.hpp"
#include "hg_gpu/keyed.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

namespace {

namespace cg = cooperative_groups;

// An add of `n` to `counter` made by the threads that reach it together: one atomicAdd for the
// coalesced group, and each thread's share starts at the exclusive prefix of the shares of the
// threads ranked below it. A lone thread makes one atomicAdd of its own `n`, and a total of 0
// makes none. Returns what the thread's own atomicAdd would have returned had the group's adds
// run in rank order; a thread asking for 0 gets an unspecified value.
__device__ __forceinline__ uint32_t coalesced_add(uint32_t* counter, uint32_t n) {
    if (__popc(__activemask()) == 1) return n ? atomicAdd(counter, n) : 0u;
    const cg::coalesced_group g = cg::coalesced_threads();
    const uint32_t prefix = cg::exclusive_scan(g, n);
    const uint32_t total = g.shfl(prefix + n, g.size() - 1);
    uint32_t base = 0;
    if (g.thread_rank() == 0 && total) base = atomicAdd(counter, total);
    return g.shfl(base, 0) + prefix;
}

// A claim of one slot from `counter`, which never passes `limit`, made by the threads that reach
// it together: the group's first thread takes as many slots as fit in one exchange, and a thread
// ranked past them gets none. A lone thread makes its own exchange. Returns the slot, or
// INVALID_ID when none fit.
__device__ __forceinline__ uint32_t coalesced_bounded_claim(uint32_t* counter, uint32_t limit) {
    const auto take_from = [&](uint32_t want, uint32_t& base) {
        uint32_t cur = *counter;
        for (;;) {
            const uint32_t take = cur >= limit ? 0u : min(want, limit - cur);
            if (take == 0) return 0u;
            const uint32_t prev = atomicCAS(counter, cur, cur + take);
            if (prev == cur) { base = cur; return take; }
            cur = prev;
        }
    };
    uint32_t base = 0;
    if (__popc(__activemask()) == 1) return take_from(1u, base) ? base : INVALID_ID;
    const cg::coalesced_group g = cg::coalesced_threads();
    uint32_t take = 0;
    if (g.thread_rank() == 0) take = take_from(g.size(), base);
    base = g.shfl(base, 0);
    take = g.shfl(take, 0);
    return g.thread_rank() < take ? base + g.thread_rank() : INVALID_ID;
}

// ---------------------------------------------------------------------------
// Event + causal + branchial device helpers
// ---------------------------------------------------------------------------

__device__ uint64_t hash_causal_triple(EventId p, EventId c, EdgeId e) {
    uint64_t h = 14695981039346656037ULL;
    h ^= p; h *= 1099511628211ULL;
    h ^= c; h *= 1099511628211ULL;
    h ^= e; h *= 1099511628211ULL;
    // BOTH reserved keys, not just EMPTY. The map reserves 0 and ~0, and a hash is as able to
    // land on one as the other; hgcommon::qr_apply_key guards the pair too.
    if (h == 0) h = 1;
    if (h == ~0ULL) h = ~0ULL - 1ULL;
    return h;
}

__device__ uint64_t branchial_pair_key(EventId a, EventId b) {
    const uint32_t lo = a < b ? a : b;
    const uint32_t hi = a < b ? b : a;
    // hgcommon::id_key, which is where the offset that keeps a packed pair off the EMPTY
    // sentinel is stated for both engines. Packed here instead, the pair (0,0) would BE the
    // sentinel, and the `if (k == 0) k = 1` that guarded it was a second statement of a rule
    // that already has one.
    return hgcommon::id_key(lo, hi);
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
// The search runs in a local stack and open-addressed visited table. When either fills, the
// calling block's thread 0 runs it again in the block's slice of ds.tr_scratch, which is 8 times
// larger at tr_scratch_scale 1. Only when that fills too does the search record
// kTrScratchOverflow and answer "not reachable", which KEEPS the candidate edge: the causal
// relation stays complete and only the reduction may retain a redundant edge, until
// grow-and-retry doubles tr_scratch_scale and runs again.
constexpr uint32_t kReachStack   = 256;
constexpr uint32_t kReachVisited = 512;   // power of two; entries store id + 1, 0 = empty

__device__ bool is_reachable_preds(const DeviceState& ds, EventId p, EventId c) {
    auto preds = [&](uint32_t x, auto&& f) { ds.preds_list.for_each(x, f); };
    EventId  stack[kReachStack];
    uint32_t visited[kReachVisited];
    hgcommon::BoundedReachCtx<decltype(preds)> local(preds, stack, kReachStack, visited,
                                                     kReachVisited);
    if (hgcommon::reach_backward(local, p, c, /*topological=*/true)) return true;
    if (!local.overflow) return false;

    if (ds.tr_scratch != nullptr && threadIdx.x == 0 && blockIdx.x < ds.tr_scratch_slots) {
        uint32_t* slice = ds.tr_scratch + static_cast<size_t>(blockIdx.x) *
                                              (ds.tr_scratch_stack + ds.tr_scratch_visited);
        hgcommon::BoundedReachCtx<decltype(preds)> wide(preds, slice, ds.tr_scratch_stack,
                                                        slice + ds.tr_scratch_stack,
                                                        ds.tr_scratch_visited);
        if (hgcommon::reach_backward(wide, p, c, /*topological=*/true)) return true;
        if (!wide.overflow) return false;
    }
    ds.errors.record(ErrorKind::kTrScratchOverflow);
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
namespace {

// The views causal registration uses, loaded once per rewrite. Read through the DeviceState
// reference inside the consumer walk, each is reloaded after every atomic, since the reference
// may alias what the atomics write (the branchial walk's BranchialViews, for the same reason).
struct CausalViews {
    decltype(DeviceState::causal_pair_dedup)    pairs;
    decltype(DeviceState::causal_triple_dedup)  triples;
    decltype(DeviceState::causal_edge_pool)     pool;
    decltype(DeviceState::preds_list)           preds;
    decltype(DeviceState::edge_consumers)       consumers;
    EventId*                                    producer;
    DeviceErrors::DeviceView                    errors;
    bool                                        tr;
    __device__ explicit CausalViews(const DeviceState& ds)
        : pairs(ds.causal_pair_dedup), triples(ds.causal_triple_dedup),
          pool(ds.causal_edge_pool), preds(ds.preds_list), consumers(ds.edge_consumers),
          producer(ds.edge_producer), errors(ds.errors), tr(ds.tr_enabled != 0) {}
};

__device__ void add_causal_edge(const DeviceState& ds, const CausalViews& v, EventId p,
                                EventId c, EdgeId e) {
    if (p == INVALID_ID || c == INVALID_ID || p == c) return;

    // Mirror CPU causal_graph.cpp::add_causal_edge:
    // - TR enabled AND pair (p,c) NOT yet seen: reject if reachable (redundant)
    // - TR enabled AND pair (p,c) already seen: always add (multiplicity —
    //   different shared edges between the same pair are all kept)
    // - TR disabled: always add
    const uint64_t pair_key = hgcommon::id_key(p, c);
    if (v.tr) {
        auto pair_lookup = v.pairs.lookup(pair_key);
        if (!pair_lookup.found && is_reachable_preds(ds, p, c)) return;
    }

    uint64_t key = hash_causal_triple(p, c, e);
    auto r = v.triples.insert_if_absent(key, 1u);
    if (r.overflowed) {
        // With the map full, whether the triple is present is unknown. The edge is not added and
        // the overflow is recorded; grow-and-retry doubles causal_triple_slots.
        v.errors.record(ErrorKind::kCausalTripleMapFull);
        return;
    }
    if (!r.inserted) return;  // already present (dup) — silently skip
    uint32_t idx = v.pool.claim();
    if (idx == Pool<DeviceCausalEdge>::kInvalid) {
        v.errors.record(ErrorKind::kCausalPoolFull);
        return;
    }
    v.pool.at(idx) = DeviceCausalEdge{p, c, e};

    if (v.tr) {
        // Record the kept edge in the reduced adjacency once per unique event pair (so
        // preds_list holds no duplicate producers), and mark the pair as seen — subsequent
        // edges between the same (p, c) skip the reachability check. With the pair map full, the
        // producer is pushed: a repeated predecessor leaves reachability unchanged, and a
        // missing one leaves redundant edges in the reduction.
        auto pr = v.pairs.insert_if_absent(pair_key, 1u);
        if (pr.overflowed) v.errors.record(ErrorKind::kCausalPairMapFull);
        if (pr.inserted || pr.overflowed) {
            if (v.preds.push(c, p) == INVALID_ID) {
                v.errors.record(ErrorKind::kTrPredsNodes);
            }
        }
    }
}

}  // namespace

__device__ void try_add_causal_edge(const DeviceState& ds, EventId p, EventId c, EdgeId e) {
    add_causal_edge(ds, CausalViews(ds), p, c, e);
}

namespace {

// The views branchial registration uses, loaded once. Read through the DeviceState reference
// inside the bucket walk, each is reloaded after every atomic, since the reference may alias
// what the atomics write.
struct BranchialViews {
    decltype(DeviceState::branchial_index)       index;
    decltype(DeviceState::event_pool)            events;
    decltype(DeviceState::branchial_pair_dedup)  pairs;
    decltype(DeviceState::branchial_edge_pool)   pool;
    DeviceErrors::DeviceView                     errors;
};

__device__ __forceinline__ void try_add_branchial_edge(const BranchialViews& v, EventId a,
                                                       EventId b, EdgeId shared) {
    if (a == INVALID_ID || b == INVALID_ID || a == b) return;
    uint64_t key = branchial_pair_key(a, b);
    auto r = v.pairs.insert_if_absent(key, 1u);
    if (r.overflowed) {
        // The pair is not added and the overflow is recorded; grow-and-retry doubles
        // branchial_pair_slots.
        v.errors.record(ErrorKind::kBranchialMapFull);
        return;
    }
    if (!r.inserted) return;  // already added (dup)
    uint32_t idx = v.pool.claim();
    if (idx == Pool<DeviceBranchialEdge>::kInvalid) {
        v.errors.record(ErrorKind::kBranchialPoolFull);
        return;
    }
    EventId lo = a < b ? a : b;
    EventId hi = a < b ? b : a;
    v.pool.at(idx) = DeviceBranchialEdge{lo, hi, shared};
}

// Causal rendezvous: register this event as producer of `eid` (via atomic
// CAS on edge_producer[]), then iterate existing consumers and create causal
// edges for each.
__device__ void register_as_producer(const DeviceState& ds, const CausalViews& v,
                                     EventId my_event, EdgeId eid) {
    cuda::atomic_ref<EventId, cuda::thread_scope_device> pref(v.producer[eid]);
    EventId expected = INVALID_ID;
    bool won = pref.compare_exchange_strong(
        expected, my_event,
        cuda::memory_order_release, cuda::memory_order_acquire);
    if (!won) return;  // another event already claimed this producer slot
    // We won. Iterate consumers already registered for this edge.
    v.consumers.for_each(eid, [&](EventId consumer) {
        add_causal_edge(ds, v, my_event, consumer, eid);
    });
}

// Causal rendezvous: register this event as consumer of `eid`, then read the
// producer (acquire). If set, create the causal edge. At least one side
// (producer or consumer) always detects the other because producer writes
// the slot before iterating consumers and consumer appends to the list
// before loading the slot.
__device__ void register_as_consumer(const DeviceState& ds, const CausalViews& v,
                                     EventId my_event, EdgeId eid) {
    if (v.consumers.push(eid, my_event) == INVALID_ID) {
        v.errors.record(ErrorKind::kEdgeConsumerNodes);
        // Don't return — we still want the producer-side detection so the
        // causal edge isn't lost; the missed-listing only affects future
        // consumers of this edge.
    }
    // After append, reload producer with acquire.
    cuda::atomic_ref<EventId, cuda::thread_scope_device> pref(v.producer[eid]);
    EventId p = pref.load(cuda::memory_order_acquire);
    if (p != INVALID_ID) {
        add_causal_edge(ds, v, p, my_event, eid);
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
__device__ void register_branchial(const DeviceState& ds, EventId my_event, StateId input_state,
                                   const EdgeId* my_consumed, uint8_t my_num_consumed) {
    const BranchialViews v{ds.branchial_index, ds.event_pool, ds.branchial_pair_dedup,
                           ds.branchial_edge_pool, ds.errors};
    for (uint8_t i = 0; i < my_num_consumed; ++i) {
        EdgeId mine = my_consumed[i];
        if (mine == INVALID_ID) continue;
        uint64_t h = (static_cast<uint64_t>(input_state) << 32) | mine;
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
        uint32_t bucket = static_cast<uint32_t>(h) & (v.index.num_keys - 1u);
        uint64_t entry  = (static_cast<uint64_t>(my_event) << 32) | mine;
        if (v.index.push(bucket, entry) == INVALID_ID) {
            v.errors.record(ErrorKind::kBranchialIndexNodes);
            // Continue — co-consumers that pushed successfully still see us
            // when they walk (best-effort coverage, mirrors the old paths).
        }
        v.index.for_each(bucket, [&](uint64_t other_entry) {
            if (static_cast<EdgeId>(other_entry) != mine) return;
            EventId other = static_cast<EventId>(other_entry >> 32);
            if (other == my_event) return;
            if (v.events.at(other).input_state != input_state) return;
            try_add_branchial_edge(v, my_event, other, mine);
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
// See gpu/ARCHITECTURE.md sec 3.
__device__ AppliedMatch apply_one_match(const DeviceState& ds,
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

    // THE RESERVATIONS BELOW ARE COALESCED: the threads of a warp that reach each one together
    // make one atomic on its counter between them (coalesced_add, coalesced_bounded_claim),
    // where one atomic per thread on one address serialises the warp.
    //
    // Reserve the state slot. state_count never passes max_states, which keeps host-side
    // indexing safe without a post-hoc cap.
    const uint32_t new_sid = coalesced_bounded_claim(ds.state_count, ds.max_states);
    if (new_sid == INVALID_ID) {
        ds.errors.record(ErrorKind::kStatePoolFull);
        return AppliedMatch{};
    }

    // Reserve event slot.
    EventId my_event = ds.event_pool.settle(coalesced_add(ds.event_pool.counter, 1u), 1u);
    if (my_event == Pool<DeviceEvent>::kInvalid) {
        ds.errors.record(ErrorKind::kEventPoolFull);
        return AppliedMatch{};
    }

    // Reserve all RHS edges in one consecutive run.
    uint32_t first_eid = ds.edge_pool.settle(
        coalesced_add(ds.edge_pool.counter, rule.num_rhs_edges), rule.num_rhs_edges);
    if (rule.num_rhs_edges == 0) first_eid = 0u;
    if (rule.num_rhs_edges > 0 && first_eid == Pool<Edge>::kInvalid) {
        ds.errors.record(ErrorKind::kEdgePoolFull);
        return AppliedMatch{};
    }
    // Reserve the new state's CSR edge-list slice up front. Size is
    // parent.count - n_consumed + n_produced. Failure to reserve means
    // the per-step state-edge budget is exceeded — report and abort.
    StateEdgeSlice parent_slice = ds.state_edge_slices[m.state_id];
    // Widen before subtracting. The match invariant is that every consumed edge is in the
    // parent slice, which keeps this non-negative; computed in 32 bits, a state that broke it
    // would wrap to about four billion and reserve that.
    const uint64_t kept_and_produced =
        static_cast<uint64_t>(parent_slice.count) + static_cast<uint64_t>(rule.num_rhs_edges);
    const uint64_t consumed = static_cast<uint64_t>(rule.num_lhs_edges);
    if (kept_and_produced < consumed) {
        ds.errors.record(ErrorKind::kStatePoolFull);
        return AppliedMatch{};
    }
    const uint32_t new_slice_count = static_cast<uint32_t>(kept_and_produced - consumed);
    const uint32_t slice_at = coalesced_add(ds.state_edge_ids_counter, new_slice_count);
    const uint32_t new_slice_offset = (new_slice_count == 0) ? 0u : slice_at;
    if (new_slice_count > 0 &&
        static_cast<uint64_t>(new_slice_offset) + new_slice_count
            > ds.state_edge_ids_capacity) {
        // CLAMP THE COUNTER, because the add above happens before this check and is never
        // rolled back: every failing reservation still advances it. Left alone it climbs
        // through the whole run and eventually past 2^32, where it wraps and hands a later
        // reservation a small offset that passes this bound and writes outside the allocation.
        // Pulling it back to the ceiling on the failing path bounds the excess to what is in
        // flight, so it cannot reach the wrap. AtomicPool::claim_n widens the same comparison
        // for the same reason.
        atomicMin(ds.state_edge_ids_counter, ds.state_edge_ids_capacity);
        ds.errors.record(ErrorKind::kStatePoolFull);
        return AppliedMatch{};
    }

    // Reserve all vertex slots in one consecutive run.
    uint32_t first_vert_off = ds.vertex_pool.settle(
        coalesced_add(ds.vertex_pool.counter, vert_slots_needed), vert_slots_needed);
    if (vert_slots_needed == 0) first_vert_off = 0u;
    if (vert_slots_needed > 0 && first_vert_off == Pool<VertexId>::kInvalid) {
        ds.errors.record(ErrorKind::kVertexPoolFull);
        return AppliedMatch{};
    }

    // Reserve fresh vertex IDs (vertex_high_water bump).
    uint32_t vid_base = 0;
    const uint32_t fresh_at = coalesced_add(ds.vertex_high_water, num_new_vars);
    if (num_new_vars > 0) {
        vid_base = fresh_at;
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

    // For each RHS edge: claim edge record + indices. RHS edge r is edge first_eid + r.
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
                if (!first_occurrence([&](uint8_t k) { return binding[re.vars[k]]; }, i)) continue;
                if (ds.vertex_inverted_index.insert(v, new_eid) == INVALID_ID) {
                    ds.errors.record(ErrorKind::kInvIndexNodes);
                }
            }
        }
    }
    const unsigned long long t_emitted = clock64();

    // The new state's CSR slice is the parent's edges minus the consumed ones, in parent order,
    // then the produced ones. The parent's slice is ascending and produced ids are above every
    // parent edge (edge_pool.claim_n issued them after the parent's edges existed), so the slice
    // is ascending. The kept count is known here, so the produced ids go after it and
    // copy_kept_edges fills the kept part.
    const uint32_t n_kept = new_slice_count - rule.num_rhs_edges;
    EdgeId* new_ids = ds.state_edge_ids + new_slice_offset;
    for (uint8_t r = 0; r < rule.num_rhs_edges; ++r) new_ids[n_kept + r] = first_eid + r;
    ds.state_edge_slices[new_sid] = StateEdgeSlice{new_slice_offset, new_slice_count};
    if (!ds.maintain_indices && new_slice_count > ds.slice_scan_max_edges) {
        atomicExch(ds.needs_indices, 1u);
    }
    KeptCopy kept{};
    kept.src_offset = parent_slice.offset;
    kept.src_count  = parent_slice.count;
    kept.dst_offset = new_slice_offset;
    kept.n_consumed = rule.num_lhs_edges;
    for (uint8_t i = 0; i < rule.num_lhs_edges && i < kMaxPatternEdges; ++i)
        kept.consumed[i] = m.matched_edges[i];
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
    ev.rewrite_id     = hgcommon::REWRITE_ID_UNSET;
    ev.first_produced = first_eid;
    ev.consumed_at    = my_event * ds.event_consumed_stride;
    for (uint8_t i = 0; i < rule.num_lhs_edges && i < ds.event_consumed_stride; ++i)
        ds.event_consumed[ev.consumed_at + i] = m.matched_edges[i];

    __threadfence();  // make the event visible before any rendezvous reads it
    const uint32_t keyed =
        ds.keyed.enabled
            ? keyed_after_rewrite(ds, m, my_event, new_sid, first_eid, rule.num_rhs_edges)
            : 0u;
    const unsigned long long t_event = clock64();

    // Under the quotient route the raw-edge rendezvous is off: the replay reconstructs the
    // relations (quotient_expansion.hpp), so which raw child wins the canonical slot does not
    // decide them. Mirrors the rewriter.cpp gate. Branchial registration below stays on either
    // way, as on the host.
    if (!ds.quotient_causal) {
    // 8. Causal rendezvous — producer side (our produced edges).
    const CausalViews causal(ds);
    for (uint8_t r = 0; r < rule.num_rhs_edges; ++r) {
        if (ds.record_causal) register_as_producer(ds, causal, my_event, first_eid + r);
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
        if (eid != INVALID_ID && ds.record_causal) register_as_consumer(ds, causal, my_event, eid);
    }
    }  // end !quotient_causal (raw-edge rendezvous)
    const unsigned long long t_causal = clock64();

    // 10. Branchial scan: our sibling events in the same input state.
    if (ds.record_branchial)
        register_branchial(ds, my_event, m.state_id, m.matched_edges, rule.num_lhs_edges);

    if (sub) {
        atomicAdd(&sub[0], t_reserved - t_start);
        atomicAdd(&sub[1], t_emitted - t_reserved);
        atomicAdd(&sub[2], t_csr - t_emitted);
        atomicAdd(&sub[3], t_event - t_csr);
        atomicAdd(&sub[4], t_causal - t_event);
        atomicAdd(&sub[5], clock64() - t_causal);
    }

    return AppliedMatch{new_sid, my_event, kept, keyed};
}

namespace {

// Batch driver: one thread per match in the pool.
__global__ void k_rewrite(const __grid_constant__ DeviceState ds,
                          const DeviceRule*        rules,
                          const MatchRecord*       matches,
                          uint32_t                 num_matches,
                          uint32_t                 step,
                          uint32_t                 tid_offset) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x + tid_offset;
    if (tid >= num_matches) return;
    const AppliedMatch a = apply_one_match(ds, rules, matches[tid], step);
    if (a.state != INVALID_ID) copy_kept_edges(ds, a.kept, hgcommon::IrSerial{});
}

}  // namespace

namespace {
__global__ void k_redundant_edge_over_chain(const __grid_constant__ DeviceState ds, uint32_t n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (uint32_t k = 1; k <= n + 1; ++k) ds.preds_list.push(k + 1, k);
    try_add_causal_edge(ds, 1u, n + 2u, 0u);
}
}  // namespace

void add_redundant_edge_over_chain(EngineState& engine, uint32_t n) {
    k_redundant_edge_over_chain<<<1, kMatchBlockThreads>>>(engine.device(), n);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "add_redundant_edge_over_chain sync");
}

uint32_t run_rewrite_kernel(EngineState&                   engine,
                            const std::vector<DeviceRule>& rules,
                            const Pool<MatchRecord>&       matches,
                            uint32_t                       num_matches,
                            uint32_t                       step) {
    if (num_matches == 0) return 0;

    DeviceRule* d_rules = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_rules, sizeof(DeviceRule) * rules.size()), "rules alloc");
    HG_CUDA_CHECK(cudaMemcpy(d_rules, rules.data(), sizeof(DeviceRule) * rules.size(),
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
            HG_CUDA_CHECK(cudaDeviceSynchronize(), "k_rewrite chunk sync");
        }
    }
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "k_rewrite sync");
}

}  // namespace gpu
}  // namespace HG_NAMESPACE