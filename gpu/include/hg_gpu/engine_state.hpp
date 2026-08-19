#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/errors.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/hash_table.hpp"
#include "hg_gpu/lock_free_list.hpp"
#include "hg_gpu/signature_index.hpp"
#include "hg_gpu/types.hpp"
#include "hg_gpu/vertex_inverted_index.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// EngineConfig is declared in evolve.hpp so host-only translation units
// (the bench harness, the differential test driver) can include it without
// pulling in cuda_runtime.h. The full definition lives there; this header
// transitively re-exports it via #include "hg_gpu/evolve.hpp".

// Device-side POD passed to kernels. All pointers refer to memory owned by
// EngineState (host side); EngineState's lifetime brackets every kernel run
// that uses it.
struct DeviceState {
    // Pools
    typename Pool<VertexId>::DeviceView vertex_pool;
    typename Pool<Edge>::DeviceView     edge_pool;

    // State storage — CSR per-state edge lists.
    // state_edge_slices[sid] = {offset, count} into state_edge_ids.
    // See StateEdgeSlice comment in types.hpp.
    StateEdgeSlice* state_edge_slices;       // [max_states]
    EdgeId*         state_edge_ids;          // [state_edge_ids_capacity]
    uint32_t*       state_edge_ids_counter;  // device atomic; next slot in state_edge_ids
    uint32_t        state_edge_ids_capacity;
    uint32_t        max_states;

    // State allocator (atomic-bumped)
    uint32_t* state_count;            // device atomic; current num_states

    // Exact canonical hash per state, 0 until computed. [max_states]
    //
    // A state's hash is computed once, when the state is created; an EVENT identity needs the
    // hash of its INPUT state, which was computed long before that event existed. Without
    // somewhere to keep it, a device-resident scheduler would have to recanonicalize the parent
    // for every transition out of it -- which is the same state hashed once per child.
    uint64_t* state_canonical_hash;

    // The EXACT isomorphism hash per state, 0 until computed. [max_states]
    //
    // Separate from the above because they are different questions and only coincide in Full
    // mode. Event identity is defined over isomorphism classes INDEPENDENTLY of the
    // state-identity choice (SPEC.md sec 4), so an event needs this even when states are being
    // identified by content or not at all.
    //
    // Filled only when the run's event identity needs it. Under EventCanonicalizationMode::None
    // nothing reads it, and computing it would be an individualization-refinement pass per
    // state bought for nobody.
    uint64_t* state_exact_hash;

    // Canonical rank of each edge WITHIN its own state, laid out parallel to state_edge_ids so
    // slot i holds the rank of the edge in slot i. [state_edge_ids_capacity], UINT32_MAX where
    // no rank exists. Written by the same individualization-refinement pass that produces
    // state_exact_hash, so the ranks are free once that pass runs.
    //
    // Null unless the run's event identity keys on consumed or produced edges
    // (EVENT_SIG_AUTOMATIC). It is four bytes per edge SLOT rather than per state, so on a
    // large configuration it is the biggest single allocation here and is not worth taking for
    // a run whose events are identified by their endpoint states alone.
    uint32_t* state_edge_rank;

    // Per-slot edge automorphism ORBIT (UINT32_MAX where none) plus per-state orbit counts,
    // parallel to state_edge_ids / indexed by state. Written by the same IR pass; null unless
    // the run does quotient-causal reconstruction, whose DP keys on orbits.
    uint32_t* state_edge_orbit;
    uint32_t* state_num_orbits;

    // Consumed or produced edges whose rank was unavailable when an event was stamped, so the
    // raw edge id stood in. Such a signature is not an isomorphism invariant; counting them is
    // what lets a caller comparing event counts across devices see that it happened.
    uint32_t* event_sig_raw_fallbacks;

    // Events that won their signature slot, which is the count a caller means by "how many
    // events" once an identity mode is selected. Null under EventSignatureKeys None, where no
    // signature is computed and every application is its own event.
    uint32_t* canonical_event_count;

    // Vertex allocator (atomic-bumped fresh-vertex counter)
    uint32_t* vertex_high_water;      // monotonic max VertexId issued + 1

    // Indices
    SignatureIndex::DeviceView        signature_index;
    VertexInvertedIndex::DeviceView   vertex_inverted_index;

    // Events and causal/branchial structures
    typename Pool<DeviceEvent>::DeviceView          event_pool;
    typename Pool<DeviceCausalEdge>::DeviceView     causal_edge_pool;
    typename Pool<DeviceBranchialEdge>::DeviceView  branchial_edge_pool;

    // Per-edge producer slot (atomic EventId, INVALID_ID if not yet produced)
    EventId* edge_producer;
    // Per-edge consumer list (LockFreeList keyed by EdgeId)
    // Which artifacts this run records. Read by the rewrite kernel before each rendezvous, so
    // an artifact nobody asked for costs no map inserts and no list nodes.
    // Automorphism generators the IR search may retain, per thread. Carried here rather than
    // read from a constant so grow-and-retry can raise it: orbits are fused over the
    // generators found, so a short table yields orbits that are too FINE and the quotient
    // reconstruction keys instance identity on them.
    uint32_t ir_generators;
    // Individualization depth the IR search explores. Carried here for the same reason as the
    // generator budget: grow-and-retry raises it, because the alternative to searching deeper
    // is a key that merges non-isomorphic states.
    uint32_t ir_depth;
    uint32_t record_causal;
    uint32_t record_branchial;

    typename LockFreeList<EventId>::DeviceView edge_consumers;
    // Per-state event list (LockFreeList keyed by raw StateId) — used by
    // branchial scan to find prior sibling events from the same input state
    // Branchial co-consumer index: bucket = hash(state, edge) & (num_keys - 1),
    // entry packs (event << 32 | edge). Bucket collisions across states are
    // disambiguated by the input-state check in register_branchial.
    typename LockFreeList<uint64_t>::DeviceView branchial_index;

    // Dedup maps. causal_triple_dedup ensures exactly one CausalEdge record
    // per (p, c, shared_edge) triple (preserves multiplicity across distinct
    // shared edges). causal_pair_dedup tracks whether the (p, c) pair has
    // been seen at all — used by online TR so the first-time-reachability
    // check is skipped for subsequent edges between the same pair.
    ConcurrentMap<uint64_t, uint32_t>::DeviceView causal_triple_dedup;
    ConcurrentMap<uint64_t, uint32_t>::DeviceView causal_pair_dedup;
    ConcurrentMap<uint64_t, uint32_t>::DeviceView branchial_pair_dedup;

    // Transitive reduction: Desc[event] (events reachable from event) and
    // Anc[event] (events that reach event). Lists are iterable; sets are
    // O(1) "is x in Desc[e]".
    // Online TR's reduced predecessor adjacency: preds_list[c] holds the producer events of
    // c's kept causal edges, one node per unique kept (producer, consumer) pair. Reachability
    // queries walk it backward; no closure is stored. Device twin of CausalGraph::preds_.
    typename LockFreeList<EventId>::DeviceView    preds_list;

    // Flags
    bool tr_enabled;
    // Quotient-causal route: when set, apply_one_match SKIPS the raw-edge producer/consumer
    // rendezvous -- causal edges come from the orbit-keyed DP instead (quotient_causal.hpp),
    // which is schedule-independent where the raw rendezvous under quotient is not. Branchial
    // registration stays on either way, as on the host.
    bool quotient_causal;

    // Index maintenance is lazy. Small states are matched by scanning their own
    // CSR slice, so the signature and vertex-inverted indices are read only once
    // some state exceeds slice_scan_max_edges. Until then inserts are skipped;
    // the rewrite kernel raises *needs_indices when it publishes a larger state,
    // and the host rebuilds both indices from the edge pool before the next
    // match launch, then keeps them maintained.
    uint32_t  slice_scan_max_edges;
    uint32_t  maintain_indices;   // 0/1, host-set, read per launch
    uint32_t* needs_indices;      // device flag, raised by the rewrite kernel

    // Error channel: kernels record overflow reasons here instead of silently
    // bailing on partial work. Host inspects after every kernel sync.
    DeviceErrors::DeviceView errors;
};

// Position of `edge` in state `sid`'s CSR slice, or UINT32_MAX if the edge is not in it. The
// slice is sorted ascending, so this is a binary search.
//
// ONE function, because EVERY per-edge array above is laid out parallel to state_edge_ids and
// indexed by the same position: the orbit, the canonical rank, and the frame slot derived from
// the orbit. Three callers each binary-searching the same slice for a different array is three
// copies of one lookup.
__device__ __forceinline__ uint32_t state_edge_index(const DeviceState& ds, StateId sid,
                                                     EdgeId edge) {
    if (sid >= ds.max_states) return UINT32_MAX;
    const StateEdgeSlice sl = ds.state_edge_slices[sid];
    uint32_t lo = 0, hi = sl.count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (ds.state_edge_ids[sl.offset + mid] < edge) lo = mid + 1; else hi = mid;
    }
    if (lo >= sl.count || ds.state_edge_ids[sl.offset + lo] != edge) return UINT32_MAX;
    return sl.offset + lo;
}

class EngineState {
public:
    // Per-thread device stack. See the constructor for why the default is not enough.
    //
    // The floor covers the kernels whose stack need is fixed: match_state_rule's DFS recurses
    // to the LHS edge count and apply_one_match holds several kMaxPatternEdges arrays, both
    // bounded by kMaxPatternEdges.
    static constexpr size_t kDeviceStackFloorBytes = 32u * 1024u;

    // The reconstruction's replay is different in kind: qe_apply -> qr_apply -> descend ->
    // qe_add_instance -> qe_drive_instance -> qe_apply is a cycle, and it descends once per
    // reconstruction DEPTH, which the caller chooses through num_steps.
    //
    // DERIVED FROM TWO MEASUREMENTS, because the cycle gained frames when the replay moved to
    // hgcommon and a per-level cost measured for the old shape does not carry to the new one.
    //
    //   1. Fault bisection on the FOUR-frame cycle, the method this constant was first set by:
    //      on sm_89 a 32 KB stack faults entering depth 7 and a 64 KB stack entering depth 13,
    //      so 32768/6 == 65536/12 == 5461 bytes per level, linear with no significant intercept.
    //   2. Per-frame `.local` depots, read out of the built PTX by
    //      tools/dev/ptx_frame_sizes.py --cycle:
    //        four-frame cycle (replay open-coded here)  2000 bytes over 4 frames
    //        six-frame cycle  (replay in hgcommon)      3168 bytes over 6 frames
    //
    // A level costs its depots PLUS the ABI save area, which the depots do not include and the
    // PTX does not name. The first measurement pins that: (5461 - 2000) / 4 == 865 bytes of ABI
    // per frame. So six frames cost 3168 + 6*865 == 8360, and this is that rounded up to the
    // next multiple of 512 -- a 4.1% margin, against the 3.1% the four-frame value carried.
    //
    // THE ABI TERM IS AN AVERAGE. It varies per function with the registers each saves, so 865
    // is not a constant of the hardware. If the cycle changes shape again, re-run the depot tool
    // AND re-do the fault bisection rather than reusing this arithmetic.
    //
    // TO RE-DERIVE IT after changing anything the cycle calls, run
    //
    //     tools/dev/ptx_frame_sizes.py <build>/gpu/CMakeFiles/hg_gpu.dir/src/persistent.cu.o --cycle
    //
    // which reads each frame's `.local` depot out of the PTX of an object that is already built
    // -- no GPU, no run, and it names WHICH frame moved. Two terms make up a level: the depot
    // sum it reports, and the ABI save area of the frames in the cycle, which it cannot see. So
    // a change that only adds BYTES moves the reported sum and a change that adds a CALL does
    // not, while costing a level's worth of ABI frame all the same. Both invalidate this number,
    // and the fault-bisection above is what settles the total.
    static constexpr size_t kDeviceStackBytesPerDepth = 8704;

    // Stack is per-thread and the driver reserves it for every resident thread, so this is
    // multiplied by the occupancy of the whole device -- it cannot simply be made large. Past
    // this the replay is bounded instead and records kScratchOverflow, which is the overflow
    // contract: partial work and a warning, never a fault.
    static constexpr size_t kDeviceStackCapBytes = 256u * 1024u;

    static size_t stack_bytes_for_depth(uint32_t depth) {
        const size_t want = kDeviceStackFloorBytes +
                            static_cast<size_t>(depth + 1u) * kDeviceStackBytesPerDepth;
        if (want < kDeviceStackFloorBytes) return kDeviceStackFloorBytes;
        return want > kDeviceStackCapBytes ? kDeviceStackCapBytes : want;
    }

    // How deep the replay may recurse on the stack this engine actually got. One level is
    // reserved so the guard fires before the frame that would fault.
    uint32_t qe_max_recursion_depth() const { return qe_max_recursion_depth_; }

    explicit EngineState(EngineConfig cfg)
        : cfg_(cfg)
        , vertex_pool_(cfg.max_vertex_slots)
        , edge_pool_(cfg.max_edges)
        , signature_index_(cfg.sig_index_buckets, cfg.sig_index_pool)
        , vertex_inverted_index_(cfg.max_vertices, cfg.inverted_pool)
        , event_pool_(cfg.max_events)
        , causal_edge_pool_(cfg.max_causal_edges)
        , branchial_edge_pool_(cfg.max_branchial_edges)
        , edge_consumers_(cfg.max_edges, cfg.edge_consumer_nodes)
        , branchial_index_(cfg.branchial_index_buckets, cfg.branchial_index_nodes)
        , causal_triple_dedup_(cfg.causal_triple_slots)
        , causal_pair_dedup_(cfg.causal_pair_slots)
        , branchial_pair_dedup_(cfg.branchial_pair_slots)
        , preds_list_(cfg.max_events, cfg.tr_preds_nodes)
    {
        // Every kernel that runs against an EngineState needs more per-thread stack than the
        // 1 KB default: match_state_rule's DFS recurses to the LHS edge count, apply_one_match
        // holds several kMaxPatternEdges arrays, and a scheduler that calls both from one
        // kernel carries the sum. Raising it here rather than in one scheduler's constructor
        // is what makes it hold for every entry point -- a scheduler that missed it would fail
        // as a stack overflow reported as an illegal memory access.
        // Checked, and then READ BACK. A driver may clamp the request rather than refuse it, so a
        // successful return does not mean the stack is the size that was asked for -- and the
        // failure mode either way is a stack overflow surfacing as an illegal memory access,
        // which reads like a pointer bug and is diagnosed as one.
        // ASK FOR LESS RATHER THAN FAIL. The driver reserves this per-thread size across every
        // thread the device can hold resident -- not across the launch grid -- so a deep run's
        // request can exceed device memory outright, and cudaDeviceSetLimit then returns
        // out-of-memory. Throwing there turns a run that could have proceeded shallower into no
        // run at all, which is the opposite of this class's contract: past the depth it can
        // support it RECORDS and carries on, and qe_max_recursion_depth_ below is already derived
        // from what the driver ACTUALLY gave rather than from what was asked.
        //
        // Measured cause: an 80-step configuration asks for the 256 KB per-thread cap, and the
        // memory budget (which counts this since abe5f732) then exceeds the cap by more than the
        // pools can give back -- fit_config_to_cap scales pools only, and the stack is not a pool.
        // Halving the request until it is accepted reaches the largest stack the device will
        // actually grant.
        size_t want_stack = stack_bytes_for_depth(cfg.reconstruction_max_depth);
        cudaError_t st_rc = cudaDeviceSetLimit(cudaLimitStackSize, want_stack);
        while (st_rc != cudaSuccess && want_stack > kDeviceStackFloorBytes) {
            cudaGetLastError();                      // clear the sticky error before retrying
            want_stack = want_stack / 2 > kDeviceStackFloorBytes ? want_stack / 2
                                                                 : kDeviceStackFloorBytes;
            st_rc = cudaDeviceSetLimit(cudaLimitStackSize, want_stack);
        }
        HG_CUDA_CHECK(st_rc, "set device stack size");
        size_t actual_stack = 0;
        HG_CUDA_CHECK(cudaDeviceGetLimit(&actual_stack, cudaLimitStackSize), "read device stack size");
        // The depth the replay may reach is derived from what the driver ACTUALLY gave, not from
        // what was asked for, because a driver may clamp a request rather than refuse it.
        qe_max_recursion_depth_ =
            actual_stack > kDeviceStackFloorBytes
                ? static_cast<uint32_t>((actual_stack - kDeviceStackFloorBytes) /
                                        kDeviceStackBytesPerDepth)
                : 0u;
        if (actual_stack < kDeviceStackFloorBytes) {
            throw std::runtime_error(
                "EngineState: device stack is " + std::to_string(actual_stack) +
                " bytes after requesting " + std::to_string(want_stack) +
                "; match_state_rule's DFS would overflow it and report an illegal memory access");
        }
        slice_scan_max_edges_ = cfg.slice_scan_max_edges;
        HG_CUDA_CHECK(cudaMalloc(&state_edge_slices_,
              sizeof(StateEdgeSlice) * cfg_.max_states),
              "EngineState state_edge_slices alloc");
        HG_CUDA_CHECK(cudaMalloc(&state_edge_ids_,
              sizeof(EdgeId) * cfg_.max_state_edge_total),
              "EngineState state_edge_ids alloc");
        // ONE ALLOCATION FOR EVERY SCALAR COUNTER THE HOST READS BACK.
        //
        // Each of these is four bytes, and read on its own each costs a `cudaMemcpy` API call.
        // The transfer is instant; the CALL is not. Measured over one steady-state evolution of
        // `multirule`: 42 cudaMemcpy calls totalling 1.884 ms against a 4.74 ms window -- 39.8%
        // of the call -- at a median of 23.5 us each, 27 of them moving eight bytes or fewer.
        // Reading four bytes six times costs six times 23.5 us; reading twenty-four bytes once
        // costs 23.5 us. Contiguity is what makes the second possible, so these six live in one
        // block and the individual pointers are offsets into it.
        //
        // Device code is unaffected: it still writes through the same typed pointers.
        HG_CUDA_CHECK(cudaMalloc(&counter_block_, sizeof(uint32_t) * kCounterSlots),
              "EngineState counter block alloc");
        // counters_snapshot_host() transfers all six slots at once, while slots 4 and 5 are bound
        // to their pointers on first use of the feature that owns them. Zeroing the whole block
        // here is what makes a slot read as zero when its feature never runs.
        HG_CUDA_CHECK(cudaMemset(counter_block_, 0, sizeof(uint32_t) * kCounterSlots),
              "EngineState counter block init");
        state_edge_ids_counter_ = counter_block_ + 0;
        state_count_            = counter_block_ + 1;
        HG_CUDA_CHECK(cudaMalloc(&state_canonical_hash_, sizeof(uint64_t) * cfg_.max_states),
              "EngineState state_canonical_hash alloc");
        HG_CUDA_CHECK(cudaMalloc(&state_exact_hash_, sizeof(uint64_t) * cfg_.max_states),
              "EngineState state_exact_hash alloc");
        needs_indices_          = counter_block_ + 2;
        vertex_high_water_      = counter_block_ + 3;
        HG_CUDA_CHECK(cudaMalloc(&edge_producer_,     sizeof(EventId) * cfg_.max_edges),
              "EngineState edge_producer alloc");
        clear();
    }

    ~EngineState() {
        if (state_edge_slices_)      cudaFree(state_edge_slices_);
        if (state_edge_ids_)         cudaFree(state_edge_ids_);
        // The six scalar counters are slices of counter_block_, not separate allocations, so
        // the block is freed once here and none of them is freed individually.
        if (counter_block_)          cudaFree(counter_block_);
        if (state_canonical_hash_)   cudaFree(state_canonical_hash_);
        if (state_exact_hash_)       cudaFree(state_exact_hash_);
        if (state_edge_rank_)        cudaFree(state_edge_rank_);
        if (state_edge_orbit_)       cudaFree(state_edge_orbit_);
        if (state_num_orbits_)       cudaFree(state_num_orbits_);
        if (edge_producer_)          cudaFree(edge_producer_);
    }

    EngineState(const EngineState&)            = delete;
    EngineState& operator=(const EngineState&) = delete;

    // Take the per-edge rank array, which only a run keying events on consumed or produced
    // edges reads. Four bytes per edge SLOT, so on a large configuration this is the biggest
    // allocation the engine makes and taking it up front would charge every run for a mode
    // most do not select. Idempotent; call before launching, never from a kernel.
    void ensure_edge_ranks() {
        if (state_edge_rank_) return;
        HG_CUDA_CHECK(cudaMalloc(&state_edge_rank_, sizeof(uint32_t) * cfg_.max_state_edge_total),
              "EngineState state_edge_rank alloc");
        event_sig_fallbacks_ = counter_block_ + 4;
        HG_CUDA_CHECK(cudaMemset(state_edge_rank_, 0xFF,
              sizeof(uint32_t) * cfg_.max_state_edge_total),
              "EngineState init state_edge_rank");
        HG_CUDA_CHECK(cudaMemset(event_sig_fallbacks_, 0, sizeof(uint32_t)),
              "EngineState init event_sig_raw_fallbacks");
    }

    // Take the per-slot edge orbit array and the per-state orbit counts, which only a
    // quotient-causal run reads (its DP keys on orbits). Idempotent; call before launching.
    void ensure_edge_orbits() {
        if (state_edge_orbit_) return;
        HG_CUDA_CHECK(cudaMalloc(&state_edge_orbit_, sizeof(uint32_t) * cfg_.max_state_edge_total),
              "EngineState state_edge_orbit alloc");
        HG_CUDA_CHECK(cudaMalloc(&state_num_orbits_, sizeof(uint32_t) * cfg_.max_states),
              "EngineState state_num_orbits alloc");
        HG_CUDA_CHECK(cudaMemset(state_edge_orbit_, 0xFF,
              sizeof(uint32_t) * cfg_.max_state_edge_total),
              "EngineState init state_edge_orbit");
        HG_CUDA_CHECK(cudaMemset(state_num_orbits_, 0, sizeof(uint32_t) * cfg_.max_states),
              "EngineState init state_num_orbits");
    }

    // Take the canonical-event counter. Called once the event mode is known, alongside the
    // signature map the scheduler carries; under EventSignatureKeys None no signature is
    // computed and every application is its own event, so nothing counts.
    void ensure_event_identity() {
        if (canonical_event_count_) return;
        canonical_event_count_ = counter_block_ + 5;
        HG_CUDA_CHECK(cudaMemset(canonical_event_count_, 0, sizeof(uint32_t)),
              "EngineState init canonical_event_count");
    }

    // Events that won their signature slot. Under an identity mode this is what "how many
    // events" means; 0 when no mode is selected, where the raw count is the answer.
    uint32_t canonical_event_count() const {
        if (!canonical_event_count_) return 0;
        uint32_t n = 0;
        HG_CUDA_CHECK(cudaMemcpy(&n, canonical_event_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "EngineState read canonical_event_count");
        return n;
    }

    // The IR scratch arena, owned by the ENGINE rather than by a run.
    //
    // It is the largest single allocation the engine makes: sized from the grid times the state
    // budget, which is 134 MB at the default configuration and measured to be genuinely needed
    // (a wide workload consumed 87% of it). Constructing it per run charged that cudaMalloc to
    // every call, which is the whole of the persistent scheduler's loss on short and interactive
    // runs -- and interactive use reuses one Engine across many calls, so it was being paid over
    // and over for a buffer whose contents never outlive a run.
    //
    // Grows but never shrinks: a later run wanting more gets a bigger one, a later run wanting
    // less reuses what is there. reset() is a single cursor store, so reuse costs nothing.
    DeviceArena& ir_arena(uint64_t needed_words) {
        if (!ir_arena_ || ir_arena_->capacity_words() < needed_words) {
            ir_arena_ = std::make_unique<DeviceArena>(needed_words);
        }
        ir_arena_->reset();
        return *ir_arena_;
    }

    // Consumed or produced edges stamped with a raw edge id because no rank was available.
    // Nonzero means some event signature is not an isomorphism invariant.
    uint32_t event_sig_raw_fallbacks() const {
        if (!event_sig_fallbacks_) return 0;
        uint32_t n = 0;
        HG_CUDA_CHECK(cudaMemcpy(&n, event_sig_fallbacks_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "EngineState read event_sig_raw_fallbacks");
        return n;
    }

    DeviceState device() const {
        DeviceState d;
        d.vertex_pool             = vertex_pool_.view();
        d.edge_pool               = edge_pool_.view();
        d.state_edge_slices       = state_edge_slices_;
        d.state_edge_ids          = state_edge_ids_;
        d.state_edge_ids_counter  = state_edge_ids_counter_;
        d.state_edge_ids_capacity = cfg_.max_state_edge_total;
        d.max_states              = cfg_.max_states;
        d.ir_generators           = cfg_.ir_generators;
        d.ir_depth                = cfg_.ir_depth;
        d.state_count             = state_count_;
        d.state_canonical_hash    = state_canonical_hash_;
        d.state_exact_hash        = state_exact_hash_;
        d.state_edge_rank         = state_edge_rank_;
        d.state_edge_orbit        = state_edge_orbit_;
        d.state_num_orbits        = state_num_orbits_;
        d.event_sig_raw_fallbacks = event_sig_fallbacks_;
        d.canonical_event_count   = canonical_event_count_;
        d.vertex_high_water       = vertex_high_water_;
        d.signature_index         = signature_index_.view();
        d.vertex_inverted_index   = vertex_inverted_index_.view();
        d.event_pool              = event_pool_.view();
        d.causal_edge_pool        = causal_edge_pool_.view();
        d.branchial_edge_pool     = branchial_edge_pool_.view();
        d.edge_producer           = edge_producer_;
        d.edge_consumers          = edge_consumers_.view();
        d.branchial_index         = branchial_index_.view();
        d.causal_triple_dedup     = causal_triple_dedup_.view();
        d.causal_pair_dedup       = causal_pair_dedup_.view();
        d.branchial_pair_dedup    = branchial_pair_dedup_.view();
        d.preds_list              = preds_list_.view();
        d.tr_enabled              = tr_enabled_;
        d.quotient_causal         = quotient_causal_;
        d.slice_scan_max_edges    = slice_scan_max_edges_;
        d.maintain_indices        = maintain_indices_ ? 1u : 0u;
        d.record_causal           = record_.causal ? 1u : 0u;
        d.record_branchial        = record_.branchial ? 1u : 0u;
        d.needs_indices           = needs_indices_;
        d.errors                  = errors_.view();
        return d;
    }

    // Error channel — sync + drain into the caller's warnings list. Call
    // after every kernel launch that writes to DeviceState. Non-throwing:
    // capacity overflows are warnings, not errors. (Genuine driver
    // failures inside the d2h still throw std::runtime_error.)
    void collect_warnings_into(std::vector<OverflowWarning>& out,
                               const char* context) {
        errors_.collect_warnings_into(out, context);
    }

    // Legacy fail-fast variant for unit tests. Production code should use
    // collect_warnings_into instead.
    void throw_on_errors(const char* context) const {
        errors_.throw_if_any(context);
    }
    void clear_errors() { errors_.clear(); }

    // Which artifacts this run records; read into DeviceState by device().
    void set_record_set(hgcommon::RecordSet r) { record_ = r; }
    hgcommon::RecordSet record_set() const { return record_; }

    void set_tr_enabled(bool enabled) { tr_enabled_ = enabled; }
    void set_quotient_causal(bool enabled) { quotient_causal_ = enabled; }
    bool quotient_causal() const { return quotient_causal_; }

    uint32_t config_slice_scan_max_edges() const { return slice_scan_max_edges_; }
    void set_maintain_indices(bool on) { maintain_indices_ = on; }
    bool maintain_indices() const { return maintain_indices_; }
    bool needs_indices_host() const {
        uint32_t v = 0;
        HG_CUDA_CHECK(cudaMemcpy(&v, needs_indices_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "EngineState needs_indices read");
        return v != 0;
    }

    void clear() {
        // CLEAR WHAT THE LAST RUN DIRTIED, NOT WHAT THE CONFIG RESERVED.
        //
        // The per-edge-slot arrays are sized from the workload ESTIMATE -- config_from_input
        // reserves max_state_edge_total slots -- while a run writes only as many as it produced.
        // Clearing the reservation made every call pay for the estimate: nsys on a depth-3 run
        // producing THIRTEEN states measured 9.8 GB of cudaMemset across 981 operations, the
        // largest single one 538 MB, which is exactly 4 bytes x max_state_edge_total. That is
        // the fixed floor a small run cannot get under, and it is why sizing the pools generously
        // to avoid grow-and-retry made a depth-7 run slower rather than faster.
        //
        // At this point state_edge_ids_counter_ still holds the PREVIOUS run's final value, so it
        // names exactly the prefix that can be dirty. Slots above it were never written and still
        // carry the fill from construction, which zeroes the whole reservation once.
        // The same argument for the head arrays of the lists whose key is a dense id. Each
        // counter still holds the previous run's value here, so it names the prefix that can be
        // dirty; heads above it were never written and still carry the fill from construction.
        // Read them all before anything below resets them.
        const uint32_t dirty_vertices = vertex_high_water_host();
        const uint32_t dirty_edges_lf = edge_pool_.size_host();
        const uint32_t dirty_events   = event_pool_.size_host();

        uint32_t dirty_edge_slots = cfg_.max_state_edge_total;
        if (state_edge_ids_counter_) {
            uint32_t n = 0;
            if (cudaMemcpy(&n, state_edge_ids_counter_, sizeof(uint32_t),
                           cudaMemcpyDeviceToHost) == cudaSuccess && n <= cfg_.max_state_edge_total)
                dirty_edge_slots = n;
        }

        HG_CUDA_CHECK(cudaMemset(state_edge_slices_, 0,
              sizeof(StateEdgeSlice) * cfg_.max_states),
              "EngineState clear state_edge_slices");
        HG_CUDA_CHECK(cudaMemset(state_edge_ids_counter_, 0, sizeof(uint32_t)),
              "EngineState clear state_edge_ids_counter");
        HG_CUDA_CHECK(cudaMemset(state_count_,       0, sizeof(uint32_t)), "EngineState clear state_count");
        // 0 means "not yet computed", which is why the empty state has its own reserved hash
        // rather than 0 -- see EMPTY_STATE_CANONICAL_HASH.
        HG_CUDA_CHECK(cudaMemset(state_canonical_hash_, 0, sizeof(uint64_t) * cfg_.max_states),
              "EngineState clear state_canonical_hash");
        HG_CUDA_CHECK(cudaMemset(state_exact_hash_, 0, sizeof(uint64_t) * cfg_.max_states),
              "EngineState clear state_exact_hash");
        if (state_edge_rank_) {
            // UINT32_MAX, not 0: 0 is a valid rank (the canonically first edge), so a zeroed
            // array would read as "every edge ranks first" instead of "no ranks yet".
            HG_CUDA_CHECK(cudaMemset(state_edge_rank_, 0xFF,
                  sizeof(uint32_t) * dirty_edge_slots),
                  "EngineState clear state_edge_rank");
        }
        if (state_edge_orbit_) {
            HG_CUDA_CHECK(cudaMemset(state_edge_orbit_, 0xFF,
                  sizeof(uint32_t) * dirty_edge_slots),
                  "EngineState clear state_edge_orbit");
            HG_CUDA_CHECK(cudaMemset(state_num_orbits_, 0, sizeof(uint32_t) * cfg_.max_states),
                  "EngineState clear state_num_orbits");
        }
        if (event_sig_fallbacks_) {
            HG_CUDA_CHECK(cudaMemset(event_sig_fallbacks_, 0, sizeof(uint32_t)),
                  "EngineState clear event_sig_raw_fallbacks");
        }
        if (canonical_event_count_) {
            HG_CUDA_CHECK(cudaMemset(canonical_event_count_, 0, sizeof(uint32_t)),
                  "EngineState clear canonical_event_count");
        }
        HG_CUDA_CHECK(cudaMemset(needs_indices_,     0, sizeof(uint32_t)), "EngineState clear needs_indices");
        HG_CUDA_CHECK(cudaMemset(vertex_high_water_, 0, sizeof(uint32_t)), "EngineState clear vertex_high_water");
        // edge_producer init to INVALID_ID (0xFF bytes).
        HG_CUDA_CHECK(cudaMemset(edge_producer_, 0xFF, sizeof(EventId) * cfg_.max_edges),
              "EngineState clear edge_producer");
        vertex_pool_.reset();
        edge_pool_.reset();
        signature_index_.clear();
        vertex_inverted_index_.clear(dirty_vertices);
        event_pool_.reset();
        causal_edge_pool_.reset();
        branchial_edge_pool_.reset();
        edge_consumers_.clear(dirty_edges_lf);
        branchial_index_.clear();
        causal_triple_dedup_.clear();
        causal_pair_dedup_.clear();
        branchial_pair_dedup_.clear();
        preds_list_.clear(dirty_events);
        errors_.clear();
    }

    const EngineConfig& config() const { return cfg_; }

    // ------------------------------------------------------------------
    // Host-side inspection helpers (slow; for tests / final readout only)
    // ------------------------------------------------------------------

    uint32_t num_edges_host() const  { return edge_pool_.size_host(); }
    uint32_t num_states_host() const {
        uint32_t v = 0;
        cudaMemcpy(&v, state_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return v;
    }
    uint32_t vertex_high_water_host() const {
        uint32_t v = 0;
        cudaMemcpy(&v, vertex_high_water_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return v;
    }

    Edge edge_at_host(EdgeId eid) const {
        Edge e{};
        cudaMemcpy(&e, edge_pool_view_data() + eid, sizeof(Edge), cudaMemcpyDeviceToHost);
        return e;
    }

    std::vector<VertexId> edge_vertices_host(EdgeId eid) const {
        Edge e = edge_at_host(eid);
        std::vector<VertexId> out(e.arity);
        cudaMemcpy(out.data(), vertex_pool_view_data() + e.vertex_offset,
                   sizeof(VertexId) * e.arity, cudaMemcpyDeviceToHost);
        return out;
    }

    // Read back every state's edge-vertex-tuple list from the device via
    // four bulk cudaMemcpy calls (slices, ids, edges, vertices) then
    // reconstructs on host. O(total state-edge slots) on the wire rather
    // than the O(max_states × max_edges/32) bitset readback.
    std::vector<std::vector<std::vector<VertexId>>> all_state_edges_host() const {
        uint32_t n_states = num_states_host();
        std::vector<std::vector<std::vector<VertexId>>> out(n_states);
        if (n_states == 0) return out;

        uint32_t n_edges      = edge_pool_.size_host();
        uint32_t n_vert_slots = vertex_pool_.size_host();
        uint32_t n_id_slots   = 0;
        cudaMemcpy(&n_id_slots, state_edge_ids_counter_, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        std::vector<Edge>           edges(n_edges);
        std::vector<VertexId>       verts(n_vert_slots);
        std::vector<StateEdgeSlice> slices(n_states);
        std::vector<EdgeId>         ids(n_id_slots);

        if (n_edges > 0) {
            cudaMemcpy(edges.data(), edge_pool_.view().data,
                       sizeof(Edge) * n_edges, cudaMemcpyDeviceToHost);
        }
        if (n_vert_slots > 0) {
            cudaMemcpy(verts.data(), vertex_pool_.view().data,
                       sizeof(VertexId) * n_vert_slots, cudaMemcpyDeviceToHost);
        }
        cudaMemcpy(slices.data(), state_edge_slices_,
                   sizeof(StateEdgeSlice) * n_states, cudaMemcpyDeviceToHost);
        if (n_id_slots > 0) {
            cudaMemcpy(ids.data(), state_edge_ids_,
                       sizeof(EdgeId) * n_id_slots, cudaMemcpyDeviceToHost);
        }

        for (uint32_t s = 0; s < n_states; ++s) {
            const StateEdgeSlice& sl = slices[s];
            if (static_cast<size_t>(sl.offset) + sl.count > ids.size()) continue;
            for (uint32_t k = 0; k < sl.count; ++k) {
                EdgeId eid = ids[sl.offset + k];
                if (eid >= n_edges) continue;
                const Edge& e = edges[eid];
                std::vector<VertexId> vs(e.arity);
                for (uint8_t i = 0; i < e.arity; ++i) {
                    vs[i] = verts[e.vertex_offset + i];
                }
                out[s].push_back(std::move(vs));
            }
        }
        return out;
    }

    // Read back one state's EdgeId list.
    std::vector<EdgeId> state_edges_host(StateId sid) const {
        StateEdgeSlice sl{0, 0};
        cudaMemcpy(&sl, state_edge_slices_ + sid, sizeof(StateEdgeSlice),
                   cudaMemcpyDeviceToHost);
        std::vector<EdgeId> out(sl.count);
        if (sl.count > 0) {
            cudaMemcpy(out.data(), state_edge_ids_ + sl.offset,
                       sizeof(EdgeId) * sl.count, cudaMemcpyDeviceToHost);
        }
        return out;
    }

    // Friend access for kernels that need raw pointers (rare; prefer DeviceView).
    Edge*     edge_pool_view_data()    const { return edge_pool_.view().data; }
    VertexId* vertex_pool_view_data()  const { return vertex_pool_.view().data; }

private:

    EngineConfig                       cfg_;
    Pool<VertexId>                     vertex_pool_;
    Pool<Edge>                         edge_pool_;
    StateEdgeSlice*                    state_edge_slices_      = nullptr;
    EdgeId*                            state_edge_ids_         = nullptr;
    // The six host-read scalar counters, contiguous so one transfer fetches them all. The
    // pointers below are offsets into this and are NOT separately freed.
    static constexpr uint32_t          kCounterSlots = 6;
    uint32_t*                          counter_block_          = nullptr;
public:
    // Every host-read scalar counter, in ONE transfer.
    //
    // Read individually these cost one cudaMemcpy API call each, and the call dominates: a
    // four-byte transfer is instant while the call runs about 23.5 us. A caller that needs more
    // than one of these should take a snapshot rather than several accessors.
    struct CounterSnapshot {
        uint32_t state_edge_ids = 0;   // slot 0
        uint32_t states         = 0;   // slot 1
        uint32_t needs_indices  = 0;   // slot 2
        uint32_t vertex_high    = 0;   // slot 3
        uint32_t sig_fallbacks  = 0;   // slot 4
        uint32_t canonical_ev   = 0;   // slot 5
    };
    CounterSnapshot counters_snapshot_host() const {
        uint32_t raw[kCounterSlots] = {};
        HG_CUDA_CHECK(cudaMemcpy(raw, counter_block_, sizeof(raw), cudaMemcpyDeviceToHost),
              "EngineState counter block d2h");
        CounterSnapshot c;
        c.state_edge_ids = raw[0]; c.states        = raw[1]; c.needs_indices = raw[2];
        c.vertex_high    = raw[3]; c.sig_fallbacks = raw[4]; c.canonical_ev  = raw[5];
        return c;
    }
private:
    uint32_t*                          state_edge_ids_counter_ = nullptr;
    uint32_t*                          state_count_            = nullptr;
    uint64_t*                          state_canonical_hash_   = nullptr;
    uint64_t*                          state_exact_hash_       = nullptr;
    hgcommon::RecordSet                record_{};
    uint32_t*                          state_edge_rank_        = nullptr;
    uint32_t*                          state_edge_orbit_       = nullptr;
    uint32_t*                          state_num_orbits_       = nullptr;
    uint32_t*                          event_sig_fallbacks_    = nullptr;
    uint32_t*                          canonical_event_count_  = nullptr;
    // Owned by the engine, not by a run. See ir_arena().
    std::unique_ptr<DeviceArena>       ir_arena_;
    uint32_t*                          vertex_high_water_      = nullptr;
    SignatureIndex                     signature_index_;
    VertexInvertedIndex                vertex_inverted_index_;
    Pool<DeviceEvent>                  event_pool_;
    Pool<DeviceCausalEdge>             causal_edge_pool_;
    Pool<DeviceBranchialEdge>          branchial_edge_pool_;
    EventId*                           edge_producer_ = nullptr;
    LockFreeList<EventId>              edge_consumers_;
    LockFreeList<uint64_t>             branchial_index_;
    ConcurrentMap<uint64_t, uint32_t>  causal_triple_dedup_;
    ConcurrentMap<uint64_t, uint32_t>  causal_pair_dedup_;
    ConcurrentMap<uint64_t, uint32_t>  branchial_pair_dedup_;
    LockFreeList<EventId>              preds_list_;
    DeviceErrors                       errors_;
    bool                               tr_enabled_ = false;
    bool                               quotient_causal_ = false;
    uint32_t                           qe_max_recursion_depth_ = 0;
    uint32_t slice_scan_max_edges_ = 256;
    bool maintain_indices_ = true;
    uint32_t* needs_indices_ = nullptr;

public:
    // Host readers for tests / EvolveResult population.
    uint32_t num_events_host()          const { return event_pool_.size_host(); }
    uint32_t num_causal_edges_host()    const { return causal_edge_pool_.size_host(); }
    uint32_t num_branchial_edges_host() const { return branchial_edge_pool_.size_host(); }

    std::vector<DeviceEvent> events_host() const {
        uint32_t n = num_events_host();
        std::vector<DeviceEvent> out(n);
        if (n > 0) cudaMemcpy(out.data(), event_pool_.view().data,
                              sizeof(DeviceEvent) * n, cudaMemcpyDeviceToHost);
        return out;
    }
    std::vector<DeviceCausalEdge> causal_edges_host() const {
        uint32_t n = num_causal_edges_host();
        std::vector<DeviceCausalEdge> out(n);
        if (n > 0) cudaMemcpy(out.data(), causal_edge_pool_.view().data,
                              sizeof(DeviceCausalEdge) * n, cudaMemcpyDeviceToHost);
        return out;
    }
    std::vector<DeviceBranchialEdge> branchial_edges_host() const {
        uint32_t n = num_branchial_edges_host();
        std::vector<DeviceBranchialEdge> out(n);
        if (n > 0) cudaMemcpy(out.data(), branchial_edge_pool_.view().data,
                              sizeof(DeviceBranchialEdge) * n, cudaMemcpyDeviceToHost);
        return out;
    }
};

}  // namespace gpu
}  // namespace HG_NAMESPACE