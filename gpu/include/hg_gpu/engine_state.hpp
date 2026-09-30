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
#include <type_traits>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// Defined in match.hpp, which includes this header; the launch scratch holds only a pointer.
struct DeviceRule;

// EngineConfig is declared in evolve.hpp so host-only translation units
// (the bench harness, the differential test driver) can include it without
// pulling in cuda_runtime.h. The full definition lives there; this header
// transitively re-exports it via #include "hg_gpu/evolve.hpp".

// Keyed rewrites on the device (keyed.hpp; the rule is hgcommon/token_core.hpp). A Full-mode
// persistent run sets `enabled` and the two maps (its own, or its session's) on the DeviceState it
// launches with; every other launch has it clear.
struct KeyedView {
    uint64_t* state_token_sum      = nullptr;   // [max_states], 0 until computed
    uint32_t* state_first_new_edge = nullptr;   // [max_states], INVALID_ID for a root
    // Children waiting on a twin whose results are not yet published (keyed_take_twin), a stack
    // per twin: follow_head [max_states] is the top event, FOLLOW_EMPTY or FOLLOW_CLOSED;
    // follow_next [max_events] the event below it.
    uint32_t* follow_head = nullptr;
    uint32_t* follow_next = nullptr;
    // Rewrite key -> the first event that applied the rewrite; its id + 1 is the rewrite id.
    ConcurrentMap<uint64_t, uint32_t>::DeviceView rewrites{};
    // Token sum -> the first state with that token set.
    ConcurrentMap<uint64_t, uint32_t>::DeviceView twins{};
    // [0] KEYED_ARMED / KEYED_INTERNING / KEYED_OFF, [1] claims made, [2] a twin was taken,
    // [3] twins taken.
    uint32_t* words       = nullptr;
    uint32_t  claim_limit = 1024;
    // Test lever (EngineConfig::keyed_sum_mask): the twin claim key is the token sum ANDed with
    // this, so a narrow mask makes states with different token sets share keys.
    uint64_t  sum_mask    = ~uint64_t{0};
    uint32_t  enabled     = 0;
};

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

    // SAMPLING AND CAPPING. The decisions live in hgcommon/sampling_core.hpp. match_state_rule
    // calls the draw, the spine and the per-(state, rule) cap when it emits a match;
    // state_retained calls the two bounds. A rate of 1.0 with no weights and no cap takes
    // the unsampled path.
    double        transition_rate;      // 1.0 = every transition is taken
    const double* rule_weights;         // null = every rule weighted 1
    uint32_t      num_rule_weights;
    uint64_t      sampling_seed;
    // Hard bounds; 0 is unlimited. The counters are device atomics, cleared per run.
    uint32_t  max_states_per_step;
    uint32_t* states_per_step;                  // [max_steps + 2]
    uint32_t  max_states_per_step_slots;
    uint32_t  max_successor_states_per_parent;
    uint32_t* successors_per_parent;            // [max_states]
    // Kept per (state, rule), chosen by rank at the point one block has found every match for
    // that pair -- the device's drain. 0 is unlimited.
    uint32_t  matches_per_state_rule;
    // Rules in the run. Under sampling one block matches every rule of a state, because the
    // spine chooses among all of the state's transitions.
    uint32_t  num_rules;

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
    uint32_t max_edge_arity;   // EngineConfig::max_edge_arity, kMaxArity when unknown
    // EngineConfig::canonical_key_mask: ANDed into a Full-mode canonical hash before it is the
    // first dedup probe key. All ones except in tests.
    uint64_t canonical_key_mask;
    // EngineConfig::event_key_mask: the same for an event signature, a replay class's run
    // signature and the exact hash None/Automatic event identity reads.
    uint64_t event_key_mask;
    // EngineConfig::replay_id_limit: the replay refuses raw event ids at or past it.
    uint32_t replay_id_limit;
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
    // The reduction's reachability search runs in local arrays and, when they fill, again in the
    // calling block's slice of this buffer: tr_scratch_slots slices of tr_scratch_stack stack
    // words followed by tr_scratch_visited visited-table words (a power of two).
    uint32_t* tr_scratch;
    uint32_t  tr_scratch_stack;
    uint32_t  tr_scratch_visited;
    uint32_t  tr_scratch_slots;
    // Each block's slice of survivor_scratch_cap entries for survivor lists longer than
    // kLocalSurvivors (survivor_buffer below); null when EngineConfig::survivor_scratch is 0.
    uint64_t* survivor_scratch;
    uint32_t  survivor_scratch_cap;
    uint32_t  survivor_scratch_slots;

    // Flags
    bool tr_enabled;
    // Quotient route: when set, apply_one_match SKIPS the raw-edge producer/consumer
    // rendezvous; the replay reconstructs the relations (quotient_expansion.hpp), which is
    // schedule-independent where the raw rendezvous under quotient is not. Branchial
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

    KeyedView keyed;

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

// A survivor list's storage, for a list of at most `need` entries: `local` (kLocalSurvivors
// entries) when it fits, else the calling block's slice of ds.survivor_scratch. Null when neither
// holds it; the caller then records its survivors-overflow kind, which grow-and-retry grows.
// Called by a persistent block's warp, whose block index is `slice`; `local` is then block-shared.
constexpr uint32_t kLocalSurvivors = 256;
__device__ __forceinline__ uint64_t* survivor_buffer(const DeviceState& ds, uint64_t* local,
                                                     uint32_t need, uint32_t slice) {
    if (need <= kLocalSurvivors) return local;
    if (ds.survivor_scratch == nullptr || slice >= ds.survivor_scratch_slots ||
        need > ds.survivor_scratch_cap)
        return nullptr;
    return ds.survivor_scratch + static_cast<size_t>(slice) * ds.survivor_scratch_cap;
}

class EngineState {
public:
    // Per-thread device stack. See the constructor for why the default is not enough.
    //
    // The floor covers the kernels whose stack need is fixed: match_state_rule's DFS recurses
    // to the LHS edge count and apply_one_match holds several kMaxPatternEdges arrays, both
    // bounded by kMaxPatternEdges.
    static constexpr size_t kDeviceStackFloorBytes = 32u * 1024u;

    // The replay carries reconstruction depth in a worklist in device memory, so the request is
    // this constant however deep the run is.
    static constexpr size_t kDeviceStackBytes = kDeviceStackFloorBytes;


    // Per-block global scratch of the reachability search at tr_scratch_scale 1: 8 times the
    // search's local arrays (rewrite.cu, kReachStack and kReachVisited).
    static constexpr uint32_t kTrScratchStack   = 2048;
    static constexpr uint32_t kTrScratchVisited = 4096;   // a power of two

    explicit EngineState(EngineConfig cfg);

    ~EngineState();

    EngineState(const EngineState&)            = delete;
    EngineState& operator=(const EngineState&) = delete;

    // Take the per-edge rank array, which only a run keying events on consumed or produced
    // edges reads. Four bytes per edge SLOT, so on a large configuration this is the biggest
    // allocation the engine makes and taking it up front would charge every run for a mode
    // most do not select. Idempotent; call before launching, never from a kernel.
    void ensure_edge_ranks();

    // Take the per-slot edge orbit array and the per-state orbit counts, which only a
    // quotient-causal run reads (its DP keys on orbits). Idempotent; call before launching.
    void ensure_edge_orbits();

    // Take the per-state keyed-rewrite arrays (KeyedView: token sum, first produced edge), which
    // only a keyed run reads. They need no clearing: a state's entries are written when it is
    // created. Idempotent; call before launching.
    void ensure_keyed();

    // Take the canonical-event counter. Called once the event mode is known, alongside the
    // signature map the scheduler carries; under EventSignatureKeys None no signature is
    // computed and every application is its own event, so nothing counts.
    void ensure_event_identity();

    // Events that won their signature slot. Under an identity mode this is what "how many
    // events" means; 0 when no mode is selected, where the raw count is the answer.
    uint32_t canonical_event_count() const;

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
    DeviceArena& ir_arena(uint64_t needed_words);

    // Consumed or produced edges stamped with a raw edge id because no rank was available.
    // Nonzero means some event signature is not an isomorphism invariant.
    uint32_t event_sig_raw_fallbacks() const;

    // SET THE SAMPLING AND CAPPING PARAMETERS FOR THIS RUN, allocating only what is asked for.
    // A run that samples nothing and caps nothing allocates nothing here and the device takes
    // the fast path in both places.
    //
    // The two counters are cleared on EVERY call, not on the first: they accumulate across a
    // run and a session's second Step must not inherit the first's tallies. Clearing costs one
    // memset of (steps + 2) and one of max_states, both of which the run is about to write.
    void set_sampling(double transition_rate, const double* weights, uint32_t num_weights,
                      uint64_t seed, uint32_t max_states_per_step,
                      uint32_t max_successor_states_per_parent, uint32_t matches_per_state_rule,
                      uint32_t num_steps, uint32_t num_rules);

    DeviceState device() const;

    // Error channel — sync + drain into the caller's warnings list. Call
    // after every kernel launch that writes to DeviceState. Non-throwing:
    // capacity overflows are warnings, not errors. (Genuine driver
    // failures inside the d2h still throw std::runtime_error.)
    void collect_warnings_into(std::vector<OverflowWarning>& out,
                               const char* context);

    // The device counts event signatures it had to build from a raw edge id because the edge's
    // canonical rank was UINT32_MAX when the event was stamped. It is not recorded through the
    // error channel -- nothing failed and nothing was dropped -- so it is reported from here,
    // beside the errors, rather than at each call site that would otherwise have to remember it.
    // A signature built that way is not an isomorphism invariant, so the event total it produces
    // is otherwise indistinguishable from a disagreement in the evolution.
    void report_event_sig_fallbacks(std::vector<OverflowWarning>& out, const char* context) const;
    // The same, for a caller that already holds the count (CounterSnapshot::sig_fallbacks).
    static void report_event_sig_fallbacks(std::vector<OverflowWarning>& out, const char* context,
                                           uint32_t fallbacks);

    // Legacy fail-fast variant for unit tests. Production code should use
    // collect_warnings_into instead.
    void throw_on_errors(const char* context) const;
    void clear_errors();

    // Which artifacts this run records; read into DeviceState by device().
    void set_record_set(hgcommon::RecordSet r);
    hgcommon::RecordSet record_set() const;

    void set_tr_enabled(bool enabled);
    void set_quotient_causal(bool enabled);
    bool quotient_causal() const;

    uint32_t config_slice_scan_max_edges() const;
    void set_maintain_indices(bool on);
    bool maintain_indices() const;
    bool needs_indices_host() const;

    void clear();

    const EngineConfig& config() const;

    // ------------------------------------------------------------------
    // Host-side inspection helpers (slow; for tests / final readout only)
    // ------------------------------------------------------------------

    uint32_t num_edges_host() const;
    uint32_t num_states_host() const;
    uint32_t vertex_high_water_host() const;

    Edge edge_at_host(EdgeId eid) const;

    std::vector<VertexId> edge_vertices_host(EdgeId eid) const;

    // Read back one state's EdgeId list.
    std::vector<EdgeId> state_edges_host(StateId sid) const;

    // Friend access for kernels that need raw pointers (rare; prefer DeviceView).
    Edge*     edge_pool_view_data()    const;
    VertexId* vertex_pool_view_data()  const;

private:

    // Every host-read scalar counter, contiguous so ONE transfer fetches them all. Declared
    // FIRST because the pools below keep their counters in its slots: a pool member's
    // constructor runs in declaration order and needs the block to exist. Slots 0..5 are the
    // engine's own counters; slots 6..10 are the five pools' live counters, so the snapshot
    // read needs no staging of any kind -- no kernel launch, which on this platform was
    // measured COSTLIER than the seven cudaMemcpy calls it replaced (triangle floor,
    // interleaved x7: staged 9.8 ms median against 8.7 unstaged).
    struct CounterBlock {
        uint32_t* p = nullptr;
        explicit CounterBlock(uint32_t slots);
        ~CounterBlock();
        CounterBlock(const CounterBlock&) = delete;
        CounterBlock& operator=(const CounterBlock&) = delete;
    };
    static constexpr uint32_t          kCounterSlots = 11;
    CounterBlock                       block_{kCounterSlots};

    EngineConfig                       cfg_;
    Pool<VertexId>                     vertex_pool_;
    Pool<Edge>                         edge_pool_;
    StateEdgeSlice*                    state_edge_slices_      = nullptr;
    EdgeId*                            state_edge_ids_         = nullptr;
    // Alias of block_.p, kept because the slot pointers below are offsets into it.
    uint32_t*                          counter_block_          = nullptr;
public:
    // GROW-ONLY DEVICE SCRATCH for the launch chain's per-run buffers (rules, seed states,
    // counters, phase cycles). Allocating these per run cost ~34
    // cudaMalloc + 34 cudaFree API calls per call on the per-call floor; sized to the largest
    // run seen, they cost the first run what they always cost and later runs nothing. Mutable
    // through a const accessor because a scratch buffer is not logical engine state.
    struct LaunchScratch {
        DeviceRule*         rules        = nullptr;  uint32_t rules_cap  = 0;
        StateId*            states       = nullptr;  uint32_t states_cap = 0;
        uint32_t*           cursor       = nullptr;  // 2 slots
        unsigned long long* phase_cycles = nullptr;  // 16 slots
    };
    // The persistent launch's match ring, dedup maps and termination detector, kept with the
    // engine and reused by every later launch of the same sizes (each is cleared per launch).
    // Defined in persistent.cu, the only file that builds them.
    struct PersistentScratch;
    struct PersistentScratchFree { void operator()(PersistentScratch* p) const; };
    PersistentScratch& persistent_scratch() const;
private:
    mutable LaunchScratch              launch_scratch_;
    mutable std::unique_ptr<PersistentScratch, PersistentScratchFree> persistent_scratch_;
public:
    // Every host-read scalar counter, in ONE transfer.
    //
    // Read individually these cost one cudaMemcpy API call each, and the call dominates: a
    // four-byte transfer is instant while the call runs about 23.5 us. A caller that needs more
    // than one of these should take a snapshot rather than several accessors. Slots 6..10 are
    // the five pools' LIVE counters -- the pools are constructed onto slots of the block -- so
    // the snapshot is one transfer and nothing else.
    struct CounterSnapshot {
        uint32_t state_edge_ids = 0;   // slot 0
        uint32_t states         = 0;   // slot 1
        uint32_t needs_indices  = 0;   // slot 2
        uint32_t vertex_high    = 0;   // slot 3
        uint32_t sig_fallbacks  = 0;   // slot 4
        uint32_t canonical_ev   = 0;   // slot 5
        uint32_t edges          = 0;   // slot 6, staged from edge_pool_
        uint32_t vertex_slots   = 0;   // slot 7, staged from vertex_pool_
        uint32_t events         = 0;   // slot 8, staged from event_pool_
        uint32_t causal         = 0;   // slot 9, staged from causal_edge_pool_
        uint32_t branchial      = 0;   // slot 10, staged from branchial_edge_pool_
    };
    CounterSnapshot counters_snapshot_host() const;
    // The counter block on the device (kCounterSlots words) and the parse of a host copy of it:
    // counters_snapshot_host is one read followed by snapshot_from.
    const uint32_t* counter_block_device() const { return counter_block_; }
    static CounterSnapshot snapshot_from(const uint32_t* raw);
    static constexpr uint32_t counter_block_words() { return kCounterSlots; }
    DeviceErrors& errors() { return errors_; }
    LaunchScratch& launch_scratch(uint32_t num_rules, uint32_t num_states) const;

    // DEVICE-TO-HOST READS ISSUED TOGETHER. add() and add_raw() record a region; finish() issues
    // every region as a cudaMemcpyAsync into the engine's pinned staging buffer, synchronizes
    // once, and fills each vector with one assign() from the staging copy (add_raw: one memcpy).
    // A synchronous cudaMemcpy costs about 25 us per call here whatever its size, and the batch
    // pays that once. The vectors and raw destinations must outlive finish().
    class ReadbackBatch {
    public:
        explicit ReadbackBatch(const EngineState& engine) : engine_(engine) {}
        template <class T>
        void add(std::vector<T>& dst, const T* src, size_t n) {
            static_assert(std::is_trivially_copyable_v<T>, "readback copies bytes");
            if (!n) { dst.clear(); return; }
            regions_.push_back(Region{&dst, src, sizeof(T) * n, &assign_into<T>});
        }
        void add_raw(void* dst, const void* src, size_t bytes) {
            if (bytes) regions_.push_back(Region{dst, src, bytes, nullptr});
        }
        void finish();
    private:
        template <class T>
        static void assign_into(void* dst, const void* staged, size_t bytes) {
            const T* p = static_cast<const T*>(staged);
            static_cast<std::vector<T>*>(dst)->assign(p, p + bytes / sizeof(T));
        }
        // `fill` is null for an add_raw region, which is copied with memcpy.
        struct Region {
            void* host;
            const void* device;
            size_t bytes;
            void (*fill)(void* dst, const void* staged, size_t bytes);
        };
        const EngineState& engine_;
        std::vector<Region> regions_;
    };

    // The state-edge readback, sized from a snapshot: the per-state slices into `slices`, and
    // the edge id lists, edge table and vertex pool into `out`'s state_edge_ids, edge_records
    // and vertex_pool.
    void add_state_edges(ReadbackBatch& batch, const CounterSnapshot& snap,
                         std::vector<StateEdgeSlice>& slices, EvolveResult& out) const;
    void add_events(ReadbackBatch& batch, uint32_t n, std::vector<DeviceEvent>& out) const;
    void add_causal_edges(ReadbackBatch& batch, uint32_t n,
                          std::vector<DeviceCausalEdge>& out) const;
    void add_branchial_edges(ReadbackBatch& batch, uint32_t n,
                             std::vector<DeviceBranchialEdge>& out) const;
private:
    // ReadbackBatch's staging buffer: pinned, grown to the largest batch, freed with the engine.
    mutable void*  pinned_       = nullptr;
    mutable size_t pinned_bytes_ = 0;
    uint32_t*                          state_edge_ids_counter_ = nullptr;
    uint32_t*                          state_count_            = nullptr;
    uint64_t*                          state_canonical_hash_   = nullptr;
    uint64_t*                          state_exact_hash_       = nullptr;
    hgcommon::RecordSet                record_{};
    uint32_t*                          state_edge_rank_        = nullptr;
    uint32_t*                          tr_scratch_             = nullptr;
    uint32_t                           tr_scratch_slots_       = 0;
    uint64_t*                          survivor_scratch_       = nullptr;
    // Sampling and capping. The weights and the two counters are the only device memory these
    // options need; everything else the draw reads was already here.
    double                             transition_rate_        = 1.0;
    double*                            rule_weights_dev_       = nullptr;
    uint32_t                           num_rule_weights_       = 0;
    uint64_t                           sampling_seed_          = 0;
    uint32_t                           max_states_per_step_    = 0;
    uint32_t*                          states_per_step_        = nullptr;
    uint32_t                           states_per_step_slots_  = 0;
    uint32_t                           max_succ_per_parent_    = 0;
    uint32_t*                          successors_per_parent_  = nullptr;
    uint32_t                           matches_per_state_rule_ = 0;
    uint32_t                           num_rules_              = 0;
    uint32_t*                          state_edge_orbit_       = nullptr;
    uint32_t*                          state_num_orbits_       = nullptr;
    uint64_t*                          keyed_token_sum_        = nullptr;
    uint32_t*                          keyed_first_new_edge_   = nullptr;
    uint32_t*                          keyed_follow_head_      = nullptr;
    uint32_t*                          keyed_follow_next_      = nullptr;
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
    uint32_t slice_scan_max_edges_ = 256;
    bool maintain_indices_ = true;
    uint32_t* needs_indices_ = nullptr;

public:
    // Host readers for tests / EvolveResult population.
    uint32_t num_events_host()          const;
    uint32_t num_causal_edges_host()    const;
    uint32_t num_branchial_edges_host() const;

    std::vector<DeviceEvent> events_host() const;
    std::vector<DeviceCausalEdge> causal_edges_host() const;
    std::vector<DeviceBranchialEdge> branchial_edges_host() const;
    // The same readbacks with the count already in hand (from a counter snapshot); the
    // zero-argument forms above delegate here after one size read.
    std::vector<DeviceEvent> events_host(uint32_t n) const;
    std::vector<DeviceCausalEdge> causal_edges_host(uint32_t n) const;
    std::vector<DeviceBranchialEdge> branchial_edges_host(uint32_t n) const;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE