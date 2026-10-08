#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/overflow.hpp"   // ErrorKind / OverflowWarning
#include "hg_gpu/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include "hgcommon/core.hpp"
#include "hgcommon/quotient_replay_core.hpp"  // QR_ID_LIMIT

#include <set>
#include <utility>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

struct RewriteRule {
    std::vector<std::vector<uint8_t>> lhs;
    std::vector<std::vector<uint8_t>> rhs;
    uint8_t num_lhs_vars = 0;
    uint8_t num_rhs_vars = 0;
};

struct EvolveInput {
    // Which artifacts this run must RECORD. An artifact turned off is never built: the
    // rendezvous that would produce it does not run. Defaults to everything, so a caller that
    // states nothing gets what it always got.
    hgcommon::RecordSet record;

    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> initial_state;
    // Multiple initial states (multiway with several roots). When non-empty this
    // takes precedence over initial_state. Each becomes a separate root state;
    // isomorphic roots merge under explore_from_canonical_states_only.
    std::vector<std::vector<std::vector<VertexId>>> initial_states;

    uint32_t num_steps = 0;
    // Canonicalization is always McKay individualization-refinement (IR):
    // only IR is correct on graphs with non-trivial automorphism.
    CanonicalizationMode      canonicalization     = CanonicalizationMode::Full;
    EventCanonicalizationMode event_canonicalization = EventCanonicalizationMode::None;
    bool transitive_reduction = true;

    // MATERIALISE the reconstructed relations, rather than only reporting their sizes.
    //
    // The counts come from the device's own counters and cost nothing. The PAIRS are an
    // expansion of the applications they are derived from -- 133,218,996 against 970,584 on
    // disc-l3a2g2r2 depth 3 -- so building the vectors is 3,798 ms of a 4,114 ms run there,
    // against 309 ms of actual evolution. Off by default because a caller that reads only the
    // counts must not pay for the expansion; the host draws the same line, between a counter and
    // an enumeration the caller drives.
    bool materialize_relations = false;
    // The reconstructed applications' input class, output class and rule, for a caller that
    // reads them as events or graphs (EvolveResult::reconstructed_event_from_class ...).
    bool materialize_events = false;
    // Read back the state-edge arrays (EvolveResult::state_edge_ids, edge_records, vertex_pool).
    // Off, a state carries its id and hash and no edges, and the four copies are skipped.
    bool materialize_state_edges = true;

    // Quotient exploration: expand each canonical state exactly once, at its
    // shortest depth, so the run costs the canonical closure rather than the
    // provenance count, claiming each state at its minimum depth. Canonical
    // states and the (input, output, rule)
    // transition multiset match the CPU engine's quotient mode; exact causal
    // and branchial multisets of the full expansion are reconstructed offline
    // from this skeleton (tools/quotient_reconstruction_probe.cpp). False
    // expands every provenance, the reference/MultiwayReference.wl semantics.
    // GPU defaults to true for bounded state growth on deep evolutions.
    bool explore_from_canonical_states_only = true;

    // Stochastic exploration pruning. Each newly-deduped state at the end
    // of a step is admitted to the next-step frontier with probability
    // `exploration_probability`. Mirrors the CPU
    // ParallelEvolutionEngine::set_exploration_probability option: the
    // state and its event are still recorded in EvolveResult; only the
    // *expansion* from that state is suppressed when the coin lands
    // unfavourable. 1.0 = always explore (default, equivalent to no
    // pruning); 0.0 = never expand any new state (only the initial state
    // is matched). Values outside [0,1] are clamped.
    //
    // `exploration_seed` is the run's seed (RandomSeed): it keys the exploration coin, the
    // transition draw and the caps' ranks, as random_seed does on the host. 0 is a seed like any
    // other. The coin is hgcommon::explore_survives on the host's key: the class's canonical hash
    // under quotient exploration, the creating transition's key under full capture.
    double   exploration_probability = 1.0;
    uint64_t exploration_seed        = 0;

    // SAMPLING AND CAPPING, the same options the host accepts and applies. These were reported
    // to the caller as unimplemented on the device; the rules are in hgcommon/sampling_core.hpp
    // and the inputs they need were already here.
    //
    // transition_rate scaled by rule_weights[rule] decides whether a transition is taken at all,
    // drawn from the transition's own identity so the kept subgraph is the same at any thread
    // count and on either engine. rule_weights may be empty (every rule weighted 1) or shorter
    // than the rule set (rules past its end weighted 1).
    double   transition_rate = 1.0;
    std::vector<double> rule_weights;
    // Hard bounds. 0 means unlimited, as on the host.
    uint32_t max_states_per_step = 0;
    uint32_t max_successor_states_per_parent = 0;
    uint32_t matches_per_state_rule = 0;

    // Override for EngineConfig::slice_scan_max_edges (0 keeps the default).
    // Tests set a tiny value to force the index-backed match path and the
    // lazy index rebuild on small workloads.
    uint32_t slice_scan_max_edges = 0;

    // Override for EngineConfig::max_blocks_per_launch (0 keeps the default).
    uint32_t max_blocks_per_launch = 0;

    // Hard ceiling on device memory the engine may allocate, in bytes. The
    // one-shot evolve() stops its grow-and-retry before a config's estimated
    // footprint would exceed this, returning the best partial result so far with
    // a kDeviceOutOfMemory warning rather than pushing the GPU to a real OOM.
    // 0 (default) resolves to 90% of total device memory at run start. Ignored by
    // the direct Engine(cfg) path, which allocates exactly what its cfg asks for.
    uint64_t max_device_memory_bytes = 0;

    // Test lever (EngineConfig::canonical_key_mask): narrows a Full-mode state's first dedup
    // probe key so that non-isomorphic states share keys.
    uint64_t canonical_key_mask = ~uint64_t{0};
    // Test lever (EngineConfig::event_key_mask): the same for event identities.
    uint64_t event_key_mask = ~uint64_t{0};
    // Test lever (EngineConfig::replay_id_limit): the replay refuses raw event ids at or past it.
    uint32_t replay_id_limit = hgcommon::QR_ID_LIMIT;
    // Keyed rewrites (hgcommon/token_core.hpp, EngineConfig::keyed_rewrites): in Full mode a
    // state whose token set an earlier state holds takes its canonical results without IR.
    bool keyed_rewrites = true;
    // Test lever (EngineConfig::keyed_claim_limit): twin claims without a twin before a run
    // stops keying.
    uint32_t keyed_claim_limit = 1024;
    // Test lever (EngineConfig::keyed_sum_mask): narrows the twin claim key.
    uint64_t keyed_sum_mask = ~uint64_t{0};
};

// One (class, depth) point of the multiplicity count: m raw states of the class at the depth.
struct ClassMultiplicity {
    uint64_t class_hash;
    uint32_t depth;
    uint64_t multiplicity;
};

// How many of a class's captured matches apply a rule.
struct ClassRuleMatches {
    uint64_t class_hash;
    uint32_t rule;
    uint64_t count;
};

// A run of ids in a buffer the result owns: one edge's vertices, or an event's edges.
template <class T>
struct IdSpan {
    const T* first = nullptr;
    uint32_t count = 0;
    const T* begin() const { return first; }
    const T* end() const { return first + count; }
    uint32_t size() const { return count; }
    bool empty() const { return count == 0; }
    const T& operator[](uint32_t i) const { return first[i]; }
};
using VertexSpan = IdSpan<VertexId>;
using EdgeSpan = IdSpan<EdgeId>;

// A state and where its edge ids sit in EvolveResult::state_edge_ids: first_edge ..
// first_edge + num_edges. EvolveResult::edge_ids, edge and edges_of read them.
struct CanonicalState {
    StateId id = INVALID_ID;
    uint64_t canonical_hash = 0;
    uint32_t first_edge = 0;
    uint32_t num_edges = 0;
};

// Events and relations as the device records them (types.hpp), read back without conversion.
// Event::signature is 0 under EventCanonicalizationMode::None. EvolveResult::consumed_of and
// produced_of give an event's edge ids.
using Event = DeviceEvent;
using CausalEdge = DeviceCausalEdge;
using BranchialEdge = DeviceBranchialEdge;

// The consecutive edge ids first .. first + count - 1: an event's produced edges.
struct EdgeRange {
    EdgeId first = 0;
    uint32_t count = 0;
    struct iterator {
        EdgeId id;
        EdgeId operator*() const { return id; }
        iterator& operator++() { ++id; return *this; }
        bool operator!=(const iterator& o) const { return id != o.id; }
    };
    iterator begin() const { return {first}; }
    iterator end() const { return {first + count}; }
    uint32_t size() const { return count; }
    bool empty() const { return count == 0; }
    EdgeId operator[](uint32_t i) const { return first + i; }
};

struct EvolveResult {
    std::vector<CanonicalState> states;
    // The device's state-edge arrays, as read back when EvolveInput::materialize_state_edges is
    // set, and empty otherwise. state_edge_ids holds every state's global edge ids, each state's
    // run at CanonicalState::first_edge; edge_records is the global edge table, indexed by
    // EdgeId; an edge's vertices are vertex_pool[vertex_offset .. vertex_offset + arity).
    std::vector<EdgeId> state_edge_ids;
    std::vector<Edge> edge_records;
    std::vector<VertexId> vertex_pool;

    // State `s`'s global edge ids.
    EdgeSpan edge_ids(const CanonicalState& s) const {
        return {state_edge_ids.data() + s.first_edge, s.num_edges};
    }
    // Global edge `eid`'s vertices; empty for an id past the table or a record past the pool.
    VertexSpan edge_vertices(EdgeId eid) const {
        if (eid >= edge_records.size()) return {};
        const Edge& e = edge_records[eid];
        if (static_cast<size_t>(e.vertex_offset) + e.arity > vertex_pool.size()) return {};
        return {vertex_pool.data() + e.vertex_offset, e.arity};
    }
    // Edge `i` of state `s`.
    VertexSpan edge(const CanonicalState& s, uint32_t i) const {
        return edge_vertices(state_edge_ids[s.first_edge + i]);
    }
    // State `s`'s edges as nested vectors, for callers that take that form.
    std::vector<std::vector<VertexId>> edges_of(const CanonicalState& s) const {
        std::vector<std::vector<VertexId>> out;
        out.reserve(s.num_edges);
        for (uint32_t i = 0; i < s.num_edges; ++i) {
            const VertexSpan e = edge(s, i);
            out.emplace_back(e.begin(), e.end());
        }
        return out;
    }
    std::vector<Event> events;
    // Every event's consumed edge ids; Event::consumed_at indexes it.
    std::vector<EdgeId> event_consumed;
    std::vector<CausalEdge> causal_edges;
    std::vector<BranchialEdge> branchial_edges;

    // Event `e`'s consumed edge ids, in match order, and its produced ones.
    EdgeSpan consumed_of(const Event& e) const {
        return {event_consumed.data() + e.consumed_at, e.num_consumed};
    }
    static EdgeRange produced_of(const Event& e) { return {e.first_produced, e.num_produced}; }

    // Takes `from`'s eight large vectors, emptied with their capacity kept, so the readback
    // fills memory that is already paged in. `from` is left with empty vectors.
    void adopt_storage(EvolveResult& from) {
        auto take = [](auto& mine, auto& theirs) { mine.swap(theirs); mine.clear(); };
        take(states, from.states);
        take(state_edge_ids, from.state_edge_ids);
        take(edge_records, from.edge_records);
        take(vertex_pool, from.vertex_pool);
        take(events, from.events);
        take(event_consumed, from.event_consumed);
        take(causal_edges, from.causal_edges);
        take(branchial_edges, from.branchial_edges);
    }

    // Capacity overflows observed during the run. Empty on a successful
    // (uncapped) run; otherwise contains one OverflowWarning per
    // (kernel-launch × ErrorKind) overflow event with a `context` string
    // identifying the phase ("match kernel step 3", "rewrite kernel
    // step 5", "ir hash", etc.). The result still contains whatever was
    // successfully computed before the overflow — it's a partial result,
    // not an error. The free `evolve()` wrapper inspects this list to
    // drive its grow-and-retry loop; explicit `Engine.run()` callers can
    // inspect it themselves and decide whether the partial result is
    // good enough.
    std::vector<OverflowWarning> warnings;

    // Class-frame expansion records captured (quotient reconstruction only, 0 otherwise). The
    // per-instance replay reads these, so a device that captures a different number from the
    // host cannot agree with it on event identity. Compared against the host's
    // for_each_expansion_match total in the differential suite.
    uint32_t expansion_matches = 0;

    // Per-class instances recorded (quotient reconstruction only, 0 otherwise). One per raw
    // occurrence of a class at a depth: one per root class before the replay lands, one more
    // per application afterwards.
    uint32_t expansion_instances = 0;

    // Raw events the per-instance replay minted: one per (instance, match) application. The
    // host's num_raw_events under the reconstruction. 0 when the route is off.
    uint64_t reconstructed_raw_events = 0;

    // Frame alignment: slots the class frame MOVED off the recording state's own labelling, and
    // slots for which no frame image existed (the capture is dropped, so events reachable only
    // through it are missing).
    // Did this run reconstruct its events from the class-frame expansion? True on the quotient
    // route and under Automatic identity, where a class holds many raw states and each would
    // otherwise be ranked from its own presentation.
    bool reconstruction_ran = false;

    // The reconstruction's event count: distinct identities under the run's mode, or the raw
    // application count under EVENT_SIG_NONE, where every application is its own event.
    uint64_t reconstructed_events = 0;

    // The reconstruction's causal relation: distinct (producer, consumer) pairs, and the
    // consumed-edge occurrences behind them. 0 when the route is off.
    uint32_t reconstructed_causal_pairs = 0;
    uint32_t reconstructed_causal_edges = 0;
    // Distinct branchial pairs: sibling applications of one instance whose consumed edges
    // overlap. 0 when the route is off.
    // Pairs tagged in-reduction: the TR view of the same relation.
    uint32_t reconstructed_causal_pairs_reduced = 0;
    uint64_t reconstructed_branchial = 0;

    // The reconstructed relations as pairs of content triples hash(input class, output class,
    // rule) -- the identity a cross-engine comparison is made on, since raw event ids are
    // minted in application order and mean nothing between engines. Empty when the route is off.
    std::vector<std::pair<uint64_t, uint64_t>> reconstructed_causal_relation;
    std::vector<std::pair<uint64_t, uint64_t>> reconstructed_causal_relation_reduced;
    std::vector<std::pair<uint64_t, uint64_t>> reconstructed_branchial_relation;
    // The same relations as RAW application-id pairs -- the reply's data form. The triple-pair
    // vectors above identify endpoints by content, which two distinct applications can share;
    // a caller serving "the raw causal edge lists" needs one entry per raw pair with endpoints
    // it can join against the event list, and that is the application id.
    std::vector<std::pair<uint32_t, uint32_t>> reconstructed_causal_raw;
    std::vector<std::pair<uint32_t, uint32_t>> reconstructed_causal_raw_reduced;
    std::vector<std::pair<uint32_t, uint32_t>> reconstructed_branchial_raw;

    // Per RAW event id, the reconstruction's schedule-stable content signature -- the same
    // hash(input class, output class, rule) its relations are keyed on, and 0 for an id the
    // replay never minted.
    //
    // WITHOUT THIS a caller cannot build a graph over the events the COUNT describes. Under an
    // identity mode observable_num_events reports the reconstruction's distinct identities,
    // while a graph built by mapping materialised events through their own canonical_id
    // describes a different set: measured 25 vertices against a count of 24 on the device, the
    // same discrepancy the host removed by routing identity through the reconstruction instead.
    std::vector<uint64_t> reconstructed_event_signature;
    // With materialize_events on the reconstruction route: per raw event, its input class, output
    // class and rule. Empty otherwise.
    std::vector<uint64_t> reconstructed_event_from_class;
    std::vector<uint64_t> reconstructed_event_to_class;
    std::vector<uint32_t> reconstructed_event_rule;

    // With record.multiplicities under quotient exploration: the (class, depth) multiplicities
    // and each class's matches per rule (QeState::class_multiplicities_host).
    std::vector<ClassMultiplicity> class_multiplicities;
    std::vector<ClassRuleMatches> class_rule_matches;

    uint32_t frame_alignments = 0;
    uint32_t frame_align_failures = 0;

    // The number a caller is told, mirroring the host's Hypergraph::observable_num_events.
    // Off the reconstruction route an event that lost a signature slot carries the winner's id,
    // so counting events that are their own canonical gives the identity count under Full or
    // Automatic and the raw count under None, without a mode test.
    // The causal pair count a caller is told, mirroring Hypergraph::observable_num_causal_pairs.
    // `reduced` selects the TR view; both come from the same base, so either is available at no
    // extra cost. Off the reconstruction route this is the materialised relation, deduplicated
    // by (from, to) as the FFI reports it.
    size_t observable_num_causal_pairs(bool reduced) const;

    // The branchial pair count a caller is told, mirroring
    // Hypergraph::observable_num_branchial.
    size_t observable_num_branchial() const;

    size_t observable_num_events() const;
};

// Sizing knobs, chosen by config_from_input from the workload and adjusted by the
// host's grow-and-retry when a run overflows. Defaults handle the differential corpus.
// All POD so this header stays host-includable without CUDA dependencies.
struct EngineConfig {
    // No padding (PersistentEvolver compares configs with memcmp): the 64-bit fields first, then
    // the 32-bit ones, then the two 16-bit ones.
    // Test lever: the first dedup probe key of a Full-mode state is its canonical hash ANDed
    // with this. All ones except in tests, which narrow it so non-isomorphic states share keys.
    uint64_t canonical_key_mask   = ~uint64_t{0};

    // Test lever: the first probe key of an event signature, a replay class's run signature and
    // the exact hash None/Automatic event identity reads is ANDed with this.
    uint64_t event_key_mask       = ~uint64_t{0};
    // Test lever: the twin claim key is the token sum ANDed with this.
    uint64_t keyed_sum_mask       = ~uint64_t{0};
    uint32_t max_edges            = 1u << 16;   // 65K edge slots
    uint32_t max_vertices         = 1u << 16;   // 65K vertex IDs (atomic counter ceiling)
    uint32_t max_vertex_slots     = 1u << 18;   // 256K flat vertex-tuple slots (avg arity ≤ 4)
    uint32_t max_states           = 1u << 14;   // 16K state slots
    // CSR-packed per-state edge lists. The flat ids pool is sized for the
    // sum of all state edge counts over the course of the evolution;
    // empirically max_states * avg_state_edges fits most workloads. Sizing
    // is linear in the total edge-slot count, not max_states * max_edges.
    uint32_t max_state_edge_total = 1u << 22;   // 4M EdgeId slots (16 MB)
    uint32_t inverted_pool        = 1u << 18;   // shared LockFreeList node capacity
    // Words of canonical-form records: one record per Full-mode canonical state, a 3-word
    // header and its IR canonical form at 1, 2 or 4 bytes per word (hgcommon/
    // canonical_form_core.hpp). Grown on kCanonicalFormsFull.
    uint32_t canonical_form_words = 1u << 22;
    // The replay refuses raw event ids at or past this (kReplayIdsExhausted).
    // hgcommon::QR_ID_LIMIT except in tests.
    uint32_t replay_id_limit      = hgcommon::QR_ID_LIMIT;
    // Keyed rewrites in Full-mode persistent runs (keyed.hpp), and the claims without a twin after
    // which a run stops keying.
    uint32_t keyed_rewrites       = 1;
    uint32_t keyed_claim_limit    = 1024;
    uint32_t match_dedup_slots    = 1u << 16;
    // States at or below this edge count are matched by scanning their own CSR
    // slice; the global indices are only consulted (and therefore maintained)
    // once some state exceeds it. See DeviceState::slice_scan_max_edges.
    uint32_t slice_scan_max_edges = 256;

    // Cap on the number of blocks per match/rewrite kernel launch. When a step's
    // grid exceeds this, the launch is split into consecutive chunks with a sync
    // between, bounding any single kernel's duration so a very deep/wide step
    // cannot trip the display driver's watchdog (WDDM TDR, ~2 s). 0 = no limit
    // (single launch). Tune to the target GPU's TDR budget; current workloads
    // (match a few ms, rewrite ~100 ms at depth 8) are far below it.
    uint32_t max_blocks_per_launch = 0;
    uint32_t event_canon_slots    = 1u << 16;


    // Event / causal / branchial sizing.
    uint32_t max_events           = 1u << 16;
    uint32_t max_causal_edges     = 1u << 18;
    uint32_t max_branchial_edges  = 1u << 18;
    uint32_t causal_triple_slots  = 1u << 19;   // dedup map for (p,c,e) triples
    uint32_t causal_pair_slots    = 1u << 18;   // dedup map for (p,c) pairs
    uint32_t branchial_pair_slots = 1u << 19;
    uint32_t edge_consumer_nodes  = 1u << 18;   // LockFreeList node pool
    // Branchial co-consumer index: buckets hashed by (state, consumed edge),
    // entries pack (event, edge). Sized like edge_consumer_nodes: one entry
    // per consumed-edge registration.
    uint32_t branchial_index_buckets = 1u << 16;  // power of two
    uint32_t branchial_index_nodes   = 1u << 18;

    // Online transitive reduction stores ONE structure: the reduced predecessor adjacency
    // (preds[c] = producers of c's kept causal edges), one list node per unique kept causal
    // pair. Reachability is answered by backward search over it, so no closure is stored.
    uint32_t tr_preds_nodes = 1u << 20;

    // THE QUOTIENT REPLAY'S TABLES, in five groups, each with its own capacity and its own
    // overflow kind, so grow-and-retry doubles only the group that filled. 0 means the default
    // in qe_entries(). The groups fill at different rates: on bigpath n256 at three steps there
    // are 769 classes and 16,842,496 raw instances.
    //   qe_class_entries     captured matches, class representatives and match counts,
    //                        multiplicity points (kQcNodes); default max_events
    //   qe_instance_entries  instances, bound list, instance lists (kQeInstancesFull);
    //                        default max_events
    //   qe_event_entries     per raw event: task log (2 per event), applied lists (2), content,
    //                        identity and kept producers (kQeEventsFull); default max_events
    //   qe_pair_entries      (instance, match) claims and causal pairs, 4 slots per entry
    //                        (kQePairsFull); default max_events
    //   qe_word_entries      the word arena (kQeWordsFull); default 16 * max_events
    uint32_t qe_class_entries    = 0;
    uint32_t qe_instance_entries = 0;
    uint32_t qe_event_entries    = 0;
    uint32_t qe_pair_entries     = 0;
    uint32_t qe_word_entries     = 0;

    // Multiplies the per-driver queues of the multiplicity cascade (64 items per level, at least
    // 256). Grow-and-retry doubles this on kQeWorkOverflow, which would otherwise drop a point's
    // pass.
    uint32_t descent_work_scale = 1u;

    // Multiplies each block's global scratch for the transitive reduction's reachability search
    // (kTrScratchStack stack words and kTrScratchVisited visited words, engine_state.hpp). A search
    // that fills it records kTrScratchOverflow, and grow-and-retry doubles this.
    uint32_t tr_scratch_scale = 1u;

    // Survivor-list entries in each block's global survivor scratch, for states with more than
    // kLocalSurvivors (256) edges under quotient exploration. 0 allocates none; grow-and-retry sets
    // it on kQeSurvivorsOverflow and doubles it after.
    uint32_t survivor_scratch = 0u;

    // Average IR-arena words per concurrent slot holder (the arena is one shared bump pool, so
    // the average share is what matters, not a per-worker partition). The default is ~6x the
    // measured average demand on multiway state sizes; a big-state workload that outgrows the
    // pool records kIRArenaExhausted, which grow-and-retry answers by doubling THIS.
    uint32_t ir_arena_share_words = 65536;

    // Automorphism generators the device IR search may retain, per thread. NOT a correctness
    // limit and not a tuning nicety: for search PRUNING a short table costs time only, but for
    // ORBITS it changes the answer, since orbits are fused over the generators found and a
    // short table gives orbits that are too FINE. A state whose automorphism group outruns
    // this records kIRGeneratorsExceeded, which grow-and-retry answers by doubling THIS rather
    // than by publishing a finer partition than the group licenses. Scratch cost is
    // generators x n_verts words PER THREAD, which is why the device default is far below the
    // host's.
    uint32_t ir_generators = 32;

    // Individualization depth the device IR search explores before giving up. A path fixes at
    // least one vertex per level, so n levels always suffice; this bounds the per-thread slot
    // instead, since the depth blocks are its bulk. A state needing more records
    // kIRDepthExceeded, which grow-and-retry answers by doubling THIS -- the device never keys
    // a state by a coarser hash, because a coarser key MERGES non-isomorphic states.
    uint32_t ir_depth = 8;
    // The largest edge arity the run can hold (initial edges and every rule's right-hand side);
    // 0 when unknown, read as kMaxArity. Bounds a state's vertex occurrences by edges x this.
    uint16_t max_edge_arity = 0;
    // The largest left-hand side among the rules; 0 when unknown, read as kMaxPatternEdges.
    uint16_t max_lhs_edges = 0;
};

// The replay's group capacities with the defaults resolved (EngineConfig::qe_class_entries and
// the four after it). Each is at most kQeEntryLimit, and words at most kQeWordLimit. `states`
// is max_states: the per-class arrays indexed by a class's representative, which is a raw
// state, are sized by it.
struct QeEntries {
    uint32_t classes, instances, events, pairs, words;
    uint32_t states = 1;
};
inline constexpr uint32_t kQeEntryLimit = 1u << 28;
inline constexpr uint32_t kQeWordLimit  = 1u << 31;
QeEntries qe_entries(const EngineConfig& cfg);

// An event's consumed ids take this many words of DeviceState::event_consumed.
inline uint32_t event_consumed_stride(const EngineConfig& cfg) {
    return cfg.max_lhs_edges && cfg.max_lhs_edges < kMaxPatternEdges ? cfg.max_lhs_edges
                                                                      : kMaxPatternEdges;
}

// One-shot evolve: constructs a fresh Engine for `input`, runs once,
// destructs. Each call pays the per-Engine CUDA setup cost (allocating
// pools, indices, lock-free lists — ~5–20 ms on a warmed-up driver, but
// significantly more on the first CUDA call of the process). For repeated
// evaluations of similar workloads use Engine + run() directly to amortise
// the setup.
//
// Capacity-overflow handling (run_with_growth, evolve.cu): if the kernels report overflow
// warnings, this wrapper doubles the relevant EngineConfig field(s) and retries (up to 8 times,
// 256x growth), with a new Engine at the bigger size each time. The result carries its own
// warnings only, so a clean attempt returns none; the discarded attempts and the sizes that
// worked are logged to stderr. A result that still has an overflow warning is partial.
EvolveResult evolve(const EvolveInput& input);

// Engine: persistent device-state container that can run() multiple
// EvolveInputs back-to-back, amortising the per-call CUDA setup. Use
// Engine when benchmarking, when running a parameter sweep, or whenever
// the caller controls the workload lifecycle.
//
// Lifecycle:
//   Engine engine(cfg);          // one-time pool/index allocation
//   for each input:
//       auto result = engine.run(input);
//       // engine is auto-reset between run() calls.
//
// `run()` calls `reset()` internally before processing, so the caller does
// not need to reset between runs. The caller may also call reset()
// explicitly if they want to clear results without running.
//
// engine.run(input) tolerates capacity overflow gracefully — the kernels
// keep running on whatever budget they have, the underlying error
// channel records each overflow into result.warnings, and the partial
// result (whatever states/events/causal/branchial were successfully
// computed before the overflow point) is returned. Engine.run() never
// throws for capacity reasons; it only throws on genuine programmer
// errors (invalid EvolveInput, CUDA driver failures). If the caller
// wants auto-grow-and-retry behaviour, use the free `evolve()` wrapper
// — Engine.run() is the explicit-control path that benchmarks use.
class EngineState;  // forward decl
// Declared in persistent.hpp, which includes THIS header, so a forward declaration is what
// keeps the two from cycling. Only ever used through a pointer here.
struct SessionView;

// Build a sensible EngineConfig for a given workload. Used by the
// one-shot evolve() and exposed publicly so callers building their own
// Engine can size it consistently. Conservative: oversizes to handle
// pre-dedup state-blow-up for the worst step. A workload that outgrows the
// estimate is answered by grow-and-retry, which logs the config that worked.
EngineConfig config_from_input(const EvolveInput& input);

// Conservative estimate of the device memory an engine built from `cfg` will
// allocate, in bytes. Used to enforce EvolveInput::max_device_memory_bytes
// before construction. Monotonic in every capacity field.
uint64_t estimated_device_bytes(const EngineConfig& cfg);

// Double the EngineConfig fields that bound `kind`. False for a kind no config field bounds
// (kScratchOverflow is a fixed per-thread limit): the grow-and-retry ladder cannot fix it.
bool grow_config_for(EngineConfig& cfg, ErrorKind kind);

// A device session, as HOST code can hold one.
//
// SessionState itself lives in persistent.hpp and reaches for cuda/atomic, so a host translation
// unit -- hg_gpu_backend.cpp is compiled by the host compiler, not nvcc -- cannot include it.
// This is the same object behind a pimpl, so the backend can own a session without seeing a
// single device type. view() hands back a pointer the evolver understands and the caller never
// dereferences.
class GpuSession {
public:
    GpuSession(uint32_t max_states, uint32_t max_events);
    ~GpuSession();
    GpuSession(const GpuSession&)            = delete;
    GpuSession& operator=(const GpuSession&) = delete;

    SessionView* view();
    // The state ids its per-state arrays cover: an engine it runs on holds at most this many.
    uint32_t state_capacity() const;
    // Boundary states the last call's budget refused, so a caller can tell a converged run from
    // one that stopped at its budget.
    uint32_t frontier_size() const;
    // The frontier's entries (device state ids) with each entry's recorded depth, read between
    // launches so a caller can resolve a steered continuation against them.
    void frontier_host(std::vector<StateId>& ids, std::vector<uint32_t>& steps) const;
    // Replaces the frontier: a steered Step writes the selected subset before the run, and the
    // merge of the new boundary with the retained entries after it. Entries beyond the session's
    // capacity are dropped, the same partial-result discipline the device append uses.
    void set_frontier_host(const StateId* ids, const uint32_t* steps, uint32_t n);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Engine {
public:
    explicit Engine(EngineConfig cfg);
    ~Engine();
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    // `session` non-null makes the run CONTINUABLE: identity is remembered across calls and
    // the budget's boundary states are recorded. `start_step` non-zero continues from that
    // frontier instead of re-seeding the roots. See persistent.hpp for why a frontier is
    // required rather than simply re-running. `storage` non-null lends its vectors to the
    // result (EvolveResult::adopt_storage).
    EvolveResult run(const EvolveInput& input, SessionView* session = nullptr,
                     uint32_t start_step = 0, EvolveResult* storage = nullptr);
    void reset();

    const EngineConfig& config() const;

private:
    struct Impl;
    Impl* impl_;
};

// PersistentEvolver: the same grow-and-retry robustness as the free evolve(),
// but it keeps its device Engine alive across run() calls. The free evolve()
// builds and destroys a whole Engine every call, and that per-call allocation
// (tens of ms of cudaMalloc/cudaFree) dominates small and medium workloads --
// 6-13x of the wall time on interactive-sized runs. A caller that evolves many
// times (the persistent worker process, a benchmark, a notebook session) should
// hold one PersistentEvolver: the engine is sized on the first run, only ever
// grows (on overflow), never shrinks, so every subsequent run reuses it. Results
// are identical to evolve(); run() resets the engine internally between calls.
class PersistentEvolver {
public:
    PersistentEvolver();
    ~PersistentEvolver();
    PersistentEvolver(const PersistentEvolver&)            = delete;
    PersistentEvolver& operator=(const PersistentEvolver&) = delete;

    EvolveResult run(const EvolveInput& input);

    // A SESSION PINS THE ENGINE. run() above may rebuild engine_ -- on a config change, or after
    // an overflow throw, which deliberately discards it so a poisoned device does not infect
    // every later call. Either would destroy a session's accumulated states while returning a
    // result that looks like a continuation, so a session run REFUSES rather than rebuilds:
    // `ok` false means the handle is dead and the caller must reopen. That is the device twin of
    // the host invalidating its session slot on a throw.
    struct SessionRun {
        EvolveResult result;
        bool ok = false;
        std::string error;
    };
    SessionRun run_session(const EvolveInput& input, SessionView* session, uint32_t start_step);

    // The config the engine was built with, so a session can tell whether a later call would
    // rebuild it.
    bool has_engine() const;
    const EngineConfig& engine_config() const;

    // Hands a result the caller has finished with back to the evolver. The next run() or
    // run_session() reads back into its vectors, which are already paged in: on cycle4 depth 6
    // the host fill of 24 MB takes 3.1 ms into new allocations and 1.2 ms into reused ones.
    // The evolver holds that storage until the next run.
    void recycle(EvolveResult&& done);

private:
    std::unique_ptr<Engine> engine_;
    EngineConfig            cfg_{};
    bool                    has_engine_ = false;
    EvolveResult            spare_;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE