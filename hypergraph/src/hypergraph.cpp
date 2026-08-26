#include "hgcommon/phase_timing.hpp"
#include "hgcommon/namespace.hpp"
// hypergraph.cpp - Implementation of Hypergraph class non-template methods

#include "hypergraph/hypergraph.hpp"

#include <unordered_map>
#include "hypergraph/ir_canonicalization.hpp"
#include "hgcommon/ir_core.hpp"
#include "hgcommon/slot_core.hpp"
#include "hypergraph/atomic_compat.hpp"
#include <thread>

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// Edge Management
// =============================================================================

EdgeId Hypergraph::create_edge(
    const VertexId* vertices,
    uint8_t arity,
    EventId creator_event,
    uint32_t step
) {
    // Downstream code (pattern matcher, EdgeSignature) uses fixed-size MAX_ARITY
    // buffers on the stack. Reject over-arity edges rather than silently corrupt.
    if (arity > MAX_ARITY) {
        throw std::length_error("Hypergraph::create_edge: arity exceeds MAX_ARITY");
    }

    EdgeId eid = counters_.alloc_edge();

    // Small-arity edges store their vertices inline in the Edge; only higher-arity
    // edges spill to an arena array. The Edge constructor copies from `vertices` into
    // whichever storage applies, so no separate allocation happens on the common path.
    VertexId* spill = (arity > Edge::INLINE_ARITY)
                          ? arena_.allocate_array<VertexId>(arity)
                          : nullptr;

    // Directly construct edge at slot eid using emplace_at
    edges_.emplace_at(eid, arena_, eid, vertices, arity, spill, creator_event, step);

    // CRITICAL: Release fence to ensure vertex data and edge struct are visible
    std::atomic_thread_fence(std::memory_order_release);

    // Compute and cache edge signature (immutable after creation)
    edge_signatures_.emplace_at(eid, arena_, EdgeSignature::from_edge(vertices, arity));

    // Update indices
    match_index_.add_edge(eid, vertices, arity, arena_);

    return eid;
}

EdgeId Hypergraph::create_edge(std::initializer_list<VertexId> vertices,
                               EventId creator_event,
                               uint32_t step) {
    // Fail loudly on over-arity rather than silently dropping vertices past
    // MAX_ARITY. The pointer/arity overload does the same check.
    if (vertices.size() > MAX_ARITY) {
        throw std::length_error("Hypergraph::create_edge: arity exceeds MAX_ARITY");
    }
    VertexId verts[MAX_ARITY];
    uint8_t arity = 0;
    for (VertexId v : vertices) {
        verts[arity++] = v;
    }
    return create_edge(verts, arity, creator_event, step);
}

// =============================================================================
// State Management
// =============================================================================

StateId Hypergraph::create_state(
    SparseBitset&& edge_set,
    uint32_t step,
    uint64_t canonical_hash,
    EventId parent_event
) {
    StateId sid = counters_.alloc_state();

    // Directly construct state at slot sid using emplace_at
    states_.emplace_at(sid, arena_, sid, std::move(edge_set), step, canonical_hash, parent_event);

    // CRITICAL: Release fence to ensure state data is visible
    std::atomic_thread_fence(std::memory_order_release);

    return sid;
}

StateId Hypergraph::create_state(
    const EdgeId* edge_ids,
    uint32_t num_edges,
    uint32_t step,
    uint64_t canonical_hash,
    EventId parent_event
) {
    SparseBitset edge_set;
    for (uint32_t i = 0; i < num_edges; ++i) {
        edge_set.set(edge_ids[i], arena_);
    }
    return create_state(std::move(edge_set), step, canonical_hash, parent_event);
}

StateId Hypergraph::create_state(std::initializer_list<EdgeId> edge_ids,
                                 uint32_t step,
                                 uint64_t canonical_hash,
                                 EventId parent_event) {
    SparseBitset edge_set;
    for (EdgeId eid : edge_ids) {
        edge_set.set(eid, arena_);
    }
    return create_state(std::move(edge_set), step, canonical_hash, parent_event);
}

StateId Hypergraph::get_or_create_genesis_state() {
    // The empty state every initial state descends from. Created on demand, and NOBODY WAITS
    // for it: a thread that finds it uninitialised builds one and offers it, and whichever
    // offer wins is the one everyone uses. Losing threads discard their candidate.
    //
    // Electing an initialiser and having the others spin until it published was the previous
    // shape, and it made every other thread's progress depend on one thread being scheduled --
    // and on it not throwing, since a claim abandoned mid-flight parked them permanently.
    // A discarded empty state costs one state id and nothing else, which is a better trade
    // than a dependency on someone else's timeline.
    StateId current = genesis_state_.load(std::memory_order_acquire);
    if (current != INVALID_ID) return current;

    // EMPTY_STATE_CANONICAL_HASH, not 0, because that is what compute_canonical_hash gives an
    // empty edge set -- the empty state must have ONE hash however it came to exist. Zero is
    // also the ConcurrentMap EMPTY sentinel, so a genesis keyed by it made every map that keys
    // on a canonical hash throw the moment genesis reached one (quotient exploration with a
    // rule that empties the state).
    SparseBitset empty_edges;
    const StateId candidate =
        create_state(std::move(empty_edges), 0, EMPTY_STATE_CANONICAL_HASH, INVALID_ID);

    StateId expected = INVALID_ID;
    if (genesis_state_.compare_exchange_strong(expected, candidate,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
        return candidate;
    }
    return expected;   // another thread's genesis won; ours is simply unused
}

// =============================================================================
// Canonical State Deduplication
// =============================================================================

Hypergraph::CanonicalStateResult Hypergraph::create_or_get_canonical_state(
    SparseBitset&& edge_set,
    uint32_t step,
    EventId parent_event,
    StateId incr_parent,
    const EdgeId* incr_consumed, uint8_t incr_num_consumed,
    const EdgeId* incr_produced, uint8_t incr_num_produced
) {
    // Create the state; its canonical hash is filled in below.
    StateId new_sid = create_state(std::move(edge_set), step, 0, parent_event);
    const SparseBitset& edges = get_state(new_sid).edges;

    // Reported hash for None/Automatic modes: the exact invariant when event
    // canonicalization is on (see compute_reported_canonical_hash), the fast WL hash
    // otherwise. Separate from map_key, which is what actually decides state identity.
    // One individualization-refinement pass serves both when event canonicalization is on:
    // the exact hash the event path resolves representatives through, and the per-edge ranks
    // it identifies consumed/produced edges by.
    const bool need_ranks = (event_signature_keys_ != EVENT_SIG_NONE);
    uint64_t ranked_hash = 0;
    if (need_ranks) ranked_hash = cache_state_edge_ranks(new_sid, edges);

    auto reported_child = [&]() -> uint64_t {
        return need_ranks ? ranked_hash : compute_canonical_hash(edges);
    };

    // Canonical identity + dedup key. In Full mode the exact IR hash is BOTH the
    // canonical identity and the dedup key, computed once (no redundant WL pass);
    // other modes use the fast WL hash for identity + a mode-specific dedup key.
    // Use atomic load with acquire to ensure we see the mode set by the main thread.
    uint64_t map_key, canonical_hash;
    switch (state_canonicalization_mode_.load(std::memory_order_acquire)) {
        case StateCanonicalizationMode::None:
            // +1: the dedup key is the raw state id, and canonical_state_map_ reserves 0 as its
            // EMPTY-slot sentinel. Without the offset the first state (id 0) keys to 0, which
            // count_unique() cannot store or count, silently undercounting None by one. The offset
            // keeps ids unique (None never dedups) while lifting id 0 off the sentinel.
            map_key = static_cast<uint64_t>(new_sid) + 1;
            canonical_hash = reported_child();
            break;
        case StateCanonicalizationMode::Automatic:
            map_key = compute_content_ordered_hash(edges);
            canonical_hash = reported_child();
            break;
        case StateCanonicalizationMode::Full:
        default:
            // In quotient mode compute the edge-orbit table and take the canonical hash
            // from the same IR canonicalization (the quotient causal reconstruction needs
            // the orbits; there is no extra canon pass). Otherwise just the dedup hash.
            if (quotient_causal_.load(std::memory_order_relaxed))
                canonical_hash = compute_and_cache_state_orbits(new_sid, edges);
            else if (need_ranks)
                canonical_hash = ranked_hash;   // exact IR, already computed with the ranks
            else
                canonical_hash = compute_canonical_hash(edges);   // exact IR
            map_key = canonical_hash;
            break;
    }
    // Any mode whose key hashes to 0 would hit the same EMPTY=0 sentinel; nudge it off (mirrors the
    // GPU's h==0?1:h guard). None is already offset above, so this only ever affects a 0-valued hash.
    if (map_key == 0) map_key = 1;
    // create_state has already published new_sid, so another thread can be reading this
    // state's canonical_hash (get_or_compute_canonical_hash, get_canonical_state_for_event)
    // while this store runs. Both sides go through atomic_ref: the store carries the
    // computed hash, and the acquire loads pick it up.
    hgcommon::atomic_ref<uint64_t>(states_[new_sid].canonical_hash)
        .store(canonical_hash, std::memory_order_release);

    // Try to insert into canonical map (lock-free, waiting for LOCKED slots)
    auto [existing_or_new, was_inserted] = canonical_state_map_.insert_if_absent_waiting(map_key, new_sid);

    // Insert into event_canonical_state_map_ only when event canonicalization is on:
    // its sole reader (get_canonical_state_for_event) runs only under
    // event_signature_keys_ != EVENT_SIG_NONE, and the keys are fixed at config time
    // before any state is created, so gating here never drops a needed entry. When
    // event canon is off this saves ~16 B/state + the map's resize chain + a per-state
    // hash+probe insert.
    // A 0 hash means the mode computed none (WL selected with no hasher configured);
    // get_canonical_state_for_event reads that as "fall back to the raw state", so there
    // is nothing to key an entry on. Every state with a hash has a non-zero one --
    // the empty state included, via EMPTY_STATE_CANONICAL_HASH.
    if (event_signature_keys_ != EVENT_SIG_NONE && canonical_hash != 0) {
        event_canonical_state_map_.insert_if_absent_waiting(canonical_hash, new_sid);
    }

    // In Full mode, the map key is the IR canonical hash which is exact —
    // hash collisions are genuine isomorphisms, no verification needed
    bool verified_duplicate = !was_inserted;

    // Cache the canonical ID in the state for fast lookup. Released here and acquired by
    // get_canonical_state(); the store itself is what carries the edge, since a bare fence
    // pairs with another fence only through an intervening atomic on the same object.
    hgcommon::atomic_ref<StateId>(states_[new_sid].canonical_id)
        .store(existing_or_new, std::memory_order_release);

    if (verified_duplicate) {
        return {existing_or_new, new_sid, false};
    }

    return {new_sid, new_sid, true};
}

bool Hypergraph::try_lower_explore_depth(StateId canonical_id, uint32_t depth) {
    if (canonical_id == INVALID_ID) return false;
    hgcommon::atomic_ref<uint32_t> known(states_[canonical_id].explore_depth);
    uint32_t cur = known.load(std::memory_order_acquire);
    while (depth < cur) {
        if (known.compare_exchange_weak(cur, depth,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

bool Hypergraph::try_claim_expanded(StateId canonical_id) {
    if (canonical_id == INVALID_ID) return false;
    hgcommon::atomic_ref<uint32_t> flag(states_[canonical_id].expanded);
    uint32_t expected = 0;
    return flag.compare_exchange_strong(expected, 1,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
}

uint32_t Hypergraph::explore_depth_of(StateId canonical_id) const {
    if (canonical_id == INVALID_ID) return INVALID_ID;
    hgcommon::atomic_ref<uint32_t> known(const_cast<uint32_t&>(states_[canonical_id].explore_depth));
    return known.load(std::memory_order_acquire);
}

// Build the state's canonical rank table, returning the exact canonical hash from the SAME
// individualization-refinement pass. Both are wanted whenever event canonicalization is on --
// the hash for the representative lookup, the ranks for the edge identity -- and computing
// them together is what keeps this to ONE pass per state rather than two per event.
//
// The state's edges are taken in EdgeId order, which is the "original index" the rank's
// tie-break uses: deterministic, and a property of the state rather than of the schedule that
// built it. Called once on the creating thread; insert_if_absent guards the rest.
uint64_t Hypergraph::cache_state_edge_ranks(StateId state_id, const SparseBitset& edges) {
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> edge_vectors;
    SVec<EdgeId> ids;
    std::atomic_thread_fence(std::memory_order_acquire);
    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        edge_vectors.emplace_back(e.vertices, e.vertices + e.arity);
        ids.push_back(eid);
    });

    const uint32_t n = static_cast<uint32_t>(ids.size());
    uint64_t hash = EMPTY_STATE_CANONICAL_HASH;
    EdgeId* arr_edges = arena_.allocate_array<EdgeId>(n ? n : 1);
    uint32_t* arr_rank = arena_.allocate_array<uint32_t>(n ? n : 1);
    if (n > 0) {
        // Flatten to the shared core's convention and take the hash AND the ranks from one
        // pass. The core is the code the device runs, so an event identity built on these
        // ranks means the same thing on both.
        SVec<uint8_t> ea;
        SVec<uint32_t> eoff, ev;
        ea.reserve(n); eoff.reserve(n); ev.reserve(n * 2);
        for (uint32_t i = 0; i < n; ++i) {
            eoff.push_back(static_cast<uint32_t>(ev.size()));
            ea.push_back(static_cast<uint8_t>(edge_vectors[i].size()));
            for (VertexId v : edge_vectors[i]) ev.push_back(v);
        }
        SVec<uint32_t> verts(ev.begin(), ev.end());
        std::sort(verts.begin(), verts.end());
        verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
        const uint32_t n_verts = static_cast<uint32_t>(verts.size());
        for (uint32_t& x : ev)
            x = static_cast<uint32_t>(std::lower_bound(verts.begin(), verts.end(), x) - verts.begin());

        const uint32_t total_occ = static_cast<uint32_t>(ev.size());
        SVec<uint32_t> ranks(n);
        bool ok = false;
        for (uint32_t depth : {1u, 8u, hgcommon::IR_MAX_DEPTH_DEFAULT}) {
            const uint64_t words = hgcommon::ir_scratch_words(n_verts, n, total_occ, depth);
            auto* scratch = static_cast<uint32_t*>(
                worker_scratch().allocate_raw((words + 2) * sizeof(uint32_t), alignof(uint64_t)));
            auto r = hgcommon::ir_canonical_hash(ea.data(), eoff.data(), ev.data(),
                                                 n, n_verts, total_occ, scratch, depth,
                                                 ranks.data());
            if (r.status == hgcommon::IR_OK) { hash = r.hash; ok = true; break; }
            if (r.status == hgcommon::IR_EMPTY) break;
        }
        if (!ok) {
            IRCanonicalizer ir;
            thread_local std::vector<uint32_t> fallback_ranks;
            hash = ir.compute_canonical_hash_with_edge_rank(edge_vectors, fallback_ranks);
            for (uint32_t i = 0; i < n; ++i) ranks[i] = fallback_ranks[i];
        }
        for (uint32_t i = 0; i < n; ++i) { arr_edges[i] = ids[i]; arr_rank[i] = ranks[i]; }
    }
    worker_scratch().release(mk);

    EdgeRankTable* tbl = arena_.template create<EdgeRankTable>();
    tbl->n = n; tbl->edges = arr_edges; tbl->rank = arr_rank;
    // +1: the map reserves key 0 as its EMPTY-slot sentinel, so state 0 needs the offset.
    state_edge_rank_tables_.insert_if_absent(static_cast<uint64_t>(state_id) + 1, tbl);
    return hash;
}

void Hypergraph::ensure_state_edge_ranks(StateId state_id, const SparseBitset& edges) {
    if (state_edge_rank_tables_.lookup(static_cast<uint64_t>(state_id) + 1).has_value()) return;
    cache_state_edge_ranks(state_id, edges);
}

StateId Hypergraph::get_canonical_state_for_event(StateId raw_state) const {
        if (raw_state == INVALID_ID) return INVALID_ID;

        // Get the isomorphism-invariant hash for this state. Written concurrently by
        // create_or_get_canonical_state and get_or_compute_canonical_hash, so acquire it.
        const State& state = get_state(raw_state);
        uint64_t hash = hgcommon::atomic_ref<uint64_t>(const_cast<uint64_t&>(state.canonical_hash))
            .load(std::memory_order_acquire);

        // If hash is 0, the state's hash wasn't computed - fall back to raw state
        if (hash == 0) return raw_state;

        // Lookup in event_canonical_state_map_ which is always keyed by canonical_hash
        auto result = event_canonical_state_map_.lookup_waiting(hash);
        return result.value_or(raw_state);
    }

uint32_t Hypergraph::edge_rank_in_state(StateId state_id, EdgeId edge) const {
        auto r = state_edge_rank_tables_.lookup(static_cast<uint64_t>(state_id) + 1);
        if (!r.has_value()) return UINT32_MAX;
        const EdgeRankTable* t = *r;
        uint32_t lo = 0, hi = t->n;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (t->edges[mid] < edge) lo = mid + 1; else hi = mid;
        }
        return (lo < t->n && t->edges[lo] == edge) ? t->rank[lo] : UINT32_MAX;
    }

void Hypergraph::reserve_vertices(VertexId max_id) {
        VertexId current = counters_.next_vertex.load(std::memory_order_relaxed);
        while (current <= max_id) {
            if (counters_.next_vertex.compare_exchange_weak(
                    current, max_id + 1, std::memory_order_relaxed)) {
                break;
            }
        }
    }

uint64_t Hypergraph::get_or_compute_canonical_hash(StateId state_id) {
    if (state_id == INVALID_ID) return 0;

    State& state = states_[state_id];

    // canonical_hash can be written (by this function) concurrently with reads
    // elsewhere (e.g. event canonicalization, match forwarding). Use atomic_ref
    // for the fast-path read and the publishing store so the concurrent access
    // is not a formal data race. On 64-bit targets the underlying load/store
    // are already single instructions, so this compiles to the same code plus
    // the appropriate fences.
    hgcommon::atomic_ref<uint64_t> atomic_hash(state.canonical_hash);
    uint64_t cached = atomic_hash.load(std::memory_order_acquire);
    if (cached != 0) {
        return cached;
    }

    // On-demand, and it must agree with what create_or_get_canonical_state published --
    // the event path reads both.
    uint64_t hash = compute_reported_canonical_hash(state.edges);

    // Publish with release; racing writers may all compute the same value and
    // the final stored value is deterministic across threads.
    atomic_hash.store(hash, std::memory_order_release);
    return hash;
}

// =============================================================================
// Event Management
// =============================================================================

Hypergraph::CreateEventResult Hypergraph::create_event(
    StateId input_state,
    StateId output_state,
    RuleIndex rule_index,
    const EdgeId* consumed,
    uint8_t num_consumed,
    const EdgeId* produced,
    uint8_t num_produced
) {
    // Allocate event ID
    EventId eid = counters_.alloc_event();

    bool is_canonical = true;
    EventId canonical_eid = eid;
    uint64_t event_signature_value = 0;

    // Event canonicalization: check if this event signature already exists
    if (event_signature_keys_ != EVENT_SIG_NONE) {
        const EventSignatureKeys keys = event_signature_keys_;

        // Get canonical state IDs for event canonicalization
        StateId canonical_input = get_canonical_state_for_event(input_state);
        StateId canonical_output = get_canonical_state_for_event(output_state);
        const State& canonical_out_state = get_state(canonical_output);

        // Ranks of the consumed and produced edges, in match and RHS order. A missing rank
        // means no rank table for that state; the raw edge id stands in and is COUNTED,
        // because such a signature is not an isomorphism invariant and a caller comparing
        // event counts across runs needs to know it happened.
        uint32_t consumed_ranks[MAX_PATTERN_EDGES];
        uint32_t produced_ranks[MAX_PATTERN_EDGES];
        if (keys & EventKey_ConsumedEdges) {
            for (uint8_t i = 0; i < num_consumed; ++i) {
                uint32_t r = edge_rank_in_state(input_state, consumed[i]);
                if (r == UINT32_MAX) {
                    event_sig_raw_fallbacks_.fetch_add(1, std::memory_order_relaxed);
                    r = consumed[i];
                }
                consumed_ranks[i] = r;
            }
        }
        if (keys & EventKey_ProducedEdges) {
            for (uint8_t i = 0; i < num_produced; ++i) {
                uint32_t r = edge_rank_in_state(output_state, produced[i]);
                if (r == UINT32_MAX) {
                    event_sig_raw_fallbacks_.fetch_add(1, std::memory_order_relaxed);
                    r = produced[i];
                }
                produced_ranks[i] = r;
            }
        }

        const uint64_t sig_key = hgcommon::event_signature(
            keys,
            (keys & EventKey_InputState)  ? get_or_compute_canonical_hash(input_state)  : 0,
            (keys & EventKey_OutputState) ? get_or_compute_canonical_hash(output_state) : 0,
            canonical_out_state.step, rule_index,
            consumed_ranks, num_consumed, produced_ranks, num_produced);

        // Try to insert this signature
        auto [existing_or_new, was_inserted] = canonical_event_map_.insert_if_absent_waiting(sig_key, eid);

        if (!was_inserted) {
            is_canonical = false;
            canonical_eid = existing_or_new;
        } else {
            canonical_event_count_.fetch_add(1, std::memory_order_relaxed);
        }
        event_signature_value = sig_key;
    }

    // Allocate and copy edge arrays
    EdgeId* cons = arena_.allocate_array<EdgeId>(num_consumed);
    std::memcpy(cons, consumed, num_consumed * sizeof(EdgeId));

    EdgeId* prod = arena_.allocate_array<EdgeId>(num_produced);
    std::memcpy(prod, produced, num_produced * sizeof(EdgeId));

    // Directly construct event at slot eid using emplace_at
    EventId canonical_id_for_event = is_canonical ? INVALID_ID : canonical_eid;
    events_.emplace_at(eid, arena_, eid, input_state, output_state, rule_index,
                       cons, num_consumed, prod, num_produced, canonical_id_for_event);
    events_[eid].signature = event_signature_value;

    // CRITICAL: Release fence to ensure event data is visible
    std::atomic_thread_fence(std::memory_order_release);

    return {eid, canonical_eid, is_canonical};
}

EventId Hypergraph::create_genesis_event(StateId initial_state, const EdgeId* edges, uint8_t num_edges) {
    // Ensure genesis state exists
    StateId genesis = get_or_create_genesis_state();

    // Allocate event ID
    EventId eid = counters_.alloc_event();

    // Event canonicalization for genesis events
    bool is_canonical = true;
    EventId canonical_eid = eid;

    if (event_signature_keys_ != EVENT_SIG_NONE) {
        const EventSignatureKeys keys = event_signature_keys_;

        // Get canonical state IDs
        StateId canonical_output = get_canonical_state(initial_state);
        const State& canonical_out_state = get_state(canonical_output);

        // Build signature from selected keys
        uint64_t sig_key = FNV_OFFSET;

        if (keys & EventKey_InputState) {
            uint64_t input_hash = get_or_compute_canonical_hash(genesis);
            sig_key = fnv_hash(sig_key, input_hash);
        }
        if (keys & EventKey_OutputState) {
            uint64_t output_hash = get_or_compute_canonical_hash(initial_state);
            sig_key = fnv_hash(sig_key, output_hash);
        }
        if (keys & EventKey_Step) {
            sig_key = fnv_hash(sig_key, static_cast<uint64_t>(canonical_out_state.step));
        }
        if (keys & EventKey_ProducedEdges) {
            for (uint8_t i = 0; i < num_edges; ++i) {
                sig_key = fnv_hash(sig_key, static_cast<uint64_t>(edges[i]));
            }
        }

        if (sig_key == 0 || sig_key == FNV_OFFSET) sig_key = 1;

        auto [existing_or_new, was_inserted] = canonical_event_map_.insert_if_absent_waiting(sig_key, eid);

        if (!was_inserted) {
            is_canonical = false;
            canonical_eid = existing_or_new;
        } else {
            canonical_event_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Allocate produced edges array
    EdgeId* produced = arena_.allocate_array<EdgeId>(num_edges);
    std::memcpy(produced, edges, num_edges * sizeof(EdgeId));

    // Directly construct event at slot eid using emplace_at
    EventId canonical_id_for_event = is_canonical ? INVALID_ID : canonical_eid;
    events_.emplace_at(eid, arena_, eid, genesis, initial_state,
                       static_cast<RuleIndex>(-1),
                       nullptr, 0,  // consumed_edges (none)
                       produced, num_edges,  // produced_edges
                       canonical_id_for_event);

    // CRITICAL: Release fence
    std::atomic_thread_fence(std::memory_order_release);

    // Register this event as the producer of all initial edges, keyed by the initial
    // state's canonical edge identities (the same keys consumers of those edges will mint).
    // num_edges is a uint8_t, so a 256-slot buffer holds every initial edge without a cap.
    CanonicalEdgeKey init_keys[256];
    causal_edge_keys(initial_state, edges, num_edges, init_keys);
    for (uint8_t i = 0; i < num_edges; ++i) {
        set_edge_producer(init_keys[i], eid, edges[i]);
    }

    return eid;
}


// =============================================================================
// Canonical Hash Computation
// =============================================================================

uint64_t Hypergraph::compute_content_ordered_hash(const SparseBitset& edges) const {
    // The rule is hgcommon::ContentHasher; only the ITERATION is ours. The device walks an edge
    // slice with a liveness filter and cannot share this loop, but it must share every constant
    // and every mixing step, which is what the hasher holds.
    hgcommon::ContentHasher ch(static_cast<uint32_t>(edges.count()));
    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        ch.edge_begin(e.arity);
        for (uint8_t i = 0; i < e.arity; ++i) ch.vertex(static_cast<uint64_t>(e.vertices[i]));
        ch.edge_end();
    });
    return ch.value();
}

uint64_t Hypergraph::compute_canonical_hash(const SparseBitset& edges) const {
    // Full mode uses the exact IR hash as the canonical identity (it is also the
    // dedup key), so there is no separate WL pass. Other modes use the fast WL hash.
    bool full = state_canonicalization_mode_.load(std::memory_order_acquire)
                == StateCanonicalizationMode::Full;
    if (!full && use_wl_hash_ && wl_hash_) {
        return compute_wl_hash(edges);
    }
    return compute_exact_canonical_hash(edges);
}

// The hash a state REPORTS, and the one the event path resolves representatives through.
//
// Event identity is defined over ISOMORPHISM classes and SPEC.md sec 4 makes it independent
// of the state-identity choice. Resolving it through the mode-aware hash breaks that: the WL
// hash is isomorphism-invariant but COARSER than IR, so outside Full mode more states share a
// representative, the edge correspondence resolves to a coarser one, and the event identity
// derived from it coarsens with it -- measured on the binary-growth corpus case as 8 events
// under Full against 6 under Automatic, so the count moved with the state mode.
//
// So when event canonicalization is on, the reported hash is the exact invariant in every
// state mode. It is only paid for when it is asked for; with event canonicalization off this
// is the mode-aware hash and nothing changes.
uint64_t Hypergraph::compute_reported_canonical_hash(const SparseBitset& edges) const {
    if (event_signature_keys_ != EVENT_SIG_NONE) return compute_exact_canonical_hash(edges);
    return compute_canonical_hash(edges);
}

uint64_t Hypergraph::compute_exact_canonical_hash(const SparseBitset& edges) const {
    hgcommon::PhaseTimer _pt(hgcommon::Phase::Canon);
    canonical_hash_computations_.fetch_add(1, std::memory_order_relaxed);
    // Exact canonical hash via individualization-refinement.
    // Flattened straight from the edge set into the per-worker scratch arena (no heap) and
    // handed to the shared CPU/GPU core, so both devices agree bit for bit.
    auto mk = worker_scratch().mark();

    std::atomic_thread_fence(std::memory_order_acquire);

    // Reserved from the edge count so the three buffers are bumped once each rather than
    // grown by repeated doubling; MAX_ARITY bounds the occurrences.
    const size_t edge_count = edges.count();
    SVec<uint8_t> ea;
    SVec<uint32_t> eoff, ev;
    ea.reserve(edge_count);
    eoff.reserve(edge_count);
    ev.reserve(edge_count * 2);
    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        eoff.push_back(static_cast<uint32_t>(ev.size()));
        ea.push_back(e.arity);
        for (uint8_t p = 0; p < e.arity; ++p) ev.push_back(e.vertices[p]);
    });

    if (ea.empty()) {
        worker_scratch().release(mk);
        return EMPTY_STATE_CANONICAL_HASH;
    }

    // Local vertex indices, assigned in encounter order through a direct-mapped table.
    //
    // The core's result does not depend on which order they are assigned in: the only place
    // an index is read as a value is the initial partition's tie-break, which orders vertices
    // WITHIN a cell, and no output reads within-cell order. (The equivalence probe checks
    // this the other way round, by relabeling every state three times.) So the indices need
    // not be ranks, and this costs one pass instead of a sort plus a binary search per
    // occurrence.
    //
    // The table is per-worker and grows monotonically; a generation stamp makes reuse O(1)
    // instead of clearing it, so its cost amortises to nothing across states.
    static thread_local std::vector<uint32_t> local_index;
    static thread_local std::vector<uint32_t> stamp;
    static thread_local uint32_t generation = 0;
    uint32_t max_vid = 0;
    for (uint32_t x : ev) max_vid = std::max(max_vid, x);
    if (stamp.size() <= max_vid) {
        stamp.assign(static_cast<size_t>(max_vid) * 2 + 64, 0);
        local_index.resize(stamp.size());
        generation = 0;
    }
    ++generation;
    uint32_t n_verts = 0;
    for (uint32_t& x : ev) {
        if (stamp[x] != generation) { stamp[x] = generation; local_index[x] = n_verts++; }
        x = local_index[x];
    }

    const uint32_t n_edges = static_cast<uint32_t>(ea.size());
    const uint32_t total_occ = static_cast<uint32_t>(ev.size());

    // Escalating depth. Almost every state is discrete straight after refinement, and at
    // depth 1 the core sizes for exactly that: no per-level partition blocks, no generator
    // rows. Only a state that actually needs the individualization search pays for it, and
    // it pays on the retry -- where the search dominates the re-run anyway.
    for (uint32_t depth : {1u, 8u, hgcommon::IR_MAX_DEPTH_DEFAULT}) {
        const uint64_t words =
            hgcommon::ir_scratch_words(n_verts, n_edges, total_occ, depth);
        // Raw, so the buffer is not zeroed on the way in: the core writes every word it
        // later reads. 8-byte aligned for the uint64 views it takes inside the span.
        auto* scratch = static_cast<uint32_t*>(
            worker_scratch().allocate_raw((words + 2) * sizeof(uint32_t), alignof(uint64_t)));
        auto r = hgcommon::ir_canonical_hash(
            ea.data(), eoff.data(), ev.data(), n_edges, n_verts, total_occ, scratch, depth);
        if (r.status == hgcommon::IR_OK) {
            worker_scratch().release(mk);
            return r.hash;
        }
        if (r.status == hgcommon::IR_EMPTY) break;
    }

    // A state whose individualization path outruns even the largest depth: fall back to the
    // unbounded-depth implementation, which allocates per level.
    SVec<SVec<VertexId>> edge_vectors;
    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        edge_vectors.emplace_back(e.vertices, e.vertices + e.arity);
    });
    IRCanonicalizer ir;
    uint64_t h = ir.compute_canonical_hash(edge_vectors);
    worker_scratch().release(mk);
    return h;
}

uint64_t Hypergraph::compute_wl_hash(const SparseBitset& edges) const {
    hgcommon::PhaseTimer _pt(hgcommon::Phase::Canon);
    canonical_hash_computations_.fetch_add(1, std::memory_order_relaxed);
    if (edges.empty()) {
        return EMPTY_STATE_CANONICAL_HASH;
    }

    std::atomic_thread_fence(std::memory_order_acquire);

    if (!wl_hash_) {
        return 0;
    }
    EdgeVertexAccessorRaw vert_acc(this);
    EdgeArityAccessorRaw arity_acc(this);
    return wl_hash_->compute_state_hash(edges, vert_acc, arity_acc);
}


// =============================================================================
// Edge Correspondence Dispatch
// =============================================================================

namespace {

// Canonical hash and per-edge ORBITS for one state's edges, escalating both bounds.
//
// ONE BODY, because there are two callers and they must agree: compute_and_cache_state_orbits
// builds the slot table the quotient reconstruction identifies instances by, and
// causal_edge_keys mints the causal edge keys from the same orbits. Two implementations of
// "which edges are the same up to automorphism" is exactly the divergence the prime directive
// exists to prevent -- and it is not hypothetical here, since these two sites already differed
// in which implementation they called.
//
// BOTH BOUNDS ESCALATE. Depth reports IR_NEED_DEPTH; the generator table reports
// IR_NEED_GENERATORS, but only when orbits were requested -- which they always are here.
// Orbits are fused over the generators found, so a short table fuses less and yields orbits
// that are too FINE: a wrong identity, not a slow run. More generators cannot rescue a depth
// failure, so the inner loop stops on IR_NEED_DEPTH.
//
// `orbit` and `klass` are resized to the edge count and filled. Returns the canonical hash.
uint64_t ir_hash_and_orbits(const SVec<SVec<VertexId>>& edge_vecs,
                            std::vector<uint32_t>& orbit,
                            std::vector<uint32_t>& klass) {
    const uint32_t n = static_cast<uint32_t>(edge_vecs.size());
    orbit.assign(n, 0);
    klass.assign(n, 0);
    if (n == 0) return EMPTY_STATE_CANONICAL_HASH;

    SVec<uint8_t> ea;
    SVec<uint32_t> eoff, ev;
    ea.reserve(n); eoff.reserve(n); ev.reserve(size_t(n) * 2);
    for (uint32_t i = 0; i < n; ++i) {
        eoff.push_back(static_cast<uint32_t>(ev.size()));
        ea.push_back(static_cast<uint8_t>(edge_vecs[i].size()));
        for (VertexId v : edge_vecs[i]) ev.push_back(static_cast<uint32_t>(v));
    }
    // Local vertex indices in encounter order. The core's result does not depend on the order
    // they are assigned in: the only place an index is read as a value is the initial
    // partition's tie-break, which orders vertices WITHIN a cell, and no output reads
    // within-cell order.
    std::unordered_map<uint32_t, uint32_t> local;
    uint32_t n_verts = 0;
    for (uint32_t& x : ev) {
        auto it = local.find(x);
        if (it == local.end()) { local.emplace(x, n_verts); x = n_verts++; }
        else x = it->second;
    }
    const uint32_t total_occ = static_cast<uint32_t>(ev.size());

    for (uint32_t depth : {1u, 8u, hgcommon::IR_MAX_DEPTH_DEFAULT}) {
        for (uint32_t gens = hgcommon::IR_HOST_GENERATORS; gens <= (1u << 16); gens *= 4u) {
            const uint64_t words = hgcommon::ir_scratch_words(n_verts, n, total_occ, depth, gens);
            auto* scratch = static_cast<uint32_t*>(worker_scratch().allocate_raw(
                (words + 2) * sizeof(uint32_t), alignof(uint64_t)));
            auto r = hgcommon::ir_canonical_hash(
                ea.data(), eoff.data(), ev.data(), n, n_verts, total_occ, scratch, depth,
                nullptr, gens, orbit.data(), klass.data());
            if (r.status == hgcommon::IR_OK)    return r.hash;
            if (r.status == hgcommon::IR_EMPTY) return EMPTY_STATE_CANONICAL_HASH;
            if (r.status == hgcommon::IR_NEED_DEPTH) break;
        }
    }
    // Past every depth AND generator budget above: the unbounded implementation rather than
    // orbits the automorphism group does not license.
    IRCanonicalizer ir;
    return ir.compute_canonical_hash_with_edge_orbits(edge_vecs, orbit, &klass);
}

}  // namespace

uint64_t Hypergraph::compute_and_cache_state_orbits(StateId s, const SparseBitset& edges) {
    hgcommon::PhaseTimer _pt(hgcommon::Phase::Canon);
    canonical_hash_computations_.fetch_add(1, std::memory_order_relaxed);
    // Materialize the state's edges (id-sorted via SparseBitset iteration) into scratch,
    // run the exact IR canonicalization with edge orbits, then copy a compact table into
    // the persistent arena and publish it under the state id. Called once per state on its
    // creating thread, so no same-state race; insert_if_absent is a belt-and-braces guard.
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> edge_vecs;
    SVec<EdgeId> ids;
    std::atomic_thread_fence(std::memory_order_acquire);
    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        edge_vecs.emplace_back(e.vertices, e.vertices + e.arity);
        ids.push_back(eid);
    });

    const uint32_t n = static_cast<uint32_t>(ids.size());
    EdgeId* arr_edges = arena_.allocate_array<EdgeId>(n ? n : 1);
    uint32_t* arr_orbit = arena_.allocate_array<uint32_t>(n ? n : 1);
    uint32_t* arr_slot  = arena_.allocate_array<uint32_t>(n ? n : 1);
    uint32_t* arr_class = arena_.allocate_array<uint32_t>(n ? n : 1);
    // The empty state's hash, for the n == 0 case that skips the canonicalizer below. It is the
    // same value compute_state_ranks_and_hash gives an empty edge set, because the empty state
    // is one state and must have one hash however it was reached. Zero is additionally the
    // ConcurrentMap EMPTY sentinel, and this hash is a key in every quotient map.
    uint64_t hash = EMPTY_STATE_CANONICAL_HASH;
    uint32_t num_orbits = 0;

    if (n > 0) {
        // Reused per worker rather than allocated per state: this runs once for every state
        // created under quotient, and the vectors would otherwise be a heap round-trip each
        // time.
        thread_local std::vector<uint32_t> orbit, klass;
        orbit.assign(n, 0);
        klass.assign(n, 0);

        hash = ir_hash_and_orbits(edge_vecs, orbit, klass);
        // ids are already ascending (SparseBitset iterates in id order), orbit is parallel.
        for (uint32_t i = 0; i < n; ++i) {
            arr_edges[i] = ids[i];
            arr_orbit[i] = orbit[i];
            arr_class[i] = klass[i];
            if (orbit[i] + 1 > num_orbits) num_orbits = orbit[i] + 1;
        }
        // Slot = rank under (ORBIT, EdgeId). The rule and its rationale live in
        // hgcommon/slot_core.hpp because the device records its expansion in the same
        // coordinates: two readings of this that drift by one tie-break produce replayed
        // events that are wrong and invisible. Bulk form here; the device reads one edge at a
        // time through slot_rank, and the two are asserted equal.
        {
            SVec<uint32_t> counts;
            counts.resize(num_orbits ? num_orbits : 1);
            hgcommon::slots_from_orbits(arr_orbit, n, arr_slot, counts.data(), num_orbits);
        }
    }
    uint32_t* arr_osize = arena_.allocate_array<uint32_t>(num_orbits ? num_orbits : 1);
    for (uint32_t j = 0; j < num_orbits; ++j) arr_osize[j] = 0;
    for (uint32_t i = 0; i < n; ++i) arr_osize[arr_orbit[i]]++;
    worker_scratch().release(mk);

    EdgeOrbitTable* tbl = arena_.template create<EdgeOrbitTable>();
    tbl->n = n; tbl->num_orbits = num_orbits;
    tbl->edges = arr_edges; tbl->orbit = arr_orbit; tbl->orbit_size = arr_osize;
    tbl->slot = arr_slot; tbl->klass = arr_class;
    // +1: the map reserves 0 as its EMPTY-slot sentinel, so a raw key of StateId 0 can never
    // be stored or found -- the initial state would silently have no orbit table, which
    // skipped INIT seeding in the producer-set DP and dropped the root class's matches from
    // the reconstruction. Same offset, same reason, as the None-mode dedup key.
    state_orbit_tables_.insert_if_absent(static_cast<uint64_t>(s) + 1, tbl);
    return hash;
}

// =============================================================================
// Quotient causal reconstruction (online depth-indexed producer-set propagation)
// =============================================================================

LockFreeList<EventId>* Hypergraph::qc_dsup_list(uint64_t key) {
    auto r = qc_dsup_.lookup(key);
    if (r.has_value()) return *r;
    auto* nl = arena_.template create<LockFreeList<EventId>>();
    auto ins = qc_dsup_.insert_if_absent(key, nl);
    return ins.second ? nl : ins.first;
}

void Hypergraph::qc_emit(EventId producer, EventId consumer) {
    // The INIT sentinel (INVALID_ID) marks initial edges, which have no producer. A
    // producer == consumer pair is NOT dropped: it is a canonical self-loop (two distinct
    // raw events of the same canonical type, a producer and a consumer, that collapse to
    // one representative here) and is present in the full-capture causal graph too.
    if (producer == INVALID_ID || consumer == INVALID_ID) return;
    causal_graph_.add_causal_edge(producer, consumer, 0);  // dedups by (producer,consumer)
}

void Hypergraph::qc_add_producer(uint64_t state_hash, uint32_t depth, uint32_t orbit,
                                 EventId producer) {
    auto c = qc_ctx();
    hgcommon::qc_add_producer(c, state_hash, depth, orbit, producer);
}

void Hypergraph::qc_process_transition(const CanonicalTransition& t, uint64_t from_hash,
                                       uint32_t depth) {
    auto c = qc_ctx();
    hgcommon::qc_process_transition(c, t, from_hash, depth);
}

void Hypergraph::qc_reach(uint64_t state_hash, uint32_t depth) {
    auto c = qc_ctx();
    hgcommon::qc_reach(c, state_hash, depth);
}

void Hypergraph::raise_quotient_max_steps(int max_steps) {
    int old = qc_max_steps_.load(std::memory_order_relaxed);
    while (max_steps > old &&
           !qc_max_steps_.compare_exchange_weak(old, max_steps, std::memory_order_relaxed)) {
    }
    if (max_steps <= old) return;

    // The points at depths that the old bound made terminal were reached and then left
    // unexpanded: qc_reach scanned their transitions and qc_process_transition declined every
    // one, so their producers never propagated and their instances never met a match. Under
    // the raised bound they are ordinary interior points, and driving them again from the
    // reached list restarts the cascade -- the deeper points it creates drive themselves,
    // since the bound is already raised when they are reached. Every step of the re-drive is
    // claimed (qc_reached_, qc_dsup_seen_, qc_applied_), so revisiting a driven point is a
    // no-op. Called between runs, with the workers drained.
    qc_reached_list_.for_each([&](const QcReachPoint& p) {
        // [old, max_steps): below the old bound the point was already driven, and AT the new
        // bound it must not be -- the final depth is produced into and never read, so
        // expanding it would replay a step the run was not asked for. The cascade pushes its
        // own deeper points onto this list as it goes, so the guard is what keeps the walk
        // from expanding the frontier it is creating.
        if (static_cast<int>(p.depth) < old || static_cast<int>(p.depth) >= max_steps) return;
        for_each_transition_from(p.state_hash, [&](const CanonicalTransition& t) {
            qc_process_transition(t, p.state_hash, p.depth);
        });
        if (!quotient_reconstruction_.load(std::memory_order_relaxed)) return;
        auto ri = qc_instances_.lookup(qc_key(p.state_hash, p.depth, 0));
        if (!ri.has_value()) return;
        (*ri)->for_each([&](const QcInstance& inst) {
            for_each_expansion_match(p.state_hash, [&](const SlotMatch& m) {
                qc_apply(inst, m, p.state_hash, p.depth);
            });
        });
    });
}

void Hypergraph::quotient_causal_seed(StateId initial_state, int max_steps) {
    qc_max_steps_.store(max_steps, std::memory_order_relaxed);
    const EdgeOrbitTable* orb = state_orbits(initial_state);
    const uint64_t h = get_state(initial_state).canonical_hash;
    if (orb) for (uint32_t j = 0; j < orb->num_orbits; ++j)
        qc_add_producer(h, 0, j, INVALID_ID);   // INIT sentinel producer
    qc_reach(h, 0);

    // Seed the per-instance reconstruction with the one instance of the initial state; its
    // edges have no producer.
    if (orb && quotient_reconstruction_.load(std::memory_order_relaxed)) {
        // Claim the initial state as its class's frame before any instance exists, so the root
        // producer vector and the expansion captured from it agree by construction.
        auto mk = worker_scratch().mark();
        SVec<uint32_t> slots(orb->n ? orb->n : 1);
        qc_frame_slots(h, initial_state, orb, slots.data());
        worker_scratch().release(mk);

        uint32_t* p0 = arena_.allocate_array<uint32_t>(orb->n ? orb->n : 1);
        for (uint32_t i = 0; i < orb->n; ++i) p0[i] = QC_NO_PRODUCER;
        qc_add_instance(h, 0, p0, orb->n);
    }
}


void Hypergraph::qc_record_causal(uint32_t producer, uint32_t consumer) {
    // Per-consumed-edge relationships (the T1 multiset) count every occurrence.
    qc_num_causal_edges_.fetch_add(1, std::memory_order_relaxed);

    const uint64_t pk = qc_pair_key(producer, consumer);
    if (!qc_causal_pairs_.insert(pk)) return;                          // pair already recorded
    qc_num_causal_pairs_.fetch_add(1, std::memory_order_relaxed);

}

void Hypergraph::qc_apply(const QcInstance& inst, const SlotMatch& m, uint64_t state_hash,
                          uint32_t depth) {
    QrCtx c{*this};
    hgcommon::qr_apply(c, inst, m, state_hash, depth);
}

void Hypergraph::qc_add_instance(uint64_t state_hash, uint32_t depth,
                                 const uint32_t* prod, uint32_t nslots) {
    const int maxs = qc_max_steps_.load(std::memory_order_relaxed);
    if (static_cast<int>(depth) > maxs) return;

    QcInstance inst;
    inst.id = qc_next_instance_.fetch_add(1, std::memory_order_relaxed);
    inst.nslots = nslots;
    inst.prod = prod;

    const uint64_t key = qc_key(state_hash, depth, 0);
    LockFreeList<QcInstance>* lst;
    auto r = qc_instances_.lookup(key);
    if (r.has_value()) lst = *r;
    else {
        auto* nl = arena_.template create<LockFreeList<QcInstance>>();
        auto ins = qc_instances_.insert_if_absent(key, nl);
        lst = ins.second ? nl : ins.first;
    }
    lst->push(inst, arena_);

    // Instances at the final depth are recorded but never expanded: the DP runs its match
    // loop over depths 0..steps-1, producing into depth steps and never reading it.
    if (static_cast<int>(depth) >= maxs) return;

    // Publish before scanning, so a match captured concurrently cannot be missed by both
    // sides (pairs with the fence in qc_capture_expansion).
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for_each_expansion_match(state_hash, [&](const SlotMatch& m) { qc_apply(inst, m, state_hash, depth); });
}

bool Hypergraph::qc_frame_slots(uint64_t state_hash, StateId s, const EdgeOrbitTable* orb,
                                uint32_t* out) {
    const uint64_t claim = static_cast<uint64_t>(s) + 1;
    auto r = qc_frame_.insert_if_absent(state_hash, claim);
    const uint64_t held = r.second ? claim : r.first;
    if (held == claim) {                       // this state defines the class's frame
        for (uint32_t i = 0; i < orb->n; ++i) out[i] = orb->slot[i];
        qc_check_frame_stable(s, out, orb->n);
        return true;
    }
    const StateId frame = static_cast<StateId>(held - 1);
    const EdgeOrbitTable* forb = state_orbits(frame);
    if (!forb || !forb->slot || forb->n != orb->n) return false;

    // Align this state's edges onto the frame's. The two states are isomorphic, so the
    // correspondence exists; it is defined up to an automorphism, which is exactly the
    // freedom that is harmless -- an automorphism permutes the frame coherently, mapping
    // matches to matches. What is NOT harmless is each state using its own labeling, which
    // is what this removes.
    EdgeCorrespondence c =
        find_edge_correspondence_dispatch(get_state(s).edges, get_state(frame).edges);
    if (!c.valid || c.count != orb->n) { qc_align_badcorr_.fetch_add(1, std::memory_order_relaxed); return false; }
    for (uint32_t i = 0; i < orb->n; ++i) out[i] = UINT32_MAX;
    for (uint32_t k = 0; k < c.count; ++k) {
        const uint32_t idx = orb->index_of(c.state1_edges[k]);
        if (idx < orb->n) out[idx] = forb->slot_of(c.state2_edges[k]);
    }
    for (uint32_t i = 0; i < orb->n; ++i) if (out[i] == UINT32_MAX) return false;
    qc_check_frame_stable(s, out, orb->n);
    return true;
}

void Hypergraph::qc_check_frame_stable(StateId s, const uint32_t* slots, uint32_t n) {
    uint64_t h = hgcommon::FNV_OFFSET;
    for (uint32_t i = 0; i < n; ++i) h = hgcommon::fnv_hash(h, slots[i]);
    h = h ? h : 1;
    auto r = qc_frame_sig_.insert_if_absent(static_cast<uint64_t>(s) + 1, h);
    if (!r.second && r.first != h) qc_frame_disagree_.fetch_add(1, std::memory_order_relaxed);
}

void Hypergraph::qc_capture_expansion(EventId e) {
    // Record this match of the expanded representative, in slots, undeduplicated. One
    // canonical state's expansion is defined by exactly one raw state: the first to publish
    // itself here wins, and events of any other raw state in the same class are ignored, so a
    // dedup race cannot double the expansion.
    const Event& ev = get_event(e);
    const EdgeOrbitTable* in_orb = state_orbits(ev.input_state);
    const EdgeOrbitTable* out_orb = state_orbits(ev.output_state);
    if (!in_orb || !out_orb || !in_orb->slot || !out_orb->slot) {
        qc_capture_no_orbits_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const uint64_t from = get_state(ev.input_state).canonical_hash;

    const uint64_t claim = static_cast<uint64_t>(ev.input_state) + 1;
    auto rep = qc_expansion_rep_.insert_if_absent(from, claim);
    if (!rep.second && rep.first != claim) {         // a different raw state owns this class
        qc_capture_not_rep_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const uint32_t nprod = ev.num_produced;

    // Resolve both endpoints into their class's frame. Every slot recorded below is a frame
    // slot, so a match captured on one raw state replays correctly against an instance built
    // from any other raw state of the same class.
    // Everything below is recorded in FRAME slots, so a match captured on one raw state
    // replays correctly against an instance built from any other raw state of the same class.
    // The scratch vectors live in an inner scope: the arena mark may only be released once
    // they are destroyed, or the rendezvous scan further down would allocate over them.
    uint32_t *cs = nullptr, *ps = nullptr, *sfs = nullptr, *sts = nullptr;
    uint32_t nsurv = 0;
    {
        auto mk = worker_scratch().mark();
        {
            SVec<uint32_t> in_slot(in_orb->n ? in_orb->n : 1),
                           out_slot(out_orb->n ? out_orb->n : 1);
            const uint64_t to = get_state(ev.output_state).canonical_hash;
            if (!qc_frame_slots(from, ev.input_state, in_orb, in_slot.data()) ||
                !qc_frame_slots(to, ev.output_state, out_orb, out_slot.data())) {
                qc_align_fail_.fetch_add(1, std::memory_order_relaxed);
                return;                          // cannot align; drop rather than mix frames
            }
            auto in_slot_of  = [&](EdgeId x) { const uint32_t i = in_orb->index_of(x);
                                               return i < in_orb->n ? in_slot[i] : 0u; };
            auto out_slot_of = [&](EdgeId x) { const uint32_t i = out_orb->index_of(x);
                                               return i < out_orb->n ? out_slot[i] : 0u; };

            cs = ev.num_consumed ? arena_.allocate_array<uint32_t>(ev.num_consumed) : nullptr;
            ps = nprod ? arena_.allocate_array<uint32_t>(nprod) : nullptr;
            for (uint8_t i = 0; i < ev.num_consumed; ++i) cs[i] = in_slot_of(ev.consumed_edges[i]);
            for (uint8_t i = 0; i < nprod; ++i) ps[i] = out_slot_of(ev.produced_edges[i]);

            // Survivors: output edges present in the input and not freshly produced.
            const SparseBitset& in_edges = get_state(ev.input_state).edges;
            SVec<std::pair<uint32_t,uint32_t>> surv;
            for (uint32_t i = 0; i < out_orb->n; ++i) {
                const EdgeId oe = out_orb->edges[i];
                bool produced_here = false;
                for (uint8_t j = 0; j < nprod; ++j)
                    if (ev.produced_edges[j] == oe) { produced_here = true; break; }
                if (produced_here || !in_edges.contains(oe)) continue;
                surv.push_back({in_slot_of(oe), out_slot[i]});
            }
            nsurv = static_cast<uint32_t>(surv.size());
            sfs = nsurv ? arena_.allocate_array<uint32_t>(nsurv) : nullptr;
            sts = nsurv ? arena_.allocate_array<uint32_t>(nsurv) : nullptr;
            for (uint32_t i = 0; i < nsurv; ++i) { sfs[i] = surv[i].first; sts[i] = surv[i].second; }
        }
        worker_scratch().release(mk);
    }

    SlotMatch m;
    m.to_hash = get_state(ev.output_state).canonical_hash;
    m.id = qc_next_match_id_.fetch_add(1, std::memory_order_relaxed);
    m.rule = ev.rule_index;
    m.from_slots = in_orb->n; m.to_slots = out_orb->n;
    m.num_consumed = ev.num_consumed; m.num_produced = nprod; m.num_survivors = nsurv;
    m.consumed_slots = cs; m.produced_slots = ps;
    m.surv_from_slot = sfs; m.surv_to_slot = sts;

    LockFreeList<SlotMatch>* lst;
    auto r = qc_expansion_.lookup(from);
    if (r.has_value()) lst = *r;
    else {
        auto* nl = arena_.template create<LockFreeList<SlotMatch>>();
        auto ins = qc_expansion_.insert_if_absent(from, nl);
        lst = ins.second ? nl : ins.first;
    }
    lst->push(m, arena_);

    // Match side of the rendezvous: replay this newly-captured match against every instance
    // already standing at this state, at every depth. Publish (the push above) before the
    // scan; pairs with the fence in qc_add_instance so a concurrent instance and match cannot
    // both miss each other. The per-pair claim in qc_apply makes the overlap harmless.
    if (!quotient_reconstruction_.load(std::memory_order_relaxed)) return;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const int maxs = qc_max_steps_.load(std::memory_order_relaxed);
    for (int d = 0; d < maxs; ++d) {
        auto ri = qc_instances_.lookup(qc_key(from, static_cast<uint32_t>(d), 0));
        if (!ri.has_value()) continue;
        (*ri)->for_each([&](const QcInstance& inst) {
            qc_apply(inst, m, from, static_cast<uint32_t>(d));
        });
    }
}

void Hypergraph::register_quotient_transition(EventId e) {
    hgcommon::PhaseTimer _pt(hgcommon::Phase::Quotient);
    qc_capture_expansion(e);
    const Event& ev = get_event(e);
    const EdgeOrbitTable* in_orb = state_orbits(ev.input_state);
    const EdgeOrbitTable* out_orb = state_orbits(ev.output_state);
    if (!in_orb || !out_orb) return;
    const uint64_t from = get_state(ev.input_state).canonical_hash;
    const uint64_t to   = get_state(ev.output_state).canonical_hash;

    auto mk = worker_scratch().mark();
    SVec<uint32_t> consumed, produced;
    for (uint8_t i = 0; i < ev.num_consumed; ++i) consumed.push_back(in_orb->orbit_of(ev.consumed_edges[i]));
    std::sort(consumed.begin(), consumed.end());
    for (uint8_t i = 0; i < ev.num_produced; ++i) produced.push_back(out_orb->orbit_of(ev.produced_edges[i]));
    std::sort(produced.begin(), produced.end());

    // Survivors: output edges that also live in the input state and were not freshly
    // produced -- they passed through, carrying their producer forward. Packed as
    // (orbit in `from` << 32 | orbit in `to`) and sorted, which is the order and the layout
    // hgcommon::qc_transition_sig reads them in.
    const SparseBitset& in_edges = get_state(ev.input_state).edges;
    SVec<uint64_t> survivors;
    for (uint32_t i = 0; i < out_orb->n; ++i) {
        const EdgeId oe = out_orb->edges[i];
        bool produced_here = false;
        for (uint8_t j = 0; j < ev.num_produced; ++j) if (ev.produced_edges[j] == oe) { produced_here = true; break; }
        if (produced_here) continue;
        if (in_edges.contains(oe))
            survivors.push_back((static_cast<uint64_t>(in_orb->orbit_of(oe)) << 32) |
                                out_orb->orbit[i]);
    }
    std::sort(survivors.begin(), survivors.end());

    const uint64_t sig = hgcommon::qc_transition_sig(
        from, to, ev.rule_index,
        consumed.data(), static_cast<uint32_t>(consumed.size()),
        survivors.data(), static_cast<uint32_t>(survivors.size()));

    if (!seen_transitions_.insert(sig)) { worker_scratch().release(mk); return; }  // already captured

    // Copy the orbit arrays into the persistent arena, then publish the transition.
    auto copy = [&](const SVec<uint32_t>& src) -> const uint32_t* {
        if (src.empty()) return nullptr;
        uint32_t* a = arena_.allocate_array<uint32_t>(src.size());
        for (size_t i = 0; i < src.size(); ++i) a[i] = src[i];
        return a;
    };
    const uint32_t nsurv = static_cast<uint32_t>(survivors.size());
    uint32_t* sf = nsurv ? arena_.allocate_array<uint32_t>(nsurv) : nullptr;
    uint32_t* st = nsurv ? arena_.allocate_array<uint32_t>(nsurv) : nullptr;
    for (uint32_t i = 0; i < nsurv; ++i) {
        sf[i] = static_cast<uint32_t>(survivors[i] >> 32);
        st[i] = static_cast<uint32_t>(survivors[i] & 0xFFFFFFFFu);
    }


    CanonicalTransition* t = arena_.template create<CanonicalTransition>();
    t->to_hash = to; t->sig = sig; t->canon_event = get_canonical_event(e); t->rule = ev.rule_index;
    t->num_consumed = static_cast<uint32_t>(consumed.size());
    t->num_produced = static_cast<uint32_t>(produced.size());
    t->num_survivors = nsurv;
    t->consumed_orbits = copy(consumed);
    t->produced_orbits = copy(produced);
    t->surv_from_orbits = sf; t->surv_to_orbits = st;
    worker_scratch().release(mk);

    LockFreeList<CanonicalTransition>* lst;
    auto r = transitions_from_.lookup(from);
    if (r.has_value()) lst = *r;
    else {
        auto* nl = arena_.template create<LockFreeList<CanonicalTransition>>();
        auto ins = transitions_from_.insert_if_absent(from, nl);
        lst = ins.second ? nl : ins.first;
    }
    lst->push(*t, arena_);

    // Drive the reconstruction: apply this newly-discovered transition at every depth its
    // source state is already reachable at. seq_cst fence pairs with qc_reach's fence so a
    // concurrent "reach (from,d)" and "register t from `from`" cannot both miss each other
    // (whichever publishes second sees the other and processes the (t, d) pair).
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const int maxs = qc_max_steps_.load(std::memory_order_relaxed);
    for (int d = 0; d <= maxs; ++d)
        if (qc_reached_.contains(qc_rkey(from, static_cast<uint32_t>(d))))
            qc_process_transition(*t, from, static_cast<uint32_t>(d));
}

void Hypergraph::causal_edge_keys(StateId state, const EdgeId* edges, uint32_t n,
                                  CanonicalEdgeKey* out) const {
    // Without quotient, or when orbits are unavailable (WL / non-Full canonicalization),
    // the key is the raw EdgeId: isomorphic-but-distinct raw states must keep disjoint
    // causal edges, and there is no automorphism collapse to account for. A 32-bit EdgeId
    // is always below the storage map's reserved high sentinel band, so no adjustment.
    auto raw_key = [](EdgeId e) { return CanonicalEdgeKey{static_cast<uint64_t>(e)}; };
    const bool full = state_canonicalization_mode_.load(std::memory_order_acquire)
                      == StateCanonicalizationMode::Full;
    if (!quotient_causal_.load(std::memory_order_relaxed) || !full) {
        for (uint32_t i = 0; i < n; ++i) out[i] = raw_key(edges[i]);
        return;
    }

    // Quotient + Full: key by iso-invariant canonical edge orbit. Extract the state's
    // edges once, compute the canonical hash and per-edge orbit ids (numbered canonically,
    // so the numbering itself is invariant), then key each queried edge by
    // fnv(canonical_hash, orbit). All scratch lives in the per-worker arena.
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> edge_vecs;
    SVec<EdgeId> ids;
    std::atomic_thread_fence(std::memory_order_acquire);
    get_state_edges(state).for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        edge_vecs.emplace_back(e.vertices, e.vertices + e.arity);
        ids.push_back(eid);
    });

    if (edge_vecs.empty()) {
        worker_scratch().release(mk);
        for (uint32_t i = 0; i < n; ++i) out[i] = raw_key(edges[i]);
        return;
    }

    thread_local std::vector<uint32_t> orbit, klass;
    const uint64_t chash = ir_hash_and_orbits(edge_vecs, orbit, klass);

    for (uint32_t i = 0; i < n; ++i) {
        // Find the queried edge's orbit via the parallel id array (edge counts are small).
        uint32_t orb = 0;
        bool found = false;
        for (size_t k = 0; k < ids.size(); ++k) {
            if (ids[k] == edges[i]) { orb = orbit[k]; found = true; break; }
        }
        uint64_t key = 14695981039346656037ULL;
        key ^= chash;                    key *= 1099511628211ULL;
        // A queried edge always belongs to `state`; the guarded fallback keeps a stray
        // edge from silently colliding on orbit 0 rather than crashing.
        key ^= found ? static_cast<uint64_t>(orb)
                     : (0xFFFFFFFF00000000ULL | edges[i]);
        key *= 1099511628211ULL;
        // Clear the top bit so the key lands in [0, 2^63), below the storage map's reserved
        // sentinel band -- costs one hash bit, still ample for collision resistance.
        key &= ~(1ULL << 63);
        out[i] = CanonicalEdgeKey{key};
    }
    worker_scratch().release(mk);
}

EdgeCorrespondence Hypergraph::find_edge_correspondence_dispatch(
    const SparseBitset& state1_edges,
    const SparseBitset& state2_edges
) const {
    EdgeVertexAccessorRaw vert_acc(this);
    EdgeArityAccessorRaw arity_acc(this);

    if (is_full_canonicalization()) {
        // Materialize both states' edges into the per-worker scratch arena (no heap),
        // reclaimed after the result (which uses the persistent global arena) is built.
        auto mk = worker_scratch().mark();
        auto extract_edges = [&](const SparseBitset& state_edges,
                                 SVec<SVec<VertexId>>& edge_vecs, SVec<EdgeId>& edge_ids) {
            state_edges.for_each([&](EdgeId eid) {
                const Edge& e = edges_[eid];
                edge_vecs.emplace_back(e.vertices, e.vertices + e.arity);
                edge_ids.push_back(eid);
            });
        };

        SVec<SVec<VertexId>> vecs1, vecs2;
        SVec<EdgeId> ids1, ids2;
        extract_edges(state1_edges, vecs1, ids1);
        extract_edges(state2_edges, vecs2, ids2);

        IRCanonicalizer ir;
        auto r1 = ir.canonicalize_edges(vecs1);
        auto r2 = ir.canonicalize_edges(vecs2);

        if (r1.canonical_form != r2.canonical_form) {
            worker_scratch().release(mk);
            return EdgeCorrespondence{};
        }

        EdgeCorrespondence result;
        result.count = static_cast<uint32_t>(ids1.size());
        result.state1_edges = const_cast<ConcurrentHeterogeneousArena&>(arena_).allocate_array<EdgeId>(result.count);
        result.state2_edges = const_cast<ConcurrentHeterogeneousArena&>(arena_).allocate_array<EdgeId>(result.count);

        for (uint32_t ci = 0; ci < result.count; ++ci) {
            size_t orig1 = r1.vertex_mapping.canonical_edge_to_original[ci];
            size_t orig2 = r2.vertex_mapping.canonical_edge_to_original[ci];
            result.state1_edges[ci] = ids1[orig1];
            result.state2_edges[ci] = ids2[orig2];
        }
        result.valid = true;
        worker_scratch().release(mk);
        return result;
    }

    if (wl_hash_) {
        return wl_hash_->find_edge_correspondence(state1_edges, state2_edges, vert_acc, arity_acc);
    }
    return EdgeCorrespondence{};
}

// =============================================================================
// Reconstruction observables and replay diagnostics
// =============================================================================
// Read by tests, by the determinism fingerprint and by the FFI's count path -- never from a
// matching or hashing loop -- so these live here rather than in the header.

void Hypergraph::set_quotient_reconstruction(bool on) {
    quotient_reconstruction_.store(on, std::memory_order_relaxed);
}

bool Hypergraph::quotient_reconstruction() const {
    return quotient_reconstruction_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_reconstructed_events() const {
    // Under an event-identity mode the observable is the count of distinct identities; with no
    // identity selected every application is its own event and the raw count IS the answer.
    // Mirrors num_events() on the full-capture side.
    if (event_signature_keys() == hgcommon::EVENT_SIG_NONE)
        return qc_next_raw_event_.load(std::memory_order_relaxed);
    return qc_num_canon_events_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_reconstructed_raw_events() const {
    return qc_next_raw_event_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_reconstructed_instances() const {
    return qc_next_instance_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_reconstructed_causal_edges() const {
    return qc_num_causal_edges_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_reconstructed_causal_pairs(bool transitively_reduced) const {
    if (transitively_reduced) {
        size_t n = 0;
        for_each_reconstructed_causal_as(
            /*reduced=*/true, [](uint32_t e) { return e; },
            [&](uint64_t, uint64_t) { ++n; });
        return n;
    }
    return qc_causal_pairs_.count_enumerated();
}

size_t Hypergraph::applied_scans() const {
    return qc_applied_scans_.load(std::memory_order_relaxed);
}

size_t Hypergraph::applied_claims() const { return qc_applied_.size(); }

std::vector<uint32_t> Hypergraph::applied_shape() const {
    std::vector<uint32_t> lens;
    const uint32_t n = qc_inst_applied_.size();
    lens.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const LockFreeList<QcAppliedMatch>* lst = qc_inst_applied_.get(i);
        if (!lst) continue;
        uint32_t c = 0;
        lst->for_each([&](const QcAppliedMatch&) { ++c; });
        if (c) lens.push_back(c);
    }
    std::sort(lens.begin(), lens.end());
    return lens;
}

uint64_t Hypergraph::applied_shape_fingerprint() const {
    // hgcommon::FNV_OFFSET, not a retyped literal: the digit-dropped 1469598103934665603 sat
    // here (17 digits against the basis's 20), so this fold started from a value that is not
    // the FNV-1a basis while two other folds in this file used the real one.
    uint64_t h = hgcommon::FNV_OFFSET;
    for (uint32_t v : applied_shape()) { h ^= v; h *= 1099511628211ULL; }
    return h;
}

size_t Hypergraph::capture_dropped_no_orbits() const {
    return qc_capture_no_orbits_.load(std::memory_order_relaxed);
}

size_t Hypergraph::capture_skipped_not_representative() const {
    return qc_capture_not_rep_.load(std::memory_order_relaxed);
}

size_t Hypergraph::applied_visits() const {
    return qc_applied_visits_.load(std::memory_order_relaxed);
}

size_t Hypergraph::captured_matches() const {
    return qc_next_match_id_.load(std::memory_order_relaxed);
}

size_t Hypergraph::reconstruction_instances() const {
    return qc_next_instance_.load(std::memory_order_relaxed);
}

size_t Hypergraph::applied_unique() const {
    return qc_applied_.count_enumerated();
}

// Simple hash of a state's edge SET -- fast, and not isomorphism-invariant. Its neighbours
// (compute_content_ordered_hash, compute_canonical_hash, compute_exact_canonical_hash) were
// already defined here; this was the outlier left in the header.
uint64_t Hypergraph::compute_state_hash(const SparseBitset& edges) {
    uint64_t h = 14695981039346656037ULL;
    edges.for_each([&](EdgeId eid) {
        h ^= eid;
        h *= 1099511628211ULL;
    });
    return h;
}

// The schedule-stable content triple of ONE reconstructed event: hash(input class, output class,
// rule). 0 when the event has no recorded triple.
uint64_t Hypergraph::reconstructed_raw_triple(uint32_t e) const {
    const QcEventContent* c = qc_event_sig_.get(e);
    return c ? c->triple_hash() : 0;
}

// THE HOTTEST ACCESSORS IN THE ENGINE: the matcher and the WL hash read edges through these on
// every candidate. They are here rather than in the class to test the premise of this work --
// with link-time optimisation the linker still inlines them, so where a body lives stops being a
// performance decision. The instruction count beside this commit is that test.
const Edge& Hypergraph::get_edge(EdgeId eid) const { return edges_[eid]; }
Edge& Hypergraph::get_edge(EdgeId eid) { return edges_[eid]; }
const VertexId* Hypergraph::edge_vertices(EdgeId eid) const { return edges_[eid].vertices; }
uint8_t Hypergraph::edge_arity(EdgeId eid) const { return edges_[eid].arity; }

// =============================================================================
// Vertex, edge and state accessors
// =============================================================================

VertexId Hypergraph::alloc_vertex() { return counters_.alloc_vertex(); }

VertexId Hypergraph::alloc_vertices(uint32_t count) {
    return counters_.next_vertex.fetch_add(count, std::memory_order_relaxed);
}

uint32_t Hypergraph::num_vertices() const {
    return counters_.next_vertex.load(std::memory_order_relaxed);
}

uint32_t Hypergraph::num_edges() const {
    return counters_.next_edge.load(std::memory_order_relaxed);
}

// PUBLISHED edges, the bound for enumeration. See num_published_states for why the claim counter
// is not that bound.
uint32_t Hypergraph::num_published_edges() const { return edges_.size(); }

const EdgeSignature& Hypergraph::edge_signature(EdgeId eid) const { return edge_signatures_[eid]; }

Hypergraph::EdgeVertexAccessorRaw Hypergraph::edge_vertex_accessor_raw() const {
    return EdgeVertexAccessorRaw(this);
}

Hypergraph::EdgeArityAccessorRaw Hypergraph::edge_arity_accessor_raw() const {
    return EdgeArityAccessorRaw(this);
}

// The acquire fence pairs with the release fence in create_state: it is what makes every field
// the creating thread wrote visible to this reader.
const State& Hypergraph::get_state(StateId sid) const {
    std::atomic_thread_fence(std::memory_order_acquire);
    return states_[sid];
}

State& Hypergraph::get_state(StateId sid) {
    std::atomic_thread_fence(std::memory_order_acquire);
    return states_[sid];
}

const SparseBitset& Hypergraph::get_state_edges(StateId sid) const {
    std::atomic_thread_fence(std::memory_order_acquire);
    return states_[sid].edges;
}

// The same hash evolution deduplicates on in Automatic mode, so display and evolution agree.
uint64_t Hypergraph::get_state_content_hash(StateId sid) const {
    std::atomic_thread_fence(std::memory_order_acquire);
    return compute_content_ordered_hash(states_[sid].edges);
}

uint32_t Hypergraph::num_states() const {
    return counters_.next_state.load(std::memory_order_relaxed);
}

// The bound for ENUMERATING states, as against num_states(), which is the claim counter and runs
// ahead of what exists.
uint32_t Hypergraph::num_published_states() const { return states_.size(); }

// INVALID_ID until a genesis state is published, and no state id equals INVALID_ID, so the
// comparison alone answers both questions.
bool Hypergraph::is_genesis_state(StateId sid) const {
    return sid == genesis_state_.load(std::memory_order_acquire);
}

bool Hypergraph::is_genesis_event(EventId eid) const {
    const StateId genesis = genesis_state_.load(std::memory_order_acquire);
    if (genesis == INVALID_ID) return false;
    if (eid >= events_.size()) return false;
    return events_[eid].input_state == genesis;
}

StateId Hypergraph::genesis_state() const {
    return genesis_state_.load(std::memory_order_acquire);
}

std::optional<StateId> Hypergraph::find_canonical_state(uint64_t canonical_hash) const {
    return canonical_state_map_.lookup_waiting(canonical_hash);
}

// None: the raw state IS the answer. Automatic/Full: the cached canonical_id, acquired -- the
// load carries the edge released by create_or_get_canonical_state, which matters on ARM64.
StateId Hypergraph::get_canonical_state(StateId raw_state) const {
    if (raw_state == INVALID_ID) return INVALID_ID;
    if (state_canonicalization_mode_.load(std::memory_order_acquire) ==
        StateCanonicalizationMode::None) {
        return raw_state;
    }
    const State& state = get_state(raw_state);
    return hgcommon::atomic_ref<StateId>(const_cast<StateId&>(state.canonical_id))
        .load(std::memory_order_acquire);
}

// Non-zero means at least one event identity is approximate rather than canonical.
uint64_t Hypergraph::event_signature_raw_fallbacks() const {
    return event_sig_raw_fallbacks_.load(std::memory_order_relaxed);
}

uint64_t Hypergraph::canonical_hash_computations() const {
    return canonical_hash_computations_.load(std::memory_order_relaxed);
}

// =============================================================================
// Event accessors, identity settings and index access
// =============================================================================

const Event& Hypergraph::get_event(EventId eid) const { return events_[eid]; }
Event& Hypergraph::get_event(EventId eid) { return events_[eid]; }

// The CANONICAL count once an event identity is selected, the raw count otherwise. The acquire
// synchronises with the release stores in alloc_event.
uint32_t Hypergraph::num_events() const {
    if (event_signature_keys_ != EVENT_SIG_NONE) {
        return canonical_event_count_.load(std::memory_order_acquire);
    }
    return counters_.next_event.load(std::memory_order_acquire);
}

uint32_t Hypergraph::num_raw_events() const {
    return counters_.next_event.load(std::memory_order_acquire);
}

// PUBLISHED events, the bound for enumeration. See num_published_states for why the claim
// counter is not that bound.
uint32_t Hypergraph::num_published_events() const { return events_.size(); }

bool Hypergraph::is_event_canonical(EventId eid) const {
    if (eid >= num_raw_events()) return false;
    return events_[eid].is_canonical();
}

EventId Hypergraph::get_canonical_event(EventId eid) const {
    if (eid >= num_raw_events()) return INVALID_ID;
    const Event& event = events_[eid];
    return event.is_canonical() ? eid : event.canonical_event_id;
}

void Hypergraph::set_event_signature_keys(EventSignatureKeys keys) {
    event_signature_keys_ = keys;
}

EventSignatureKeys Hypergraph::event_signature_keys() const { return event_signature_keys_; }

void Hypergraph::set_positional_event_identity(bool on) {
    positional_event_identity_.store(on, std::memory_order_relaxed);
}

bool Hypergraph::positional_event_identity() const {
    return positional_event_identity_.load(std::memory_order_relaxed);
}

const SignatureIndex& Hypergraph::signature_index() const {
    return match_index_.signature_index();
}

const InvertedVertexIndex& Hypergraph::inverted_index() const {
    return match_index_.inverted_index();
}

const PatternMatchingIndex& Hypergraph::match_index() const { return match_index_; }

CausalGraph& Hypergraph::causal_graph() { return causal_graph_; }
const CausalGraph& Hypergraph::causal_graph() const { return causal_graph_; }

void Hypergraph::set_edge_producer(CanonicalEdgeKey key, EventId producer, EdgeId raw_edge) {
    causal_graph_.set_edge_producer(key, producer, raw_edge);
}

// The cached edge-orbit table for a state, or null when there is none -- full-capture mode, or
// before canonicalization. The +1 keeps the key off the map's EMPTY sentinel.
const EdgeOrbitTable* Hypergraph::state_orbits(StateId s) const {
    auto r = state_orbit_tables_.lookup(static_cast<uint64_t>(s) + 1);
    return r.has_value() ? *r : nullptr;
}

// =============================================================================
// Observables (SPEC section 5)
// =============================================================================
// The engine reaches the same observable two ways: full capture explores every raw state,
// quotient explores one per isomorphism class and reconstructs the rest. These hide that choice.
// Deliberately NOT the num_events()/causal_graph() accessors, which report what is MATERIALISED
// -- internal code iterates records by id against those and would break if they reported counts
// with no records behind them.

size_t Hypergraph::observable_num_events() const {
    return quotient_reconstruction() ? num_reconstructed_events() : num_events();
}

size_t Hypergraph::observable_num_causal_edges() const {
    return quotient_reconstruction() ? num_reconstructed_causal_edges()
                                     : causal_graph_.num_causal_edges();
}

size_t Hypergraph::observable_num_causal_pairs(bool transitively_reduced) const {
    return quotient_reconstruction() ? num_reconstructed_causal_pairs(transitively_reduced)
                                     : causal_graph_.num_causal_event_pairs();
}

size_t Hypergraph::observable_num_branchial() const {
    return quotient_reconstruction() ? num_reconstructed_branchial()
                                     : causal_graph_.num_branchial_edges();
}

EventId Hypergraph::get_edge_producer(CanonicalEdgeKey key) const {
    return causal_graph_.get_edge_producer(key);
}

void Hypergraph::add_edge_consumer(CanonicalEdgeKey key, EventId consumer, EdgeId raw_edge) {
    causal_graph_.add_edge_consumer(key, consumer, raw_edge);
}

void Hypergraph::propagate_producers(CanonicalEdgeKey from, CanonicalEdgeKey to,
                                     EdgeId raw_edge) {
    causal_graph_.propagate_producers(from, to, raw_edge);
}

void Hypergraph::set_quotient_causal(bool q) {
    quotient_causal_.store(q, std::memory_order_relaxed);
}

bool Hypergraph::quotient_causal() const {
    return quotient_causal_.load(std::memory_order_relaxed);
}

// Set before evolving and read by the workers, so both components are atomics like every other
// pre-evolution switch.
void Hypergraph::set_record_set(RecordSet r) {
    record_causal_.store(r.causal, std::memory_order_relaxed);
    record_branchial_.store(r.branchial, std::memory_order_relaxed);
    record_state_events_.store(r.state_events, std::memory_order_relaxed);
    record_raw_events_.store(r.raw_events, std::memory_order_relaxed);
}

RecordSet Hypergraph::record_set() const {
    return RecordSet{record_causal_.load(std::memory_order_relaxed),
                     record_branchial_.load(std::memory_order_relaxed),
                     record_state_events_.load(std::memory_order_relaxed),
                     record_raw_events_.load(std::memory_order_relaxed)};
}

// The per-state event list and the branchial pair relation are recorded independently: they feed
// different outputs, so a run that needs one need not build the other.
void Hypergraph::record_state_event(EventId event, StateId input_state) {
    causal_graph_.record_state_event(event, input_state);
}

void Hypergraph::record_branchial_overlaps(EventId event, StateId input_state,
                                           const EdgeId* consumed_edges, uint8_t num_consumed) {
    causal_graph_.record_branchial_overlaps(event, input_state, consumed_edges, num_consumed);
}

size_t Hypergraph::num_causal_edges() const { return causal_graph_.num_causal_edges(); }
size_t Hypergraph::num_causal_event_pairs() const { return causal_graph_.num_causal_event_pairs(); }
size_t Hypergraph::num_branchial_edges() const { return causal_graph_.num_branchial_edges(); }

ConcurrentHeterogeneousArena& Hypergraph::arena() { return arena_; }
const ConcurrentHeterogeneousArena& Hypergraph::arena() const { return arena_; }

GlobalCounters& Hypergraph::counters() { return counters_; }
const GlobalCounters& Hypergraph::counters() const { return counters_; }

// =============================================================================
// QcCtx / QrCtx -- the storage face the shared quotient cores drive
// =============================================================================
// WHERE a producer vector, an applied list or a claim set lives is here; what an application
// DOES is in hgcommon, which is the body the device runs too. Both cores are instantiated in
// this translation unit and nowhere else, which is what lets these bodies live here.

uint32_t Hypergraph::QcCtx::max_steps() const { return steps; }

bool Hypergraph::QcCtx::enter(uint32_t) const { return true; }

void Hypergraph::QcCtx::defer_reach(uint64_t state_hash, uint32_t depth) {
    hgcommon::qc_reach(*this, state_hash, depth);
}

void Hypergraph::QcCtx::defer_producer(uint64_t state_hash, uint32_t depth, uint32_t orbit,
                                       uint32_t producer) {
    hgcommon::qc_add_producer(*this, state_hash, depth, orbit, producer);
}

bool Hypergraph::QcCtx::mark_reached(uint64_t rkey, uint64_t state_hash, uint32_t depth) {
    if (!hg.qc_reached_.insert(rkey)) return false;
    // Recorded so raise_quotient_max_steps can re-drive the depths the old bound made terminal;
    // nothing else reads the list.
    hg.qc_reached_list_.push(QcReachPoint{state_hash, depth}, hg.arena_);
    return true;
}

bool Hypergraph::QcCtx::mark_producer_seen(uint64_t seen_key) {
    return hg.qc_dsup_seen_.insert(seen_key);
}

void Hypergraph::QcCtx::push_producer(uint64_t key, uint32_t producer) {
    hg.qc_dsup_list(key)->push(producer, hg.arena_);
}

void Hypergraph::QcCtx::emit(uint32_t producer, uint32_t consumer) {
    hg.qc_emit(producer, consumer);
}

void Hypergraph::QcCtx::fence() { std::atomic_thread_fence(std::memory_order_seq_cst); }

Hypergraph::QcCtx Hypergraph::qc_ctx() {
    return QcCtx{*this, static_cast<uint32_t>(qc_max_steps_.load(std::memory_order_relaxed))};
}

bool Hypergraph::QrCtx::claim(uint64_t apply_key) { return hg.qc_applied_.insert(apply_key); }

uint32_t Hypergraph::QrCtx::mint_event() {
    return hg.qc_next_raw_event_.fetch_add(1, std::memory_order_relaxed);
}

void Hypergraph::QrCtx::record_content(uint32_t ev, uint64_t from_class, uint64_t to_class,
                                       uint32_t rule) {
    hg.qc_event_sig_.emplace_at(ev, hg.arena_, QcEventContent{from_class, to_class, rule});
}

hgcommon::EventSignatureKeys Hypergraph::QrCtx::keys() const {
    return hg.event_signature_keys();
}

uint32_t Hypergraph::QrCtx::frame_step(uint64_t class_hash, uint32_t fallback) const {
    if (auto fo = hg.qc_frame_.lookup(class_hash))
        return hg.get_state(static_cast<StateId>(*fo - 1)).step;
    return fallback;
}

void Hypergraph::QrCtx::record_runsig(uint32_t ev, uint64_t csig) {
    hg.qc_event_runsig_.emplace_at(ev, hg.arena_, csig);
    if (hg.qc_canon_event_seen_.insert(csig))
        hg.qc_num_canon_events_.fetch_add(1, std::memory_order_relaxed);
}

bool Hypergraph::QrCtx::want_causal() const    { return hg.record_set().causal; }
bool Hypergraph::QrCtx::want_branchial() const { return hg.record_set().branchial; }

uint32_t Hypergraph::QrCtx::producer_at(const QcInstance& inst, uint32_t slot) const {
    return inst.prod[slot];
}

void Hypergraph::QrCtx::record_causal(uint32_t producer, uint32_t consumer) {
    hg.qc_record_causal(producer, consumer);
}

bool Hypergraph::QrCtx::applied_ref_valid(AppliedRef r) { return r != nullptr; }

Hypergraph::QrCtx::AppliedRef Hypergraph::QrCtx::publish_applied(const QcInstance& inst,
                                                                 const SlotMatch& m,
                                                                 uint32_t ev) {
    auto& applied = hg.qc_inst_applied_.get_or_default(inst.id, hg.arena_);
    return applied.push(QcAppliedMatch{m.id, ev, m.num_consumed, m.consumed_slots}, hg.arena_);
}

void Hypergraph::QrCtx::record_branchial_pair(uint32_t lo, uint32_t hi) {
    (void)lo; (void)hi;
    hg.qc_num_branchial_.fetch_add(1, std::memory_order_relaxed);
}

// The child instance: survivors carry their producer across, produced slots take THIS event.
void Hypergraph::QrCtx::descend(const SlotMatch& m, uint32_t depth, uint32_t ev,
                                const QcInstance& parent) {
    uint32_t* cp = hg.arena_.allocate_array<uint32_t>(m.to_slots ? m.to_slots : 1);
    for (uint32_t i = 0; i < m.to_slots; ++i) cp[i] = hgcommon::QR_NO_PRODUCER;
    for (uint32_t i = 0; i < m.num_survivors; ++i) {
        const uint32_t a = m.surv_from(i), b = m.surv_to(i);
        if (a < parent.nslots && b < m.to_slots) cp[b] = parent.prod[a];
    }
    for (uint32_t i = 0; i < m.num_produced; ++i) {
        const uint32_t s = m.produced(i);
        if (s < m.to_slots) cp[s] = ev;
    }
    hg.qc_add_instance(m.to_hash, depth + 1, cp, m.to_slots);
}

// count_unique rather than size: ConcurrentMap can hold duplicate keys when two threads insert
// the same canonical hash, and the unique count is the answer once evolution is complete.
size_t Hypergraph::num_canonical_states() const { return canonical_state_map_.count_unique(); }

// Release/acquire: the mode is set on the main thread and read by workers, which matters on a
// weak model like ARM64.
void Hypergraph::set_state_canonicalization_mode(StateCanonicalizationMode mode) {
    state_canonicalization_mode_.store(mode, std::memory_order_release);
}

StateCanonicalizationMode Hypergraph::state_canonicalization_mode() const {
    return state_canonicalization_mode_.load(std::memory_order_acquire);
}

void Hypergraph::enable_wl_hash()  { use_wl_hash_ = true; }
void Hypergraph::disable_wl_hash() { use_wl_hash_ = false; }
bool Hypergraph::wl_hash_enabled() const { return use_wl_hash_; }

bool Hypergraph::is_full_canonicalization() const {
    return state_canonicalization_mode_.load(std::memory_order_acquire) ==
           StateCanonicalizationMode::Full;
}

size_t Hypergraph::num_reconstructed_branchial() const {
    return qc_num_branchial_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_frame_alignment_disagreements() const {
    return qc_frame_disagree_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_alignment_failures() const {
    return qc_align_fail_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_bad_correspondences() const {
    return qc_align_badcorr_.load(std::memory_order_relaxed);
}

// The state whose labelling defines a canonical class -- the class FRAME. INVALID_ID when the
// class has no frame, which happens for a class no captured transition touched.
StateId Hypergraph::class_frame_state(uint64_t class_hash) const {
    auto r = qc_frame_.lookup(class_hash);
    return r.has_value() ? static_cast<StateId>(*r - 1) : INVALID_ID;
}

// Falls back to the internal (input, output, rule) triple when no identity mode is selected:
// full capture leaves Event::signature at 0 there, so neither value is comparable and the
// internal one at least distinguishes events.
uint64_t Hypergraph::event_pair_signature(uint32_t e) const {
    if (event_signature_keys() != hgcommon::EVENT_SIG_NONE) {
        const uint64_t* r = qc_event_runsig_.get(e);
        if (r) return *r;
    }
    return reconstructed_raw_triple(e);
}

// The event's content itself, for a caller that must DESCRIBE the event rather than identify it.
const QcEventContent* Hypergraph::reconstructed_event_content(uint32_t e) const {
    return qc_event_sig_.get(e);
}

uint32_t Hypergraph::count_state_edges(StateId sid) const {
    uint32_t count = 0;
    states_[sid].edges.for_each([&](EdgeId) { count++; });
    return count;
}

// Route every map's table storage through the arena (no malloc, no per-map heap contention).
// The initialiser order follows member declaration order; arena_ is declared before these maps,
// so it is fully constructed by the time they take its address.
Hypergraph::Hypergraph()
    : canonical_state_map_(decltype(canonical_state_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
    , event_canonical_state_map_(
          decltype(event_canonical_state_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
    , wl_hash_(std::make_unique<WLHash>(&arena_))
    , canonical_event_map_(decltype(canonical_event_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
{
    causal_graph_.set_arena(&arena_);
}

// An ordered pair of event ids as one map key. Both ids are offset by one before packing, which
// makes the key injective and never zero -- ConcurrentMap reserves 0 as EMPTY.
uint64_t Hypergraph::qc_pair_key(uint32_t a, uint32_t b) { return id_key(a, b); }

// The DP's key spaces come from hgcommon so the device indexes the same ones.
uint64_t Hypergraph::qc_key(uint64_t state_hash, uint32_t depth, uint32_t orbit) {
    return hgcommon::qc_key(state_hash, depth, orbit);
}

uint64_t Hypergraph::qc_rkey(uint64_t state_hash, uint32_t depth) {
    return hgcommon::qc_rkey(state_hash, depth);
}

Hypergraph::EdgeVertexAccessorRaw::EdgeVertexAccessorRaw(const Hypergraph* hg) : hg_(hg) {}

const VertexId* Hypergraph::EdgeVertexAccessorRaw::operator[](EdgeId eid) const {
    return hg_->edges_[eid].vertices;
}

Hypergraph::EdgeArityAccessorRaw::EdgeArityAccessorRaw(const Hypergraph* hg) : hg_(hg) {}

uint8_t Hypergraph::EdgeArityAccessorRaw::operator[](EdgeId eid) const {
    return hg_->edges_[eid].arity;
}

uint32_t Hypergraph::QcAppliedMatch::consumed(uint32_t j) const {
    return consumed_slots[j];
}

}  // namespace engine
}  // namespace HG_NAMESPACE