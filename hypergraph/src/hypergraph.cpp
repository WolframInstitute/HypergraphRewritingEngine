// hypergraph.cpp - Implementation of Hypergraph class non-template methods

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hypergraph/atomic_compat.hpp"
#include <thread>

namespace hypergraph {

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
    // Lock-free initialization using CAS
    // States: 0=uninit, 1=in_progress, 2=done

    // Fast path: already created
    int state = genesis_state_init_.load(std::memory_order_acquire);
    if (state == 2) {
        return genesis_state_;
    }

    // Try to become the initializer (CAS 0 -> 1)
    int expected = 0;
    if (genesis_state_init_.compare_exchange_strong(expected, 1,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        // We are the initializer - create the genesis state
        try {
            SparseBitset empty_edges;
            genesis_state_ = create_state(std::move(empty_edges), 0, 0, INVALID_ID);
        } catch (...) {
            // Hand the claim back before propagating: a claim left at "in progress"
            // parks every other thread in the wait below with nothing left to finish it.
            genesis_state_init_.store(0, std::memory_order_release);
            throw;
        }
        genesis_state_init_.store(2, std::memory_order_release);
        return genesis_state_;
    }

    // Someone else is initializing or already done - spin until done
    while (genesis_state_init_.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();  // Allow other threads to progress
    }
    return genesis_state_;
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

    // WL canonical hash for None/Automatic modes.
    auto wl_child = [&]() -> uint64_t {
        return compute_canonical_hash(edges);
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
            canonical_hash = wl_child();
            break;
        case StateCanonicalizationMode::Automatic:
            map_key = compute_content_ordered_hash(edges);
            canonical_hash = wl_child();
            break;
        case StateCanonicalizationMode::Full:
        default:
            // In quotient mode compute the edge-orbit table and take the canonical hash
            // from the same IR canonicalization (the quotient causal reconstruction needs
            // the orbits; there is no extra canon pass). Otherwise just the dedup hash.
            if (quotient_causal_.load(std::memory_order_relaxed))
                canonical_hash = compute_and_cache_state_orbits(new_sid, edges);
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
    hg::atomic_ref<uint64_t>(states_[new_sid].canonical_hash)
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
    hg::atomic_ref<StateId>(states_[new_sid].canonical_id)
        .store(existing_or_new, std::memory_order_release);

    if (verified_duplicate) {
        return {existing_or_new, new_sid, false};
    }

    return {new_sid, new_sid, true};
}

bool Hypergraph::try_lower_explore_depth(StateId canonical_id, uint32_t depth) {
    if (canonical_id == INVALID_ID) return false;
    hg::atomic_ref<uint32_t> known(states_[canonical_id].explore_depth);
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
    hg::atomic_ref<uint32_t> flag(states_[canonical_id].expanded);
    uint32_t expected = 0;
    return flag.compare_exchange_strong(expected, 1,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
}

uint32_t Hypergraph::explore_depth_of(StateId canonical_id) const {
    if (canonical_id == INVALID_ID) return INVALID_ID;
    hg::atomic_ref<uint32_t> known(const_cast<uint32_t&>(states_[canonical_id].explore_depth));
    return known.load(std::memory_order_acquire);
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
    hg::atomic_ref<uint64_t> atomic_hash(state.canonical_hash);
    uint64_t cached = atomic_hash.load(std::memory_order_acquire);
    if (cached != 0) {
        return cached;
    }

    // Compute the canonical hash on-demand (mode-aware: exact IR in Full mode, WL otherwise)
    uint64_t hash = compute_canonical_hash(state.edges);

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

    // Event canonicalization: check if this event signature already exists
    if (event_signature_keys_ != EVENT_SIG_NONE) {
        const EventSignatureKeys keys = event_signature_keys_;

        // Get canonical state IDs for event canonicalization
        StateId canonical_input = get_canonical_state_for_event(input_state);
        StateId canonical_output = get_canonical_state_for_event(output_state);
        const State& canonical_out_state = get_state(canonical_output);

        uint64_t sig_key = FNV_OFFSET;

        // Add isomorphism-invariant state hashes to signature if requested
        if (keys & EventKey_InputState) {
            uint64_t input_hash = get_or_compute_canonical_hash(input_state);
            sig_key = fnv_hash(sig_key, input_hash);
        }
        if (keys & EventKey_OutputState) {
            uint64_t output_hash = get_or_compute_canonical_hash(output_state);
            sig_key = fnv_hash(sig_key, output_hash);
        }
        if (keys & EventKey_Step) {
            sig_key = fnv_hash(sig_key, static_cast<uint64_t>(canonical_out_state.step));
        }
        if (keys & EventKey_Rule) {
            sig_key = fnv_hash(sig_key, static_cast<uint64_t>(rule_index));
        }

        // Add edge signatures if requested
        if (keys & (EventKey_ConsumedEdges | EventKey_ProducedEdges)) {
            const State& in_state = get_state(input_state);
            const State& out_state = get_state(output_state);
            const State& canonical_in_state = get_state(canonical_input);

            // Compute edge correspondence using hash dispatch
            EdgeCorrespondence input_correspondence = find_edge_correspondence_dispatch(
                in_state.edges, canonical_in_state.edges);
            EdgeCorrespondence output_correspondence = find_edge_correspondence_dispatch(
                out_state.edges, canonical_out_state.edges);

            // Build edge mappings (worker-local scratch during event creation)
            SUMap<EdgeId, EdgeId> input_edge_map, output_edge_map;
            if (input_correspondence.valid) {
                for (uint32_t i = 0; i < input_correspondence.count; ++i) {
                    input_edge_map[input_correspondence.state1_edges[i]] =
                        input_correspondence.state2_edges[i];
                }
            }
            if (output_correspondence.valid) {
                for (uint32_t i = 0; i < output_correspondence.count; ++i) {
                    output_edge_map[output_correspondence.state1_edges[i]] =
                        output_correspondence.state2_edges[i];
                }
            }

            // Map edges to canonical equivalents and compute signatures
            if (keys & EventKey_ConsumedEdges) {
                for (uint8_t i = 0; i < num_consumed; ++i) {
                    auto it = input_edge_map.find(consumed[i]);
                    EdgeId canonical_edge = (it != input_edge_map.end()) ? it->second : consumed[i];
                    sig_key = fnv_hash(sig_key, static_cast<uint64_t>(canonical_edge));
                }
            }

            if (keys & EventKey_ProducedEdges) {
                for (uint8_t i = 0; i < num_produced; ++i) {
                    auto it = output_edge_map.find(produced[i]);
                    EdgeId canonical_edge = (it != output_edge_map.end()) ? it->second : produced[i];
                    sig_key = fnv_hash(sig_key, static_cast<uint64_t>(canonical_edge));
                }
            }
        }

        // Avoid key=0 (reserved as EMPTY_KEY in ConcurrentMap)
        if (sig_key == 0 || sig_key == FNV_OFFSET) sig_key = 1;

        // Try to insert this signature
        auto [existing_or_new, was_inserted] = canonical_event_map_.insert_if_absent_waiting(sig_key, eid);

        if (!was_inserted) {
            is_canonical = false;
            canonical_eid = existing_or_new;
        } else {
            canonical_event_count_.fetch_add(1, std::memory_order_relaxed);
        }
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

void Hypergraph::register_event_for_branchial(
    EventId event,
    StateId input_state,
    const EdgeId* consumed_edges,
    uint8_t num_consumed,
    EventId canonical_event
) {
    // Branchial relationships are exact-consumed-edge overlaps at a shared input
    // state, independent of event canonicalization: two events branch iff they
    // consumed a common edge. (Both former paths used exact edge equality and no
    // canonical skip, so they were identical.) The inverted-index registration below
    // handles this in O(co-consumers); canonical_event is unused for branchial.
    (void)canonical_event;
    causal_graph_.register_event_from_state_with_overlap_check(
        event, input_state, consumed_edges, num_consumed,
        [this](EventId eid, const EdgeId*& edges, uint8_t& num) {
            const Event& ev = events_[eid];
            edges = ev.consumed_edges;
            num = ev.num_consumed;
        }
    );
}

// =============================================================================
// Canonical Hash Computation
// =============================================================================

uint64_t Hypergraph::compute_content_ordered_hash(const SparseBitset& edges) const {
    uint64_t h = FNV_OFFSET;

    // Hash edge count first
    h = fnv_hash(h, mix64(edges.count()));

    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        h = fnv_hash(h, mix64(static_cast<uint64_t>(e.arity)));
        for (uint8_t i = 0; i < e.arity; ++i) {
            h = fnv_hash(h, mix64(static_cast<uint64_t>(e.vertices[i])));
        }
        h = fnv_hash(h, 0xDEADBEEFCAFEBABEULL);
    });

    return h;
}

uint64_t Hypergraph::compute_canonical_hash(const SparseBitset& edges) const {
    // Full mode uses the exact IR hash as the canonical identity (it is also the
    // dedup key), so there is no separate WL pass. Other modes use the fast WL hash.
    bool full = state_canonicalization_mode_.load(std::memory_order_acquire)
                == StateCanonicalizationMode::Full;
    if (!full && use_wl_hash_ && wl_hash_) {
        return compute_wl_hash(edges);
    }

    // Full mode, or WL disabled: exact canonical hash via IR (polynomial for low-symmetry graphs).
    // Materialize into the per-worker scratch arena (no heap), reclaimed after.
    auto mk = worker_scratch().mark();
    SVec<SVec<VertexId>> edge_vectors;

    std::atomic_thread_fence(std::memory_order_acquire);

    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        edge_vectors.emplace_back(e.vertices, e.vertices + e.arity);
    });

    if (edge_vectors.empty()) {
        worker_scratch().release(mk);
        return EMPTY_STATE_CANONICAL_HASH;
    }

    IRCanonicalizer ir;
    uint64_t h = ir.compute_canonical_hash(edge_vectors);
    worker_scratch().release(mk);
    return h;
}

uint64_t Hypergraph::compute_wl_hash(const SparseBitset& edges) const {
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

uint64_t Hypergraph::compute_and_cache_state_orbits(StateId s, const SparseBitset& edges) {
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
    uint64_t hash = 0;
    uint32_t num_orbits = 0;

    if (n > 0) {
        IRCanonicalizer ir;
        // Reused per worker rather than allocated per state: this runs once for every state
        // created under quotient, and the two vectors would otherwise be a heap round-trip
        // each time. The IR entry point takes std::vector, so the storage is kept alive here
        // instead of being handed to the scratch arena.
        thread_local std::vector<uint32_t> orbit, klass;
        hash = ir.compute_canonical_hash_with_edge_orbits(edge_vecs, orbit, &klass);
        // ids are already ascending (SparseBitset iterates in id order), orbit is parallel.
        for (uint32_t i = 0; i < n; ++i) {
            arr_edges[i] = ids[i];
            arr_orbit[i] = orbit[i];
            arr_class[i] = klass[i];
            if (orbit[i] + 1 > num_orbits) num_orbits = orbit[i] + 1;
        }
        // Slot = rank under (ORBIT, EdgeId). Orbit, not content class: an automorphism can
        // permute edges between content classes, so a class is defined only up to the Aut
        // action and two raw states of one canonical class can disagree about which class an
        // edge belongs to. Orbits are Aut-invariant, so the block structure is the same in
        // every instance. Order within an orbit is arbitrary (ids ascend, so a stable sort
        // fixes it), and that is harmless: the match set is closed under Aut, so permuting
        // within an orbit maps matches to matches and leaves the reconstructed causal multiset
        // unchanged -- which is exactly the property class-rank lacks.
        SVec<uint32_t> order;
        order.reserve(n);
        for (uint32_t i = 0; i < n; ++i) order.push_back(i);
        std::stable_sort(order.begin(), order.end(),
                         [&](uint32_t a, uint32_t b) { return arr_class[a] < arr_class[b]; });
        for (uint32_t r = 0; r < n; ++r) arr_slot[order[r]] = r;
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
    const int maxs = qc_max_steps_.load(std::memory_order_relaxed);
    if (static_cast<int>(depth) > maxs) return;
    const uint64_t key = qc_key(state_hash, depth, orbit);
    // Newly added producer for this (state, depth, orbit)?
    uint64_t seenk = key ^ (static_cast<uint64_t>(producer) + 0x9e3779b97f4a7c15ULL);
    seenk *= 1099511628211ULL; if (seenk == 0 || seenk == ~0ULL) seenk = 1;
    if (!qc_dsup_seen_.insert_if_absent(seenk, true).second) return;
    qc_dsup_list(key)->push(producer, arena_);

    // A producer landing at (state, depth) witnesses that (state, depth) is reachable, so
    // mark it and process its transitions once. Without this a producer arriving via the
    // survivor cascade would leave (state, depth) unreached, and a consuming transition
    // registered later would be skipped by the register trigger. Idempotent; bounded by depth.
    qc_reach(state_hash, depth);

    // A state is only *processed* (emits, propagates survivors) at depth < steps -- the DP
    // runs its transition loop for depths 0..steps-1, producing into depth steps but never
    // reading depth steps. Producers landing at the final depth are stored and dead.
    if (static_cast<int>(depth) >= maxs) return;

    // Rendezvous with transitions already known from this state: publish before scan.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for_each_transition_from(state_hash, [&](const CanonicalTransition& t) {
        for (uint32_t i = 0; i < t.num_consumed; ++i)
            if (t.consumed_orbits[i] == orbit) { qc_emit(producer, t.canon_event); break; }
        for (uint32_t i = 0; i < t.num_survivors; ++i)
            if (t.surv_from[i] == orbit)
                qc_add_producer(t.to_hash, depth + 1, t.surv_to[i], producer);
    });
}

void Hypergraph::qc_process_transition(const CanonicalTransition& t, uint64_t from_hash,
                                       uint32_t depth) {
    const int maxs = qc_max_steps_.load(std::memory_order_relaxed);
    if (static_cast<int>(depth) + 1 > maxs) return;
    qc_reach(t.to_hash, depth + 1);
    // Produced edges are produced by this canonical event, at the child depth.
    for (uint32_t i = 0; i < t.num_produced; ++i)
        qc_add_producer(t.to_hash, depth + 1, t.produced_orbits[i], t.canon_event);
    // Rendezvous with producers already present at (from, depth): publish (reach/produce
    // above) before this scan.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (uint32_t i = 0; i < t.num_consumed; ++i) {
        auto r = qc_dsup_.lookup(qc_key(from_hash, depth, t.consumed_orbits[i]));
        if (r.has_value()) (*r)->for_each([&](EventId p) { qc_emit(p, t.canon_event); });
    }
    for (uint32_t i = 0; i < t.num_survivors; ++i) {
        auto r = qc_dsup_.lookup(qc_key(from_hash, depth, t.surv_from[i]));
        if (r.has_value()) (*r)->for_each([&](EventId p) {
            qc_add_producer(t.to_hash, depth + 1, t.surv_to[i], p);
        });
    }
}

void Hypergraph::qc_reach(uint64_t state_hash, uint32_t depth) {
    const int maxs = qc_max_steps_.load(std::memory_order_relaxed);
    if (static_cast<int>(depth) > maxs) return;
    if (!qc_reached_.insert_if_absent(qc_rkey(state_hash, depth), true).second) return;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for_each_transition_from(state_hash, [&](const CanonicalTransition& t) {
        qc_process_transition(t, state_hash, depth);
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

bool Hypergraph::qc_reachable(uint32_t producer, uint32_t consumer) const {
    // Backward walk from `consumer` over the KEPT predecessors, looking for `producer`. Ids
    // increase along every causal edge, so anything below `producer` is out of the cone.
    if (producer >= consumer) return false;
    SVec<uint32_t> stack;
    SUSet<uint32_t> visited;
    stack.push_back(consumer);
    visited.insert(consumer);
    while (!stack.empty()) {
        const uint32_t x = stack.back();
        stack.pop_back();
        auto r = qc_preds_.lookup(static_cast<uint64_t>(x) + 1);
        if (!r.has_value()) continue;
        bool found = false;
        (*r)->for_each([&](uint32_t q) {
            if (found) return;
            if (q == producer) { found = true; return; }
            if (q > producer && visited.insert(q).second) stack.push_back(q);
        });
        if (found) return true;
    }
    return false;
}

void Hypergraph::qc_record_causal(uint32_t producer, uint32_t consumer) {
    // Per-consumed-edge relationships (the T1 multiset) count every occurrence.
    qc_num_causal_edges_.fetch_add(1, std::memory_order_relaxed);

    uint64_t pk = (static_cast<uint64_t>(producer) << 32) | consumer;
    pk = pk ? pk : 1;
    if (!qc_causal_pairs_.insert_if_absent(pk, true).second) return;   // pair already recorded
    qc_num_causal_pairs_.fetch_add(1, std::memory_order_relaxed);

    // One base, two views: tag whether this pair survives reduction. A pair bypassed by a
    // longer path is not in the reduction; otherwise it is kept and becomes part of the
    // predecessor adjacency later decisions walk.
    if (qc_reachable(producer, consumer)) return;                      // redundant: not tagged
    qc_num_tr_pairs_.fetch_add(1, std::memory_order_relaxed);
    LockFreeList<uint32_t>* lst;
    const uint64_t k = static_cast<uint64_t>(consumer) + 1;
    auto r = qc_preds_.lookup(k);
    if (r.has_value()) lst = *r;
    else {
        auto* nl = arena_.template create<LockFreeList<uint32_t>>();
        auto ins = qc_preds_.insert_if_absent(k, nl);
        lst = ins.second ? nl : ins.first;
    }
    lst->push(producer, arena_);
}

void Hypergraph::qc_apply(const QcInstance& inst, const SlotMatch& m, uint64_t state_hash,
                          uint32_t depth) {
    // Claim the (instance, match) pair. Both sides of the rendezvous drive application -- an
    // instance arriving replays the known matches, and a match being captured replays the
    // known instances -- and unlike the producer-set DP this is NOT idempotent: every
    // application mints a raw event. Exactly-once is therefore enforced here.
    uint64_t ck = 1469598103934665603ULL;
    ck ^= inst.id;  ck *= 1099511628211ULL;
    ck ^= m.id;     ck *= 1099511628211ULL;
    if (ck == 0 || ck == ~0ULL) ck = 1;
    if (!qc_applied_.insert_if_absent(ck, true).second) return;
    if (m.from_slots != inst.nslots) return;   // capture/instance disagree; drop rather than corrupt

    // The raw event this instance's copy of the match stands for. An id suffices: counts and
    // causal edges are expressed over ids, so no Event record (and hence no raw state or raw
    // edge) has to be materialised here.
    const uint32_t ev = qc_next_raw_event_.fetch_add(1, std::memory_order_relaxed);
    {   // The event's only identity: isomorphism-invariant, so schedule-independence can be
        // fingerprinted on it and a later materialisation can key off it.
        uint64_t s = 1469598103934665603ULL;
        s ^= state_hash; s *= 1099511628211ULL;
        s ^= m.to_hash;  s *= 1099511628211ULL;
        s ^= m.rule;     s *= 1099511628211ULL;
        qc_event_sig_.emplace_at(ev, arena_, s);
    }

    // Causal: one relationship per consumed edge that has a producer. Feed them in DESCENDING
    // producer order so nearer producers enter the kept adjacency before farther ones are
    // tested -- the same discipline the full-capture rendezvous uses, and what makes the
    // reduction tag exact rather than insertion-order dependent.
    {
        auto mk = worker_scratch().mark();
        SVec<uint32_t> producers;
        for (uint32_t i = 0; i < m.num_consumed; ++i) {
            const uint32_t s = m.consumed_slots[i];
            if (s >= inst.nslots) continue;
            const uint32_t p = inst.prod[s];
            if (p != QC_NO_PRODUCER) producers.push_back(p);
        }
        std::sort(producers.begin(), producers.end(), std::greater<uint32_t>());
        for (uint32_t p : producers) qc_record_causal(p, ev);
        worker_scratch().release(mk);
    }

    // Branchial: two events are siblings when they expand the SAME instance and their consumed
    // edges overlap. Each unordered pair is counted once, by only considering matches whose id
    // is below this one -- so whichever of the two applies second reports the pair, and the
    // claim above guarantees that happens exactly once per (instance, match).
    if (m.num_consumed) {
        for_each_expansion_match(state_hash, [&](const SlotMatch& other) {
            if (other.id >= m.id) return;
            bool overlaps = false;
            for (uint32_t i = 0; i < m.num_consumed && !overlaps; ++i)
                for (uint32_t j = 0; j < other.num_consumed; ++j)
                    if (m.consumed_slots[i] == other.consumed_slots[j]) { overlaps = true; break; }
            if (overlaps) qc_num_branchial_.fetch_add(1, std::memory_order_relaxed);
        });
    }

    // Child instance: survivors carry their producer across, produced slots take this event.
    uint32_t* cp = arena_.allocate_array<uint32_t>(m.to_slots ? m.to_slots : 1);
    for (uint32_t i = 0; i < m.to_slots; ++i) cp[i] = QC_NO_PRODUCER;
    for (uint32_t i = 0; i < m.num_survivors; ++i) {
        const uint32_t a = m.surv_from_slot[i], b = m.surv_to_slot[i];
        if (a < inst.nslots && b < m.to_slots) cp[b] = inst.prod[a];
    }
    for (uint32_t i = 0; i < m.num_produced; ++i) {
        const uint32_t s = m.produced_slots[i];
        if (s < m.to_slots) cp[s] = ev;
    }
    qc_add_instance(m.to_hash, depth + 1, cp, m.to_slots);
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
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t i = 0; i < n; ++i) { h ^= slots[i]; h *= 1099511628211ULL; }
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
    if (!in_orb || !out_orb || !in_orb->slot || !out_orb->slot) return;
    const uint64_t from = get_state(ev.input_state).canonical_hash;

    const uint64_t claim = static_cast<uint64_t>(ev.input_state) + 1;
    auto rep = qc_expansion_rep_.insert_if_absent(from, claim);
    if (!rep.second && rep.first != claim) return;   // a different raw state owns this class

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
    // produced -- they passed through, carrying their producer forward. Recorded as the
    // (orbit in `from`, orbit in `to`) pair, sorted.
    const SparseBitset& in_edges = get_state(ev.input_state).edges;
    SVec<std::pair<uint32_t,uint32_t>> survivors;
    for (uint32_t i = 0; i < out_orb->n; ++i) {
        const EdgeId oe = out_orb->edges[i];
        bool produced_here = false;
        for (uint8_t j = 0; j < ev.num_produced; ++j) if (ev.produced_edges[j] == oe) { produced_here = true; break; }
        if (produced_here) continue;
        if (in_edges.contains(oe)) survivors.push_back({in_orb->orbit_of(oe), out_orb->orbit[i]});
    }
    std::sort(survivors.begin(), survivors.end());

    // Dedup signature over (from, to, rule, consumed orbits, survivor orbit pairs) -- the
    // same key the validated prototype dedups canonical transitions on.
    uint64_t sig = 1469598103934665603ULL;
    auto mix = [&](uint64_t v){ sig ^= v; sig *= 1099511628211ULL; };
    mix(from); mix(to); mix(ev.rule_index);
    for (uint32_t o : consumed) { mix(0x1111); mix(o); }
    for (auto& pr : survivors) { mix(0x2222); mix(pr.first); mix(pr.second); }

    auto seen = seen_transitions_.insert_if_absent(sig, true);
    if (!seen.second) { worker_scratch().release(mk); return; }  // already captured

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
    for (uint32_t i = 0; i < nsurv; ++i) { sf[i] = survivors[i].first; st[i] = survivors[i].second; }


    CanonicalTransition* t = arena_.template create<CanonicalTransition>();
    t->to_hash = to; t->sig = sig; t->canon_event = get_canonical_event(e); t->rule = ev.rule_index;
    t->num_consumed = static_cast<uint32_t>(consumed.size());
    t->num_produced = static_cast<uint32_t>(produced.size());
    t->num_survivors = nsurv;
    t->consumed_orbits = copy(consumed);
    t->produced_orbits = copy(produced);
    t->surv_from = sf; t->surv_to = st;
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
        if (qc_reached_.lookup(qc_rkey(from, static_cast<uint32_t>(d))).has_value())
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

    IRCanonicalizer ir;
    std::vector<uint32_t> orbit;
    uint64_t chash = ir.compute_canonical_hash_with_edge_orbits(edge_vecs, orbit);

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

}  // namespace hypergraph
