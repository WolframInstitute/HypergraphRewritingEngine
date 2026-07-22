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

    // Register in global vertex adjacency index
    for (uint8_t i = 0; i < arity; ++i) {
        VertexId v = vertices[i];
        EdgeOccurrence occ(eid, i, arity);
        vertex_adjacency_.get_or_default(v, arena_).push(occ, arena_);
    }

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
        SparseBitset empty_edges;
        genesis_state_ = create_state(std::move(empty_edges), 0, 0, INVALID_ID);
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
            map_key = static_cast<uint64_t>(new_sid);
            canonical_hash = wl_child();
            break;
        case StateCanonicalizationMode::Automatic:
            map_key = compute_content_ordered_hash(edges);
            canonical_hash = wl_child();
            break;
        case StateCanonicalizationMode::Full:
        default:
            canonical_hash = compute_canonical_hash(edges);   // exact IR
            map_key = canonical_hash;
            break;
    }
    states_[new_sid].canonical_hash = canonical_hash;

    // Try to insert into canonical map (lock-free, waiting for LOCKED slots)
    auto [existing_or_new, was_inserted] = canonical_state_map_.insert_if_absent_waiting(map_key, new_sid);

    // Insert into event_canonical_state_map_ only when event canonicalization is on:
    // its sole reader (get_canonical_state_for_event) runs only under
    // event_signature_keys_ != EVENT_SIG_NONE, and the keys are fixed at config time
    // before any state is created, so gating here never drops a needed entry. When
    // event canon is off this saves ~16 B/state + the map's resize chain + a per-state
    // hash+probe insert.
    if (event_signature_keys_ != EVENT_SIG_NONE) {
        event_canonical_state_map_.insert_if_absent_waiting(canonical_hash, new_sid);
    }

    // In Full mode, the map key is the IR canonical hash which is exact —
    // hash collisions are genuine isomorphisms, no verification needed
    bool verified_duplicate = !was_inserted;

    // Cache the canonical ID in the state for fast lookup
    states_[new_sid].canonical_id = existing_or_new;

    // CRITICAL: Release fence ensures canonical_id write is visible to other threads
    // on ARM64's weak memory model. Pairs with acquire fence in get_canonical_state().
    std::atomic_thread_fence(std::memory_order_release);

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

    // Register this event as the producer of all initial edges
    for (uint8_t i = 0; i < num_edges; ++i) {
        set_edge_producer(edges[i], eid);
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
        return 0;
    }

    IRCanonicalizer ir;
    uint64_t h = ir.compute_canonical_hash(edge_vectors);
    worker_scratch().release(mk);
    return h;
}

uint64_t Hypergraph::compute_wl_hash(const SparseBitset& edges) const {
    if (edges.empty()) {
        return 0;
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
