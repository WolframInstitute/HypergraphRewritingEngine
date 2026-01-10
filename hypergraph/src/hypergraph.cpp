// hypergraph.cpp - Implementation of Hypergraph class non-template methods

#include "hypergraph/hypergraph.hpp"

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
    EdgeId eid = counters_.alloc_edge();

    // Allocate and copy vertex array
    VertexId* verts = arena_.allocate_array<VertexId>(arity);
    std::memcpy(verts, vertices, arity * sizeof(VertexId));

    // Directly construct edge at slot eid using emplace_at
    edges_.emplace_at(eid, arena_, eid, verts, arity, creator_event, step);

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

    // Register with hash implementations
    if (unified_tree_) {
        unified_tree_->register_edge(eid, vertices, arity);
    }
    if (incremental_tree_) {
        incremental_tree_->register_edge(eid, vertices, arity);
    }
    if (wl_hash_) {
        wl_hash_->register_edge(eid, vertices, arity);
    }

    return eid;
}

EdgeId Hypergraph::create_edge(std::initializer_list<VertexId> vertices,
                               EventId creator_event,
                               uint32_t step) {
    VertexId verts[MAX_ARITY];
    uint8_t arity = 0;
    for (VertexId v : vertices) {
        if (arity < MAX_ARITY) {
            verts[arity++] = v;
        }
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

    // Ensure auxiliary arrays are large enough (thread-safe)
    state_children_.ensure_size(sid + 1, arena_);
    state_matches_.ensure_size(sid + 1, arena_);

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
        // Spin (could add pause/yield here for better performance)
    }
    return genesis_state_;
}

// =============================================================================
// Canonical State Deduplication
// =============================================================================

Hypergraph::CanonicalStateResult Hypergraph::create_or_get_canonical_state(
    SparseBitset&& edge_set,
    uint64_t canonical_hash,
    uint32_t step,
    EventId parent_event
) {
    // First, create the state unconditionally
    StateId new_sid = create_state(std::move(edge_set), step, canonical_hash, parent_event);

    // Determine the key for canonical map based on mode
    uint64_t map_key;
    switch (state_canonicalization_mode_) {
        case StateCanonicalizationMode::None:
            map_key = static_cast<uint64_t>(new_sid);
            break;
        case StateCanonicalizationMode::Automatic:
            map_key = compute_content_ordered_hash(get_state(new_sid).edges);
            break;
        case StateCanonicalizationMode::Full:
        default:
            map_key = canonical_hash;
            break;
    }

    // Try to insert into canonical map (lock-free, waiting for LOCKED slots)
    auto [existing_or_new, was_inserted] = canonical_state_map_.insert_if_absent_waiting(map_key, new_sid);

    // Also insert into event_canonical_state_map_ using the isomorphism-invariant hash
    event_canonical_state_map_.insert_if_absent_waiting(canonical_hash, new_sid);

    // Cache the canonical ID in the state for fast lookup
    states_[new_sid].canonical_id = existing_or_new;

    if (!was_inserted) {
        return {existing_or_new, new_sid, false};
    }

    return {new_sid, new_sid, true};
}

uint64_t Hypergraph::get_or_compute_canonical_hash(StateId state_id) {
    if (state_id == INVALID_ID) return 0;

    State& state = states_[state_id];

    // If hash is already computed, return it
    if (state.canonical_hash != 0) {
        return state.canonical_hash;
    }

    // Compute hash on-demand using hash dispatch
    auto [hash, cache] = compute_hash_with_cache_dispatch(state.edges);

    // Cache the hash for future use
    state.canonical_hash = hash;
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
    uint8_t num_produced,
    const VariableBinding& binding
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

            // Build edge mappings
            std::unordered_map<EdgeId, EdgeId> input_edge_map, output_edge_map;
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
                       cons, num_consumed, prod, num_produced, binding, canonical_id_for_event);

    // CRITICAL: Release fence to ensure event data is visible
    std::atomic_thread_fence(std::memory_order_release);

    // Track parent-child relationship
    add_state_child(input_state, output_state);

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
                       VariableBinding{},
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
    if (event_signature_keys_ != EVENT_SIG_NONE) {
        // Use edge equivalence-aware branchial registration
        causal_graph_.register_event_from_state_with_canonicalization(
            event, input_state, consumed_edges, num_consumed,
            // Get consumed edges callback
            [this](EventId eid, const EdgeId*& edges, uint8_t& num) {
                const Event& ev = events_[eid];
                edges = ev.consumed_edges;
                num = ev.num_consumed;
            },
            // Same canonical event check
            [this, canonical_event](EventId e1, EventId e2) -> bool {
                (void)e1; (void)e2; (void)canonical_event;
                return false;
            },
            // Edge equivalence check
            [](EdgeId e1, EdgeId e2) -> bool {
                return e1 == e2;
            }
        );
    } else {
        causal_graph_.register_event_from_state_with_overlap_check(
            event, input_state, consumed_edges, num_consumed,
            [this](EventId eid, const EdgeId*& edges, uint8_t& num) {
                const Event& ev = events_[eid];
                edges = ev.consumed_edges;
                num = ev.num_consumed;
            }
        );
    }
}

// =============================================================================
// Match Management
// =============================================================================

MatchId Hypergraph::register_match(
    RuleIndex rule_index,
    const EdgeId* matched_edges,
    uint8_t num_edges,
    const VariableBinding& binding,
    StateId origin_state
) {
    MatchId mid = counters_.alloc_match();

    // Allocate and copy edge array
    EdgeId* edges = arena_.allocate_array<EdgeId>(num_edges);
    std::memcpy(edges, matched_edges, num_edges * sizeof(EdgeId));

    // Directly construct match at slot mid using emplace_at
    matches_.emplace_at(mid, arena_, mid, rule_index, edges, num_edges, binding, origin_state);

    // Add to state's match list
    if (origin_state < state_matches_.size()) {
        state_matches_[origin_state].push(mid, arena_);
    }

    return mid;
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
    // Use unified uniqueness tree if enabled
    if (use_shared_tree_ && unified_tree_) {
        return compute_canonical_hash_shared(edges);
    }

    // Build edge vectors for canonicalizer
    std::vector<std::vector<std::size_t>> edge_vectors;

    std::atomic_thread_fence(std::memory_order_acquire);

    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];

        std::vector<std::size_t> verts;
        verts.reserve(e.arity);
        for (uint8_t i = 0; i < e.arity; ++i) {
            verts.push_back(static_cast<std::size_t>(e.vertices[i]));
        }
        edge_vectors.push_back(std::move(verts));
    });

    if (edge_vectors.empty()) {
        return 0;
    }

    // Use exact canonicalization
    hypergraph::Canonicalizer canonicalizer;
    auto result = canonicalizer.canonicalize_edges(edge_vectors);

    // FNV-style hash of canonical form
    uint64_t hash = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;

    for (const auto& edge : result.canonical_form.edges) {
        for (auto vertex : edge) {
            hash ^= static_cast<uint64_t>(vertex);
            hash *= FNV_PRIME;
        }
        hash ^= 0xDEADBEEF;
        hash *= FNV_PRIME;
    }

    return hash;
}

std::string Hypergraph::get_canonical_form_string(const SparseBitset& edges) const {
    std::vector<std::vector<std::size_t>> edge_vectors;
    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        std::vector<std::size_t> verts;
        verts.reserve(e.arity);
        for (uint8_t i = 0; i < e.arity; ++i) {
            verts.push_back(static_cast<std::size_t>(e.vertices[i]));
        }
        edge_vectors.push_back(std::move(verts));
    });

    if (edge_vectors.empty()) {
        return "{}";
    }

    hypergraph::Canonicalizer canonicalizer;
    auto result = canonicalizer.canonicalize_edges(edge_vectors);

    std::string s = "{";
    for (size_t i = 0; i < result.canonical_form.edges.size(); ++i) {
        if (i > 0) s += ", ";
        s += "{";
        for (size_t j = 0; j < result.canonical_form.edges[i].size(); ++j) {
            if (j > 0) s += ",";
            s += std::to_string(result.canonical_form.edges[i][j]);
        }
        s += "}";
    }
    s += "}";
    return s;
}

std::string Hypergraph::get_raw_edges_string(const SparseBitset& edges) const {
    std::string s = "{";
    bool first = true;
    edges.for_each([&](EdgeId eid) {
        if (!first) s += ", ";
        first = false;
        const Edge& e = edges_[eid];
        s += "{";
        for (uint8_t i = 0; i < e.arity; ++i) {
            if (i > 0) s += ",";
            s += std::to_string(e.vertices[i]);
        }
        s += "}";
    });
    s += "}";
    return s;
}

uint64_t Hypergraph::compute_canonical_hash_shared(const SparseBitset& edges) const {
    if (edges.empty()) {
        return 0;
    }

    std::atomic_thread_fence(std::memory_order_acquire);

    auto [hash, cache] = compute_hash_with_cache_dispatch(edges);
    return hash;
}

// =============================================================================
// Hash Cache Management
// =============================================================================

VertexHashCache Hypergraph::get_or_compute_wl_cache(StateId state_id) {
    WLHashCacheEntry& entry = wl_hash_cache_.get_or_default(state_id, arena_);

    // Fast path: already computed
    VertexHashCache* cached = entry.cache_ptr.load(std::memory_order_acquire);
    if (cached) {
        return *cached;
    }

    // Slow path: compute cache
    const State& state = get_state(state_id);
    EdgeVertexAccessorRaw vert_acc(this);
    EdgeArityAccessorRaw arity_acc(this);

    auto [hash, cache] = wl_hash_->compute_state_hash_with_cache(
        state.edges, vert_acc, arity_acc);

    // Allocate cache on arena and copy data
    VertexHashCache* new_cache = arena_.create<VertexHashCache>(cache);

    // Try to set the pointer atomically
    VertexHashCache* expected = nullptr;
    if (entry.cache_ptr.compare_exchange_strong(expected, new_cache,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
        return *new_cache;
    } else {
        return *expected;
    }
}

VertexHashCache Hypergraph::get_or_compute_ut_cache(StateId state_id) {
    StateIncrementalCache& entry = state_incremental_cache_.get_or_default(state_id, arena_);

    // Fast path: already computed
    StateIncrementalCacheData* cached = entry.data_ptr.load(std::memory_order_acquire);
    if (cached) {
        return cached->vertex_cache;
    }

    // Slow path: compute cache
    const State& state = get_state(state_id);
    EdgeVertexAccessorRaw vert_acc(this);
    EdgeArityAccessorRaw arity_acc(this);

    auto [hash, cache] = incremental_tree_->compute_state_hash_with_cache(
        state.edges, vert_acc, arity_acc);

    // Allocate cache data on arena
    StateIncrementalCacheData* new_data = arena_.create<StateIncrementalCacheData>();
    new_data->vertex_cache = cache;

    // Try to set the pointer atomically
    StateIncrementalCacheData* expected = nullptr;
    if (entry.data_ptr.compare_exchange_strong(expected, new_data,
                                                std::memory_order_release,
                                                std::memory_order_acquire)) {
        return new_data->vertex_cache;
    } else {
        return expected->vertex_cache;
    }
}

VertexHashCache Hypergraph::get_or_compute_ut_plain_cache(StateId state_id) {
    WLHashCacheEntry& entry = wl_hash_cache_.get_or_default(state_id, arena_);

    // Fast path: already computed
    VertexHashCache* cached = entry.cache_ptr.load(std::memory_order_acquire);
    if (cached) {
        return *cached;
    }

    // Slow path: compute cache
    const State& state = get_state(state_id);
    EdgeVertexAccessorRaw vert_acc(this);
    EdgeArityAccessorRaw arity_acc(this);

    auto [hash, cache] = unified_tree_->compute_state_hash_with_cache(
        state.edges, vert_acc, arity_acc);

    // Allocate cache on arena and copy data
    VertexHashCache* new_cache = arena_.create<VertexHashCache>(cache);

    // Try to set the pointer atomically
    VertexHashCache* expected = nullptr;
    if (entry.cache_ptr.compare_exchange_strong(expected, new_cache,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
        return *new_cache;
    } else {
        return *expected;
    }
}

// =============================================================================
// Hash Strategy Dispatch
// =============================================================================

std::pair<uint64_t, VertexHashCache> Hypergraph::compute_hash_with_cache_dispatch(
    const SparseBitset& edges
) const {
    if (edges.empty()) {
        return {0, VertexHashCache()};
    }

    EdgeVertexAccessorRaw vert_acc(this);
    EdgeArityAccessorRaw arity_acc(this);

    if ((hash_strategy_ == HashStrategy::WL || hash_strategy_ == HashStrategy::IncrementalWL) && wl_hash_) {
        return wl_hash_->compute_state_hash_with_cache(edges, vert_acc, arity_acc);
    } else if (hash_strategy_ == HashStrategy::IncrementalUniquenessTree && incremental_tree_) {
        return incremental_tree_->compute_state_hash_with_cache(edges, vert_acc, arity_acc);
    } else if (unified_tree_) {
        return unified_tree_->compute_state_hash_with_cache(edges, vert_acc, arity_acc);
    }
    return {0, VertexHashCache()};
}

VertexHashCache Hypergraph::get_or_compute_hash_cache_dispatch(StateId state_id) {
    switch (hash_strategy_) {
        case HashStrategy::WL:
        case HashStrategy::IncrementalWL:
            return get_or_compute_wl_cache(state_id);
        case HashStrategy::IncrementalUniquenessTree:
            return get_or_compute_ut_cache(state_id);
        case HashStrategy::UniquenessTree:
        default:
            return get_or_compute_ut_plain_cache(state_id);
    }
}

EdgeCorrespondence Hypergraph::find_edge_correspondence_dispatch(
    const SparseBitset& state1_edges,
    const SparseBitset& state2_edges
) const {
    EdgeVertexAccessorRaw vert_acc(this);
    EdgeArityAccessorRaw arity_acc(this);

    if ((hash_strategy_ == HashStrategy::WL || hash_strategy_ == HashStrategy::IncrementalWL) && wl_hash_) {
        return wl_hash_->find_edge_correspondence(state1_edges, state2_edges, vert_acc, arity_acc);
    } else if (hash_strategy_ == HashStrategy::IncrementalUniquenessTree && incremental_tree_) {
        return incremental_tree_->find_edge_correspondence(state1_edges, state2_edges, vert_acc, arity_acc);
    } else if (unified_tree_) {
        return unified_tree_->find_edge_correspondence(state1_edges, state2_edges, vert_acc, arity_acc);
    }
    return EdgeCorrespondence{};
}

// =============================================================================
// Incremental Hash Computation
// =============================================================================

std::pair<uint64_t, VertexHashCache> Hypergraph::compute_canonical_hash_incremental(
    const SparseBitset& new_edges,
    StateId parent_state,
    const EdgeId* consumed_edges, uint8_t num_consumed,
    const EdgeId* produced_edges, uint8_t num_produced
) {
    if (new_edges.empty()) {
        return {0, VertexHashCache()};
    }

    EdgeVertexAccessorRaw verts_accessor(this);
    EdgeArityAccessorRaw arities_accessor(this);

    // Handle WL strategy (non-incremental)
    if (hash_strategy_ == HashStrategy::WL && wl_hash_) {
        auto [hash, cache] = wl_hash_->compute_state_hash_with_cache(
            new_edges, verts_accessor, arities_accessor);
        return {hash, cache};
    }

    // Handle IncrementalWL strategy
    if (hash_strategy_ == HashStrategy::IncrementalWL && wl_hash_) {
        // Try to get parent cache for incremental computation
        const VertexHashCache* parent_wl_cache = nullptr;
        if (parent_state != INVALID_ID && parent_state < wl_hash_cache_.size()) {
            const WLHashCacheEntry& entry = wl_hash_cache_[parent_state];
            VertexHashCache* cached = entry.cache_ptr.load(std::memory_order_acquire);
            if (cached && cached->count > 0) {
                parent_wl_cache = cached;
            }
        }

        if (parent_wl_cache) {
            auto [hash, cache] = wl_hash_->compute_state_hash_incremental_with_cache(
                new_edges, *parent_wl_cache,
                consumed_edges, num_consumed,
                produced_edges, num_produced,
                verts_accessor, arities_accessor);
            return {hash, cache};
        } else {
            auto [hash, cache] = wl_hash_->compute_state_hash_with_cache(
                new_edges, verts_accessor, arities_accessor);
            return {hash, cache};
        }
    }

    // Use incremental path only for IncrementalUniquenessTree strategy
    if (hash_strategy_ != HashStrategy::IncrementalUniquenessTree || !incremental_tree_) {
        if (unified_tree_) {
            auto [hash, cache] = unified_tree_->compute_state_hash_with_cache(
                new_edges, verts_accessor, arities_accessor);
            return {hash, cache};
        }
        return {0, VertexHashCache()};
    }

    std::atomic_thread_fence(std::memory_order_acquire);

    // Get parent's vertex hash cache for incremental reuse
    const VertexHashCache* parent_vertex_cache = nullptr;
    if (parent_state != INVALID_ID && parent_state < state_incremental_cache_.size()) {
        const StateIncrementalCache* pcache = state_incremental_cache_.get(parent_state);
        if (pcache) {
            StateIncrementalCacheData* data = pcache->data_ptr.load(std::memory_order_acquire);
            if (data) {
                const VertexHashCache& vc = data->vertex_cache;
                if (vc.count > 0 && vc.vertices != nullptr &&
                    vc.hashes != nullptr && vc.subtree_filters != nullptr) {
                    parent_vertex_cache = &vc;
                }
            }
        }
    }

    // Collect directly affected vertices
    ArenaVector<VertexId> affected_vertices(arena_);
    for (uint8_t i = 0; i < num_consumed; ++i) {
        EdgeId eid = consumed_edges[i];
        uint8_t arity = arities_accessor[eid];
        const VertexId* verts = verts_accessor[eid];
        for (uint8_t j = 0; j < arity; ++j) {
            affected_vertices.push_back(verts[j]);
        }
    }
    for (uint8_t i = 0; i < num_produced; ++i) {
        EdgeId eid = produced_edges[i];
        uint8_t arity = arities_accessor[eid];
        const VertexId* verts = verts_accessor[eid];
        for (uint8_t j = 0; j < arity; ++j) {
            affected_vertices.push_back(verts[j]);
        }
    }

    // Remove duplicates
    std::sort(affected_vertices.begin(), affected_vertices.end());
    auto new_end = std::unique(affected_vertices.begin(), affected_vertices.end());
    affected_vertices.resize(new_end - affected_vertices.begin());

    // If no parent cache with bloom filters, use the full computation
    if (!parent_vertex_cache) {
        return incremental_tree_->compute_state_hash_with_cache(
            new_edges, verts_accessor, arities_accessor);
    }

    // Bloom filter reuse path

    // Collect all vertices in child state
    ArenaVector<VertexId> vertices(arena_);
    std::unordered_set<VertexId> seen_vertices;
    new_edges.for_each([&](EdgeId eid) {
        uint8_t arity = arities_accessor[eid];
        const VertexId* verts = verts_accessor[eid];
        for (uint8_t j = 0; j < arity; ++j) {
            if (seen_vertices.insert(verts[j]).second) {
                vertices.push_back(verts[j]);
            }
        }
    });
    std::sort(vertices.begin(), vertices.end());

    if (vertices.empty()) {
        return {0, VertexHashCache()};
    }

    // Prepare result cache with space for bloom filters
    VertexHashCache result_cache;
    result_cache.capacity = static_cast<uint32_t>(vertices.size());
    result_cache.vertices = arena_.allocate_array<VertexId>(result_cache.capacity);
    result_cache.hashes = arena_.allocate_array<uint64_t>(result_cache.capacity);
    result_cache.subtree_filters = arena_.allocate_array<SubtreeBloomFilter>(result_cache.capacity);
    result_cache.count = 0;

    ArenaVector<uint64_t> tree_hashes(arena_, vertices.size());

    size_t local_reused = 0;
    size_t local_recomputed = 0;

    // Lazy adjacency building
    std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>> adjacency;
    bool adjacency_built = false;

    auto build_adjacency_if_needed = [&]() {
        if (adjacency_built) return;
        adjacency_built = true;

        new_edges.for_each([&](EdgeId eid) {
            uint8_t arity = arities_accessor[eid];
            const VertexId* verts = verts_accessor[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                VertexId v = verts[i];
                auto it = adjacency.find(v);
                if (it == adjacency.end()) {
                    it = adjacency.emplace(v, ArenaVector<std::pair<EdgeId, uint8_t>>(arena_)).first;
                }
                it->second.push_back({eid, i});
            }
        });
    };

    // Build O(1) lookup map from parent cache
    std::unordered_map<VertexId, uint32_t> parent_cache_index;
    parent_cache_index.reserve(parent_vertex_cache->count);
    for (uint32_t i = 0; i < parent_vertex_cache->count; ++i) {
        parent_cache_index[parent_vertex_cache->vertices[i]] = i;
    }

    // Compute tree hash for each vertex, reusing where possible
    for (VertexId root : vertices) {
        auto cache_it = parent_cache_index.find(root);
        if (cache_it != parent_cache_index.end()) {
            uint32_t idx = cache_it->second;
            uint64_t parent_hash = parent_vertex_cache->hashes[idx];
            const SubtreeBloomFilter* bloom = parent_vertex_cache->subtree_filters
                ? &parent_vertex_cache->subtree_filters[idx] : nullptr;

            if (bloom != nullptr && parent_hash != 0) {
                bool might_be_affected = false;
                for (VertexId affected : affected_vertices) {
                    if (bloom->might_contain(affected)) {
                        might_be_affected = true;
                        break;
                    }
                }

                if (!might_be_affected) {
                    tree_hashes.push_back(parent_hash);
                    result_cache.insert_with_subtree(root, parent_hash, *bloom);
                    ++local_reused;
                    continue;
                }
            }
        }

        // Need to recompute this vertex's hash
        build_adjacency_if_needed();
        ++local_recomputed;

        DirectAdjacencyWithArity<decltype(adjacency), EdgeArityAccessorRaw> adj_provider(adjacency, arities_accessor);

        SparseBitset visited;
        SubtreeBloomFilter new_bloom;
        new_bloom.clear();

        uint64_t tree_hash = incremental_tree_->compute_tree_hash_with_bloom(
            root, new_edges, verts_accessor, arities_accessor,
            adj_provider, visited, new_bloom);

        tree_hashes.push_back(tree_hash);
        result_cache.insert_with_subtree(root, tree_hash, new_bloom);
    }

    // Update member atomics for stats reporting
    bloom_reused_.fetch_add(local_reused, std::memory_order_relaxed);
    bloom_recomputed_.fetch_add(local_recomputed, std::memory_order_relaxed);

    // Combine tree hashes into state hash
    std::sort(tree_hashes.begin(), tree_hashes.end());
    uint64_t state_hash = FNV_OFFSET;
    state_hash = fnv_hash(state_hash, tree_hashes.size());
    for (uint64_t h : tree_hashes) {
        state_hash = fnv_hash(state_hash, h);
    }

    return {state_hash, result_cache};
}

void Hypergraph::store_state_cache(StateId state, const VertexHashCache& cache) {
    // Store for IncrementalWL strategy
    if (hash_strategy_ == HashStrategy::IncrementalWL) {
        WLHashCacheEntry& slot = wl_hash_cache_.get_or_default(state, arena_);

        VertexHashCache* new_cache = arena_.create<VertexHashCache>(cache);

        VertexHashCache* expected = nullptr;
        slot.cache_ptr.compare_exchange_strong(expected, new_cache,
                std::memory_order_release, std::memory_order_relaxed);
        return;
    }

    // Store for iUT strategy
    if (hash_strategy_ != HashStrategy::IncrementalUniquenessTree) {
        return;
    }

    StateIncrementalCache& slot = state_incremental_cache_.get_or_default(state, arena_);

    StateIncrementalCacheData* new_data = arena_.create<StateIncrementalCacheData>();
    new_data->vertex_cache = cache;

    StateIncrementalCacheData* expected = nullptr;
    slot.data_ptr.compare_exchange_strong(expected, new_data,
            std::memory_order_release, std::memory_order_relaxed);
}

size_t Hypergraph::num_stored_caches() const {
    size_t count = 0;
    for (size_t i = 0; i < state_incremental_cache_.size(); ++i) {
        if (state_incremental_cache_[i].data_ptr.load(std::memory_order_relaxed) != nullptr) {
            ++count;
        }
    }
    return count;
}

std::pair<size_t, size_t> Hypergraph::incremental_tree_stats() const {
    size_t reused = bloom_reused_.load(std::memory_order_relaxed);
    size_t recomputed = bloom_recomputed_.load(std::memory_order_relaxed);
    if (incremental_tree_) {
        reused += incremental_tree_->stats_reused();
        recomputed += incremental_tree_->stats_recomputed();
    }
    return {reused, recomputed};
}

void Hypergraph::reset_incremental_tree_stats() {
    bloom_reused_.store(0, std::memory_order_relaxed);
    bloom_recomputed_.store(0, std::memory_order_relaxed);
    if (incremental_tree_) {
        incremental_tree_->reset_stats();
    }
}

}  // namespace hypergraph
