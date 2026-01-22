// parallel_evolution.cpp - Implementation of ParallelEvolutionEngine class

#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

namespace hypergraph {

// =============================================================================
// Constructor / Destructor
// =============================================================================

ParallelEvolutionEngine::ParallelEvolutionEngine(Hypergraph* hg, size_t num_threads)
    : hg_(hg)
    , rewriter_(hg)
    , num_threads_(num_threads > 0 ? num_threads : std::thread::hardware_concurrency())
{
    job_system_ = std::make_unique<job_system::JobSystem<EvolutionJobType>>(num_threads_);
    job_system_->start();
}

ParallelEvolutionEngine::~ParallelEvolutionEngine() {
    if (job_system_) {
        // Defensive: ensure all jobs complete before destruction
        // This prevents use-after-free if caller forgot to wait
        request_stop();
        job_system_->wait_for_completion();
        job_system_->shutdown();
    }
    // CRITICAL: Clear abort flag pointer in hg_ before should_stop_ is destroyed
    // Otherwise hg_ and its trees hold a dangling pointer
    if (hg_) {
        hg_->set_abort_flag(nullptr);
    }
}

// =============================================================================
// Main Evolution Loop
// =============================================================================

void ParallelEvolutionEngine::evolve(
    const std::vector<std::vector<VertexId>>& initial_edges,
    size_t steps
) {
    if (!hg_ || rules_.empty()) return;

    max_steps_ = steps;
    should_stop_.store(false, std::memory_order_relaxed);

    // Propagate abort flag to hypergraph for long-running hash computations
    hg_->set_abort_flag(&should_stop_);

    // Create initial state
    std::vector<EdgeId> edge_ids;
    for (const auto& edge : initial_edges) {
        EdgeId eid = hg_->create_edge(edge.data(), static_cast<uint8_t>(edge.size()));
        edge_ids.push_back(eid);

        // Track max vertex ID to ensure fresh vertices don't collide
        for (VertexId v : edge) {
            hg_->reserve_vertices(v);
        }
    }

    SparseBitset initial_edge_set;
    for (EdgeId eid : edge_ids) {
        initial_edge_set.set(eid, hg_->arena());
    }

    // For initial state, use incremental path with no parent (will compute from scratch)
    // This also returns the cache to store for future incremental computation
    auto [canonical_hash, vertex_cache] = hg_->compute_canonical_hash_incremental(
        initial_edge_set,
        INVALID_ID,  // No parent state
        nullptr, 0,  // No consumed edges
        edge_ids.data(), static_cast<uint8_t>(edge_ids.size())  // All edges are "produced"
    );
    auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
        std::move(initial_edge_set), canonical_hash, 0, INVALID_ID);

    // Store cache for the initial state
    hg_->store_state_cache(raw_state, vertex_cache);

    // Emit visualization event for initial state
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
    {
        const auto& state_data = hg_->get_state(raw_state);
        VIZ_EMIT_STATE_CREATED(
            raw_state,                // state id
            0,                        // parent state id (0 = none)
            0,                        // generation (initial state is gen 0)
            state_data.edges.count(), // edge count
            0                         // vertex count (not tracked per-state)
        );
        // Emit hyperedge data for each edge in the initial state
        uint32_t edge_idx = 0;
        state_data.edges.for_each([&](EdgeId eid) {
            const Edge& edge = hg_->get_edge(eid);
            VIZ_EMIT_HYPEREDGE(raw_state, edge_idx++, edge.vertices, edge.arity);
        });
    }
#endif

    // Create genesis event if enabled
    // This allows causal edges from initial state edges to be tracked
    if (enable_genesis_events_) {
        [[maybe_unused]] EventId genesis_event = hg_->create_genesis_event(
            raw_state,
            edge_ids.data(),
            static_cast<uint8_t>(edge_ids.size())
        );

        // Emit visualization event for the genesis event
        // Genesis events are always canonical (unique by definition)
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
        VIZ_EMIT_REWRITE_APPLIED(
            viz::VIZ_NO_SOURCE_STATE,  // source_state (none - genesis)
            raw_state,      // target_state (initial state)
            static_cast<RuleIndex>(-1),  // rule_index (none)
            genesis_event,  // event_id (raw)
            genesis_event,  // canonical_event_id (same as raw for genesis)
            0,              // destroyed edges (none)
            static_cast<uint8_t>(edge_ids.size())  // created edges
        );
#endif
    }

    // Mark initial state as matched (waiting version for correctness)
    matched_raw_states_.insert_if_absent_waiting(raw_state, true);

    // Submit MATCH task for initial state - this kicks off the dataflow
    submit_match_task(raw_state, 1);

    // Single synchronization point at the end
    job_system_->wait_for_completion();

    finalize_evolution();
}

void ParallelEvolutionEngine::evolve(
    const std::vector<std::vector<std::vector<VertexId>>>& initial_states,
    size_t steps
) {
    if (!hg_ || rules_.empty() || initial_states.empty()) return;

    max_steps_ = steps;
    should_stop_.store(false, std::memory_order_relaxed);

    // Propagate abort flag to hypergraph for long-running hash computations
    hg_->set_abort_flag(&should_stop_);

    // Create all initial states - they will all be explored
    for (const auto& state_edges : initial_states) {
        create_and_register_initial_state(state_edges);
    }

    // Single synchronization point at the end
    job_system_->wait_for_completion();

    finalize_evolution();
}

// =============================================================================
// Uniform Random Evolution
// =============================================================================

void ParallelEvolutionEngine::evolve_uniform_random(
    const std::vector<std::vector<VertexId>>& initial_edges,
    size_t steps,
    size_t matches_per_step
) {
    if (!hg_ || rules_.empty()) return;

    max_steps_ = steps;
    should_stop_.store(false, std::memory_order_relaxed);
    hg_->set_abort_flag(&should_stop_);

    // Create initial state
    StateId initial_state = create_initial_state_only(initial_edges);
    if (initial_state == INVALID_ID) return;

    fprintf(stderr, "[uniform_random] Initial: arena=%zu bytes, edges=%u, states=%u\n",
            hg_->arena().bytes_allocated(), hg_->num_edges(), hg_->num_states());

    std::vector<StateId> current_states;
    current_states.push_back(initial_state);

    std::vector<MatchRecord> step_matches;
    step_matches.reserve(matches_per_step * 10);

    std::vector<StateId> next_states;
    next_states.reserve(matches_per_step);

    // Thread-local RNG seeded once (avoid repeated random_device calls)
    std::random_device rd;
    std::mt19937 rng(rd());

    auto get_edge = [this](EdgeId eid) -> const Edge& { return hg_->get_edge(eid); };
    auto get_signature = [this](EdgeId eid) -> const EdgeSignature& { return hg_->edge_signature(eid); };

    size_t total_rewrites_applied = 0;
    size_t total_states_created = 0;

    // Match forwarding data structures
    // state_matches: matches found for each state in current frontier (cleared each step)
    std::unordered_map<StateId, std::vector<MatchRecord>> state_matches;

    // Child creation info for match forwarding
    struct ChildInfo {
        StateId parent;
        EdgeId consumed_edges[16];
        uint8_t num_consumed;
        EdgeId new_edges[16];
        uint8_t num_new;
    };
    std::unordered_map<StateId, ChildInfo> child_info;
    std::unordered_map<StateId, ChildInfo> next_child_info;

    for (size_t step = 1; step <= steps && !current_states.empty(); ++step) {
        if (should_stop_.load(std::memory_order_relaxed)) break;

        step_matches.clear();

        // Prepare per-state match storage for this step
        std::unordered_map<StateId, std::vector<MatchRecord>> next_state_matches;

        const size_t num_jobs = current_states.size() * rules_.size();
        const size_t max_matches_per_job = std::max(size_t(16), (matches_per_step + num_jobs - 1) / num_jobs);

        std::vector<std::vector<MatchRecord>> job_outputs(num_jobs);
        for (auto& v : job_outputs) {
            v.reserve(std::min(max_matches_per_job, size_t(100)));
        }

        // Seed for each job (deterministic from step to avoid random_device in jobs)
        uint64_t step_seed = rng();

        size_t job_idx = 0;
        for (StateId state : current_states) {
            if (should_stop_.load(std::memory_order_relaxed)) break;

            StateId canonical = hg_->get_canonical_state(state);
            uint64_t canonical_hash = hg_->get_state(state).canonical_hash;

            // Check if we have parent info for match forwarding
            auto ci_it = child_info.find(state);
            bool has_parent = (ci_it != child_info.end());

            // Get parent's matches if available
            const std::vector<MatchRecord>* parent_matches = nullptr;
            const ChildInfo* ci = nullptr;
            if (has_parent) {
                ci = &ci_it->second;
                auto pm_it = state_matches.find(ci->parent);
                if (pm_it != state_matches.end()) {
                    parent_matches = &pm_it->second;
                }
            }

            for (uint16_t r = 0; r < rules_.size(); ++r) {
                std::vector<MatchRecord>* output = &job_outputs[job_idx];
                uint64_t job_seed = step_seed ^ (job_idx * 0x9e3779b97f4a7c15ULL);
                job_idx++;

                // Copy data needed for forwarding (avoid dangling pointers in lambda)
                std::vector<MatchRecord> forwarded_matches;
                std::vector<EdgeId> new_edge_list;

                if (parent_matches && ci) {
                    // Forward parent matches for this rule, filtering consumed edges
                    for (const auto& pm : *parent_matches) {
                        if (pm.rule_index != r) continue;

                        // Check if any matched edge was consumed
                        bool uses_consumed = false;
                        for (uint8_t i = 0; i < pm.num_edges && !uses_consumed; ++i) {
                            for (uint8_t j = 0; j < ci->num_consumed; ++j) {
                                if (pm.matched_edges[i] == ci->consumed_edges[j]) {
                                    uses_consumed = true;
                                    break;
                                }
                            }
                        }

                        if (!uses_consumed) {
                            // Match is still valid - update source state
                            MatchRecord fwd = pm;
                            fwd.source_state = state;
                            fwd.canonical_source = canonical;
                            fwd.source_canonical_hash = canonical_hash;
                            forwarded_matches.push_back(fwd);
                        }
                    }

                    // Collect new edges for delta matching
                    for (uint8_t i = 0; i < ci->num_new; ++i) {
                        new_edge_list.push_back(ci->new_edges[i]);
                    }
                }

                job_system_->submit_function(
                    [this, output, &get_edge, &get_signature,
                     state, canonical, canonical_hash, r, max_matches_per_job, job_seed,
                     forwarded = std::move(forwarded_matches),
                     new_edges = std::move(new_edge_list),
                     has_parent]() {
                        if (should_stop_.load(std::memory_order_relaxed)) return;

                        const State& s = hg_->get_state(state);
                        output->reserve(std::min(max_matches_per_job, size_t(128)));

                        std::mt19937 local_rng(job_seed);
                        size_t match_count = 0;

                        auto on_match = [&](uint16_t rule_index, const EdgeId* edges, uint8_t num_edges,
                                           const VariableBinding& binding, StateId /*src*/) {
                            MatchRecord match;
                            match.rule_index = rule_index;
                            match.num_edges = num_edges;
                            match.binding = binding;
                            match.source_state = state;
                            match.canonical_source = canonical;
                            match.source_canonical_hash = canonical_hash;
                            for (uint8_t i = 0; i < num_edges; ++i) {
                                match.matched_edges[i] = edges[i];
                            }

                            // Reservoir sampling
                            if (output->size() < max_matches_per_job) {
                                output->push_back(match);
                            } else {
                                std::uniform_int_distribution<size_t> dist(0, match_count);
                                size_t j = dist(local_rng);
                                if (j < max_matches_per_job) {
                                    (*output)[j] = match;
                                }
                            }
                            match_count++;
                        };

                        if (has_parent && !forwarded.empty()) {
                            // Add forwarded matches first
                            for (const auto& fm : forwarded) {
                                on_match(fm.rule_index, fm.matched_edges, fm.num_edges,
                                        fm.binding, fm.source_state);
                            }

                            // Delta match: only search for matches involving new edges
                            if (!new_edges.empty()) {
                                find_delta_matches(
                                    rules_[r], r, state, s.edges,
                                    hg_->signature_index(), hg_->inverted_index(),
                                    get_edge, get_signature, on_match,
                                    new_edges.data(), static_cast<uint8_t>(new_edges.size())
                                );
                            }
                        } else {
                            // Full pattern matching (initial state or no parent info)
                            find_matches(
                                rules_[r], r, state, s.edges,
                                hg_->signature_index(), hg_->inverted_index(),
                                get_edge, get_signature, on_match
                            );
                        }
                    },
                    EvolutionJobType::MATCH
                );
            }
        }

        job_system_->wait_for_completion();

        // Merge job outputs and store per-state
        size_t job_i = 0;
        for (StateId state : current_states) {
            auto& state_match_vec = next_state_matches[state];
            for (uint16_t r = 0; r < rules_.size(); ++r) {
                for (auto& m : job_outputs[job_i]) {
                    step_matches.push_back(m);
                    state_match_vec.push_back(std::move(m));
                }
                job_i++;
            }
        }

        if (step_matches.empty()) {
            fprintf(stderr, "[uniform_random] Step %zu: no matches, stopping\n", step);
            break;
        }

        // Shuffle for uniform random selection
        std::shuffle(step_matches.begin(), step_matches.end(), rng);

        // Apply matches
        next_states.clear();
        next_child_info.clear();
        size_t matches_tried = 0;
        size_t duplicates_rejected = 0;
        size_t target_states = (matches_per_step == 0) ? SIZE_MAX : matches_per_step;
        size_t max_attempts = target_states * 5;

        for (size_t i = 0; i < step_matches.size() && next_states.size() < target_states && matches_tried < max_attempts; ++i) {
            const auto& match = step_matches[i];
            matches_tried++;

            RewriteResult rr = rewriter_.apply(
                rules_[match.rule_index],
                match.source_state,
                match.matched_edges,
                match.num_edges,
                match.binding,
                static_cast<uint32_t>(step)
            );

            if (rr.raw_state != INVALID_ID) {
                total_rewrites_applied++;
                auto [existing, inserted] = matched_raw_states_.insert_if_absent_waiting(rr.raw_state, true);
                if (inserted) {
                    next_states.push_back(rr.raw_state);
                    total_states_created++;

                    // Record child info for match forwarding
                    ChildInfo ci;
                    ci.parent = match.source_state;
                    ci.num_consumed = match.num_edges;
                    for (uint8_t j = 0; j < match.num_edges && j < 16; ++j) {
                        ci.consumed_edges[j] = match.matched_edges[j];
                    }
                    ci.num_new = std::min(rr.num_produced, uint8_t(16));
                    for (uint8_t j = 0; j < ci.num_new; ++j) {
                        ci.new_edges[j] = rr.produced_edges[j];
                    }
                    next_child_info[rr.raw_state] = ci;
                } else {
                    duplicates_rejected++;
                }
            }
        }

        fprintf(stderr, "[uniform_random] Step %zu: %zu matches, tried %zu, accepted %zu (dup %zu), forwarding %zu\n",
                step, step_matches.size(), matches_tried, next_states.size(), duplicates_rejected, next_child_info.size());

        // Move to next step
        current_states.swap(next_states);
        state_matches.swap(next_state_matches);
        child_info.swap(next_child_info);
    }

    fprintf(stderr, "[uniform_random] Complete: %zu steps, %zu rewrites, %zu states, arena=%.2f MB\n",
            steps, total_rewrites_applied, total_states_created,
            hg_->arena().bytes_allocated() / (1024.0 * 1024.0));

    finalize_evolution();
}

// =============================================================================
// Private Helper Methods
// =============================================================================

void ParallelEvolutionEngine::finalize_evolution() {
    // CRITICAL: Acquire fence to ensure all writes from worker threads are visible
    // This pairs with release semantics of atomic operations in worker threads
    std::atomic_thread_fence(std::memory_order_acquire);

    // Emit visualization event for evolution completion
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
    VIZ_EMIT_EVOLUTION_COMPLETE(
        hg_->num_states(),      // total states
        hg_->num_events(),      // total events
        max_steps_,             // max generation
        hg_->num_states()       // final state count (approximation)
    );
#endif
}

StateId ParallelEvolutionEngine::create_initial_state_only(
    const std::vector<std::vector<VertexId>>& edges
) {
    // Create edges
    std::vector<EdgeId> edge_ids;
    for (const auto& edge : edges) {
        EdgeId eid = hg_->create_edge(edge.data(), static_cast<uint8_t>(edge.size()));
        edge_ids.push_back(eid);
        for (VertexId v : edge) {
            hg_->reserve_vertices(v);
        }
    }

    SparseBitset initial_edge_set;
    for (EdgeId eid : edge_ids) {
        initial_edge_set.set(eid, hg_->arena());
    }

    // Compute canonical hash
    auto [canonical_hash, vertex_cache] = hg_->compute_canonical_hash_incremental(
        initial_edge_set,
        INVALID_ID,  // No parent state
        nullptr, 0,  // No consumed edges
        edge_ids.data(), static_cast<uint8_t>(edge_ids.size())
    );
    auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
        std::move(initial_edge_set), canonical_hash, 0, INVALID_ID);

    // Store cache for the initial state
    hg_->store_state_cache(raw_state, vertex_cache);

    // Create genesis event if enabled
    if (enable_genesis_events_) {
        hg_->create_genesis_event(raw_state, edge_ids.data(), static_cast<uint8_t>(edge_ids.size()));
    }

    // Mark as seen but do NOT submit match task
    matched_raw_states_.insert_if_absent_waiting(raw_state, true);

    return raw_state;
}

StateId ParallelEvolutionEngine::create_and_register_initial_state(
    const std::vector<std::vector<VertexId>>& edges
) {
    // Create edges
    std::vector<EdgeId> edge_ids;
    for (const auto& edge : edges) {
        EdgeId eid = hg_->create_edge(edge.data(), static_cast<uint8_t>(edge.size()));
        edge_ids.push_back(eid);
        for (VertexId v : edge) {
            hg_->reserve_vertices(v);
        }
    }

    SparseBitset initial_edge_set;
    for (EdgeId eid : edge_ids) {
        initial_edge_set.set(eid, hg_->arena());
    }

    // Compute canonical hash
    auto [canonical_hash, vertex_cache] = hg_->compute_canonical_hash_incremental(
        initial_edge_set,
        INVALID_ID,  // No parent state
        nullptr, 0,  // No consumed edges
        edge_ids.data(), static_cast<uint8_t>(edge_ids.size())
    );
    auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
        std::move(initial_edge_set), canonical_hash, 0, INVALID_ID);

    // Store cache for the initial state
    hg_->store_state_cache(raw_state, vertex_cache);

    // Create genesis event if enabled
    if (enable_genesis_events_) {
        hg_->create_genesis_event(raw_state, edge_ids.data(), static_cast<uint8_t>(edge_ids.size()));
    }

    // Mark initial state as matched and submit for pattern matching
    matched_raw_states_.insert_if_absent_waiting(raw_state, true);
    submit_match_task(raw_state, 1);

    return raw_state;
}

LockFreeList<MatchRecord>* ParallelEvolutionEngine::get_or_create_state_matches(StateId state) {
    uint64_t key = state;

    // First, try to look up existing list
    auto result = state_matches_.lookup(key);
    if (result.has_value()) {
        return *result;
    }

    // Need to create - allocate new list from arena
    auto* new_list = hg_->arena().template create<LockFreeList<MatchRecord>>();

    // Try to insert - if another thread beat us, use theirs
    auto [existing, inserted] = state_matches_.insert_if_absent(key, new_list);

    // Return whichever list is now in the map
    return inserted ? new_list : existing;
}

uint64_t ParallelEvolutionEngine::store_match_for_state(
    StateId state,
    MatchRecord& match,
    bool with_fence
) {
    // Get epoch BEFORE storing (for ordering)
    uint64_t epoch = global_epoch_.fetch_add(1, std::memory_order_acq_rel);
    match.storage_epoch = epoch;

    LockFreeList<MatchRecord>* list = get_or_create_state_matches(state);
    list->push(match, hg_->arena());

    // In non-batched (eager) mode, we need a fence after each store to ensure
    // visibility before push_match_to_children runs. In batched mode, we use
    // a single fence after all stores.
    if (with_fence) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    return epoch;
}

LockFreeList<ChildInfo>* ParallelEvolutionEngine::get_or_create_state_children(StateId state) {
    uint64_t key = state;

    // First, try to look up existing list
    auto result = state_children_.lookup(key);
    if (result.has_value()) {
        return *result;
    }

    // Need to create - allocate new list from arena
    auto* new_list = hg_->arena().template create<LockFreeList<ChildInfo>>();

    // Try to insert - if another thread beat us, use theirs
    auto [existing, inserted] = state_children_.insert_if_absent(key, new_list);

    // Return whichever list is now in the map
    return inserted ? new_list : existing;
}

uint64_t ParallelEvolutionEngine::register_child_with_parent(
    StateId parent,
    StateId child,
    const EdgeId* consumed_edges,
    uint8_t num_consumed,
    uint32_t child_step
) {
    if (parent == INVALID_ID) return 0;

    // Build ChildInfo (epoch will be set after push)
    ChildInfo info;
    info.child_state = child;
    info.num_consumed = num_consumed;
    info.creation_step = child_step;  // Step at which child was created
    info.registration_epoch = 0;  // Temporary, will update after push
    for (uint8_t i = 0; i < num_consumed; ++i) {
        info.consumed_edges[i] = consumed_edges[i];
    }

    // Push child to list (for push_match_to_children if called recursively)
    LockFreeList<ChildInfo>* children = get_or_create_state_children(parent);
    children->push(info, hg_->arena());

    // Get epoch (for tracking, though less critical now with batching)
    uint64_t epoch = global_epoch_.fetch_add(1, std::memory_order_acq_rel);

    // Store registration epoch for this child
    state_registration_epoch_.insert_if_absent(child, epoch);

    // Track child's parent (for walking ancestor chain during forward)
    ParentInfo pi_init;
    pi_init.parent_state = parent;
    pi_init.num_consumed = num_consumed;
    for (uint8_t i = 0; i < num_consumed; ++i) {
        pi_init.consumed_edges[i] = consumed_edges[i];
    }
    ParentInfo* parent_info = hg_->arena().template create<ParentInfo>(pi_init);
    state_parent_.insert_if_absent(child, parent_info);

    return epoch;
}

void ParallelEvolutionEngine::push_match_to_children(
    StateId parent,
    const MatchRecord& match,
    uint32_t step
) {
    if (batched_matching_) {
        // With batched matching, no retry loop needed
        push_match_to_children_impl(parent, match, step);
    } else {
        // EAGER MODE: Retry loop to close race window
        uint64_t epoch_before = global_epoch_.load(std::memory_order_acquire);
        push_match_to_children_impl(parent, match, step);

        // Retry if new children may have registered during push
        uint64_t epoch_after = global_epoch_.load(std::memory_order_acquire);
        while (epoch_after != epoch_before) {
            epoch_before = epoch_after;
            push_match_to_children_impl(parent, match, step);
            epoch_after = global_epoch_.load(std::memory_order_acquire);
        }
    }
}

void ParallelEvolutionEngine::push_match_to_children_impl(
    StateId parent,
    const MatchRecord& match,
    [[maybe_unused]] uint32_t step
) {
    auto result = state_children_.lookup_waiting(parent);
    if (!result.has_value()) return;  // No children registered

    LockFreeList<ChildInfo>* children = *result;
    children->for_each([&](const ChildInfo& child_info) {
        // Skip if match overlaps with consumed edges
        if (child_info.match_overlaps_consumed(match.matched_edges, match.num_edges)) {
            stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Create forwarded match with child as source
        MatchRecord forwarded = match;
        forwarded.source_state = child_info.child_state;
        forwarded.canonical_source = hg_->get_canonical_state(child_info.child_state);
        forwarded.source_canonical_hash = hg_->get_state(child_info.child_state).canonical_hash;

        // Deduplicate
        uint64_t h = forwarded.hash();
        auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
        if (!inserted) {
            return;  // Already processed
        }

        // Check if this was a "missing" match that arrived late via push
        if (validate_match_forwarding_) {
            auto missing = missing_match_hashes_.lookup(h);
            if (missing.has_value()) {
                late_arrivals_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

        DEBUG_LOG("PUSH parent=%u -> child=%u rule=%u hash=%lu step=%u epoch=%lu",
                  parent, child_info.child_state, match.rule_index, h, step, match.storage_epoch);

        // Store match in child
        store_match_for_state(child_info.child_state, forwarded);

        // CRITICAL FIX: Use child's MATCH step, not parent's step!
        uint32_t child_step = child_info.creation_step + 1;

        // RECURSIVE: Push to child's existing children (grandchildren)
        push_match_to_children(child_info.child_state, forwarded, child_step);

        // Spawn REWRITE task for this forwarded match
        if (!uniform_random_mode_) {
            submit_rewrite_task(forwarded, child_step);
        }
    });
}

void ParallelEvolutionEngine::forward_existing_parent_matches(
    StateId parent,
    StateId child,
    const EdgeId* consumed_edges,
    uint8_t num_consumed,
    uint32_t step,
    std::vector<MatchRecord>& batch
) {
    // Accumulate all consumed edges along the path from child to ancestors
    EdgeId accumulated_consumed[MAX_PATTERN_EDGES * 8];
    uint8_t total_consumed = 0;
    for (uint8_t i = 0; i < num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
        accumulated_consumed[total_consumed++] = consumed_edges[i];
    }

    // Walk up the ancestor chain, forwarding matches from each ancestor
    StateId current_ancestor = parent;
    while (current_ancestor != INVALID_ID) {
        forward_matches_from_single_ancestor(current_ancestor, child,
                                              accumulated_consumed, total_consumed, step, batch);

        // Move to the next ancestor and accumulate its consumed edges
        auto parent_result = state_parent_.lookup_waiting(current_ancestor);
        if (!parent_result.has_value()) break;

        ParentInfo* pi = *parent_result;
        if (!pi || !pi->has_parent()) break;

        // Add this ancestor's consumed edges to the accumulated set
        for (uint8_t i = 0; i < pi->num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
            accumulated_consumed[total_consumed++] = pi->consumed_edges[i];
        }
        current_ancestor = pi->parent_state;
    }
}

void ParallelEvolutionEngine::forward_matches_from_single_ancestor(
    StateId ancestor,
    StateId child,
    const EdgeId* accumulated_consumed,
    uint8_t total_consumed,
    uint32_t step,
    std::vector<MatchRecord>& batch
) {
    forward_matches_from_single_ancestor_impl(
        ancestor, child, accumulated_consumed, total_consumed, step,
        0,  // child_registration_epoch unused
        batch);
}

void ParallelEvolutionEngine::forward_matches_from_single_ancestor_impl(
    StateId ancestor,
    StateId child,
    const EdgeId* accumulated_consumed,
    uint8_t total_consumed,
    uint32_t /* step */,
    uint64_t /* child_registration_epoch - unused */,
    std::vector<MatchRecord>& batch
) {
    auto result = state_matches_.lookup_waiting(ancestor);
    if (!result.has_value()) return;  // Ancestor has no matches yet

    LockFreeList<MatchRecord>* ancestor_matches = *result;
    ancestor_matches->for_each([&](const MatchRecord& ancestor_match) {
        // Skip if match overlaps with ANY consumed edge along the path
        bool overlaps = false;
        for (uint8_t i = 0; i < ancestor_match.num_edges && !overlaps; ++i) {
            for (uint8_t j = 0; j < total_consumed; ++j) {
                if (ancestor_match.matched_edges[i] == accumulated_consumed[j]) {
                    overlaps = true;
                    break;
                }
            }
        }

        if (overlaps) {
            stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Create forwarded match for child state
        MatchRecord forwarded = ancestor_match;
        forwarded.source_state = child;
        forwarded.canonical_source = hg_->get_canonical_state(child);
        forwarded.source_canonical_hash = hg_->get_state(child).canonical_hash;

        // Deduplicate
        uint64_t h = forwarded.hash();
        auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
        if (!inserted) {
            DEBUG_LOG("FWD_DUP ancestor=%u -> child=%u rule=%u hash=%lu",
                      ancestor, child, ancestor_match.rule_index, h);
            return;  // Already seen
        }

        // Check if this was a "missing" match that arrived late via forward_existing
        if (validate_match_forwarding_) {
            auto missing = missing_match_hashes_.lookup(h);
            if (missing.has_value()) {
                late_arrivals_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

        DEBUG_LOG("FWD ancestor=%u -> child=%u rule=%u hash=%lu epoch=%lu",
                  ancestor, child, ancestor_match.rule_index, h, ancestor_match.storage_epoch);

        // Add to batch
        batch.push_back(forwarded);
    });
}

void ParallelEvolutionEngine::forward_existing_parent_matches_eager(
    StateId parent,
    StateId child,
    const EdgeId* consumed_edges,
    uint8_t num_consumed,
    uint32_t step
) {
    // Accumulate all consumed edges along the path from child to ancestors
    EdgeId accumulated_consumed[MAX_PATTERN_EDGES * 8];
    uint8_t total_consumed = 0;
    for (uint8_t i = 0; i < num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
        accumulated_consumed[total_consumed++] = consumed_edges[i];
    }

    // RETRY LOOP: Pull with epoch tracking to close race window
    uint64_t epoch_before = global_epoch_.load(std::memory_order_acquire);

    // Walk up the ancestor chain, forwarding matches from each ancestor
    StateId current_ancestor = parent;
    while (current_ancestor != INVALID_ID) {
        forward_matches_from_single_ancestor_eager(
            current_ancestor, child,
            accumulated_consumed, total_consumed, step);

        // Move to next ancestor and accumulate consumed edges
        auto parent_result = state_parent_.lookup_waiting(current_ancestor);
        if (!parent_result.has_value()) break;

        ParentInfo* pi = *parent_result;
        if (!pi || !pi->has_parent()) break;

        for (uint8_t i = 0; i < pi->num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
            accumulated_consumed[total_consumed++] = pi->consumed_edges[i];
        }
        current_ancestor = pi->parent_state;
    }

    // Check if epoch changed - if so, retry to catch any new matches
    uint64_t epoch_after = global_epoch_.load(std::memory_order_acquire);
    while (epoch_after != epoch_before) {
        epoch_before = epoch_after;

        // Reset consumed edges and re-walk
        total_consumed = 0;
        for (uint8_t i = 0; i < num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
            accumulated_consumed[total_consumed++] = consumed_edges[i];
        }

        current_ancestor = parent;
        while (current_ancestor != INVALID_ID) {
            forward_matches_from_single_ancestor_eager(
                current_ancestor, child,
                accumulated_consumed, total_consumed, step);

            auto parent_result = state_parent_.lookup_waiting(current_ancestor);
            if (!parent_result.has_value()) break;

            ParentInfo* pi = *parent_result;
            if (!pi || !pi->has_parent()) break;

            for (uint8_t i = 0; i < pi->num_consumed && total_consumed < sizeof(accumulated_consumed)/sizeof(EdgeId); ++i) {
                accumulated_consumed[total_consumed++] = pi->consumed_edges[i];
            }
            current_ancestor = pi->parent_state;
        }

        epoch_after = global_epoch_.load(std::memory_order_acquire);
    }
}

void ParallelEvolutionEngine::forward_matches_from_single_ancestor_eager(
    StateId ancestor,
    StateId child,
    const EdgeId* accumulated_consumed,
    uint8_t total_consumed,
    uint32_t step
) {
    auto result = state_matches_.lookup_waiting(ancestor);
    if (!result.has_value()) return;

    // Build consumed set once for O(1) amortized overlap checks
    std::unordered_set<EdgeId> consumed_set(accumulated_consumed, accumulated_consumed + total_consumed);

    LockFreeList<MatchRecord>* ancestor_matches = *result;
    ancestor_matches->for_each([&](const MatchRecord& ancestor_match) {
        // Skip if match overlaps with ANY consumed edge - O(n) vs O(n*m)
        bool overlaps = false;
        for (uint8_t i = 0; i < ancestor_match.num_edges && !overlaps; ++i) {
            if (consumed_set.count(ancestor_match.matched_edges[i])) {
                overlaps = true;
            }
        }

        if (overlaps) {
            stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Create forwarded match for child state
        MatchRecord forwarded = ancestor_match;
        forwarded.source_state = child;
        forwarded.canonical_source = hg_->get_canonical_state(child);
        forwarded.source_canonical_hash = hg_->get_state(child).canonical_hash;

        // Deduplicate - seen_match_hashes_ protects against both push and pull duplicates
        uint64_t h = forwarded.hash();
        auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
        if (!inserted) {
            DEBUG_LOG("FWD_EAGER_DUP ancestor=%u -> child=%u rule=%u hash=%lu",
                      ancestor, child, ancestor_match.rule_index, h);
            return;  // Already seen (possibly via push)
        }

        // Check if this was a "missing" match that arrived late
        if (validate_match_forwarding_) {
            auto missing = missing_match_hashes_.lookup(h);
            if (missing.has_value()) {
                late_arrivals_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

        DEBUG_LOG("FWD_EAGER ancestor=%u -> child=%u rule=%u hash=%lu step=%u",
                  ancestor, child, ancestor_match.rule_index, h, step);

        // EAGER: Immediately spawn REWRITE task
        if (!uniform_random_mode_) {
            submit_rewrite_task(forwarded, step);
        }
    });
}

// =============================================================================
// Task Submission
// =============================================================================

void ParallelEvolutionEngine::submit_match_task(StateId state, uint32_t step) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && step > max_steps_) return;
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(state)) return;

    DEBUG_LOG("SUBMIT_MATCH state=%u step=%u (full)", state, step);

    auto job = job_system::make_job<EvolutionJobType>(
        [this, state, step]() {
            execute_match_task(state, step, MatchContext{});
        },
        EvolutionJobType::MATCH
    );
    job_system_->submit(std::move(job));
}

void ParallelEvolutionEngine::submit_match_task_with_context(
    StateId state,
    uint32_t step,
    const MatchContext& ctx
) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && step > max_steps_) return;
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(state)) return;

    DEBUG_LOG("SUBMIT_MATCH state=%u step=%u parent=%u produced=%u consumed=%u (delta)",
              state, step, ctx.parent_state, ctx.num_produced, ctx.num_consumed);

    auto job = job_system::make_job<EvolutionJobType>(
        [this, state, step, ctx]() {
            execute_match_task(state, step, ctx);
        },
        EvolutionJobType::MATCH
    );
    job_system_->submit(std::move(job));
}

void ParallelEvolutionEngine::submit_rewrite_task(const MatchRecord& match, uint32_t step) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && step > max_steps_) return;
    // Early check (non-reserving) - execute_rewrite_task does the actual atomic reservation
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(match.source_state)) return;

    DEBUG_LOG("SUBMIT_REWRITE state=%u rule=%u step=%u", match.source_state, match.rule_index, step);

    auto job = job_system::make_job<EvolutionJobType>(
        [this, match, step]() {
            execute_rewrite_task(match, step);
        },
        EvolutionJobType::REWRITE
    );
    job_system_->submit(std::move(job));
}

void ParallelEvolutionEngine::submit_scan_task(const ScanTaskData& data) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && data.step > max_steps_) return;
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("SUBMIT_SCAN state=%u rule=%u step=%u delta=%d",
              data.state, data.rule_index, data.step, data.is_delta);

    auto job = job_system::make_job<EvolutionJobType>(
        [this, data]() {
            execute_scan_task(data);
        },
        EvolutionJobType::SCAN
    );
    // SCAN tasks use FIFO - start broadly, then depth-first via EXPAND
    job_system_->submit(std::move(job));
}

void ParallelEvolutionEngine::submit_expand_task(const ExpandTaskData& data) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && data.step > max_steps_) return;
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("SUBMIT_EXPAND state=%u rule=%u matched=%u/%u step=%u",
              data.state, data.rule_index, data.num_matched, data.num_pattern_edges, data.step);

    auto job = job_system::make_job<EvolutionJobType>(
        [this, data]() {
            execute_expand_task(data);
        },
        EvolutionJobType::EXPAND
    );
    // LIFO scheduling: depth-first traversal, bounded memory O(|E(q)|² × |E(H)|)
    job_system_->submit(std::move(job), job_system::ScheduleMode::LIFO);
}

void ParallelEvolutionEngine::submit_sink_task(const ExpandTaskData& data) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && data.step > max_steps_) return;
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("SUBMIT_SINK state=%u rule=%u matched=%u step=%u",
              data.state, data.rule_index, data.num_matched, data.step);

    auto job = job_system::make_job<EvolutionJobType>(
        [this, data]() {
            execute_sink_task(data);
        },
        EvolutionJobType::SINK
    );
    job_system_->submit(std::move(job));
}

// =============================================================================
// Pruning Helpers
// =============================================================================

bool ParallelEvolutionEngine::can_create_states_at_step(uint32_t step) const {
    if (max_states_per_step_ == 0) return true;

    auto result = states_per_step_.lookup(step);
    if (!result.has_value()) return true;

    return (*result)->load(std::memory_order_relaxed) < max_states_per_step_;
}

bool ParallelEvolutionEngine::can_have_more_children(StateId parent) const {
    if (max_successor_states_per_parent_ == 0) return true;

    auto result = parent_successor_count_.lookup(parent);
    if (!result.has_value()) return true;

    return (*result)->load(std::memory_order_relaxed) < max_successor_states_per_parent_;
}

bool ParallelEvolutionEngine::try_reserve_successor_slot(StateId parent) {
    if (max_successor_states_per_parent_ == 0) return true;

    // Get or create atomic counter for this parent
    uint64_t key = parent;
    auto result = parent_successor_count_.lookup(key);
    std::atomic<size_t>* counter = nullptr;

    if (result.has_value()) {
        counter = *result;
    } else {
        // Allocate new counter from arena
        counter = hg_->arena().template create<std::atomic<size_t>>(0);
        auto [existing, inserted] = parent_successor_count_.insert_if_absent(key, counter);
        if (!inserted) {
            counter = existing;  // Another thread beat us
        }
    }

    // Try to increment, fail if at limit
    size_t old_val = counter->fetch_add(1, std::memory_order_relaxed);
    if (old_val >= max_successor_states_per_parent_) {
        counter->fetch_sub(1, std::memory_order_relaxed);  // Rollback
        return false;
    }
    return true;
}

bool ParallelEvolutionEngine::try_reserve_step_slot(uint32_t step) {
    if (max_states_per_step_ == 0) return true;  // Unlimited

    // Get or create atomic counter for this step
    uint64_t key = step;
    auto result = states_per_step_.lookup(key);
    std::atomic<size_t>* counter = nullptr;

    if (result.has_value()) {
        counter = *result;
    } else {
        // Allocate new counter from arena
        counter = hg_->arena().template create<std::atomic<size_t>>(0);
        auto [existing, inserted] = states_per_step_.insert_if_absent(key, counter);
        if (!inserted) {
            counter = existing;  // Another thread beat us
        }
    }

    // Try to increment, fail if at limit
    size_t old_val = counter->fetch_add(1, std::memory_order_relaxed);
    if (old_val >= max_states_per_step_) {
        counter->fetch_sub(1, std::memory_order_relaxed);  // Rollback
        return false;
    }
    return true;
}

void ParallelEvolutionEngine::release_step_slot(uint32_t step) {
    if (max_states_per_step_ == 0) return;  // Unlimited, nothing to release

    auto result = states_per_step_.lookup(step);
    if (result.has_value()) {
        (*result)->fetch_sub(1, std::memory_order_relaxed);
    }
}

bool ParallelEvolutionEngine::should_explore() {
    if (exploration_probability_ >= 1.0) return true;
    if (exploration_probability_ <= 0.0) return false;

    // Thread-local random number generation
    thread_local std::mt19937 rng(std::random_device{}());
    thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

    return dist(rng) < exploration_probability_;
}

std::vector<uint16_t> ParallelEvolutionEngine::get_shuffled_rule_indices() const {
    std::vector<uint16_t> indices(rules_.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Thread-local random number generation for shuffling
    thread_local std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    return indices;
}

// =============================================================================
// REWRITE Task Execution
// =============================================================================

void ParallelEvolutionEngine::execute_rewrite_task(const MatchRecord& match, uint32_t step) {
    if (should_stop_.load(std::memory_order_relaxed)) return;

    // Check step limit - don't spawn REWRITEs past max_steps
    if (max_steps_ > 0 && step > max_steps_) return;

    // Check limits before applying
    if (max_states_ > 0 && hg_->num_states() >= max_states_) {
        should_stop_.store(true, std::memory_order_relaxed);
        return;
    }
    if (max_events_ > 0 && hg_->num_events() >= max_events_) {
        should_stop_.store(true, std::memory_order_relaxed);
        return;
    }

    // Pruning: check max_successor_states_per_parent
    if (!try_reserve_successor_slot(match.source_state)) {
        return;  // Parent has too many children already
    }

    // Pruning: check max_states_per_step (child will be at step+1)
    if (!try_reserve_step_slot(step + 1)) {
        return;  // Too many states at this generation
    }

    const RewriteRule& rule = rules_[match.rule_index];

    // Apply the rewrite
    RewriteResult rr = rewriter_.apply(
        rule,
        match.source_state,
        match.matched_edges,
        match.num_edges,
        match.binding,
        step
    );

    if (rr.new_state == INVALID_ID) {
        // Rewrite failed - release the reserved slot
        release_step_slot(step + 1);
        return;
    }

    total_rewrites_.fetch_add(1, std::memory_order_relaxed);
    total_events_.fetch_add(1, std::memory_order_relaxed);

    if (rr.was_new_state) {
        total_new_states_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Duplicate state - release the reserved slot (only count unique states)
        release_step_slot(step + 1);
    }

    // Emit visualization events for canonical states only
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
    if (rr.was_new_state) {
        // Emit StateCreated only for new canonical states
        const auto& state_data = hg_->get_state(rr.new_state);
        VIZ_EMIT_STATE_CREATED(
            rr.new_state,             // state id (canonical)
            match.source_state,       // parent state id
            step + 1,                 // generation
            state_data.edges.count(), // edge count
            0                         // vertex count (not tracked)
        );
        // Emit hyperedge data for each edge in the new state
        uint32_t edge_idx = 0;
        state_data.edges.for_each([&](EdgeId eid) {
            const Edge& edge = hg_->get_edge(eid);
            VIZ_EMIT_HYPEREDGE(rr.new_state, edge_idx++, edge.vertices, edge.arity);
        });
    }
    // Emit RewriteApplied for ALL events
    VIZ_EMIT_REWRITE_APPLIED(
        match.source_state,       // source state
        rr.new_state,             // target state (canonical)
        match.rule_index,         // rule index
        rr.event,                 // raw event id (for tracking)
        rr.canonical_event,       // canonical event id (for deduplication)
        match.num_edges,          // destroyed edges count
        rr.num_produced           // created edges count
    );
#endif

    // Spawn MATCH task for the new raw state if it hasn't been matched yet
    auto [existing, inserted] = matched_raw_states_.insert_if_absent_waiting(
        rr.raw_state, true);

    if (inserted) {
        DEBUG_LOG("STATE parent=%u -> child=%u (canonical=%u) rule=%u step=%u new=%d",
                  match.source_state, rr.raw_state, rr.new_state, match.rule_index, step, rr.was_new_state);

        // Build MatchContext for match forwarding
        MatchContext ctx;
        ctx.parent_state = match.source_state;
        ctx.num_consumed = match.num_edges;
        for (uint8_t i = 0; i < match.num_edges; ++i) {
            ctx.consumed_edges[i] = match.matched_edges[i];
        }
        ctx.num_produced = rr.num_produced;
        for (uint8_t i = 0; i < rr.num_produced; ++i) {
            ctx.produced_edges[i] = rr.produced_edges[i];
        }

        // Pruning: check exploration_probability (v1 style)
        if (!should_explore()) {
            return;
        }

        // Exploration deduplication: only explore from canonical representatives
        if (explore_from_canonical_states_only_ && !rr.was_new_state) {
            return;
        }

        // Register child's parent pointer for ancestor chain walking
        if (enable_match_forwarding_) {
            register_child_with_parent(
                match.source_state, rr.raw_state,
                match.matched_edges, match.num_edges,
                step);
        }

        // In uniform random mode, push to pending list for main loop to collect
        if (uniform_random_mode_ && pending_new_states_simple_) {
            pending_new_states_simple_->push(rr.raw_state, hg_->arena());
        } else {
            // Submit MATCH task with context for match forwarding
            submit_match_task_with_context(rr.raw_state, step + 1, ctx);
        }
    }
}

// =============================================================================
// MATCH Task Execution
// =============================================================================

void ParallelEvolutionEngine::execute_match_task(
    StateId state,
    uint32_t step,
    const MatchContext& ctx
) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && step > max_steps_) return;

    // Early exit if rewrites are impossible due to limits
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(state)) return;

    const State& s = hg_->get_state(state);

    // Get canonical state for deterministic deduplication
    StateId canonical_state = hg_->get_canonical_state(state);

    // Edge accessor
    auto get_edge = [this](EdgeId eid) -> const Edge& {
        return hg_->get_edge(eid);
    };

    // Signature accessor (cached signatures for O(1) lookup)
    auto get_signature = [this](EdgeId eid) -> const EdgeSignature& {
        return hg_->edge_signature(eid);
    };

    // Batched matching: collect all matches first, then spawn REWRITEs
    std::vector<MatchRecord> batch;
    batch.reserve(32);
    size_t delta_start = 0;  // Index where delta (discovered) matches start

    // Collector callback
    auto collect_match = [&, state, canonical_state](
        uint16_t rule_index,
        const EdgeId* edges,
        uint8_t num_edges,
        const VariableBinding& binding
    ) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        MatchRecord match;
        match.rule_index = rule_index;
        match.num_edges = num_edges;
        match.binding = binding;
        match.source_state = state;
        match.canonical_source = canonical_state;
        match.source_canonical_hash = s.canonical_hash;
        for (uint8_t i = 0; i < num_edges; ++i) {
            match.matched_edges[i] = edges[i];
        }

        // Deduplicate
        uint64_t h = match.hash();
        auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
        if (!inserted) {
            rejected_duplicates_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.new_matches_discovered.fetch_add(1, std::memory_order_relaxed);

        DEBUG_LOG("NEW state=%u rule=%u hash=%lu step=%u", state, rule_index, h, step);

        if (batched_matching_) {
            batch.push_back(match);
        } else {
            if (enable_match_forwarding_) {
                store_match_for_state(state, match, true);
                push_match_to_children(state, match, step);
            }
            if (!uniform_random_mode_) {
                submit_rewrite_task(match, step);
            }
        }
    };

    // Match callback for pattern matching
    auto on_match = [&](
        uint16_t rule_index,
        const EdgeId* edges,
        uint8_t num_edges,
        const VariableBinding& binding,
        StateId /*source_state*/
    ) {
        collect_match(rule_index, edges, num_edges, binding);
    };

    if (enable_match_forwarding_ && ctx.has_parent()) {
        // DELTA MATCHING MODE (child state)
        stats_.delta_pattern_matches.fetch_add(1, std::memory_order_relaxed);

        if (batched_matching_) {
            forward_existing_parent_matches(
                ctx.parent_state, state,
                ctx.consumed_edges, ctx.num_consumed, step, batch);
        } else {
            forward_existing_parent_matches_eager(
                ctx.parent_state, state,
                ctx.consumed_edges, ctx.num_consumed, step);
        }

        delta_start = batch.size();

        if (task_based_matching_) {
            // Task-based delta matching: spawn SCAN tasks for each rule
            // Shuffle rule order to mitigate bias in pruning modes
            auto shuffled_rules = get_shuffled_rule_indices();
            for (uint16_t r : shuffled_rules) {
                ScanTaskData scan_data;
                scan_data.state = state;
                scan_data.rule_index = r;
                scan_data.step = step;
                scan_data.canonical_state = canonical_state;
                scan_data.source_canonical_hash = s.canonical_hash;
                scan_data.is_delta = true;
                scan_data.num_produced = ctx.num_produced;
                for (uint8_t i = 0; i < ctx.num_produced; ++i) {
                    scan_data.produced_edges[i] = ctx.produced_edges[i];
                }
                submit_scan_task(scan_data);
            }
            // Spawn REWRITEs for forwarded matches
            if (!uniform_random_mode_) {
                for (size_t i = 0; i < delta_start; ++i) {
                    submit_rewrite_task(batch[i], step);
                }
            }
            return;
        }

        // Synchronous delta matching
        DEBUG_LOG("SYNC_DELTA_MATCH state=%u step=%u rules=%zu produced=%u",
                  state, step, rules_.size(), ctx.num_produced);
        for (uint16_t r = 0; r < rules_.size(); ++r) {
            find_delta_matches(
                rules_[r], r, state, s.edges,
                hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature, on_match,
                ctx.produced_edges, ctx.num_produced
            );
        }

        // VALIDATION: Compare forwarded+delta vs full matching
        if (validate_match_forwarding_) {
            size_t missing = 0;
            auto count_missing = [&, state, canonical_state](
                uint16_t rule_index,
                const EdgeId* edges,
                uint8_t num_edges,
                const VariableBinding& binding,
                StateId /*source_state*/
            ) {
                MatchRecord match;
                match.rule_index = rule_index;
                match.num_edges = num_edges;
                match.binding = binding;
                match.source_state = state;
                match.canonical_source = canonical_state;
                match.source_canonical_hash = s.canonical_hash;
                for (uint8_t i = 0; i < num_edges; ++i) {
                    match.matched_edges[i] = edges[i];
                }
                uint64_t h = match.hash();
                if (!seen_match_hashes_.contains(h)) {
                    ++missing;
                    uint64_t debug_info = (static_cast<uint64_t>(state) << 16) | rule_index;
                    missing_match_hashes_.insert_if_absent(h, debug_info);
                }
            };
            for (uint16_t r = 0; r < rules_.size(); ++r) {
                find_matches(
                    rules_[r], r, state, s.edges,
                    hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature, count_missing
                );
            }
            if (missing > 0) {
                validation_mismatches_.fetch_add(missing, std::memory_order_relaxed);
            }
        }
    } else {
        // FULL MATCHING MODE (initial state or forwarding disabled)
        stats_.full_pattern_matches.fetch_add(1, std::memory_order_relaxed);

        if (task_based_matching_) {
            // Task-based matching: spawn SCAN tasks for each rule
            // Shuffle rule order to mitigate bias in pruning modes
            auto shuffled_rules = get_shuffled_rule_indices();
            for (uint16_t r : shuffled_rules) {
                ScanTaskData scan_data;
                scan_data.state = state;
                scan_data.rule_index = r;
                scan_data.step = step;
                scan_data.canonical_state = canonical_state;
                scan_data.source_canonical_hash = s.canonical_hash;
                scan_data.is_delta = false;
                scan_data.num_produced = 0;
                submit_scan_task(scan_data);
            }
            return;
        }

        // Synchronous matching
        DEBUG_LOG("SYNC_MATCH state=%u step=%u rules=%zu",
                  state, step, rules_.size());
        for (uint16_t r = 0; r < rules_.size(); ++r) {
            find_matches(
                rules_[r], r, state, s.edges,
                hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature, on_match
            );
        }
    }

    // Phase 2: Store all matches, then spawn all REWRITEs (BATCHED MODE ONLY)
    if (batched_matching_ || uniform_random_mode_) {
        if (uniform_random_mode_ && pending_matches_) {
            for (const auto& match : batch) {
                pending_matches_->push(match, hg_->arena());
            }
        } else if (enable_match_forwarding_) {
            for (size_t i = delta_start; i < batch.size(); ++i) {
                store_match_for_state(state, batch[i]);
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }

        if (!uniform_random_mode_) {
            for (const auto& match : batch) {
                submit_rewrite_task(match, step);
            }
        }
    }
}

// =============================================================================
// SCAN Task Execution
// =============================================================================

void ParallelEvolutionEngine::execute_scan_task(const ScanTaskData& data) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && data.step > max_steps_) return;

    // Early exit if rewrites are impossible due to limits
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("EXEC_SCAN state=%u rule=%u step=%u delta=%d",
              data.state, data.rule_index, data.step, data.is_delta);

    const State& s = hg_->get_state(data.state);
    const RewriteRule& rule = rules_[data.rule_index];

    if (rule.num_lhs_edges == 0) return;

    // Edge accessor
    auto get_edge = [this](EdgeId eid) -> const Edge& {
        return hg_->get_edge(eid);
    };

    // Signature accessor
    auto get_signature = [this](EdgeId eid) -> const EdgeSignature& {
        return hg_->edge_signature(eid);
    };

    // Pre-compute pattern signatures
    EdgeSignature pattern_sigs[MAX_PATTERN_EDGES];
    CompatibleSignatureCache sig_caches[MAX_PATTERN_EDGES];
    for (uint8_t i = 0; i < rule.num_lhs_edges; ++i) {
        pattern_sigs[i] = rule.lhs[i].signature();
        sig_caches[i] = CompatibleSignatureCache::from_pattern(pattern_sigs[i]);
    }

    if (data.is_delta) {
        // Delta matching: start from produced edges
        for (uint8_t p = 0; p < data.num_produced; ++p) {
            EdgeId produced = data.produced_edges[p];
            if (!s.edges.contains(produced)) continue;

            // Try this produced edge at each pattern position
            for (uint8_t pos = 0; pos < rule.num_lhs_edges; ++pos) {
                if (should_stop_.load(std::memory_order_relaxed)) return;

                const PatternEdge& pattern_edge = rule.lhs[pos];
                const auto& edge = get_edge(produced);

                // Check signature compatibility
                const EdgeSignature& data_sig = get_signature(produced);
                if (!signature_compatible(data_sig, pattern_sigs[pos])) continue;

                // Validate candidate
                VariableBinding binding;
                if (!validate_candidate(edge.vertices, edge.arity, pattern_edge, binding)) continue;

                // Create EXPAND task data
                ExpandTaskData expand_data;
                expand_data.state = data.state;
                expand_data.rule_index = data.rule_index;
                expand_data.num_pattern_edges = rule.num_lhs_edges;
                expand_data.next_pattern_idx = 0;
                expand_data.matched_edges[0] = produced;
                expand_data.match_order[0] = pos;
                expand_data.num_matched = 1;
                expand_data.binding = binding;
                expand_data.step = data.step;
                expand_data.canonical_state = data.canonical_state;
                expand_data.source_canonical_hash = data.source_canonical_hash;

                if (rule.num_lhs_edges == 1) {
                    submit_sink_task(expand_data);
                } else {
                    submit_expand_task(expand_data);
                }
            }
        }
    } else {
        // Full matching: start from first pattern edge
        const PatternEdge& first_edge = rule.lhs[0];
        const EdgeSignature& first_sig = pattern_sigs[0];
        const CompatibleSignatureCache& first_cache = sig_caches[0];

        // Generate candidates for first edge
        generate_candidates(
            first_edge, first_sig, first_cache,
            VariableBinding{}, s.edges,
            hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature,
            [&](EdgeId candidate) {
                if (should_stop_.load(std::memory_order_relaxed)) return;

                // Validate candidate
                const auto& edge = get_edge(candidate);
                VariableBinding binding;
                if (!validate_candidate(edge.vertices, edge.arity, first_edge, binding)) return;

                // Create EXPAND task data
                ExpandTaskData expand_data;
                expand_data.state = data.state;
                expand_data.rule_index = data.rule_index;
                expand_data.num_pattern_edges = rule.num_lhs_edges;
                expand_data.next_pattern_idx = 1;
                expand_data.matched_edges[0] = candidate;
                expand_data.match_order[0] = 0;
                expand_data.num_matched = 1;
                expand_data.binding = binding;
                expand_data.step = data.step;
                expand_data.canonical_state = data.canonical_state;
                expand_data.source_canonical_hash = data.source_canonical_hash;

                if (rule.num_lhs_edges == 1) {
                    submit_sink_task(expand_data);
                } else {
                    submit_expand_task(expand_data);
                }
            }
        );
    }
}

// =============================================================================
// EXPAND Task Execution
// =============================================================================

void ParallelEvolutionEngine::execute_expand_task(const ExpandTaskData& data) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && data.step > max_steps_) return;

    // Early exit if rewrites are impossible due to limits
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("EXEC_EXPAND state=%u rule=%u matched=%u/%u step=%u",
              data.state, data.rule_index, data.num_matched, data.num_pattern_edges, data.step);

    // Check if complete (shouldn't happen, but safety check)
    if (data.is_complete()) {
        submit_sink_task(data);
        return;
    }

    const State& s = hg_->get_state(data.state);
    const RewriteRule& rule = rules_[data.rule_index];

    // Edge accessor
    auto get_edge = [this](EdgeId eid) -> const Edge& {
        return hg_->get_edge(eid);
    };

    // Signature accessor
    auto get_signature = [this](EdgeId eid) -> const EdgeSignature& {
        return hg_->edge_signature(eid);
    };

    // Get next pattern edge to match
    uint8_t pattern_idx = data.get_next_pattern_idx();
    if (pattern_idx >= rule.num_lhs_edges) return;

    const PatternEdge& pattern_edge = rule.lhs[pattern_idx];
    EdgeSignature pattern_sig = pattern_edge.signature();
    CompatibleSignatureCache sig_cache = CompatibleSignatureCache::from_pattern(pattern_sig);

    // Generate candidates
    generate_candidates(
        pattern_edge, pattern_sig, sig_cache,
        data.binding, s.edges,
        hg_->signature_index(), hg_->inverted_index(), get_edge, get_signature,
        [&](EdgeId candidate) {
            if (should_stop_.load(std::memory_order_relaxed)) return;

            // Skip if already matched
            if (data.contains_edge(candidate)) return;

            // Validate candidate
            const auto& edge = get_edge(candidate);
            VariableBinding extended = data.binding;
            if (!validate_candidate(edge.vertices, edge.arity, pattern_edge, extended)) return;

            // Create new EXPAND task with extended match
            ExpandTaskData new_data = data;
            new_data.matched_edges[new_data.num_matched] = candidate;
            new_data.match_order[new_data.num_matched] = pattern_idx;
            new_data.num_matched++;
            new_data.binding = extended;

            if (new_data.is_complete()) {
                submit_sink_task(new_data);
            } else {
                submit_expand_task(new_data);
            }
        }
    );
}

// =============================================================================
// SINK Task Execution
// =============================================================================

void ParallelEvolutionEngine::execute_sink_task(const ExpandTaskData& data) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (max_steps_ > 0 && data.step > max_steps_) return;

    // Early exit if rewrites are impossible due to limits
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("EXEC_SINK state=%u rule=%u matched=%u step=%u",
              data.state, data.rule_index, data.num_matched, data.step);

    // Convert matched edges to pattern order
    EdgeId edges_in_order[MAX_PATTERN_EDGES];
    data.to_pattern_order(edges_in_order);

    // Build MatchRecord
    MatchRecord match;
    match.rule_index = data.rule_index;
    match.num_edges = data.num_pattern_edges;
    match.binding = data.binding;
    match.source_state = data.state;
    match.canonical_source = data.canonical_state;
    match.source_canonical_hash = data.source_canonical_hash;
    for (uint8_t i = 0; i < data.num_pattern_edges; ++i) {
        match.matched_edges[i] = edges_in_order[i];
    }

    // Deduplicate using lock-free ConcurrentMap
    uint64_t h = match.hash();
    auto [existing, inserted] = seen_match_hashes_.insert_if_absent_waiting(h, true);
    if (!inserted) {
        rejected_duplicates_.fetch_add(1, std::memory_order_relaxed);
        return;  // Already seen
    }

    total_matches_found_.fetch_add(1, std::memory_order_relaxed);
    stats_.new_matches_discovered.fetch_add(1, std::memory_order_relaxed);

    DEBUG_LOG("SINK state=%u rule=%u hash=%lu step=%u", data.state, data.rule_index, h, data.step);

    // Store match for collection/forwarding
    if (uniform_random_mode_ && pending_matches_) {
        pending_matches_->push(match, hg_->arena());
    } else if (enable_match_forwarding_) {
        store_match_for_state(data.state, match, true);
        push_match_to_children(data.state, match, data.step);
    }

    // Spawn REWRITE task (unless uniform random mode)
    if (!uniform_random_mode_) {
        submit_rewrite_task(match, data.step);
    }
}

}  // namespace hypergraph
