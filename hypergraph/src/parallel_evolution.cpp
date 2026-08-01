// parallel_evolution.cpp - Implementation of ParallelEvolutionEngine class

#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
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
    // Route every engine map's table storage through the hypergraph arena (no malloc,
    // no per-map heap contention). These maps are append-only across the engine's
    // lifetime; re-homing here is single-threaded setup, before any task runs.
    ConcurrentHeterogeneousArena* arena = &hg_->arena();
    seen_match_hashes_.set_arena(arena);
    matched_raw_states_.set_arena(arena);
    state_matches_.set_arena(arena);
    state_children_.set_arena(arena);
    state_parent_.set_arena(arena);
    missing_match_hashes_.set_arena(arena);
    parent_successor_count_.set_arena(arena);
    states_per_step_.set_arena(arena);
    match_join_.set_arena(arena);

    job_system_ = std::make_unique<job_system::JobSystem<EvolutionJobType>>(num_threads_);
    // Recycle each worker's scratch arena after every job — temporaries allocated
    // during a task are reclaimed in bulk, keeping malloc off the hot path.
    job_system_->set_on_job_complete([] { worker_scratch().reset(); });
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
    configure_identity_and_quotient();
    // New run: re-seed the per-thread sampling RNGs from random_seed_.
    sampling_generation_.fetch_add(1, std::memory_order_relaxed);

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

    // Create or get the canonical state (canonical hash computed inside, mode-aware).
    auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
        std::move(initial_edge_set), 0, INVALID_ID);

    // Seed the quotient causal reconstruction at this root (depth 0). The reconstruction runs
    // whenever quotient_causal is on -- which includes full capture under Automatic identity --
    // so the seed follows that switch, not the exploration strategy.
    if (hg_->quotient_causal())
        hg_->quotient_causal_seed(
            canonical_state, static_cast<int>(std::min<size_t>(steps, std::numeric_limits<int>::max())));

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

    // Under quotient exploration the initial state sits at depth zero and is expanded here.
    if (explore_from_canonical_states_only_) {
        hg_->try_lower_explore_depth(canonical_state, 0);
        hg_->try_claim_expanded(canonical_state);
    }

    // Submit MATCH task for initial state - this kicks off the dataflow
    submit_match_task(raw_state, 1);

    // Single synchronization point at the end
    job_system_->wait_for_completion();
    raise_worker_error();

    finalize_evolution();
}

void ParallelEvolutionEngine::evolve(
    const std::vector<std::vector<std::vector<VertexId>>>& initial_states,
    size_t steps
) {
    if (!hg_ || rules_.empty() || initial_states.empty()) return;

    max_steps_ = steps;
    should_stop_.store(false, std::memory_order_relaxed);
    configure_identity_and_quotient();
    // New run: re-seed the per-thread sampling RNGs from random_seed_.
    sampling_generation_.fetch_add(1, std::memory_order_relaxed);

    // Create all initial states - they will all be explored
    for (const auto& state_edges : initial_states) {
        create_and_register_initial_state(state_edges);
    }

    // Single synchronization point at the end
    job_system_->wait_for_completion();
    raise_worker_error();

    finalize_evolution();
}


// =============================================================================
// Private Helper Methods
// =============================================================================

void ParallelEvolutionEngine::raise_worker_error() const {
    switch (last_error()) {
        case job_system::ErrorType::None:
        case job_system::ErrorType::Aborted:
            return;
        default:
            throw std::runtime_error(
                std::string("evolution failed in a worker: ")
                + job_system_->get_error_description()
                + " (the graph is truncated at the point of failure)");
    }
}

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

    // Create or get the canonical state (canonical hash computed inside, mode-aware).
    auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
        std::move(initial_edge_set), 0, INVALID_ID);

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

    // Create or get the canonical state (canonical hash computed inside, mode-aware).
    auto [canonical_state, raw_state, was_new] = hg_->create_or_get_canonical_state(
        std::move(initial_edge_set), 0, INVALID_ID);

    // Create genesis event if enabled
    if (enable_genesis_events_) {
        hg_->create_genesis_event(raw_state, edge_ids.data(), static_cast<uint8_t>(edge_ids.size()));
    }

    // Seed the quotient causal reconstruction at this root (depth 0). The reconstruction runs
    // whenever quotient_causal is on -- which includes full capture under Automatic identity --
    // so the seed follows that switch, not the exploration strategy.
    if (hg_->quotient_causal())
        hg_->quotient_causal_seed(
            canonical_state, static_cast<int>(std::min<size_t>(max_steps_, std::numeric_limits<int>::max())));

    // Mark initial state as matched and submit for pattern matching
    matched_raw_states_.insert_if_absent_waiting(raw_state, true);

    // Under quotient exploration every initial state sits at depth zero. The claim
    // succeeds for the first root of each canonical class; when quotienting the
    // initial states, isomorphic later roots are not expanded (they collapse into
    // the first). Default keeps every provided root as a distinct entry point.
    bool expand_root = true;
    if (explore_from_canonical_states_only_) {
        hg_->try_lower_explore_depth(canonical_state, 0);
        bool first = hg_->try_claim_expanded(canonical_state);
        if (quotient_initial_states_) expand_root = first;
    }

    if (expand_root) {
        submit_match_task(raw_state, 1);
    }

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
    // Publish the match, THEN bump the rendezvous epoch. Pulls re-walk their
    // ancestor chain while match_epoch_ keeps changing; pushing before the bump
    // guarantees that a pull woken by this bump also observes the pushed match.
    // (Bumping first opens a window where the pull re-reads the list before the
    // push lands, sees no further epoch change, and quiesces without the match.)
    LockFreeList<MatchRecord>* list = get_or_create_state_matches(state);
    list->push(match, hg_->arena());

    // In non-batched (eager) mode, we need a fence after each store to ensure
    // visibility before push_match_to_children runs. In batched mode, we use
    // a single fence after all stores.
    if (with_fence) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    return match_epoch_.fetch_add(1, std::memory_order_acq_rel);
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

    // Publish the child's parent link (ancestor-chain data for pulls) BEFORE the
    // child becomes visible in the parent's children list. Ordering invariant: the
    // pull's ancestor walk treats an absent state_parent_ entry as "reached a root"
    // and stops. Once the child is push-visible below, a forwarded match can create a
    // GRANDCHILD on another worker whose pull walks up through this child; if this
    // link were published after the push (as a later step), that walk could find the
    // link absent, silently truncate, and permanently miss every match stored only in
    // higher ancestors (pulled matches are not re-stored in descendants and pushes
    // are one-shot at discovery). Publishing the link first makes "absent" mean
    // "root" -- every reachable ancestor's link is visible to any walk that can
    // reach it (the insert happens-before the child's visibility, which
    // happens-before any descendant's existence).
    ParentInfo pi_init;
    pi_init.parent_state = parent;
    pi_init.num_consumed = num_consumed;
    for (uint8_t i = 0; i < num_consumed; ++i) {
        pi_init.consumed_edges[i] = consumed_edges[i];
    }
    ParentInfo* parent_info = hg_->arena().template create<ParentInfo>(pi_init);
    state_parent_.insert_if_absent(child, parent_info);

    // Now make the child push-visible (for push_match_to_children, incl. recursive)
    LockFreeList<ChildInfo>* children = get_or_create_state_children(parent);
    children->push(info, hg_->arena());

    // Bump the rendezvous epoch (pushers retry their child scan on epoch change)
    uint64_t epoch = child_epoch_.fetch_add(1, std::memory_order_acq_rel);

    return epoch;
}

void ParallelEvolutionEngine::note_late_arrival(uint64_t match_hash) {
    if (validate_match_forwarding_) {
        auto missing = missing_match_hashes_.lookup(match_hash);
        if (missing.has_value()) {
            late_arrivals_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void ParallelEvolutionEngine::push_match_to_children(
    StateId parent,
    const MatchRecord& match,
    uint32_t step,
    PushSite site
) {
    if (batched_matching_) {
        // With batched matching, no retry loop needed
        push_match_to_children_impl(parent, match, step, site);
        return;
    }

    // EAGER MODE: re-run while THIS parent is still gaining children.
    //
    // The retry is not redundant with for_each's own stability loop: that loop covers nodes
    // appended to a list being walked, but the impl returns early when the parent has no
    // children list AT ALL yet, and then there is nothing to iterate.
    //
    // It is scoped to this parent's own list. Watching a counter bumped by every child
    // registration anywhere in the graph made an unrelated registration force a re-walk here,
    // and since the impl recurses into each child -- each level opening its own loop on the
    // same shared counter -- one outer retry re-executed a whole subtree. Termination then
    // required a moment in which NO thread anywhere registered a child, which on a loaded run
    // is not something to rely on. A single parent gains finitely many children, so scoped
    // this way the loop is bounded by that.
    auto child_epoch = [this](StateId p) -> uintptr_t {
        auto r = state_children_.lookup(p);
        return r.has_value() ? (*r)->head_token() : 0;
    };

    uintptr_t before = child_epoch(parent);
    push_match_to_children_impl(parent, match, step, site);
    uintptr_t after = child_epoch(parent);
    while (after != before) {
        stats_.forwarding_rewalks.fetch_add(1, std::memory_order_relaxed);
        before = after;
        push_match_to_children_impl(parent, match, step, site);
        after = child_epoch(parent);
    }
}

void ParallelEvolutionEngine::push_match_to_children_impl(
    StateId parent,
    const MatchRecord& match,
    [[maybe_unused]] uint32_t step,
    PushSite site
) {
    // Counted before the early return, so the denominator is every call and not only the ones
    // that found work. The question this answers is how often the call has anything to do.
    auto& calls = site == PushSite::Discovery ? stats_.push_discovery_calls
                                              : stats_.push_forwarding_calls;
    auto& empty = site == PushSite::Discovery ? stats_.push_discovery_empty
                                              : stats_.push_forwarding_empty;
    calls.fetch_add(1, std::memory_order_relaxed);

    auto result = state_children_.lookup_waiting(parent);
    if (!result.has_value()) {
        empty.fetch_add(1, std::memory_order_relaxed);
        return;  // No children registered
    }

    LockFreeList<ChildInfo>* children = *result;
    // A registered-but-empty list counts the same as an absent one: either way there is nothing
    // to push to. Detected by riding the walk that has to happen anyway rather than by a second
    // pass, so measuring this costs one stack bool and no extra traversal.
    bool any_child = false;
    children->for_each([&](const ChildInfo& child_info) {
        any_child = true;
        // Skip if match overlaps with consumed edges
        if (child_info.match_overlaps_consumed(match.matched_edges(), match.num_edges())) {
            stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Forward by reference: share the immutable core, only the per-descendant
        // source_state differs.
        MatchRecord forwarded = match;
        forwarded.source_state = child_info.child_state;

        // Deduplicate
        uint64_t h = forwarded.hash();
        const bool inserted = claim_match(h, forwarded, [&] {
            return hg_->arena().template create<MatchRecord>(forwarded);
        });
        if (!inserted) {
            return;  // Already processed
        }

        // Check if this was a "missing" match that arrived late via push
        note_late_arrival(h);

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

        DEBUG_LOG("PUSH parent=%u -> child=%u rule=%u hash=%lu step=%u",
                  parent, child_info.child_state, match.rule_index(), h, step);

        // Store match in child
        store_match_for_state(child_info.child_state, forwarded);

        // CRITICAL FIX: Use child's MATCH step, not parent's step!
        uint32_t child_step = child_info.creation_step + 1;

        // RECURSIVE: Push to child's existing children (grandchildren)
        push_match_to_children(child_info.child_state, forwarded, child_step);

        // Thin this transition, downstream of the store and the recursion for the same reason
        // as the pull side: the match stays available further down, where it is a different
        // transition with its own draw. Only the (this child, this match) transition is at
        // stake here.
        if (transition_rate_ < 1.0 &&
            !transition_survives(canonical_transition_key(child_info.child_state, forwarded), 0))
            return;

        // Spawn REWRITE task for this forwarded match
        submit_rewrite_task(forwarded, child_step);
    });

    if (!any_child) empty.fetch_add(1, std::memory_order_relaxed);
}

void ParallelEvolutionEngine::forward_existing_parent_matches(
    StateId parent,
    StateId child,
    const EdgeId* consumed_edges,
    uint8_t num_consumed,
    uint32_t step,
    SVec<MatchRecord>& batch
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
    SVec<MatchRecord>& batch
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
    uint32_t step,
    uint64_t /* child_registration_epoch - unused */,
    SVec<MatchRecord>& batch
) {
    auto result = state_matches_.lookup_waiting(ancestor);
    if (!result.has_value()) return;  // Ancestor has no matches yet

    LockFreeList<MatchRecord>* ancestor_matches = *result;
    ancestor_matches->for_each([&](const MatchRecord& ancestor_match) {
        // Skip if match overlaps with ANY consumed edge along the path
        bool overlaps = false;
        for (uint8_t i = 0; i < ancestor_match.num_edges() && !overlaps; ++i) {
            for (uint8_t j = 0; j < total_consumed; ++j) {
                if (ancestor_match.matched_edges()[i] == accumulated_consumed[j]) {
                    overlaps = true;
                    break;
                }
            }
        }

        if (overlaps) {
            stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Forward by reference: share the immutable core, set this child as source.
        MatchRecord forwarded = ancestor_match;
        forwarded.source_state = child;

        // Deduplicate
        uint64_t h = forwarded.hash();
        const bool inserted = claim_match(h, forwarded, [&] {
            return hg_->arena().template create<MatchRecord>(forwarded);
        });
        if (!inserted) {
            DEBUG_LOG("FWD_DUP ancestor=%u -> child=%u rule=%u hash=%lu",
                      ancestor, child, ancestor_match.rule_index(), h);
            return;  // Already seen
        }

        // Check if this was a "missing" match that arrived late via forward_existing
        note_late_arrival(h);

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

        DEBUG_LOG("FWD ancestor=%u -> child=%u rule=%u hash=%lu",
                  ancestor, child, ancestor_match.rule_index(), h);

        // CLAIM-WINNER OWNS THE MATCH AT THIS NODE (same invariant as the eager
        // pull): store the copy and propagate to this child's own children, exactly
        // as the push side does when it wins. A pull that claims the hash without
        // this leaves already-registered grandchildren uncovered (the racing push
        // sees the hash taken and skips its store+recursion). The rewrite itself is
        // submitted by the caller's batch phase.
        store_match_for_state(child, forwarded, true);
        push_match_to_children(child, forwarded, step);

        // Thin this transition, on the SAME terms as the eager pull and as a discovered match.
        // A forwarded match takes its own draw because it is its own transition; storing and
        // propagating above are deliberately upstream of it, so the match stays available to this
        // child's own children where it is a different transition and draws again.
        //
        // Without this the sampled subgraph depends on which submission mode is in use, since
        // forwarded matches would enter the batch unthinned while discovered ones are thinned.
        if (transition_rate_ < 1.0 &&
            !transition_survives(canonical_transition_key(child, forwarded), 1)) return;

        batch.push_back(forwarded);
    });
}


// Fold the match-list head of every ancestor on `parent`'s chain into one token. Two reads
// returning the same value mean no ancestor ON THIS CHAIN gained a match between them.
//
// Scoping matters twice over here. Watching a counter bumped by every match store anywhere
// coupled this walk to unrelated states, and -- worse -- the walk itself calls
// store_match_for_state, which bumped that same counter, so the walker perturbed the very
// quantity it was waiting to see settle and guaranteed at least one extra full re-walk every
// pass. Those stores go to the CHILD, which is not on the ancestor chain, so a chain-scoped
// token is untouched by them.
uintptr_t ParallelEvolutionEngine::ancestor_match_epoch(StateId parent) const {
    uintptr_t token = 0;
    StateId cur = parent;
    for (uint32_t hops = 0; cur != INVALID_ID && hops < kMaxAncestorHops; ++hops) {
        auto m = state_matches_.lookup(cur);
        token = token * 1099511628211ULL + (m.has_value() ? (*m)->head_token() : 0);
        auto p = state_parent_.lookup(cur);
        if (!p.has_value() || !*p || !(*p)->has_parent()) break;
        cur = (*p)->parent_state;
    }
    return token;
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

    // RETRY LOOP: re-walk while an ancestor ON THIS CHAIN is still gaining matches.
    uintptr_t epoch_before = ancestor_match_epoch(parent);

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
    uintptr_t epoch_after = ancestor_match_epoch(parent);
    while (epoch_after != epoch_before) {
        stats_.forwarding_rewalks.fetch_add(1, std::memory_order_relaxed);
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

        epoch_after = ancestor_match_epoch(parent);
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
    SUSet<EdgeId> consumed_set(accumulated_consumed, accumulated_consumed + total_consumed);

    LockFreeList<MatchRecord>* ancestor_matches = *result;
    ancestor_matches->for_each([&](const MatchRecord& ancestor_match) {
        // Skip if match overlaps with ANY consumed edge - O(n) vs O(n*m)
        bool overlaps = false;
        for (uint8_t i = 0; i < ancestor_match.num_edges() && !overlaps; ++i) {
            if (consumed_set.count(ancestor_match.matched_edges()[i])) {
                overlaps = true;
            }
        }

        if (overlaps) {
            stats_.matches_invalidated.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Forward by reference: share the immutable core, set this child as source.
        MatchRecord forwarded = ancestor_match;
        forwarded.source_state = child;

        // Deduplicate - seen_match_hashes_ protects against both push and pull duplicates
        uint64_t h = forwarded.hash();
        const bool inserted = claim_match(h, forwarded, [&] {
            return hg_->arena().template create<MatchRecord>(forwarded);
        });
        if (!inserted) {
            DEBUG_LOG("FWD_EAGER_DUP ancestor=%u -> child=%u rule=%u hash=%lu",
                      ancestor, child, ancestor_match.rule_index(), h);
            return;  // Already seen (possibly via push)
        }

        // Check if this was a "missing" match that arrived late
        note_late_arrival(h);

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.matches_forwarded.fetch_add(1, std::memory_order_relaxed);

        DEBUG_LOG("FWD_EAGER ancestor=%u -> child=%u rule=%u hash=%lu step=%u",
                  ancestor, child, ancestor_match.rule_index(), h, step);

        // CLAIM-WINNER OWNS THE MATCH AT THIS NODE: store the copy and propagate to
        // this child's own children, exactly as the push side does when it wins the
        // claim. Without this, a pull that wins the (match, child) claim races the
        // ancestor's push out of its store+recursion (the push sees the hash taken
        // and returns), so a GRANDCHILD whose pull already completed is covered by
        // nobody and the transition is permanently lost. Symmetric ownership makes
        // the coverage inductive: every claim winner re-establishes the invariant
        // one level down.
        store_match_for_state(child, forwarded, true);
        push_match_to_children(child, forwarded, step);

        // Thin this transition. A forwarded match takes the SAME draw as a discovered one --
        // that is the property a per-state count could not have, because forwarding is what
        // stops a state's match population from ever closing. Storing and propagating above
        // are deliberately upstream of it: the match stays available to this child's own
        // children, where it is a different transition and gets its own draw.
        if (transition_rate_ < 1.0 &&
            !transition_survives(canonical_transition_key(child, forwarded), 2)) return;

        // EAGER: Immediately spawn REWRITE task
        submit_rewrite_task(forwarded, step);
    });
}

// =============================================================================
// Task Submission
// =============================================================================

void ParallelEvolutionEngine::propagate_explore_depth(StateId canonical_state, uint32_t depth) {
    // Depth strictly decreases on every accepted relaxation and is bounded below by zero,
    // so this terminates. Matching still runs at most once per canonical state, because
    // the claim is separate from the depth.
    // The caller has already lowered canonical_state's depth. Order that store before the
    // scan of its child list so a child registered concurrently (which pushes itself, then
    // reads this parent's depth, both across a seq_cst fence) is either seen here or sees
    // the lowered depth itself -- never stranded at a stale depth. Same fence guards each
    // deeper scan against its own just-completed relaxation store.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const LockFreeList<StateId>* kids = canon_children_.get(canonical_state);
    if (!kids) return;

    const uint32_t budget =
        static_cast<uint32_t>(std::min<size_t>(max_steps_, INVALID_ID));

    // The worklist draws from the per-worker scratch arena and is reclaimed in bulk. It is
    // walked as a queue with a cursor, which is the breadth-first order relaxation wants.
    auto mark = worker_scratch().mark();
    ArenaVector<std::pair<StateId, uint32_t>> pending(worker_scratch(), 16);
    kids->for_each([&](StateId child) { pending.emplace_back(child, depth + 1); });
    for (size_t i = 0; i < pending.size(); ++i) {
        const StateId s = pending[i].first;
        const uint32_t d = pending[i].second;
        if (s == INVALID_ID) continue;
        if (!hg_->try_lower_explore_depth(s, d)) continue;
        // exploration_probability_ is tested BEFORE the key is built: the key costs an
        // individualization-refinement pass on first use for a state, and at p == 1 the answer
        // is yes regardless. An argument evaluated eagerly here would put that pass on the
        // default, unsampled path.
        if (d < budget && hg_->try_claim_expanded(s) &&
            (exploration_probability_ >= 1.0 ||
             should_explore(hg_->get_or_compute_canonical_hash(s)))) {
            submit_match_task(s, d + 1);  // a canonical state is its own representative
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (const LockFreeList<StateId>* more = canon_children_.get(s)) {
            more->for_each([&](StateId child) { pending.emplace_back(child, d + 1); });
        }
    }
    worker_scratch().release(mark);
}

void ParallelEvolutionEngine::submit_match_task(StateId state, uint32_t step) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (step > max_steps_) return;
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(state)) return;

    DEBUG_LOG("SUBMIT_MATCH state=%u step=%u (full)", state, step);

    note_match_task_pushed(state);
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
    if (step > max_steps_) return;
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(state)) return;

    DEBUG_LOG("SUBMIT_MATCH state=%u step=%u parent=%u produced=%u consumed=%u (delta)",
              state, step, ctx.parent_state, ctx.num_produced, ctx.num_consumed);

    note_match_task_pushed(state);
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
    if (step > max_steps_) return;
    // Early check (non-reserving) - execute_rewrite_task does the actual atomic reservation
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(match.source_state)) return;

    DEBUG_LOG("SUBMIT_REWRITE state=%u rule=%u step=%u", match.source_state, match.rule_index(), step);

    auto job = job_system::make_job<EvolutionJobType>(
        [this, match, step]() {
            execute_rewrite_task(match, step);
        },
        EvolutionJobType::REWRITE
    );
    job_system_->submit(std::move(job));
}

void ParallelEvolutionEngine::execute_expand_chunk(const ExpandChunk& chunk) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    for (uint32_t i = chunk.begin; i < chunk.end; ++i) {
        if (should_stop_.load(std::memory_order_relaxed)) return;
        execute_rewrite_task(chunk.matches[i], chunk.step);
    }
}

void ParallelEvolutionEngine::submit_expand_chunk(const ExpandChunk& chunk) {
    auto job = job_system::make_job<EvolutionJobType>(
        [this, chunk]() { execute_expand_chunk(chunk); },
        EvolutionJobType::REWRITE
    );
    job_system_->submit(std::move(job));
}

void ParallelEvolutionEngine::dispatch_expansion(StateId state, uint32_t step,
                                                 const MatchRecord* matches, size_t count) {
    if (count == 0) return;
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (step > max_steps_) return;
    // Whole-state gates, checked once here rather than once per match. execute_rewrite_task
    // still does the reserving check per child, so this is a filter, not the decision.
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(state)) return;

    if (count <= kExpandChunkSize) {
        // Everything on this thread: no arena copy, no job, and the parent's data is already
        // in this core's cache from the matching that produced these records.
        ExpandChunk chunk{matches, 0, static_cast<uint32_t>(count), step};
        execute_expand_chunk(chunk);
        return;
    }

    // Wide state. The match records outlive this frame, so they move to the arena; every
    // chunk then reads one shared immutable array.
    MatchRecord* shared = hg_->arena().template allocate_array<MatchRecord>(count);
    for (size_t i = 0; i < count; ++i) shared[i] = matches[i];

    // Submit the tail first and keep the head for this thread. The submitted chunks are
    // available to steal while this one runs, so the wide case both spreads and stays warm.
    for (size_t begin = kExpandChunkSize; begin < count; begin += kExpandChunkSize) {
        const size_t end = std::min(begin + kExpandChunkSize, count);
        submit_expand_chunk(ExpandChunk{shared, static_cast<uint32_t>(begin),
                                        static_cast<uint32_t>(end), step});
    }
    execute_expand_chunk(ExpandChunk{shared, 0, kExpandChunkSize, step});
}

void ParallelEvolutionEngine::submit_scan_task(const ScanTaskData& data) {
    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (data.step > max_steps_) return;
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("SUBMIT_SCAN state=%u rule=%u step=%u delta=%d",
              data.state, data.rule_index, data.step, data.is_delta);

    note_match_task_pushed(data.state);
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
    if (data.step > max_steps_) return;
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("SUBMIT_EXPAND state=%u rule=%u matched=%u/%u step=%u",
              data.state, data.rule_index, data.num_matched, data.num_pattern_edges, data.step);

    note_match_task_pushed(data.state);
    auto job = job_system::make_job<EvolutionJobType>(
        [this, data]() {
            execute_expand_task(data);
        },
        EvolutionJobType::EXPAND
    );
    // LIFO scheduling: depth-first traversal, bounded memory O(|E(q)|² × |E(H)|)
    job_system_->submit(std::move(job), job_system::ScheduleMode::LIFO);
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

// Find, or install, the budget counter a key shares across threads. Templated because
// the step and successor maps carry different reserved sentinel bands.
template <typename Map>
static std::atomic<size_t>* budget_counter(Map& counters, uint64_t key,
                                           ConcurrentHeterogeneousArena& arena) {
    auto result = counters.lookup(key);
    if (result.has_value()) return *result;

    auto* counter = arena.template create<std::atomic<size_t>>(0);
    auto [existing, inserted] = counters.insert_if_absent(key, counter);
    return inserted ? counter : existing;  // another thread installed one first
}

// Claim one unit of a budget, or report it exhausted.
//
// A fetch_add followed by a rollback would publish a count above the limit for as long
// as the rollback takes, and the plain readers (can_create_states_at_step,
// can_have_more_children) prune on exactly that value -- so N concurrent claimants would
// make each other's in-budget work look out-of-budget, and which work got pruned would
// depend on the interleaving. Claiming by CAS never publishes a count above the limit,
// so the budget prunes the same work whatever the schedule.
bool ParallelEvolutionEngine::try_claim_budget(std::atomic<size_t>* counter, size_t limit) {
    size_t cur = counter->load(std::memory_order_relaxed);
    while (cur < limit) {
        if (counter->compare_exchange_weak(cur, cur + 1,
                                           std::memory_order_acq_rel,
                                           std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

// =============================================================================
// Per-State Match Join
// =============================================================================

ParallelEvolutionEngine::MatchJoin* ParallelEvolutionEngine::match_join_for(StateId state) {
    const uint64_t key = static_cast<uint64_t>(state);
    auto result = match_join_.lookup(key);
    if (result.has_value()) return *result;

    auto* join = hg_->arena().template create<MatchJoin>();
    auto [existing, inserted] = match_join_.insert_if_absent(key, join);
    return inserted ? join : existing;   // another thread installed one first
}

uint64_t ParallelEvolutionEngine::canonical_transition_key(StateId state,
                                                          const MatchRecord& match) {
    const State& s = hg_->get_state(state);

    // Ranks come from the same individualization-refinement pass as the state's canonical
    // hash, so asking for them costs one IR pass per state and nothing per match after that.
    hg_->ensure_state_edge_ranks(state, s.edges);
    const uint64_t input_hash = hg_->get_or_compute_canonical_hash(state);

    uint32_t ranks[MAX_PATTERN_EDGES];
    const uint8_t n = match.num_edges();
    for (uint8_t i = 0; i < n && i < MAX_PATTERN_EDGES; ++i) {
        ranks[i] = hg_->edge_rank_in_state(state, match.matched_edges()[i]);
    }

    return hgcommon::event_signature(hgcommon::EVENT_SIG_TRANSITION,
                                     input_hash, /*output_state_hash=*/0,
                                     /*step=*/0, match.rule_index(),
                                     ranks, n, /*produced_ranks=*/nullptr, 0);
}

bool ParallelEvolutionEngine::transition_survives_spined(StateId source, uint64_t canonical_key,
                                                         int site) {
    // Own-found draws only. Track the running minimum own key -- complete exactly at the
    // state's drain, because its own matching is what the drain joins on -- and mark whether
    // any own draw passed. Forwarded draws go through transition_survives directly and touch
    // neither: their arrival order races the drain, and a spine that read them decided WHICH
    // transition survives by schedule (caught at 8 workers by SamplingReproducibility).
    MatchJoin* join = match_join_for(source);
    // The spine's order is SEEDED: min over splitmix(key ^ seed), not over the bare key.
    // A seed-independent spine keeps the same skeleton every run, so the union over seeds
    // saturates at skeleton-plus-bushes; seeding it makes each seed explore a different
    // skeleton, which is what lets repeated sampling recover the graph. Still a pure
    // function of (transition, seed): schedule-stable, device-stable.
    const uint64_t ranked = spine_rank(canonical_key);
    uint64_t seen = join->own_min_key.load(std::memory_order_relaxed);
    while (ranked < seen &&
           !join->own_min_key.compare_exchange_weak(seen, ranked,
                                                    std::memory_order_relaxed)) {}
    if (transition_survives(canonical_key, site)) {
        join->own_spawned.store(1, std::memory_order_release);
        return true;
    }
    return false;
}

void ParallelEvolutionEngine::spine_at_drain(StateId state, uint32_t step, MatchJoin* join) {
    if (join->own_spawned.load(std::memory_order_acquire) != 0) return;
    const uint64_t want = join->own_min_key.load(std::memory_order_acquire);
    if (want == ~0ULL) return;   // no own-found matches: no spine for this state
    auto stored = state_matches_.lookup(static_cast<uint64_t>(state));
    if (!stored.has_value()) return;

    // The own-minimum's record is in the stored list (every site stores before drawing);
    // find it by its seeded rank.
    bool found = false;
    MatchRecord best{};
    (*stored)->for_each([&](const MatchRecord& m) {
        if (!found && spine_rank(canonical_transition_key(state, m)) == want) {
            found = true; best = m;
        }
    });
    if (!found) return;

    stats_.spine_forced.fetch_add(1, std::memory_order_relaxed);
    join->own_spawned.store(1, std::memory_order_release);
    submit_rewrite_task(best, step);
}


bool ParallelEvolutionEngine::transition_survives(uint64_t transition_key, int site) const {
    if (transition_rate_ >= 1.0) return true;
    if (transition_rate_ <= 0.0) return false;
    draws_taken_.fetch_add(1, std::memory_order_relaxed);
    if (site >= 0 && site < 5) draws_by_site_[site].fetch_add(1, std::memory_order_relaxed);

    // splitmix64 of (seed, transition). Deliberately NOT a worker RNG: drawing from thread
    // state would make the surviving subgraph depend on which thread happened to reach the
    // transition, so the same run would sample differently at a different thread count and
    // "representative" would have nothing to be reproducible about.
    uint64_t x = transition_key ^ (random_seed_ * 0x9E3779B97F4A7C15ULL);
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= (x >> 31);

    // Compare in [0,1) via the top 53 bits, so the threshold means the same thing at any q.
    const double u = static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
    const bool survives = u < transition_rate_;
    if (survives) draws_survived_.fetch_add(1, std::memory_order_relaxed);
    return survives;
}

size_t ParallelEvolutionEngine::matches_found_for_state(StateId state) const {
    auto result = match_join_.lookup(static_cast<uint64_t>(state));
    if (!result.has_value()) return 0;
    return (*result)->matches.load(std::memory_order_acquire);
}

void ParallelEvolutionEngine::note_match_task_pushed(StateId state) {
    match_join_for(state)->pushed.fetch_add(1, std::memory_order_release);
}

void ParallelEvolutionEngine::note_match_task_done(StateId state, uint32_t step) {
    MatchJoin* join = match_join_for(state);
    const size_t done = join->completed.fetch_add(1, std::memory_order_acq_rel) + 1;

    // Read `pushed` AFTER booking the completion. A task that will still spawn more has not
    // reached its own guard, so anything it pushes is already counted here; and if `pushed`
    // has moved on since, this task is simply not the last one and whichever is will fire.
    if (done != join->pushed.load(std::memory_order_acquire)) return;

    states_drained_.fetch_add(1, std::memory_order_relaxed);
    if (transition_rate_ > 0.0 && transition_rate_ < 1.0) spine_at_drain(state, step, join);
    if (on_state_matches_complete_) on_state_matches_complete_(state, step);
}

bool ParallelEvolutionEngine::try_reserve_successor_slot(StateId parent) {
    if (max_successor_states_per_parent_ == 0) return true;  // Unlimited
    return try_claim_budget(budget_counter(parent_successor_count_, parent, hg_->arena()),
                            max_successor_states_per_parent_);
}

void ParallelEvolutionEngine::release_successor_slot(StateId parent) {
    if (max_successor_states_per_parent_ == 0) return;  // Unlimited, nothing to release

    auto result = parent_successor_count_.lookup(parent);
    if (result.has_value()) {
        (*result)->fetch_sub(1, std::memory_order_relaxed);
    }
}

bool ParallelEvolutionEngine::try_reserve_step_slot(uint32_t step) {
    if (max_states_per_step_ == 0) return true;  // Unlimited
    return try_claim_budget(budget_counter(states_per_step_, step, hg_->arena()),
                            max_states_per_step_);
}

void ParallelEvolutionEngine::release_step_slot(uint32_t step) {
    if (max_states_per_step_ == 0) return;  // Unlimited, nothing to release

    auto result = states_per_step_.lookup(step);
    if (result.has_value()) {
        (*result)->fetch_sub(1, std::memory_order_relaxed);
    }
}

std::mt19937& ParallelEvolutionEngine::sampling_rng() const {
    thread_local std::mt19937 rng;
    thread_local uint64_t seen_gen = std::numeric_limits<uint64_t>::max();
    uint64_t gen = sampling_generation_.load(std::memory_order_relaxed);
    if (seen_gen != gen) {
        uint64_t s = random_seed_
            ? (random_seed_ ^ (0x9e3779b97f4a7c15ULL *
                 static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()))))
            : static_cast<uint64_t>(std::random_device{}());
        rng.seed(static_cast<std::mt19937::result_type>(s));
        seen_gen = gen;
    }
    return rng;
}


void ParallelEvolutionEngine::configure_identity_and_quotient() {
    // Positional identity reads ranks from each raw state's own canonical labelling, and the
    // quotient never materialises raw presentations -- the two cannot agree by construction.
    // The quotient is an optimisation, so the REQUEST wins: full capture runs and the disabled
    // optimisation is reported.
    if (hg_->positional_event_identity() && explore_from_canonical_states_only_) {
        explore_from_canonical_states_only_ = false;
        warnings_.push_back(
            "ExploreFromCanonicalStatesOnly disabled: Positional event identity requires raw "
            "presentations, which quotient exploration does not materialise.");
    }

    // Automatic identity is the linked-hypergraph convention, computed by the reconstruction's
    // class-frame signing. It runs under BOTH exploration strategies, so quotient and full
    // capture produce the same event identities, causal relation and branchial relation by
    // construction -- adjudicated step-exact against Wolfram/Multicomputation
    // (reference/adjudicate_gap1_authority.wls). Set before any state (incl. genesis) is
    // created so edge keys use the right mode.
    const bool qc = explore_from_canonical_states_only_ ||
                    (!hg_->positional_event_identity() &&
                     hg_->event_signature_keys() == hgcommon::EVENT_SIG_AUTOMATIC);
    hg_->set_quotient_causal(qc);
    hg_->set_quotient_reconstruction(qc);
    guard_quotient_transitive_reduction();
}

void ParallelEvolutionEngine::guard_quotient_transitive_reduction() {
    // Quotient emits causal edges between CANONICAL event ids, which are schedule-dependent
    // (canonical_event_map_ is first-writer-wins) and non-topological under recurrence, so the
    // online transitive reduction -- which assumes id order == topological order -- would
    // produce a non-deterministic, wrong reduced graph. Serving the correct un-reduced graph is
    // strictly better, hence this guard.
    //
    // The per-instance raw reconstruction (Hypergraph::set_quotient_reconstruction) recovers
    // every raw observable COUNT exactly, but its causal edge IDENTITIES are not yet canonical:
    // slots tie-break on EdgeId within a content class, and when same-content edges have
    // different producers the wiring varies run to run. Until that tie-break is canonical the
    // reconstruction cannot replace this guard. See docs/VERIFICATION_PLAN.md.
    if (explore_from_canonical_states_only_ &&
        hg_ && hg_->causal_graph().transitive_reduction_enabled()) {
        hg_->causal_graph().set_transitive_reduction(false);
        DEBUG_LOG("WARN: transitive reduction is disabled under quotient exploration; returning "
                  "the correct un-reduced causal graph. Use full-capture for the reduced graph.");
    }
}

bool ParallelEvolutionEngine::should_explore(uint64_t invariant_key) const {
    if (exploration_probability_ >= 1.0) return true;
    if (exploration_probability_ <= 0.0) return false;

    // Same construction as the transition draw, on a separate stream. The two samplers must
    // not correlate: under full capture they are keyed on the same object, so sharing a stream
    // would make "explored" and "transition kept" the same coin rather than two.
    constexpr uint64_t kStateStream = 0xD1B54A32D192ED03ULL;
    uint64_t x = (invariant_key ^ kStateStream) ^ (random_seed_ * 0x9E3779B97F4A7C15ULL);
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= (x >> 31);
    const double u = static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
    return u < exploration_probability_;
}

bool ParallelEvolutionEngine::should_explore() {
    if (exploration_probability_ >= 1.0) return true;
    if (exploration_probability_ <= 0.0) return false;

    auto& rng = sampling_rng();
    thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

    return dist(rng) < exploration_probability_;
}

SVec<uint16_t> ParallelEvolutionEngine::get_shuffled_rule_indices() const {
    SVec<uint16_t> indices(rules_.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), sampling_rng());

    return indices;
}

// =============================================================================
// REWRITE Task Execution
// =============================================================================

void ParallelEvolutionEngine::execute_rewrite_task(const MatchRecord& match, uint32_t step) {
    if (should_stop_.load(std::memory_order_relaxed)) return;

    // Check step limit - don't spawn REWRITEs past max_steps
    if (step > max_steps_) return;

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
        release_successor_slot(match.source_state);  // no child will occupy it
        return;  // Too many states at this generation
    }

    const RewriteRule& rule = rules_[match.rule_index()];

    // Apply the rewrite
    RewriteResult rr = rewriter_.apply(
        rule,
        match.source_state,
        match.matched_edges(),
        match.num_edges(),
        match.binding(),
        step
    );

    // Both budgets count states this parent actually contributed, so a rewrite that
    // produces none gives both slots back. Holding the successor slot here would retire
    // a parent's child budget on work that never became a child.
    if (rr.new_state == INVALID_ID) {
        // Rewrite failed - release the reserved slots
        release_step_slot(step + 1);
        release_successor_slot(match.source_state);
        return;
    }

    total_rewrites_.fetch_add(1, std::memory_order_relaxed);
    total_events_.fetch_add(1, std::memory_order_relaxed);

    if (!rr.was_new_state) {
        // Duplicate state - release the reserved slots (only count unique states)
        release_step_slot(step + 1);
        release_successor_slot(match.source_state);
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
        match.rule_index(),       // rule index
        rr.event,                 // raw event id (for tracking)
        rr.canonical_event,       // canonical event id (for deduplication)
        match.num_edges(),        // destroyed edges count
        rr.num_produced           // created edges count
    );
#endif

    // Spawn MATCH task for the new raw state if it hasn't been matched yet
    auto [existing, inserted] = matched_raw_states_.insert_if_absent_waiting(
        rr.raw_state, true);

    if (inserted) {
        DEBUG_LOG("STATE parent=%u -> child=%u (canonical=%u) rule=%u step=%u new=%d",
                  match.source_state, rr.raw_state, rr.new_state, match.rule_index(), step, rr.was_new_state);

        // Build MatchContext for match forwarding
        MatchContext ctx;
        ctx.parent_state = match.source_state;
        ctx.num_consumed = match.num_edges();
        for (uint8_t i = 0; i < match.num_edges(); ++i) {
            ctx.consumed_edges[i] = match.matched_edges()[i];
        }
        ctx.num_produced = rr.num_produced;
        for (uint8_t i = 0; i < rr.num_produced; ++i) {
            ctx.produced_edges[i] = rr.produced_edges[i];
        }

        // Quotient exploration: record the canonical transition, then relax the child's
        // depth. Expansion is driven by relaxation, so a state is matched once, at the
        // shortest depth that reaches it, whatever order the paths arrive in. The claim is
        // taken here rather than inside the walk so the expansion still carries this
        // rewrite's match context, which is what lets forwarding skip a full rematch.
        if (explore_from_canonical_states_only_) {
            const StateId parent_canonical = hg_->get_canonical_state(match.source_state);
            // Publish the canonical transition BEFORE reading the parent's depth. A racing
            // relaxation of the parent lowers the parent's depth and then scans this child
            // list; this side pushes the child and then reads the parent's depth. The
            // seq_cst fences on both sides forbid the outcome where the scan misses this
            // child AND this read misses the lowered depth, so the child always learns the
            // parent's true minimum depth even across the publish/relax race.
            canon_children_.get_or_default(parent_canonical, hg_->arena())
                           .push(rr.new_state, hg_->arena());
            std::atomic_thread_fence(std::memory_order_seq_cst);

            // The child's arrival depth is one past the parent's CURRENT minimum depth, not
            // the depth the parent happened to be expanded at. A shorter path to the parent
            // found after it was first expanded must pull the child (and its subtree) into
            // budget; deriving the depth from the parent's live minimum, together with the
            // relaxation cascade below, makes expansion depend only on the shortest-path
            // depth, never on the order arrivals race.
            const uint32_t parent_depth = hg_->explore_depth_of(parent_canonical);
            const uint32_t child_depth =
                (parent_depth == INVALID_ID) ? step : parent_depth + 1;

            if (!hg_->try_lower_explore_depth(rr.new_state, child_depth)) return;

            const uint32_t budget =
        static_cast<uint32_t>(std::min<size_t>(max_steps_, INVALID_ID));
            if (child_depth < budget && hg_->try_claim_expanded(rr.new_state)) {
                // Exploration-probability pruning: flip the coin ONCE per canonical
                // state, at its first (shortest-depth) claim, so a state reached by N
                // transitions is kept with probability p, not 1-(1-p)^N. Matches the
                // GPU, which flips once per deduped state. A pruned state stays
                // claimed, so no later transition re-flips for it.
                //
                // Keyed on the class's canonical hash, so WHICH classes survive is the same
                // at any worker count -- and so it stays one draw per class even though the
                // claim that reaches here is whichever transition won the race.
                if (exploration_probability_ >= 1.0 ||
                    should_explore(hg_->get_or_compute_canonical_hash(rr.new_state))) {
                    if (enable_match_forwarding_) {
                        register_child_with_parent(
                            match.source_state, rr.raw_state,
                            match.matched_edges(), match.num_edges(),
                            child_depth);
                    }
                    submit_match_task_with_context(rr.raw_state, child_depth + 1, ctx);
                }
            }
            propagate_explore_depth(rr.new_state, child_depth);
            return;
        }

        // Exploration-probability pruning: full multiway expands every raw state, so one coin
        // flip per raw state is one per state.
        //
        // Keyed on the transition that created this state, because that is the only
        // isomorphism-invariant name a RAW state has -- its own id is an allocation order.
        // Raw states stand in bijection with the events that create them, so this is also the
        // exact sense in which ExplorationProbability and TransitionRate are one knob under
        // full capture and two only under quotient; the separate stream inside should_explore
        // is what keeps them from being literally the same coin.
        if (exploration_probability_ < 1.0 &&
            !should_explore(canonical_transition_key(match.source_state, match))) {
            return;
        }

        // Register child's parent pointer for ancestor chain walking
        if (enable_match_forwarding_) {
            register_child_with_parent(
                match.source_state, rr.raw_state,
                match.matched_edges(), match.num_edges(),
                step);
        }

        // Submit MATCH task with context for match forwarding
        submit_match_task_with_context(rr.raw_state, step + 1, ctx);
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
    // First statement, so every exit below books this task's completion. The push happened at
    // submit time, so an early return here still has to be accounted for.
    MatchTaskGuard join_guard(*this, state, step);

    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (step > max_steps_) return;

    // Early exit if rewrites are impossible due to limits
    if (!can_create_states_at_step(step + 1)) return;
    if (!can_have_more_children(state)) return;

    const State& s = hg_->get_state(state);

    // Edge accessor
    auto get_edge = [this](EdgeId eid) -> const Edge& {
        return hg_->get_edge(eid);
    };

    // Signature accessor (cached signatures for O(1) lookup)
    auto get_signature = [this](EdgeId eid) -> const EdgeSignature& {
        return hg_->edge_signature(eid);
    };

    // Batched matching: collect all matches first, then spawn REWRITEs. The batch
    // is only touched in batched mode, so reserve only then -- the default eager
    // path would otherwise bump ~32 MatchRecords of per-task scratch for nothing.
    SVec<MatchRecord> batch;
    if (batched_matching_) batch.reserve(32);
    size_t delta_start = 0;  // Index where delta (discovered) matches start

    // Collector callback
    auto collect_match = [&, state](
        uint16_t rule_index,
        const EdgeId* edges,
        uint8_t num_edges,
        const VariableBinding& binding
    ) {
        if (should_stop_.load(std::memory_order_relaxed)) return;

        // Build the payload on the stack so a duplicate costs no arena; the winner
        // promotes it to an immutable arena-resident MatchCore that forwarded copies
        // can share.
        MatchCore core_tmp;
        core_tmp.rule_index = rule_index;
        core_tmp.num_edges = num_edges;
        core_tmp.binding = binding;
        for (uint8_t i = 0; i < num_edges; ++i) {
            core_tmp.matched_edges[i] = edges[i];
        }
        MatchRecord match;
        match.core = &core_tmp;
        match.source_state = state;

        // Deduplicate on CONTENT, not on the hash alone.
        //
        // The map must hold a pointer that outlives this frame so other threads can compare
        // against it, and it has no claim-then-publish protocol -- so the core is promoted
        // before the insert rather than after winning it. A true duplicate would then waste an
        // arena slot, and true duplicates are routine (delta matching finds a match on k
        // produced edges k times, once anchored on each), so they are answered by a lookup
        // first and never reach the promotion.
        uint64_t h = match.hash();
        if (!claim_match(h, match, [&] {
                // Reached only when the match may be new, so a duplicate never promotes.
                match.core = hg_->arena().template create<MatchCore>(core_tmp);
                return hg_->arena().template create<MatchRecord>(match);
            })) {
            rejected_duplicates_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        total_matches_found_.fetch_add(1, std::memory_order_relaxed);
        stats_.new_matches_discovered.fetch_add(1, std::memory_order_relaxed);
        match_join_for(state)->matches.fetch_add(1, std::memory_order_acq_rel);

        DEBUG_LOG("NEW state=%u rule=%u hash=%lu step=%u", state, rule_index, h, step);

        if (enable_match_forwarding_ && !batched_matching_) {
            store_match_for_state(state, match, true);
            push_match_to_children(state, match, step, PushSite::Discovery);
        }

        // Thin this transition. Same reason forwarding above is unaffected: dropping the
        // transition (S -> S') does not drop the match, and the same match at a different
        // source state is a DIFFERENT transition that gets its own independent draw.
        if (transition_rate_ < 1.0 &&
            !transition_survives_spined(state, canonical_transition_key(state, match), 3)) return;

        if (batched_matching_) {
            batch.push_back(match);
        } else {
            submit_rewrite_task(match, step);
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
                scan_data.is_delta = true;
                scan_data.num_produced = ctx.num_produced;
                for (uint8_t i = 0; i < ctx.num_produced; ++i) {
                    scan_data.produced_edges[i] = ctx.produced_edges[i];
                }
                submit_scan_task(scan_data);
            }
            // Spawn REWRITEs for forwarded matches
            for (size_t i = 0; i < delta_start; ++i) {
                submit_rewrite_task(batch[i], step);
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
            validations_performed_.fetch_add(1, std::memory_order_relaxed);
            size_t missing = 0;
            auto count_missing = [&, state](
                uint16_t rule_index,
                const EdgeId* edges,
                uint8_t num_edges,
                const VariableBinding& binding,
                StateId /*source_state*/
            ) {
                // Transient: only the hash is needed, so the core stays on the stack.
                MatchCore core_tmp;
                core_tmp.rule_index = rule_index;
                core_tmp.num_edges = num_edges;
                core_tmp.binding = binding;
                for (uint8_t i = 0; i < num_edges; ++i) {
                    core_tmp.matched_edges[i] = edges[i];
                }
                MatchRecord match;
                match.core = &core_tmp;
                match.source_state = state;
                uint64_t h = match.hash();
                if (!contains_match(h, match)) {
                    ++missing;
                    // Attribute the miss: does it touch an edge this rewrite produced?
                    bool touches_produced = false;
                    for (uint8_t i = 0; i < num_edges && !touches_produced; ++i)
                        for (uint8_t j = 0; j < ctx.num_produced; ++j)
                            if (edges[i] == ctx.produced_edges[j]) { touches_produced = true; break; }
                    if (touches_produced) missing_owed_by_delta_.fetch_add(1, std::memory_order_relaxed);
                    else missing_owed_by_forwarding_.fetch_add(1, std::memory_order_relaxed);
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
    if (batched_matching_) {
        if (enable_match_forwarding_) {
            for (size_t i = delta_start; i < batch.size(); ++i) {
                store_match_for_state(state, batch[i]);
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }

        dispatch_expansion(state, step, batch.data(), batch.size());
    }
}

// =============================================================================
// SCAN Task Execution
// =============================================================================

void ParallelEvolutionEngine::execute_scan_task(const ScanTaskData& data) {
    MatchTaskGuard join_guard(*this, data.state, data.step);

    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (data.step > max_steps_) return;

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

    // Per-edge pattern signatures and compatible-signature caches are precomputed
    // once on the rule (rule.lhs_sig / rule.lhs_cache); read them here.

    // Single-edge rules complete a match on the seed itself. Collect them here and expand
    // them together at the end rather than one task per match: they are all children of the
    // same parent, so running them back to back keeps that parent's edge set and
    // canonicalization inputs in this core's cache for the whole run.
    SVec<MatchRecord> completed;

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
                if (!signature_compatible(data_sig, rule.lhs_sig[pos])) continue;

                // Validate candidate
                VariableBinding binding;
                if (!validate_candidate(edge.vertices, edge.arity, pattern_edge, binding)) continue;

                // Create EXPAND task data
                ExpandTaskData expand_data;
                expand_data.state = data.state;
                expand_data.rule_index = data.rule_index;
                expand_data.num_pattern_edges = rule.num_lhs_edges;
                expand_data.matched_edges[0] = produced;
                expand_data.match_order[0] = pos;
                expand_data.num_matched = 1;
                expand_data.binding = binding;
                expand_data.step = data.step;

                if (rule.num_lhs_edges == 1) {
                    MatchRecord m;
                    if (complete_match(expand_data, m)) completed.push_back(m);
                } else {
                    submit_expand_task(expand_data);
                }
            }
        }
    } else {
        // Full matching: seed with the rule's most-constrained edge (match_order[0]).
        const uint8_t first_pidx = rule.match_order[0];
        const PatternEdge& first_edge = rule.lhs[first_pidx];
        const EdgeSignature& first_sig = rule.lhs_sig[first_pidx];
        const CompatibleSignatureCache& first_cache = rule.lhs_cache[first_pidx];

        // Generate candidates for first edge
        generate_candidates(
            first_edge, first_sig, first_cache,
            VariableBinding{}, s.edges,
            hg_->signature_index(), hg_->inverted_index(), get_edge,
            [&](EdgeId candidate, const auto& edge) {
                if (should_stop_.load(std::memory_order_relaxed)) return;

                VariableBinding binding;
                if (!validate_candidate(edge.vertices, edge.arity, first_edge, binding)) return;

                // Create EXPAND task data
                ExpandTaskData expand_data;
                expand_data.state = data.state;
                expand_data.rule_index = data.rule_index;
                expand_data.num_pattern_edges = rule.num_lhs_edges;
                expand_data.matched_edges[0] = candidate;
                expand_data.match_order[0] = first_pidx;
                expand_data.num_matched = 1;
                expand_data.binding = binding;
                expand_data.step = data.step;

                if (rule.num_lhs_edges == 1) {
                    MatchRecord m;
                    if (complete_match(expand_data, m)) completed.push_back(m);
                } else {
                    submit_expand_task(expand_data);
                }
            }
        );
    }

    dispatch_expansion(data.state, data.step, completed.data(), completed.size());
}

// =============================================================================
// EXPAND Task Execution
// =============================================================================

void ParallelEvolutionEngine::execute_expand_task(const ExpandTaskData& data) {
    MatchTaskGuard join_guard(*this, data.state, data.step);

    if (should_stop_.load(std::memory_order_relaxed)) return;
    if (data.step > max_steps_) return;

    // Early exit if rewrites are impossible due to limits
    if (!can_create_states_at_step(data.step + 1)) return;
    if (!can_have_more_children(data.state)) return;

    DEBUG_LOG("EXEC_EXPAND state=%u rule=%u matched=%u/%u step=%u",
              data.state, data.rule_index, data.num_matched, data.num_pattern_edges, data.step);

    // Check if complete (shouldn't happen, but safety check)
    if (data.is_complete()) {
        MatchRecord m;
        if (complete_match(data, m)) dispatch_expansion(data.state, data.step, &m, 1);
        return;
    }

    const State& s = hg_->get_state(data.state);
    const RewriteRule& rule = rules_[data.rule_index];

    // Edge accessor
    auto get_edge = [this](EdgeId eid) -> const Edge& {
        return hg_->get_edge(eid);
    };

    // Next pattern edge in the rule's optimized join order, skipping edges already
    // matched (the seed may be an arbitrary position for delta matching).
    uint32_t matched_mask = 0;
    for (uint8_t i = 0; i < data.num_matched; ++i) {
        matched_mask |= (1u << data.match_order[i]);
    }
    uint8_t pattern_idx = rule.num_lhs_edges;
    for (uint8_t k = 0; k < rule.num_lhs_edges; ++k) {
        uint8_t pidx = rule.match_order[k];
        if (!(matched_mask & (1u << pidx))) { pattern_idx = pidx; break; }
    }
    if (pattern_idx >= rule.num_lhs_edges) return;

    const PatternEdge& pattern_edge = rule.lhs[pattern_idx];
    const EdgeSignature& pattern_sig = rule.lhs_sig[pattern_idx];
    const CompatibleSignatureCache& sig_cache = rule.lhs_cache[pattern_idx];

    // Completions of the LAST pattern edge are siblings: same parent, same rule, differing in
    // one edge. They are the natural expansion batch -- collected here and applied together
    // below, so the parent's data is loaded once for the whole family instead of once per
    // child on whichever worker happened to steal it.
    SVec<MatchRecord> completed;

    // Generate candidates
    generate_candidates(
        pattern_edge, pattern_sig, sig_cache,
        data.binding, s.edges,
        hg_->signature_index(), hg_->inverted_index(), get_edge,
        [&](EdgeId candidate, const auto& edge) {
            if (should_stop_.load(std::memory_order_relaxed)) return;

            // Skip if already matched
            if (data.contains_edge(candidate)) return;

            VariableBinding extended = data.binding;
            if (!validate_candidate(edge.vertices, edge.arity, pattern_edge, extended)) return;

            // Create new EXPAND task with extended match
            ExpandTaskData new_data = data;
            new_data.matched_edges[new_data.num_matched] = candidate;
            new_data.match_order[new_data.num_matched] = pattern_idx;
            new_data.num_matched++;
            new_data.binding = extended;

            if (new_data.is_complete()) {
                MatchRecord m;
                if (complete_match(new_data, m)) completed.push_back(m);
            } else {
                submit_expand_task(new_data);
            }
        }
    );

    dispatch_expansion(data.state, data.step, completed.data(), completed.size());
}

// =============================================================================
// SINK Task Execution
// =============================================================================

bool ParallelEvolutionEngine::complete_match(const ExpandTaskData& data, MatchRecord& out) {
    if (should_stop_.load(std::memory_order_relaxed)) return false;
    if (data.step > max_steps_) return false;

    // Early exit if rewrites are impossible due to limits
    if (!can_create_states_at_step(data.step + 1)) return false;
    if (!can_have_more_children(data.state)) return false;

    DEBUG_LOG("EXEC_SINK state=%u rule=%u matched=%u step=%u",
              data.state, data.rule_index, data.num_matched, data.step);

    // Convert matched edges to pattern order
    EdgeId edges_in_order[MAX_PATTERN_EDGES];
    data.to_pattern_order(edges_in_order);

    // Build the payload on the stack; a duplicate costs no arena. The winner
    // promotes it to an immutable arena-resident MatchCore shared by forwarded copies.
    MatchCore core_tmp;
    core_tmp.rule_index = data.rule_index;
    core_tmp.num_edges = data.num_pattern_edges;
    core_tmp.binding = data.binding;
    for (uint8_t i = 0; i < data.num_pattern_edges; ++i) {
        core_tmp.matched_edges[i] = edges_in_order[i];
    }
    MatchRecord match;
    match.core = &core_tmp;
    match.source_state = data.state;

    // Deduplicate using lock-free ConcurrentMap
    uint64_t h = match.hash();
    // Content dedup, with the same shape as the other claim sites: answer a true duplicate from
    // a lookup so it costs no arena, and only promote when the match may be new.
    if (!claim_match(h, match, [&] {
            // Reached only when the match may be new, so a duplicate never promotes.
            match.core = hg_->arena().template create<MatchCore>(core_tmp);
            return hg_->arena().template create<MatchRecord>(match);
        })) {
        rejected_duplicates_.fetch_add(1, std::memory_order_relaxed);
        return false;  // Already seen
    }

    total_matches_found_.fetch_add(1, std::memory_order_relaxed);
    stats_.new_matches_discovered.fetch_add(1, std::memory_order_relaxed);
    match_join_for(data.state)->matches.fetch_add(1, std::memory_order_acq_rel);

    DEBUG_LOG("SINK state=%u rule=%u hash=%lu step=%u", data.state, data.rule_index, h, data.step);

    // Store match for collection/forwarding
    if (enable_match_forwarding_) {
        store_match_for_state(data.state, match, true);
        push_match_to_children(data.state, match, data.step, PushSite::Discovery);
    }

    // Thinned out: the caller gets nothing to expand. Returning false here is not "duplicate"
    // -- it is "not yours to rewrite", which is the same instruction to the caller.
    if (transition_rate_ < 1.0 &&
        !transition_survives_spined(data.state, canonical_transition_key(data.state, match), 4)) return false;

    out = match;
    return true;
}

}  // namespace hypergraph
