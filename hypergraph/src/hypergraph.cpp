#include "hgcommon/core.hpp"
#include "hgcommon/explore_depth_core.hpp"
#include "hgcommon/rendezvous.hpp"
#include "hgcommon/quotient_multiplicity_core.hpp"
#include "hgcommon/phase_timing.hpp"
#include "hgcommon/namespace.hpp"
// hypergraph.cpp - Implementation of Hypergraph class non-template methods

#include "hypergraph/hypergraph.hpp"
#include "hgcommon/reach_core.hpp"
#include "hypergraph/scratch_alloc.hpp"

#include "hypergraph/ir_canonicalization.hpp"
#include "hgcommon/ir_core.hpp"
#include "hgcommon/content_core.hpp"
#include "hgcommon/canonical_form_core.hpp"
#include "hgcommon/dedup_claim_core.hpp"
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
    size_t requested_arity,
    EventId creator_event,
    uint32_t step
) {
    return create_edge_at(counters_.alloc_edge(), vertices, requested_arity, creator_event, step);
}

void Hypergraph::alloc_edges_and_vertices(uint32_t num_edges, uint32_t num_vertices,
                                          EdgeId& first_edge, VertexId& first_vertex) {
    counters_.alloc_edges_and_vertices(num_edges, num_vertices, first_edge, first_vertex);
}

EdgeId Hypergraph::create_edge_at(
    EdgeId eid,
    const VertexId* vertices,
    size_t requested_arity,
    EventId creator_event,
    uint32_t step
) {
    // Downstream code (pattern matcher, EdgeSignature) uses fixed-size MAX_ARITY
    // buffers on the stack. Reject over-arity edges rather than silently corrupt.
    //
    // The parameter is wide enough to hold whatever the caller counted: a caller that
    // narrowed to the storage width first would present 260 vertices as 4 and pass a check
    // MAX_ARITY makes on the true count.
    if (requested_arity > MAX_ARITY) {
        throw std::length_error("Hypergraph::create_edge: arity exceeds MAX_ARITY");
    }
    const uint8_t arity = static_cast<uint8_t>(requested_arity);

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

// The depth rung an individualisation-refinement search starts at, per thread: one above the
// deepest level the previous search on this thread reached. The search reports IR_NEED_DEPTH
// when its path reaches the rung, and the attempt below the need is a search run to its limit
// and thrown away -- two of every three attempts on disc-l3a2g2r2 at depth 2 -- while a rung
// above the need costs scratch words and nothing else, since the search stops at a discrete
// partition. Escalation past the hint stays: a state deeper than its predecessor climbs.
namespace {
constexpr uint32_t kIrDepthRungs[] = {1u, 8u, hgcommon::IR_MAX_DEPTH_DEFAULT};
uint32_t& ir_depth_hint() {
    HG_THREAD_LOCAL(uint32_t, hint);
    return hint;
}
inline bool ir_rung_below_hint(uint32_t rung) { return rung < ir_depth_hint(); }
inline void ir_note_search(const hgcommon::IrWork& work) { ir_depth_hint() = work.max_depth + 1; }

// The IR canonical form of a state whose search outran every bounded rung, in the core's flat
// [arity, v0, v1, ...] layout, from IRCanonicalizer::canonicalize_edges. That runs the same core
// to its unbounded depth, so the form equals the one a bounded call would have emitted.
void fallback_canonical_form(const SVec<SVec<VertexId>>& edge_vectors,
                             std::vector<uint32_t>& out) {
    IRCanonicalizer ir;
    const CanonicalizationResult cr = ir.canonicalize_edges(edge_vectors);
    out.clear();
    for (const auto& e : cr.canonical_form.edges) {
        out.push_back(static_cast<uint32_t>(e.size()));
        for (VertexId v : e) out.push_back(static_cast<uint32_t>(v));
    }
}
}  // namespace

StateId Hypergraph::create_state(
    SparseBitset&& edge_set,
    uint32_t step,
    uint64_t canonical_hash,
    EventId parent_event,
    StateId parent_state,
    const EdgeId* produced,
    uint8_t num_produced
) {
    StateId sid = counters_.alloc_state();
    // The state's contribution to its chain, indexed by vertex once, here: every edge of a
    // root, the produced edges of a derived state (ancestry.hpp walks the chain of these).
    const RootVertexEntry* vertex_index = nullptr;
    uint32_t vertex_index_size = 0;
    const EdgeId* delta_edges = nullptr;
    uint32_t num_delta_edges = 0;
    auto index_edges = [&](auto&& for_each_edge) {
        uint32_t n = 0;
        for_each_edge([&](EdgeId eid) { n += get_edge(eid).arity; });
        if (n == 0) return;
        auto* entries = arena_.allocate_array<RootVertexEntry>(n);
        uint32_t k = 0;
        for_each_edge([&](EdgeId eid) {
            const Edge& e = get_edge(eid);
            for (uint8_t i = 0; i < e.arity; ++i) entries[k++] = RootVertexEntry{e.vertices[i], eid};
        });
        std::sort(entries, entries + n, [](const RootVertexEntry& x, const RootVertexEntry& y) {
            return x.vertex != y.vertex ? x.vertex < y.vertex : x.edge < y.edge;
        });
        // A vertex repeated within one edge lists that edge once.
        vertex_index_size = static_cast<uint32_t>(
            std::unique(entries, entries + n, [](const RootVertexEntry& x, const RootVertexEntry& y) {
                return x.vertex == y.vertex && x.edge == y.edge;
            }) - entries);
        vertex_index = entries;
    };
    if (parent_state == INVALID_ID) {
        index_edges([&](auto&& f) { edge_set.for_each(f); });
    } else if (num_produced > 0) {
        index_edges([&](auto&& f) { for (uint8_t i = 0; i < num_produced; ++i) f(produced[i]); });
        auto* ids = arena_.allocate_array<EdgeId>(num_produced);
        for (uint8_t i = 0; i < num_produced; ++i) ids[i] = produced[i];
        delta_edges = ids;
        num_delta_edges = num_produced;
    }
    // Directly construct state at slot sid using emplace_at
    states_.emplace_at(sid, arena_, sid, std::move(edge_set), step, canonical_hash, parent_event);
    note_published_state(sid);
    State& st = states_[sid];
    st.vertex_index = vertex_index;
    st.vertex_index_size = vertex_index_size;
    st.delta_edges = delta_edges;
    st.num_delta_edges = num_delta_edges;
    st.parent_state = parent_state;
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
    StateId new_sid = create_state(std::move(edge_set), step, 0, parent_event,
                                   incr_parent, incr_produced, incr_num_produced);
    const SparseBitset& edges = get_state(new_sid).edges;

    // Reported hash for None/Automatic modes: stored at creation only when event
    // canonicalization is on -- from the same individualization-refinement pass that
    // computes the per-edge ranks the event path identifies consumed/produced edges by.
    // With event canonicalization off it stays 0 and get_or_compute_canonical_hash
    // computes it on first query, so no per-state hash is computed that nothing reads.
    // Separate from map_key, which is what actually decides state identity.
    const bool need_ranks = (event_signature_keys_ != EVENT_SIG_NONE);
    // Use atomic load with acquire to ensure we see the mode set by the main thread.
    const StateCanonicalizationMode mode =
        state_canonicalization_mode_.load(std::memory_order_acquire);
    const bool full = mode != StateCanonicalizationMode::None &&
                      mode != StateCanonicalizationMode::Automatic;
    const bool quotient = full && quotient_causal_.load(std::memory_order_relaxed);
    // The state's IR canonical form, filled by whichever call below computes its IR hash: in
    // Full mode claim_canonical_state compares it, in None and Automatic with event identity on
    // the claim of the IR key in event_canonical_state_map_ does.
    HG_THREAD_LOCAL(std::vector<uint32_t>, form);
    uint64_t ranked_hash = 0;
    if (need_ranks)
        ranked_hash = cache_state_edge_ranks(new_sid, edges, quotient ? nullptr : &form);

    // Canonical identity + dedup key. In Full mode the IR canonical hash is BOTH the
    // canonical identity and the first probe key, computed once (no redundant WL pass);
    // other modes use the fast WL hash for identity + a mode-specific dedup key.
    uint64_t map_key = 0, canonical_hash = 0;
    switch (mode) {
        case StateCanonicalizationMode::None:
            // +1: the dedup key is the raw state id, and canonical_state_map_ reserves 0 as its
            // EMPTY-slot sentinel. Without the offset the first state (id 0) keys to 0, which
            // count_unique() cannot store or count, silently undercounting None by one. The offset
            // keeps ids unique (None never dedups) while lifting id 0 off the sentinel.
            map_key = static_cast<uint64_t>(new_sid) + 1;
            canonical_hash = need_ranks ? ranked_hash : 0;
            break;
        case StateCanonicalizationMode::Automatic:
            // The content hash selects the key and the content words decide it (below).
            map_key = compute_content_ordered_hash(edges);
            canonical_hash = need_ranks ? ranked_hash : 0;
            break;
        case StateCanonicalizationMode::Full:
        default:
            // In quotient mode compute the edge-orbit table and take the canonical hash
            // from the same IR canonicalization (the quotient causal reconstruction needs
            // the orbits; there is no extra canon pass). Otherwise just the dedup hash.
            if (quotient)
                // WARM FILL. Suppressed by HG_CALIBRATE_ORBIT_CACHE_COLD so that every capture
                // misses and rebuilds, which is how the rebuild path gets exercised at all --
                // it is unreachable on a run whose cache is always warm, and an unexercised
                // path is not one the reconstruction may depend on.
#if defined(HG_CALIBRATE_ORBIT_CACHE_COLD)
                canonical_hash = compute_and_cache_state_orbits(new_sid, edges, /*cache=*/false,
                                                                &form);
#else
                canonical_hash = compute_and_cache_state_orbits(new_sid, edges, /*cache=*/true,
                                                                &form);
#endif
            else if (need_ranks)
                canonical_hash = ranked_hash;   // IR, computed with the ranks and the form
            else
                canonical_hash = compute_canonical_hash(edges, &form);
            break;   // no map_key: claim_canonical_state derives the probe keys
    }
    // Any mode whose key hashes to 0 would hit the same EMPTY=0 sentinel; nudge it off (mirrors the
    // GPU's h==0?1:h guard). None is already offset above, so this only ever affects a 0-valued hash.
    map_key = hgcommon::avoid_reserved_keys(map_key);
    StateId existing_or_new;
    bool was_inserted;
    if (full) {
        // The class's key is its identity. It differs from the hash only when the hash is a
        // map sentinel, the key mask is narrowed, or another class held the hash's key.
        const CanonicalClaim claim = claim_canonical_state(new_sid, canonical_hash, form);
        existing_or_new = claim.rep;
        was_inserted = claim.won;
        canonical_hash = claim.key;
    } else if (mode == StateCanonicalizationMode::Automatic) {
        // Automatic identity is equal edge content, so two states whose content hashes collide
        // stay two states.
        const CanonicalClaim claim = claim_content_state(new_sid, map_key, edges);
        existing_or_new = claim.rep;
        was_inserted = claim.won;
    } else {
        auto r = canonical_state_map_.insert_if_absent_waiting(map_key, new_sid);
        existing_or_new = r.first;
        was_inserted = r.second;
    }
    // Under None and Automatic with event identity on, the reported IR key is claimed on the IR
    // form in event_canonical_state_map_, whose record resolves the event path's representative;
    // under Full the class's key above is that key already.
    if (!full && need_ranks && ranked_hash != 0) {
        const CanonicalClaim ir =
            claim_identity(event_canonical_state_map_, ranked_hash & event_key_mask_,
                           form.data(), static_cast<uint32_t>(form.size()), new_sid);
        canonical_hash = ir.key;
    }
    // create_state has already published new_sid, so another thread can be reading this
    // state's canonical_hash (get_or_compute_canonical_hash, get_canonical_state_for_event)
    // while this store runs. Both sides go through atomic_ref: the store carries the hash, or
    // in Full mode the class's key, stored once, and the acquire loads pick it up.
    hgcommon::atomic_ref<uint64_t>(states_[new_sid].canonical_hash)
        .store(canonical_hash, std::memory_order_release);

    // Cache the canonical ID in the state for fast lookup. Released here and acquired by
    // get_canonical_state(); the store itself is what carries the edge, since a bare fence
    // pairs with another fence only through an intervening atomic on the same object.
    hgcommon::atomic_ref<StateId>(states_[new_sid].canonical_id)
        .store(existing_or_new, std::memory_order_release);

    if (!was_inserted) {
        return {existing_or_new, new_sid, false};
    }

    return {new_sid, new_sid, true};
}

namespace {

// The walk is hgcommon::dedup_claim, the rule the match set is claimed by: the hash selects
// the key, and the class's identity decides it at both points the walk can conclude. A probe
// that finds a different class moves to the next key, so two classes whose hashes collide both
// get a key.
//
// `same(v)` is whether the claimant belongs to the class whose map value is `v`; `make()`
// returns the value to publish, called only after a lookup missed and before the first offer;
// `rep_of(v)` is the class's representative. A value is fully written before the offer that
// publishes it, and the map's acquire load of it makes what it refers to visible to the reader.
//
// max_probes is unbounded: every key visited holds a distinct class, and a claim that stopped
// early would have no key to be found under.
template <class V, class Map, class Same, class Make, class RepOf, class OnCollision>
struct KeyedClaim {
    Map& map;
    uint64_t h;
    Same& same;
    Make& make;
    RepOf& rep_of;
    OnCollision& on_collision;
    V mine{};
    bool made = false;
    bool won = false;
    uint32_t rep = INVALID_ID;
    uint64_t key = 0;

    uint32_t max_probes() const { return UINT32_MAX; }
    uint64_t probe_key(uint32_t k) const { return hgcommon::dedup_probe_key(h, k, 0, ~uint64_t{0}); }

    hgcommon::ProbeState probe(uint64_t k) {
        const auto v = map.lookup(k);
        if (!v) return hgcommon::ProbeState::Miss;
        if (!same(*v)) return hgcommon::ProbeState::Collision;
        rep = rep_of(*v); key = k;
        return hgcommon::ProbeState::Duplicate;
    }
    void make_stable() { mine = make(); made = true; }
    hgcommon::ClaimState offer(uint64_t k) {
        const auto [existing, inserted] = map.insert_if_absent(k, mine);
        if (inserted) { rep = rep_of(mine); key = k; return hgcommon::ClaimState::Won; }
        if (!same(existing)) return hgcommon::ClaimState::Collision;
        rep = rep_of(existing); key = k;
        return hgcommon::ClaimState::Duplicate;
    }
    void note_collision() { on_collision(); }
    void note_exhausted() {}
};

template <class V, class Map, class Same, class Make, class RepOf, class OnCollision>
KeyedClaim<V, Map, Same, Make, RepOf, OnCollision>
keyed_claim(Map& map, uint64_t first_key, Same& same, Make& make, RepOf& rep_of,
            OnCollision& on_collision) {
    KeyedClaim<V, Map, Same, Make, RepOf, OnCollision> c{map, first_key, same, make, rep_of,
                                                         on_collision};
    c.won = hgcommon::dedup_claim(c);
    return c;
}

}  // namespace

// A record map: the value is the class's record (representative id and words). A record that
// lost every offer is in the map under no key; it is given back when it is still the top of
// this worker's arena cursor.
Hypergraph::CanonicalClaim Hypergraph::claim_identity(IdentityMap& map, uint64_t first_key,
                                                      const uint32_t* words, uint32_t n,
                                                      uint32_t id) {
    using Rec = const hgcommon::CanonicalFormRecord*;
    uint64_t rec_bytes = 0;
    auto same = [&](Rec r) { return hgcommon::canonical_form_equals(r, words, n); };
    auto make = [&]() -> Rec {
        const uint32_t width = hgcommon::canonical_form_width(words, n);
        rec_bytes = hgcommon::canonical_form_record_bytes(n, width);
        auto* rec = static_cast<hgcommon::CanonicalFormRecord*>(
            arena_.allocate_raw(rec_bytes, alignof(hgcommon::CanonicalFormRecord)));
        hgcommon::canonical_form_encode(id, words, n, width, rec);
        return rec;
    };
    auto rep_of = [](Rec r) { return r->id; };
    auto on_collision = [&] {
        HG_STAT(canonical_key_collisions_.fetch_add(1, std::memory_order_relaxed));
    };
    auto c = keyed_claim<Rec>(map, first_key, same, make, rep_of, on_collision);
    if (!c.won && c.made)
        arena_.release_last(const_cast<hgcommon::CanonicalFormRecord*>(c.mine), rec_bytes);
    return {c.rep, c.key, c.won};
}

Hypergraph::CanonicalClaim Hypergraph::claim_canonical_state(StateId sid, uint64_t hash,
                                                             const std::vector<uint32_t>& form) {
    return claim_identity(canonical_form_map_, hash & canonical_key_mask_, form.data(),
                          static_cast<uint32_t>(form.size()), sid);
}

// A state's edges in content order (hgcommon::content_equal), from its edge ids in id order.
struct HostContentCursor {
    const SegmentedArray<Edge>& edges;
    const SVec<EdgeId>& ids;
    uint32_t at = 0;
    bool next(uint32_t& arity, const uint32_t*& vertices) {
        if (at >= ids.size()) return false;
        const Edge& e = edges[ids[at++]];
        arity = e.arity;
        vertices = e.vertices;
        return true;
    }
};

Hypergraph::CanonicalClaim Hypergraph::claim_content_state(StateId sid, uint64_t hash,
                                                           const SparseBitset& edges) {
    auto mk = worker_scratch().mark();
    // The claimant's edge ids, listed on the first key hit only: a claim that misses never
    // compares.
    SVec<EdgeId> mine, theirs;
    bool listed = false;
    auto same = [&](StateId rep) {
        if (!listed) {
            edges.for_each([&](EdgeId e) { mine.push_back(e); });
            listed = true;
        }
        theirs.clear();
        states_[rep].edges.for_each([&](EdgeId e) { theirs.push_back(e); });
        HostContentCursor a{edges_, mine}, b{edges_, theirs};
        return hgcommon::content_equal(a, b);
    };
    auto make = [&] { return sid; };
    auto rep_of = [](StateId r) { return r; };
    auto on_collision = [&] {
        HG_STAT(canonical_key_collisions_.fetch_add(1, std::memory_order_relaxed));
    };
    auto c = keyed_claim<StateId>(canonical_state_map_, hash & canonical_key_mask_, same, make,
                                  rep_of, on_collision);
    worker_scratch().release(mk);
    return {c.rep, c.key, c.won};
}

uint32_t Hypergraph::event_values_of(EventId e, uint64_t* out, bool count_fallbacks) {
    const Event& ev = events_[e];
    const EventSignatureKeys keys = event_signature_keys_;
    // Ranks of the consumed and produced edges, in match and RHS order. A missing rank means no
    // rank table for that state; the raw edge id stands in and is COUNTED, because such a
    // signature is not an isomorphism invariant and a caller comparing event counts across runs
    // needs to know it happened. Counted in every build: the FFI surfaces the count as the
    // EventSigRawFallback warning.
    uint32_t consumed_ranks[MAX_PATTERN_EDGES];
    uint32_t produced_ranks[MAX_PATTERN_EDGES];
    const uint8_t nc = ev.num_consumed < MAX_PATTERN_EDGES
                           ? ev.num_consumed : static_cast<uint8_t>(MAX_PATTERN_EDGES);
    const uint8_t np = ev.num_produced < MAX_PATTERN_EDGES
                           ? ev.num_produced : static_cast<uint8_t>(MAX_PATTERN_EDGES);
    auto rank = [&](const EdgeRankTable* t, EdgeId edge) {
        uint32_t r = edge_rank_in(t, edge);
        if (r == UINT32_MAX) {
            if (count_fallbacks) event_sig_raw_fallbacks_.fetch_add(1, std::memory_order_relaxed);
            r = edge;
        }
        return r;
    };
    if (keys & EventKey_ConsumedEdges) {
        const EdgeRankTable* t = edge_rank_table(ev.input_state);
        for (uint8_t i = 0; i < nc; ++i) consumed_ranks[i] = rank(t, ev.consumed_edges[i]);
    }
    if (keys & EventKey_ProducedEdges) {
        const EdgeRankTable* t = edge_rank_table(ev.output_state);
        for (uint8_t i = 0; i < np; ++i) produced_ranks[i] = rank(t, ev.produced_edges[i]);
    }
    const State& canonical_out =
        get_state(get_canonical_state_for_event(ev.output_state));
    return hgcommon::event_signature_values(
        keys,
        (keys & EventKey_InputState)  ? get_or_compute_canonical_hash(ev.input_state)  : 0,
        (keys & EventKey_OutputState) ? get_or_compute_canonical_hash(ev.output_state) : 0,
        canonical_out.step, ev.rule_index, consumed_ranks, nc, produced_ranks, np, out);
}

// Under an event identity mode: the event's signature values, claimed; a duplicate records the
// class's first event as its canonical id. Every event's reported signature is its class's key.
Hypergraph::EventIdentity Hypergraph::assign_event_identity(EventId eid) {
    if (event_signature_keys_ == EVENT_SIG_NONE) return {eid, true};
    uint64_t values[hgcommon::EVENT_SIG_MAX_VALUES];
    const uint32_t n = event_values_of(eid, values, true);
    const CanonicalClaim claim =
        claim_event(eid, values, n, hgcommon::event_signature_of_values(values, n));
    Event& ev = events_[eid];
    ev.signature = claim.key;
    if (!claim.won) {
        ev.canonical_event_id = claim.rep;
        return {claim.rep, false};
    }
    canonical_event_count_.fetch_add(1, std::memory_order_relaxed);
    return {eid, true};
}

Hypergraph::CanonicalClaim Hypergraph::claim_event(EventId e, const uint64_t* values, uint32_t n,
                                                   uint64_t sig) {
    auto same = [&](EventId rep) {
        uint64_t theirs[hgcommon::EVENT_SIG_MAX_VALUES];
        if (event_values_of(rep, theirs, false) != n) return false;
        for (uint32_t i = 0; i < n; ++i)
            if (theirs[i] != values[i]) return false;
        return true;
    };
    auto make = [&] { return e; };
    auto rep_of = [](EventId r) { return r; };
    auto on_collision = [&] {
        HG_STAT(canonical_key_collisions_.fetch_add(1, std::memory_order_relaxed));
    };
    auto c = keyed_claim<EventId>(canonical_event_map_, sig & event_key_mask_, same, make, rep_of,
                                  on_collision);
    return {c.rep, c.key, c.won};
}

bool Hypergraph::explore_depth_cas(StateId canonical_id, uint32_t& expected, uint32_t desired) {
    hgcommon::atomic_ref<uint32_t> known(states_[canonical_id].explore_depth);
    return known.compare_exchange_weak(expected, desired, std::memory_order_acq_rel,
                                       std::memory_order_acquire);
}

bool Hypergraph::try_lower_explore_depth(StateId canonical_id, uint32_t depth) {
    if (canonical_id == INVALID_ID) return false;
    struct Ops {
        Hypergraph* hg;
        uint32_t depth_load(uint32_t s) const { return hg->explore_depth_of(s); }
        bool depth_cas(uint32_t s, uint32_t& e, uint32_t d) { return hg->explore_depth_cas(s, e, d); }
    } ops{this};
    return hgcommon::explore_try_lower(ops, canonical_id, depth);
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
uint64_t Hypergraph::cache_state_edge_ranks(StateId state_id, const SparseBitset& edges,
                                            std::vector<uint32_t>* out_form) {
    if (out_form) out_form->clear();
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
        uint32_t* form = nullptr;
        if (out_form) {
            out_form->resize(hgcommon::ir_canonical_form_words(n, total_occ));
            form = out_form->data();
        }
        for (uint32_t depth : kIrDepthRungs) {
            if (ir_rung_below_hint(depth)) continue;
            const uint64_t words = hgcommon::ir_scratch_words(n_verts, n, total_occ, depth);
            auto* scratch = static_cast<uint32_t*>(
                worker_scratch().allocate_raw((words + 2) * sizeof(uint32_t), alignof(uint64_t)));
            hgcommon::IrWork work{};
            auto r = hgcommon::ir_canonical_hash(ea.data(), eoff.data(), ev.data(),
                                                 n, n_verts, total_occ, scratch, depth,
                                                 ranks.data(), hgcommon::IR_HOST_GENERATORS,
                                                 nullptr, nullptr, form, nullptr, &work);
            book_ir_call(work, r.status == hgcommon::IR_NEED_DEPTH);
            if (r.status == hgcommon::IR_OK) { ir_note_search(work); hash = r.hash; ok = true; break; }
            if (r.status == hgcommon::IR_EMPTY) break;
        }
        if (!ok) {
            HG_STAT(ir_fallbacks_.fetch_add(1, std::memory_order_relaxed));
            IRCanonicalizer ir;
            HG_THREAD_LOCAL(std::vector<uint32_t>, fallback_ranks);
            hash = ir.compute_canonical_hash_with_edge_rank(edge_vectors, fallback_ranks);
            for (uint32_t i = 0; i < n; ++i) ranks[i] = fallback_ranks[i];
            if (out_form) fallback_canonical_form(edge_vectors, *out_form);
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

// The state whose class resolves the event path's representative: under Full the class's
// record in canonical_form_map_, otherwise the IR key's record in event_canonical_state_map_,
// both keyed by the state's stored canonical hash.
StateId Hypergraph::get_canonical_state_for_event(StateId raw_state) const {
        if (raw_state == INVALID_ID) return INVALID_ID;

        // Get the isomorphism-invariant hash for this state. Written concurrently by
        // create_or_get_canonical_state and get_or_compute_canonical_hash, so acquire it.
        const State& state = get_state(raw_state);
        uint64_t hash = hgcommon::atomic_ref<uint64_t>(const_cast<uint64_t&>(state.canonical_hash))
            .load(std::memory_order_acquire);

        // If hash is 0, the state's hash wasn't computed - fall back to raw state
        if (hash == 0) return raw_state;

        const IdentityMap& map = is_full_canonicalization() ? canonical_form_map_
                                                             : event_canonical_state_map_;
        if (auto rec = map.lookup(hash)) return (*rec)->id;
        return raw_state;
    }

const EdgeRankTable* Hypergraph::edge_rank_table(StateId state_id) const {
    auto r = state_edge_rank_tables_.lookup(static_cast<uint64_t>(state_id) + 1);
    return r.has_value() ? *r : nullptr;
}

uint32_t Hypergraph::edge_rank_in(const EdgeRankTable* t, EdgeId edge) {
    if (!t) return UINT32_MAX;
    uint32_t lo = 0, hi = t->n;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (t->edges[mid] < edge) lo = mid + 1; else hi = mid;
    }
    return (lo < t->n && t->edges[lo] == edge) ? t->rank[lo] : UINT32_MAX;
}

void Hypergraph::reserve_vertices(VertexId max_id) {
        uint64_t w = counters_.next_edge_vertex.load(std::memory_order_relaxed);
        while (GlobalCounters::vertex_field(w) <= max_id) {
            const uint64_t raised = (uint64_t(max_id) + 1) * GlobalCounters::kVertexOne +
                                    GlobalCounters::edge_field(w);
            if (counters_.next_edge_vertex.compare_exchange_weak(w, raised,
                                                                 std::memory_order_relaxed)) {
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
    uint64_t hash = compute_canonical_hash(state.edges);

    // Published only over 0: racing on-demand writers compute the same value, and a value the
    // state's creator stored (in Full mode the class's key, which can differ from the hash) is
    // never overwritten. A failed exchange returns the stored value.
    uint64_t expected = 0;
    if (!atomic_hash.compare_exchange_strong(expected, hash, std::memory_order_acq_rel,
                                             std::memory_order_acquire))
        return expected;
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

    // Allocate and copy edge arrays
    EdgeId* cons = arena_.allocate_array<EdgeId>(num_consumed);
    std::memcpy(cons, consumed, num_consumed * sizeof(EdgeId));

    EdgeId* prod = arena_.allocate_array<EdgeId>(num_produced);
    std::memcpy(prod, produced, num_produced * sizeof(EdgeId));

    // Stored before its identity is claimed: a claim that hits this event's key reads its
    // signature values from the Event.
    events_.emplace_at(eid, arena_, eid, input_state, output_state, rule_index,
                       cons, num_consumed, prod, num_produced, INVALID_ID);
    const EventIdentity id = assign_event_identity(eid);
    note_published_event(eid);

    // CRITICAL: Release fence to ensure event data is visible
    std::atomic_thread_fence(std::memory_order_release);

    return {eid, id.canonical, id.is_canonical};
}

EventId Hypergraph::create_genesis_event(StateId initial_state, const EdgeId* edges,
                                         size_t requested_num_edges) {
    // A genesis event produces every initial edge, and Event::num_produced is one byte, so
    // 255 is the largest initial state this event can describe. The count arrives at full
    // width and is checked here: narrowing it at the call site would present a 256-edge state
    // as zero, and the event would then register a producer for none of its edges while the
    // run reported no error.
    if (requested_num_edges > MAX_GENESIS_EDGES) {
        throw std::length_error(
            "Hypergraph::create_genesis_event: an initial state of more than "
            "255 edges cannot be described by a genesis event; evolve with "
            "genesis events disabled");
    }
    const uint8_t num_edges = static_cast<uint8_t>(requested_num_edges);

    // Ensure genesis state exists
    StateId genesis = get_or_create_genesis_state();

    // Allocate event ID
    EventId eid = counters_.alloc_event();

    EdgeId* produced = arena_.allocate_array<EdgeId>(num_edges);
    std::memcpy(produced, edges, num_edges * sizeof(EdgeId));

    // Directly construct event at slot eid using emplace_at; its identity is claimed like every
    // other event's.
    events_.emplace_at(eid, arena_, eid, genesis, initial_state,
                       static_cast<RuleIndex>(-1),
                       nullptr, 0,  // consumed_edges (none)
                       produced, num_edges,  // produced_edges
                       INVALID_ID);
    assign_event_identity(eid);
    note_published_event(eid);

    // CRITICAL: Release fence
    std::atomic_thread_fence(std::memory_order_release);

    // Register this event as the producer of all initial edges, keyed by the initial
    // state's canonical edge identities (the same keys consumers of those edges will mint).
    // num_edges is bounded by MAX_GENESIS_EDGES above, so the buffer holds every initial edge.
    CanonicalEdgeKey init_keys[MAX_GENESIS_EDGES + 1];
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

uint64_t Hypergraph::compute_canonical_hash(const SparseBitset& edges,
                                            std::vector<uint32_t>* out_form) const {
    hgcommon::PhaseTimer _pt(hgcommon::Phase::Canon);
    HG_STAT(canonical_hash_computations_.fetch_add(1, std::memory_order_relaxed));
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
        if (out_form) out_form->clear();
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
    HG_THREAD_LOCAL(std::vector<uint32_t>, local_index);
    HG_THREAD_LOCAL(std::vector<uint32_t>, stamp);
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
    uint32_t* form = nullptr;
    if (out_form) {
        out_form->resize(hgcommon::ir_canonical_form_words(n_edges, total_occ));
        form = out_form->data();
    }
    for (uint32_t depth : kIrDepthRungs) {
        if (ir_rung_below_hint(depth)) continue;
        const uint64_t words =
            hgcommon::ir_scratch_words(n_verts, n_edges, total_occ, depth);
        // Raw, so the buffer is not zeroed on the way in: the core writes every word it
        // later reads. 8-byte aligned for the uint64 views it takes inside the span.
        auto* scratch = static_cast<uint32_t*>(
            worker_scratch().allocate_raw((words + 2) * sizeof(uint32_t), alignof(uint64_t)));
        hgcommon::IrWork work{};
        auto r = hgcommon::ir_canonical_hash(
            ea.data(), eoff.data(), ev.data(), n_edges, n_verts, total_occ, scratch, depth,
            nullptr, hgcommon::IR_HOST_GENERATORS, nullptr, nullptr, form, nullptr, &work);
        book_ir_call(work, r.status == hgcommon::IR_NEED_DEPTH);
        if (r.status == hgcommon::IR_OK) {
            ir_note_search(work);
            worker_scratch().release(mk);
            return r.hash;
        }
        if (r.status == hgcommon::IR_EMPTY) break;
    }
    HG_STAT(ir_fallbacks_.fetch_add(1, std::memory_order_relaxed));

    // A state whose individualization path outruns even the largest depth: fall back to the
    // unbounded-depth implementation, which allocates per level.
    SVec<SVec<VertexId>> edge_vectors;
    edges.for_each([&](EdgeId eid) {
        const Edge& e = edges_[eid];
        edge_vectors.emplace_back(e.vertices, e.vertices + e.arity);
    });
    IRCanonicalizer ir;
    uint64_t h = ir.compute_canonical_hash(edge_vectors);
    if (out_form) fallback_canonical_form(edge_vectors, *out_form);
    worker_scratch().release(mk);
    return h;
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
//
// The edges come as ids into the edge table and are read into the core's flat form directly:
// the run per edge is [arity | vertices...], so nothing is copied twice and nothing is
// allocated per edge (an intermediate vector of vectors was one scratch allocation per edge
// per state, ~33 per state on wpp depth 7).
uint64_t ir_hash_and_orbits(const SegmentedArray<Edge>& edge_table,
                            const EdgeId* ids, uint32_t n,
                            std::vector<uint32_t>& orbit,
                            std::vector<uint32_t>& klass,
                            std::vector<uint32_t>& rank,
                            Hypergraph::IrBooking& booking,
                            std::vector<uint32_t>* out_form = nullptr) {
    orbit.assign(n, 0);
    klass.assign(n, 0);
    if (out_form) out_form->clear();
    if (n == 0) return EMPTY_STATE_CANONICAL_HASH;

    uint32_t total_occ = 0;
    for (uint32_t i = 0; i < n; ++i) total_occ += edge_table[ids[i]].arity;
    auto* ea   = static_cast<uint8_t*>(worker_scratch().allocate_raw(n, alignof(uint8_t)));
    auto* eoff = static_cast<uint32_t*>(worker_scratch().allocate_raw(sizeof(uint32_t) * n, alignof(uint32_t)));
    auto* ev   = static_cast<uint32_t*>(worker_scratch().allocate_raw(sizeof(uint32_t) * total_occ, alignof(uint32_t)));
    // Local vertex indices in encounter order. The core's result does not depend on the order
    // they are assigned in: the only place an index is read as a value is the initial
    // partition's tie-break, which orders vertices WITHIN a cell, and no output reads
    // within-cell order.
    ScratchIdMap local(total_occ * 2);
    uint32_t n_verts = 0;
    uint32_t occ = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const Edge& e = edge_table[ids[i]];
        eoff[i] = occ;
        ea[i] = e.arity;
        for (uint8_t j = 0; j < e.arity; ++j) {
            uint32_t idx;
            if (local.find_or_insert(static_cast<uint32_t>(e.vertices[j]), n_verts, idx)) ++n_verts;
            ev[occ++] = idx;
        }
    }

    // The static rungs, then the true bound: a search individualises at most one vertex per
    // level, so n_verts + 1 admits every state -- a maximal-symmetry star needs a level per
    // leaf and fell through every static rung into the reference fallback (over 300 s at 256
    // leaves against milliseconds in the search).
    const uint32_t rung_cap = n_verts + 1;
    const uint32_t rungs[] = {kIrDepthRungs[0], kIrDepthRungs[1], kIrDepthRungs[2],
                              rung_cap > hgcommon::IR_MAX_DEPTH_DEFAULT ? rung_cap : 0u};
    uint32_t* form = nullptr;
    if (out_form) {
        out_form->resize(hgcommon::ir_canonical_form_words(n, total_occ));
        form = out_form->data();
    }
    for (uint32_t depth : rungs) {
        if (depth == 0u) continue;
        if (ir_rung_below_hint(depth)) continue;
        for (uint32_t gens = hgcommon::IR_HOST_GENERATORS; gens <= (1u << 16); gens *= 4u) {
            const uint64_t words = hgcommon::ir_scratch_words(n_verts, n, total_occ, depth, gens);
            auto* scratch = static_cast<uint32_t*>(worker_scratch().allocate_raw(
                (words + 2) * sizeof(uint32_t), alignof(uint64_t)));
            hgcommon::IrWork work{};
            auto r = hgcommon::ir_canonical_hash(
                ea, eoff, ev, n, n_verts, total_occ, scratch, depth,
                rank.data(), gens, orbit.data(), klass.data(), form, nullptr, &work);
            booking.calls += 1;
            booking.searched += work.searched;
            booking.leaves += work.leaves;
            booking.nodes += work.nodes;
            booking.depth_sum += work.max_depth;
            if (r.status == hgcommon::IR_NEED_DEPTH || r.status == hgcommon::IR_NEED_GENERATORS)
                booking.retries += 1;
            if (r.status == hgcommon::IR_OK)    { ir_note_search(work); return r.hash; }
            if (r.status == hgcommon::IR_EMPTY) {
                // Every edge has arity 0: the form is one arity word per edge.
                if (out_form) out_form->assign(n, 0u);
                return EMPTY_STATE_CANONICAL_HASH;
            }
            if (r.status == hgcommon::IR_NEED_DEPTH) break;
        }
    }
    booking.fallback = true;
    // Past every depth AND generator budget above: the unbounded implementation rather than
    // orbits the automorphism group does not license. It takes the edges as vectors, built
    // here and only here.
    SVec<SVec<VertexId>> edge_vecs;
    edge_vecs.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const Edge& e = edge_table[ids[i]];
        edge_vecs.emplace_back(e.vertices, e.vertices + e.arity);
    }
    IRCanonicalizer ir;
    const uint64_t h =
        ir.compute_canonical_hash_with_edge_orbits(edge_vecs, orbit, &klass, rank.data());
    if (out_form) fallback_canonical_form(edge_vecs, *out_form);
    return h;
}

}  // namespace

// `cache` is false only under HG_CALIBRATE_ORBIT_CACHE_COLD, which suppresses the warm fill so
// that every capture has to take qc_orbits_or_build's rebuild path. The hash is returned either
// way, so the state's identity does not depend on the arm.
uint64_t Hypergraph::compute_and_cache_state_orbits(StateId s, const SparseBitset& edges,
                                                    bool cache, std::vector<uint32_t>* out_form) {
    hgcommon::PhaseTimer _pt(hgcommon::Phase::Canon);
    if (out_form) out_form->clear();
    HG_STAT(canonical_hash_computations_.fetch_add(1, std::memory_order_relaxed));
    // Materialize the state's edges (id-sorted via SparseBitset iteration) into scratch,
    // run the exact IR canonicalization with edge orbits, then copy a compact table into
    // the persistent arena and publish it under the state id. Called once per state on its
    // creating thread, so no same-state race; insert_if_absent is a belt-and-braces guard.
    auto mk = worker_scratch().mark();
    SVec<EdgeId> ids;
    std::atomic_thread_fence(std::memory_order_acquire);
    edges.for_each([&](EdgeId eid) { ids.push_back(eid); });

    const uint32_t n = static_cast<uint32_t>(ids.size());
    EdgeId* arr_edges = arena_.allocate_array<EdgeId>(n ? n : 1);
    uint32_t* arr_orbit = arena_.allocate_array<uint32_t>(n ? n : 1);
    uint32_t* arr_slot  = arena_.allocate_array<uint32_t>(n ? n : 1);
    uint32_t* arr_class = arena_.allocate_array<uint32_t>(n ? n : 1);
    uint32_t* arr_rank  = arena_.allocate_array<uint32_t>(n ? n : 1);
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
        HG_THREAD_LOCAL(std::vector<uint32_t>, orbit);
        HG_THREAD_LOCAL(std::vector<uint32_t>, klass);
        HG_THREAD_LOCAL(std::vector<uint32_t>, rank);
        orbit.assign(n, 0);
        klass.assign(n, 0);
        rank.assign(n, 0);
        IrBooking booking{};
        hash = ir_hash_and_orbits(edges_, ids.data(), n, orbit, klass, rank, booking, out_form);
        book_ir(booking);
        // ids are already ascending (SparseBitset iterates in id order), orbit is parallel.
        for (uint32_t i = 0; i < n; ++i) {
            arr_edges[i] = ids[i];
            arr_orbit[i] = orbit[i];
            arr_class[i] = klass[i];
            arr_rank[i] = rank[i];
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
    tbl->slot = arr_slot; tbl->klass = arr_class; tbl->rank = arr_rank;
    // +1: the map reserves 0 as its EMPTY-slot sentinel, so a raw key of StateId 0 can never
    // be stored or found -- the initial state would silently have no orbit table, which
    // skipped INIT seeding in the producer-set DP and dropped the root class's matches from
    // the reconstruction. Same offset, same reason, as the None-mode dedup key.
    if (cache) state_orbit_tables_.insert_if_absent(static_cast<uint64_t>(s) + 1, tbl);
    return hash;
}

// =============================================================================
// Quotient reconstruction: the depth bound
// =============================================================================

int Hypergraph::raise_quotient_max_steps(int max_steps) {
    int old = qc_max_steps_.load(std::memory_order_relaxed);
    while (max_steps > old &&
           !qc_max_steps_.compare_exchange_weak(old, max_steps, std::memory_order_relaxed)) {
    }
    if (max_steps <= old) return -1;
    return old;
}

void Hypergraph::quotient_redrive_point(uint64_t state_hash, uint32_t depth) {
    // The point stood at the old bound, so its mass was not passed on and its instances never
    // met a match. With the bound already raised, driving it restarts the cascade, and the
    // deeper points it reaches drive themselves inline from here.
    if (!quotient_reconstruction_.load(std::memory_order_relaxed)) return;
    if (quotient_multiplicity()) {
        qm_cascade([&](QmCtx& c) {
            if (c.claim_queued(state_hash, depth)) c.push(state_hash, depth);
        });
    }
    if (!quotient_replay()) return;
    for_each_instance_at(state_hash, depth, [&](const QcInstance& inst) {
        for_each_expansion_match(state_hash, [&](const SlotMatch& m) {
            qc_apply(inst, m, state_hash, depth);
        });
    });
}

uint32_t Hypergraph::alloc_instance_id() {
    const int w = arena_worker_index();
    if (w < 0) {
        qc_instances_made_outside_.fetch_add(1, std::memory_order_relaxed);
        return qc_next_instance_.fetch_add(1, std::memory_order_relaxed);
    }
    IdBlock& b = qc_inst_blocks_[w];
    if (b.next == b.end) {
        b.next = qc_next_instance_.fetch_add(kIdBlock, std::memory_order_relaxed);
        b.end = b.next + kIdBlock;
    }
    ++b.made;
    return b.next++;
}

void Hypergraph::quotient_causal_seed(StateId initial_state, int max_steps) {
    qc_max_steps_.store(max_steps, std::memory_order_relaxed);
    // Through the builder: a miss here leaves the root class with no root instance, so the whole
    // reconstruction hangs off nothing and every relation under it is absent.
    const EdgeOrbitTable* orb = qc_orbits_or_build(initial_state);
    const uint64_t h = get_state(initial_state).canonical_hash;

    // Seed the per-instance reconstruction with the one instance of the initial state; its
    // edges have no producer.
    if (orb && quotient_multiplicity())
        qm_cascade([&](QmCtx& c) { hgcommon::qm_credit(c, h, 0, 1); });
    if (orb && quotient_replay()) {
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


void Hypergraph::qc_record_causal(uint32_t producer, uint32_t consumer, bool distinct_pair) {
    // Per-consumed-edge relationships (the T1 multiset) count every occurrence. The count is
    // a SEMANTIC observable (num_reconstructed_causal_edges, the full-capture twin of
    // causal_graph().num_causal_edges()), not a cost diagnostic, so it is maintained in every
    // build: the occurrences are deliberately not stored -- adjacent repeats are skipped, not
    // recorded -- so no enumeration can recover this number after the fact. A plain increment
    // on this worker's own 64-byte slot, no shared line, no RMW.
    ++qc_slot(qc_ctr_).causal_edges;

    // NO DEDUP STRUCTURE, and none is needed. `consumer` is the event this application just
    // minted, so the pair cannot repeat across applications; within this one the caller has
    // already marked the adjacent repeats, which are the only other way it can repeat.
    // Appending to this worker's own list touches no line another worker reads.
    if (!distinct_pair) return;
    const int w = arena_worker_index();
    qc_causal_pairs_[w >= 0 ? w : 0].push(qc_pair_key(producer, consumer), arena_);
    HG_STAT(++qc_slot(qc_ctr_).causal_pairs);

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
    inst.id = alloc_instance_id();
    inst.nslots = nslots;
    inst.prod = prod;
    // Claim words only for an instance that will be expanded; one at the bound claims nothing.
    if (static_cast<int>(depth) < maxs) {
        uint32_t class_matches = 0;
        if (auto xr = qc_expansion_.lookup(state_hash))
            class_matches = (*xr)->n.load(std::memory_order_acquire);
        const uint32_t words = hgcommon::qr_claim_words(class_matches);
        inst.claim_cap = hgcommon::qr_claim_bits(words);
        inst.claim_bits = arena_.allocate_array<std::atomic<uint64_t>>(words);
        for (uint32_t i = 0; i < words; ++i)
            inst.claim_bits[i].store(0, std::memory_order_relaxed);
    }

    const uint64_t key = qc_key(state_hash, depth, 0);
    QcInstanceShards* sh;
    auto r = qc_instances_.lookup(key);
    if (r.has_value()) sh = *r;
    else {
        auto* ns = arena_.template create<QcInstanceShards>();
        auto ins = qc_instances_.insert_if_absent(key, ns);
        sh = ins.second ? ns : ins.first;
        if (ins.second && static_cast<int>(depth) >= maxs)
            qc_blocked_.push(QcPoint{state_hash, depth}, arena_);
    }
    const int w = arena_worker_index();
    sh->shard[w < 0 ? 0u : static_cast<uint32_t>(w) % kInstShards].list.push(inst, arena_);

    // Instances at the final depth are recorded but never expanded: the DP runs its match
    // loop over depths 0..steps-1, producing into depth steps and never reading it.
    if (static_cast<int>(depth) >= maxs) return;

    // Publish before scanning, so a match captured concurrently cannot be missed by both
    // sides. The push above is this side's publish; the partner is in qc_capture_expansion.
    hgcommon::rendezvous_barrier<hgcommon::rv::QuotientInstanceMatch>();
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
    // Through the builder for the same reason: a state cannot be aligned onto a frame whose
    // slots are not there, and returning false here drops the capture. A differing edge count
    // is a real mismatch and stays a refusal.
    const EdgeOrbitTable* forb = qc_orbits_or_build(frame);
    if (!forb || forb->n != orb->n) return false;

    // Align this state's edges onto the frame's. The two states are one canonical class, so
    // their canonical forms are equal and the edge at rank r here is the edge at rank r in
    // the frame: that is the isomorphism, defined up to an automorphism, which is exactly the
    // freedom that is harmless -- an automorphism permutes the frame coherently, mapping
    // matches to matches. What is NOT harmless is each state using its own labeling, which
    // is what this removes. Both ranks come from the search each state ran at creation.
    auto mk = worker_scratch().mark();
    SVec<uint32_t> frame_at_rank(orb->n, UINT32_MAX);
    for (uint32_t i = 0; i < forb->n; ++i)
        if (forb->rank[i] < forb->n) frame_at_rank[forb->rank[i]] = i;
    bool aligned = true;
    for (uint32_t i = 0; i < orb->n && aligned; ++i) {
        const uint32_t rk = orb->rank[i];
        const uint32_t fi = rk < orb->n ? frame_at_rank[rk] : UINT32_MAX;
        if (fi == UINT32_MAX) aligned = false;
        else out[i] = forb->slot[fi];
    }
    worker_scratch().release(mk);
    if (!aligned) { HG_STAT(qc_align_badcorr_.fetch_add(1, std::memory_order_relaxed)); return false; }
    qc_check_frame_stable(s, out, orb->n);
    return true;
}

void Hypergraph::qc_check_frame_stable(StateId s, const uint32_t* slots, uint32_t n) {
    uint64_t h = hgcommon::FNV_OFFSET;
    for (uint32_t i = 0; i < n; ++i) h = hgcommon::fnv_hash(h, slots[i]);
    h = hgcommon::avoid_reserved_keys(h);
    auto r = qc_frame_sig_.insert_if_absent(static_cast<uint64_t>(s) + 1, h);
    if (!r.second && r.first != h) HG_STAT(qc_frame_disagree_.fetch_add(1, std::memory_order_relaxed));
}

// The edge-orbit table of a state, built here if it is not cached yet.
//
// state_orbit_tables_ is a CACHE, filled when a state is canonicalized so the reconstruction
// does not re-run IR canonicalization for every event. A miss therefore has to be FILLED. Read
// as "this state has no orbits" it silently removes the match from its class frame, and the
// replay then produces neither the raw events that match would have made nor the causal and
// branchial pairs under them -- a shortfall that leaves the state and canonical event counts
// untouched, so it looks like a run that simply found less.
//
// REBUILDING NEEDS NOTHING FROM ANY OTHER THREAD, which is what makes it the right answer here
// rather than waiting for the table to appear. The table is a function of the state's edge set,
// that set is immutable once the state exists, and insert_if_absent keeps whichever copy lands
// first -- so two threads that both rebuild publish identical slots and the reconstruction is a
// function of the inputs either way.
const EdgeOrbitTable* Hypergraph::qc_orbits_or_build(StateId s) {
    // A lookup on an id that names no state answered null; a rebuild would index the state
    // array with it. Every caller already treats null as "no orbits here", so the bound is
    // checked once rather than at each of them.
    if (s == INVALID_ID || s >= num_states()) return nullptr;
    const EdgeOrbitTable* t = state_orbits(s);
    if (t && t->slot) return t;

    // The first miss keeps its evidence: the state, and whether the entry was absent or present
    // with no slot array. Those have different causes, and one count reports them as one thing.
    HG_STAT(qc_capture_orbit_rebuilds_.fetch_add(1, std::memory_order_relaxed));
    uint64_t none = ~uint64_t{0};
    qc_no_orbits_witness_.compare_exchange_strong(
        none, (static_cast<uint64_t>(t ? 2u : 1u) << 32) | s,
        std::memory_order_release, std::memory_order_relaxed);

    compute_and_cache_state_orbits(s, get_state(s).edges);
    t = state_orbits(s);
    return (t && t->slot) ? t : nullptr;
}

// Both orbit tables list their edges in ascending id: the shared merge walk pairs them
// (hgcommon::qc_for_each_survivor), and the callers read orbit or slot through the indices.
template <typename F>
static void for_each_survivor(const EdgeOrbitTable& in_orb, const EdgeOrbitTable& out_orb,
                              const EdgeId* produced, uint8_t num_produced, F&& f) {
    hgcommon::qc_for_each_survivor(in_orb.edges, in_orb.n, out_orb.edges, out_orb.n,
                                   produced, num_produced, f);
}

void Hypergraph::qc_capture_expansion(EventId e) {
    // Record this match of the expanded representative, in slots, undeduplicated. One
    // canonical state's expansion is defined by exactly one raw state: the first to publish
    // itself here wins, and events of any other raw state in the same class are ignored, so a
    // dedup race cannot double the expansion.
    const Event& ev = get_event(e);
    const EdgeOrbitTable* in_orb = qc_orbits_or_build(ev.input_state);
    const EdgeOrbitTable* out_orb = qc_orbits_or_build(ev.output_state);
    if (!in_orb || !out_orb) {
        // A REBUILD THAT ALSO CAME BACK EMPTY. qc_orbits_or_build recomputes from the state's
        // own edges, so reaching here means the state has no usable edge set at all, which is
        // not something a schedule can produce. Counted rather than asserted because a capture
        // lost is a quiet shortfall in the relations, and a count is what makes it audible.
        HG_STAT(qc_capture_no_orbits_.fetch_add(1, std::memory_order_relaxed));
        return;
    }
    const uint64_t from = get_state(ev.input_state).canonical_hash;

    const uint64_t claim = static_cast<uint64_t>(ev.input_state) + 1;
    auto rep = qc_expansion_rep_.insert_if_absent(from, claim);
    if (!rep.second && rep.first != claim) {         // a different raw state owns this class
        HG_STAT(qc_capture_not_rep_.fetch_add(1, std::memory_order_relaxed));
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
                HG_STAT(qc_align_fail_.fetch_add(1, std::memory_order_relaxed));
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

            SVec<std::pair<uint32_t,uint32_t>> surv;
            for_each_survivor(*in_orb, *out_orb, ev.produced_edges, nprod, [&](uint32_t j, uint32_t i) {
                surv.push_back({in_slot[j], out_slot[i]});
            });
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

    QcExpansion* xp;
    auto r = qc_expansion_.lookup(from);
    if (r.has_value()) xp = *r;
    else {
        auto* nx = arena_.template create<QcExpansion>();
        auto ins = qc_expansion_.insert_if_absent(from, nx);
        xp = ins.second ? nx : ins.first;
    }
    m.local = xp->n.fetch_add(1, std::memory_order_acq_rel);
    LockFreeList<SlotMatch>* lst = &xp->list;
    const auto* node = lst->push(m, arena_);

    if (!quotient_reconstruction_.load(std::memory_order_relaxed)) return;
    if (quotient_multiplicity()) {
        // b_j over the matches linked before this one, then ready, then the mass already
        // standing at this class at every depth. Partner: the run in qm_drain.
        uint64_t b = 0;
        lst->for_each_before(node, [&](const SlotMatch& other) {
            if (hgcommon::qr_consumed_overlap(cs, m.num_consumed, other)) ++b;
        });
        qm_overlaps_.insert_if_absent(static_cast<uint64_t>(m.id) + 1, b + 1);
        // The list's copy: the pass may keep a reference to the match (claim_replay_event).
        qm_cascade([&](QmCtx& c) {
            c.fence();
            for (uint32_t d = 0; d < c.max_steps(); ++d)
                hgcommon::qm_pass(c, node->value, from, d);
        });
    }
    if (!quotient_replay()) return;

    // Match side of the rendezvous: replay this newly-captured match against every instance
    // already standing at this state, at every depth. Publish (the push above) before the
    // scan, so a concurrent instance and match cannot both miss each other. The per-pair claim
    // in qc_apply makes the overlap harmless.
    hgcommon::rendezvous_barrier<hgcommon::rv::QuotientInstanceMatch>();
    // THE SCAN IS SPLIT BY (depth, list). A match captured after its class holds many instances
    // is applied to all of them, and on a rule with few classes that is most of the replay; one
    // thread applying them all is the serial part of the run. The first non-empty unit runs
    // here and the rest go to qc_spawn_ as jobs, which scan after the publish above and so keep
    // the rendezvous; the claim in qc_apply keeps each pair to one application.
    const SlotMatch* stored = &node->value;
    const int maxs = qc_max_steps_.load(std::memory_order_relaxed);
    bool ran_one = false;
    for (int d = 0; d < maxs; ++d) {
        auto ri = qc_instances_.lookup(qc_key(from, static_cast<uint32_t>(d), 0));
        if (!ri.has_value()) continue;
        for (uint32_t l = 0; l < kInstShards; ++l) {
            if ((*ri)->shard[l].list.empty()) continue;
            if (ran_one && qc_spawn_) {
                qc_spawn_(qc_spawn_ctx_, this, stored, from, static_cast<uint32_t>(d), l);
                continue;
            }
            ran_one = true;
            qc_apply_list(stored, from, static_cast<uint32_t>(d), l);
        }
    }
}

void Hypergraph::qc_apply_list(const SlotMatch* m, uint64_t from, uint32_t depth, uint32_t list) {
    auto ri = qc_instances_.lookup(qc_key(from, depth, 0));
    if (!ri.has_value()) return;
    (*ri)->shard[list].list.for_each([&](const QcInstance& inst) { qc_apply(inst, *m, from, depth); });
}

void Hypergraph::register_quotient_transition(EventId e) {
    hgcommon::PhaseTimer _pt(hgcommon::Phase::Quotient);
    qc_capture_expansion(e);
}

void Hypergraph::causal_edge_keys(StateId state, const EdgeId* edges, uint32_t n,
                                  CanonicalEdgeKey* out) const {
    auto raw_key = [](EdgeId e) { return CanonicalEdgeKey{static_cast<uint64_t>(e)}; };
    const bool full = state_canonicalization_mode_.load(std::memory_order_acquire)
                      == StateCanonicalizationMode::Full;
    if (!quotient_causal_.load(std::memory_order_relaxed) || !full) {
        for (uint32_t i = 0; i < n; ++i) out[i] = raw_key(edges[i]);
        return;
    }
    // The key of an edge is its state's canonical hash and its automorphism orbit in that
    // state; an edge the state does not hold keys on its raw id, and is counted.
    auto mint = [](uint64_t chash, bool found, uint32_t orb, EdgeId e) {
        uint64_t key = 14695981039346656037ULL;
        key ^= chash;                    key *= 1099511628211ULL;
        key ^= found ? static_cast<uint64_t>(orb)
                     : (0xFFFFFFFF00000000ULL | e);
        key *= 1099511628211ULL;
        // Bit 63 clear keeps the key below the storage map's reserved sentinel band, and bit 62
        // set keeps it above every raw edge id, which is what tells the causal graph's storage
        // the two kinds apart (CausalGraph::key_is_edge_id). Two hash bits, 62 remain.
        key &= ~(1ULL << 63);
        key |= (1ULL << 62);
        return CanonicalEdgeKey{key};
    };
    // The orbits were computed and cached when the state was created
    // (compute_and_cache_state_orbits), by the same search that set its canonical hash, so
    // the keys are read from that table: one search per state, not one per event end.
    if (const EdgeOrbitTable* tbl = state_orbits(state); tbl && tbl->edges) {
        const uint64_t chash =
            hgcommon::atomic_ref<uint64_t>(const_cast<uint64_t&>(get_state(state).canonical_hash))
                .load(std::memory_order_acquire);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t idx = tbl->index_of(edges[i]);
            out[i] = mint(chash, idx < tbl->n, idx < tbl->n ? tbl->orbit[idx] : 0, edges[i]);
        }
        return;
    }
    // No cached table for this state (a cold cache under HG_CALIBRATE_ORBIT_CACHE_COLD): the
    // same search, run here.
    auto mk = worker_scratch().mark();
    SVec<EdgeId> ids;
    std::atomic_thread_fence(std::memory_order_acquire);
    get_state_edges(state).for_each([&](EdgeId eid) { ids.push_back(eid); });
    if (ids.empty()) {
        worker_scratch().release(mk);
        for (uint32_t i = 0; i < n; ++i) out[i] = raw_key(edges[i]);
        return;
    }
    HG_THREAD_LOCAL(std::vector<uint32_t>, orbit);
    HG_THREAD_LOCAL(std::vector<uint32_t>, klass);
    HG_THREAD_LOCAL(std::vector<uint32_t>, rank);
    rank.assign(ids.size(), 0);
    IrBooking booking{};
    const uint64_t chash = ir_hash_and_orbits(edges_, ids.data(), static_cast<uint32_t>(ids.size()),
                                              orbit, klass, rank, booking);
    book_ir(booking);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t orb = 0;
        bool found = false;
        for (size_t k = 0; k < ids.size(); ++k) {
            if (ids[k] == edges[i]) { orb = orbit[k]; found = true; break; }
        }
        out[i] = mint(chash, found, orb, edges[i]);
    }
    worker_scratch().release(mk);
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

void Hypergraph::set_quotient_multiplicity(bool on) {
    quotient_multiplicity_.store(on, std::memory_order_relaxed);
}

bool Hypergraph::quotient_multiplicity() const {
    return quotient_multiplicity_.load(std::memory_order_relaxed);
}

void Hypergraph::set_quotient_replay(bool on) {
    quotient_replay_.store(on, std::memory_order_relaxed);
}

bool Hypergraph::quotient_replay() const {
    return quotient_replay_.load(std::memory_order_relaxed);
}

bool Hypergraph::quotient_counts_saturated() const {
    return qm_saturated_.load(std::memory_order_relaxed);
}

uint64_t Hypergraph::num_reconstructed_events() const {
    // Under an event-identity mode the observable is the count of distinct identities; with no
    // identity selected every application is its own event and the raw count IS the answer.
    // Mirrors num_events() on the full-capture side.
    if (event_signature_keys() == hgcommon::EVENT_SIG_NONE) return num_reconstructed_raw_events();
    return qc_num_canon_events_.load(std::memory_order_relaxed);
}

uint64_t Hypergraph::num_reconstructed_raw_events() const {
    if (!quotient_replay()) return qm_events_.load(std::memory_order_relaxed);
    uint64_t n = qc_events_made_outside_.load(std::memory_order_relaxed);
    for (int i = 0; i < MAX_ARENA_WORKERS; ++i) n += qc_event_blocks_[i].made;
    return n;
}

size_t Hypergraph::num_reconstructed_instances() const {
    uint64_t n = qc_instances_made_outside_.load(std::memory_order_relaxed);
    for (int i = 0; i < MAX_ARENA_WORKERS; ++i) n += qc_inst_blocks_[i].made;
    return n;
}

size_t Hypergraph::num_reconstructed_causal_edges() const {
    return qc_ctr_total(&QcCounterSlot::causal_edges);
}

size_t Hypergraph::num_reconstructed_causal_pairs(bool transitively_reduced) const {
    if (transitively_reduced) return qc_ctr_total(&QcCounterSlot::reduced_pairs);
    return qc_causal_pairs_count();
}

size_t Hypergraph::applied_scans() const {
    return qc_ctr_total(&QcCounterSlot::applied_scans);
}

size_t Hypergraph::applied_claims() const {
    return qc_applied_.size() + qc_ctr_total(&QcCounterSlot::bit_claims);
}

std::vector<uint32_t> Hypergraph::applied_shape() const {
    std::vector<uint32_t> lens;
    const uint32_t n = qc_applied_slot_bound();
    for (uint32_t i = 0; i < n; ++i) {
        const LockFreeList<QcAppliedMatch>* lst = qc_inst_applied_.find(i);
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

#if HG_ENGINE_STATS
size_t Hypergraph::capture_dropped_no_orbits() const {
    return qc_capture_no_orbits_.load(std::memory_order_relaxed);
}

std::vector<size_t> Hypergraph::reconstruction_applications_by_worker() const {
    std::vector<size_t> out;
    for (const QcCounterSlot& s : qc_ctr_) out.push_back(s.applications);
    return out;
}
#endif

#if HG_ENGINE_STATS
size_t Hypergraph::capture_skipped_not_representative() const {
    return qc_capture_not_rep_.load(std::memory_order_relaxed);
}

size_t Hypergraph::capture_orbit_rebuilds() const {
    return qc_capture_orbit_rebuilds_.load(std::memory_order_relaxed);
}
#endif

StateId Hypergraph::capture_no_orbits_state() const {
    const uint64_t w = qc_no_orbits_witness_.load(std::memory_order_acquire);
    return w == ~uint64_t{0} ? INVALID_ID : static_cast<StateId>(w & 0xFFFFFFFFULL);
}

uint32_t Hypergraph::capture_no_orbits_reason() const {
    const uint64_t w = qc_no_orbits_witness_.load(std::memory_order_acquire);
    return w == ~uint64_t{0} ? 0u : static_cast<uint32_t>(w >> 32);
}

size_t Hypergraph::applied_visits() const {
    return qc_ctr_total(&QcCounterSlot::applied_visits);
}

size_t Hypergraph::captured_matches() const {
    return qc_next_match_id_.load(std::memory_order_relaxed);
}

size_t Hypergraph::reconstruction_instances() const { return num_reconstructed_instances(); }

size_t Hypergraph::applied_unique() const {
    return qc_applied_.count_enumerated() + qc_ctr_total(&QcCounterSlot::bit_claims);
}

// Simple hash of a state's edge SET -- fast, and not isomorphism-invariant. Its neighbours
// (compute_content_ordered_hash, compute_canonical_hash, compute_canonical_hash) were
// already defined here; this was the outlier left in the header.
// The schedule-stable content triple of ONE reconstructed event: hash(input class, output class,
// rule). 0 when the event has no recorded triple.
uint64_t Hypergraph::reconstructed_raw_triple(uint32_t e) const {
    const QcEventContent* c = reconstructed_event_content(e);
    return c ? c->triple_hash() : 0;
}

// THE HOTTEST ACCESSORS IN THE ENGINE: the matcher reads edges through these on
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
    return GlobalCounters::vertex_field(counters_.next_edge_vertex.fetch_add(
        uint64_t(count) * GlobalCounters::kVertexOne, std::memory_order_relaxed));
}

uint32_t Hypergraph::num_vertices() const { return counters_.next_vertex_id(); }

uint32_t Hypergraph::num_edges() const { return counters_.next_edge_id(); }


const EdgeSignature& Hypergraph::edge_signature(EdgeId eid) const { return edge_signatures_[eid]; }

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
uint32_t Hypergraph::num_published_states() const {
    uint32_t n = published_states_outside_.load(std::memory_order_acquire);
    for (int i = 0; i < MAX_ARENA_WORKERS; ++i) n = std::max(n, published_[i].states);
    return n;
}

namespace {
void raise_to(std::atomic<uint32_t>& a, uint32_t v) {
    uint32_t cur = a.load(std::memory_order_relaxed);
    while (cur < v && !a.compare_exchange_weak(cur, v, std::memory_order_release,
                                               std::memory_order_relaxed)) {}
}
}  // namespace

void Hypergraph::note_published_state(StateId sid) {
    const int w = arena_worker_index();
    if (w < 0) { raise_to(published_states_outside_, sid + 1); return; }
    if (published_[w].states < sid + 1) published_[w].states = sid + 1;
}

void Hypergraph::note_published_event(EventId eid) {
    const int w = arena_worker_index();
    if (w < 0) { raise_to(published_events_outside_, eid + 1); return; }
    if (published_[w].events < eid + 1) published_[w].events = eid + 1;
}

// INVALID_ID until a genesis state is published, and no state id equals INVALID_ID, so the
// comparison alone answers both questions.
bool Hypergraph::is_genesis_state(StateId sid) const {
    return sid == genesis_state_.load(std::memory_order_acquire);
}

bool Hypergraph::is_genesis_event(EventId eid) const {
    const StateId genesis = genesis_state_.load(std::memory_order_acquire);
    if (genesis == INVALID_ID) return false;
    if (eid >= num_published_events()) return false;
    return events_[eid].input_state == genesis;
}

StateId Hypergraph::genesis_state() const {
    return genesis_state_.load(std::memory_order_acquire);
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

#if HG_ENGINE_STATS
uint64_t Hypergraph::invalid_matches() const {
    return invalid_matches_.load(std::memory_order_relaxed);
}
#endif

void Hypergraph::note_invalid_match() {
    HG_STAT(invalid_matches_.fetch_add(1, std::memory_order_relaxed));
}

#if HG_ENGINE_STATS
uint64_t Hypergraph::canonical_key_collisions() const {
    return canonical_key_collisions_.load(std::memory_order_relaxed);
}
uint64_t Hypergraph::canonical_hash_computations() const {
    return canonical_hash_computations_.load(std::memory_order_relaxed);
}
#endif

void Hypergraph::book_ir_call(const hgcommon::IrWork& work, bool retried) const {
    HG_STAT(ir_calls_.fetch_add(1, std::memory_order_relaxed));
    HG_STAT(ir_searched_.fetch_add(work.searched, std::memory_order_relaxed));
    HG_STAT(ir_leaves_.fetch_add(work.leaves, std::memory_order_relaxed));
    HG_STAT(ir_nodes_.fetch_add(work.nodes, std::memory_order_relaxed));
    HG_STAT(ir_depth_sum_.fetch_add(work.max_depth, std::memory_order_relaxed));
    HG_STAT(if (retried) ir_retries_.fetch_add(1, std::memory_order_relaxed));
#if !HG_ENGINE_STATS
    (void)work; (void)retried;
#endif
}

void Hypergraph::book_ir(const IrBooking& b) const {
    HG_STAT(ir_calls_.fetch_add(b.calls, std::memory_order_relaxed));
    HG_STAT(ir_searched_.fetch_add(b.searched, std::memory_order_relaxed));
    HG_STAT(ir_leaves_.fetch_add(b.leaves, std::memory_order_relaxed));
    HG_STAT(ir_nodes_.fetch_add(b.nodes, std::memory_order_relaxed));
    HG_STAT(ir_depth_sum_.fetch_add(b.depth_sum, std::memory_order_relaxed));
    HG_STAT(ir_retries_.fetch_add(b.retries, std::memory_order_relaxed));
    HG_STAT(if (b.fallback) ir_fallbacks_.fetch_add(1, std::memory_order_relaxed));
#if !HG_ENGINE_STATS
    (void)b;
#endif
}

#if HG_ENGINE_STATS
Hypergraph::IrWorkTotals Hypergraph::ir_work() const {
    return IrWorkTotals{ir_calls_.load(std::memory_order_relaxed),
                        ir_searched_.load(std::memory_order_relaxed),
                        ir_leaves_.load(std::memory_order_relaxed),
                        ir_nodes_.load(std::memory_order_relaxed),
                        ir_depth_sum_.load(std::memory_order_relaxed),
                        ir_retries_.load(std::memory_order_relaxed),
                        ir_fallbacks_.load(std::memory_order_relaxed)};
}
#endif

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
uint32_t Hypergraph::num_published_events() const {
    uint32_t n = published_events_outside_.load(std::memory_order_acquire);
    for (int i = 0; i < MAX_ARENA_WORKERS; ++i) n = std::max(n, published_[i].events);
    return n;
}

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

uint64_t Hypergraph::observable_num_events() const {
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

uint64_t Hypergraph::observable_num_branchial() const {
    return quotient_reconstruction() ? num_reconstructed_branchial()
                                     : causal_graph_.num_branchial_edges();
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
    record_raw_counts_only_.store(r.raw_counts_only, std::memory_order_relaxed);
    record_multiplicities_.store(r.multiplicities, std::memory_order_relaxed);
}

RecordSet Hypergraph::record_set() const {
    return RecordSet{record_causal_.load(std::memory_order_relaxed),
                     record_branchial_.load(std::memory_order_relaxed),
                     record_state_events_.load(std::memory_order_relaxed),
                     record_raw_events_.load(std::memory_order_relaxed),
                     record_raw_counts_only_.load(std::memory_order_relaxed),
                     record_multiplicities_.load(std::memory_order_relaxed)};
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
// QrCtx -- the storage face the shared replay core drives
// =============================================================================
// WHERE a producer vector, an applied list or a claim set lives is here; what an application
// DOES is in hgcommon, which is the body the device runs too. The core is instantiated in this
// translation unit and nowhere else, which is what lets these bodies live here.

bool Hypergraph::QrCtx::claim(const QcInstance& inst, const SlotMatch& m) {
    if (m.local < inst.claim_cap) {
        const uint64_t bit = uint64_t{1} << (m.local & 63u);
        if (inst.claim_bits[m.local >> 6].fetch_or(bit, std::memory_order_acq_rel) & bit)
            return false;
        ++qc_slot(hg.qc_ctr_).bit_claims;
        return true;
    }
    return hg.qc_applied_.insert(hgcommon::qr_apply_key(inst.id, m.id));
}

uint32_t Hypergraph::QrCtx::mint_event(uint32_t above) {
    HG_STAT(++qc_slot(hg.qc_ctr_).applications);
    return hg.alloc_event_id(above);
}

uint32_t Hypergraph::alloc_event_id(uint32_t above) {
    const int w = arena_worker_index();
    if (w < 0) {
        qc_events_made_outside_.fetch_add(1, std::memory_order_relaxed);
        return qc_next_raw_event_.fetch_add(1, std::memory_order_relaxed);
    }
    IdBlock& b = qc_event_blocks_[w];
    if (b.next == b.end || (above != hgcommon::QR_NO_PRODUCER && b.next <= above)) {
        b.next = qc_next_raw_event_.fetch_add(kEventIdBlock, std::memory_order_relaxed);
        b.end = b.next + kEventIdBlock;
    }
    ++b.made;
    return b.next++;
}

void Hypergraph::QrCtx::record_content(uint32_t ev, uint64_t from_class, uint64_t to_class,
                                       uint32_t rule) {
    hg.qc_event_sig_.emplace_at(Hypergraph::qc_ev_slot(ev), hg.arena_,
                                QcEventContent{from_class, to_class, rule, 1u});
}

hgcommon::EventSignatureKeys Hypergraph::QrCtx::keys() const {
    return hg.event_signature_keys();
}

uint32_t Hypergraph::QrCtx::frame_step(uint64_t class_hash, uint32_t fallback) const {
    return hg.qc_frame_step(class_hash, fallback);
}

// The step of the class's frame state, which is what an event signature records as the output
// step; `fallback` when the class has no frame yet.
uint32_t Hypergraph::qc_frame_step(uint64_t class_hash, uint32_t fallback) const {
    if (auto fo = qc_frame_.lookup(class_hash))
        return get_state(static_cast<StateId>(*fo - 1)).step;
    return fallback;
}

// The replay's event class, claimed on the signature's values like every event identity; the
// class's key is the signature recorded for the event.
void Hypergraph::QrCtx::record_runsig(uint32_t ev, const SlotMatch& m, uint64_t from_class,
                                      uint32_t out_step) {
    const CanonicalClaim claim = hg.claim_replay_event(m, from_class, out_step);
    hg.qc_event_runsig_.emplace_at(Hypergraph::qc_ev_slot(ev), hg.arena_, claim.key);
    if (claim.won) hg.qc_num_canon_events_.fetch_add(1, std::memory_order_relaxed);
}

// The replay's event class, claimed on the signature's values like every event identity; a key
// hit recomputes the class's values from its first application. The class's key is the
// signature recorded for the event. The signature is a function of (m, from_class, out_step)
// and a match has one from class, so the key is kept on the match for its output step and later
// applications read it there.
Hypergraph::CanonicalClaim Hypergraph::claim_replay_event(const SlotMatch& m, uint64_t from_class,
                                                          uint32_t out_step) {
    struct Cells {
        const SlotMatch& m;
        uint32_t step_load() const {
            return hgcommon::atomic_ref<uint32_t>(m.runsig_step).load(std::memory_order_relaxed);
        }
        bool step_cas(uint32_t expected, uint32_t desired) {
            return hgcommon::atomic_ref<uint32_t>(m.runsig_step)
                .compare_exchange_strong(expected, desired, std::memory_order_relaxed);
        }
        uint64_t key_load() const {
            return hgcommon::atomic_ref<uint64_t>(m.runsig_key).load(std::memory_order_acquire);
        }
        void key_store(uint64_t key) {
            hgcommon::atomic_ref<uint64_t>(m.runsig_key).store(key, std::memory_order_release);
        }
    } cells{m};
    if (uint64_t key = 0; hgcommon::qr_cached_key(cells, out_step, key)) return {0, key, false};
    const EventSignatureKeys keys = event_signature_keys_;
    hgcommon::QrRunSignature sig;
    hgcommon::qr_signature_values(keys, m, from_class, out_step, sig);
    auto same = [&](const QrEventRef* r) {
        // The values are a function of these three, and on a rule with few classes most hits are
        // the class's first match applied to another instance.
        if (r->m == &m && r->from_hash == sig.from_hash && r->out_step == sig.out_step)
            return true;
        hgcommon::QrRunSignature theirs;
        hgcommon::qr_signature_values(keys, *r->m, r->from_hash, r->out_step, theirs);
        return hgcommon::qr_same_values(theirs, sig);
    };
    QrEventRef* made = nullptr;
    auto make = [&]() -> const QrEventRef* {
        made = static_cast<QrEventRef*>(arena_.allocate_raw(sizeof(QrEventRef), alignof(QrEventRef)));
        *made = QrEventRef{&m, sig.from_hash, sig.out_step};
        return made;
    };
    auto rep_of = [](const QrEventRef*) { return uint32_t{0}; };
    auto on_collision = [&] {
        HG_STAT(canonical_key_collisions_.fetch_add(1, std::memory_order_relaxed));
    };
    auto c = keyed_claim<const QrEventRef*>(qc_canon_events_, sig.sig & event_key_mask_, same, make,
                                            rep_of, on_collision);
    if (!c.won && made) arena_.release_last(made, sizeof(QrEventRef));
    hgcommon::qr_cache_key(cells, out_step, c.key);
    return {c.rep, c.key, c.won};
}

// One flag each, read per application: record_set() loads all five.
bool Hypergraph::QrCtx::want_causal() const {
    return hg.record_causal_.load(std::memory_order_relaxed);
}
bool Hypergraph::QrCtx::want_branchial() const {
    return hg.record_branchial_.load(std::memory_order_relaxed);
}

uint32_t Hypergraph::QrCtx::producer_at(const QcInstance& inst, uint32_t slot) const {
    return inst.prod[slot];
}

void Hypergraph::QrCtx::record_causal(uint32_t producer, uint32_t consumer, bool distinct_pair) {
    hg.qc_record_causal(producer, consumer, distinct_pair);
}

uint32_t Hypergraph::QrCtx::redundant(const uint32_t* producers, uint32_t n) const {
    if (n < 2) return 0;
    const Hypergraph& g = hg;
    auto ctx = make_scratch_reach_ctx([&](uint32_t x, auto&& f) {
        if (const QcKept* k = g.qc_kept_->find(qc_ev_slot(x)))
            for (uint32_t i = 0; i < k->n; ++i) f(k->at(i));
    });
    // A producer's application minted its id before the descent that led here, so ids increase
    // along every edge of this relation.
    return hgcommon::redundant_producers(ctx, producers, n, /*topological=*/true);
}

void Hypergraph::QrCtx::record_kept(uint32_t ev, const uint32_t* kept, uint32_t nkept) {
    if (nkept == 0) return;
    QcKept k{nkept, {0, 0, 0}, nullptr};
    for (uint32_t i = 0; i < nkept && i < 3; ++i) k.inl[i] = kept[i];
    if (nkept > 3) {
        uint32_t* rest = hg.arena_.allocate_array<uint32_t>(nkept - 3);
        for (uint32_t i = 3; i < nkept; ++i) rest[i - 3] = kept[i];
        k.more = rest;
    }
    hg.qc_kept_->emplace_at(qc_ev_slot(ev), hg.arena_, k);
    qc_slot(hg.qc_ctr_).reduced_pairs += nkept;
}

bool Hypergraph::QrCtx::applied_ref_valid(AppliedRef r) { return r != nullptr; }

Hypergraph::QrCtx::AppliedRef Hypergraph::QrCtx::publish_applied(const QcInstance& inst,
                                                                 const SlotMatch& m,
                                                                 uint32_t ev) {
    auto& applied = hg.qc_inst_applied_.slot(qc_ev_slot(inst.id), hg.arena_);
    return applied.push(QcAppliedMatch{m.id, ev, m.num_consumed, m.consumed_slots}, hg.arena_);
}

void Hypergraph::QrCtx::record_branchial_pair(uint32_t lo, uint32_t hi) {
    (void)lo; (void)hi;
    ++branchial_seen;
}

Hypergraph::QrCtx::~QrCtx() {
    // Same contract as the causal-edge count above: num_reconstructed_branchial is a semantic
    // observable (read by the bench and four probes), so the flush runs in every build.
    if (branchial_seen)
        qc_slot(hg.qc_ctr_).branchial += branchial_seen;
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
// One of the two maps is empty unless the mode was changed between runs on this Hypergraph.
size_t Hypergraph::num_canonical_states() const {
    return canonical_state_map_.count_unique() + canonical_form_map_.count_unique();
}

// Release/acquire: the mode is set on the main thread and read by workers, which matters on a
// weak model like ARM64.
void Hypergraph::set_state_canonicalization_mode(StateCanonicalizationMode mode) {
    state_canonicalization_mode_.store(mode, std::memory_order_release);
}

StateCanonicalizationMode Hypergraph::state_canonicalization_mode() const {
    return state_canonicalization_mode_.load(std::memory_order_acquire);
}

bool Hypergraph::is_full_canonicalization() const {
    return state_canonicalization_mode_.load(std::memory_order_acquire) ==
           StateCanonicalizationMode::Full;
}

uint64_t Hypergraph::num_reconstructed_branchial() const {
    if (!quotient_replay()) return qm_branchial_.load(std::memory_order_relaxed);
    return qc_ctr_total(&QcCounterSlot::branchial);
}

// =============================================================================
// Raw counts from class multiplicities: the storage behind quotient_multiplicity_core.hpp.

uint64_t Hypergraph::qm_point_key(uint64_t class_hash, uint32_t depth) {
    return hgcommon::avoid_reserved_keys(hgcommon::qc_key(class_hash, depth, 0));
}

uint64_t Hypergraph::qm_consumed_key(uint32_t match_id, uint32_t depth) {
    return hgcommon::qr_apply_key(match_id, depth);
}

Hypergraph::QmPoint* Hypergraph::qm_point(uint64_t class_hash, uint32_t depth) {
    const uint64_t key = qm_point_key(class_hash, depth);
    if (auto r = qm_points_.lookup(key)) return *r;
    QmPoint* p = arena_.template create<QmPoint>();
    p->depth = depth;
    p->class_hash = class_hash;
    const auto ins = qm_points_.insert_if_absent(key, p);
    if (ins.second && static_cast<int>(depth) >= qc_max_steps_.load(std::memory_order_relaxed))
        qc_blocked_.push(QcPoint{class_hash, depth}, arena_);
    return ins.first;
}

std::atomic<uint64_t>* Hypergraph::qm_consumed_cell(uint32_t match_id, uint32_t depth) {
    const uint64_t key = qm_consumed_key(match_id, depth);
    if (auto r = qm_consumed_.lookup(key)) return *r;
    auto* cell = arena_.template create<std::atomic<uint64_t>>(0);
    return qm_consumed_.insert_if_absent(key, cell).first;
}

void Hypergraph::qm_add(std::atomic<uint64_t>& counter, uint64_t delta) {
    uint64_t old = counter.load(std::memory_order_relaxed);
    uint64_t next;
    do {
        next = hgcommon::qm_sat_add(old, delta);
    } while (!counter.compare_exchange_weak(old, next, std::memory_order_acq_rel,
                                            std::memory_order_relaxed));
    if (next == hgcommon::QM_SATURATED && old + delta != next)
        qm_saturated_.store(true, std::memory_order_relaxed);
}

// The cascade's queued points, a min-heap on depth (hgcommon::qm_heap_push / qm_heap_pop).
struct Hypergraph::QmQueue {
    struct Task { uint64_t class_hash; uint32_t depth; };
    SVec<Task> heap;
};

template <class F>
void Hypergraph::qm_cascade(F&& start) {
    auto mk = worker_scratch().mark();
    {
        QmQueue q;
        QmCtx c{*this, q};
        start(c);
        hgcommon::qm_drain(c);
    }
    worker_scratch().release(mk);
}

uint32_t Hypergraph::QmCtx::max_steps() const {
    return static_cast<uint32_t>(hg.qc_max_steps_.load(std::memory_order_relaxed));
}

bool Hypergraph::QmCtx::ready(const SlotMatch& m, uint64_t& b) const {
    auto r = hg.qm_overlaps_.lookup(static_cast<uint64_t>(m.id) + 1);
    if (!r.has_value()) return false;
    b = *r - 1;
    return true;
}

uint64_t Hypergraph::QmCtx::mass(uint64_t class_hash, uint32_t depth) const {
    auto r = hg.qm_points_.lookup(qm_point_key(class_hash, depth));
    return r.has_value() ? (*r)->mass.load(std::memory_order_acquire) : 0;
}

void Hypergraph::QmCtx::add_mass(uint64_t class_hash, uint32_t depth, uint64_t delta) {
    hg.qm_add(hg.qm_point(class_hash, depth)->mass, delta);
}

uint64_t Hypergraph::QmCtx::consumed(const SlotMatch& m, uint32_t depth) {
    auto r = hg.qm_consumed_.lookup(qm_consumed_key(m.id, depth));
    return r.has_value() ? (*r)->load(std::memory_order_acquire) : 0;
}

bool Hypergraph::QmCtx::advance(const SlotMatch& m, uint32_t depth, uint64_t& expected,
                                uint64_t desired) {
    return hg.qm_consumed_cell(m.id, depth)->compare_exchange_strong(
        expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
}

void Hypergraph::QmCtx::count(uint64_t events, uint64_t branchial) {
    hg.qm_add(hg.qm_events_, events);
    if (branchial) hg.qm_add(hg.qm_branchial_, branchial);
}

hgcommon::EventSignatureKeys Hypergraph::QmCtx::keys() const { return hg.event_signature_keys(); }

uint32_t Hypergraph::QmCtx::frame_step(uint64_t class_hash, uint32_t fallback) const {
    return hg.qc_frame_step(class_hash, fallback);
}

void Hypergraph::QmCtx::note_signature(const SlotMatch& m, uint64_t from_class,
                                        uint32_t out_step) {
    if (hg.claim_replay_event(m, from_class, out_step).won)
        hg.qc_num_canon_events_.fetch_add(1, std::memory_order_relaxed);
}

bool Hypergraph::QmCtx::claim_queued(uint64_t class_hash, uint32_t depth) {
    return hg.qm_point(class_hash, depth)->queued.exchange(1, std::memory_order_acq_rel) == 0;
}

void Hypergraph::QmCtx::push(uint64_t class_hash, uint32_t depth) {
    queue.heap.push_back({class_hash, depth});
    hgcommon::qm_heap_push(queue.heap.data(), static_cast<uint32_t>(queue.heap.size()));
}

bool Hypergraph::QmCtx::pop(uint64_t& class_hash, uint32_t& depth) {
    if (queue.heap.empty()) return false;
    const QmQueue::Task t =
        hgcommon::qm_heap_pop(queue.heap.data(), static_cast<uint32_t>(queue.heap.size()));
    queue.heap.pop_back();
    class_hash = t.class_hash;
    depth = t.depth;
    hg.qm_point(class_hash, depth)->queued.store(0, std::memory_order_release);
    return true;
}

// Partners: qm_credit's add-then-claim against qm_drain's clear-then-read, and the
// ready-then-read in qc_capture_expansion against both.
void Hypergraph::QmCtx::fence() { hgcommon::rendezvous_barrier<hgcommon::rv::QuotientMassMatch>(); }

#if HG_ENGINE_STATS
size_t Hypergraph::num_frame_alignment_disagreements() const {
    return qc_frame_disagree_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_alignment_failures() const {
    return qc_align_fail_.load(std::memory_order_relaxed);
}

size_t Hypergraph::num_bad_correspondences() const {
    return qc_align_badcorr_.load(std::memory_order_relaxed);
}
#endif

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
        if (e < qc_next_raw_event_.load(std::memory_order_relaxed))
            if (const uint64_t* r = qc_event_runsig_.find(qc_ev_slot(e))) return *r;
    }
    return reconstructed_raw_triple(e);
}

// The event's content itself, for a caller that must DESCRIBE the event rather than identify it.
const QcEventContent* Hypergraph::reconstructed_event_content(uint32_t e) const {
    if (e >= qc_next_raw_event_.load(std::memory_order_relaxed)) return nullptr;
    const QcEventContent* c = qc_event_sig_.find(qc_ev_slot(e));
    return c && c->written ? c : nullptr;
}

uint32_t Hypergraph::count_state_edges(StateId sid) const {
    uint32_t count = 0;
    states_[sid].edges.for_each([&](EdgeId) { count++; });
    return count;
}

// Route every map's table storage through the arena (no malloc, no per-map heap contention).
// The initialiser order follows member declaration order; arena_ is declared before these maps,
// so it is fully constructed by the time they take its address.
namespace {
// Segment SHIFT for a given capacity scale. The arrays take log2 of the segment size rather than
// the size, so a scale that is not a power of two is rounded up here and a segment size that is
// not one cannot be constructed at all.
uint32_t seg_shift_for(uint32_t scale) {
    uint32_t extra = 0;
    for (uint32_t s = 1; s < (scale ? scale : 1u); s <<= 1) ++extra;
    return SegmentedArray<Edge>::DEFAULT_SEGMENT_SHIFT + extra;
}
}  // namespace

Hypergraph::Hypergraph(uint32_t capacity_scale)
    : edges_(seg_shift_for(capacity_scale))
    , edge_signatures_(seg_shift_for(capacity_scale))
    , states_(seg_shift_for(capacity_scale))
    , events_(seg_shift_for(capacity_scale))
    , canonical_state_map_(decltype(canonical_state_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
    , canonical_form_map_(decltype(canonical_form_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
    , event_canonical_state_map_(
          decltype(event_canonical_state_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)
    , qc_inst_applied_(seg_shift_for(capacity_scale))
    , qc_canon_events_(decltype(qc_canon_events_)::DEFAULT_INITIAL_CAPACITY, &arena_)
    , qc_event_sig_(seg_shift_for(capacity_scale))
    , qc_kept_(std::make_unique<SegmentedArray<QcKept>>(seg_shift_for(capacity_scale)))
    , qc_event_runsig_(seg_shift_for(capacity_scale))
    , canonical_event_map_(decltype(canonical_event_map_)::DEFAULT_INITIAL_CAPACITY, &arena_)

{
    // Edges and their signatures are read by id only; nothing enumerates them or asks their extent.
    edges_.set_uncounted();
    qc_inst_applied_.set_uncounted();
    qc_event_sig_.set_uncounted();
    qc_kept_->set_uncounted();
    qc_event_runsig_.set_uncounted();
    edge_signatures_.set_uncounted();
    // Their extent is num_published_states/events, from the per-worker marks.
    states_.set_uncounted();
    events_.set_uncounted();
    causal_graph_.set_arena(&arena_);
    // The dedup sets are seated in the arena like every other member: a table on fresh arena
    // bytes needs no sentinel fill, and every table is reclaimed with the arena.
    qc_applied_.set_arena(&arena_);
}

// An ordered pair of event ids as one map key. Both ids are offset by one before packing, which
// makes the key injective and never zero -- ConcurrentMap reserves 0 as EMPTY.
uint64_t Hypergraph::qc_pair_key(uint32_t a, uint32_t b) { return id_key(a, b); }

// The DP's key spaces come from hgcommon so the device indexes the same ones.
uint64_t Hypergraph::qc_key(uint64_t state_hash, uint32_t depth, uint32_t orbit) {
    return hgcommon::qc_key(state_hash, depth, orbit);
}


uint32_t Hypergraph::QcAppliedMatch::consumed(uint32_t j) const {
    return consumed_slots[j];
}

}  // namespace engine
}  // namespace HG_NAMESPACE