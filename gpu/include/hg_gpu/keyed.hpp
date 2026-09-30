#pragma once
#include "hgcommon/namespace.hpp"
// KEYED REWRITES ON THE DEVICE. The rule is the host's (hgcommon/token_core.hpp,
// Hypergraph::intern_rewrite, claim_twin, take_twin): an edge's token is its id plus one for an
// initial edge and (rewrite id, RHS index) for a produced one; a state whose token set an earlier
// state holds takes that state's canonical results without IR. What differs is storage:
//   - a token is read from the edge's creator event, which holds its produced edges inline and
//     its rewrite id (DeviceEvent::rewrite_id); there is no per-edge token array;
//   - the rewrite map holds the first event that applied a rewrite, and a key hit is decided by
//     that event's rule and consumed tokens; its id plus one is the rewrite id;
//   - a match is inherited when all its edges predate its state (state_first_new_edge), because
//     the device matches every state in full.
// All of it runs on one thread: the rewrite's (apply_one_match) and the twin check's (lane 0 of
// the persistent block, before the warp's IR).

#include <cstdint>

#include "hgcommon/core.hpp"
#include "hgcommon/token_core.hpp"
#include "hg_gpu/atomic_pool.hpp"
#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/exploration.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/ring_buffer.hpp"
#include "hg_gpu/types.hpp"

namespace HG_NAMESPACE {
namespace gpu {

// Values of KeyedView::words[0].
enum : uint32_t { KEYED_OFF = 0, KEYED_ARMED = 1, KEYED_INTERNING = 2 };

// A state made by a rewrite that produced no edge: every edge it holds predates it.
constexpr uint32_t kKeyedNoNewEdge = 0xFFFFFFFEu;

__device__ inline uint32_t keyed_load(const uint32_t* p) {
    return *reinterpret_cast<const volatile uint32_t*>(p);
}

__device__ inline uint32_t keyed_rid_of(const DeviceState& ds, EventId ev) {
    return keyed_load(&ds.event_pool.data[ev].rewrite_id);
}

// The token of edge `e` when its creator event's rewrite id is set: 0 otherwise, and when the
// id space is exhausted.
__device__ inline uint64_t keyed_token_resolved(const DeviceState& ds, EdgeId e) {
    const EventId ev = ds.edge_pool.data[e].creator_event;
    if (ev == INVALID_ID) return hgcommon::token_initial(e);
    const uint32_t rid = keyed_rid_of(ds, ev);
    if (rid == hgcommon::REWRITE_ID_UNSET || rid == hgcommon::REWRITE_ID_NONE) return 0;
    // A rewrite's produced edges have consecutive ids (apply_one_match).
    return hgcommon::token_produced(rid, e - ds.event_pool.data[ev].first_produced);
}

// Interns event `ev`'s rewrite, given its consumed tokens in match order: the rewrite id, or
// REWRITE_ID_NONE. `repeated` is set when an earlier event holds the key. The event is written
// and fenced before this is called, so an event found in the map can be read.
__device__ inline uint32_t keyed_intern_event(const DeviceState& ds, EventId ev,
                                              const uint64_t* tokens, bool& repeated) {
    repeated = false;
    const DeviceEvent& x = ds.event_pool.data[ev];
    uint32_t words[1 + 2 * kMaxPatternEdges];
    uint64_t h = 0;
    if (hgcommon::rewrite_key(static_cast<uint16_t>(x.rule), tokens, x.num_consumed, words, h) ==
        0)
        return hgcommon::REWRITE_ID_NONE;
    struct P {
        const DeviceState& ds;
        const DeviceEvent& x;
        const uint64_t* tokens;
        EventId ev;
        __device__ bool same(uint32_t holder) const {
            const DeviceEvent& y = ds.event_pool.data[holder];
            if (y.rule != x.rule || y.num_consumed != x.num_consumed) return false;
            for (uint8_t i = 0; i < y.num_consumed; ++i)
                if (keyed_token_resolved(ds, event_consumed_edge(ds, y, i)) != tokens[i]) return false;
            return true;
        }
        __device__ bool make(uint32_t& v) { v = ev; return true; }
        __device__ uint32_t rep_of(uint32_t v) const { return v; }
    } p{ds, x, tokens, ev};
    const StateClaim c = keyed_claim_device(ds, ev, h, ds.keyed.rewrites, p);
    // Ids at or past 2^31 would overlap REWRITE_TWIN_CANDIDATE and REWRITE_ID_NONE.
    if (c.canonical >= 0x7FFFFFFFu) return hgcommon::REWRITE_ID_NONE;
    repeated = !c.fresh;
    return c.canonical + 1u;
}

// The rewrite id of event `ev`, interning it and the events it depends on when unset. The
// events whose ids are missing are resolved oldest first: an event's consumed edges were
// produced by older events, so the walk stops at initial edges or at events with ids. A walk
// deeper than its stack leaves the id unset, and the edges it would name have token 0.
__device__ inline uint32_t keyed_event_rid(const DeviceState& ds, EventId ev) {
    uint32_t rid = keyed_rid_of(ds, ev);
    if (rid != hgcommon::REWRITE_ID_UNSET) return rid;
    constexpr uint32_t kDepth = 32;
    EventId pending[kDepth];
    uint32_t top = 0;
    pending[top++] = ev;
    while (top > 0) {
        const EventId cur = pending[top - 1];
        const DeviceEvent& x = ds.event_pool.data[cur];
        EventId missing = INVALID_ID;
        for (uint8_t i = 0; i < x.num_consumed && missing == INVALID_ID; ++i) {
            const EventId c = ds.edge_pool.data[event_consumed_edge(ds, x, i)].creator_event;
            if (c != INVALID_ID && keyed_rid_of(ds, c) == hgcommon::REWRITE_ID_UNSET) missing = c;
        }
        if (missing != INVALID_ID) {
            if (top == kDepth) return hgcommon::REWRITE_ID_UNSET;
            pending[top++] = missing;
            continue;
        }
        uint64_t tokens[kMaxPatternEdges];
        for (uint8_t i = 0; i < x.num_consumed; ++i)
            tokens[i] = keyed_token_resolved(ds, event_consumed_edge(ds, x, i));
        bool repeated = false;
        const uint32_t r = keyed_intern_event(ds, cur, tokens, repeated);
        atomicExch(&ds.event_pool.data[cur].rewrite_id, r);
        --top;
    }
    return keyed_rid_of(ds, ev);
}

// The token of edge `e`, resolving its creator event's rewrite id first.
__device__ inline uint64_t keyed_edge_token(const DeviceState& ds, EdgeId e) {
    const EventId ev = ds.edge_pool.data[e].creator_event;
    if (ev != INVALID_ID) keyed_event_rid(ds, ev);
    return keyed_token_resolved(ds, e);
}

// The token sum of state `s`, computed over its edges on first use; 0 when a token is 0.
__device__ inline uint64_t keyed_state_sum(const DeviceState& ds, StateId s) {
    volatile uint64_t* slot = ds.keyed.state_token_sum + s;
    uint64_t sum = *slot;
    if (sum != 0) return sum;
    const StateEdgeSlice sl = ds.state_edge_slices[s];
    for (uint32_t i = 0; i < sl.count; ++i) {
        const uint64_t t = keyed_edge_token(ds, ds.state_edge_ids[sl.offset + i]);
        if (t == 0) return 0;
        sum += hgcommon::token_term(t);
    }
    sum = hgcommon::token_sum_nonzero(sum);
    *slot = sum;
    return sum;
}

// After a rewrite has written event `ev` and state `sid` (produced edges first_eid ..
// first_eid + num_produced - 1): records the state's first produced edge, and from the run's
// first inherited match on interns the rewrite. Returns the rewrite id with
// REWRITE_TWIN_CANDIDATE when the rewrite may have been applied before, or 0.
__device__ inline uint32_t keyed_after_rewrite(const DeviceState& ds, const MatchRecord& m,
                                               EventId ev, StateId sid, uint32_t first_eid,
                                               uint32_t num_produced) {
    const KeyedView& k = ds.keyed;
    k.state_token_sum[sid] = 0;
    k.state_first_new_edge[sid] = num_produced ? first_eid : kKeyedNoNewEdge;
    const uint32_t st = keyed_load(k.words);
    if (st == KEYED_OFF) return 0;
    // Every edge of the match predates its state, so the parent holds the match too and
    // applies it: this application repeats a rewrite (Hypergraph::match_predates_state).
    const uint32_t first = k.state_first_new_edge[m.state_id];
    bool inherited = first != INVALID_ID;
    for (uint8_t i = 0; i < m.num_edges && inherited; ++i)
        if (m.matched_edges[i] >= first) inherited = false;
    if (st == KEYED_ARMED) {
        if (!inherited) return 0;
        atomicCAS(k.words, KEYED_ARMED, KEYED_INTERNING);
    }
    uint64_t tokens[kMaxPatternEdges];
    for (uint8_t i = 0; i < m.num_edges; ++i) tokens[i] = keyed_edge_token(ds, m.matched_edges[i]);
    bool repeated = false;
    const uint32_t rid = keyed_intern_event(ds, ev, tokens, repeated);
    atomicExch(&ds.event_pool.data[ev].rewrite_id, rid);
    if (rid == hgcommon::REWRITE_ID_NONE) return 0;
    return rid | ((inherited || repeated) ? hgcommon::REWRITE_TWIN_CANDIDATE : 0u);
}

// A twin's follower stack: FOLLOW_EMPTY until a child waits on it, FOLLOW_CLOSED once its
// results are published and its followers handed on (keyed_close_followers).
constexpr uint32_t FOLLOW_EMPTY  = 0xFFFFFFFFu;
constexpr uint32_t FOLLOW_CLOSED = 0xFFFFFFFEu;

enum class TwinResult : uint32_t {
    kNone = 0,        // no twin, or a twin that will never publish: the child runs its IR
    kTaken = 1,       // the twin's results are the child's
    kFollowing = 2,   // the child waits on the twin's stack and is completed from `ready`
};

// Pushes event `ev` (whose child waits) onto twin `t`'s follower stack; false when the stack is
// closed, and the twin's hash, if it has one, is then published.
__device__ inline bool keyed_follow(const DeviceState& ds, StateId t, EventId ev) {
    uint32_t head = *reinterpret_cast<const volatile uint32_t*>(ds.keyed.follow_head + t);
    for (;;) {
        if (head == FOLLOW_CLOSED) {
            __threadfence();
            return false;
        }
        ds.keyed.follow_next[ev] = head;
        __threadfence();   // the link before the push that makes it reachable
        const uint32_t prev = atomicCAS(ds.keyed.follow_head + t, head, ev);
        if (prev == head) return true;
        head = prev;
    }
}

// The twin check for state `child`, made by event `ev` from `parent` with `keyed` from
// keyed_after_rewrite, run by one thread before the state's IR. Stores the state's token sum.
// When a candidate finds an earlier state with the same token set whose canonical hash is
// published, the child takes its hash (`h`), its class (`rep`) and, where the run keeps
// them, its ranks and orbits carried across by token, and the IR is skipped (kTaken). When
// that state has not published yet and `may_follow`, the child waits on its follower stack
// (kFollowing). Scratch for the token index comes from `slot` (grown from `arena`).
__device__ inline TwinResult keyed_take_twin(const DeviceState& ds, StateId child, StateId parent,
                                       EventId ev, uint32_t keyed, DeviceArena::View arena,
                                       uint32_t*& slot, uint64_t& slot_words, bool want_ranks,
                                       bool want_orbits, DedupMap::DeviceView states,
                                       typename Pool<uint32_t>::DeviceView forms, uint64_t& h,
                                       StateId& rep, bool may_follow) {
    const KeyedView& k = ds.keyed;
    const uint32_t rid = keyed & ~hgcommon::REWRITE_TWIN_CANDIDATE;
    const DeviceEvent& x = ds.event_pool.data[ev];
    uint64_t tokens[kMaxPatternEdges];
    for (uint8_t i = 0; i < x.num_consumed; ++i)
        tokens[i] = keyed_edge_token(ds, event_consumed_edge(ds, x, i));
    const uint64_t sum = hgcommon::child_token_sum(keyed_state_sum(ds, parent), tokens,
                                                   x.num_consumed, rid, x.num_produced);
    *reinterpret_cast<volatile uint64_t*>(k.state_token_sum + child) = sum;
    if (sum == 0 || !(keyed & hgcommon::REWRITE_TWIN_CANDIDATE)) return TwinResult::kNone;

    // The token index over a candidate twin's edges and, for the i-th edge of the child, the
    // position of the twin's edge with the same token (at[i]).
    const StateEdgeSlice sl = ds.state_edge_slices[child];
    const uint32_t n = sl.count;
    const uint32_t cap = hgcommon::token_index_capacity(n);
    const uint64_t need = uint64_t(cap) * 3u + n;
    if (slot_words < need) {
        uint32_t* p = arena.claim(need);
        if (!p) return TwinResult::kNone;
        slot = p;
        slot_words = need;
    }
    uint64_t* keys = reinterpret_cast<uint64_t*>(slot);
    uint32_t* pos = slot + 2u * cap;
    uint32_t* at = pos + cap;
    struct P {
        const DeviceState& ds;
        StateEdgeSlice sl;
        uint64_t* keys;
        uint32_t* pos;
        uint32_t* at;
        uint32_t cap;
        StateId child;
        __device__ bool same(uint32_t t) const {
            const StateEdgeSlice tl = ds.state_edge_slices[t];
            if (tl.count != sl.count) return false;
            hgcommon::TokenIndex idx = hgcommon::token_index_open(keys, pos, cap);
            for (uint32_t j = 0; j < tl.count; ++j)
                hgcommon::token_index_add(idx, keyed_edge_token(ds, ds.state_edge_ids[tl.offset + j]));
            if (!idx.valid) return false;
            for (uint32_t i = 0; i < sl.count; ++i) {
                at[i] = hgcommon::token_index_find(
                    idx, keyed_edge_token(ds, ds.state_edge_ids[sl.offset + i]));
                if (at[i] == 0xFFFFFFFFu) return false;
            }
            return true;
        }
        __device__ bool make(uint32_t& v) { v = child; return true; }
        __device__ uint32_t rep_of(uint32_t v) const { return v; }
    } p{ds, sl, keys, pos, at, cap, child};
    __threadfence();   // the child's slice and sum before it can be found as a twin
    const StateClaim c = keyed_claim_device(ds, child, sum & k.sum_mask, k.twins, p);
    const StateId t = c.canonical;
    TwinResult result = TwinResult::kNone;
    if (t != child && t != INVALID_ID) {
        uint64_t th = *reinterpret_cast<const volatile uint64_t*>(ds.state_canonical_hash + t);
        // Not published yet: wait on the twin's stack, or, when the twin has closed it since,
        // read the hash it published before closing.
        if (th == 0) {
            if (may_follow && keyed_follow(ds, t, ev)) result = TwinResult::kFollowing;
            else th = *reinterpret_cast<const volatile uint64_t*>(ds.state_canonical_hash + t);
        }
        if (th != 0) {
            __threadfence();   // the twin's tables were written before its hash was published
            const auto r = states.lookup(th);
            if (r.found) {
                const StateEdgeSlice tl = ds.state_edge_slices[t];
                if (want_ranks && ds.state_edge_rank)
                    for (uint32_t i = 0; i < n; ++i)
                        ds.state_edge_rank[sl.offset + i] = ds.state_edge_rank[tl.offset + at[i]];
                if (want_orbits && ds.state_edge_orbit) {
                    for (uint32_t i = 0; i < n; ++i)
                        ds.state_edge_orbit[sl.offset + i] = ds.state_edge_orbit[tl.offset + at[i]];
                    ds.state_num_orbits[child] = ds.state_num_orbits[t];
                }
                __threadfence();
                rep = reinterpret_cast<const hgcommon::CanonicalFormRecord*>(forms.data + r.value)->id;
                h = th;
                result = TwinResult::kTaken;
            }
        }
    }
    if (result == TwinResult::kTaken) atomicAdd(k.words + 3, 1u);
    struct Words {
        const KeyedView& k;
        __device__ bool seen() const { return keyed_load(k.words + 2) != 0; }
        __device__ void set_seen() { atomicExch(k.words + 2, 1u); }
        __device__ uint32_t add_claim() { return atomicAdd(k.words + 1, 1u) + 1u; }
        __device__ void switch_off() { atomicExch(k.words, static_cast<uint32_t>(KEYED_OFF)); }
    } w{k};
    hgcommon::keyed_note_claim(w, t != child && t != INVALID_ID, k.claim_limit);
    return result;
}

// Closes `sid`'s follower stack once its canonical results are published (or will never be:
// its key failed) and hands every child waiting on it to `ready`, where a block completes it.
// One thread; the caller has published the state's hash, and the fence orders that before the
// close a follower reads.
__device__ inline void keyed_close_followers(const DeviceState& ds, StateId sid,
                                             typename RingBuffer<uint32_t>::DeviceView ready) {
    __threadfence();
    uint32_t e = atomicExch(ds.keyed.follow_head + sid, FOLLOW_CLOSED);
    __threadfence();   // each follower linked its next before the exchange that pushed it
    while (e != FOLLOW_EMPTY && e != FOLLOW_CLOSED) {
        const uint32_t next = *reinterpret_cast<const volatile uint32_t*>(ds.keyed.follow_next + e);
        // The ring holds max_states entries and a child waits on one twin, so it has room.
        if (!ready.try_push(e)) ds.errors.record(ErrorKind::kScratchOverflow);
        e = next;
    }
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
