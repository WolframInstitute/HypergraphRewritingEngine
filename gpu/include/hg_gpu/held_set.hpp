#pragma once
#include "hgcommon/namespace.hpp"
// THE DEVICE'S HELD MATCH SETS, the host's HeldMatches: under hgcommon::held_matches_apply a
// state's matches are held in a set that the step's selection hands to the child each kept
// transition makes, and the child updates it by the rewrite's consumed and produced edges. A step
// then costs the matches that touch its event.
//
// A set holds its matches in a slab (rule and edge count, the lineage key, the matched edges in
// pattern order), ordered by key in an order-statistic set (hgcommon/key_order_core.hpp), and the
// state's edges at each vertex (the adjacency), over which the update's seeded joins enumerate
// candidates (hgcommon::join_seed). The matches on a consumed edge are found by a join anchored
// at it in the parent state and dropped; the matches on a produced edge by a join anchored at it
// in the child, each from its first produced edge.
//
// Storage comes from a pool of words in power-of-two size classes (HeldPoolView): a freed array
// goes on its class's stack and the next claim of that class takes it, so the live memory follows
// the live sets. One thread updates a set at a time: a set belongs to one state of one step.

#include <cstdint>

#include <cuda/atomic>

#include "hgcommon/join_core.hpp"
#include "hgcommon/key_order_core.hpp"
#include "hgcommon/sampling_core.hpp"
#include "hgcommon/step_select_core.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/match.hpp"

namespace HG_NAMESPACE {
namespace gpu {

// The record word that names the set a selected record hands on (DeviceState::held_handoff).
constexpr uint32_t kHeldRecordWord = kMaxPatternEdges - 1u;
constexpr uint32_t kHeldNone = 0xFFFFFFFFu;
// Entries per bucket of a set's order (the host's buckets are vectors).
constexpr uint32_t kHeldBucketCap = 48;
// Size classes of the pool: 2^(kHeldMinClass + c) words for c < kHeldClasses.
constexpr uint32_t kHeldMinClass = 6;
constexpr uint32_t kHeldClasses = 32;

struct HeldPoolView {
    uint32_t*           base = nullptr;
    unsigned long long* cursor = nullptr;   // words handed out from base
    uint64_t            capacity = 0;       // words
    unsigned long long* free_heads = nullptr;   // [kHeldClasses]: tag << 32 | (block index + 1)
};

__device__ inline uint32_t held_class_of(uint64_t words) {
    uint32_t c = 0;
    while ((uint64_t{1} << (kHeldMinClass + c)) < words) ++c;
    return c;
}

// A block of at least `words` words, or null when the pool is exhausted. The block's index in
// units of 2^kHeldMinClass words fits 32 bits.
__device__ inline uint32_t* held_alloc(const HeldPoolView& p, uint64_t words) {
    const uint32_t c = held_class_of(words);
    if (c >= kHeldClasses) return nullptr;
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> head(p.free_heads[c]);
    unsigned long long h = head.load(cuda::memory_order_acquire);
    while (static_cast<uint32_t>(h) != 0u) {
        const uint32_t idx = static_cast<uint32_t>(h) - 1u;
        uint32_t* block = p.base + (uint64_t{idx} << kHeldMinClass);
        const uint32_t next = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(block[0])
                                  .load(cuda::memory_order_relaxed);
        const unsigned long long want = ((h >> 32) + 1ull) << 32 | next;
        if (head.compare_exchange_weak(h, want, cuda::memory_order_acq_rel,
                                       cuda::memory_order_acquire))
            return block;
    }
    const unsigned long long size = 1ull << (kHeldMinClass + c);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> cur(*p.cursor);
    unsigned long long off = cur.load(cuda::memory_order_relaxed);
    for (;;) {
        if (off > p.capacity || size > p.capacity - off) return nullptr;
        if (cur.compare_exchange_weak(off, off + size, cuda::memory_order_relaxed)) break;
    }
    return p.base + off;
}

__device__ inline void held_free(const HeldPoolView& p, uint32_t* block, uint64_t words) {
    if (!block) return;
    const uint32_t c = held_class_of(words);
    const uint32_t idx = static_cast<uint32_t>(static_cast<uint64_t>(block - p.base) >> kHeldMinClass);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> head(p.free_heads[c]);
    unsigned long long h = head.load(cuda::memory_order_relaxed);
    for (;;) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(block[0])
            .store(static_cast<uint32_t>(h), cuda::memory_order_relaxed);
        const unsigned long long want = ((h >> 32) + 1ull) << 32 | (uint64_t{idx} + 1u);
        if (head.compare_exchange_weak(h, want, cuda::memory_order_release,
                                       cuda::memory_order_relaxed))
            return;
    }
}

using HeldEntry = hgcommon::KeyOrderEntry<uint32_t>;

// The order of a set, a Store of hgcommon/key_order_core.hpp with kHeldBucketCap entries per
// bucket.
struct DeviceKeyOrder {
    uint32_t   bits = 0;
    uint64_t*  tree = nullptr;     // [2^bits + 1]
    uint64_t   size = 0;
    uint32_t*  counts = nullptr;   // [2^bits]
    HeldEntry* runs = nullptr;     // [2^bits * kHeldBucketCap]
    __host__ __device__ HeldEntry* run(uint32_t b) { return runs + uint64_t{b} * kHeldBucketCap; }
    __host__ __device__ const HeldEntry* run(uint32_t b) const { return runs + uint64_t{b} * kHeldBucketCap; }
    __host__ __device__ uint32_t count(uint32_t b) const { return counts[b]; }
    __host__ __device__ void set_count(uint32_t b, uint32_t n) { counts[b] = n; }
    __host__ __device__ bool reserve(uint32_t, uint32_t n) const { return n <= kHeldBucketCap; }
    // While set, the tree is left alone and rebuilt from the counts afterwards
    // (held_order_rebuild_tree): a block editing many buckets at once would otherwise add to the
    // tree's top nodes from every lane.
    uint32_t   defer_tree = 0;
    __device__ void tree_add(uint32_t i, int64_t d) {
        if (defer_tree) return;
        atomicAdd(reinterpret_cast<unsigned long long*>(&tree[i]),
                  static_cast<unsigned long long>(d));
    }
    __device__ void size_add(int64_t d) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&size), static_cast<unsigned long long>(d));
    }
};

// The tree of `o` from its counts, by the whole block: node i (1-based) is the sum of the counts
// of buckets i - lowbit(i) .. i - 1, a lane per node.
__device__ inline void held_order_rebuild_tree(DeviceKeyOrder& o) {
    const uint32_t nb = 1u << o.bits;
    for (uint32_t i = threadIdx.x + 1u; i <= nb; i += blockDim.x) {
        uint64_t sum = 0;
        for (uint32_t b = i - (i & (0u - i)); b < i; ++b) sum += o.counts[b];
        o.tree[i] = sum;
    }
    __syncthreads();
}

__device__ inline uint64_t held_order_words(uint32_t bits) {
    const uint64_t nb = uint64_t{1} << bits;
    return 2u * (nb + 1u) + nb + nb * kHeldBucketCap * (sizeof(HeldEntry) / 4u);
}

// The order's arrays in one block of held_order_words(bits) words.
__device__ inline void held_order_point(DeviceKeyOrder& o, uint32_t* block, uint32_t bits) {
    const uint64_t nb = uint64_t{1} << bits;
    o.bits = bits;
    o.tree = reinterpret_cast<uint64_t*>(block);
    o.runs = reinterpret_cast<HeldEntry*>(block + 2u * (nb + 1u));
    o.counts = block + 2u * (nb + 1u) + nb * kHeldBucketCap * (sizeof(HeldEntry) / 4u);
}

// The same, cleared.
__device__ inline void held_order_carve(DeviceKeyOrder& o, uint32_t* block, uint32_t bits) {
    held_order_point(o, block, bits);
    hgcommon::ko_clear(o);
}

// One state's held matches.
struct DeviceHeldSet {
    DeviceKeyOrder order;
    uint32_t* order_block = nullptr;
    // The slab: `stride` words per slot (rule | count << 16, key low, key high, a word the update
    // uses, the matched edges in pattern order), `slots` of them; `used` handed out; the freed
    // slots a stack of `free_n` in free_slots[slots].
    uint32_t* slab = nullptr;
    uint32_t  stride = 0;
    uint32_t  slots = 0;
    uint32_t  used = 0;
    uint32_t* free_slots = nullptr;
    uint32_t  free_n = 0;
    // The adjacency: head[v] for v < vcap the first node of vertex v's list; node i is
    // nodes[2i] (the edge) and nodes[2i + 1] (the next node); freed nodes chained from free_node.
    uint32_t* head = nullptr;
    uint32_t  vcap = 0;
    uint32_t* nodes = nullptr;
    uint32_t  ncap = 0;
    uint32_t  nused = 0;
    uint32_t  free_node = kHeldNone;
    StateId   state = INVALID_ID;
    uint32_t  next = kHeldNone;   // the next set in its step's list (DeviceState::held_step_head)
    uint32_t  live = 0;
    uint32_t  pending = 0;        // the candidates a selection is about to hold (held_select_step)
};

// The arrays of DeviceState's held sets, for the update and the selection.
struct HeldCtx {
    const DeviceState& ds;
    const DeviceRule*  rules;
    HeldPoolView       pool;
    DeviceHeldSet*     sets;   // [ds.held_set_cap]
};

__device__ inline HeldCtx held_ctx(const DeviceState& ds, const DeviceRule* rules) {
    return HeldCtx{ds, rules,
                   HeldPoolView{ds.held_words, ds.held_cursor, ds.held_capacity, ds.held_free_heads},
                   static_cast<DeviceHeldSet*>(ds.held_sets)};
}

// A set's live values: 1 while it is a step's, 2 when its update failed and its state is
// matched in full (the selection frees it).
constexpr uint32_t kHeldLive = 1u;
constexpr uint32_t kHeldFailed = 2u;

__device__ inline uint64_t held_slab_words(uint32_t slots, uint32_t stride) {
    return uint64_t{slots} * stride;
}

__device__ inline bool held_grow_order(const HeldCtx& c, DeviceHeldSet& s) {
    const uint32_t bits = s.order.bits + 1u;
    uint32_t* block = held_alloc(c.pool, held_order_words(bits));
    if (!block) return false;
    DeviceKeyOrder next;
    held_order_carve(next, block, bits);
    if (!hgcommon::ko_rebuild(s.order, next)) {
        held_free(c.pool, block, held_order_words(bits));
        return false;
    }
    held_free(c.pool, s.order_block, held_order_words(s.order.bits));
    s.order = next;
    s.order_block = block;
    return true;
}

__device__ inline uint32_t held_rule_at(const DeviceHeldSet& s, uint32_t slot) {
    return s.slab[uint64_t{slot} * s.stride] & 0xFFFFu;
}
__device__ inline uint32_t held_count_at(const DeviceHeldSet& s, uint32_t slot) {
    return s.slab[uint64_t{slot} * s.stride] >> 16;
}
__device__ inline const EdgeId* held_edges_at(const DeviceHeldSet& s, uint32_t slot) {
    return s.slab + uint64_t{slot} * s.stride + 4u;
}
__device__ inline uint64_t held_key_at(const DeviceHeldSet& s, uint32_t slot) {
    const uint32_t* w = s.slab + uint64_t{slot} * s.stride;
    return w[1] | (uint64_t{w[2]} << 32);
}
__device__ inline bool held_same(const DeviceHeldSet& s, uint32_t slot, uint32_t rule,
                                 const EdgeId* edges, uint32_t n) {
    if (held_rule_at(s, slot) != rule || held_count_at(s, slot) != n) return false;
    const EdgeId* e = held_edges_at(s, slot);
    for (uint32_t i = 0; i < n; ++i)
        if (e[i] != edges[i]) return false;
    return true;
}

// The lineage key of a match (hgcommon::lineage_transition_key over the edges' labels), the
// key the host's HeldMatches and step_order_device order by.
__device__ inline uint64_t held_key(const DeviceState& ds, uint32_t rule, const EdgeId* edges,
                                    uint32_t n) {
    uint64_t labels[kMaxPatternEdges];
    for (uint32_t i = 0; i < n; ++i) labels[i] = ds.edge_label[edges[i]];
    return hgcommon::lineage_transition_key(rule, labels, n);
}

// Room for `need` slots in the slab and its free stack; false when the pool cannot hold them.
__device__ inline bool held_grow_slab(const HeldCtx& c, DeviceHeldSet& s, uint32_t need) {
    if (need <= s.slots) return true;
    uint32_t slots = s.slots ? s.slots : 64u;
    while (slots < need) slots *= 2u;
    uint32_t* slab = held_alloc(c.pool, held_slab_words(slots, s.stride));
    uint32_t* free_slots = held_alloc(c.pool, slots);
    if (!slab || !free_slots) {
        held_free(c.pool, slab, held_slab_words(slots, s.stride));
        held_free(c.pool, free_slots, slots);
        return false;
    }
    for (uint64_t w = 0; w < held_slab_words(s.used, s.stride); ++w) slab[w] = s.slab[w];
    for (uint32_t i = 0; i < s.free_n; ++i) free_slots[i] = s.free_slots[i];
    held_free(c.pool, s.slab, held_slab_words(s.slots, s.stride));
    held_free(c.pool, s.free_slots, s.slots);
    s.slab = slab;
    s.free_slots = free_slots;
    s.slots = slots;
    return true;
}

// Holds the match unless it is held; false when the set could not grow (the match is not held).
__device__ inline bool held_add(const HeldCtx& c, DeviceHeldSet& s, uint32_t rule,
                                const EdgeId* edges, uint32_t n) {
    const uint64_t key = held_key(c.ds, rule, edges, n);
    if (hgcommon::ko_contains(s.order, key,
                              [&](uint32_t slot) { return held_same(s, slot, rule, edges, n); }))
        return true;
    uint32_t slot;
    if (s.free_n != 0u) {
        slot = s.free_slots[--s.free_n];
    } else {
        if (!held_grow_slab(c, s, s.used + 1u)) return false;
        slot = s.used++;
    }
    uint32_t* w = s.slab + uint64_t{slot} * s.stride;
    w[0] = rule | (n << 16);
    w[1] = static_cast<uint32_t>(key);
    w[2] = static_cast<uint32_t>(key >> 32);
    w[3] = kHeldNone;
    for (uint32_t i = 0; i < n; ++i) w[4 + i] = edges[i];
    if (hgcommon::ko_needs_grow(s.order) && !held_grow_order(c, s)) return false;
    while (!hgcommon::ko_insert(s.order, key, slot))
        if (!held_grow_order(c, s)) return false;
    return true;
}

// Drops the held match (rule, edges), when held.
__device__ inline void held_drop(const HeldCtx& c, DeviceHeldSet& s, uint32_t rule,
                                 const EdgeId* edges, uint32_t n) {
    const uint64_t key = held_key(c.ds, rule, edges, n);
    uint32_t found = kHeldNone;
    if (!hgcommon::ko_erase(s.order, key, [&](uint32_t slot) {
            if (!held_same(s, slot, rule, edges, n)) return false;
            found = slot;
            return true;
        }))
        return;
    s.free_slots[s.free_n++] = found;
}

__device__ inline bool held_grow_vertices(const HeldCtx& c, DeviceHeldSet& s, uint32_t v) {
    uint32_t vcap = s.vcap ? s.vcap : 64u;
    while (vcap <= v) vcap *= 2u;
    uint32_t* head = held_alloc(c.pool, vcap);
    if (!head) return false;
    for (uint32_t i = 0; i < s.vcap; ++i) head[i] = s.head[i];
    for (uint32_t i = s.vcap; i < vcap; ++i) head[i] = kHeldNone;
    held_free(c.pool, s.head, s.vcap);
    s.head = head;
    s.vcap = vcap;
    return true;
}

// Edge e with vertices v[0..arity) enters the state, at each distinct vertex.
__device__ inline bool held_link(const HeldCtx& c, DeviceHeldSet& s, EdgeId e, const VertexId* v,
                                 uint32_t arity) {
    for (uint32_t i = 0; i < arity; ++i) {
        bool seen = false;
        for (uint32_t j = 0; j < i && !seen; ++j) seen = v[j] == v[i];
        if (seen) continue;
        if (v[i] >= s.vcap && !held_grow_vertices(c, s, v[i])) return false;
        uint32_t node;
        if (s.free_node != kHeldNone) {
            node = s.free_node;
            s.free_node = s.nodes[2u * node + 1u];
        } else {
            if (s.nused == s.ncap) {
                const uint32_t ncap = s.ncap ? 2u * s.ncap : 64u;
                uint32_t* nodes = held_alloc(c.pool, 2ull * ncap);
                if (!nodes) return false;
                for (uint64_t w = 0; w < 2ull * s.nused; ++w) nodes[w] = s.nodes[w];
                held_free(c.pool, s.nodes, 2ull * s.ncap);
                s.nodes = nodes;
                s.ncap = ncap;
            }
            node = s.nused++;
        }
        s.nodes[2u * node] = e;
        s.nodes[2u * node + 1u] = s.head[v[i]];
        s.head[v[i]] = node;
    }
    return true;
}

// Edge e with vertices v[0..arity) leaves the state.
__device__ inline void held_unlink(DeviceHeldSet& s, EdgeId e, const VertexId* v, uint32_t arity) {
    for (uint32_t i = 0; i < arity; ++i) {
        bool seen = false;
        for (uint32_t j = 0; j < i && !seen; ++j) seen = v[j] == v[i];
        if (seen || v[i] >= s.vcap) continue;
        uint32_t* at = &s.head[v[i]];
        while (*at != kHeldNone && s.nodes[2u * *at] != e) at = &s.nodes[2u * *at + 1u];
        if (*at == kHeldNone) continue;
        const uint32_t node = *at;
        *at = s.nodes[2u * node + 1u];
        s.nodes[2u * node + 1u] = s.free_node;
        s.free_node = node;
    }
}

// The join over a held set's adjacency (hgcommon::join_seed): a pattern edge with a bound
// variable takes its candidates from that vertex's list, one without from the state's slice.
// A candidate among `excluded` is not usable.
struct HeldJoinCtx {
    using JoinState = hgcommon::JoinState<kMaxPatternEdges, kMaxVars, EdgeId, VertexId>;
    const DeviceState&   ds;
    const DeviceRule&    rule;
    const DeviceHeldSet& s;
    StateEdgeSlice       slice;
    const EdgeId*        excluded;
    uint32_t             num_excluded;
    // The candidates of the position bound after the anchor are dealt to `split` parts, and this
    // join takes those of part `part`: a seed's parts together are the seed's join.
    uint32_t             split = 1;
    uint32_t             part = 0;

    struct Cand {
        EdgeId          id;
        const VertexId* v;
        uint8_t         arity;
    };
    struct Cursor {
        uint32_t at;     // the next list node, or the next slice index
        uint8_t  scan;
        uint8_t  dealt;  // 1 at the depth after the anchor
        uint32_t index;  // candidates yielded there so far
    };

    __device__ uint8_t num_lhs_edges() const { return rule.num_lhs_edges; }
    __device__ uint8_t order_at(uint8_t k) const { return rule.order[k]; }
    __device__ const uint8_t* pattern_vars(uint8_t p) const { return rule.lhs[p].vars; }
    __device__ uint8_t pattern_arity(uint8_t p) const { return rule.lhs[p].arity; }
    __device__ Cand candidate_of(EdgeId e) const {
        const Edge& ed = ds.edge_pool.data[e];
        return Cand{e, ds.vertex_pool.data + ed.vertex_offset, ed.arity};
    }
    __device__ EdgeId candidate_id(const Cand& c) const { return c.id; }
    __device__ const VertexId* edge_vertices(const Cand& c) const { return c.v; }
    __device__ uint8_t edge_arity(const Cand& c) const { return c.arity; }
    __device__ bool usable(EdgeId e) const {
        for (uint32_t i = 0; i < num_excluded; ++i)
            if (excluded[i] == e) return false;
        return true;
    }
    __device__ bool aborted() const { return false; }
    __device__ void cursor_open(uint8_t p, const JoinState& st, Cursor& c) const {
        c.dealt = st.depth == 1u && split > 1u ? 1u : 0u;
        c.index = 0;
        const DevicePatternEdge& pe = rule.lhs[p];
        for (uint8_t k = 0; k < pe.arity; ++k) {
            const uint8_t var = pe.vars[k];
            if (!((st.bound_mask >> var) & 1u)) continue;
            const VertexId v = st.binding[var];
            c.scan = 0;
            c.at = v < s.vcap ? s.head[v] : kHeldNone;
            return;
        }
        c.scan = 1;
        c.at = 0;
    }
    __device__ bool next_candidate(Cursor& c, Cand& out) const {
        if (c.scan) {
            if (c.at >= slice.count) return false;
            out = candidate_of(ds.state_edge_ids[slice.offset + c.at++]);
            return true;
        }
        if (c.at == kHeldNone) return false;
        out = candidate_of(s.nodes[2u * c.at]);
        c.at = s.nodes[2u * c.at + 1u];
        return true;
    }
    __device__ bool cursor_next(Cursor& c, Cand& out) const {
        if (!c.dealt) return next_candidate(c, out);
        while (next_candidate(c, out))
            if (c.index++ % split == part) return true;
        return false;
    }
};

// The update's work items: each (anchor, rule, pattern position, part), the part one of
// kHeldSplit deals of the candidates bound after the anchor (HeldJoinCtx::split). Item i is part
// i % kHeldSplit of seed i / kHeldSplit, and seed t is anchor t / P with the (t % P)-th
// (rule, position) in rule order, P the rules' left-hand edges together.
constexpr uint32_t kHeldSplit = 8;

__device__ inline uint32_t held_positions(const HeldCtx& c) {
    uint32_t n = 0;
    for (uint32_t r = 0; r < c.ds.num_rules; ++r) n += c.rules[r].num_lhs_edges;
    return n;
}

__device__ inline uint32_t held_item_count(const HeldCtx& c, uint32_t anchors) {
    return anchors * held_positions(c) * kHeldSplit;
}

// The matches of work item i (held_item_count) in `state` over the set's adjacency, with the
// anchors before its own unused when `exclude` holds, to emit(rule, edges by pattern position,
// count).
template <class Emit>
__device__ inline void held_item(const HeldCtx& c, const DeviceHeldSet& s, StateId state,
                                 const EdgeId* anchors, uint32_t i, bool exclude, Emit&& emit) {
    const uint32_t positions = held_positions(c);
    const uint32_t seed = i / kHeldSplit, part = i % kHeldSplit;
    const uint32_t a = seed / positions;
    uint32_t rem = seed % positions, r = 0;
    while (rem >= c.rules[r].num_lhs_edges) rem -= c.rules[r++].num_lhs_edges;
    const DeviceRule& rule = c.rules[r];
    // A one-edge rule's match is its anchor, which every part would emit.
    if (rule.num_lhs_edges == 1u && part != 0u) return;
    HeldJoinCtx ctx{c.ds, rule, s, c.ds.state_edge_slices[state], exclude ? anchors : nullptr,
                    exclude ? a : 0u};
    ctx.split = kHeldSplit;
    ctx.part = part;
    HeldJoinCtx::JoinState st;
    hgcommon::join_seed(ctx, st, anchors[a], static_cast<uint8_t>(rem),
                        [&](const HeldJoinCtx::JoinState& done) {
                            EdgeId edges[kMaxPatternEdges];
                            for (uint8_t d = 0; d < done.depth; ++d)
                                edges[done.pattern[d]] = done.matched[d];
                            emit(r, edges, rule.num_lhs_edges);
                        });
}

// Matches one pass of an update holds in its buffer, and the words of each: the rule, the key's
// two words, the slot, then the edges in pattern order.
constexpr uint32_t kHeldFound = 16384;
constexpr uint32_t kHeldFoundWords = 4u + kMaxPatternEdges;
// Work items a pass reruns on one thread when the buffer filled during them.
constexpr uint32_t kHeldOverflowItems = 64;

// The child's update of the set handed to it, by the whole block (one warp): the parent's matches
// on each consumed edge go, the consumed edges leave the adjacency and the produced ones enter,
// and the child's matches on each produced edge are held, each found from its first produced
// edge. The lanes run the seeded joins of a pass into a buffer, then edit the order by bucket
// (bucket % 32 is a lane's), so no bucket is edited by two lanes; the tree and the size take
// atomic adds. A seed whose matches did not fit the buffer is run again on thread 0. False when
// the pool could not hold the set (the caller matches the state itself).
__device__ inline bool held_update_block(const HeldCtx& c, uint32_t h, StateId child,
                                         EventId event) {
    DeviceHeldSet& s = c.sets[h];
    const uint32_t lane = threadIdx.x, nt = blockDim.x;
    __shared__ uint32_t* buf;
    __shared__ uint32_t s_found;
    __shared__ uint32_t s_failed;
    __shared__ uint32_t s_overflow[kHeldOverflowItems];
    __shared__ uint32_t s_overflow_n;
    __shared__ bool s_ok;
    const uint64_t buf_words = uint64_t{kHeldFound} * kHeldFoundWords;
    const DeviceEvent& ev = c.ds.event_pool.at(event);
    const EdgeId* consumed = c.ds.event_consumed + ev.consumed_at;
    const uint32_t nc = ev.num_consumed;
    const StateId parent = ev.input_state;
    EdgeId produced[kMaxPatternEdges];
    const uint32_t np = ev.num_produced;
    for (uint32_t i = 0; i < np; ++i) produced[i] = ev.first_produced + i;
    if (lane == 0) {
        buf = held_alloc(c.pool, buf_words);
        s_ok = buf != nullptr;
    }
    __syncthreads();
    if (!s_ok) return false;

    // One pass: every seed of `anchors` into the buffer.
    auto collect = [&](StateId state, const EdgeId* anchors, uint32_t num_anchors, bool exclude) {
        if (lane == 0) { s_found = 0; s_overflow_n = 0; }
        __syncthreads();
        const uint32_t items = held_item_count(c, num_anchors);
        for (uint32_t t = lane; t < items; t += nt) {
            bool over = false;
            held_item(c, s, state, anchors, t, exclude,
                      [&](uint32_t r, const EdgeId* e, uint32_t n) {
                          const uint32_t at = atomicAdd(&s_found, 1u);
                          if (at >= kHeldFound) { over = true; return; }
                          uint32_t* w = buf + uint64_t{at} * kHeldFoundWords;
                          const uint64_t key = held_key(c.ds, r, e, n);
                          w[0] = r | (n << 16);
                          w[1] = static_cast<uint32_t>(key);
                          w[2] = static_cast<uint32_t>(key >> 32);
                          w[3] = kHeldNone;
                          for (uint32_t i = 0; i < n; ++i) w[4 + i] = e[i];
                      });
            if (over) {
                const uint32_t k = atomicAdd(&s_overflow_n, 1u);
                if (k < kHeldOverflowItems) s_overflow[k] = t;
            }
        }
        __syncthreads();
    };
    auto entry_key = [&](uint32_t e) {
        const uint32_t* w = buf + uint64_t{e} * kHeldFoundWords;
        return w[1] | (uint64_t{w[2]} << 32);
    };

    // DROP: the parent's matches on the consumed edges, over the parent's adjacency.
    collect(parent, consumed, nc, false);
    // An edit of more than one entry per eight buckets rebuilds the tree once.
    auto defer_tree = [&](uint32_t m) {
        if (lane == 0) s.order.defer_tree = m > (1u << s.order.bits) / 8u ? 1u : 0u;
        __syncthreads();
    };
    auto settle_tree = [&]() {
        if (s.order.defer_tree) {
            held_order_rebuild_tree(s.order);
            if (lane == 0) s.order.defer_tree = 0u;
        }
        __syncthreads();
    };
    defer_tree(min(s_found, kHeldFound));
    {
        const uint32_t m = min(s_found, kHeldFound);
        for (uint32_t e = 0; e < m; ++e) {
            const uint64_t key = entry_key(e);
            if (hgcommon::ko_bucket_of(key, s.order.bits) % nt != lane) continue;
            const uint32_t* w = buf + uint64_t{e} * kHeldFoundWords;
            const uint32_t rule = w[0] & 0xFFFFu, n = w[0] >> 16;
            uint32_t found = kHeldNone;
            if (hgcommon::ko_erase(s.order, key, [&](uint32_t slot) {
                    if (!held_same(s, slot, rule, w + 4, n)) return false;
                    found = slot;
                    return true;
                }))
                s.free_slots[atomicAdd(&s.free_n, 1u)] = found;
        }
    }
    __syncthreads();
    settle_tree();
    if (lane == 0) {
        // Items whose matches did not fit: run again here, dropping directly.
        const bool all = s_overflow_n > kHeldOverflowItems;
        const uint32_t items = held_item_count(c, nc);
        for (uint32_t k = 0; k < (all ? items : s_overflow_n); ++k)
            held_item(c, s, parent, consumed, all ? k : s_overflow[k], false,
                      [&](uint32_t r, const EdgeId* e, uint32_t n) { held_drop(c, s, r, e, n); });
        // The adjacency: the consumed edges leave and the produced enter.
        for (uint32_t i = 0; i < nc; ++i) {
            const Edge& ed = c.ds.edge_pool.data[consumed[i]];
            held_unlink(s, consumed[i], c.ds.vertex_pool.data + ed.vertex_offset, ed.arity);
        }
        for (uint32_t i = 0; i < np && s_ok; ++i) {
            const Edge& ed = c.ds.edge_pool.data[produced[i]];
            s_ok = held_link(c, s, produced[i], c.ds.vertex_pool.data + ed.vertex_offset, ed.arity);
        }
    }
    __syncthreads();
    if (!s_ok) {
        if (lane == 0) held_free(c.pool, buf, buf_words);
        __syncthreads();
        return false;
    }

    // ADD: the child's matches on the produced edges, each from its first produced edge.
    collect(child, produced, np, true);
    __shared__ uint32_t s_reuse;
    if (lane == 0) {
        const uint32_t m = min(s_found, kHeldFound);
        s_failed = 0;
        // Room for every match in the order and the slab: the free stack's top m slots, then
        // fresh ones.
        while (s_ok && s.order.size + m > (uint64_t{hgcommon::ko_load} << s.order.bits))
            s_ok = held_grow_order(c, s);
        s_reuse = min(m, s.free_n);
        if (s_ok) s_ok = held_grow_slab(c, s, s.used + (m - s_reuse));
    }
    __syncthreads();
    if (s_ok) {
        const uint32_t m = min(s_found, kHeldFound);
        const uint32_t reuse = s_reuse;
        // A slot and the slab entry for each match, a lane each.
        for (uint32_t e = lane; e < m; e += nt) {
            uint32_t* w = buf + uint64_t{e} * kHeldFoundWords;
            w[3] = e < reuse ? s.free_slots[s.free_n - 1u - e] : s.used + (e - reuse);
            uint32_t* d = s.slab + uint64_t{w[3]} * s.stride;
            d[0] = w[0];
            d[1] = w[1];
            d[2] = w[2];
            d[3] = kHeldNone;
            for (uint32_t i = 0; i < (w[0] >> 16); ++i) d[4 + i] = w[4 + i];
        }
        __syncthreads();
        defer_tree(m);
        // The order, by bucket. A full bucket leaves the entry to thread 0.
        for (uint32_t e = 0; e < m; ++e) {
            const uint64_t key = entry_key(e);
            if (hgcommon::ko_bucket_of(key, s.order.bits) % nt != lane) continue;
            if (!hgcommon::ko_insert(s.order, key, buf[uint64_t{e} * kHeldFoundWords + 3u])) {
                buf[uint64_t{e} * kHeldFoundWords] |= 0x80000000u;
                atomicAdd(&s_failed, 1u);
            }
        }
        __syncthreads();
        settle_tree();
    }
    __syncthreads();
    if (lane == 0 && s_ok) {
        const uint32_t m = min(s_found, kHeldFound);
        s.free_n -= s_reuse;
        s.used += m - s_reuse;
        for (uint32_t e = 0; e < m && s_failed && s_ok; ++e) {
            uint32_t* w = buf + uint64_t{e} * kHeldFoundWords;
            if (!(w[0] & 0x80000000u)) continue;
            w[0] &= 0x7FFFFFFFu;
            while (s_ok && !hgcommon::ko_insert(s.order, entry_key(e), w[3]))
                s_ok = held_grow_order(c, s);
        }
        const bool all = s_overflow_n > kHeldOverflowItems;
        const uint32_t items = held_item_count(c, np);
        for (uint32_t k = 0; k < (all ? items : s_overflow_n) && s_ok; ++k)
            held_item(c, s, child, produced, all ? k : s_overflow[k], true,
                      [&](uint32_t r, const EdgeId* e, uint32_t n) {
                          if (s_ok) s_ok = held_add(c, s, r, e, n);
                      });
        s.state = child;
        held_free(c.pool, buf, buf_words);
    }
    __syncthreads();
    return s_ok;
}

// SET LIFECYCLE. The selection of a step creates, copies and frees the step's sets on one
// thread, and only a selection does, so the stack of free headers needs no atomics: the
// selections of a run are ordered by the step tokens (persistent.cu, step_release).

__device__ inline void held_free_arrays(const HeldCtx& c, DeviceHeldSet& s) {
    held_free(c.pool, s.order_block, held_order_words(s.order.bits));
    held_free(c.pool, s.slab, held_slab_words(s.slots, s.stride));
    held_free(c.pool, s.free_slots, s.slots);
    held_free(c.pool, s.head, s.vcap);
    held_free(c.pool, s.nodes, 2ull * s.ncap);
    s = DeviceHeldSet{};
}

__device__ inline void held_free_set(const HeldCtx& c, uint32_t h) {
    held_free_arrays(c, c.sets[h]);
    uint32_t* stack = c.ds.held_free_sets;
    stack[1u + stack[0]] = h;
    ++stack[0];
}

// A header with an empty order of ko_min_bits buckets, or kHeldNone.
__device__ inline uint32_t held_new_header(const HeldCtx& c, StateId state) {
    uint32_t* stack = c.ds.held_free_sets;
    if (stack[0] == 0u) return kHeldNone;
    const uint32_t h = stack[stack[0]];
    --stack[0];
    DeviceHeldSet& s = c.sets[h];
    s = DeviceHeldSet{};
    s.stride = c.ds.held_stride;
    s.state = state;
    s.live = kHeldLive;
    s.order_block = held_alloc(c.pool, held_order_words(hgcommon::ko_min_bits));
    if (!s.order_block) {
        held_free_set(c, h);
        return kHeldNone;
    }
    held_order_carve(s.order, s.order_block, hgcommon::ko_min_bits);
    return h;
}

// A set for `state` with its edges in the adjacency and no match, or kHeldNone.
__device__ inline uint32_t held_new_set(const HeldCtx& c, StateId state) {
    const uint32_t h = held_new_header(c, state);
    if (h == kHeldNone) return h;
    DeviceHeldSet& s = c.sets[h];
    const StateEdgeSlice slice = c.ds.state_edge_slices[state];
    for (uint32_t i = 0; i < slice.count; ++i) {
        const EdgeId e = c.ds.state_edge_ids[slice.offset + i];
        const Edge& ed = c.ds.edge_pool.data[e];
        if (!held_link(c, s, e, c.ds.vertex_pool.data + ed.vertex_offset, ed.arity)) {
            held_free_set(c, h);
            return kHeldNone;
        }
    }
    return h;
}

// A copy of set h, or kHeldNone.
__device__ inline uint32_t held_copy_set(const HeldCtx& c, uint32_t h) {
    const DeviceHeldSet& from = c.sets[h];
    uint32_t* stack = c.ds.held_free_sets;
    if (stack[0] == 0u) return kHeldNone;
    const uint32_t k = stack[stack[0]];
    --stack[0];
    DeviceHeldSet& s = c.sets[k];
    s = from;
    s.next = kHeldNone;
    s.order_block = held_alloc(c.pool, held_order_words(from.order.bits));
    s.slab = from.slots ? held_alloc(c.pool, held_slab_words(from.slots, from.stride)) : nullptr;
    s.free_slots = from.slots ? held_alloc(c.pool, from.slots) : nullptr;
    s.head = from.vcap ? held_alloc(c.pool, from.vcap) : nullptr;
    s.nodes = from.ncap ? held_alloc(c.pool, 2ull * from.ncap) : nullptr;
    if (!s.order_block || (from.slots && (!s.slab || !s.free_slots)) || (from.vcap && !s.head) ||
        (from.ncap && !s.nodes)) {
        if (s.order_block) held_free(c.pool, s.order_block, held_order_words(from.order.bits));
        if (s.slab) held_free(c.pool, s.slab, held_slab_words(from.slots, from.stride));
        if (s.free_slots) held_free(c.pool, s.free_slots, from.slots);
        if (s.head) held_free(c.pool, s.head, from.vcap);
        if (s.nodes) held_free(c.pool, s.nodes, 2ull * from.ncap);
        s = DeviceHeldSet{};
        stack[1u + stack[0]] = k;
        ++stack[0];
        return kHeldNone;
    }
    const uint64_t ow = held_order_words(from.order.bits);
    for (uint64_t w = 0; w < ow; ++w) s.order_block[w] = from.order_block[w];
    held_order_point(s.order, s.order_block, from.order.bits);
    s.order.size = from.order.size;
    for (uint64_t w = 0; w < held_slab_words(from.used, from.stride); ++w) s.slab[w] = from.slab[w];
    for (uint32_t i = 0; i < from.free_n; ++i) s.free_slots[i] = from.free_slots[i];
    for (uint32_t w = 0; w < from.vcap; ++w) s.head[w] = from.head[w];
    for (uint64_t w = 0; w < 2ull * from.nused; ++w) s.nodes[w] = from.nodes[w];
    return k;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
