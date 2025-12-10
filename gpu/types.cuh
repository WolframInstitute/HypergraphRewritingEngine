#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// CUDA error checking macro
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
} while(0)
#endif

namespace hypergraph::gpu {

// =============================================================================
// Constants (match unified/)
// =============================================================================

static constexpr uint32_t MAX_EDGE_ARITY = 16;
static constexpr uint32_t MAX_PATTERN_EDGES = 16;
static constexpr uint32_t MAX_VARS = 32;
static constexpr uint32_t INVALID_ID = 0xFFFFFFFF;

// Pool sizes (reduced for faster testing - increase for production)
static constexpr size_t MAX_EDGES = 10000;       // Was 500000
static constexpr size_t MAX_STATES = 10000;      // Was 50000
static constexpr size_t MAX_EVENTS = 50000;      // Was 200000
static constexpr size_t MAX_MATCHES_PER_STEP = 100000;  // Was 2000000
static constexpr size_t WORK_QUEUE_SIZE = 10000; // Was 100000
static constexpr size_t HASH_TABLE_SIZE = 16384; // Was 131072 (16K power of 2)

// Bitmap words for state edge membership (supports up to MAX_EDGES edges)
// With MAX_EDGES=10000, this is 157 words = 1256 bytes per state
static constexpr size_t BITMAP_WORDS = (MAX_EDGES + 63) / 64;

// =============================================================================
// Type Aliases (match unified/)
// =============================================================================

using VertexId = uint32_t;
using EdgeId = uint32_t;
using StateId = uint32_t;
using EventId = uint32_t;

// =============================================================================
// Edge Storage (Struct-of-Arrays for coalescing)
// =============================================================================
// Edges stored in CSR-style format for efficient GPU access.
// vertex_offsets[i] gives start index into vertex_data for edge i.

struct DeviceEdges {
    uint32_t* vertex_offsets;    // [MAX_EDGES+1] Start offset for each edge's vertices
    uint32_t* vertex_data;       // Flattened vertex data (sum of all arities)
    uint8_t* arities;            // [MAX_EDGES] Arity of each edge
    uint32_t* creator_events;    // [MAX_EDGES] Event that produced this edge (INVALID_ID for initial)
    uint32_t* num_edges;         // Pointer to device counter for atomic edge allocation
    uint32_t* num_vertex_data;   // Pointer to device counter for vertex_data allocation
    uint32_t cached_num_edges;   // Cached count for read-only access (set before kernel launch)

    // Device-side helpers
    __device__ uint8_t get_arity(EdgeId eid) const {
        return arities[eid];
    }

    __device__ VertexId get_vertex(EdgeId eid, uint8_t pos) const {
        return vertex_data[vertex_offsets[eid] + pos];
    }

    __device__ const uint32_t* get_vertices(EdgeId eid) const {
        return &vertex_data[vertex_offsets[eid]];
    }

    // Allocate a new edge, returns edge ID or INVALID_ID if full
    __device__ uint32_t alloc_edge() {
        uint32_t eid = atomicAdd(num_edges, 1);
        if (eid >= MAX_EDGES) {
            atomicSub(num_edges, 1);
            return INVALID_ID;
        }
        return eid;
    }

    // Allocate space in vertex_data array, returns offset
    __device__ uint32_t alloc_vertex_data(uint32_t count) {
        return atomicAdd(num_vertex_data, count);
    }
};

// =============================================================================
// State Storage
// =============================================================================
// Each state has a bitmap indicating which edges it contains.
// Bitmaps use 64-bit words for efficient atomics.

struct DeviceState {
    uint64_t* bitmap;           // [BITMAP_WORDS] Bit i set if edge i in state
    uint64_t canonical_hash;    // WL canonical hash
    uint32_t parent_state;      // Parent state (INVALID_ID for initial)
    uint32_t parent_event;      // Event that created this state
    uint32_t step;              // Evolution step when created
    uint32_t edge_count;        // Number of edges in state

    // Device-side helpers
    __device__ bool has_edge(EdgeId eid) const {
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        return (bitmap[word] & (1ULL << bit)) != 0;
    }

    __device__ void set_edge(EdgeId eid) {
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        atomicOr((unsigned long long*)&bitmap[word], 1ULL << bit);
    }

    __device__ void clear_edge(EdgeId eid) {
        uint32_t word = eid / 64;
        uint32_t bit = eid % 64;
        atomicAnd((unsigned long long*)&bitmap[word], ~(1ULL << bit));
    }
};

// State pool for bulk allocation
struct StatePool {
    DeviceState* states;        // [MAX_STATES] State metadata
    uint64_t* all_bitmaps;      // [MAX_STATES * BITMAP_WORDS] All bitmaps contiguous
    uint32_t* num_states;       // Pointer to device counter for atomic updates

    __device__ DeviceState* get_state(StateId sid) {
        return &states[sid];
    }

    __device__ uint64_t* get_bitmap(StateId sid) {
        return &all_bitmaps[sid * BITMAP_WORDS];
    }

    // Allocate a new state, returns state ID or INVALID_ID if full
    __device__ uint32_t alloc_state() {
        uint32_t idx = atomicAdd(num_states, 1);
        if (idx >= MAX_STATES) {
            atomicSub(num_states, 1);
            return INVALID_ID;
        }
        return idx;
    }

    // Get current count (device side)
    __device__ uint32_t size() const {
        return *num_states;
    }
};

// =============================================================================
// Match Storage (fixed size, no pointers)
// =============================================================================

struct DeviceMatch {
    uint32_t matched_edges[MAX_PATTERN_EDGES];  // Edge IDs matched by pattern
    uint32_t bindings[MAX_VARS];                // Variable bindings (var -> vertex)
    uint32_t bound_mask;                        // Bitmask of bound variables
    uint32_t source_state;                      // State this match was found in
    uint64_t source_hash;                       // Canonical hash of source state
    uint16_t rule_index;                        // Which rule matched
    uint8_t num_edges;                          // Number of edges in match

    __device__ bool is_bound(uint8_t var) const {
        return (bound_mask & (1u << var)) != 0;
    }

    __device__ VertexId get_binding(uint8_t var) const {
        return bindings[var];
    }

    __device__ void bind(uint8_t var, VertexId vertex) {
        bindings[var] = vertex;
        bound_mask |= (1u << var);
    }

    // Hash for deduplication (FNV-1a style)
    __device__ uint64_t hash() const {
        uint64_t h = 14695981039346656037ULL;
        constexpr uint64_t FNV_PRIME = 1099511628211ULL;

        h ^= rule_index;
        h *= FNV_PRIME;
        h ^= source_state;
        h *= FNV_PRIME;

        for (uint8_t i = 0; i < num_edges; ++i) {
            h ^= matched_edges[i];
            h *= FNV_PRIME;
        }

        h ^= bound_mask;
        h *= FNV_PRIME;

        for (uint8_t i = 0; i < MAX_VARS; ++i) {
            if (is_bound(i)) {
                h ^= (uint64_t(i) << 32) | bindings[i];
                h *= FNV_PRIME;
            }
        }

        return h;
    }
};

// =============================================================================
// Event Storage
// =============================================================================

struct DeviceEvent {
    uint32_t input_state;       // State this event transforms
    uint32_t output_state;      // State produced by this event
    uint32_t consumed_offset;   // Offset into consumed_edges array
    uint32_t produced_offset;   // Offset into produced_edges array
    uint8_t num_consumed;       // Number of edges consumed
    uint8_t num_produced;       // Number of edges produced
    uint16_t rule_index;        // Which rule was applied
    uint32_t step;              // Evolution step
};

struct EventPool {
    DeviceEvent* events;        // [MAX_EVENTS] Event metadata
    uint32_t* consumed_edges;   // Flattened consumed edge IDs
    uint32_t* produced_edges;   // Flattened produced edge IDs
    uint32_t* num_events;       // Pointer to device counter
    uint32_t* consumed_offset;  // Pointer to device counter for next offset
    uint32_t* produced_offset;  // Pointer to device counter for next offset

    // Allocate a new event, returns event ID or INVALID_ID if full
    __device__ uint32_t alloc_event() {
        uint32_t idx = atomicAdd(num_events, 1);
        if (idx >= MAX_EVENTS) {
            atomicSub(num_events, 1);
            return INVALID_ID;
        }
        return idx;
    }

    // Allocate space for consumed edges
    __device__ uint32_t alloc_consumed(uint32_t count) {
        return atomicAdd(consumed_offset, count);
    }

    // Allocate space for produced edges
    __device__ uint32_t alloc_produced(uint32_t count) {
        return atomicAdd(produced_offset, count);
    }

    __device__ uint32_t size() const {
        return *num_events;
    }
};

// =============================================================================
// Rewrite Rule (device-side)
// =============================================================================

struct DevicePatternEdge {
    uint8_t vars[MAX_EDGE_ARITY];   // Variable indices for each position
    uint8_t arity;
};

struct DeviceRewriteRule {
    DevicePatternEdge lhs[MAX_PATTERN_EDGES];   // Left-hand side pattern
    DevicePatternEdge rhs[MAX_PATTERN_EDGES];   // Right-hand side pattern
    uint8_t num_lhs;                            // Number of LHS edges
    uint8_t num_rhs;                            // Number of RHS edges
    uint8_t first_fresh_var;                    // First fresh variable index
};

// =============================================================================
// Causal/Branchial Edges
// =============================================================================

struct CausalEdge {
    uint32_t producer;      // Producer event ID
    uint32_t consumer;      // Consumer event ID
    uint32_t edge;          // Edge that links them
};

struct BranchialEdge {
    uint32_t event1;        // First event
    uint32_t event2;        // Second event
    uint32_t shared_edge;   // Edge consumed by both
};

// =============================================================================
// Online Causal/Branchial Tracking (GPU-side data structures)
// =============================================================================

// Consumer node for per-edge consumer linked list
struct ConsumerNode {
    uint32_t event_id;      // Event that consumed this edge
    uint32_t next;          // Index of next node (INVALID_ID = end of list)
};

// Pool for consumer nodes (one allocation for all edges)
static constexpr size_t MAX_CONSUMER_NODES = MAX_EVENTS * MAX_PATTERN_EDGES;

// State event node for per-state event linked list (for branchial tracking)
struct StateEventNode {
    uint32_t event_id;      // Event from this input state
    uint32_t next;          // Index of next node (INVALID_ID = end of list)
};

// Pool for state event nodes
static constexpr size_t MAX_STATE_EVENT_NODES = MAX_EVENTS;

// Online causal graph tracking
struct OnlineCausalGraph {
    // Per-edge consumer list heads (linked list using ConsumerNode)
    uint32_t* edge_consumer_heads;      // [MAX_EDGES] Head of consumer list per edge

    // Consumer node pool
    ConsumerNode* consumer_nodes;        // [MAX_CONSUMER_NODES] Pool of consumer nodes
    uint32_t* num_consumer_nodes;        // Atomic counter for allocation

    // Per-state event list heads (for branchial tracking)
    // Key: canonical state ID (from state_pool), Value: head of event list
    uint32_t* state_event_heads;         // [MAX_STATES] Head of event list per state

    // State event node pool
    StateEventNode* state_event_nodes;   // [MAX_STATE_EVENT_NODES] Pool
    uint32_t* num_state_event_nodes;     // Atomic counter for allocation

    // Output arrays (same as before)
    CausalEdge* causal_edges;
    uint32_t* num_causal_edges;
    BranchialEdge* branchial_edges;
    uint32_t* num_branchial_edges;

    // Causal triple deduplication hash set (producer, consumer, edge)
    // Uses inline hash set to avoid circular includes with hash_table.cuh
    uint64_t* seen_causal_triples;      // Hash set keys (EMPTY = 0xFFFFFFFFFFFFFFFF)
    uint32_t seen_causal_capacity;      // Must be power of 2
    uint32_t seen_causal_mask;          // capacity - 1

    // Branchial pair deduplication hash set (event1, event2)
    uint64_t* seen_branchial_pairs;     // Hash set keys (EMPTY = 0xFFFFFFFFFFFFFFFF)
    uint32_t seen_branchial_capacity;   // Must be power of 2
    uint32_t seen_branchial_mask;       // capacity - 1

    // Causal event pair tracking (for v1 semantics: count unique (producer, consumer) pairs)
    uint64_t* seen_causal_event_pairs;  // Hash set keys (EMPTY = 0xFFFFFFFFFFFFFFFF)
    uint32_t seen_causal_pairs_capacity;  // Must be power of 2
    uint32_t seen_causal_pairs_mask;    // capacity - 1
    uint32_t* num_causal_event_pairs;   // Atomic counter for unique pairs

    // Allocate a consumer node, returns index or INVALID_ID if full
    __device__ uint32_t alloc_consumer_node() {
        uint32_t idx = atomicAdd(num_consumer_nodes, 1);
        if (idx >= MAX_CONSUMER_NODES) {
            atomicSub(num_consumer_nodes, 1);
            return INVALID_ID;
        }
        return idx;
    }

    // Allocate a state event node
    __device__ uint32_t alloc_state_event_node() {
        uint32_t idx = atomicAdd(num_state_event_nodes, 1);
        if (idx >= MAX_STATE_EVENT_NODES) {
            atomicSub(num_state_event_nodes, 1);
            return INVALID_ID;
        }
        return idx;
    }

    // Hash function for causal triple (same as CPU unified)
    __device__ static uint64_t hash_causal_triple(uint32_t producer, uint32_t consumer, uint32_t edge) {
        uint64_t h = 14695981039346656037ULL;  // FNV offset basis
        h ^= producer;
        h *= 1099511628211ULL;  // FNV prime
        h ^= consumer;
        h *= 1099511628211ULL;
        h ^= edge;
        h *= 1099511628211ULL;
        return h;
    }

    // Try to insert a causal triple into dedup hash set
    // Returns true if newly inserted, false if already existed
    __device__ bool insert_causal_triple(uint64_t triple_key) {
        constexpr uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;
        uint32_t slot = static_cast<uint32_t>(triple_key ^ (triple_key >> 32)) & seen_causal_mask;

        for (uint32_t i = 0; i < seen_causal_capacity; ++i) {
            uint64_t prev = atomicCAS(
                (unsigned long long*)&seen_causal_triples[slot],
                EMPTY_KEY,
                triple_key
            );

            if (prev == EMPTY_KEY) {
                return true;  // Newly inserted
            }
            if (prev == triple_key) {
                return false;  // Already existed
            }

            slot = (slot + 1) & seen_causal_mask;
        }
        return false;  // Table full
    }

    // Try to insert a branchial pair into dedup hash set
    // Returns true if newly inserted, false if already existed
    __device__ bool insert_branchial_pair(uint32_t e1, uint32_t e2) {
        // Ensure consistent ordering (smaller ID first)
        if (e1 > e2) {
            uint32_t tmp = e1;
            e1 = e2;
            e2 = tmp;
        }
        uint64_t pair_key = (static_cast<uint64_t>(e1) << 32) | e2;

        constexpr uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;
        uint32_t slot = static_cast<uint32_t>(pair_key ^ (pair_key >> 32)) & seen_branchial_mask;

        for (uint32_t i = 0; i < seen_branchial_capacity; ++i) {
            uint64_t prev = atomicCAS(
                (unsigned long long*)&seen_branchial_pairs[slot],
                EMPTY_KEY,
                pair_key
            );

            if (prev == EMPTY_KEY) {
                return true;  // Newly inserted
            }
            if (prev == pair_key) {
                return false;  // Already existed
            }

            slot = (slot + 1) & seen_branchial_mask;
        }
        return false;  // Table full
    }

    // Try to insert a causal event pair into dedup hash set
    // Returns true if newly inserted, false if already existed
    __device__ bool insert_causal_event_pair(uint32_t producer, uint32_t consumer) {
        uint64_t pair_key = (static_cast<uint64_t>(producer) << 32) | consumer;

        constexpr uint64_t EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;
        uint32_t slot = static_cast<uint32_t>(pair_key ^ (pair_key >> 32)) & seen_causal_pairs_mask;

        for (uint32_t i = 0; i < seen_causal_pairs_capacity; ++i) {
            uint64_t prev = atomicCAS(
                (unsigned long long*)&seen_causal_event_pairs[slot],
                EMPTY_KEY,
                pair_key
            );

            if (prev == EMPTY_KEY) {
                atomicAdd(num_causal_event_pairs, 1);
                return true;  // Newly inserted
            }
            if (prev == pair_key) {
                return false;  // Already existed
            }

            slot = (slot + 1) & seen_causal_pairs_mask;
        }
        return false;  // Table full
    }

    // Add a causal edge (thread-safe with deduplication)
    __device__ void add_causal_edge(uint32_t producer, uint32_t consumer, uint32_t edge) {
        // Compute triple hash for deduplication
        uint64_t triple_key = hash_causal_triple(producer, consumer, edge);

        // Only add if not already seen
        if (!insert_causal_triple(triple_key)) {
            return;  // Duplicate
        }

        uint32_t idx = atomicAdd(num_causal_edges, 1);
        if (idx < MAX_EVENTS * 4) {  // Same limit as before
            causal_edges[idx].producer = producer;
            causal_edges[idx].consumer = consumer;
            causal_edges[idx].edge = edge;
        }

        // Also track unique event pairs (for v1 compatibility)
        insert_causal_event_pair(producer, consumer);
    }

    // Add a branchial edge (thread-safe with deduplication)
    __device__ void add_branchial_edge(uint32_t event1, uint32_t event2, uint32_t shared_edge) {
        // Ensure consistent ordering (smaller ID first)
        if (event1 > event2) {
            uint32_t tmp = event1;
            event1 = event2;
            event2 = tmp;
        }

        // Deduplicate: only add if this pair hasn't been seen
        if (!insert_branchial_pair(event1, event2)) {
            return;  // Already exists
        }

        uint32_t idx = atomicAdd(num_branchial_edges, 1);
        if (idx < MAX_EVENTS) {
            branchial_edges[idx].event1 = event1;
            branchial_edges[idx].event2 = event2;
            branchial_edges[idx].shared_edge = shared_edge;
        }
    }

    // Register that an event produced an edge (set producer, check consumers)
    // Called after edge is created and creator_events[edge] is set
    __device__ void register_edge_producer(uint32_t edge, uint32_t event_id) {
        // Traverse consumer list and create causal edges
        uint32_t node_idx = edge_consumer_heads[edge];
        while (node_idx != INVALID_ID) {
            ConsumerNode& node = consumer_nodes[node_idx];
            add_causal_edge(event_id, node.event_id, edge);
            node_idx = node.next;
        }
    }

    // Register that an event consumed an edge (add to consumer list, check producer)
    __device__ void register_edge_consumer(uint32_t edge, uint32_t event_id, uint32_t producer_event) {
        // Create causal edge if producer exists
        if (producer_event != INVALID_ID) {
            add_causal_edge(producer_event, event_id, edge);
        }

        // Add to consumer list (prepend)
        uint32_t node_idx = alloc_consumer_node();
        if (node_idx != INVALID_ID) {
            consumer_nodes[node_idx].event_id = event_id;

            // CAS to prepend to list
            uint32_t old_head;
            do {
                old_head = edge_consumer_heads[edge];
                consumer_nodes[node_idx].next = old_head;
            } while (atomicCAS(&edge_consumer_heads[edge], old_head, node_idx) != old_head);
        }
    }

    // Register event from a state (for branchial tracking)
    // Matches unified semantics: add self FIRST, then check ALL events
    // Deduplication prevents duplicate edges when both sides detect overlap
    // Returns number of branchial edges created
    __device__ uint32_t register_event_from_state(
        uint32_t state_id,
        uint32_t event_id,
        const uint32_t* consumed_edges,
        uint8_t num_consumed,
        const EventPool* event_pool
    ) {
        // Add this event to the state's event list FIRST (before checking)
        // This ensures both sides of a pair can detect each other
        uint32_t new_node = alloc_state_event_node();
        if (new_node != INVALID_ID) {
            state_event_nodes[new_node].event_id = event_id;

            // CAS to prepend
            uint32_t old_head;
            do {
                old_head = state_event_heads[state_id];
                state_event_nodes[new_node].next = old_head;
            } while (atomicCAS(&state_event_heads[state_id], old_head, new_node) != old_head);
        }

        // Memory fence to ensure our addition is visible before we check
        __threadfence();

        // Re-read head with atomic to ensure we see latest value
        // (Simple load might be cached by compiler)
        uint32_t node_idx = atomicAdd(&state_event_heads[state_id], 0);

        // Now check ALL events in the list (including ones added concurrently)
        // The deduplication in add_branchial_edge prevents duplicates
        uint32_t branchial_count = 0;
        while (node_idx != INVALID_ID) {
            StateEventNode& node = state_event_nodes[node_idx];
            uint32_t other_event = node.event_id;

            if (other_event != event_id) {
                // Get other event's consumed edges
                const DeviceEvent& other = event_pool->events[other_event];
                const uint32_t* other_consumed = &event_pool->consumed_edges[other.consumed_offset];

                // Check for overlap
                for (uint8_t i = 0; i < num_consumed; ++i) {
                    for (uint8_t j = 0; j < other.num_consumed; ++j) {
                        if (consumed_edges[i] == other_consumed[j]) {
                            add_branchial_edge(event_id, other_event, consumed_edges[i]);
                            ++branchial_count;
                            goto next_event;  // Only one branchial edge per event pair
                        }
                    }
                }
            }
            next_event:
            node_idx = node.next;
        }

        return branchial_count;
    }
};

// =============================================================================
// Work Items for Megakernel
// =============================================================================

struct MatchTask {
    StateId state_id;
    uint32_t step;
};

struct RewriteTask {
    DeviceMatch match;
    uint32_t step;
};

// =============================================================================
// Adjacency Index (for WL hash and pattern matching)
// =============================================================================
// CSR format: for each vertex, list of (edge_id, position) pairs

struct DeviceAdjacency {
    uint32_t* row_offsets;      // [max_vertex+2] Start index per vertex
    uint32_t* edge_ids;         // Edge IDs containing each vertex
    uint8_t* positions;         // Position of vertex within each edge
    uint8_t* edge_arities;      // Cached arity for each entry
    uint32_t num_vertices;      // Number of vertices in index
    uint32_t num_entries;       // Total entries in edge_ids/positions

    __device__ uint32_t degree(VertexId v) const {
        return row_offsets[v + 1] - row_offsets[v];
    }

    __device__ uint32_t neighbor_edge(VertexId v, uint32_t idx) const {
        return edge_ids[row_offsets[v] + idx];
    }

    __device__ uint8_t neighbor_position(VertexId v, uint32_t idx) const {
        return positions[row_offsets[v] + idx];
    }
};

// =============================================================================
// Event Canonicalization Mode (forward declaration for EvolutionContext)
// =============================================================================
// Controls how events are deduplicated. Full definition at end of file.

enum class EventCanonicalizationMode : uint8_t {
    None,              // No event deduplication - all events created
    ByState,           // Deduplicate by (canonical_input, canonical_output) only
    ByStateAndEdges    // Deduplicate by canonical states + edge correspondence
};

// =============================================================================
// Task Granularity Mode
// =============================================================================
// Controls how pattern matching work is distributed across GPU threads.
// This is a key performance tuning parameter.

enum class TaskGranularity : uint8_t {
    // COARSE: Each warp handles one full state
    // - Warp-cooperative: all 32 lanes scan edges together
    // - Better memory coalescing, less queue contention
    // - Good for: small patterns, many states, high edge counts
    Coarse,

    // MEDIUM: Each warp handles one (state, rule) pair
    // - Still warp-cooperative within a rule
    // - Moderate queue pressure
    // - Good for: multiple rules with different complexities
    Medium,

    // FINE: Each task is one partial match expansion (HGMatch style)
    // - Maximum parallelism potential
    // - Higher queue contention, less coalescing
    // - Good for: large patterns, few states, low edge counts
    Fine
};

// =============================================================================
// Evolution Context (all pointers for megakernel)
// =============================================================================

struct EvolutionContext {
    // Data pools
    DeviceEdges* edges;
    StatePool* states;
    EventPool* events;
    DeviceAdjacency* adjacency;

    // Rules
    DeviceRewriteRule* rules;
    uint32_t num_rules;

    // WL hash scratch space: per-block allocation for concurrent hash computation
    // Total size: num_blocks * wl_scratch_per_block
    uint64_t* wl_scratch;
    uint32_t wl_scratch_per_block;  // Size of scratch per block (max_vertex * 2)

    // Per-warp match output buffer
    // Total size: num_warps * matches_per_warp
    DeviceMatch* matches_buffer;
    uint32_t matches_per_warp;  // Number of matches per warp (e.g., 1024)

    // Hash tables for deduplication
    void* canonical_state_map;      // Hash -> StateId
    void* seen_match_hashes;        // Match hash -> bool
    void* seen_event_hashes;        // Event hash -> bool (for deduplication)

    // Event canonicalization mode
    EventCanonicalizationMode event_canon_mode;

    // Online causal/branchial tracking (replaces phased computation)
    OnlineCausalGraph* online_causal;

    // Legacy causal/branchial output (kept for compatibility)
    CausalEdge* causal_edges;
    uint32_t* num_causal_edges;
    BranchialEdge* branchial_edges;
    uint32_t* num_branchial_edges;

    // Edge producer tracking (for causal) - redundant with online_causal but kept for phased fallback
    uint32_t* edge_producer_map;    // Edge -> producing event

    // Vertex counter for fresh allocations
    uint32_t* vertex_counter;

    // Limits
    uint32_t max_steps;
    uint32_t max_states;
    uint32_t max_events;

    // Configuration
    bool transitive_reduction_enabled;
    TaskGranularity task_granularity;
};

// =============================================================================
// Result structure for host download
// =============================================================================
// Note: HostCausalEdge and HostBranchialEdge are defined in gpu_evolution_host.hpp

struct EvolutionResults {
    size_t num_states;
    size_t num_canonical_states;
    size_t num_events;
    size_t num_causal_edges;
    size_t num_branchial_edges;
    size_t num_redundant_edges_skipped;
    // Note: Actual edges available via get_causal_edges() / get_branchial_edges()
};

// =============================================================================
// Match Forwarding Support (GPU implementation matching unified/)
// =============================================================================
//
// Key concepts from unified/:
// - Eager rewriting: MATCH spawns REWRITE immediately, REWRITE spawns MATCH immediately
// - Match storage: Per-state list of matches found (for forwarding to children)
// - Parent tracking: Each state tracks its parent + consumed edges (for ancestor chain)
// - Epoch-based coordination: Determines push vs pull responsibility
// - Pull from ancestors: Child walks ancestor chain, forwarding valid matches
// - Push to children: Parent pushes new matches to registered children
// - Delta matching: Child only matches on produced edges (not full state)

// Global epoch counter for ordering matches and child registrations
// Stored in device memory, accessed atomically

// Match record for storage and forwarding (mirrors unified/MatchRecord)
struct MatchRecord {
    uint16_t rule_index;
    EdgeId matched_edges[MAX_PATTERN_EDGES];
    uint8_t num_edges;
    uint32_t bindings[MAX_VARS];
    uint32_t bound_mask;
    StateId source_state;           // Raw state where match was found
    StateId canonical_source;       // Canonical state ID
    uint64_t source_canonical_hash; // Hash of source state
    uint64_t storage_epoch;         // Epoch when this match was stored

    // Hash for deduplication - uses source_state (raw), not canonical_hash
    // This ensures matches in different raw states don't incorrectly deduplicate
    __device__ uint64_t hash() const {
        uint64_t h = 14695981039346656037ULL;  // FNV offset basis
        constexpr uint64_t FNV_PRIME = 1099511628211ULL;

        h ^= rule_index;
        h *= FNV_PRIME;
        h ^= source_state;
        h *= FNV_PRIME;
        h ^= (source_state >> 16);
        h *= FNV_PRIME;

        for (uint8_t i = 0; i < num_edges; ++i) {
            h ^= matched_edges[i];
            h *= FNV_PRIME;
        }

        h ^= bound_mask;
        h *= FNV_PRIME;

        for (uint8_t i = 0; i < MAX_VARS; ++i) {
            if (bound_mask & (1u << i)) {
                h ^= (static_cast<uint64_t>(i) << 32) | bindings[i];
                h *= FNV_PRIME;
            }
        }

        return h;
    }

    __device__ bool overlaps_edges(const EdgeId* consumed, uint8_t num_consumed) const {
        for (uint8_t i = 0; i < num_edges; ++i) {
            for (uint8_t j = 0; j < num_consumed; ++j) {
                if (matched_edges[i] == consumed[j]) return true;
            }
        }
        return false;
    }
};

// Lock-free list node for per-state match storage
struct MatchNode {
    MatchRecord match;
    MatchNode* next;
};

// Per-state match list head (lock-free linked list)
struct StateMatchList {
    MatchNode* head;  // Atomic pointer to head of list

    __device__ void push(MatchNode* node) {
        MatchNode* old_head;
        do {
            old_head = head;
            node->next = old_head;
        } while (atomicCAS((unsigned long long*)&head,
                           (unsigned long long)old_head,
                           (unsigned long long)node) != (unsigned long long)old_head);
    }

    // Iterate over all matches (read-only traversal safe during concurrent push)
    template<typename Func>
    __device__ void for_each(Func&& fn) const {
        MatchNode* current = head;
        while (current != nullptr) {
            fn(current->match);
            current = current->next;
        }
    }
};

// Child info for push-based match forwarding
struct ChildInfo {
    StateId child_state;
    EdgeId consumed_edges[MAX_PATTERN_EDGES];
    uint8_t num_consumed;
    uint32_t creation_step;
    uint64_t registration_epoch;

    __device__ bool match_overlaps_consumed(const EdgeId* matched, uint8_t num_matched) const {
        for (uint8_t i = 0; i < num_matched; ++i) {
            for (uint8_t j = 0; j < num_consumed; ++j) {
                if (matched[i] == consumed_edges[j]) return true;
            }
        }
        return false;
    }
};

// Lock-free list node for children
struct ChildNode {
    ChildInfo info;
    ChildNode* next;
};

// Per-state children list (for push forwarding)
struct StateChildrenList {
    ChildNode* head;

    __device__ void push(ChildNode* node) {
        ChildNode* old_head;
        do {
            old_head = head;
            node->next = old_head;
        } while (atomicCAS((unsigned long long*)&head,
                           (unsigned long long)old_head,
                           (unsigned long long)node) != (unsigned long long)old_head);
    }

    template<typename Func>
    __device__ void for_each(Func&& fn) const {
        ChildNode* current = head;
        while (current != nullptr) {
            fn(current->info);
            current = current->next;
        }
    }
};

// Parent info for pull-based match forwarding (ancestor chain walking)
struct ParentInfo {
    StateId parent_state;           // INVALID_ID if no parent (initial state)
    EdgeId consumed_edges[MAX_PATTERN_EDGES];
    uint8_t num_consumed;

    __device__ bool has_parent() const {
        return parent_state != INVALID_ID;
    }
};

// Match context passed from REWRITE to child's MATCH task
struct MatchContext {
    StateId parent_state;
    EdgeId consumed_edges[MAX_PATTERN_EDGES];
    uint8_t num_consumed;
    EdgeId produced_edges[MAX_PATTERN_EDGES];
    uint8_t num_produced;

    __device__ bool has_parent() const {
        return parent_state != INVALID_ID;
    }

    __device__ bool edge_was_consumed(EdgeId eid) const {
        for (uint8_t i = 0; i < num_consumed; ++i) {
            if (consumed_edges[i] == eid) return true;
        }
        return false;
    }

    __device__ bool edge_was_produced(EdgeId eid) const {
        for (uint8_t i = 0; i < num_produced; ++i) {
            if (produced_edges[i] == eid) return true;
        }
        return false;
    }
};

// Extended MATCH task with context for match forwarding
struct MatchTaskWithContext {
    StateId state_id;
    uint32_t step;
    MatchContext context;
};

// Extended REWRITE task with match record
struct RewriteTaskWithMatch {
    MatchRecord match;
    uint32_t step;
};

// Configuration for match forwarding
struct MatchForwardingConfig {
    bool enabled;                   // Whether match forwarding is active
    bool batched_matching;          // Batch all matches before spawning REWRITEs (vs eager)
};

// =============================================================================
// HGMatch Task Types and Generic Task Queue
// =============================================================================
// Following HGMatch paper: SCAN, EXPAND, SINK, REWRITE are task types.
// All tasks go into ONE generic task queue. Workers pop tasks and dispatch
// based on task type.

// Task types following HGMatch paper and unified/ implementation:
// SCAN → EXPAND → SINK → REWRITE → (SCAN for new state)
// Causal/branchial edges are computed ONLINE during REWRITE (not as separate tasks)
enum class TaskType : uint8_t {
    SCAN,       // Find initial candidates for first pattern edge, spawn EXPAND tasks
    EXPAND,     // Extend partial match with next pattern edge, spawn EXPAND or SINK
    SINK,       // Process complete match: count it, check limits, spawn REWRITE task
    REWRITE     // Apply rewrite rule: create new state, compute causal/branchial online, spawn SCAN tasks
};

// Generic task that can represent any HGMatch task type
struct Task {
    TaskType type;

    // Common fields
    StateId source_state;
    uint64_t source_hash;
    uint16_t rule_index;
    uint32_t step;

    // Binding state (for EXPAND/SINK/REWRITE)
    uint32_t bindings[MAX_VARS];
    uint32_t bound_mask;

    // Matched edges (in pattern order)
    EdgeId matched_edges[MAX_PATTERN_EDGES];
    uint8_t num_matched;

    // For delta matching: starting position (0xFF if not delta)
    uint8_t delta_start_position;

    // For REWRITE: the complete match to apply
    // (bindings and matched_edges already hold this)

    __device__ bool is_delta() const {
        return delta_start_position != 0xFF;
    }

    __device__ bool contains_edge(EdgeId eid) const {
        for (uint8_t i = 0; i < num_matched; ++i) {
            if (matched_edges[i] == eid) return true;
        }
        return false;
    }

    // Factory functions for creating tasks
    __device__ static Task make_scan(StateId state, uint64_t hash, uint16_t rule, uint32_t step_num) {
        Task t;
        t.type = TaskType::SCAN;
        t.source_state = state;
        t.source_hash = hash;
        t.rule_index = rule;
        t.step = step_num;
        t.bound_mask = 0;
        t.num_matched = 0;
        t.delta_start_position = 0xFF;
        return t;
    }

    __device__ static Task make_expand(
        StateId state, uint64_t hash, uint16_t rule, uint32_t step_num,
        const uint32_t* bind, uint32_t mask,
        const EdgeId* edges, uint8_t n_matched,
        uint8_t delta_pos = 0xFF
    ) {
        Task t;
        t.type = TaskType::EXPAND;
        t.source_state = state;
        t.source_hash = hash;
        t.rule_index = rule;
        t.step = step_num;
        for (uint8_t i = 0; i < MAX_VARS; ++i) t.bindings[i] = bind[i];
        t.bound_mask = mask;
        for (uint8_t i = 0; i < n_matched; ++i) t.matched_edges[i] = edges[i];
        t.num_matched = n_matched;
        t.delta_start_position = delta_pos;
        return t;
    }

    __device__ static Task make_sink(
        StateId state, uint64_t hash, uint16_t rule, uint32_t step_num,
        const uint32_t* bind, uint32_t mask,
        const EdgeId* edges, uint8_t n_matched
    ) {
        Task t;
        t.type = TaskType::SINK;
        t.source_state = state;
        t.source_hash = hash;
        t.rule_index = rule;
        t.step = step_num;
        for (uint8_t i = 0; i < MAX_VARS; ++i) t.bindings[i] = bind[i];
        t.bound_mask = mask;
        for (uint8_t i = 0; i < n_matched; ++i) t.matched_edges[i] = edges[i];
        t.num_matched = n_matched;
        t.delta_start_position = 0xFF;
        return t;
    }

    __device__ static Task make_rewrite(
        StateId state, uint64_t hash, uint16_t rule, uint32_t step_num,
        const uint32_t* bind, uint32_t mask,
        const EdgeId* edges, uint8_t n_matched
    ) {
        Task t;
        t.type = TaskType::REWRITE;
        t.source_state = state;
        t.source_hash = hash;
        t.rule_index = rule;
        t.step = step_num;
        for (uint8_t i = 0; i < MAX_VARS; ++i) t.bindings[i] = bind[i];
        t.bound_mask = mask;
        for (uint8_t i = 0; i < n_matched; ++i) t.matched_edges[i] = edges[i];
        t.num_matched = n_matched;
        t.delta_start_position = 0xFF;
        return t;
    }
};

// Legacy ExpandTask alias for compatibility during transition
using ExpandTask = Task;

// =============================================================================
// Event Signature for ByStateAndEdges mode deduplication
// =============================================================================
// Note: EventCanonicalizationMode enum is defined earlier (before EvolutionContext)

// Event signature for ByStateAndEdges mode deduplication
// Uses WL vertex hashes to compute canonical edge signatures
struct DeviceEventSignatureFull {
    uint64_t canonical_input_hash;
    uint64_t canonical_output_hash;
    uint64_t consumed_signature;    // Hash of consumed edges (using vertex WL hashes)
    uint64_t produced_signature;    // Hash of produced edges (using vertex WL hashes)

    __device__ uint64_t hash() const {
        uint64_t h = 14695981039346656037ULL;  // FNV offset basis
        constexpr uint64_t FNV_PRIME = 1099511628211ULL;
        h ^= canonical_input_hash;
        h *= FNV_PRIME;
        h ^= canonical_output_hash;
        h *= FNV_PRIME;
        h ^= consumed_signature;
        h *= FNV_PRIME;
        h ^= produced_signature;
        h *= FNV_PRIME;
        return h;
    }

    __device__ bool operator==(const DeviceEventSignatureFull& other) const {
        return canonical_input_hash == other.canonical_input_hash &&
               canonical_output_hash == other.canonical_output_hash &&
               consumed_signature == other.consumed_signature &&
               produced_signature == other.produced_signature;
    }
};

}  // namespace hypergraph::gpu
