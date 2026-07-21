#pragma once

#include <cstdint>
#include "hgcommon/core.hpp"

namespace hg_gpu {

using hgcommon::VertexId;
using hgcommon::EdgeId;
using hgcommon::StateId;
using hgcommon::EventId;
using hgcommon::MatchId;
using hgcommon::INVALID_ID;
using RuleId = uint32_t;  // GPU-local width (host engine uses a 16-bit RuleIndex)

// =============================================================================
// Storage layout — Edge, State edge CSR, Pools
// =============================================================================
//
// Edges and vertices live in pre-allocated mega-pools (sized by the auto-
// tuner). State membership is a CSR per-state edge list.
//
//   vertex_pool: uint32_t[MAX_VERTEX_SLOTS]
//     Flat per-edge vertex tuples. Each edge's vertices occupy a contiguous
//     run starting at Edge::vertex_offset. Reused across all edges (edges
//     never share vertex slots — each edge gets its own run allocated from
//     a pool's atomic counter).
//
//   edge_pool: Edge[MAX_EDGES]
//     Metadata per edge. Immutable once written.
//
//   state_edge_slices: StateEdgeSlice[MAX_STATES]
//   state_edge_ids:    EdgeId[MAX_STATE_EDGE_TOTAL]
//     CSR: each state's edges are a sorted-ascending run in state_edge_ids,
//     pointed to by (offset, count) in state_edge_slices. Total memory is
//     linear in the sum of per-state edge counts, not MAX_STATES * MAX_EDGES.
//     Sortedness is preserved by rewrite
//     because new edges (via edge_pool.claim_n) always have IDs greater
//     than any parent edge.

constexpr uint32_t kMaxArity         = hgcommon::MAX_ARITY;
constexpr uint32_t kMaxPatternEdges  = hgcommon::MAX_PATTERN_EDGES;
constexpr uint32_t kMaxVars          = hgcommon::MAX_VARS;
constexpr uint32_t kMaxConsumedBits  = 1024; // per-match consumed bitmap width
constexpr uint32_t kMaxConsumedWords = kMaxConsumedBits / 32;

// WL hash scratch bounds. A single state's canonicalization uses these as
// upper bounds on the per-state vertex and edge count — auto-tuner bumps
// them if a workload needs more.
// Per-thread WL-hash scratch bounds. Each thread needs (edges + verts +
// 2*verts_for_colors) worth of local memory; the neighbour array dominates
// at ~edges*15 entries. Keep tight — these go into local memory (GDDR).
constexpr uint32_t kMaxWlVertices = 64;
constexpr uint32_t kMaxWlEdges    = 64;
constexpr uint32_t kMaxWlRefineIters = 16;

// Event-related device structs. An Event is created per successful rewrite
// and carries the full (input, output, consumed, produced) record.
struct DeviceEvent {
    EventId id              = INVALID_ID;
    EventId canonical_id    = INVALID_ID;  // INVALID_ID when this event is itself canonical
    StateId input_state     = INVALID_ID;
    StateId output_state    = INVALID_ID;
    RuleId  rule            = 0;
    uint32_t step           = 0;
    uint8_t num_consumed    = 0;
    uint8_t num_produced    = 0;
    EdgeId consumed_edges[kMaxPatternEdges] = {INVALID_ID};
    EdgeId produced_edges[kMaxPatternEdges] = {INVALID_ID};
};

// Causal edge: producer event → consumer event via a shared data edge.
// Multiplicity is preserved — the same (from, to) pair appears multiple
// times if the consumer consumed multiple edges the producer created.
struct DeviceCausalEdge {
    EventId from;           // producer event
    EventId to;             // consumer event
    EdgeId  shared_edge;    // the hyperedge that established causality
};

// Branchial edge: two events that share an input state and overlap in at
// least one consumed edge. Canonical orientation: a = min(e1, e2), b = max.
struct DeviceBranchialEdge {
    EventId a;
    EventId b;
    EdgeId  shared_edge;
};

struct Edge {
    uint8_t  arity;          // number of vertex slots occupied
    uint8_t  pad0_[3];
    uint32_t vertex_offset;  // index into vertex_pool for this edge's first vertex
    uint64_t signature;      // precomputed arity-order-signature
    EventId  creator_event;  // event that produced this edge; INVALID_ID for initial
    uint32_t step;           // step at which this edge was created
};
static_assert(sizeof(Edge) == 24, "Edge layout must pack to 24 bytes");

// Per-state edge list descriptor: a compressed-sparse-row layout where
// `state_edge_slices[sid] = {offset, count}` points at a sorted run of EdgeIds
// in `state_edge_ids[]`.
// The flat ids pool is append-only; each rewrite allocates a consecutive
// slice via atomic claim_n. Slices stay sorted because (a) initial-state
// edges are inserted in ascending order, (b) each rewrite appends produced
// edges (whose IDs are guaranteed greater than all parent edge IDs via
// edge_pool.claim_n) at the end of its surviving-parent-edges prefix.
struct StateEdgeSlice {
    uint32_t offset;   // start index into state_edge_ids
    uint32_t count;    // number of EdgeIds in this state's row
};

enum class CanonicalizationMode : uint8_t {
    None = 0,
    Automatic,
    Full,
};

enum class EventCanonicalizationMode : uint8_t {
    None = 0,
    Full,
    Automatic,
};

}  // namespace hg_gpu
