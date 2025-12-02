# v2.3 System Design Specification

## Overview

This document specifies the complete design for v2.3 of the hypergraph rewriting engine.
Key goals:
- Lock-free concurrent execution
- Avoid copying states (track edge membership efficiently)
- Incremental causal/branchial edge computation
- Support multiple levels of canonicalization
- Use uniqueness trees to speed up canonicalization

---

## 1. Canonicalization Levels

### Level 0: No Canonicalization
- Every state is unique based on creation path
- Multiway graph is a tree/forest (from multiple initial states)
- Events only consume edges from their direct ancestor chain
- Causal ordering guaranteed by parent→child structure
- Simplest case, no equivalence tracking needed

### Level 1: State Equivalencing (Hash-Based)
- States with same canonical hash are "equivalent"
- Detects isomorphic states via uniqueness tree hash
- Edges retain distinct identities (E1 in S1 ≠ E4 in S2, even if S1 ≡ S2)
- Evolution continues from canonical representative only (avoids duplicate work)
- Causal relationships stay within each branch's edge lineage
- Branchial: events from same canonical input state with overlapping edge IDs

### Level 2: Full Canonicalization with Edge Correspondence
- States equivalent AND edges are identified across isomorphic states
- Uniqueness tree computes edge correspondence: E1 ↔ E4, E2 ↔ E5, etc.
- Edges grouped into equivalence classes
- Causal edges span across branches via edge correspondence:
  - Event A produces E1, Event B produces E4, E1 ↔ E4
  - Event C consumes E1 → causal edges: A→C AND B→C
- Branchial edges also span branches:
  - Events from equivalent input states consuming corresponding edges are branchial
  - Events can be branchial with themselves (after event canonicalization)
- Self-branchial: E1 ≡ E2 (event canonicalization), E1-E2 branchial → self-loop on canonical event

---

## 2. Core Data Types

### 2.1 Identifiers

All IDs are `uint32_t` (4 billion sufficient, less cache pressure than uint64).
Allocated via atomic fetch_add from global counters.

```cpp
using VertexId = uint32_t;
using EdgeId = uint32_t;
using StateId = uint32_t;
using EventId = uint32_t;
using MatchId = uint32_t;
using EquivClassId = uint32_t;

constexpr uint32_t INVALID_ID = UINT32_MAX;
```

### 2.2 Core Structures

```cpp
struct Edge {
    EdgeId id;
    VertexId* vertices;      // Arena-allocated array
    uint8_t arity;
    EventId creator_event;   // INVALID_ID for initial edges
    uint32_t step;
    std::atomic<EquivClassId> equiv_class;  // Updated when correspondence found
};
// Mutability: Immutable except equiv_class (atomic update)
// Allocation: Arena

struct State {
    StateId id;
    Bitset edges;            // Which edges present (arena-allocated)
    uint32_t step;
    uint64_t canonical_hash; // Computed once via uniqueness tree
    EventId parent_event;    // Event that created this, INVALID_ID for initial
};
// Mutability: Fully immutable after creation
// Allocation: Arena

struct Event {
    EventId id;
    StateId input_state;
    StateId output_state;
    uint16_t rule_index;
    EdgeId* consumed_edges;  // Arena-allocated array
    EdgeId* produced_edges;  // Arena-allocated array
    uint8_t num_consumed;
    uint8_t num_produced;
    VariableBinding binding;
};
// Mutability: Fully immutable after creation
// Allocation: Arena

struct Match {
    MatchId id;
    uint16_t rule_index;
    EdgeId* matched_edges;   // Arena-allocated array
    uint8_t num_edges;
    VariableBinding binding;
    StateId origin_state;    // Where first discovered
};
// Mutability: Fully immutable after creation
// Allocation: Arena

struct VariableBinding {
    VertexId bindings[MAX_VARS];  // Inline array, MAX_VARS = 32
    uint32_t bound_mask;          // Bitmask of which vars are bound
};
// No heap allocation, fixed size
```

### 2.3 Bitset for Edge Membership

```cpp
struct Bitset {
    uint64_t* chunks;     // Arena-allocated
    uint32_t num_chunks;

    bool contains(EdgeId e) const;      // O(1)
    void set(EdgeId e);                 // O(1)
    void clear(EdgeId e);               // O(1)
    size_t count() const;               // O(chunks) or cached
    template<typename F> void for_each(F&& f) const;  // Iterate set bits
};
```

**Size:** E/8 bytes per state where E = max EdgeId.
With 1M edges: 125KB per state.

**Copy-on-Write consideration:** For derived states, most chunks identical to parent.
Could share chunks and only copy on modification. Keep this optimization in mind.

**Creation for derived state:**
1. Allocate chunk array from arena
2. Copy parent's chunks (or share with COW)
3. Clear bits for consumed edges
4. Set bits for produced edges

### 2.4 Edge Equivalence Classes (Level 2 only)

```cpp
struct EdgeEquivalenceClass {
    EquivClassId id;
    EdgeId representative;
    LockFreeList<EventId> producers;   // Events that produced edges in this class
    LockFreeList<EventId> consumers;   // Events that consumed edges in this class
};
```

**Union-Find** for merging equivalence classes when edge correspondence discovered.

---

## 3. Indices and Collections

### 3.1 Append-Only Collections

| Collection | Type | Notes |
|------------|------|-------|
| `edges` | SegmentedArray<Edge> | Random access by EdgeId |
| `states` | SegmentedArray<State> | Random access by StateId |
| `events` | SegmentedArray<Event> | Random access by EventId |
| `matches` | SegmentedArray<Match> | Random access by MatchId |

**SegmentedArray:** Array of fixed-size segments. Avoids reallocation.
New segments allocated from arena when needed (rare).
Lock-free append via atomic count increment.

### 3.2 Indices

| Index | Key → Value | Purpose | Thread Safety |
|-------|-------------|---------|---------------|
| `vertex_to_edges` | VertexId → List<EdgeId> | Pattern matching | Lock-free append |
| `canonical_states` | uint64_t → StateId | State deduplication | Concurrent map |
| `state_matches` | StateId → List<MatchId> | Match lookup | Per-state, set at creation |
| `equiv_classes` | EquivClassId → EdgeEquivalenceClass | Causal tracking (Level 2) | Lock-free |

### 3.3 Causal Tracking (Level 0/1)

Per edge:
```cpp
struct EdgeCausalInfo {
    std::atomic<EventId> producer;           // Set once when edge created
    LockFreeList<EventId> consumers;         // Appended when edge consumed
};
```

### 3.4 Causal Tracking (Level 2)

Per equivalence class (instead of per edge):
```cpp
struct EquivClassCausalInfo {
    LockFreeList<EventId> producers;         // Multiple due to correspondence
    LockFreeList<EventId> consumers;
};
```

### 3.5 Branchial Tracking

Per canonical state:
```cpp
struct StateBranchialInfo {
    LockFreeList<EventId> events_from_here;  // Events with this input state
};
```

---

## 4. Memory Management

### 4.1 Per-Thread Arena Allocator

```cpp
class Arena {
    static constexpr size_t BLOCK_SIZE = 1024 * 1024;  // 1MB blocks

    struct Block {
        Block* next;
        size_t capacity;
        std::atomic<size_t> offset;
        char data[];
    };

    std::atomic<Block*> current_block;

    void* allocate(size_t size, size_t alignment);
    template<typename T, typename... Args> T* create(Args&&...);
};
```

**Per-thread arenas:** Each worker thread has its own arena.
No contention on allocation. Memory freed in bulk at end of evolution phase.

### 4.2 Lock-Free Append List

```cpp
template<typename T>
class LockFreeList {
    struct Node {
        T value;
        std::atomic<Node*> next;
    };
    std::atomic<Node*> head;

    void push(T value);                    // CAS on head
    template<typename F> void for_each(F&& f) const;
};
```

Nodes allocated from arena. Never freed individually.

### 4.3 Concurrent Map (for canonical_states)

Lock-free map supporting:
- `insert_if_absent(key, value)` → (value, was_inserted)
- `lookup(key)` → optional<value>

No delete needed (append-only usage).

Options:
- Split-ordered list (v2 approach, fix memory management with arena)
- Simple lock-free hash map with fixed bucket count + overflow lists

---

## 5. Task Types and Data Flow

### 5.1 MATCH Task

**Input:**
- `state_id: StateId`
- `rule_index: uint16_t`

**Reads:**
- State's edge bitset
- Edges (vertex data)
- `vertex_to_edges` index
- Signature index (for filtering)

**Output:**
- Spawns REWRITE task for each complete match

**Operations:**
1. Get state's edge bitset
2. Use signature index to find candidate edges
3. Build partial matches, extend using vertex_to_edges
4. For each complete match, spawn REWRITE task

**Parallelism:** Read-only access to shared structures. Multiple MATCH tasks concurrent.

### 5.2 REWRITE Task

**Input:**
- `state_id: StateId`
- `match: Match`
- `rule_index: uint16_t`

**Output:**
- New edges, new event, new state
- Causal edges (immediate)
- Possibly spawns MATCH tasks for new state

**Operations:**

```
1. ALLOCATE NEW VERTICES
   For each fresh variable in RHS:
     vertex_id = atomic_fetch_add(next_vertex_id)

2. CREATE NEW EDGES
   For each edge in RHS pattern:
     edge_id = atomic_fetch_add(next_edge_id)
     Substitute binding into pattern vertices
     Allocate edge from arena
     Append to edges collection
     Update vertex_to_edges index (lock-free append)

3. REGISTER PRODUCED EDGES FOR CAUSAL TRACKING
   For each produced edge:
     Set self as producer (atomic store)
     If consumers exist (Level 2 canonicalization race):
       Create causal edges to each consumer

4. REGISTER CONSUMED EDGES
   For each consumed edge:
     Append self to consumers list (lock-free)
     Read producer (atomic load)
     If producer exists:
       Create causal edge (producer → self)

5. CREATE EVENT
   event_id = atomic_fetch_add(next_event_id)
   Allocate event from arena
   Append to events collection

6. CREATE NEW STATE
   state_id = atomic_fetch_add(next_state_id)
   Create bitset: copy parent's, clear consumed, set produced
   Compute canonical hash via uniqueness tree
   Allocate state from arena
   Append to states collection

7. CHECK CANONICAL EQUIVALENCE
   result = canonical_states.insert_if_absent(hash, state_id)
   If was_inserted (new canonical state):
     → Go to step 8
   Else (equivalent state exists):
     → Go to step 9

8. NEW CANONICAL STATE
   Register for branchial tracking:
     Append event to input_state's events_from_here
     Check for branchial relationships with other events there
   Inherit matches from parent:
     For each match in parent's state_matches:
       If all matched_edges in new bitset:
         Add to new state's state_matches
   Spawn MATCH tasks:
     For each rule, spawn MATCH(new_state, rule)
   DONE

9. EQUIVALENT STATE EXISTS (Level 2 only)
   canonical_state = result.value
   Compute edge correspondence via uniqueness tree
   For each (new_edge, canonical_edge) correspondence:
     Merge equivalence classes
     Union producer/consumer sets
     Create cross-branch causal edges from merged sets
   Update branchial tracking with equivalence
   (No new MATCH tasks - canonical representative handles evolution)
   DONE
```

**Parallelism:** Multiple REWRITE tasks concurrent. Contention points:
- Atomic ID allocations (fast, no contention)
- Arena allocation (per-thread, no contention)
- Index appends (lock-free)
- Canonical map insert (lock-free, potential CAS retry)

---

## 6. Uniqueness Tree for Canonicalization

### 6.1 Purpose

The uniqueness tree spans the entire unified hypergraph (all edges across all states).
Used for:
1. Computing state canonical hashes
2. Finding edge correspondence between isomorphic states
3. Potentially speeding up canonicalization (avoid factorial complexity)

### 6.2 Structure

```cpp
struct VertexNeighborhoodData {
    VertexId vertex;
    std::vector<EdgeOccurrence> occurrences;  // All edges containing this vertex
};

struct EdgeOccurrence {
    EdgeId edge_id;
    uint8_t position;    // Position of vertex within edge
    uint8_t arity;       // Total arity of edge
};

class SharedUniquenessTree {
    // Global vertex data - built incrementally as edges added
    ConcurrentMap<VertexId, VertexNeighborhoodData> vertex_data;

    void on_edge_added(EdgeId, const Edge&);
    uint64_t compute_state_hash(const State&, const Edges&) const;
    EdgeCorrespondence find_edge_correspondence(const State& s1, const State& s2) const;
};
```

### 6.3 State Hash Computation

Weisfeiler-Lehman style hashing:

1. Collect vertices present in state (from edges in bitset)
2. For each vertex, compute vertex hash:
   - Based on: incident edges (filtered to state), positions, neighbor degrees
   - Only considers edges present in this state
3. Sort vertex hashes
4. Combine into single state hash (FNV-1a or similar)

**Complexity:** O(V * avg_degree) where V = vertices in state

### 6.4 Edge Correspondence

Given two isomorphic states S1 and S2:

1. Compute canonical vertex ordering for each state
   - Vertices sorted by (hash, degree, vertex_id)
2. Build vertex mapping: S1.vertex[i] ↔ S2.vertex[i]
3. For each edge in S1:
   - Map its vertices to S2's vertex space
   - Find matching edge in S2 with same mapped vertex sequence
4. Return edge mapping: S1.edge → S2.edge

**Complexity:** O(E) where E = edges in state

### 6.5 Speeding Up Canonicalization

**The problem:** Full graph canonicalization is factorial in worst case.

**How uniqueness tree helps:**

1. **Vertex Classification:**
   - Vertices with different hashes CANNOT be mapped to each other
   - Partitions vertices into equivalence classes
   - Only need to try permutations within each class

2. **Incremental Refinement:**
   - WL-style iteration refines vertex classes
   - After k iterations, two vertices in different classes are guaranteed non-equivalent
   - Most graphs stabilize after few iterations

3. **Early Termination:**
   - If hash computation gives different results → states not isomorphic
   - Cheap O(V * E) check before expensive canonicalization

4. **Guided Search:**
   - When exploring permutations, use vertex hashes to prune
   - Only try mapping v1 → v2 if hash(v1) == hash(v2)

**Question:** Can we achieve polynomial-time canonicalization for "most" hypergraphs using the uniqueness tree, reserving factorial only for highly symmetric cases?

### 6.6 Hash Collision Handling

Two non-isomorphic states might have same hash (collision).

**Detection:** After finding candidate equivalent state, verify with full isomorphism check.

**Options for full check:**
1. Brute force (factorial) - only for small states or high symmetry
2. Use uniqueness tree to guide search (much faster in practice)
3. Store canonical form, compare directly

**Trade-off:** Hash collisions are rare. Optimize for common case (hash differs or hash matches and truly isomorphic).

---

## 7. Causal Edge Computation (Detailed)

### 7.1 Level 0/1: Per-Edge Tracking

```cpp
struct EdgeCausalInfo {
    std::atomic<EventId> producer{INVALID_ID};
    LockFreeList<EventId> consumers;
};

// Global array indexed by EdgeId
SegmentedArray<EdgeCausalInfo> edge_causal_info;
```

**On edge production (in REWRITE):**
```cpp
EdgeId e = create_new_edge(...);
edge_causal_info[e].producer.store(current_event_id, memory_order_release);

// Check for consumers (Level 1 with canonicalization - possible race)
edge_causal_info[e].consumers.for_each([&](EventId consumer) {
    create_causal_edge(current_event_id, consumer);
});
```

**On edge consumption (in REWRITE):**
```cpp
for (EdgeId e : consumed_edges) {
    edge_causal_info[e].consumers.push(current_event_id);  // Lock-free append

    EventId producer = edge_causal_info[e].producer.load(memory_order_acquire);
    if (producer != INVALID_ID) {
        create_causal_edge(producer, current_event_id);
    }
}
```

**Rendezvous guarantee:** Both sides write then read. At least one sees the other.

### 7.2 Level 2: Per-Equivalence-Class Tracking

When edges A and B are found to correspond:

```cpp
void merge_edge_equivalence(EdgeId a, EdgeId b) {
    EquivClassId class_a = edges[a].equiv_class.load();
    EquivClassId class_b = edges[b].equiv_class.load();

    if (class_a == class_b) return;  // Already same class

    // Union-find merge (with CAS for thread safety)
    EquivClassId merged = union_find_merge(class_a, class_b);

    // Merge producer/consumer sets
    auto& info_a = equiv_class_causal_info[class_a];
    auto& info_b = equiv_class_causal_info[class_b];
    auto& info_merged = equiv_class_causal_info[merged];

    // Create cross-branch causal edges
    info_a.producers.for_each([&](EventId prod_a) {
        info_b.consumers.for_each([&](EventId cons_b) {
            create_causal_edge(prod_a, cons_b);
        });
    });
    info_b.producers.for_each([&](EventId prod_b) {
        info_a.consumers.for_each([&](EventId cons_a) {
            create_causal_edge(prod_b, cons_a);
        });
    });
}
```

---

## 8. Branchial Edge Computation (Detailed)

### 8.1 Level 0/1: Per-State Tracking

```cpp
struct StateBranchialInfo {
    LockFreeList<EventId> events_from_here;
};

// Map from canonical StateId to branchial info
ConcurrentMap<StateId, StateBranchialInfo> state_branchial_info;
```

**On event creation:**
```cpp
StateId canonical_input = get_canonical_state(event.input_state);
auto& info = state_branchial_info[canonical_input];

// Check for branchial relationships with existing events
info.events_from_here.for_each([&](EventId other_event) {
    if (edges_overlap(event.consumed_edges, events[other_event].consumed_edges)) {
        create_branchial_edge(event.id, other_event);
    }
});

// Add self to list
info.events_from_here.push(event.id);
```

### 8.2 Level 2: With Edge Correspondence

"Edges overlap" now means: edges in same equivalence class.

```cpp
bool edges_overlap_level2(EdgeId* edges1, uint8_t n1, EdgeId* edges2, uint8_t n2) {
    for (int i = 0; i < n1; i++) {
        EquivClassId class1 = get_equiv_class(edges1[i]);
        for (int j = 0; j < n2; j++) {
            EquivClassId class2 = get_equiv_class(edges2[j]);
            if (class1 == class2) return true;
        }
    }
    return false;
}
```

### 8.3 Self-Branchial Events

After event canonicalization, E1 ≡ E2 means:
- They had equivalent input states
- They consumed corresponding edges
- They produced equivalent output states

If E1 and E2 were branchial before canonicalization (same input, overlapping consumed),
then the canonical event is branchial with itself (self-loop).

---

## 9. Match Cascading

### 9.1 Storage

Each state has a list of valid match IDs:

```cpp
// Could be stored in State struct or separate index
ConcurrentMap<StateId, LockFreeList<MatchId>> state_matches;
```

### 9.2 Inheritance

When new state S_child created from S_parent:

```cpp
state_matches[S_child.id] = new LockFreeList<MatchId>();

state_matches[S_parent.id].for_each([&](MatchId mid) {
    const Match& m = matches[mid];

    // Check if all matched edges still present
    bool valid = true;
    for (int i = 0; i < m.num_edges; i++) {
        if (!S_child.edges.contains(m.matched_edges[i])) {
            valid = false;
            break;
        }
    }

    if (valid) {
        state_matches[S_child.id].push(mid);
    }
});
```

### 9.3 New Match Discovery

When MATCH task finds a new match in state S:

```cpp
MatchId mid = create_match(...);

// Add to origin state
state_matches[S.id].push(mid);

// Cascade to descendants (if any exist)
// This requires tracking state children, or we defer to when descendants are created
```

**Decision:** Cascade at descendant creation time (pull model) rather than at match discovery (push model). Simpler, no need to track children.

---

## 10. Open Questions and Decisions

### 10.1 Resolved

- **ID size:** uint32_t (4 billion sufficient, better cache efficiency)
- **Arena:** Per-thread allocation (no contention)
- **Match storage:** Per-state lists (efficient lookup)
- **Causal computation:** Incremental with rendezvous pattern
- **Branchial computation:** Per-canonical-state event lists

### 10.2 To Decide

1. **Bitset COW:** Implement chunked copy-on-write for large states?
   - Depends on expected state count and edge count

2. **Concurrent map implementation:**
   - Split-ordered list (complex but scalable)
   - Fixed buckets + lock-free lists (simpler, may have hot buckets)

3. **Uniqueness tree integration:**
   - How exactly to use it for speeding up canonicalization?
   - Can we avoid factorial complexity in most cases?

4. **Transitive reduction:**
   - Apply during edge creation (incremental) or post-process?
   - Incremental is complex with concurrent updates

5. **Event canonicalization details:**
   - When exactly are events identified as equivalent?
   - How does this interact with self-branchial?

---

## 11. Dynamic Transitive Reduction (Goranci et al.)

### 11.1 The Problem

The causal graph may have redundant edges. If A→B→C and A→C both exist, the direct A→C is redundant (reachability preserved without it). Transitive reduction removes such redundant edges.

v1 does this as batch post-processing. We want incremental/online computation.

### 11.2 Goranci Algorithm Overview

Based on "Dynamic Transitive Reduction" (Goranci et al. 2025).

**Key insight:** For DAGs with vertex-centered insertions (our case - event created with edges to predecessors), O(m) amortized per insertion is achievable.

**Core data structures per vertex u:**
- **Desc_u**: vertices reachable FROM u in snapshot G_u
- **Anc_u**: vertices that can REACH u in snapshot G_u
- Snapshots are nested: G_v1 ⊆ G_v2 ⊆ ... ⊆ G_vn = G

**Redundancy test (Lemma 5.2):** Edge xy is redundant iff ONE of:
1. ∃ vertex z ∉ {x,y} such that x ∈ Anc_z AND y ∈ Desc_z in G_z
2. y has an in-neighbor z ∈ Desc_x \ {x} in G_x
3. x has an out-neighbor z ∈ Anc_y \ {y} in G_y

**Per-edge counters:**
- c(xy): count of vertices z where alternate path exists through z
- t(xy): binary flag for conditions 2 or 3

**Invariant:** xy ∈ reduced graph iff c(xy) = 0 AND t(xy) = 0

### 11.3 Applicability to Our Causal Graph

Our causal graph is a DAG (events ordered by creation/step).

**Insertion pattern:** When event e is created with predecessors P, this is vertex-centered insertion (all new edges incident to new vertex e). Fits Theorem 1.1 exactly.

**For incremental-only (no deletions):**
- Each insertion: O(m) worst-case to rebuild snapshot
- O(m) total amortized over all insertions

**Implementation requirements:**
1. For each event vertex u, maintain Desc_u and Anc_u
2. Maintain counters c(e) and flags t(e) for each edge
3. On insertion: initialize new snapshot, update counters
4. Edge in reduced graph iff c=0 AND t=0

### 11.4 Space Optimization

Virtual adjacency lists with timestamps:
- Single adjacency list, each edge timestamped
- time(v) = last insertion time centered at v
- Virtual G_v = edges with timestamp ≤ time(v)
- Reduces space from O(nm) to O(n²)

---

## 12. Rendezvous Pattern for Concurrent Causal Tracking

### 12.1 The Problem

With concurrent event creation, producer and consumer may race:
- Event E1 produces edge X
- Event E2 consumes edge X
- If E2 checks "who produced X?" before E1 registers, causal edge is missed

This is especially relevant with state canonicalization where "future" events (in causal sense) may be created before "past" events due to branch merging.

### 12.2 The Solution

Each edge has:
```cpp
struct EdgeCausalInfo {
    std::atomic<EventId> producer;      // Written by producer
    LockFreeList<EventId> consumers;    // Appended by consumers
};
```

**On produce (E1 produces X):**
1. WRITE: set X.producer = E1 (atomic store)
2. READ: for each consumer in X.consumers, create causal edge E1→consumer

**On consume (E2 consumes X):**
1. WRITE: append E2 to X.consumers (lock-free)
2. READ: if X.producer exists, create causal edge producer→E2

### 12.3 Why It Works

Both sides write-then-read. Regardless of interleaving:
- If E1 writes producer before E2 reads → E2 sees producer, creates edge
- If E2 writes consumer before E1 reads → E1 sees consumer, creates edge
- At least one side always sees the other

The edge acts as a "rendezvous point" where producer and consumer discover each other regardless of arrival order.

---

## 13. Memory Management Details

### 13.1 Bump Allocator vs Arena

**Bump allocator:** Specific allocation strategy
- Single pointer into contiguous memory
- Allocate = increment pointer ("bump" it)
- O(1) allocation, no individual frees
- When block exhausted, allocate new block

**Arena allocator:** General term
- Any allocator grouping allocations for bulk deallocation
- May use bump allocation internally

Our implementation is a **bump allocator** (arena.hpp uses atomic bump within blocks).

### 13.2 Why Bump Allocators Solve ABA

**ABA problem:** In lock-free algorithms, pointer A is read, changed to B by another thread, then changed back to A. Original thread's CAS succeeds incorrectly.

**With bump allocation:**
- Memory addresses are never reused during an evolution phase
- New allocations always get fresh addresses (pointer only moves forward)
- ABA cannot occur because address A is never recycled back to A

**Cleanup:** At end of evolution phase, entire bump allocator can be reset/freed at once.

### 13.3 Per-Thread Bump Allocators

Each worker thread has its own bump allocator:
- No synchronization needed for allocation
- No contention between threads
- Thread-local bump pointer, thread-local blocks

Global data (edges, states, events) stored in shared segmented arrays, but the individual objects are allocated from thread-local bump allocators.

---

## 14. Lock-Free Hash Map with Resizing

### 14.1 The Problem

v1's ConcurrentHashMap has fixed bucket count (1024 default). Performance degrades with many entries.

v2 attempted split-ordered lists but had memory leak issues.

### 14.2 Solution with Bump Allocators

Split-ordered list approach, but memory management simplified:
- Nodes allocated from bump allocator
- Old bucket arrays retained (not freed) when resizing
- No reclamation needed - bump allocator freed in bulk at phase end

**Resizing:**
1. Allocate new larger bucket array
2. Copy bucket pointers (or initialize new sentinels)
3. CAS to install new array
4. Old array safely ignored (not freed, no ABA risk)

### 14.3 Append-Only Simplification

Our usage is append-only (insert_if_absent, lookup - no delete):
- No need for marked deletion
- No need for hazard pointers or epoch-based reclamation
- Simpler correctness reasoning

---

## 15. Implementation Order

1. **Arena allocator** - foundation for all allocations
2. **Basic types** - IDs, Edge, State, Event, Match, VariableBinding
3. **Bitset** - edge membership
4. **SegmentedArray** - append-only collections
5. **LockFreeList** - append-only linked list
6. **Concurrent map** - for canonical_states
7. **Uniqueness tree** - hash computation and correspondence
8. **MATCH task** - pattern matching
9. **REWRITE task** - rule application
10. **Causal tracking** - Level 0/1
11. **Branchial tracking** - Level 0/1
12. **Level 2 extensions** - edge correspondence, equivalence classes
13. **Testing** - verify against v1 outputs

---

## 16. Current Implementation Status

### ✅ COMPLETED

| Component | File | Status |
|-----------|------|--------|
| Arena allocator | `arena.hpp` | Complete - HeterogeneousArena with bump allocation |
| Basic types | `types.hpp` | Complete - All IDs, Edge, State, Event, Match, VariableBinding |
| Bitset | `bitset.hpp` | Complete - SparseBitset for edge membership |
| SegmentedArray | `segmented_array.hpp` | Complete - Lock-free append-only collection |
| LockFreeList | `lock_free_list.hpp` | Complete - Lock-free singly-linked list |
| Concurrent map | `concurrent_map.hpp` | Complete - Lock-free hash map with insert_if_absent |
| Uniqueness tree | `uniqueness_tree.hpp` | Complete - WL-style hashing, vertex classification, edge correspondence |
| Pattern matching | `pattern.hpp`, `pattern_matcher.hpp`, `signature.hpp` | Complete - Signature-based matching with inverted indices |
| Indices | `index.hpp` | Complete - SignatureIndex, InvertedVertexIndex, PatternMatchingIndex |
| UnifiedHypergraph | `unified_hypergraph.hpp` | Complete - Central storage with indices |
| Rewriter | `rewriter.hpp` | Complete - Apply matches to create new states |
| StateRegistry | `state_registry.hpp` | Complete - State management with match inheritance |
| CausalGraph | `causal_graph.hpp` | Complete - Online causal/branchial computation with rendezvous pattern |
| Transitive Reduction | `transitive_reduction.hpp` | Complete - Online Goranci algorithm for DAG reduction |
| EvolutionEngine | `evolution_engine.hpp` | Complete - Full evolution orchestration with match forwarding |
| ParallelEvolutionEngine | `parallel_evolution.hpp` | Complete - Lock-free parallel evolution with job system |

### ❌ REMAINING / DEFERRED

| Component | Priority | Description |
|-----------|----------|-------------|
| **Level 2 Edge Correspondence** | LOW | UniquenessTree has find_correspondence but not integrated |
| **Edge Equivalence Classes** | LOW | No EdgeEquivalenceClass or union-find for Level 2 |
| **Match Persistence/Resumption** | LOW | Partial matches discarded when no candidates |
| **Online Transitive Reduction** | LOW | Goranci algorithm not integrated into causal graph |

---

## 17. Implementation Plan

### Phase 1: Online Causal/Branchial Computation ✅ COMPLETE

**Goal:** Compute causal and branchial edges incrementally as events are created.

**Implemented in:** `causal_graph.hpp`, `types.hpp`, `unified_hypergraph.hpp`, `rewriter.hpp`

**Design Approach:**

1. **Rendezvous Pattern for Causal Edges:**
   - When edge is produced: WRITE producer, READ consumers, create causal edges to any consumers
   - When edge is consumed: WRITE to consumers list, READ producer, create causal edge from producer if exists
   - Guarantees: At least one side sees the other, no edges missed despite concurrency

2. **Per-Edge Tracking:**
   - `EdgeCausalInfo`: atomic producer EventId, consumers stored in parallel SegmentedArray
   - `edge_consumers_`: SegmentedArray<LockFreeList<EventId>> for O(1) append
   - Memory: O(total_edges) + O(causal_relationships)

3. **Branchial Detection:**
   - Per-state event lists track all events originating from each state
   - When new event created, check overlap with existing events from same input state
   - Only events with overlapping consumed edges create branchial edges

4. **Integration with Rewriter:**
   - After event creation, register produced edges as having this event as producer
   - Register consumed edges with this event as consumer (triggers causal edge creation)
   - Register for branchial tracking (checks overlap with sibling events)

**Tests:** 16 tests in `test_v2_3_causal_branchial.cpp` covering:
- CausalGraph unit tests (construction, producer/consumer, rendezvous, multiple consumers)
- Rewriter integration tests (single rewrite, chained causal chain, branchial detection)

**Complexity Analysis:**
- Setting producer: O(1) atomic CAS + O(consumers) for edge creation
- Adding consumer: O(1) lock-free append + O(1) for edge creation
- Branchial check: O(events_from_state × consumed_edges²)

---

### Phase 2: Uniqueness Tree Integration ✅ COMPLETE

**Goal:** Replace simple hash with proper canonical hashing.

**Implemented:**

1. **Edge Accessors in UnifiedHypergraph:**
   - `edge_vertices(EdgeId)` - Returns vertex array pointer
   - `edge_arity(EdgeId)` - Returns edge arity
   - `EdgeVertexAccessor` / `EdgeArityAccessor` - Indexed access classes

2. **Canonical Hash Computation:**
   - `compute_canonical_hash(SparseBitset)` - Uses UniquenessTree for isomorphism-invariant hash
   - `compute_canonical_info(SparseBitset)` - Returns full StateCanonicalInfo with vertex classes

3. **Integration with Rewriter:**
   - `rewriter.hpp:apply()` now calls `compute_canonical_hash()` instead of simple FNV hash
   - States created during rewriting have proper isomorphism-invariant hashes

**Tests:** 6 new tests in `test_v2_3_causal_branchial.cpp`:
- Empty state handling
- Single edge hashing
- Isomorphic states (same hash)
- Non-isomorphic states (different hash)
- Self-loop isomorphism
- Vertex equivalence class verification

**Complexity Analysis:**
- UniquenessTree compute: O(V² × E) per WL iteration, typically O(V² × E × depth)
- Memory: O(V + E) temporary arrays
- vs Simple FNV: O(E) - but doesn't handle isomorphism

### Phase 3: Transitive Reduction ✅ COMPLETE

**Goal:** Online transitive reduction for DAG causal graphs.

**Implemented in:** `transitive_reduction.hpp`

**Design Approach:**

1. **Per-Vertex Snapshot Data:**
   - `VertexSnapshot`: Per-event data including:
     - `desc`: Set of vertices reachable FROM this vertex
     - `anc`: Set of vertices that can REACH this vertex
     - `out_neighbors`: Direct successors
     - `in_neighbors`: Direct predecessors

2. **Redundancy Detection:**
   - Edge x→y is redundant if there exists an alternate path x→...→y
   - Checked via:
     - y already in x's descendant set (path exists before adding direct edge)
     - Some in-neighbor of y is reachable from x (path x→z→y exists)
     - Intersection of x's descendants with y's ancestors (general alternate path)

3. **Algorithm Flow:**
   - On vertex insertion with predecessors:
     1. Update adjacency lists
     2. Update ancestor sets (BEFORE redundancy checking)
     3. Check redundancy for each incoming edge
     4. Update descendant sets (AFTER redundancy checking)
   - Propagate descendant information up through ancestors

4. **Thread Safety:**
   - Per-vertex mutex locks
   - Consistent lock ordering to avoid deadlock
   - Global lock for edge storage growth

**Tests:** 17 tests in `test_v2_3_transitive_reduction.cpp` covering:
- Empty graph, single edge, linear chains
- Triangle/diamond patterns with redundancy detection
- Multiple redundant edges
- Parallel branches (no false redundancy)
- Fork-join patterns
- Complex DAGs
- Statistics tracking
- Edge ID preservation

**Complexity Analysis:**
- Redundancy check: O(|desc_x| + |in_neighbors_y|) per edge
- Descendant propagation: O(|V|) amortized over all insertions
- Space: O(V²) for Anc/Desc sets (worst case, typically much smaller)

### Phase 4: Evolution Engine ✅ COMPLETE

**Goal:** Full evolution orchestration with match forwarding and determinism.

**Implemented in:** `evolution_engine.hpp`

**Key Components:**

1. **MatchRecord:**
   - Stores rule index, matched edges, binding, source state
   - Hash function for deduplication
   - Max 16 edges per pattern (MAX_PATTERN_EDGES)

2. **StateMatchCache:**
   - Per-state storage of valid matches
   - Tracks deleted edges for match invalidation
   - `is_match_valid()` checks if match overlaps deleted edges

3. **EvolutionStats:**
   - Atomic counters for tracking:
     - `matches_found`, `matches_forwarded`, `matches_invalidated`
     - `new_matches_discovered`, `full_pattern_matches`
     - `states_created`, `events_created`

4. **Match Forwarding Algorithm:**
   - **Full pattern matching** only on initial states
   - **Forward valid matches** from parent to child states:
     - Inherits all parent matches
     - Invalidates matches overlapping consumed edges
     - Discovers new matches from newly created edges only
   - **Deterministic ordering:** Sorted by (rule_index, edge_ids)

5. **Integration:**
   - Uses CausalGraph for online causal/branchial tracking
   - Uses OnlineTransitiveReduction for DAG compression
   - Uses UniquenessTree for canonical hashing via UnifiedHypergraph

**Tests:** 14 tests in `test_v2_3_evolution_engine.cpp`:
- Basic evolution (1-2 steps)
- No-match rules
- Two-edge rules with chains
- Determinism tests (10 runs each)
- Match forwarding statistics
- Causal/branchial tracking
- Multi-rule evolution
- Max states/events limits

**DeterminismFuzzing:** 18 tests in `test_v2_3_determinism_fuzzing.cpp`:
- All test cases run 50-100 times
- Tracks 7 metrics: states, events, causal, branchial, vertex IDs, edge IDs, event IDs
- All tests pass with 100% determinism (1 unique value per metric)

### Phase 5: Parallel Evolution ✅ COMPLETE

**Goal:** Lock-free parallel evolution with deterministic results.

**Implemented in:** `parallel_evolution.hpp`

**Key Components:**

1. **ParallelEvolutionEngine:**
   - Uses job_system for concurrent match application
   - Lock-free result collection via LockFreeList
   - Deterministic state selection (smallest raw_state per canonical)

2. **Determinism Guarantees:**
   - **Canonical-based match deduplication:** Hash uses canonical_source + binding (not raw StateId + EdgeIds)
   - **Deterministic raw state selection:** When multiple raw states map to same canonical, pick smallest StateId
   - **Branchial edge deduplication:** ConcurrentMap with ordered pair key prevents duplicates

3. **Thread Safety:**
   - ConcurrentMap for match hash deduplication
   - ConcurrentMap for branchial pair deduplication
   - LockFreeList for step result collection
   - Atomic counters for statistics

4. **Race Condition Fixes (Dec 2025):**
   - Branchial edges: Both threads can detect overlap, deduplicate via `seen_branchial_pairs_`
   - Match deduplication: Use canonical state ID, not raw state ID
   - Result collection: Use std::map to deterministically select smallest raw_state per canonical

**Tests:** All 18 V2_3_DeterminismFuzzingTest tests pass with 100% determinism across 30-100 runs.

### Phase 6: Level 2 Extensions (Deferred)

- Edge equivalence classes with union-find
- Cross-branch causal edges via correspondence
- Event canonicalization

---

## 18. Key Design Principles (v1 Improvements)

This section summarizes the core improvements v2.3 aims to achieve over v1.

### 18.1 No Edge Copying

**v1 Problem:** States copy all their edges when created. Memory grows as O(states × edges_per_state).

**v2.3 Solution:** Single unified hypergraph stores ALL edges. States only track membership via bitsets. Memory is O(total_edges + states × bitset_size).

### 18.2 Match Forwarding (Incremental Pattern Matching)

**v1 Problem:** Full pattern matching on every new state.

**v2.3 Solution:**
1. **Full pattern matching only on initial states**
2. **Eagerly begin rewriting** as soon as matches are found (while matching continues)
3. **Forward matches from parent to child states:**
   - All parent matches potentially valid in child
   - **Only invalidate** matches that overlap deleted (consumed) edges
   - Valid inherited matches don't require re-matching
4. **Minimal work on new states:**
   - Only discover NEW matches induced by newly ADDED edges
   - Use signature index to find patterns that could match new edges
   - Extend partial matches from existing edge context

### 18.3 Global Uniqueness Tree

**v1 Problem:** Uniqueness tree built per-state for hashing. Canonicalization is O(V!) worst case.

**v2.3 Solution:**
1. Single SharedUniquenessTree spans ALL states
2. Vertex neighborhood data shared across states (computed once per vertex)
3. State hash = filtered view of global tree (only edges in state's bitset)
4. **Canonicalization speedup:** Vertex classification from tree prunes permutation search
5. Benchmark old canonicalization vs uniqueness-tree-assisted to measure improvement

### 18.4 Online Causal/Branchial Computation

**v1 Problem:** Causal and branchial edges computed as post-processing phase.

**v2.3 Solution:**
1. Compute incrementally as events are created (see Sections 7, 8)
2. Rendezvous pattern ensures no edges missed despite concurrency
3. Dynamic transitive reduction (Goranci algorithm) runs online as edges added

### 18.5 Thread Safety Requirements

All shared data structures must be thread-safe:
- Lock-free append for collections
- Atomic updates for IDs and flags
- Per-thread arenas eliminate allocation contention
- Rendezvous pattern for producer/consumer synchronization

**Verification:** Run tests with sanitizers enabled:
- ThreadSanitizer (TSAN) for race detection
- AddressSanitizer (ASAN) for memory errors
- UndefinedBehaviorSanitizer (UBSAN) for undefined behavior

**Sanitizer configuration added to top-level CMakeLists.txt:**
```bash
cmake .. -DENABLE_ASAN=ON  # AddressSanitizer
cmake .. -DENABLE_TSAN=ON  # ThreadSanitizer
cmake .. -DENABLE_UBSAN=ON # UndefinedBehaviorSanitizer
```

**Status:** All 44 v2.3 tests pass with ASAN enabled (no memory errors detected).

---

## 19. Validation Against v1

### 19.1 Correctness Criteria

For identical inputs (rules + initial states + step count), v2.3 must produce:
- **Same number of canonical states** as v1
- **Same number of events** as v1
- **Same causal graph structure** (after transitive reduction)
- **Same branchial graph structure**

### 19.2 Comparison Test Suite

Create tests that:
1. Run v1 evolution, capture (state_count, event_count, causal_edges, branchial_edges)
2. Run v2.3 evolution with same parameters
3. Assert all counts match
4. Optionally verify state hashes match (accounting for ID differences)

Test cases should cover:
- Simple rules (e.g., `{{x,y},{y,z}} -> {{x,z}}`)
- Fresh vertex creation
- Self-loops and unary edges
- Multiple rules
- Multiple initial states
- Various step counts (1, 2, 5, 10)

### 19.3 Performance Comparison

Benchmark v2.3 vs v1 for:
- Total evolution time
- Memory usage (peak RSS)
- Pattern matching time
- State creation time

---

## 20. Benchmarking Plan

### 20.1 Canonicalization Comparison

Use existing benchmark framework (`benchmarks/canonicalization_benchmarks.cpp` and `benchmarks/uniqueness_tree_benchmarks.cpp`) to compare:

1. **Old method (v1 Canonicalizer):**
   - Full permutation-based canonicalization
   - Measure time vs edge count, arity, symmetry

2. **Uniqueness tree for isomorphism only:**
   - Just compute hash for equivalence checking
   - Already benchmarked

3. **Uniqueness tree for canonicalization speedup:**
   - Use vertex classification to prune permutation search
   - Compare against (1) for same graphs
   - Measure speedup factor

New benchmarks to add:
```cpp
BENCHMARK(uniqueness_assisted_canonicalization_by_edge_count, "Canonicalization with UT pruning")
BENCHMARK(canonicalization_comparison, "Old vs UT-assisted canonicalization")
```

### 20.2 Evolution Benchmarks

Compare v1 vs v2.3 full evolution:
- Time to reach N states
- Memory usage at N states
- Throughput (events/second)

---

## 21. Test Status

**All v2.3 tests pass** (73 tests, ASAN verified):

| Test Suite | Test Count | Description |
|------------|------------|-------------|
| V2_3_Signature | 6 | Edge signature computation and compatibility |
| V2_3_PatternMatching | 6 | Pattern matching with various patterns |
| V2_3_Index | 4 | SignatureIndex and InvertedVertexIndex |
| V2_3_UnifiedHypergraph | 3 | Edge/state creation, vertex allocation |
| V2_3_Rewriter | 8 | Rewriting with causal/branchial tracking |
| V2_3_Integration | 3 | End-to-end evolution tests (updated for canonical deduplication) |
| V2_3_CausalGraph | 8 | CausalGraph unit tests, rendezvous pattern |
| V2_3_UniquenessTree | 6 | Canonical hash, isomorphism, vertex classes |
| V1_V2_3_Comparison | 6 | v1 vs v2.3 behavior comparison |
| TransitiveReductionTest | 17 | Online transitive reduction algorithm |
| V2_3_EvolutionEngine | 14 | Evolution engine with match forwarding |
| V2_3_DeterminismFuzzing | 18 | Comprehensive determinism testing (30-100 runs per test) |
| V2_3_ParallelEvolution | 4 | Parallel evolution with job system |
| V1_V2_3_CountComparisonTest | 8 | Exact state/event/causal/branchial count matching |
| V2_3_Level2 | 4 | Level 2 edge correspondence and event canonicalization tests |

**Completed:**
- ✅ Causal edge computation tests
- ✅ Branchial edge computation tests
- ✅ Uniqueness tree integration tests
- ✅ ASAN test runs (pass)
- ✅ Transitive reduction tests
- ✅ v1 vs v2.3 comparison tests
- ✅ EvolutionEngine integration tests
- ✅ Match forwarding tests
- ✅ Determinism fuzzing tests (all tests pass with 100% determinism)
- ✅ **v1 vs v2.3 count comparison tests** - Exact match of state/event counts
- ✅ **Level 2 edge correspondence** - Edge equivalence classes via union-find

**Recent Fixes (Dec 2025):**
1. **Vertex counter initialization**: Fixed `create_initial_state()` to properly initialize the vertex counter, preventing self-loop bugs where fresh vertices collided with initial vertex IDs.
2. **Event counting for duplicate states**: Fixed `apply_match()` to return canonical state IDs for ALL successful rewrites (including duplicates), ensuring events from duplicate states are counted correctly.
3. **Exact canonicalization**: Changed from WL-style polynomial hashing to exact canonicalization using v1's `Canonicalizer`, ensuring isomorphic states are correctly identified.
4. **Level 2 edge correspondence**: Implemented lock-free union-find for edge equivalence classes. When duplicate states are detected, edge correspondence is computed and edges are merged into equivalence classes.
5. **Parallel evolution determinism** (Dec 2 2025):
   - Fixed branchial edge race condition: Added `seen_branchial_pairs_` ConcurrentMap for deduplication
   - Fixed raw state selection: Use `std::map<canonical, raw>` to pick smallest raw_state ID per canonical
   - Fixed match deduplication: Use `canonical_source` + binding in hash, not raw StateId + EdgeIds
   - All 18 determinism fuzzing tests now pass with 100% determinism

**Level 2 Implementation Status:**
- ✅ `LockFreeUnionFind` - Thread-safe union-find with path compression
- ✅ `EdgeEquivalenceManager` - Manages edge equivalence with producer/consumer tracking
- ✅ `register_edge_correspondence()` - Computes edge mapping between isomorphic states
- ✅ `enable_level2()` - API to enable Level 2 canonicalization
- ✅ `EventCanonicalizer` - Computes canonical hashes for events based on edge equivalence
- 🔲 Cross-branch causal edges via edge equivalence (infrastructure in place, needs integration)

**Next tests needed:**
- **TSAN test runs** - ThreadSanitizer verification for parallel evolution
- Event canonicalization tests (Level 2)
- Cross-branch causal edge tests (Level 2)

