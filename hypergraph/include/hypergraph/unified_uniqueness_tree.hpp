#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "types.hpp"
#include "bitset.hpp"
#include "arena.hpp"
#include "segmented_array.hpp"
#include "lock_free_list.hpp"

namespace hypergraph {

// =============================================================================
// UnifiedUniquenessTree: Gorard-style Uniqueness Trees for Unified Hypergraph
// =============================================================================
//
// Based on Gorard (2016) "Uniqueness Trees: A Possible Polynomial Approach to
// the Graph Isomorphism Problem" (arXiv:1606.06399).
//
// This is a TRUE uniqueness tree implementation (not WL hashing):
// - Each vertex has a BFS tree rooted at it
// - subtree_hash computed bottom-up from actual tree structure
// - Vertices with identical subtree_hash are automorphically equivalent
//
// Key difference from WL hashing:
// - WL: Iterative message passing, all vertices update simultaneously
// - Uniqueness trees: BFS tree structure, hash computed bottom-up from leaves
//
// Uses arena allocation patterns from unified/:
// - ArenaVector for temporary storage during tree construction
// - No std::vector in hot paths
// - Tree nodes computed on-the-fly, not stored permanently
//
// Note: Common types (EdgeOccurrence, VertexHashCache, EdgeCorrespondence,
// EventSignature, FNV constants) are defined in types.hpp

// =============================================================================
// UnifiedUniquenessTree: The main class
// =============================================================================

class UnifiedUniquenessTree {
public:
    static constexpr uint32_t MAX_TREE_DEPTH = 100;

    explicit UnifiedUniquenessTree(ConcurrentHeterogeneousArena* arena)
        : arena_(arena) {}

    // Abort flag for early termination of long-running hash computations
    void set_abort_flag(std::atomic<bool>* flag) { abort_flag_ = flag; }
    bool should_abort() const {
        return abort_flag_ && abort_flag_->load(std::memory_order_relaxed);
    }

    // =========================================================================
    // Edge Registration (called when edges are added to unified hypergraph)
    // =========================================================================

    void register_edge(EdgeId edge_id, const VertexId* vertices, uint8_t arity) {
        // Store edge data for adjacency lookups (grows as needed)
        if (edge_id >= edge_vertices_.size()) {
            edge_vertices_.get_or_default(edge_id, *arena_);
            edge_arities_.get_or_default(edge_id, *arena_);
        }

        // Copy vertices to arena-allocated storage
        VertexId* verts = arena_->allocate_array<VertexId>(arity);
        for (uint8_t i = 0; i < arity; ++i) {
            verts[i] = vertices[i];
        }
        edge_vertices_[edge_id] = verts;
        edge_arities_[edge_id] = arity;

        // Record edge occurrence for each vertex
        for (uint8_t i = 0; i < arity; ++i) {
            VertexId v = vertices[i];
            EdgeOccurrence occ(edge_id, i, arity);
            vertex_occurrences_.get_or_default(v, *arena_).push(occ, *arena_);
        }
    }

    // =========================================================================
    // State Hash Computation
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_state_hash(
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        auto [hash, cache] = compute_state_hash_with_cache(state_edges, edge_vertices, edge_arities);
        return hash;
    }

    template<typename EdgeAccessor, typename ArityAccessor>
    std::pair<uint64_t, VertexHashCache> compute_state_hash_with_cache(
        const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        VertexHashCache cache;

        if (state_edges.empty()) {
            return {0, cache};
        }

        // Collect vertices and track min/max for dense index optimization
        ArenaVector<VertexId> vertices(*arena_);
        VertexId max_vertex = 0;
        VertexId min_vertex = UINT32_MAX;

        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                vertices.push_back(verts[i]);
                if (verts[i] > max_vertex) max_vertex = verts[i];
                if (verts[i] < min_vertex) min_vertex = verts[i];
            }
        });

        // Remove duplicates
        std::sort(vertices.begin(), vertices.end());
        auto new_end = std::unique(vertices.begin(), vertices.end());
        vertices.resize(new_end - vertices.begin());

        if (vertices.empty()) {
            return {0, cache};
        }

        const size_t num_vertices = vertices.size();

        // Always use sparse index for simplicity and predictable behavior
        // Dense index optimization was removed as an unprincipled heuristic
        const bool use_dense_index = false;
        ArenaVector<size_t> dense_index(*arena_);  // Empty, unused

        std::unordered_map<VertexId, size_t> sparse_index;
        sparse_index.reserve(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            sparse_index[vertices[i]] = i;
        }

        // Lambda for vertex lookup
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            return sparse_index[v];
        };

        // Count total edge occurrences for pre-allocation
        size_t total_occurrences = 0;
        state_edges.for_each([&](EdgeId eid) {
            total_occurrences += edge_arities[eid];
        });

        // Arena-allocate flat adjacency structure
        ArenaVector<std::pair<EdgeId, uint8_t>> adj_data(*arena_, total_occurrences);
        ArenaVector<uint32_t> adj_counts(*arena_, num_vertices);
        adj_counts.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; ++i) adj_counts[i] = 0;

        // First pass: count occurrences per vertex
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                adj_counts[idx]++;
            }
        });

        // Compute offsets (prefix sum)
        ArenaVector<uint32_t> adj_offsets(*arena_, num_vertices + 1);
        adj_offsets.resize(num_vertices + 1);
        adj_offsets[0] = 0;
        for (size_t i = 0; i < num_vertices; ++i) {
            adj_offsets[i + 1] = adj_offsets[i] + adj_counts[i];
        }

        // Reset counts for second pass
        for (size_t i = 0; i < num_vertices; ++i) adj_counts[i] = 0;

        // Second pass: fill adjacency data
        adj_data.resize(total_occurrences);
        state_edges.for_each([&](EdgeId eid) {
            uint8_t arity = edge_arities[eid];
            const VertexId* verts = edge_vertices[eid];
            for (uint8_t i = 0; i < arity; ++i) {
                size_t idx = get_vertex_idx(verts[i]);
                size_t pos = adj_offsets[idx] + adj_counts[idx];
                adj_data[pos] = {eid, i};
                adj_counts[idx]++;
            }
        });

        // Allocate cache arrays
        cache.capacity = static_cast<uint32_t>(num_vertices);
        cache.vertices = arena_->allocate_array<VertexId>(cache.capacity);
        cache.hashes = arena_->allocate_array<uint64_t>(cache.capacity);
        cache.count = 0;

        // Compute uniqueness tree hash for each vertex
        ArenaVector<uint64_t> tree_hashes(*arena_, num_vertices);

        // Reusable buffers for tree hash computation - allocated once for all vertices
        TreeHashBuffers buffers(*arena_);
        buffers.init_if_needed(*arena_, MAX_TREE_DEPTH);

        // Shared visited set (cleared during DFS backtracking)
        SparseBitset visited;

        for (size_t vi = 0; vi < num_vertices; ++vi) {
            if (should_abort()) throw AbortedException{};
            VertexId root = vertices[vi];
            uint64_t tree_hash = compute_tree_hash_flat(
                root, min_vertex, use_dense_index, dense_index, sparse_index,
                visited, buffers,
                adj_data, adj_offsets, state_edges, edge_vertices, edge_arities);
            tree_hashes.push_back(tree_hash);
            cache.insert(root, tree_hash);
        }

        // Combine tree hashes for state hash (sorted for canonical form)
        insertion_sort(tree_hashes.begin(), tree_hashes.end());

        uint64_t state_hash = FNV_OFFSET;
        state_hash = fnv_hash(state_hash, tree_hashes.size());
        for (uint64_t h : tree_hashes) {
            state_hash = fnv_hash(state_hash, h);
        }

        return {state_hash, cache};
    }

    // =========================================================================
    // Edge Correspondence
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    EdgeCorrespondence find_edge_correspondence(
        const SparseBitset& state1_edges,
        const SparseBitset& state2_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        EdgeCorrespondence result;

        auto [hash1, cache1] = compute_state_hash_with_cache(state1_edges, edge_vertices, edge_arities);
        auto [hash2, cache2] = compute_state_hash_with_cache(state2_edges, edge_vertices, edge_arities);

        if (hash1 != hash2) {
            return result;  // Not isomorphic
        }

        // Build edge signature -> edge list map for state2
        std::unordered_map<uint64_t, ArenaVector<EdgeId>> edge2_by_sig;

        state2_edges.for_each([&](EdgeId eid) {
            uint64_t sig = compute_edge_signature(eid, cache2, edge_vertices, edge_arities);
            auto it = edge2_by_sig.find(sig);
            if (it == edge2_by_sig.end()) {
                it = edge2_by_sig.emplace(sig, ArenaVector<EdgeId>(*arena_)).first;
            }
            it->second.push_back(eid);
        });

        // Collect edges from state1
        ArenaVector<EdgeId> edges1(*arena_);
        state1_edges.for_each([&](EdgeId eid) {
            edges1.push_back(eid);
        });

        // Count edges in state2
        size_t edge2_count = 0;
        state2_edges.for_each([&](EdgeId) { ++edge2_count; });

        if (edges1.size() != edge2_count) {
            return result;
        }

        // Allocate result arrays
        result.count = static_cast<uint32_t>(edges1.size());
        result.state1_edges = arena_->allocate_array<EdgeId>(result.count);
        result.state2_edges = arena_->allocate_array<EdgeId>(result.count);

        // Track which edges from state2 have been used
        std::unordered_map<uint64_t, size_t> sig_next_idx;

        for (size_t i = 0; i < edges1.size(); ++i) {
            EdgeId e1 = edges1[i];
            uint64_t sig = compute_edge_signature(e1, cache1, edge_vertices, edge_arities);

            auto e2_it = edge2_by_sig.find(sig);
            if (e2_it == edge2_by_sig.end()) {
                result.valid = false;
                return result;
            }

            size_t& idx = sig_next_idx[sig];
            if (idx >= e2_it->second.size()) {
                result.valid = false;
                return result;
            }

            result.state1_edges[i] = e1;
            result.state2_edges[i] = e2_it->second[idx];
            ++idx;
        }

        result.valid = true;
        return result;
    }

    // =========================================================================
    // Event Signature Computation
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    EventSignature compute_event_signature(
        uint64_t input_state_hash,
        uint64_t output_state_hash,
        const EdgeId* consumed_edges, uint8_t num_consumed,
        const EdgeId* produced_edges, uint8_t num_produced,
        const VertexHashCache& input_cache,
        const VertexHashCache& output_cache,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        EventSignature sig;
        sig.input_state_hash = input_state_hash;
        sig.output_state_hash = output_state_hash;

        sig.consumed_edges_sig = compute_edge_set_signature(
            consumed_edges, num_consumed, input_cache, edge_vertices, edge_arities);
        sig.produced_edges_sig = compute_edge_set_signature(
            produced_edges, num_produced, output_cache, edge_vertices, edge_arities);

        return sig;
    }

private:
    ConcurrentHeterogeneousArena* arena_;
    std::atomic<bool>* abort_flag_{nullptr};

    // Per-vertex edge occurrences (for global edge registration)
    SegmentedArray<LockFreeList<EdgeOccurrence>> vertex_occurrences_;

    // Edge data (for adjacency lookups)
    SegmentedArray<VertexId*> edge_vertices_;
    SegmentedArray<uint8_t> edge_arities_;

    // =========================================================================
    // Tree Hash Computation (the core algorithm) - ITERATIVE VERSION
    // =========================================================================
    //
    // Computes the uniqueness tree hash for a vertex without storing the tree.
    // Uses an explicit stack instead of recursion to avoid per-call allocations.
    //
    // The hash captures:
    // 1. Vertex level in tree
    // 2. Edge occurrences (how this vertex connects to parent)
    // 3. Child hashes (computed iteratively via explicit stack)
    //

    // Stack frame for iterative DFS
    struct DFSFrame {
        VertexId vertex;
        uint32_t level;
        uint32_t child_idx;           // Which child we're currently processing
        uint32_t num_children;        // Total number of children
        uint64_t partial_hash;        // Hash accumulated so far (for own_positions)
        size_t own_positions_count;   // Number of own positions (already hashed)
        // Child info is stored separately in parallel arrays to avoid allocations per frame
    };

    // Child descriptor for DFS
    struct ChildInfo {
        VertexId vertex;
        bool is_unique;
        uint8_t num_occurrences;           // Number of occurrences (edge connections)
        uint16_t occurrence_positions[8];  // Sorted (arity << 8) | position values
    };

    // Reusable buffers for tree hash computation - allocated once per state hash
    struct TreeHashBuffers {
        ArenaVector<DFSFrame> stack;
        ArenaVector<ArenaVector<ChildInfo>> children_by_level;
        ArenaVector<ArenaVector<uint64_t>> child_hashes_by_level;
        ArenaVector<uint8_t> scratch_positions;
        ArenaVector<uint16_t> scratch_occ_positions;
        ArenaVector<std::pair<VertexId, std::pair<uint8_t, uint8_t>>> adj_vertices;
        bool initialized;

        explicit TreeHashBuffers(ConcurrentHeterogeneousArena& arena)
            : stack(arena)
            , children_by_level(arena)
            , child_hashes_by_level(arena)
            , scratch_positions(arena)
            , scratch_occ_positions(arena)
            , adj_vertices(arena)
            , initialized(false)
        {}

        void init_if_needed(ConcurrentHeterogeneousArena& arena, uint32_t max_depth) {
            if (initialized) return;
            stack.reserve(max_depth + 1);
            children_by_level.reserve(max_depth + 1);
            child_hashes_by_level.reserve(max_depth + 1);
            for (uint32_t i = 0; i <= max_depth; ++i) {
                children_by_level.push_back(ArenaVector<ChildInfo>(arena));
                child_hashes_by_level.push_back(ArenaVector<uint64_t>(arena));
            }
            initialized = true;
        }

        void reset_for_new_tree() {
            stack.clear();
            // children_by_level and child_hashes_by_level are cleared per-level during traversal
            scratch_positions.clear();
            scratch_occ_positions.clear();
            adj_vertices.clear();
        }
    };

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_tree_hash(
        VertexId root,
        const SparseBitset& state_edges,
        const std::unordered_map<VertexId, ArenaVector<std::pair<EdgeId, uint8_t>>>& adjacency,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        // Use SparseBitset for visited tracking with arena allocation
        SparseBitset visited;

        // Explicit stack - pre-allocate for typical tree depth
        ArenaVector<DFSFrame> stack(*arena_);
        stack.reserve(MAX_TREE_DEPTH + 1);

        // Per-frame children storage (indexed by stack depth)
        // Each level can have its own children list
        ArenaVector<ArenaVector<ChildInfo>> children_by_level(*arena_);
        children_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            children_by_level.push_back(ArenaVector<ChildInfo>(*arena_));
        }

        // Per-frame child hashes storage (indexed by stack depth)
        ArenaVector<ArenaVector<uint64_t>> child_hashes_by_level(*arena_);
        child_hashes_by_level.reserve(MAX_TREE_DEPTH + 1);
        for (uint32_t i = 0; i <= MAX_TREE_DEPTH; ++i) {
            child_hashes_by_level.push_back(ArenaVector<uint64_t>(*arena_));
        }

        // Scratch buffer for positions (reused across all frames)
        ArenaVector<uint8_t> scratch_positions(*arena_);
        ArenaVector<uint16_t> scratch_occ_positions(*arena_);

        // Push initial frame
        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            // First time visiting this frame? Initialize it
            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                // Check max depth
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    // Return result to parent
                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        // Finalize child hash with is_unique flag and occurrence info
                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);

                // Collect own positions
                scratch_positions.clear();
                auto adj_it = adjacency.find(frame.vertex);
                if (adj_it != adjacency.end()) {
                    for (size_t i = 0; i < adj_it->second.size(); ++i) {
                        scratch_positions.push_back(adj_it->second[i].second);
                    }
                }
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                // Build partial hash with level, is_unique=1, and own_positions
                // We'll add child count and child hashes later
                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);  // is_unique = true
                // Note: We hash own_positions now but need child count first
                frame.own_positions_count = scratch_positions.size();

                // Find adjacent vertices and build children list
                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                // Group adjacent vertices by vertex ID to detect non-unique children
                std::unordered_map<VertexId, ArenaVector<std::pair<uint8_t, uint8_t>>> adjacent_map;

                if (adj_it != adjacency.end()) {
                    for (size_t j = 0; j < adj_it->second.size(); ++j) {
                        EdgeId eid = adj_it->second[j].first;
                        uint8_t my_pos = adj_it->second[j].second;
                        uint8_t arity = edge_arities[eid];
                        const VertexId* verts = edge_vertices[eid];

                        for (uint8_t i = 0; i < arity; ++i) {
                            if (i != my_pos) {
                                VertexId adj_v = verts[i];
                                if (!visited.contains(adj_v)) {
                                    auto it = adjacent_map.find(adj_v);
                                    if (it == adjacent_map.end()) {
                                        it = adjacent_map.emplace(adj_v, ArenaVector<std::pair<uint8_t, uint8_t>>(*arena_)).first;
                                    }
                                    it->second.push_back({arity, i});
                                }
                            }
                        }
                    }
                }

                // Convert to children list with pre-computed occurrence positions
                for (const auto& [adj_v, occs] : adjacent_map) {
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.is_unique = (occs.size() == 1);
                    child.num_occurrences = static_cast<uint8_t>(occs.size());

                    // Pre-compute sorted occurrence positions
                    scratch_occ_positions.clear();
                    for (size_t k = 0; k < occs.size(); ++k) {
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(occs[k].first) << 8) | occs[k].second);
                    }
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    // Store sorted positions (up to 8 per child - should be enough)
                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());

                // Store own_positions for later (we need to hash them after child count)
                // Actually, we need to preserve them. Store count and re-collect later.
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            // Process next child
            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                if (child.is_unique) {
                    // Push child frame for recursive processing
                    DFSFrame child_frame;
                    child_frame.vertex = child.vertex;
                    child_frame.level = level + 1;
                    child_frame.child_idx = 0;
                    child_frame.num_children = 0;
                    child_frame.partial_hash = 0;
                    child_frame.own_positions_count = 0;
                    stack.push_back(child_frame);
                    continue;
                } else {
                    // Non-unique child: compute hash directly without recursion
                    uint64_t child_hash = FNV_OFFSET;
                    child_hash = fnv_hash(child_hash, level + 1);
                    child_hash = fnv_hash(child_hash, 0);  // is_unique = false
                    child_hash = fnv_hash(child_hash, 0);  // is_unique flag again (for compatibility)
                    for (uint8_t k = 0; k < child.num_occurrences && k < 8; ++k) {
                        child_hash = fnv_hash(child_hash, child.occurrence_positions[k]);
                    }
                    child_hashes.push_back(child_hash);
                    frame.child_idx++;
                    continue;
                }
            }

            // All children processed - finalize this frame's hash
            visited.clear(frame.vertex);

            // Re-collect own_positions (we stored the count but not the actual values)
            scratch_positions.clear();
            auto adj_it = adjacency.find(frame.vertex);
            if (adj_it != adjacency.end()) {
                for (size_t i = 0; i < adj_it->second.size(); ++i) {
                    scratch_positions.push_back(adj_it->second[i].second);
                }
            }
            insertion_sort(scratch_positions.begin(), scratch_positions.end());

            // Build final hash
            uint64_t hash = FNV_OFFSET;
            hash = fnv_hash(hash, level);
            hash = fnv_hash(hash, 1);  // is_unique = true
            hash = fnv_hash(hash, child_hashes.size());

            // Hash own positions
            hash = fnv_hash(hash, scratch_positions.size());
            for (size_t i = 0; i < scratch_positions.size(); ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }

            // Sort and hash child hashes
            insertion_sort(child_hashes.begin(), child_hashes.end());
            for (size_t i = 0; i < child_hashes.size(); ++i) {
                hash = fnv_hash(hash, child_hashes[i]);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            // Return result to parent
            if (!stack.empty()) {
                DFSFrame& parent = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent.level];
                auto& parent_child_hashes = child_hashes_by_level[parent.level];

                // Finalize child hash with is_unique flag and occurrence info
                ChildInfo& child_info = parent_children[parent.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent.child_idx++;
            }
        }

        return result;
    }

    // Insertion sort for small arrays (faster than std::sort for n < 16)
    template<typename Iter>
    static void insertion_sort(Iter begin, Iter end) {
        for (Iter i = begin; i != end; ++i) {
            auto key = *i;
            Iter j = i;
            while (j != begin && *(j - 1) > key) {
                *j = *(j - 1);
                --j;
            }
            *j = key;
        }
    }

    // =========================================================================
    // Tree Hash Computation - FLAT ADJACENCY VERSION
    // =========================================================================
    //
    // Uses flat adjacency arrays instead of std::unordered_map for better performance.
    //

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_tree_hash_flat(
        VertexId root,
        VertexId min_vertex,
        bool use_dense_index,
        const ArenaVector<size_t>& dense_index,
        const std::unordered_map<VertexId, size_t>& sparse_index,
        SparseBitset& visited,  // Passed from caller, reused across calls (already empty/clean)
        TreeHashBuffers& buffers,  // Reusable buffers passed from caller
        const ArenaVector<std::pair<EdgeId, uint8_t>>& adj_data,
        const ArenaVector<uint32_t>& adj_offsets,
        [[maybe_unused]] const SparseBitset& state_edges,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        // Lambda for vertex lookup
        auto get_vertex_idx = [&](VertexId v) -> size_t {
            if (use_dense_index) {
                size_t idx = v - min_vertex;
                return (idx < dense_index.size()) ? dense_index[idx] : SIZE_MAX;
            } else {
                auto it = sparse_index.find(v);
                return (it != sparse_index.end()) ? it->second : SIZE_MAX;
            }
        };

        // Use reusable buffers from caller
        buffers.reset_for_new_tree();
        auto& stack = buffers.stack;
        auto& children_by_level = buffers.children_by_level;
        auto& child_hashes_by_level = buffers.child_hashes_by_level;
        auto& scratch_positions = buffers.scratch_positions;
        auto& scratch_occ_positions = buffers.scratch_occ_positions;

        // Push initial frame
        DFSFrame initial_frame;
        initial_frame.vertex = root;
        initial_frame.level = 0;
        initial_frame.child_idx = 0;
        initial_frame.num_children = 0;
        initial_frame.partial_hash = 0;
        initial_frame.own_positions_count = 0;
        stack.push_back(initial_frame);

        uint64_t result = 0;

        while (!stack.empty()) {
            if (should_abort()) throw AbortedException{};

            DFSFrame& frame = stack[stack.size() - 1];
            uint32_t level = frame.level;

            // First time visiting this frame? Initialize it
            if (frame.child_idx == 0 && frame.num_children == 0 && frame.partial_hash == 0) {
                // Check max depth
                if (level >= MAX_TREE_DEPTH) {
                    result = fnv_hash(FNV_OFFSET, level);
                    stack.resize(stack.size() - 1);

                    // Return result to parent
                    if (!stack.empty()) {
                        DFSFrame& parent = stack[stack.size() - 1];
                        auto& parent_children = children_by_level[parent.level];
                        auto& parent_child_hashes = child_hashes_by_level[parent.level];

                        ChildInfo& child_info = parent_children[parent.child_idx];
                        uint64_t child_hash = result;
                        child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                        for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                            child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                        }
                        parent_child_hashes.push_back(child_hash);
                        parent.child_idx++;
                    }
                    continue;
                }

                visited.set(frame.vertex, *arena_);

                // Get adjacency for this vertex using flat structure
                size_t v_idx = get_vertex_idx(frame.vertex);
                uint32_t adj_start = adj_offsets[v_idx];
                uint32_t adj_end = adj_offsets[v_idx + 1];

                // Collect own positions
                scratch_positions.clear();
                for (uint32_t j = adj_start; j < adj_end; ++j) {
                    scratch_positions.push_back(adj_data[j].second);
                }
                insertion_sort(scratch_positions.begin(), scratch_positions.end());

                frame.partial_hash = FNV_OFFSET;
                frame.partial_hash = fnv_hash(frame.partial_hash, level);
                frame.partial_hash = fnv_hash(frame.partial_hash, 1);  // is_unique = true
                frame.own_positions_count = scratch_positions.size();

                // Find adjacent vertices and build children list
                auto& children = children_by_level[level];
                children.clear();
                child_hashes_by_level[level].clear();

                // Group adjacent vertices by vertex ID using reusable buffer
                auto& adj_vertices = buffers.adj_vertices;
                adj_vertices.clear();

                for (uint32_t j = adj_start; j < adj_end; ++j) {
                    EdgeId eid = adj_data[j].first;
                    uint8_t my_pos = adj_data[j].second;
                    uint8_t arity = edge_arities[eid];
                    const VertexId* verts = edge_vertices[eid];

                    for (uint8_t i = 0; i < arity; ++i) {
                        if (i != my_pos) {
                            VertexId adj_v = verts[i];
                            if (!visited.contains(adj_v)) {
                                adj_vertices.push_back({adj_v, {arity, i}});
                            }
                        }
                    }
                }

                // Sort by vertex ID to group them
                std::sort(adj_vertices.begin(), adj_vertices.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });

                // Build children from grouped vertices
                size_t i = 0;
                while (i < adj_vertices.size()) {
                    VertexId adj_v = adj_vertices[i].first;
                    ChildInfo child;
                    child.vertex = adj_v;
                    child.num_occurrences = 0;

                    scratch_occ_positions.clear();

                    // Collect all occurrences for this vertex
                    while (i < adj_vertices.size() && adj_vertices[i].first == adj_v) {
                        auto [arity, pos] = adj_vertices[i].second;
                        scratch_occ_positions.push_back(
                            (static_cast<uint16_t>(arity) << 8) | pos);
                        child.num_occurrences++;
                        ++i;
                    }

                    child.is_unique = (child.num_occurrences == 1);
                    insertion_sort(scratch_occ_positions.begin(), scratch_occ_positions.end());

                    for (size_t k = 0; k < scratch_occ_positions.size() && k < 8; ++k) {
                        child.occurrence_positions[k] = scratch_occ_positions[k];
                    }

                    children.push_back(child);
                }

                frame.num_children = static_cast<uint32_t>(children.size());
            }

            auto& children = children_by_level[level];
            auto& child_hashes = child_hashes_by_level[level];

            // Process next child
            if (frame.child_idx < frame.num_children) {
                ChildInfo& child = children[frame.child_idx];

                if (child.is_unique) {
                    // Push child frame for recursive processing
                    DFSFrame child_frame;
                    child_frame.vertex = child.vertex;
                    child_frame.level = level + 1;
                    child_frame.child_idx = 0;
                    child_frame.num_children = 0;
                    child_frame.partial_hash = 0;
                    child_frame.own_positions_count = 0;
                    stack.push_back(child_frame);
                    continue;
                } else {
                    // Non-unique child: compute hash directly without recursion
                    uint64_t child_hash = FNV_OFFSET;
                    child_hash = fnv_hash(child_hash, level + 1);
                    child_hash = fnv_hash(child_hash, 0);  // is_unique = false
                    child_hash = fnv_hash(child_hash, 0);  // is_unique flag again
                    for (uint8_t k = 0; k < child.num_occurrences && k < 8; ++k) {
                        child_hash = fnv_hash(child_hash, child.occurrence_positions[k]);
                    }
                    child_hashes.push_back(child_hash);
                    frame.child_idx++;
                    continue;
                }
            }

            // All children processed - finalize this frame's hash
            visited.clear(frame.vertex);

            // Re-collect own_positions
            size_t v_idx = get_vertex_idx(frame.vertex);
            uint32_t adj_start = adj_offsets[v_idx];
            uint32_t adj_end = adj_offsets[v_idx + 1];

            scratch_positions.clear();
            for (uint32_t j = adj_start; j < adj_end; ++j) {
                scratch_positions.push_back(adj_data[j].second);
            }
            insertion_sort(scratch_positions.begin(), scratch_positions.end());

            // Build final hash
            uint64_t hash = FNV_OFFSET;
            hash = fnv_hash(hash, level);
            hash = fnv_hash(hash, 1);  // is_unique = true
            hash = fnv_hash(hash, child_hashes.size());

            // Hash own positions
            hash = fnv_hash(hash, scratch_positions.size());
            for (size_t i = 0; i < scratch_positions.size(); ++i) {
                hash = fnv_hash(hash, scratch_positions[i]);
            }

            // Sort and hash child hashes
            insertion_sort(child_hashes.begin(), child_hashes.end());
            for (size_t i = 0; i < child_hashes.size(); ++i) {
                hash = fnv_hash(hash, child_hashes[i]);
            }

            result = hash;
            stack.resize(stack.size() - 1);

            // Return result to parent
            if (!stack.empty()) {
                DFSFrame& parent = stack[stack.size() - 1];
                auto& parent_children = children_by_level[parent.level];
                auto& parent_child_hashes = child_hashes_by_level[parent.level];

                ChildInfo& child_info = parent_children[parent.child_idx];
                uint64_t child_hash = result;
                child_hash = fnv_hash(child_hash, child_info.is_unique ? 1 : 0);
                for (uint8_t k = 0; k < child_info.num_occurrences && k < 8; ++k) {
                    child_hash = fnv_hash(child_hash, child_info.occurrence_positions[k]);
                }
                parent_child_hashes.push_back(child_hash);
                parent.child_idx++;
            }
        }

        return result;
    }

    // =========================================================================
    // Edge Signature Computation
    // =========================================================================

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_edge_signature(
        EdgeId eid,
        const VertexHashCache& cache,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        uint8_t arity = edge_arities[eid];
        const VertexId* verts = edge_vertices[eid];

        uint64_t sig = FNV_OFFSET;
        sig = fnv_hash(sig, arity);

        // Include vertex hashes IN ORDER (directed hyperedge)
        for (uint8_t i = 0; i < arity; ++i) {
            sig = fnv_hash(sig, cache.lookup(verts[i]));
        }

        return sig;
    }

    template<typename EdgeAccessor, typename ArityAccessor>
    uint64_t compute_edge_set_signature(
        const EdgeId* edges,
        uint8_t count,
        const VertexHashCache& cache,
        const EdgeAccessor& edge_vertices,
        const ArityAccessor& edge_arities
    ) {
        ArenaVector<uint64_t> edge_sigs(*arena_, count);

        for (uint8_t i = 0; i < count; ++i) {
            edge_sigs.push_back(compute_edge_signature(edges[i], cache, edge_vertices, edge_arities));
        }

        std::sort(edge_sigs.begin(), edge_sigs.end());

        uint64_t sig = FNV_OFFSET;
        sig = fnv_hash(sig, count);
        for (uint64_t es : edge_sigs) {
            sig = fnv_hash(sig, es);
        }

        return sig;
    }
};

}  // namespace hypergraph
