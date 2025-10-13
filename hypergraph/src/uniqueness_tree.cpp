#include <hypergraph/uniqueness_tree.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <algorithm>

namespace hypergraph {

// COMPLEXITY ANALYSIS (rigorous, based on actual implementation):
//
// Based on Gorard (2016) "Uniqueness Trees: A Possible Polynomial Approach to the
// Graph Isomorphism Problem" (arXiv:1606.06399). Paper proves O(n^7) total for
// simple graphs with adjacency lists: O(n^6) generation + O(n^7) comparison.
// We use hashing instead of comparison, so comparison cost is O(1).
//
// NOTATION:
// V = number of vertices
// E = number of edges
// a = maximum edge arity
// For complexity bounds, assume worst case: E = Θ(V²) (dense graph), a = Θ(V) (large hyperedges)
//
// ============================================================================
// IMPLEMENTATION (WITH OPTIMIZATIONS):
// ============================================================================
//
// Our implementation achieves O(V⁷ log V) through several optimizations:
//
// OPTIMIZATION 1: Pre-built adjacency index (passed from WolframState)
//   - Built once in WolframState constructor: O(E·a) = O(V³)
//   - Copied when cloning states: O(E) (std::unordered_map assignment)
//   - Updated incrementally on edge add/remove: O(modified_edges · a)
//   - Threaded through: WolframState → HashStrategy → UniquenessTreeSet → UniquenessTree
//
// OPTIMIZATION 2: Shared edge_map across all trees
//   - Built once in UniquenessTreeSet::build_trees(): O(E) = O(V²)
//   - Shared by pointer with all V trees (no per-tree rebuild)
//   - Saves O(V·E) = O(V³) work
//
// OPTIMIZATION 3: Extract vertices from adjacency_index keys
//   - O(V) iteration over adjacency_index keys (already built)
//   - Avoids O(E·a) = O(V³) scan over all edges
//
// OPTIMIZATION 4: Use unordered_map in build_tree()
//   - std::unordered_map for vertex grouping: O(1) expected inserts
//   - Replaces std::map with O(log V) inserts
//
// OPTIMIZATION 5: Reserve allocations to avoid reallocations
//   - node->children.reserve(vertex_positions.size())
//   - Eliminates vector reallocation overhead
//
// OPTIMIZATION 6: Merged build/hash into single bottom-up pass
//   - compute_subtree_hash() called immediately after building each node's children
//   - Eliminates separate O(V³) tree traversal for hash computation
//   - Child hashes already available (computed bottom-up), no recursive recomputation
//
// 1. UniquenessTreeSet::build_trees():
//    - Build edge_map once: O(E) = O(V²)
//    - Extract V vertices from adjacency_index keys: O(V)
//    - Sort vertices: O(V log V)
//    - Build V trees (analyzed below)
//
// 2. UniquenessTree::UniquenessTree() per tree:
//    - Constructor receives adjacency_index and edge_map by reference/pointer
//    - build_tree(root, visited, max_depth=100) - constructs tree AND computes hashes
//
// 3. build_tree() - recursive tree construction with bottom-up hashing:
//
//    Tree structure bounds (from paper):
//    - Maximum height: h = min(V, 100)
//    - Maximum width at any level: w = O(V²)
//      (n vertices can each appear in up to n-1 edges, giving n(n-1) children)
//    - Maximum total nodes per tree: T = h × w = O(V³) for V ≤ 100
//                                               = O(100·V²) = O(V²) for V > 100
//
//    Work per node created:
//    a) find_adjacent_vertices(v, visited):
//       - Lookup adjacency_index[v]: O(1) expected (unordered_map)
//       - Iterate over degree(v) edge positions: O(degree(v))
//       - Per position, lookup edge via edge_map: O(1) expected
//       - Scan edge (arity a) for other vertices: O(a)
//       - Check visited set: O(1) expected (unordered_set)
//       - Per node: O(degree(v) · a)
//       - **AMORTIZED across all nodes in tree: Σ degree(v) · a = O(E·a) = O(V³)**
//         (since Σ degree(v) over all V vertices = O(E))
//
//    b) Group adjacencies into std::unordered_map<GlobalVertexId, ...>:
//       - |adjacent| = O(degree(v) · a) per node
//       - Insert into unordered_map: O(1) expected per insert
//       - Per node: O(degree(v) · a)
//       - **Amortized across all T nodes: O(E·a) = O(V³)**
//
//    c) Reserve and create child nodes: O(|unique_vertices|) ≤ O(V) per node
//       - Across all T = O(V³) nodes: O(V⁴)
//
//    d) Mark uniqueness: O(|children|) ≤ O(V) per node, O(V⁴) total
//
//    e) Recurse on unique children (depth limited by h) - builds subtrees and computes their hashes
//
//    f) compute_subtree_hash() called at end of build_tree() for this node:
//       - Extract and sort |occurrences| positions:
//         * |occurrences| ≤ E·a = O(V³) (vertex can appear in many edges/positions)
//         * std::sort: O(V³ log V³) = O(V³ log V)
//       - Collect child hashes (already computed during recursion): O(|children|)
//       - Sort |children| child hashes:
//         * |children| ≤ V² (paper's bound on width)
//         * std::sort: O(V² log V²) = O(V² log V)
//       - Per node total: O(V³ log V) (dominated by sorting occurrences)
//
//    **Per tree total:**
//    - Tree construction steps (a-e): O(V³) + O(V³) + O(V⁴) = O(V⁴)
//    - Hash computation (f) for all T = O(V³) nodes: T × O(V³ log V) = O(V⁶ log V)
//    - **= O(V⁶ log V) per tree** (hash computation dominates)
//
// 4. compute_canonical_hash():
//    - Collect and sort V tree hashes: O(V log V)
//
// IMPLEMENTATION TOTAL:
// - One-time setup (build_trees): O(E) edge_map + O(V log V) vertex sort = O(V² log V)
// - Build+hash (merged): V trees × O(V⁶ log V) per tree = O(V⁷ log V)
//   (Dominated by O(V³) nodes × O(V³ log V) per node for sorting occurrences)
// - Final hash combination: O(V log V)
// - **TOTAL: O(V⁷ log V)** ✓
//
// This is a factor of V improvement over the naive O(V⁸ log V) implementation that
// would scan all E edges for every tree node. The adjacency index optimization
// amortizes the edge lookups across all nodes in a tree.
//
// For a = O(1) (constant arity, e.g., binary edges):
// - Tree construction: O(V⁵)
// - Hash computation: O(V⁷ log V) (dominates)
// - This matches the paper's theoretical bound for simple graphs
//
// ============================================================================
// EXTENSIONS BEYOND PAPER:
// ============================================================================
// Original paper: simple graphs (no loops, no multiple edges, binary edges only)
// Our implementation extends to:
// 1. Edge multiplicity: Distinguished via EdgePosition (edge_id + position)
// 2. Self-loops: Vertex appearing multiple times in same edge
// 3. Hypergraphs: Arbitrary arity a (not just a=2)
//
// These extensions preserve polynomial complexity with the adjacency index.

// ============================================================================
// TreeNode Implementation
// ============================================================================

TreeNode::TreeNode(const TreeNode& other)
    : vertex(other.vertex), level(other.level),
      occurrences(other.occurrences), subtree_hash(other.subtree_hash),
      is_unique(other.is_unique) {
    // Deep copy children
    children.reserve(other.children.size());
    for (const auto& child : other.children) {
        children.push_back(std::make_unique<TreeNode>(*child));
    }
}

TreeNode& TreeNode::operator=(const TreeNode& other) {
    if (this != &other) {
        vertex = other.vertex;
        level = other.level;
        occurrences = other.occurrences;
        subtree_hash = other.subtree_hash;
        is_unique = other.is_unique;

        // Deep copy children
        children.clear();
        children.reserve(other.children.size());
        for (const auto& child : other.children) {
            children.push_back(std::make_unique<TreeNode>(*child));
        }
    }
    return *this;
}

uint64_t TreeNode::compute_subtree_hash() {
    // Include structural properties with level information
    uint64_t hash = std::hash<std::size_t>{}(level);

    // Include uniqueness flag
    hash ^= (std::hash<bool>{}(is_unique) << 1);

    // Include number of children as a structural property
    hash ^= (std::hash<std::size_t>{}(children.size()) << 2);

    // CRITICAL: Include position information to handle edge multiplicity
    // We hash the positions (structural info) but NOT the edge IDs (arbitrary labels)
    // The number of occurrences captures multiplicity, edge IDs are just for bookkeeping
    std::vector<std::size_t> positions;
    positions.reserve(occurrences.size());
    for (const auto& occ : occurrences) {
        positions.push_back(occ.position);
    }

    // Sort to ensure canonical ordering (edge_id is arbitrary, only position matters)
    std::sort(positions.begin(), positions.end());

    // Hash the occurrence count (edge multiplicity) and positions
    hash ^= std::hash<std::size_t>{}(positions.size()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

    for (std::size_t pos : positions) {
        hash ^= std::hash<std::size_t>{}(pos) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    // Include child hashes (sorted for canonical form)
    // Child hashes are already computed during tree construction (bottom-up)
    std::vector<uint64_t> child_hashes;
    child_hashes.reserve(children.size());
    for (const auto& child : children) {
        child_hashes.push_back(child->subtree_hash);
    }
    std::sort(child_hashes.begin(), child_hashes.end());

    for (uint64_t child_hash : child_hashes) {
        hash ^= child_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    subtree_hash = hash;
    return hash;
}

std::size_t TreeNode::height() const {
    if (children.empty()) {
        return 0;
    }

    std::size_t max_child_height = 0;
    for (const auto& child : children) {
        max_child_height = std::max(max_child_height, child->height());
    }
    return max_child_height + 1;
}

// ============================================================================
// UniquenessTree Implementation
// ============================================================================

UniquenessTree::UniquenessTree(GlobalVertexId root_vertex,
                              const std::vector<GlobalHyperedge>& edges,
                              const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index,
                              const std::unordered_map<GlobalEdgeId, const GlobalHyperedge*>* edge_map)
    : root_vertex_(root_vertex), edges_(&edges), adjacency_index_(&adjacency_index) {
    root_ = std::make_unique<TreeNode>(root_vertex, 0);

    // Use provided edge_map if available, otherwise build locally
    if (edge_map) {
        edge_map_ = edge_map;
    } else {
        // Build edge_id -> edge pointer map once - O(E) one-time cost
        owned_edge_map_.reserve(edges.size());
        for (const auto& edge : edges) {
            owned_edge_map_[edge.global_id] = &edge;
        }
        edge_map_ = &owned_edge_map_;
    }

    // Find all occurrences of root vertex using adjacency index O(degree(v))
    if (adjacency_index.count(root_vertex)) {
        const auto& positions = adjacency_index.at(root_vertex);
        root_->occurrences.reserve(positions.size());
        for (const auto& [edge_id, pos] : positions) {
            root_->occurrences.emplace_back(edge_id, pos);
        }
    }

    // Build the tree and compute hashes in one pass (optimization #2)
    std::unordered_set<GlobalVertexId> visited;
    build_tree(root_.get(), visited);
}

UniquenessTree::UniquenessTree(const UniquenessTree& other)
    : root_vertex_(other.root_vertex_), edges_(other.edges_), adjacency_index_(other.adjacency_index_),
      owned_edge_map_(other.owned_edge_map_) {
    // Repoint edge_map_ to correct location
    edge_map_ = other.edge_map_ == &other.owned_edge_map_ ? &owned_edge_map_ : other.edge_map_;

    if (other.root_) {
        root_ = std::make_unique<TreeNode>(*other.root_);
    }
}

UniquenessTree::UniquenessTree(UniquenessTree&& other) noexcept
    : root_vertex_(other.root_vertex_), root_(std::move(other.root_)),
      edges_(other.edges_), adjacency_index_(other.adjacency_index_),
      owned_edge_map_(std::move(other.owned_edge_map_)), edge_map_(other.edge_map_) {
    // Repoint if it was pointing to owned map
    if (other.edge_map_ == &other.owned_edge_map_) {
        edge_map_ = &owned_edge_map_;
    }
    other.edges_ = nullptr;
    other.adjacency_index_ = nullptr;
    other.edge_map_ = nullptr;
}

UniquenessTree& UniquenessTree::operator=(const UniquenessTree& other) {
    if (this != &other) {
        root_vertex_ = other.root_vertex_;
        edges_ = other.edges_;
        adjacency_index_ = other.adjacency_index_;
        owned_edge_map_ = other.owned_edge_map_;
        edge_map_ = other.edge_map_ == &other.owned_edge_map_ ? &owned_edge_map_ : other.edge_map_;

        if (other.root_) {
            root_ = std::make_unique<TreeNode>(*other.root_);
        } else {
            root_.reset();
        }
    }
    return *this;
}

UniquenessTree& UniquenessTree::operator=(UniquenessTree&& other) noexcept {
    if (this != &other) {
        root_vertex_ = other.root_vertex_;
        root_ = std::move(other.root_);
        edges_ = other.edges_;
        adjacency_index_ = other.adjacency_index_;
        owned_edge_map_ = std::move(other.owned_edge_map_);
        edge_map_ = other.edge_map_;

        if (other.edge_map_ == &other.owned_edge_map_) {
            edge_map_ = &owned_edge_map_;
        }

        other.edges_ = nullptr;
        other.adjacency_index_ = nullptr;
        other.edge_map_ = nullptr;
    }
    return *this;
}

void UniquenessTree::build_tree(TreeNode* node, std::unordered_set<GlobalVertexId>& visited,
                                std::size_t max_depth) {
    if (node->level >= max_depth) {
        // Compute hash even at max depth
        node->compute_subtree_hash();
        return; // Prevent infinite recursion
    }

    visited.insert(node->vertex);

    // Find adjacent vertices that haven't been visited
    auto adjacent = find_adjacent_vertices(node->vertex, visited);

    if (adjacent.empty()) {
        visited.erase(node->vertex);
        // Compute hash even for leaf nodes (e.g., self-loops with no children)
        node->compute_subtree_hash();
        return;
    }

    // Group adjacent vertices by their ID to detect duplicates
    // Use unordered_map for O(1) expected insert instead of map's O(log V)
    std::unordered_map<GlobalVertexId, std::vector<EdgePosition>> vertex_positions;
    for (const auto& [vertex, position] : adjacent) {
        vertex_positions[vertex].push_back(position);
    }

    // Create child nodes - reserve to avoid reallocations
    node->children.reserve(vertex_positions.size());

    for (const auto& [vertex, positions] : vertex_positions) {
        auto child = std::make_unique<TreeNode>(vertex, node->level + 1);
        child->occurrences = positions;

        // Mark as non-unique if vertex appears at multiple positions (edge multiplicity)
        if (positions.size() > 1) {
            child->is_unique = false;
        }

        node->children.push_back(std::move(child));
    }

    // Recursively build subtrees for unique vertices only
    for (auto& child : node->children) {
        if (child->is_unique) {
            build_tree(child.get(), visited, max_depth);
        } else {
            // Non-unique children still need hashes computed
            child->compute_subtree_hash();
        }
    }

    visited.erase(node->vertex);

    // Compute hash immediately after building subtree (optimization #2)
    // This saves a separate tree traversal
    node->compute_subtree_hash();
}

std::vector<std::pair<GlobalVertexId, EdgePosition>>
UniquenessTree::find_adjacent_vertices(GlobalVertexId vertex,
                                      const std::unordered_set<GlobalVertexId>& visited) const {
    std::vector<std::pair<GlobalVertexId, EdgePosition>> adjacent;

    // Use adjacency index for O(degree(v)·a) lookup instead of O(E·a²) scan
    if (!adjacency_index_->count(vertex)) {
        return adjacent;
    }

    const auto& positions = adjacency_index_->at(vertex);

    // Use pre-built edge_map_ for O(1) edge lookup instead of O(E) rebuild
    // For each occurrence of this vertex
    for (const auto& [edge_id, vertex_pos] : positions) {
        const GlobalHyperedge* edge = edge_map_->at(edge_id);

        // Find adjacent vertices at positions after this vertex's position
        for (std::size_t i = vertex_pos + 1; i < edge->global_vertices.size(); ++i) {
            GlobalVertexId adj_vertex = edge->global_vertices[i];
            if (visited.find(adj_vertex) == visited.end()) {
                adjacent.emplace_back(adj_vertex, EdgePosition(edge_id, i));
            }
        }
    }

    return adjacent;
}

void UniquenessTree::update_hashes() {
    if (root_) {
        root_->compute_subtree_hash();
    }
}

// ============================================================================
// UniquenessTreeSet Implementation
// ============================================================================

UniquenessTreeSet::UniquenessTreeSet(
    const std::vector<GlobalHyperedge>& edges,
    const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index)
    : edges_(&edges), adjacency_index_(&adjacency_index), canonical_hash_(0), hash_valid_(false) {
    build_trees();
}

UniquenessTreeSet::UniquenessTreeSet(const UniquenessTreeSet& other)
    : trees_(other.trees_), vertex_to_tree_index_(other.vertex_to_tree_index_),
      edges_(other.edges_), adjacency_index_(other.adjacency_index_),
      canonical_hash_(other.canonical_hash_), hash_valid_(other.hash_valid_) {
}

UniquenessTreeSet::UniquenessTreeSet(UniquenessTreeSet&& other) noexcept
    : trees_(std::move(other.trees_)),
      vertex_to_tree_index_(std::move(other.vertex_to_tree_index_)),
      edges_(other.edges_), adjacency_index_(other.adjacency_index_),
      canonical_hash_(other.canonical_hash_), hash_valid_(other.hash_valid_) {
    other.edges_ = nullptr;
    other.adjacency_index_ = nullptr;
}

UniquenessTreeSet& UniquenessTreeSet::operator=(const UniquenessTreeSet& other) {
    if (this != &other) {
        trees_ = other.trees_;
        vertex_to_tree_index_ = other.vertex_to_tree_index_;
        edges_ = other.edges_;
        adjacency_index_ = other.adjacency_index_;
        canonical_hash_ = other.canonical_hash_;
        hash_valid_ = other.hash_valid_;
    }
    return *this;
}

UniquenessTreeSet& UniquenessTreeSet::operator=(UniquenessTreeSet&& other) noexcept {
    if (this != &other) {
        trees_ = std::move(other.trees_);
        vertex_to_tree_index_ = std::move(other.vertex_to_tree_index_);
        edges_ = other.edges_;
        adjacency_index_ = other.adjacency_index_;
        canonical_hash_ = other.canonical_hash_;
        hash_valid_ = other.hash_valid_;
        other.edges_ = nullptr;
        other.adjacency_index_ = nullptr;
    }
    return *this;
}

void UniquenessTreeSet::build_trees() {
    if (!edges_ || !adjacency_index_) return;

    // Build edge_id -> edge map once - O(E), shared by all V trees
    // Avoids O(V·E) = O(V³) redundant work
    edge_map_.clear();
    edge_map_.reserve(edges_->size());
    for (const auto& edge : *edges_) {
        edge_map_[edge.global_id] = &edge;
    }

    // Collect all unique vertices from adjacency_index_ keys - O(V)
    // More efficient than scanning all edges O(E·a)
    std::vector<GlobalVertexId> vertices;
    vertices.reserve(adjacency_index_->size());
    for (const auto& [vertex, _] : *adjacency_index_) {
        vertices.push_back(vertex);
    }

    // Sort for deterministic ordering
    std::sort(vertices.begin(), vertices.end());

    trees_.clear();
    vertex_to_tree_index_.clear();

    trees_.reserve(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        GlobalVertexId vertex = vertices[i];
        // Pass edge_map_ pointer to avoid rebuilding in each tree
        trees_.emplace_back(vertex, *edges_, *adjacency_index_, &edge_map_);
        vertex_to_tree_index_[vertex] = i;
    }

    hash_valid_ = false;
}

uint64_t UniquenessTreeSet::canonical_hash() const {
    if (!hash_valid_) {
        compute_canonical_hash();
    }
    return canonical_hash_;
}

void UniquenessTreeSet::compute_canonical_hash() const {
    uint64_t result = 0;

    // Collect tree hashes
    std::vector<uint64_t> tree_hashes;
    tree_hashes.reserve(trees_.size());

    for (const auto& tree : trees_) {
        tree_hashes.push_back(tree.hash());
    }

    // Sort hashes to ensure canonical ordering (isomorphism-invariant)
    std::sort(tree_hashes.begin(), tree_hashes.end());

    // Combine tree hashes using FNV-style hashing
    for (uint64_t hash : tree_hashes) {
        result ^= hash + 0x9e3779b9 + (result << 6) + (result >> 2);
    }

    canonical_hash_ = result;
    hash_valid_ = true;
}

} // namespace hypergraph
