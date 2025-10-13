#include <hypergraph/uniqueness_tree.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <algorithm>
#include <map>

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
// CURRENT IMPLEMENTATION (WITHOUT ADJACENCY INDEX):
// ============================================================================
//
// 1. UniquenessTreeSet::build_trees():
//    - Collect V unique vertices from edges: O(E·a) = O(V³)
//    - Sort vertices: O(V log V)
//    - Build V trees (analyzed below)
//
// 2. UniquenessTree::UniquenessTree() per tree:
//    - Find root vertex in all edges: O(E·a) = O(V³)
//    - build_tree(root, visited, max_depth=100)
//    - update_hashes()
//
// 3. build_tree() - recursive tree construction (lines 271-330):
//
//    Tree structure bounds (from paper):
//    - Maximum height: h = min(V, 100)
//    - Maximum width at any level: w = O(V²)
//      (n vertices can each appear in up to n-1 edges, giving n(n-1) children)
//    - Maximum total nodes per tree: T = h × w = O(V³) for V ≤ 100
//                                               = O(100·V²) = O(V²) for V > 100
//
//    Work per node created:
//    a) find_adjacent_vertices(v, visited) (line 280, impl lines 332-364):
//       **CRITICAL BOTTLENECK: Scans ALL E edges for EVERY node**
//       - Loop over ALL E edges (line 338): O(E)
//       - Per edge, scan all a positions to find v (lines 340-345): O(a)
//       - Per position where v found, scan subsequent positions (lines 353-360): O(a)
//       - Check visited set (line 356): O(1) expected (unordered_set)
//       - Total: O(E·a²) = O(V⁴) assuming E=O(V²), a=O(V)
//
//    b) Group adjacencies into std::map<GlobalVertexId, ...> (lines 288-291):
//       - |adjacent| ≤ E·a² = O(V⁴) entries returned from find_adjacent
//       - Insert each into std::map (red-black tree): O(log V) per insert
//       - Total: O(|adjacent| · log V) = O(V⁴ log V)
//
//    c) Create child nodes (lines 293-306): O(|unique_vertices|) ≤ O(V)
//
//    d) Mark uniqueness (lines 308-320): O(|children|) ≤ O(V)
//
//    e) Recurse on unique children (lines 322-327)
//
//    **Per node work: O(V⁴ log V)** (dominated by grouping step b)
//
//    **Per tree total: T nodes × O(V⁴ log V) per node**
//                  **= O(V³) × O(V⁴ log V) = O(V⁷ log V)**
//
// 4. update_hashes() per tree (called once after construction, line 232):
//    Calls TreeNode::compute_subtree_hash() on root (line 368)
//
//    TreeNode::compute_subtree_hash() is RECURSIVE (lines 153-195):
//    - Post-order traversal: visits each node once
//    - Per node (lines 166-191):
//      a) Extract and sort |occurrences| positions (lines 166-172):
//         - |occurrences| ≤ E·a = O(V³) (vertex can appear in many edges/positions)
//         - std::sort: O(V³ log V³) = O(V³ log V)
//      b) Recursively compute child hashes (line 185): handled by recursion
//      c) Sort |children| child hashes (lines 182-187):
//         - |children| ≤ V² (paper's bound on width)
//         - std::sort: O(V² log V²) = O(V² log V)
//      - Per node total: O(V³ log V) (dominated by sorting occurrences)
//
//    Total for T = O(V³) nodes: T × O(V³ log V) = O(V³ · V³ log V) = O(V⁶ log V)
//
// 5. compute_canonical_hash():
//    - Collect and sort V tree hashes: O(V log V)
//
// CURRENT IMPLEMENTATION TOTAL:
// - Tree construction: V trees × O(V⁷ log V) per tree = O(V⁸ log V)
//   (Dominated by O(V³) nodes × O(V⁴ log V) per node for find_adjacent + grouping)
// - Hash computation: V trees × O(V⁶ log V) per tree = O(V⁷ log V)
//   (O(V³) nodes × O(V³ log V) per node for sorting occurrences/children)
// - Final hash combination: O(V log V)
// - **TOTAL: O(V⁸ log V)**
//
// ============================================================================
// WITH ADJACENCY LIST OPTIMIZATION (PLANNED):
// ============================================================================
//
// Build adjacency list once at UniquenessTreeSet construction:
//   std::unordered_map<GlobalVertexId, std::vector<EdgePosition>> adjacency;
//   Cost: O(E·a) = O(V³) one-time
//
// Modified find_adjacent_vertices(v, visited):
//   - Lookup adjacency[v]: O(1) expected
//   - Filter by visited: O(|adjacency[v]|) = O(degree(v) · a)
//   - Sum over all nodes in tree: Σ degree(v) · a = O(E·a) amortized
//     (since Σ degree(v) = 2E for undirected, or E for directed)
//
// Modified per-tree cost:
//   - find_adjacent for all T nodes: O(E·a) = O(V³) amortized across all nodes
//   - Grouping for all nodes: O(V³ log V) worst case
//   - Uniqueness checking: O(T·V) = O(V³·V) = O(V⁴)
//   - Per tree: O(V⁴)
//
// With optimization:
// - Tree construction: V trees × O(V⁴) = O(V⁵)
// - Hash computation: O(V⁷ log V) (unchanged)
// - TOTAL: O(V⁷ log V)
//
// For a = O(1) (constant arity, e.g., binary edges):
// - Current: O(V⁸)
// - Optimized: O(V⁵) for construction, O(V⁷ log V) for hashing
// - This would be BETTER than paper's O(V⁶) for generation!
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
// These extensions preserve polynomial complexity with adjacency optimization.

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
    for (const auto& occ : occurrences) {
        positions.push_back(occ.position);
    }

    // Sort to ensure canonical ordering
    std::sort(positions.begin(), positions.end());

    // Hash the occurrence count (edge multiplicity) and positions
    hash ^= std::hash<std::size_t>{}(positions.size()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

    for (std::size_t pos : positions) {
        hash ^= std::hash<std::size_t>{}(pos) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    // Include child hashes (sorted for canonical form)
    std::vector<uint64_t> child_hashes;
    child_hashes.reserve(children.size());
    for (const auto& child : children) {
        child_hashes.push_back(child->compute_subtree_hash());
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
                              const std::vector<GlobalHyperedge>& edges)
    : root_vertex_(root_vertex), edges_(&edges) {
    root_ = std::make_unique<TreeNode>(root_vertex, 0);

    // Find all occurrences of root vertex
    for (const auto& edge : edges) {
        for (std::size_t pos = 0; pos < edge.global_vertices.size(); ++pos) {
            if (edge.global_vertices[pos] == root_vertex) {
                root_->occurrences.emplace_back(edge.global_id, pos);
            }
        }
    }

    // Build the tree
    std::unordered_set<GlobalVertexId> visited;
    build_tree(root_.get(), visited);

    // Compute hashes
    update_hashes();
}

UniquenessTree::UniquenessTree(const UniquenessTree& other)
    : root_vertex_(other.root_vertex_), edges_(other.edges_) {
    if (other.root_) {
        root_ = std::make_unique<TreeNode>(*other.root_);
    }
}

UniquenessTree::UniquenessTree(UniquenessTree&& other) noexcept
    : root_vertex_(other.root_vertex_), root_(std::move(other.root_)),
      edges_(other.edges_) {
    other.edges_ = nullptr;
}

UniquenessTree& UniquenessTree::operator=(const UniquenessTree& other) {
    if (this != &other) {
        root_vertex_ = other.root_vertex_;
        edges_ = other.edges_;
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
        other.edges_ = nullptr;
    }
    return *this;
}

void UniquenessTree::build_tree(TreeNode* node, std::unordered_set<GlobalVertexId>& visited,
                                std::size_t max_depth) {
    if (node->level >= max_depth) {
        return; // Prevent infinite recursion
    }

    visited.insert(node->vertex);

    // Find adjacent vertices that haven't been visited
    auto adjacent = find_adjacent_vertices(node->vertex, visited);

    if (adjacent.empty()) {
        visited.erase(node->vertex);
        return;
    }

    // Group adjacent vertices by their ID to detect duplicates
    std::map<GlobalVertexId, std::vector<EdgePosition>> vertex_positions;
    for (const auto& [vertex, position] : adjacent) {
        vertex_positions[vertex].push_back(position);
    }

    // Create child nodes for unique vertices only
    std::vector<TreeNode*> children_at_level;
    for (const auto& [vertex, positions] : vertex_positions) {
        auto child = std::make_unique<TreeNode>(vertex, node->level + 1);
        child->occurrences = positions;

        // Mark as non-unique if it appears multiple times at this level
        if (positions.size() > 1) {
            child->is_unique = false;
        }

        children_at_level.push_back(child.get());
        node->children.push_back(std::move(child));
    }

    // Further mark vertices as non-unique if they appear in multiple child nodes
    // Count occurrences of each vertex at this level
    std::map<GlobalVertexId, std::size_t> vertex_counts;
    for (const auto* child_node : children_at_level) {
        vertex_counts[child_node->vertex]++;
    }

    // Mark vertices as non-unique if they appear multiple times
    for (auto* child_node : children_at_level) {
        if (vertex_counts[child_node->vertex] > 1) {
            child_node->is_unique = false;
        }
    }

    // Recursively build subtrees for unique vertices only
    for (auto& child : node->children) {
        if (child->is_unique) {
            build_tree(child.get(), visited, max_depth);
        }
    }

    visited.erase(node->vertex);
}

std::vector<std::pair<GlobalVertexId, EdgePosition>>
UniquenessTree::find_adjacent_vertices(GlobalVertexId vertex,
                                      const std::unordered_set<GlobalVertexId>& visited) const {
    std::vector<std::pair<GlobalVertexId, EdgePosition>> adjacent;

    // Find all edges containing this vertex
    for (const auto& edge : *edges_) {
        // Find positions of this vertex in the edge
        std::vector<std::size_t> vertex_positions;
        for (std::size_t pos = 0; pos < edge.global_vertices.size(); ++pos) {
            if (edge.global_vertices[pos] == vertex) {
                vertex_positions.push_back(pos);
            }
        }

        if (vertex_positions.empty()) {
            continue; // Vertex not in this edge
        }

        // For hypergraphs, all other vertices in the same edge are adjacent
        // We consider vertices at positions after the first occurrence of this vertex
        for (std::size_t vertex_pos : vertex_positions) {
            for (std::size_t i = vertex_pos + 1; i < edge.global_vertices.size(); ++i) {
                GlobalVertexId adj_vertex = edge.global_vertices[i];
                if (visited.find(adj_vertex) == visited.end()) {
                    adjacent.emplace_back(adj_vertex, EdgePosition(edge.global_id, i));
                }
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

UniquenessTreeSet::UniquenessTreeSet(const std::vector<GlobalHyperedge>& edges)
    : edges_(&edges), canonical_hash_(0), hash_valid_(false) {
    build_trees();
}

UniquenessTreeSet::UniquenessTreeSet(const UniquenessTreeSet& other)
    : trees_(other.trees_), vertex_to_tree_index_(other.vertex_to_tree_index_),
      edges_(other.edges_), canonical_hash_(other.canonical_hash_),
      hash_valid_(other.hash_valid_) {
}

UniquenessTreeSet::UniquenessTreeSet(UniquenessTreeSet&& other) noexcept
    : trees_(std::move(other.trees_)),
      vertex_to_tree_index_(std::move(other.vertex_to_tree_index_)),
      edges_(other.edges_), canonical_hash_(other.canonical_hash_),
      hash_valid_(other.hash_valid_) {
    other.edges_ = nullptr;
}

UniquenessTreeSet& UniquenessTreeSet::operator=(const UniquenessTreeSet& other) {
    if (this != &other) {
        trees_ = other.trees_;
        vertex_to_tree_index_ = other.vertex_to_tree_index_;
        edges_ = other.edges_;
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
        canonical_hash_ = other.canonical_hash_;
        hash_valid_ = other.hash_valid_;
        other.edges_ = nullptr;
    }
    return *this;
}

void UniquenessTreeSet::build_trees() {
    if (!edges_) return;

    // Collect all unique vertices
    std::unordered_set<GlobalVertexId> unique_vertices;
    for (const auto& edge : *edges_) {
        for (GlobalVertexId vertex : edge.global_vertices) {
            unique_vertices.insert(vertex);
        }
    }

    // Convert to sorted vector for deterministic ordering
    std::vector<GlobalVertexId> vertices(unique_vertices.begin(), unique_vertices.end());
    std::sort(vertices.begin(), vertices.end());

    trees_.clear();
    vertex_to_tree_index_.clear();

    trees_.reserve(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        GlobalVertexId vertex = vertices[i];
        trees_.emplace_back(vertex, *edges_);
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
