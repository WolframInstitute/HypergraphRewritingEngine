#ifndef HYPERGRAPH_UNIQUENESS_TREE_HPP
#define HYPERGRAPH_UNIQUENESS_TREE_HPP

#include <hypergraph/types.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cstdint>

namespace hypergraph {

// Forward declaration
struct GlobalHyperedge;

/**
 * Position of a vertex within a hyperedge.
 * Handles edge multiplicity and self-loops.
 */
struct EdgePosition {
    GlobalEdgeId edge_id;
    std::size_t position;

    EdgePosition(GlobalEdgeId id, std::size_t pos) : edge_id(id), position(pos) {}

    bool operator==(const EdgePosition& other) const {
        return edge_id == other.edge_id && position == other.position;
    }

    bool operator<(const EdgePosition& other) const {
        if (edge_id != other.edge_id) return edge_id < other.edge_id;
        return position < other.position;
    }
};

/**
 * Node in a uniqueness tree.
 * Extended to handle edge multiplicity and self-loops.
 */
struct TreeNode {
    GlobalVertexId vertex;
    std::size_t level;
    std::vector<EdgePosition> occurrences;
    std::vector<std::unique_ptr<TreeNode>> children;
    uint64_t subtree_hash;
    bool is_unique;

    TreeNode(GlobalVertexId v, std::size_t l) : vertex(v), level(l), subtree_hash(0), is_unique(true) {}

    // Deep copy
    TreeNode(const TreeNode& other);
    TreeNode& operator=(const TreeNode& other);

    // Move operations
    TreeNode(TreeNode&& other) noexcept = default;
    TreeNode& operator=(TreeNode&& other) noexcept = default;

    /**
     * Compute hash of subtree with support for edge multiplicity.
     * Includes edge IDs and positions to distinguish identical edges.
     */
    uint64_t compute_subtree_hash();

    std::size_t height() const;
    bool is_leaf() const { return children.empty(); }
};

/**
 * Uniqueness tree for a single vertex.
 * Adapted to work with GlobalHyperedge instead of OrderedHypergraph.
 */
class UniquenessTree {
public:
    /**
     * Construct uniqueness tree for a vertex in a hypergraph.
     * @param root_vertex The vertex to build tree around
     * @param edges All edges in the hypergraph
     * @param adjacency_index Adjacency index for O(degree(v)) edge lookups
     * @param edge_map Pre-built edge_id->edge map (nullptr to build locally, or shared from UniquenessTreeSet)
     */
    UniquenessTree(GlobalVertexId root_vertex,
                  const std::vector<GlobalHyperedge>& edges,
                  const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index,
                  const std::unordered_map<GlobalEdgeId, const GlobalHyperedge*>* edge_map = nullptr);

    // Copy and move operations
    UniquenessTree(const UniquenessTree& other);
    UniquenessTree(UniquenessTree&& other) noexcept;
    UniquenessTree& operator=(const UniquenessTree& other);
    UniquenessTree& operator=(UniquenessTree&& other) noexcept;

    GlobalVertexId root_vertex() const { return root_vertex_; }
    const TreeNode& root() const { return *root_; }
    std::size_t height() const { return root_->height(); }
    uint64_t hash() const { return root_->subtree_hash; }

private:
    GlobalVertexId root_vertex_;
    std::unique_ptr<TreeNode> root_;
    const std::vector<GlobalHyperedge>* edges_;
    const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>* adjacency_index_;

    // Edge map for O(1) edge_id->edge lookup
    // Either owned (if built locally) or borrowed (if passed from UniquenessTreeSet)
    std::unordered_map<GlobalEdgeId, const GlobalHyperedge*> owned_edge_map_;
    const std::unordered_map<GlobalEdgeId, const GlobalHyperedge*>* edge_map_; // Points to owned_edge_map_ or external

    void build_tree(TreeNode* node, std::unordered_set<GlobalVertexId>& visited,
                   std::size_t max_depth = 100);

    std::vector<std::pair<GlobalVertexId, EdgePosition>> find_adjacent_vertices(
        GlobalVertexId vertex, const std::unordered_set<GlobalVertexId>& visited) const;

    void update_hashes();
};

/**
 * Collection of uniqueness trees for all vertices in a hypergraph.
 * Provides canonical hash for isomorphism-invariant hashing.
 */
class UniquenessTreeSet {
public:
    /**
     * Construct trees for all vertices in the hypergraph.
     * @param edges All edges in the hypergraph
     * @param adjacency_index Adjacency index from the graph (vertex -> (edge_id, position) pairs)
     * @param edge_map Optional pre-built edge_id->edge map (built if nullptr, shared across trees)
     */
    UniquenessTreeSet(const std::vector<GlobalHyperedge>& edges,
                     const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index);

    // Copy and move operations
    UniquenessTreeSet(const UniquenessTreeSet& other);
    UniquenessTreeSet(UniquenessTreeSet&& other) noexcept;
    UniquenessTreeSet& operator=(const UniquenessTreeSet& other);
    UniquenessTreeSet& operator=(UniquenessTreeSet&& other) noexcept;

    /**
     * Get canonical hash of the entire hypergraph.
     * This hash is isomorphism-invariant.
     */
    uint64_t canonical_hash() const;

    const std::vector<UniquenessTree>& get_trees() const { return trees_; }

private:
    std::vector<UniquenessTree> trees_;
    std::unordered_map<GlobalVertexId, std::size_t> vertex_to_tree_index_;
    const std::vector<GlobalHyperedge>* edges_;
    const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>* adjacency_index_;
    std::unordered_map<GlobalEdgeId, const GlobalHyperedge*> edge_map_; // Built once, shared by all trees

    mutable uint64_t canonical_hash_;
    mutable bool hash_valid_;

    void build_trees();
    void compute_canonical_hash() const;
};

} // namespace hypergraph

#endif // HYPERGRAPH_UNIQUENESS_TREE_HPP
