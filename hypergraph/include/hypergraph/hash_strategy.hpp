#ifndef HYPERGRAPH_HASH_STRATEGY_HPP
#define HYPERGRAPH_HASH_STRATEGY_HPP

#include <hypergraph/types.hpp>
#include <hypergraph/wolfram_states.hpp>
#include <hypergraph/canonicalization.hpp>
#include <hypergraph/uniqueness_tree.hpp>
#include <vector>
#include <cstdint>

namespace hypergraph {

/**
 * Strategy interface for computing isomorphism-invariant hashes of hypergraphs.
 * Allows runtime or compile-time selection between different hashing algorithms.
 */
class HashStrategy {
public:
    virtual ~HashStrategy() = default;

    /**
     * Compute isomorphism-invariant hash for a collection of edges.
     * @param edges The edges to hash
     * @param adjacency_index Adjacency index mapping vertices to (edge_id, position) pairs
     * @return Hash value that is the same for isomorphic hypergraphs
     */
    virtual std::size_t compute_hash(
        const std::vector<GlobalHyperedge>& edges,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index) const = 0;

    /**
     * Check if two edge collections represent isomorphic hypergraphs.
     * Optional method - not all strategies need to implement this.
     */
    virtual bool are_isomorphic(
        const std::vector<GlobalHyperedge>& edges1,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index1,
        const std::vector<GlobalHyperedge>& edges2,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index2) const {
        // Default implementation: compare hashes (may have false positives)
        return compute_hash(edges1, adjacency_index1) == compute_hash(edges2, adjacency_index2);
    }
};

/**
 * Canonicalization-based hash strategy.
 * Uses full vertex renaming and edge sorting (O(n!) worst case).
 * Accurate but slow for large graphs.
 */
class CanonicalizationHashStrategy : public HashStrategy {
public:
    std::size_t compute_hash(
        const std::vector<GlobalHyperedge>& edges,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index) const override {
        (void)adjacency_index; // Unused by canonicalization

        if (edges.empty()) {
            return 0;
        }

        // Convert global edges to vector format
        std::vector<std::vector<GlobalVertexId>> edge_vectors;
        edge_vectors.reserve(edges.size());
        for (const auto& global_edge : edges) {
            edge_vectors.push_back(global_edge.global_vertices);
        }

        // Canonicalize
        Canonicalizer canonicalizer;
        auto result = canonicalizer.canonicalize_edges(edge_vectors);

        // FNV-style hash of canonical form
        std::size_t hash = 14695981039346656037ULL; // FNV offset basis
        constexpr std::size_t FNV_PRIME = 1099511628211ULL;

        for (const auto& edge : result.canonical_form.edges) {
            for (std::size_t vertex : edge) {
                hash ^= vertex;
                hash *= FNV_PRIME;
            }
            // Add separator between edges
            hash ^= 0xDEADBEEF;
            hash *= FNV_PRIME;
        }

        return hash;
    }

    bool are_isomorphic(
        const std::vector<GlobalHyperedge>& edges1,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index1,
        const std::vector<GlobalHyperedge>& edges2,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index2) const override {
        (void)adjacency_index1; // Unused by canonicalization
        (void)adjacency_index2; // Unused by canonicalization
        // Convert to vector format
        auto to_vectors = [](const std::vector<GlobalHyperedge>& edges) {
            std::vector<std::vector<GlobalVertexId>> result;
            result.reserve(edges.size());
            for (const auto& edge : edges) {
                result.push_back(edge.global_vertices);
            }
            return result;
        };

        Canonicalizer canonicalizer;
        auto canonical1 = canonicalizer.canonicalize_edges(to_vectors(edges1));
        auto canonical2 = canonicalizer.canonicalize_edges(to_vectors(edges2));

        return canonical1.canonical_form.edges == canonical2.canonical_form.edges;
    }
};

/**
 * Uniqueness tree-based hash strategy.
 * Uses polynomial-time (O(n^7)) graph hashing algorithm.
 * Fast but may have rare hash collisions (check with canonicalization if needed).
 */
class UniquenessTreeHashStrategy : public HashStrategy {
public:
    std::size_t compute_hash(
        const std::vector<GlobalHyperedge>& edges,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index) const override {
        if (edges.empty()) {
            return 0;
        }

        // Build uniqueness tree set and get canonical hash
        UniquenessTreeSet tree_set(edges, adjacency_index);
        return static_cast<std::size_t>(tree_set.canonical_hash());
    }

    bool are_isomorphic(
        const std::vector<GlobalHyperedge>& edges1,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index1,
        const std::vector<GlobalHyperedge>& edges2,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index2) const override {
        // Basic checks first
        if (edges1.size() != edges2.size()) {
            return false;
        }

        // Compare uniqueness tree hashes
        // Note: This may have false positives (hash collisions) in rare cases
        return compute_hash(edges1, adjacency_index1) == compute_hash(edges2, adjacency_index2);
    }
};

/**
 * Compile-time hash strategy selector for zero-overhead abstraction.
 * Use this for performance-critical code where strategy is known at compile time.
 */
template<HashStrategyType Strategy>
struct HashStrategySelector;

template<>
struct HashStrategySelector<HashStrategyType::CANONICALIZATION> {
    using type = CanonicalizationHashStrategy;
};

template<>
struct HashStrategySelector<HashStrategyType::UNIQUENESS_TREE> {
    using type = UniquenessTreeHashStrategy;
};

/**
 * Helper function to compute hash with compile-time strategy selection.
 */
template<HashStrategyType Strategy>
inline std::size_t compute_hash_with_strategy(const std::vector<GlobalHyperedge>& edges) {
    typename HashStrategySelector<Strategy>::type strategy;
    return strategy.compute_hash(edges);
}

/**
 * Default hash strategy (can be changed via compile-time flag).
 */
#ifndef DEFAULT_HASH_STRATEGY
#define DEFAULT_HASH_STRATEGY HashStrategyType::UNIQUENESS_TREE
#endif

using DefaultHashStrategy = typename HashStrategySelector<DEFAULT_HASH_STRATEGY>::type;

/**
 * Runtime-selectable hash strategy for benchmarking and comparison.
 * Allows switching between canonicalization and uniqueness tree hashing at runtime.
 */
class RuntimeHashStrategy : public HashStrategy {
private:
    HashStrategyType strategy_type_;

public:
    explicit RuntimeHashStrategy(HashStrategyType type = DEFAULT_HASH_STRATEGY)
        : strategy_type_(type) {}

    void set_strategy_type(HashStrategyType type) {
        strategy_type_ = type;
    }

    HashStrategyType get_strategy_type() const {
        return strategy_type_;
    }

    std::size_t compute_hash(
        const std::vector<GlobalHyperedge>& edges,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index) const override {
        switch (strategy_type_) {
            case HashStrategyType::CANONICALIZATION: {
                CanonicalizationHashStrategy strategy;
                return strategy.compute_hash(edges, adjacency_index);
            }
            case HashStrategyType::UNIQUENESS_TREE: {
                UniquenessTreeHashStrategy strategy;
                return strategy.compute_hash(edges, adjacency_index);
            }
        }
        return 0; // unreachable
    }

    bool are_isomorphic(
        const std::vector<GlobalHyperedge>& edges1,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index1,
        const std::vector<GlobalHyperedge>& edges2,
        const std::unordered_map<GlobalVertexId, std::vector<std::pair<GlobalEdgeId, std::size_t>>>& adjacency_index2) const override {
        switch (strategy_type_) {
            case HashStrategyType::CANONICALIZATION: {
                CanonicalizationHashStrategy strategy;
                return strategy.are_isomorphic(edges1, adjacency_index1, edges2, adjacency_index2);
            }
            case HashStrategyType::UNIQUENESS_TREE: {
                UniquenessTreeHashStrategy strategy;
                return strategy.are_isomorphic(edges1, adjacency_index1, edges2, adjacency_index2);
            }
        }
        return false; // unreachable
    }
};

} // namespace hypergraph

#endif // HYPERGRAPH_HASH_STRATEGY_HPP
