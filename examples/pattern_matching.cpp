/**
 * Pattern Matching Example
 *
 * Demonstrates hypergraph pattern matching using the unified API:
 * - Creating patterns (rules) with variables
 * - Using the evolution engine to find and apply matches
 * - Understanding rule syntax with make_rule()
 */

#include <hypergraph/parallel_evolution.hpp>
#include <hypergraph/unified_hypergraph.hpp>
#include <hypergraph/pattern_matcher.hpp>
#include <iostream>

using namespace hypergraph;

int main() {
    std::cout << "=== Pattern Matching Example ===\n\n";

    // Create hypergraph
    UnifiedHypergraph hg;
    ParallelEvolutionEngine engine(&hg, 1);  // Single thread for determinism

    // Target hypergraph: triangle {1,2}, {2,3}, {3,1} + extra edge {4,5}
    std::cout << "Creating target hypergraph:\n";
    std::cout << "  {{1, 2}, {2, 3}, {3, 1}, {4, 5}}\n";
    std::cout << "  (Triangle + isolated edge)\n\n";

    std::vector<std::vector<VertexId>> initial_edges = {{1, 2}, {2, 3}, {3, 1}, {4, 5}};

    // Create pattern: two connected edges {x, y}, {y, z}
    // In unified API, patterns are defined as rules
    std::cout << "Creating pattern: {x, y}, {y, z} where x=0, y=1, z=2 are variables\n";
    std::cout << "This pattern matches two edges sharing a common vertex.\n\n";

    // Pattern as a rule that preserves edges (identity transform)
    auto pattern_rule = make_rule(0)
        .lhs({0, 1})      // {x, y}
        .lhs({1, 2})      // {y, z}
        .rhs({0, 1})      // {x, y} (preserved)
        .rhs({1, 2})      // {y, z} (preserved)
        .build();

    engine.add_rule(pattern_rule);

    std::cout << "Running evolution to find matches...\n";
    engine.evolve(initial_edges, 1);

    // The evolution creates new states for each match found
    std::cout << "\nResults:\n";
    std::cout << "  Initial states: 1\n";
    std::cout << "  Final states: " << hg.num_states() << "\n";
    std::cout << "  Events (matches found): " << hg.num_events() << "\n\n";

    // A match occurs when two connected edges are found
    // In triangle {1,2}, {2,3}, {3,1}:
    // - Match 1: {1,2}, {2,3} (sharing vertex 2)
    // - Match 2: {2,3}, {3,1} (sharing vertex 3)
    // - Match 3: {3,1}, {1,2} (sharing vertex 1)
    std::cout << "The pattern {x,y},{y,z} matches pairs of connected edges.\n";
    std::cout << "In the triangle, there are 3 such pairs (one for each corner).\n\n";

    // ===== Demonstrate a rule that creates new structure =====
    std::cout << std::string(50, '=') << "\n\n";
    std::cout << "Now demonstrating pattern matching with transformation:\n";
    std::cout << "Rule: {x,y},{y,z} -> {x,y},{y,z},{z,x} (close the triangle)\n\n";

    UnifiedHypergraph hg2;
    ParallelEvolutionEngine engine2(&hg2, 1);

    auto close_triangle_rule = make_rule(0)
        .lhs({0, 1})      // {x, y}
        .lhs({1, 2})      // {y, z}
        .rhs({0, 1})      // {x, y}
        .rhs({1, 2})      // {y, z}
        .rhs({2, 0})      // {z, x} - new edge closing the triangle
        .build();

    engine2.add_rule(close_triangle_rule);

    // Start with just two edges forming an "L" shape
    std::cout << "Initial state: {{1,2}, {2,3}} (L-shaped path)\n";
    std::vector<std::vector<VertexId>> initial2 = {{1, 2}, {2, 3}};
    engine2.evolve(initial2, 1);

    std::cout << "\nAfter 1 step:\n";
    std::cout << "  States: " << hg2.num_states() << "\n";
    std::cout << "  Events: " << hg2.num_events() << "\n";

    // Print the resulting state's edges
    std::cout << "\nThe rule adds edge {3,1} to complete the triangle.\n";

    std::cout << "\n=== Example completed successfully ===\n";
    return 0;
}
