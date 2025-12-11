// Mock evolution data generator for testing visualization
// Creates various test cases without needing the actual rewriting engine

#pragma once

#include "hypergraph_data.hpp"
#include <random>
#include <functional>

namespace viz::scene {

// Mock evolution generator
class MockEvolutionGenerator {
public:
    MockEvolutionGenerator(uint32_t seed = 42) : rng_(seed) {}

    // Generate a simple tree evolution (no canonicalization)
    // Each state branches into `branching_factor` children
    Evolution generate_tree(uint32_t depth, uint32_t branching_factor = 2) {
        Evolution evo;

        // Initial state: simple triangle hypergraph {1,2,3}
        Hypergraph initial;
        initial.add_edge({0, 1, 2});  // Triangle
        StateId root = evo.add_state(initial, true);

        generate_tree_recursive(evo, root, depth, branching_factor);
        return evo;
    }

    // Generate an evolution with canonicalization (DAG structure)
    // Some states will map to the same canonical representative
    Evolution generate_dag(uint32_t num_states, float canonicalization_rate = 0.3f) {
        Evolution evo;

        // Initial state
        Hypergraph initial;
        initial.add_edge({0, 1});
        initial.add_edge({1, 2});
        StateId root = evo.add_state(initial, true);

        std::vector<StateId> frontier = {root};
        std::vector<StateId> canonical_states = {root};

        for (uint32_t i = 1; i < num_states; ++i) {
            // Pick a random parent from frontier
            StateId parent = frontier[rng_() % frontier.size()];

            // Create child hypergraph (slight modification of parent)
            const State* parent_state = evo.get_state(parent);
            Hypergraph child_hg = mutate_hypergraph(parent_state->hypergraph);

            StateId child = evo.add_state(child_hg);
            evo.add_event(parent, child);

            // With some probability, map to existing canonical state
            if (!canonical_states.empty() && uniform_real() < canonicalization_rate) {
                StateId canonical = canonical_states[rng_() % canonical_states.size()];
                evo.set_canonical(child, canonical);
            } else {
                canonical_states.push_back(child);
            }

            frontier.push_back(child);

            // Occasionally remove old states from frontier to control width
            if (frontier.size() > 10) {
                frontier.erase(frontier.begin());
            }
        }

        return evo;
    }

    // Generate evolution with causal edges
    Evolution generate_with_causal(uint32_t num_states) {
        Evolution evo = generate_tree(3, 2);  // Base tree

        // Add causal edges between events
        for (size_t i = 0; i + 1 < evo.events.size(); ++i) {
            // Events that share consumed/produced edges are causally related
            if (uniform_real() < 0.4f) {
                evo.add_causal_edge(
                    evo.events[i].id,
                    evo.events[i + 1].id
                );
            }
        }

        return evo;
    }

    // Generate evolution with branchial edges
    Evolution generate_with_branchial(uint32_t depth, uint32_t branching = 2) {
        Evolution evo = generate_tree(depth, branching);

        // Add branchial edges between sibling states (same depth)
        // Group states by depth
        std::vector<std::vector<StateId>> by_depth(depth + 1);
        by_depth[0].push_back(0);  // Root

        for (const auto& event : evo.events) {
            // Find depth of output state
            for (size_t d = 0; d < depth; ++d) {
                for (StateId s : by_depth[d]) {
                    if (s == event.input_state) {
                        by_depth[d + 1].push_back(event.output_state);
                        break;
                    }
                }
            }
        }

        // Add branchial edges between states at same depth
        for (const auto& level : by_depth) {
            for (size_t i = 0; i < level.size(); ++i) {
                for (size_t j = i + 1; j < level.size(); ++j) {
                    evo.add_branchial_edge(level[i], level[j]);
                }
            }
        }

        return evo;
    }

    // Generate a hypergraph with self-loops for testing virtual vertices
    Evolution generate_self_loop_test() {
        Evolution evo;

        Hypergraph hg;
        // Normal edge: 0 → 1
        hg.add_edge({0, 1});
        // Self-loop: 1 → 1 (as part of hyperedge {1, 1, 2})
        hg.add_edge({1, 1, 2});
        // Double self-loop: {2, 2, 2, 3}
        hg.add_edge({2, 2, 2, 3});
        // Pure self-loop: {3, 3}
        hg.add_edge({3, 3});

        // Multiple 2-edge loops incident on vertex 4:
        // These should spread out around vertex 4
        hg.add_edge({4, 4});  // Loop 1 at vertex 4
        hg.add_edge({4, 4});  // Loop 2 at vertex 4
        hg.add_edge({4, 4});  // Loop 3 at vertex 4
        hg.add_edge({3, 4});  // Normal edge connecting to the loop vertex

        // Another vertex (5) with mixed loops from different hyperedges
        hg.add_edge({5, 5, 6});  // Loop at 5, then to 6
        hg.add_edge({5, 5});     // Another pure loop at 5
        hg.add_edge({4, 5});     // Connect 4 and 5

        evo.add_state(hg, true);
        return evo;
    }

    // Generate a larger hypergraph for stress testing layout
    Evolution generate_large_hypergraph(uint32_t num_vertices, uint32_t num_edges) {
        Evolution evo;

        Hypergraph hg;
        hg.vertex_count = num_vertices;

        for (uint32_t i = 0; i < num_edges; ++i) {
            // Random arity 2-4
            uint32_t arity = 2 + (rng_() % 3);
            std::vector<VertexId> verts;
            for (uint32_t j = 0; j <= arity; ++j) {
                verts.push_back(rng_() % num_vertices);
            }
            hg.add_edge(verts);
        }

        evo.add_state(hg, true);
        return evo;
    }

    // Generate edge bundle test (multiple events between same states)
    Evolution generate_bundle_test() {
        Evolution evo;

        Hypergraph hg1, hg2;
        hg1.add_edge({0, 1, 2});
        hg2.add_edge({0, 1, 2});
        hg2.add_edge({2, 3});

        StateId s1 = evo.add_state(hg1, true);
        StateId s2 = evo.add_state(hg2);

        // Multiple events between same states (creates bundle)
        evo.add_event(s1, s2, {0}, {0, 1});
        evo.add_event(s1, s2, {0}, {0, 1});
        evo.add_event(s1, s2, {0}, {0, 1});

        // Mark bundle multiplicity
        for (auto& edge : evo.evolution_edges) {
            if (edge.type == EvolutionEdgeType::Event &&
                edge.source == s1 && edge.target == s2) {
                edge.multiplicity = 3;
                break;
            }
        }

        return evo;
    }

private:
    std::mt19937 rng_;

    float uniform_real() {
        return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_);
    }

    void generate_tree_recursive(Evolution& evo, StateId parent, uint32_t depth, uint32_t branching) {
        if (depth == 0) return;

        // IMPORTANT: Don't hold pointers across add_state() calls - vector may reallocate!
        // Copy the parent hypergraph by value before any modifications to evo.states
        Hypergraph parent_hg = evo.get_state(parent)->hypergraph;

        for (uint32_t i = 0; i < branching; ++i) {
            Hypergraph child_hg = mutate_hypergraph(parent_hg);
            StateId child = evo.add_state(child_hg);

            // Track consumed/produced edges
            std::vector<EdgeId> consumed, produced;
            if (!parent_hg.edges.empty()) {
                consumed.push_back(0);  // Consume first edge
            }
            if (!child_hg.edges.empty()) {
                produced.push_back(static_cast<EdgeId>(child_hg.edges.size() - 1));
            }

            evo.add_event(parent, child, consumed, produced);
            generate_tree_recursive(evo, child, depth - 1, branching);
        }
    }

    Hypergraph mutate_hypergraph(const Hypergraph& hg) {
        Hypergraph result = hg;

        // Random mutation: add an edge
        if (result.vertex_count > 0) {
            uint32_t arity = 2 + (rng_() % 2);
            std::vector<VertexId> verts;
            for (uint32_t i = 0; i <= arity; ++i) {
                // Sometimes add new vertex, sometimes reuse existing
                if (uniform_real() < 0.3f) {
                    verts.push_back(result.vertex_count++);
                } else {
                    verts.push_back(rng_() % result.vertex_count);
                }
            }
            result.add_edge(verts);
        }

        return result;
    }
};

} // namespace viz::scene
