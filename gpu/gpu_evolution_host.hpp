#pragma once

// Pure C++ header for GPU Evolution Engine host interface.
// This can be included from C++20 code without CUDA dependencies.

#include <cstdint>
#include <cstddef>
#include <vector>

namespace hypergraph::gpu {

// Host-side causal edge (for results)
struct HostCausalEdge {
    uint32_t producer;      // Producer event ID
    uint32_t consumer;      // Consumer event ID
    uint32_t edge;          // Edge that links them
};

// Host-side branchial edge (for results)
struct HostBranchialEdge {
    uint32_t event1;        // First event
    uint32_t event2;        // Second event
    uint32_t shared_edge;   // Edge consumed by both
};

// Evolution results returned from GPU (counts only - fast)
struct EvolutionResultsHost {
    size_t num_states;
    size_t num_canonical_states;
    size_t num_events;
    size_t num_causal_edges;
    size_t num_branchial_edges;
    size_t num_redundant_edges_skipped;
    // Note: Actual edges available via get_causal_edges() / get_branchial_edges()
};

// Opaque handle to GPU evolution engine (pimpl pattern)
class GPUEvolutionEngineHost {
public:
    GPUEvolutionEngineHost();
    ~GPUEvolutionEngineHost();

    // Non-copyable
    GPUEvolutionEngineHost(const GPUEvolutionEngineHost&) = delete;
    GPUEvolutionEngineHost& operator=(const GPUEvolutionEngineHost&) = delete;

    // Configuration
    void add_rule(const std::vector<std::vector<uint8_t>>& lhs,
                  const std::vector<std::vector<uint8_t>>& rhs,
                  uint8_t first_fresh_var);

    void set_max_steps(uint32_t max);
    void set_max_states(uint32_t max);
    void set_max_events(uint32_t max);
    void set_transitive_reduction(bool enable);

    // Evolution
    void evolve(const std::vector<std::vector<uint32_t>>& initial_edges, uint32_t steps);

    // Results (counts only - fast)
    EvolutionResultsHost get_results() const;
    size_t num_states() const;
    size_t num_canonical_states() const;
    size_t num_events() const;
    size_t num_causal_edges() const;
    size_t num_branchial_edges() const;

    // On-demand edge retrieval (downloads from GPU when called)
    std::vector<HostCausalEdge> get_causal_edges() const;
    std::vector<HostBranchialEdge> get_branchial_edges() const;

private:
    void* impl_;  // Opaque pointer to GPUEvolutionEngine
};

}  // namespace hypergraph::gpu
