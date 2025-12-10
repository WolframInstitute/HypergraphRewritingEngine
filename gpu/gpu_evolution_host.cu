// Implementation of the pure C++ host interface for GPU Evolution Engine

#include "gpu_evolution_host.hpp"
#include "evolution.cuh"

namespace hypergraph::gpu {

GPUEvolutionEngineHost::GPUEvolutionEngineHost() {
    impl_ = new GPUEvolutionEngine();
}

GPUEvolutionEngineHost::~GPUEvolutionEngineHost() {
    delete static_cast<GPUEvolutionEngine*>(impl_);
}

void GPUEvolutionEngineHost::add_rule(
    const std::vector<std::vector<uint8_t>>& lhs,
    const std::vector<std::vector<uint8_t>>& rhs,
    uint8_t first_fresh_var
) {
    static_cast<GPUEvolutionEngine*>(impl_)->add_rule(lhs, rhs, first_fresh_var);
}

void GPUEvolutionEngineHost::set_max_steps(uint32_t max) {
    static_cast<GPUEvolutionEngine*>(impl_)->set_max_steps(max);
}

void GPUEvolutionEngineHost::set_max_states(uint32_t max) {
    static_cast<GPUEvolutionEngine*>(impl_)->set_max_states(max);
}

void GPUEvolutionEngineHost::set_max_events(uint32_t max) {
    static_cast<GPUEvolutionEngine*>(impl_)->set_max_events(max);
}

void GPUEvolutionEngineHost::set_transitive_reduction(bool enable) {
    static_cast<GPUEvolutionEngine*>(impl_)->set_transitive_reduction(enable);
}

void GPUEvolutionEngineHost::evolve(
    const std::vector<std::vector<uint32_t>>& initial_edges,
    uint32_t steps
) {
    static_cast<GPUEvolutionEngine*>(impl_)->evolve(initial_edges, steps);
}

EvolutionResultsHost GPUEvolutionEngineHost::get_results() const {
    auto r = static_cast<GPUEvolutionEngine*>(impl_)->get_results();
    return {
        r.num_states,
        r.num_canonical_states,
        r.num_events,
        r.num_causal_edges,
        r.num_branchial_edges,
        r.num_redundant_edges_skipped
    };
}

size_t GPUEvolutionEngineHost::num_states() const {
    return static_cast<GPUEvolutionEngine*>(impl_)->num_states();
}

size_t GPUEvolutionEngineHost::num_canonical_states() const {
    return static_cast<GPUEvolutionEngine*>(impl_)->num_canonical_states();
}

size_t GPUEvolutionEngineHost::num_events() const {
    return static_cast<GPUEvolutionEngine*>(impl_)->num_events();
}

size_t GPUEvolutionEngineHost::num_causal_edges() const {
    return static_cast<GPUEvolutionEngine*>(impl_)->num_causal_edges();
}

size_t GPUEvolutionEngineHost::num_branchial_edges() const {
    return static_cast<GPUEvolutionEngine*>(impl_)->num_branchial_edges();
}

std::vector<HostCausalEdge> GPUEvolutionEngineHost::get_causal_edges() const {
    return static_cast<GPUEvolutionEngine*>(impl_)->get_causal_edges();
}

std::vector<HostBranchialEdge> GPUEvolutionEngineHost::get_branchial_edges() const {
    return static_cast<GPUEvolutionEngine*>(impl_)->get_branchial_edges();
}

}  // namespace hypergraph::gpu
