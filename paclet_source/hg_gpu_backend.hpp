#pragma once

#ifdef HG_GPU_BACKEND

#include "hg_core.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// A parsed evolution job routed to the GPU backend. Field names mirror the
// FFI's parsed option variables so run_rewriting_core can hand them straight
// through. Only present in the GPU binary (hg_evolve_gpu), which links hg_gpu.
struct GpuJob {
    // Rules: name -> {lhs_edges, rhs_edges}; each edge is a list of vertex ids
    // that double as pattern-variable indices (same convention as the FFI).
    const std::vector<std::pair<std::string,
        std::vector<std::vector<std::vector<int64_t>>>>>& rules;
    // Each entry is one initial (root) state: a list of edges.
    const std::vector<std::vector<std::vector<int64_t>>>& initial_states;

    int steps = 0;
    int event_canon_mode = 0;   // 0 None, 1 Full, 2 Automatic
    int state_canon_mode = 2;   // 0 None, 1 Automatic, 2 Full (hg_gpu::CanonicalizationMode order)
    bool transitive_reduction = true;
    bool explore_from_canonical_states_only = false;
    bool quotient_initial_states = false;
    double exploration_probability = 1.0;
    uint64_t max_device_memory_bytes = 0;

    // Output selection (mirrors the FFI include_* flags).
    bool include_states = true;
    bool include_events = true;
    bool include_causal_edges = true;
    bool include_branchial_edges = true;
    bool include_canonical_hashes = false;
};

// Run the job on the GPU (hg_gpu::evolve) and marshal the result into the same
// WXF association the CPU FFI produces (States / Events / CausalEdges /
// BranchialEdges / Num* [+ Warnings on a capacity overflow]). Because the GPU
// result is a raw per-provenance space while the FFI emits a canonical-class
// space, states are grouped by their host-recomputed IR canonical hash and one
// entry is emitted per class; events stay raw so multiplicity (and counts)
// match the CPU. Throws std::exception on error.
std::vector<uint8_t> run_gpu_evolution(const GpuJob& job, const HostBridge& host);

#endif  // HG_GPU_BACKEND
