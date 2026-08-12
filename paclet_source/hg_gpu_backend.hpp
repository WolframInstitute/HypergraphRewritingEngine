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

    // THE TWO MODE CODES DO NOT SHARE AN ORDER, and that is the whole reason they are named.
    // event is None/Full/Automatic; state is None/Automatic/Full, following
    // hg_gpu::CanonicalizationMode. Written as bare integers in one file and read as bare
    // integers in another, the two orders look interchangeable and are not: the encoder collapsed
    // an Automatic request to 1, the decoder read 1 as Full, and a caller asking for Automatic
    // event identity silently received a coarser one.
    struct EventCanonCode { static constexpr int kNone = 0, kFull = 1, kAutomatic = 2; };
    struct StateCanonCode { static constexpr int kNone = 0, kAutomatic = 1, kFull = 2; };

    int steps = 0;
    int event_canon_mode = EventCanonCode::kNone;
    int state_canon_mode = StateCanonCode::kFull;
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

    // Graph properties (StatesGraph / CausalGraph / BranchialGraph / Evolution* and
    // their Structure variants) and the options that shape them. Marshalled through
    // the shared hgmarshal::build_graph_data so GPU GraphData matches the CPU FFI.
    std::vector<std::string> graph_properties;
    bool edge_deduplication = true;
    int  branchial_step = 0;      // 0=all, >0 1-based step, <0 from end (-1=final)
    bool show_genesis_events = false;

    // THE SESSION ENVELOPE, carried through so the device can answer the same four verbs the
    // host does. Empty `op` means Evolve, which is a one-shot run and the whole of the older
    // protocol, so a caller sending neither field is served exactly as before.
    //
    // ONE SESSION PER PROCESS. The worker runs jobs serially against one device engine, and a
    // session pins that engine (a rebuild would drop its accumulated states while returning
    // something shaped like a continuation), so a second Open before a Close is refused rather
    // than silently answering about a different evolution.
    std::string session_op;
    uint64_t    session_handle = 0;

    // Which frontier states a `Step` expands, if the caller named a subset. Carried so the
    // device can REFUSE a steered continuation rather than run it unsteered -- the device
    // session holds its frontier as device state ids with no host-visible identity, so the
    // selection cannot be resolved there, and running it anyway would explore the branches the
    // caller asked to leave alone and answer a different question in the right shape.
    std::vector<int64_t> session_from;
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
