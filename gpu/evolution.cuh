#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <set>
#include "types.cuh"
#include "gpu_evolution_host.hpp"  // For HostCausalEdge, HostBranchialEdge
#include "hash_table.cuh"
#include "memory_pool.cuh"
#include "work_queue.cuh"
#include "wl_hash.cuh"
#include "pattern_match.cuh"
#include "rewrite.cuh"
#include "causal.cuh"
#include "signature_index.cuh"

namespace hypergraph::gpu {

// =============================================================================
// GPU Evolution Engine
// =============================================================================
// Main controller for GPU-based hypergraph evolution.
// Uses persistent megakernel with work queues for minimal CPU-GPU communication.

class GPUEvolutionEngine {
public:
    GPUEvolutionEngine();
    ~GPUEvolutionEngine();

    // Non-copyable
    GPUEvolutionEngine(const GPUEvolutionEngine&) = delete;
    GPUEvolutionEngine& operator=(const GPUEvolutionEngine&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    // Add a rewrite rule (host-side, before evolution)
    void add_rule(const std::vector<std::vector<uint8_t>>& lhs,
                  const std::vector<std::vector<uint8_t>>& rhs,
                  uint8_t first_fresh_var);

    void set_max_steps(uint32_t max) { max_steps_ = max; }
    void set_max_states(uint32_t max) { max_states_ = max; }
    void set_max_events(uint32_t max) { max_events_ = max; }
    void set_transitive_reduction(bool enable) { tr_enabled_ = enable; }
    void set_match_forwarding(bool enable) { match_forwarding_enabled_ = enable; }
    void set_batched_matching(bool enable) { batched_matching_ = enable; }
    void set_event_canonicalization(EventCanonicalizationMode mode) { event_canon_mode_ = mode; }
    void set_task_granularity(TaskGranularity granularity) { task_granularity_ = granularity; }

    // =========================================================================
    // Evolution
    // =========================================================================

    // Run evolution from initial edges
    void evolve(const std::vector<std::vector<uint32_t>>& initial_edges, uint32_t steps);

    // =========================================================================
    // Results
    // =========================================================================

    EvolutionResults get_results() const;

    size_t num_states() const { return results_.num_states; }
    size_t num_canonical_states() const { return results_.num_canonical_states; }
    size_t num_events() const { return results_.num_events; }
    size_t num_causal_edges() const { return results_.num_causal_edges; }
    size_t num_branchial_edges() const { return results_.num_branchial_edges; }
    size_t num_redundant_edges_skipped() const { return results_.num_redundant_edges_skipped; }

    // On-demand edge retrieval (to minimize transfer costs when only counts needed)
    std::vector<HostCausalEdge> get_causal_edges() const;
    std::vector<HostBranchialEdge> get_branchial_edges() const;

private:
    // =========================================================================
    // Initialization
    // =========================================================================

    void init_gpu_resources();
    void destroy_gpu_resources();
    void upload_rules();
    StateId upload_initial_state(const std::vector<std::vector<uint32_t>>& edges);
    void build_adjacency_index(const std::vector<std::vector<uint32_t>>& edges, uint32_t num_vertices);
    void download_results();

    // =========================================================================
    // Evolution Phases
    // =========================================================================

    void run_megakernel_evolution();

    // =========================================================================
    // Member Variables
    // =========================================================================

    // Configuration
    std::vector<DeviceRewriteRule> h_rules_;
    uint32_t max_steps_;
    uint32_t max_states_;
    uint32_t max_events_;
    bool tr_enabled_;
    bool match_forwarding_enabled_;
    bool batched_matching_;
    EventCanonicalizationMode event_canon_mode_;
    TaskGranularity task_granularity_;
    bool initialized_;

    // Device memory
    GPUMemoryManager memory_;
    DeviceRewriteRule* d_rules_;
    uint32_t num_rules_;

    // Data structures
    DeviceEdges d_edges_;
    StatePool d_state_pool_;
    EventPool d_event_pool_;
    DeviceAdjacency d_adjacency_;

    // Hash tables
    GPUHashTable<> canonical_state_map_;
    GPUHashSet<> seen_match_hashes_;
    GPUHashSet<> seen_event_hashes_;    // For event deduplication (ByState/ByStateAndEdges)
    EdgeProducerMap edge_producer_map_;

    // Work queues (context-aware for match forwarding)
    WorkQueue<MatchTaskWithContext> match_queue_;
    WorkQueue<RewriteTaskWithMatch> rewrite_queue_;
    TerminationDetector termination_;

    // Match forwarding: per-state match storage (indexed by state ID)
    StateMatchList* d_state_matches_;         // Array of per-state match lists
    StateChildrenList* d_state_children_;     // Array of per-state children lists
    ParentInfo* d_state_parents_;             // Array of parent info per state

    // Global epoch counter for push/pull coordination
    uint64_t* d_global_epoch_;

    // Node allocators for lock-free lists
    MatchNode* d_match_node_pool_;
    uint32_t* d_match_node_next_;
    ChildNode* d_child_node_pool_;
    uint32_t* d_child_node_next_;

    // Causal graph (phased - kept as fallback)
    CausalGraphGPU causal_graph_;
    CausalEdge* d_causal_edges_;
    uint32_t* d_num_causal_edges_;
    BranchialEdge* d_branchial_edges_;
    uint32_t* d_num_branchial_edges_;

    // Online causal/branchial tracking
    OnlineCausalGraph* d_online_causal_;          // Device-resident struct
    uint32_t* d_edge_consumer_heads_;             // [MAX_EDGES]
    ConsumerNode* d_consumer_nodes_;              // [MAX_CONSUMER_NODES]
    uint32_t* d_num_consumer_nodes_;
    uint32_t* d_state_event_heads_;               // [MAX_STATES]
    StateEventNode* d_state_event_nodes_;         // [MAX_STATE_EVENT_NODES]
    uint32_t* d_num_state_event_nodes_;
    uint64_t* d_seen_causal_triples_;             // Dedup hash set for (producer, consumer, edge)
    uint64_t* d_seen_branchial_pairs_;            // Dedup hash set for (event1, event2)
    uint64_t* d_seen_causal_event_pairs_;         // Dedup hash set for (producer, consumer) pairs
    uint32_t* d_num_causal_event_pairs_;          // Counter for unique causal event pairs
    static constexpr uint32_t CAUSAL_DEDUP_CAPACITY = 65536;  // Power of 2
    static constexpr uint32_t BRANCHIAL_DEDUP_CAPACITY = 65536;  // Power of 2
    static constexpr uint32_t CAUSAL_PAIRS_CAPACITY = 65536;  // Power of 2
    bool use_online_causal_;                      // Flag to switch between online/phased

    // Signature-based index for pattern matching acceleration
    GPUSignatureIndexBuilder sig_index_builder_;
    DeviceSignatureIndex d_sig_index_;
    DeviceInvertedVertexIndex d_inv_vertex_index_;

    // WL hasher
    WLHasher wl_hasher_;

    // Scratch buffers
    DeviceMatch* d_matches_;
    RewriteOutput* d_rewrite_outputs_;
    uint64_t* d_wl_scratch_;
    uint32_t wl_scratch_per_block_;  // Size per block for WL scratch
    uint32_t matches_per_warp_;      // Matches buffer size per warp (determined at runtime)

    // Results
    EvolutionResults results_;

    // CUDA resources
    cudaStream_t stream_;
    int num_sms_;
};

// =============================================================================
// Megakernel Declaration (Task-Parallel with Match Forwarding)
// =============================================================================
// Persistent kernel with work queues for eager rewriting:
// - MATCH tasks find matches, store for forwarding, push to children, spawn REWRITEs
// - REWRITE tasks apply matches, register with parent, pull from ancestors, spawn MATCHes

__global__ void evolution_megakernel_with_forwarding(
    EvolutionContext* ctx,
    WorkQueueView<MatchTaskWithContext>* match_queue,
    WorkQueueView<RewriteTaskWithMatch>* rewrite_queue,
    StateMatchList* state_matches,
    StateChildrenList* state_children,
    ParentInfo* state_parents,
    uint64_t* global_epoch,
    MatchNode* match_node_pool,
    uint32_t* match_node_next,
    ChildNode* child_node_pool,
    uint32_t* child_node_next,
    TerminationDetectorView* termination,
    uint32_t max_steps,
    bool match_forwarding_enabled,
    bool batched_matching
);

}  // namespace hypergraph::gpu
