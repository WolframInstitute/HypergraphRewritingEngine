#pragma once

#include "hg_gpu/overflow.hpp"   // ErrorKind / OverflowWarning
#include "hg_gpu/types.hpp"

#include <cstdint>
#include <vector>

namespace hg_gpu {

struct RewriteRule {
    std::vector<std::vector<uint8_t>> lhs;
    std::vector<std::vector<uint8_t>> rhs;
    uint8_t num_lhs_vars = 0;
    uint8_t num_rhs_vars = 0;
};

struct EvolveInput {
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> initial_state;
    uint32_t num_steps = 0;
    // Canonicalization is always McKay individualization-refinement (IR):
    // only IR is correct on graphs with non-trivial automorphism.
    CanonicalizationMode      canonicalization     = CanonicalizationMode::Full;
    EventCanonicalizationMode event_canonicalization = EventCanonicalizationMode::None;
    bool transitive_reduction = true;
    // Quotient exploration: expand each canonical state exactly once, at its
    // shortest depth, so the run costs the canonical closure rather than the
    // provenance count. The level-synchronised step loop gives the shortest
    // depth by construction. Canonical states and the (input, output, rule)
    // transition multiset match the CPU engine's quotient mode; exact causal
    // and branchial multisets of the full expansion are reconstructed offline
    // from this skeleton (tools/quotient_reconstruction_probe.cpp). False
    // expands every provenance, the reference/MultiwayReference.wl semantics.
    // GPU defaults to true for bounded state growth on deep evolutions.
    bool explore_from_canonical_states_only = true;

    // Stochastic exploration pruning. Each newly-deduped state at the end
    // of a step is admitted to the next-step frontier with probability
    // `exploration_probability`. Mirrors the CPU
    // ParallelEvolutionEngine::set_exploration_probability option: the
    // state and its event are still recorded in EvolveResult; only the
    // *expansion* from that state is suppressed when the coin lands
    // unfavourable. 1.0 = always explore (default, equivalent to no
    // pruning); 0.0 = never expand any new state (only the initial state
    // is matched). Values outside [0,1] are clamped.
    //
    // `exploration_seed`: deterministic seed for the per-(step, state)
    // coin flip. When 0 (default), a non-deterministic seed is drawn
    // from std::random_device at run start, mirroring the CPU side which
    // uses a thread_local mt19937 with random_device seeding. Set to a
    // non-zero value for reproducible runs.
    float    exploration_probability = 1.0f;
    uint64_t exploration_seed        = 0;

    // Override for EngineConfig::slice_scan_max_edges (0 keeps the default).
    // Tests set a tiny value to force the index-backed match path and the
    // lazy index rebuild on small workloads.
    uint32_t slice_scan_max_edges = 0;
};

struct CanonicalState {
    StateId id = INVALID_ID;
    uint64_t canonical_hash = 0;
    std::vector<std::vector<VertexId>> edges;
};

struct Event {
    EventId id = INVALID_ID;
    EventId canonical_id = INVALID_ID;
    StateId input_state  = INVALID_ID;
    StateId output_state = INVALID_ID;
    RuleId  rule = 0;
    uint32_t step = 0;
    std::vector<EdgeId> consumed_edges;
    std::vector<EdgeId> produced_edges;
};

struct CausalEdge {
    EventId from;
    EventId to;
};

struct BranchialEdge {
    EventId a;
    EventId b;
};

struct EvolveResult {
    std::vector<CanonicalState> states;
    std::vector<Event> events;
    std::vector<CausalEdge> causal_edges;
    std::vector<BranchialEdge> branchial_edges;

    // Capacity overflows observed during the run. Empty on a successful
    // (uncapped) run; otherwise contains one OverflowWarning per
    // (kernel-launch × ErrorKind) overflow event with a `context` string
    // identifying the phase ("match kernel step 3", "rewrite kernel
    // step 5", "ir hash", etc.). The result still contains whatever was
    // successfully computed before the overflow — it's a partial result,
    // not an error. The free `evolve()` wrapper inspects this list to
    // drive its grow-and-retry loop; explicit `Engine.run()` callers can
    // inspect it themselves and decide whether the partial result is
    // good enough.
    std::vector<OverflowWarning> warnings;
};

// Sizing knobs. The auto-tuner (M9) will pick these per device + workload;
// for now, sensible defaults that handle the M1.7 differential corpus.
// All POD so this header stays host-includable without CUDA dependencies.
struct EngineConfig {
    uint32_t max_edges            = 1u << 16;   // 65K edge slots
    uint32_t max_vertices         = 1u << 16;   // 65K vertex IDs (atomic counter ceiling)
    uint32_t max_vertex_slots     = 1u << 18;   // 256K flat vertex-tuple slots (avg arity ≤ 4)
    uint32_t max_states           = 1u << 14;   // 16K state slots
    // CSR-packed per-state edge lists. The flat ids pool is sized for the
    // sum of all state edge counts over the course of the evolution —
    // empirically max_states × avg_state_edges fits most workloads.
    // Replaces the legacy O(max_states × max_edges) bitset.
    uint32_t max_state_edge_total = 1u << 22;   // 4M EdgeId slots (16 MB)
    uint32_t sig_index_buckets    = 1024;       // power of two; sig_hash & (n-1)
    uint32_t inverted_pool        = 1u << 18;   // shared LockFreeList node capacity
    uint32_t sig_index_pool       = 1u << 16;   // shared LockFreeList node capacity
    uint32_t canonical_map_slots  = 1u << 14;   // capacity 4× expected dedup'd states
    uint32_t match_dedup_slots    = 1u << 16;
    // States at or below this edge count are matched by scanning their own CSR
    // slice; the global indices are only consulted (and therefore maintained)
    // once some state exceeds it. See DeviceState::slice_scan_max_edges.
    uint32_t slice_scan_max_edges = 256;
    uint32_t event_canon_slots    = 1u << 16;

    // Event / causal / branchial sizing.
    uint32_t max_events           = 1u << 16;
    uint32_t max_causal_edges     = 1u << 18;
    uint32_t max_branchial_edges  = 1u << 18;
    uint32_t causal_triple_slots  = 1u << 19;   // dedup map for (p,c,e) triples
    uint32_t causal_pair_slots    = 1u << 18;   // dedup map for (p,c) pairs
    uint32_t branchial_pair_slots = 1u << 19;
    uint32_t edge_consumer_nodes  = 1u << 18;   // LockFreeList node pool
    uint32_t state_event_nodes    = 1u << 17;

    // Transitive-reduction sizing. Desc[e] and Anc[e] per event.
    uint32_t tr_desc_nodes  = 1u << 20;
    uint32_t tr_anc_nodes   = 1u << 20;
    uint32_t tr_desc_slots  = 1u << 20;
    uint32_t tr_anc_slots   = 1u << 20;
};

// One-shot evolve: constructs a fresh Engine for `input`, runs once,
// destructs. Each call pays the per-Engine CUDA setup cost (allocating
// pools, indices, lock-free lists — ~5–20 ms on a warmed-up driver, but
// significantly more on the first CUDA call of the process). For repeated
// evaluations of similar workloads use Engine + run() directly to amortise
// the setup.
//
// Capacity-overflow handling: if the kernels report any overflow
// warnings, this wrapper doubles the relevant EngineConfig field(s) and
// retries (up to 6× / 64× total growth). Each retry destructs the old
// Engine and constructs a new one at the bigger size. The returned
// result's `warnings` list is the cumulative trail across retries — so
// the caller can see what was bumped without code-archaeology.
EvolveResult evolve(const EvolveInput& input);

// Engine: persistent device-state container that can run() multiple
// EvolveInputs back-to-back, amortising the per-call CUDA setup. Use
// Engine when benchmarking, when running a parameter sweep, or whenever
// the caller controls the workload lifecycle.
//
// Lifecycle:
//   Engine engine(cfg);          // one-time pool/index allocation
//   for each input:
//       auto result = engine.run(input);
//       // engine is auto-reset between run() calls.
//
// `run()` calls `reset()` internally before processing, so the caller does
// not need to reset between runs. The caller may also call reset()
// explicitly if they want to clear results without running.
//
// engine.run(input) tolerates capacity overflow gracefully — the kernels
// keep running on whatever budget they have, the underlying error
// channel records each overflow into result.warnings, and the partial
// result (whatever states/events/causal/branchial were successfully
// computed before the overflow point) is returned. Engine.run() never
// throws for capacity reasons; it only throws on genuine programmer
// errors (invalid EvolveInput, CUDA driver failures). If the caller
// wants auto-grow-and-retry behaviour, use the free `evolve()` wrapper
// — Engine.run() is the explicit-control path that benchmarks use.
class EngineState;  // forward decl

// Build a sensible EngineConfig for a given workload. Used by the
// one-shot evolve() and exposed publicly so callers building their own
// Engine can size it consistently. Conservative: oversizes to handle
// pre-dedup state-blow-up for the worst step. Auto-tuner (M9) will
// replace this heuristic with cached per-device best-fit values.
EngineConfig config_from_input(const EvolveInput& input);

class Engine {
public:
    explicit Engine(EngineConfig cfg);
    ~Engine();
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    EvolveResult run(const EvolveInput& input);
    void reset();

    const EngineConfig& config() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace hg_gpu
