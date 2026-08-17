#include "hgcommon/namespace.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/exploration.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/rewrite.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

namespace {

}  // namespace (close anon — config_from_input has external linkage)

EngineConfig config_from_input(const EvolveInput& in) {
    EngineConfig cfg;
    size_t n_init   = in.initial_state.size();
    uint32_t steps  = in.num_steps;
    // The replay descends one stack frame per reconstruction depth, so the step count sizes the
    // per-thread stack as well as the pools.
    cfg.reconstruction_max_depth = steps;

    // Estimate growth per step. A typical Wolfram-style rule produces 2–4
    // new edges per match; matches grow ~linearly with edge count; states
    // grow by a branching factor.
    uint32_t growth = 1u;
    for (uint32_t s = 0; s < steps && growth < 32u; ++s) growth *= 4u;

    // Raw (pre-dedup) state production in a single step can blow past
    // canonical final-step counts by 10× due to within-step branching
    // before dedup collapses isomorphic states. CSR per-state edge lists
    // (Stream 2) size linearly in the total edge-slot count rather than
    // quadratically in max_states * max_edges, so max_states and max_edges
    // can be large without a memory blow-up.
    uint32_t expected_edges  = std::max<uint32_t>(1u << 20, static_cast<uint32_t>(n_init) * growth * 512u);
    uint32_t expected_states = std::max<uint32_t>(1u << 17, static_cast<uint32_t>(n_init) * growth * 32u);

    cfg.max_edges              = expected_edges;
    cfg.max_states             = expected_states;
    cfg.max_vertex_slots       = expected_edges * 4u;
    // Total edge-ID slots across all states' CSR rows. Each rewrite
    // consumes parent.count + rhs slots; assume average state size ~
    // max(n_init, 64) and room for ~16 edges per state on average.
    uint32_t avg_state_edges   = std::max<uint32_t>(64u,
                                 static_cast<uint32_t>(n_init) + growth * 16u);
    cfg.max_state_edge_total   = static_cast<uint32_t>(
        std::min<uint64_t>(
            static_cast<uint64_t>(expected_states) * avg_state_edges,
            1ull << 30));  // ≤ 4 GB × sizeof(EdgeId)=4 → ≤ 1G slots
    // Each event allocates ≤ kMaxVars fresh vertices, so vertex IDs bound
    // by n_init-vertices + events × kMaxVars. Be generous.
    cfg.max_vertices           = std::max<uint32_t>(expected_edges,
                                 static_cast<uint32_t>(n_init) * 4u + expected_states * 4u);
    cfg.sig_index_buckets      = 1024;
    cfg.sig_index_pool         = expected_edges * 2u;
    cfg.inverted_pool          = expected_edges * 4u;

    if (in.slice_scan_max_edges) cfg.slice_scan_max_edges = in.slice_scan_max_edges;
    if (in.max_blocks_per_launch) cfg.max_blocks_per_launch = in.max_blocks_per_launch;

    uint32_t expected_events   = expected_states;
    cfg.max_events             = expected_events;
    cfg.max_causal_edges       = expected_events * 8u;
    cfg.max_branchial_edges    = expected_events * 8u;
    cfg.causal_triple_slots    = expected_events * 16u;
    cfg.causal_pair_slots      = expected_events * 8u;
    cfg.branchial_pair_slots   = expected_events * 16u;
    cfg.edge_consumer_nodes    = expected_edges * 4u;
    cfg.branchial_index_buckets = 1u << 20;
    cfg.branchial_index_nodes   = expected_events * 4u;
    // One preds node per unique kept causal pair; kept pairs are a subset of causal pairs.
    cfg.tr_preds_nodes         = expected_events * 8u;
    return cfg;
}

// Which components of the shared event-identity lattice a mode asks for.
EventSignatureKeys event_keys_for(EventCanonicalizationMode m) {
    switch (m) {
        case EventCanonicalizationMode::Full:      return EVENT_SIG_FULL;
        case EventCanonicalizationMode::Automatic: return EVENT_SIG_AUTOMATIC;
        case EventCanonicalizationMode::None:
        default:                                   return EVENT_SIG_NONE;
    }
}

namespace {  // re-open anon namespace for kernel + helper definitions

// splitmix64 — deterministic, header-quality scalar hash. Used to derive
// a per-(seed, step, sid) coin-flip value for stochastic exploration
// pruning. Cheap (~1 ns) and avoids needing a curand state per thread.
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}


}  // namespace


namespace {

}  // namespace

// ---------------------------------------------------------------------------
// Engine::Impl
//
// Holds every device-side resource that's reusable across run() calls. The
// per-input data (initial state, rules, frontier seeding) is uploaded /
// reset on each run() — pools, indices, and lock-free lists are NOT
// reallocated.
//
// On overflow, the underlying error channel throws via
// engine_state.throw_on_errors with a specific pool name; caller can
// destruct the Engine and construct a new one with a larger config.
// (Auto-grow-on-overflow is Stream 5.)
// ---------------------------------------------------------------------------
struct Engine::Impl {
    explicit Impl(EngineConfig cfg)
        : cfg_(cfg)
        , state_(cfg)
        , matches_(cfg.max_states * 8u)
    {}

    ~Impl() {
        if (d_rules_) cudaFree(d_rules_);
    }

    void reset() {
        state_.clear();
        matches_.reset();
    }

    EvolveResult run(const EvolveInput& in, SessionView* session = nullptr,
                     uint32_t start_step = 0);

    EngineConfig                       cfg_;
    EngineState                        state_;
    Pool<MatchRecord>                  matches_;
    // Engine-lifetime, cleared per run: rebuilding its maps costs tens of MB of cudaMalloc
    // per evolve. Constructed on the first run that routes quotient causal.
    std::unique_ptr<QcState>           qc_state_;
    std::unique_ptr<QeState>           qe_state_;
    DeviceRule*                        d_rules_          = nullptr;
    uint32_t                           d_rules_capacity_ = 0;
};

Engine::Engine(EngineConfig cfg) : impl_(new Impl(cfg)) {}
Engine::~Engine() { delete impl_; }
void Engine::reset() { impl_->reset(); }
const EngineConfig& Engine::config() const { return impl_->cfg_; }
EvolveResult Engine::run(const EvolveInput& in, SessionView* session,
                         uint32_t start_step) {
    return impl_->run(in, session, start_step);
}

namespace {
// Fill each state's dedup key with a unique value so NONE of them merge —
}  // namespace

EvolveResult Engine::Impl::run(const EvolveInput& in, SessionView* session,
                               uint32_t start_step) {
    // Reset device state from any prior run() -- EXCEPT when continuing a session. A Step's
    // accumulated states ARE the graph being extended, and the frontier it seeds from holds ids
    // into those pools, so clearing them leaves the run seeding ids that no longer name
    // anything and it produces nothing. Opening (start_step 0) still resets, which is what
    // keeps a session from inheriting a previous job's graph.
    if (start_step == 0) reset();

    EvolveResult out;
    if (in.rules.empty() && in.num_steps == 0 && in.initial_state.empty()) {
        return out;
    }

    auto t_total_start = std::chrono::steady_clock::now();
    auto t_init_start = std::chrono::steady_clock::now();
    EngineState& engine = state_;
    // WHICH RUNS RECONSTRUCT. This must be the same predicate the host uses
    // (ParallelEvolutionEngine::configure_identity_and_quotient), because it decides where event
    // identity comes from: the class frame, or each raw state's own labelling. The device used
    // to require quotient EXPLORATION, so an Automatic-identity run under full capture took the
    // raw-labelling path here and the class-frame path on the host -- which is the whole of the
    // CPU 21 / GPU 23 divergence.
    //
    // Full state canonicalization is required by both: the reconstruction is defined over
    // canonical states and their edge orbits, and no other mode computes orbit tables.
    const bool qc_route = in.canonicalization == CanonicalizationMode::Full &&
                          in.num_steps > 0 &&
                          (in.explore_from_canonical_states_only ||
                           event_keys_for(in.event_canonicalization) == EVENT_SIG_AUTOMATIC);
    engine.set_quotient_causal(qc_route);
    engine.set_record_set(in.record);
    engine.set_tr_enabled(in.transitive_reduction && !qc_route);
    if (qc_route) { engine.ensure_edge_orbits(); engine.ensure_edge_ranks(); }
    double t_init = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_init_start).count();

    // Lazy index maintenance: skip index inserts until some state exceeds the
    // slice-scan threshold (the match kernels never read the indices below it).
    // Normalize to a list of initial states (plural takes precedence).
    const std::vector<std::vector<std::vector<VertexId>>> roots =
        !in.initial_states.empty()
            ? in.initial_states
            : std::vector<std::vector<std::vector<VertexId>>>{in.initial_state};
    size_t max_root_edges = 0;
    for (const auto& r : roots) max_root_edges = std::max(max_root_edges, r.size());
    // Index maintenance cannot flip on mid-run: the evolution is one launch with no host in
    // the loop, so there is no point at which a lazy flip could happen -- and an index that
    // missed inserts is not stale but WRONG (missed candidates).
    // So the decision is made up front, and it can be made exactly: a state's edge count along
    // any path is bounded by root_edges + steps * max(rhs - lhs) over the rules, and the match
    // kernels read the indices only for states past the slice-scan threshold. When even that
    // bound cannot reach the threshold, no state in the run can ever be matched through the
    // indices, and every insert would be bought for nobody.
    //
    // Decided HERE, before the upload, and that placement is the point. upload_initial_states
    // populates the indices only when maintenance is already on, and rebuild_indices INSERTS
    // without clearing -- so turning maintenance on after the upload and rebuilding puts every
    // root edge in its bucket twice, which surfaces as duplicate candidates, duplicate matches
    // and duplicate events on any state large enough to be matched through the indices.
    size_t max_growth = 0;
    for (const auto& r : in.rules) {
        const size_t lhs = r.lhs.size(), rhs = r.rhs.size();
        if (rhs > lhs) max_growth = std::max(max_growth, rhs - lhs);
    }
    const size_t max_state_edges =
        max_root_edges + max_growth * static_cast<size_t>(in.num_steps);
    engine.set_maintain_indices(max_state_edges > engine.config_slice_scan_max_edges());
    // Continuing: the roots are already in the pools from the call that opened the session, so
    // uploading them again would add a second copy of every root and re-seed the evolution from
    // depth 0 alongside the frontier.
    const uint32_t num_roots = (start_step == 0)
                                   ? upload_initial_states(engine, roots)
                                   : static_cast<uint32_t>(roots.size());

    // Upload rules. Resize the device-side rules buffer if this run has more
    // rules than any prior run. Re-upload every run so the caller can pass a
    // different rule set per call without surprises.
    std::vector<DeviceRule> rules;
    rules.reserve(in.rules.size());
    for (const auto& r : in.rules) rules.push_back(make_device_rule(r));
    const uint32_t num_rules = static_cast<uint32_t>(rules.size());

    if (num_rules > d_rules_capacity_) {
        if (d_rules_) cudaFree(d_rules_);
        d_rules_ = nullptr;
        d_rules_capacity_ = 0;
        if (num_rules > 0) {
            HG_CUDA_CHECK(cudaMalloc(&d_rules_, sizeof(DeviceRule) * num_rules), "d_rules alloc");
            d_rules_capacity_ = num_rules;
        }
    }
    if (num_rules > 0) {
        HG_CUDA_CHECK(cudaMemcpy(d_rules_, rules.data(), sizeof(DeviceRule) * num_rules,
                         cudaMemcpyHostToDevice), "d_rules copy");
    }
    DeviceRule* d_rules = d_rules_;

    const EngineConfig& cfg = engine.config();
    Pool<MatchRecord>& matches = matches_;

    // The per-state hash lives on DeviceState, not in this driver: the persistent kernel
    // writes it when it hashes a child for dedup, and the readback at the end reads that same
    // array. A buffer owned by Engine::Impl would have made the assembly private to the host.
    uint64_t* d_state_hashes  = engine.device().state_canonical_hash;

    // Resolve exploration-probability parameters once per run.
    //   threshold == UINT32_MAX → fast path: always explore (zero overhead).
    //   threshold == 0          → never expand any new state.
    //   else                    → admit with probability ≈ threshold / 2^32.
    float clamped_p = in.exploration_probability;
    if (!(clamped_p > 0.0f)) clamped_p = 0.0f;
    if (clamped_p > 1.0f)    clamped_p = 1.0f;
    uint32_t explore_threshold_u32;
    if (clamped_p >= 1.0f) {
        explore_threshold_u32 = 0xFFFFFFFFu;
    } else if (clamped_p <= 0.0f) {
        explore_threshold_u32 = 0u;
    } else {
        explore_threshold_u32 = static_cast<uint32_t>(
            static_cast<double>(clamped_p) * 4294967296.0);
    }
    // The quotient-causal DP's device structures, one body of state whichever scheduler
    // drives it; token-sized when the route is off, engine-lifetime and cleared per run.
    if (!qc_state_ || qc_state_->enabled() != qc_route)
        qc_state_ = std::make_unique<QcState>(qc_route, cfg.max_events);
    else
        qc_state_->clear();
    QcView qc_view = qc_state_->view(in.num_steps, state_.qe_max_recursion_depth());

    // The class-frame expansion capture rides the same route decision as the causal DP: both
    // ARE the quotient reconstruction, and a run that reconstructs causality is exactly a run
    // whose event identity comes from the class frame rather than each raw state's labelling.
    if (!qe_state_ || qe_state_->enabled() != qc_route)
        qe_state_ = std::make_unique<QeState>(qc_route, cfg.max_events);
    else
        qe_state_->clear();
    // Capture always; REPLAY only when the caller records something the raw unfolding answers.
    // The replay is the device twin of the host's instance cascade, and it is the term measured
    // exponential in depth against an answer that is linear (b98a943c). Capture is untouched, so
    // Automatic event identity -- signed from the class frame -- is unchanged either way.
    const bool qe_replay = in.record.causal || in.record.branchial || in.record.raw_events;
    QeView qe_view = qe_state_->view(in.num_steps, event_keys_for(in.event_canonicalization),
                                     state_.qe_max_recursion_depth(), qe_replay);

    uint64_t resolved_seed = in.exploration_seed;
    if (resolved_seed == 0 && clamped_p < 1.0f) {
        std::random_device rd;
        resolved_seed = (static_cast<uint64_t>(rd()) << 32) | rd();
        if (resolved_seed == 0) resolved_seed = 0xA5A5A5A5A5A5A5A5ull;
    }

    const bool dbg = std::getenv("HG_GPU_DBG_TIME") != nullptr;
    double t_match = 0, t_rewrite = 0, t_hash = 0, t_dedup = 0;

    // Cache the running state_count on host so we only D2H once per step
    // (instead of twice via num_states_host around the rewrite call).
    uint32_t state_count_host = engine.num_states_host();

    // Tag warnings with a step-aware context, e.g. "match kernel step 3".
    // Reused across all four phases per step.
    char ctx_buf[64];

    // The whole evolution in ONE launch: the device decides what work exists, who takes it, and
    // when it is finished. Everything below the loop is unchanged -- the readback is post-hoc
    // and reads the same per-state hash array either scheduler filled, which is what makes one
    // assembly path serve both.
    {
        const EventSignatureKeys ekeys = event_keys_for(in.event_canonicalization);

        // Automatic event identity keys on the canonical ranks of the consumed and produced
        // edges, which live in a per-edge-slot array no other mode reads. Taken here, once the
        // mode is known, so a run identifying events by their endpoint states alone is not
        // charged four bytes per edge slot for it.
        if (ekeys & (hgcommon::EventKey_ConsumedEdges | hgcommon::EventKey_ProducedEdges)) {
            engine.ensure_edge_ranks();
        }

        std::vector<StateId> roots(num_roots);
        for (uint32_t i = 0; i < num_roots; ++i) roots[i] = i;

        // The device arena the exact hash claims its IR scratch from. Sized from the GRID: each
        // resident worker holds one slot at a time and grows it to the largest state it
        // personally canonicalizes, so demand scales with the worker count rather than with the
        // state budget. Exhaustion is a recorded capacity overflow (kIRArenaExhausted, which the
        // wrapper can grow and retry), never a coarser hash.
        DeviceArena& arena = engine.ir_arena(
            persistent_arena_words(cfg.ir_arena_share_words, default_persistent_grid()));

        PersistentEvolveStats st = run_persistent_evolve(
            engine, rules, roots, in.num_steps, matches, arena,
            /*dedup=*/in.explore_from_canonical_states_only,
            explore_threshold_u32, resolved_seed,
            in.canonicalization, ekeys, /*blocks=*/0,
            /*quotient_roots=*/in.quotient_initial_states,
            qc_route ? &qc_view : nullptr,
            qc_route ? &qe_view : nullptr,
            session, start_step);

        state_count_host = engine.num_states_host();
        // One transfer for every scalar below, instead of one per field. The counters share a
        // single device allocation precisely so this is possible.
        const auto qc_counts = qc_route ? qe_state_->counters_host()
                                        : hg_gpu::QeState::Counters{};
        out.expansion_matches   = qc_route ? qe_state_->num_matches_host()   : 0u;
        out.expansion_instances = qc_counts.instances;
        out.reconstructed_raw_events = qc_counts.raw_events;
        // The REPLAY is what produces a reconstructed answer, so this reports the replay and
        // not merely the route. A run that captured the class frames but never replayed them
        // has no reconstructed raw events, and a caller that read this as "reconstruction ran"
        // would treat empty relations as a result rather than as an artifact not requested.
        out.reconstruction_ran = qc_route && qe_replay;
        // Under EVENT_SIG_NONE no identity is computed and every application is its own event,
        // so the raw count IS the answer -- the same rule as the host's num_reconstructed_events.
        out.reconstructed_events =
            !qc_route ? 0u
            : (event_keys_for(in.event_canonicalization) == EVENT_SIG_NONE
                   ? qc_counts.raw_events
                   : qc_counts.canon_events);
        out.reconstructed_causal_pairs = qc_counts.causal_pairs;
        out.reconstructed_causal_edges = qc_counts.causal_edges;
        out.reconstructed_causal_pairs_reduced =
            qc_counts.reduced_pairs;
        out.reconstructed_branchial = qc_counts.branchial;
        if (qc_route)
            qe_state_->reconstructed_pairs_host(out.reconstructed_causal_relation,
                                                out.reconstructed_causal_relation_reduced,
                                                out.reconstructed_branchial_relation);
        out.frame_alignments = qc_counts.aligned;
        out.frame_align_failures = qc_counts.align_failures;
        engine.collect_warnings_into(out.warnings, "persistent evolve");
        if (dbg) {
            const double tot = double(st.cycles_match) + double(st.cycles_rewrite) +
                               double(st.cycles_canon) + double(st.cycles_idle) +
                               double(st.cycles_wait);
            const double pct = tot > 0 ? 100.0 / tot : 0.0;
            std::fprintf(stderr,
                         "[persistent] states=%u matches=%u arena_words=%llu cycles: "
                         "match=%.1f%% rewrite=%.1f%% canon=%.1f%% idle=%.1f%% wait=%.1f%%\n",
                         st.states_after, st.matches_found,
                         (unsigned long long)st.arena_words_used,
                         st.cycles_match * pct, st.cycles_rewrite * pct,
                         st.cycles_canon * pct, st.cycles_idle * pct,
                         st.cycles_wait * pct);
            const double rw = double(st.cycles_rw_sub[0]) + double(st.cycles_rw_sub[1]) +
                              double(st.cycles_rw_sub[2]) + double(st.cycles_rw_sub[3]) +
                              double(st.cycles_rw_sub[4]) + double(st.cycles_rw_sub[5]);
            const double rpct = rw > 0 ? 100.0 / rw : 0.0;
            std::fprintf(stderr,
                         "[persistent rw] reserve=%.1f%% emit=%.1f%% csr=%.1f%% event=%.1f%% "
                         "causal=%.1f%% branchial=%.1f%%\n",
                         st.cycles_rw_sub[0] * rpct, st.cycles_rw_sub[1] * rpct,
                         st.cycles_rw_sub[2] * rpct, st.cycles_rw_sub[3] * rpct,
                         st.cycles_rw_sub[4] * rpct, st.cycles_rw_sub[5] * rpct);
        }
    }

    auto t_readback_start = std::chrono::steady_clock::now();

    // Readback — hashes were persisted across steps, no re-hashing needed.
    uint32_t total_states = engine.num_states_host();
    std::vector<uint64_t> h_hashes(total_states);
    if (total_states > 0) {
        HG_CUDA_CHECK(cudaMemcpy(h_hashes.data(), d_state_hashes, sizeof(uint64_t) * total_states,
                         cudaMemcpyDeviceToHost), "final hashes d2h");
    }
    double t_readback_hashes = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_readback_start).count();

    auto t_readback_states_start = std::chrono::steady_clock::now();
    auto all_edges = engine.all_state_edges_host();
    out.states.reserve(all_edges.size());
    for (uint32_t s = 0; s < all_edges.size(); ++s) {
        CanonicalState cs;
        cs.id             = s;
        cs.canonical_hash = (s < h_hashes.size()) ? h_hashes[s] : 0;
        cs.edges          = std::move(all_edges[s]);
        out.states.push_back(std::move(cs));
    }

    double t_readback_states = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_readback_states_start).count();

    auto t_readback_evcb_start = std::chrono::steady_clock::now();
    auto d_events = engine.events_host();
    out.events.reserve(d_events.size());
    for (const auto& de : d_events) {
        Event e;
        e.id            = de.id;
        e.canonical_id  = de.canonical_id;
        e.signature     = de.signature;
        e.input_state   = de.input_state;
        e.output_state  = de.output_state;
        e.rule          = de.rule;
        e.step          = de.step;
        for (uint8_t i = 0; i < de.num_consumed; ++i) e.consumed_edges.push_back(de.consumed_edges[i]);
        for (uint8_t i = 0; i < de.num_produced; ++i) e.produced_edges.push_back(de.produced_edges[i]);
        out.events.push_back(std::move(e));
    }

    auto d_causal = engine.causal_edges_host();
    out.causal_edges.reserve(d_causal.size());
    for (const auto& c : d_causal) out.causal_edges.push_back(CausalEdge{c.from, c.to});
    auto d_branch = engine.branchial_edges_host();
    out.branchial_edges.reserve(d_branch.size());
    for (const auto& b : d_branch) out.branchial_edges.push_back(BranchialEdge{b.a, b.b});

    double t_readback_evcb = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_readback_evcb_start).count();

    // Note: Impl-owned device buffers (d_rules_, d_frontier_, etc) are NOT
    // freed here — they live for the Engine's lifetime and are reused on
    // subsequent run() calls. Impl's destructor handles cleanup.

    double t_total = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_total_start).count();

    if (dbg) {
        std::fprintf(stderr,
            "[evolve dbg] total=%.2f init=%.2f match=%.2f rewrite=%.2f "
            "hash=%.2f dedup=%.2f readback{hashes=%.2f states=%.2f ev/c/b=%.2f} (ms)\n",
            t_total, t_init, t_match, t_rewrite, t_hash, t_dedup,
            t_readback_hashes, t_readback_states, t_readback_evcb);
    }

    return out;
}

// Map an ErrorKind (the pool that overflowed) to the EngineConfig field(s)
// that govern its capacity, and double them. Some kinds map to multiple
// fields (e.g. kVertexPoolFull involves both max_vertex_slots and max_vertices).
// Returns true if growth was applied; false for kinds that have no
// retryable config (kScratchOverflow is a kernel-internal limit and can't
// be grown by reconfiguring pools).
bool grow_config_for(EngineConfig& cfg, ErrorKind kind) {
    auto dbl = [](uint32_t& f) { f = (f >= (1u << 31)) ? f : (f * 2u); };
    switch (kind) {
        case ErrorKind::kEdgePoolFull:
            dbl(cfg.max_edges);
            dbl(cfg.sig_index_pool);
            dbl(cfg.inverted_pool);
            dbl(cfg.edge_consumer_nodes);
            return true;
        case ErrorKind::kStatePoolFull:
            dbl(cfg.max_states);
            dbl(cfg.max_state_edge_total);
            dbl(cfg.canonical_map_slots);
            return true;
        case ErrorKind::kEventPoolFull:
            dbl(cfg.max_events);
            dbl(cfg.tr_preds_nodes);
            return true;
        case ErrorKind::kVertexPoolFull:
            dbl(cfg.max_vertex_slots);
            dbl(cfg.max_vertices);
            dbl(cfg.inverted_pool);
            return true;
        case ErrorKind::kCausalPoolFull:
            dbl(cfg.max_causal_edges);
            return true;
        case ErrorKind::kBranchialPoolFull:
            dbl(cfg.max_branchial_edges);
            return true;
        case ErrorKind::kMatchPoolFull:
            // Match pool sized as cfg.max_states * 8 in Engine::Impl ctor;
            // bumping max_states grows both. Edge growth also helps because
            // each match uses bounded RHS edges.
            dbl(cfg.max_states);
            return true;
        case ErrorKind::kCausalTripleMapFull: dbl(cfg.causal_triple_slots);  return true;
        case ErrorKind::kCausalPairMapFull:   dbl(cfg.causal_pair_slots);    return true;
        case ErrorKind::kBranchialMapFull:    dbl(cfg.branchial_pair_slots); return true;
        case ErrorKind::kEdgeConsumerNodes:   dbl(cfg.edge_consumer_nodes);  return true;
        case ErrorKind::kBranchialIndexNodes: dbl(cfg.branchial_index_nodes); return true;
        case ErrorKind::kTrPredsNodes:        dbl(cfg.tr_preds_nodes);       return true;
        case ErrorKind::kSigIndexNodes:       dbl(cfg.sig_index_pool);       return true;
        case ErrorKind::kInvIndexNodes:       dbl(cfg.inverted_pool);        return true;
        case ErrorKind::kFrontierCapFull:     dbl(cfg.max_states);           return true;
        case ErrorKind::kIRArenaExhausted:
            // The device IR arena is sized as holders x cfg.ir_arena_share_words, so this IS
            // config-controlled and doubling the share doubles the arena. Retryable, and it
            // must be: the persistent path has no 1-WL fallback (by design -- a fallback key
            // MERGES non-isomorphic states), so without the retry the work is lost rather
            // than degraded.
            dbl(cfg.ir_arena_share_words);
            return true;
        case ErrorKind::kIRGeneratorsExceeded:
            // Config-controlled, so doubling is a real remedy -- and it MUST be retried rather
            // than reported: the alternative is orbits fused over a truncated generator table,
            // which are too fine, and the quotient reconstruction keys instance identity on
            // them. A wrong answer, not a slow one.
            dbl(cfg.ir_generators);
            return true;
        case ErrorKind::kIRDepthExceeded:
            // The individualization search wanted to go deeper than the device attempts, and
            // the depth is config-controlled, so doubling is a real remedy. It must be retried
            // rather than reported: a state the exact path cannot key is a state with no dedup
            // key, and the only alternatives are dropping it or keying it by something coarser
            // -- and a coarser key MERGES non-isomorphic states.
            dbl(cfg.ir_depth);
            return true;
        case ErrorKind::kScratchOverflow:
            // A fixed per-thread bound (the TR closure's ancestor/descendant scratch). Not
            // config-controlled, so it cannot be retried; the caller accepts the truncation.
            return false;
        default: return false;
    }
}

// One-shot wrapper. Builds an Engine sized for `in`, runs it. If the
// kernel reports any retryable overflow warnings, doubles the relevant
// EngineConfig field(s) and re-runs from scratch — up to kMaxRetries
// times (64× capacity growth ceiling). Each retry destructs and
// reconstructs the Engine (the pools have to be re-allocated at the new
// sizes; preserving in-flight state across reallocs is more engineering
// for negligible benefit on the cold-start path).
//
// The returned result accumulates warnings across all attempts — the
// caller sees the cumulative trail, not just the last attempt's warnings.
// If the final attempt still produces warnings (because we hit the retry
// ceiling, or because some warnings are non-retryable like
// kScratchOverflow), the partial result is still returned with all
// warnings attached. The caller decides whether the partial result is
// good enough.
//
// For repeated runs of the same workload prefer `Engine(cfg).run(in)`
// directly with a config you've already validated, so the retry loop
// only fires on the first call.
// Log the EngineConfig that worked (after grow-and-retry) so the user can
// pre-size on subsequent calls. Only the fields that were grown beyond
// their initial value are printed — keeps the message focused. Format is
// stable so callers can grep it.
static void log_winning_config(const EngineConfig& initial,
                               const EngineConfig& winning) {
#define LOG_FIELD(field) \
    if (winning.field != initial.field) { \
        std::fprintf(stderr, "  %s: %u → %u\n", #field, initial.field, winning.field); \
    }
    std::fprintf(stderr,
        "hg_gpu::evolve: succeeded after grow-and-retry; pass these to "
        "Engine(cfg) directly to skip the retry loop next time:\n");
    LOG_FIELD(max_edges);
    LOG_FIELD(max_vertices);
    LOG_FIELD(max_vertex_slots);
    LOG_FIELD(max_states);
    LOG_FIELD(max_state_edge_total);
    LOG_FIELD(sig_index_pool);
    LOG_FIELD(inverted_pool);
    LOG_FIELD(canonical_map_slots);
    LOG_FIELD(max_events);
    LOG_FIELD(max_causal_edges);
    LOG_FIELD(max_branchial_edges);
    LOG_FIELD(causal_triple_slots);
    LOG_FIELD(causal_pair_slots);
    LOG_FIELD(branchial_pair_slots);
    LOG_FIELD(edge_consumer_nodes);
    LOG_FIELD(branchial_index_buckets);
    LOG_FIELD(branchial_index_nodes);
    LOG_FIELD(tr_preds_nodes);
#undef LOG_FIELD
}

// Scale a config's growable capacity fields down proportionally so its estimated
// footprint fits within `cap` bytes, leaving a floor under each so a minimal run
// is still possible. Bucket counts (power-of-two) and fixed control words are
// left alone. A run under the shrunk config that needs more will overflow and
// return a partial result, which is the intended "constrain to memory" behaviour.
void fit_config_to_cap(EngineConfig& cfg, uint64_t cap) {
    uint64_t est = estimated_device_bytes(cfg);
    if (cap == 0 || est <= cap) return;
    double r = static_cast<double>(cap) / static_cast<double>(est);
    auto sc = [&](uint32_t& f, uint32_t floor) {
        uint64_t v = static_cast<uint64_t>(static_cast<double>(f) * r);
        f = static_cast<uint32_t>(v < floor ? floor : v);
    };
    sc(cfg.max_edges, 1u<<12);            sc(cfg.max_vertices, 1u<<12);
    sc(cfg.max_vertex_slots, 1u<<14);     sc(cfg.max_states, 1u<<10);
    sc(cfg.max_state_edge_total, 1u<<16); sc(cfg.sig_index_pool, 1u<<12);
    sc(cfg.inverted_pool, 1u<<12);        sc(cfg.max_events, 1u<<10);
    sc(cfg.max_causal_edges, 1u<<12);     sc(cfg.max_branchial_edges, 1u<<12);
    sc(cfg.causal_triple_slots, 1u<<12);  sc(cfg.causal_pair_slots, 1u<<12);
    sc(cfg.branchial_pair_slots, 1u<<12); sc(cfg.edge_consumer_nodes, 1u<<12);
    sc(cfg.branchial_index_nodes, 1u<<12);sc(cfg.tr_preds_nodes, 1u<<12);
    sc(cfg.canonical_map_slots, 1u<<12);
}

uint64_t estimated_device_bytes(const EngineConfig& cfg) {
    // Sum the pools EngineState allocates. Element sizes: Edge 24; DeviceEvent
    // ~160; DeviceCausal/Branchial edge 12; StateEdgeSlice 8; a LockFreeList node
    // is sizeof(value)+4 rounded up; a ConcurrentMap slot is sizeof(K)+sizeof(V).
    // A 4-byte id is the unit for most index/id pools. Approximate — a 15%
    // headroom covers the small frontier/hash scratch buffers and allocation
    // granularity, so the estimate errs high (refusing borderline growth).
    auto u64 = [](uint32_t v) { return static_cast<uint64_t>(v); };
    uint64_t b = 0;
    b += u64(cfg.max_vertex_slots)    * 4;          // vertex_pool
    b += u64(cfg.max_edges)           * 24;         // edge_pool (Edge)
    b += u64(cfg.max_edges)           * 4;          // edge_producer
    b += u64(cfg.max_states)          * 8;          // state_edge_slices
    b += u64(cfg.max_state_edge_total)* 4;          // state_edge_ids
    b += u64(cfg.sig_index_buckets)   * 4 + u64(cfg.sig_index_pool) * 8;   // signature index
    b += u64(cfg.max_vertices)        * 4 + u64(cfg.inverted_pool)  * 8;   // vertex inverted index
    b += u64(cfg.max_events)          * 160;        // event_pool (DeviceEvent)
    b += u64(cfg.max_causal_edges)    * 12;         // causal_edge_pool
    b += u64(cfg.max_branchial_edges) * 12;         // branchial_edge_pool
    b += u64(cfg.max_edges)           * 4 + u64(cfg.edge_consumer_nodes)   * 8;   // edge_consumers
    b += u64(cfg.branchial_index_buckets) * 4 + u64(cfg.branchial_index_nodes) * 16; // branchial index
    b += u64(cfg.causal_triple_slots) * 12;
    b += u64(cfg.causal_pair_slots)   * 12;
    b += u64(cfg.branchial_pair_slots)* 12;
    b += u64(cfg.max_events)          * 4  + u64(cfg.tr_preds_nodes) * 8;  // preds_list
    b += u64(cfg.canonical_map_slots) * 12;         // canonical dedup map
    b += u64(cfg.match_dedup_slots)   * 12 + u64(cfg.event_canon_slots) * 12;
    b += u64(cfg.max_states)          * 8 * 76;     // matches pool (max_states*8 records ~76B)
    b += u64(cfg.max_states)          * 16;         // d_frontier + d_next_frontier + state_canonical_hash
    return b + b / 6;   // ~17% headroom
}

EvolveResult evolve(const EvolveInput& in) {
    constexpr int kMaxRetries = 6;  // up to 64× capacity growth
    EngineConfig initial_cfg = config_from_input(in);
    EngineConfig cfg = initial_cfg;
    std::vector<OverflowWarning> trail;

    // Resolve the device-memory ceiling: explicit request, else 90% of total VRAM.
    uint64_t mem_cap = in.max_device_memory_bytes;
    if (mem_cap == 0) {
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
            mem_cap = static_cast<uint64_t>(static_cast<double>(totalB) * 0.90);
        }
        cudaGetLastError();  // clear any sticky status from the query
    }
    // Shrink the initial config to the ceiling if config_from_input over-provisioned
    // past it; the loop below then never grows back over the cap.
    if (mem_cap != 0 && estimated_device_bytes(cfg) > mem_cap) {
        fit_config_to_cap(cfg, mem_cap);
        initial_cfg = cfg;
        std::fprintf(stderr,
            "hg_gpu::evolve: initial config exceeded the memory cap (%llu MB) — "
            "scaled pools down to ~%llu MB; result may be partial.\n",
            (unsigned long long)(mem_cap >> 20),
            (unsigned long long)(estimated_device_bytes(cfg) >> 20));
    }

    // Best partial result seen so far: an attempt that overflowed still returns
    // whatever it computed, and if the next, larger engine no longer fits in
    // device memory that partial is what the caller gets, never an exception.
    EvolveResult best;

    for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
        EvolveResult result;
        try {
            Engine engine(cfg);
            result = engine.run(in);
        } catch (const std::exception& e) {
            trail.push_back(OverflowWarning{
                ErrorKind::kDeviceOutOfMemory, 1u,
                std::string("attempt ") + std::to_string(attempt + 1) + ": " + e.what()});
            std::fprintf(stderr,
                "hg_gpu::evolve: engine at the grown size no longer fits in device "
                "memory (%s) — returning the last completed attempt's partial result.\n",
                e.what());
            best.warnings = std::move(trail);
            return best;
        }

        // No overflow this attempt: success — return with the cumulative
        // trail (which is empty on the first-attempt-clean path).
        if (result.warnings.empty() && trail.empty()) {
            return result;
        }
        if (result.warnings.empty()) {
            // Clean run after one or more grow-and-retry rounds. Surface
            // the winning config to the operator and return with the
            // accumulated trail attached.
            log_winning_config(initial_cfg, cfg);
            result.warnings = std::move(trail);
            return result;
        }

        // Grow the config for any retryable warnings observed THIS
        // attempt. grow_config_for is idempotent under repeats and
        // doubling spuriously is conservative, so we just sweep every
        // warning's kind. If no warning is retryable (only
        // kScratchOverflow, say), we can't make progress — return the
        // partial result with the cumulative trail.
        bool any_retryable = false;
        ErrorKind first_grew = ErrorKind::kCount;
        for (const auto& w : result.warnings) {
            if (grow_config_for(cfg, w.kind)) {
                if (!any_retryable) first_grew = w.kind;
                any_retryable = true;
            }
        }

        // Cumulative trail across attempts; user can see what was hit
        // and how many times across the retries.
        for (auto& w : result.warnings) trail.push_back(std::move(w));

        if (!any_retryable || attempt == kMaxRetries) {
            result.warnings = std::move(trail);
            return result;
        }

        // Would the grown config exceed the memory ceiling? If so, stop here and
        // return the best partial rather than pushing toward a real device OOM.
        if (mem_cap != 0 && estimated_device_bytes(cfg) > mem_cap) {
            trail.push_back(OverflowWarning{
                ErrorKind::kDeviceOutOfMemory,
                static_cast<uint32_t>(estimated_device_bytes(cfg) >> 20),
                "grown config (~" + std::to_string(estimated_device_bytes(cfg) >> 20) +
                " MB) would exceed the device memory cap (" +
                std::to_string(mem_cap >> 20) + " MB); returning partial result"});
            std::fprintf(stderr,
                "hg_gpu::evolve: next config (~%llu MB) would exceed the memory cap "
                "(%llu MB) — returning the partial result.\n",
                (unsigned long long)(estimated_device_bytes(cfg) >> 20),
                (unsigned long long)(mem_cap >> 20));
            result.warnings = std::move(trail);
            return result;
        }

        best = std::move(result);

        std::fprintf(stderr,
            "hg_gpu::evolve: overflow on %s — growing relevant config and "
            "retrying (attempt %d/%d).\n",
            error_kind_name(first_grew), attempt + 2, kMaxRetries + 1);
    }
    // Unreachable: loop returns on every path.
    return EvolveResult{};
}

// ---------------------------------------------------------------------------
// PersistentEvolver: evolve() with the device Engine kept alive across calls.
// ---------------------------------------------------------------------------
// GpuSession: the pimpl that lets a HOST translation unit own a device session. Holds the
// SessionState and hands out the view; nothing else.
struct GpuSession::Impl {
    SessionState state;
    SessionView  view;
    Impl(uint32_t max_states, uint32_t max_events)
        : state(max_states, max_events), view(state.view()) {}
};

GpuSession::GpuSession(uint32_t max_states, uint32_t max_events)
    : impl_(std::make_unique<Impl>(max_states, max_events)) {}
GpuSession::~GpuSession() = default;

SessionView* GpuSession::view() { return &impl_->view; }
uint32_t GpuSession::frontier_size() const { return impl_->state.frontier_size(); }

PersistentEvolver::PersistentEvolver()  = default;
PersistentEvolver::~PersistentEvolver() = default;

// A session run REFUSES to rebuild the engine. run() below may replace engine_ -- when the
// config changes, or after an overflow throw, which discards it deliberately so a poisoned
// device cannot infect every later call in this worker. Either would drop a session's
// accumulated states while handing back something shaped exactly like a continuation, so this
// reports a dead handle instead and the caller reopens. The host does the same thing by
// invalidating its session slot on a throw.
PersistentEvolver::SessionRun PersistentEvolver::run_session(const EvolveInput& in,
                                                             SessionView* session,
                                                             uint32_t start_step) {
    SessionRun out;
    if (session == nullptr) {
        out.error = "run_session called without a session";
        return out;
    }

    // Opening: size the engine from this input, exactly as a first run would.
    if (!has_engine_) {
        if (start_step != 0) {
            out.error = "this session has no engine to continue; it was never opened, or a "
                        "previous call discarded it";
            return out;
        }
        EngineConfig cfg = config_from_input(in);
        try {
            engine_ = std::make_unique<Engine>(cfg);
        } catch (const std::exception& e) {
            out.error = std::string("could not build a device engine for this session: ") + e.what();
            return out;
        }
        cfg_        = cfg;
        has_engine_ = true;
    } else if (start_step == 0) {
        // OPENING ONTO A LIVE ENGINE IS NORMAL, not an error: this evolver is reused for every
        // job the worker handles, so it has an engine after any prior Evolve. The session must
        // not inherit that run's graph, and Engine::Impl::run already clears it whenever
        // start_step is 0 -- one place decides that, rather than two that can disagree.
        // Refusing instead would make a plain Evolve poison the next Open, which is what the
        // process-boundary gate caught.
    }

    try {
        out.result = engine_->run(in, session, start_step);
    } catch (const std::exception& e) {
        // Same reasoning as run(): the engine may be inconsistent, so it goes. The session dies
        // with it rather than continuing against a rebuilt device.
        engine_.reset();
        has_engine_ = false;
        out.error = std::string("the device run failed and the session is no longer valid: ")
                  + e.what();
        return out;
    }
    out.ok = true;
    return out;
}

EvolveResult PersistentEvolver::run(const EvolveInput& in) {
    constexpr int kMaxRetries = 6;
    // Never shrink: start from the live engine's config if we have one, else size
    // to this input. The loop only rebuilds when the config actually changes, so a
    // run whose input fits the current engine reuses it and pays no allocation.
    EngineConfig cfg = has_engine_ ? cfg_ : config_from_input(in);
    std::vector<OverflowWarning> trail;

    uint64_t mem_cap = in.max_device_memory_bytes;
    if (mem_cap == 0) {
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess)
            mem_cap = static_cast<uint64_t>(static_cast<double>(totalB) * 0.90);
        cudaGetLastError();
    }
    if (mem_cap != 0 && estimated_device_bytes(cfg) > mem_cap)
        fit_config_to_cap(cfg, mem_cap);

    EvolveResult best;
    for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
        // (Re)build only when the config differs from the live engine's. On a grow
        // the old engine is freed before the larger one is built, so peak VRAM is
        // bounded by the larger config, not their sum.
        if (!has_engine_ || std::memcmp(&cfg, &cfg_, sizeof(EngineConfig)) != 0) {
            try {
                engine_.reset();
                engine_ = std::make_unique<Engine>(cfg);
            } catch (const std::exception& e) {
                has_engine_ = false;
                trail.push_back(OverflowWarning{
                    ErrorKind::kDeviceOutOfMemory, 1u,
                    std::string("attempt ") + std::to_string(attempt + 1) + ": " + e.what()});
                best.warnings = std::move(trail);
                return best;
            }
            cfg_        = cfg;
            has_engine_ = true;
        }

        EvolveResult result;
        try {
            result = engine_->run(in);
        } catch (const std::exception& e) {
            // Engine::run throws on a hard overflow (and on a genuine device fault). Unlike
            // the free evolve() -- which builds a throwaway Engine per attempt -- this evolver
            // REUSES engine_ across calls, so a throw that leaves the device state inconsistent
            // would poison every later call in this worker (the reported "works, then fails and
            // never recovers until a kernel reset"). Discard the engine so the next attempt/call
            // rebuilds a clean one, clear any non-sticky device error, and return the best
            // partial so far with an overflow warning instead of propagating -- mirroring the
            // graceful partial-result contract of the free evolve().
            engine_.reset();
            has_engine_ = false;
            cudaGetLastError();
            trail.push_back(OverflowWarning{
                ErrorKind::kDeviceOutOfMemory, 1u,
                std::string("attempt ") + std::to_string(attempt + 1) + ": " + e.what()});
            best.warnings = std::move(trail);
            return best;
        }

        if (result.warnings.empty() && trail.empty()) return result;
        if (result.warnings.empty()) {
            result.warnings = std::move(trail);
            return result;
        }

        bool any_retryable = false;
        for (const auto& w : result.warnings)
            if (grow_config_for(cfg, w.kind)) any_retryable = true;
        for (auto& w : result.warnings) trail.push_back(std::move(w));

        if (!any_retryable || attempt == kMaxRetries) {
            result.warnings = std::move(trail);
            return result;
        }
        if (mem_cap != 0 && estimated_device_bytes(cfg) > mem_cap) {
            trail.push_back(OverflowWarning{
                ErrorKind::kDeviceOutOfMemory,
                static_cast<uint32_t>(estimated_device_bytes(cfg) >> 20),
                "grown config (~" + std::to_string(estimated_device_bytes(cfg) >> 20) +
                    " MB) would exceed the device memory cap (" +
                    std::to_string(mem_cap >> 20) + " MB); returning partial result"});
            result.warnings = std::move(trail);
            return result;
        }
        best = std::move(result);
    }
    return EvolveResult{};
}

}  // namespace gpu
}  // namespace HG_NAMESPACE