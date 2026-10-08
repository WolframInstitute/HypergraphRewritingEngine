#include "hgcommon/namespace.hpp"
#include <limits>
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/exploration.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/rewrite.hpp"
#include "hg_gpu/cuda_check.hpp"
#include "hgcommon/quotient_route.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>
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

// The most edges a state can hold `steps` rewrites from a root of `largest_root` edges: a rewrite
// changes a state's edge count by |rhs| - |lhs| of its rule.
static size_t max_state_edges(size_t largest_root, const std::vector<RewriteRule>& rules,
                              uint32_t steps) {
    size_t max_growth = 0;
    for (const auto& r : rules)
        if (r.rhs.size() > r.lhs.size()) max_growth = std::max(max_growth, r.rhs.size() - r.lhs.size());
    return largest_root + max_growth * static_cast<size_t>(steps);
}

EngineConfig config_from_input(const EvolveInput& in) {
    EngineConfig cfg;
    size_t n_init   = in.initial_state.size();
    uint32_t steps  = in.num_steps;

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
    //
    // Computed in 64 bits and each field clamped to 2^31, the ceiling grow_config_for doubles
    // to: a 32,768-edge root at three steps asks for 2^30 edges and 2^32 vertex slots. A config
    // past the device is scaled to it by fit_config_to_cap.
    auto field = [](uint64_t v) { return static_cast<uint32_t>(std::min<uint64_t>(v, 1ull << 31)); };
    const uint64_t expected_edges  =
        std::max<uint64_t>(1u << 20, static_cast<uint64_t>(n_init) * growth * 512u);
    const uint64_t expected_states =
        std::max<uint64_t>(1u << 17, static_cast<uint64_t>(n_init) * growth * 32u);

    cfg.max_edges              = field(expected_edges);
    cfg.max_states             = field(expected_states);
    cfg.max_vertex_slots       = field(expected_edges * 4u);
    // Total edge-ID slots across all states' CSR rows: one slice per state id, each at most
    // max_state_edges. At most 1G slots (4 GB).
    size_t largest_init = n_init;
    for (const auto& s : in.initial_states) largest_init = std::max(largest_init, s.size());
    const uint64_t state_edges_bound =
        std::max<uint64_t>(1u, max_state_edges(largest_init, in.rules, steps));
    cfg.max_state_edge_total   = static_cast<uint32_t>(std::min<uint64_t>(
        static_cast<uint64_t>(cfg.max_states) * state_edges_bound, 1ull << 30));
    // Each event allocates ≤ kMaxVars fresh vertices, so vertex IDs bound
    // by n_init-vertices + events × kMaxVars. Be generous.
    cfg.max_vertices           = field(std::max<uint64_t>(
        expected_edges, static_cast<uint64_t>(n_init) * 4u + expected_states * 4u));
    cfg.sig_index_buckets      = 1024;
    cfg.sig_index_pool         = field(expected_edges * 2u);
    cfg.inverted_pool          = field(expected_edges * 4u);

    if (in.slice_scan_max_edges) cfg.slice_scan_max_edges = in.slice_scan_max_edges;
    if (in.max_blocks_per_launch) cfg.max_blocks_per_launch = in.max_blocks_per_launch;

    const uint64_t expected_events = expected_states;
    cfg.max_events             = field(expected_events);
    cfg.max_causal_edges       = field(expected_events * 8u);
    cfg.max_branchial_edges    = field(expected_events * 8u);
    cfg.causal_triple_slots    = field(expected_events * 16u);
    cfg.causal_pair_slots      = field(expected_events * 8u);
    cfg.branchial_pair_slots   = field(expected_events * 16u);
    cfg.edge_consumer_nodes    = field(expected_edges * 4u);
    cfg.branchial_index_buckets = 1u << 20;
    cfg.branchial_index_nodes   = field(expected_events * 4u);
    // One preds node per unique kept causal pair; kept pairs are a subset of causal pairs.
    cfg.tr_preds_nodes         = field(expected_events * 8u);
    cfg.canonical_key_mask     = in.canonical_key_mask;
    cfg.event_key_mask         = in.event_key_mask;
    cfg.replay_id_limit        = in.replay_id_limit;
    cfg.keyed_rewrites         = in.keyed_rewrites;
    cfg.keyed_claim_limit      = in.keyed_claim_limit;
    cfg.keyed_sum_mask         = in.keyed_sum_mask;
    auto widen = [](uint16_t& f, size_t n) {
        f = static_cast<uint16_t>(std::max<size_t>(f, std::min<size_t>(n, 0xFFFFu)));
    };
    for (const auto& e : in.initial_state) widen(cfg.max_edge_arity, e.size());
    for (const auto& r : in.rules) {
        for (const auto& e : r.rhs) widen(cfg.max_edge_arity, e.size());
        widen(cfg.max_lhs_edges, r.lhs.size());
    }
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



namespace {

// The setting in which `in` differs from the session's opening call, or nullptr.
const char* continuation_mismatch(const EvolveInput& open, const EvolveInput& in) {
    if (!hgcommon::same_record_set(open.record, in.record)) return "record set";
    if (open.canonicalization != in.canonicalization) return "canonicalization";
    if (open.event_canonicalization != in.event_canonicalization)
        return "event canonicalization";
    if (open.transitive_reduction != in.transitive_reduction) return "transitive_reduction";
    if (open.explore_from_canonical_states_only != in.explore_from_canonical_states_only)
        return "explore_from_canonical_states_only";
    // The per-event content arrays are written as events are minted, only when this is set.
    if (open.materialize_events != in.materialize_events) return "materialize_events";
    return nullptr;
}

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
        , matches_(static_cast<uint32_t>(
              std::min<uint64_t>(uint64_t{cfg.max_states} * 8u, 1ull << 31)))
    {}

    void reset() {
        state_.clear();
        // Cleared here, while the counter still bounds what the last run wrote; the launch's
        // own reset_and_clear then finds the pool clean.
        matches_.reset_and_clear();
    }

    EvolveResult run(const EvolveInput& in, SessionView* session = nullptr,
                     uint32_t start_step = 0, EvolveResult* storage = nullptr);

    EngineConfig                       cfg_;
    EngineState                        state_;
    Pool<MatchRecord>                  matches_;
    // The settings of the call that opened the current session (start_step 0), which a
    // continuation must repeat: the captures, instances and counts it extends were built under
    // them.
    EvolveInput                        opening_;
    // Engine-lifetime, cleared per run: rebuilding its maps costs tens of MB of cudaMalloc
    // per evolve. Constructed on the first run that routes quotient causal.
    std::unique_ptr<QcState>           qc_state_;
    std::unique_ptr<QeState>           qe_state_;
};

Engine::Engine(EngineConfig cfg) : impl_(new Impl(cfg)) {}
Engine::~Engine() { delete impl_; }
void Engine::reset() { impl_->reset(); }
const EngineConfig& Engine::config() const { return impl_->cfg_; }
EvolveResult Engine::run(const EvolveInput& in, SessionView* session,
                         uint32_t start_step, EvolveResult* storage) {
    return impl_->run(in, session, start_step, storage);
}

namespace {
// Fill each state's dedup key with a unique value so NONE of them merge —
}  // namespace

EvolveResult Engine::Impl::run(const EvolveInput& in, SessionView* session,
                               uint32_t start_step, EvolveResult* storage) {
    // Reset device state from any prior run() -- EXCEPT when continuing a session. A Step's
    // accumulated states ARE the graph being extended, and the frontier it seeds from holds ids
    // into those pools, so clearing them leaves the run seeding ids that no longer name
    // anything and it produces nothing. Opening (start_step 0) still resets, which is what
    // keeps a session from inheriting a previous job's graph.
    if (start_step == 0) reset();
    if (start_step == 0) {
        opening_.record = in.record;
        opening_.canonicalization = in.canonicalization;
        opening_.event_canonicalization = in.event_canonicalization;
        opening_.transitive_reduction = in.transitive_reduction;
        opening_.explore_from_canonical_states_only = in.explore_from_canonical_states_only;
        opening_.materialize_events = in.materialize_events;
    } else if (const char* what = continuation_mismatch(opening_, in)) {
        throw std::invalid_argument(std::string("a session continuation must use the opening "
                                                "call's ") + what);
    }

    EvolveResult out;
    if (storage) out.adopt_storage(*storage);
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
                          hgcommon::quotient_route_requested(
                              in.explore_from_canonical_states_only, /*positional=*/false,
                              event_keys_for(in.event_canonicalization));
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
    //
    // A continuation extends the bound with its larger num_steps. Maintenance that was off until
    // now turns on with the indices empty -- nothing was inserted since the opening call's clear
    // -- so every edge the session holds is inserted once (rebuild_indices) before the run.
    // Maintenance that was on stays on.
    const bool maintain = max_state_edges(max_root_edges, in.rules, in.num_steps) >
                          engine.config_slice_scan_max_edges();
    if (start_step == 0) {
        engine.set_maintain_indices(maintain);
    } else if (maintain && !engine.maintain_indices()) {
        engine.set_maintain_indices(true);
        rebuild_indices(engine, std::min(engine.num_edges_host(), engine.config().max_edges));
    }
    // Continuing: the roots are already in the pools from the call that opened the session, so
    // uploading them again would add a second copy of every root and re-seed the evolution from
    // depth 0 alongside the frontier.
    const uint32_t num_roots = (start_step == 0)
                                   ? upload_initial_states(engine, roots)
                                   : static_cast<uint32_t>(roots.size());

    // The fallback count lives in the counter block, which a continuation keeps; the warnings
    // are per call (DeviceErrors::warnings_from clears them), so the count is this call's too.
    const uint32_t fallbacks_before = start_step != 0 ? engine.event_sig_raw_fallbacks() : 0u;

    // The launch uploads the rules into its own scratch (run_persistent_evolve).
    std::vector<DeviceRule> rules;
    rules.reserve(in.rules.size());
    for (const auto& r : in.rules) rules.push_back(make_device_rule(r));

    const EngineConfig& cfg = engine.config();
    Pool<MatchRecord>& matches = matches_;

    // The per-state hash lives on DeviceState, not in this driver: the persistent kernel
    // writes it when it hashes a child for dedup, and the readback at the end reads that same
    // array. A buffer owned by Engine::Impl would have made the assembly private to the host.
    uint64_t* d_state_hashes  = engine.device().state_canonical_hash;

    // ExplorationProbability, clamped to [0, 1] as the host clamps it.
    double clamped_p = in.exploration_probability;
    if (!(clamped_p > 0.0)) clamped_p = 0.0;
    if (clamped_p > 1.0)    clamped_p = 1.0;
    // The quotient route flag: on the route every state also computes its edge orbits.
    auto t_qcsetup_start = std::chrono::steady_clock::now();
    if (!qc_state_ || qc_state_->enabled() != qc_route)
        qc_state_ = std::make_unique<QcState>(qc_route);
    QcView qc_view = qc_state_->view();

    // The class-frame expansion capture rides the route decision: a run that reconstructs
    // causality is exactly a run whose event identity comes from the class frame rather than
    // each raw state's labelling.
    if (!qe_state_ || qe_state_->enabled() != qc_route) {
        qe_state_ = std::make_unique<QeState>(qc_route, qe_entries(cfg));
    } else if (start_step == 0 && qc_route) {
        // A continuation keeps the reconstruction: its captures, instances, counts and the
        // points the old bound left standing, which the run drives (k_qe_redrive). Off the
        // route the state is never written (every device entry returns on !qe.enabled), so
        // there is nothing to clear.
        qe_state_->clear();
    }
    // Capture always; REPLAY only when the caller records something the raw unfolding answers.
    // The replay is the device twin of the host's instance cascade, and it is the term measured
    // exponential in depth against an answer that is linear (b98a943c). Capture is untouched, so
    // Automatic event identity -- signed from the class frame -- is unchanged either way.
    //
    // Counts only: the raw counts come from class multiplicities and no instance is built. The
    // multiplicities are also counted when the caller asks for them. The host's rule in
    // ParallelEvolutionEngine (set_quotient_multiplicity, set_quotient_replay).
    const bool qe_counts_only = in.record.raw_counts_only && !in.record.causal;
    const bool qe_raw = in.record.causal || in.record.branchial || in.record.raw_events;
    const bool qe_replay = qe_raw && !qe_counts_only;
    const bool qe_multiplicity = (qe_raw && qe_counts_only) || in.record.multiplicities;
    // The multiplicity cascade's queue, one slice per driver: one driver per persistent block and
    // one per root, so the arena covers whichever launch starts more of them. The replay's lanes
    // each take a reachability slice on first need, from a table with one entry per lane.
    if (qe_multiplicity) {
        const uint32_t drivers =
            default_persistent_grid() > static_cast<uint32_t>(roots.size())
                ? default_persistent_grid()
                : static_cast<uint32_t>(roots.size());
        qe_state_->ensure_work(drivers, in.num_steps, cfg.descent_work_scale);
    }
    if (qe_replay) qe_state_->ensure_lanes(default_persistent_grid() * kMatchBlockThreads);
    const bool qe_event_content = qc_route && qe_replay && in.materialize_events;
    if (qe_event_content) qe_state_->ensure_event_content();
    qe_state_->set_id_limit(in.replay_id_limit);
    QeView qe_view = qe_state_->view(in.num_steps, event_keys_for(in.event_canonicalization),
                                     qe_replay, qe_multiplicity, qe_event_content);

    double t_qcsetup = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_qcsetup_start).count();

    // Sampling and capping parameters for this run. Allocates only what is asked for, and
    // clears the two counters every call so a session's later Step does not inherit the
    // tallies of its first.
    engine.set_sampling(in.transition_rate,
                        in.rule_weights.empty() ? nullptr : in.rule_weights.data(),
                        static_cast<uint32_t>(in.rule_weights.size()),
                        in.exploration_seed,
                        clamped_p,
                        in.max_states_per_step,
                        in.max_successor_states_per_parent,
                        in.matches_per_state_rule,
                        in.num_steps,
                        static_cast<uint32_t>(in.rules.size()));

    const bool dbg = std::getenv("HG_GPU_DBG_TIME") != nullptr;
    // Carried out of the persistent branch so the summary below can attribute the whole call.
    // Without these the branch between init and readback -- kernel, quotient setup and the
    // reconstruction marshalling -- was a single untimed region holding about 30% of the call.
    double t_persist_call = 0.0, t_recon = 0.0;
    // Every scalar counter of the engine, read once after the kernel; the warnings and the
    // readback below both size from it.
    EngineState::CounterSnapshot snap{};
    double t_match = 0, t_rewrite = 0, t_hash = 0, t_dedup = 0;


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
        // The transition draw keys on the consumed edges' ranks too, so a SAMPLED run needs the
        // array whatever event identity it asked for. Taking it in the kernel's need_ranks alone
        // is not enough: that flag decides whether the ranks are COMPUTED, and this decides
        // whether there is anywhere to put them. Without it the kernel writes through a null
        // pointer, the run faults, and the grow-and-retry loop reports the engine as too large
        // for the device rather than naming the real fault.
        const bool sampling_needs_ranks = hgcommon::sampling_active(
            in.transition_rate, in.rule_weights.data(),
            static_cast<uint32_t>(in.rule_weights.size()),
            hgcommon::drain_selects(in.matches_per_state_rule,
                                    in.max_successor_states_per_parent,
                                    in.max_states_per_step)) ||
            (clamped_p < 1.0 && !in.explore_from_canonical_states_only);
        if (sampling_needs_ranks ||
            (ekeys & (hgcommon::EventKey_ConsumedEdges | hgcommon::EventKey_ProducedEdges))) {
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

        auto t_kern_start = std::chrono::steady_clock::now();
        PersistentEvolveStats st = run_persistent_evolve(
            engine, rules, roots, in.num_steps, matches, arena,
            /*dedup=*/in.explore_from_canonical_states_only,
            in.canonicalization, ekeys, /*blocks=*/0,
            qc_route ? &qc_view : nullptr,
            qc_route ? &qe_view : nullptr,
            session, start_step, /*read_stats=*/dbg);

        t_persist_call = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_kern_start).count();

        auto t_recon_start = std::chrono::steady_clock::now();
        // Branchial pairs are counted from the points after the run (qe_count_branchial).
        const bool qe_branchial =
            qc_route && (qe_multiplicity || (qe_replay && in.record.branchial));
        if (qe_branchial) qe_count_branchial(engine.device(), qe_view, qe_multiplicity);
        // Every scalar the result needs, in one batch: the engine's counter block, the error
        // counters, and on the reconstruction route the QeState counters, its multiplicity
        // counts and its capture count.
        std::vector<uint32_t> eng_raw(EngineState::counter_block_words());
        std::vector<uint32_t> err_raw(DeviceErrors::kMaxKinds);
        std::vector<uint32_t> qe_raw(hg_gpu::QeState::counter_words());
        unsigned long long qm_raw[2] = {};
        uint32_t qe_matches = 0;
        {
            EngineState::ReadbackBatch counters(engine);
            counters.add_raw(eng_raw.data(), engine.counter_block_device(),
                             sizeof(uint32_t) * eng_raw.size());
            counters.add_raw(err_raw.data(), engine.errors().counters_device(),
                             sizeof(uint32_t) * err_raw.size());
            if (qc_route) {
                counters.add_raw(qe_raw.data(), qe_state_->counters_device(),
                                 sizeof(uint32_t) * qe_raw.size());
                if (qe_multiplicity || qe_branchial)
                    counters.add_raw(qm_raw, qe_state_->qm_counts_device(), sizeof(qm_raw));
                counters.add_raw(&qe_matches, qe_state_->num_matches_device(), sizeof(uint32_t));
            }
            counters.finish();
        }
        snap = EngineState::snapshot_from(eng_raw.data());
        auto qc_counts = qc_route ? hg_gpu::QeState::counters_from(qe_raw.data(), qm_raw)
                                  : hg_gpu::QeState::Counters{};
        // The id counter passes the limit by the refused attempts; the ids issued are below it.
        if (qc_counts.raw_events > in.replay_id_limit) qc_counts.raw_events = in.replay_id_limit;
        out.expansion_matches   = qe_matches;
        out.expansion_instances = qc_counts.instances;
        out.reconstructed_raw_events =
            qe_replay ? qc_counts.raw_events : qc_counts.qm_raw_events;
        // The REPLAY is what produces a reconstructed answer, so this reports the replay and
        // not merely the route. A run that captured the class frames but never replayed them
        // has no reconstructed raw events, and a caller that read this as "reconstruction ran"
        // would treat empty relations as a result rather than as an artifact not requested.
        out.reconstruction_ran = qc_route && (qe_replay || qe_multiplicity);
        // Under EVENT_SIG_NONE no identity is computed and every application is its own event,
        // so the raw count IS the answer -- the same rule as the host's num_reconstructed_events.
        out.reconstructed_events =
            !qc_route ? 0u
            : (event_keys_for(in.event_canonicalization) == EVENT_SIG_NONE
                   ? out.reconstructed_raw_events
                   : qc_counts.canon_events);
        out.reconstructed_causal_pairs = qc_counts.causal_pairs;
        out.reconstructed_causal_edges = qc_counts.causal_edges;
        out.reconstructed_branchial = qc_counts.qm_branchial;
        if (qc_route && in.record.multiplicities)
            qe_state_->class_multiplicities_host(out.class_multiplicities, out.class_rule_matches);
        // Causal and its reduction are built whenever the route ran, because the reduced COUNT
        // is the size of that relation and deriving it is the only way to know it. Branchial is
        // the expansion, so it is built only for a caller that will read the pairs.
        if (qc_route)
            qe_state_->reconstructed_pairs_host(out.reconstructed_causal_relation,
                                                out.reconstructed_causal_relation_reduced,
                                                out.reconstructed_branchial_relation,
                                                in.materialize_relations,
                                                qc_counts.raw_events,
                                                &out.reconstructed_event_signature,
                                                &out.reconstructed_causal_raw,
                                                &out.reconstructed_causal_raw_reduced,
                                                in.materialize_relations
                                                    ? &out.reconstructed_branchial_raw : nullptr);
        if (qe_event_content)
            qe_state_->reconstructed_event_content_host(out.reconstructed_event_from_class,
                                                        out.reconstructed_event_to_class,
                                                        out.reconstructed_event_rule);
        // DERIVED from the relation the caller receives, not counted beside it: the reduction
        // is computed during that readback, so a separate tally could only ever disagree.
        out.reconstructed_causal_pairs_reduced =
            static_cast<uint32_t>(out.reconstructed_causal_relation_reduced.size());
        out.frame_alignments = qc_counts.aligned;
        out.frame_align_failures = qc_counts.align_failures;
        engine.errors().warnings_from(err_raw.data(), out.warnings, "persistent evolve");
        EngineState::report_event_sig_fallbacks(out.warnings, "persistent evolve",
                                                snap.sig_fallbacks - fallbacks_before);
        t_recon = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_recon_start).count();
        if (dbg) {
            const double tot = double(st.cycles_match) + double(st.cycles_rewrite) +
                               double(st.cycles_canon) + double(st.cycles_idle) +
                               double(st.cycles_wait);
            const double pct = tot > 0 ? 100.0 / tot : 0.0;
            std::fprintf(stderr,
                         "[persistent] states=%u matches=%u arena_words=%llu cycles: "
                         "match=%.1f%% rewrite=%.1f%% canon=%.1f%% idle=%.1f%% wait=%.1f%% "
                         "keyed_twins=%u\n",
                         st.states_after, st.matches_found,
                         (unsigned long long)st.arena_words_used,
                         st.cycles_match * pct, st.cycles_rewrite * pct,
                         st.cycles_canon * pct, st.cycles_idle * pct,
                         st.cycles_wait * pct, st.keyed_twins);
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
            // The canon bucket's parts, in the order the kernel accumulates them. An
            // optimisation aimed at "canonicalization" has to know which of the five it is
            // aiming at, and the bucket is the largest single phase on the device.
            const double cb = double(st.cycles_canon_sub[0]) + double(st.cycles_canon_sub[1]) +
                              double(st.cycles_canon_sub[2]) + double(st.cycles_canon_sub[3]) +
                              double(st.cycles_canon_sub[4]);
            const double cpct = cb > 0 ? 100.0 / cb : 0.0;
            std::fprintf(stderr,
                         "[persistent canon] irkey=%.1f%% evkey=%.1f%% qc=%.1f%% qe=%.1f%% "
                         "dedup=%.1f%%\n",
                         st.cycles_canon_sub[0] * cpct, st.cycles_canon_sub[1] * cpct,
                         st.cycles_canon_sub[2] * cpct, st.cycles_canon_sub[3] * cpct,
                         st.cycles_canon_sub[4] * cpct);
        }
    }

    auto t_readback_start = std::chrono::steady_clock::now();

    // Readback. The kernel wrote the state hashes; `snap` sizes the states, the per-state edge
    // arrays and the three relation pools, and every region is read in one batch: one
    // synchronization for all of them.
    const uint32_t total_states = snap.states;
    EngineState::ReadbackBatch batch(engine);
    std::vector<uint64_t> h_hashes;
    batch.add(h_hashes, static_cast<const uint64_t*>(d_state_hashes), total_states);
    std::vector<StateEdgeSlice> slices;
    // The slices are read without the edge arrays too: a state a failed rewrite claimed carries
    // the slice {INVALID_ID, 0} (apply_one_match), and reads back with id INVALID_ID.
    if (in.materialize_state_edges) engine.add_state_edges(batch, snap, slices, out);
    else if (total_states)
        batch.add(slices, static_cast<const StateEdgeSlice*>(engine.device().state_edge_slices),
                  total_states);
    engine.add_events(batch, snap.events, out.events, out.event_consumed);
    engine.add_causal_edges(batch, snap.causal, out.causal_edges);
    engine.add_branchial_edges(batch, snap.branchial, out.branchial_edges);
    batch.finish();
    // A rewrite that claimed its event and then failed a later claim leaves the event with id
    // INVALID_ID (apply_one_match). Only a run that recorded an overflow has one.
    if (!out.warnings.empty())
        out.events.erase(std::remove_if(out.events.begin(), out.events.end(),
                                        [](const Event& e) { return e.id == INVALID_ID; }),
                         out.events.end());
    double t_readback_copy = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_readback_start).count();

    auto t_readback_states_start = std::chrono::steady_clock::now();
    out.states.resize(total_states);
    for (uint32_t s = 0; s < total_states; ++s) {
        CanonicalState& cs = out.states[s];
        cs.id             = (s < slices.size() && slices[s].offset == INVALID_ID) ? INVALID_ID : s;
        cs.canonical_hash = (s < h_hashes.size()) ? h_hashes[s] : 0;
        // A slice past the id array describes no edges.
        if (s < slices.size() &&
            static_cast<size_t>(slices[s].offset) + slices[s].count <= out.state_edge_ids.size()) {
            cs.first_edge = slices[s].offset;
            cs.num_edges  = slices[s].count;
        }
    }

    double t_readback_states = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_readback_states_start).count();

    double t_total = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_total_start).count();

    if (dbg) {
        std::fprintf(stderr,
            "[evolve dbg] total=%.2f init=%.2f qcsetup=%.2f persist=%.2f recon=%.2f "
            "match=%.2f rewrite=%.2f hash=%.2f dedup=%.2f "
            "readback{copy=%.2f states=%.2f} unattributed=%.2f (ms)\n",
            t_total, t_init, t_qcsetup, t_persist_call, t_recon,
            t_match, t_rewrite, t_hash, t_dedup,
            t_readback_copy, t_readback_states,
            t_total - t_init - t_qcsetup - t_persist_call - t_recon - t_match - t_rewrite
                    - t_hash - t_dedup - t_readback_copy - t_readback_states);
    }

    return out;
}

// Map an ErrorKind (the pool that overflowed) to the EngineConfig field(s)
// that govern its capacity, and double them. Some kinds map to multiple
// fields (e.g. kVertexPoolFull involves both max_vertex_slots and max_vertices).
// Returns true if growth was applied; false for kinds that have no
// retryable config (kScratchOverflow is a kernel-internal limit and can't
// be grown by reconfiguring pools).
// The pools a run fills in proportion to its size: states, events, edges and vertices, and the
// match pool, which is sized from max_states. When one overflows the run was cut short, so the
// others' needs were never reached and they overflow on the next attempts in turn (wpp depth 7:
// state, event, state, edge -- five attempts). They grow together, once per attempt.
bool is_size_pool(ErrorKind kind) {
    return kind == ErrorKind::kStatePoolFull || kind == ErrorKind::kEventPoolFull ||
           kind == ErrorKind::kEdgePoolFull || kind == ErrorKind::kVertexPoolFull ||
           kind == ErrorKind::kMatchPoolFull;
}

bool grow_config_for(EngineConfig& cfg, ErrorKind kind);

void grow_size_pools(EngineConfig& cfg) {
    for (ErrorKind k : {ErrorKind::kStatePoolFull, ErrorKind::kEventPoolFull,
                        ErrorKind::kEdgePoolFull, ErrorKind::kVertexPoolFull})
        grow_config_for(cfg, k);
}

QeEntries qe_entries(const EngineConfig& cfg) {
    auto pick = [&](uint32_t set, uint64_t dflt, uint32_t limit) {
        return static_cast<uint32_t>(std::min<uint64_t>(set ? set : dflt, limit));
    };
    const uint64_t e = cfg.max_events;
    return QeEntries{pick(cfg.qe_class_entries, e, kQeEntryLimit),
                     pick(cfg.qe_instance_entries, e, kQeEntryLimit),
                     pick(cfg.qe_event_entries, e, kQeEntryLimit),
                     pick(cfg.qe_pair_entries, e, kQeEntryLimit),
                     pick(cfg.qe_word_entries, 16u * e, kQeWordLimit),
                     cfg.max_states ? cfg.max_states : 1u};
}

bool grow_config_for(EngineConfig& cfg, ErrorKind kind) {
    auto dbl = [](uint32_t& f) { f = (f >= (1u << 31)) ? f : (f * 2u); };
    // A replay group grows from its resolved size, so a group left at its default doubles
    // what the run had rather than what max_events would give after other growth.
    auto dbl_qe = [&](uint32_t& f, uint32_t resolved, uint32_t limit) {
        f = static_cast<uint32_t>(std::min<uint64_t>(2ull * resolved, limit));
    };
    const QeEntries qe = qe_entries(cfg);
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
            dbl(cfg.canonical_form_words);
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
        // Retryable, and it must be: a full dedup map cannot decide whether a state has been
        // seen, so the run keeps states it might have merged and reports an over-complete answer.
        // Growing the map is what turns that warning back into an exact result.
        // Every claim map (states, event signatures, exact hashes, keyed rewrites and twins) is
        // sized from max_states or max_events, so both grow. The replay's run-signature and
        // frame maps report this kind too and are in the class group.
        case ErrorKind::kCanonicalMapFull:
            dbl(cfg.max_states);
            dbl(cfg.max_events);
            if (cfg.qe_class_entries) dbl_qe(cfg.qe_class_entries, qe.classes, kQeEntryLimit);
            return true;
        case ErrorKind::kCanonicalFormsFull:  dbl(cfg.canonical_form_words); return true;
        case ErrorKind::kCausalTripleMapFull: dbl(cfg.causal_triple_slots);  return true;
        case ErrorKind::kCausalPairMapFull:   dbl(cfg.causal_pair_slots);    return true;
        case ErrorKind::kBranchialMapFull:    dbl(cfg.branchial_pair_slots); return true;
        case ErrorKind::kEdgeConsumerNodes:   dbl(cfg.edge_consumer_nodes);  return true;
        case ErrorKind::kBranchialIndexNodes: dbl(cfg.branchial_index_nodes); return true;
        case ErrorKind::kTrPredsNodes:        dbl(cfg.tr_preds_nodes);       return true;
        // Each replay group has its own kind and grows alone. Retryable, and it must be: a full
        // replay table loses applications or causal pairs, and the run reports a relation
        // smaller than the one the host computes.
        case ErrorKind::kQcNodes:
            dbl_qe(cfg.qe_class_entries, qe.classes, kQeEntryLimit);       return true;
        case ErrorKind::kQeInstancesFull:
            dbl_qe(cfg.qe_instance_entries, qe.instances, kQeEntryLimit);  return true;
        case ErrorKind::kQeEventsFull:
            dbl_qe(cfg.qe_event_entries, qe.events, kQeEntryLimit);        return true;
        case ErrorKind::kQePairsFull:
            dbl_qe(cfg.qe_pair_entries, qe.pairs, kQeEntryLimit);          return true;
        case ErrorKind::kQeWordsFull:
            dbl_qe(cfg.qe_word_entries, qe.words, kQeWordLimit);           return true;
        case ErrorKind::kQeWorkOverflow:      dbl(cfg.descent_work_scale);   return true;
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
        case ErrorKind::kTrScratchOverflow:   dbl(cfg.tr_scratch_scale);     return true;
        case ErrorKind::kQeSurvivorsOverflow:
            cfg.survivor_scratch = cfg.survivor_scratch ? cfg.survivor_scratch * 2u : 1024u;
            return true;
        case ErrorKind::kScratchOverflow:
            // A fixed bound no config field sets. It cannot be retried; the caller accepts the
            // truncation.
            return false;
        default: return false;
    }
}

// The fields grow-and-retry changed, after `header`, and the estimated size of `winning`:
// logged for the config that worked, so a caller can pre-size the next call, and for the one
// that did not fit. The format is stable so callers can grep it.
static void log_config_growth(const char* header, const EngineConfig& initial,
                              const EngineConfig& winning) {
#define LOG_FIELD(field) \
    if (winning.field != initial.field) { \
        std::fprintf(stderr, "  %s: %u → %u\n", #field, initial.field, winning.field); \
    }
    std::fprintf(stderr, "hg_gpu::evolve: %s (estimated %llu MB):\n", header,
                 (unsigned long long)(estimated_device_bytes(winning) >> 20));
    LOG_FIELD(max_edges);
    LOG_FIELD(max_vertices);
    LOG_FIELD(max_vertex_slots);
    LOG_FIELD(max_states);
    LOG_FIELD(max_state_edge_total);
    LOG_FIELD(sig_index_pool);
    LOG_FIELD(inverted_pool);
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
    LOG_FIELD(qe_class_entries);
    LOG_FIELD(qe_instance_entries);
    LOG_FIELD(qe_event_entries);
    LOG_FIELD(qe_pair_entries);
    LOG_FIELD(qe_word_entries);
    LOG_FIELD(descent_work_scale);
    LOG_FIELD(tr_scratch_scale);
    LOG_FIELD(survivor_scratch);
    LOG_FIELD(canonical_form_words);
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
    sc(cfg.canonical_form_words, 1u<<14);
}

uint64_t estimated_device_bytes(const EngineConfig& cfg) {
    // Sum the pools EngineState allocates. Element sizes: Edge 24; DeviceEvent
    // 48; DeviceCausal/Branchial edge 12; StateEdgeSlice 8; a LockFreeList node
    // is sizeof(value)+4 rounded up; a ConcurrentMap slot is 16 B at a power-of-two capacity.
    // A 4-byte id is the unit for most index/id pools. Approximate — a 15%
    // headroom covers the small frontier/hash scratch buffers and allocation
    // granularity, so the estimate errs high (refusing borderline growth).
    auto u64 = [](uint32_t v) { return static_cast<uint64_t>(v); };
    // A hash map: 16-byte slots (hash_table.hpp MapSlot), its capacity rounded up to a power of two.
    auto map_bytes = [](uint64_t slots) {
        uint64_t cap = 1;
        while (cap < slots) cap <<= 1;
        return cap * 16;
    };
    uint64_t b = 0;
    b += u64(cfg.max_vertex_slots)    * 4;          // vertex_pool
    b += u64(cfg.max_edges)           * 24;         // edge_pool (Edge)
    b += u64(cfg.max_edges)           * 4;          // edge_producer
    b += u64(cfg.max_states)          * 8;          // state_edge_slices
    // state_edge_ids, and the edge ranks and orbits the canonical and quotient routes allocate
    // (EngineState::ensure_edge_ranks, ensure_edge_orbits).
    b += u64(cfg.max_state_edge_total) * 12;
    b += u64(cfg.sig_index_buckets)   * 4 + u64(cfg.sig_index_pool) * 8;   // signature index
    b += u64(cfg.max_vertices)        * 4 + u64(cfg.inverted_pool)  * 8;   // vertex inverted index
    b += u64(cfg.max_events)          * sizeof(DeviceEvent);   // event_pool
    b += u64(cfg.max_events)          * 4 * event_consumed_stride(cfg);   // event_consumed
    b += u64(cfg.max_causal_edges)    * 12;         // causal_edge_pool
    b += u64(cfg.max_branchial_edges) * 12;         // branchial_edge_pool
    b += u64(cfg.max_edges)           * 4 + u64(cfg.edge_consumer_nodes)   * 8;   // edge_consumers
    b += u64(cfg.branchial_index_buckets) * 4 + u64(cfg.branchial_index_nodes) * 16; // branchial index
    b += map_bytes(cfg.causal_triple_slots);
    b += map_bytes(cfg.causal_pair_slots);
    b += map_bytes(cfg.branchial_pair_slots);
    b += u64(cfg.max_events)          * 4  + u64(cfg.tr_preds_nodes) * 8;  // preds_list
    // QeState, per group entry (quotient.cu QeState::QeState; map slots are 16 B and rounded up
    // to a power of two, which the factor 2 on the maps covers). Omitting it let the
    // grow-and-retry memory cap approve a config the device could not hold.
    //   class:    matches 72, by_from 24, rep 16, frame 32, canon_seen 32, multiplicity maps
    //             96 and arrays 32: 304, maps 176 of it
    //   per state: class_nmatch 4 and class_pairs 8, indexed by a class's representative
    //   instance: instance 28, bound item 16, instance list 24: 68
    //   event:    tasks 2 x 24, applied lists 2 x 24, content 8 + 8, kept 20: 132
    //   pair:     applied 4 x 16, causal pairs 4 x 16: 128, all map
    //   word:     4
    {
        const QeEntries qe = qe_entries(cfg);
        b += u64(qe.classes) * (304 + 176) + u64(qe.instances) * 68 + u64(qe.events) * 132 +
             u64(qe.pairs) * 256 + u64(qe.words) * 4 + u64(qe.states) * 12;
    }
    // The multiplicity queues at their minimum per-driver size (256 items), which
    // descent_work_scale multiplies; a deep run's queues are larger still.
    b += u64(default_persistent_grid()) * 256u * u64(cfg.descent_work_scale) *
         sizeof(QeWorkItem);
    // Reachability scratch and its busy words.
    b += u64(default_persistent_grid()) * 4u *
         (u64(EngineState::tr_scratch_scale_of(cfg)) *
              (EngineState::kTrScratchStack + EngineState::kTrScratchVisited) + 1u);
    b += u64(default_persistent_grid()) * u64(cfg.survivor_scratch) * 8u;    // survivor scratch
    // The claim maps: states and exact hashes at two slots per state, event signatures at two
    // per event (persistent.cu reuse_map).
    b += map_bytes(u64(cfg.max_states) * 2) * 2 + map_bytes(u64(cfg.max_events) * 2);
    b += u64(cfg.canonical_form_words) * 4;         // canonical form records
    if (cfg.keyed_rewrites) {
        // Keyed rewrites: per-state token sum and first produced edge, the rewrite map (two
        // slots per event) and the twin map (two slots per state).
        b += u64(cfg.max_states) * 12 + map_bytes(u64(cfg.max_events) * 2) +
             map_bytes(u64(cfg.max_states) * 2);
    }
    b += map_bytes(cfg.match_dedup_slots) + map_bytes(cfg.event_canon_slots);
    b += u64(cfg.max_states)          * 8 * 76;     // matches pool (max_states*8 records ~76B)
    b += u64(cfg.max_states)          * 16;         // d_frontier + d_next_frontier + state_canonical_hash

    // The device stack: EngineState's constructor sets cudaLimitStackSize, and the driver
    // reserves that per-thread size for every thread the device can hold resident, not for the
    // grid launched (CapacityOverflow.TheEstimateCoversTheAllocation).
    b += static_cast<uint64_t>(EngineState::device_stack_bytes()) * device_resident_threads();

    return b + b / 6;   // ~17% headroom
}

// THE GROW-AND-RETRY LADDER, for evolve() and PersistentEvolver::run.
//
// `attempt(cfg)` runs the input on an engine of that config and returns the result, or throws
// when the engine cannot be built or the device run fails. A result with retryable overflow
// warnings doubles the config fields those warnings name and runs again, up to kMaxRetries
// times. The result returned carries its own warnings only: the attempts before it were
// discarded, and each is logged to stderr with the overflow that ended it. A clean attempt
// therefore returns no warnings. When the ladder stops with an overflow still present (no
// retryable warning, the retry ceiling, the memory cap, or a throw), the caller gets that
// attempt's partial result and its warnings, plus one saying why the ladder stopped.
//
// Eight retries, not six. The ladder doubles ONE knob per retry and the replay's groups size
// pools that are filled per APPLICATION while their base counts EVENTS -- measured on
// disc-l3a2g2r2 depth 3, 970,584 applications against 4,512 events, 215 to 1. At a 64x ceiling
// qe_events reaches 288,768 and the applied pool 577,536, and the run reported needing at least
// 354,113 / 402,555 / 390,638 more slots across its attempts: short by one doubling, three times
// over.
template <class Attempt>
static EvolveResult run_with_growth(EngineConfig cfg, uint64_t mem_cap, Attempt&& attempt) {
    constexpr int kMaxRetries = 8;  // up to 256x capacity growth

    // The device-memory ceiling: explicit request, else 90% of total VRAM. The total does not
    // change, so it is queried once per process.
    if (mem_cap == 0) {
        static const uint64_t total_vram = [] {
            size_t freeB = 0, totalB = 0;
            const bool ok = cudaMemGetInfo(&freeB, &totalB) == cudaSuccess;
            cudaGetLastError();  // clear any sticky status from the query
            return ok ? static_cast<uint64_t>(totalB) : uint64_t{0};
        }();
        mem_cap = static_cast<uint64_t>(static_cast<double>(total_vram) * 0.90);
    }
    // Shrink the initial config to the ceiling if it was sized past it; the ladder then never
    // grows back over the cap.
    if (mem_cap != 0 && estimated_device_bytes(cfg) > mem_cap) {
        fit_config_to_cap(cfg, mem_cap);
        std::fprintf(stderr,
            "hg_gpu::evolve: initial config exceeded the memory cap (%llu MB) — "
            "scaled pools down to ~%llu MB; result may be partial.\n",
            (unsigned long long)(mem_cap >> 20),
            (unsigned long long)(estimated_device_bytes(cfg) >> 20));
    }
    const EngineConfig initial_cfg = cfg;

    // Best partial result seen so far: an attempt that overflowed still returns whatever it
    // computed, and if the next, larger engine cannot be built that partial is what the caller
    // gets, never an exception.
    EvolveResult best;
    int first_shrinks = 0;

    for (int attempt_no = 0; attempt_no <= kMaxRetries; ++attempt_no) {
        EvolveResult result;
        try {
            result = attempt(cfg);
        } catch (const std::exception& e) {
            // No attempt has completed, so there is no partial result to return. The first
            // config is sized from Steps and can be larger than the free device memory (Steps
            // 1,000,000 asks for 16.7 GB): it is halved, up to kFirstShrinks times, and the run
            // tried again, so a run that outgrows the smaller engine returns partial work.
            constexpr int kFirstShrinks = 3;
            if (attempt_no == 0 && first_shrinks < kFirstShrinks) {
                ++first_shrinks;
                fit_config_to_cap(cfg, estimated_device_bytes(cfg) / 2);
                std::fprintf(stderr,
                    "hg_gpu::evolve: the first engine failed (%s) -- halving it to ~%llu MB and "
                    "retrying.\n", e.what(),
                    (unsigned long long)(estimated_device_bytes(cfg) >> 20));
                --attempt_no;
                continue;
            }
            best.warnings.push_back(OverflowWarning{
                ErrorKind::kDeviceOutOfMemory, 1u,
                std::string("attempt ") + std::to_string(attempt_no + 1) + ": " + e.what()});
            std::fprintf(stderr,
                "hg_gpu::evolve: the engine at this size failed (%s) — returning the last "
                "completed attempt's partial result.\n", e.what());
            log_config_growth("the size that failed", initial_cfg, cfg);
            return best;
        }

        if (result.warnings.empty()) {
            if (attempt_no > 0)
                log_config_growth("succeeded after grow-and-retry; pass these to Engine(cfg) "
                                  "directly to skip the retry loop next time",
                                  initial_cfg, cfg);
            return result;
        }

        // Grow the config for every retryable warning of THIS attempt. grow_config_for is
        // idempotent under repeats, so every warning's kind is swept.
        bool any_retryable = false;
        ErrorKind first_grew = ErrorKind::kCount;
        bool size_grown = false;
        for (const auto& w : result.warnings) {
            if (is_size_pool(w.kind)) {
                if (!size_grown) grow_size_pools(cfg);
                size_grown = true;
                if (!any_retryable) first_grew = w.kind;
                any_retryable = true;
                continue;
            }
            if (grow_config_for(cfg, w.kind)) {
                if (!any_retryable) first_grew = w.kind;
                any_retryable = true;
            }
        }
        // THE REPLAY'S POOLS GROW 4x PER ATTEMPT when the grown config stays under the memory
        // cap, 2x otherwise. They start from the event budget while the replay grows with raw
        // applications, exponentially in depth: doubling alone takes multirule at 7 steps
        // through five attempts (replay tables 32x, descent_work_scale 4x), each a full run
        // from the start.
        {
            EngineConfig quad = cfg;
            bool replay_grew = false;
            for (const auto& w : result.warnings)
                if (w.kind == ErrorKind::kQcNodes || w.kind == ErrorKind::kQeInstancesFull ||
                    w.kind == ErrorKind::kQeEventsFull || w.kind == ErrorKind::kQePairsFull ||
                    w.kind == ErrorKind::kQeWordsFull || w.kind == ErrorKind::kQeWorkOverflow)
                    replay_grew = grow_config_for(quad, w.kind) || replay_grew;
            if (replay_grew && (mem_cap == 0 || estimated_device_bytes(quad) <= mem_cap))
                cfg = quad;
        }
        if (!any_retryable || attempt_no == kMaxRetries) return result;

        // Would the grown config exceed the memory ceiling? Then stop here and return the
        // partial rather than pushing toward a real device OOM.
        if (mem_cap != 0 && estimated_device_bytes(cfg) > mem_cap) {
            result.warnings.push_back(OverflowWarning{
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
            return result;
        }

        best = std::move(result);
        std::fprintf(stderr,
            "hg_gpu::evolve: overflow on %s — growing relevant config and "
            "retrying (attempt %d/%d).\n",
            error_kind_name(first_grew), attempt_no + 2, kMaxRetries + 1);
    }
    return EvolveResult{};  // unreachable: every iteration returns or continues
}

EvolveResult evolve(const EvolveInput& in) {
    // A fresh engine per attempt: the pools are re-allocated at the new sizes.
    return run_with_growth(config_from_input(in), in.max_device_memory_bytes,
                           [&](const EngineConfig& cfg) {
                               // The estimate is below the real allocation, so a config it
                               // already places past the free memory cannot fit; stopping here
                               // skips building it in paged memory (run_persistent_evolve
                               // checks the real allocation).
                               size_t free_b = 0, total_b = 0;
                               if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess &&
                                   estimated_device_bytes(cfg) + total_b / 64 > free_b)
                                   throw std::runtime_error(
                                       "the engine does not fit in device memory (" +
                                       std::to_string(free_b >> 20) + " MiB free)");
                               cudaGetLastError();
                               Engine engine(cfg);
                               return engine.run(in);
                           });
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
uint32_t GpuSession::state_capacity() const { return impl_->view.explore.max_states; }
uint32_t GpuSession::frontier_size() const { return impl_->state.frontier_size(); }
void GpuSession::frontier_host(std::vector<StateId>& ids, std::vector<uint32_t>& steps) const {
    impl_->state.frontier_host(ids, steps);
}
void GpuSession::set_frontier_host(const StateId* ids, const uint32_t* steps, uint32_t n) {
    impl_->state.set_frontier_host(ids, steps, n);
}

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

    // Opening: size the engine from this input, exactly as a first run would. A live engine
    // whose event_consumed stride is narrower than this input's largest left-hand side is
    // rebuilt too, and so is one with more state ids than the session's per-state arrays cover:
    // its states would index them past their end.
    const uint32_t session_states = session->explore.max_states;
    if (has_engine_ && start_step == 0 &&
        (config_from_input(in).max_lhs_edges > cfg_.max_lhs_edges ||
         cfg_.max_states > session_states)) {
        engine_.reset();
        has_engine_ = false;
    }
    if (!has_engine_) {
        if (start_step != 0) {
            out.error = "this session has no engine to continue; it was never opened, or a "
                        "previous call discarded it";
            return out;
        }
        EngineConfig cfg = config_from_input(in);
        if (cfg.max_states > session_states) {
            out.error = "the session covers " + std::to_string(session_states) +
                        " states and this input sizes an engine of " +
                        std::to_string(cfg.max_states) + "; open the session at that size";
            return out;
        }
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
        out.result = engine_->run(in, session, start_step, &spare_);
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

void PersistentEvolver::recycle(EvolveResult&& done) { spare_.adopt_storage(done); }

EvolveResult PersistentEvolver::run(const EvolveInput& in) {
    // Never shrink: start from the live engine's config if there is one, else size to this
    // input. The engine is rebuilt only when the config changes, so a run whose input fits the
    // current engine reuses it and pays no allocation.
    EngineConfig start = config_from_input(in);
    if (has_engine_) {
        // The live config, widened when this input's largest left-hand side needs a wider
        // event_consumed stride; a changed config rebuilds the engine below.
        const uint16_t lhs = start.max_lhs_edges;
        start = cfg_;
        start.max_lhs_edges = std::max(start.max_lhs_edges, lhs);
    }
    return run_with_growth(start, in.max_device_memory_bytes,
                           [&](const EngineConfig& cfg) {
        // On a grow the old engine is freed before the larger one is built, so peak VRAM is
        // bounded by the larger config, not their sum.
        static_assert(std::has_unique_object_representations_v<EngineConfig>,
                      "memcmp of EngineConfig compares padding");
        if (!has_engine_ || std::memcmp(&cfg, &cfg_, sizeof(EngineConfig)) != 0) {
            engine_.reset();
            has_engine_ = false;
            engine_ = std::make_unique<Engine>(cfg);
            cfg_        = cfg;
            has_engine_ = true;
        }
        try {
            return engine_->run(in, nullptr, 0, &spare_);
        } catch (...) {
            // This evolver REUSES engine_ across calls, so an engine a throw left inconsistent
            // would poison every later call in this worker. It is discarded, and the next
            // attempt or call builds a clean one.
            engine_.reset();
            has_engine_ = false;
            cudaGetLastError();
            throw;
        }
    });
}


// =============================================================================
// evolve.hpp host bodies
// =============================================================================
//
// EvolveResult's observable_* answer what a caller is TOLD -- the same question
// Hypergraph::observable_num_causal_pairs answers on the host -- by folding the reconstruction's
// relations or deduplicating the materialised ones. evolve.hpp is the header a host-only
// translation unit includes to avoid cuda_runtime.h, so keeping these out of it is what that
// separation is for.

size_t EvolveResult::observable_num_causal_pairs(bool reduced) const {
        if (reconstruction_ran)
            return reduced ? reconstructed_causal_relation_reduced.size()
                           : reconstructed_causal_relation.size();
        std::set<std::pair<EventId, EventId>> seen;
        for (const auto& c : causal_edges) seen.insert({c.from, c.to});
        return seen.size();
    }

// The reconstruction's count is the device counter: the pair relation is built only when
// in.materialize_relations is set, and a counts-only request leaves it empty.
size_t EvolveResult::observable_num_branchial() const {
        return reconstruction_ran ? reconstructed_branchial : branchial_edges.size();
    }

size_t EvolveResult::observable_num_events() const {
        if (reconstruction_ran) return reconstructed_events;
        size_t n = 0;
        for (const auto& e : events) if (e.canonical_id == INVALID_ID) ++n;
        return n;
    }

bool PersistentEvolver::has_engine() const { return has_engine_; }

const EngineConfig& PersistentEvolver::engine_config() const { return cfg_; }

}  // namespace gpu
}  // namespace HG_NAMESPACE
