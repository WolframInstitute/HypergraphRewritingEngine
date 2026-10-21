#ifdef HG_GPU_BACKEND

#include "hg_gpu_backend.hpp"
#include <algorithm>

#include "hg_gpu/device_probe.hpp"
#include "hgcommon/affinity.hpp"   // default_worker_count
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/branchial_rows.hpp"
#include "hg_gpu/step_pool.hpp"
#include "hgcommon/quotient_replay_core.hpp"
#include "hgcommon/quotient_causal_core.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "wxf.hpp"
#include "graph_marshal.hpp"
#include "state_statistics.hpp"
#include "session.hpp"

#include <algorithm>
#include <atomic>
#include <optional>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// A genesis event's "RuleIndex": the host's RuleIndex(-1), a 16-bit rule index.
constexpr int64_t kGenesisRuleIndex = 0xFFFF;

// Class points from which "StepStatistics" pools on the device.
constexpr size_t kDevicePoolMinClasses = 1024;

// Every step's pool made on the device from its classes (hg_gpu::step_pool_device), each class's
// record a word offset into `pool`, and stored as the step's device_part. Nothing changes when
// the device cannot: the host then pools.
void device_step_pool(const std::vector<uint64_t>& pool, bool weight_by_classes,
                      std::map<uint32_t, hg::stats::StepInput>& inputs) {
    std::vector<hg_gpu::StepPoolPoint> points;
    uint32_t steps = 0;
    for (const auto& [step, in] : inputs) {
        steps = std::max(steps, step + 1);
        for (const auto& [hash, c] : in.classes) {
            const uint64_t word =
                c.record ? static_cast<uint64_t>(reinterpret_cast<const uint64_t*>(c.record) - pool.data())
                         : UINT64_MAX;
            points.push_back({step, word, weight_by_classes ? 1 : c.weight, c.weight});
        }
    }
    std::vector<std::vector<hgcommon::SpEntry>> entries;
    std::vector<hg_gpu::StepPoolTotals> totals;
    if (!hg_gpu::step_pool_device(pool, points, steps, entries, totals)) return;
    for (auto& [step, in] : inputs) {
        if (in.classes.empty()) continue;
        auto part = std::make_unique<hg::stats::PoolPart>();
        part->entries = std::move(entries[step]);
        for (uint32_t k = 0; k < hgcommon::SG_DISTS; ++k) part->dist[k] = totals[step][k];
        in.device_part = std::move(part);
    }
}

// Build a hg_gpu::EvolveInput from the parsed job. Rule vertices double as
// pattern-variable indices; each initial state's vertices are remapped to
// 0..n-1 (matching the FFI, so isomorphic roots share a representation).
hg_gpu::EvolveInput build_input(const GpuJob& job) {
    hg_gpu::EvolveInput in;

    // Rule variables are validated BEFORE the uint8_t narrowing: a value at or above
    // MAX_VARS would wrap through the cast and pass the device-side check as the WRONG
    // variable, silently merging two distinct pattern variables. The contract mirrors the
    // CPU FFI exactly -- malformed shape, negative variable, or out-of-domain variable is
    // an error, an empty RHS is a legitimate deleting rule ({{x,y}} -> {} reaches the
    // empty state), and only an empty LHS is rejected. make_device_rule re-validates the
    // dimension counts downstream.
    size_t rule_index = 0;
    for (const auto& [name, parts] : job.rules) {
        (void)name;
        if (parts.size() != 2) {
            throw std::runtime_error("rule " + std::to_string(rule_index) +
                                     " is not lhs -> rhs (expected 2 parts, got " +
                                     std::to_string(parts.size()) + ")");
        }
        auto checked_var = [&](int64_t v, const char* side) -> uint8_t {
            if (v < 0) {
                throw std::runtime_error("rule " + std::to_string(rule_index) + " " + side +
                                         " has a negative pattern variable");
            }
            if (v >= static_cast<int64_t>(hgcommon::MAX_VARS)) {
                throw std::runtime_error(
                    "rule " + std::to_string(rule_index) + " " + side +
                    " uses pattern variable " + std::to_string(v) + ", but the maximum is " +
                    std::to_string(hgcommon::MAX_VARS - 1));
            }
            return static_cast<uint8_t>(v);
        };
        hg_gpu::RewriteRule r;
        uint8_t lhs_max = 0, rhs_max = 0;
        for (const auto& edge : parts[0]) {
            std::vector<uint8_t> e;
            for (int64_t v : edge) {
                const uint8_t u = checked_var(v, "LHS");
                e.push_back(u);
                lhs_max = std::max<uint8_t>(lhs_max, u);
            }
            if (!e.empty()) r.lhs.push_back(std::move(e));
        }
        for (const auto& edge : parts[1]) {
            std::vector<uint8_t> e;
            for (int64_t v : edge) {
                const uint8_t u = checked_var(v, "RHS");
                e.push_back(u);
                rhs_max = std::max<uint8_t>(rhs_max, u);
            }
            if (!e.empty()) r.rhs.push_back(std::move(e));
        }
        if (r.lhs.empty()) {
            throw std::runtime_error("rule " + std::to_string(rule_index) +
                                     " has an empty left-hand side");
        }
        r.num_lhs_vars = static_cast<uint8_t>(lhs_max + 1);
        r.num_rhs_vars = static_cast<uint8_t>(rhs_max + 1);
        in.rules.push_back(std::move(r));
        ++rule_index;
    }

    // Initial-state vertices are labels, renumbered by the rule the CPU engine uses.
    in.initial_states = hgffi::dense_initial_states<hg_gpu::VertexId>(job.initial_states);

    in.num_steps = static_cast<uint32_t>(std::max(0, job.steps));
    // State dedup / class collapse is done host-side in the marshaller, keyed by the
    // requested mode (None per-provenance, Automatic edge-content, Full IR class).
    in.canonicalization =
        job.state_canon_mode == GpuJob::StateCanonCode::kNone      ? hg_gpu::CanonicalizationMode::None :
        job.state_canon_mode == GpuJob::StateCanonCode::kAutomatic ? hg_gpu::CanonicalizationMode::Automatic :
                                                                     hg_gpu::CanonicalizationMode::Full;
    in.event_canonicalization =
        job.event_canon_mode == GpuJob::EventCanonCode::kFull      ? hg_gpu::EventCanonicalizationMode::Full :
        job.event_canon_mode == GpuJob::EventCanonCode::kAutomatic ? hg_gpu::EventCanonicalizationMode::Automatic :
                                                                     hg_gpu::EventCanonicalizationMode::None;
    // What this call must RECORD, from what it will return. Same derivation as the CPU FFI,
    // and the graph properties' needs come from the same graph_property_needs the marshaller
    // builds them with. The device has no per-state event list, so an all-siblings branchial
    // state view is built from the events it returns rather than from a recorded index.
    {
        const hgmarshal::GraphPropertyNeeds gneeds =
            hgmarshal::graph_property_needs(job.graph_properties);
        in.record = hgmarshal::record_set_for(job, gneeds);
        // The reconstructed relations as pair vectors and the applications as events: the
        // record set's enumerations (RecordSet::relation_lists, event_lists).
        in.materialize_relations = in.record.relation_lists;
        in.materialize_events = in.record.event_lists;
        // The reconstruction's genesis pairs: read by every reply that shows genesis events.
        in.genesis_pairs = job.show_genesis_events;
        // State contents: read by the state records, event records (their input and output
        // states), every graph (vertex data), step statistics, a host-computed CanonicalHash,
        // genesis events, the branchial state views, "GlobalEdges" and "StateBitvectors"; a
        // session may be asked for any later.
        in.materialize_state_edges =
            job.include_global_edges || job.include_state_bitvectors ||
            job.include_states || job.include_events || !job.graph_properties.empty() ||
            job.include_step_statistics || job.include_canonical_hashes ||
            job.show_genesis_events || job.include_branchial_state_edges ||
            job.include_branchial_state_edges_all_siblings || job.session_op == "Open";
    }
    in.transitive_reduction = job.transitive_reduction;
    in.explore_from_canonical_states_only = job.explore_from_canonical_states_only;
    in.exploration_probability = job.exploration_probability;
    in.exploration_seed = job.random_seed;
    in.transition_rate = job.transition_rate;
    in.rule_weights = job.rule_weights;
    // The device's caps are 32-bit. A larger cap saturates, where a cast would keep the low bits
    // (2^32 + 1 became a cap of 1) while the host treats it as unreachable.
    auto cap32 = [](uint64_t v) {
        return v > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(v);
    };
    in.max_states_per_step = cap32(job.max_states_per_step);
    in.step_cap_counts_matches = job.step_cap_counts_matches;
    in.max_successor_states_per_parent = cap32(job.max_successor_states_per_parent);
    in.matches_per_state_rule = cap32(job.matches_per_state_rule);
    in.max_device_memory_bytes = job.max_device_memory_bytes;
    in.persistent_blocks = job.persistent_blocks;
    in.lane_backlog = job.lane_backlog;
    in.l2_persist_limit = job.l2_persist_limit;
    return in;
}

// THE HELD SESSION, if any. One per process: a session pins the engine, because a rebuild
// would drop its accumulated states while handing back something shaped like a
// continuation, and the worker runs jobs serially against one device anyway.
//
// `last` is kept so Query costs nothing: it reports what the session holds and extends by
// nothing, which is exactly the previous result.
struct HeldSession {
    std::unique_ptr<hg_gpu::GpuSession> state;
    hg_gpu::EvolveInput  input;
    hg_gpu::EvolveResult last;
    // Host mirror of the device frontier, read back after every run. A steered Step is
    // resolved by hgffi::steered_entries against `frontier_eff`, built when the frontier was
    // last reported, so the ids it reads are the ids the caller read.
    std::vector<hg_gpu::StateId>          frontier_ids;
    std::vector<uint32_t>                 frontier_steps;
    std::vector<int64_t>                  frontier_eff;   // entry i's effective id, as reported
    uint32_t steps_done = 0;
    uint64_t handle     = 0;
    // The Open's identity and relation settings. A held verb is served under these, as the host
    // reads them back from its engine (hypergraph_ffi.cpp, read_back_session_identity).
    int  event_canon_mode = 0;
    int  state_canon_mode = 0;
    bool show_genesis_events = false;
    bool transitive_reduction = true;
    bool explore_from_canonical_states_only = false;
};
HeldSession held;

// Reuse one device Engine across every job this process handles. The
// persistent worker processes many HGEvolve calls in one process, and the
// per-call Engine allocation dominates small/medium runs, so amortizing it
// is 6-12x on interactive workloads. Jobs run serially through the worker, so
// a process-lifetime evolver is safe; the one-shot binary just uses it once.
// A job reuses the engine when the engine's config covers the job (PersistentEvolver::run).
hg_gpu::PersistentEvolver& process_evolver() {
    static hg_gpu::PersistentEvolver evolver;
    return evolver;
}

std::vector<uint8_t> run_gpu_evolution_once(const GpuJob& request, const HostBridge& host);

}  // namespace

uint64_t gpu_session_handle() { return held.handle; }

// A job that throws leaves the worker as a fresh one would be: the evolver drops its engine
// and storage and the device is reset (PersistentEvolver::release_after_failure), unless a
// session is held, whose states live on that engine.
std::vector<uint8_t> run_gpu_evolution(const GpuJob& request, const HostBridge& host) {
    try {
        return run_gpu_evolution_once(request, host);
    } catch (...) {
        if (held.handle == 0) process_evolver().release_after_failure();
        throw;
    }
}

namespace {

std::vector<uint8_t> run_gpu_evolution_once(const GpuJob& request, const HostBridge& host) {
    GpuJob job = request;
    if ((job.session_op == "Step" || job.session_op == "Query") && held.handle != 0 &&
        job.session_handle == held.handle) {
        job.event_canon_mode = held.event_canon_mode;
        job.state_canon_mode = held.state_canon_mode;
        job.show_genesis_events = held.show_genesis_events;
        job.transitive_reduction = held.transitive_reduction;
        job.explore_from_canonical_states_only = held.explore_from_canonical_states_only;
    }
    hg_gpu::EvolveInput in = build_input(job);

    hg_gpu::PersistentEvolver& evolver = process_evolver();

    const std::string& op = job.session_op;
    const bool is_open  = (op == "Open");
    const bool is_step  = (op == "Step");
    const bool is_query = (op == "Query");
    const bool is_close = (op == "Close");
    if (!op.empty() && !is_open && !is_step && !is_query && !is_close && op != "Evolve") {
        throw std::runtime_error(
            "Op '" + op + "' is not a verb; 'Evolve', 'Open', 'Step', 'Query' and 'Close' are");
    }
    if ((is_step || is_query || is_close) && (held.handle == 0 || job.session_handle != held.handle)) {
        throw std::runtime_error(
            "Op '" + op + "' names a session this worker does not hold. A GPU session lives in "
            "the worker process that opened it, so a restarted worker invalidates it rather "
            "than reissuing the handle.");
    }

    if (is_close) {
        held.state.reset();
        held = HeldSession{};
        return hgmarshal::session_ack(0);
    }

    hg_gpu::EvolveResult result;
    // An Open whose reply is not built hands the caller no handle, so a session it cannot name
    // or close would refuse every later Open (D7). It is released if the job throws before its
    // reply is returned.
    struct UndeliveredOpen {
        bool armed = false;
        ~UndeliveredOpen() {
            if (!armed) return;
            held.state.reset();
            held = HeldSession{};
        }
    } undelivered_open;
    if (is_open || is_step) {
        if (is_open && held.handle != 0) {
            // The same refusal the host gives, from the one place that spells it.
            throw std::runtime_error(hgffi::SessionSlot::already_live_message(held.handle));
        }
        if (is_open) {
            const hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
            held.state = std::make_unique<hg_gpu::GpuSession>(cfg.max_states, cfg.max_events);
            held.input = in;
            held.steps_done = 0;
            held.handle = hgffi::SessionSlot::mint_handle();
            held.event_canon_mode = job.event_canon_mode;
            held.state_canon_mode = job.state_canon_mode;
            held.show_genesis_events = job.show_genesis_events;
            held.transitive_reduction = job.transitive_reduction;
            held.explore_from_canonical_states_only = job.explore_from_canonical_states_only;
            undelivered_open.armed = true;
        }
        // A STEERED STEP: the caller's effective ids are resolved against the frontier as it
        // was last REPORTED, the selected entries are written down as the whole device
        // frontier, and the unselected ones are PUT BACK after the run -- so a later Step can
        // still resume them, exactly the host's retention contract. Resolving against the
        // frontier rather than every state is what makes an id that is not on it an ERROR
        // rather than a silent no-op.
        std::vector<hg_gpu::StateId> retained_ids;
        std::vector<uint32_t>        retained_steps;
        if (is_step && !job.session_from.empty()) {
            std::vector<char> take(held.frontier_ids.size(), 0);
            const std::vector<int64_t> none;
            const auto& entry_ids =
                held.frontier_eff.size() == held.frontier_ids.size() ? held.frontier_eff : none;
            for (size_t i : hgffi::steered_entries(entry_ids, job.session_from)) take[i] = 1;
            std::vector<hg_gpu::StateId> sel_ids;
            std::vector<uint32_t>        sel_steps;
            for (size_t i = 0; i < take.size(); ++i) {
                (take[i] ? sel_ids : retained_ids).push_back(held.frontier_ids[i]);
                (take[i] ? sel_steps : retained_steps).push_back(held.frontier_steps[i]);
            }
            held.state->set_frontier_host(sel_ids.data(), sel_steps.data(),
                                          static_cast<uint32_t>(sel_ids.size()));
        }
        // A Step's budget is the TOTAL depth, and start_step is where the last call stopped --
        // the frontier is seeded at that depth rather than the roots at zero.
        const uint32_t from = held.steps_done;
        held.input.num_steps = from + in.num_steps;
        auto sr = evolver.run_session(held.input, held.state->view(), from);
        if (!sr.ok) {
            held.state.reset();
            held = HeldSession{};
            throw std::runtime_error("GPU session: " + sr.error);
        }
        held.steps_done = held.input.num_steps;
        held.last = std::move(sr.result);
        // The run consumed the selection and appended its own boundary; the retained entries
        // rejoin it now, at the depths they were stranded at. Entries past the session's
        // capacity are dropped with the same warning kind the device append records.
        held.state->frontier_host(held.frontier_ids, held.frontier_steps);
        // The map indexes the frontier as last reported; it is rebuilt with this reply, and until
        // then a steered Step resolves nothing rather than an index into the replaced list.
        held.frontier_eff.clear();
        if (!retained_ids.empty()) {
            held.frontier_ids.insert(held.frontier_ids.end(),
                                     retained_ids.begin(), retained_ids.end());
            held.frontier_steps.insert(held.frontier_steps.end(),
                                       retained_steps.begin(), retained_steps.end());
            held.state->set_frontier_host(held.frontier_ids.data(), held.frontier_steps.data(),
                                          static_cast<uint32_t>(held.frontier_ids.size()));
            const uint32_t kept = held.state->frontier_size();
            if (kept < held.frontier_ids.size()) {
                held.last.warnings.push_back(
                    {hg_gpu::ErrorKind::kFrontierCapFull,
                     static_cast<uint32_t>(held.frontier_ids.size()) - kept,
                     "steered Step put-back"});
                held.frontier_ids.resize(kept);
                held.frontier_steps.resize(kept);
            }
        }
        result = held.last;
    } else if (is_query) {
        result = held.last;
    } else if (held.handle != 0) {
        // The held session owns the evolver's engine: running this job on it would reset or
        // rebuild the graph the session extends. It runs on an engine of its own.
        result = hg_gpu::evolve(in);
    } else {
        result = evolver.run(in);
    }
    // The depth of the evolution the reply describes: a held session's whole depth, not this
    // job's own step count, which is 0 for a Query. The host does the same (engine.max_steps()).
    const int run_steps = (is_step || is_query) ? static_cast<int>(held.steps_done) : job.steps;

    // The host workers that build the reply's sections, started by the first of them.
    std::optional<hgmarshal::HostJobs> host_jobs;
    auto reply_jobs = [&]() -> hgmarshal::HostJobs& {
        if (!host_jobs) host_jobs.emplace(job.threads);
        return *host_jobs;
    };

    // Group the GPU states by the requested canonicalization mode, mirroring the CPU:
    //   None      -> every state is distinct (per-provenance / tree mode, no grouping)
    //   Automatic -> exact edge-content (non-isomorphic) equality
    //   Full      -> IR isomorphism class
    // The class representative (first-seen id) is the stable handle emitted. state_hash always
    // holds the IR canonical hash so the optional CanonicalHash output is available in any mode.
    const hg_gpu::CanonicalizationMode canon_mode = in.canonicalization;
    std::unordered_map<uint64_t, hg_gpu::StateId> hash_to_rep;
    std::unordered_map<hg_gpu::StateId, hg_gpu::StateId> state_to_rep;
    std::unordered_map<hg_gpu::StateId, uint64_t> state_hash;
    std::unordered_map<hg_gpu::StateId, const hg_gpu::CanonicalState*> state_by_id;
    std::vector<hg_gpu::StateId> class_reps;
    // Under Full the device's key IS the exact isomorphism hash, from the same ir_core the host
    // runs, so it is read rather than recomputed. Under None and Automatic the device key is not
    // isomorphism-invariant: the device's exact hash is read when the run recorded it
    // (CanonicalState::exact_hash, "StepStatistics"), and otherwise the IR hash is computed here
    // only for the outputs that read it.
    const bool host_ir = canon_mode != hg_gpu::CanonicalizationMode::Full &&
                         (job.include_canonical_hashes || job.include_step_statistics);
    // The host IR hashes, one per state the device left without one, computed by the reply jobs.
    std::vector<uint64_t> host_hash;
    if (host_ir) {
        host_hash.assign(result.states.size(), 0);
        wxf::Writer unused;
        hgmarshal::write_items(unused, result.states.size(), reply_jobs().parallel_for(),
                               [&](wxf::Writer&, size_t i) {
            const hg_gpu::CanonicalState& s = result.states[i];
            if (s.id != hg_gpu::INVALID_ID && !s.exact_hash)
                host_hash[i] = hypergraph::IRCanonicalizer{}.compute_canonical_hash(result.edges_of(s));
        });
    }
    for (size_t si = 0; si < result.states.size(); ++si) {
        const hg_gpu::CanonicalState& s = result.states[si];
        // A state slot a failed rewrite claimed (a partial result) reads back with id INVALID_ID.
        if (s.id == hg_gpu::INVALID_ID) continue;
        state_hash[s.id] = hgmarshal::reported_state_hash(
            s.edge_count == 0,
            canon_mode == hg_gpu::CanonicalizationMode::Full ? s.canonical_hash
            : s.exact_hash ? s.exact_hash
            : host_ir ? host_hash[si] : 0);
        state_by_id[s.id] = &s;
        // Automatic groups by the key THE DEVICE DEDUPLICATED WITH. CanonicalState::canonical_hash
        // carries what state_key_device wrote for the requested mode, so under Automatic it is
        // the content hash the evolution itself used. Recomputing content identity here would be
        // a second opinion about it, and a second opinion that disagreed would merge states the
        // device had kept distinct -- one silently missing from the result.
        //
        // A hash of 0 means the device never wrote one for this state. Grouping by it would put
        // every such state in ONE class and drop the rest from the result, so an absent key
        // falls back to the state's own id: unmerged is a visible over-count, merged is a
        // silent loss. This is the EMPTY=0 collision class that has bitten four maps here.
        const uint64_t auto_key = s.canonical_hash != 0 ? s.canonical_hash
                                                        : static_cast<uint64_t>(s.id);
        uint64_t key =
            canon_mode == hg_gpu::CanonicalizationMode::None      ? static_cast<uint64_t>(s.id) :
            canon_mode == hg_gpu::CanonicalizationMode::Automatic ? auto_key
                                                                  : state_hash[s.id];
        auto it = hash_to_rep.find(key);
        if (it == hash_to_rep.end()) {
            hash_to_rep[key] = s.id;
            state_to_rep[s.id] = s.id;
            class_reps.push_back(s.id);
        } else {
            state_to_rep[s.id] = it->second;
        }
    }
    auto rep_of = [&](hg_gpu::StateId s) -> int64_t {
        auto it = state_to_rep.find(s);
        return static_cast<int64_t>(it == state_to_rep.end() ? s : it->second);
    };

    // Per-state producing step + is-initial (a root is no event's output_state).
    std::unordered_map<hg_gpu::StateId, uint32_t> state_step;
    std::unordered_set<hg_gpu::StateId> is_output;
    for (const auto& e : result.events) {
        auto it = state_step.find(e.output_state);
        if (it == state_step.end() || e.step < it->second) state_step[e.output_state] = e.step;
        is_output.insert(e.output_state);
    }

    // The Step a state record and its graph vertex data report (hgmarshal::reported_state_step,
    // the host's rule): under Full the class's explore depth under quotient exploration, the
    // least of its states' steps otherwise; outside Full the state's own step.
    const bool full_states = canon_mode == hg_gpu::CanonicalizationMode::Full;
    auto own_step = [&](hg_gpu::StateId s) -> uint32_t {
        auto it = state_step.find(s);
        return it == state_step.end() ? 0u : it->second;
    };
    std::unordered_map<hg_gpu::StateId, uint32_t> class_least, class_depth;   // by class rep
    if (full_states) {
        for (const auto& st : result.states) {
            if (st.id == hg_gpu::INVALID_ID) continue;
            const auto rep = static_cast<hg_gpu::StateId>(rep_of(st.id));
            auto [it, fresh] = class_least.emplace(rep, own_step(st.id));
            if (!fresh) it->second = std::min(it->second, own_step(st.id));
            if (!job.explore_from_canonical_states_only ||
                st.explore_depth == hgcommon::kExploreNoDepth)
                continue;
            auto [jt, first] = class_depth.emplace(rep, st.explore_depth);
            if (!first) jt->second = std::min(jt->second, st.explore_depth);
        }
    }
    auto reported_step = [&](hg_gpu::StateId s) -> uint32_t {
        const auto rep = static_cast<hg_gpu::StateId>(rep_of(s));
        const auto d = class_depth.find(rep);
        const auto l = class_least.find(rep);
        return hgmarshal::reported_state_step(
            full_states, d == class_depth.end() ? hgcommon::kExploreNoDepth : d->second,
            l == class_least.end() ? UINT32_MAX : l->second, own_step(s));
    };

    // GENESIS EVENTS, SYNTHESISED HERE BECAUSE THE DEVICE HAS NONE.
    //
    // On the host a genesis event is a real event: it connects a synthetic genesis state to an
    // initial state, produces that state's edges, carries rule index -1, and registers itself as
    // the PRODUCER of every initial edge -- which is why showing them adds causal edges as well
    // as events. The device mints no such thing, so "ShowGenesisEvents" was reported to the
    // caller as having no effect on the GPU.
    //
    // It is a presentation construct with no rewriting semantics -- nothing consumes a genesis
    // event's output beyond what already consumes the initial state -- so it is built from the
    // result rather than by the device, and costs a run that does not ask for it nothing.
    //
    // Ids are taken ABOVE every id the device issued, so a genesis event and a real one can
    // never collide in the association the caller receives.
    std::vector<hg_gpu::StateId> genesis_roots;
    std::unordered_map<hg_gpu::EdgeId, size_t> initial_edge_root;   // edge -> index in roots
    hg_gpu::StateId genesis_state_id = 0;
    hg_gpu::EventId first_genesis_event = 0;
    if (job.show_genesis_events) {
        hg_gpu::StateId max_sid = 0;
        for (const auto& st : result.states) if (st.id != hg_gpu::INVALID_ID && st.id > max_sid) max_sid = st.id;
        hg_gpu::EventId max_eid = 0;
        for (const auto& e : result.events) if (e.id != hg_gpu::INVALID_ID && e.id > max_eid) max_eid = e.id;
        genesis_state_id   = max_sid + 1u;
        first_genesis_event = max_eid + 1u;

        for (const auto& st : result.states) {
            if (st.id == hg_gpu::INVALID_ID) continue;
            if (is_output.count(st.id)) continue;            // produced by an event: not a root
            const size_t at = genesis_roots.size();
            genesis_roots.push_back(st.id);
            for (auto eid : result.edge_ids(st)) initial_edge_root[eid] = at;
        }
    }

    // THE GENESIS HALF OF THE CAUSAL RELATION (docs/SPEC.md §5.2). A genesis event produces its
    // initial state's edges, and an event that consumed one of them is paired with it; under the
    // transitive reduction only an event that consumed no produced edge
    // (hgcommon::qr_genesis_pair_kept). The device records no producer for an initial edge, so
    // the full-capture pairs are derived here: an edge in a root state's edge list was never
    // produced by a rewrite. The reconstruction's pairs come from the device
    // (EvolveResult::reconstructed_genesis_pairs). Read by "CausalEdges", NumCausalEdges and the
    // causal graphs. Ids are final only below, where the reconstruction may raise
    // first_genesis_event, so these hold root indices.
    std::vector<std::pair<size_t, uint32_t>> genesis_causal;   // (root index, consumer event)
    if (job.show_genesis_events && !result.reconstruction_ran) {
        for (const auto& e : result.events) {
            if (e.id == hg_gpu::INVALID_ID) continue;
            size_t root = SIZE_MAX;
            bool produced = false;
            for (auto c : result.consumed_of(e)) {
                if (c == hg_gpu::INVALID_ID) continue;
                auto it = initial_edge_root.find(c);
                if (it == initial_edge_root.end()) produced = true;
                else root = it->second;
            }
            if (root != SIZE_MAX &&
                hgcommon::qr_genesis_pair_kept(true, produced, job.transitive_reduction))
                genesis_causal.emplace_back(root, e.id);
        }
    }

    wxf::WXFValueAssociation full_result;

    if (job.include_states) {
        // (edge id, vertices) for each of the state's edges.
        auto record_edges = [&](hg_gpu::StateId s) {
            const hg_gpu::CanonicalState& st = *state_by_id.at(s);
            const hg_gpu::EdgeSpan ids = result.edge_ids(st);
            std::vector<std::pair<int64_t, std::vector<uint32_t>>> edges;
            edges.reserve(st.num_edges);
            for (uint32_t k = 0; k < st.num_edges; ++k) {
                const hg_gpu::VertexSpan e = result.edge_vertices(ids[k]);
                edges.emplace_back(static_cast<int64_t>(ids[k]),
                                   std::vector<uint32_t>(e.begin(), e.end()));
            }
            return edges;
        };
        // Every state, or under Full one per class: the host's rule (hypergraph_ffi.cpp, States).
        std::vector<hg_gpu::StateId> emit;
        if (canon_mode == hg_gpu::CanonicalizationMode::Full) {
            emit = class_reps;
        } else {
            for (const auto& s : result.states)
                if (s.id != hg_gpu::INVALID_ID) emit.push_back(s.id);
        }
        const hgmarshal::ParallelFor par = reply_jobs().parallel_for();
        std::vector<int64_t> listed(emit.size());
        std::vector<uint64_t> listed_hash(emit.size());
        {
            wxf::Writer unused;
            hgmarshal::write_items(unused, emit.size(), par, [&](wxf::Writer&, size_t i) {
                listed[i] = static_cast<int64_t>(emit[i]);
                listed_hash[i] = hgmarshal::content_hash_of(record_edges(emit[i]));
            });
        }
        const auto content_id = hgmarshal::lowest_id_by_content(listed, listed_hash);
        wxf::Writer w;
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(emit.size());
        hgmarshal::write_items(w, emit.size(), par, [&](wxf::Writer& out, size_t i) {
            const hg_gpu::StateId s = emit[i];
            out.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            out.write(static_cast<int64_t>(s));
            hgmarshal::StreamRecordSink sink(out);
            hgmarshal::write_state_record(sink,
                hgmarshal::StateRecordIds{
                    static_cast<int64_t>(s), rep_of(s),
                    content_id.at(listed_hash[i]),
                    static_cast<int64_t>(reported_step(s)),
                    job.include_canonical_hashes, static_cast<int64_t>(state_hash.at(s))},
                canon_mode == hg_gpu::CanonicalizationMode::Full, record_edges(s));
        });
        full_result.push_back({wxf::WXFValue("States"), wxf::WXFValue(wxf::Encoded{w.release_data()})});
    }

    // The identity a reconstructed application is REPORTED under, shared by the edge lists and
    // the graph below: the run signature when an event signature is in force, the application id
    // itself under EVENT_SIG_NONE -- where every application is its own event, and the signature
    // array would collapse them.
    const bool recon_ran = result.reconstruction_ran &&
                           !result.reconstructed_event_signature.empty();
    // Read by recon_id, which only "Events", the causal and branchial edge lists and the genesis
    // pairs call; a reply without them builds no map.
    auto recon_dense = [&]() {
        std::unordered_map<uint64_t, int64_t> dense;
        const bool read = job.include_events || job.include_causal_edges ||
                          job.include_branchial_edges || job.show_genesis_events;
        if (read && recon_ran && job.event_canon_mode != 0) {
            for (uint64_t sig : result.reconstructed_event_signature) {
                if (sig == 0ull) continue;
                dense.emplace(sig, static_cast<int64_t>(dense.size()));
            }
        }
        return dense;
    }();
    auto sig_of_run = [&](uint32_t eid) -> uint64_t {
        return eid < result.reconstructed_event_signature.size()
                   ? result.reconstructed_event_signature[eid] : 0ull;
    };
    auto recon_id = [&](uint32_t eid) -> int64_t {
        if (job.event_canon_mode == 0) return static_cast<int64_t>(eid);
        const uint64_t sig = eid < result.reconstructed_event_signature.size()
                                 ? result.reconstructed_event_signature[eid] : 0ull;
        auto it = recon_dense.find(sig);
        return it == recon_dense.end() ? -1 : it->second;
    };
    // The state that stands for each class the reconstruction names: a replayed application's
    // input and output classes are canonical hashes, and these states are "States" keys.
    std::unordered_map<uint64_t, int64_t> state_of_class;
    for (const auto& st : result.states) state_of_class.emplace(st.canonical_hash, rep_of(st.id));
    auto class_state = [&](uint64_t h) -> int64_t {
        auto it = state_of_class.find(h);
        return it == state_of_class.end() ? -1 : it->second;
    };
    // The reconstruction's applications with their classes and rules, when read back. The graphs
    // are drawn over them whenever present, as on the host, and "Events" lists them, so the
    // relations' ids join with it.
    const bool recon_content = recon_ran && !result.reconstructed_event_from_class.empty();
    const bool recon_events = recon_content;
    // "Events" and the relations are then keyed by application id or by identity, every one
    // below the count of applications, so genesis ids start there, as the host's start at its id
    // bound.
    if (recon_ran)
        first_genesis_event = std::max<hg_gpu::EventId>(
            first_genesis_event,
            static_cast<hg_gpu::EventId>(std::max<uint64_t>(
                result.reconstructed_raw_events, result.reconstructed_event_signature.size())));
    // The reconstruction's genesis pairs as (root index, application), one per (genesis event,
    // application), as full capture pairs them.
    std::vector<std::pair<size_t, uint32_t>> recon_genesis;
    if (recon_ran && job.show_genesis_events) {
        std::unordered_map<hg_gpu::StateId, size_t> root_index;
        for (size_t i = 0; i < genesis_roots.size(); ++i) root_index.emplace(genesis_roots[i], i);
        for (const auto& [root, ev] : result.reconstructed_genesis_pairs) {
            auto it = root_index.find(root);
            if (it != root_index.end()) recon_genesis.emplace_back(it->second, ev);
        }
    }
    // Every application the replay minted has content; under an event identity it must also
    // carry the identity the count groups by.
    auto recon_app_valid = [&](uint32_t e) -> bool {
        if (e >= result.reconstructed_event_from_class.size()) return false;
        return job.event_canon_mode == 0 || sig_of_run(e) != 0ull;
    };
    if (job.include_events) {
        // The records in reply order: the reconstruction's applications, or the device's
        // events, then one genesis event per initial state.
        std::vector<uint32_t> apps;
        if (recon_events)
            for (uint32_t e = 0; e < result.reconstructed_event_from_class.size(); ++e)
                if (recon_app_valid(e)) apps.push_back(e);
        const size_t raw_events = recon_events ? 0 : result.events.size();
        const size_t n = apps.size() + raw_events + genesis_roots.size();
        wxf::Writer w;
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(n);
        hgmarshal::write_items(w, n, reply_jobs().parallel_for(), [&](wxf::Writer& out, size_t i) {
            hgmarshal::StreamRecordSink sink(out);
            std::vector<int64_t> consumed, produced;
            if (i < apps.size()) {
                const uint32_t e = apps[i];
                const int64_t from = class_state(result.reconstructed_event_from_class[e]);
                const int64_t to = class_state(result.reconstructed_event_to_class[e]);
                out.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                out.write(static_cast<int64_t>(e));
                hgmarshal::write_event_record(sink,
                    hgmarshal::EventRecordIds{static_cast<int64_t>(e), recon_id(e),
                        static_cast<int64_t>(result.reconstructed_event_rule[e]), from, to, from, to},
                    job.include_events_minimal, consumed, produced);
                return;
            }
            i -= apps.size();
            if (i < raw_events) {
                const hg_gpu::Event& e = result.events[i];
                for (auto c : result.consumed_of(e))
                    if (c != hg_gpu::INVALID_ID) consumed.push_back(static_cast<int64_t>(c));
                for (auto pe : result.produced_of(e))
                    if (pe != hg_gpu::INVALID_ID) produced.push_back(static_cast<int64_t>(pe));
                out.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                out.write(static_cast<int64_t>(e.id));
                hgmarshal::write_event_record(sink,
                    hgmarshal::EventRecordIds{
                        static_cast<int64_t>(e.id),
                        e.canonical_id == hg_gpu::INVALID_ID ? static_cast<int64_t>(e.id)
                                                             : static_cast<int64_t>(e.canonical_id),
                        static_cast<int64_t>(e.rule),
                        static_cast<int64_t>(e.input_state), static_cast<int64_t>(e.output_state),
                        rep_of(e.input_state), rep_of(e.output_state)},
                    job.include_events_minimal, consumed, produced);
                return;
            }
            i -= raw_events;
            // One genesis event per initial state, in the same shape a real one has. Rule index
            // 65535 is the host's RuleIndex(-1), "no rule applied", which is what produced these
            // edges.
            const hg_gpu::StateId root = genesis_roots[i];
            const int64_t gid = static_cast<int64_t>(first_genesis_event + i);
            for (auto pe : result.edge_ids(*state_by_id.at(root)))
                produced.push_back(static_cast<int64_t>(pe));
            out.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            out.write(gid);
            hgmarshal::write_event_record(sink,
                hgmarshal::EventRecordIds{gid, gid, kGenesisRuleIndex,
                                          static_cast<int64_t>(genesis_state_id),
                                          static_cast<int64_t>(root),
                                          static_cast<int64_t>(genesis_state_id), rep_of(root)},
                job.include_events_minimal, consumed, produced);
        });
        full_result.push_back({wxf::WXFValue("Events"), wxf::WXFValue(wxf::Encoded{w.release_data()})});
    }

    // An event's id under the event identity in force: its canonical event, or itself. From and To
    // of the causal and branchial lists carry it and RawFrom/RawTo the raw ids, as on the host
    // (Hypergraph::get_canonical_event).
    std::unordered_map<uint32_t, uint32_t> canonical_event;
    for (const auto& e : result.events)
        if (e.id != hg_gpu::INVALID_ID && e.canonical_id != hg_gpu::INVALID_ID)
            canonical_event.emplace(e.id, e.canonical_id);
    auto event_id_of = [&](uint32_t raw) -> int64_t {
        auto it = canonical_event.find(raw);
        return static_cast<int64_t>(it == canonical_event.end() ? raw : it->second);
    };

    // CausalEdges: dedup by raw (from,to), matching the FFI. The deduped count feeds
    // NumCausalEdges and must be computed even when the edge list itself is not requested
    // (e.g. the counts-only "Debug" property), so the count matches the CPU in every case.
    int64_t num_causal = 0;
    std::vector<std::array<int64_t, 4>> causal;   // From, To, RawFrom, RawTo
    if (recon_ran) {
        const auto& raw = job.transitive_reduction ? result.reconstructed_causal_raw_reduced
                                                   : result.reconstructed_causal_raw;
        if (job.include_causal_edges) {
            for (const auto& [p, c] : raw)
                causal.push_back({recon_id(p), recon_id(c), static_cast<int64_t>(p),
                                  static_cast<int64_t>(c)});
            for (const auto& [root, ev] : recon_genesis) {
                const int64_t from = static_cast<int64_t>(first_genesis_event + root);
                causal.push_back({from, recon_id(ev), from, static_cast<int64_t>(ev)});
            }
        }
        num_causal = static_cast<int64_t>(raw.size() + recon_genesis.size());
    } else {
        std::unordered_set<uint64_t> seen;
        for (const auto& c : result.causal_edges) {
            uint64_t key = (static_cast<uint64_t>(c.from) << 32) | static_cast<uint32_t>(c.to);
            if (!seen.insert(key).second) continue;
            if (job.include_causal_edges)
                causal.push_back({event_id_of(c.from), event_id_of(c.to),
                                  static_cast<int64_t>(c.from), static_cast<int64_t>(c.to)});
        }
        for (const auto& [root, to] : genesis_causal) {
            const uint32_t from = static_cast<uint32_t>(first_genesis_event + root);
            const uint64_t key = (static_cast<uint64_t>(from) << 32) | to;
            if (!seen.insert(key).second) continue;
            if (job.include_causal_edges)
                causal.push_back({static_cast<int64_t>(from), static_cast<int64_t>(to),
                                  static_cast<int64_t>(from), static_cast<int64_t>(to)});
        }
        num_causal = static_cast<int64_t>(seen.size());
    }
    if (job.include_causal_edges)
        full_result.push_back({wxf::WXFValue("CausalEdges"),
                               hgmarshal::record_list(causal, {"From", "To", "RawFrom", "RawTo"},
                                                      reply_jobs().parallel_for())});

    // BranchialEdges: no dedup (multiplicity matters).
    if (job.include_branchial_edges) {
        std::vector<std::array<int64_t, 2>> branchial;
        if (recon_ran) {
            for (const auto& [a, b] : result.reconstructed_branchial_raw)
                branchial.push_back({recon_id(a), recon_id(b)});
        } else {
            for (const auto& b : result.branchial_edges)
                branchial.push_back({event_id_of(b.a), event_id_of(b.b)});
        }
        full_result.push_back({wxf::WXFValue("BranchialEdges"),
                               hgmarshal::record_list(branchial, {"From", "To"},
                                                      reply_jobs().parallel_for())});
    }

    // BranchialStateEdges / BranchialStateEdgesAllSiblings: the state-endpoint projection of the
    // branchial relation, through the same hgmarshal rules the host calls. Everything the two
    // rules need is already in EvolveResult -- the branchial event pairs, each event's input and
    // output state, the class representative and the per-state step -- so nothing is read back
    // from the device for them.
    //
    // No genesis filter: the device mints no genesis event, which is why ShowGenesisEvents
    // carries its own advisory rather than being applied here.
    if (job.include_branchial_state_edges || job.include_branchial_state_edges_all_siblings) {
        std::unordered_map<hg_gpu::EventId, hg_gpu::StateId> ev_out;
        ev_out.reserve(result.events.size());
        for (const auto& e : result.events) ev_out[e.id] = e.output_state;

        const auto out_state_of = [&](uint32_t eid) -> uint32_t {
            auto it = ev_out.find(eid);
            return it == ev_out.end() ? 0u : it->second;
        };
        const auto canonical_of = [&](uint32_t sid) { return rep_of(sid); };
        const auto step_of = [&](uint32_t sid) -> uint32_t {
            auto it = state_step.find(sid);
            return it == state_step.end() ? 0u : it->second;
        };
        hgmarshal::GraphOptions bse_opts;
        bse_opts.branchial_step = job.branchial_step;
        bse_opts.steps = run_steps;

        if (job.include_branchial_state_edges) {
            std::vector<std::pair<uint32_t, uint32_t>> pairs;
            pairs.reserve(result.branchial_edges.size());
            for (const auto& b : result.branchial_edges) pairs.emplace_back(b.a, b.b);
            hgmarshal::push_branchial_state_edges(
                full_result,
                hgmarshal::branchial_state_edges_from_pairs(
                    pairs, out_state_of, canonical_of, step_of, bse_opts));
        }

        if (job.include_branchial_state_edges_all_siblings) {
            // Grouped by input state and ordered by it, which is this engine's traversal of its
            // own storage; the host walks a per-state list instead. The rule the groups are fed
            // to is the same one.
            std::map<hg_gpu::StateId, std::vector<uint32_t>> by_input;
            for (const auto& e : result.events) by_input[e.input_state].push_back(e.id);
            const auto for_each_group = [&](auto&& emit) {
                for (const auto& kv : by_input) emit(kv.second);
            };
            hgmarshal::push_branchial_state_edges(
                full_result,
                hgmarshal::branchial_state_edges_all_siblings(
                    for_each_group, out_state_of, canonical_of, step_of, bse_opts));
        }
    }

    // NumStates mirrors the CPU's num_canonical_states() = canonical_state_map_.count_unique(),
    // verified against the CPU engine in gpu/tests/test_gpu_vs_cpu_differential.cpp
    // (CanonicalStateCount.ModesVsCpu). class_reps is grouped by the mode's dedup key, so its size
    // is exactly the CPU count in every mode: None -> raw state count, Automatic -> distinct
    // content, Full -> distinct IR class. (The CPU's None-mode sentinel undercount is fixed in
    // create_or_get_canonical_state, so no adjustment is needed here.)
    // GlobalEdges -> {edge_id, v1, v2, ...} for every edge the evolution created, and
    // StateBitvectors -> state id -> the edge ids that state holds, from the state-edge arrays
    // in EvolveResult. Same shape as the host serialises.
    if (job.include_global_edges) {
        std::vector<hg_gpu::EdgeId> ids;
        for (size_t eid = 0; eid < result.edge_records.size(); ++eid)
            if (!result.edge_vertices(static_cast<hg_gpu::EdgeId>(eid)).empty())
                ids.push_back(static_cast<hg_gpu::EdgeId>(eid));
        wxf::Writer w;
        w.write_function("List", ids.size());
        hgmarshal::write_items(w, ids.size(), reply_jobs().parallel_for(),
                               [&](wxf::Writer& out, size_t i) {
            const hg_gpu::VertexSpan vs = result.edge_vertices(ids[i]);
            out.write_function("List", vs.size() + 1);
            out.write(static_cast<int64_t>(ids[i]));
            for (auto v : vs) out.write(static_cast<int64_t>(v));
        });
        full_result.push_back({wxf::WXFValue("GlobalEdges"), wxf::WXFValue(wxf::Encoded{w.release_data()})});
    }
    if (job.include_state_bitvectors) {
        wxf::Writer w;
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(result.states.size());
        hgmarshal::write_items(w, result.states.size(), reply_jobs().parallel_for(),
                               [&](wxf::Writer& out, size_t i) {
            const hg_gpu::CanonicalState& st = result.states[i];
            out.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            out.write(static_cast<int64_t>(st.id));
            const hg_gpu::EdgeSpan ids = result.edge_ids(st);
            out.write_function("List", ids.size());
            for (auto eid : ids) out.write(static_cast<int64_t>(eid));
        });
        full_result.push_back({wxf::WXFValue("StateBitvectors"),
                               wxf::WXFValue(wxf::Encoded{w.release_data()})});
    }

    if (job.include_device) {
        const hg_gpu::DeviceProbe& probe = hg_gpu::probe_device();
        wxf::WXFValueAssociation device;
        device.push_back({wxf::WXFValue("Type"), wxf::WXFValue("GPU")});
        device.push_back({wxf::WXFValue("Name"), wxf::WXFValue(hgmarshal::valid_utf8(probe.name))});
        device.push_back({wxf::WXFValue("ComputeCapability"),
                          wxf::WXFValue(std::to_string(probe.major) + "." +
                                        std::to_string(probe.minor))});
        device.push_back({wxf::WXFValue("MemoryCap"),
                          wxf::WXFValue(static_cast<int64_t>(
                              hg_gpu::resolved_memory_cap(in.max_device_memory_bytes)))});
        {
            const hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
            device.push_back({wxf::WXFValue("Blocks"),
                              wxf::WXFValue(static_cast<int64_t>(hg_gpu::persistent_grid_of(cfg)))});
            device.push_back({wxf::WXFValue("Backlog"),
                              wxf::WXFValue(static_cast<int64_t>(hg_gpu::lane_backlog_of(cfg)))});
            device.push_back({wxf::WXFValue("L2Persist"),
                              wxf::WXFValue(static_cast<int64_t>(
                                  hg_gpu::l2_persist_limit_of(cfg.l2_persist_limit)))});
        }
        device.push_back({wxf::WXFValue("Threads"),
                          wxf::WXFValue(static_cast<int64_t>(
                              job.threads ? job.threads : hgcommon::default_worker_count()))});
        full_result.push_back({wxf::WXFValue("Device"), wxf::WXFValue(device)});
    }

    if (job.include_num_states) {
        full_result.push_back({wxf::WXFValue("NumStates"),
                               wxf::WXFValue(static_cast<int64_t>(class_reps.size()))});
    }
    // NumEvents mirrors the CPU's hg.observable_num_events(): the RECONSTRUCTION's count
    // wherever the reconstruction ran, and the identity count from the events themselves
    // otherwise. The rule lives in EvolveResult so this and the differential harness cannot
    // disagree about which number the device serves.
    if (job.include_num_events) {
        // Under ShowGenesisEvents the genesis events are counted, one per initial state, on both
        // routes (docs/SPEC.md §5.2).
        full_result.push_back({wxf::WXFValue("NumEvents"),
                               wxf::WXFValue(static_cast<int64_t>(result.observable_num_events() +
                                                                  genesis_roots.size()))});
    }
    // The RECONSTRUCTION's relations wherever it ran, as hypergraph_ffi.cpp:1319 does -- on that
    // route the materialised edges counted above belong to the explored representatives alone.
    // The rule lives on EvolveResult beside observable_num_events, so the shipped number and the
    // number the differential gates cannot diverge.
    if (job.include_num_causal_edges) {
        full_result.push_back({wxf::WXFValue("NumCausalEdges"),
                               wxf::WXFValue(result.reconstruction_ran
                                   ? static_cast<int64_t>(result.observable_num_causal_pairs(
                                         job.transitive_reduction) + recon_genesis.size())
                                   : num_causal)});
    }
    if (job.include_num_branchial_edges) {
        full_result.push_back({wxf::WXFValue("NumBranchialEdges"),
                               wxf::WXFValue(static_cast<int64_t>(
                                   result.observable_num_branchial()))});
    }

    // StepStatistics: as on the host (hypergraph_ffi.cpp), from the multiplicities under
    // quotient exploration and from every raw state under full capture.
    if (job.include_step_statistics) {
        std::map<uint32_t, hg::stats::StepInput> inputs;
        // Each class's record, written on the device by the warp that created the class.
        std::unordered_map<uint64_t, const hgcommon::StateInvariantRecord*> class_invariants;
        auto take_record = [&](const hg_gpu::CanonicalState& st) {
            auto& r = class_invariants[state_hash[st.id]];
            if (!r) r = result.invariants(st);
            return r;
        };
        if (!result.class_multiplicities.empty()) {
            for (const auto& st : result.states)
                if (st.id != hg_gpu::INVALID_ID) take_record(st);
            std::vector<hg::stats::StepPoint> points;
            for (const auto& p : result.class_multiplicities)
                points.push_back({p.depth, p.class_hash, p.multiplicity});
            for (const auto& p : points) {
                const auto r = class_invariants.find(p.class_hash);
                hg::stats::add_class(inputs[p.step], p.class_hash, p.weight,
                                     r == class_invariants.end() ? nullptr : r->second);
            }
            std::unordered_map<uint64_t, std::map<int64_t, uint64_t>> matches_by_rule;
            for (const auto& c : result.class_rule_matches)
                matches_by_rule[c.expansion_key][static_cast<int64_t>(c.rule)] += c.count;
            std::map<uint32_t, uint64_t> events;
            std::map<uint32_t, std::map<int64_t, uint64_t>> rule_counts;
            hg::stats::events_from_multiplicities(
                points, matches_by_rule, hgcommon::qc_expansions_by_depth(job.max_states_per_step),
                static_cast<uint32_t>(run_steps), events, rule_counts);
            for (const auto& [step, n] : events) inputs[step].events = n;
            for (auto& [step, by_rule] : rule_counts) inputs[step].rule_counts = std::move(by_rule);
        } else {
            // Every class's record first: only the state that created the class wrote one, and
            // its id need not be the lowest of the class.
            for (const auto& st : result.states)
                if (st.id != hg_gpu::INVALID_ID) take_record(st);
            for (const auto& st : result.states) {
                const auto it = state_step.find(st.id);
                const uint32_t step = it == state_step.end() ? 0u : it->second;
                hg::stats::add_class(inputs[step], state_hash[st.id], 1, take_record(st));
            }
            for (const auto& e : result.events) {
                hg::stats::StepInput& at = inputs[e.step];
                ++at.events;
                ++at.rule_counts[static_cast<int64_t>(e.rule)];
            }
        }
        // "StepStatisticsBranchial": from the device's branchial pairs and state contents.
        const uint32_t which = job.step_statistics_branchial;
        if (which) {
            std::unordered_map<hg_gpu::EventId, hg_gpu::StateId> output_of;
            for (const auto& e : result.events) output_of[e.id] = e.output_state;
            std::vector<std::pair<uint32_t, uint32_t>> pairs;
            if (which & hg::stats::kBranchialGraph)
                for (const auto& b : result.branchial_edges)
                    pairs.emplace_back(static_cast<uint32_t>(b.a), static_cast<uint32_t>(b.b));
            auto bsteps = hgmarshal::branchial_steps(
                which, static_cast<uint32_t>(result.states.size()),
                [&](uint32_t sid) { return result.states[sid].id != hg_gpu::INVALID_ID; },
                [&](uint32_t sid) { return rep_of(sid); },
                [&](uint32_t sid) { return own_step(sid); },
                [&](uint32_t sid, std::vector<uint32_t>& vs) {
                    const hg_gpu::CanonicalState& st = result.states[sid];
                    for (uint32_t k = 0; k < st.num_edges; ++k) {
                        const hg_gpu::VertexSpan e = result.edge(st, k);
                        vs.push_back(e.size());
                        vs.insert(vs.end(), e.begin(), e.end());
                    }
                },
                pairs,
                [&](uint32_t eid) {
                    const auto it = output_of.find(eid);
                    return it == output_of.end() ? 0u : static_cast<uint32_t>(it->second);
                },
                static_cast<int>(run_steps));
            for (auto& [step, b] : bsteps) inputs[step].branchial = std::move(b);
        }
        // The pool of a run of kDevicePoolMinClasses class points or more is made on the device
        // (hg_gpu::step_pool_device); the host then reads the classes for the signature
        // histograms only.
        size_t class_points = 0;
        for (const auto& [step, step_in] : inputs) class_points += step_in.classes.size();
        if (class_points >= kDevicePoolMinClasses && !result.invariant_pool.empty())
            device_step_pool(result.invariant_pool, job.step_statistics_by_class, inputs);
        // The overlap rows of a step of 2048 states or more run on the device
        // (hg_gpu::branchial_rows_device), one step at a time; a step that finds the device in
        // use runs its rows on the host. The edge rows of the state matrices run first: they
        // only fill the edge matrices, which the host's rows fill again with the same values
        // when the vertex rows then fall back to the host.
        std::atomic<bool> device_busy{false};
        auto device_rows = [&device_busy](hg::stats::BranchialStepWork& work) {
            if (device_busy.exchange(true, std::memory_order_acquire)) return false;
            bool done = true;
            try {
                if (hgcommon::BoMatrices* m = work.edge_matrices())
                    done = hg_gpu::branchial_rows_device(work.edge_rows_view(), nullptr, m);
                if (done && work.wants_vertex_rows())
                    done = hg_gpu::branchial_rows_device(
                        work.rows_view(),
                        [&work](uint32_t distance, uint32_t size_a, uint32_t size_b,
                                uint32_t shared, uint64_t n) {
                            work.add_counts(distance, size_a, size_b, shared, n);
                        },
                        work.vertex_matrices());
            } catch (...) {
                device_busy.store(false, std::memory_order_release);
                throw;
            }
            device_busy.store(false, std::memory_order_release);
            return done;
        };
        std::vector<std::pair<uint32_t, uint32_t>> over_limit;
        full_result.push_back({wxf::WXFValue("StepStatistics"),
                               hg::stats::step_statistics_on_host(
                                   reply_jobs(), std::move(inputs), job.step_statistics_by_class,
                                   which, device_rows, 2048, job.step_statistics_matrix_limit,
                                   &over_limit)});
        for (const auto& [step, states] : over_limit)
            job.job_warnings.push_back(
                {"StepStatistics", 1,
                 hg::stats::matrix_limit_warning(step, states, job.step_statistics_matrix_limit)});
    }

    // GraphData for the requested *Graph properties, built through the SAME shared
    // marshaller (graph_marshal.hpp) as the CPU FFI so the two devices emit identical
    // graph structure. The adapter exposes the device result as effective (class-rep)
    // ids plus CPU-matching vertex tooltips.
    if (!job.graph_properties.empty()) {
        std::unordered_map<hg_gpu::EventId, const hg_gpu::Event*> event_by_id;
        uint32_t max_state = 0, max_event = 0;
        for (const auto& s : result.states) max_state = std::max<uint32_t>(max_state, s.id);
        for (const auto& e : result.events) {
            event_by_id[e.id] = &e;
            max_event = std::max<uint32_t>(max_event, e.id);
        }
        const bool full = (canon_mode == hg_gpu::CanonicalizationMode::Full);

        // A state's edges numbered from 0 in stored order; under Full its IR canonical form
        // (hgmarshal::state_record_edges).
        auto state_edges_of = [&](hg_gpu::StateId sid,
                                  std::vector<std::pair<int64_t, std::vector<uint32_t>>>& out) {
            auto it = state_by_id.find(sid);
            if (it == state_by_id.end()) return;
            int64_t idx = 0;
            for (const auto& ce : result.edges_of(*it->second))
                out.emplace_back(idx++, std::vector<uint32_t>(ce.begin(), ce.end()));
            hgmarshal::state_record_edges(full, out);
        };
        auto step_of = [&](hg_gpu::StateId sid) -> uint32_t {
            auto it = state_step.find(sid);
            return it == state_step.end() ? 0u : it->second;
        };
        // THE EVENT IDENTITY THE COUNT USES, when the reconstruction ran.
        //
        // observable_num_events reports the reconstruction's distinct identities. Mapping a
        // materialised event through its own canonical_id answers a different question -- that
        // id is computed from each raw state's own labelling, which is the per-state convention
        // the reconstruction exists to replace -- and the two sets differ: 25 graph vertices
        // against a count of 24. So when the replay ran, identity comes from its signature, and
        // an event it never minted stands for no vertex at all. This is the host's rule
        // (hypergraph_ffi.cpp's recon.dense_of_sig), reached through the signature array the
        // device now hands back.
        std::unordered_map<uint64_t, int64_t> dense_of_sig;
        const bool recon_identity = result.reconstruction_ran &&
                                    !result.reconstructed_event_signature.empty();
        if (recon_identity) {
            for (uint64_t sig : result.reconstructed_event_signature) {
                if (sig == 0ull) continue;
                dense_of_sig.emplace(sig, static_cast<int64_t>(dense_of_sig.size()));
            }
        }
        auto sig_of_event = [&](hg_gpu::EventId eid) -> uint64_t {
            return eid < result.reconstructed_event_signature.size()
                       ? result.reconstructed_event_signature[eid] : 0ull;
        };
        auto eff_event = [&](hg_gpu::EventId eid) -> int64_t {
            if (recon_identity) {
                // Under EVENT_SIG_NONE every application is its own event; the signature array
                // would collapse distinct applications into one vertex.
                if (job.event_canon_mode == 0) return static_cast<int64_t>(eid);
                auto it = dense_of_sig.find(sig_of_event(eid));
                return it == dense_of_sig.end() ? -1 : it->second;
            }
            if (job.state_canon_mode == 0 || job.event_canon_mode == 0) return static_cast<int64_t>(eid);
            auto it = event_by_id.find(eid);
            if (it == event_by_id.end() || it->second->canonical_id == hg_gpu::INVALID_ID)
                return static_cast<int64_t>(eid);
            return static_cast<int64_t>(it->second->canonical_id);
        };

        struct GpuGraphSource {
            uint32_t n_states, n_events;
            std::function<bool(uint32_t)> state_valid_;
            std::function<int64_t(uint32_t)> eff_state_;
            std::function<uint32_t(uint32_t)> step_;
            std::function<void(uint32_t, hgmarshal::GraphStateFields&)> state_data_;
            std::function<void(uint32_t, std::vector<std::pair<int64_t, std::vector<uint32_t>>>&)>
                state_edges_;
            std::function<bool(uint32_t)> event_valid_;
            std::function<int64_t(uint32_t)> eff_event_;
            std::function<uint32_t(uint32_t)> in_state_;
            std::function<uint32_t(uint32_t)> out_state_;
            std::function<bool(uint32_t, hgmarshal::GraphEventFields&)> event_data_;
            std::vector<std::pair<uint32_t, uint32_t>> causal_pairs_;
            std::vector<std::pair<uint32_t, uint32_t>> branchial_pairs_;

            uint32_t num_states() const { return n_states; }
            bool state_valid(uint32_t sid) const { return state_valid_(sid); }
            int64_t effective_state_id(uint32_t sid) const { return eff_state_(sid); }
            uint32_t state_step(uint32_t sid) const { return step_(sid); }
            void state_fields(uint32_t sid, hgmarshal::GraphStateFields& f) const { state_data_(sid, f); }
            void state_edges(uint32_t sid,
                             std::vector<std::pair<int64_t, std::vector<uint32_t>>>& out) const {
                state_edges_(sid, out);
            }
            uint32_t num_raw_events() const { return n_events; }
            bool is_valid_event(uint32_t eid) const { return event_valid_(eid); }
            int64_t effective_event_id(uint32_t eid) const { return eff_event_(eid); }
            uint32_t event_input_state(uint32_t eid) const { return in_state_(eid); }
            uint32_t event_output_state(uint32_t eid) const { return out_state_(eid); }
            bool event_fields(uint32_t eid, hgmarshal::GraphEventFields& f) const { return event_data_(eid, f); }
            std::vector<std::pair<uint32_t, uint32_t>> causal_event_pairs() const { return causal_pairs_; }
            std::vector<std::pair<uint32_t, uint32_t>> branchial_event_pairs() const { return branchial_pairs_; }
        };

        // The genesis state and events are drawn when shown, over the device's events or the
        // reconstruction's applications, as the host draws them. The genesis state is the host's:
        // no edges, step 0.
        const bool graph_genesis = !genesis_roots.empty();
        const uint32_t genesis_end =
            graph_genesis ? static_cast<uint32_t>(first_genesis_event + genesis_roots.size()) : 0u;
        auto is_genesis_event = [&](uint32_t eid) {
            return graph_genesis && eid >= first_genesis_event && eid < genesis_end;
        };
        auto is_genesis_state = [&](uint32_t sid) { return graph_genesis && sid == genesis_state_id; };

        GpuGraphSource gsrc;
        gsrc.n_states = std::max<uint32_t>(max_state + 1, graph_genesis ? genesis_state_id + 1 : 0u);
        gsrc.n_events = recon_content
                            ? std::max<uint32_t>(
                                  static_cast<uint32_t>(result.reconstructed_event_from_class.size()),
                                  genesis_end)
                            : std::max<uint32_t>(max_event + 1, genesis_end);
        gsrc.state_valid_ = [&](uint32_t sid) {
            return is_genesis_state(sid) || state_by_id.find(sid) != state_by_id.end();
        };
        gsrc.eff_state_ = [&](uint32_t sid) {
            return is_genesis_state(sid) ? static_cast<int64_t>(sid) : rep_of(sid);
        };
        gsrc.step_ = [&](uint32_t sid) { return is_genesis_state(sid) ? 0u : step_of(sid); };
        gsrc.state_data_ = [&](uint32_t sid, hgmarshal::GraphStateFields& f) {
            f.id = static_cast<int64_t>(sid);
            if (is_genesis_state(sid)) {
                f.canonical_id = static_cast<int64_t>(sid);
                f.step = 0;
                return;
            }
            f.canonical_id = rep_of(sid);
            f.step = static_cast<int64_t>(reported_step(sid));
        };
        // The genesis state has no edges.
        gsrc.state_edges_ = [&](uint32_t sid,
                                std::vector<std::pair<int64_t, std::vector<uint32_t>>>& out) {
            if (!is_genesis_state(sid)) state_edges_of(sid, out);
        };
        gsrc.event_valid_ = [&](uint32_t eid) {
            // An application whose identity the replay never registered stands for no vertex,
            // which is what keeps the vertex set equal to the set the count describes.
            if (is_genesis_event(eid)) return true;
            if (recon_content) return recon_app_valid(eid);
            if (recon_identity) {
                if (job.event_canon_mode == 0)
                    return eid < result.reconstructed_event_signature.size() &&
                           result.reconstructed_event_signature[eid] != 0ull;
                return dense_of_sig.count(sig_of_event(eid)) != 0;
            }
            return event_by_id.find(eid) != event_by_id.end();
        };
        gsrc.eff_event_ = [&](uint32_t eid) -> int64_t {
            return is_genesis_event(eid) ? static_cast<int64_t>(eid) : eff_event(eid);
        };
        // A reconstructed application's endpoints are the states standing for its classes.
        gsrc.in_state_ = [&](uint32_t eid) -> uint32_t {
            if (is_genesis_event(eid)) return genesis_state_id;
            if (recon_content) return static_cast<uint32_t>(class_state(result.reconstructed_event_from_class[eid]));
            auto it = event_by_id.find(eid); return it == event_by_id.end() ? 0u : it->second->input_state; };
        gsrc.out_state_ = [&](uint32_t eid) -> uint32_t {
            if (is_genesis_event(eid)) return genesis_roots[eid - first_genesis_event];
            if (recon_content) return static_cast<uint32_t>(class_state(result.reconstructed_event_to_class[eid]));
            auto it = event_by_id.find(eid); return it == event_by_id.end() ? 0u : it->second->output_state; };
        gsrc.event_data_ = [&](uint32_t eid, hgmarshal::GraphEventFields& f) -> bool {
            if (is_genesis_event(eid)) {
                const hg_gpu::StateId root = genesis_roots[eid - first_genesis_event];
                for (auto pe : result.edge_ids(*state_by_id.at(root)))
                    f.produced.push_back(static_cast<int64_t>(pe));
                f.id = static_cast<int64_t>(eid);
                f.canonical_id = static_cast<int64_t>(eid);
                f.rule = kGenesisRuleIndex;
                f.input = static_cast<int64_t>(genesis_state_id);
                f.output = static_cast<int64_t>(root);
                f.input_raw = genesis_state_id;
                f.output_raw = root;
                return true;
            }
            if (recon_content) {
                f.reconstructed = true;
                f.id = eff_event(eid);
                f.rule = static_cast<int64_t>(result.reconstructed_event_rule[eid]);
                f.input = class_state(result.reconstructed_event_from_class[eid]);
                f.output = class_state(result.reconstructed_event_to_class[eid]);
                return true;
            }
            auto eit = event_by_id.find(eid);
            if (eit == event_by_id.end()) return false;
            const hg_gpu::Event& e = *eit->second;
            for (auto c : result.consumed_of(e)) f.consumed.push_back(static_cast<int64_t>(c));
            for (auto p : result.produced_of(e)) f.produced.push_back(static_cast<int64_t>(p));
            f.id = static_cast<int64_t>(eid);
            f.canonical_id = eff_event(eid);
            f.rule = static_cast<int64_t>(e.rule);
            f.input = static_cast<int64_t>(e.input_state);
            f.output = static_cast<int64_t>(e.output_state);
            f.input_raw = e.input_state;
            f.output_raw = e.output_state;
            return true;
        };
        if (result.reconstruction_ran) {
            // The reconstruction's raw relation, reduced per the option -- the same source the
            // edge lists serve, so the graph is their rendered form and not a second relation.
            const auto& raw = job.transitive_reduction ? result.reconstructed_causal_raw_reduced
                                                       : result.reconstructed_causal_raw;
            gsrc.causal_pairs_.assign(raw.begin(), raw.end());
            for (const auto& [root, ev] : recon_genesis)
                gsrc.causal_pairs_.emplace_back(static_cast<uint32_t>(first_genesis_event + root), ev);
            gsrc.branchial_pairs_.assign(result.reconstructed_branchial_raw.begin(),
                                         result.reconstructed_branchial_raw.end());
        } else {
        for (const auto& c : result.causal_edges)
            gsrc.causal_pairs_.emplace_back(static_cast<uint32_t>(c.from), static_cast<uint32_t>(c.to));
        if (graph_genesis)
            for (const auto& [root, to] : genesis_causal)
                gsrc.causal_pairs_.emplace_back(static_cast<uint32_t>(first_genesis_event + root), to);
        for (const auto& b : result.branchial_edges)
            gsrc.branchial_pairs_.emplace_back(static_cast<uint32_t>(b.a), static_cast<uint32_t>(b.b));
        }

        hgmarshal::GraphOptions gopts;
        gopts.edge_deduplication = job.edge_deduplication;
        gopts.branchial_step = job.branchial_step;
        gopts.steps = run_steps;
        full_result.push_back({wxf::WXFValue("GraphData"),
                               hgmarshal::build_graph_data(gsrc, job.graph_properties, gopts, nullptr,
                                                           reply_jobs().parallel_for())});
    }

    // The warning trail: capacity overflows mark a partial result, the other kinds do not.
    std::vector<HG_NAMESPACE::ffi::FfiWarning> count_warnings;
    if (result.reconstruction_ran)
        HG_NAMESPACE::ffi::append_saturation_warnings(
            job.include_num_events, result.observable_num_events(),
            job.include_num_branchial_edges, result.observable_num_branchial(), count_warnings);
    if (!job.job_warnings.empty() || !count_warnings.empty() || !result.warnings.empty()) {
        wxf::WXFValueList warn;
        bool partial = false;
        for (const auto& w : job.job_warnings)
            warn.push_back(hgmarshal::warning_record(w.kind, w.count, w.context, w.partial));
        for (const auto& w : count_warnings)
            warn.push_back(hgmarshal::warning_record(w.kind, w.count, w.context, w.partial));
        for (const auto& w : result.warnings) {
            const bool p = hg_gpu::error_kind_is_partial(w.kind);
            partial = partial || p;
            warn.push_back(hgmarshal::warning_record(hg_gpu::error_kind_name(w.kind),
                                                     static_cast<int64_t>(w.count), w.context, p));
        }
        if (partial && host.progress)
            host.progress("HGEvolve (GPU): capacity overflow -- returning partial result");
        full_result.push_back({wxf::WXFValue("Warnings"), wxf::WXFValue(warn)});
    }

    wxf::Writer writer;
    writer.write_header();
    // A session's result names the session, so the caller can Step it. Same key the CPU emits,
    // because a caller cannot tell which device answered and must not have to.
    if (held.handle != 0 && (is_open || is_step || is_query)) {
        // THE FRONTIER, in every reply a session gives -- the same contract the CPU serves: a
        // caller cannot steer a continuation toward states it cannot name. Effective ids, the
        // identity every other field uses, deduplicated because several device states of one
        // class can sit on the frontier together while the caller sees one id. The resolution
        // map is rebuilt HERE, where the ids are minted, so a later steered Step reads exactly
        // the identity this reply reported.
        held.frontier_eff.clear();
        std::set<int64_t> seen;
        wxf::WXFValueList frontier;
        for (size_t i = 0; i < held.frontier_ids.size(); ++i) {
            const int64_t eff = rep_of(held.frontier_ids[i]);
            held.frontier_eff.push_back(eff);
            if (seen.insert(eff).second) frontier.push_back(wxf::WXFValue(eff));
        }
        full_result.push_back({wxf::WXFValue("Frontier"), wxf::WXFValue(std::move(frontier))});
        full_result.push_back({wxf::WXFValue("Session"),
                               wxf::WXFValue(static_cast<int64_t>(held.handle))});
    }
    writer.write(wxf::WXFValue(std::move(full_result)));
    undelivered_open.armed = false;
    // An engine that could not be built at the grown size ends the run partial; the worker then
    // goes back to a fresh worker's footprint, as after a throw (run_gpu_evolution).
    const bool grow_failed =
        std::any_of(result.warnings.begin(), result.warnings.end(),
                    [](const hg_gpu::OverflowWarning& w) {
                        return w.kind == hg_gpu::ErrorKind::kDeviceOutOfMemory;
                    });
    if (grow_failed && held.handle == 0) {
        std::vector<uint8_t> out = writer.release_data();
        result = hg_gpu::EvolveResult{};
        evolver.release_after_failure();
        return out;
    }
    evolver.recycle(std::move(result));
    return writer.release_data();
}

}  // namespace

#endif  // HG_GPU_BACKEND
