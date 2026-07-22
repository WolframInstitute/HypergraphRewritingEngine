#ifdef HG_GPU_BACKEND

#include "hg_gpu_backend.hpp"

#include "hg_gpu/evolve.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "wxf.hpp"
#include "graph_marshal.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Build a hg_gpu::EvolveInput from the parsed job. Rule vertices double as
// pattern-variable indices; each initial state's vertices are remapped to
// 0..n-1 (matching the FFI, so isomorphic roots share a representation).
hg_gpu::EvolveInput build_input(const GpuJob& job) {
    hg_gpu::EvolveInput in;

    for (const auto& [name, parts] : job.rules) {
        (void)name;
        if (parts.size() != 2) continue;
        hg_gpu::RewriteRule r;
        uint8_t lhs_max = 0, rhs_max = 0;
        for (const auto& edge : parts[0]) {
            std::vector<uint8_t> e;
            for (int64_t v : edge) {
                if (v < 0) continue;
                e.push_back(static_cast<uint8_t>(v));
                lhs_max = std::max<uint8_t>(lhs_max, static_cast<uint8_t>(v));
            }
            if (!e.empty()) r.lhs.push_back(std::move(e));
        }
        for (const auto& edge : parts[1]) {
            std::vector<uint8_t> e;
            for (int64_t v : edge) {
                if (v < 0) continue;
                e.push_back(static_cast<uint8_t>(v));
                rhs_max = std::max<uint8_t>(rhs_max, static_cast<uint8_t>(v));
            }
            if (!e.empty()) r.rhs.push_back(std::move(e));
        }
        if (r.lhs.empty() || r.rhs.empty()) continue;
        r.num_lhs_vars = static_cast<uint8_t>(lhs_max + 1);
        r.num_rhs_vars = static_cast<uint8_t>(rhs_max + 1);
        in.rules.push_back(std::move(r));
    }

    for (const auto& state : job.initial_states) {
        std::unordered_map<int64_t, hg_gpu::VertexId> vmap;
        hg_gpu::VertexId next = 0;
        std::vector<std::vector<hg_gpu::VertexId>> edges;
        for (const auto& edge : state) {
            std::vector<hg_gpu::VertexId> e;
            for (int64_t v : edge) {
                if (v < 0) continue;
                auto it = vmap.find(v);
                if (it == vmap.end()) { vmap[v] = next; e.push_back(next); ++next; }
                else e.push_back(it->second);
            }
            if (!e.empty()) edges.push_back(std::move(e));
        }
        if (!edges.empty()) in.initial_states.push_back(std::move(edges));
    }

    in.num_steps = static_cast<uint32_t>(std::max(0, job.steps));
    // State dedup / class collapse is done host-side in the marshaller, keyed by the
    // requested mode (None per-provenance, Automatic edge-content, Full IR class).
    in.canonicalization =
        job.state_canon_mode == 0 ? hg_gpu::CanonicalizationMode::None :
        job.state_canon_mode == 1 ? hg_gpu::CanonicalizationMode::Automatic :
                                    hg_gpu::CanonicalizationMode::Full;
    in.event_canonicalization =
        job.event_canon_mode == 1 ? hg_gpu::EventCanonicalizationMode::Full :
        job.event_canon_mode == 2 ? hg_gpu::EventCanonicalizationMode::Automatic :
                                    hg_gpu::EventCanonicalizationMode::None;
    in.transitive_reduction = job.transitive_reduction;
    in.explore_from_canonical_states_only = job.explore_from_canonical_states_only;
    in.quotient_initial_states = job.quotient_initial_states;
    in.exploration_probability = static_cast<float>(job.exploration_probability);
    in.max_device_memory_bytes = job.max_device_memory_bytes;
    return in;
}

}  // namespace

std::vector<uint8_t> run_gpu_evolution(const GpuJob& job, const HostBridge& host) {
    hg_gpu::EvolveInput in = build_input(job);

    // Reuse one device Engine across every job this process handles. The
    // persistent worker processes many HGEvolve calls in one process, and the
    // per-call Engine allocation dominates small/medium runs, so amortizing it
    // is 6-12x on interactive workloads. Jobs run serially through the worker, so
    // a process-lifetime evolver is safe; the one-shot binary just uses it once.
    // The evolver grows on overflow and never shrinks (high-water-mark).
    static hg_gpu::PersistentEvolver evolver;
    hg_gpu::EvolveResult result = evolver.run(in);

    hypergraph::IRCanonicalizer ir;

    // Group the GPU states by the requested canonicalization mode, mirroring the CPU:
    //   None      -> every state is distinct (per-provenance / tree mode, no grouping)
    //   Automatic -> exact edge-content (non-isomorphic) equality
    //   Full      -> IR isomorphism class
    // The class representative (first-seen id) is the stable handle emitted. state_hash always
    // holds the IR canonical hash so the optional CanonicalHash output is available in any mode.
    const hg_gpu::CanonicalizationMode canon_mode = in.canonicalization;
    auto content_key = [](const std::vector<std::vector<hg_gpu::VertexId>>& edges) -> uint64_t {
        std::vector<std::vector<hg_gpu::VertexId>> e = edges;
        std::sort(e.begin(), e.end());
        uint64_t h = 1469598103934665603ull;               // FNV-1a over the sorted edge multiset
        for (const auto& ed : e) {
            for (auto v : ed) h = (h ^ static_cast<uint64_t>(v)) * 1099511628211ull;
            h = (h ^ 0x2Cull) * 1099511628211ull;          // edge separator
        }
        return h;
    };
    std::unordered_map<uint64_t, hg_gpu::StateId> hash_to_rep;
    std::unordered_map<hg_gpu::StateId, hg_gpu::StateId> state_to_rep;
    std::unordered_map<hg_gpu::StateId, uint64_t> state_hash;
    std::unordered_map<hg_gpu::StateId, const std::vector<std::vector<hg_gpu::VertexId>>*> state_edges;
    std::vector<hg_gpu::StateId> class_reps;
    for (const auto& s : result.states) {
        state_hash[s.id] = ir.compute_canonical_hash(s.edges);
        state_edges[s.id] = &s.edges;
        uint64_t key =
            canon_mode == hg_gpu::CanonicalizationMode::None      ? static_cast<uint64_t>(s.id) :
            canon_mode == hg_gpu::CanonicalizationMode::Automatic ? content_key(s.edges)
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

    wxf::WXFValueAssociation full_result;

    if (job.include_states) {
        wxf::WXFValueAssociation states_assoc;
        for (hg_gpu::StateId rep : class_reps) {
            wxf::WXFValueList edge_list;
            int64_t idx = 0;
            // Full relabels to the IR canonical form; None/Automatic keep the state's own labels
            // (there is no canonical relabelling when states are not identified by isomorphism).
            if (canon_mode == hg_gpu::CanonicalizationMode::Full) {
                auto canon = ir.canonicalize_edges(*state_edges[rep]);
                for (const auto& ce : canon.canonical_form.edges) {
                    wxf::WXFValueList ed;
                    ed.push_back(wxf::WXFValue(idx++));
                    for (auto v : ce) ed.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                    edge_list.push_back(wxf::WXFValue(ed));
                }
            } else {
                for (const auto& ce : *state_edges[rep]) {
                    wxf::WXFValueList ed;
                    ed.push_back(wxf::WXFValue(idx++));
                    for (auto v : ce) ed.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                    edge_list.push_back(wxf::WXFValue(ed));
                }
            }
            bool is_init = is_output.find(rep) == is_output.end();
            int64_t step = is_init ? 0 : static_cast<int64_t>(state_step[rep]);
            wxf::WXFValueAssociation sa;
            sa.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(rep))});
            sa.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(static_cast<int64_t>(rep))});
            sa.push_back({wxf::WXFValue("ContentStateId"), wxf::WXFValue(static_cast<int64_t>(rep))});
            sa.push_back({wxf::WXFValue("Step"), wxf::WXFValue(step)});
            sa.push_back({wxf::WXFValue("Edges"), wxf::WXFValue(edge_list)});
            sa.push_back({wxf::WXFValue("IsInitial"), wxf::WXFValue(is_init)});
            if (job.include_canonical_hashes) {
                sa.push_back({wxf::WXFValue("CanonicalHash"),
                              wxf::WXFValue(static_cast<int64_t>(state_hash[rep]))});
            }
            states_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(rep)), wxf::WXFValue(sa)});
        }
        full_result.push_back({wxf::WXFValue("States"), wxf::WXFValue(states_assoc)});
    }

    if (job.include_events) {
        wxf::WXFValueAssociation events_assoc;
        for (const auto& e : result.events) {
            int64_t canon_event = (e.canonical_id == hg_gpu::INVALID_ID)
                ? static_cast<int64_t>(e.id) : static_cast<int64_t>(e.canonical_id);
            wxf::WXFValueList consumed, produced;
            for (auto c : e.consumed_edges)
                if (c != hg_gpu::INVALID_ID) consumed.push_back(wxf::WXFValue(static_cast<int64_t>(c)));
            for (auto p : e.produced_edges)
                if (p != hg_gpu::INVALID_ID) produced.push_back(wxf::WXFValue(static_cast<int64_t>(p)));
            wxf::WXFValueAssociation ea;
            ea.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(e.id))});
            ea.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(canon_event)});
            ea.push_back({wxf::WXFValue("RuleIndex"), wxf::WXFValue(static_cast<int64_t>(e.rule))});
            ea.push_back({wxf::WXFValue("InputState"), wxf::WXFValue(static_cast<int64_t>(e.input_state))});
            ea.push_back({wxf::WXFValue("OutputState"), wxf::WXFValue(static_cast<int64_t>(e.output_state))});
            ea.push_back({wxf::WXFValue("CanonicalInputState"), wxf::WXFValue(rep_of(e.input_state))});
            ea.push_back({wxf::WXFValue("CanonicalOutputState"), wxf::WXFValue(rep_of(e.output_state))});
            ea.push_back({wxf::WXFValue("ConsumedEdges"), wxf::WXFValue(consumed)});
            ea.push_back({wxf::WXFValue("ProducedEdges"), wxf::WXFValue(produced)});
            events_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(e.id)), wxf::WXFValue(ea)});
        }
        full_result.push_back({wxf::WXFValue("Events"), wxf::WXFValue(events_assoc)});
    }

    // CausalEdges: dedup by raw (from,to), matching the FFI. The deduped count feeds
    // NumCausalEdges and must be computed even when the edge list itself is not requested
    // (e.g. the counts-only "Debug" property), so the count matches the CPU in every case.
    int64_t num_causal = 0;
    {
        wxf::WXFValueList causal;
        std::unordered_set<uint64_t> seen;
        for (const auto& c : result.causal_edges) {
            uint64_t key = (static_cast<uint64_t>(c.from) << 32) | static_cast<uint32_t>(c.to);
            if (!seen.insert(key).second) continue;
            if (job.include_causal_edges) {
                wxf::WXFValueAssociation ed;
                ed.push_back({wxf::WXFValue("From"), wxf::WXFValue(static_cast<int64_t>(c.from))});
                ed.push_back({wxf::WXFValue("To"), wxf::WXFValue(static_cast<int64_t>(c.to))});
                ed.push_back({wxf::WXFValue("RawFrom"), wxf::WXFValue(static_cast<int64_t>(c.from))});
                ed.push_back({wxf::WXFValue("RawTo"), wxf::WXFValue(static_cast<int64_t>(c.to))});
                causal.push_back(wxf::WXFValue(ed));
            }
        }
        num_causal = static_cast<int64_t>(seen.size());
        if (job.include_causal_edges)
            full_result.push_back({wxf::WXFValue("CausalEdges"), wxf::WXFValue(causal)});
    }

    // BranchialEdges: no dedup (multiplicity matters).
    if (job.include_branchial_edges) {
        wxf::WXFValueList branchial;
        for (const auto& b : result.branchial_edges) {
            wxf::WXFValueAssociation ed;
            ed.push_back({wxf::WXFValue("From"), wxf::WXFValue(static_cast<int64_t>(b.a))});
            ed.push_back({wxf::WXFValue("To"), wxf::WXFValue(static_cast<int64_t>(b.b))});
            branchial.push_back(wxf::WXFValue(ed));
        }
        full_result.push_back({wxf::WXFValue("BranchialEdges"), wxf::WXFValue(branchial)});
    }

    // NumStates mirrors the CPU's num_canonical_states() = canonical_state_map_.count_unique(),
    // verified against the CPU engine in gpu/tests/test_gpu_vs_cpu_differential.cpp
    // (CanonicalStateCount.ModesVsCpu). class_reps is grouped by the mode's dedup key, so its size
    // is exactly the CPU count in every mode: None -> raw state count, Automatic -> distinct
    // content, Full -> distinct IR class. (The CPU's None-mode sentinel undercount is fixed in
    // create_or_get_canonical_state, so no adjustment is needed here.)
    full_result.push_back({wxf::WXFValue("NumStates"),
                           wxf::WXFValue(static_cast<int64_t>(class_reps.size()))});
    full_result.push_back({wxf::WXFValue("NumEvents"),
                           wxf::WXFValue(static_cast<int64_t>(result.events.size()))});
    full_result.push_back({wxf::WXFValue("NumCausalEdges"), wxf::WXFValue(num_causal)});
    full_result.push_back({wxf::WXFValue("NumBranchialEdges"),
                           wxf::WXFValue(static_cast<int64_t>(result.branchial_edges.size()))});

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

        auto serialize_edges = [&](hg_gpu::StateId sid) -> wxf::WXFValueList {
            wxf::WXFValueList edge_list;
            auto it = state_edges.find(sid);
            if (it == state_edges.end()) return edge_list;
            int64_t idx = 0;
            if (full) {
                auto canon = ir.canonicalize_edges(*it->second);
                for (const auto& ce : canon.canonical_form.edges) {
                    wxf::WXFValueList ed; ed.push_back(wxf::WXFValue(idx++));
                    for (auto v : ce) ed.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                    edge_list.push_back(wxf::WXFValue(ed));
                }
            } else {
                for (const auto& ce : *it->second) {
                    wxf::WXFValueList ed; ed.push_back(wxf::WXFValue(idx++));
                    for (auto v : ce) ed.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                    edge_list.push_back(wxf::WXFValue(ed));
                }
            }
            return edge_list;
        };
        auto step_of = [&](hg_gpu::StateId sid) -> uint32_t {
            auto it = state_step.find(sid);
            return it == state_step.end() ? 0u : it->second;
        };
        auto is_init = [&](hg_gpu::StateId sid) -> bool {
            return is_output.find(sid) == is_output.end();
        };
        auto eff_event = [&](hg_gpu::EventId eid) -> int64_t {
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
            std::function<wxf::WXFValueAssociation(uint32_t)> state_data_;
            std::function<bool(uint32_t)> event_valid_;
            std::function<int64_t(uint32_t)> eff_event_;
            std::function<uint32_t(uint32_t)> in_state_;
            std::function<uint32_t(uint32_t)> out_state_;
            std::function<wxf::WXFValueAssociation(uint32_t)> event_data_;
            std::vector<std::pair<uint32_t, uint32_t>> causal_pairs_;
            std::vector<std::pair<uint32_t, uint32_t>> branchial_pairs_;

            uint32_t num_states() const { return n_states; }
            bool state_valid(uint32_t sid) const { return state_valid_(sid); }
            int64_t effective_state_id(uint32_t sid) const { return eff_state_(sid); }
            uint32_t state_step(uint32_t sid) const { return step_(sid); }
            wxf::WXFValueAssociation serialize_state_data(uint32_t sid) const { return state_data_(sid); }
            uint32_t num_raw_events() const { return n_events; }
            bool is_valid_event(uint32_t eid) const { return event_valid_(eid); }
            int64_t effective_event_id(uint32_t eid) const { return eff_event_(eid); }
            uint32_t event_input_state(uint32_t eid) const { return in_state_(eid); }
            uint32_t event_output_state(uint32_t eid) const { return out_state_(eid); }
            wxf::WXFValueAssociation serialize_event_data(uint32_t eid) const { return event_data_(eid); }
            std::vector<std::pair<uint32_t, uint32_t>> causal_event_pairs() const { return causal_pairs_; }
            std::vector<std::pair<uint32_t, uint32_t>> branchial_event_pairs() const { return branchial_pairs_; }
        };

        GpuGraphSource gsrc;
        gsrc.n_states = max_state + 1;
        gsrc.n_events = max_event + 1;
        gsrc.state_valid_ = [&](uint32_t sid) { return state_edges.find(sid) != state_edges.end(); };
        gsrc.eff_state_ = [&](uint32_t sid) { return rep_of(sid); };
        gsrc.step_ = step_of;
        gsrc.state_data_ = [&](uint32_t sid) -> wxf::WXFValueAssociation {
            wxf::WXFValueAssociation d;
            d.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(sid))});
            d.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(rep_of(sid))});
            d.push_back({wxf::WXFValue("Step"), wxf::WXFValue(static_cast<int64_t>(step_of(sid)))});
            d.push_back({wxf::WXFValue("Edges"), wxf::WXFValue(serialize_edges(sid))});
            d.push_back({wxf::WXFValue("IsInitial"), wxf::WXFValue(is_init(sid))});
            return d;
        };
        gsrc.event_valid_ = [&](uint32_t eid) { return event_by_id.find(eid) != event_by_id.end(); };
        gsrc.eff_event_ = eff_event;
        gsrc.in_state_ = [&](uint32_t eid) -> uint32_t {
            auto it = event_by_id.find(eid); return it == event_by_id.end() ? 0u : it->second->input_state; };
        gsrc.out_state_ = [&](uint32_t eid) -> uint32_t {
            auto it = event_by_id.find(eid); return it == event_by_id.end() ? 0u : it->second->output_state; };
        gsrc.event_data_ = [&](uint32_t eid) -> wxf::WXFValueAssociation {
            auto eit = event_by_id.find(eid);
            wxf::WXFValueAssociation d;
            if (eit == event_by_id.end()) return d;
            const hg_gpu::Event& e = *eit->second;
            wxf::WXFValueList consumed, produced;
            for (auto c : e.consumed_edges) if (c != hg_gpu::INVALID_ID) consumed.push_back(wxf::WXFValue(static_cast<int64_t>(c)));
            for (auto p : e.produced_edges) if (p != hg_gpu::INVALID_ID) produced.push_back(wxf::WXFValue(static_cast<int64_t>(p)));
            d.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(eid))});
            d.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(eff_event(eid))});
            d.push_back({wxf::WXFValue("RuleIndex"), wxf::WXFValue(static_cast<int64_t>(e.rule))});
            d.push_back({wxf::WXFValue("InputState"), wxf::WXFValue(static_cast<int64_t>(e.input_state))});
            d.push_back({wxf::WXFValue("OutputState"), wxf::WXFValue(static_cast<int64_t>(e.output_state))});
            d.push_back({wxf::WXFValue("ConsumedEdges"), wxf::WXFValue(consumed)});
            d.push_back({wxf::WXFValue("ProducedEdges"), wxf::WXFValue(produced)});
            d.push_back({wxf::WXFValue("InputStateEdges"), wxf::WXFValue(serialize_edges(e.input_state))});
            d.push_back({wxf::WXFValue("OutputStateEdges"), wxf::WXFValue(serialize_edges(e.output_state))});
            return d;
        };
        for (const auto& c : result.causal_edges)
            gsrc.causal_pairs_.emplace_back(static_cast<uint32_t>(c.from), static_cast<uint32_t>(c.to));
        for (const auto& b : result.branchial_edges)
            gsrc.branchial_pairs_.emplace_back(static_cast<uint32_t>(b.a), static_cast<uint32_t>(b.b));

        hgmarshal::GraphOptions gopts;
        gopts.edge_deduplication = job.edge_deduplication;
        gopts.branchial_step = job.branchial_step;
        gopts.steps = job.steps;
        full_result.push_back({wxf::WXFValue("GraphData"),
                               hgmarshal::build_graph_data(gsrc, job.graph_properties, gopts)});
    }

    // Surface capacity overflows as a partial-result warning trail.
    if (!result.warnings.empty()) {
        wxf::WXFValueList warn;
        for (const auto& w : result.warnings) {
            wxf::WXFValueAssociation wa;
            wa.push_back({wxf::WXFValue("Kind"), wxf::WXFValue(std::string(hg_gpu::error_kind_name(w.kind)))});
            wa.push_back({wxf::WXFValue("Count"), wxf::WXFValue(static_cast<int64_t>(w.count))});
            wa.push_back({wxf::WXFValue("Context"), wxf::WXFValue(w.context)});
            warn.push_back(wxf::WXFValue(wa));
        }
        if (host.progress) host.progress("HGEvolve (GPU): capacity overflow -- returning partial result");
        full_result.push_back({wxf::WXFValue("Warnings"), wxf::WXFValue(warn)});
    }

    wxf::Writer writer;
    writer.write_header();
    writer.write(wxf::WXFValue(full_result));
    return writer.data();
}

#endif  // HG_GPU_BACKEND
