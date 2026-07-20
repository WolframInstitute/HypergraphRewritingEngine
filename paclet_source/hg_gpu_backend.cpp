#ifdef HG_GPU_BACKEND

#include "hg_gpu_backend.hpp"

#include "hg_gpu/evolve.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "wxf.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
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
    // The GPU canonicalizes states with McKay IR only; Full is the sole correct
    // mode. State dedup / class collapse is done host-side in the marshaler.
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;
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

    // Group the raw per-provenance GPU states into canonical classes by their
    // host-recomputed IR hash; the class representative (first-seen id) is the
    // stable handle emitted for that canonical state.
    std::unordered_map<uint64_t, hg_gpu::StateId> hash_to_rep;
    std::unordered_map<hg_gpu::StateId, hg_gpu::StateId> state_to_rep;
    std::unordered_map<hg_gpu::StateId, uint64_t> state_hash;
    std::unordered_map<hg_gpu::StateId, const std::vector<std::vector<hg_gpu::VertexId>>*> state_edges;
    std::vector<hg_gpu::StateId> class_reps;
    for (const auto& s : result.states) {
        uint64_t h = ir.compute_canonical_hash(s.edges);
        state_hash[s.id] = h;
        state_edges[s.id] = &s.edges;
        auto it = hash_to_rep.find(h);
        if (it == hash_to_rep.end()) {
            hash_to_rep[h] = s.id;
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
            auto canon = ir.canonicalize_edges(*state_edges[rep]);
            int64_t idx = 0;
            for (const auto& ce : canon.canonical_form.edges) {
                wxf::WXFValueList ed;
                ed.push_back(wxf::WXFValue(idx++));
                for (auto v : ce) ed.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                edge_list.push_back(wxf::WXFValue(ed));
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

    // CausalEdges: dedup by raw (from,to), matching the FFI.
    int64_t num_causal = 0;
    if (job.include_causal_edges) {
        wxf::WXFValueList causal;
        std::unordered_set<uint64_t> seen;
        for (const auto& c : result.causal_edges) {
            uint64_t key = (static_cast<uint64_t>(c.from) << 32) | static_cast<uint32_t>(c.to);
            if (!seen.insert(key).second) continue;
            wxf::WXFValueAssociation ed;
            ed.push_back({wxf::WXFValue("From"), wxf::WXFValue(static_cast<int64_t>(c.from))});
            ed.push_back({wxf::WXFValue("To"), wxf::WXFValue(static_cast<int64_t>(c.to))});
            ed.push_back({wxf::WXFValue("RawFrom"), wxf::WXFValue(static_cast<int64_t>(c.from))});
            ed.push_back({wxf::WXFValue("RawTo"), wxf::WXFValue(static_cast<int64_t>(c.to))});
            causal.push_back(wxf::WXFValue(ed));
        }
        num_causal = static_cast<int64_t>(causal.size());
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

    full_result.push_back({wxf::WXFValue("NumStates"),
                           wxf::WXFValue(static_cast<int64_t>(class_reps.size()))});
    full_result.push_back({wxf::WXFValue("NumEvents"),
                           wxf::WXFValue(static_cast<int64_t>(result.events.size()))});
    full_result.push_back({wxf::WXFValue("NumCausalEdges"), wxf::WXFValue(num_causal)});
    full_result.push_back({wxf::WXFValue("NumBranchialEdges"),
                           wxf::WXFValue(static_cast<int64_t>(result.branchial_edges.size()))});

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
