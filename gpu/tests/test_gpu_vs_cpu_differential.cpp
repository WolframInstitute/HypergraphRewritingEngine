// Differential test harness: same workload, CPU vs GPU, must produce identical
// results when compared by isomorphism-key. Coverage starts as just canonical
// state hashes; M3.6 / M4.10 / M5.8 / M6.7 add events, causal edges, branchial
// edges, and event canonicalization keys.
//
// All-empty workloads pass today. Anything non-empty fails until M5.8 lands the
// first end-to-end GPU pipeline (match → rewrite → IR-dedup). That is the
// intended state — the harness exists *before* the kernels so each kernel
// closes one column.

#include <gtest/gtest.h>

#include "hg_gpu/evolve.hpp"

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"

#include <set>
#include <string>
#include <vector>

namespace {

struct Workload {
    std::string name;
    std::vector<hg_gpu::RewriteRule> rules;
    std::vector<std::vector<hg_gpu::VertexId>> initial_state;
    uint32_t num_steps = 0;
    hg_gpu::CanonicalizationMode canon_mode = hg_gpu::CanonicalizationMode::Full;
    hg_gpu::EventCanonicalizationMode event_canon_mode = hg_gpu::EventCanonicalizationMode::None;
    bool transitive_reduction = true;
    // Reference semantics (reference/MultiwayReference.wl): every state is
    // expanded. The engine reproduces the reference exactly on this path
    // (wolfram_canonical_steps5: states=302, events=1174, causal(TR)=1332).
    bool explore_from_canonical_states_only = false;
};

// Result normalized for cross-engine comparison. States compare by
// isomorphism key (IR canonical hash on raw edges). Events are normalised
// to (input_state_hash, output_state_hash, rule, step) keys — this collapses
// events whose input/output states are isomorphic while preserving
// multiplicity (multiset). Causal/branchial edges are normalised as pairs of
// event-keys, multiplicity-preserving.
struct NormalizedResult {
    std::set<uint64_t>      canonical_state_hashes;
    std::multiset<uint64_t> event_keys;
    std::multiset<uint64_t> causal_edge_keys;
    std::multiset<uint64_t> branchial_edge_keys;

    bool operator==(const NormalizedResult& o) const {
        return canonical_state_hashes == o.canonical_state_hashes
            && event_keys              == o.event_keys
            && causal_edge_keys        == o.causal_edge_keys
            && branchial_edge_keys     == o.branchial_edge_keys;
    }
};

uint64_t mix_fnv(uint64_t h, uint64_t x) {
    h ^= x;
    h *= 1099511628211ULL;
    return h;
}
uint64_t event_key(uint64_t input_hash, uint64_t output_hash, uint32_t rule, uint32_t step) {
    uint64_t h = 14695981039346656037ULL;
    h = mix_fnv(h, input_hash);
    h = mix_fnv(h, output_hash);
    h = mix_fnv(h, rule);
    h = mix_fnv(h, step);
    return h;
}
uint64_t edge_pair_key(uint64_t a_key, uint64_t b_key) {
    uint64_t lo = a_key < b_key ? a_key : b_key;
    uint64_t hi = a_key < b_key ? b_key : a_key;
    uint64_t h = 14695981039346656037ULL;
    h = mix_fnv(h, lo);
    h = mix_fnv(h, hi);
    return h;
}
uint64_t causal_key(uint64_t from_key, uint64_t to_key) {
    // Directed: (from, to), not min/max.
    uint64_t h = 14695981039346656037ULL;
    h = mix_fnv(h, from_key);
    h = mix_fnv(h, to_key);
    return h;
}

hypergraph::StateCanonicalizationMode to_cpu_canon(hg_gpu::CanonicalizationMode m) {
    switch (m) {
        case hg_gpu::CanonicalizationMode::None:      return hypergraph::StateCanonicalizationMode::None;
        case hg_gpu::CanonicalizationMode::Automatic: return hypergraph::StateCanonicalizationMode::Automatic;
        case hg_gpu::CanonicalizationMode::Full:      return hypergraph::StateCanonicalizationMode::Full;
    }
    return hypergraph::StateCanonicalizationMode::None;
}

hypergraph::RewriteRule convert_rule(const hg_gpu::RewriteRule& src, uint16_t index) {
    hypergraph::RuleBuilder b(index);
    for (const auto& edge : src.lhs) b.lhs(edge);
    for (const auto& edge : src.rhs) b.rhs(edge);
    return b.build();
}

NormalizedResult run_cpu(const Workload& w) {
    NormalizedResult out;

    hypergraph::Hypergraph hg;
    // The internal CPU exploration-time edge hash is WL (fast + deterministic,
    // only affects per-step CPU dedup, not cross-engine comparison). The test's
    // correctness comes from the final IRCanonicalizer on both sides.
    hg.set_state_canonicalization_mode(to_cpu_canon(w.canon_mode));

    hypergraph::ParallelEvolutionEngine engine(&hg, /*num_threads=*/0);
    for (size_t i = 0; i < w.rules.size(); ++i) {
        engine.add_rule(convert_rule(w.rules[i], static_cast<uint16_t>(i)));
    }
    engine.set_transitive_reduction(w.transitive_reduction);
    engine.set_explore_from_canonical_states_only(w.explore_from_canonical_states_only);

    engine.evolve(w.initial_state, w.num_steps);

    if (false /* diag off */) {
        std::printf("[diag %s] cpu_num_states=%u cpu_num_events=%u\n",
                    w.name.c_str(), hg.num_states(), hg.num_events());
        const uint32_t n_cpu = hg.num_states();
        for (uint32_t sid = 0; sid < n_cpu; ++sid) {
            const auto& state = hg.get_state(sid);
            if (state.id == hypergraph::INVALID_ID) continue;
            std::printf("  cpu state %u edges:", sid);
            state.edges.for_each([&](hypergraph::EdgeId eid) {
                const auto& e = hg.get_edge(eid);
                std::printf(" (");
                for (uint8_t i = 0; i < e.arity; ++i) std::printf("%u,", e.vertices[i]);
                std::printf(")");
            });
            std::printf("\n");
        }
        std::fflush(stdout);
    }

    hypergraph::IRCanonicalizer ir;

    // Per-state canonical hashes: also build StateId → hash map for event
    // normalisation.
    std::unordered_map<uint32_t, uint64_t> state_hash_by_id;
    const uint32_t n = hg.num_states();
    for (uint32_t sid = 0; sid < n; ++sid) {
        const auto& state = hg.get_state(sid);
        if (state.id == hypergraph::INVALID_ID) continue;

        std::vector<std::vector<hg_gpu::VertexId>> edges;
        state.edges.for_each([&](hypergraph::EdgeId eid) {
            const auto& e = hg.get_edge(eid);
            std::vector<hg_gpu::VertexId> vs;
            vs.reserve(e.arity);
            for (uint8_t i = 0; i < e.arity; ++i) vs.push_back(e.vertices[i]);
            edges.push_back(std::move(vs));
        });
        uint64_t h = ir.compute_canonical_hash(edges);
        state_hash_by_id[sid] = h;
        out.canonical_state_hashes.insert(h);
    }

    // EventId → event_key (input_hash, output_hash, rule, step).
    std::unordered_map<uint32_t, uint64_t> event_key_by_id;
    for (uint32_t eid = 0; eid < hg.num_events(); ++eid) {
        const auto& ev = hg.get_event(eid);
        if (ev.id == hypergraph::INVALID_ID) continue;
        uint64_t ih = state_hash_by_id.count(ev.input_state)  ? state_hash_by_id[ev.input_state]  : 0ULL;
        uint64_t oh = state_hash_by_id.count(ev.output_state) ? state_hash_by_id[ev.output_state] : 0ULL;
        uint32_t step = hg.get_state(ev.output_state).step;
        // Quotient mode compares transitions without the step: each canonical state
        // is expanded once, so (input, output, rule) determines the step within a
        // run, but WHICH depth the CPU's dataflow claims a state at is arrival-
        // dependent, while the GPU's level-synchronised loop always claims at the
        // minimum. The step-less multiset is the canonical transition multiset.
        uint64_t ek = event_key(ih, oh, ev.rule_index,
                                w.explore_from_canonical_states_only ? 0u : step);
        event_key_by_id[eid] = ek;
        out.event_keys.insert(ek);
    }

    for (const auto& ce : hg.causal_graph().get_causal_edges()) {
        if (ce.producer == hypergraph::INVALID_ID || ce.consumer == hypergraph::INVALID_ID) continue;
        auto pit = event_key_by_id.find(ce.producer);
        auto cit = event_key_by_id.find(ce.consumer);
        if (pit == event_key_by_id.end() || cit == event_key_by_id.end()) continue;
        out.causal_edge_keys.insert(causal_key(pit->second, cit->second));
    }

    for (const auto& be : hg.causal_graph().get_branchial_edges()) {
        if (be.event1 == hypergraph::INVALID_ID || be.event2 == hypergraph::INVALID_ID) continue;
        auto it1 = event_key_by_id.find(be.event1);
        auto it2 = event_key_by_id.find(be.event2);
        if (it1 == event_key_by_id.end() || it2 == event_key_by_id.end()) continue;
        out.branchial_edge_keys.insert(edge_pair_key(it1->second, it2->second));
    }
    return out;
}

NormalizedResult run_gpu(const Workload& w) {
    NormalizedResult out;

    hg_gpu::EvolveInput in;
    in.rules                  = w.rules;
    in.initial_state          = w.initial_state;
    in.num_steps              = w.num_steps;
    in.canonicalization       = w.canon_mode;
    in.event_canonicalization = w.event_canon_mode;
    in.transitive_reduction   = w.transitive_reduction;
    in.explore_from_canonical_states_only = w.explore_from_canonical_states_only;

    auto result = hg_gpu::evolve(in);

    if (false /* diag off */) {
        std::printf("[diag %s] gpu_states=%zu gpu_events=%zu\n",
                    w.name.c_str(), result.states.size(), result.events.size());
        for (const auto& s : result.states) {
            std::printf("  gpu state %u edges:", s.id);
            for (const auto& e : s.edges) {
                std::printf(" (");
                for (auto v : e) std::printf("%u,", v);
                std::printf(")");
            }
            std::printf("\n");
        }
        std::fflush(stdout);
    }

    hypergraph::IRCanonicalizer ir;
    std::unordered_map<uint32_t, uint64_t> state_hash_by_id;
    for (const auto& s : result.states) {
        uint64_t h = ir.compute_canonical_hash(s.edges);
        state_hash_by_id[s.id] = h;
        out.canonical_state_hashes.insert(h);
    }

    std::unordered_map<uint32_t, uint64_t> event_key_by_id;
    for (const auto& ev : result.events) {
        uint64_t ih = state_hash_by_id.count(ev.input_state)  ? state_hash_by_id[ev.input_state]  : 0ULL;
        uint64_t oh = state_hash_by_id.count(ev.output_state) ? state_hash_by_id[ev.output_state] : 0ULL;
        uint64_t ek = event_key(ih, oh, ev.rule,
                                w.explore_from_canonical_states_only ? 0u : ev.step);
        event_key_by_id[ev.id] = ek;
        out.event_keys.insert(ek);
    }

    for (const auto& ce : result.causal_edges) {
        auto pit = event_key_by_id.find(ce.from);
        auto cit = event_key_by_id.find(ce.to);
        if (pit == event_key_by_id.end() || cit == event_key_by_id.end()) continue;
        out.causal_edge_keys.insert(causal_key(pit->second, cit->second));
    }
    for (const auto& be : result.branchial_edges) {
        auto it1 = event_key_by_id.find(be.a);
        auto it2 = event_key_by_id.find(be.b);
        if (it1 == event_key_by_id.end() || it2 == event_key_by_id.end()) continue;
        out.branchial_edge_keys.insert(edge_pair_key(it1->second, it2->second));
    }
    return out;
}

class DifferentialEvolution : public ::testing::TestWithParam<Workload> {};

TEST_P(DifferentialEvolution, BitIdenticalCanonicalForm) {
    const Workload& w = GetParam();
    NormalizedResult cpu = run_cpu(w);
    NormalizedResult gpu = run_gpu(w);
    EXPECT_EQ(cpu.canonical_state_hashes, gpu.canonical_state_hashes)
        << "Workload: " << w.name
        << " state sets differ; cpu=" << cpu.canonical_state_hashes.size()
        << " gpu=" << gpu.canonical_state_hashes.size();
    EXPECT_EQ(cpu.event_keys, gpu.event_keys)
        << "Workload: " << w.name
        << " event multisets differ; cpu=" << cpu.event_keys.size()
        << " gpu=" << gpu.event_keys.size();
    // Quotient mode records only the expanded representative's causal/branchial
    // edges, and which raw representative carries them is a claim race on the CPU.
    // Exact causal and branchial multisets are reconstructed offline from the
    // skeleton (tools/quotient_reconstruction_probe.cpp), so the online sets are
    // not a cross-engine invariant in this mode.
    if (!w.explore_from_canonical_states_only) {
        EXPECT_EQ(cpu.causal_edge_keys, gpu.causal_edge_keys)
            << "Workload: " << w.name
            << " causal multisets differ; cpu=" << cpu.causal_edge_keys.size()
            << " gpu=" << gpu.causal_edge_keys.size();
        EXPECT_EQ(cpu.branchial_edge_keys, gpu.branchial_edge_keys)
            << "Workload: " << w.name
            << " branchial multisets differ; cpu=" << cpu.branchial_edge_keys.size()
            << " gpu=" << gpu.branchial_edge_keys.size();
    }
}

// =============================================================================
// Workload corpus
// =============================================================================
// Lifted from hypergraph/tests/test_determinism_fuzzing.cpp plus additions
// per gpu/ARCHITECTURE.md §10 (2-edge, 3-edge, mixed-arity, self-loop,
// Wolfram canonical, multi-rule). All non-trivial cases will fail until
// M5.8 lands the first end-to-end GPU pipeline.

hg_gpu::RewriteRule rule(std::vector<std::vector<uint8_t>> lhs,
                         std::vector<std::vector<uint8_t>> rhs) {
    hg_gpu::RewriteRule r;
    r.lhs = std::move(lhs);
    r.rhs = std::move(rhs);
    uint8_t lhs_max = 0;
    for (auto& e : r.lhs) for (auto v : e) lhs_max = std::max<uint8_t>(lhs_max, v);
    uint8_t rhs_max = 0;
    for (auto& e : r.rhs) for (auto v : e) rhs_max = std::max<uint8_t>(rhs_max, v);
    r.num_lhs_vars = static_cast<uint8_t>(r.lhs.empty() ? 0 : lhs_max + 1);
    r.num_rhs_vars = static_cast<uint8_t>(r.rhs.empty() ? 0 : rhs_max + 1);
    return r;
}

std::vector<Workload> build_corpus() {
    using V = std::vector<std::vector<hg_gpu::VertexId>>;
    std::vector<Workload> ws;

    ws.push_back({.name = "empty_rules_empty_initial_zero_steps", .num_steps = 0});

    // 1-edge LHS, branching rule, simple initial.
    ws.push_back({
        .name = "1edge_branching_steps1",
        .rules = {rule({{0,1}}, {{0,1},{1,2}})},
        .initial_state = V{{0u,1u}},
        .num_steps = 1,
    });
    ws.push_back({
        .name = "1edge_branching_steps3",
        .rules = {rule({{0,1}}, {{0,1},{1,2}})},
        .initial_state = V{{0u,1u}},
        .num_steps = 3,
    });

    // 2-edge LHS.
    ws.push_back({
        .name = "2edge_lhs_triangle_init",
        .rules = {rule({{0,1},{1,2}}, {{0,1},{1,2},{1,3}})},
        .initial_state = V{{0u,1u},{1u,2u},{2u,0u}},
        .num_steps = 1,
    });
    ws.push_back({
        .name = "2edge_lhs_path_init_steps2",
        .rules = {rule({{0,1},{1,2}}, {{0,1},{1,2},{1,3}})},
        .initial_state = V{{0u,1u},{1u,2u}},
        .num_steps = 2,
    });

    // 3-arity LHS — mixed arity.
    ws.push_back({
        .name = "3arity_lhs_to_2arity_rhs",
        .rules = {rule({{0,1,2}}, {{0,1},{0,2},{0,3}})},
        .initial_state = V{{0u,1u,2u}},
        .num_steps = 1,
    });

    // Multi-rule with mixed arity in the same engine.
    ws.push_back({
        .name = "multirule_arity3_plus_arity2",
        .rules = {
            rule({{0,1,2}}, {{0,1},{0,2},{0,3}}),
            rule({{0,1}}, {{0,1},{0,2}}),
        },
        .initial_state = V{{0u,1u,2u}},
        .num_steps = 2,
    });

    // Self-loop initial (Wolfram non-distinct binding stress).
    ws.push_back({
        .name = "selfloop_initial_2edge_lhs",
        .rules = {rule({{0,1},{1,2}}, {{0,1},{1,2},{1,3}})},
        .initial_state = V{{0u,0u},{0u,0u}},
        .num_steps = 1,
    });

    // Wolfram canonical: {{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}}
    ws.push_back({
        .name = "wolfram_canonical_steps1",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = V{{0u,1u},{0u,2u}},
        .num_steps = 1,
    });
    ws.push_back({
        .name = "wolfram_canonical_steps3",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = V{{0u,1u},{0u,2u}},
        .num_steps = 3,
    });
    ws.push_back({
        .name = "wolfram_canonical_steps5",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = V{{0u,1u},{0u,2u}},
        .num_steps = 5,
    });

    // Cover canon mode None too — exercises a different CPU code path.
    Workload none_mode = {
        .name = "1edge_branching_canon_none_steps2",
        .rules = {rule({{0,1}}, {{0,1},{1,2}})},
        .initial_state = V{{0u,1u}},
        .num_steps = 2,
        .canon_mode = hg_gpu::CanonicalizationMode::None,
    };
    ws.push_back(none_mode);

    // Quotient exploration (explore_from_canonical_states_only): each canonical
    // state is expanded once, at its shortest depth. The CPU reaches that via
    // depth relaxation over its dataflow; the GPU's level-synchronised step loop
    // gives it by construction. Compared on canonical states and the step-less
    // transition multiset; causal/branchial are reconstructed offline in this
    // mode. The multi-rule workloads put loops in the multiway states graph,
    // which is where first-discovery ordering used to diverge.
    ws.push_back({
        .name = "quotient_wolfram_steps5",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = V{{0u,1u},{0u,2u}},
        .num_steps = 5,
        .explore_from_canonical_states_only = true,
    });
    ws.push_back({
        .name = "quotient_all_three_triangle",
        .rules = {rule({{0,1}}, {{0,2},{2,1}}),
                  rule({{0,1}}, {{1,0}}),
                  rule({{0,1},{1,2}}, {{0,2}})},
        .initial_state = V{{0u,1u},{1u,2u},{2u,0u}},
        .num_steps = 3,
        .explore_from_canonical_states_only = true,
    });
    ws.push_back({
        .name = "quotient_all_three_two_edges",
        .rules = {rule({{0,1}}, {{0,2},{2,1}}),
                  rule({{0,1}}, {{1,0}}),
                  rule({{0,1},{1,2}}, {{0,2}})},
        .initial_state = V{{0u,1u},{0u,2u}},
        .num_steps = 3,
        .explore_from_canonical_states_only = true,
    });
    ws.push_back({
        .name = "quotient_dupe_dedup",
        .rules = {rule({{0,1}}, {{0,1},{0,1}}),
                  rule({{0,1},{0,1}}, {{0,1}})},
        .initial_state = V{{0u,1u},{0u,1u}},
        .num_steps = 3,
        .explore_from_canonical_states_only = true,
    });

    return ws;
}

INSTANTIATE_TEST_SUITE_P(InitialCorpus, DifferentialEvolution,
    ::testing::ValuesIn(build_corpus()),
    [](const ::testing::TestParamInfo<Workload>& info) { return info.param.name; });

}  // namespace
