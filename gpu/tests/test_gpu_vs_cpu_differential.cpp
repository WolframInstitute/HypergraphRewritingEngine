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
    // Multiple initial states (multiway with several roots). Takes precedence
    // over initial_state when non-empty.
    std::vector<std::vector<std::vector<hg_gpu::VertexId>>> initial_states;
    uint32_t num_steps = 0;
    hg_gpu::CanonicalizationMode canon_mode = hg_gpu::CanonicalizationMode::Full;
    hg_gpu::EventCanonicalizationMode event_canon_mode = hg_gpu::EventCanonicalizationMode::None;
    bool transitive_reduction = true;
    // Reference semantics (reference/MultiwayReference.wl): every state is
    // expanded. The engine reproduces the reference exactly on this path
    // (wolfram_canonical_steps5: states=302, events=1174, causal(TR)=1332).
    bool explore_from_canonical_states_only = false;
    // Forwarded to EvolveInput::slice_scan_max_edges (0 keeps the default).
    // A tiny value forces the index-backed match path and the lazy index
    // rebuild on small workloads, cross-checking that regime against the CPU.
    uint32_t slice_scan_max_edges = 0;
    // Forwarded to EvolveInput::max_blocks_per_launch (0 = single launch). A
    // small value forces the match/rewrite kernels to run in chunks.
    uint32_t max_blocks_per_launch = 0;
    // Collapse isomorphic initial states under quotient exploration (default off).
    bool quotient_initial_states = false;
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

    // Count diagnostics for the NumStates convention (deliberately NOT in operator==):
    // NumStates as reported by HGEvolve is the CPU's num_canonical_states().
    size_t num_canonical_states = 0;  // cpu: hg.num_canonical_states(); gpu: n/a
    size_t raw_states = 0;            // cpu: hg.num_states(); gpu: result.states.size()
    size_t produced_states = 0;       // states that are the output of some event (non-root)
    size_t distinct_content = 0;      // distinct content-ordered (non-iso) hashes among states
    // Events, counted the way each device's user-facing NumEvents counts them: the CANONICAL
    // count once an event-identity mode is selected, the raw count otherwise.
    size_t num_events = 0;
    size_t raw_events = 0;
    std::multiset<uint64_t> content_multiset;  // per-state raw edge content (labels intact)
    std::multiset<uint64_t> iso_multiset;      // per-state IR canonical hash (labels normalised)
    // The hash each ENGINE reports for itself, as opposed to iso_multiset, which this harness
    // recomputes on the host from the returned edges. Recomputing on the host checks that the two
    // engines explored the same STATES; it cannot check that the device computed the same HASH
    // for them -- and the device's hash is what it deduplicates on and what a caller reads back.
    std::multiset<uint64_t> engine_state_hashes;

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
// Byte/content key: hashes a state's edge multiset with its LABELS intact (edges sorted for
// order-independence, but vertices NOT relabelled). Two states share this key iff their raw edge
// content is identical — a strictly stronger equivalence than the iso-invariant IR hash.
uint64_t content_key(std::vector<std::vector<hg_gpu::VertexId>> edges) {
    std::sort(edges.begin(), edges.end());
    uint64_t h = 14695981039346656037ULL;
    for (const auto& e : edges) {
        for (auto v : e) h = mix_fnv(h, static_cast<uint64_t>(v));
        h = mix_fnv(h, 0x2CULL);   // edge separator
    }
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

// The event-identity mode as the shared lattice spells it, mirroring evolve.cu's
// event_keys_for so both devices answer the same question.
hypergraph::EventSignatureKeys to_cpu_event_keys(hg_gpu::EventCanonicalizationMode m) {
    switch (m) {
        case hg_gpu::EventCanonicalizationMode::Full:      return hgcommon::EVENT_SIG_FULL;
        case hg_gpu::EventCanonicalizationMode::Automatic: return hgcommon::EVENT_SIG_AUTOMATIC;
        case hg_gpu::EventCanonicalizationMode::None:
        default:                                           return hgcommon::EVENT_SIG_NONE;
    }
}

NormalizedResult run_cpu(const Workload& w) {
    NormalizedResult out;

    hypergraph::Hypergraph hg;
    // The internal CPU exploration-time edge hash is WL (fast + deterministic,
    // only affects per-step CPU dedup, not cross-engine comparison). The test's
    // correctness comes from the final IRCanonicalizer on both sides.
    hg.set_state_canonicalization_mode(to_cpu_canon(w.canon_mode));
    // The event axis, which this harness used to leave at its default whatever the workload
    // asked for -- so every CPU run was EVENT_SIG_NONE and no comparison of the event modes
    // could have been meaningful.
    hg.set_event_signature_keys(to_cpu_event_keys(w.event_canon_mode));

    hypergraph::ParallelEvolutionEngine engine(&hg, /*num_threads=*/0);
    for (size_t i = 0; i < w.rules.size(); ++i) {
        engine.add_rule(convert_rule(w.rules[i], static_cast<uint16_t>(i)));
    }
    engine.set_transitive_reduction(w.transitive_reduction);
    engine.set_explore_from_canonical_states_only(w.explore_from_canonical_states_only);

    if (!w.initial_states.empty()) {
        std::vector<std::vector<std::vector<hypergraph::VertexId>>> roots;
        for (const auto& r : w.initial_states) {
            std::vector<std::vector<hypergraph::VertexId>> st;
            for (const auto& e : r) st.emplace_back(e.begin(), e.end());
            roots.push_back(std::move(st));
        }
        engine.set_quotient_initial_states(w.quotient_initial_states);
        engine.evolve(roots, w.num_steps);
    } else {
        engine.evolve(w.initial_state, w.num_steps);
    }

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
        out.content_multiset.insert(content_key(edges));
        out.iso_multiset.insert(h);
        out.engine_state_hashes.insert(hg.get_or_compute_canonical_hash(sid));
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
        // dependent, while the GPU claims at the minimum depth. The step-less
        // multiset is the canonical transition multiset.
        uint64_t ek = event_key(ih, oh, ev.rule_index,
                                w.explore_from_canonical_states_only ? 0u : step);
        event_key_by_id[eid] = ek;
        out.event_keys.insert(ek);
    }

    // Count diagnostics. NumStates as HGEvolve reports it is hg.num_canonical_states().
    out.num_canonical_states = hg.num_canonical_states();
    out.raw_states = state_hash_by_id.size();
    // NOT the count the FFI serves. HGEvolve reports hg.observable_num_events()
    // (hypergraph_ffi.cpp:1279), which under quotient exploration or EVENT_SIG_AUTOMATIC is the
    // RECONSTRUCTION's count, not this one. The device has no reconstruction, so switching this
    // line to observable_num_events() turns two assertions red -- see the pinned reproducer
    // ReconstructionGapIsStillOpen below, which holds the measured numbers. That switch is the
    // last step of the device port, not the first.
    out.num_events = hg.num_events();
    out.raw_events = hg.num_raw_events();
    {
        std::set<uint32_t> outs;
        for (uint32_t eid = 0; eid < hg.num_events(); ++eid) {
            const auto& ev = hg.get_event(eid);
            if (ev.id != hypergraph::INVALID_ID) outs.insert(ev.output_state);
        }
        out.produced_states = outs.size();
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

// One mapping from a Workload to a device run. Any test that needs the raw EvolveResult rather
// than the normalized comparison uses this, so the two cannot describe different runs.
hg_gpu::EvolveInput make_input(const Workload& w) {
    hg_gpu::EvolveInput in;
    in.rules                  = w.rules;
    in.initial_state          = w.initial_state;
    in.initial_states         = w.initial_states;
    in.quotient_initial_states = w.quotient_initial_states;
    in.num_steps              = w.num_steps;
    in.canonicalization       = w.canon_mode;
    in.event_canonicalization = w.event_canon_mode;
    // The scheduler is part of the workload, so it is stated rather than inherited: this suite's
    // job is to validate the GPU against the CPU, and which GPU scheduler answered has to be a
    // property of the case rather than of whatever the default happens to be.
    in.transitive_reduction   = w.transitive_reduction;
    in.explore_from_canonical_states_only = w.explore_from_canonical_states_only;
    in.slice_scan_max_edges = w.slice_scan_max_edges;
    in.max_blocks_per_launch = w.max_blocks_per_launch;
    return in;
}

NormalizedResult run_gpu(const Workload& w) {
    NormalizedResult out;

    hg_gpu::EvolveInput in = make_input(w);

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
        out.content_multiset.insert(content_key(s.edges));
        out.iso_multiset.insert(h);
        out.engine_state_hashes.insert(s.canonical_hash);
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

    // Count diagnostics (GPU side).
    out.raw_states = result.states.size();
    // Mirrors hg_gpu_backend.cpp's NumEvents: events that are their own canonical.
    out.raw_events = result.events.size();
    out.num_events = 0;
    for (const auto& e : result.events)
        if (e.canonical_id == hg_gpu::INVALID_ID) ++out.num_events;
    {
        std::set<uint32_t> outs;
        for (const auto& ev : result.events) outs.insert(ev.output_state);
        out.produced_states = outs.size();
        std::set<uint64_t> content;
        for (const auto& s : result.states) {
            auto e = s.edges;
            std::sort(e.begin(), e.end());
            uint64_t h = 1469598103934665603ULL;
            for (const auto& ed : e) {
                for (auto v : ed) h = (h ^ static_cast<uint64_t>(v)) * 1099511628211ULL;
                h = (h ^ 0x2CULL) * 1099511628211ULL;
            }
            content.insert(h);
        }
        out.distinct_content = content.size();
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

    // THE WORKLOAD THAT SEPARATES THE TWO RANK CONVENTIONS.
    //
    // An Automatic event signature hashes the RANK of each consumed and produced edge -- the
    // edge's position in a canonical labelling. Which state's labelling supplies that rank is a
    // choice, and the two choices are not the same function:
    //
    //   class frame  -- one labelling pinned per isomorphism class, shared by every raw state in
    //                   it. This is the linked-hypergraph convention of Wolfram/Multicomputation,
    //                   and it is what the CPU computes for EVENT_SIG_AUTOMATIC (the reconstruction
    //                   does the signing; see parallel_evolution.cpp, `qc`).
    //   per-state    -- each raw state's own labelling. This is the CPU's Positional identity, and
    //                   it is what edge_rank_in_state_device reads on the GPU, which has no
    //                   class-level frame at all.
    //
    // Where a state's automorphism group makes the canonical labelling a coset, the two can pick
    // different representatives, and then the class frame calls two applications ONE event while
    // per-state calls them TWO. On this pair of rules at 3 steps the CPU measures exactly that
    // split: 21 events by class frame, 23 by per-state (hypergraph.hpp
    // set_positional_event_identity; test_event_identity_authority "two-rules-overlap").
    //
    // Every other workload in this corpus happens to agree under both conventions, so without
    // this row the equality below is satisfied by two engines that were never asked to differ.
    ws.push_back({
        .name = "two_rules_overlap_rank_frame",
        .rules = {rule({{0,1}}, {{0,2},{2,1}}),
                  rule({{0,1}}, {{1,2},{2,0}})},
        .initial_state = V{{0u,1u}},
        .num_steps = 3,
    });

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

    // Multiple initial states: two structurally distinct roots evolved together.
    ws.push_back({
        .name = "multi_initial_two_roots",
        .rules = {rule({{0,1}}, {{0,2},{2,1}})},
        .initial_state = {},
        .initial_states = { V{{0u,1u}}, V{{2u,3u},{3u,4u}} },
        .num_steps = 3,
    });
    // Two isomorphic roots under quotient, DEFAULT (quotient_initial_states=false):
    // both engines keep every provided root as a distinct entry point (reference
    // MultiwaySystem semantics), so results agree.
    ws.push_back({
        .name = "multi_initial_iso_roots_kept",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = {},
        .initial_states = { V{{0u,1u},{0u,2u}}, V{{5u,6u},{5u,7u}} },
        .num_steps = 3,
        .explore_from_canonical_states_only = true,
    });
    // Same, but quotient_initial_states=true: isomorphic roots collapse to one on
    // both engines.
    ws.push_back({
        .name = "multi_initial_iso_roots_quotiented",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = {},
        .initial_states = { V{{0u,1u},{0u,2u}}, V{{5u,6u},{5u,7u}} },
        .num_steps = 3,
        .explore_from_canonical_states_only = true,
        .quotient_initial_states = true,
    });

    // Multi-initial x multi-rule under full multiway: the combined corner of the
    // single/multi initial x single/multi rule space, validated against the
    // reference oracle (2init x 2rule: states=28, eventsNone=144 at depth 3).
    ws.push_back({
        .name = "multi_init_multi_rule",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}}), rule({{0,1}}, {{1,0}})},
        .initial_state = {},
        .initial_states = { V{{0u,1u},{0u,2u}}, V{{0u,1u},{1u,2u}} },
        .num_steps = 3,
    });

    // Chunked launches: 3 blocks per match/rewrite launch forces the kernels to
    // run in many consecutive chunks, cross-checking the watchdog-bounding path
    // against the CPU (results must be identical to a single launch).
    ws.push_back({
        .name = "chunked_launch_wolfram_steps4",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = V{{0u,1u},{0u,2u}},
        .num_steps = 4,
        .max_blocks_per_launch = 3,
    });

    // Quotient exploration (explore_from_canonical_states_only): each canonical
    // state is expanded once, at its shortest depth. The CPU reaches that via
    // depth relaxation over its dataflow; the GPU's single-launch persistent loop
    // gives it by construction. Compared on canonical states and the step-less
    // transition multiset; causal/branchial are reconstructed offline in this
    // mode. The multi-rule workloads put loops in the multiway states graph,
    // which is where first-discovery ordering is most prone to diverge across
    // the two engines -- exactly what this comparison guards.
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
    // Depth 6 pushes hub-vertex inverted-index buckets past the match kernel's
    // 256-entry seen buffer, exercising the signature-walk overflow path.
    ws.push_back({
        .name = "quotient_wolfram_steps6",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = V{{0u,1u},{0u,2u}},
        .num_steps = 6,
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

    // Force the index-backed match regime on a small workload: threshold 2 means
    // the step-1 children (4 edges) exceed it, flipping lazy index maintenance to
    // a mid-run rebuild, and every later state matches through the indices.
    ws.push_back({
        .name = "index_regime_wolfram_steps5",
        .rules = {rule({{0,1},{0,2}}, {{0,1},{0,3},{1,3},{2,3}})},
        .initial_state = V{{0u,1u},{0u,2u}},
        .num_steps = 5,
        .slice_scan_max_edges = 2,
    });
    ws.push_back({
        .name = "index_regime_all_three_triangle",
        .rules = {rule({{0,1}}, {{0,2},{2,1}}),
                  rule({{0,1}}, {{1,0}}),
                  rule({{0,1},{1,2}}, {{0,2}})},
        .initial_state = V{{0u,1u},{1u,2u},{2u,0u}},
        .num_steps = 3,
        .explore_from_canonical_states_only = true,
        .slice_scan_max_edges = 2,
    });

    return ws;
}

// Every corpus workload against the CPU. The device has ONE scheduler, and the CPU -- not a
// second device path -- is its reference: two GPU paths sharing code can agree while both are
// wrong, which is why the authority for the device is the host engine and the oracle behind it.
std::vector<Workload> build_corpus_both_schedulers() {
    return build_corpus();
}

// Running build_corpus_both_schedulers() here is the next step and is NOT yet green: it exposes a
// persistent-path defect on index_regime_all_three_triangle (quotient + 3 rules), where the event
// multiset is 106 against the CPU's 75 while the state set matches. See the task notes for #70 for
// the reproducer. Wired but not instantiated, rather than instantiated with that case excluded,
// because an exclusion list is how a known failure becomes invisible.
INSTANTIATE_TEST_SUITE_P(InitialCorpus, DifferentialEvolution,
    ::testing::ValuesIn(build_corpus_both_schedulers()),
    [](const ::testing::TestParamInfo<Workload>& info) { return info.param.name; });

// The event-identity axis, compared the way a caller sees it.
//
// NumStates has been checked against the CPU per state mode since the marshalling was written.
// NumEvents never was, and the gap that hid was total: the device stamped event signatures and
// applied none of them, so its event count was the raw application count in every mode while the
// CPU returned the canonical count in two of the three. HGEvolve reported different numbers for
// the same question depending on which device answered.
//
// Compared per event mode. The raw application count must agree too -- if it does not, the two
// engines disagree about the evolution itself and any event-count comparison on top of that is
// measuring the wrong thing.
// A CANONICAL RANK MUST NOT DEPEND ON HOW THE STATE WAS PRESENTED, on either device.
//
// A rank is a position in a canonical LABELLING, and on a state with a nontrivial automorphism
// group that labelling is a COSET: interchangeable edges can take each other's positions. Which
// member an engine settles on is decided by its within-cell tie-break, and that tie-break reads
// the vertex numbering it was handed. So an engine that numbers vertices from the order the
// edges happened to arrive will return a DIFFERENT rank assignment for the same graph presented
// differently -- each internally consistent, and none of them an isomorphism invariant.
//
// That is exactly the defect #66 turned out to be: the device numbered vertices in encounter
// order where the host numbered them by sorted id, so the two picked different representatives
// and Automatic reported 19 against 15. Four other hypotheses were measured and refuted before
// that one, and this check would have named it immediately -- an equality between devices can be
// satisfied by two engines that are both presentation-dependent in the same way, whereas this
// cannot be satisfied by any engine that is presentation-dependent at all.
//
// Every presentation below is the same directed 4-cycle.
// THE DEVICE'S OWN CANONICAL HASH must equal the host's, not merely partition the states the
// same way.
//
// Everywhere else this harness recomputes each state's hash ON THE HOST from the edges the run
// returned. That checks the two engines explored the same STATES, and it is deliberately blind to
// what the device actually computed: a device whose canonical hash was wrong in a
// label-consistent way would still return the same edge sets and still pass.
//
// The device hash is not a diagnostic. It is what the device DEDUPLICATES on, and it is what a
// caller reads back from CanonicalState::canonical_hash, so two devices that disagree about it
// give different answers to "is this the same state as that one" across a session. Under Full
// canonicalization it is the exact IR hash on both sides and must agree VALUE for VALUE.
//
// A multiset, not a set: two states sharing a hash is the thing being asserted about, so
// collapsing duplicates would hide a device that merged a class the host split.
TEST(CanonicalHash, DeviceHashEqualsHostHash) {
    auto r = rule({{0, 1}, {1, 2}}, {{0, 1}, {1, 3}, {3, 2}});

    struct Case { const char* name; std::vector<std::vector<hg_gpu::VertexId>> init; uint32_t steps; };
    const std::vector<Case> cases = {
        {"path",              {{0,1},{1,2},{2,3}},          3},
        {"cycle4-automorphic",{{0,1},{1,2},{2,3},{3,0}},     3},
        {"star4-automorphic", {{0,1},{0,2},{0,3},{0,4}},     2},
    };

    for (const Case& c : cases) {
        for (bool quotient : {false, true}) {
            Workload w;
            w.name = std::string("hash/") + c.name + (quotient ? "/quotient" : "/full");
            w.rules = {r};
            w.initial_state = c.init;
            w.num_steps = c.steps;
            w.canon_mode = hg_gpu::CanonicalizationMode::Full;   // exact IR on both sides
            w.explore_from_canonical_states_only = quotient;

            const NormalizedResult cpu = run_cpu(w);
            const NormalizedResult gpu = run_gpu(w);

            ASSERT_FALSE(cpu.engine_state_hashes.empty()) << w.name;
            EXPECT_EQ(gpu.engine_state_hashes, cpu.engine_state_hashes)
                << w.name << ": the device's own canonical hashes differ from the host's. The "
                << "state SETS may still match -- this compares what each engine COMPUTED, which "
                << "is what it deduplicates on and what a caller reads back.";
        }
    }
}

TEST(CanonicalEventCount, RanksAreIndependentOfPresentation) {
    using EM = hg_gpu::EventCanonicalizationMode;
    auto r = rule({{0, 1}, {1, 2}}, {{0, 1}, {1, 3}, {3, 2}});

    struct Presentation {
        const char* name;
        std::vector<std::vector<hg_gpu::VertexId>> init;
    };
    const std::vector<Presentation> presentations = {
        {"as written",       {{0,1},{1,2},{2,3},{3,0}}},
        {"edges rotated",    {{1,2},{2,3},{3,0},{0,1}}},
        {"edges reversed",   {{3,0},{2,3},{1,2},{0,1}}},
        {"vertices +10",     {{10,11},{11,12},{12,13},{13,10}}},
        {"vertices relabel", {{7,3},{3,9},{9,5},{5,7}}},
    };

    size_t cpu_baseline = 0, gpu_baseline = 0;
    for (size_t i = 0; i < presentations.size(); ++i) {
        Workload w;
        w.name = std::string("presentation/") + presentations[i].name;
        w.rules = {r};
        w.initial_state = presentations[i].init;
        w.num_steps = 3;
        w.canon_mode = hg_gpu::CanonicalizationMode::Full;
        w.event_canon_mode = EM::Automatic;   // the only mode that keys on ranks

        const NormalizedResult cpu = run_cpu(w);
        const NormalizedResult gpu = run_gpu(w);

        if (i == 0) {
            cpu_baseline = cpu.num_events;
            gpu_baseline = gpu.num_events;
            ASSERT_GT(cpu_baseline, 0u);
        }
        EXPECT_EQ(cpu.num_events, cpu_baseline)
            << "CPU: presenting the same 4-cycle as \"" << presentations[i].name
            << "\" changed the Automatic event count from " << cpu_baseline << " to "
            << cpu.num_events << ", so its ranks follow the presentation and are not an "
            << "isomorphism invariant";
        EXPECT_EQ(gpu.num_events, gpu_baseline)
            << "GPU: presenting the same 4-cycle as \"" << presentations[i].name
            << "\" changed the Automatic event count from " << gpu_baseline << " to "
            << gpu.num_events << ", so its ranks follow the presentation";
        EXPECT_EQ(gpu.num_events, cpu.num_events)
            << "devices disagree on " << presentations[i].name;
    }
}

// PINNED REPRODUCER for the device's missing reconstruction. It passes exactly while the
// defect is present, so the suite notices if the gap silently moves or closes -- the same
// contract verification/genmc/run.sh gives a `GENMC-EXPECT: violation` harness.
//
// WHAT IS BROKEN. An Automatic event signature hashes each consumed/produced edge's RANK, and
// which state's canonical labelling supplies that rank is a choice. The CPU uses ONE labelling
// pinned per isomorphism class, served by quotient reconstruction; the GPU has no class frame
// and reads each state's own labelling (edge_rank_in_state_device). Measured on this workload:
//
//     hg.observable_num_events()  -- what HGEvolve returns --  CPU 21
//     the GPU's canonical event count                          GPU 23
//     hg.num_events()             -- what this harness reads -- CPU 23   <- why it passes
//
// The same gap under quotient exploration in mode None is larger: CPU 144, GPU 15, because the
// CPU serves the reconstructed RAW count and the device has no reconstruction at all.
//
// TO CLOSE: port the expansion capture and per-instance reconstruction to the device, then
// switch run_cpu to observable_num_events() and replace the two EXPECT_EQs below with an
// equality between the devices.
TEST(CanonicalEventCount, ReconstructionGapIsStillOpen) {
    using EM = hg_gpu::EventCanonicalizationMode;
    Workload w;
    w.name = "two_rules_overlap_automatic";
    w.rules = {rule({{0,1}}, {{0,2},{2,1}}), rule({{0,1}}, {{1,2},{2,0}})};
    w.initial_state = {{0u, 1u}};
    w.num_steps = 3;
    w.canon_mode = hg_gpu::CanonicalizationMode::Full;
    w.event_canon_mode = EM::Automatic;

    NormalizedResult cpu = run_cpu(w);
    NormalizedResult gpu = run_gpu(w);
    hypergraph::Hypergraph probe;
    probe.set_state_canonicalization_mode(hypergraph::StateCanonicalizationMode::Full);
    probe.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
    hypergraph::ParallelEvolutionEngine pe(&probe, 1);
    for (size_t i = 0; i < w.rules.size(); ++i)
        pe.add_rule(convert_rule(w.rules[i], static_cast<uint16_t>(i)));
    pe.evolve({{0u, 1u}}, w.num_steps);

    // The two engines agree on the EVOLUTION -- same applications -- and differ only on identity.
    EXPECT_EQ(gpu.raw_events, cpu.raw_events);

    // The shipped CPU answer, from the class-pinned frame.
    EXPECT_EQ(probe.observable_num_events(), 21u)
        << "the CPU's shipped count moved; re-derive the pinned numbers";
    // The device answer, from each state's own labelling. Equality with the line above is the
    // goal; this inequality is the defect, pinned so it cannot close unnoticed.
    EXPECT_EQ(gpu.num_events, 23u)
        << "the device's count moved; if it now reports 21 the port has landed -- replace this "
           "reproducer with an equality against observable_num_events()";
}

// P2.1: the device captures the SAME class-frame expansion the host does.
//
// The expansion is the input to the per-instance replay, so a device that captures a different
// number of frame matches cannot agree with the host on event identity however good the replay
// is. This gates the capture on its own, before any replay exists -- otherwise the first
// disagreement would surface as an event-count difference with two candidate causes.
//
// One record per match of each class's FRAME state, so the host total is the sum of
// for_each_expansion_match over the distinct canonical hashes it captured.
TEST(CanonicalEventCount, DeviceReplaysTheClassFrameExpansion) {
    // The two doors into the reconstruction: quotient exploration, and Automatic identity under
    // full capture. Both must reach the same population and mint the same raw events as the host.
    std::vector<Workload> ws;
    {
        Workload w;
        w.name = "two_rules_overlap_automatic";
        w.rules = {rule({{0,1}}, {{0,2},{2,1}}), rule({{0,1}}, {{1,2},{2,0}})};
        w.initial_state = {{0u, 1u}};
        w.num_steps = 3;
        w.event_canon_mode = hg_gpu::EventCanonicalizationMode::Automatic;
        ws.push_back(w);
    }
    {
        Workload w;
        w.name = "quotient_4cycle_none";
        w.rules = {rule({{0,1},{1,2}}, {{0,1},{1,3},{3,2}})};
        w.initial_state = {{0u,1u},{1u,2u},{2u,3u},{3u,0u}};
        w.num_steps = 3;
        w.explore_from_canonical_states_only = true;
        ws.push_back(w);
    }

    for (const Workload& w : ws) {
        hg_gpu::EvolveInput in = make_input(w);
        hg_gpu::EvolveResult gpu = hg_gpu::evolve(in);

        hypergraph::Hypergraph hg;
        hg.set_state_canonicalization_mode(hypergraph::StateCanonicalizationMode::Full);
        hg.set_event_signature_keys(
            w.event_canon_mode == hg_gpu::EventCanonicalizationMode::Automatic
                ? hgcommon::EVENT_SIG_AUTOMATIC : hgcommon::EVENT_SIG_NONE);
        hypergraph::ParallelEvolutionEngine pe(&hg, 1);
        pe.set_explore_from_canonical_states_only(w.explore_from_canonical_states_only);
        for (size_t i = 0; i < w.rules.size(); ++i)
            pe.add_rule(convert_rule(w.rules[i], static_cast<uint16_t>(i)));
        pe.evolve(w.initial_state, w.num_steps);

        size_t host_matches = 0;
        std::set<uint64_t> seen;
        for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
            const uint64_t h = hg.get_state(sid).canonical_hash;
            if (!seen.insert(h).second) continue;
            hg.for_each_expansion_match(h, [&](const hypergraph::SlotMatch&) { ++host_matches; });
        }
        const size_t host_raw = hg.num_reconstructed_raw_events();

        std::printf("%-28s frame matches %4zu  raw events %4zu\n", w.name.c_str(),
                    host_matches, host_raw);

        EXPECT_GT(host_matches, 0u) << w.name << ": the host captured no expansion at all -- "
                                       "every comparison below would pass on a device that "
                                       "captures nothing";
        EXPECT_GT(host_raw, 0u) << w.name << ": the host reconstructed no raw events";

        EXPECT_EQ(gpu.expansion_matches, host_matches)
            << w.name << ": device captured " << gpu.expansion_matches
            << " class-frame matches, host " << host_matches;
        EXPECT_EQ(gpu.reconstructed_raw_events, host_raw)
            << w.name << ": the replay minted " << gpu.reconstructed_raw_events
            << " raw events, host " << host_raw;
    }
}

TEST(CanonicalEventCount, ModesVsCpu) {
    using EM = hg_gpu::EventCanonicalizationMode;
    // A workload on which the modes actually SEPARATE. An edge-splitting rule on a path merges
    // nothing, so every mode returns the raw count and the comparison would pass on a device that
    // applies no identity at all -- which is the defect being gated. The directed 4-cycle under a
    // rule that reaches the same state by several applications does merge.
    auto r = rule({{0, 1}, {1, 2}}, {{0, 1}, {1, 3}, {3, 2}});
    const char* mn[] = {"None", "Automatic", "Full"};
    EM modes[] = {EM::None, EM::Automatic, EM::Full};
    const char* mode_names[] = {"None", "Automatic", "Full"};
    size_t merged_somewhere = 0;

    // Run under full capture AND under quotient exploration. Under full capture a canonical
    // class holds many raw states, each ranked from its own presentation; under quotient there
    // is one. If the device's Automatic over-count is raw states of one class disagreeing about
    // ranks, it appears in the first and not the second -- which is why both are here rather
    // than only the default.
    std::printf("\n%-10s %-8s | cpu: events raw | gpu: events raw\n", "event mode", "explore");
    for (bool quotient : {false, true})
    for (int mi = 0; mi < 3; ++mi) {
        Workload w;
        w.name = std::string("events_") + mn[mi] + (quotient ? "_quotient" : "_full");
        w.rules = {r};
        w.initial_state = {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}};
        w.num_steps = 3;
        w.canon_mode = hg_gpu::CanonicalizationMode::Full;
        w.event_canon_mode = modes[mi];
        w.explore_from_canonical_states_only = quotient;

        NormalizedResult cpu = run_cpu(w);
        NormalizedResult gpu = run_gpu(w);
        std::printf("%-10s %-8s |      %5zu %4zu |      %5zu %4zu%s\n",
                    mn[mi], quotient ? "quotient" : "full",
                    cpu.num_events, cpu.raw_events, gpu.num_events, gpu.raw_events,
                    gpu.num_events != cpu.num_events ? "   <-- differs" : "");

        EXPECT_EQ(gpu.raw_events, cpu.raw_events)
            << "raw application count differs for " << w.name
            << ", so the two engines disagree about the evolution and not merely about identity";
        // ALL THREE identity modes must agree exactly, Automatic included.
        //
        // Automatic keys on canonical edge RANKS, and a rank is a position in a canonical
        // LABELLING. On this workload -- a directed 4-cycle, rotation group of order 4 -- that
        // labelling is a COSET, so which member an engine settles on is decided by its
        // within-cell tie-break, which reads the vertex numbering it was handed. The device
        // numbered vertices in encounter order while the host numbered them by sorted id, the
        // two picked different coset representatives, and the GPU reported 19 against the CPU's
        // 15 (f72ed00). Both flatteners now number by sorted id.
        //
        // So this is an equality and not a printed number. A rank-keyed identity that agrees
        // across devices is the whole claim; if it regresses, the coset is being chosen by
        // presentation again.
        EXPECT_EQ(gpu.num_events, cpu.num_events)
            << "NumEvents mismatch for " << w.name << " in mode " << mode_names[mi]
            << ": the CPU reports " << cpu.num_events << " and the GPU " << gpu.num_events;
        if (cpu.num_events < cpu.raw_events) ++merged_somewhere;
    }

    // Without this the three equalities above are satisfied by two devices that both merge
    // nothing, which is precisely the state this gate exists to detect.
    EXPECT_GT(merged_somewhere, 0u)
        << "no event mode merged anything on this workload, so the comparison is vacuous";
}

// Diagnostic: print the count conventions across modes and root counts so the GPU marshalling's
// NumStates (= the CPU's num_canonical_states()) can be reproduced exactly, not reverse-engineered.
TEST(CanonicalStateCount, ModesVsCpu) {
    using M = hg_gpu::CanonicalizationMode;
    auto r = rule({{0, 1}}, {{0, 2}, {2, 1}});   // binary edge splitting
    std::vector<std::vector<std::vector<hg_gpu::VertexId>>> root1 = {{{0u, 1u}}};
    std::vector<std::vector<std::vector<hg_gpu::VertexId>>> root2 = {{{0u, 1u}}, {{0u, 1u}, {1u, 2u}}};
    std::vector<std::vector<std::vector<hg_gpu::VertexId>>> root3 =
        {{{0u, 1u}}, {{0u, 1u}, {1u, 2u}}, {{0u, 1u}, {1u, 2u}, {2u, 3u}}};
    struct C { const char* name; std::vector<std::vector<std::vector<hg_gpu::VertexId>>> roots; uint32_t steps; };
    std::vector<C> cases = {{"1root", root1, 4}, {"2root", root2, 3}, {"3root", root3, 3}};
    const char* mn[] = {"None", "Automatic", "Full"};
    M modes[] = {M::None, M::Automatic, M::Full};
    std::printf("\n%-6s %-10s | cpu: canon raw prod | gpu: raw prod content ir\n", "roots", "mode");
    for (auto& c : cases) for (int mi = 0; mi < 3; ++mi) {
        Workload w;
        w.name = std::string(c.name) + "_" + mn[mi];
        w.rules = {r};
        w.initial_states = c.roots;
        w.num_steps = c.steps;
        w.canon_mode = modes[mi];
        NormalizedResult cpu = run_cpu(w);
        NormalizedResult gpu = run_gpu(w);
        std::printf("%-6s %-10s |     %4zu %4zu %4zu | %4zu %4zu %4zu %4zu\n",
            c.name, mn[mi],
            cpu.num_canonical_states, cpu.raw_states, cpu.produced_states,
            gpu.raw_states, gpu.produced_states, gpu.distinct_content,
            gpu.canonical_state_hashes.size());

        // NumStates the GPU marshalling (hg_gpu_backend.cpp) reports, by the same per-mode rule:
        //   None -> raw - 1, Automatic -> distinct content, Full -> distinct IR class.
        size_t gpu_num_states =
            modes[mi] == M::None      ? gpu.raw_states :
            modes[mi] == M::Automatic ? gpu.distinct_content
                                      : gpu.canonical_state_hashes.size();
        EXPECT_EQ(gpu_num_states, cpu.num_canonical_states)
            << "NumStates mismatch for " << w.name;
        // Correctness invariant for the raw modes: the two engines produce the same states up to
        // isomorphism, WITH the same multiplicity (strictly stronger than the set comparison in
        // DifferentialEvolution). Byte-equivalence of the raw labels does NOT hold and is not
        // expected — None/Automatic keep each engine's own vertex numbering, which is assignment-
        // order dependent; only Full canonicalises labels. Report byte-equivalence as a diagnostic.
        if (modes[mi] != M::Full) {
            EXPECT_EQ(cpu.iso_multiset, gpu.iso_multiset)
                << "iso multiset (states up to isomorphism, with multiplicity) differ for " << w.name;
            std::printf("       byte-equiv(%-9s): %s\n", mn[mi],
                cpu.content_multiset == gpu.content_multiset
                    ? "yes" : "no (iso-equivalent; raw vertex labels differ by engine)");
        }
    }
}

}  // namespace
