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

#include "hgcommon/quotient_replay_core.hpp"
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"

#include <algorithm>
#include <iterator>
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
    // The RECONSTRUCTED relations, when the run routes the reconstruction. Kept apart from the
    // full-capture sets above because the endpoint identity differs: these are pairs of content
    // triples hash(input class, output class, rule), which is what both engines can produce for
    // an event that was never materialised.
    std::multiset<uint64_t> recon_causal, recon_branchial;
    // The TR view of the same relation: the pairs tagged in-reduction.
    std::multiset<uint64_t> recon_causal_reduced;
    // The same relations' ENDPOINTS, flattened, before they are combined into a pair key.
    // Separates two failures a pair-key comparison cannot: endpoints that disagree (the two
    // engines identify the reconstructed events differently) from endpoints that agree while the
    // pairs do not (they identify the same events and relate them differently). The pair key
    // hashes both into one number, so a mismatch there says only that something differs.
    std::multiset<uint64_t> recon_causal_endpoints, recon_branchial_endpoints;
    // The COUNTS HGEvolve returns for the two relations, from each engine's observable_*
    // accessor -- the same one its FFI reads, so the gated number is the shipped number.
    size_t observable_causal = 0, observable_branchial = 0;

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
    for (uint32_t eid = 0; eid < hg.num_published_events(); ++eid) {
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
    // The count the FFI serves (hypergraph_ffi.cpp:1305): under quotient exploration or
    // EVENT_SIG_AUTOMATIC it is the RECONSTRUCTION's, which is a different number from the
    // full-capture count and is what a caller of HGEvolve is told.
    out.num_events = hg.observable_num_events();
    out.raw_events = hg.num_raw_events();
    {
        std::set<uint32_t> outs;
        for (uint32_t eid = 0; eid < hg.num_published_events(); ++eid) {
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

    // The relations the run SERVES when it routes the reconstruction. The sets above are full
    // capture's, which on that route holds only what the explored representatives left behind.
    out.observable_causal =
        hg.observable_num_causal_pairs(hg.causal_graph().transitive_reduction_enabled());
    out.observable_branchial = hg.observable_num_branchial();

    if (hg.quotient_reconstruction()) {
        auto triple = [&](uint32_t e) { return hg.reconstructed_raw_triple(e); };
        hg.for_each_reconstructed_causal_as(/*reduced=*/false, triple,
            [&](uint64_t p, uint64_t c) {
                out.recon_causal.insert(causal_key(p, c));
                out.recon_causal_endpoints.insert(p);
                out.recon_causal_endpoints.insert(c);
            });
        hg.for_each_reconstructed_causal_as(/*reduced=*/true, triple,
            [&](uint64_t p, uint64_t c) { out.recon_causal_reduced.insert(causal_key(p, c)); });
        hg.for_each_reconstructed_branchial_as(triple,
            [&](uint64_t a, uint64_t b) {
                out.recon_branchial.insert(edge_pair_key(a, b));
                out.recon_branchial_endpoints.insert(a);
                out.recon_branchial_endpoints.insert(b);
            });
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
    // This suite COMPARES the relations pair by pair, so it needs them built. A caller that
    // reads only the counts leaves this off and does not pay for the expansion.
    in.materialize_relations  = true;
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

    for (const auto& p : result.reconstructed_causal_relation) {
        out.recon_causal.insert(causal_key(p.first, p.second));
        out.recon_causal_endpoints.insert(p.first);
        out.recon_causal_endpoints.insert(p.second);
    }
    out.observable_causal = result.observable_num_causal_pairs(w.transitive_reduction);
    out.observable_branchial = result.observable_num_branchial();

    for (const auto& p : result.reconstructed_causal_relation_reduced)
        out.recon_causal_reduced.insert(causal_key(p.first, p.second));
    for (const auto& p : result.reconstructed_branchial_relation) {
        out.recon_branchial.insert(edge_pair_key(p.first, p.second));
        out.recon_branchial_endpoints.insert(p.first);
        out.recon_branchial_endpoints.insert(p.second);
    }

    // Count diagnostics (GPU side).
    out.raw_states = result.states.size();
    out.raw_events = result.events.size();
    // The same accessor hg_gpu_backend.cpp serves as NumEvents, so the harness cannot compare a
    // number no caller receives.
    out.num_events = result.observable_num_events();
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
    // Whether this workload routes the reconstruction at all -- the same condition both
    // engines apply (Full states, and either quotient exploration or Automatic identity).
    // Stated here so the equality below cannot pass by both engines reconstructing nothing.
    const bool routes_reconstruction =
        w.canon_mode == hg_gpu::CanonicalizationMode::Full && w.num_steps > 0 &&
        (w.explore_from_canonical_states_only ||
         w.event_canon_mode == hg_gpu::EventCanonicalizationMode::Automatic);
    if (routes_reconstruction) {
        EXPECT_FALSE(cpu.recon_causal.empty())
            << "Workload: " << w.name << " routes the reconstruction but the host reconstructed "
            << "no causal relation, so the equality below constrains nothing";
        std::printf("[recon %s] causal cpu=%zu gpu=%zu (reduced cpu=%zu)  branchial cpu=%zu"
                    " gpu=%zu\n", w.name.c_str(), cpu.recon_causal.size(),
                    gpu.recon_causal.size(), cpu.recon_causal_reduced.size(),
                    cpu.recon_branchial.size(), gpu.recon_branchial.size());
    }

    // The RECONSTRUCTED relations, compared as SETS. Both engines reconstruct them online from
    // the class-frame expansion, and both report endpoints as the schedule-stable content
    // triple, so this is a cross-engine invariant wherever the route is taken.
    // Reported BEFORE the pair comparisons, because it says which of two very different faults
    // the pair mismatch below is. A pair key hashes both endpoints into one number, so a
    // difference there is silent about whether the two engines identified the reconstructed
    // events differently or identified them identically and related them differently.
    if (cpu.recon_causal != gpu.recon_causal ||
        cpu.recon_branchial != gpu.recon_branchial) {
        const bool causal_ends_agree = cpu.recon_causal_endpoints == gpu.recon_causal_endpoints;
        const bool branchial_ends_agree =
            cpu.recon_branchial_endpoints == gpu.recon_branchial_endpoints;
        // How MANY distinct identities the two share. Zero means the disagreement is systematic
        // -- every reconstructed event is identified differently, so some component of the
        // content triple is computed from a different quantity on one side. A partial overlap
        // would mean the opposite: the rule is shared and some particular states break it.
        std::set<uint64_t> ca(cpu.recon_causal_endpoints.begin(),
                              cpu.recon_causal_endpoints.end());
        std::set<uint64_t> ga(gpu.recon_causal_endpoints.begin(),
                              gpu.recon_causal_endpoints.end());
        std::vector<uint64_t> shared;
        std::set_intersection(ca.begin(), ca.end(), ga.begin(), ga.end(),
                              std::back_inserter(shared));
        std::printf("[recon-endpoints %s] causal endpoints %s (cpu=%zu gpu=%zu), "
                    "branchial endpoints %s (cpu=%zu gpu=%zu); DISTINCT causal identities "
                    "cpu=%zu gpu=%zu shared=%zu\n",
                    w.name.c_str(),
                    causal_ends_agree ? "AGREE -> the pairing differs, not the identities"
                                      : "DIFFER -> the two engines identify events differently",
                    cpu.recon_causal_endpoints.size(), gpu.recon_causal_endpoints.size(),
                    branchial_ends_agree ? "AGREE -> the pairing differs, not the identities"
                                         : "DIFFER -> the two engines identify events differently",
                    cpu.recon_branchial_endpoints.size(), gpu.recon_branchial_endpoints.size(),
                    ca.size(), ga.size(), shared.size());
        // Resolve each side's smallest identity back to the triple that made it. Both engines
        // agree on canonical_state_hashes (that assertion passes), and qr_content_hash is the
        // shared mixing, so a search over that set x the rule indices names WHICH component of
        // (from_class, to_class, rule) the two disagree on -- which is the thing a hash
        // comparison cannot say.
        // Every class hash either engine could plausibly have mixed. The harness-recomputed set
        // is NOT sufficient on its own: it is a fresh IR pass over the returned edges, whereas
        // the reconstruction mixes what the ENGINE computed and stored. Zero is admitted too,
        // because DeviceState::state_canonical_hash is documented "0 until computed" and a
        // capture racing that fill would mix a class no canonical hash ever takes.
        std::set<uint64_t> cand;
        cand.insert(0ULL);
        cand.insert(cpu.canonical_state_hashes.begin(), cpu.canonical_state_hashes.end());
        cand.insert(cpu.engine_state_hashes.begin(), cpu.engine_state_hashes.end());
        cand.insert(gpu.engine_state_hashes.begin(), gpu.engine_state_hashes.end());

        auto resolve = [&](uint64_t want, const char* side) {
            if (cand.size() > 96) return;                         // keep the search trivial
            for (uint64_t a : cand)
                for (uint64_t b : cand)
                    for (uint32_t r = 0; r < 16; ++r)
                        if (hgcommon::qr_content_hash(a, b, r) == want) {
                            std::printf("[recon-triple %s] %s identity %llu = "
                                        "(from=%llu, to=%llu, rule=%u)\n",
                                        w.name.c_str(), side, (unsigned long long)want,
                                        (unsigned long long)a, (unsigned long long)b, r);
                            return;
                        }
            std::printf("[recon-triple %s] %s identity %llu is NOT any (class, class, rule<16) "
                        "over the %zu candidate class hashes (harness-recomputed + BOTH engines' "
                        "own + zero) -- that side mixes something this run never called a class\n",
                        w.name.c_str(), side, (unsigned long long)want, cand.size());
        };
        if (!ca.empty()) resolve(*ca.begin(), "cpu");
        if (!ga.empty()) resolve(*ga.begin(), "gpu");

        // The device's OWN canonical hashes against the host's. canonical_state_hashes above
        // cannot answer this: the harness recomputes it on the host from each engine's returned
        // edges, so it agrees whenever the state SETS agree, however each engine labelled them.
        // engine_state_hashes is what each engine actually COMPUTED -- and it is what the
        // reconstruction mixes into the content triple, so if these differ the identities must.
        std::printf("[engine-hashes %s] device's own canonical hashes %s the host's "
                    "(cpu=%zu gpu=%zu distinct); harness-recomputed sets %s\n",
                    w.name.c_str(),
                    cpu.engine_state_hashes == gpu.engine_state_hashes ? "MATCH" : "DIFFER",
                    cpu.engine_state_hashes.size(), gpu.engine_state_hashes.size(),
                    cpu.canonical_state_hashes == gpu.canonical_state_hashes ? "match"
                                                                             : "differ");
    }

    EXPECT_EQ(cpu.recon_causal, gpu.recon_causal)
        << "Workload: " << w.name << " reconstructed causal relations differ; cpu="
        << cpu.recon_causal.size() << " gpu=" << gpu.recon_causal.size();
    // The numbers HGEvolve returns, on either route.
    EXPECT_EQ(gpu.observable_causal, cpu.observable_causal)
        << "Workload: " << w.name << " NumCausalEdges differs; cpu=" << cpu.observable_causal
        << " gpu=" << gpu.observable_causal;
    EXPECT_EQ(gpu.observable_branchial, cpu.observable_branchial)
        << "Workload: " << w.name << " NumBranchialEdges differs; cpu="
        << cpu.observable_branchial << " gpu=" << gpu.observable_branchial;

    EXPECT_EQ(cpu.recon_causal_reduced, gpu.recon_causal_reduced)
        << "Workload: " << w.name << " reconstructed REDUCED causal relations differ; cpu="
        << cpu.recon_causal_reduced.size() << " gpu=" << gpu.recon_causal_reduced.size();
    EXPECT_EQ(cpu.recon_branchial, gpu.recon_branchial)
        << "Workload: " << w.name << " reconstructed branchial relations differ; cpu="
        << cpu.recon_branchial.size() << " gpu=" << gpu.recon_branchial.size();

    // Full capture's own causal/branchial records. Under quotient exploration these hold only
    // what the explored representatives left behind, and which raw representative carries them
    // is a claim race, so they are compared on the full-capture route alone -- the relation a
    // caller receives there is the reconstruction, compared above.
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

    // A rule that CONSUMES its match and produces nothing, so the evolution reaches the empty
    // state. The empty state's canonical hash is a value both devices must agree on: it is what
    // dedup keys on and what IncludeCanonicalHashes returns, and no other workload in this
    // corpus reaches it.
    ws.push_back({
        .name = "emptying_rule",
        .rules = {rule({{0, 1}}, {})},
        .initial_state = V{{0u, 1u}, {1u, 2u}},
        .num_steps = 3,
    });

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
    // A DEEP CONE, which is where the two reachability walks can differ.
    //
    // The reduction asks "is this pair already bypassed" by walking backward over the KEPT
    // predecessors. A rule that consumes two edges gives every reconstructed event two
    // predecessors, so at depth d the cone holds O(d) nodes and 2^d PATHS through them. A walk
    // that dedups visits the nodes; one that does not visits the paths. Both answer the same
    // question, so this workload does not separate them by RESULT -- it is here because it is
    // the only shape in the corpus that reaches the regime at all, and because the reduced sets
    // must agree in it.
    //
    // The rule is the disconnected-LHS one: it matches any two edges, so the class has two
    // matches and the instance count doubles per depth, which is what builds the cone.
    ws.push_back({
        .name = "deep_cone_reduction_d6",
        .rules = {rule({{0, 1}, {2, 3}}, {{0, 2}, {1, 3}})},
        .initial_state = V{{0u, 1u}, {2u, 3u}},
        .num_steps = 6,
        .explore_from_canonical_states_only = true,
    });

    ws.push_back({
        .name = "deep_cone_reduction_d13",
        .rules = {rule({{0, 1}, {2, 3}}, {{0, 2}, {1, 3}})},
        .initial_state = V{{0u, 1u}, {2u, 3u}},
        .num_steps = 13,
        .explore_from_canonical_states_only = true,
    });

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
    // `p`, not `info`: INSTANTIATE_TEST_SUITE_P expands to a body that already binds `info`, so
    // the obvious name shadows it and -Wshadow says so on every expansion of this macro.
    [](const ::testing::TestParamInfo<Workload>& p) { return p.param.name; });

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
    // A rule that consumes its match and produces nothing, so the evolution reaches the EMPTY
    // state. The empty state has no edges for a canonicalizer to work on, so its hash is a
    // reserved value rather than a computed one, and the two engines must reserve the same one.
    auto empt = rule({{0, 1}}, {});

    struct Case {
        const char* name;
        std::vector<std::vector<hg_gpu::VertexId>> init;
        uint32_t steps;
        bool emptying = false;
    };
    const std::vector<Case> cases = {
        {"path",              {{0,1},{1,2},{2,3}},          3},
        {"cycle4-automorphic",{{0,1},{1,2},{2,3},{3,0}},     3},
        {"star4-automorphic", {{0,1},{0,2},{0,3},{0,4}},     2},
        {"empties",           {{0,1},{1,2}},                3, true},
    };

    for (const Case& c : cases) {
        for (bool quotient : {false, true}) {
            Workload w;
            w.name = std::string("hash/") + c.name + (quotient ? "/quotient" : "/full");
            w.rules = {c.emptying ? empt : r};
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

// The two devices agree on the number HGEvolve returns, on the workload where the identity
// question has teeth.
//
// An Automatic event signature hashes each consumed/produced edge's RANK, and which state's
// canonical labelling supplies that rank is a choice. Both engines now answer from ONE labelling
// pinned per isomorphism class -- the class frame -- rather than from whichever raw state of the
// class a schedule happened to rank first. Reading each state's own labelling gives 23 here
// where the class frame gives 21, so this workload separates the two conventions and an
// equality on it is a real constraint.
TEST(CanonicalEventCount, BothDevicesServeTheReconstructedCount) {
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

    // Same applications: the engines agree on the EVOLUTION before any identity is applied.
    EXPECT_EQ(gpu.raw_events, cpu.raw_events);

    // 21, not 23. Stated as a literal as well as an equality: two engines that both regressed to
    // per-state ranks would agree with each other and say nothing.
    EXPECT_EQ(cpu.num_events, 21u)
        << "the CPU's shipped count moved; re-derive the pinned number";
    EXPECT_EQ(gpu.num_events, cpu.num_events)
        << "device serves " << gpu.num_events << " events, host " << cpu.num_events;
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

    // At least one workload must have redundancy for the reduction to remove; on a relation
    // that is already its own reduction the equality above is satisfied by an engine that does
    // not reduce at all.
    bool reduction_bites = false;

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
        for (uint32_t sid = 0; sid < hg.num_published_states(); ++sid) {
            const uint64_t h = hg.get_state(sid).canonical_hash;
            if (!seen.insert(h).second) continue;
            hg.for_each_expansion_match(h, [&](const hypergraph::SlotMatch&) { ++host_matches; });
        }
        const size_t host_raw = hg.num_reconstructed_raw_events();
        const size_t host_ids = hg.num_reconstructed_events();
        const size_t host_cp = hg.num_reconstructed_causal_pairs(false);
        const size_t host_ce = hg.num_reconstructed_causal_edges();
        const size_t host_br = hg.num_reconstructed_branchial();
        const size_t host_cr = hg.num_reconstructed_causal_pairs(true);

        std::printf("%-28s matches %3zu  raw %4zu  ids %4zu  causal %4zu/%4zu  branchial %5zu"
                    "  reduced %4zu  moved %3u  align-fail %u\n", w.name.c_str(), host_matches,
                    host_raw, host_ids, host_cp, host_ce, host_br, host_cr,
                    gpu.frame_alignments, gpu.frame_align_failures);

        EXPECT_GT(host_matches, 0u) << w.name << ": the host captured no expansion at all -- "
                                       "every comparison below would pass on a device that "
                                       "captures nothing";
        EXPECT_GT(host_raw, 0u) << w.name << ": the host reconstructed no raw events";
        EXPECT_GT(host_cp, 0u) << w.name << ": the host reconstructed no causal pairs, so the "
                                  "equality below holds for both engines doing nothing";
        EXPECT_GT(host_br, 0u) << w.name << ": the host reconstructed no branchial pairs, so "
                                  "the equality below holds for both engines doing nothing";

        EXPECT_EQ(gpu.expansion_matches, host_matches)
            << w.name << ": device captured " << gpu.expansion_matches
            << " class-frame matches, host " << host_matches;
        EXPECT_EQ(gpu.reconstructed_raw_events, host_raw)
            << w.name << ": the replay minted " << gpu.reconstructed_raw_events
            << " raw events, host " << host_raw;
        // Under EVENT_SIG_NONE every application is its own event, so the reconstruction
        // reports no identity count and num_reconstructed_events returns the raw count.
        const uint32_t gpu_ids = w.event_canon_mode == hg_gpu::EventCanonicalizationMode::None
                                     ? gpu.reconstructed_raw_events : gpu.reconstructed_events;
        EXPECT_EQ(gpu_ids, host_ids)
            << w.name << ": the replay carries " << gpu_ids << " distinct event identities, "
            << "host " << host_ids;

        EXPECT_EQ(gpu.reconstructed_causal_pairs, host_cp)
            << w.name << ": the replay recorded " << gpu.reconstructed_causal_pairs
            << " distinct causal pairs, host " << host_cp;
        EXPECT_EQ(gpu.reconstructed_causal_edges, host_ce)
            << w.name << ": the replay recorded " << gpu.reconstructed_causal_edges
            << " causal edge occurrences, host " << host_ce;

        EXPECT_EQ(gpu.reconstructed_causal_pairs_reduced, host_cr)
            << w.name << ": the replay tagged " << gpu.reconstructed_causal_pairs_reduced
            << " pairs in-reduction, host " << host_cr;
        if (host_cr < host_cp) reduction_bites = true;

        EXPECT_EQ(gpu.reconstructed_branchial, host_br)
            << w.name << ": the replay recorded " << gpu.reconstructed_branchial
            << " branchial pairs, host " << host_br;

        EXPECT_EQ(gpu.frame_align_failures, 0u)
            << w.name << ": " << gpu.frame_align_failures << " slots had no image in their "
            << "class's frame, so their captures were dropped";
    }

    EXPECT_TRUE(reduction_bites)
        << "no workload here has a causal relation with redundancy in it, so the reduced-pair "
           "equality is satisfied by an engine that never reduces";
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
// THE EDGE IDENTITY THE READBACK CARRIES OUT MUST DESCRIBE THE STATES IT CARRIES OUT.
//
// all_state_edges_host() builds each state's edge CONTENTS by mapping that state's edge id list
// through the global edge table, and returns those two alongside the contents when asked. The
// three are therefore redundant by construction, which is exactly what makes the check sharp: if
// the ids came back in a different order from the contents built out of them, or an entry were
// dropped, then "StateBitvectors" would name edges belonging to another state and "GlobalEdges"
// would give them the wrong vertices -- and NOTHING else in the result would move, because every
// other consumer reads the contents.
TEST(EdgeIdentity, TheIdsDescribeTheContents) {
    Workload w;
    w.name = "edge_identity";
    w.rules = {rule({{0, 1}}, {{0, 2}, {2, 1}})};
    w.initial_states = {{{0u, 1u}}};
    w.num_steps = 4;
    w.canon_mode = hg_gpu::CanonicalizationMode::Full;

    hg_gpu::EvolveInput in = make_input(w);
    in.edge_identity = true;
    auto result = hg_gpu::evolve(in);

    ASSERT_FALSE(result.states.empty());
    ASSERT_EQ(result.state_edge_ids.size(), result.states.size())
        << "one id list per state, indexed the same way";
    ASSERT_FALSE(result.global_edges.empty());

    size_t checked = 0;
    for (size_t s = 0; s < result.states.size(); ++s) {
        const auto& contents = result.states[s].edges;
        const auto& ids = result.state_edge_ids[s];
        ASSERT_EQ(ids.size(), contents.size())
            << "state " << s << ": " << ids.size() << " edge ids for "
            << contents.size() << " edges";
        for (size_t k = 0; k < ids.size(); ++k) {
            ASSERT_LT(static_cast<size_t>(ids[k]), result.global_edges.size())
                << "state " << s << " slot " << k << ": edge id past the global table";
            EXPECT_EQ(result.global_edges[ids[k]], contents[k])
                << "state " << s << " slot " << k
                << ": the id list and the contents built from it disagree";
            ++checked;
        }
    }
    EXPECT_GT(checked, 0u) << "the workload produced no edges to check";
}

// And a run that does not ask for it does not carry it. The two arrays roughly double what the
// result holds about edges, and only "GlobalEdges" and "StateBitvectors" ask.
TEST(EdgeIdentity, AbsentUnlessAskedFor) {
    Workload w;
    w.name = "edge_identity_off";
    w.rules = {rule({{0, 1}}, {{0, 2}, {2, 1}})};
    w.initial_states = {{{0u, 1u}}};
    w.num_steps = 4;
    w.canon_mode = hg_gpu::CanonicalizationMode::Full;

    hg_gpu::EvolveInput in = make_input(w);   // edge_identity defaults false
    auto result = hg_gpu::evolve(in);

    ASSERT_FALSE(result.states.empty()) << "the run must still produce states";
    EXPECT_TRUE(result.state_edge_ids.empty());
    EXPECT_TRUE(result.global_edges.empty());
}

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

// The device records what it was asked for, and nothing else moves.
//
// EvolveInput::record mirrors the host's RecordSet. An artifact turned off must vanish from the
// result and leave every other artifact exactly as it was -- compared as SETS, since two runs
// can agree on how many pairs there are and disagree about which. Both routes are covered: the
// full-capture rendezvous in rewrite.cu and the reconstruction's replay in qe_apply.
// Not recording the raw unfolding must remove WORK and nothing else.
//
// The device's expansion subsystem does two jobs: it CAPTURES each class's frame and its matches
// in frame slots, which is what Automatic event identity is signed from, and it REPLAYS that
// capture against instances to recover the raw event set. Only the replay is exponential in
// depth, and only the replay is gated by the record set. Capture must therefore be identical
// whether or not the caller asked for raw events -- so the canonical states a run reports, and
// their canonical hashes, must not move.
//
// This is the configuration the gate exists for and the one the default record set never
// exercises: every other test here runs with raw_events on.
TEST(RecordSet, NotRecordingRawEventsLeavesTheCanonicalAnswerUnchanged) {
    // An automorphic initial state, where the class frame does real work and a mistake in the
    // capture/replay split would show up as a different canonical answer rather than as a count.
    Workload w;
    w.name = "record/raw-events-off";
    w.rules = {rule({{0, 1}, {1, 2}}, {{0, 1}, {1, 3}, {3, 2}})};
    w.initial_state = {{0u,1u},{1u,2u},{2u,3u},{3u,0u}};
    w.num_steps = 3;
    w.canon_mode = hg_gpu::CanonicalizationMode::Full;
    w.explore_from_canonical_states_only = true;   // the route that has a replay at all

    auto run = [&](bool raw) {
        hg_gpu::EvolveInput in = make_input(w);
        in.record.causal = in.record.branchial = in.record.raw_events = raw;
        return hg_gpu::evolve(in);
    };

    const auto with_raw = run(true);
    const auto no_raw   = run(false);

    // Canonical states by CONTENT. Device state ids are arrival-ordered and differ between two
    // runs of the same request, so an id-keyed comparison would call every run a change.
    auto state_hashes = [](const hg_gpu::EvolveResult& x) {
        hypergraph::IRCanonicalizer ir;
        std::multiset<uint64_t> h;
        for (const auto& st : x.states) h.insert(ir.compute_canonical_hash(st.edges));
        return h;
    };

    EXPECT_EQ(state_hashes(with_raw), state_hashes(no_raw))
        << "gating the replay changed the canonical states, so it removed more than work";
    EXPECT_EQ(with_raw.states.size(), no_raw.states.size());

    // The replay is what the flag governs: it runs in one arm and not the other. If both report
    // the same thing here the flag is not reaching the device and the test above proves nothing.
    EXPECT_TRUE(with_raw.reconstruction_ran);
    EXPECT_FALSE(no_raw.reconstruction_ran)
        << "raw_events=false still ran the reconstruction: the gate is not wired";
}

TEST(RecordSet, DeviceSkipsOnlyWhatItWasNotAskedFor) {
    struct Case { const char* name; bool quotient; };
    const Case cases[] = {{"full-capture", false}, {"reconstruction", true}};

    auto r = rule({{0, 1}, {1, 2}}, {{0, 1}, {1, 3}, {3, 2}});

    for (const Case& c : cases) {
        Workload w;
        w.name = std::string("record/") + c.name;
        w.rules = {r};
        w.initial_state = {{0u,1u},{1u,2u},{2u,3u},{3u,0u}};
        w.num_steps = 3;
        w.canon_mode = hg_gpu::CanonicalizationMode::Full;
        w.explore_from_canonical_states_only = c.quotient;

        auto run = [&](hgcommon::RecordSet rs) {
            hg_gpu::EvolveInput in = make_input(w);
            in.record = rs;
            return hg_gpu::evolve(in);
        };

        const auto all  = run(hgcommon::RecordSet{true, true, true});
        const auto no_c = run(hgcommon::RecordSet{false, true, true});
        const auto no_b = run(hgcommon::RecordSet{true, false, true});

        // Endpoints by CONTENT, never by device event id: ids are handed out in arrival
        // order, so two runs of the same request already disagree on them and an id-keyed
        // comparison would report every run as a change.
        auto causal_set = [](const hg_gpu::EvolveResult& x) {
            std::multiset<uint64_t> s;
            if (x.reconstruction_ran) {
                for (const auto& p : x.reconstructed_causal_relation)
                    s.insert(causal_key(p.first, p.second));   // already content triples
                return s;
            }
            hypergraph::IRCanonicalizer ir;
            std::unordered_map<uint32_t, uint64_t> state_hash;
            for (const auto& st : x.states) state_hash[st.id] = ir.compute_canonical_hash(st.edges);
            std::unordered_map<uint32_t, uint64_t> ekey;
            for (const auto& ev : x.events) {
                const uint64_t ih = state_hash.count(ev.input_state) ? state_hash[ev.input_state] : 0ull;
                const uint64_t oh = state_hash.count(ev.output_state) ? state_hash[ev.output_state] : 0ull;
                ekey[ev.id] = event_key(ih, oh, ev.rule, ev.step);
            }
            for (const auto& e : x.causal_edges) {
                if (!ekey.count(e.from) || !ekey.count(e.to)) continue;
                s.insert(causal_key(ekey[e.from], ekey[e.to]));
            }
            return s;
        };
        auto branchial_size = [](const hg_gpu::EvolveResult& x) {
            return x.reconstruction_ran ? x.reconstructed_branchial_relation.size()
                                        : x.branchial_edges.size();
        };

        ASSERT_FALSE(causal_set(all).empty()) << w.name << ": nothing to drop";
        ASSERT_GT(branchial_size(all), 0u) << w.name << ": nothing to drop";

        EXPECT_TRUE(causal_set(no_c).empty())
            << w.name << ": the device recorded causal when it was not asked to";
        EXPECT_EQ(branchial_size(no_c), branchial_size(all))
            << w.name << ": dropping causal moved the branchial relation";
        EXPECT_EQ(no_c.states.size(), all.states.size())
            << w.name << ": dropping causal moved the states";

        EXPECT_EQ(branchial_size(no_b), 0u)
            << w.name << ": the device recorded branchial when it was not asked to";
        EXPECT_EQ(causal_set(no_b), causal_set(all))
            << w.name << ": dropping branchial changed the causal relation";
        EXPECT_EQ(no_b.states.size(), all.states.size())
            << w.name << ": dropping branchial moved the states";
    }
}

// Past the depth the per-thread stack holds, the replay STOPS AND SAYS SO.
//
// The replay descends one frame triple per reconstruction depth, so its stack need is linear in
// the run's step count -- a quantity the caller chooses. EngineState sizes the stack from that
// count, but stack is reserved per resident thread, so it is capped, and past the cap there is a
// depth the device cannot reach. What must not happen there is a fault: an illegal memory access
// takes the whole run's result with it, and the caller gets nothing back and no reason.
//
// A chain rule is what reaches the regime cheaply. Consuming one edge and producing one gives a
// single application per depth, so the recursion is deep while the work stays small -- the wide
// rules in the corpus above would need 2^depth applications to get here.
// A DEPTH WHERE THE DEVICE POOLS SATURATE, which is where a partial answer used to be returned
// as though it were the answer.
//
// The device's quotient pools are sized off the EVENT count and filled per APPLICATION, and on a
// dense workload those differ by two orders of magnitude: 970,584 applications against 4,512
// events on a three-edge disconnected left side at depth 3. Two things followed. The branchial
// pairs were stored as an expansion in a 2^22 map, so 133,218,996 of them came back as
// 4,194,304; and the grow-and-retry ladder stopped at 64x, one doubling short, so the run still
// truncated after the map was removed.
//
// The suite passed through both, because nothing compared the device's reconstruction against
// the host's at a depth that saturates anything. This is that comparison. It asserts on COUNTS
// rather than on the relations themselves: the relation is 133 million pairs, and materialising
// it on both sides to compare them would cost more than the run.
TEST(QuotientReconstruction, ADepthThatSaturatesThePoolsStillAgreesWithTheHost) {
    Workload w;
    w.name = "saturating/disc-l3a2g2r2";
    // Three left edges in two components, arity 2, two growth edges: a cartesian product in the
    // matcher, which is what makes the instance count explode away from the event count.
    w.rules = {rule({{0, 1}, {2, 3}, {4, 5}},
                    {{0, 1}, {2, 3}, {4, 5}, {0, 6}, {2, 7}})};
    w.initial_state = {{0u, 1u}, {1u, 2u}, {2u, 0u}, {3u, 4u}};
    w.num_steps = 3;
    w.canon_mode = hg_gpu::CanonicalizationMode::Full;
    w.explore_from_canonical_states_only = true;

    hg_gpu::EvolveInput in = make_input(w);
    in.record = hgcommon::RecordSet{true, true, true};
    const auto gpu = hg_gpu::evolve(in);
    ASSERT_TRUE(gpu.reconstruction_ran) << "the device did not reconstruct, so there is nothing "
                                           "to compare and the gate asserts nothing";

    // The host side is run DIRECTLY rather than through run_cpu, and only its counters are
    // read. run_cpu normalises every relation into a multiset to compare them element by
    // element, which at 133 million branchial pairs is the relation materialised twice over --
    // more than the run costs, and the reason both engines stopped storing those pairs at all.
    // The counters are exact and O(1) on both sides, which is the whole point of deriving them
    // from the structure rather than from a stored expansion.
    hypergraph::Hypergraph hg;
    hg.set_state_canonicalization_mode(to_cpu_canon(w.canon_mode));
    hg.set_event_signature_keys(to_cpu_event_keys(w.event_canon_mode));
    {
        hypergraph::ParallelEvolutionEngine engine(&hg, /*num_threads=*/0);
        for (size_t i = 0; i < w.rules.size(); ++i)
            engine.add_rule(convert_rule(w.rules[i], static_cast<uint16_t>(i)));
        engine.set_transitive_reduction(w.transitive_reduction);
        engine.set_explore_from_canonical_states_only(w.explore_from_canonical_states_only);
        std::vector<std::vector<hypergraph::VertexId>> init;
        for (const auto& e : w.initial_state)
            init.emplace_back(e.begin(), e.end());
        engine.evolve(init, static_cast<int>(w.num_steps));
    }
    const size_t host_branchial = hg.num_reconstructed_branchial();
    const size_t host_causal    = hg.num_reconstructed_causal_pairs(false);

    // The gate is worthless if the workload is small enough to fit without growing anything.
    ASSERT_GT(host_branchial, 4u * 1024u * 1024u)
        << "this workload no longer exceeds the 2^22 ceiling the gate exists for; make it denser";

    EXPECT_EQ(gpu.reconstructed_branchial, host_branchial)
        << "the device returned a different number of branchial pairs than the host. A device "
           "figure that is exactly a power of two is a container ceiling, not an answer";
    EXPECT_EQ(gpu.reconstructed_causal_pairs, host_causal)
        << "the device returned a different number of causal pairs than the host";
}

TEST(QuotientReconstruction, PastTheStackDepthItRecordsRatherThanFaults) {
    const uint32_t deep = 80;   // beyond any stack the cap allows
    hg_gpu::EvolveInput in;
    in.rules = {rule({{0, 1}}, {{1, 2}})};
    in.initial_state = {{0u, 1u}};
    in.num_steps = deep;
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;
    in.explore_from_canonical_states_only = true;

    auto result = hg_gpu::evolve(in);

    // The run came back. This is the whole point: before the bound existed this faulted, and a
    // faulted context returns an empty result for every later call in the process too.
    EXPECT_FALSE(result.states.empty())
        << "the deep run returned no states at all, which is what a device fault looks like";

    bool faulted = false, bounded = false;
    for (const auto& w : result.warnings) {
        if (w.context.find("illegal memory access") != std::string::npos) faulted = true;
        if (w.kind == hg_gpu::ErrorKind::kScratchOverflow) bounded = true;
    }
    EXPECT_FALSE(faulted) << "the replay faulted instead of stopping at its depth bound";
    EXPECT_TRUE(bounded)
        << "a run " << deep << " deep neither reached that depth nor recorded that it could not";
}
