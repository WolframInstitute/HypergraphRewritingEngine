#include <gtest/gtest.h>
#include <vector>
#include <set>
#include <tuple>
#include <algorithm>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

// =============================================================================
// Canonical determinism gate.
//
// The engine's SEMANTIC output must be a schedule-independent function of
// (rules, initial state, options) -- identical across runs, thread counts, and
// RNG seeds. We fingerprint the output *canonically* (iso-invariant): states by
// their canonical hash, causal/branchial edges as sorted pairs of iso-invariant
// event signatures. This factors out benign id/order churn and detects only
// genuine structural non-determinism.
//
// Crucially the matrix includes LOOP-FORMING / recurrence rulesets under quotient
// exploration -- the case the earlier growing-rule determinism gates never
// exercised, which is exactly why the quotient causal-attribution non-determinism
// hid. See docs/VERIFICATION_PLAN.md.
// =============================================================================

namespace hg = hypergraph;

namespace {

uint64_t fnv(uint64_t h, uint64_t x) { h ^= x; h *= 1099511628211ULL; return h; }

struct Fingerprint {
    uint64_t states = 0, causal = 0, branchial = 0;
    long num_states = 0, num_events = 0, num_causal = 0, num_branchial = 0;
};

Fingerprint fingerprint(hg::Hypergraph& g) {
    auto canon = [&](hg::StateId s) -> uint64_t {
        return s == hg::INVALID_ID ? 0 : g.get_or_compute_canonical_hash(s);
    };
    auto esig = [&](hg::EventId e) -> uint64_t {
        const hg::Event& x = g.get_event(e);
        return fnv(fnv(fnv(1469598103934665603ULL, canon(x.input_state)),
                       canon(x.output_state)), x.rule_index);
    };

    Fingerprint fp;
    std::vector<uint64_t> sh;
    for (uint32_t s = 0; s < g.num_states(); ++s)
        if (g.get_state(s).id != hg::INVALID_ID) sh.push_back(canon(s));
    std::sort(sh.begin(), sh.end());
    fp.states = 1469598103934665603ULL; for (uint64_t v : sh) fp.states = fnv(fp.states, v);
    fp.num_states = static_cast<long>(sh.size());

    std::vector<uint64_t> ce;
    for (const auto& c : g.causal_graph().get_causal_edges()) {
        if (c.producer == hg::INVALID_ID || c.consumer == hg::INVALID_ID) continue;
        ce.push_back(fnv(fnv(0, esig(c.producer)), esig(c.consumer)));
    }
    std::sort(ce.begin(), ce.end());
    fp.causal = 1469598103934665603ULL; for (uint64_t v : ce) fp.causal = fnv(fp.causal, v);
    fp.num_causal = static_cast<long>(ce.size());

    std::vector<uint64_t> be;
    for (const auto& b : g.causal_graph().get_branchial_edges()) {
        uint64_t a = esig(b.event1), d = esig(b.event2);
        if (a > d) std::swap(a, d);
        be.push_back(fnv(fnv(0, a), d));
    }
    std::sort(be.begin(), be.end());
    fp.branchial = 1469598103934665603ULL; for (uint64_t v : be) fp.branchial = fnv(fp.branchial, v);
    fp.num_branchial = static_cast<long>(be.size());

    for (uint32_t e = 0; e < g.num_raw_events(); ++e)
        if (g.get_event(e).id != hg::INVALID_ID) ++fp.num_events;
    return fp;
}

Fingerprint run(const std::vector<hg::RewriteRule>& rules,
                const std::vector<std::vector<hg::VertexId>>& init,
                bool quotient, int threads, uint64_t seed, int steps) {
    hg::Hypergraph g;
    g.set_state_canonicalization_mode(hg::StateCanonicalizationMode::Full);
    hg::ParallelEvolutionEngine e(&g, threads);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(quotient);
    e.set_random_seed(seed);
    for (const auto& r : rules) e.add_rule(r);
    e.evolve(init, steps);
    return fingerprint(g);
}

struct Workload {
    const char* name;
    std::vector<hg::RewriteRule> rules;
    std::vector<std::vector<hg::VertexId>> init;
    int steps;
};

std::vector<Workload> workloads() {
    std::vector<Workload> w;
    w.push_back({"WPP",
        {hg::make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
        {{0,1},{0,2}}, 6});
    w.push_back({"mixed1",
        {hg::make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
         hg::make_rule(1).lhs({0,1}).rhs({1,0}).build(),
         hg::make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()},
        {{0,1}}, 6});
    w.push_back({"mixed2",
        {hg::make_rule(0).lhs({0,1}).rhs({1,0}).build(),
         hg::make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()},
        {{0,1}}, 6});
    return w;
}

// Collect the distinct value of each fingerprint component over runs × threads × seeds.
struct Spread { std::set<uint64_t> states, causal, branchial; std::set<long> ns, ne, nc, nb; };
Spread spread(const Workload& w, bool quotient) {
    Spread s;
    for (uint64_t seed : {uint64_t(0xABCDEF), uint64_t(0)})   // fixed then random
        for (int rep = 0; rep < 4; ++rep)
            for (int th : {1, 2, 8}) {
                Fingerprint f = run(w.rules, w.init, quotient, th, seed, w.steps);
                s.states.insert(f.states); s.causal.insert(f.causal); s.branchial.insert(f.branchial);
                s.ns.insert(f.num_states); s.ne.insert(f.num_events);
                s.nc.insert(f.num_causal); s.nb.insert(f.num_branchial);
            }
    return s;
}

}  // namespace

// Without quotient the entire semantic output is a pure function of the input.
TEST(CausalDeterminism, NonQuotientFullyDeterministic) {
    for (const auto& w : workloads()) {
        Spread s = spread(w, /*quotient=*/false);
        EXPECT_EQ(s.states.size(), 1u)    << w.name << ": state set non-deterministic";
        EXPECT_EQ(s.causal.size(), 1u)    << w.name << ": causal graph non-deterministic";
        EXPECT_EQ(s.branchial.size(), 1u) << w.name << ": branchial graph non-deterministic";
    }
}

// Under quotient, states / events / branchial are already deterministic; only causal
// attribution is not (the first-writer-wins single producer per canonical edge).
TEST(CausalDeterminism, QuotientStatesEventsBranchialDeterministic) {
    for (const auto& w : workloads()) {
        Spread s = spread(w, /*quotient=*/true);
        EXPECT_EQ(s.states.size(), 1u)    << w.name << ": state set non-deterministic under quotient";
        EXPECT_EQ(s.branchial.size(), 1u) << w.name << ": branchial non-deterministic under quotient";
        EXPECT_EQ(s.ne.size(), 1u)        << w.name << ": event count non-deterministic under quotient";
        EXPECT_EQ(s.nb.size(), 1u)        << w.name << ": branchial count non-deterministic under quotient";
    }
}

// The Phase-2 target: causal attribution must be order-independent under quotient too.
// DISABLED until the producer-set (order-independent) attribution lands; enabling this
// (rename off DISABLED_) is the acceptance test for that fix. See docs/VERIFICATION_PLAN.md.
TEST(CausalDeterminism, DISABLED_QuotientCausalAttribution) {
    for (const auto& w : workloads()) {
        Spread s = spread(w, /*quotient=*/true);
        EXPECT_EQ(s.causal.size(), 1u) << w.name << ": causal attribution non-deterministic under quotient";
    }
}
