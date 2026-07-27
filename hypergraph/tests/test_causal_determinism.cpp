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

namespace hgraph = hypergraph;   // hg is the engine's atomic_compat namespace

namespace {

uint64_t fnv(uint64_t h, uint64_t x) { h ^= x; h *= 1099511628211ULL; return h; }

struct Fingerprint {
    uint64_t states = 0, causal = 0, branchial = 0;
    long num_states = 0, num_events = 0, num_causal = 0, num_branchial = 0;
};

Fingerprint fingerprint(hgraph::Hypergraph& g) {
    auto canon = [&](hgraph::StateId s) -> uint64_t {
        return s == hgraph::INVALID_ID ? 0 : g.get_or_compute_canonical_hash(s);
    };
    auto esig = [&](hgraph::EventId e) -> uint64_t {
        const hgraph::Event& x = g.get_event(e);
        return fnv(fnv(fnv(1469598103934665603ULL, canon(x.input_state)),
                       canon(x.output_state)), x.rule_index);
    };

    Fingerprint fp;
    std::vector<uint64_t> sh;
    for (uint32_t s = 0; s < g.num_states(); ++s)
        if (g.get_state(s).id != hgraph::INVALID_ID) sh.push_back(canon(s));
    std::sort(sh.begin(), sh.end());
    fp.states = 1469598103934665603ULL; for (uint64_t v : sh) fp.states = fnv(fp.states, v);
    fp.num_states = static_cast<long>(sh.size());

    // Under quotient the causal relation is reconstructed rather than explored, so it is read
    // from the reconstruction -- which is also what the engine reports. Reading the materialised
    // causal graph here instead would fingerprint an empty set and pass vacuously.
    std::vector<uint64_t> ce;
    if (g.quotient_reconstruction()) {
        g.for_each_reconstructed_causal(/*reduced=*/true, [&](uint64_t p, uint64_t c) {
            ce.push_back(fnv(fnv(0, p), c));
        });
    } else {
        for (const auto& c : g.causal_graph().get_causal_edges()) {
            if (c.producer == hgraph::INVALID_ID || c.consumer == hgraph::INVALID_ID) continue;
            ce.push_back(fnv(fnv(0, esig(c.producer)), esig(c.consumer)));
        }
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
        if (g.get_event(e).id != hgraph::INVALID_ID) ++fp.num_events;
    return fp;
}

Fingerprint run(const std::vector<hgraph::RewriteRule>& rules,
                const std::vector<std::vector<hgraph::VertexId>>& init,
                bool quotient, int threads, uint64_t seed, int steps) {
    hgraph::Hypergraph g;
    g.set_state_canonicalization_mode(hgraph::StateCanonicalizationMode::Full);
    hgraph::ParallelEvolutionEngine e(&g, threads);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(quotient);
    e.set_random_seed(seed);
    for (const auto& r : rules) e.add_rule(r);
    e.evolve(init, steps);
    return fingerprint(g);
}

struct Workload {
    const char* name;
    std::vector<hgraph::RewriteRule> rules;
    std::vector<std::vector<hgraph::VertexId>> init;
    int steps;
};

std::vector<Workload> workloads() {
    std::vector<Workload> w;
    w.push_back({"WPP",
        {hgraph::make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
        {{0,1},{0,2}}, 6});
    w.push_back({"mixed1",
        {hgraph::make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
         hgraph::make_rule(1).lhs({0,1}).rhs({1,0}).build(),
         hgraph::make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()},
        {{0,1}}, 6});
    w.push_back({"mixed2",
        {hgraph::make_rule(0).lhs({0,1}).rhs({1,0}).build(),
         hgraph::make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()},
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

// Quotient causal attribution must be order-independent. The engine serves the correct
// TR-OFF causal graph under quotient (the online producer-set reconstruction, qc_*); the
// run() harness requests TR on, but the engine's guard_quotient_transitive_reduction()
// downgrades it to TR-off, because the transitively-reduced RAW causal graph is PROVEN not
// reconstructable from the quotient skeleton (the raw instance wiring is discarded --
// docs/VERIFICATION_PLAN.md, tools/quotient_causal_tr_deadend_probe.cpp). So under quotient
// this verifies TR-OFF causal determinism; TR-on causal determinism is covered by
// NonQuotientFullyDeterministic (full-capture, the only mode that can produce reduce(raw)).
TEST(CausalDeterminism, QuotientCausalAttribution) {
    for (const auto& w : workloads()) {
        Spread s = spread(w, /*quotient=*/true);
        EXPECT_EQ(s.causal.size(), 1u) << w.name << ": causal attribution non-deterministic under quotient";
    }
}

// Branchial siblings of one instance must be counted exactly once, whatever order the
// two matches of a pair are applied in.
//
// The pairing used to elect a reporter by match id ("count it if other.id < m.id"), which
// silently assumed id order matched the order matches become visible in the expansion
// list. It does not -- ids come from a global counter while the list is appended
// concurrently -- so a lower-id match could reach the list after a higher-id match had
// already scanned: the higher one never saw it, the lower one dismissed the higher as not
// below it, and the pair was lost by BOTH sides. It reproduced as a branchial count short
// by 2 on roughly one matrix run in four, never on events or causal pairs.
//
// Electing the reporter by application order instead is necessary but not sufficient:
// both sides can observe the other's application claim (claim a, claim b, scan a, scan b),
// so the pair itself has to be claimed. This drives the two smallest configurations that
// exhibited the loss, at the thread count that produced it.
TEST(CausalDeterminism, QuotientBranchialCountedExactlyOnce) {
    // iters is sized from the measured loss rate WITHOUT the fix: dup+dedup/selfloop lost a
    // pair about once in 700 evolutions (observed at iterations 137 and 1310 on two runs), so
    // 4000 catches a regression with high probability in ~2 s. The fan case races far less and
    // is kept short -- it is here because it was one of the two configurations observed to
    // fail, not because it carries the detection.
    struct Case { const char* name; int iters; std::vector<hgraph::RewriteRule> rules;
                  std::vector<std::vector<hgraph::VertexId>> init; };
    // dup+dedup: {{x,y}} -> {{x,y}},{{x,y}} together with {{x,y}},{{x,y}} -> {{x,y}}.
    auto dup_dedup = [] {
        return std::vector<hgraph::RewriteRule>{
            hgraph::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build(),
            hgraph::make_rule(1).lhs({0,1}).lhs({0,1}).rhs({0,1}).build()};
    };
    const std::vector<Case> cases = {
        {"dup+dedup/selfloop", 4000, dup_dedup(), {{0,0}}},
        {"dup+dedup/fan",       400, dup_dedup(), {{0,1},{0,2}}},
    };

    for (const auto& c : cases) {
        size_t expected = 0;
        {   // full capture, single-threaded: the reference the reconstruction must match
            hgraph::Hypergraph hg;
            hg.set_state_canonicalization_mode(hgraph::StateCanonicalizationMode::Full);
            hgraph::ParallelEvolutionEngine e(&hg, 1);
            e.set_transitive_reduction(false);
            e.set_explore_from_canonical_states_only(false);
            for (const auto& r : c.rules) e.add_rule(r);
            auto in = c.init; e.evolve(in, 3);
            expected = hg.causal_graph().num_branchial_edges();
        }
        ASSERT_GT(expected, 0u) << c.name << ": no branchial edges to compare";

        for (int iter = 0; iter < c.iters; ++iter) {
            hgraph::Hypergraph hg;
            hg.set_state_canonicalization_mode(hgraph::StateCanonicalizationMode::Full);
            hgraph::ParallelEvolutionEngine e(&hg, 8);
            e.set_transitive_reduction(false);
            e.set_explore_from_canonical_states_only(true);
            hg.set_quotient_reconstruction(true);
            for (const auto& r : c.rules) e.add_rule(r);
            auto in = c.init; e.evolve(in, 3);
            ASSERT_EQ(hg.num_reconstructed_branchial(), expected)
                << c.name << ": branchial count differs from full capture on iteration " << iter;
        }
    }
}
