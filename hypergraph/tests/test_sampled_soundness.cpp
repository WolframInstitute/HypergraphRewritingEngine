#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

// =============================================================================
// RELEASE GATE: a sampled evolution is a SOUND, DEPTH-COMPLETE, SCHEDULE-STABLE
// subgraph of the unpruned evolution.
//
// Equality is not the property -- a sample is smaller by construction. What a
// release can promise is:
//
//   SOUND       nothing is invented: every sampled canonical state, event
//               identity, causal pair and branchial pair exists in the unpruned
//               run of the same model.
//   DEPTH       the spine guarantee holds: the sample reaches the requested
//               depth at every rate (the pre-spine behaviour was extinction at
//               depth 1 below the critical rate).
//   STABLE      the sampled sets are identical across thread counts -- the
//               draws key on canonical transitions and the spine picks minimum
//               canonical keys, so the scheduler must not be able to change
//               WHICH subgraph was sampled.
// =============================================================================

namespace {

using namespace hypergraph;

uint64_t fnv(uint64_t h, uint64_t x) { h ^= x; h *= 1099511628211ULL; return h; }

struct Workload {
    const char* name;
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> init;
    int steps;
};

std::vector<Workload> workloads() {
    std::vector<Workload> w;
    w.push_back({"wolfram-2to4",
        {make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).rhs({2,0}).build()},
        {{0,1},{1,2}}, 4});
    w.push_back({"WPP",
        {make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
        {{0,1},{0,2}}, 4});
    return w;
}

struct Sets {
    std::set<uint64_t> states;
    std::set<uint64_t> events;                       // EVENT_SIG_FULL identities
    std::set<std::pair<uint64_t, uint64_t>> causal;   // (producer sig, consumer sig)
    std::set<std::pair<uint64_t, uint64_t>> branchial;
    uint32_t max_depth = 0;
};

Sets run(const Workload& w, double q, int threads) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    g.set_event_signature_keys(hgcommon::EVENT_SIG_FULL);
    ParallelEvolutionEngine e(&g, threads);
    e.set_transitive_reduction(false);
    e.set_transition_rate(q);
    e.set_random_seed(0xABCDEF);
    for (const auto& r : w.rules) e.add_rule(r);
    e.evolve(w.init, w.steps);

    auto esig = [&](EventId id) -> uint64_t {
        const Event& ev = g.get_event(id);
        const uint64_t in = ev.input_state == INVALID_ID
            ? 0 : g.get_or_compute_canonical_hash(ev.input_state);
        const uint64_t out = ev.output_state == INVALID_ID
            ? 0 : g.get_or_compute_canonical_hash(ev.output_state);
        return fnv(fnv(1469598103934665603ULL, in), out);
    };

    Sets s;
    for (uint32_t sid = 0; sid < g.num_states(); ++sid) {
        if (g.get_state(sid).id == INVALID_ID) continue;
        s.states.insert(g.get_or_compute_canonical_hash(sid));
        if (g.get_state(sid).step > s.max_depth) s.max_depth = g.get_state(sid).step;
    }
    for (uint32_t eid = 0; eid < g.num_raw_events(); ++eid) {
        if (g.get_event(eid).id == INVALID_ID) continue;
        s.events.insert(esig(eid));
    }
    for (const auto& c : g.causal_graph().get_causal_edges()) {
        if (c.producer == INVALID_ID || c.consumer == INVALID_ID) continue;
        s.causal.insert({esig(c.producer), esig(c.consumer)});
    }
    for (const auto& b : g.causal_graph().get_branchial_edges()) {
        uint64_t a = esig(b.event1), d = esig(b.event2);
        if (a > d) std::swap(a, d);
        s.branchial.insert({a, d});
    }
    return s;
}

template <typename T>
bool subset(const std::set<T>& a, const std::set<T>& b) {
    for (const T& x : a) if (!b.count(x)) return false;
    return true;
}

}  // namespace

TEST(SampledSoundness, SoundDepthCompleteAndScheduleStable) {
    for (const auto& w : workloads()) {
        const Sets full = run(w, 1.0, 4);
        for (double q : {0.25, 0.5}) {
            const Sets s1 = run(w, q, 1);
            const Sets s4 = run(w, q, 4);

            // SOUND: the sample invents nothing.
            EXPECT_TRUE(subset(s1.states, full.states)) << w.name << " q=" << q << ": states";
            EXPECT_TRUE(subset(s1.events, full.events)) << w.name << " q=" << q << ": events";
            EXPECT_TRUE(subset(s1.causal, full.causal)) << w.name << " q=" << q << ": causal";
            EXPECT_TRUE(subset(s1.branchial, full.branchial))
                << w.name << " q=" << q << ": branchial";

            // DEPTH: the spine carries the sample to the requested depth.
            EXPECT_EQ(s1.max_depth, static_cast<uint32_t>(w.steps))
                << w.name << " q=" << q << ": extinct before depth";

            // STABLE: the scheduler cannot change which subgraph was sampled.
            EXPECT_EQ(s1.states, s4.states) << w.name << " q=" << q << ": state set by threads";
            EXPECT_EQ(s1.events, s4.events) << w.name << " q=" << q << ": event set by threads";

            // And it is genuinely a sample, not the whole thing.
            EXPECT_LT(s1.states.size(), full.states.size()) << w.name << " q=" << q;
        }
    }
}
