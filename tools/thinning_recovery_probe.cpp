// Does repeated thinning RECOVER the full multiway graph?
//
// The thinning contract, as stated for the release: prune transitions while leaving the overall
// structure in place, such that sampling enough times -- proportional to the pruning amount --
// recovers the original graph. This probe turns that from an intuition into a curve.
//
// Each seed produces a DIFFERENT deterministic subgraph (the draw is a pure function of
// (canonical transition, seed)), so the union over seeds grows toward the full graph. If a
// transition survives a given seed with probability ~q independently across seeds, coverage
// after k seeds is ~1-(1-q)^k: half-coverage around k = ln 2 / q, near-full around k = ln N / q
// -- "proportional to the pruning amount". Departures from that curve are the structure the
// spine and the cascade add (a transition can only be drawn if its source state was reached),
// which slows early coverage of deep regions.
//
// Reported per (q, k): union coverage of canonical STATES and Full-identity EVENTS against the
// unpruned run. Soundness (union <= full) is asserted, not assumed.
//
// Usage: thinning_recovery_probe [steps] [max_seeds]

#include "hypergraph/parallel_evolution.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

using namespace hypergraph;

namespace {

uint64_t fnv(uint64_t h, uint64_t x) { h ^= x; h *= 1099511628211ULL; return h; }

struct Sets {
    std::set<uint64_t> states, events;
};

Sets run(double q, uint64_t seed, int steps) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    g.set_event_signature_keys(hgcommon::EVENT_SIG_FULL);
    ParallelEvolutionEngine e(&g, 4);
    e.set_transition_rate(q);
    e.set_random_seed(seed);
    e.add_rule(make_rule(0).lhs({0,1}).lhs({1,2})
                   .rhs({0,1}).rhs({1,3}).rhs({3,2}).rhs({2,0}).build());
    e.evolve({{0,1},{1,2}}, steps);

    Sets s;
    for (uint32_t sid = 0; sid < g.num_states(); ++sid) {
        if (g.get_state(sid).id == INVALID_ID) continue;
        s.states.insert(g.get_or_compute_canonical_hash(sid));
    }
    for (uint32_t eid = 0; eid < g.num_raw_events(); ++eid) {
        const Event& ev = g.get_event(eid);
        if (ev.id == INVALID_ID) continue;
        const uint64_t in = ev.input_state == INVALID_ID
            ? 0 : g.get_or_compute_canonical_hash(ev.input_state);
        const uint64_t out = ev.output_state == INVALID_ID
            ? 0 : g.get_or_compute_canonical_hash(ev.output_state);
        s.events.insert(fnv(fnv(1469598103934665603ULL, in), out));
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 5;
    const int max_seeds = argc > 2 ? std::atoi(argv[2]) : 24;

    const Sets full = run(1.0, 1, steps);
    std::printf("thinning recovery: wolfram-2to4, %d steps; full = %zu states, %zu events\n",
                steps, full.states.size(), full.events.size());
    std::printf("%-6s %-5s %-16s %-16s\n", "q", "k", "state coverage", "event coverage");

    for (double q : {0.125, 0.25}) {
        Sets uni;
        for (int k = 1; k <= max_seeds; ++k) {
            Sets s = run(q, 0x1000 + static_cast<uint64_t>(k), steps);
            for (uint64_t v : s.states) {
                if (!full.states.count(v)) { std::printf("UNSOUND: sampled state not in full\n"); return 1; }
                uni.states.insert(v);
            }
            for (uint64_t v : s.events) {
                if (!full.events.count(v)) { std::printf("UNSOUND: sampled event not in full\n"); return 1; }
                uni.events.insert(v);
            }
            if (k == 1 || k == 2 || k == 4 || k == 8 || k == 16 || k == max_seeds)
                std::printf("%-6.3f %-5d %6zu/%-6zu %.3f  %6zu/%-6zu %.3f\n",
                            q, k,
                            uni.states.size(), full.states.size(),
                            double(uni.states.size()) / double(full.states.size()),
                            uni.events.size(), full.events.size(),
                            double(uni.events.size()) / double(full.events.size()));
        }
    }
    return 0;
}
