// Does sampled evolution REACH DEPTH at any rate, and is the surviving set schedule-stable?
//
// The evidence that created #63: at fixed q the sampled evolution is a branching process, and
// below 1/branching it is subcritical -- wolfram-2to4 at q=1/8 and 1/4 went extinct at depth 1.
// The spine closes that: every state's drain guarantees one surviving outgoing transition (the
// minimum canonical-key stored match), so depth is reached at ANY q and the rate controls the
// bushiness, not survival.
//
// Two questions, so two outputs per (q, threads):
//   reached   states per depth -- extinction is a zero before max depth
//   fp        canonical-state-set fingerprint -- compared across thread counts, because the
//             spine's pick at a state is the min over the matches STORED BY ITS DRAIN, and late
//             forwarded arrivals make that set potentially schedule-relative. The claim that the
//             spine is schedule-stable is MEASURED here, not assumed; a fingerprint mismatch
//             localises to the workload and rate that break it.
//
// Usage: spine_probe [steps]

#include "hypergraph/parallel_evolution.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

using namespace hypergraph;

namespace {

uint64_t fnv(uint64_t h, uint64_t x) { h ^= x; h *= 1099511628211ULL; return h; }

struct Out {
    std::map<uint32_t, size_t> per_depth;
    size_t states = 0;
    uint64_t fp = 0;
    size_t spine = 0, taken = 0, survived = 0, drained = 0;
};

Out run(double q, int threads, int steps) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&g, threads);
    e.set_transition_rate(q);
    e.set_random_seed(0xABCDEF);
    e.add_rule(make_rule(0).lhs({0,1}).lhs({1,2})
                   .rhs({0,1}).rhs({1,3}).rhs({3,2}).rhs({2,0}).build());
    e.evolve({{0,1},{1,2}}, steps);

    Out o;
    // DISTINCT canonical hashes: an XOR fold over raw states cancels even multiplicities, and
    // two different runs can then share a fingerprint -- measured, q=0.5 and q=1.0 collided
    // before this was a set. Order-free because a set iterates sorted.
    std::set<uint64_t> hashes;
    for (uint32_t s = 0; s < g.num_published_states(); ++s) {
        if (g.get_state(s).id == INVALID_ID) continue;
        o.per_depth[g.get_state(s).step]++;
        hashes.insert(g.get_or_compute_canonical_hash(s));
        ++o.states;
    }
    o.fp = 1469598103934665603ULL;
    for (uint64_t h : hashes) o.fp = fnv(o.fp, h);
    o.spine = e.stats().spine_forced.load();
    o.taken = e.draws_taken();
    o.survived = e.draws_survived();
    o.drained = e.states_drained();
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 6;
    std::printf("spine probe: wolfram-2to4, %d steps\n", steps);
    std::printf("%-6s %-4s %-8s %-18s per-depth states\n", "q", "thr", "states", "set-fp");

    for (double q : {0.125, 0.25, 0.5, 1.0}) {
        uint64_t fp1 = 0;
        for (int thr : {1, 4}) {
            Out o = run(q, thr, steps);
            std::printf("%-6.3f %-4d %-8zu %016llx sp=%zu dr=%zu %zu/%zu  ", q, thr, o.states,
                        static_cast<unsigned long long>(o.fp), o.spine, o.drained,
                        o.survived, o.taken);
            for (auto& [d, n] : o.per_depth) std::printf("%u:%zu ", d, n);
            const uint32_t max_d = o.per_depth.empty() ? 0 : o.per_depth.rbegin()->first;
            if (max_d < static_cast<uint32_t>(steps)) std::printf(" EXTINCT@%u", max_d);
            if (thr == 1) fp1 = o.fp;
            else if (o.fp != fp1) std::printf(" THREAD-UNSTABLE");
            std::printf("\n");
        }
    }
    return 0;
}
