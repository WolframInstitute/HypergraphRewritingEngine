// Why does the kept fraction read LOW under batched submission?
//
// AUDIT NOTE (post-spine). Spine-forced survivals (the drain's minimum-key spawn and the late
// spine) pass OUTSIDE the draw, so draws_survived_ undercounts kept transitions by the spine
// count on sampled runs; stats_.spine_forced carries that number. The set-vs-repeat inference
// below is unaffected -- it compares draw denominators, which the spine does not touch.
//
// The draw is deterministic in canonical_transition_key, the denominators were verified
// symmetric, and the forwarded prefix takes exactly one draw. So a low numerator means the two
// modes are drawing on DIFFERENT SETS of keys, not drawing more often on the same ones. Counting
// draws taken and draws survived separates those two possibilities without dumping every key:
//
//   same draws, fewer survivors  -> different KEYS (the sets differ)
//   more draws, same ratio       -> the same key is being drawn on repeatedly
#include <cstdio>
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
using namespace hypergraph;

int main() {
    // The same growth rule the failing test uses, built the same way.
    const RewriteRule rule = make_rule(0).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();
    std::vector<std::vector<VertexId>> init;
    for (int i = 0; i < 24; ++i) init.push_back({(VertexId)i, (VertexId)(i+1)});

    std::printf("%-9s %5s | %8s %8s | %8s %8s | %7s | %8s %8s %8s %8s %8s\n",
                "mode","q","draws","survived","matches","events","kept",
                "push","batPull","eagPull","collect","sink");
    for (bool batched : {false, true}) {
        for (double q : {0.25, 0.5}) {
            size_t draws=0, surv=0, matches=0, events=0; size_t site[5]={0,0,0,0,0};
            for (uint64_t seed = 1; seed <= 12; ++seed) {
                Hypergraph hg;
                hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
                ParallelEvolutionEngine e(&hg, 4);
                e.set_random_seed(seed);
                e.set_match_forwarding(true);
                e.set_batched_matching(batched);
                e.set_transition_rate(q);
                e.add_rule(rule);
                e.evolve(init, 4);
                draws += e.draws_taken(); surv += e.draws_survived();
                for (int k = 0; k < 5; ++k) site[k] += e.draws_at_site(k);
                matches += e.total_matches(); events += hg.num_events();
            }
            std::printf("%-9s %5.2f | %8zu %8zu | %8zu %8zu | %7.4f | %8zu %8zu %8zu %8zu %8zu\n",
                        batched?"batched":"eager", q, draws, surv, matches, events,
                        matches? (double)events/matches : 0.0,
                        site[0],site[1],site[2],site[3],site[4]);
        }
    }
    return 0;
}
