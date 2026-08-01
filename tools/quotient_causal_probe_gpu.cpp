// Is the persistent scheduler's causal-edge count schedule-dependent, and is the quotient
// expansion race the mechanism?
//
// The cross-scheduler gate (Rewrite.PersistentSchedulerThroughEngineRunMatchesTheStepLoop)
// intermittently reads one fewer causal edge from the persistent run in state-mode Full.
// The candidate mechanism: under explore_from_canonical_states_only, isomorphic children race
// for the canonical slot and only the WINNER's raw instance is expanded; downstream consumers
// then attach their causal edges to the winner's raw edges, whose producers differ between
// winners, so the kept TR set follows the race. If that is the mechanism, the count must be
// CONSTANT with exploration pruning off (full capture: every raw instance expanded, no race to
// win) and vary with it on.
//
// Usage: quotient_causal_probe_gpu [reps]   (HG_GPU_PERSISTENT_BLOCKS picks the grid)

#include "hg_gpu/evolve.hpp"
#include "hypergraph/parallel_evolution.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;

    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;

    for (bool quotient : {true, false}) {
        std::map<size_t, int> counts;
        std::map<size_t, int> tallies_states;
        int warned = 0;
        for (int i = 0; i < reps; ++i) {
            hg_gpu::EvolveInput in;
            in.rules = {r};
            in.initial_state = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
            in.num_steps = 3;
            in.canonicalization = hg_gpu::CanonicalizationMode::Full;
            in.event_canonicalization = hg_gpu::EventCanonicalizationMode::Full;
            in.explore_from_canonical_states_only = quotient;
            in.persistent_scheduler = true;
            const auto res = hg_gpu::evolve(in);
            counts[res.causal_edges.size()]++;
            tallies_states[res.states.size()]++;
            if (!res.warnings.empty()) ++warned;
        }
        std::printf("quotient=%d reps=%d warned=%d causal counts:", quotient ? 1 : 0, reps,
                    warned);
        for (auto& [k, v] : counts) std::printf(" %zu x%d", k, v);
        std::printf("  state counts:");
        for (auto& [k, v] : tallies_states) std::printf(" %zu x%d", k, v);
        std::printf("\n");
    }

    // CPU reference on the identical workload, both routes: the quotient DP's canonical-pair
    // count must agree across devices, and full capture is the sanity anchor.
    for (bool quotient : {true, false}) {
        std::map<size_t, int> counts;
        for (int i = 0; i < reps; ++i) {
            hypergraph::Hypergraph g;
            g.set_state_canonicalization_mode(hypergraph::StateCanonicalizationMode::Full);
            g.set_event_signature_keys(hgcommon::EVENT_SIG_FULL);
            g.causal_graph().set_transitive_reduction(true);
            hypergraph::ParallelEvolutionEngine e(&g, 4);
            e.set_explore_from_canonical_states_only(quotient);
            e.add_rule(hypergraph::make_rule(0).lhs({0, 1}).lhs({1, 2})
                           .rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build());
            e.evolve({{0, 1}, {1, 2}, {2, 3}}, 3);
            counts[g.causal_graph().get_causal_edges().size()]++;
        }
        std::printf("cpu quotient=%d reps=%d causal counts:", quotient ? 1 : 0, reps);
        for (auto& [k, v] : counts) std::printf(" %zu x%d", k, v);
        std::printf("\n");
    }
    return 0;
}
