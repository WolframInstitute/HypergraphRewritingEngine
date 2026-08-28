// GENMC-LINK: engine
// GENMC-ARGS: --unroll=2
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16
//
// GenMC harness: THE WHOLE ENGINE, one evolve() call. Every job-system path (submit, steal,
// park, wake, quiescence), the matcher, the rewriter, the causal graph and the state and event
// registries run as they ship, under RC11, and the checker explores every interleaving of the
// workers. The property is the run's own contract -- the event and state counts of the input --
// and no memory error on the way.
//
// TWO ARMS, chosen by defines (HG_HARNESS_DEFINES):
//
//   default                     two workers, rule {0,1} -> {0,1},{1,2}, one edge, one step, no
//                               canonicalisation. The smallest run that goes through everything
//                               once: 1 event, 2 states. Sized 2026-08-28: 1 execution.
//   -DHG_EVOLVE_LIVE_SHAPE=1    the shape of the live nondeterminism failures (cycle4-automorphic
//                               at 16 threads, Full canonicalisation): THREE workers, the
//                               two-edge rule {0,1},{1,2} -> {0,1},{1,3},{3,2}, a two-edge path,
//                               TWO steps, Full canonicalisation. The child state pulls the
//                               parent's surviving matches and delta-matches its produced edges,
//                               the second step's rewrites consume edges the first produced
//                               (causal in-edges, transitive reduction), the two second-step
//                               children are isomorphic (canonical dedup under IR + orbits), and
//                               a third worker can interleave every rendezvous the other two
//                               are in. 3 events, 4 raw states, 3 canonical.
//
// WHAT IS BOUNDED. Every loop unrolled twice, which ends a thread that exceeds it as blocked,
// never as an error. The module is the fully inlined engine (43M LLVM instructions before the
// checker's own passes), so this is run with HG_GENMC_PROGRESS and sized with --mode=estimate
// before an exhaustive run.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"
#include <cassert>
#include <vector>

int main() {
    hg::engine::Hypergraph g;
#if defined(HG_EVOLVE_LIVE_SHAPE)
    g.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::Full);
    hg::engine::RewriteRule rule = hg::engine::make_rule(0)
        .lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build();
    hg::engine::ParallelEvolutionEngine e(&g, 3);
    e.add_rule(rule);
    std::vector<std::vector<hg::engine::VertexId>> init = {{0, 1}, {1, 2}};
    e.evolve(init, 2);
    assert(e.num_events() == 3);
    assert(g.num_states() == 4);
#else
    g.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::None);
    hg::engine::RewriteRule rule = hg::engine::make_rule(0)
        .lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();
    hg::engine::ParallelEvolutionEngine e(&g, 2);
    e.add_rule(rule);
    std::vector<std::vector<hg::engine::VertexId>> init = {{0, 1}};
    e.evolve(init, 1);
    assert(e.num_events() == 1);
    assert(g.num_states() == 2);
#endif
    return 0;
}
