// GENMC-LINK: engine
// GENMC-ARGS: --unroll=2
//
// GenMC harness: THE COMPOSED ENGINE, constructed. Every engine translation unit is linked, and
// what main reaches is what the checker is handed -- a Hypergraph, a ParallelEvolutionEngine with
// two workers, and therefore a started JobSystem: two worker threads spawned, each entering its
// loop, finding nothing, and parking on the gate hgcommon/park_gate.hpp checks as a unit.
//
// This is the first rung of the ladder that ends at evolve() (engine_rule, engine_evolve), and
// it exists on its own because for a long time it was the ceiling: the interpreter died before
// the first thread ran, on globals it could not materialise. What lifted it is recorded in
// README.md under "What HG_VERIFICATION changes" and in run.sh at the link step, and every one of
// those rewrites is a pipeline step applied to the code as it is, not an edit to a module.
//
// --unroll=2 bounds every loop to two iterations. The workers' loops are spin loops on the park
// word and the job deques, which the spin-assume transformation turns into assumes; the bound is
// what keeps a worker that never receives work from being explored forever. A bounded loop that
// exceeds its bound ends that thread's execution as BLOCKED, never as an error, so the bound can
// hide a behaviour but cannot manufacture one.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"

#include <cassert>
#include <vector>

int main() {
    hg::engine::Hypergraph g;
    g.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::None);
    hg::engine::ParallelEvolutionEngine e(&g, 2);
    assert(e.num_events() == 0);
    return 0;
}
