// GENMC-LINK: engine
// GENMC-ARGS: --disable-estimation --check-liveness --weak-cas-spurious-total=1
// GENMC-BOUND: weak CAS spurious-failure bound of 1 per execution: 60,112 complete executions, K = 561 dropped (bounded, not exhaustive).
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=4 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16 -DHG_JOB_QUEUE_CAPACITY=16 -DHG_JOB_INJECTOR_CAPACITY=64 -DHG_MAX_ARENA_WORKERS=8 -DHG_KEY_SET_SHARDS=4 -DHG_MAX_PATTERN_EDGES=4 -DHG_ARENA_BLOCK_SIZE=512
// GENMC-CALIBRATE: -DHG_HARNESS_CALIBRATE_END
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
// NO --unroll. The workers' loops are spin loops on the park word and the job deques, which the
// spin-assume transformation turns into assumes; every other loop is finite, so no execution is
// cut at a bound. Every weak compare-exchange has its spurious failures explored, at most one
// per execution over all threads (--weak-cas-spurious-total=1): with one in a row per thread
// and address instead, the run gave no verdict in 3,600 s, because construction and worker
// start alone run dozens of weak CASes and each doubles the executions.
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
#if defined(HG_HARNESS_CALIBRATE_END)
    // A bound reaches the end of this harness iff the checker reports this assertion; a bound
    // under which it does not kills a thread earlier, and the verdict covers that prefix alone.
    assert(!"the end of the harness is reachable under this bound");
#endif
    return 0;
}
