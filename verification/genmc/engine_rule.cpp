// GENMC-LINK: engine
// GENMC-ARGS: --unroll=2
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=4 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16 -DHG_QC_CANON_EVENT_SEEN_CAPACITY=16 -DHG_JOB_QUEUE_CAPACITY=16 -DHG_JOB_INJECTOR_CAPACITY=64 -DHG_MAX_ARENA_WORKERS=8 -DHG_KEY_SET_SHARDS=4 -DHG_MAX_PATTERN_EDGES=4 -DHG_MAX_CACHED_SIGS=8 -DHG_ARENA_BLOCK_SIZE=512
//
// GenMC harness: the composed engine with a rule added. Second rung of the ladder
// engine_construct -> engine_rule -> engine_evolve; see engine_construct.cpp for what the ladder
// is and why each rung is its own harness.
//
// add_rule runs the rule analysis (rule_analysis.hpp) and the join planning, which is the first
// engine code past construction that touches shared state -- the rule table the workers will
// read. Measured: 19,477 lines after prune, and the rung at which the interpreter used to stop.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"

#include <cassert>
#include <vector>

int main() {
    hg::engine::Hypergraph g;
    g.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::None);
    hg::engine::RewriteRule rule = hg::engine::make_rule(0)
        .lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();
    hg::engine::ParallelEvolutionEngine e(&g, 2);
    e.add_rule(rule);
    assert(e.num_events() == 0);
#if defined(HG_HARNESS_CALIBRATE_END)
    // A bound reaches the end of this harness iff the checker reports this assertion; a bound
    // under which it does not kills a thread earlier, and the verdict covers that prefix alone.
    assert(!"the end of the harness is reachable under this bound");
#endif
    return 0;
}
