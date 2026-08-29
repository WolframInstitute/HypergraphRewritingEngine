// GENMC-LINK: engine
// GENMC-ARGS: --unroll=2048
// GENMC-DEFINES: -DHG_SEGMENTED_ARRAY_MAX_SEGMENTS=8 -DHG_SEGMENTED_ARRAY_MAX_SHIFT=4 -DHG_CONCURRENT_MAP_INITIAL_CAPACITY=16 -DHG_QC_CANON_EVENT_SEEN_CAPACITY=16 -DHG_JOB_QUEUE_CAPACITY=16 -DHG_JOB_INJECTOR_CAPACITY=64 -DHG_MAX_ARENA_WORKERS=8 -DHG_KEY_SET_SHARDS=4 -DHG_MAX_PATTERN_EDGES=4 -DHG_MAX_CACHED_SIGS=8 -DHG_ARENA_BLOCK_SIZE=512
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
//                               once: 1 event, 2 states.
//   -DHG_EVOLVE_TWO_STEP=1      two workers, two steps, no canonicalisation, the one-edge rule:
//                               the smallest run that forwards a parent's matches to its child
//                               and delta-matches produced edges (3 events, 4 raw states).
//   -DHG_EVOLVE_CANON_SHAPE=1   the two-edge rule under Full canonicalisation, two workers, one
//                               step: the middle rung between the default and the live shape.
//                               Measured 2026-08-29: the live shape's first execution did not
//                               complete in 23 minutes of exploration at --unroll=2048.
//   -DHG_EVOLVE_TWO_STEP_CANON=1 two workers, two steps, Full canonicalisation, the two-edge
//                               rule: forwarding and delta matching under the canonicaliser and
//                               the dedup rendezvous. 3 events, 4 raw states, 3 canonical.
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
//   -DHG_HARNESS_CALIBRATE_END=1 either arm, with an assertion that FAILS after the contract
//                               asserts. The checker must report it: a bound under which it
//                               does not is a bound that never reaches the end of evolve(),
//                               and the verdict of the property arm under that bound covers
//                               only the prefix the bound allows.
//
// WHAT IS BOUNDED. Every loop is bounded to --unroll iterations per entry; a thread that
// exceeds the bound is KILLED there, and an execution whose threads were killed still counts
// as complete with no error. So a bound has to be shown to reach the end before its verdict
// means anything, which is what the calibration arm is for. Measured 2026-08-29 on the live
// arm at --unroll=2 (execution graph of the saved module): the main thread is killed after
// 1,237 events inside construction, before any worker thread exists, so "No errors, 2
// executions" at that bound covered construction alone. The bound is baked into the module by
// the checker's unroll pass, so each bound is its own transform. The module is the fully
// inlined engine (43M LLVM instructions before the checker's own passes), so this is run with
// HG_GENMC_PROGRESS and saved with --output-llvm-after=<file>.bc once per bound and arm.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"
#include <cassert>
#include <vector>

int main() {
    hg::engine::Hypergraph g;
#if defined(HG_EVOLVE_TWO_STEP)
    // Two workers, TWO steps, no canonicalisation, the one-edge rule: the smallest run in
    // which a child inherits its parent's matches (forwarding) and delta-matches its produced
    // edges -- the path the lost-event defect lived on. Step 1: one match, one child with two
    // edges; step 2: two matches on that child, two grandchildren. 3 events, 4 raw states.
    g.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::None);
    hg::engine::RewriteRule rule = hg::engine::make_rule(0)
        .lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();
    hg::engine::ParallelEvolutionEngine e(&g, 2);
    e.add_rule(rule);
    std::vector<std::vector<hg::engine::VertexId>> init = {{0, 1}};
    e.evolve(init, 2);
    assert(e.num_events() == 3);
    assert(g.num_states() == 4);
#elif defined(HG_EVOLVE_CANON_SHAPE)
    // The middle rung: the two-edge rule under Full canonicalisation, TWO workers, ONE step --
    // the canonicaliser and the dedup rendezvous on the smallest run that reaches them.
    g.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::Full);
    hg::engine::RewriteRule rule = hg::engine::make_rule(0)
        .lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build();
    hg::engine::ParallelEvolutionEngine e(&g, 2);
    e.add_rule(rule);
    std::vector<std::vector<hg::engine::VertexId>> init = {{0, 1}, {1, 2}};
    e.evolve(init, 1);
    assert(e.num_events() == 1);
    assert(g.num_states() == 2);
#elif defined(HG_EVOLVE_TWO_STEP_CANON)
    // Two workers, TWO steps, Full canonicalisation, the two-edge rule: forwarding, delta
    // matching, the canonicaliser and the dedup rendezvous on one run. Step 1: one match, one
    // child (the path 0-1-3-2). Step 2: two matches on that child, two grandchildren that are
    // both the five-vertex path, so the canonical dedup merges them. 3 events, 4 raw states,
    // 3 canonical.
    g.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::Full);
    hg::engine::RewriteRule rule = hg::engine::make_rule(0)
        .lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build();
    hg::engine::ParallelEvolutionEngine e(&g, 2);
    e.add_rule(rule);
    std::vector<std::vector<hg::engine::VertexId>> init = {{0, 1}, {1, 2}};
    e.evolve(init, 2);
    assert(e.num_events() == 3);
    assert(g.num_states() == 4);
#elif defined(HG_EVOLVE_LIVE_SHAPE)
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
#if defined(HG_HARNESS_CALIBRATE_END)
    assert(!"the end of evolve() is reachable under this bound");
#endif
    return 0;
}
