// GENMC-LINK: engine
// GenMC harness: the COMPOSED ENGINE, linked from every engine translation unit.
//
// One rule, one initial edge, one step, two workers. The rule matches exactly once, so the run
// has exactly one event and two states -- a function of the rule and the input, so it must hold
// in whichever order the workers interleave. GenMC enumerates the interleavings RC11 permits;
// any that yields a different count trips the assertion.
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

    std::vector<std::vector<hg::engine::VertexId>> init = {{0, 1}};
    e.evolve(init, 1);

    assert(e.num_events() == 1);
    assert(g.num_states() == 2);
    return 0;
}
