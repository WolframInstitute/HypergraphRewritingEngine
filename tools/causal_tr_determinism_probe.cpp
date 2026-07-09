#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <set>
#include <vector>
using namespace hypergraph;
static RewriteRule wr(){ return make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build(); }
static size_t run(int th){ Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, th); e.set_transitive_reduction(true); e.set_explore_from_canonical_states_only(true);
    e.add_rule(wr()); std::vector<std::vector<VertexId>> in={{0u,1u},{0u,2u}}; e.evolve(in,5);
    return hg.causal_graph().get_causal_edges().size(); }
int main(){ printf("single-thread (reference): %zu\n", run(1));
    std::set<size_t> s; for(int i=0;i<20;++i) s.insert(run(8));
    printf("8-thread over 20 runs: "); for(size_t v:s) printf("%zu ",v); printf("(%zu distinct)\n", s.size()); return 0; }
