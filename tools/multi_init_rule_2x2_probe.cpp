#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <vector>
using namespace hypergraph;
static RewriteRule r1(){ return make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build(); }
static RewriteRule r2(){ return make_rule(1).lhs({0,1}).rhs({1,0}).build(); }
using Init = std::vector<std::vector<VertexId>>;
static void run(const char* name, std::vector<RewriteRule> rules,
                std::vector<Init> roots, int steps){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, 1); e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(false);  // full multiway = reference
    for (auto& r : rules) e.add_rule(r);
    if (roots.size()==1) e.evolve(roots[0], steps);
    else {
        std::vector<Init> rs = roots; e.evolve(rs, steps);
    }
    printf("%-14s | states=%zu events=%zu\n", name, hg.num_canonical_states(), hg.num_events());
}
int main(){
    Init i1={{0u,1u},{0u,2u}}, i2={{0u,1u},{1u,2u}};
    int s=3;
    printf("ORACLE: 1x1 states=10 ev=22 | 1x2 states=28 ev=116 | 2x1 states=11 ev=22 | 2x2 states=28 ev=144\n");
    run("1init x 1rule", {r1()}, {i1}, s);
    run("1init x 2rule", {r1(),r2()}, {i1}, s);
    run("2init x 1rule", {r1()}, {i1,i2}, s);
    run("2init x 2rule", {r1(),r2()}, {i1,i2}, s);
    return 0;
}
