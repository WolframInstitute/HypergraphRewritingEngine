// S2 GATE: does the ONLINE per-instance reconstruction recover full-capture's raw event count?
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
using namespace hypergraph;
using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;
struct WL { const char* name; Rules (*rules)(); Init init; int steps; };
static Rules pSplit(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules pWolfram(){ return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build() }; }
static Rules pDupe(){ return { make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build() }; }
static Rules selfLoop(){ return { make_rule(0).lhs({0,0}).rhs({0,0}).rhs({0,0}).build() }; }
static Rules iFlip(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build() }; }
static Rules iShift(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules rMerge(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mixed1(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mixed2(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules TC6(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,2}).rhs({1,3}).build() }; }
static size_t full_events(const WL& w){
  Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
  ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(false);
  e.set_explore_from_canonical_states_only(false);
  for(auto&r:w.rules()) e.add_rule(r); auto in=w.init; e.evolve(in,w.steps);
  return hg.num_events();
}
static void check(const WL& w,int threads){
  size_t f=full_events(w);
  Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
  ParallelEvolutionEngine e(&hg,threads); e.set_transitive_reduction(false);
  e.set_explore_from_canonical_states_only(true);
  hg.set_quotient_reconstruction(true);
  for(auto&r:w.rules()) e.add_rule(r); auto in=w.init; e.evolve(in,w.steps);
  size_t r=hg.num_reconstructed_events();
  printf("%-9s s=%d T=%-2d | full %6zu | skeleton %5zu | reconstructed %6zu  %s\n",
    w.name,w.steps,threads,f,hg.num_events(),r, r==f?"OK":"** MISMATCH **");
}
int main(){
  WL wls[]={ {"pSplit",pSplit,{{0,1}},4}, {"pWolfram",pWolfram,{{0,1},{0,2}},4},
             {"pDupe",pDupe,{{0,1}},4}, {"selfLoop",selfLoop,{{0,0}},4},
             {"iFlip",iFlip,{{0,1}},4}, {"iShift",iShift,{{0,1},{1,2}},4},
             {"rMerge",rMerge,{{0,1},{1,2},{2,0}},3}, {"mixed1",mixed1,{{0,1}},4},
             {"mixed2",mixed2,{{0,1}},4}, {"TC6",TC6,{{1,2},{2,3}},4} };
  for(int t : {1,8}) { for(auto&w:wls) check(w,t); printf("\n"); }
  return 0;
}
