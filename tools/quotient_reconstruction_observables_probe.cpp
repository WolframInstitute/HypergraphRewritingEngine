// S3 GATE: does the online reconstruction recover ALL FOUR observables, in BOTH TR views?
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
struct Ref { size_t events, edges, pairs_off, pairs_on, branchial; };
static Ref reference(const WL& w){
  Ref r{};
  { Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(false);
    for(auto&x:w.rules()) e.add_rule(x); auto in=w.init; e.evolve(in,w.steps);
    r.events=hg.num_events(); r.edges=hg.causal_graph().num_causal_edges();
    r.pairs_off=hg.causal_graph().num_causal_event_pairs();
    r.branchial=hg.causal_graph().num_branchial_edges(); }
  { Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(false);
    for(auto&x:w.rules()) e.add_rule(x); auto in=w.init; e.evolve(in,w.steps);
    r.pairs_on=hg.causal_graph().num_causal_event_pairs(); }
  return r;
}
static void check(const WL& w,int th){
  Ref f=reference(w);
  Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
  ParallelEvolutionEngine e(&hg,th); e.set_transitive_reduction(false);
  e.set_explore_from_canonical_states_only(true);
  hg.set_quotient_reconstruction(true);
  for(auto&x:w.rules()) e.add_rule(x); auto in=w.init; e.evolve(in,w.steps);
  size_t ev=hg.num_reconstructed_events(), ed=hg.num_reconstructed_causal_edges();
  size_t poff=hg.num_reconstructed_causal_pairs(false), pon=hg.num_reconstructed_causal_pairs(true);
  size_t br=hg.num_reconstructed_branchial();
  auto m=[](size_t a,size_t b){ return a==b?"ok":"**"; };
  printf("%-9s T=%-2d | ev %5zu/%-5zu %s | edges %5zu/%-5zu %s | pairs(TRoff) %5zu/%-5zu %s | pairs(TRon) %5zu/%-5zu %s | br %5zu/%-5zu %s\n",
    w.name,th, f.events,ev,m(f.events,ev), f.edges,ed,m(f.edges,ed),
    f.pairs_off,poff,m(f.pairs_off,poff), f.pairs_on,pon,m(f.pairs_on,pon), f.branchial,br,m(f.branchial,br));
}
int main(){
  printf("S3 gate: full-capture / reconstructed   (ev, per-edge causal, pairs un-reduced, pairs reduced, branchial)\n\n");
  WL wls[]={ {"pSplit",pSplit,{{0,1}},4}, {"pWolfram",pWolfram,{{0,1},{0,2}},4},
             {"pDupe",pDupe,{{0,1}},4}, {"selfLoop",selfLoop,{{0,0}},4},
             {"iFlip",iFlip,{{0,1}},4}, {"iShift",iShift,{{0,1},{1,2}},4},
             {"rMerge",rMerge,{{0,1},{1,2},{2,0}},3}, {"mixed1",mixed1,{{0,1}},4},
             {"mixed2",mixed2,{{0,1}},4}, {"TC6",TC6,{{1,2},{2,3}},4} };
  for(int t:{1,8}){ for(auto&w:wls) check(w,t); printf("\n"); }
  return 0;
}
