// S3 GATE: does the online reconstruction recover ALL FOUR observables, in BOTH TR views, under
// EVERY EVENT IDENTITY MODE?
//
// The identity axis is what P1.5 turns on. There are two causal attribution mechanisms: the
// raw-edge-id rendezvous in rewriter.cpp, and this reconstruction. Which one runs is decided by
// ParallelEvolutionEngine::configure -- quotient exploration OR Automatic identity picks the
// reconstruction, anything else picks the rendezvous. Collapsing to one mechanism requires the
// reconstruction to reproduce the rendezvous under the modes it does not currently serve, and
// each column below is that comparison for one mode.
//
// Full capture is the reference in every column, so a "**" means the two mechanisms disagree on
// an observable a user can read.
//
// BOTH ARMS READ observable_num_*, not num_events()/causal_graph(). The latter report what is
// MATERIALISED, and under Automatic identity full capture ALSO routes causal through the
// reconstruction -- so reading the materialised counts on one side and the reconstructed counts
// on the other compares two different quantities and calls the difference a defect.
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
struct Mode { const char* name; hgcommon::EventSignatureKeys keys; bool positional; };
static const Mode kModes[] = {
  {"None",      hgcommon::EVENT_SIG_NONE,      false},
  {"Automatic", hgcommon::EVENT_SIG_AUTOMATIC, false},
  {"Full",      hgcommon::EVENT_SIG_FULL,      false},
  {"Transition",hgcommon::EVENT_SIG_TRANSITION,false},
};
static void apply(Hypergraph& hg, const Mode& m){
  hg.set_event_signature_keys(m.keys);
  hg.set_positional_event_identity(m.positional);
}
struct Ref { size_t events, edges, pairs_off, pairs_on, branchial; };
static Ref reference(const WL& w, const Mode& mo){
  Ref r{};
  { Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    apply(hg,mo);
    ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(false);
    for(auto&x:w.rules()) e.add_rule(x); auto in=w.init; e.evolve(in,w.steps);
    r.events=hg.observable_num_events(); r.edges=hg.observable_num_causal_edges();
    r.pairs_off=hg.observable_num_causal_pairs(false);
    r.branchial=hg.observable_num_branchial(); }
  { Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    apply(hg,mo);
    ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(false);
    for(auto&x:w.rules()) e.add_rule(x); auto in=w.init; e.evolve(in,w.steps);
    r.pairs_on=hg.observable_num_causal_pairs(true); }
  return r;
}
static int check(const WL& w,int th,const Mode& mo){
  Ref f=reference(w,mo);
  Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
  apply(hg,mo);
  ParallelEvolutionEngine e(&hg,th); e.set_transitive_reduction(false);
  e.set_explore_from_canonical_states_only(true);
  hg.set_quotient_reconstruction(true);
  for(auto&x:w.rules()) e.add_rule(x); auto in=w.init; e.evolve(in,w.steps);
  size_t ev=hg.observable_num_events(), ed=hg.observable_num_causal_edges();
  size_t poff=hg.observable_num_causal_pairs(false), pon=hg.observable_num_causal_pairs(true);
  size_t br=hg.observable_num_branchial();
  auto m=[](size_t a,size_t b){ return a==b?"ok":"**"; };
  const int bad = (f.events!=ev)+(f.edges!=ed)+(f.pairs_off!=poff)+(f.pairs_on!=pon)+(f.branchial!=br);
  printf("%-10s %-9s T=%-2d | ev %5zu/%-5zu %s | edges %5zu/%-5zu %s | pairs(TRoff) %5zu/%-5zu %s | pairs(TRon) %5zu/%-5zu %s | br %5zu/%-5zu %s\n",
    mo.name,w.name,th, f.events,ev,m(f.events,ev), f.edges,ed,m(f.edges,ed),
    f.pairs_off,poff,m(f.pairs_off,poff), f.pairs_on,pon,m(f.pairs_on,pon), f.branchial,br,m(f.branchial,br));
  fflush(stdout);
  return bad;
}
int main(){
  printf("S3 gate: full-capture / reconstructed   (ev, per-edge causal, pairs un-reduced, pairs reduced, branchial)\n\n");
  WL wls[]={ {"pSplit",pSplit,{{0,1}},4}, {"pWolfram",pWolfram,{{0,1},{0,2}},4},
             {"pDupe",pDupe,{{0,1}},4}, {"selfLoop",selfLoop,{{0,0}},4},
             {"iFlip",iFlip,{{0,1}},4}, {"iShift",iShift,{{0,1},{1,2}},4},
             {"rMerge",rMerge,{{0,1},{1,2},{2,0}},3}, {"mixed1",mixed1,{{0,1}},4},
             {"mixed2",mixed2,{{0,1}},4}, {"TC6",TC6,{{1,2},{2,3}},4} };
  int bad=0, cells=0;
  for(const auto& mo:kModes){
    printf("--- event identity: %s ---\n", mo.name);
    for(int t:{1,8}){ for(auto&w:wls){ bad+=check(w,t,mo); ++cells; } printf("\n"); }
  }
  printf("TOTAL: %d disagreeing observables over %d configurations\n", bad, cells);
  return bad==0?0:1;
}
