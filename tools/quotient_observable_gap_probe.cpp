// What does quotient ACTUALLY report today vs full-capture, on every observable?
// This is the "us vs our oracle" question for quotient mode: the contract (SPEC 5.4) says the
// observable output must be IDENTICAL. Measure the gap so the reconstruction wiring has an exact target.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
using namespace hypergraph;
using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;
struct WL { const char* name; Rules (*rules)(); Init init; int steps; };

static Rules pSplit(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules pWolfram(){ return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build() }; }
static Rules iFlip(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build() }; }
static Rules iShift(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules rMerge(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mixed1(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules TC6(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,2}).rhs({1,3}).build() }; }

struct Obs { size_t states, events, causal_pairs, causal_edges, branchial; };
static Obs run(const WL& w, bool quotient, bool tr){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,4);
    e.set_transitive_reduction(tr);
    e.set_explore_from_canonical_states_only(quotient);
    for(auto&r:w.rules()) e.add_rule(r);
    auto in=w.init; e.evolve(in,w.steps);
    Obs o;
    o.states = hg.num_canonical_states();
    o.events = hg.num_events();
    o.causal_pairs = hg.causal_graph().num_causal_event_pairs();
    o.causal_edges = hg.causal_graph().num_causal_edges();
    o.branchial = hg.causal_graph().num_branchial_edges();
    return o;
}
static void cmp(const WL& w){
    Obs f = run(w,false,false), q = run(w,true,false);
    auto mk=[](size_t a,size_t b){ return a==b ? "  " : "<<"; };
    printf("%-9s s=%d | states %5zu/%-5zu%s | events %6zu/%-6zu%s | causalPairs %6zu/%-6zu%s | branchial %6zu/%-6zu%s\n",
           w.name,w.steps,
           f.states,q.states,mk(f.states,q.states),
           f.events,q.events,mk(f.events,q.events),
           f.causal_pairs,q.causal_pairs,mk(f.causal_pairs,q.causal_pairs),
           f.branchial,q.branchial,mk(f.branchial,q.branchial));
}
int main(){
    printf("format: full/quotient  ('<<' = MISMATCH, quotient differs from full-capture)  [TR off]\n\n");
    WL wls[]={ {"pSplit",pSplit,{{0,1}},4}, {"pWolfram",pWolfram,{{0,1},{0,2}},4},
               {"iFlip",iFlip,{{0,1}},4}, {"iShift",iShift,{{0,1},{1,2}},4},
               {"rMerge",rMerge,{{0,1},{1,2},{2,0}},3}, {"mixed1",mixed1,{{0,1}},4},
               {"TC6",TC6,{{1,2},{2,3}},4} };
    for(auto&w:wls) cmp(w);
    return 0;
}
