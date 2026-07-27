// Isolate: is the quotient causal non-determinism the TR-on reduction order-dependence,
// or the underlying qc_* reconstruction? Replicates the determinism gate's fingerprint
// but sweeps TR on AND off. If causal spread==1 with TR off, the reconstruction is
// deterministic and the bug is purely the online TR reduction order (stage 3).
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include <vector>
#include <set>
#include <algorithm>
#include <cstdio>
namespace hgraph = hypergraph;   // hg is the engine's atomic_compat namespace
static uint64_t fnv(uint64_t h, uint64_t x){ h^=x; h*=1099511628211ULL; return h; }

static uint64_t causal_fp(hgraph::Hypergraph& g){
    auto canon=[&](hgraph::StateId s){ return s==hgraph::INVALID_ID?0:g.get_or_compute_canonical_hash(s); };
    auto esig=[&](hgraph::EventId e){ const hgraph::Event& x=g.get_event(e);
        return fnv(fnv(fnv(1469598103934665603ULL,canon(x.input_state)),canon(x.output_state)),x.rule_index); };
    std::vector<uint64_t> ce;
    for(const auto& c: g.causal_graph().get_causal_edges()){
        if(c.producer==hgraph::INVALID_ID||c.consumer==hgraph::INVALID_ID) continue;
        ce.push_back(fnv(fnv(0,esig(c.producer)),esig(c.consumer))); }
    std::sort(ce.begin(),ce.end());
    uint64_t fp=1469598103934665603ULL; for(uint64_t v:ce) fp=fnv(fp,v); return fp;
}

static void run_wl(const char* name, const std::vector<hgraph::RewriteRule>& rules,
                   const std::vector<std::vector<hgraph::VertexId>>& init, int steps){
    for(bool tr : {false, true}){
        std::set<uint64_t> spread; std::set<long> ncount;
        for(uint64_t seed : {uint64_t(0xABCDEF), uint64_t(0)})
            for(int rep=0; rep<4; ++rep)
                for(int th : {1,2,8}){
                    hgraph::Hypergraph g; g.set_state_canonicalization_mode(hgraph::StateCanonicalizationMode::Full);
                    hgraph::ParallelEvolutionEngine e(&g, th);
                    e.set_transitive_reduction(tr);
                    e.set_explore_from_canonical_states_only(true);
                    e.set_random_seed(seed);
                    for(const auto& r: rules) e.add_rule(r);
                    auto in=init; e.evolve(in, steps);
                    spread.insert(causal_fp(g));
                    long n=0; for(const auto& c: g.causal_graph().get_causal_edges())
                        if(c.producer!=hgraph::INVALID_ID&&c.consumer!=hgraph::INVALID_ID) ++n;
                    ncount.insert(n);
                }
        printf("%-8s TR=%-3s  distinct_causal_fp=%-2zu  distinct_edge_count=%-2zu  %s\n",
               name, tr?"on":"off", spread.size(), ncount.size(),
               spread.size()==1?"DETERMINISTIC":"*** NON-DET ***");
    }
}
int main(){
    run_wl("WPP", {hgraph::make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
           {{0,1},{0,2}}, 6);
    run_wl("mixed1", {hgraph::make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                      hgraph::make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                      hgraph::make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()}, {{0,1}}, 6);
    run_wl("mixed2", {hgraph::make_rule(0).lhs({0,1}).rhs({1,0}).build(),
                      hgraph::make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()}, {{0,1}}, 6);
    return 0;
}
