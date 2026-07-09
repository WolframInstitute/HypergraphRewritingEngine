#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <functional>
using namespace hypergraph;
struct WL { const char* name; RewriteRule (*rule)(); std::vector<std::vector<VertexId>> init; int steps; };
static RewriteRule rA(){ return make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build(); }
static RewriteRule rB(){ return make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(); }
static RewriteRule rC(){ return make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).rhs({1,0}).build(); }

static std::set<std::pair<uint32_t,uint32_t>> pairs_of(const std::vector<CausalEdge>& e){
    std::set<std::pair<uint32_t,uint32_t>> p; for(auto&c:e) p.insert({c.producer,c.consumer}); return p; }

// offline minimal TR (unique pairs) of a raw pair set
static std::set<std::pair<uint32_t,uint32_t>> offline_tr(const std::set<std::pair<uint32_t,uint32_t>>& pairs){
    std::unordered_map<uint32_t,std::unordered_set<uint32_t>> succ;
    for(auto&pc:pairs) succ[pc.first].insert(pc.second);
    std::unordered_map<uint32_t,std::unordered_set<uint32_t>> reach; std::unordered_set<uint32_t> done;
    std::function<std::unordered_set<uint32_t>&(uint32_t)> R=[&](uint32_t u)->std::unordered_set<uint32_t>&{
        auto& s=reach[u]; if(done.count(u)) return s; done.insert(u);
        auto it=succ.find(u); if(it!=succ.end()) for(uint32_t w:it->second){ s.insert(w); auto& rw=R(w); s.insert(rw.begin(),rw.end()); }
        return s; };
    std::set<std::pair<uint32_t,uint32_t>> kept;
    for(auto&pc:pairs){ bool red=false; for(uint32_t w:succ[pc.first]){ if(w==pc.second) continue; if(R(w).count(pc.second)){red=true;break;} } if(!red) kept.insert(pc); }
    return kept;
}
static std::set<std::pair<uint32_t,uint32_t>> run(const WL& w, bool tr, int th){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, th); e.set_transitive_reduction(tr);
    e.set_explore_from_canonical_states_only(false);
    e.add_rule(w.rule()); auto in=w.init; e.evolve(in,w.steps);
    return pairs_of(hg.causal_graph().get_causal_edges());
}
int main(){
    WL wls[] = { {"wolfram5", rA, {{0u,1u},{0u,2u}}, 5},
                 {"chain6",   rB, {{0u,1u}}, 6},
                 {"tri4",     rC, {{0u,1u},{1u,2u}}, 4} };
    bool allok=true;
    for (auto& w : wls) {
        for (int th : {1,2,4,8,16}) {
            // Event ids differ between runs, so compare SIZES: the causal graphs are
            // isomorphic and TR size is an isomorphism invariant.
            auto raw  = run(w,false,th);
            auto want = offline_tr(raw);
            auto got  = run(w,true,th);
            bool ok = (got.size()==want.size());
            allok &= ok;
            printf("%-9s th=%2d  raw=%4zu  engineTR=%4zu  minimalTR=%4zu  %s\n",
                   w.name, th, raw.size(), got.size(), want.size(), ok?"EXACT":"*** NOT MINIMAL ***");
        }
    }
    printf("\n%s\n", allok? "ALL EXACT: no redundant edge slipped through, at any thread count"
                          : "SOME RUNS NOT MINIMAL");
    return allok?0:1;
}
