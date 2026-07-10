#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include <cstdio>
#include <map>
#include <unordered_map>
#include <vector>
#include <set>
using namespace hypergraph;
static RewriteRule wr(){ return make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build(); }
static uint64_t mix(uint64_t h, uint64_t v){ h^=v; h*=1099511628211ull; return h; }

// canonical causal multiset: multiset of (canonical producer event, canonical consumer event)
struct Res { size_t triples; size_t distinct_pairs; uint64_t fp; size_t events; size_t canon_events; uint64_t ce_fp; };
static Res run(bool explore, bool tr, int th){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, th);
    e.set_transitive_reduction(tr);
    e.set_explore_from_canonical_states_only(explore);
    e.add_rule(wr());
    std::vector<std::vector<VertexId>> in={{0u,1u},{0u,2u}};
    e.evolve(in,5);

    IRCanonicalizer ir;
    std::unordered_map<uint32_t,uint64_t> sh;
    for (uint32_t sid=0; sid<hg.num_states(); ++sid){
        const auto& st = hg.get_state(sid);
        if (st.id == INVALID_ID) continue;
        std::vector<std::vector<VertexId>> edges;
        st.edges.for_each([&](EdgeId eid){ const auto& ed=hg.get_edge(eid);
            std::vector<VertexId> v; v.reserve(ed.arity);
            for(uint8_t i=0;i<ed.arity;++i) v.push_back(ed.vertices[i]);
            edges.push_back(std::move(v)); });
        sh[sid] = ir.compute_canonical_hash(edges);
    }
    std::unordered_map<uint32_t,uint64_t> ek;
    size_t nev=0;
    for (uint32_t eid=0; eid<hg.num_events(); ++eid){
        const auto& ev = hg.get_event(eid);
        if (ev.id == INVALID_ID) continue;
        ++nev;
        uint64_t ih = sh.count(ev.input_state)? sh[ev.input_state]:0;
        uint64_t oh = sh.count(ev.output_state)? sh[ev.output_state]:0;
        uint32_t step = hg.get_state(ev.output_state).step;
        uint64_t k = 1469598103934665603ull;
        k=mix(k,ih); k=mix(k,oh); k=mix(k,ev.rule_index); k=mix(k,step);
        ek[eid]=k;
    }
    std::map<std::pair<uint64_t,uint64_t>, size_t> ms;
    size_t tri=0;
    for (auto& c : hg.causal_graph().get_causal_edges()){
        if (c.producer==INVALID_ID||c.consumer==INVALID_ID) continue;
        auto p=ek.find(c.producer), q=ek.find(c.consumer);
        if(p==ek.end()||q==ek.end()) continue;
        ms[{p->second,q->second}]++; ++tri;
    }
    uint64_t fp=1469598103934665603ull;
    for (auto& kv : ms){ fp=mix(fp,kv.first.first); fp=mix(fp,kv.first.second); fp=mix(fp,kv.second); }
    std::set<uint64_t> ce; for (auto& kv : ek) ce.insert(kv.second);
    uint64_t cefp=1469598103934665603ull; for (uint64_t v : ce) cefp=mix(cefp,v);
    return {tri, ms.size(), fp, nev, ce.size(), cefp};
}
int main(){
    for (bool ex : {false,true}) for (bool tr : {false,true}) {
        std::set<uint64_t> fps, cefps; std::set<size_t> tris, dps, evs, ces;
        for(int th : {1,8}) for(int i=0;i<3;++i){ auto r=run(ex,tr,th); fps.insert(r.fp); tris.insert(r.triples); dps.insert(r.distinct_pairs); evs.insert(r.events); ces.insert(r.canon_events); cefps.insert(r.ce_fp); }
        printf("explore=%d TR=%d | raw events:", ex,tr); for(size_t v:evs)printf(" %zu",v);
        printf(" | CANONICAL events:"); for(size_t v:ces)printf(" %zu",v);
        printf(" (set %s)", cefps.size()==1?"STABLE":"VARIES");
        printf(" | canon-causal triples:"); for(size_t v:tris)printf(" %zu",v);
        printf(" | distinct canon pairs:"); for(size_t v:dps)printf(" %zu",v);
        printf(" | multiset %s\n", fps.size()==1?"STABLE":"VARIES");
    }
    return 0;
}
