// De-risk: does depth-ordered TR of the reconstructed quotient causal graph reproduce
// full-capture's TR-on causal set (projected to canonical event signatures)?
// Nodes = (transition signature te=fnv(from,to,rule), depth). Edges reconstructed by the
// depth-indexed producer-support DP over the quotient skeleton. TR uses depth as the
// (deterministic, acyclic) topological order. Project (drop depth) -> esig pairs, compare
// to full-capture(TR on) -> esig pairs.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
using namespace hypergraph;
using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;
struct WL { const char* name; Rules (*rules)(); Init init; int steps; };

static Rules WPP(){ return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build() }; }
static Rules mixed1(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mixed2(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules TC6(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,2}).rhs({1,3}).build() }; }

static uint64_t mixh(uint64_t h,uint64_t v){ h^=v; h*=1099511628211ull; return h; }
static uint64_t te_of(uint64_t from,uint64_t to,uint32_t rule){
    uint64_t k=1469598103934665603ull; k=mixh(k,from); k=mixh(k,to); k=mixh(k,rule); return k; }

struct Node { uint64_t te; int d; bool operator<(const Node&o)const{ return te!=o.te?te<o.te:d<o.d; }
              bool operator==(const Node&o)const{ return te==o.te&&d==o.d; } };
static const Node INIT{0,-1};

// generic depth-ordered TR of a (Node) edge set -> projected esig-pair set
static std::set<std::pair<uint64_t,uint64_t>> reduce_depthTR(const std::set<std::pair<Node,Node>>& edges){
    std::map<Node,std::vector<Node>> adj; for(auto&e:edges) adj[e.first].push_back(e.second);
    auto reach2=[&](const Node&u,const Node&v)->bool{
        std::set<Node> vis; std::vector<Node> st;
        for(auto&w2:adj[u]) if(!(w2==v)){ st.push_back(w2); vis.insert(w2); }
        while(!st.empty()){ Node x=st.back(); st.pop_back(); if(x==v) return true;
            auto a=adj.find(x); if(a==adj.end()) continue;
            for(auto&y:a->second) if(y.d<=v.d && vis.insert(y).second) st.push_back(y); }
        return false; };
    std::set<std::pair<uint64_t,uint64_t>> R;
    for(auto&e:edges){ if(e.first==e.second){ R.insert({e.first.te,e.second.te}); continue; }
        if(!reach2(e.first,e.second)) R.insert({e.first.te,e.second.te}); }
    return R;
}

// ---- full-capture, TWO reductions ----
// (a) reduce(raw) then project  = the reference/raw-forest convention
// (b) reduce over (esig,depth)   = the canonical-graph convention (should == quotient)
static void fullcap(const WL& w, std::set<std::pair<uint64_t,uint64_t>>& raw, std::set<std::pair<uint64_t,uint64_t>>& depth){
    // (a) reduce(raw): engine TR on, project
    { Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
      ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(true); e.set_explore_from_canonical_states_only(false);
      for(auto&r:w.rules()) e.add_rule(r); auto in=w.init; e.evolve(in,w.steps);
      auto esig=[&](EventId ev){ const Event& x=hg.get_event(ev);
          return te_of(hg.get_or_compute_canonical_hash(x.input_state),hg.get_or_compute_canonical_hash(x.output_state),x.rule_index); };
      for(const auto& c: hg.causal_graph().get_causal_edges()){ if(c.producer==INVALID_ID||c.consumer==INVALID_ID) continue;
          raw.insert({esig(c.producer),esig(c.consumer)}); } }
    // (b) reduce over (esig,depth): engine TR OFF, tag each event by (esig, output step), reduce
    { Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
      ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(false); e.set_explore_from_canonical_states_only(false);
      for(auto&r:w.rules()) e.add_rule(r); auto in=w.init; e.evolve(in,w.steps);
      auto node=[&](EventId ev)->Node{ const Event& x=hg.get_event(ev);
          uint64_t te=te_of(hg.get_or_compute_canonical_hash(x.input_state),hg.get_or_compute_canonical_hash(x.output_state),x.rule_index);
          return Node{te, (int)hg.get_state(x.output_state).step}; };
      std::set<std::pair<Node,Node>> edges;
      for(const auto& c: hg.causal_graph().get_causal_edges()){ if(c.producer==INVALID_ID||c.consumer==INVALID_ID) continue;
          edges.insert({node(c.producer), node(c.consumer)}); }
      depth = reduce_depthTR(edges); }
}

struct StateInfo { uint64_t hash; std::unordered_map<uint32_t,uint32_t> edge_orbit; std::vector<uint32_t> orbit_size; };
static StateInfo describe(const Hypergraph& hg, IRCanonicalizer& ir, uint32_t sid){
    StateInfo si; std::vector<std::vector<VertexId>> es; std::vector<uint32_t> ids;
    hg.get_state(sid).edges.for_each([&](EdgeId eid){ const auto& e=hg.get_edge(eid);
        std::vector<VertexId> v; for(uint8_t i=0;i<e.arity;++i) v.push_back(e.vertices[i]); es.push_back(v); ids.push_back(eid); });
    std::vector<uint32_t> orb; si.hash=ir.compute_canonical_hash_with_edge_orbits(es,orb);
    uint32_t no=0; for(uint32_t o:orb) no=std::max(no,o+1); si.orbit_size.assign(no,0);
    for(uint32_t o:orb) si.orbit_size[o]++; for(size_t i=0;i<ids.size();++i) si.edge_orbit[ids[i]]=orb.empty()?0:orb[i];
    return si;
}

// Node = (te, depth); INIT sentinel = {0,-1}
static void check(const WL& w){
    std::set<std::pair<uint64_t,uint64_t>> fc_raw, fc_depth; fullcap(w, fc_raw, fc_depth);

    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,8); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(true);
    for(auto&r:w.rules()) e.add_rule(r);
    auto init=w.init; e.evolve(init,w.steps);
    IRCanonicalizer ir;
    std::unordered_map<uint32_t,StateInfo> si;
    for(uint32_t s=0;s<hg.num_states();++s) if(hg.get_state(s).id!=INVALID_ID) si[s]=describe(hg,ir,s);
    std::unordered_map<uint64_t,std::vector<uint32_t>> osz; for(auto&kv:si) osz[kv.second.hash]=kv.second.orbit_size;

    struct CE { uint64_t from,to; uint32_t rule; std::vector<uint32_t> consumed, produced_orbit;
                std::vector<std::pair<uint32_t,uint32_t>> survivors; uint64_t te()const{return te_of(from,to,rule);} };
    std::map<std::string,CE> ces;
    for(uint32_t i=0;i<hg.num_events();++i){ const auto& ev=hg.get_event(i); if(ev.id==INVALID_ID) continue;
        const StateInfo& in_s=si[ev.input_state]; const StateInfo& out_s=si[ev.output_state];
        CE c; c.from=in_s.hash; c.to=out_s.hash; c.rule=ev.rule_index;
        for(uint8_t j=0;j<ev.num_consumed;++j) c.consumed.push_back(in_s.edge_orbit.at(ev.consumed_edges[j]));
        std::sort(c.consumed.begin(),c.consumed.end());
        std::set<uint32_t> prod(ev.produced_edges,ev.produced_edges+ev.num_produced);
        for(uint8_t j=0;j<ev.num_produced;++j) c.produced_orbit.push_back(out_s.edge_orbit.at(ev.produced_edges[j]));
        for(auto&kv:out_s.edge_orbit) if(!prod.count(kv.first)&&in_s.edge_orbit.count(kv.first))
            c.survivors.push_back({in_s.edge_orbit.at(kv.first),kv.second});
        std::sort(c.survivors.begin(),c.survivors.end());
        char buf[512]; int n=snprintf(buf,sizeof buf,"%llu|%llu|%u|",(unsigned long long)c.from,(unsigned long long)c.to,c.rule);
        for(uint32_t o:c.consumed) n+=snprintf(buf+n,sizeof(buf)-n,"c%u",o);
        for(auto&pr:c.survivors) n+=snprintf(buf+n,sizeof(buf)-n,"s%u>%u",pr.first,pr.second);
        ces.emplace(buf,c);
    }
    std::unordered_map<uint64_t,std::vector<const CE*>> out_of; for(auto&kv:ces) out_of[kv.second.from].push_back(&kv.second);

    // support DP over (state,depth,orbit) -> set of producer Nodes; emit depth-tagged causal edges
    uint64_t s0=si[0].hash;
    std::map<std::tuple<uint64_t,int,uint32_t>, std::set<Node>> D;
    std::set<std::pair<uint64_t,int>> reached; reached.insert({s0,0});
    for(uint32_t j=0;j<osz[s0].size();++j) D[{s0,0,j}].insert(INIT);
    std::set<std::pair<Node,Node>> edges;
    for(int k=0;k<w.steps;++k){
        for(auto it=reached.begin(); it!=reached.end(); ++it){
            if(it->second!=k) continue; uint64_t s=it->first;
            auto oit=out_of.find(s); if(oit==out_of.end()) continue;
            for(const CE* c: oit->second){
                Node cnode{c->te(), k+1};
                // causal: each consumed orbit's producers -> this event
                std::set<uint32_t> cons(c->consumed.begin(),c->consumed.end());
                for(uint32_t j: cons){ auto d=D.find({s,k,j}); if(d==D.end()) continue;
                    for(const Node& p: d->second) if(!(p==INIT)) edges.insert({p, cnode}); }
                reached.insert({c->to,k+1});
                // produced edges produced by this event at child depth
                for(uint32_t o: c->produced_orbit) D[{c->to,k+1,o}].insert(cnode);
                // survivors carry producer sets forward
                for(auto&pr:c->survivors){ auto d=D.find({s,k,pr.first}); if(d==D.end()) continue;
                    for(const Node& p: d->second) D[{c->to,k+1,pr.second}].insert(p); }
            }
        }
    }

    std::set<std::pair<uint64_t,uint64_t>> C = reduce_depthTR(edges);   // quotient depth-TR

    bool eq_depth = (C==fc_depth);     // the clean-recovery claim: quotient == full-capture (esig,depth) reduction
    bool eq_raw   = (C==fc_raw);       // vs the raw-forest reference convention
    printf("%-8s steps=%d | quotient depthTR=%-4zu  fullcap depthTR=%-4zu  [%s]   |  fullcap reduce(raw)=%-4zu  [%s]\n",
           w.name, w.steps, C.size(), fc_depth.size(),
           eq_depth?"MATCH: clean recovery holds":"*** MISMATCH ***",
           fc_raw.size(), eq_raw?"also matches raw":"differs from raw (expected)");
    if(!eq_depth){ size_t miss=0,extra=0; for(auto&p:fc_depth) if(!C.count(p)) miss++; for(auto&p:C) if(!fc_depth.count(p)) extra++;
        printf("           vs fullcap-depthTR: missing=%zu extra=%zu\n", miss, extra); }
}
int main(){
    WL wls[]={ {"WPP",WPP,{{0,1},{0,2}},6}, {"mixed1",mixed1,{{0,1}},6},
               {"mixed2",mixed2,{{0,1}},6}, {"TC6",TC6,{{1,2},{2,3}},4} };
    for(auto&w:wls) check(w);
    return 0;
}
