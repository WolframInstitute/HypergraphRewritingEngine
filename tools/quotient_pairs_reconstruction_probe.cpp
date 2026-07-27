// De-risk: reconstruct num_causal_event_pairs from the quotient skeleton as
//   pairs = T1 - T2   (rules consume <=2 edges, so higher terms vanish)
// T1 = per-edge causal (first-moment D, already proven exact) ;
// T2 = same-producer co-consumed edge-pairs (a JOINT tracker J, propagated one
// order up from D). Marginal orbit fractions in J's survivor term and in the
// consumer selection are the independence assumption under test -- if pairs match
// the engine's num_causal_event_pairs on the co-consumption workloads, it holds.
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <cmath>
using namespace hypergraph;
using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;
struct WL { const char* name; Rules (*rules)(); Init init; int steps; };

static Rules pSplit(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules pWolfram(){ return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build() }; }
static Rules iShift(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules rMerge(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mProdRed(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                  make_rule(1).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mAllThree(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                   make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                                   make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules TC6(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,2}).rhs({1,3}).build() }; }

struct StateInfo { uint64_t hash; std::unordered_map<uint32_t,uint32_t> edge_orbit; std::vector<uint32_t> orbit_size; uint32_t step; };
static StateInfo describe(const Hypergraph& hg, IRCanonicalizer& ir, uint32_t sid){
    StateInfo si; si.step=hg.get_state(sid).step;
    std::vector<std::vector<VertexId>> edges; std::vector<uint32_t> ids;
    hg.get_state(sid).edges.for_each([&](EdgeId eid){ const auto& e=hg.get_edge(eid);
        std::vector<VertexId> v; for(uint8_t i=0;i<e.arity;++i) v.push_back(e.vertices[i]);
        edges.push_back(v); ids.push_back(eid); });
    std::vector<uint32_t> orbit; si.hash=ir.compute_canonical_hash_with_edge_orbits(edges, orbit);
    uint32_t no=0; for(uint32_t o:orbit) no=std::max(no,o+1);
    si.orbit_size.assign(no,0); for(uint32_t o:orbit) si.orbit_size[o]++;
    for(size_t i=0;i<ids.size();++i) si.edge_orbit[ids[i]] = orbit.empty()?0:orbit[i];
    return si;
}

static uint64_t mixh(uint64_t h,uint64_t v){h^=v;h*=1099511628211ull;return h;}
static uint64_t ekey(uint64_t ih,uint64_t oh,uint32_t rule,uint32_t step){
    uint64_t k=1469598103934665603ull;k=mixh(k,ih);k=mixh(k,oh);k=mixh(k,rule);k=mixh(k,step);return k;}

static void check(const WL& w){
    // ---- oracle: engine full-capture ----
    size_t tgt_pairs, tgt_edges;
    { Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
      ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(false);
      e.set_explore_from_canonical_states_only(false); for(auto&r:w.rules()) e.add_rule(r);
      auto in=w.init; e.evolve(in,w.steps);
      tgt_pairs=hg.causal_graph().num_causal_event_pairs(); tgt_edges=hg.causal_graph().num_causal_edges(); }

    // ---- skeleton ----
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,8); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(true); for(auto&r:w.rules()) e.add_rule(r);
    auto init=w.init; e.evolve(init,w.steps);

    IRCanonicalizer ir;
    std::unordered_map<uint32_t,StateInfo> si;
    for(uint32_t s=0;s<hg.num_states();++s) if(hg.get_state(s).id!=INVALID_ID) si[s]=describe(hg,ir,s);
    std::unordered_map<uint64_t,std::vector<uint32_t>> osz_of;
    for(auto&kv:si) osz_of[kv.second.hash]=kv.second.orbit_size;

    struct CE { uint64_t from,to; uint32_t rule; std::vector<uint32_t> consumed, produced_orbit;
                std::vector<std::pair<uint32_t,uint32_t>> survivors; long long w=0; uint64_t okey_at(int k)const{return ekey(from,to,rule,(uint32_t)(k+1));} };
    std::map<std::string,CE> ces;
    for(uint32_t i=0;i<hg.num_events();++i){ const auto& ev=hg.get_event(i); if(ev.id==INVALID_ID) continue;
        const StateInfo& in_s=si[ev.input_state]; const StateInfo& out_s=si[ev.output_state];
        CE c; c.from=in_s.hash; c.to=out_s.hash; c.rule=ev.rule_index;
        for(uint8_t j=0;j<ev.num_consumed;++j) c.consumed.push_back(in_s.edge_orbit.at(ev.consumed_edges[j]));
        std::sort(c.consumed.begin(),c.consumed.end());
        std::set<uint32_t> produced(ev.produced_edges,ev.produced_edges+ev.num_produced);
        for(uint8_t j=0;j<ev.num_produced;++j) c.produced_orbit.push_back(out_s.edge_orbit.at(ev.produced_edges[j]));
        std::sort(c.produced_orbit.begin(),c.produced_orbit.end());
        for(auto&kv:out_s.edge_orbit) if(!produced.count(kv.first)&&in_s.edge_orbit.count(kv.first))
            c.survivors.push_back({in_s.edge_orbit.at(kv.first),kv.second});
        std::sort(c.survivors.begin(),c.survivors.end());
        char buf[512]; int n=snprintf(buf,sizeof buf,"%llu|%llu|%u|",(unsigned long long)c.from,(unsigned long long)c.to,c.rule);
        for(uint32_t o:c.consumed) n+=snprintf(buf+n,sizeof(buf)-n,"c%u",o);
        for(auto&pr:c.survivors) n+=snprintf(buf+n,sizeof(buf)-n,"s%u>%u",pr.first,pr.second);
        auto it=ces.find(buf); if(it==ces.end()) ces.emplace(buf,c); ces[buf].w+=1;
    }
    std::unordered_map<uint64_t,std::vector<const CE*>> out_of;
    for(auto&kv:ces) out_of[kv.second.from].push_back(&kv.second);

    const int STEPS=w.steps; const uint64_t INIT=0; uint64_t s0=si[0].hash;
    std::map<std::pair<uint64_t,int>,long double> mult; mult[{s0,0}]=1;
    std::map<std::tuple<uint64_t,int,uint32_t>,std::map<uint64_t,long double>> D;         // (s,k,orbit)->prod->count
    std::map<std::tuple<uint64_t,int,uint32_t,uint32_t>,std::map<uint64_t,long double>> J; // (s,k,oa<=ob)->prod->same-producer pair count
    for(uint32_t j=0;j<osz_of[s0].size();++j) D[{s0,0,j}][INIT]=osz_of[s0][j];

    long double T1=0, T2=0;
    auto Jkey=[](uint64_t s,int k,uint32_t a,uint32_t b){ if(a>b) std::swap(a,b); return std::make_tuple(s,k,a,b); };

    for(int k=0;k<STEPS;++k){
        for(auto& kv:mult){ if(kv.first.second!=k) continue; uint64_t s=kv.first.first; long double M=kv.second; if(M<=0) continue;
            auto oit=out_of.find(s); if(oit==out_of.end()) continue; const auto& msz=osz_of[s];
            for(const CE* c:oit->second){
                long double firings=M*(long double)c->w;
                std::map<uint32_t,int> cj; for(uint32_t o:c->consumed) cj[o]++;
                // T1: per-edge causal
                for(auto&[j,cnt]:cj){ auto d=D.find({s,k,j}); if(d==D.end()) continue;
                    long double frac=(long double)cnt/(long double)msz[j];
                    for(auto&[p,val]:d->second){ if(p==INIT) continue; T1 += (long double)c->w*frac*val; } }
                // T2: same-producer co-consumed pairs (marginal selection fractions)
                std::vector<uint32_t> orbs; for(auto&[j,cnt]:cj) orbs.push_back(j);
                for(size_t a=0;a<orbs.size();++a) for(size_t b=a;b<orbs.size();++b){
                    uint32_t ja=orbs[a], jb=orbs[b]; auto jit=J.find(Jkey(s,k,ja,jb)); if(jit==J.end()) continue;
                    long double selfrac;
                    if(ja==jb){ long double m=msz[ja]; if(m<2) continue; selfrac=(long double)cj[ja]*(cj[ja]-1)/(m*(m-1)); }
                    else selfrac=((long double)cj[ja]/msz[ja])*((long double)cj[jb]/msz[jb]);
                    for(auto&[p,val]:jit->second){ if(p==INIT) continue; T2 += (long double)c->w*selfrac*val; }
                }
                mult[{c->to,k+1}] += firings;
                // ---- propagate D ----
                std::map<std::pair<uint32_t,uint32_t>,int> sm; for(auto&pr:c->survivors) sm[pr]++;
                for(auto&[jj,cnt]:sm){ auto d=D.find({s,k,jj.first}); if(d==D.end()) continue;
                    long double frac=(long double)cnt/(long double)msz[jj.first];
                    for(auto&[p,val]:d->second) D[{c->to,k+1,jj.second}][p] += (long double)c->w*frac*val; }
                for(uint32_t o:c->produced_orbit) D[{c->to,k+1,o}][c->okey_at(k)] += firings;
                // ---- propagate J ----
                // produced-produced pairs, all by this event
                for(size_t a=0;a<c->produced_orbit.size();++a) for(size_t b=a+1;b<c->produced_orbit.size();++b)
                    J[Jkey(c->to,k+1,c->produced_orbit[a],c->produced_orbit[b])][c->okey_at(k)] += firings;
                // survivor-survivor pairs carry parent's same-producer pairs forward (marginal joint survival)
                // per parent orbit-pair, distribute into child orbit-pairs by survival fractions
                std::map<uint32_t,std::map<uint32_t,int>> smap; for(auto&pr:c->survivors) smap[pr.first][pr.second]++;
                for(auto& jkv:J){ if(std::get<0>(jkv.first)!=s||std::get<1>(jkv.first)!=k) continue;
                    uint32_t pa=std::get<2>(jkv.first), pb=std::get<3>(jkv.first);
                    auto ia=smap.find(pa), ib=smap.find(pb); if(ia==smap.end()||ib==smap.end()) continue;
                    for(auto&[ca,na]:ia->second) for(auto&[cb,nb]:ib->second){
                        long double fa=(long double)na/msz[pa], fb=(long double)nb/msz[pb];
                        long double f = (pa==pb)? // same parent orbit: avoid pairing an edge with itself
                            ((msz[pa]<2)?0.0L:(long double)na*(na-1)/((long double)msz[pa]*(msz[pa]-1))) : fa*fb;
                        if(f<=0) continue;
                        for(auto&[p,val]:jkv.second) J[Jkey(c->to,k+1,ca,cb)][p] += (long double)c->w*f*val;
                    }
                }
            }
        }
    }
    long long pairs=(long long)llroundl(T1-T2);
    printf("%-11s steps=%d  target_pairs=%-5zu (edges=%-5zu)  T1=%-6.0Lf T2=%-6.0Lf  recon=%-5lld  %s\n",
           w.name,w.steps,tgt_pairs,tgt_edges,T1,T2,pairs,((size_t)pairs==tgt_pairs)?"MATCH":"*** MISMATCH ***");
}
int main(){
    WL wls[]={ {"pSplit",pSplit,{{1,2}},4}, {"rMerge",rMerge,{{1,2},{2,3},{3,4}},3},
        {"pWolfram",pWolfram,{{1,2},{1,3}},4}, {"iShift",iShift,{{1,2},{2,3}},4},
        {"mProdRed",mProdRed,{{1,2}},4}, {"mAllThree",mAllThree,{{1,2}},3}, {"TC6",TC6,{{1,2},{2,3}},4} };
    for(auto&w:wls) check(w);
    return 0;
}
