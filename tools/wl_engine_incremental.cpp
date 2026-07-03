// tools/wl_engine_incremental.cpp
//
// Validates a SPARSE incremental WL against the ENGINE's real full WL
// (WLHash::compute_state_hash_with_cache, now a commutative multiset hash), and
// measures the de-confounded speedup. The incremental reimplements the engine's
// exact colour formula (init: FNV,deg,sorted(arity,pos); refine: fold sorted
// fnv_combine(current[nbr], k); distinct-count fixpoint) so its child hash must be
// BIT-IDENTICAL to the engine's. Vertex-id indexed (stable across parent->child),
// dirty-frontier refinement. The child's OWN fixpoint round is detected from its
// distinct-colour trajectory, lazily EXTENDING the parent history when the child
// needs more rounds -> always correct, NO fallback (B2). O(delta) commutative patch
// when the child fixpoint matches the parent's, full rebuild otherwise.
//
// Build: g++ -O2 -std=c++17 -I hypergraph/include tools/wl_engine_incremental.cpp -o /tmp/wl_engine_incremental

#include <hypergraph/wl_hash.hpp>
#include <hypergraph/arena.hpp>
#include <hypergraph/bitset.hpp>
#include <hypergraph/types.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace hypergraph;
using clk = std::chrono::high_resolution_clock;
static double msd(clk::time_point a, clk::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }

// engine's commutative-hash finaliser (must match WLHash::smix64)
static inline uint64_t smix64(uint64_t z){ z+=0x9e3779b97f4a7c15ull; z=(z^(z>>30))*0xbf58476d1ce4e5b9ull; z=(z^(z>>27))*0x94d049bb133111ebull; return z^(z>>31); }
static inline uint64_t fc(uint64_t h, uint64_t x){ return fnv_hash(h, x); }   // engine fnv_combine

static std::vector<std::vector<VertexId>> STORE;   // edge id -> vertices
static int add_edge(const std::vector<VertexId>& v){ STORE.push_back(v); return (int)STORE.size()-1; }

// ---- engine full WL (real oracle) over a set of edge ids ----
struct EV { const VertexId* operator[](EdgeId e) const { return STORE[e].data(); } };
struct EA { uint8_t operator[](EdgeId e) const { return (uint8_t)STORE[e].size(); } };

// ---- id-indexed sparse WL (reimplements the engine formula) ----
struct Occ { int eid; uint8_t pos; };
struct Hist {
    int maxid=0;
    std::vector<char> present;                 // by id
    std::vector<std::vector<Occ>> adj;         // by id
    std::vector<std::vector<uint64_t>> rounds; // [r][id] colour (present ids only meaningful)
    int fixp=0; uint64_t vsum=0, esum=0; size_t m=0;
    std::vector<int> verts;                    // present ids
};

static uint64_t init_col(const std::vector<Occ>& occ){
    std::vector<std::pair<uint8_t,uint8_t>> ap; ap.reserve(occ.size());
    for(auto&o:occ) ap.push_back({(uint8_t)STORE[o.eid].size(), o.pos});
    std::sort(ap.begin(),ap.end());
    uint64_t h=FNV_OFFSET; h=fc(h,occ.size()); for(auto&p:ap){ h=fc(h,p.first); h=fc(h,p.second);} return h;
}
template<typename Col>
static uint64_t refine_col(int id, const std::vector<Occ>& occ, Col&& col){
    uint64_t h=col(id); std::vector<uint64_t> nb;
    for(auto&o:occ){ const auto& ev=STORE[o.eid]; uint8_t ar=(uint8_t)ev.size();
        for(uint8_t k=0;k<ar;++k) if(k!=o.pos) nb.push_back(fc(col(ev[k]), k)); }
    std::sort(nb.begin(),nb.end()); for(uint64_t x:nb) h=fc(h,x); return h;
}
static int distinct_present(const std::vector<uint64_t>& col, const std::vector<int>& verts){
    std::vector<uint64_t> t; t.reserve(verts.size()); for(int v:verts) t.push_back(col[v]);
    std::sort(t.begin(),t.end()); int d=0; for(size_t i=0;i<t.size();++i) if(i==0||t[i]!=t[i-1])++d; return d;
}
static uint64_t comm_hash(size_t nv, size_t m, uint64_t vsum, uint64_t esum){
    uint64_t h=FNV_OFFSET; h=fc(h,nv); h=fc(h,m); h=fc(h,vsum); h=fc(h,esum); return h;
}

// full id-indexed WL over edge ids, recording per-round colours -> Hist
static uint64_t full_hist(const std::vector<int>& eids, Hist& H){
    int maxid=0; for(int e:eids) for(VertexId v:STORE[e]) maxid=std::max(maxid,(int)v);
    H.maxid=maxid; H.present.assign(maxid+1,0); H.adj.assign(maxid+1,{}); H.verts.clear();
    for(int e:eids){ for(uint8_t p=0;p<STORE[e].size();++p){ int v=STORE[e][p]; if(!H.present[v]){H.present[v]=1;H.verts.push_back(v);} H.adj[v].push_back({e,p}); } }
    std::sort(H.verts.begin(),H.verts.end());
    std::vector<uint64_t> cur(maxid+1,0), nxt(maxid+1,0);
    for(int v:H.verts) cur[v]=init_col(H.adj[v]);
    H.rounds.clear(); H.rounds.push_back(cur);
    int prev=distinct_present(cur,H.verts), iter=0, fixp=0;
    while(iter<100){ ++iter;
        for(int v:H.verts) nxt[v]=refine_col(v,H.adj[v],[&](int u){return cur[u];});
        for(int v:H.verts) cur[v]=nxt[v];
        H.rounds.push_back(cur); fixp=iter;
        int d=distinct_present(cur,H.verts); if(d==prev) break; prev=d;
    }
    H.fixp=fixp;
    uint64_t vsum=0; for(int v:H.verts) vsum+=smix64(cur[v]);
    uint64_t esum=0; for(int e:eids) { uint64_t eh=FNV_OFFSET; eh=fc(eh,STORE[e].size()); for(VertexId v:STORE[e]) eh=fc(eh,cur[v]); esum+=smix64(eh); }
    H.vsum=vsum; H.esum=esum; H.m=eids.size();
    return comm_hash(H.verts.size(), eids.size(), vsum, esum);
}

// Sparse incremental child hash from parent Hist + delta. Detects the CHILD's own
// fixpoint round via its distinct-colour trajectory and lazily EXTENDS the parent
// history when the child needs more rounds -> always correct, no fallback. Uses the
// O(delta) commutative patch when the child fixpoint matches the parent's (mode 0);
// otherwise a full commutative rebuild at the child fixpoint (mode 1). Returns
// {hash, mode}. `P` is by non-const ref because the lazy extension caches new rounds.
static std::pair<uint64_t,int> sparse_incr(Hist& P, const std::vector<int>& child_eids,
                                           const std::vector<int>& consumed, const std::vector<int>& produced){
    std::unordered_set<int> cons(consumed.begin(),consumed.end()), prod(produced.begin(),produced.end());
    const int Rp=P.fixp;
    auto ppres=[&](int v){ return v<=P.maxid && P.present[v]; };
    std::unordered_set<int> affected;
    for(int e:consumed) for(VertexId v:STORE[e]) affected.insert(v);
    for(int e:produced) for(VertexId v:STORE[e]) affected.insert(v);
    std::unordered_map<int,std::vector<Occ>> oadj;
    for(int id:affected){ std::vector<Occ> lst;
        if(ppres(id)) for(auto&o:P.adj[id]) if(!cons.count(o.eid)) lst.push_back(o);
        for(int e:produced){ const auto&ev=STORE[e]; for(uint8_t p=0;p<ev.size();++p) if((int)ev[p]==id) lst.push_back({e,p}); }
        oadj[id]=std::move(lst);
    }
    auto adj=[&](int id)->const std::vector<Occ>&{ auto it=oadj.find(id); return it!=oadj.end()?it->second:P.adj[id]; };
    auto cpres=[&](int id){ auto it=oadj.find(id); if(it!=oadj.end()) return !it->second.empty(); return ppres(id); };
    auto isnew=[&](int id){ return !ppres(id) && cpres(id); };
    std::vector<int> cverts; { std::unordered_set<int> seen;
        for(int e:child_eids) for(VertexId v:STORE[e]) if(seen.insert(v).second) cverts.push_back(v); }

    // lazy parent history extension: parent colours past its own fixpoint are still
    // well-defined (its partition is stable but content-hash colours keep re-hashing)
    auto ext_parent=[&](int r){
        while((int)P.rounds.size()<=r){
            int pr=(int)P.rounds.size()-1;
            std::vector<uint64_t> nx=P.rounds[pr];
            for(int v:P.verts) nx[v]=refine_col(v,P.adj[v],[&](int u){return P.rounds[pr][u];});
            P.rounds.push_back(std::move(nx));
        }
    };
    auto pcol=[&](int r,int id)->uint64_t{ ext_parent(r); return ppres(id)? P.rounds[r][id]:0; };
    std::vector<std::unordered_map<int,uint64_t>> ov; ov.emplace_back();
    auto ccol=[&](int r,int id)->uint64_t{ if(r<(int)ov.size()){auto it=ov[r].find(id); if(it!=ov[r].end()) return it->second;} return pcol(r,id); };
    auto child_distinct=[&](int r)->int{ std::vector<uint64_t> t; t.reserve(cverts.size()); for(int id:cverts) t.push_back(ccol(r,id));
        std::sort(t.begin(),t.end()); int d=0; for(size_t i=0;i<t.size();++i) if(i==0||t[i]!=t[i-1])++d; return d; };

    std::unordered_set<int> changed;
    for(int id:affected) if(cpres(id)){ ov[0][id]=init_col(adj(id)); changed.insert(id); }
    int prev=child_distinct(0), Rc=0;
    for(int r=1;r<=100;++r){
        if((int)ov.size()<=r) ov.emplace_back();
        std::unordered_set<int> recompute=changed;
        for(int id:changed) for(auto&o:adj(id)){ const auto&ev=STORE[o.eid]; for(uint8_t k=0;k<ev.size();++k) recompute.insert(ev[k]); }
        std::unordered_set<int> nchanged;
        for(int id:recompute){ uint64_t val=refine_col(id,adj(id),[&](int u){return ccol(r-1,u);}); ov[r][id]=val;
            uint64_t pv=ppres(id)?pcol(r,id):(val+1); if(val!=pv) nchanged.insert(id); }
        changed.swap(nchanged);
        int d=child_distinct(r); Rc=r; if(d==prev) break; prev=d;
    }
    auto cf=[&](int id){ return ccol(Rc,id); };
    size_t mchild=child_eids.size();

    if(Rc==Rp){   // O(delta) patch from the parent's stored components (both at Rp)
        uint64_t vsum=P.vsum, esum=P.esum; std::vector<int> cchg;
        for(auto&kv:ov[Rc]){ int id=kv.first; uint64_t c=kv.second;
            if(isnew(id)){ vsum+=smix64(c); cchg.push_back(id); }
            else { uint64_t pf=P.rounds[Rc][id]; if(c!=pf){ vsum+=smix64(c)-smix64(pf); cchg.push_back(id);} } }
        std::unordered_set<int> rm;
        for(int e:consumed) for(VertexId v:STORE[e]) if(ppres(v)&&!cpres(v)&&rm.insert(v).second) vsum-=smix64(P.rounds[Rc][v]);
        auto ceh=[&](int e){ uint64_t h=FNV_OFFSET;h=fc(h,STORE[e].size());for(VertexId v:STORE[e])h=fc(h,cf(v));return h; };
        auto peh=[&](int e){ uint64_t h=FNV_OFFSET;h=fc(h,STORE[e].size());for(VertexId v:STORE[e])h=fc(h,P.rounds[Rc][v]);return h; };
        for(int e:consumed) esum-=smix64(peh(e));
        for(int e:produced) esum+=smix64(ceh(e));
        std::unordered_set<int> patched;
        for(int id:cchg) for(auto&o:adj(id)){ int e=o.eid; if(prod.count(e))continue; if(!patched.insert(e).second)continue; esum+=smix64(ceh(e))-smix64(peh(e)); }
        int nnew=0,nrem=0; for(int id:affected){ if(isnew(id))++nnew; if(ppres(id)&&!cpres(id))++nrem; }
        size_t nchild=P.verts.size()+nnew-nrem;
        return { comm_hash(nchild, mchild, vsum, esum), 0 };
    }
    // Rc != Rp: full commutative rebuild from child colours at the child fixpoint
    uint64_t vsum=0; for(int id:cverts) vsum+=smix64(cf(id));
    uint64_t esum=0; for(int e:child_eids){ uint64_t h=FNV_OFFSET;h=fc(h,STORE[e].size());for(VertexId v:STORE[e])h=fc(h,cf(v)); esum+=smix64(h); }
    return { comm_hash(cverts.size(), mchild, vsum, esum), 1 };
}

int main(){
    setvbuf(stdout,nullptr,_IONBF,0);
    ConcurrentHeterogeneousArena arena; WLHash wl(&arena);
    EV ev; EA ea;
    std::mt19937 rng(12345);
    std::printf("Sparse incremental WL vs ENGINE full WL (commutative). Validates bit-identical + speedup.\n\n");
    struct Fam{ const char* name; int lo,hi; };
    for (Fam f : { Fam{"random sparse n=40..80",40,80}, Fam{"random sparse n=200..400",200,400}, Fam{"random sparse n=800..1600",800,1600} }){
        double t_full=0,t_incr=0; int cases=0, ok=0, formula_ok=0, m0=0, m1=0;
        for(int t=0;t<300;++t){
            STORE.clear();
            int n=f.lo+(int)(rng()%(f.hi-f.lo+1)); int m=(int)(1.5*n);
            std::vector<int> pe; for(int i=0;i<m;++i) pe.push_back(add_edge({(VertexId)(rng()%n),(VertexId)(rng()%n)}));
            SparseBitset pbs; for(int e:pe) pbs.set(e,arena); if(pbs.empty()) continue;
            // rewrite: consume up to 2, produce {a,w},{w,b}
            std::vector<int> consumed; consumed.push_back(pe[rng()%pe.size()]);
            if(pe.size()>1){ int e2=pe[rng()%pe.size()]; if(e2!=consumed[0]) consumed.push_back(e2); }
            int a=STORE[consumed[0]][0], b=STORE[consumed[0]].back(), w=n;
            std::vector<int> produced={ add_edge({(VertexId)a,(VertexId)w}), add_edge({(VertexId)w,(VertexId)b}) };
            std::unordered_set<int> cs(consumed.begin(),consumed.end());
            std::vector<int> child; for(int e:pe) if(!cs.count(e)) child.push_back(e); for(int e:produced) child.push_back(e);
            if(child.empty()) continue;
            SparseBitset cbs; for(int e:child) cbs.set(e,arena);

            // engine full WL on child (ground truth) + parent history (amortised)
            auto t0=clk::now(); uint64_t engine_full=wl.compute_state_hash_with_cache(cbs,ev,ea).first; auto t1=clk::now();
            Hist P; uint64_t my_parent=full_hist(pe,P);
            SparseBitset ebs; for(int e:pe) ebs.set(e,arena);
            uint64_t engine_parent=wl.compute_state_hash_with_cache(ebs,ev,ea).first;
            auto t2=clk::now(); auto sr=sparse_incr(P,child,consumed,produced); auto t3=clk::now();
            ++cases;
            if(my_parent==engine_parent) ++formula_ok;               // my formula matches engine
            if(sr.first==engine_full) ++ok;                          // bit-identical to engine full WL
            if(sr.second==0) ++m0; else ++m1;                        // 0=O(delta) patch, 1=full rebuild at Rc!=Rp
            t_full+=msd(t0,t1); t_incr+=msd(t2,t3);
        }
        std::printf("== %s ==\n", f.name);
        std::printf("   formula-match(parent): %d/%d  (my WL == engine WL)\n", formula_ok, cases);
        std::printf("   incremental bit-identical: %d/%d  (child-fixpoint detect + lazy extend, no fallback)\n", ok, cases);
        std::printf("   O(delta) patch: %d   full-rebuild (Rc!=Rp): %d   speedup: %.2fx  (understated: map-heavy vs arena-fast full)\n\n",
                    m0, m1, t_incr>0? t_full/t_incr : 0.0);
    }
    return 0;
}
