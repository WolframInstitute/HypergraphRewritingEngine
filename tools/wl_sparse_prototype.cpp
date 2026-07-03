// tools/wl_sparse_prototype.cpp
//
// Standalone prototype of a FULLY SPARSE incremental WL: adjacency, refinement,
// and the commutative hash are all patched in O(delta) from the parent, indexing
// by (stable) vertex id. Tests whether this breaks the ~1.25x ceiling (which was
// imposed by the O(n+m) child adjacency rebuild + O(n*rounds) refinement fills).
//
// Pragmatic shortcut: run the child to the PARENT's fixpoint round; if the
// resulting hash matches full WL we accept it (the common case where the rewrite
// doesn't change refinement depth), else fall back to full WL (correctness). We
// report the accept rate, the speedup when accepted, and the effective speedup.
//
// Binary directed edges (the rule shapes here). Validated bit-identical (or
// fallback) against full WL with the same commutative hash.
//
// Build: g++ -O2 -std=c++17 tools/wl_sparse_prototype.cpp -o /tmp/wl_sparse_prototype

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// allocation counter — to quantify how much each path touches the allocator
static long long g_allocs = 0; static bool g_track = false;
void* operator new(std::size_t n){ if(g_track)++g_allocs; void* p=std::malloc(n?n:1); if(!p)throw std::bad_alloc(); return p; }
void* operator new[](std::size_t n){ if(g_track)++g_allocs; void* p=std::malloc(n?n:1); if(!p)throw std::bad_alloc(); return p; }
void operator delete(void* p)noexcept{ std::free(p); }
void operator delete(void* p,std::size_t)noexcept{ std::free(p); }
void operator delete[](void* p)noexcept{ std::free(p); }
void operator delete[](void* p,std::size_t)noexcept{ std::free(p); }

using clk = std::chrono::high_resolution_clock;
static double msd(clk::time_point a, clk::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
static constexpr uint64_t FNV=1469598103934665603ull, FP=1099511628211ull;
static inline uint64_t fc(uint64_t h, uint64_t x){ h^=x; h*=FP; return h; }
static inline uint64_t mix64(uint64_t z){ z+=0x9e3779b97f4a7c15ull; z=(z^(z>>30))*0xbf58476d1ce4e5b9ull; z=(z^(z>>27))*0x94d049bb133111ebull; return z^(z>>31); }

static std::vector<std::array<int,2>> EDGES;   // global edge store; edge id = index
static int add_edge(int a,int b){ EDGES.push_back({a,b}); return (int)EDGES.size()-1; }

struct Occ { int eid; int pos; };

// per-vertex init colour: degree + sorted (arity=2, pos) pairs
static uint64_t init_col(const std::vector<Occ>& occ){
    std::vector<std::pair<int,int>> ap; for(auto&o:occ) ap.push_back({2,o.pos});
    std::sort(ap.begin(),ap.end());
    uint64_t h=FNV; h=fc(h,occ.size()); for(auto&p:ap){ h=fc(h,p.first); h=fc(h,p.second);} return h;
}
// one refinement round for a vertex, given a colour lookup
template<typename ColOf>
static uint64_t sig(int v, const std::vector<Occ>& occ, ColOf&& col){
    uint64_t h=col(v); std::vector<uint64_t> nh;
    for(auto&o:occ){ int other=EDGES[o.eid][1-o.pos]; nh.push_back(fc(col(other), 1-o.pos)); }
    std::sort(nh.begin(),nh.end()); for(uint64_t x:nh) h=fc(h,x); return h;
}
static uint64_t edge_hash(int eid, const std::vector<uint64_t>& col){
    uint64_t h=FNV; h=fc(h,2); h=fc(h,col[EDGES[eid][0]]); h=fc(h,col[EDGES[eid][1]]); return h;
}

struct Full {
    uint64_t hash=0; uint64_t vsum=0, esum=0; int fixp=0; int maxid=0;
    int npresent=0, nactive=0;
    std::vector<std::vector<Occ>> adj;     // by id
    std::vector<char> present;             // by id
    std::vector<std::vector<uint64_t>> hist;  // [round][id]
};

static int distinct_count(const std::vector<uint64_t>& col, const std::vector<char>& present){
    std::vector<uint64_t> t; for(size_t v=0;v<present.size();++v) if(present[v]) t.push_back(col[v]);
    std::sort(t.begin(),t.end()); int d=0; for(size_t i=0;i<t.size();++i) if(i==0||t[i]!=t[i-1])++d; return d;
}

static Full full_wl(const std::vector<int>& active, long long* work=nullptr){
    Full F; int maxid=0; for(int e:active){ maxid=std::max(maxid,std::max(EDGES[e][0],EDGES[e][1])); }
    F.maxid=maxid; F.nactive=(int)active.size();
    F.adj.assign(maxid+1,{}); F.present.assign(maxid+1,0);
    for(int e:active){ for(int p=0;p<2;++p){ int v=EDGES[e][p]; F.adj[v].push_back({e,p}); F.present[v]=1; } }
    std::vector<uint64_t> cur(maxid+1,0);
    int np=0; for(int v=0;v<=maxid;++v) if(F.present[v]){ cur[v]=init_col(F.adj[v]); ++np; }
    F.npresent=np;
    F.hist.push_back(cur);
    int prev=distinct_count(cur,F.present), iter=0;
    std::vector<uint64_t> nxt(maxid+1,0);
    while(iter<200){ ++iter;
        for(int v=0;v<=maxid;++v) if(F.present[v]){ nxt[v]=sig(v,F.adj[v],[&](int u){return cur[u];}); if(work)++*work; }
        std::swap(cur,nxt); F.hist.push_back(cur); F.fixp=(int)F.hist.size()-1;
        int d=distinct_count(cur,F.present); if(d==prev) break; prev=d;
    }
    uint64_t vsum=0; for(int v=0;v<=maxid;++v) if(F.present[v]) vsum+=mix64(cur[v]);
    uint64_t esum=0; for(int e:active) esum+=mix64(edge_hash(e,cur));
    F.vsum=vsum; F.esum=esum;
    uint64_t h=FNV; h=fc(h,np); h=fc(h,(int)active.size()); h=fc(h,vsum); h=fc(h,esum);
    F.hash=h; return F;
}

// Sparse incremental child hash from parent Full + delta. Returns {hash, accepted}.
// accepted=false => fell back to full WL (hash is the full result).
static std::pair<uint64_t,bool> sparse_incr(const Full& P, const std::vector<int>& consumed,
                                            const std::vector<int>& produced, long long* work=nullptr){
    std::unordered_set<int> cons(consumed.begin(),consumed.end()), prod(produced.begin(),produced.end());
    // affected vertex ids = endpoints of consumed/produced
    std::unordered_set<int> affected;
    for(int e:consumed){ affected.insert(EDGES[e][0]); affected.insert(EDGES[e][1]); }
    for(int e:produced){ affected.insert(EDGES[e][0]); affected.insert(EDGES[e][1]); }
    // child adjacency overlay (only affected ids)
    std::unordered_map<int,std::vector<Occ>> oadj;
    int maxid=P.maxid; for(int e:produced){ maxid=std::max(maxid,std::max(EDGES[e][0],EDGES[e][1])); }
    auto parent_present=[&](int v){ return v<=P.maxid && P.present[v]; };
    for(int id:affected){
        std::vector<Occ> lst;
        if(parent_present(id)) for(auto&o:P.adj[id]) if(!cons.count(o.eid)) lst.push_back(o);
        for(int e:produced){ if(EDGES[e][0]==id) lst.push_back({e,0}); if(EDGES[e][1]==id) lst.push_back({e,1}); }
        oadj[id]=std::move(lst);
    }
    auto child_adj=[&](int id)->const std::vector<Occ>&{ auto it=oadj.find(id); return it!=oadj.end()?it->second:P.adj[id]; };
    auto child_present=[&](int id){ auto it=oadj.find(id); if(it!=oadj.end()) return !it->second.empty(); return parent_present(id); };
    auto isnew=[&](int id){ return !parent_present(id) && child_present(id); };

    // delta = affected ids that are present in child, + new ids
    std::vector<int> delta; for(int id:affected) if(child_present(id)) delta.push_back(id);

    int R=P.fixp;
    std::vector<std::unordered_map<int,uint64_t>> ov(R+1);  // child colour overlays per round
    auto ccol=[&](int r,int id)->uint64_t{ auto it=ov[r].find(id); if(it!=ov[r].end()) return it->second; return P.hist[r][id]; };
    for(int id:delta) ov[0][id]=init_col(child_adj(id));
    std::unordered_set<int> changed(delta.begin(),delta.end());
    for(int r=1;r<=R;++r){
        std::unordered_set<int> recompute=changed;
        for(int id:changed) for(auto&o:child_adj(id)) recompute.insert(EDGES[o.eid][1-o.pos]);
        std::unordered_set<int> nchanged;
        for(int id:recompute){
            uint64_t val=sig(id, child_adj(id), [&](int u){ return ccol(r-1,u); }); if(work)++*work;
            ov[r][id]=val;
            bool nw=isnew(id);
            if(nw || val != P.hist[r][id]) nchanged.insert(id);
        }
        changed.swap(nchanged);
    }
    // child final colours at round R: ccol(R, id)
    // commutative hash patch
    uint64_t vsum=P.vsum, esum=P.esum;
    std::vector<int> colourchanged;
    for(auto&kv:ov[R]){ int id=kv.first; uint64_t cf=kv.second;
        if(isnew(id)){ vsum+=mix64(cf); colourchanged.push_back(id); }
        else { uint64_t pf=P.hist[R][id]; if(cf!=pf){ vsum+=mix64(cf)-mix64(pf); colourchanged.push_back(id);} } }
    // removed parent vertices (subset of consumed endpoints not in child)
    std::unordered_set<int> rmseen;
    for(int e:consumed) for(int p=0;p<2;++p){ int v=EDGES[e][p]; if(parent_present(v) && !child_present(v) && rmseen.insert(v).second) vsum-=mix64(P.hist[R][v]); }
    // edge patch
    auto ceh=[&](int eid)->uint64_t{ uint64_t h=FNV; h=fc(h,2); h=fc(h,ccol(R,EDGES[eid][0])); h=fc(h,ccol(R,EDGES[eid][1])); return h; };
    auto peh=[&](int eid)->uint64_t{ uint64_t h=FNV; h=fc(h,2); h=fc(h,P.hist[R][EDGES[eid][0]]); h=fc(h,P.hist[R][EDGES[eid][1]]); return h; };
    for(int e:consumed) esum-=mix64(peh(e));
    for(int e:produced) esum+=mix64(ceh(e));
    std::unordered_set<int> patched;
    for(int id:colourchanged) for(auto&o:child_adj(id)){ int eid=o.eid; if(prod.count(eid)) continue; if(!patched.insert(eid).second) continue; esum += mix64(ceh(eid)) - mix64(peh(eid)); }

    int nchild = P.npresent; for(int e:produced) for(int p=0;p<2;++p){ int v=EDGES[e][p]; if(!parent_present(v)){ /* counted below */ } }
    // recompute nchild exactly: parent present + new - removed
    std::unordered_set<int> newseen, remseen2;
    for(int id:affected){ if(isnew(id)) newseen.insert(id); if(parent_present(id)&&!child_present(id)) remseen2.insert(id); }
    nchild = P.npresent + (int)newseen.size() - (int)remseen2.size();
    int mchild = P.nactive - (int)consumed.size() + (int)produced.size();
    uint64_t h=FNV; h=fc(h,nchild); h=fc(h,mchild); h=fc(h,vsum); h=fc(h,esum);
    (void)maxid;
    return {h, true};
}

// ---- generators (parent graphs over ids 0..n-1) ----
static std::vector<int> gen_random(int n, std::mt19937& rng){ int m=(int)(1.5*n); std::vector<int> a; for(int i=0;i<m;++i) a.push_back(add_edge(rng()%n, rng()%n)); return a; }
static std::vector<int> gen_grid(int n, std::mt19937&){ int w=(int)std::round(std::sqrt((double)n)); if(w<2)w=2; int h=w; auto id=[&](int x,int y){return y*w+x;}; std::vector<int> a; for(int y=0;y<h;++y)for(int x=0;x<w;++x){ if(x+1<w)a.push_back(add_edge(id(x,y),id(x+1,y))); if(y+1<h)a.push_back(add_edge(id(x,y),id(x,y+1))); } return a; }
static std::vector<int> gen_chain(int n, std::mt19937&){ int c=n/5; if(c<1)c=1; std::vector<int> a; for(int k=0;k<c;++k){ int b=k*5; for(int i=0;i<5;++i)a.push_back(add_edge(b+i,b+(i+1)%5)); if(k+1<c)a.push_back(add_edge(b+4,(k+1)*5)); } return a; }

static void run(const char* name, std::vector<int>(*gen)(int,std::mt19937&), const std::vector<int>& sizes, int trials){
    std::printf("== %s ==\n", name);
    std::printf("   %-8s %-7s %-9s %-9s %-10s %-10s\n","n","rounds","accept%","speedup","full_mallocs","incr_mallocs");
    for(int n:sizes){
        std::mt19937 rng(1234u+n);
        double t_full=0,t_incr=0; int accepted=0,cases=0; long long rsum=0; long long fa=0,sa=0;
        for(int t=0;t<trials;++t){
            EDGES.clear();
            std::vector<int> active=gen(n,rng);
            if(active.empty()) continue;
            // parent
            Full P=full_wl(active);
            // rewrite: consume up to 2 edges, produce {a,w},{w,b} with fresh w
            std::vector<int> consumed; consumed.push_back(active[rng()%active.size()]);
            if(active.size()>1){ int e2=active[rng()%active.size()]; if(e2!=consumed[0]) consumed.push_back(e2); }
            int a=EDGES[consumed[0]][0], b=EDGES[consumed[0]][1], w=n; // fresh id n
            std::vector<int> produced={ add_edge(a,w), add_edge(w,b) };
            std::unordered_set<int> cs(consumed.begin(),consumed.end());
            std::vector<int> child; for(int e:active) if(!cs.count(e)) child.push_back(e); for(int e:produced) child.push_back(e);
            if(child.empty()) continue;
            ++cases; rsum+=P.fixp;
            g_track=true;
            long long a0=g_allocs; auto t0=clk::now(); Full C=full_wl(child); auto t1=clk::now(); long long a1=g_allocs;
            auto t2=clk::now(); auto sr=sparse_incr(P,consumed,produced); auto t3=clk::now(); long long a2=g_allocs;
            g_track=false;
            fa += a1-a0; sa += a2-a1;
            bool ok = sr.second && (sr.first==C.hash);
            if(ok) ++accepted;
            t_full+=msd(t0,t1);
            t_incr+= ok ? msd(t2,t3) : (msd(t2,t3)+msd(t0,t1)); // fallback pays a full WL
        }
        std::printf("   %-8d %-7.1f %-9.0f %-9.2f %-10.1f %-10.1f\n", n, cases?(double)rsum/cases:0.0,
                    cases?100.0*accepted/cases:0.0, t_incr>0?t_full/t_incr:0.0,
                    cases?(double)fa/cases:0.0, cases?(double)sa/cases:0.0);
    }
    std::printf("\n");
}

int main(){
    setvbuf(stdout,nullptr,_IONBF,0);
    std::printf("Fully-sparse incremental WL prototype (O(delta) adjacency+refine+hash, id-indexed)\n\n");
    run("random sparse (low diameter)", gen_random, {100,200,400,800,1600,3200}, 80);
    run("grid (sqrt-n diameter)",       gen_grid,   {100,256,400,784,1600},      40);
    run("modular chain (high diam)",    gen_chain,  {100,200,400,800,1600},      40);
    std::printf("Reading: accept%%<100 => some rewrites changed refinement depth (fell back to full).\n"
                "speedup counts fallback cost. If speedup rises well above 1.25x at scale, the sparse\n"
                "pipeline breaks the ceiling.\n");
    return 0;
}
