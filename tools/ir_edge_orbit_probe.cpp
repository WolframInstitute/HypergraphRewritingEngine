#include "hypergraph/ir_canonicalization.hpp"
#include <cstdio>
#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <set>
using namespace hypergraph;
using Edges = std::vector<std::vector<VertexId>>;

// brute-force orbits of DISTINCT EDGE CONTENTS under Aut(edge multiset).
// Positions are ill-defined for duplicate edges: a vertex automorphism induces no
// unique permutation of identical edges, so orbits are taken over contents.
static std::map<std::vector<VertexId>,int> brute_content_orbits(const Edges& e, uint32_t nv){
    std::vector<std::vector<VertexId>> uniq(e.begin(), e.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    size_t m = uniq.size();
    std::vector<int> uf(m); std::iota(uf.begin(),uf.end(),0);
    std::function<int(int)> f=[&](int x){ while(uf[x]!=x){uf[x]=uf[uf[x]];x=uf[x];} return x; };
    auto idx=[&](const std::vector<VertexId>& c){ return (int)(std::lower_bound(uniq.begin(),uniq.end(),c)-uniq.begin()); };
    std::vector<VertexId> pv(nv); std::iota(pv.begin(),pv.end(),0);
    std::multiset<std::vector<VertexId>> base(e.begin(), e.end());
    do {
        std::multiset<std::vector<VertexId>> img;
        for (auto& ed : e){ std::vector<VertexId> mm; for(VertexId v:ed) mm.push_back(pv[v]); img.insert(mm); }
        if (img!=base) continue;
        for (size_t c=0;c<m;++c){
            std::vector<VertexId> mm; for(VertexId v:uniq[c]) mm.push_back(pv[v]);
            int a=f((int)c), b=f(idx(mm)); if(a!=b) uf[a]=b;
        }
    } while (std::next_permutation(pv.begin(), pv.end()));
    std::map<std::vector<VertexId>,int> out;
    for(size_t c=0;c<m;++c) out[uniq[c]]=f((int)c);
    return out;
}

static bool same_partition(const std::vector<uint32_t>& a, const std::vector<int>& b){
    size_t m=a.size(); if(b.size()!=m) return false;
    for(size_t i=0;i<m;++i) for(size_t j=0;j<m;++j)
        if((a[i]==a[j]) != (b[i]==b[j])) return false;
    return true;
}
int main(){
    std::mt19937 rng(999);
    IRCanonicalizer ir;
    size_t trials=3000, bad_hash=0, bad_iso=0, bad_brute=0, nontriv=0;
    for(size_t t=0;t<trials;++t){
        uint32_t nv=2+rng()%5, ne=1+rng()%6;
        Edges e;
        for(uint32_t i=0;i<ne;++i){ uint32_t ar=2+rng()%2; std::vector<VertexId> ed;
            for(uint32_t j=0;j<ar;++j) ed.push_back(rng()%nv); e.push_back(ed); }
        if(rng()%3==0 && !e.empty()) e.push_back(e[rng()%e.size()]);

        std::vector<uint32_t> orb; uint64_t h1=ir.compute_canonical_hash_with_edge_orbits(e,orb);
        if(h1!=ir.compute_canonical_hash(e)) ++bad_hash;

        auto bco = brute_content_orbits(e,nv);
        std::vector<int> bo(e.size());
        for(size_t i=0;i<e.size();++i) bo[i]=bco[e[i]];
        { std::set<int> br(bo.begin(),bo.end()); std::set<std::vector<VertexId>> uc(e.begin(),e.end());
          if(br.size()!=uc.size()) ++nontriv; }
        if(!same_partition(orb,bo)) ++bad_brute;

        // invariance under relabel + reorder
        std::vector<VertexId> perm(nv); std::iota(perm.begin(),perm.end(),0);
        std::shuffle(perm.begin(),perm.end(),rng);
        std::vector<size_t> ep(e.size()); std::iota(ep.begin(),ep.end(),0);
        std::shuffle(ep.begin(),ep.end(),rng);
        Edges e2(e.size());
        for(size_t i=0;i<e.size();++i){ std::vector<VertexId> ed; for(VertexId v:e[ep[i]]) ed.push_back(perm[v]); e2[i]=ed; }
        std::vector<uint32_t> orb2; uint64_t h2=ir.compute_canonical_hash_with_edge_orbits(e2,orb2);
        if(h2!=h1) ++bad_hash;
        for(size_t i=0;i<e.size();++i) if(orb2[i]!=orb[ep[i]]){ ++bad_iso; break; }
    }
    printf("trials=%zu  (states where Aut fuses distinct edge contents: %zu)\n",trials,nontriv);
    printf("hash mismatches                                : %zu\n",bad_hash);
    printf("orbit ids NOT invariant under relabel+reorder  : %zu\n",bad_iso);
    printf("orbits disagree with brute-force Aut orbits    : %zu\n",bad_brute);
    bool ok=!bad_hash&&!bad_iso&&!bad_brute;
    printf("\n%s\n", ok?"EDGE ORBITS ARE INVARIANT AND MATCH BRUTE-FORCE Aut":"*** FAILURE ***");
    return ok?0:1;
}
