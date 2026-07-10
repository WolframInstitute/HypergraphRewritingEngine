// Checks compute_canonical_hash_with_edge_map against compute_canonical_hash and
// against the properties the multiplicity propagation relies on:
//   - the hash is unchanged
//   - edge content classes are isomorphism invariant (vertex relabel + edge reorder)
//   - edges with identical content share a class
//   - classes are contiguous from zero
#include "hypergraph/ir_canonicalization.hpp"
#include <cstdio>
#include <random>
#include <algorithm>
#include <map>
#include <numeric>
#include <vector>
#include <set>
using namespace hypergraph;
using Edges = std::vector<std::vector<VertexId>>;

int main(){
    std::mt19937 rng(12345);
    IRCanonicalizer ir;
    size_t trials=3000, bad_hash=0, bad_iso=0, bad_dup=0, bad_contig=0, dup_states=0;
    size_t iso_fail_with_aut=0, iso_fail_no_aut=0;

    for (size_t t=0;t<trials;++t){
        uint32_t nv = 2 + rng()%6;
        uint32_t ne = 1 + rng()%7;
        Edges e;
        for (uint32_t i=0;i<ne;++i){
            uint32_t ar = 2 + rng()%2;
            std::vector<VertexId> ed;
            for (uint32_t j=0;j<ar;++j) ed.push_back(rng()%nv);
            e.push_back(ed);
        }
        // sometimes force a duplicate edge
        if (rng()%3==0 && !e.empty()){ e.push_back(e[rng()%e.size()]); }

        std::vector<uint32_t> cls;
        uint64_t h1 = ir.compute_canonical_hash_with_edge_map(e, cls);
        uint64_t h0 = ir.compute_canonical_hash(e);
        if (h0!=h1) ++bad_hash;

        // duplicates share a class
        std::map<std::vector<VertexId>, uint32_t> seen;
        bool dup_here=false;
        for (size_t i=0;i<e.size();++i){
            auto it=seen.find(e[i]);
            if (it==seen.end()) seen[e[i]]=cls[i];
            else { dup_here=true; if (it->second != cls[i]) ++bad_dup; }
        }
        if (dup_here) ++dup_states;

        // contiguous classes from 0
        if (!cls.empty()){
            std::vector<uint32_t> u=cls; std::sort(u.begin(),u.end()); u.erase(std::unique(u.begin(),u.end()),u.end());
            if (u.front()!=0 || u.back()+1!=u.size()) ++bad_contig;
        }

        // isomorphism invariance: relabel vertices, permute edge order
        std::vector<VertexId> perm(nv); std::iota(perm.begin(),perm.end(),0);
        std::shuffle(perm.begin(),perm.end(),rng);
        std::vector<size_t> eperm(e.size()); std::iota(eperm.begin(),eperm.end(),0);
        std::shuffle(eperm.begin(),eperm.end(),rng);
        Edges e2(e.size());
        for (size_t i=0;i<e.size();++i){
            std::vector<VertexId> ed;
            for (VertexId v : e[eperm[i]]) ed.push_back(perm[v]);
            e2[i]=ed;
        }
        std::vector<uint32_t> cls2;
        uint64_t h2 = ir.compute_canonical_hash_with_edge_map(e2, cls2);
        if (h2!=h1) ++bad_hash;
        bool iso_fail=false;
        for (size_t i=0;i<e.size();++i) if (cls2[i]!=cls[eperm[i]]) { iso_fail=true; break; }
        if (iso_fail){
            ++bad_iso;
            // brute-force |Aut| of the edge multiset over vertex permutations
            std::vector<VertexId> pv(nv); std::iota(pv.begin(),pv.end(),0);
            std::multiset<std::vector<VertexId>> base(e.begin(), e.end());
            size_t aut=0;
            do {
                std::multiset<std::vector<VertexId>> img;
                for (auto& ed : e){ std::vector<VertexId> m; for(VertexId v:ed) m.push_back(pv[v]); img.insert(m); }
                if (img==base) ++aut;
            } while (std::next_permutation(pv.begin(), pv.end()));
            if (aut>1) ++iso_fail_with_aut; else ++iso_fail_no_aut;
        }
    }
    printf("trials=%zu (with duplicate edges: %zu)\n", trials, dup_states);
    printf("hash mismatches vs compute_canonical_hash / under iso : %zu\n", bad_hash);
    printf("edge classes varying under relabel+reorder            : %zu\n", bad_iso);
    printf("duplicate edges assigned different classes            : %zu\n", bad_dup);
    printf("class ids not contiguous from zero                    : %zu\n", bad_contig);
    printf("  of the iso failures: %zu have |Aut|>1, %zu have trivial Aut\n", iso_fail_with_aut, iso_fail_no_aut);
    // Edge content classes are canonical only up to Aut acting on edges: with a
    // nontrivial automorphism group several labelings reach the same canonical form
    // and permute edges between classes. A violation with trivial Aut would be a bug.
    bool ok = !bad_hash && !bad_dup && !bad_contig && iso_fail_no_aut==0;
    printf("\n%s\n", ok?"ALL PROPERTIES HOLD (class invariance holds exactly when |Aut|==1)"
                       :"*** FAILURE ***");
    return ok?0:1;
}
