#include "hypergraph/hypergraph.hpp"
#include "hgcommon/wl_core.hpp"
#include <cstdio>
#include <vector>
#include <map>
using namespace hypergraph;

// Flatten an edge list (vertex tuples) to local-index form and run the shared core.
static uint64_t core_hash(const std::vector<std::vector<VertexId>>& edges){
    std::map<VertexId,uint32_t> idx; 
    std::vector<uint8_t> ea; std::vector<uint32_t> eoff, ev;
    for(auto& e: edges){ eoff.push_back(ev.size()); ea.push_back(e.size());
        for(VertexId v: e){ auto it=idx.find(v); uint32_t li; if(it==idx.end()){li=idx.size(); idx[v]=li;} else li=it->second; ev.push_back(li);} }
    uint32_t nv=idx.size(), ne=edges.size();
    std::vector<uint64_t> cur(nv),nxt(nv),dscr(nv),nbr(ne*16+16),out(nv);
    std::vector<uint32_t> occ_off(nv+1),occ_edge(ev.size()); std::vector<uint8_t> occ_pos(ev.size());
    return hgcommon::wl_canonical_hash(ea.data(),eoff.data(),ev.data(),ne,nv,16,
        cur.data(),nxt.data(),occ_off.data(),occ_edge.data(),occ_pos.data(),nbr.data(),nbr.size(),dscr.data(),out.data());
}
static uint64_t cpu_hash(const std::vector<std::vector<VertexId>>& edges){
    Hypergraph hg; std::vector<EdgeId> eids;
    for(auto& e: edges) eids.push_back(hg.create_edge(e.data(), (uint8_t)e.size(), INVALID_ID, 0));
    // build a state bitset
    SparseBitset bs; for(EdgeId id: eids) bs.set(id, hg.arena());
    return hg.compute_wl_hash(bs);
}
int main(){
    std::vector<std::pair<const char*,std::vector<std::vector<VertexId>>>> tests = {
        {"triangle", {{0,1},{1,2},{2,0}}},
        {"path4",    {{0,1},{1,2},{2,3}}},
        {"star",     {{0,1},{0,2},{0,3},{0,4}}},
        {"arity3",   {{0,1,2},{2,3,4},{4,5,0}}},
        {"grid-ish", {{0,1},{1,2},{3,4},{4,5},{0,3},{1,4},{2,5}}},
        {"selfloop", {{0,0},{0,1},{1,1}}},
        {"multi",    {{0,1},{0,1},{1,2}}},
    };
    int pass=0,fail=0;
    for(auto& [name,edges]: tests){
        uint64_t c=cpu_hash(edges), k=core_hash(edges);
        bool ok=(c==k); ok?++pass:++fail;
        std::printf("  %-10s cpu=%016llx core=%016llx  %s\n", name,(unsigned long long)c,(unsigned long long)k, ok?"MATCH":"*** MISMATCH ***");
    }
    std::printf("BIT-IDENTITY: %d match, %d mismatch\n", pass, fail);
    return fail?1:0;
}
