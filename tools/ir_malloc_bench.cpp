// tools/ir_malloc_bench.cpp
//
// Measures heap allocations per IR canonicalization, across symmetry regimes.
// Build (links the IR canonicalizer source directly):
//   g++ -O2 -std=c++17 -I hypergraph/include tools/ir_malloc_bench.cpp \
//       hypergraph/src/ir_canonicalization.cpp -o /tmp/ir_malloc_bench

#include <hypergraph/ir_canonicalization.hpp>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

static long long g_allocs = 0; static bool g_track = false;
void* operator new(std::size_t n){ if(g_track)++g_allocs; void* p=std::malloc(n?n:1); if(!p)throw std::bad_alloc(); return p; }
void operator delete(void* p)noexcept{ std::free(p); }
void operator delete(void* p,std::size_t)noexcept{ std::free(p); }

using namespace hypergraph;
using Edges = std::vector<std::vector<VertexId>>;

static Edges gen_path(int n){ Edges e; for(int i=0;i+1<n;++i) e.push_back({(VertexId)i,(VertexId)(i+1)}); if(n>3){e.push_back({(VertexId)0,(VertexId)(n/2)});} return e; }
static Edges gen_cycle(int n){ Edges e; for(int i=0;i<n;++i) e.push_back({(VertexId)i,(VertexId)((i+1)%n)}); return e; }
static Edges gen_grid(int w){ Edges e; auto id=[&](int x,int y){return (VertexId)(y*w+x);}; for(int y=0;y<w;++y)for(int x=0;x<w;++x){ if(x+1<w)e.push_back({id(x,y),id(x+1,y)}); if(y+1<w)e.push_back({id(x,y),id(x,y+1)}); } return e; }

static void measure(const char* name, const Edges& edges, int iters){
    IRCanonicalizer ir;
    uint64_t h = ir.compute_canonical_hash(edges);   // warmup (fills worker_scratch)
    g_track = true; long long a0 = g_allocs;
    for(int i=0;i<iters;++i) h ^= ir.compute_canonical_hash(edges);
    long long da = g_allocs - a0; g_track = false;
    std::printf("  %-22s edges=%-4zu  mallocs/canon=%8.1f   (hash=%016llx)\n",
                name, edges.size(), (double)da/iters, (unsigned long long)h);
}

int main(){
    setvbuf(stdout,nullptr,_IONBF,0);
    std::printf("IR canonicalization heap allocations per call (lower = better):\n");
    measure("path-40 (asym)",   gen_path(40),  2000);
    measure("cycle-40 (sym)",   gen_cycle(40), 2000);
    measure("grid-7x7 (sym)",   gen_grid(7),   2000);
    measure("cycle-80 (sym)",   gen_cycle(80), 1000);
    return 0;
}
