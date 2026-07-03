// determinism_forwarding_repro.cpp — regression checker for match-forwarding
// completeness. Runs the same evolution many times under heavy thread
// oversubscription with forwarding on and off, and reports the distinct
// (states, events, causal, branchial) count-tuples for each. The forwarding
// rendezvous must be exhaustive: both modes must produce exactly ONE tuple, and the
// same one (for the self-preserving rule at 5 steps: states=events=153, causal=152).
// More than one tuple with forwarding on means a forwarded match was lost under
// contention. Build:
//   g++ -O2 -std=c++20 -pthread -I hypergraph/include -I job_system/include \
//       -I lockfree_deque/include tools/determinism_forwarding_repro.cpp \
//       hypergraph/src/*.cpp -o /tmp/det_repro && /tmp/det_repro 16 3000
#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <set>
#include <tuple>
using namespace hypergraph;
static RewriteRule rule(){ return make_rule(0).lhs({0,1}).rhs({0,1}).rhs({1,2}).build(); }
static std::tuple<size_t,size_t,size_t,size_t> run1(bool fwd, int threads){
    Hypergraph hg;
    ParallelEvolutionEngine engine(&hg, threads);
    engine.set_match_forwarding(fwd);
    engine.add_rule(rule());
    std::vector<std::vector<VertexId>> initial = {{0,1}};
    engine.evolve(initial, 5);
    return std::make_tuple((size_t)engine.num_canonical_states(), (size_t)engine.num_events(),
                           (size_t)engine.num_causal_edges(), (size_t)engine.num_branchial_edges());
}
int main(int argc, char** argv){
    int threads = argc>1 ? atoi(argv[1]) : 16;
    int runs = argc>2 ? atoi(argv[2]) : 3000;
    for(int fi=0; fi<2; ++fi){ bool fwd = (fi==0);
        std::set<std::tuple<size_t,size_t,size_t,size_t>> seen;
        for(int i=0;i<runs;++i) seen.insert(run1(fwd, threads));
        std::printf("threads=%d forwarding=%d: %zu distinct tuples over %d runs\n", threads, (int)fwd, seen.size(), runs);
        for(auto& t : seen) std::printf("  (s=%zu e=%zu c=%zu b=%zu)\n", std::get<0>(t),std::get<1>(t),std::get<2>(t),std::get<3>(t));
    }
    return 0;
}
