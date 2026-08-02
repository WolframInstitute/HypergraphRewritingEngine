// unfolding_ratio_probe.cpp -- how many states does one match live in?
//
// A match is a morphism L -> G_inf into the global edge pool, and it is valid in every state
// whose edge set contains its matched edges. The engine materialises one MatchRecord per
// (state, match) PAIR; the match itself -- the MatchCore -- is one object shared by pointer.
// The ratio between the two is the up-set size: the number of states a single match lives in,
// averaged over the run.
//
//   records / cores  ==  1     one match, one state: per-pair materialisation costs nothing,
//                              and a shared representation buys nothing.
//   records / cores  >>  1     each match is re-materialised once per descendant that inherits
//                              it, and that multiple is what a membership representation removes.
//
// The same ratio measures the compression of the grammar's UNFOLDING against the transition
// system we emit: an unfolding event is determined by (rule, consumed occurrences) alone, so it
// is one per MatchCore, while the multiway graph carries one event per (state, match).
//
// Instrument: EvolutionStats counters, exact, no sampling. new_matches_discovered counts matches
// the matcher found; matches_forwarded counts those inherited from an ancestor. Cores are
// created only on discovery, so cores == discovered and records == discovered + forwarded.
//
// Build:
//   g++ -O2 -std=c++20 -pthread -I hypergraph/include -I common/include -I job_system/include \
//       -I lockfree_deque/include tools/unfolding_ratio_probe.cpp hypergraph/src/*.cpp \
//       -o /tmp/unfolding_ratio_probe && /tmp/unfolding_ratio_probe

#include "hypergraph/parallel_evolution.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace hypergraph;

namespace {

struct Workload {
    const char* name;
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> initial;
    size_t steps;
};

RewriteRule growth() {   // {{x,y}} -> {{x,y},{y,z}}: single-edge LHS, wide branching
    return make_rule(0).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();
}

RewriteRule pair_rule() {  // {{x,y},{y,z}} -> {{x,y},{y,w},{w,z}}: two-edge LHS, deeper join
    return make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build();
}

RewriteRule triangle() {  // {{x,y},{y,z},{z,x}} -> {{x,y},{y,z},{z,w},{w,x}}: cyclic LHS
    return make_rule(0).lhs({0, 1}).lhs({1, 2}).lhs({2, 0})
                       .rhs({0, 1}).rhs({1, 2}).rhs({2, 3}).rhs({3, 0}).build();
}

std::vector<std::vector<VertexId>> path(size_t n) {
    std::vector<std::vector<VertexId>> out;
    for (size_t i = 0; i < n; ++i)
        out.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});
    return out;
}

std::vector<std::vector<VertexId>> cycle(size_t n) {
    std::vector<std::vector<VertexId>> out;
    for (size_t i = 0; i < n; ++i)
        out.push_back({static_cast<VertexId>(i), static_cast<VertexId>((i + 1) % n)});
    return out;
}

std::vector<std::vector<VertexId>> disjoint(size_t n) {
    std::vector<std::vector<VertexId>> out;
    for (size_t i = 0; i < n; ++i)
        out.push_back({static_cast<VertexId>(2 * i), static_cast<VertexId>(2 * i + 1)});
    return out;
}

void run(const Workload& w, size_t threads, bool quotient) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    if (quotient) hg.set_quotient_causal(true);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_random_seed(12345);
    for (const auto& r : w.rules) e.add_rule(r);

    e.evolve(w.initial, w.steps);

    const size_t discovered = e.stats().new_matches_discovered.load();
    const size_t forwarded  = e.stats().matches_forwarded.load();
    const size_t records    = e.total_matches();
    const double ratio = discovered ? double(records) / double(discovered) : 0.0;

    std::printf("%-22s %-9s th=%zu  states=%-6zu canon=%-6zu events=%-7zu  "
                "cores=%-8zu records=%-9zu fwd=%-9zu  up-set=%.2f\n",
                w.name, quotient ? "quotient" : "full", threads,
                hg.num_states(), hg.num_canonical_states(), hg.num_events(),
                discovered, records, forwarded, ratio);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<Workload> workloads = {
        {"growth/path(8)",      {growth()},      path(8),      4},
        {"growth/path(16)",     {growth()},      path(16),     4},
        {"growth/disjoint(8)",  {growth()},      disjoint(8),  4},
        {"pair/path(12)",       {pair_rule()},   path(12),     4},
        {"pair/path(24)",       {pair_rule()},   path(24),     4},
        {"pair/cycle(8)",       {pair_rule()},   cycle(8),     4},
        {"triangle/cycle(6)",   {triangle()},    cycle(6),     3},
        {"two-rule/path(12)",   {growth(), pair_rule()}, path(12), 3},
    };

    std::printf("up-set = records/cores: the number of states one match lives in.\n"
                "1.00 means per-pair materialisation costs nothing.\n\n");

    for (const auto& w : workloads) run(w, 4, /*quotient=*/false);
    std::printf("\n");
    for (const auto& w : workloads) run(w, 4, /*quotient=*/true);
    return 0;
}
