// Does the RECONSTRUCTION already equal full capture on the rows that disagree?
//
// #4 assumed the reconstruction had to be ported from tools/quotient_reconstruction_probe.cpp.
// It does not: the engine computes it (num_reconstructed_*), labelled "the full-capture counts".
// What the golden matrix reads is causal_graph_, which under quotient is fed by qc_emit and is
// multiplicity-free. If the reconstructed numbers match full capture, #4 is a ROUTING change.
#include <cstdio>
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
using namespace hypergraph;

struct R { size_t states, events, causal, pairs, rec_ev, rec_causal, rec_pairs; };

static R run(const std::vector<RewriteRule>& rules,
             std::vector<std::vector<VertexId>> init, int steps, bool quotient) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, 4);
    e.set_explore_from_canonical_states_only(quotient);
    for (auto& r : rules) e.add_rule(r);
    e.evolve(init, steps);
    return R{hg.num_canonical_states(), hg.num_events(),
             hg.causal_graph().num_causal_edges(), hg.causal_graph().num_causal_event_pairs(),
             hg.num_reconstructed_events(), hg.num_reconstructed_causal_edges(),
             hg.num_reconstructed_causal_pairs(false)};
}

int main() {
    struct C { const char* n; std::vector<RewriteRule> r; std::vector<std::vector<VertexId>> i; int s; };
    std::vector<C> cs = {
      {"binary-growth", {make_rule(0).lhs({0,1}).rhs({0,2}).rhs({1,2}).build()}, {{0,1}}, 3},
      {"wolfram-2to4",  {make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).build()}, {{0,1},{1,2}}, 3},
      {"reductive-2to1",{make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()}, {{0,1},{1,2},{2,3},{3,4}}, 3},
      {"idempotent-2to2",{make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build()}, {{0,1},{1,2}}, 3},
    };
    std::printf("%-18s | %-22s | %-22s | %s\n", "case", "FULL ev/causal/pairs",
                "QUOT graph ev/caus/prs", "QUOT reconstructed");
    for (auto& c : cs) {
        R f = run(c.r, c.i, c.s, false);
        R q = run(c.r, c.i, c.s, true);
        std::printf("%-18s | %6zu %6zu %6zu   | %6zu %6zu %6zu   | %6zu %6zu %6zu %s\n",
            c.n, f.events, f.causal, f.pairs,
            q.events, q.causal, q.pairs,
            q.rec_ev, q.rec_causal, q.rec_pairs,
            (q.rec_ev == f.events && q.rec_causal == f.causal) ? "<= MATCHES FULL" : "");
    }
    std::printf("\n# The reconstructed columns are the engine's own num_reconstructed_*(),\n"
                "# documented as \"the full-capture counts\". They read ZERO: the per-instance\n"
                "# replay (qc_instances_/qc_apply, built by S1-S3) is WIRED but DORMANT.\n"
                "# So #4 is not a port of the 345-line DP -- the machinery exists and produces\n"
                "# nothing. Find why it is silent before writing any new propagation code.\n");
    return 0;
}
