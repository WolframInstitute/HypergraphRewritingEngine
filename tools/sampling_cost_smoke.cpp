// sampling_cost_smoke.cpp — bisect which arm of sampling_cost_probe does not return.
//
// One configuration per invocation, unbuffered, so a hang is attributable rather than inferred
// from an empty output file.
//
// Run: /tmp/smoke <arm: off|cap|rate|ratenofwd|fwdon|fwdoff> <rule> <edges> <steps> <threads> <k> <canon: full|automatic|none>
//
// `off` and the canon mode exist to answer what sampling COSTS, which is a different question
// from what it PRESERVES. The sampling draw is keyed on canonical_transition_key, whose first
// act is ensure_state_edge_ranks -- an individualization-refinement pass. Under Full that pass
// happens anyway for dedup, so the key is nearly free. Under Automatic (content hash) and None
// (a per-state counter) NO IR pass otherwise runs, so switching sampling on ADDS the engine's
// most expensive operation per state. Whether the pruning repays it is a crossover, and this is
// the arm pair that locates it: `off` against `rate` at the same canon mode.

#include "hypergraph/parallel_evolution.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace hypergraph;

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string arm  = (argc > 1) ? argv[1] : "cap";
    const std::string rule = (argc > 2) ? argv[2] : "growth";
    const size_t edges     = (argc > 3) ? std::stoul(argv[3]) : 1;
    const size_t steps     = (argc > 4) ? std::stoul(argv[4]) : 7;
    const size_t threads   = (argc > 5) ? std::stoul(argv[5]) : 4;
    const size_t k         = (argc > 6) ? std::stoul(argv[6]) : 4;
    const std::string canon = (argc > 7) ? argv[7] : "full";

    std::printf("arm=%s rule=%s edges=%zu steps=%zu threads=%zu k=%zu canon=%s\n",
                arm.c_str(), rule.c_str(), edges, steps, threads, k, canon.c_str());

    // LHS SIZE IS THE SWEPT VARIABLE, so the shapes have to differ in that and in nothing else.
    // chain_rule(n) is an n-edge CONNECTED left-hand side -- edges (0,1),(1,2),...,(n-1,n),
    // consecutive edges sharing a vertex -- rewritten by subdividing the LAST edge through one
    // fresh vertex. Every n therefore has the same net effect on the state (+1 edge) and the
    // same right-hand side shape, and the only thing that changes with n is how much of the
    // state a match must bind. chain_rule(2) IS the two-edge rule the published sweep uses;
    // it is generated here rather than written out again so the family cannot drift apart.
    auto chain_rule = [](size_t n) {
        auto b = make_rule(0);
        for (size_t i = 0; i < n; ++i)
            b.lhs({static_cast<uint8_t>(i), static_cast<uint8_t>(i + 1)});
        for (size_t i = 0; i + 1 < n; ++i)
            b.rhs({static_cast<uint8_t>(i), static_cast<uint8_t>(i + 1)});
        const auto fresh = static_cast<uint8_t>(n + 1);
        b.rhs({static_cast<uint8_t>(n - 1), fresh});
        b.rhs({fresh, static_cast<uint8_t>(n)});
        return b.build();
    };

    RewriteRule r =
        rule == "pair"   ? chain_rule(2) :
        rule == "triple" ? chain_rule(3) :
        rule == "quad"   ? chain_rule(4) :
        // TWO COMPONENTS SHARING NO VARIABLE. Matching this cannot join -- there is no shared
        // vertex to join on -- so it enumerates the cartesian product of the state's edges,
        // quadratic in state size. It is the shape the engine warns about, and it belongs in
        // the LHS-size sweep as the point where connectedness rather than size is what changed.
        rule == "disc"   ? make_rule(0).lhs({0, 1}).lhs({2, 3})
                                       .rhs({0, 1}).rhs({2, 4}).rhs({4, 3}).build() :
        // "growth": the one-edge LHS, which keeps its edge and appends rather than subdividing.
                           make_rule(0).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();

    std::vector<std::vector<VertexId>> init;
    for (size_t i = 0; i < edges; ++i)
        init.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});

    Hypergraph hg;
    hg.set_state_canonicalization_mode(
        canon == "none"      ? StateCanonicalizationMode::None :
        canon == "automatic" ? StateCanonicalizationMode::Automatic
                             : StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_random_seed(12345);
    e.add_rule(r);
    if      (arm == "off")    { /* the baseline: no sampling of any kind */ }
    else if (arm == "cap")    e.set_max_successor_states_per_parent(k);
    else if (arm == "rate")   e.set_transition_rate(1.0 / static_cast<double>(k));
    else if (arm == "fwdon")  e.set_match_forwarding(true);
    else if (arm == "fwdoff") e.set_match_forwarding(false);
    // The rate with forwarding disabled. Forwarded matches reach a state through
    // push_match_to_children / forward_matches_from_single_ancestor, and a sampler that
    // did not reach those dispatches would bound the run here and not with forwarding on --
    // which is how the per-state reservoir this arm replaced was caught.
    else if (arm == "ratenofwd") {
        e.set_transition_rate(1.0 / static_cast<double>(k));
        e.set_match_forwarding(false);
    }

    std::printf("evolving...\n");
    const auto t0 = std::chrono::steady_clock::now();
    e.evolve(init, steps);
    const auto t1 = std::chrono::steady_clock::now();

    // Per-depth width and the branching factor between depths. A fixed thinning rate q makes
    // the expected width evolve as the product of m(d)*q, so whether q must depend on depth is
    // entirely the question of whether m does -- which is this histogram, not an argument.
    size_t max_width = 0, depth_reached = 0;
    {
        std::vector<size_t> width;
        for (uint32_t sid = 0; sid < hg.num_published_states(); ++sid) {
            const auto& st = hg.get_state(sid);
            if (st.id == INVALID_ID) continue;
            if (st.step >= width.size()) width.resize(st.step + 1, 0);
            width[st.step]++;
        }
        for (size_t d = 0; d < width.size(); ++d)
            if (width[d] > max_width) max_width = width[d];
        depth_reached = width.empty() ? 0 : width.size() - 1;
        std::printf("depth  width      m(d)=width(d+1)/width(d)\n");
        for (size_t d = 0; d < width.size(); ++d) {
            if (d + 1 < width.size() && width[d] > 0) {
                std::printf("%5zu  %-9zu %.2f\n", d, width[d],
                            static_cast<double>(width[d + 1]) / width[d]);
            } else {
                std::printf("%5zu  %-9zu -\n", d, width[d]);
            }
        }
    }

    std::printf("done %.2f ms  states=%zu canon=%zu events=%zu matches=%zu drained=%zu\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                hg.num_states(), hg.num_canonical_states(), hg.num_events(),
                e.total_matches(), e.states_drained());

    // ONE MACHINE-READABLE LINE CARRYING EVERY COUNT THIS RUN ALREADY COMPUTED. The `done` line
    // above reports the size of the state space; these are the sizes of the RELATIONS over it,
    // and they are what distinguishes one workload from another at equal state count -- a run
    // with 200k states and 40k causal edges is a different object from one with 200k states and
    // 4M. They cost nothing here: the causal and branchial accessors are derived from sets the
    // evolution already built, so reporting them adds no work to the measured region, which ends
    // at t1 above.
    //
    // key=value, one line, stable key order, so a sweep can parse it without a format per plot.
    // The causal and branchial sizes come from the CausalGraph the full-capture run actually
    // built. The num_reconstructed_* accessors report the quotient reconstruction and read zero
    // outside quotient mode, so they are the wrong instrument for this sweep.
    const auto& cg = hg.causal_graph();
    std::printf("RICH rule=%s lhs_edges=%zu init_edges=%zu steps=%zu threads=%zu canon_mode=%s"
                " ms=%.3f states=%zu canonical=%zu events=%zu matches=%zu"
                " causal_edges=%zu causal_pairs=%zu branchial_edges=%zu branchial_claimed=%zu"
                " max_width=%zu depth_reached=%zu\n",
                rule.c_str(),
                rule == "pair" ? size_t{2} : rule == "triple" ? size_t{3}
                    : rule == "quad" ? size_t{4} : rule == "disc" ? size_t{2} : size_t{1},
                edges, steps, threads, canon.c_str(),
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                hg.num_states(), hg.num_canonical_states(),
                static_cast<size_t>(hg.num_events()), e.total_matches(),
                cg.num_causal_edges(), cg.num_causal_event_pairs(),
                cg.num_branchial_edges(), cg.num_branchial_pairs_claimed(),
                max_width, depth_reached);
    return 0;
}
