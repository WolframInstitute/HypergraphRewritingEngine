// sampling_cost_smoke.cpp — bisect which arm of sampling_cost_probe does not return.
//
// One configuration per invocation, unbuffered, so a hang is attributable rather than inferred
// from an empty output file.
//
// Build:
//   g++ -O2 -std=c++20 -pthread -I hypergraph/include -I job_system/include \
//       -I lockfree_deque/include -I common/include tools/sampling_cost_smoke.cpp \
//       hypergraph/src/*.cpp -o /tmp/smoke
// Run: /tmp/smoke <arm: cap|rate|ratenofwd|fwdon|fwdoff> <rule> <edges> <steps> <threads> <k>

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

    std::printf("arm=%s rule=%s edges=%zu steps=%zu threads=%zu k=%zu\n",
                arm.c_str(), rule.c_str(), edges, steps, threads, k);

    RewriteRule r = (rule == "pair")
        ? make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build()
        : make_rule(0).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();

    std::vector<std::vector<VertexId>> init;
    for (size_t i = 0; i < edges; ++i)
        init.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});

    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_random_seed(12345);
    e.add_rule(r);
    if      (arm == "cap")    e.set_max_successor_states_per_parent(k);
    else if (arm == "rate")   e.set_transition_rate(1.0 / static_cast<double>(k));
    else if (arm == "fwdon")  e.set_match_forwarding(true);
    else if (arm == "fwdoff") e.set_match_forwarding(false);
    // The rate with forwarding disabled. Forwarded matches reach a state through
    // push_match_to_children / forward_matches_from_single_ancestor_eager, and a sampler that
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

    std::printf("done %.2f ms  states=%zu canon=%zu events=%zu matches=%zu drained=%zu\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                hg.num_states(), hg.num_canonical_states(), hg.num_events(),
                e.total_matches(), e.states_drained());
    return 0;
}
