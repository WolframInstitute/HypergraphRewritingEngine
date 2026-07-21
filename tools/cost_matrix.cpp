// Cost matrix: for a diverse rule corpus, prove EXACTNESS against the brute-force
// oracle and MEASURE the engine's memory + output size, so optimizations can be
// shown to preserve correctness while reducing cost. Memory is the arena's
// bytes_allocated() (durable, noise-free) — not RSS — so results are reproducible.
//
// Usage: cost_matrix [measure_steps_override]
//   Run before and after a change; the exactness column must stay EXACT and the
//   memory column proves the win. This is the harness the paper's ablation uses.

#include "../reference/oracle_corpus.hpp"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

using namespace hypergraph;

namespace {

struct Measured {
    size_t canonical_states;
    size_t raw_states;
    size_t events;
    size_t causal_edges;
    size_t branchial_edges;
    size_t arena_bytes;
};

// Full mode, single-threaded (deterministic memory), online causal+branchial+TR.
Measured measure(const oracle::Case& c, int steps) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, 1);
    for (const auto& r : c.rules) engine.add_rule(r);
    engine.evolve(c.init, steps);

    Measured m;
    m.canonical_states = hg.num_canonical_states();
    m.raw_states       = hg.num_states();
    m.events           = hg.num_events();
    m.causal_edges     = hg.causal_graph().num_causal_edges();
    m.branchial_edges  = hg.causal_graph().num_branchial_edges();
    m.arena_bytes      = hg.arena().bytes_allocated();
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    int steps_override = (argc > 1) ? std::atoi(argv[1]) : -1;

    auto cases = oracle::corpus();

    std::printf("%-18s %-22s %6s %8s %8s %8s %8s %8s %10s %9s\n",
                "case", "type", "oracle", "canon", "raw", "events",
                "causal", "branch", "arenaB", "B/state");
    std::printf("%s\n", std::string(120, '-').c_str());

    bool all_exact = true;
    for (const auto& c : cases) {
        // Exactness: engine Full-count vs brute-force iso count at the oracle depth.
        bool all_small = true;
        size_t brute = oracle::brute_force_iso_count(c.rules, c.init, c.oracle_steps, &all_small);
        size_t full  = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 1);
        const char* verdict;
        if (!all_small)          verdict = "oversz";
        else if (full == brute)  verdict = "EXACT";
        else { verdict = "FAIL"; all_exact = false; }

        int steps = (steps_override > 0) ? steps_override : c.measure_steps;
        Measured m = measure(c, steps);
        double b_per_state = m.canonical_states ? double(m.arena_bytes) / double(m.canonical_states) : 0.0;

        std::printf("%-18s %-22s %6s %8zu %8zu %8zu %8zu %8zu %10zu %9.1f\n",
                    c.name, c.type, verdict,
                    m.canonical_states, m.raw_states, m.events,
                    m.causal_edges, m.branchial_edges, m.arena_bytes, b_per_state);
    }

    std::printf("%s\n", std::string(120, '-').c_str());
    std::printf("exactness (oracle depth): %s\n", all_exact ? "ALL EXACT" : "*** MISMATCH ***");
    return all_exact ? 0 : 1;
}
