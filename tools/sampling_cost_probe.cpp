// sampling_cost_probe.cpp — the two measurements that were being decided by argument.
//
// A. WHAT UNIFORMITY COSTS. A uniform k-of-M sample of a state's matches cannot be applied
//    eagerly: exactly-k needs the rank, the rank needs the population, and an eager decision
//    knows only its own position. TransitionRate is the shape that survives that -- a rate
//    decides per transition and needs no population -- and the control is
//    MaxSuccessorStatesPerParent at the matching expected width: it also bounds successors,
//    but by arrival order rather than by an independent draw. What separates them is not
//    speed but what they preserve: a rate keeps the offspring DISTRIBUTION, a cap clips it to
//    a point mass.
//
// B. WHETHER MATCH FORWARDING PAYS. A child inherits its parent's still-valid matches instead
//    of re-matching. It is the single largest source of concurrency subtlety in the engine
//    (per-state epochs, the forwarding rendezvous, push_match_to_children), and whether it is
//    worth that has never been measured -- only argued. ON == OFF equality is already proven
//    by determinism_forwarding_repro, so this is purely throughput.
//
// Wall clock is the only instrument for either. Both effects are LOST PARALLELISM: the
// deferral idles workers that had work a moment ago, and forwarding trades matching work for
// coordination. A serialising profiler reports instructions, which is precisely the quantity
// that does not move here. So: interleaved A/B (adjacent in time, so drift hits both arms) and
// minimum-of-N (the minimum is the run least disturbed by the rest of the machine).
//
// Build:
//   g++ -O2 -std=c++20 -pthread -I hypergraph/include -I job_system/include \
//       -I lockfree_deque/include tools/sampling_cost_probe.cpp hypergraph/src/*.cpp \
//       -o /tmp/sampling_cost_probe && /tmp/sampling_cost_probe

#include "hypergraph/parallel_evolution.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

// Unbuffered: a probe that hangs must say which arm it hung in, not leave an empty file.

using namespace hypergraph;

namespace {

struct Workload {
    const char* name;
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> initial;
    size_t steps;
};

struct Run {
    double ms;
    size_t states;
    size_t canonical;
    size_t events;
    size_t matches;
};

// One arm of a comparison. `configure` sets whatever knob the arm is testing.
template <typename Configure>
Run once(const Workload& w, size_t threads, Configure configure) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_random_seed(12345);
    for (const auto& r : w.rules) e.add_rule(r);
    configure(e);

    const auto t0 = std::chrono::steady_clock::now();
    e.evolve(w.initial, w.steps);
    const auto t1 = std::chrono::steady_clock::now();

    Run out;
    out.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.states = hg.num_states();
    out.canonical = hg.num_canonical_states();
    out.events = hg.num_events();
    out.matches = e.total_matches();
    return out;
}

// Interleaved A/B, minimum-of-N. Runs are adjacent in time so machine drift lands on both
// arms; the minimum is reported because a slow run means something else ran, not that the
// code is slower.
template <typename ConfigA, typename ConfigB>
void compare(const char* label, const Workload& w, size_t threads, size_t reps,
             const char* name_a, ConfigA ca, const char* name_b, ConfigB cb) {
    Run best_a{1e18, 0, 0, 0, 0};
    Run best_b{1e18, 0, 0, 0, 0};
    for (size_t i = 0; i < reps; ++i) {
        Run a = once(w, threads, ca);
        Run b = once(w, threads, cb);
        if (a.ms < best_a.ms) best_a = a;
        if (b.ms < best_b.ms) best_b = b;
    }

    const double delta = 100.0 * (best_b.ms - best_a.ms) / best_a.ms;
    std::printf("%-14s %-26s  %s %8.2f ms  states=%-6zu canon=%-6zu events=%-6zu matches=%zu\n",
                label, w.name, name_a, best_a.ms,
                best_a.states, best_a.canonical, best_a.events, best_a.matches);
    std::printf("%-14s %-26s  %s %8.2f ms  states=%-6zu canon=%-6zu events=%-6zu matches=%zu"
                "   [%+.1f%%]\n",
                "", "", name_b, best_b.ms,
                best_b.states, best_b.canonical, best_b.events, best_b.matches, delta);

    // A comparison across different amounts of work is not a comparison. Say so rather than
    // letting the percentage stand on its own.
    if (best_a.events != best_b.events) {
        std::printf("%-14s %-26s  NOTE: the arms did different work (%zu vs %zu events), so the "
                    "percentage above mixes cost with work\n",
                    "", "", best_a.events, best_b.events);
    }
    std::printf("\n");
}

RewriteRule growth() {   // {{x,y}} -> {{x,y},{y,z}}: one match per edge, branches wide
    return make_rule(0).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();
}

RewriteRule pair_rule() {  // {{x,y},{y,z}} -> {{x,y},{y,w},{w,z}}: two-edge LHS, deeper join
    return make_rule(0).lhs({0, 1}).lhs({1, 2}).rhs({0, 1}).rhs({1, 3}).rhs({3, 2}).build();
}

std::vector<std::vector<VertexId>> path(size_t n) {
    std::vector<std::vector<VertexId>> out;
    for (size_t i = 0; i < n; ++i)
        out.push_back({static_cast<VertexId>(i), static_cast<VertexId>(i + 1)});
    return out;
}

std::vector<std::vector<VertexId>> disjoint(size_t n) {
    std::vector<std::vector<VertexId>> out;
    for (size_t i = 0; i < n; ++i)
        out.push_back({static_cast<VertexId>(2 * i), static_cast<VertexId>(2 * i + 1)});
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const size_t threads = (argc > 1) ? std::stoul(argv[1]) : 4;
    const size_t reps    = (argc > 2) ? std::stoul(argv[2]) : 5;
    const size_t k       = (argc > 3) ? std::stoul(argv[3]) : 4;

    std::printf("threads=%zu reps=%zu k=%zu  (minimum of %zu interleaved runs per arm)\n\n",
                threads, reps, k, reps);

    std::vector<Workload> workloads = {
        // Narrow start: one state at the root, so the deferral has nothing to overlap with and
        // its cost should be at its worst.
        {"narrow/growth/d7",   {growth()},    path(1),      7},
        {"narrow/pair/d6",     {pair_rule()}, path(3),      6},
        // Wide start: many states in flight immediately, so a per-state join should be hidden
        // by inter-state parallelism. If the deferral costs the same here as above, the "it is
        // only intra-state" argument is wrong.
        //
        // Depths are set from measurement, not from symmetry with the narrow cases: the
        // two-edge LHS joins over a 24-edge path, so each extra step multiplies the state
        // count by ~26 (14k states at d3, 373k at d4, 11.4 s). d3 is the last depth that
        // leaves the whole probe in the tens of seconds.
        //
        // The wide-growth start is a PATH, not disjoint edges: n identical disjoint edges
        // have Aut containing S_n, and Full-mode IR on that wall dominates everything this
        // probe measures (measured: depth-1 canonicalization alone is 0.4 / 5.7 / 178 ms at
        // n = 6 / 12 / 24, and the tree multiplies it past any budget). A path gives the
        // same 24-match width with trivial automorphisms.
        {"wide/growth/d4",     {growth()},    path(24),     4},
        {"wide/pair/d3",       {pair_rule()}, path(24),     3},
    };

    std::printf("=== A. cap vs rate: same expected width, different offspring distribution ===\n");
    std::printf("    cap: MaxSuccessorStatesPerParent=%zu, eager, first-k by arrival\n", k);
    std::printf("    rte: TransitionRate=1/%zu, eager, independent per transition\n\n", k);
    for (const auto& w : workloads) {
        compare("A/eagerness", w, threads, reps,
                "cap", [&](ParallelEvolutionEngine& e) {
                    e.set_max_successor_states_per_parent(k);
                },
                "rte", [&](ParallelEvolutionEngine& e) {
                    e.set_transition_rate(1.0 / static_cast<double>(k));
                });
    }

    std::printf("=== B. whether match forwarding pays (identical output, throughput only) ===\n\n");
    for (const auto& w : workloads) {
        compare("B/forwarding", w, threads, reps,
                " on", [](ParallelEvolutionEngine& e) { e.set_match_forwarding(true); },
                "off", [](ParallelEvolutionEngine& e) { e.set_match_forwarding(false); });
    }

    return 0;
}
