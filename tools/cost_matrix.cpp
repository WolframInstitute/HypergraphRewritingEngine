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
#include <atomic>
#include <new>
#include <string>
#include <vector>

// Process-wide heap-allocation counter, so we can MEASURE (not assume) how much
// malloc/new the engine does — the surface we are driving to zero. Counts every
// global operator new/new[]; snapshot around the measured evolution.
namespace {
std::atomic<uint64_t> g_alloc_count{0};
std::atomic<uint64_t> g_alloc_bytes{0};
}  // namespace

void* operator new(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace hypergraph;

namespace {

struct Measured {
    size_t canonical_states;
    size_t raw_states;
    size_t events;
    size_t causal_edges;
    size_t branchial_edges;
    size_t arena_bytes;
    uint64_t heap_allocs;   // global new calls during the evolution
    uint64_t heap_bytes;    // global new bytes during the evolution
};

// Full mode, single-threaded (deterministic memory), online causal+branchial+TR.
Measured measure(const oracle::Case& c, int steps) {
    uint64_t a0 = g_alloc_count.load(std::memory_order_relaxed);
    uint64_t b0 = g_alloc_bytes.load(std::memory_order_relaxed);

    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, 1);
    engine.set_transitive_reduction(true);  // exercise the Desc/Anc closure (the O(N^2) term)
    for (const auto& r : c.rules) engine.add_rule(r);
    engine.evolve(c.init, steps);

    Measured m;
    m.canonical_states = hg.num_canonical_states();
    m.raw_states       = hg.num_states();
    m.events           = hg.num_events();
    m.causal_edges     = hg.causal_graph().num_causal_edges();
    m.branchial_edges  = hg.causal_graph().num_branchial_edges();
    m.arena_bytes      = hg.arena().bytes_allocated();
    m.heap_allocs      = g_alloc_count.load(std::memory_order_relaxed) - a0;
    m.heap_bytes       = g_alloc_bytes.load(std::memory_order_relaxed) - b0;
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    int steps_override = (argc > 1) ? std::atoi(argv[1]) : -1;

    auto cases = oracle::corpus();

    std::printf("%-18s %-20s %6s %7s %7s %7s %7s %10s %10s %9s\n",
                "case", "type", "oracle", "canon", "events",
                "causal", "branch", "arenaB", "heapB", "heapAllocs");
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

        std::printf("%-18s %-20s %6s %7zu %7zu %7zu %7zu %10zu %10llu %9llu\n",
                    c.name, c.type, verdict,
                    m.canonical_states, m.events,
                    m.causal_edges, m.branchial_edges, m.arena_bytes,
                    (unsigned long long)m.heap_bytes, (unsigned long long)m.heap_allocs);
    }

    std::printf("%s\n", std::string(120, '-').c_str());
    std::printf("exactness (oracle depth): %s\n", all_exact ? "ALL EXACT" : "*** MISMATCH ***");
    return all_exact ? 0 : 1;
}
