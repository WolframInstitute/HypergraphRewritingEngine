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

// The ALIGNED overloads are separate functions, and an over-aligned type routes to them
// rather than to the plain ones above. Replacing only the plain pair left this counter blind
// to every such allocation -- including the arena's `new LocalCursor[MAX_ARENA_WORKERS]`,
// where LocalCursor is alignas(64), so 16 KB per arena went uncounted by the very instrument
// the de-heap numbers were measured with.
void* operator new(std::size_t n, std::align_val_t a) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
    if (void* p = std::aligned_alloc(static_cast<std::size_t>(a),
                                     (n + static_cast<std::size_t>(a) - 1) &
                                         ~(static_cast<std::size_t>(a) - 1))) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

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
//
// `rec` selects which artifacts the run RECORDS. Measuring both settings on the same workload
// is what turns "the causal graph is built even when unrequested" into a number: the states and
// events must be identical between the two, and the difference is what recording them costs.
// Which engine the probe drives. Serial spawns no thread, which is the only mode available on
// a target without them -- and running the corpus there is what says the engine WORKS there,
// as against merely compiling.
ParallelEvolutionEngine::ExecutionMode g_mode = ParallelEvolutionEngine::ExecutionMode::Parallel;

Measured measure(const oracle::Case& c, int steps, RecordSet rec = RecordSet{}) {
    uint64_t a0 = g_alloc_count.load(std::memory_order_relaxed);
    uint64_t b0 = g_alloc_bytes.load(std::memory_order_relaxed);

    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_record_set(rec);
    ParallelEvolutionEngine engine(&hg, 1, g_mode);
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
    int steps_override = -1;
    std::string only;            // --case NAME: run this workload alone
    const char* record = "";     // --record all|none: run it under one record set alone

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--case" && i + 1 < argc) only = argv[++i];
        else if (a == "--record" && i + 1 < argc) record = argv[++i];
        else if (a == "--serial") g_mode = ParallelEvolutionEngine::ExecutionMode::Serial;
        else steps_override = std::atoi(argv[i]);
    }

    // One case, one record set, nothing else: the shape a profiler can attribute. The exactness
    // check and the paired second run are skipped, because both would land in the same profile.
    if (!only.empty() && *record) {
        const RecordSet rs = (std::string(record) == "none")
                                 ? RecordSet{false, false, false} : RecordSet{};
        for (const auto& c : oracle::corpus()) {
            if (only != c.name) continue;
            const int steps = (steps_override > 0) ? steps_override : c.measure_steps;
            const Measured m = measure(c, steps, rs);
            std::printf("%s record=%s states=%zu events=%zu causal=%zu branchial=%zu arenaB=%zu\n",
                        c.name, record, m.canonical_states, m.events, m.causal_edges,
                        m.branchial_edges, m.arena_bytes);
            return 0;
        }
        std::fprintf(stderr, "no such case: %s\n", only.c_str());
        return 2;
    }

    auto cases = oracle::corpus();

    // The last column is the SAME evolution recording NEITHER relation, so the gap against
    // arenaB is what the causal and branchial graphs cost a caller who asked for neither.
    std::printf("engine: %s\n",
                g_mode == ParallelEvolutionEngine::ExecutionMode::Serial ? "serial" : "threaded");
    std::printf("%-18s %-20s %6s %7s %7s %7s %7s %10s %10s %9s %10s\n",
                "case", "type", "oracle", "canon", "events",
                "causal", "branch", "arenaB", "heapB", "heapAllocs", "noRelB");
    // The last column is the SAME evolution recording neither relation, so the difference
    // against arenaB is what the causal and branchial graphs cost a caller who asked for
    // neither.
    std::printf("%s\n", std::string(120, '-').c_str());

    bool all_exact = true;
    bool any_unverified = false;   // a row the oracle could not check
    size_t total_all = 0, total_states_only = 0;
    for (const auto& c : cases) {
        // Exactness: engine Full-count vs brute-force iso count at the oracle depth.
        bool all_small = true;
        size_t brute = oracle::brute_force_iso_count(c.rules, c.init, c.oracle_steps, &all_small,
                                                    g_mode);
        size_t full  = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 1, g_mode);
        const char* verdict;
        // "oversz" means the brute-force oracle could not run, so exactness was NOT checked
        // for this row. Leaving all_exact alone let the summary print ALL EXACT having
        // verified nothing -- a not-measured reported as a pass.
        if (!all_small)        { verdict = "oversz"; any_unverified = true; }
        else if (full == brute)  verdict = "EXACT";
        else { verdict = "FAIL"; all_exact = false; }

        int steps = (steps_override > 0) ? steps_override : c.measure_steps;
        Measured m = measure(c, steps);
        // The same evolution recording neither relation. States and events must not move.
        Measured s = measure(c, steps, RecordSet{false, false});
        if (s.canonical_states != m.canonical_states || s.events != m.events) {
            std::printf("  %-16s STATES/EVENTS MOVED when the relations were not recorded: "
                        "%zu/%zu against %zu/%zu\n", c.name, s.canonical_states, s.events,
                        m.canonical_states, m.events);
            all_exact = false;
        }
        total_all += m.arena_bytes;
        total_states_only += s.arena_bytes;

        std::printf("%-18s %-20s %6s %7zu %7zu %7zu %7zu %10zu %10llu %9llu %10zu\n",
                    c.name, c.type, verdict,
                    m.canonical_states, m.events,
                    m.causal_edges, m.branchial_edges, m.arena_bytes,
                    (unsigned long long)m.heap_bytes, (unsigned long long)m.heap_allocs,
                    s.arena_bytes);
    }

    std::printf("%s\n", std::string(120, '-').c_str());
    std::printf("arena bytes, all artifacts / neither relation: %zu / %zu  (%.1f%% of the arena "
                "is the causal and branchial graphs)\n", total_all, total_states_only,
                total_all ? 100.0 * double(total_all - total_states_only) / double(total_all) : 0.0);
    std::printf("exactness (oracle depth): %s%s\n",
                all_exact ? "ALL EXACT" : "*** MISMATCH ***",
                any_unverified ? "  (some rows UNVERIFIED: oversized for the brute-force oracle)" : "");
    // An unverified row is not a pass. Exit non-zero so nothing can gate on a run that
    // silently checked less than it appears to.
    return (all_exact && !any_unverified) ? 0 : 1;
}
