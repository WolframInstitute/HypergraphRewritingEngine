// Does push_match_to_children have anything to do when it is called at DISCOVERY?
//
// WHY THIS EXISTS. Batched submission costs 13.52% more arena than eager (cost_matrix, 17 cases;
// worst case star4-automorphic at 20.88%) and produces identical output. Batched is the default
// regardless, because eager LOSES MATCHES -- 1 to 7 of 204 runs against 0 of 51 -- and forwarding
// is inductive, so a lost match deletes its whole subtree while the run stays self-consistent.
// The memory is recoverable; the lost matches are not. #77 is the recovery.
//
// THE HYPOTHESIS UNDER TEST. Two mechanisms forward matches, covering complementary windows:
//
//   PULL   at child creation, the child walks its ancestor chain and takes each ancestor's
//          matches, filtering by the consumed edges accumulated along the path. Covers matches
//          that existed BEFORE the child did.
//   PUSH   a match arriving in a state is forwarded to that state's registered children,
//          recursively. Covers matches arriving AFTER the child exists.
//
// Under batched submission a state's matching COMPLETES before any of its rewrites are submitted,
// so no child of that state should exist while its matches are being discovered. If that holds,
// the DISCOVERY-time push always finds an empty child registry and is a no-op, and everything it
// would have delivered is delivered instead by the child's pull at creation -- when the parent's
// match set is already complete. Only the FORWARDING-time push can find children.
//
// WHAT WOULD FALSIFY IT. A discovery-site empty fraction below 1.0. That would mean children
// become visible earlier than the batching argument says, and the 13.52% has a different cause --
// in which case DO NOT skip the call on the strength of the argument.
//
// This measures; it does not change behaviour. Removing the call is a separate step gated on
// this number, on cost_matrix over the same 17 cases, and on test_match_completeness holding at
// zero misses.
//
// Usage: push_site_probe [steps]

#include "hypergraph/parallel_evolution.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace hypergraph;

namespace {

struct Workload {
    const char* name;
    std::vector<RewriteRule> rules;
    std::vector<std::vector<VertexId>> init;
};

std::vector<Workload> workloads() {
    std::vector<Workload> w;
    w.push_back({"wolfram-2to4",
        {make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).rhs({2,0}).build()},
        {{0,1},{1,2}}});
    w.push_back({"binary-growth",
        {make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()},
        {{0,1}}});
    w.push_back({"WPP",
        {make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
        {{0,1},{0,2}}});
    w.push_back({"star4-automorphic",
        {make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,2}).build()},
        {{0,1},{0,2},{0,3},{0,4}}});
    return w;
}

struct Row {
    std::string name;
    int threads;
    size_t d_calls, d_empty, f_calls, f_empty, matches;
};

Row run(const Workload& w, int threads, int steps, bool batched) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&g, threads);
    e.set_batched_matching(batched);
    for (const auto& r : w.rules) e.add_rule(r);
    e.evolve(w.init, steps);

    const auto& s = e.stats();
    return {w.name, threads,
            s.push_discovery_calls.load(), s.push_discovery_empty.load(),
            s.push_forwarding_calls.load(), s.push_forwarding_empty.load(),
            s.matches_found.load()};
}

void report(const char* mode, const std::vector<Row>& rows) {
    std::printf("\n=== %s ===\n", mode);
    std::printf("%-20s %-4s %-22s %-22s %s\n",
                "workload", "thr", "discovery empty/calls", "forwarding empty/calls", "matches");
    size_t td = 0, tde = 0, tf = 0, tfe = 0;
    for (const Row& r : rows) {
        auto frac = [](size_t e, size_t c) { return c ? double(e) / double(c) : 1.0; };
        std::printf("%-20s %-4d %8zu/%-8zu %.3f  %8zu/%-8zu %.3f  %zu\n",
                    r.name.c_str(), r.threads,
                    r.d_empty, r.d_calls, frac(r.d_empty, r.d_calls),
                    r.f_empty, r.f_calls, frac(r.f_empty, r.f_calls),
                    r.matches);
        td += r.d_calls; tde += r.d_empty; tf += r.f_calls; tfe += r.f_empty;
    }
    std::printf("%-20s %-4s %8zu/%-8zu %.3f  %8zu/%-8zu %.3f\n", "TOTAL", "",
                tde, td, td ? double(tde) / double(td) : 1.0,
                tfe, tf, tf ? double(tfe) / double(tf) : 1.0);
}

}  // namespace

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 4;

    std::vector<Row> batched, eager;
    for (const auto& w : workloads()) {
        for (int t : {1, 4}) {
            batched.push_back(run(w, t, steps, /*batched=*/true));
            eager.push_back(run(w, t, steps, /*batched=*/false));
        }
    }

    std::printf("push_match_to_children: does it find children, split by call site (steps=%d)\n",
                steps);
    report("BATCHED (the default)", batched);
    report("EAGER (for contrast)", eager);

    size_t d = 0, de = 0;
    for (const Row& r : batched) { d += r.d_calls; de += r.d_empty; }
    std::printf("\nBatched discovery-site empty fraction: %.4f over %zu calls.\n",
                d ? double(de) / double(d) : 1.0, d);
    std::printf("1.0000 means the discovery-time push never has anything to do under batched,\n"
                "so it can be skipped. Anything less falsifies that and the 13.52%% is elsewhere.\n");
    return 0;
}
