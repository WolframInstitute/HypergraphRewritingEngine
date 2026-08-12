// Where does the batched default's extra cost actually go?
//
// WHY THIS EXISTS. Batched submission costs 13.52% more arena than eager (cost_matrix, 17 cases;
// worst case star4-automorphic at 20.88%) and produces identical output. Batched is the default
// regardless, because eager LOSES MATCHES -- 1 to 7 of 204 runs against 0 of 51 -- and forwarding
// is inductive, so a lost match deletes its whole subtree while the run stays self-consistent.
// The memory is recoverable; the lost matches are not. #77 is the recovery, and this locates it.
//
// TWO MECHANISMS FORWARD MATCHES, covering complementary windows:
//
//   PULL   at child creation, the child walks its ancestor chain and takes each ancestor's
//          matches, filtering by the consumed edges accumulated along the path. Covers matches
//          that existed BEFORE the child did.
//   PUSH   a match arriving in a state is forwarded to that state's registered children,
//          recursively. Covers matches arriving AFTER the child exists.
//
// WHAT THIS MEASURED, AND WHAT IT REFUTED. The reasoning going in was that under batched
// submission a state's matching completes before any of its rewrites are submitted, so no child
// exists during discovery, so the DISCOVERY-time push is a no-op the child's pull would cover.
// The first run reported a discovery-site empty fraction of 0.062: 93.8% of discovery pushes DO
// find children, so children become visible earlier than that argument claims and the call cannot
// be removed. The waste is at the FORWARDING site, which finds nothing 92.2% of the time.
//
// SO THE COLUMNS BELOW ARE THE ANSWER, NOT THE QUESTION:
//
//   empty/calls per site -- how often a push has anything to push to. Cheap to fix (a probe that
//   returns), but it is not where the arena goes.
//
//   dedup allocs and WASTED allocs. claim_match must have a stable MatchRecord copy before the
//   exchange that publishes it, so it allocates on the strength of a lookup that just missed;
//   another thread can claim the key in that window, and the arena is a bump pointer with no
//   per-object free, so a copy that loses is permanent.
//
// AND THAT IS NOT WHERE THE ARENA GOES EITHER -- this probe's own output refutes the sentence
// that used to stand here ("a fix for the 13.52% has to move the wasted count"). Measured,
// steps=4, threads {1,4}:
//
//   batched   8487 stable copies allocated, 3 lost the claim   (0.035%)
//   eager     4350 stable copies allocated, 0 lost the claim   (0%)
//
// Three lost copies cannot be 13.52% of anything. The gap is the ALLOCATION COUNT, 8487 against
// 4350, and essentially every one of those allocations WINS its claim -- they are distinct
// (state, match) records, not race debris.
//
// So batched stores about twice as many forwarded match records as eager while producing
// IDENTICAL output. The mechanism is visible in the same table: batched makes 7200 forwarding-
// site calls against eager's 3066, because by the time a match is pushed more of the parent's
// children exist to push it to.
//
// AN OVERLAPPING PUSH/PULL WINDOW IS NOT THE EXPLANATION. Testing it needs a per-node insertion
// position: the match list is prepend-only, so a node's position in the chain IS its insertion
// order, a reader holding a head covers exactly the positions at or below that head's, and a
// child that publishes the position it pulled up to lets a later push skip any record already
// covered. That was built end to end -- position derived from the predecessor the winning CAS
// observed, a per-child watermark published after each pull, and the position carried through
// both discovery sites and both pull re-propagations. Measured on the same configuration:
//
//   batched   8487 -> 8455 stable copies (0.4%), with 0 to 1 copies losing a claim throughout
//
// A partition can only remove records that two mechanisms both cover, and the claim-loss rate
// says there are none: every allocation wins, so each is the unique cover for a distinct
// (state, match) pair. The extra records are coverage batched genuinely performs and eager does
// not, and the arena gap is not recoverable by scheduling push against pull.
//
// This measures; it does not change behaviour. Removing or reordering anything is a separate step
// gated on these numbers, on cost_matrix over the same 17 cases, and on test_match_completeness
// holding at zero misses across the corpus x workers {1,2,4,8} x 3 reps.
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
    size_t d_calls, d_empty, f_calls, f_empty;
    size_t allocs, allocs_wasted;
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
            e.dedup_allocs(), e.dedup_allocs_wasted()};
}

double frac(size_t part, size_t whole) { return whole ? double(part) / double(whole) : 0.0; }

void report(const char* mode, const std::vector<Row>& rows) {
    std::printf("\n=== %s ===\n", mode);
    std::printf("%-20s %-4s %-21s %-21s %s\n",
                "workload", "thr", "discovery empty/calls", "forwarding empty/calls",
                "dedup wasted/allocs");
    size_t td = 0, tde = 0, tf = 0, tfe = 0, ta = 0, taw = 0;
    for (const Row& r : rows) {
        std::printf("%-20s %-4d %7zu/%-7zu %.3f  %7zu/%-7zu %.3f  %7zu/%-7zu %.3f\n",
                    r.name.c_str(), r.threads,
                    r.d_empty, r.d_calls, frac(r.d_empty, r.d_calls),
                    r.f_empty, r.f_calls, frac(r.f_empty, r.f_calls),
                    r.allocs_wasted, r.allocs, frac(r.allocs_wasted, r.allocs));
        td += r.d_calls; tde += r.d_empty;
        tf += r.f_calls; tfe += r.f_empty;
        ta += r.allocs;  taw += r.allocs_wasted;
    }
    std::printf("%-20s %-4s %7zu/%-7zu %.3f  %7zu/%-7zu %.3f  %7zu/%-7zu %.3f\n", "TOTAL", "",
                tde, td, frac(tde, td), tfe, tf, frac(tfe, tf), taw, ta, frac(taw, ta));
}

size_t total_allocs(const std::vector<Row>& rows, bool wasted) {
    size_t t = 0;
    for (const Row& r : rows) t += wasted ? r.allocs_wasted : r.allocs;
    return t;
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

    std::printf("push_match_to_children and claim_match, by call site (steps=%d)\n", steps);
    report("BATCHED (the default)", batched);
    report("EAGER (for contrast)", eager);

    const size_t ba = total_allocs(batched, false), bw = total_allocs(batched, true);
    const size_t ea = total_allocs(eager, false),   ew = total_allocs(eager, true);
    std::printf("\nStable MatchRecord copies allocated inside claim_match:\n");
    std::printf("  batched  %zu allocated, %zu lost the claim (%.1f%%)\n",
                ba, bw, 100.0 * frac(bw, ba));
    std::printf("  eager    %zu allocated, %zu lost the claim (%.1f%%)\n",
                ea, ew, 100.0 * frac(ew, ea));
    std::printf("A lost copy is permanent arena -- the allocator is a bump pointer with no\n"
                "per-object free. The difference between these two rows is what the batched\n"
                "default's extra 13.52%% is made of, and what a fix has to move.\n");
    return 0;
}
