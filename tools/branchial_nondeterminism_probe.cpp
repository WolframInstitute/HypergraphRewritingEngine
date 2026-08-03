// branchial_nondeterminism_probe.cpp — localise the determinism-gate failure (#65).
//
// CausalDeterminism.NonQuotientFullyDeterministic fails ~5% on WPP with a differing canonical
// state fingerprint. That fingerprint is the sorted multiset of canonical state hashes, so two
// very different faults produce the identical symptom and they live in different subsystems:
//
//   the state SET differs        -- exploration raced: a state was created in one run and not
//                                   the other, or a duplicate escaped dedup. Fault is in
//                                   matching / rewriting / the dedup map.
//   the same states, different
//   canonical HASHES             -- canonicalization raced: the same edge set hashed two ways.
//                                   Fault is in the canonical hash path, and the state set is
//                                   fine.
//
// The count alone separates them, so this reports the distinct (count, fingerprint) pairs and
// then, when the sets differ, the symmetric difference of the hash multisets -- which states
// appeared or vanished. Knowing WHICH state is missing points at a mechanism far faster than
// knowing how often.
//
// Run: /tmp/probe [runs] [threads] [steps]
//
// STATUS: this does NOT yet reproduce. WPP is clean over 1200 runs and mixed1 over 360, at
// threads {1,2,8} and both seeds, while the gate binary fails about 1 invocation in 30. So the
// trigger is not either workload in isolation, and the remaining difference is the process
// context: the gate runs WPP, mixed1 and mixed2 in sequence in one process, each leaving behind
// worker threads, thread-local scratch and a used allocator. Recorded here so the next attempt
// starts from what has been excluded rather than repeating it.

#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace hypergraph;

namespace {

struct Outcome {
    size_t num_states;
    size_t num_events;
    size_t num_causal;
    size_t num_branchial;
    std::vector<uint64_t> state_hashes;      // sorted
    std::vector<uint64_t> branchial_sigs;    // sorted, iso-invariant

    bool operator<(const Outcome& o) const {
        if (num_states != o.num_states) return num_states < o.num_states;
        if (num_events != o.num_events) return num_events < o.num_events;
        if (num_causal != o.num_causal) return num_causal < o.num_causal;
        if (num_branchial != o.num_branchial) return num_branchial < o.num_branchial;
        if (state_hashes != o.state_hashes) return state_hashes < o.state_hashes;
        return branchial_sigs < o.branchial_sigs;
    }
};

Outcome run(int threads, int steps, uint64_t seed) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&g, threads);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(false);
    e.set_random_seed(seed);
    // mixed1: three rules over a single initial edge. This is the workload that actually
    // failed, and only its BRANCHIAL fingerprint did -- branchial edges pair sibling events
    // from one input state, so several rules firing on the same edge is exactly the shape that
    // stresses the sibling rendezvous. The first probe here aimed at WPP and found nothing in
    // 1200 runs, because WPP has one rule and never produces that contention.
    e.add_rule(make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build());
    e.add_rule(make_rule(1).lhs({0,1}).rhs({1,0}).build());
    e.add_rule(make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build());
    e.evolve(std::vector<std::vector<VertexId>>{{0,1}}, steps);

    Outcome o;
    for (uint32_t s = 0; s < g.num_states(); ++s) {
        if (g.get_state(s).id == INVALID_ID) continue;
        o.state_hashes.push_back(g.get_or_compute_canonical_hash(s));
    }
    std::sort(o.state_hashes.begin(), o.state_hashes.end());
    o.num_states    = o.state_hashes.size();

    // Branchial by iso-invariant event signature, matching the gate. Raw event ids are
    // allocation order and are allowed to differ; the PAIRS they represent are not.
    auto canon = [&](StateId s) -> uint64_t {
        return s == INVALID_ID ? 0 : g.get_or_compute_canonical_hash(s);
    };
    auto esig = [&](EventId ev) -> uint64_t {
        const Event& x = g.get_event(ev);
        uint64_t h = 1469598103934665603ULL;
        for (uint64_t v : {canon(x.input_state), canon(x.output_state),
                           uint64_t(x.rule_index)}) { h ^= v; h *= 1099511628211ULL; }
        return h;
    };
    for (const auto& b : g.causal_graph().get_branchial_edges()) {
        uint64_t a = esig(b.event1), d = esig(b.event2);
        if (a > d) std::swap(a, d);
        uint64_t h = 0; h ^= a; h *= 1099511628211ULL; h ^= d; h *= 1099511628211ULL;
        o.branchial_sigs.push_back(h);
    }
    std::sort(o.branchial_sigs.begin(), o.branchial_sigs.end());
    o.num_events    = g.num_events();
    o.num_causal    = g.causal_graph().num_causal_edges();
    o.num_branchial = g.causal_graph().num_branchial_edges();
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int runs    = (argc > 1) ? std::atoi(argv[1]) : 200;
    const int threads = (argc > 2) ? std::atoi(argv[2]) : 8;
    const int steps   = (argc > 3) ? std::atoi(argv[3]) : 6;

    std::printf("mixed1: runs=%d threads=%d steps=%d\n", runs, threads, steps);

    // Mirror the gate's spread exactly: it pools seeds x reps x THREAD COUNTS into one set,
    // so a result that is stable at 8 threads and different at 1 is precisely the failure. A
    // probe that fixes the thread count cannot see it -- which is how the first pass here came
    // back clean over 200 runs.
    std::map<Outcome, int> seen;
    // Which (threads, seed) produced each outcome. A divergence confined to one thread count
    // is a different bug from one that appears at every count, so record the provenance rather
    // than only the tally.
    std::map<Outcome, std::set<std::string>> origin;
    (void)threads;
    for (int i = 0; i < runs; ++i)
        for (uint64_t seed : {uint64_t(0xABCDEF), uint64_t(0)})
            for (int th : {1, 2, 8}) {
                Outcome o = run(th, steps, seed);
                ++seen[o];
                origin[o].insert("t" + std::to_string(th) +
                                 "/seed" + (seed ? "fixed" : "random"));
            }

    std::printf("%zu distinct outcome(s)\n", seen.size());
    for (const auto& [o, n] : seen) {
        std::printf("  x%-4d states=%zu events=%zu causal=%zu branchial=%zu   from:",
                    n, o.num_states, o.num_events, o.num_causal, o.num_branchial);
        for (const auto& s : origin[o]) std::printf(" %s", s.c_str());
        std::printf("\n");
    }
    if (seen.size() < 2) { std::printf("deterministic over these runs\n"); return 0; }

    // Two outcomes: is it the SET or the HASHES? If every outcome has the same count, the sets
    // are the same size and something hashed differently -- a canonicalization fault. If the
    // counts differ, exploration raced.
    bool counts_agree = true;
    const size_t first = seen.begin()->first.num_states;
    for (const auto& [o, n] : seen) if (o.num_states != first) counts_agree = false;
    std::printf("\nstate COUNT %s across outcomes -> fault is in %s\n",
                counts_agree ? "AGREES" : "DIFFERS",
                counts_agree ? "CANONICALIZATION (same states, different hashes)"
                             : "EXPLORATION (a state was created in one run and not another)");

    // The symmetric difference names the state that came or went.
    const auto& a = seen.begin()->first.state_hashes;
    const auto& b = std::next(seen.begin())->first.state_hashes;
    std::vector<uint64_t> only_a, only_b;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(only_a));
    std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(only_b));
    std::printf("only in outcome 1: %zu hash(es)", only_a.size());
    for (uint64_t h : only_a) std::printf(" %llu", (unsigned long long)h);
    std::printf("\nonly in outcome 2: %zu hash(es)", only_b.size());
    for (uint64_t h : only_b) std::printf(" %llu", (unsigned long long)h);
    std::printf("\n");
    return 1;
}
