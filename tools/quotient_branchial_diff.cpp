// WHICH branchial pair does the quotient reconstruction miss?
//
// WHY THIS EXISTS. #4: quotient's observable output must equal full capture's (SPEC 5.4) and does
// not. The branchial count disagrees -- 2 against 3 on wolfram-2to4 -- and until now only the
// COUNTS could be compared, because the reconstruction exposed num_reconstructed_branchial() and
// nothing to enumerate. A count says the two paths disagree and can never say about what.
//
// THE TWO IMPLEMENTATIONS SCOPE THE SAME OVERLAP RELATION DIFFERENTLY, which is the suspected
// cause and what this is built to test:
//
//   full capture     an inverted index keyed (input_state, consumed_edge). An event is paired
//                    with every other occupant of each bucket it lands in, deduped on (e1,e2).
//                    No boundary beyond the input state.
//   reconstruction   pairs are formed by scanning ONE INSTANCE's applied-match list, requiring
//                    overlapping consumed SLOTS. A pair whose two events land in DIFFERENT
//                    instances cannot form, however much they overlap.
//
// So the reconstruction can only ever be a subset, and the question is whether the missing pairs
// are cross-instance (the boundary is wrong) or intra-instance (the overlap test is wrong). The
// two have different fixes, and the diff below distinguishes them: a missing pair whose two
// events also appear in full capture's event set, but never together in one instance, is the
// first; anything else is the second.
//
// Both sides are reported as pairs of ISOMORPHISM-INVARIANT EVENT SIGNATURES, so the comparison
// does not depend on raw ids matching between two runs that explore different state sets.
//
// Usage: quotient_branchial_diff [steps]

#include "hypergraph/parallel_evolution.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace hypergraph;

namespace {

using Pair = std::pair<uint64_t, uint64_t>;

Pair ordered(uint64_t a, uint64_t b) { return a <= b ? Pair{a, b} : Pair{b, a}; }

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
    // Two rules whose applications can coincide up to canonical form. The authority's Automatic
    // key set carries no rule index (MultiwaySystem.m:365), so such applications MERGE there;
    // the single-rule workloads above cannot exercise that. Authority per step: 1, 5, 21.
    w.push_back({"two-rules-overlap",
        {make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
         make_rule(1).lhs({0,1}).rhs({1,2}).rhs({2,0}).build()},
        {{0,1}}});
    return w;
}

struct Side {
    std::set<Pair> branchial;
    std::set<uint64_t> events;
};

// Full capture: every raw state explored, branchial edges read off the materialised graph and
// re-keyed by event signature so they are comparable with the reconstruction's.
Side full_capture(const Workload& w, int steps, EventSignatureKeys ekeys) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    g.set_event_signature_keys(ekeys);
    ParallelEvolutionEngine e(&g, 1);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(false);
    for (const auto& r : w.rules) e.add_rule(r);
    e.evolve(w.init, steps);

    Side s;
    for (uint32_t i = 0; i < g.num_raw_events(); ++i) {
        const Event& ev = g.get_event(i);
        if (ev.id != INVALID_ID) s.events.insert(ev.signature);
    }
    for (const auto& b : g.causal_graph().get_branchial_edges()) {
        const Event& e1 = g.get_event(b.event1);
        const Event& e2 = g.get_event(b.event2);
        if (e1.id == INVALID_ID || e2.id == INVALID_ID) continue;
        s.branchial.insert(ordered(e1.signature, e2.signature));
    }
    return s;
}

// Quotient: each canonical state expanded once, observables reconstructed from the quotient.
Side quotient(const Workload& w, int steps, EventSignatureKeys ekeys) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    g.set_event_signature_keys(ekeys);
    ParallelEvolutionEngine e(&g, 1);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(true);
    for (const auto& r : w.rules) e.add_rule(r);
    e.evolve(w.init, steps);

    Side s;
    g.for_each_reconstructed_event_signature([&](uint64_t sig) { s.events.insert(sig); });
    g.for_each_reconstructed_branchial([&](uint64_t a, uint64_t b) {
        s.branchial.insert(ordered(a, b));
    });
    return s;
}

void diff_one(const Workload& w, int steps, const char* mode_name, EventSignatureKeys ekeys) {
    // An event identity mode must be SET on both sides. Under EVENT_SIG_NONE full capture leaves
    // Event::signature at zero, so every event collapses to one value and the comparison silently
    // compares nothing, while the reconstruction populates no signature set at all.
    const Side F = full_capture(w, steps, ekeys);
    const Side Q = quotient(w, steps, ekeys);

    std::vector<Pair> f_only, q_only, both;
    for (const Pair& p : F.branchial)
        (Q.branchial.count(p) ? both : f_only).push_back(p);
    for (const Pair& p : Q.branchial)
        if (!F.branchial.count(p)) q_only.push_back(p);

    std::printf("\n%s  [event identity: %s]  (steps=%d)\n", w.name, mode_name, steps);
    std::printf("  branchial   |F|=%zu  |Q|=%zu   both=%zu  F-only=%zu  Q-only=%zu\n",
                F.branchial.size(), Q.branchial.size(), both.size(),
                f_only.size(), q_only.size());
    // Event-set agreement first. If the two sides do not even agree on WHICH events exist, a
    // branchial diff is comparing pairs drawn from different universes and says nothing about
    // the pairing. This line is what makes the rest of the output interpretable.
    size_t ev_both = 0;
    for (uint64_t s : F.events) if (Q.events.count(s)) ++ev_both;
    std::printf("  events      |F|=%zu  |Q|=%zu  shared=%zu  F-only=%zu  Q-only=%zu\n",
                F.events.size(), Q.events.size(), ev_both,
                F.events.size() - ev_both, Q.events.size() - ev_both);

    // The discriminator. A pair full capture has and the reconstruction does not, whose BOTH
    // endpoints the reconstruction nonetheless knows as events, is a pair the reconstruction
    // could have formed and did not -- so the two events never met in one instance, and the
    // instance boundary is the cause. A pair with an endpoint the reconstruction never produced
    // is a different and worse problem: an event is missing, not just a pairing.
    size_t boundary = 0, missing_endpoint = 0;
    for (const Pair& p : f_only) {
        const bool have_a = Q.events.count(p.first) > 0;
        const bool have_b = Q.events.count(p.second) > 0;
        if (have_a && have_b) ++boundary; else ++missing_endpoint;
        std::printf("    F-only  (%016lx, %016lx)  Q-knows-endpoints: %s/%s\n",
                    p.first, p.second, have_a ? "yes" : "NO", have_b ? "yes" : "NO");
    }
    for (const Pair& p : q_only)
        std::printf("    Q-only  (%016lx, %016lx)\n", p.first, p.second);

    if (!f_only.empty()) {
        std::printf("  -> %zu missing pair(s) with BOTH endpoints known to Q: the two events never\n"
                    "     met in one instance, so the INSTANCE BOUNDARY is the cause.\n", boundary);
        if (missing_endpoint)
            std::printf("  -> %zu missing pair(s) have an endpoint Q never produced: an EVENT is\n"
                        "     missing, which is a different and larger defect than the pairing.\n",
                        missing_endpoint);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 4;
    std::printf("Quotient vs full-capture branchial edges, as event-signature pairs\n");
    for (const auto& w : workloads()) {
        diff_one(w, steps, "Full", EVENT_SIG_FULL);
        diff_one(w, steps, "Automatic", EVENT_SIG_AUTOMATIC);
    }
    return 0;
}
