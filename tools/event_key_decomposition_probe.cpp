// Which COMPONENT of the Automatic event key makes the CPU and the GPU disagree?
//
// On the directed 4-cycle under full capture the CPU reports 15 canonical events and the GPU
// reports 19. The reference adjudicates 15 for Automatic-Positional, so the CPU is right. Under
// QUOTIENT the two agree at 15, and that split is the whole clue: a canonical class holds exactly
// one raw state under quotient and many under full capture.
//
// EventKey_Step is the component that behaves differently between those two regimes. The CPU
// signs with canonical_out_state.step -- the depth of the class REPRESENTATIVE, one value per
// class. The GPU signs with the raw child's own depth, and a class reached at several depths
// therefore gets several values, splitting events the CPU merges. 19 > 15 is the right direction.
//
// This drives the CPU alone across subsets of the key lattice. If dropping Step from the CPU's
// key set moves it 15 -> 19, then Step is what merges on the CPU, the GPU's raw-depth convention
// is what fails to merge, and the ranks are innocent. If it does not move, the hypothesis is
// wrong and the ranks are back in scope.
//
// The CPU is used on both sides deliberately: it isolates the COMPONENT without also varying the
// device, which is the confound that made the earlier rank hypothesis look plausible.

#include <cstdio>
#include <utility>
#include <vector>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

using namespace hypergraph;

namespace {

struct Counts { size_t canonical_events; size_t raw_events; };

Counts run(hgcommon::EventSignatureKeys keys, bool quotient,
           const std::vector<std::vector<VertexId>>* init_override = nullptr) {
    // The directed 4-cycle: the adjudicated case, and the one with a nontrivial automorphism
    // group, which is where a class is reached at several depths.
    // Exactly the corpus case cycle4-automorphic (reference/oracle_corpus.hpp), including its
    // oracle depth -- an invented workload does not reproduce the adjudicated 15 and its numbers
    // are not comparable to the GPU's 19.
    std::vector<std::vector<VertexId>> init = init_override ? *init_override
                                          : std::vector<std::vector<VertexId>>{{0,1},{1,2},{2,3},{3,0}};
    RewriteRule rule =
        make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,3}).rhs({3,2}).build();

    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    hg.set_event_signature_keys(keys);
    ParallelEvolutionEngine e(&hg, 1);
    e.set_explore_from_canonical_states_only(quotient);
    e.add_rule(rule);
    e.evolve(init, 3);   // the corpus oracle depth

    return Counts{hg.num_events(), hg.num_raw_events()};
}

struct Row { const char* name; hgcommon::EventSignatureKeys keys; };

}  // namespace

int main() {
    using namespace hgcommon;
    const EventSignatureKeys automatic = EVENT_SIG_AUTOMATIC;
    const EventSignatureKeys no_step   = automatic & ~EventKey_Step;
    const EventSignatureKeys no_ranks  = automatic & ~(EventKey_ConsumedEdges | EventKey_ProducedEdges);

    const std::vector<Row> rows = {
        {"Automatic (all)",          automatic},
        {"Automatic minus Step",     no_step},
        {"Automatic minus ranks",    no_ranks},
        {"endpoints only (Full)",    EVENT_SIG_FULL},
        {"endpoints + Step",         (EventSignatureKeys)(EVENT_SIG_FULL | EventKey_Step)},
    };

    std::printf("# CPU only, corpus case cycle4-automorphic (3 steps). GPU reports 19 for Automatic under full\n");
    std::printf("# capture; the reference adjudicates 15. Does dropping Step move the CPU to 19?\n\n");
    std::printf("%-24s | %-18s | %-18s\n", "key set", "full capture", "quotient");
    std::printf("%-24s | %8s %9s | %8s %9s\n", "", "canon", "raw", "canon", "raw");

    // IS THE RANK AN ISOMORPHISM INVARIANT, or does it depend on how the state was PRESENTED?
    //
    // A canonical edge rank is a position in a canonical LABELLING, and on a state with a
    // nontrivial automorphism group that labelling is a COSET -- interchangeable edges can take
    // each other's positions. If the engine's within-cell tie-break falls back on input order,
    // then presenting the SAME graph differently selects a different coset representative, the
    // ranks move, and an event identity keyed on them is not an invariant at all.
    //
    // Every presentation below is the same directed 4-cycle. An invariant identity must return
    // the same count for all of them. This is the check that says whether the CPU's agreement
    // with the reference is BY CONSTRUCTION or merely luck on one presentation -- and if it is
    // luck, #66 is a defect in both engines rather than a GPU defect.
    const std::vector<std::pair<const char*, std::vector<std::vector<VertexId>>>> presentations = {
        {"as written",        {{0,1},{1,2},{2,3},{3,0}}},
        {"edges rotated",     {{1,2},{2,3},{3,0},{0,1}}},
        {"edges reversed",    {{3,0},{2,3},{1,2},{0,1}}},
        {"vertices +10",      {{10,11},{11,12},{12,13},{13,10}}},
        {"vertices relabel",  {{7,3},{3,9},{9,5},{5,7}}},
    };
    std::printf("\n# same 4-cycle, presented differently. Automatic, full capture.\n");
    std::printf("# an isomorphism-invariant identity gives ONE number down this column.\n");
    std::printf("%-20s | %8s %9s\n", "presentation", "canon", "raw");
    for (const auto& pr : presentations) {
        const Counts c = run(automatic, /*quotient=*/false, &pr.second);
        std::printf("%-20s | %8zu %9zu\n", pr.first, c.canonical_events, c.raw_events);
    }

    // BINARY-GROWTH under FULL CAPTURE, presented five ways. #4's remaining gap is that the
    // reconstruction reports 6 Automatic identities where full capture reports 8. Before treating
    // 8 as the target, ask whether 8 is itself an invariant: a rank is a position in a canonical
    // LABELLING, and on an automorphic state that labelling is a coset. If full capture's count
    // moves with presentation then 8 is presentation-dependent and the reconstruction's 6 -- read
    // off the pinned class frame -- may be the invariant answer.
    {
        RewriteRule bg = make_rule(0).lhs({0,1}).rhs({0,2}).rhs({1,2}).build();
        const std::vector<std::pair<const char*, std::vector<std::vector<VertexId>>>> bgp = {
            {"as written",       {{0,1}}},
            {"vertices +10",     {{10,11}}},
            {"vertices swapped", {{1,0}}},
            {"vertices relabel", {{7,3}}},
        };
        std::printf("\n# binary-growth, FULL CAPTURE, Automatic, five presentations of one graph\n");
        std::printf("%-20s | %8s %9s\n", "presentation", "canon", "raw");
        for (const auto& pr : bgp) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
            hg.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
            ParallelEvolutionEngine e(&hg, 1);
            e.set_explore_from_canonical_states_only(false);
            e.add_rule(bg);
            auto init2 = pr.second;
            e.evolve(init2, 3);
            std::printf("%-20s | %8zu %9zu\n", pr.first, hg.num_events(), hg.num_raw_events());
        }
    }

    for (const Row& r : rows) {
        const Counts f = run(r.keys, /*quotient=*/false);
        const Counts q = run(r.keys, /*quotient=*/true);
        std::printf("%-24s | %8zu %9zu | %8zu %9zu\n",
                    r.name, f.canonical_events, f.raw_events, q.canonical_events, q.raw_events);
    }
    return 0;
}
