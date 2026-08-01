// WHICH event identities do quotient and full capture disagree about?
//
// binary-growth Full/Automatic: quotient reports 6 identities, full capture 8. The mode
// decomposition already says the events all exist (raw 9 = 9) and endpoint keying agrees
// (Full 5 = 5), so the extra Automatic keys -- Step, ConsumedEdges, ProducedEdges -- under-split
// in the reconstruction. Counts cannot say which key collapses which pair; the SETS can.
//
// Prints |Q|, |F|, and the set differences. An identity in F but not Q is one full capture keeps
// apart and the reconstruction merges.
#include <cstdio>
#include <set>
#include <vector>
#include "hypergraph/hypergraph.hpp"
using namespace hgcommon;
#include "hypergraph/parallel_evolution.hpp"
using namespace hypergraph;

static void measure(hgcommon::EventSignatureKeys keys, std::set<uint64_t>& F,
                    std::set<uint64_t>& Q) {
    const RewriteRule rule = make_rule(0).lhs({0,1}).rhs({0,2}).rhs({1,2}).build();
    const std::vector<std::vector<VertexId>> init = {{0,1}};
    const int steps = 3;
    {   // full capture
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        hg.set_event_signature_keys(keys);
        ParallelEvolutionEngine e(&hg, 1);
        e.set_explore_from_canonical_states_only(false);
        e.add_rule(rule); auto i = init; e.evolve(i, steps);
        for (uint32_t k = 0; k < hg.num_raw_events(); ++k) {
            const Event& ev = hg.get_event(k);
            if (ev.id == INVALID_ID || !ev.is_canonical()) continue;
            F.insert(ev.signature);
        }
    }
    {   // quotient + reconstruction
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        hg.set_event_signature_keys(keys);
        ParallelEvolutionEngine e(&hg, 1);
        e.set_explore_from_canonical_states_only(true);
        e.add_rule(rule); auto i = init; e.evolve(i, steps);
        hg.for_each_reconstructed_event_signature([&](uint64_t s) { Q.insert(s); });
        if (keys & (EventKey_ConsumedEdges | EventKey_ProducedEdges))
            std::printf("   [rank xlat] fired=%zu fell_through=%zu\n",
                        hg.rank_translation_fired(), hg.rank_translation_fell_through());
    }

}

int main() {
    using namespace hgcommon;
    // Isolate each key component. Both paths honour set_event_signature_keys, so subsetting the
    // keys says WHICH component makes the signature sets disjoint -- without touching the engine.
    struct Row { const char* name; EventSignatureKeys keys; };
    const Row rows[] = {
        {"Full (endpoints only)",   EVENT_SIG_FULL},
        {"Full + Step",             (EventSignatureKeys)(EVENT_SIG_FULL | EventKey_Step)},
        {"Automatic minus Step",    (EventSignatureKeys)(EVENT_SIG_AUTOMATIC & ~EventKey_Step)},
        {"Automatic (all)",         EVENT_SIG_AUTOMATIC},
    };
    std::printf("%-24s | %4s %4s %6s %8s %6s\n", "key set", "|F|", "|Q|", "both", "F-only", "Q-only");
    for (const Row& r : rows) {
        std::set<uint64_t> F, Q;
        measure(r.keys, F, Q);
        size_t both = 0;
        for (uint64_t v : F) if (Q.count(v)) ++both;
        std::printf("%-24s | %4zu %4zu %6zu %8zu %6zu\n",
                    r.name, F.size(), Q.size(), both, F.size() - both, Q.size() - both);
    }
    std::printf("\n# The first row where 'both' collapses to 0 names the component that diverges.\n");
    return 0;
}
