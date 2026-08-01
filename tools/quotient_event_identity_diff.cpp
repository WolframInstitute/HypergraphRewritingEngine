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
#include "hypergraph/parallel_evolution.hpp"
using namespace hypergraph;

int main() {
    const RewriteRule rule = make_rule(0).lhs({0,1}).rhs({0,2}).rhs({1,2}).build();
    const std::vector<std::vector<VertexId>> init = {{0,1}};
    const int steps = 3;

    std::set<uint64_t> F, Q;
    {   // full capture
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        hg.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
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
        hg.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
        ParallelEvolutionEngine e(&hg, 1);
        e.set_explore_from_canonical_states_only(true);
        e.add_rule(rule); auto i = init; e.evolve(i, steps);
        hg.for_each_reconstructed_event_signature([&](uint64_t s) { Q.insert(s); });
    }

    std::printf("full-capture identities |F| = %zu\n", F.size());
    std::printf("reconstructed identities |Q| = %zu\n\n", Q.size());
    size_t both = 0;
    for (uint64_t s : F) if (Q.count(s)) ++both;
    std::printf("in BOTH            : %zu\n", both);
    std::printf("in F only (merged) : %zu\n", F.size() - both);
    std::printf("in Q only          : %zu\n", Q.size() - both);
    std::printf("\n# 'in BOTH' near zero means the two paths compute DIFFERENT signature values\n"
                "# for the same events -- a keying mismatch. A large overlap with a few F-only\n"
                "# entries means the reconstruction genuinely merges identities full capture keeps.\n"
                "# These are different defects and the counts alone cannot tell them apart.\n");
    return 0;
}
