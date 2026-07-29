// The identity matrix gate: every corpus workload under every state x event identity mode,
// plus quotient exploration, compared against cached expectations in milliseconds.
//
// What this adds over the existing oracle gate, which runs Full mode only and compares counts:
//
//   ALL NINE CELLS. The two identity axes are what the engine is FOR -- they decide which
//   states are the same state and which events are the same event -- and only one of the nine
//   combinations was covered. A change to canonicalization could alter the other eight without
//   a single test noticing.
//
//   FINGERPRINTS, NOT COUNTS. A count cannot distinguish two different state sets of the same
//   size, and these modes differ precisely in which states they identify. The fingerprint folds
//   the multiset of canonical state hashes commutatively, so it is invariant under creation
//   order (which the scheduler decides) and sensitive to WHICH states and to their
//   multiplicity.
//
//   SPEED. The independent checks are O(V!) brute force and wolframscript. Neither can run per
//   build, so they run when the golden is regenerated and this gate compares the result.
//
// The gate also reports the PROVENANCE mix, because a cached expectation produced by the engine
// and compared against the engine proves stability rather than correctness. A row is only
// called checked if something that shares no code with the engine's canonicalization agreed
// with it. The rest are regression tripwires and are counted separately so that the coverage
// stays visible instead of being implied by the tests passing.

#include <gtest/gtest.h>

#include "reference/golden_matrix.hpp"
#include "reference/oracle_corpus.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace hypergraph;

namespace {

std::vector<golden::Row> load_rows(bool* found) {
    std::vector<golden::Row> rows;
    *found = false;
    // Run from the build directory or the source root; try both rather than depend on cwd.
    for (const char* path : {"reference/golden_matrix.txt", "../reference/golden_matrix.txt"}) {
        std::ifstream in(path);
        if (!in) continue;
        *found = true;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            golden::Row r;
            std::string sm, ek, prov;
            int quotient = 0;
            ls >> r.case_name >> sm >> ek >> quotient >> r.steps
               >> r.states >> r.events >> r.causal_edges >> r.causal_event_pairs
               >> r.branchial_edges >> r.state_fingerprint >> prov;
            if (!ls && !ls.eof()) continue;
            if (!golden::state_mode_from_name(sm, r.state_mode)) continue;
            if (!golden::event_keys_from_name(ek, r.event_keys)) continue;
            r.quotient = (quotient != 0);
            r.provenance = (prov == "oracle")    ? golden::Provenance::Oracle
                         : (prov == "reference") ? golden::Provenance::Reference
                                                 : golden::Provenance::Pin;
            rows.push_back(std::move(r));
        }
        break;
    }
    return rows;
}

const oracle::Case* find_case(const std::vector<oracle::Case>& corpus, const std::string& name) {
    for (const auto& c : corpus) if (name == c.name) return &c;
    return nullptr;
}

}  // namespace

TEST(GoldenMatrix, EveryIdentityCellMatchesItsCachedExpectation) {
    bool found = false;
    const std::vector<golden::Row> rows = load_rows(&found);
    ASSERT_TRUE(found) << "reference/golden_matrix.txt is missing; regenerate it with "
                          "tools/generate_golden_matrix.cpp";
    ASSERT_FALSE(rows.empty()) << "the golden matrix parsed to no rows";

    const std::vector<oracle::Case> corpus = oracle::corpus();
    size_t checked = 0, pinned = 0;

    for (const auto& r : rows) {
        const oracle::Case* c = find_case(corpus, r.case_name);
        ASSERT_NE(c, nullptr) << "golden names a case the corpus no longer has: " << r.case_name;

        Hypergraph hg;
        hg.set_state_canonicalization_mode(r.state_mode);
        hg.set_event_signature_keys(r.event_keys);
        ParallelEvolutionEngine e(&hg, 4);
        e.set_explore_from_canonical_states_only(r.quotient);
        for (const auto& rule : c->rules) e.add_rule(rule);
        e.evolve(c->init, r.steps);

        uint64_t fingerprint = 0;
        for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
            if (hg.get_state(sid).id == INVALID_ID) continue;
            fingerprint = golden::fold_fingerprint(fingerprint,
                                                   hg.get_or_compute_canonical_hash(sid));
        }

        const std::string where = r.case_name + std::string(" state=") +
            golden::state_mode_name(r.state_mode) + " event=" +
            golden::event_keys_name(r.event_keys) +
            (r.quotient ? " quotient" : "") + " [" +
            golden::provenance_name(r.provenance) + "]";

        EXPECT_EQ(hg.num_canonical_states(), r.states) << where << ": states";
        EXPECT_EQ(hg.num_events(), r.events) << where << ": events";
        EXPECT_EQ(hg.causal_graph().num_causal_edges(), r.causal_edges) << where << ": causal";
        EXPECT_EQ(hg.causal_graph().num_causal_event_pairs(), r.causal_event_pairs)
            << where << ": causal pairs";
        EXPECT_EQ(hg.causal_graph().num_branchial_edges(), r.branchial_edges)
            << where << ": branchial";
        EXPECT_EQ(fingerprint, r.state_fingerprint)
            << where << ": the state SET differs while its size may not -- this is the check a "
            << "count cannot make";

        if (r.provenance == golden::Provenance::Pin) ++pinned; else ++checked;
    }

    // Not an assertion: a statement of how much of the matrix has an independent check behind
    // it. Left visible so the number has to be argued down rather than quietly assumed.
    std::printf("[ golden ] %zu rows independently checked, %zu pinned to engine output "
                "(regression only)\n", checked, pinned);
}

// A pin is only a tripwire if the thing it pins is stable. Every cell must also be independent
// of the worker count -- the identity modes are defined on the graph, not on the schedule.
TEST(GoldenMatrix, EveryIdentityCellIsIndependentOfWorkerCount) {
    bool found = false;
    const std::vector<golden::Row> rows = load_rows(&found);
    ASSERT_TRUE(found);
    const std::vector<oracle::Case> corpus = oracle::corpus();

    for (const auto& r : rows) {
        const oracle::Case* c = find_case(corpus, r.case_name);
        ASSERT_NE(c, nullptr);

        auto run = [&](unsigned threads) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(r.state_mode);
            hg.set_event_signature_keys(r.event_keys);
            ParallelEvolutionEngine e(&hg, threads);
            e.set_explore_from_canonical_states_only(r.quotient);
            for (const auto& rule : c->rules) e.add_rule(rule);
            e.evolve(c->init, r.steps);
            uint64_t fp = 0;
            for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
                if (hg.get_state(sid).id == INVALID_ID) continue;
                fp = golden::fold_fingerprint(fp, hg.get_or_compute_canonical_hash(sid));
            }
            return fp;
        };

        const std::string where = r.case_name + std::string(" state=") +
            golden::state_mode_name(r.state_mode) + " event=" +
            golden::event_keys_name(r.event_keys) + (r.quotient ? " quotient" : "");
        const uint64_t one = run(1);
        EXPECT_EQ(run(8), one) << where << ": the state set depends on the worker count";
    }
}
