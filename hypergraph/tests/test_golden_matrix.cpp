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
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace hypergraph;

namespace {

std::vector<golden::Row> load_rows(bool* found) {
    std::vector<golden::Row> rows;
    *found = false;
    // From the source tree CMake configured, not from the working directory. A prefix guessed
    // from the caller's cwd finds the file from some directories and not others, and "not found"
    // here means the gate abstains rather than fails.
    {
        std::ifstream in(std::string(HG_SOURCE_DIR) + "/reference/golden_matrix.txt");
        if (!in) return rows;
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

        // Fold each DISTINCT canonical hash once: per-raw folding weights each class by its
        // member count, which differs between exploration strategies while the state SET is
        // identical. Must mirror the producer's definition exactly.
        uint64_t fingerprint = 0;
        {
            std::set<uint64_t> canon;
            for (uint32_t sid = 0; sid < hg.num_published_states(); ++sid) {
                if (hg.get_state(sid).id == INVALID_ID) continue;
                canon.insert(hg.get_or_compute_canonical_hash(sid));
            }
            for (uint64_t h : canon) fingerprint = golden::fold_fingerprint(fingerprint, h);
        }

        const std::string where = r.case_name + std::string(" state=") +
            golden::state_mode_name(r.state_mode) + " event=" +
            golden::event_keys_name(r.event_keys) +
            (r.quotient ? " quotient" : "") + " [" +
            golden::provenance_name(r.provenance) + "]";

        EXPECT_EQ(hg.num_canonical_states(), r.states) << where << ": states";
        EXPECT_EQ(hg.observable_num_events(), r.events) << where << ": events";
        EXPECT_EQ(hg.observable_num_causal_edges(), r.causal_edges) << where << ": causal";
        EXPECT_EQ(hg.observable_num_causal_pairs(
                      hg.causal_graph().transitive_reduction_enabled()),
                  r.causal_event_pairs)
            << where << ": causal pairs";
        EXPECT_EQ(hg.observable_num_branchial(), r.branchial_edges)
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

        // Both axes of the cell, not just the state one. An event-identity mode can be
        // schedule-dependent while the state set is perfectly stable -- the two are separate
        // questions and only comparing both asks the second one.
        struct Shot {
            uint64_t state_fingerprint;
            uint64_t event_fingerprint;
            uint64_t events;
            uint64_t causal_edges;
            uint64_t branchial_edges;
        };
        auto run = [&](unsigned threads) {
            Hypergraph hg;
            hg.set_state_canonicalization_mode(r.state_mode);
            hg.set_event_signature_keys(r.event_keys);
            ParallelEvolutionEngine e(&hg, threads);
            e.set_explore_from_canonical_states_only(r.quotient);
            for (const auto& rule : c->rules) e.add_rule(rule);
            e.evolve(c->init, r.steps);
            Shot s{};
            {
                std::set<uint64_t> canon;
                for (uint32_t sid = 0; sid < hg.num_published_states(); ++sid) {
                    if (hg.get_state(sid).id == INVALID_ID) continue;
                    canon.insert(hg.get_or_compute_canonical_hash(sid));
                }
                for (uint64_t h : canon)
                    s.state_fingerprint = golden::fold_fingerprint(s.state_fingerprint, h);
            }
            // Order-independent over the CANONICAL events' signature VALUES. A permutation of
            // signatures across events leaves every count intact, so the counts below cannot
            // ask whether the two runs agree on which event is which -- only this can.
            if (hg.quotient_reconstruction()) {
                // Raw content triples, not identity signatures: identity values embed
                // frame-relative slots and legitimately vary across schedules on symmetric
                // classes, while the triples are a function of the multiway structure alone.
                hg.for_each_reconstructed_raw_triple([&](uint64_t sig) {
                    s.event_fingerprint =
                        golden::fold_fingerprint(s.event_fingerprint, sig);
                });
            } else {
                for (uint32_t eid = 0; eid < hg.num_published_events(); ++eid) {
                    const Event& ev = hg.get_event(eid);
                    if (ev.id == INVALID_ID || !ev.is_canonical()) continue;
                    s.event_fingerprint =
                        golden::fold_fingerprint(s.event_fingerprint, ev.signature);
                }
            }
            s.events          = hg.observable_num_events();
            s.causal_edges    = hg.observable_num_causal_edges();
            s.branchial_edges = hg.observable_num_branchial();
            return s;
        };

        const std::string where = r.case_name + std::string(" state=") +
            golden::state_mode_name(r.state_mode) + " event=" +
            golden::event_keys_name(r.event_keys) + (r.quotient ? " quotient" : "");
        const Shot one = run(1);
        const Shot eight = run(8);
        EXPECT_EQ(eight.state_fingerprint, one.state_fingerprint)
            << where << ": the state set depends on the worker count";
        EXPECT_EQ(eight.events, one.events)
            << where << ": the event count depends on the worker count (" << one.events
            << " at 1 worker, " << eight.events << " at 8)";
        EXPECT_EQ(eight.event_fingerprint, one.event_fingerprint)
            << where << ": the event count matches but the SIGNATURES differ, so the two runs "
            << "disagree about which event is which";
        EXPECT_EQ(eight.causal_edges, one.causal_edges)
            << where << ": the causal edge count depends on the worker count";
        EXPECT_EQ(eight.branchial_edges, one.branchial_edges)
            << where << ": the branchial edge count depends on the worker count";
    }
}
