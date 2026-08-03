#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "wxf.hpp"
#include "paclet_source/hg_core.hpp"

// Pin test for the FFI WXF serialization (run_rewriting_core), the LibraryLink /
// standalone-binary output contract. This path has no wolframscript-free coverage
// otherwise, so these tests exercise it end to end: craft a WXF input, run a small
// multiway evolution through run_rewriting_core, and assert the output structure by
// parsing it back with wxf::Parser.
//
// The evolution is multi-threaded, so run-local state/event IDs (and hence the exact
// byte order of the States/Events associations) vary between runs; the assertions
// below are invariant to that ordering (element counts, key presence, per-entry
// fields). The byte-level equivalence of the streaming serializer to the prior
// value-tree serializer was verified out of band by diffing hg_evolve output against
// the pre-change build under a single worker (deterministic IDs): byte-identical.
namespace {

using Edge = std::vector<int64_t>;
using EdgeList = std::vector<Edge>;      // one hypergraph state / rule side
using StateList = std::vector<EdgeList>; // list of states

// Serialize an input association identical to the LibraryLink performRewriting
// contract: InitialStates, Rules[Rule[lhs, rhs]], Steps, Options.
std::vector<uint8_t> build_input(const StateList& initial_states,
                                 const EdgeList& rule_lhs,
                                 const EdgeList& rule_rhs,
                                 int64_t steps,
                                 const std::function<void(wxf::Writer&)>& write_options,
                                 std::size_t option_count) {
    wxf::Writer w;
    w.write_header();

    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(4);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("InitialStates"));
    w.write(initial_states);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Rules"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(1);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("r0"));
    w.write_function("Rule", 2);
    w.write(rule_lhs);
    w.write(rule_rhs);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Steps"));
    w.write(steps);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Options"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(option_count);
    write_options(w);

    return w.release_data();
}

// The raw BYTES of one top-level key's value. Two runs are then compared payload for payload,
// which is what "the gating changed nothing" has to mean -- equal entry counts would also hold
// for two runs that returned different states.
std::vector<uint8_t> value_bytes(const std::vector<uint8_t>& out, const std::string& key) {
    std::vector<uint8_t> got;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        const size_t begin = vp.position();
        vp.skip_value();
        if (k == key) got.assign(out.begin() + static_cast<long>(begin),
                                 out.begin() + static_cast<long>(vp.position()));
    });
    return got;
}

void put_str_list_option(wxf::Writer& w, const char* key,
                         const std::vector<std::string>& values) {
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string(key));
    w.write(values);
}

void put_str_option(wxf::Writer& w, const char* key, const char* value) {
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string(key));
    w.write(std::string(value));
}

// Count the entries of the association stored under top-level `key` (e.g. "States").
// Returns -1 if the key is absent from the output association.
int64_t count_assoc_entries(const std::vector<uint8_t>& out, const std::string& key) {
    int64_t result = -1;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k == key) {
            int64_t n = 0;
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
                kp.skip_value();
                valp.skip_value();
                ++n;
            });
            result = n;
        } else {
            vp.skip_value();
        }
    });
    return result;
}

int64_t read_int_key(const std::vector<uint8_t>& out, const std::string& key) {
    int64_t result = -1;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k == key) {
            result = vp.read<int64_t>();
        } else {
            vp.skip_value();
        }
    });
    return result;
}

// A -> B -> C chain rule: {{1,2}} -> {{1,2},{2,3}}, from a 2-edge seed. Three steps
// of full-multiway rewriting; the reached state/event set is deterministic.
const StateList kSeed = {{{1, 2}, {2, 3}}};
const EdgeList kLhs = {{1, 2}};
const EdgeList kRhs = {{1, 2}, {2, 3}};

}  // namespace

TEST(WxfSerializationPin, DefaultStatesAndEvents) {
    auto input = build_input(kSeed, kLhs, kRhs, 3, [](wxf::Writer&) {}, 0);
    HostBridge host;
    auto out = run_rewriting_core(input, host);
    ASSERT_FALSE(out.empty());

    int64_t states_entries = count_assoc_entries(out, "States");
    int64_t events_entries = count_assoc_entries(out, "Events");
    int64_t num_states = read_int_key(out, "NumStates");
    int64_t num_events = read_int_key(out, "NumEvents");

    // Default mode is CanonicalizeStates -> None: States carries every raw state and NumStates is
    // the canonical count, which in None equals the raw count (every provenance is its own state).
    EXPECT_EQ(states_entries, 33);
    EXPECT_EQ(num_states, 33);
    EXPECT_GT(events_entries, 0);
    EXPECT_GE(num_events, 0);

    // Each state entry is an association carrying the fixed field set; verify an
    // initial state (Step == 0, IsInitial) is present and every state carries Edges.
    int64_t seen = 0, initial_states = 0, with_edges = 0;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "States") { vp.skip_value(); return; }
        vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
            kp.skip_value();
            ++seen;
            int64_t step = -1;
            bool has_edges = false;
            valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                if (fk == "Step") { step = fvp.read<int64_t>(); }
                else if (fk == "Edges") { has_edges = true; fvp.skip_value(); }
                else { fvp.skip_value(); }
            });
            if (step == 0) ++initial_states;
            if (has_edges) ++with_edges;
        });
    });
    EXPECT_EQ(seen, states_entries);
    EXPECT_EQ(with_edges, states_entries);
    EXPECT_GE(initial_states, 1);
}

TEST(WxfSerializationPin, FullCanonicalizationWithHashes) {
    auto input = build_input(kSeed, kLhs, kRhs, 3,
                             [](wxf::Writer& w) {
                                 put_str_option(w, "CanonicalizeStates", "Full");
                                 put_str_option(w, "IncludeCanonicalHashes", "True");
                             },
                             2);
    HostBridge host;
    auto out = run_rewriting_core(input, host);
    ASSERT_FALSE(out.empty());

    EXPECT_GT(count_assoc_entries(out, "States"), 0);
    EXPECT_GT(count_assoc_entries(out, "Events"), 0);

    // Under IncludeCanonicalHashes -> True every state carries a CanonicalHash field.
    int64_t seen_states = 0, with_hash = 0;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "States") { vp.skip_value(); return; }
        vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
            kp.skip_value();
            ++seen_states;
            bool has_hash = false;
            valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                if (fk == "CanonicalHash") has_hash = true;
                fvp.skip_value();
            });
            if (has_hash) ++with_hash;
        });
    });
    EXPECT_GT(seen_states, 0);
    EXPECT_EQ(with_hash, seen_states);
}

TEST(WxfSerializationPin, MinimalEvents) {
    // RequestedData -> {"EventsMinimal"}: only the minimal Events association is emitted.
    auto input = build_input(kSeed, kLhs, kRhs, 2,
                             [](wxf::Writer& w) {
                                 w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                                 w.write(std::string("RequestedData"));
                                 w.write_function("List", 1);
                                 w.write(std::string("EventsMinimal"));
                             },
                             1);
    HostBridge host;
    auto out = run_rewriting_core(input, host);
    ASSERT_FALSE(out.empty());

    EXPECT_GT(count_assoc_entries(out, "Events"), 0);
    EXPECT_EQ(count_assoc_entries(out, "States"), -1);  // States not requested

    // Minimal event entries omit ConsumedEdges / ProducedEdges (7 fields, not 9).
    bool any_event = false, all_minimal = true;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "Events") { vp.skip_value(); return; }
        vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
            kp.skip_value();
            any_event = true;
            bool has_consumed = false;
            valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                if (fk == "ConsumedEdges" || fk == "ProducedEdges") has_consumed = true;
                fvp.skip_value();
            });
            if (has_consumed) all_minimal = false;
        });
    });
    EXPECT_TRUE(any_event);
    EXPECT_TRUE(all_minimal);
}

// Asking for less returns the same thing, and does not build what nobody asked for.
//
// RequestedData used to gate SERIALIZATION only: a caller asking for States alone still paid
// for the causal and branchial relations in full. Now the request decides what is RECORDED, so
// this pins the other half of that change.
//
// Compared by COUNT, not by payload bytes: the state and event ids in the output are raw
// engine ids, handed out in arrival order by whichever worker got there first, so two runs of
// the SAME request already disagree on them. Only CanonicalHash is stable across runs. What
// the contents are is gated a layer down, where the recording happens, by
// OracleCorpus.RecordSetSkipsOnlyWhatItWasNotAskedFor comparing canonical-hash multisets.
TEST(WxfSerializationPin, RequestedDataChangesNothingItReturns) {
    HostBridge host;

    auto full_in = build_input(kSeed, kLhs, kRhs, 3, [](wxf::Writer&) {}, 0);
    auto full = run_rewriting_core(full_in, host);
    ASSERT_FALSE(full.empty());

    auto lean_in = build_input(kSeed, kLhs, kRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData", {"States", "NumStates", "Events", "NumEvents"});
    }, 1);
    auto lean = run_rewriting_core(lean_in, host);
    ASSERT_FALSE(lean.empty());

    ASSERT_GT(count_assoc_entries(full, "States"), 0)
        << "the all-on run returned no States, so every equality below is vacuous";
    EXPECT_EQ(count_assoc_entries(lean, "States"), count_assoc_entries(full, "States"))
        << "asking for States alone changed how many states came back";
    EXPECT_EQ(count_assoc_entries(lean, "Events"), count_assoc_entries(full, "Events"))
        << "asking for a subset changed how many events came back";
    EXPECT_EQ(read_int_key(lean, "NumStates"), read_int_key(full, "NumStates"));
    EXPECT_EQ(read_int_key(lean, "NumEvents"), read_int_key(full, "NumEvents"));

    // The relations nobody asked for are absent from the output, not merely empty -- and the
    // run did not build them either, which is what the record set changed.
    EXPECT_EQ(count_assoc_entries(lean, "CausalEdges"), -1)
        << "CausalEdges came back from a request that did not ask for it";
    EXPECT_EQ(read_int_key(lean, "NumCausalEdges"), -1)
        << "NumCausalEdges came back from a request that did not ask for it";
    EXPECT_EQ(read_int_key(lean, "NumBranchialEdges"), -1)
        << "NumBranchialEdges came back from a request that did not ask for it";
}

// A content-bearing graph property under Full canonicalization: the path where every event
// carries its two endpoint states' edge lists, so the same state's canonical form is asked for
// once as a state and again by every event incident to it.
TEST(WxfSerializationPin, StatesGraphUnderFullCanonicalization) {
    HostBridge host;
    auto in = build_input(kSeed, kLhs, kRhs, 4, [](wxf::Writer& w) {
        put_str_list_option(w, "GraphProperties", {"StatesGraph"});
        put_str_option(w, "CanonicalizeStates", "Full");
    }, 2);
    auto out = run_rewriting_core(in, host);
    ASSERT_FALSE(out.empty());

    // GraphData carries one entry per requested property.
    EXPECT_EQ(count_assoc_entries(out, "GraphData"), 1)
        << "StatesGraph was requested and no GraphData came back";
    EXPECT_GT(read_int_key(out, "NumStates"), 0);
}

// PINNED REPRODUCER for P4.6: the causal GRAPH and the event COUNT beside it describe
// different event sets. It passes exactly while the gap is present, so the suite notices if it
// moves or closes -- the same contract ReconstructionGapIsStillOpen carried until P2 closed it.
//
// WHAT IS BROKEN. NumEvents routes through observable_num_events, which under an identity mode
// is the RECONSTRUCTION's count: distinct identities over the class frame. The graph is built by
// scanning MATERIALISED raw events and mapping each through its canonical_event_id, which full
// capture computed from each raw state's own labelling -- the per-state convention the
// reconstruction exists to replace. Measured on this workload under Automatic:
//
//     NumEvents (reconstruction, class frame)     24
//     causal graph vertices (materialised)        25
//
// TO CLOSE: serve the graph's event vertices from the reconstruction when it ran. That needs a
// vertex payload for an identity with no Event record -- the reconstruction deliberately
// materialises none -- built from the class hashes and rule it does hold, or the lean payload
// the Structure variants already use.
TEST(WxfSerializationPin, CausalGraphAndNumEventsStillDescribeDifferentSets) {
    HostBridge host;
    auto in = build_input(kSeed, kLhs, kRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "GraphProperties", {"CausalGraph"});
        put_str_option(w, "CanonicalizeStates", "Full");
        put_str_option(w, "CanonicalizeEvents", "Automatic");
    }, 3);
    auto out = run_rewriting_core(in, host);
    ASSERT_FALSE(out.empty());

    const int64_t num_events = read_int_key(out, "NumEvents");

    int64_t vertex_count = -1;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "GraphData") { vp.skip_value(); return; }
        vp.read_association([&](const std::string&, wxf::Parser& gp) {
            gp.read_association([&](const std::string& field, wxf::Parser& fp) {
                if (field != "Vertices") { fp.skip_value(); return; }
                // A WXF list is a Function with head List; its arg count is the length.
                fp.read_function([&](const std::string&, size_t n, wxf::Parser& ep) {
                    for (size_t i = 0; i < n; ++i) ep.skip_value();
                    vertex_count = static_cast<int64_t>(n);
                });
            });
        });
    });
    ASSERT_GE(vertex_count, 0) << "no Vertices came back for the requested CausalGraph";

    EXPECT_EQ(num_events, 24)
        << "the reconstruction's count moved; re-derive the pinned numbers";
    EXPECT_EQ(vertex_count, 25)
        << "the graph's vertex count moved; if it now equals NumEvents the gap has closed -- "
           "replace this reproducer with the equality";
}

// RandomSeed reaches the sampler it is documented to control.
//
// ExplorationProbability is Monte-Carlo sampling of the multiway system, and the engine's
// contract (parallel_evolution.hpp) is that a NONZERO seed is what makes that sample
// reproducible. The option reached the initial-condition generators only, so a sampled
// evolution asked for with a fixed seed returned a different sample every run and nothing said
// so. Serial, because the contract is stated for a single thread.
TEST(WxfSerializationPin, RandomSeedMakesASampledEvolutionReproducible) {
    HostBridge host;
    auto sampled = [&](int64_t seed) {
        auto in = build_input(kSeed, kLhs, kRhs, 5, [&](wxf::Writer& w) {
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("ExplorationProbability"));
            w.write(0.5);
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("RandomSeed"));
            w.write(seed);
        }, 2);
        auto out = run_rewriting_core(in, host);
        return read_int_key(out, "NumStates");
    };

    // A fixed seed pins the sample.
    const int64_t a = sampled(12345), b = sampled(12345);
    ASSERT_GT(a, 0);
    EXPECT_EQ(a, b) << "the same RandomSeed gave " << a << " then " << b
                    << " states: the seed does not reach the sampling draws";

    // A different seed is allowed to differ; what must not happen is the seed being ignored,
    // which would make every seed give the same answer for the wrong reason. Sampling at 0.5
    // over 5 steps separates them on this workload.
    bool any_different = false;
    for (int64_t s : {7, 99, 4242, 31337}) if (sampled(s) != a) { any_different = true; break; }
    EXPECT_TRUE(any_different)
        << "every seed gave " << a << " states, so the draw is not seeded at all";
}
