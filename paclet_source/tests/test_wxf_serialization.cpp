#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
// The process gate below drives a worker through a pair of FIFOs, which needs fork, mkfifo and
// waitpid. None of them exists on Windows, and the gate is compiled out there rather than
// emulated: what it checks -- that the SHIPPED binary serves the four verbs over the wire -- is
// a property of the binary, and the Windows leg has no such binary to point at.
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
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
        // vp is a sub-parser over the value, so its position() counts from the value's first
        // byte. The offset into `out` is where vp's view begins, which is what data() gives.
        const uint8_t* begin = vp.data();
        vp.skip_value();
        if (k == key) got.assign(begin, begin + vp.position());
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

// Vertices in the FIRST GraphData entry. The graph properties are requested one at a time
// here, so "first" is "the one asked for". Returns -1 if no Vertices field came back, which
// is distinct from a graph that legitimately has none.
int64_t graph_vertex_count(const std::vector<uint8_t>& out) {
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
    return vertex_count;
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

std::vector<int64_t> read_int_list_key(const std::vector<uint8_t>& out, const std::string& key) {
    std::vector<int64_t> result;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k == key) {
            result = vp.read<std::vector<int64_t>>();
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

// A rule whose LHS has TWO edges, so two distinct matches can share a consumed edge and
// can_branch is true. kLhs above is a single edge and is the provably branchial-free case.
// Both are needed: the engine takes a different path for each, and only one of them was
// covered when a regression made every no-property call return nothing.
const StateList kBranchSeed = {{{1, 2}, {1, 3}}};
const EdgeList kBranchLhs = {{1, 2}, {1, 3}};
const EdgeList kBranchRhs = {{1, 2}, {1, 3}, {2, 3}};

// The graph-shaped properties, and the identity modes each must work under.
const char* const kGraphProperties[] = {
    "StatesGraph", "CausalGraph", "BranchialGraph",
    "EvolutionGraph", "EvolutionCausalGraph", "EvolutionBranchialGraph",
    "EvolutionCausalBranchialGraph",
    "StatesGraphStructure", "EvolutionGraphStructure",
};
const char* const kIdentityModes[] = {"None", "Automatic", "Full"};

}  // namespace

// The same job with one extra top-level key, so the session envelope can be exercised without
// disturbing the builder every other pin test uses.
//
// `with_rules` is false for the verbs that address a HELD engine: a session's rule set was fixed
// when it opened, so `Step` and `Query` carry none and sending them is an error rather than a
// no-op. Keeping it a parameter is what lets that error be gated too.
// `from`, when non-empty, is the frontier subset a steered Step names. Sent as the wire's
// "From" key so the gate exercises the same envelope a caller builds, not a shortcut past it.
// The same envelope, with a RequestedData list in Options. That option is what makes the FFI
// derive its RecordSet, so it is the only way to ask "was this run charged for something the
// caller did not ask for" from outside.
std::vector<uint8_t> build_input_requesting(int64_t steps, const std::string& op,
                                            const std::vector<std::string>& requested,
                                            int64_t session = 0, bool with_rules = true,
                                            bool quotient = false) {
    wxf::Writer w;
    w.write_header();

    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(4 + (session ? 1 : 0) + (with_rules ? 1 : 0));

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("InitialStates"));
    w.write(kSeed);

    if (with_rules) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Rules"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(1);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r0"));
        w.write_function("Rule", 2);
        w.write(kLhs);
        w.write(kRhs);
    }

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Steps"));
    w.write(steps);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Options"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(quotient ? 3 : 1);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("RequestedData"));
    w.write(requested);
    if (quotient) {
        // The RAW UNFOLDING only exists as a separate quantity under quotient exploration, and
        // that needs the exact identity: in tree mode every state is its own, the reconstruction
        // never runs, and record.raw_events decides nothing. A gate for it that leaves these at
        // their defaults cannot fail.
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("CanonicalizeStates"));
        w.write_symbol("Full");
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("ExploreFromCanonicalStatesOnly"));
        w.write_symbol("True");
    }

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Op"));
    w.write(op);

    if (session) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Session"));
        w.write(session);
    }

    return w.release_data();
}

std::vector<uint8_t> build_input_with_op(int64_t steps, const std::string& op,
                                         int64_t session = 0, bool with_rules = true,
                                         const std::vector<int64_t>& from = {}) {
    wxf::Writer w;
    w.write_header();

    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(4 + (session ? 1 : 0) + (with_rules ? 1 : 0) + (from.empty() ? 0 : 1));

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("InitialStates"));
    w.write(kSeed);

    if (with_rules) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Rules"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(1);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r0"));
        w.write_function("Rule", 2);
        w.write(kLhs);
        w.write(kRhs);
    }

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Steps"));
    w.write(steps);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Options"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(0);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Op"));
    w.write(op);

    if (!from.empty()) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("From"));
        w.write(from);
    }

    if (session) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Session"));
        w.write(session);
    }

    return w.release_data();
}

// The session envelope's compatibility guarantee, which is the whole of its first commit: a job
// that names no `Op` is an `Evolve` job. Asserted on BYTES rather than counts, because equal
// counts would also hold for two runs that returned different states.
//
// Every other test in this suite sends an envelope with neither key, so they already gate the
// absent case. What they cannot gate is the two things below: that naming `Evolve` explicitly
// changes nothing, and that a word which is not a verb is REFUSED. A silently ignored `Op` is
// the failure that matters -- a caller would read a one-shot result as a session's.
TEST(WxfSerializationPin, SessionEnvelopeIsOptionalAndNonVerbsAreRefused) {
    HostBridge host;

    const auto plain = run_rewriting_core(build_input(kSeed, kLhs, kRhs, 3,
                                                      [](wxf::Writer&) {}, 0), host);
    const auto plain_again = run_rewriting_core(build_input(kSeed, kLhs, kRhs, 3,
                                                            [](wxf::Writer&) {}, 0), host);
    const auto explicit_evolve = run_rewriting_core(build_input_with_op(3, "Evolve"), host);
    ASSERT_FALSE(plain.empty());
    ASSERT_FALSE(explicit_evolve.empty());

    // WHICH PAYLOADS A BYTE COMPARISON CAN SPEAK ABOUT AT ALL.
    //
    // Ids are deliberately not deterministic. The engine does not fix a frontier or an
    // evaluation order -- it is not supposed to -- so `States` and `Events` carry ids assigned
    // in discovery order and two runs of the same job need not agree on them byte for byte.
    // What two runs owe each other is FORM equivalence: the same states graph up to
    // isomorphism, the same evolution structure, the same automorphism content. Byte equality
    // of an id-bearing payload is a stronger claim than the engine makes, and asserting it
    // asserts on the scheduler.
    //
    // Sampling does not rescue it. An earlier version admitted a payload to the comparison when
    // two runs of the same job happened to agree; under genuine non-determinism two runs can
    // coincide and a third diverge, which is how this test failed in a whole-suite run while
    // passing in isolation on a box whose other tenant was saturating a core.
    //
    // So the comparison is confined to the counts, which ARE form invariants: two isomorphic
    // evolutions have the same number of states and events whatever ids they handed out.
    for (const char* key : {"NumStates", "NumEvents"}) {
        EXPECT_EQ(value_bytes(plain, key), value_bytes(explicit_evolve, key))
            << "naming Op -> Evolve changed the " << key << " payload; that is a count, which is "
            << "invariant under the id assignment, so the envelope is not inert";
    }
    // At least the counts must be stable, or the comparison above skipped everything and the
    // test asserts nothing.
    ASSERT_EQ(value_bytes(plain, "NumStates"), value_bytes(plain_again, "NumStates"));
    ASSERT_EQ(value_bytes(plain, "NumEvents"), value_bytes(plain_again, "NumEvents"));

    // Not a verb at all. Refused, not ignored, and refused BEFORE any engine is built: a job
    // whose Op the worker does not recognise is a caller and a worker that disagree about the
    // protocol, and answering it as an Evolve would hide that for as long as the answer looked
    // plausible.
    EXPECT_THROW(run_rewriting_core(build_input_with_op(3, "Nonsense"), host), std::runtime_error);

    // A verb that addresses a held engine, with no session live and no handle given. The slot is
    // what refuses this, so it is refused whether or not the verb is wired.
    EXPECT_THROW(run_rewriting_core(build_input_with_op(3, "Step", 0, /*with_rules=*/false), host),
                 std::runtime_error);
    EXPECT_THROW(run_rewriting_core(build_input_with_op(0, "Query", 0, /*with_rules=*/false), host),
                 std::runtime_error);
}

// Open and Close against a live engine: the LIFETIME, asserted against the real worker slot
// rather than only against SessionSlot in isolation -- that a session is retained, that
// NARROWING THE REQUEST MUST NOT CHANGE THE ANSWER.
//
// RequestedData drives the FFI's RecordSet, and record.raw_events in particular decides whether
// the run reconstructs the raw unfolding at all -- 25x on multirule at depth 6, and 99.57% of
// engine cycles by RecordSet's own measurement. A derivation that turns it off for a request
// that needed it would return a smaller answer, not a slower one, and the counts are where that
// shows. Asked narrowly or asked broadly, the same question has the same answer.
TEST(WxfSerializationPin, AskingForLessDoesNotAnswerLess) {
    HostBridge host;

  for (const bool quotient : {false, true}) {
    const auto broad = run_rewriting_core(
        build_input_requesting(3, "Evolve",
                               {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"},
                               0, true, quotient),
        host);
    ASSERT_FALSE(broad.empty());

    // Each component asked for ALONE. The narrow run derives a smaller RecordSet than the broad
    // one; if that derivation drops something the component needed, this is where it shows.
    struct Case { const char* key; };
    for (const Case c : {Case{"NumStates"}, Case{"NumEvents"},
                         Case{"NumCausalEdges"}, Case{"NumBranchialEdges"}}) {
        const auto narrow = run_rewriting_core(
            build_input_requesting(3, "Evolve", {c.key}, 0, true, quotient), host);
        ASSERT_FALSE(narrow.empty()) << c.key;
        EXPECT_EQ(read_int_key(narrow, c.key), read_int_key(broad, c.key))
            << "asking for " << c.key << " alone answered differently from asking for it "
            << "alongside the others, so the record set derived from the narrow request "
            << "dropped something that component needed"
            << (quotient ? " (quotient exploration)" : " (tree mode)");
    }
  }
}

// A CONTINUATION MUST NOT DEPEND ON THE ORDER THE CALLER ASKED THINGS IN.
//
// The FFI derives its RecordSet from the properties a job requests, which is what stops a
// one-shot call paying for the raw unfolding it will not report -- 25x on multirule at depth 6.
// Deriving a SESSION's record set the same way makes the answer to a later Query depend on what
// the Open happened to name: open for "NumStates", ask for the causal relation three steps
// later, and the evolution that would have built it has already run. The relation comes back
// empty, which reads exactly like a system that has none.
//
// So a session records everything. This is the gate for that, and it is a C++ gate on purpose:
// the WL-layer session script runs under Windows wolframscript, which loads the Windows engine
// binary, so it cannot exercise a change made to the Linux one.
TEST(WxfSerializationPin, ASessionAnswersAQueryItsOpenDidNotNameTheOptionFor) {
    HostBridge host;

    // What the answer IS, asked for from the start by a one-shot call.
    const auto direct = run_rewriting_core(
        build_input_requesting(3, "Evolve", {"NumCausalEdges", "NumBranchialEdges"}), host);
    ASSERT_FALSE(direct.empty());
    const int64_t want_causal    = read_int_key(direct, "NumCausalEdges");
    const int64_t want_branchial = read_int_key(direct, "NumBranchialEdges");

    // The gate asserts nothing if the workload has no relation to lose.
    ASSERT_GT(want_causal, 0) << "this workload must HAVE causal edges, or an empty answer "
                                 "would pass for the wrong reason";

    // Open naming only the cheapest property there is, then ask for the two it did not name.
    const auto opened = run_rewriting_core(build_input_requesting(3, "Open", {"NumStates"}), host);
    ASSERT_FALSE(opened.empty());
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_NE(handle, 0);

    const auto queried = run_rewriting_core(
        build_input_requesting(3, "Query", {"NumCausalEdges", "NumBranchialEdges"}, handle,
                               /*with_rules=*/false), host);
    run_rewriting_core(build_input_with_op(0, "Close", handle, /*with_rules=*/false), host);

    EXPECT_EQ(read_int_key(queried, "NumCausalEdges"), want_causal)
        << "the session was opened naming NumStates and queried for the causal relation; it "
           "must answer what a call that asked from the start answers, not an empty relation";
    EXPECT_EQ(read_int_key(queried, "NumBranchialEdges"), want_branchial)
        << "same for the branchial relation";
}

// retaining it does not change the answer, and that the one-at-a-time rule holds here too.
TEST(WxfSerializationPin, OpenRetainsASessionAndCloseReleasesIt) {
    HostBridge host;

    const auto evolved = run_rewriting_core(build_input(kSeed, kLhs, kRhs, 3,
                                                        [](wxf::Writer&) {}, 0), host);
    const auto opened = run_rewriting_core(build_input_with_op(3, "Open"), host);
    ASSERT_FALSE(opened.empty());

    // The reply must name the session, or the caller has a handle it cannot close.
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_NE(handle, 0) << "Open must return a non-zero Session handle; 0 means 'no session'";

    // Opening returns the same ANSWER as evolving. The session is something the caller gains,
    // not a different result -- and the counts are the payloads that are byte-stable run to run
    // (States/Events carry raw ids, which follow discovery order across threads).
    EXPECT_EQ(value_bytes(evolved, "NumStates"), value_bytes(opened, "NumStates"));
    EXPECT_EQ(value_bytes(evolved, "NumEvents"), value_bytes(opened, "NumEvents"));

    // One at a time (D7): the second Open is refused, and refusing it does not disturb the
    // first. The message has to say a session is already live, or a caller cannot tell this
    // from a malformed job.
    try {
        run_rewriting_core(build_input_with_op(3, "Open"), host);
        ADD_FAILURE() << "a second Open while one is live must be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("already live"), std::string::npos) << e.what();
    }

    // A handle this worker never issued is refused, so a stale or invented one cannot close
    // somebody else's session.
    EXPECT_THROW(run_rewriting_core(build_input_with_op(0, "Close", handle + 1000), host),
                 std::runtime_error);

    // Close releases it, and a second Close is an error rather than a silent success -- closing
    // what is not open means the caller's model has diverged from the worker's.
    EXPECT_NO_THROW(run_rewriting_core(build_input_with_op(0, "Close", handle), host));
    EXPECT_THROW(run_rewriting_core(build_input_with_op(0, "Close", handle), host),
                 std::runtime_error);

    // With the slot empty, opening succeeds again -- and issues a DIFFERENT handle, because a
    // reissued one would let a stale caller address a session that is not its own.
    const auto reopened = run_rewriting_core(build_input_with_op(2, "Open"), host);
    const int64_t handle2 = read_int_key(reopened, "Session");
    EXPECT_NE(handle2, handle);
    EXPECT_NO_THROW(run_rewriting_core(build_input_with_op(0, "Close", handle2), host));
}

// EVERY HELPER IN THIS FILE THAT REDUCES A RESULT TO A COMPARABLE VALUE, FED TWO RESULTS KNOWN TO
// DIFFER. Nothing else in this suite can catch a helper that returns the same thing for different
// inputs: such a helper makes assertions PASS, so the suite goes green and says nothing.
//
// This is not hypothetical. `value_bytes` computed its slice offset from a SUB-parser's position,
// which is always 0, so it returned the first N bytes of the whole stream for every key -- for
// NumStates that is `8:`, the two-byte WXF header, identical for every run. Three assertions here
// were comparing the header to itself, and the defect surfaced only because an unrelated new test
// asserted two runs must differ before comparing them.
//
// So each helper below gets the same treatment: a pair that MUST come out different, asserted
// before any test relies on the helper to tell two things apart.
TEST(WxfSerializationPin, EveryResultHelperDistinguishesTwoResultsThatDiffer) {
    HostBridge host;

    // Two runs that differ in every count, by construction: one step against three.
    const auto shallow = run_rewriting_core(build_input(kSeed, kLhs, kRhs, 1,
                                                        [](wxf::Writer&) {}, 0), host);
    const auto deep = run_rewriting_core(build_input(kSeed, kLhs, kRhs, 3,
                                                     [](wxf::Writer&) {}, 0), host);
    ASSERT_FALSE(shallow.empty());
    ASSERT_FALSE(deep.empty());

    // read_int_key: the counts must differ, and must not be the -1 the helper returns for an
    // absent key -- which would also "differ" from a real count and prove nothing.
    for (const char* key : {"NumStates", "NumEvents"}) {
        const int64_t a = read_int_key(shallow, key), b = read_int_key(deep, key);
        EXPECT_NE(a, -1) << key << " is absent from the shallow run, so read_int_key is "
                            "reporting absence rather than a value";
        EXPECT_NE(b, -1) << key << " is absent from the deep run";
        EXPECT_NE(a, b) << "read_int_key returns the same " << key << " for a 1-step and a "
                           "3-step run, so it cannot tell two results apart";
    }

    // value_bytes: the payload of a key that differs must itself differ. This is the exact
    // assertion the old implementation failed.
    EXPECT_NE(value_bytes(shallow, "NumStates"), value_bytes(deep, "NumStates"))
        << "value_bytes returns identical bytes for two runs with different NumStates";
    EXPECT_FALSE(value_bytes(shallow, "NumStates").empty())
        << "value_bytes found nothing for a key that is present";
    // And it must address the KEY, not a fixed offset: two different keys of the same run have
    // no reason to share a payload, and a helper that slices from position 0 returns the same
    // prefix for both.
    EXPECT_NE(value_bytes(deep, "NumStates"), value_bytes(deep, "NumEvents"))
        << "value_bytes returns the same bytes for two different keys of one result";

    // count_assoc_entries: States is an association whose size tracks the run.
    const int64_t sa = count_assoc_entries(shallow, "States");
    const int64_t sb = count_assoc_entries(deep, "States");
    EXPECT_GT(sa, 0) << "count_assoc_entries reports no States entries at all";
    EXPECT_NE(sa, sb) << "count_assoc_entries returns the same States size for a 1-step and a "
                         "3-step run";
    EXPECT_EQ(count_assoc_entries(deep, "NoSuchKey"), -1)
        << "an absent key must be distinguishable from an empty association, or 'nothing was "
           "returned' reads as 'the system has none'";

    // graph_vertex_count: the same property on the two runs, which have different state counts.
    auto with_graph = [&](int64_t steps) {
        return run_rewriting_core(
            build_input(kSeed, kLhs, kRhs, steps,
                        [](wxf::Writer& w) {
                            put_str_list_option(w, "GraphProperties", {"StatesGraph"});
                            put_str_option(w, "CanonicalizeStates", "Full");
                        }, 2), host);
    };
    const int64_t va = graph_vertex_count(with_graph(1));
    const int64_t vb = graph_vertex_count(with_graph(3));
    EXPECT_NE(va, -1) << "graph_vertex_count found no Vertices field, which it reports the same "
                         "way whether the field is missing or the graph is empty";
    EXPECT_NE(va, vb) << "graph_vertex_count returns the same vertex count for a 1-step and a "
                         "3-step StatesGraph";
}

// THE CLAIM A SESSION EXISTS TO MAKE: an exploration continued in pieces is the exploration run
// whole. Open at depth 1, Step by 2, and the counts must equal a plain 3-step Evolve's -- if
// `Step` re-ran instead of resuming it would still return a 3-deep graph, with new raw ids and a
// second copy of every state, and only a comparison against the one-shot run distinguishes them.
//
// `Query` is asserted to change NOTHING, twice: once against the Open it follows and once against
// the Step. A verb that reports on a session must be a pure read, or a caller cannot look at its
// own exploration without perturbing it.
TEST(WxfSerializationPin, StepContinuesTheHeldExplorationAndQueryOnlyReportsIt) {
    HostBridge host;

    const auto whole = run_rewriting_core(build_input(kSeed, kLhs, kRhs, 3,
                                                      [](wxf::Writer&) {}, 0), host);

    const auto opened = run_rewriting_core(build_input_with_op(1, "Open"), host);
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_NE(handle, 0);

    // Non-vacuity: depth 1 and depth 3 have to be DIFFERENT graphs, or every equality below is
    // satisfied by a Step that did nothing at all.
    ASSERT_NE(read_int_key(opened, "NumStates"), read_int_key(whole, "NumStates"))
        << "this system converges before depth 3, so the continuation is not being tested";

    // A held verb carries no rules: the session's rule set was fixed at Open, and rules sent now
    // would describe a system the session is not exploring. Refused rather than ignored, and
    // refused without disturbing the session -- the Query below still has to work.
    EXPECT_THROW(run_rewriting_core(build_input_with_op(1, "Step", handle, /*with_rules=*/true),
                                    host),
                 std::runtime_error);

    const auto queried = run_rewriting_core(
        build_input_with_op(0, "Query", handle, /*with_rules=*/false), host);
    EXPECT_EQ(read_int_key(queried, "Session"), handle)
        << "a reply that came from a session must name it";
    EXPECT_EQ(read_int_key(opened, "NumStates"), read_int_key(queried, "NumStates"));
    EXPECT_EQ(read_int_key(opened, "NumEvents"), read_int_key(queried, "NumEvents"));

    const auto stepped = run_rewriting_core(
        build_input_with_op(2, "Step", handle, /*with_rules=*/false), host);
    EXPECT_EQ(read_int_key(whole, "NumStates"), read_int_key(stepped, "NumStates"))
        << "1 step then 2 more is not the same exploration as 3 steps: either the continuation "
           "resumed from the wrong frontier or it re-ran from the initial states";
    EXPECT_EQ(read_int_key(whole, "NumEvents"), read_int_key(stepped, "NumEvents"));
    EXPECT_EQ(read_int_key(whole, "NumCausalEdges"), read_int_key(stepped, "NumCausalEdges"));
    EXPECT_EQ(read_int_key(whole, "NumBranchialEdges"), read_int_key(stepped, "NumBranchialEdges"));

    const auto after = run_rewriting_core(
        build_input_with_op(0, "Query", handle, /*with_rules=*/false), host);
    EXPECT_EQ(read_int_key(stepped, "NumStates"), read_int_key(after, "NumStates"));
    EXPECT_EQ(read_int_key(stepped, "NumEvents"), read_int_key(after, "NumEvents"));

    // A handle this worker never issued reaches no engine, whichever verb names it.
    EXPECT_THROW(run_rewriting_core(build_input_with_op(1, "Step", handle + 1000,
                                                       /*with_rules=*/false), host),
                 std::runtime_error);

    ASSERT_NO_THROW(run_rewriting_core(build_input_with_op(0, "Close", handle), host));

    // The engine is gone; the verbs that addressed it say so rather than answering from a fresh
    // one, which would report an empty exploration as the caller's own.
    EXPECT_THROW(run_rewriting_core(build_input_with_op(0, "Query", handle, /*with_rules=*/false),
                                    host),
                 std::runtime_error);
}

// A Step naming frontier states continues from those and no others. The device refuses this
// verb because it holds its frontier as device ids a caller cannot name; the host resolves the
// selection against the frontier, so here it must be ANSWERED, and answered narrowly.
TEST(WxfSerializationPin, AStepNamingPartOfTheFrontierContinuesFromThatPartOnly) {
    HostBridge host;

    const auto opened = run_rewriting_core(build_input_with_op(1, "Open"), host);
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_NE(handle, 0);

    const std::vector<int64_t> frontier = read_int_list_key(opened, "Frontier");

#ifdef HG_GPU_BACKEND
    // The device session carries its frontier as device state ids with no host-visible identity,
    // so it names none of them and resolves no selection. Both halves are asserted: reporting a
    // frontier the caller could not steer by would be as wrong as answering the steered Step.
    EXPECT_TRUE(frontier.empty())
        << "the device named " << frontier.size() << " frontier states, so a caller can steer by "
           "an id the device cannot resolve";
    EXPECT_THROW(run_rewriting_core(
                     build_input_with_op(1, "Step", handle, /*with_rules=*/false, /*from=*/{0}),
                     host),
                 std::runtime_error);
    EXPECT_NO_THROW(run_rewriting_core(
        build_input_with_op(0, "Query", handle, /*with_rules=*/false), host))
        << "the refused Step took the session with it";
    ASSERT_NO_THROW(run_rewriting_core(build_input_with_op(0, "Close", handle), host));
#else
    ASSERT_GE(frontier.size(), 2u)
        << "this seed reaches a single frontier state, so steering cannot exclude anything and "
           "the comparison below would hold for a selection that was never read";

    // An id that is not on the frontier is an error, not an empty step: a caller steering toward
    // a state the exploration has already passed would otherwise get a silent no-op it cannot
    // distinguish from a branch that genuinely had no successors.
    EXPECT_THROW(run_rewriting_core(
                     build_input_with_op(1, "Step", handle, /*with_rules=*/false,
                                         /*from=*/{1000000}), host),
                 std::runtime_error);

    // The refusal leaves the session usable, so the error path is not a disguised invalidation.
    EXPECT_NO_THROW(run_rewriting_core(
        build_input_with_op(0, "Query", handle, /*with_rules=*/false), host));

    const auto steered = run_rewriting_core(
        build_input_with_op(1, "Step", handle, /*with_rules=*/false, /*from=*/{frontier[0]}), host);
    ASSERT_NO_THROW(run_rewriting_core(build_input_with_op(0, "Close", handle), host));

    // The same continuation again, naming nothing. A second session rather than the same one,
    // because the Step above already advanced this one; the slot holds a single session, so it
    // is opened after the first is closed and its equal depth-1 size is asserted, not assumed.
    const auto other = run_rewriting_core(build_input_with_op(1, "Open"), host);
    const int64_t other_handle = read_int_key(other, "Session");
    ASSERT_NE(other_handle, 0);
    ASSERT_EQ(read_int_key(other, "NumStates"), read_int_key(opened, "NumStates"))
        << "the two sessions did not open on the same exploration";
    const auto unsteered = run_rewriting_core(
        build_input_with_op(1, "Step", other_handle, /*with_rules=*/false), host);
    ASSERT_NO_THROW(run_rewriting_core(build_input_with_op(0, "Close", other_handle), host));

    EXPECT_LT(read_int_key(steered, "NumStates"), read_int_key(unsteered, "NumStates"))
        << "naming one of " << frontier.size() << " frontier states reached as many states as "
           "continuing from all of them, so the selection was not applied";
#endif
}

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

// The causal graph a caller receives has one vertex per event a caller is TOLD about.
//
// NumEvents routes through observable_num_events, which under an identity mode is the
// RECONSTRUCTION's count: distinct identities over the class frame. The graph used to be built
// by scanning MATERIALISED raw events and mapping each through its canonical_event_id, which
// full capture computes from each raw state's own labelling -- the per-state convention the
// reconstruction exists to replace. Measured then: 24 against 25, a graph and a count over
// different event sets with nothing marking the difference.
TEST(WxfSerializationPin, CausalGraphVerticesAreTheEventsTheCountReports) {
    HostBridge host;
    auto in = build_input(kSeed, kLhs, kRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "GraphProperties", {"CausalGraph"});
        put_str_option(w, "CanonicalizeStates", "Full");
        put_str_option(w, "CanonicalizeEvents", "Automatic");
    }, 3);
    auto out = run_rewriting_core(in, host);
    ASSERT_FALSE(out.empty());

    const int64_t num_events = read_int_key(out, "NumEvents");
    ASSERT_GT(num_events, 0);

    const int64_t vertex_count = graph_vertex_count(out);
    ASSERT_GE(vertex_count, 0) << "no Vertices came back for the requested CausalGraph";

    EXPECT_EQ(vertex_count, num_events)
        << "the causal graph has " << vertex_count << " event vertices while NumEvents reports "
        << num_events << ": the graph and the count describe different event sets";
}

// A relation PROVED empty is still a relation the caller asked for.
//
// record_set().branchial answers "the caller asked for this". The critical-pair work added a
// second writer (parallel_evolution.cpp, configure_identity_and_quotient): when can_branch
// proves no two matches can share a consumed edge, the flag is cleared so the run does not
// build a relation whose answer is empty. The FFI then read the cleared flag as "the caller did
// not ask" and threw, so HGEvolve returned $Failed for every no-property call -- the default
// property is "EvolutionCausalBranchialGraph" -- on any rule that cannot branch.
//
// kLhs is a SINGLE edge, which is exactly the provable case: a match IS that edge, so two
// distinct matches are two distinct edges and none can share one. That makes this corpus's
// own rule the reproducer, and it is why the branchial-free proof's own gate stayed green --
// it checks that the branchial COUNT is 0, which it correctly is.
TEST(WxfSerializationPin, ProvablyBranchialFreeRulesStillServeTheBranchialGraph) {
    HostBridge host;
    auto in = build_input(kSeed, kLhs, kRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "GraphProperties", {"EvolutionCausalBranchialGraph"});
    }, 1);
    auto out = run_rewriting_core(in, host);

    // Empty output is the failure this pins: the engine aborted rather than returning.
    ASSERT_FALSE(out.empty())
        << "the default graph property returned nothing on a rule that provably cannot branch";
    EXPECT_EQ(count_assoc_entries(out, "GraphData"), 1)
        << "EvolutionCausalBranchialGraph was requested and no GraphData came back";
}

// EVERY GRAPH PROPERTY, EVERY IDENTITY MODE, BRANCHING AND NON-BRANCHING.
//
// This surface had almost no coverage, and that is how a regression that made HGEvolve's
// DEFAULT call return nothing survived: the oracle and golden gates request "States",
// "Events", "CausalEdges" and "BranchialEdges" -- counts and lists -- so none of them enters
// hgmarshal::build_graph_data at all. The two graph properties that were pinned,
// StatesGraph and CausalGraph, happen to be the two that do not need the branchial relation.
//
// 54 cases, each cheap. The assertion is deliberately weak per case -- the engine returned
// something, and it returned one GraphData entry for the one property asked for -- because
// the failure this exists to catch is the engine returning NOTHING.
TEST(GraphPropertySurface, EveryPropertyInEveryModeOnBranchingAndNonBranchingRules) {
    HostBridge host;
    for (const char* prop : kGraphProperties) {
        for (const char* mode : kIdentityModes) {
            for (int branching = 0; branching < 2; ++branching) {
                SCOPED_TRACE(std::string(prop) + "  CanonicalizeStates -> " + mode +
                             (branching ? "  [two-edge LHS, can branch]"
                                        : "  [one-edge LHS, provably branchial-free]"));
                auto in = build_input(branching ? kBranchSeed : kSeed,
                                      branching ? kBranchLhs : kLhs,
                                      branching ? kBranchRhs : kRhs, 3,
                                      [&](wxf::Writer& w) {
                                          put_str_list_option(w, "GraphProperties", {prop});
                                          put_str_option(w, "CanonicalizeStates", mode);
                                      }, 2);
                auto out = run_rewriting_core(in, host);
                ASSERT_FALSE(out.empty()) << "the engine returned nothing for this property";
                EXPECT_EQ(count_assoc_entries(out, "GraphData"), 1)
                    << "requested one graph property and did not get one GraphData entry";
            }
        }
    }
}

// The graph a caller receives has the vertices the counts promise.
//
// A property can return a well-formed but WRONG graph, which the surface test above cannot
// see. This pins the one invariant that ties the graph to the numbers reported beside it:
// StatesGraph has one vertex per state, and EvolutionGraph has one per state plus one per
// event. CausalGraph is already pinned separately (its vertices are the events NumEvents
// reports), which is the same invariant for the third shape.
TEST(GraphPropertySurface, GraphVerticesAgreeWithTheCountsReportedBesideThem) {
    HostBridge host;
    // ALL THREE MODES, because all three now name an identity the EVOLUTION applies. The
    // graph's vertices and the count are two readings of one population only when the run
    // deduplicated by the mode it was asked for: while Automatic deduplicated nothing and was
    // regrouped afterwards, the two disagreed (17 vertices against 19 states on the two-edge
    // rule at 3 steps), because the map held keys that no surviving state's content reproduced.
    const char* const kModesWhereEstablished[] = {"None", "Automatic", "Full"};
    for (const char* mode : kModesWhereEstablished) {
        for (int branching = 0; branching < 2; ++branching) {
            const StateList& seed = branching ? kBranchSeed : kSeed;
            const EdgeList& lhs = branching ? kBranchLhs : kLhs;
            const EdgeList& rhs = branching ? kBranchRhs : kRhs;
            auto run = [&](const char* prop) {
                auto in = build_input(seed, lhs, rhs, 3, [&](wxf::Writer& w) {
                    put_str_list_option(w, "GraphProperties", {prop});
                    put_str_option(w, "CanonicalizeStates", mode);
                }, 2);
                return run_rewriting_core(in, host);
            };
            SCOPED_TRACE(std::string("CanonicalizeStates -> ") + mode +
                         (branching ? "  [can branch]" : "  [branchial-free]"));

            auto sg = run("StatesGraph");
            ASSERT_FALSE(sg.empty());
            EXPECT_EQ(graph_vertex_count(sg), read_int_key(sg, "NumStates"))
                << "StatesGraph vertices disagree with NumStates";

            auto eg = run("EvolutionGraph");
            ASSERT_FALSE(eg.empty());
            EXPECT_EQ(graph_vertex_count(eg),
                      read_int_key(eg, "NumStates") + read_int_key(eg, "NumEvents"))
                << "EvolutionGraph vertices are not the states plus the events";
        }
    }
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

// ---------------------------------------------------------------------------------------
// THE PROCESS BOUNDARY, which nothing else in this suite crosses.
//
// The GPU answers through a SEPARATE BINARY (hg_evolve_gpu). gpu_differential_tests links
// hg_gpu directly and never runs hg_gpu_backend.cpp at all, so everything that file does --
// the marshalling, the state grouping, and the session verbs -- had no gate whatsoever. That
// gap is why three disagreeing implementations of Automatic content identity survived here
// until one was read by hand.
//
// A SESSION NEEDS ONE PROCESS, which is what makes the framing matter rather than being an
// implementation detail. The device session lives in the worker that opened it: a handle is a
// pointer into that process's memory, so driving the four verbs through four one-shot
// invocations opens a session in the first and finds nothing in the second. The binary's
// --serve mode is exactly the shape a session requires -- one process, a stream of 8-byte
// length-prefixed frames -- and it is what the paclet's hgWorkerStart uses.
//
// SKIPPED, NOT FAILED, when the binary is absent: a machine without CUDA cannot build it, and
// a skip that says why is honest where a failure would be noise.
#ifndef _WIN32
namespace {

std::string gpu_binary_path() {
    return std::string(HG_SOURCE_DIR) + "/paclet/LibraryResources/Linux-x86-64/hg_evolve_gpu";
}


// One --serve worker, driven for the life of the fixture. Frames are 8-byte little-endian
// lengths followed by the WXF payload, matching run_serve exactly; a ZERO-length reply is how
// the worker reports that a job threw, which is distinct from a reply that happens to be empty.
class ServeWorker {
public:
    explicit ServeWorker(const std::string& exe) {
        const std::string cmd = "\"" + exe + "\" --serve 2>/dev/null";
        pipe_ = popen(cmd.c_str(), "w");   // write jobs; replies come back on a second channel
    }
    ~ServeWorker() { if (pipe_) pclose(pipe_); }
    bool ok() const { return pipe_ != nullptr; }
private:
    FILE* pipe_ = nullptr;
};

// popen gives one direction only, so the worker is driven through a pair of FIFOs instead:
// jobs in, replies out, one process for all four verbs.
struct WorkerPipes {
    std::string dir, in_path, out_path;
    pid_t pid = -1;
    int in_fd = -1, out_fd = -1;
    bool started = false;
};

bool worker_start(WorkerPipes& w, const std::string& exe) {
    w.dir = std::string(HG_SOURCE_DIR) + "/.gpu_gate_fifo";
    ::mkdir(w.dir.c_str(), 0700);
    w.in_path  = w.dir + "/in";
    w.out_path = w.dir + "/out";
    ::unlink(w.in_path.c_str());
    ::unlink(w.out_path.c_str());
    if (::mkfifo(w.in_path.c_str(), 0600) != 0) return false;
    if (::mkfifo(w.out_path.c_str(), 0600) != 0) return false;

    w.pid = ::fork();
    if (w.pid < 0) return false;
    if (w.pid == 0) {
        const int fin  = ::open(w.in_path.c_str(),  O_RDONLY);
        const int fout = ::open(w.out_path.c_str(), O_WRONLY);
        if (fin >= 0)  ::dup2(fin, 0);
        if (fout >= 0) ::dup2(fout, 1);
        ::execl(exe.c_str(), exe.c_str(), "--serve", (char*)nullptr);
        ::_exit(127);
    }
    w.in_fd  = ::open(w.in_path.c_str(),  O_WRONLY);
    w.out_fd = ::open(w.out_path.c_str(), O_RDONLY);
    w.started = (w.in_fd >= 0 && w.out_fd >= 0);
    return w.started;
}

void worker_stop(WorkerPipes& w) {
    if (w.in_fd  >= 0) ::close(w.in_fd);
    if (w.out_fd >= 0) ::close(w.out_fd);
    if (w.pid > 0) { int st = 0; ::waitpid(w.pid, &st, 0); }
    ::unlink(w.in_path.c_str());
    ::unlink(w.out_path.c_str());
    ::rmdir(w.dir.c_str());
}

bool read_exact_fd(int fd, size_t n, std::vector<uint8_t>& out) {
    out.assign(n, 0);
    size_t got = 0;
    while (got < n) {
        const ssize_t r = ::read(fd, out.data() + got, n - got);
        if (r <= 0) return false;
        got += static_cast<size_t>(r);
    }
    return true;
}

// Send one job, read one reply. Empty reply means the worker reported an error for that job.
std::vector<uint8_t> worker_call(WorkerPipes& w, const std::vector<uint8_t>& job) {
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>((job.size() >> (8 * i)) & 0xFF);
    if (::write(w.in_fd, len, 8) != 8) return {};
    size_t sent = 0;
    while (sent < job.size()) {
        const ssize_t r = ::write(w.in_fd, job.data() + sent, job.size() - sent);
        if (r <= 0) return {};
        sent += static_cast<size_t>(r);
    }
    std::vector<uint8_t> lenbuf;
    if (!read_exact_fd(w.out_fd, 8, lenbuf)) return {};
    uint64_t reply_len = 0;
    for (int i = 0; i < 8; ++i) reply_len |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
    if (reply_len == 0) return {};
    std::vector<uint8_t> reply;
    if (!read_exact_fd(w.out_fd, reply_len, reply)) return {};
    return reply;
}

}  // namespace

TEST(GpuBinaryGate, SessionVerbsThroughTheWorkerMatchOneEvolveOfTheSameDepth) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) {
            GTEST_SKIP() << "hg_evolve_gpu is not built here; this gate covers the process "
                            "boundary and needs the binary";
        }
    }

    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }

    const auto one_shot = worker_call(w, build_input_with_op(3, "Evolve"));
    if (one_shot.empty()) {
        worker_stop(w);
        GTEST_SKIP() << "the worker returned no result for a plain Evolve (no usable device?); "
                        "the gate abstains rather than reporting a device absence as a defect";
    }
    const int64_t ref_states = read_int_key(one_shot, "NumStates");
    const int64_t ref_events = read_int_key(one_shot, "NumEvents");
    ASSERT_GT(ref_states, 0) << "the one-shot run returned no states, so there is nothing to "
                                "compare a session against";

    const auto opened = worker_call(w, build_input_with_op(1, "Open"));
    ASSERT_FALSE(opened.empty()) << "Open through the worker errored";
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_GT(handle, 0) << "Open returned no session handle, so nothing can be stepped";

    // TWO Steps, not one: a single extend cannot tell a consumed frontier from an accumulated
    // one, which is exactly how that defect survived its first gate.
    const auto s1 = worker_call(w, build_input_with_op(1, "Step", handle, /*with_rules=*/false));
    ASSERT_FALSE(s1.empty()) << "the first Step errored";
    const auto s2 = worker_call(w, build_input_with_op(1, "Step", handle, /*with_rules=*/false));
    ASSERT_FALSE(s2.empty()) << "the second Step errored";

    EXPECT_EQ(read_int_key(s2, "NumStates"), ref_states)
        << "a GPU session stepped to depth 3 does not hold what one Evolve to depth 3 returns";
    EXPECT_EQ(read_int_key(s2, "NumEvents"), ref_events)
        << "a GPU session stepped to depth 3 does not hold the events one Evolve returns";

    const auto q = worker_call(w, build_input_with_op(0, "Query", handle, /*with_rules=*/false));
    EXPECT_FALSE(q.empty()) << "Query errored";
    // The Session key is emitted only when the session branch answered. Its absence says the
    // verb was not recognised and a fresh zero-step run replied instead, which is a different
    // defect from the session holding the wrong graph.
    EXPECT_EQ(read_int_key(q, "Session"), handle)
        << "Query was not answered from the held session";
    EXPECT_EQ(read_int_key(q, "NumStates"), ref_states)
        << "Query must report what the session holds and extend it by nothing";

    // A STEERED STEP IS REFUSED ON THE DEVICE, and refusal is the only correct answer: the
    // device session holds its frontier as device state ids with no host-visible identity, so a
    // caller's selection cannot be resolved there. Running it unsteered would explore the
    // branches the caller asked to leave alone and return a graph that is a correct answer to a
    // DIFFERENT question -- indistinguishable, at the wire, from the right one.
    //
    // The check is that the worker ERRORS the job (an empty reply) rather than answering it.
    // Asserting on the resulting state count instead would pass on a device that happened to
    // converge, which is the case a steered run is least likely to be asked about.
    const auto steered = worker_call(
        w, build_input_with_op(1, "Step", handle, /*with_rules=*/false, /*from=*/{0}));
    EXPECT_TRUE(steered.empty())
        << "a Step naming a frontier subset was ANSWERED by the GPU worker; the device cannot "
           "resolve the selection, so answering it means the branches the caller excluded were "
           "explored anyway";

    // The refusal must not take the session with it: the worker is alive and the session is
    // still the one that was opened.
    const auto after = worker_call(w, build_input_with_op(0, "Query", handle, /*with_rules=*/false));
    EXPECT_FALSE(after.empty()) << "the refused Step killed the session it refused";
    EXPECT_EQ(read_int_key(after, "NumStates"), ref_states)
        << "the refused Step changed what the session holds";

    const auto c = worker_call(w, build_input_with_op(0, "Close", handle, /*with_rules=*/false));
    EXPECT_FALSE(c.empty()) << "Close errored";

    worker_stop(w);
}

#endif  // _WIN32
