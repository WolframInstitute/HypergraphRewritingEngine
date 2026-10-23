#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
// The process gate below drives a worker through a pair of FIFOs, which needs fork, mkfifo and
// waitpid. None of them exists on Windows, and the gate is compiled out there rather than
// emulated: what it checks -- that the SHIPPED binary serves the four verbs over the wire -- is
// a property of the binary, and the Windows leg has no such binary to point at.
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#include <string>
#include <vector>

#include "wxf.hpp"
#include "paclet_source/hg_core.hpp"
#include "paclet_source/state_statistics.hpp"
#include "paclet_source/cpu_engine_holder.hpp"
#ifdef HG_GPU_BACKEND
#include "hg_gpu/branchial_rows.hpp"
#include "hg_gpu/step_pool.hpp"
#include "hg_gpu/vertex_curvature.hpp"
#endif
#include "hgcommon/affinity.hpp"
#include "hgcommon/core.hpp"
#include "paclet_source/graph_marshal.hpp"
#include "job_system/job_system.hpp"
#include "reference/oracle_corpus.hpp"

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

// Entries in a LIST-valued key. -1 when the key is absent, which is distinct from a key that
// came back empty -- the difference between "this device does not serve it" and "this run had
// none", and the first is what went unnoticed.
int64_t count_list_entries(const std::vector<uint8_t>& out, const std::string& key) {
    int64_t result = -1;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k == key) {
            // A WXF list is a Function with head List; its argument count is the length.
            vp.read_function([&](const std::string&, size_t n, wxf::Parser& ep) {
                for (size_t i = 0; i < n; ++i) ep.skip_value();
                result = static_cast<int64_t>(n);
            });
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
                                            bool quotient = false,
                                            const StateList& seed = kSeed) {
    wxf::Writer w;
    w.write_header();

    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(4 + (session ? 1 : 0) + (with_rules ? 1 : 0));

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("InitialStates"));
    w.write(seed);

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

// A job envelope for any verb: the seed and the one rule (sent when `with_rules`), the steps, the
// options, the verb, and the optional From, Session and Delivery -> "Delta" keys.
std::vector<uint8_t> session_envelope(const StateList& seed, const EdgeList& lhs,
                                      const EdgeList& rhs, int64_t steps, const std::string& op,
                                      int64_t session, bool with_rules,
                                      const std::vector<int64_t>& from,
                                      const std::function<void(wxf::Writer&)>& opts,
                                      uint64_t n_opts, bool delta, bool send_empty_from = false) {
    wxf::Writer w;
    w.write_header();

    const bool send_from = !from.empty() || send_empty_from;
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(4 + (session ? 1 : 0) + (with_rules ? 1 : 0) + (send_from ? 1 : 0) +
                   (delta ? 1 : 0));

    if (delta) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Delivery"));
        w.write(std::string("Delta"));
    }

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("InitialStates"));
    w.write(seed);

    if (with_rules) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Rules"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(1);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r0"));
        w.write_function("Rule", 2);
        w.write(lhs);
        w.write(rhs);
    }

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Steps"));
    w.write(steps);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Options"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(n_opts);
    if (opts) opts(w);

    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Op"));
    w.write(op);

    if (send_from) {
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

// A job on the single-edge rule kLhs -> kRhs from kSeed.
std::vector<uint8_t> build_input_with_op(int64_t steps, const std::string& op,
                                         int64_t session = 0, bool with_rules = true,
                                         const std::vector<int64_t>& from = {},
                                         const std::function<void(wxf::Writer&)>& opts = {},
                                         uint64_t n_opts = 0, bool delta = false) {
    return session_envelope(kSeed, kLhs, kRhs, steps, op, session, with_rules, from, opts,
                            n_opts, delta);
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
// AN INITIAL-STATE VERTEX IS A LABEL, AND A LABEL'S SIGN CARRIES NO MEANING.
//
// Negative integers are refused on the rule side, where they would be pattern variables, and
// the device remaps them on the initial-state side (hg_gpu_backend.cpp). A host that instead
// dropped them evolved a different hypergraph from the one the caller wrote -- {{1,-2},{3,4}}
// became a one-edge plus a two-edge state -- and answered without complaint, while the same
// job on the device answered for the state as written.
TEST(WxfSerializationPin, ANegativeInitialStateVertexIsRefused) {
    HostBridge host;
    const std::vector<std::string> props = {"NumStates", "NumEvents", "NumCausalEdges"};

    // Refused wherever it sits: alone in an edge, beside a non-negative vertex, and in a state
    // whose other edges are well formed. Dropping such a vertex instead evolves a hypergraph the
    // caller did not write -- {{1,-2},{3,4}} becomes a one-edge plus a two-edge state -- and an
    // all-negative state leaves nothing to evolve and returns an empty answer with no error.
    const StateList all_negative = {{{-1, -2}, {-2, -3}}};
    const StateList one_negative = {{{1, -2}, {3, 4}}};
    const StateList negative_late = {{{1, 2}, {2, 3}, {3, -1}}};

    for (const StateList& seed : {all_negative, one_negative, negative_late}) {
        EXPECT_THROW(run_rewriting_core(
                         build_input_requesting(3, "Evolve", props, 0, true, false, seed), host),
                     std::runtime_error);
    }

    // The same shape with every vertex non-negative still evolves, so what is refused is the
    // sign and not the shape.
    const auto ok = run_rewriting_core(
        build_input_requesting(3, "Evolve", props, 0, true, false, kSeed), host);
    ASSERT_FALSE(ok.empty());
    EXPECT_GT(read_int_key(ok, "NumStates"), 1);
}

// One vertex numbering serves every initial state, so a label names one vertex in all of them.
// Under CanonicalizeStates -> Automatic, {{1,3},{3,3}} and {{11,12},{12,12}} are two states with
// different edge lists (reference/MultiwayReference.wl "Content"), and {{1,2}} and {{2,1}} too.
TEST(WxfSerializationPin, InitialStatesShareOneVertexNumbering) {
    HostBridge host;
    auto automatic_states = [&](const StateList& seed) {
        const auto out = run_rewriting_core(
            build_input(seed, {{1, 2}}, {{2, 1}}, 0,
                        [](wxf::Writer& w) {
                            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                            w.write(std::string("CanonicalizeStates"));
                            w.write_symbol("Automatic");
                            put_str_list_option(w, "RequestedData", {"NumStates"});
                        },
                        2),
            host);
        return read_int_key(out, "NumStates");
    };
    EXPECT_EQ(automatic_states({{{1, 3}, {3, 3}}, {{11, 12}, {12, 12}}}), 2);
    EXPECT_EQ(automatic_states({{{1, 2}}, {{2, 1}}}), 2);
    // The same labels are the same state.
    EXPECT_EQ(automatic_states({{{1, 2}}, {{1, 2}}}), 1);
}

// Under ShowGenesisEvents each initial state has its own genesis event under every event
// identity, and a genesis event is paired with every application that consumed an initial edge,
// on full capture and under quotient exploration alike (docs/SPEC.md 5.2).
TEST(WxfSerializationPin, GenesisEventsAreOnePerInitialStateUnderEveryIdentity) {
    HostBridge host;
    auto run = [&](const StateList& seed, const EdgeList& lhs, const EdgeList& rhs, int64_t steps,
                   const char* events, bool quotient, const char* key) {
        const auto out = run_rewriting_core(
            build_input(seed, lhs, rhs, steps,
                        [&](wxf::Writer& w) {
                            for (auto [k, v] : {std::pair<const char*, const char*>{"CanonicalizeStates", "Full"},
                                                {"CanonicalizeEvents", events},
                                                {"ShowGenesisEvents", "True"},
                                                {"ExploreFromCanonicalStatesOnly", quotient ? "True" : "False"}}) {
                                w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                                w.write(std::string(k));
                                w.write_symbol(v);
                            }
                            put_str_list_option(w, "RequestedData", {"NumEvents", "NumCausalEdges"});
                        },
                        5),
            host);
        return read_int_key(out, key);
    };
    for (bool quotient : {false, true}) {
        SCOPED_TRACE(quotient ? "quotient exploration" : "full capture");
        // Two isomorphic roots: 2 event classes under Full and one genesis event per root.
        EXPECT_EQ(run({{{1, 2}}, {{3, 4}}}, {{1, 2}}, {{1, 3}, {3, 2}}, 2, "Full", quotient, "NumEvents"), 4);
        // Two applications of one class consume the root's edges: two genesis pairs
        // (reference/MultiwayReference.wl with "GenesisEvents" -> True).
        EXPECT_EQ(run({{{1}, {1}}}, {{1}, {1}}, {{1}}, 3, "Full", quotient, "NumCausalEdges"), 2);
    }
}

// A key list compares the consumed and produced edges as marked forms (docs/SPEC.md 4.2), also
// when it names exactly Automatic's five components. The two applications of
// {{1},{2}} -> {{2,3},{2,3},{2}} to {{2},{4}} consume every edge, produce isomorphic states with
// every produced edge marked, and are one event (reference/MultiwayReference.wl).
TEST(WxfSerializationPin, AKeyListOfAutomaticsComponentsComparesMarkedForms) {
    HostBridge host;
    auto events = [&](const std::function<void(wxf::Writer&)>& identity) {
        const auto out = run_rewriting_core(
            build_input({{{2}, {4}}}, {{1}, {2}}, {{2, 3}, {2, 3}, {2}}, 1,
                        [&](wxf::Writer& w) {
                            put_str_option(w, "CanonicalizeStates", "Full");
                            identity(w);
                            put_str_list_option(w, "RequestedData", {"NumEvents"});
                        },
                        3),
            host);
        return read_int_key(out, "NumEvents");
    };
    EXPECT_EQ(events([](wxf::Writer& w) {
                  put_str_list_option(w, "CanonicalizeEvents",
                                      {"InputState", "OutputState", "Step", "ConsumedEdges",
                                       "ProducedEdges"});
              }),
              1);
    EXPECT_EQ(events([](wxf::Writer& w) {
                  put_str_list_option(w, "CanonicalizeEvents",
                                      {"InputState", "OutputState", "Step", "ConsumedEdges",
                                       "ProducedEdges", "Rule"});
              }),
              1);
}

// Automatic event identity ranks the edges of {{1},{2}} by first appearance on both devices
// (hgcommon::ir_renumber_encounter): {{1},{2}} -> {{1},{2}} and {{1},{2}} -> {{2},{1}} give two
// events in one step, as Wolfram/Multicomputation's MultiwaySystem counts them.
TEST(WxfSerializationPin, AutomaticRanksComeFromTheFirstAppearanceNumbering) {
    HostBridge host;
    wxf::Writer w;
    w.write_header();
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(4);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("InitialStates"));
    w.write(StateList{{{1}, {2}}});
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Rules"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(2);
    for (auto [name, rhs] : {std::pair<const char*, EdgeList>{"r0", {{1}, {2}}}, {"r1", {{2}, {1}}}}) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string(name));
        w.write_function("Rule", 2);
        w.write(EdgeList{{1}, {2}});
        w.write(rhs);
    }
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Steps"));
    w.write(int64_t{1});
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Options"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(3);
    put_str_option(w, "CanonicalizeStates", "Full");
    put_str_option(w, "CanonicalizeEvents", "Automatic");
    put_str_list_option(w, "RequestedData", {"NumEvents"});
    EXPECT_EQ(read_int_key(run_rewriting_core(w.release_data(), host), "NumEvents"), 2);
}

// MatchesPerStep counts transitions also when MaxSuccessorStatesPerParent is set (docs/SPEC.md 5).
// {{1,1}} -> {} on {{2,2},{2,2}} under Automatic states makes two transitions into one state;
// MatchesPerStep -> 1 keeps one of them, and a per-parent cap of 4 keeps both of them.
TEST(WxfSerializationPin, MatchesPerStepCountsTransitionsUnderAPerParentCap) {
    HostBridge host;
    auto events = [&](bool per_parent) {
        const auto out = run_rewriting_core(
            build_input({{{2, 2}, {2, 2}}}, {{1, 1}}, {}, 1,
                        [&](wxf::Writer& w) {
                            put_str_option(w, "CanonicalizeStates", "Automatic");
                            put_str_option(w, "UniformRandom", "True");
                            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                            w.write(std::string("MatchesPerStep"));
                            w.write(int64_t{1});
                            if (per_parent) {
                                w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                                w.write(std::string("MaxSuccessorStatesPerParent"));
                                w.write(int64_t{4});
                            }
                            put_str_list_option(w, "RequestedData", {"NumEvents"});
                        },
                        per_parent ? 5 : 4),
            host);
        return read_int_key(out, "NumEvents");
    };
    EXPECT_EQ(events(false), 1);
    EXPECT_EQ(events(true), 1);
}

// The genesis events' input state is not a state of the evolution (docs/SPEC.md 5.2), so
// "StepStatistics" is the same with and without ShowGenesisEvents.
TEST(WxfSerializationPin, StepStatisticsLeaveOutTheGenesisState) {
    HostBridge host;
    auto stats = [&](bool genesis) {
        return value_bytes(
            run_rewriting_core(
                build_input({{{1, 2}}}, {{1, 2}}, {{1, 3}, {3, 2}}, 2,
                            [&](wxf::Writer& w) {
                                put_str_option(w, "ShowGenesisEvents", genesis ? "True" : "False");
                                put_str_list_option(w, "RequestedData", {"StepStatistics"});
                            },
                            2),
                host),
            "StepStatistics");
    };
    const auto without = stats(false);
    ASSERT_FALSE(without.empty());
    EXPECT_EQ(stats(true), without);
}

// "StepStatistics" under full capture is the same on every run. On the device only the state
// that creates an isomorphism class writes the class's record, and which state that is depends
// on the schedule; a state summarised before its class's record was found counted as an empty
// state (tools/dev/worker_fuzz.py seed 4627: 4 of 30 runs differed at step 3).
TEST(WxfSerializationPin, StepStatisticsAreTheSameOnEveryRun) {
    HostBridge host;
    auto job = [] {
        wxf::Writer w;
        w.write_header();
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(4);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("InitialStates"));
        w.write(StateList{{{1, 3}, {3, 3, 3}}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Rules"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(3);
        const std::pair<EdgeList, EdgeList> rules[] = {
            {{{2, 3, 4}}, {}}, {{{2, 3, 3}}, {{2, 2, 2}}}, {{{2, 1}}, {{2, 4}, {2, 1}, {2, 1}}}};
        for (size_t i = 0; i < 3; ++i) {
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write("r" + std::to_string(i));
            w.write_function("Rule", 2);
            w.write(rules[i].first);
            w.write(rules[i].second);
        }
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Steps"));
        w.write(int64_t{4});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Options"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(1);
        put_str_list_option(w, "RequestedData",
                            {"NumEvents", "NumCausalEdges", "NumBranchialEdges", "StepStatistics"});
        return w.release_data();
    }();
    const auto first = value_bytes(run_rewriting_core(job, host), "StepStatistics");
    ASSERT_FALSE(first.empty());
    for (int i = 0; i < 100; ++i)
        ASSERT_EQ(value_bytes(run_rewriting_core(job, host), "StepStatistics"), first) << "run " << i;
}

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
// A session under quotient exploration, stepped 1 + 1 + 1, holds the raw events and relations
// one Evolve to depth 3 reconstructs.
void quotient_session_options(wxf::Writer& w) {
    put_str_option(w, "CanonicalizeStates", "Full");
    put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
    put_str_list_option(w, "RequestedData", {"NumEvents", "NumCausalEdges", "NumBranchialEdges"});
}
// Counts only: the raw counts come from class multiplicities, not the replay.
void quotient_counts_session_options(wxf::Writer& w) {
    put_str_option(w, "CanonicalizeStates", "Full");
    put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
    put_str_list_option(w, "RequestedData", {"NumEvents", "NumBranchialEdges"});
}

TEST(WxfSerializationPin, AQuotientSessionSteppedHoldsWhatOneEvolveReconstructs) {
    for (auto opts : {quotient_session_options, quotient_counts_session_options}) {
        HostBridge host;
        const auto one_shot = run_rewriting_core(
            build_input_with_op(3, "Evolve", 0, true, {}, opts, 3), host);
        const auto opened = run_rewriting_core(
            build_input_with_op(1, "Open", 0, true, {}, opts, 3), host);
        const int64_t handle = read_int_key(opened, "Session");
        ASSERT_GT(handle, 0);
        run_rewriting_core(build_input_with_op(1, "Step", handle, false, {}, opts, 3), host);
        const auto s2 = run_rewriting_core(
            build_input_with_op(1, "Step", handle, false, {}, opts, 3), host);
        for (const char* k : {"NumEvents", "NumCausalEdges", "NumBranchialEdges"})
            EXPECT_EQ(read_int_key(s2, k), read_int_key(one_shot, k)) << k;
        run_rewriting_core(build_input_with_op(0, "Close", handle, false), host);
    }
}

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

// A Step naming frontier states continues from those and no others, on BOTH devices: the
// selection is resolved against the frontier as it was last reported, and the unselected
// entries are put back so a later Step can still resume them.
TEST(WxfSerializationPin, AStepNamingPartOfTheFrontierContinuesFromThatPartOnly) {
    HostBridge host;

    const auto opened = run_rewriting_core(build_input_with_op(1, "Open"), host);
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_NE(handle, 0);

    const std::vector<int64_t> frontier = read_int_list_key(opened, "Frontier");
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
    // RETENTION: the entries the selection excluded are put back, not dropped -- a later Step
    // can still resume them. The excluded id was not expanded, so it is still on the frontier
    // the steered reply reports.
    const std::vector<int64_t> after = read_int_list_key(steered, "Frontier");
    EXPECT_NE(std::find(after.begin(), after.end(), frontier[1]), after.end())
        << "steering by state " << frontier[0] << " dropped unselected state " << frontier[1]
        << " from the frontier, so \"explore this branch\" meant \"abandon the others\"";
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
}

// Counts the warnings in a reply. Returns -1 when the key is absent, which is distinct from a
// reply that carries an empty warning list.
int64_t count_warnings(const std::vector<uint8_t>& out) {
    int64_t n = -1;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "Warnings") { vp.skip_value(); return; }
        vp.read_function([&](const std::string&, size_t count, wxf::Parser& ep) {
            for (size_t i = 0; i < count; ++i) ep.skip_value();
            n = static_cast<int64_t>(count);
        });
    });
    return n;
}

// A saving is not a gap. The engine drops the branchial relation when the rules provably cannot
// branch, and a session's later Query for it must not be told the empty answer is an artefact of
// what its Open asked for -- that inverts the truth, because the system genuinely has no
// branchial pairs.
//
// kLhs is the one-edge left-hand side this file already documents as the provably branchial-free
// case, which is what makes the engine take that path here.
TEST(WxfSerializationPin, AProvablyEmptyRelationIsNotReportedAsUnrecorded) {
    HostBridge host;

    const auto opened = run_rewriting_core(build_input_with_op(1, "Open"), host);
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_NE(handle, 0);

    const auto queried = run_rewriting_core(
        build_input_requesting(0, "Query", {"NumBranchialEdges"}, handle, /*with_rules=*/false),
        host);

    EXPECT_EQ(read_int_key(queried, "NumBranchialEdges"), 0)
        << "this rule set cannot branch, so the count is zero and the warning below is the only "
           "thing at issue";
    EXPECT_LE(count_warnings(queried), 0)
        << "a session was warned that it did not ask to record the branchial relation, when it "
           "opened recording everything and the relation is empty because the rules cannot "
           "branch. The warning tells the caller its empty answer is an artefact of its request, "
           "which is the opposite of what happened";

    ASSERT_NO_THROW(run_rewriting_core(build_input_with_op(0, "Close", handle), host));
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

// The empty state reports hgcommon::EMPTY_STATE_CANONICAL_HASH as its CanonicalHash in every
// state mode. Rule {{1}} -> {} from {{1}}, one step: the states are {{1}} and {}.
TEST(WxfSerializationPin, TheEmptyStateHasOneCanonicalHash) {
    for (const char* mode : {"None", "Automatic", "Full"}) {
        auto input = build_input({{{1}}}, {{1}}, {}, 1,
                                 [&](wxf::Writer& w) {
                                     put_str_option(w, "CanonicalizeStates", mode);
                                     put_str_option(w, "IncludeCanonicalHashes", "True");
                                 },
                                 2);
        HostBridge host;
        auto out = run_rewriting_core(input, host);
        std::set<int64_t> hashes;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "States") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
                kp.skip_value();
                valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                    if (fk == "CanonicalHash") hashes.insert(fvp.read<int64_t>());
                    else fvp.skip_value();
                });
            });
        });
        EXPECT_EQ(hashes.size(), 2u) << mode;
        EXPECT_EQ(hashes.count(static_cast<int64_t>(hgcommon::EMPTY_STATE_CANONICAL_HASH)), 1u)
            << mode;
        EXPECT_EQ(hashes.count(0), 0u) << mode;
    }
}

// Under Full a state record's Step is its class's least step, the same in every run. Two rules
// whose classes are reached at several depths; 12 runs at the default worker count.
TEST(WxfSerializationPin, FullStateStepIsTheClassLeastStep) {
    auto job = [] {
        wxf::Writer w;
        w.write_header();
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(4);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("InitialStates"));
        w.write(StateList{{{3, 1}, {3, 2, 1}, {3, 2}, {1, 1}, {3, 3}}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Rules"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(2);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r0"));
        w.write_function("Rule", 2);
        w.write(EdgeList{{4, 3}, {1, 2}});
        w.write(EdgeList{{5, 5}, {2, 3}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r1"));
        w.write_function("Rule", 2);
        w.write(EdgeList{{3, 2, 3}, {2, 2}});
        w.write(EdgeList{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Steps"));
        w.write(int64_t{4});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Options"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(2);
        put_str_option(w, "CanonicalizeStates", "Full");
        put_str_option(w, "IncludeCanonicalHashes", "True");
        return w.release_data();
    };
    std::set<std::set<std::pair<int64_t, int64_t>>> seen;
    for (int rep = 0; rep < 12; ++rep) {
        HostBridge host;
        const auto out = run_rewriting_core(job(), host);
        std::set<std::pair<int64_t, int64_t>> hash_step;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "States") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
                kp.skip_value();
                int64_t h = 0, step = -1;
                valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                    if (fk == "CanonicalHash") h = fvp.read<int64_t>();
                    else if (fk == "Step") step = fvp.read<int64_t>();
                    else fvp.skip_value();
                });
                hash_step.insert({h, step});
            });
        });
        seen.insert(hash_step);
    }
    EXPECT_EQ(seen.size(), 1u);
}

// Under quotient exploration a state record's Step is its class's shortest depth, the same in
// every run, as under full capture. The same case with ExploreFromCanonicalStatesOnly.
TEST(WxfSerializationPin, QuotientStateStepIsTheClassShortestDepth) {
    auto job = [](bool ecso) {
        wxf::Writer w;
        w.write_header();
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(4);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("InitialStates"));
        w.write(StateList{{{3, 1}, {3, 2, 1}, {3, 2}, {1, 1}, {3, 3}}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Rules"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(2);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r0"));
        w.write_function("Rule", 2);
        w.write(EdgeList{{4, 3}, {1, 2}});
        w.write(EdgeList{{5, 5}, {2, 3}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r1"));
        w.write_function("Rule", 2);
        w.write(EdgeList{{3, 2, 3}, {2, 2}});
        w.write(EdgeList{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Steps"));
        w.write(int64_t{4});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Options"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(3);
        put_str_option(w, "CanonicalizeStates", "Full");
        put_str_option(w, "IncludeCanonicalHashes", "True");
        put_str_option(w, "ExploreFromCanonicalStatesOnly", ecso ? "True" : "False");
        return w.release_data();
    };
    std::set<std::set<std::pair<int64_t, int64_t>>> seen;
    for (int rep = 0; rep < 13; ++rep) {
        HostBridge host;
        const auto out = run_rewriting_core(job(rep < 12), host);
        std::set<std::pair<int64_t, int64_t>> hash_step;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "States") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
                kp.skip_value();
                int64_t h = 0, step = -1;
                valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                    if (fk == "CanonicalHash") h = fvp.read<int64_t>();
                    else if (fk == "Step") step = fvp.read<int64_t>();
                    else fvp.skip_value();
                });
                hash_step.insert({h, step});
            });
        });
        seen.insert(hash_step);
    }
    EXPECT_EQ(seen.size(), 1u);
}

// Under Full states and Automatic events the causal and branchial lists' From/To are ids of the
// "Events" property, in every run (docs/SPEC.md §5.1). Full capture, two rules, 4 steps, 8 runs.
TEST(WxfSerializationPin, RelationEndpointsAreEventsUnderTheReconstruction) {
    auto job = [] {
        wxf::Writer w;
        w.write_header();
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(4);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("InitialStates"));
        w.write(StateList{{{4, 2, 2}, {5, 5, 4}}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Rules"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(2);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r0"));
        w.write_function("Rule", 2);
        w.write(EdgeList{{2, 2, 1}});
        w.write(EdgeList{{2, 2, 2}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("r1"));
        w.write_function("Rule", 2);
        w.write(EdgeList{{1, 1, 1}});
        w.write(EdgeList{{2, 3, 3, 3}, {3, 3}, {1}});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Steps"));
        w.write(int64_t{4});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Options"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(3);
        put_str_option(w, "CanonicalizeStates", "Full");
        put_str_option(w, "CanonicalizeEvents", "Automatic");
        put_str_list_option(w, "RequestedData", {"Events", "CausalEdges", "BranchialEdges"});
        return w.release_data();
    };
    std::set<std::multiset<std::pair<int64_t, int64_t>>> labelled;
    for (int rep = 0; rep < 8; ++rep) {
        HostBridge host;
        const auto out = run_rewriting_core(job(), host);
        std::set<int64_t> event_ids;
        std::map<int64_t, int64_t> rule_of;
        std::vector<int64_t> endpoints;
        std::vector<bool> unordered;   // per pair: a branchial pair
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k == "Events") {
                vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
                    kp.skip_value();
                    int64_t id = -1, rule = -1;
                    valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                        if (fk == "CanonicalId") id = fvp.read<int64_t>();
                        else if (fk == "RuleIndex") rule = fvp.read<int64_t>();
                        else fvp.skip_value();
                    });
                    event_ids.insert(id);
                    rule_of.emplace(id, rule);
                });
            } else if (k == "CausalEdges" || k == "BranchialEdges") {
                vp.read_function([&](const std::string&, size_t n, wxf::Parser& ep) {
                    for (size_t i = 0; i < n; ++i) {
                        ep.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                            if (fk == "From" || fk == "To") endpoints.push_back(fvp.read<int64_t>());
                            else fvp.skip_value();
                        });
                        unordered.push_back(k == "BranchialEdges");
                    }
                });
            } else {
                vp.skip_value();
            }
        });
        ASSERT_FALSE(endpoints.empty());
        std::multiset<std::pair<int64_t, int64_t>> pairs;
        for (size_t i = 0; i < endpoints.size(); ++i)
            EXPECT_EQ(event_ids.count(endpoints[i]), 1u) << "run " << rep << ": endpoint " << endpoints[i];
        // A branchial pair is unordered: its From is the lower id, and ids are run-local.
        for (size_t i = 0; i + 1 < endpoints.size(); i += 2) {
            int64_t a = rule_of[endpoints[i]], b = rule_of[endpoints[i + 1]];
            if (unordered[i / 2] && b < a) std::swap(a, b);
            pairs.insert({a, b});
        }
        labelled.insert(pairs);
    }
    // The relations joined to "Events" by id, as (rule of From, rule of To): one answer.
    std::string got;
    for (const auto& ps : labelled) {
        got += " {";
        for (const auto& pr : ps)
            got += "(" + std::to_string(pr.first) + "," + std::to_string(pr.second) + ")";
        got += "}";
    }
    EXPECT_EQ(labelled.size(), 1u) << got;
}

// Under ShowGenesisEvents the genesis event's input state, which stands before the initial
// states, is not one of the evolution's states: "States" lists NumStates records. Rule
// {{1,2}} -> {{1,2},{2,3}} from {{1,2}}, 1 step, None and Full.
TEST(WxfSerializationPin, GenesisInputStateIsNotListed) {
    for (const char* mode : {"None", "Full"}) {
        auto input = build_input({{{1, 2}}}, {{1, 2}}, {{1, 2}, {2, 3}}, 1,
                                 [&](wxf::Writer& w) {
                                     put_str_option(w, "CanonicalizeStates", mode);
                                     put_str_option(w, "ShowGenesisEvents", "True");
                                     put_str_list_option(w, "RequestedData", {"States", "NumStates"});
                                 },
                                 3);
        HostBridge host;
        const auto out = run_rewriting_core(input, host);
        EXPECT_EQ(count_assoc_entries(out, "States"), read_int_key(out, "NumStates")) << mode;
    }
}

// ShowGenesisEvents under Full states: quotient exploration gives full capture's NumEvents,
// NumCausalEdges and causal list, with and without CausalTransitiveReduction (docs/SPEC.md
// §5.2, §5.4): the genesis event of an initial state causes every application that consumed
// one of its edges.
TEST(WxfSerializationPin, GenesisEventsAgreeAcrossRoutes) {
    struct Case { StateList init; EdgeList lhs, rhs; int64_t steps; };
    const Case cases[] = {
        {{{{1, 2}}}, {{1, 2}}, {{1, 3}, {3, 2}}, 2},
        {{{{1, 2}}}, {{1, 2}}, {{1, 2}, {2, 3}}, 2},
        {{{{1, 2}, {2, 3}}}, {{1, 2}, {2, 3}}, {{1, 3}, {3, 4}, {4, 2}}, 2},
        {{{{1, 1}, {1, 1}}}, {{1, 2}}, {{1, 2}, {2, 3}}, 3},
        {{{{1, 2}}, {{1, 1}, {2, 1}}}, {{1, 2}}, {{1, 3}, {3, 2}}, 2},
    };
    for (const Case& c : cases) {
        for (const char* tr : {"True", "False"}) {
            int64_t counts[2][4] = {};
            for (int q = 0; q < 2; ++q) {
                auto input = build_input(c.init, c.lhs, c.rhs, c.steps,
                                         [&](wxf::Writer& w) {
                                             put_str_option(w, "CanonicalizeStates", "Full");
                                             put_str_option(w, "ShowGenesisEvents", "True");
                                             put_str_option(w, "CausalTransitiveReduction", tr);
                                             put_str_option(w, "ExploreFromCanonicalStatesOnly",
                                                            q ? "True" : "False");
                                             put_str_list_option(w, "RequestedData",
                                                 {"NumStates", "NumEvents", "NumCausalEdges",
                                                  "CausalEdges"});
                                         },
                                         5);
                HostBridge host;
                const auto out = run_rewriting_core(input, host);
                counts[q][0] = read_int_key(out, "NumStates");
                counts[q][1] = read_int_key(out, "NumEvents");
                counts[q][2] = read_int_key(out, "NumCausalEdges");
                counts[q][3] = count_list_entries(out, "CausalEdges");
            }
            for (int k = 0; k < 4; ++k)
                EXPECT_EQ(counts[1][k], counts[0][k])
                    << "case steps " << c.steps << " TR " << tr << " field " << k;
            EXPECT_EQ(counts[0][3], counts[0][2]);
            // The chain rule from {{1,2}}, 2 steps, as HGEvolve.md documents it: 4 events with
            // the genesis event, 3 causal pairs.
            if (&c == &cases[0] && std::string(tr) == "True") {
                EXPECT_EQ(counts[0][1], 4);
                EXPECT_EQ(counts[0][2], 3);
            }
        }
    }
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
// THE COMPONENTS THAT ARE DERIVED RATHER THAN REPORTED, asked of whichever engine this binary
// was compiled against.
//
// BranchialStateEdges, BranchialStateEdgesAllSiblings, GlobalEdges and StateBitvectors are built
// from the relations and the edge store rather than handed over by the engine, and each was
// served by the host and absent from a device reply -- the key simply missing, so a caller got a
// shorter association from the same request. Nothing executed them on the device to notice,
// which is why the assertions live in this file: it is compiled into all_tests against the host
// and into gpu_ffi_tests against the device.
//
// A two-edge left-hand side, so two matches can share a consumed edge and the branchial relation
// is not empty. Counts are not compared across devices -- state ids are each engine's own -- so
// what is asserted is that the key is SERVED and its two halves agree with each other.
TEST(WxfSerializationPin, TheDerivedComponentsAreServed) {
    HostBridge host;
    auto in = build_input(kBranchSeed, kBranchLhs, kBranchRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData",
            {"States", "Events", "BranchialStateEdges", "GlobalEdges", "StateBitvectors"});
    }, 1);
    auto out = run_rewriting_core(in, host);
    ASSERT_FALSE(out.empty());

    const int64_t bse = count_list_entries(out, "BranchialStateEdges");
    ASSERT_GE(bse, 0) << "BranchialStateEdges was asked for and no such key came back";
    EXPECT_GT(bse, 0) << "this rule branches, so the branchial relation is not empty";

    const auto verts = read_int_list_key(out, "BranchialStateVertices");
    EXPECT_GT(verts.size(), 0u) << "edges were served with no vertices beside them";
    EXPECT_LE(static_cast<int64_t>(verts.size()), 2 * bse)
        << "more distinct endpoints than the edges can have";

    const int64_t ge = count_list_entries(out, "GlobalEdges");
    ASSERT_GE(ge, 0) << "GlobalEdges was asked for and no such key came back";
    EXPECT_GT(ge, 0) << "an evolution that produced states produced edges";

    const int64_t sb = count_assoc_entries(out, "StateBitvectors");
    ASSERT_GE(sb, 0) << "StateBitvectors was asked for and no such key came back";
    EXPECT_GT(sb, 0) << "an evolution that produced states has an edge set for each";
}

// The all-siblings form is a DIFFERENT rule under the same key: every two events leaving one
// input state, whether or not their consumed edges overlap. It therefore cannot return fewer
// edges than the overlap form on the same run.
TEST(WxfSerializationPin, AllSiblingsIsAtLeastTheOverlapRelation) {
    HostBridge host;

    auto overlap_in = build_input(kBranchSeed, kBranchLhs, kBranchRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData", {"States", "BranchialStateEdges"});
    }, 1);
    auto overlap = run_rewriting_core(overlap_in, host);
    ASSERT_FALSE(overlap.empty());

    auto siblings_in = build_input(kBranchSeed, kBranchLhs, kBranchRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData",
            {"States", "BranchialStateEdgesAllSiblings"});
    }, 1);
    auto siblings = run_rewriting_core(siblings_in, host);
    ASSERT_FALSE(siblings.empty());

    const int64_t a = count_list_entries(overlap, "BranchialStateEdges");
    const int64_t b = count_list_entries(siblings, "BranchialStateEdges");
    ASSERT_GT(a, 0);
    ASSERT_GE(b, 0) << "BranchialStateEdgesAllSiblings was asked for and nothing came back";
    EXPECT_GE(b, a) << "all sibling pairs cannot be fewer than the overlapping ones: "
                    << b << " against " << a;
}

// SHOWING GENESIS EVENTS ADDS EXACTLY ONE EVENT PER INITIAL STATE, on whichever engine this
// binary was compiled against.
//
// A genesis event connects a synthetic genesis state to an initial state and produces that
// state's edges. The host mints them during evolution and filters them out unless asked; the
// device mints none and synthesises them here, so this asserts the SHAPE both must produce
// rather than comparing ids, which are each engine's own.
//
// It also changes the CAUSAL relation, which is the part that makes this more than a cosmetic
// list: a genesis event is the producer of every initial edge, so any event consuming one is
// caused by it. A run whose first events consume the initial edges must therefore gain causal
// pairs as well as events.
TEST(WxfSerializationPin, ShowingGenesisEventsAddsOnePerInitialState) {
    HostBridge host;

    auto plain_in = build_input(kBranchSeed, kBranchLhs, kBranchRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData", {"Events", "CausalEdges"});
    }, 1);
    auto plain = run_rewriting_core(plain_in, host);
    ASSERT_FALSE(plain.empty());

    auto shown_in = build_input(kBranchSeed, kBranchLhs, kBranchRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData", {"Events", "CausalEdges"});
        put_str_option(w, "ShowGenesisEvents", "True");
    }, 2);
    auto shown = run_rewriting_core(shown_in, host);
    ASSERT_FALSE(shown.empty());

    const int64_t plain_events = count_assoc_entries(plain, "Events");
    const int64_t shown_events = count_assoc_entries(shown, "Events");
    ASSERT_GT(plain_events, 0);
    // kBranchSeed is ONE initial state, so exactly one genesis event joins the set.
    EXPECT_EQ(shown_events, plain_events + 1)
        << "showing genesis events did not add exactly one event per initial state";

    const int64_t plain_causal = count_list_entries(plain, "CausalEdges");
    const int64_t shown_causal = count_list_entries(shown, "CausalEdges");
    ASSERT_GE(plain_causal, 0);
    EXPECT_GT(shown_causal, plain_causal)
        << "the genesis event produced the initial edges, so events consuming them are caused "
           "by it -- showing it must add causal pairs, not only an event";
}

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

// The branchial edges of an evolution graph are one per unordered pair of events, so their
// number does not depend on which event of a pair the run numbered first. Rule
// {{x,x},{x,y}} -> {{x,x},{x,y},{y,y},{y,z}} from {{0,0},{0,1}}, 4 steps, Full states with
// Automatic events (the quotient route in full capture, whose application ids follow the
// schedule): the count at 1, 2 and 8 workers, five runs each, is one number. The two-run
// difference measured before: 108 to 190 branchial edges.
TEST(GraphPropertySurface, EvolutionGraphBranchialEdgesDoNotDependOnTheSchedule) {
    const StateList seed = {{{0, 0}, {0, 1}}};
    const EdgeList lhs = {{0, 0}, {0, 1}};
    const EdgeList rhs = {{0, 0}, {0, 1}, {1, 1}, {1, 2}};
    const std::vector<uint8_t> tag = {0x53, 0x09, 'B', 'r', 'a', 'n', 'c', 'h', 'i', 'a', 'l'};
    std::set<size_t> counts;
    for (const int64_t threads : {1, 2, 8})
        for (int run = 0; run < 5; ++run) {
            HostBridge host;
            const auto out = run_rewriting_core(build_input(seed, lhs, rhs, 4,
                [&](wxf::Writer& w) {
                    put_str_list_option(w, "RequestedData", {});
                    put_str_list_option(w, "GraphProperties", {"EvolutionCausalBranchialGraph"});
                    put_str_option(w, "CanonicalizeStates", "Full");
                    put_str_option(w, "CanonicalizeEvents", "Automatic");
                    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    w.write(std::string("Threads"));
                    w.write(threads);
                }, 5), host);
            size_t n = 0;
            for (auto it = out.begin(); (it = std::search(it, out.end(), tag.begin(), tag.end())) !=
                                        out.end();
                 ++it)
                ++n;
            counts.insert(n);
        }
    EXPECT_EQ(counts.size(), 1u) << "branchial edge counts: " << *counts.begin() << " to "
                                 << *counts.rbegin();
    EXPECT_GT(*counts.begin(), 0u);
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
namespace {

// A job on the WPP rule (kBranch*) with the given options, for any verb. Rules are sent with
// Evolve and Open only; a held verb carries none.
std::vector<uint8_t> branch_job(int64_t steps, const std::string& op, int64_t session,
                                const std::function<void(wxf::Writer&)>& write_options,
                                std::size_t option_count, bool delta = false) {
    return session_envelope(kBranchSeed, kBranchLhs, kBranchRhs, steps, op, session,
                            op == "Evolve" || op == "Open", {}, write_options, option_count,
                            delta);
}

bool reply_mentions(const std::vector<uint8_t>& out, const std::string& text) {
    return std::search(out.begin(), out.end(), text.begin(), text.end()) != out.end();
}

// The per-step branchial keys a "StepStatistics" reply carries.
const std::vector<std::string> kBranchialKeys = {
    "BranchialDegree", "BranchialDistance", "BranchialComponents", "BranchialDimension",
    "StateOverlap", "StateCosineSimilarity", "StateMutualInformation",
    "InitialStateMutualInformation", "VertexSharpness", "BranchEntropy", "EdgeSharpness",
    "EdgeBranchEntropy", "OverlapByBranchialDistance"};

// Per step, each branchial key of the step's "StepStatistics" record and its value's bytes.
using KeyBytes = std::vector<std::pair<std::string, std::vector<uint8_t>>>;
std::vector<KeyBytes> branchial_record_bytes(const std::vector<uint8_t>& out) {
    const auto stats = value_bytes(out, "StepStatistics");
    std::vector<KeyBytes> rows;
    if (stats.empty()) return rows;
    wxf::Parser p(stats);
    p.read_function([&](const std::string&, size_t n, wxf::Parser& lp) {
        for (size_t i = 0; i < n; ++i) {
            KeyBytes row;
            lp.read_association([&](const std::string& k, wxf::Parser& vp) {
                const uint8_t* begin = vp.data();
                vp.skip_value();
                if (std::find(kBranchialKeys.begin(), kBranchialKeys.end(), k) !=
                    kBranchialKeys.end())
                    row.push_back({k, std::vector<uint8_t>(begin, begin + vp.position())});
            });
            rows.push_back(std::move(row));
        }
    });
    return rows;
}

}  // namespace

// "StepStatistics" asked of a session that was opened for another property is what one call
// asking for it from the start gives, under quotient exploration and under full capture.
TEST(WxfSerializationPin, ASessionServesStepStatisticsItsOpenDidNotName) {
    for (bool quotient : {true, false}) {
        auto opts = [quotient](const std::string& prop) {
            return [quotient, prop](wxf::Writer& w) {
                put_str_list_option(w, "RequestedData", {prop});
                put_str_option(w, "CanonicalizeStates", quotient ? "Full" : "None");
                put_str_option(w, "ExploreFromCanonicalStatesOnly", quotient ? "True" : "False");
            };
        };
        HostBridge host;
        const auto direct = run_rewriting_core(branch_job(3, "Evolve", 0, opts("StepStatistics"), 3), host);
        const auto opened = run_rewriting_core(branch_job(0, "Open", 0, opts("NumStates"), 3), host);
        const int64_t handle = read_int_key(opened, "Session");
        ASSERT_NE(handle, 0) << "quotient=" << quotient;
        run_rewriting_core(branch_job(3, "Step", handle, opts("NumStates"), 3), host);
        const auto queried = run_rewriting_core(branch_job(0, "Query", handle, opts("StepStatistics"), 3), host);
        run_rewriting_core(branch_job(0, "Close", handle, opts("NumStates"), 3), host);
        const auto want = value_bytes(direct, "StepStatistics");
        ASSERT_FALSE(want.empty()) << "quotient=" << quotient;
        EXPECT_EQ(value_bytes(queried, "StepStatistics"), want) << "quotient=" << quotient;
    }
}

// A Step with "From" -> {} names no frontier state and expands nothing.
TEST(WxfSerializationPin, AStepFromNoStateExpandsNothing) {
    auto opts = [](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData", {"NumStates", "NumEvents"});
    };
    auto job = [&](int64_t steps, const std::string& op, int64_t session, bool empty_from) {
        return session_envelope(kBranchSeed, kBranchLhs, kBranchRhs, steps, op, session,
                                op == "Open", {}, opts, 1, false, empty_from);
    };
    HostBridge host;
    const int64_t handle = read_int_key(run_rewriting_core(job(0, "Open", 0, false), host), "Session");
    ASSERT_NE(handle, 0);
    const auto one = run_rewriting_core(job(1, "Step", handle, false), host);
    const auto none = run_rewriting_core(job(1, "Step", handle, true), host);
    const auto two = run_rewriting_core(job(1, "Step", handle, false), host);
    run_rewriting_core(job(0, "Close", handle, false), host);
    ASSERT_GT(read_int_key(one, "NumEvents"), 0);
    EXPECT_EQ(read_int_key(none, "NumEvents"), read_int_key(one, "NumEvents"));
    EXPECT_EQ(read_int_key(none, "NumStates"), read_int_key(one, "NumStates"));
    // The frontier it left is the one the next Step expands.
    const auto direct = run_rewriting_core(
        session_envelope(kBranchSeed, kBranchLhs, kBranchRhs, 2, "Evolve", 0, true, {}, opts, 1, false),
        host);
    EXPECT_EQ(read_int_key(two, "NumEvents"), read_int_key(direct, "NumEvents"));
}

// A value of an identity option the engine does not recognise is reported as OptionSkipped,
// naming the value, and the run uses the default. A WL symbol Positional arrives with its
// context, as Global`Positional, and is one such value.
TEST(WxfSerializationPin, AnUnrecognisedIdentityValueIsReported) {
    const std::vector<std::pair<std::string, std::function<void(wxf::Writer&)>>> cases = {
        {"Global`Positional", [](wxf::Writer& w) {
             put_str_option(w, "CanonicalizeEvents", "Global`Positional"); }},
        {"Foo", [](wxf::Writer& w) {
             put_str_list_option(w, "CanonicalizeEvents", {"InputState", "Foo"}); }},
        {"Exact", [](wxf::Writer& w) { put_str_option(w, "CanonicalizeStates", "Exact"); }},
    };
    for (const auto& [value, option] : cases) {
        auto opts = [&option](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"NumEvents"});
            option(w);
        };
        HostBridge host;
        const auto out = run_rewriting_core(branch_job(2, "Evolve", 0, opts, 2), host);
        EXPECT_TRUE(reply_mentions(out, "OptionSkipped")) << value;
        EXPECT_TRUE(reply_mentions(out, "'" + value + "'")) << value;
    }
}

// "IncludeVertexCurvature" through the request: every state record has "VertexCurvature", one
// entry per vertex of the record's edges with a neighbour, under None and under Full (where the
// record lists the canonical form and the curvature is of that form). Without the option no
// record has it.
TEST(WxfSerializationPin, IncludeVertexCurvatureAddsAKeyToEachStateRecord) {
    auto run = [](const char* states, const char* include) {
        auto opts = [=](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"States"});
            put_str_option(w, "CanonicalizeStates", states);
            put_str_option(w, "IncludeVertexCurvature", include);
        };
        HostBridge host;
        return run_rewriting_core(branch_job(3, "Evolve", 0, opts, 3), host);
    };
    for (const char* mode : {"None", "Full"}) {
        const auto out = run(mode, "True");
        size_t records = 0, with_entries = 0;
        const auto states_bytes = value_bytes(out, "States");
        wxf::Parser p(states_bytes);
        p.read_association_generic([&](wxf::Parser& kp, wxf::Parser& vp) {
            kp.skip_value();
            std::vector<std::pair<int64_t, std::vector<uint32_t>>> edges;
            std::vector<std::pair<int64_t, double>> curvature;
            bool has = false;
            vp.read_association([&](const std::string& k, wxf::Parser& fp) {
                if (k == "Edges") {
                    for (const auto& e : fp.read<std::vector<std::vector<int64_t>>>()) {
                        std::vector<uint32_t> vs;
                        for (size_t i = 1; i < e.size(); ++i) vs.push_back(static_cast<uint32_t>(e[i]));
                        edges.push_back({e[0], vs});
                    }
                } else if (k == "VertexCurvature") {
                    has = true;
                    fp.read_association_generic([&](wxf::Parser& v, wxf::Parser& x) {
                        const int64_t vertex = v.read<int64_t>();
                        curvature.push_back({vertex, x.read<double>()});
                    });
                } else {
                    fp.skip_value();
                }
            });
            EXPECT_TRUE(has) << mode;
            const hg::stats::VertexCurvature want = hg::stats::vertex_curvature(edges);
            ASSERT_EQ(curvature.size(), want.size()) << mode;
            for (size_t i = 0; i < want.size(); ++i) {
                EXPECT_EQ(curvature[i].first, int64_t(want[i].first)) << mode;
                EXPECT_EQ(curvature[i].second, want[i].second) << mode;
            }
            ++records;
            with_entries += !want.empty();
        });
        EXPECT_GT(records, 1u) << mode;
        EXPECT_GT(with_entries, 0u) << mode;
    }
    EXPECT_FALSE(reply_mentions(run("None", "False"), "VertexCurvature"));
}

// "StepStatisticsMatrices" and "StepStatisticsMatrixLimit" through the request: the matrices
// asked for reach each step's record, a step above the limit warns and has none, Full states
// warn and give none, and a name that is not a matrix is reported as OptionSkipped.
TEST(WxfSerializationPin, StepStatisticsMatricesOptions) {
    auto run = [](const char* states, std::vector<std::string> names, int64_t limit) {
        auto opts = [=](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"StepStatistics"});
            put_str_option(w, "CanonicalizeStates", states);
            put_str_list_option(w, "StepStatisticsMatrices", names);
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("StepStatisticsMatrixLimit"));
            w.write(limit);
        };
        HostBridge host;
        return run_rewriting_core(branch_job(3, "Evolve", 0, opts, 4), host);
    };
    const auto all = run("None", {"StateMutualInformation", "StateEdgeOverlap"}, 0);
    EXPECT_TRUE(reply_mentions(all, "StateMatrices"));
    EXPECT_TRUE(reply_mentions(all, "StateEdgeOverlap"));
    EXPECT_FALSE(reply_mentions(all, "StateCosineSimilarity"));
    EXPECT_FALSE(reply_mentions(all, "Warnings"));
    const auto capped = run("Automatic", {"StateMutualInformation"}, 2);
    EXPECT_TRUE(reply_mentions(capped, "above StepStatisticsMatrixLimit 2;"));
    // 0 is Automatic: the matrices of these small steps fit.
    const auto automatic = run("None", {"StateMutualInformation"}, 0);
    EXPECT_TRUE(reply_mentions(automatic, "StateMatrices"));
    EXPECT_FALSE(reply_mentions(automatic, "Warnings"));
    const auto negative = run("None", {"StateMutualInformation"}, -1);
    EXPECT_TRUE(reply_mentions(negative, "StepStatisticsMatrixLimit is a byte count or Automatic"));
    const auto full = run("Full", {"StateMutualInformation"}, 0);
    EXPECT_TRUE(reply_mentions(full, "StepStatisticsMatrices needs CanonicalizeStates"));
    EXPECT_FALSE(reply_mentions(full, "StateMatrices"));
    const auto bad = run("None", {"Overlap"}, 0);
    EXPECT_TRUE(reply_mentions(bad, "OptionSkipped"));
    EXPECT_TRUE(reply_mentions(bad, "StepStatisticsMatrices 'Overlap'"));
}

// {{1,1},{1,1}} -> {{1,1},{1,1},{1,1}} at 12 steps under quotient exploration: NumEvents is
// 3,002,019,319,241,196,638 and the branchial count passes 2^63 - 1. A CountSaturated warning is
// given for a saturated count the job asked for, and it names that count.
TEST(WxfSerializationPin, ACountSaturatedWarningNamesARequestedCount) {
    const StateList seed = {{{1, 1}, {1, 1}}};
    const EdgeList lhs = {{1, 1}, {1, 1}};
    const EdgeList rhs = {{1, 1}, {1, 1}, {1, 1}};
    auto job = [&](std::vector<std::string> requested) {
        auto opts = [requested](wxf::Writer& w) {
            put_str_option(w, "CanonicalizeStates", "Full");
            put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
            put_str_list_option(w, "RequestedData", requested);
        };
        return session_envelope(seed, lhs, rhs, 12, "Evolve", 0, true, {}, opts, 3, false);
    };
    HostBridge host;
    const auto events_only = run_rewriting_core(job({"NumEvents"}), host);
    EXPECT_EQ(read_int_key(events_only, "NumEvents"), 3002019319241196638ll);
    EXPECT_LE(count_warnings(events_only), 0);
    EXPECT_FALSE(reply_mentions(events_only, "CountSaturated"));

    const auto both = run_rewriting_core(job({"NumEvents", "NumBranchialEdges"}), host);
    EXPECT_EQ(read_int_key(both, "NumBranchialEdges"), 0x7FFFFFFFFFFFFFFFll);
    EXPECT_EQ(count_warnings(both), 1);
    EXPECT_TRUE(reply_mentions(both, "NumBranchialEdges is at least 2^63 - 1"));
    EXPECT_FALSE(reply_mentions(both, "NumEvents is at least"));
}

#ifndef _WIN32
namespace {

std::string gpu_binary_path() {
    return std::string(HG_SOURCE_DIR) + "/paclet/LibraryResources/Linux-x86-64/hg_evolve_gpu";
}


// One --serve worker, driven through a pair of FIFOs: jobs in, replies out, one process for all
// four verbs. Frames are 8-byte little-endian lengths followed by the payload, matching run_serve.
struct WorkerPipes {
    std::string dir, in_path, out_path;
    pid_t pid = -1;
    int in_fd = -1, out_fd = -1;
    bool started = false;
    std::string last_error;   // the last error frame's message
};

// Each worker has its own FIFO pair, and the test's ends are close-on-exec: a second worker
// forked while the first is live would otherwise inherit the first's write end, and the first
// would never see end of input when the test closes it. A nonzero `address_space` caps the
// worker's RLIMIT_AS in bytes. A non-null `cuda_visible_devices` sets CUDA_VISIBLE_DEVICES for the
// worker.
bool worker_start(WorkerPipes& w, const std::string& exe, uint64_t address_space = 0,
                  const char* cuda_visible_devices = nullptr) {
    static int next_worker = 0;
    w.dir = std::string(HG_SOURCE_DIR) + "/.gpu_gate_fifo." + std::to_string(::getpid()) + "." +
            std::to_string(next_worker++);
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
        if (address_space) {
            const struct rlimit lim = {address_space, address_space};
            ::setrlimit(RLIMIT_AS, &lim);
        }
        if (cuda_visible_devices) ::setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices, 1);
        ::execl(exe.c_str(), exe.c_str(), "--serve", (char*)nullptr);
        ::_exit(127);
    }
    w.in_fd  = ::open(w.in_path.c_str(),  O_WRONLY | O_CLOEXEC);
    w.out_fd = ::open(w.out_path.c_str(), O_RDONLY | O_CLOEXEC);
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

// Send one job, read one reply. Empty reply means the worker reported an error for that job; its
// message is in w.last_error.
std::vector<uint8_t> worker_call(WorkerPipes& w, const std::vector<uint8_t>& job) {
    w.last_error.clear();
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
    uint64_t reply_len = 0;
    for (;;) {
        if (!read_exact_fd(w.out_fd, 8, lenbuf)) return {};
        reply_len = 0;
        for (int i = 0; i < 8; ++i) reply_len |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
        if (!(reply_len >> 63)) break;
        // A progress frame (bit 63) precedes the reply; its text is skipped.
        std::vector<uint8_t> progress;
        if (!read_exact_fd(w.out_fd, reply_len & ~(1ull << 63), progress)) return {};
    }
    if (reply_len == 0) return {};
    const bool error = (reply_len >> 62) == 1;   // bit 62 set, bit 63 clear: an error frame
    reply_len &= ~(3ull << 62);
    std::vector<uint8_t> reply;
    if (!read_exact_fd(w.out_fd, reply_len, reply)) return {};
    if (error) {
        w.last_error.assign(reply.begin(), reply.end());
        return {};
    }
    return reply;
}

// The host engine's answer. In this binary run_rewriting_core answers on the device (it is built
// with HG_GPU_BACKEND), so a gate comparing devices takes the host's answer from the CPU worker
// binary, hg_evolve, built from this tree beside hg_evolve_gpu.
std::string cpu_binary_path() {
    return std::string(HG_SOURCE_DIR) + "/paclet/LibraryResources/Linux-x86-64/hg_evolve";
}
struct CpuWorker {
    // hg_evolve is built only with BUILD_WOLFRAM_LANGUAGE_PACLET, which CI's bare runner turns off.
    static bool built() { return ::access(cpu_binary_path().c_str(), X_OK) == 0; }
    WorkerPipes w;
    bool ok = false;
    CpuWorker() { ok = worker_start(w, cpu_binary_path()); }
    ~CpuWorker() { worker_stop(w); }
    std::vector<uint8_t> operator()(const std::vector<uint8_t>& job) { return worker_call(w, job); }
};

}  // namespace

TEST(GpuBinaryGate, AQuotientSessionSteppedHoldsWhatOneEvolveReconstructs) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    for (auto opts : {quotient_session_options, quotient_counts_session_options}) {
        const auto one_shot = worker_call(w, build_input_with_op(3, "Evolve", 0, true, {}, opts, 3));
        if (one_shot.empty()) {
            worker_stop(w);
            GTEST_SKIP() << "the worker returned no result for a plain Evolve (no usable device?)";
        }
        const auto opened = worker_call(w, build_input_with_op(1, "Open", 0, true, {}, opts, 3));
        const int64_t handle = read_int_key(opened, "Session");
        ASSERT_GT(handle, 0);
        worker_call(w, build_input_with_op(1, "Step", handle, false, {}, opts, 3));
        const auto s2 = worker_call(w, build_input_with_op(1, "Step", handle, false, {}, opts, 3));
        for (const char* k : {"NumEvents", "NumCausalEdges", "NumBranchialEdges"})
            EXPECT_EQ(read_int_key(s2, k), read_int_key(one_shot, k)) << k;
        worker_call(w, build_input_with_op(0, "Close", handle, false));
    }
    worker_stop(w);
}

// One GPU worker serves four jobs in sequence, and each reply's counts equal a CPU worker's for the
// same job. Job 2 ran out of device memory after job 1 (100,736 states, CPU 137,432), and job 4
// returned [1,0,0,0] with no warning (CPU [7,6,0,8]): the device config each job started from was
// the previous job's. The worker runs under the fuzz harness's RLIMIT_AS of 16,384,000,000 bytes;
// at 48 GB job 2 fits in the stale config and the sequence passes on the defective binary.
TEST(GpuBinaryGate, AWorkerJobSequenceMatchesTheCpuJobByJob) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    if (!CpuWorker::built()) GTEST_SKIP() << "hg_evolve is not built here";
    const std::vector<std::string> counts = {"NumStates", "NumEvents", "NumCausalEdges",
                                             "NumBranchialEdges"};
    const auto counts_job = [&](const StateList& init, const EdgeList& lhs, const EdgeList& rhs,
                                const std::vector<std::string>& requested, int64_t max_states) {
        return build_input(init, lhs, rhs, 1,
                           [&](wxf::Writer& w) {
                               put_str_list_option(w, "RequestedData", requested);
                               if (max_states) {
                                   w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                                   w.write(std::string("MaxStatesPerStep"));
                                   w.write(max_states);
                               }
                           },
                           max_states ? 2 : 1);
    };
    const auto big_job = [&](const std::vector<std::string>& requested) {
        return build_input({{{1}, {1}, {1}, {1}, {1}, {1}}, {{3}, {3, 3, 5, 2}, {1}, {2}}},
                           {{1}, {1}}, {{1, 1, 1}, {1}, {1, 1}}, 5,
                           [&](wxf::Writer& w) {
                               put_str_option(w, "IncludeCanonicalHashes", "True");
                               put_str_list_option(w, "RequestedData", requested);
                               put_str_option(w, "CanonicalizeStates", "None");
                               put_str_option(w, "CanonicalizeEvents", "Automatic");
                           },
                           4);
    };
    const std::vector<std::vector<uint8_t>> jobs = {
        counts_job({{{1, 2}}}, {{1, 2}, {2, 3}, {3, 4}}, {{1, 2}}, {"NumStates"}, 0),
        big_job({"States", "Events", "CausalEdges", "BranchialEdges", "NumStates", "NumEvents",
                 "NumCausalEdges", "NumBranchialEdges"}),
        big_job(counts),
        counts_job({{{1, 1}, {1, 1}, {1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1}}},
                   {{3, 2, 2}, {2, 1}}, {{1}, {3, 2}}, counts, 6),
    };
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path(), 16000000ull * 1024)) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker cpu;
    ASSERT_TRUE(cpu.ok);
    for (size_t j = 0; j < jobs.size(); ++j) {
        const auto host = cpu(jobs[j]);
        ASSERT_FALSE(host.empty()) << "job " << j << ": " << cpu.w.last_error;
        const auto device = worker_call(w, jobs[j]);
        if (j == 0 && device.empty()) {
            worker_stop(w);
            GTEST_SKIP() << "the worker returned no result (no usable device?)";
        }
        ASSERT_FALSE(device.empty()) << "job " << j << ": " << w.last_error;
        for (const auto& k : counts)
            EXPECT_EQ(read_int_key(device, k), read_int_key(host, k)) << "job " << j << " " << k;
    }
    worker_stop(w);
}

namespace {

// A state matrix read back: a packed array of reals, or a SparseArray (Writer::write_sparse_reals)
// expanded to its rows; `sparse`, when given, says which.
std::vector<std::vector<double>> read_real_matrix(wxf::Parser& p, bool* sparse = nullptr) {
    const size_t at = p.position();
    try {
        auto m = p.read<std::vector<std::vector<double>>>();
        if (sparse) *sparse = false;
        return m;
    } catch (const wxf::WXFException&) {
        p.seek(at);
    }
    std::vector<std::vector<double>> m;
    p.read_function([&](const std::string& head, size_t n, wxf::Parser& fp) {
        EXPECT_EQ(head, "SparseArray");
        EXPECT_EQ(n, 4u);
        EXPECT_EQ(fp.read<std::string>(), "Automatic");
        const auto dims = fp.read<std::vector<int64_t>>();
        const double implicit = fp.read<double>();
        std::vector<int64_t> row_ptr;
        std::vector<std::vector<int64_t>> col;
        std::vector<double> vals;
        fp.read_function([&](const std::string&, size_t, wxf::Parser& lp) {
            EXPECT_EQ(lp.read<int64_t>(), 1);
            lp.read_function([&](const std::string&, size_t, wxf::Parser& qp) {
                row_ptr = qp.read<std::vector<int64_t>>();
                col = qp.read<std::vector<std::vector<int64_t>>>();
            });
            vals = lp.read<std::vector<double>>();
        });
        EXPECT_EQ(dims.size(), 2u);
        EXPECT_EQ(implicit, 0.0);
        m.assign(dims[0], std::vector<double>(dims[1], implicit));
        for (size_t r = 0; r + 1 < row_ptr.size(); ++r)
            for (int64_t j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
                m[r][col.at(j).at(0) - 1] = vals.at(j);
    });
    if (sparse) *sparse = true;
    return m;
}

// Per step, for each state matrix of a "StepStatistics" reply (its bytes), the multiset of rows,
// each as its diagonal cell and its cells sorted: the matrix up to a renumbering of the states.
using MatrixRows = std::multiset<std::vector<double>>;
std::vector<std::map<std::string, MatrixRows>> matrix_rows(const std::vector<uint8_t>& stats) {
    std::vector<std::map<std::string, MatrixRows>> out;
    wxf::Parser p(stats);
    p.read_function([&](const std::string&, size_t n, wxf::Parser& lp) {
        for (size_t i = 0; i < n; ++i) {
            std::map<std::string, MatrixRows> step;
            lp.read_association([&](const std::string& k, wxf::Parser& vp) {
                if (k != "StateMatrices") {
                    vp.skip_value();
                    return;
                }
                vp.read_association([&](const std::string& name, wxf::Parser& mp) {
                    if (name == "States") {
                        mp.skip_value();
                        return;
                    }
                    const auto m = read_real_matrix(mp);
                    for (size_t a = 0; a < m.size(); ++a) {
                        std::vector<double> row = m[a];
                        std::sort(row.begin(), row.end());
                        row.insert(row.begin(), m[a][a]);
                        step[name].insert(row);
                    }
                });
            });
            out.push_back(std::move(step));
        }
    });
    return out;
}

// Per step, each key of a "StepStatistics" reply (its bytes) other than "StateMatrices", and
// its value's bytes.
std::vector<std::vector<std::pair<std::string, std::vector<uint8_t>>>> keys_but_matrices(
    const std::vector<uint8_t>& stats) {
    std::vector<std::vector<std::pair<std::string, std::vector<uint8_t>>>> rows;
    wxf::Parser p(stats);
    p.read_function([&](const std::string&, size_t n, wxf::Parser& lp) {
        for (size_t i = 0; i < n; ++i) {
            std::vector<std::pair<std::string, std::vector<uint8_t>>> row;
            lp.read_association([&](const std::string& k, wxf::Parser& vp) {
                const uint8_t* begin = vp.data();
                vp.skip_value();
                if (k != "StateMatrices")
                    row.push_back({k, std::vector<uint8_t>(begin, begin + vp.position())});
            });
            rows.push_back(std::move(row));
        }
    });
    return rows;
}

}  // namespace

// The StepStatisticsBranchial metrics are admitted by one rule on both devices, so the two
// StepStatistics payloads are byte-equal across Graph/Overlap x None/Automatic/Full states x
// ExploreFromCanonicalStatesOnly. Under Full states with Automatic events the CPU dropped the
// metrics with the ExploreFromCanonicalStatesOnly warning, though the option was not set. Every
// case also asks for all six state matrices: the two devices number the states in their own
// creation order, so the matrices agree up to a renumbering of the states and every other key
// byte for byte.
TEST(GpuBinaryGate, StepStatisticsBranchialIsAdmittedAlikeOnBothDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    if (!CpuWorker::built()) GTEST_SKIP() << "hg_evolve is not built here";
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker cpu;
    ASSERT_TRUE(cpu.ok);
    const auto job = [](int64_t steps, const char* states, const char* events, bool quotient,
                        const std::vector<std::string>& branchial) {
        return build_input({{{1, 2}}}, {{1, 2}}, {{1, 2}, {2, 3}}, steps,
                           [&](wxf::Writer& o) {
                               put_str_list_option(o, "RequestedData", {"StepStatistics"});
                               put_str_list_option(o, "StepStatisticsBranchial", branchial);
                               put_str_list_option(
                                   o, "StepStatisticsMatrices",
                                   {"StateOverlap", "StateCosineSimilarity",
                                    "StateMutualInformation", "StateEdgeOverlap",
                                    "StateEdgeCosineSimilarity", "StateEdgeMutualInformation"});
                               put_str_option(o, "CanonicalizeStates", states);
                               put_str_option(o, "CanonicalizeEvents", events);
                               if (quotient)
                                   put_str_option(o, "ExploreFromCanonicalStatesOnly", "True");
                           },
                           quotient ? 6 : 5);
    };
    struct Case { int64_t steps; const char* states; bool quotient; std::vector<std::string> b; };
    std::vector<Case> cases = {{1, "Full", false, {"Graph"}}};
    for (const char* states : {"None", "Automatic", "Full"})
        for (bool quotient : {false, true})
            for (const auto& b : std::vector<std::vector<std::string>>{
                     {"Graph"}, {"Overlap"}, {"Graph", "Overlap"}})
                cases.push_back({3, states, quotient, b});
    for (const auto& c : cases) {
        const auto bytes = job(c.steps, c.states, "Automatic", c.quotient, c.b);
        const auto host = cpu(bytes);
        const auto device = worker_call(w, bytes);
        ASSERT_FALSE(host.empty()) << cpu.w.last_error;
        if (device.empty()) {
            worker_stop(w);
            GTEST_SKIP() << "the worker returned no result (no usable device?)";
        }
        const std::string what = std::string(c.states) + (c.quotient ? " quotient " : " ") +
                                 c.b.front() + (c.b.size() > 1 ? "+" + c.b.back() : "") +
                                 " steps " + std::to_string(c.steps);
        const auto host_stats = value_bytes(host, "StepStatistics");
        const auto device_stats = value_bytes(device, "StepStatistics");
        EXPECT_FALSE(host_stats.empty()) << what;
        EXPECT_EQ(keys_but_matrices(host_stats), keys_but_matrices(device_stats)) << what;
        const auto host_rows = matrix_rows(host_stats);
        EXPECT_EQ(host_rows, matrix_rows(device_stats)) << what;
        size_t with_matrices = 0;
        for (const auto& step : host_rows) with_matrices += step.size() == 6;
        // ExploreFromCanonicalStatesOnly takes effect under Full states only.
        EXPECT_EQ(with_matrices, std::string(c.states) == "Full" ? 0u : host_rows.size()) << what;
        EXPECT_EQ(reply_mentions(host, "not computed under ExploreFromCanonicalStatesOnly"),
                  reply_mentions(device, "not computed under ExploreFromCanonicalStatesOnly"))
            << what;
    }
    worker_stop(w);
}

// "IncludeVertexCurvature" gives the same curvatures from the CPU and the GPU binaries: per state
// record, the sorted list of its k_v, compared as a multiset over the records (the two devices
// number the states in their own creation order).
TEST(GpuBinaryGate, VertexCurvatureAgreesOnBothDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    if (!CpuWorker::built()) GTEST_SKIP() << "hg_evolve is not built here";
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker cpu;
    ASSERT_TRUE(cpu.ok);
    auto curvatures = [](const std::vector<uint8_t>& out) {
        std::multiset<std::vector<double>> all;
        const auto bytes = value_bytes(out, "States");
        wxf::Parser p(bytes);
        p.read_association_generic([&](wxf::Parser& kp, wxf::Parser& vp) {
            kp.skip_value();
            std::vector<double> k;
            vp.read_association([&](const std::string& key, wxf::Parser& fp) {
                if (key != "VertexCurvature") {
                    fp.skip_value();
                    return;
                }
                fp.read_association_generic([&](wxf::Parser& v, wxf::Parser& x) {
                    v.skip_value();
                    k.push_back(x.read<double>());
                });
            });
            std::sort(k.begin(), k.end());
            all.insert(k);
        });
        return all;
    };
    for (const char* states : {"None", "Full"}) {
        const auto job = build_input({{{1, 2}, {2, 3}, {3, 1}}}, {{1, 2}, {2, 3}},
                                     {{1, 2}, {2, 4}, {4, 3}, {3, 1}}, 3,
                                     [&](wxf::Writer& o) {
                                         put_str_list_option(o, "RequestedData", {"States"});
                                         put_str_option(o, "CanonicalizeStates", states);
                                         put_str_option(o, "IncludeVertexCurvature", "True");
                                     },
                                     3);
        const auto host = cpu(job);
        const auto device = worker_call(w, job);
        ASSERT_FALSE(host.empty()) << cpu.w.last_error;
        if (device.empty()) {
            worker_stop(w);
            GTEST_SKIP() << "the worker returned no result (no usable device?)";
        }
        const auto want = curvatures(host);
        EXPECT_GT(want.size(), 3u) << states;
        EXPECT_EQ(curvatures(device), want) << states;
    }
    worker_stop(w);
}

// "StatisticsGraph" on a rule of ternary hyperedges: under each rule both devices give the same
// "StepStatistics" and "VertexCurvature" bytes, and the two rules give different statistics.
TEST(GpuBinaryGate, StatisticsGraphAgreesAcrossDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    if (!CpuWorker::built()) GTEST_SKIP() << "hg_evolve is not built here";
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker cpu;
    ASSERT_TRUE(cpu.ok);
    std::vector<uint8_t> stats[2];
    int i = 0;
    for (const char* graph : {"Consecutive", "AllPairs"}) {
        const auto job = build_input({{{1, 2, 3}}}, {{1, 2, 3}}, {{1, 2, 4}, {4, 3, 2}}, 4,
                                     [&](wxf::Writer& o) {
                                         put_str_list_option(o, "RequestedData",
                                                             {"States", "StepStatistics"});
                                         put_str_option(o, "CanonicalizeStates", "Full");
                                         put_str_option(o, "IncludeVertexCurvature", "True");
                                         put_str_option(o, "StatisticsGraph", graph);
                                         put_str_list_option(
                                             o, "StepStatisticsMeasures",
                                             {"VolumeGrowth", "Geodesics", "EdgeCurvature",
                                              "Combinatorial", "Particles", "DimensionGradient"});
                                     },
                                     5);
        const auto host = cpu(job);
        const auto device = worker_call(w, job);
        ASSERT_FALSE(host.empty()) << cpu.w.last_error;
        if (device.empty()) {
            worker_stop(w);
            GTEST_SKIP() << "the worker returned no result (no usable device?)";
        }
        stats[i] = value_bytes(host, "StepStatistics");
        ASSERT_FALSE(stats[i].empty()) << graph;
        EXPECT_EQ(value_bytes(device, "StepStatistics"), stats[i]) << graph;
        EXPECT_EQ(value_bytes(device, "States"), value_bytes(host, "States")) << graph;
        ++i;
    }
    EXPECT_NE(stats[0], stats[1]);
    worker_stop(w);
}

// A quotient session opened for counts serves the relation lists a later Step asks for: the Open
// records everything a session may be asked later, the lists included.
void quotient_relation_list_options(wxf::Writer& w) {
    put_str_option(w, "CanonicalizeStates", "Full");
    put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
    put_str_list_option(w, "RequestedData", {"CausalEdges", "BranchialEdges"});
}
TEST(GpuBinaryGate, AQuotientSessionServesRelationListsItsOpenDidNotName) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    const auto one_shot = worker_call(w, branch_job(2, "Evolve", 0, quotient_relation_list_options, 3));
    if (one_shot.empty()) {
        worker_stop(w);
        GTEST_SKIP() << "the worker returned no result for a plain Evolve (no usable device?)";
    }
    const int64_t ref_causal = count_list_entries(one_shot, "CausalEdges");
    const int64_t ref_branchial = count_list_entries(one_shot, "BranchialEdges");
    ASSERT_GT(ref_branchial, 0) << "this workload branches, so one Evolve lists branchial pairs";
    const auto opened = worker_call(w, branch_job(1, "Open", 0, quotient_session_options, 3));
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_GT(handle, 0);
    const auto s1 = worker_call(w, branch_job(1, "Step", handle, quotient_relation_list_options, 3));
    ASSERT_FALSE(s1.empty());
    EXPECT_EQ(count_list_entries(s1, "CausalEdges"), ref_causal);
    EXPECT_EQ(count_list_entries(s1, "BranchialEdges"), ref_branchial);
    worker_call(w, branch_job(0, "Close", handle, quotient_session_options, 3));
    worker_stop(w);
}

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
    // A plain Evolve on the same worker between two Steps runs beside the session and leaves the
    // graph it holds alone.
    const auto between = worker_call(w, build_input_with_op(2, "Evolve"));
    ASSERT_FALSE(between.empty()) << "an Evolve while a session is open errored";
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

    // A STEERED STEP THROUGH THE WIRE. An id that is not on the frontier ERRORS the job (an
    // empty reply), and the error must not take the session with it. A member of the reported
    // frontier is ANSWERED: the worker resolves the selection against the frontier it last
    // reported, the same contract the host serves.
    const auto bad = worker_call(
        w, build_input_with_op(1, "Step", handle, /*with_rules=*/false, /*from=*/{1000000}));
    EXPECT_TRUE(bad.empty())
        << "a Step naming a state that is not on the frontier was answered rather than errored";
    const auto after = worker_call(w, build_input_with_op(0, "Query", handle, /*with_rules=*/false));
    EXPECT_FALSE(after.empty()) << "the errored Step killed the session it refused";
    EXPECT_EQ(read_int_key(after, "NumStates"), ref_states)
        << "the errored Step changed what the session holds";

    const std::vector<int64_t> fr = read_int_list_key(q, "Frontier");
    ASSERT_FALSE(fr.empty())
        << "this workload grows at every depth, so a depth-3 session must report a frontier";
    const auto steered = worker_call(
        w, build_input_with_op(1, "Step", handle, /*with_rules=*/false, /*from=*/{fr[0]}));
    EXPECT_FALSE(steered.empty())
        << "a Step naming a frontier subset was refused by the GPU worker";

    const auto c = worker_call(w, build_input_with_op(0, "Close", handle, /*with_rules=*/false));
    EXPECT_FALSE(c.empty()) << "Close errored";

    worker_stop(w);
}

#endif  // _WIN32

// =============================================================================
// Relation coherence across the identity surface
// =============================================================================
//
// One causal relation reaches the caller through three shapes -- the NumCausalEdges count, the
// CausalEdges list, and the CausalGraphStructure graph -- and they are the SAME observable, so
// they must agree in every cell of the identity surface: state mode x event mode x quotient
// exploration x transitive reduction. Measured before this gate existed (binary-growth, depth 3,
// TR on, through the paclet): quotient off + Automatic events served list 6 against graph 8;
// quotient on + None served list 6 against graph 3 where full capture serves 8; quotient on +
// Automatic served 6 against 8 -- three different answers for one relation in one reply, because
// the three shapes routed through different stores (the list read the stored skeleton with no
// reconstruction branch, the graph took the reconstruction branch whenever the identity
// machinery was on, and the two carry different event-id spaces).
//
// The quotient rows also state the exploration contract: quotient is a performance mode, never a
// semantic one, so every number it serves equals full capture's in the same cell.

namespace {

// The doc's transitive-reduction example: two LHS edges chained through a shared vertex, whose
// raw causal DAG has bypassed pairs -- the workload on which CausalTransitiveReduction must
// visibly change the graph, where a single-LHS-edge rule's raw DAG is a forest and cannot.
const StateList kSeedTR = {{{1, 1}, {1, 1}}};
const EdgeList kLhsTR = {{1, 2}, {2, 3}};
const EdgeList kRhsTR = {{1, 3}, {3, 4}, {1, 4}};

std::vector<uint8_t> build_relation_request(int64_t steps, const char* states_mode,
                                            const char* events_mode, bool quotient, bool tr,
                                            bool reducible_workload = false) {
    wxf::Writer w;
    w.write_header();
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(5);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("InitialStates"));
    w.write(reducible_workload ? kSeedTR : kSeed);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Rules"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(1);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("r0"));
    w.write_function("Rule", 2);
    w.write(reducible_workload ? kLhsTR : kLhs);
    w.write(reducible_workload ? kRhsTR : kRhs);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Steps"));
    w.write(steps);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Options"));
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(7);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("ShowProgress"));
    w.write_symbol("True");
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("RequestedData"));
    w.write(std::vector<std::string>{"CausalEdges", "NumCausalEdges"});
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("GraphProperties"));
    w.write(std::vector<std::string>{"CausalGraphStructure"});
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("CanonicalizeStates"));
    w.write_symbol(states_mode);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("CanonicalizeEvents"));
    w.write_symbol(events_mode);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("ExploreFromCanonicalStatesOnly"));
    w.write_symbol(quotient ? "True" : "False");
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("CausalTransitiveReduction"));
    w.write_symbol(tr ? "True" : "False");
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Op"));
    w.write(std::string("Evolve"));
    return w.release_data();
}

// Edges in the FIRST GraphData entry, the twin of graph_vertex_count above.
int64_t graph_edge_count(const std::vector<uint8_t>& out) {
    int64_t edge_count = -1;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "GraphData") { vp.skip_value(); return; }
        vp.read_association([&](const std::string&, wxf::Parser& gp) {
            gp.read_association([&](const std::string& field, wxf::Parser& fp) {
                if (field != "Edges") { fp.skip_value(); return; }
                fp.read_function([&](const std::string&, size_t n, wxf::Parser& ep) {
                    for (size_t i = 0; i < n; ++i) ep.skip_value();
                    edge_count = static_cast<int64_t>(n);
                });
            });
        });
    });
    return edge_count;
}

struct CausalView { int64_t num; int64_t list; int64_t graph; };

CausalView causal_view(int64_t steps, const char* states_mode, const char* events_mode,
                       bool quotient, bool tr, std::string* routing = nullptr,
                       bool reducible_workload = false) {
    HostBridge host;
    if (routing) {
        host.progress = [routing](const std::string& m) {
            const auto at = m.find("recon=");
            if (at != std::string::npos) *routing = m.substr(at);
        };
    }
    const auto out = run_rewriting_core(
        build_relation_request(steps, states_mode, events_mode, quotient, tr,
                               reducible_workload), host);
    return CausalView{read_int_key(out, "NumCausalEdges"),
                      count_list_entries(out, "CausalEdges"),
                      graph_edge_count(out)};
}

}  // namespace

TEST(RelationCoherence, CountListAndGraphAgreeInEveryIdentityCell) {
    // The invariants, per cell:
    //   count == list length     -- both are the raw relation, in every cell;
    //   graph == count           -- when event identity is None, since the graph's vertices are
    //                               then the raw events and no projection happens;
    //   graph <= count           -- otherwise: the graph is the identity-projected,
    //                               deduplicated view of the same relation;
    //   TR on <= TR off          -- CausalTransitiveReduction drops redundant edges at
    //                               registration (stored path) or on read (reconstruction), so
    //                               count, list and graph all shrink together; the strict case
    //                               is pinned by TransitiveReductionVisiblyReducesTheGraph.
    for (const char* sm : {"None", "Automatic", "Full"}) {
        for (const char* em : {"None", "Full", "Automatic"}) {
            for (bool quotient : {false, true}) {
                const CausalView off = causal_view(3, sm, em, quotient, false);
                const CausalView on  = causal_view(3, sm, em, quotient, true);
                for (const CausalView* v : {&off, &on}) {
                    EXPECT_GT(v->num, 0) << sm << "/" << em << " q=" << quotient;
                    EXPECT_EQ(v->num, v->list)
                        << "count vs list, states=" << sm << " events=" << em
                        << " quotient=" << quotient;
                }
                EXPECT_LE(on.num, off.num)
                    << "reduction only removes, states=" << sm << " events=" << em
                    << " quotient=" << quotient;
                if (std::string(em) == "None") {
                    EXPECT_EQ(off.graph, off.num) << sm << " q=" << quotient << " (tr off)";
                } else {
                    EXPECT_LE(off.graph, off.num) << sm << "/" << em << " q=" << quotient;
                    EXPECT_GT(off.graph, 0) << sm << "/" << em << " q=" << quotient;
                }
                EXPECT_LE(on.graph, off.graph)
                    << "reduction only removes, states=" << sm << " events=" << em
                    << " quotient=" << quotient;
            }
        }
    }
}

// The documented promise of CausalTransitiveReduction, held on a workload whose raw causal DAG
// has bypassed pairs: the relation is strictly smaller with the reduction on, and count, list
// and graph move together because they are one relation.
TEST(RelationCoherence, TransitiveReductionVisiblyReducesTheGraph) {
    for (const char* sm : {"None", "Full"}) {
        for (bool quotient : {false, true}) {
            const CausalView off = causal_view(3, sm, "None", quotient, false, nullptr, true);
            const CausalView on  = causal_view(3, sm, "None", quotient, true,  nullptr, true);
            EXPECT_LT(on.num, off.num)   << sm << " q=" << quotient;
            EXPECT_EQ(on.num, on.list)   << sm << " q=" << quotient;
            EXPECT_EQ(on.num, on.graph)  << sm << " q=" << quotient;
            EXPECT_EQ(off.num, off.list) << sm << " q=" << quotient;
            EXPECT_EQ(off.num, off.graph) << sm << " q=" << quotient;
        }
    }
}

TEST(RelationCoherence, QuotientServesFullCapturesNumbers) {
    // The quotient contract holds where the quotient runs: it needs Full state
    // canonicalization, and the engine says so in a warning otherwise.
    for (const char* em : {"None", "Full", "Automatic"}) {
        for (bool tr : {false, true}) {
            const CausalView full = causal_view(3, "Full", em, false, tr);
            const CausalView quot = causal_view(3, "Full", em, true, tr);
            EXPECT_EQ(full.num, quot.num)
                << "events=" << em << " tr=" << tr << " (count)";
            EXPECT_EQ(full.list, quot.list)
                << "events=" << em << " tr=" << tr << " (list)";
            EXPECT_EQ(full.graph, quot.graph)
                << "events=" << em << " tr=" << tr << " (graph)";
        }
    }
}

// The whole surface, printed. Not an assertion: the two gates above state the invariants; this
// exists so a failure reads as a table rather than as thirty scattered EXPECT lines.
TEST(RelationCoherence, PrintSurface) {
    std::printf("%-10s %-10s %-9s %-6s | %6s %6s %6s\n",
                "states", "events", "quotient", "tr", "num", "list", "graph");
    for (const char* sm : {"None", "Automatic", "Full"})
        for (const char* em : {"None", "Full", "Automatic"})
            for (bool q : {false, true})
                for (bool tr : {false, true}) {
                    std::string routing;
                    const CausalView v = causal_view(3, sm, em, q, tr, &routing);
                    std::printf("%-10s %-10s %-9s %-6s | %6lld %6lld %6lld | %s\n",
                                sm, em, q ? "on" : "off", tr ? "on" : "off",
                                (long long)v.num, (long long)v.list, (long long)v.graph,
                                routing.c_str());
                }
}

// The states graph has one edge per event NumEvents reports, under quotient exploration as under
// full capture, in every event identity mode. Under an identity mode the one-shot graph repeated
// an edge for every raw event sharing an identity: 74 edges against 16 events. The kernel asks for a states graph with an empty RequestedData, which left the
// reconstruction off, so under quotient exploration the graph was built over the explored
// representatives' events: 52 edges against 126 events on the WPP rule at depth 4.
TEST(WxfSerializationPin, StatesGraphEdgesAreTheEventsTheCountReports) {
  for (const char* events_mode : {"None", "Full", "Automatic"}) {
    for (bool quotient : {false, true}) {
        auto request = [&](bool graph) {
            return build_input(kBranchSeed, kBranchLhs, kBranchRhs, 4, [&](wxf::Writer& w) {
                if (graph) {
                    put_str_list_option(w, "RequestedData", {});
                    put_str_list_option(w, "GraphProperties", {"StatesGraph"});
                } else {
                    put_str_list_option(w, "RequestedData", {"NumEvents"});
                }
                put_str_option(w, "CanonicalizeStates", "Full");
                put_str_option(w, "CanonicalizeEvents", events_mode);
                if (quotient) put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
            }, (graph ? 4 : 3) + (quotient ? 1 : 0));
        };
        HostBridge host;
        const auto graph_out = run_rewriting_core(request(true), host);
        const auto count_out = run_rewriting_core(request(false), host);
        const int64_t num_events = read_int_key(count_out, "NumEvents");
        ASSERT_GT(num_events, 0);
        EXPECT_EQ(graph_edge_count(graph_out), num_events)
            << "events=" << events_mode << " quotient=" << quotient
            << ": the states graph and NumEvents describe different event sets";
    }
  }
}

// Under Full state canonicalization "States" holds one record per class, keyed by the class's
// canonical representative, and every event names its endpoints' classes by those keys in
// CanonicalInputState / CanonicalOutputState. The representative is the state that won the
// class's dedup claim, which on parallel workers need not be the lowest raw id, so the run is
// repeated.
namespace {
// The integer `field` of every record of the association `section` ("States" or "Events").
std::vector<int64_t> record_field_values(const std::vector<uint8_t>& out, const std::string& section,
                                         const std::string& field) {
    std::vector<int64_t> values;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != section) { vp.skip_value(); return; }
        vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& rp) {
            kp.skip_value();
            rp.read_association([&](const std::string& f, wxf::Parser& fp) {
                if (f == field) values.push_back(fp.read<int64_t>());
                else fp.skip_value();
            });
        });
    });
    return values;
}
}  // namespace

TEST(WxfSerializationPin, EventEndpointClassesAreStatesKeys) {
    auto field_values = record_field_values;
    for (bool quotient : {false, true}) {
        for (int rep = 0; rep < 10; ++rep) {
            HostBridge host;
            auto in = build_input(kBranchSeed, kBranchLhs, kBranchRhs, 4, [&](wxf::Writer& w) {
                put_str_list_option(w, "RequestedData", {"States", "Events"});
                put_str_option(w, "CanonicalizeStates", "Full");
                if (quotient) put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
            }, quotient ? 3 : 2);
            const auto out = run_rewriting_core(in, host);
            const auto ids = field_values(out, "States", "Id");
            const auto cids = field_values(out, "States", "CanonicalId");
            ASSERT_FALSE(ids.empty());
            EXPECT_EQ(ids, cids) << "quotient=" << quotient << ": a States record is not its class's representative";
            const std::set<int64_t> keys(ids.begin(), ids.end());
            for (const char* f : {"CanonicalInputState", "CanonicalOutputState"})
                for (int64_t s : field_values(out, "Events", f))
                    EXPECT_TRUE(keys.count(s)) << "quotient=" << quotient << ": event " << f << " "
                                               << s << " is not a States key";
        }
    }
}

// Under quotient exploration "Events" lists every rule application, as under full exploration:
// as many records as full exploration gives, grouped by "CanonicalId" into the NumEvents events,
// with endpoints that are States keys.
TEST(WxfSerializationPin, QuotientEventsAreTheApplicationsTheCountReports) {
    for (const char* events_mode : {"None", "Full", "Automatic"}) {
        auto run = [&](bool quotient) {
            HostBridge host;
            return run_rewriting_core(build_input(kBranchSeed, kBranchLhs, kBranchRhs, 4, [&](wxf::Writer& w) {
                put_str_list_option(w, "RequestedData", {"States", "Events", "NumEvents"});
                put_str_option(w, "CanonicalizeStates", "Full");
                put_str_option(w, "CanonicalizeEvents", events_mode);
                if (quotient) put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
            }, quotient ? 4 : 3), host);
        };
        const auto full = run(false), quot = run(true);
        const auto full_ids = record_field_values(full, "Events", "Id");
        const auto ids = record_field_values(quot, "Events", "Id");
        ASSERT_FALSE(full_ids.empty()) << events_mode;
        EXPECT_EQ(ids.size(), full_ids.size())
            << events_mode << ": quotient exploration lists " << ids.size()
            << " applications, full exploration " << full_ids.size();
        const auto cids = record_field_values(quot, "Events", "CanonicalId");
        EXPECT_EQ(static_cast<int64_t>(std::set<int64_t>(cids.begin(), cids.end()).size()),
                  read_int_key(quot, "NumEvents")) << events_mode;
        const auto keys_v = record_field_values(quot, "States", "Id");
        const std::set<int64_t> keys(keys_v.begin(), keys_v.end());
        for (const char* f : {"CanonicalInputState", "CanonicalOutputState"})
            for (int64_t st : record_field_values(quot, "Events", f))
                EXPECT_TRUE(keys.count(st)) << events_mode << ": " << f << " " << st << " is not a States key";
    }
}

// Isomorphic initial states are separate initial states, and quotient exploration gives the
// counts and statistics full exploration gives for them.
TEST(WxfSerializationPin, IsomorphicInitialStatesUnderQuotientCountLikeFullExploration) {
    const StateList roots = {{{1, 2}, {1, 3}}, {{4, 5}, {4, 6}}, {{7, 8}, {7, 9}}};
    auto run = [&](bool quotient) {
        HostBridge host;
        return run_rewriting_core(build_input(roots, kBranchLhs, kBranchRhs, 3, [&](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData",
                {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges", "StepStatistics"});
            put_str_option(w, "CanonicalizeStates", "Full");
            if (quotient) put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
        }, quotient ? 3 : 2), host);
    };
    const auto full = run(false), quot = run(true);
    for (const char* k : {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"})
        EXPECT_EQ(read_int_key(quot, k), read_int_key(full, k)) << k;
    EXPECT_EQ(value_bytes(quot, "StepStatistics"), value_bytes(full, "StepStatistics"));
}

// A session's frontier names states by the ids "States" uses. Under Full both are the class's
// canonical representative; "States" once emitted the first raw state of each class, which on
// parallel workers need not be the representative, so the run is repeated.
TEST(WxfSerializationPin, SessionFrontierIdsAreStatesKeys) {
    auto job = [](int64_t steps, const std::string& op, int64_t session) {
        return branch_job(steps, op, session, [](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"States"});
            put_str_option(w, "CanonicalizeStates", "Full");
        }, 2);
    };
    auto states_keys = [](const std::vector<uint8_t>& out) {
        std::set<int64_t> keys;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "States") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& rp) {
                keys.insert(kp.read<int64_t>());
                rp.skip_value();
            });
        });
        return keys;
    };
    for (int rep = 0; rep < 20; ++rep) {
        HostBridge host;
        const auto opened = run_rewriting_core(job(0, "Open", 0), host);
        const int64_t handle = read_int_key(opened, "Session");
        ASSERT_NE(handle, 0);
        const auto stepped = run_rewriting_core(job(4, "Step", handle), host);
        const auto keys = states_keys(stepped);
        const auto frontier = read_int_list_key(stepped, "Frontier");
        ASSERT_FALSE(frontier.empty());
        for (int64_t f : frontier)
            EXPECT_TRUE(keys.count(f)) << "run " << rep << ": frontier id " << f
                                       << " is not a States key";
        run_rewriting_core(job(0, "Close", handle), host);
    }
}

// The same invariant after every one-step Step, with the second step steered by "From" to the
// lowest frontier id, under Full with and without quotient exploration and under Automatic.
TEST(WxfSerializationPin, SteeredSessionFrontierIdsAreStatesKeys) {
    struct Mode { const char* states; bool quotient; };
    for (const Mode mode : {Mode{"Full", false}, Mode{"Full", true}, Mode{"Automatic", false}}) {
        auto job = [&](int64_t steps, const std::string& op, int64_t session,
                       const std::vector<int64_t>& from) {
            return session_envelope(kBranchSeed, kBranchLhs, kBranchRhs, steps, op, session,
                                    op == "Open", from, [&](wxf::Writer& w) {
                put_str_list_option(w, "RequestedData", {"States"});
                put_str_option(w, "CanonicalizeStates", mode.states);
                if (mode.quotient) put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
            }, mode.quotient ? 3 : 2, false);
        };
        auto states_keys = [](const std::vector<uint8_t>& out) {
            std::set<int64_t> keys;
            wxf::Parser parser(out);
            parser.skip_header();
            parser.read_association([&](const std::string& k, wxf::Parser& vp) {
                if (k != "States") { vp.skip_value(); return; }
                vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& rp) {
                    keys.insert(kp.read<int64_t>());
                    rp.skip_value();
                });
            });
            return keys;
        };
        for (int rep = 0; rep < 10; ++rep) {
            HostBridge host;
            const auto opened = run_rewriting_core(job(0, "Open", 0, {}), host);
            const int64_t handle = read_int_key(opened, "Session");
            ASSERT_NE(handle, 0) << mode.states;
            std::vector<int64_t> frontier = read_int_list_key(opened, "Frontier");
            for (int step = 1; step <= 4; ++step) {
                std::vector<int64_t> from;
                if (step == 2 && !frontier.empty())
                    from = {*std::min_element(frontier.begin(), frontier.end())};
                const auto stepped = run_rewriting_core(job(1, "Step", handle, from), host);
                const auto keys = states_keys(stepped);
                frontier = read_int_list_key(stepped, "Frontier");
                ASSERT_FALSE(frontier.empty()) << mode.states << " step " << step;
                for (int64_t f : frontier)
                    EXPECT_TRUE(keys.count(f))
                        << mode.states << (mode.quotient ? " quotient" : "") << " run " << rep
                        << " step " << step << ": frontier id " << f << " is not a States key";
            }
            run_rewriting_core(job(0, "Close", handle, {}), host);
        }
    }
}

#ifndef _WIN32
// Positional event identity has no device mode, so the GPU binary runs such a job on its CPU
// engine and says so; the counts are the CPU engine's. A session opened that way is served by the
// CPU engine for every later verb.
// A CanonicalizeEvents key list other than the Full and Automatic presets runs on the CPU engine
// in the GPU binary, and gives the CPU's count.
TEST(GpuBinaryGate, ACustomEventKeySetRunsOnTheCpuEngine) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    auto options = [](bool quotient) {
        return [quotient](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"NumStates", "NumEvents"});
            put_str_option(w, "CanonicalizeStates", "Full");
            put_str_list_option(w, "CanonicalizeEvents", {"ConsumedEdges", "ProducedEdges"});
            if (quotient) put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
        };
    };
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    for (bool quotient : {false, true}) {
        const std::size_t n = quotient ? 4 : 3;
        HostBridge host;
        const auto cpu = run_rewriting_core(branch_job(3, "Evolve", 0, options(quotient), n), host);
        const auto gpu = worker_call(w, branch_job(3, "Evolve", 0, options(quotient), n));
        ASSERT_FALSE(gpu.empty()) << "quotient=" << quotient;
        EXPECT_EQ(read_int_key(gpu, "NumEvents"), read_int_key(cpu, "NumEvents"))
            << "quotient=" << quotient;
        EXPECT_TRUE(reply_mentions(gpu, "A CanonicalizeEvents key list runs on the CPU engine"))
            << "quotient=" << quotient;
    }
    worker_stop(w);
}

TEST(GpuBinaryGate, PositionalRunsOnTheCpuEngine) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    auto options = [](bool quotient) {
        return [quotient](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"NumStates", "NumEvents"});
            put_str_option(w, "CanonicalizeStates", "Full");
            put_str_option(w, "CanonicalizeEvents", "Positional");
            if (quotient) put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
        };
    };
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    const std::string note = "Positional event identity runs on the CPU engine";
    for (bool quotient : {false, true}) {
        const std::size_t n = quotient ? 4 : 3;
        HostBridge host;
        const auto cpu = run_rewriting_core(branch_job(3, "Evolve", 0, options(quotient), n), host);
        const auto gpu = worker_call(w, branch_job(3, "Evolve", 0, options(quotient), n));
        ASSERT_FALSE(gpu.empty()) << "quotient=" << quotient;
        EXPECT_EQ(read_int_key(gpu, "NumEvents"), read_int_key(cpu, "NumEvents")) << "quotient=" << quotient;
        EXPECT_EQ(read_int_key(gpu, "NumStates"), read_int_key(cpu, "NumStates")) << "quotient=" << quotient;
        EXPECT_TRUE(reply_mentions(gpu, note)) << "quotient=" << quotient << ": no warning";
    }
    const auto opened = worker_call(w, branch_job(0, "Open", 0, options(false), 3));
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_NE(handle, 0);
    const auto stepped = worker_call(w, branch_job(3, "Step", handle, options(false), 3));
    HostBridge host;
    const auto one = run_rewriting_core(branch_job(3, "Evolve", 0, options(false), 3), host);
    EXPECT_EQ(read_int_key(stepped, "NumEvents"), read_int_key(one, "NumEvents"))
        << "a Positional session stepped 3 does not hold one evolve of 3";
    // One session per worker across both engines, and one handle sequence: a device Open is
    // refused while the CPU engine holds a session, and after Close it gets a new handle.
    auto plain = [](wxf::Writer& wr) { put_str_list_option(wr, "RequestedData", {"NumStates"}); };
    EXPECT_TRUE(worker_call(w, branch_job(0, "Open", 0, plain, 1)).empty())
        << "a device Open was accepted while the CPU engine held a session";
    EXPECT_FALSE(worker_call(w, branch_job(0, "Close", handle, options(false), 3)).empty());
    const auto device = worker_call(w, branch_job(0, "Open", 0, plain, 1));
    ASSERT_FALSE(device.empty());
    EXPECT_NE(read_int_key(device, "Session"), handle)
        << "the device session took the handle the CPU session had";
    EXPECT_FALSE(
        worker_call(w, branch_job(0, "Close", read_int_key(device, "Session"), plain, 1)).empty());
    worker_stop(w);
}
// A state record's edges carry the ids the event records name, on both devices: every event's
// ProducedEdges are edge ids of its output state (CanonicalizeStates None). The GPU once numbered
// a state's edges 0, 1, 2, ... and the ids agreed only for the initial state.
TEST(GpuBinaryGate, StateEdgeIdsAreTheIdsEventsName) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    auto opts = [](wxf::Writer& w) { put_str_list_option(w, "RequestedData", {"States", "Events"}); };
    // state id -> edge ids, and (output state, produced edges) per event.
    auto check = [](const std::vector<uint8_t>& out, const char* device) {
        std::map<int64_t, std::set<int64_t>> state_edges;
        std::vector<std::pair<int64_t, std::vector<int64_t>>> produced;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "States" && k != "Events") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& rp) {
                const int64_t key = kp.read<int64_t>();
                int64_t out_state = -1;
                std::vector<int64_t> prod;
                rp.read_association([&](const std::string& f, wxf::Parser& fp) {
                    if (k == "States" && f == "Edges") {
                        fp.read_function([&](const std::string&, size_t n, wxf::Parser& ep) {
                            for (size_t i = 0; i < n; ++i)
                                ep.read_function([&](const std::string&, size_t m, wxf::Parser& xp) {
                                    state_edges[key].insert(xp.read<int64_t>());
                                    for (size_t j = 1; j < m; ++j) xp.skip_value();
                                });
                        });
                    } else if (k == "Events" && f == "OutputState") {
                        out_state = fp.read<int64_t>();
                    } else if (k == "Events" && f == "ProducedEdges") {
                        prod = fp.read<std::vector<int64_t>>();
                    } else {
                        fp.skip_value();
                    }
                });
                if (k == "Events") produced.emplace_back(out_state, prod);
            });
        });
        ASSERT_FALSE(produced.empty()) << device;
        for (const auto& [st, prod] : produced)
            for (int64_t e : prod)
                EXPECT_TRUE(state_edges[st].count(e))
                    << device << ": produced edge " << e << " is not an edge id of state " << st;
    };
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker cpu;
    ASSERT_TRUE(cpu.ok) << "could not start hg_evolve --serve";
    check(cpu(branch_job(2, "Evolve", 0, opts, 1)), "CPU");
    check(worker_call(w, branch_job(2, "Evolve", 0, opts, 1)), "GPU");
    worker_stop(w);
}
// A job the engine refuses comes back as an error frame carrying the engine's message, and the
// worker serves the next job. The message went only to the worker's stderr, which no client reads.
TEST(Session, AWorkerReportsARefusedJobWithItsMessage) {
    if (!CpuWorker::built()) GTEST_SKIP() << "hg_evolve is not built here";
    CpuWorker w;
    ASSERT_TRUE(w.ok) << "could not start hg_evolve --serve";
    EXPECT_TRUE(w(build_input_with_op(1, "Step", 12345, false)).empty());
    EXPECT_NE(w.w.last_error.find("12345"), std::string::npos)
        << "error frame: '" << w.w.last_error << "'";
    EXPECT_FALSE(w(build_input_with_op(1, "Evolve")).empty());
    EXPECT_TRUE(w.w.last_error.empty());
}

// Two worker processes issue different session handles, so a caller holding sessions in the CPU
// and GPU workers, or in a worker and its restart, never addresses one session by the other's
// handle. Each process counted from 1, so both Opens returned 1.
TEST(Session, TwoWorkerProcessesIssueDifferentHandles) {
    if (!CpuWorker::built()) GTEST_SKIP() << "hg_evolve is not built here";
    CpuWorker a, b;
    ASSERT_TRUE(a.ok && b.ok) << "could not start hg_evolve --serve";
    const int64_t ha = read_int_key(a(build_input_with_op(1, "Open")), "Session");
    const int64_t hb = read_int_key(b(build_input_with_op(1, "Open")), "Session");
    ASSERT_GT(ha, 0);
    ASSERT_GT(hb, 0);
    EXPECT_NE(ha, hb);
    // A handle from one worker is refused by the other, and the refusal says why.
    EXPECT_TRUE(b(build_input_with_op(1, "Step", ha, false)).empty());
    EXPECT_NE(b.w.last_error.find("is not this worker's live session"), std::string::npos)
        << "error frame: '" << b.w.last_error << "'";
}

// "ContentStateId" is the lowest id among the listed states with the same edge list, on both
// devices and in every "CanonicalizeStates" mode, so it is always a "States" key. The GPU once
// gave each state its own id, and the CPU under Full could name a state "States" does not list.
TEST(GpuBinaryGate, ContentStateIdIsTheLowestListedStateOfEqualContent) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    // key -> (ContentStateId, vertex lists in edge-id order)
    using Content = std::vector<std::vector<int64_t>>;
    auto read = [](const std::vector<uint8_t>& out) {
        std::map<int64_t, std::pair<int64_t, Content>> states;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "States") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& rp) {
                const int64_t key = kp.read<int64_t>();
                int64_t cid = -1;
                std::map<int64_t, std::vector<int64_t>> by_edge;
                rp.read_association([&](const std::string& f, wxf::Parser& fp) {
                    if (f == "ContentStateId") {
                        cid = fp.read<int64_t>();
                    } else if (f == "Edges") {
                        fp.read_function([&](const std::string&, size_t n, wxf::Parser& ep) {
                            for (size_t i = 0; i < n; ++i)
                                ep.read_function([&](const std::string&, size_t m, wxf::Parser& xp) {
                                    const int64_t id = xp.read<int64_t>();
                                    for (size_t j = 1; j < m; ++j)
                                        by_edge[id].push_back(xp.read<int64_t>());
                                });
                        });
                    } else {
                        fp.skip_value();
                    }
                });
                Content c;
                for (auto& [id, vs] : by_edge) c.push_back(vs);
                states[key] = {cid, c};
            });
        });
        return states;
    };
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    for (const char* mode : {"None", "Automatic", "Full"}) {
        auto opts = [mode](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"States"});
            put_str_option(ww, "CanonicalizeStates", mode);
        };
        CpuWorker host;
        ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
        const auto cpu = read(host(branch_job(3, "Evolve", 0, opts, 2)));
        const auto gpu = read(worker_call(w, branch_job(3, "Evolve", 0, opts, 2)));
        for (const auto* side : {&cpu, &gpu}) {
            const char* device = side == &cpu ? "CPU" : "GPU";
            ASSERT_FALSE(side->empty()) << device << " " << mode;
            std::map<Content, int64_t> lowest;
            for (const auto& [key, rec] : *side) {
                auto [it, fresh] = lowest.emplace(rec.second, key);
                if (!fresh && key < it->second) it->second = key;
            }
            size_t shared = 0;
            for (const auto& [key, rec] : *side) {
                EXPECT_TRUE(side->count(rec.first))
                    << device << " " << mode << ": ContentStateId " << rec.first
                    << " of state " << key << " is not a States key";
                EXPECT_EQ(rec.first, lowest.at(rec.second)) << device << " " << mode << " state " << key;
                if (rec.first != key) ++shared;
            }
            if (std::string(mode) == "None")
                EXPECT_GT(shared, 0u) << device << ": no two states share an edge list, so the check is vacuous";
        }
    }
    worker_stop(w);
}
// "Events" with genesis events shown lists every application and every genesis event once, on
// both devices: the GPU numbers its genesis events above every application id, as the host does.
TEST(GpuBinaryGate, GenesisEventsAreListedOnceBesideEveryApplication) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    // The "Events" keys, in reply order.
    auto event_keys = [](const std::vector<uint8_t>& out) {
        std::vector<int64_t> keys;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "Events") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& rp) {
                keys.push_back(kp.read<int64_t>());
                rp.skip_value();
            });
        });
        return keys;
    };
    for (bool quotient : {false, true}) {
        auto opts = [quotient](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"Events", "NumEvents"});
            put_str_option(ww, "CanonicalizeStates", "Full");
            put_str_option(ww, "ExploreFromCanonicalStatesOnly", quotient ? "True" : "False");
            put_str_option(ww, "ShowGenesisEvents", "True");
        };
        const auto cpu = host(branch_job(3, "Evolve", 0, opts, 4));
        const auto gpu = worker_call(w, branch_job(3, "Evolve", 0, opts, 4));
        ASSERT_FALSE(gpu.empty()) << "quotient=" << quotient;
        const auto c = event_keys(cpu), g = event_keys(gpu);
        const std::set<int64_t> gs(g.begin(), g.end());
        EXPECT_EQ(gs.size(), g.size()) << "quotient=" << quotient << ": an \"Events\" key repeats";
        EXPECT_EQ(g.size(), c.size()) << "quotient=" << quotient;
    }
    worker_stop(w);
}

// The causal graphs with genesis events shown draw each genesis event and the causal edges it
// starts, on both devices: the GPU builds its genesis events from the result, and its graphs take
// them as the host's do.
TEST(GpuBinaryGate, CausalGraphsDrawGenesisEventsOnBothDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    for (const char* graph : {"CausalGraph", "EvolutionCausalGraph"}) {
        for (bool quotient : {false, true}) {
            auto opts = [graph, quotient](wxf::Writer& ww) {
                put_str_list_option(ww, "GraphProperties", {graph});
                put_str_option(ww, "CanonicalizeStates", "Full");
                put_str_option(ww, "ExploreFromCanonicalStatesOnly", quotient ? "True" : "False");
                put_str_option(ww, "ShowGenesisEvents", "True");
            };
            const auto cpu = host(branch_job(3, "Evolve", 0, opts, 4));
            const auto gpu = worker_call(w, branch_job(3, "Evolve", 0, opts, 4));
            ASSERT_FALSE(gpu.empty()) << graph << " quotient=" << quotient;
            EXPECT_EQ(graph_vertex_count(gpu), graph_vertex_count(cpu))
                << graph << " quotient=" << quotient;
            EXPECT_EQ(graph_edge_count(gpu), graph_edge_count(cpu))
                << graph << " quotient=" << quotient;
        }
    }
    worker_stop(w);
}

// ShowGenesisEvents under Full states (docs/SPEC.md §5.2) gives the host's counts on the GPU, on
// both routes and with and without CausalTransitiveReduction: NumStates, NumEvents (one genesis
// event per initial state), NumCausalEdges and the causal list, one "Events" record with
// "RuleIndex" 65535 per initial state, and a "States" list of NumStates records. The cases of
// WxfSerializationPin.GenesisEventsAgreeAcrossRoutes.
TEST(GpuBinaryGate, GenesisEventsGiveTheHostsCounts) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    struct Case { StateList init; EdgeList lhs, rhs; int64_t steps; };
    const Case cases[] = {
        {{{{1, 2}}}, {{1, 2}}, {{1, 3}, {3, 2}}, 2},
        {{{{1, 2}}}, {{1, 2}}, {{1, 2}, {2, 3}}, 2},
        {{{{1, 2}, {2, 3}}}, {{1, 2}, {2, 3}}, {{1, 3}, {3, 4}, {4, 2}}, 2},
        {{{{1, 1}, {1, 1}}}, {{1, 2}}, {{1, 2}, {2, 3}}, 3},
        {{{{1, 2}}, {{1, 1}, {2, 1}}}, {{1, 2}}, {{1, 3}, {3, 2}}, 2},
    };
    auto genesis_records = [](const std::vector<uint8_t>& out) {
        int64_t n = 0;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "Events") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
                kp.skip_value();
                valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                    if (fk == "RuleIndex") n += fvp.read<int64_t>() == 65535 ? 1 : 0;
                    else fvp.skip_value();
                });
            });
        });
        return n;
    };
    for (const Case& c : cases) {
        for (const char* tr : {"True", "False"}) {
            for (const char* quotient : {"False", "True"}) {
                auto input = build_input(c.init, c.lhs, c.rhs, c.steps,
                                         [&](wxf::Writer& ww) {
                                             put_str_option(ww, "CanonicalizeStates", "Full");
                                             put_str_option(ww, "ShowGenesisEvents", "True");
                                             put_str_option(ww, "CausalTransitiveReduction", tr);
                                             put_str_option(ww, "ExploreFromCanonicalStatesOnly",
                                                            quotient);
                                             put_str_list_option(ww, "RequestedData",
                                                 {"States", "Events", "NumStates", "NumEvents",
                                                  "NumCausalEdges", "CausalEdges"});
                                         },
                                         5);
                const auto cpu = host(input);
                const auto gpu = worker_call(w, input);
                ASSERT_FALSE(gpu.empty());
                const std::string at = "steps " + std::to_string(c.steps) + " TR " + tr +
                                       " quotient " + quotient;
                for (const char* key : {"NumStates", "NumEvents", "NumCausalEdges"})
                    EXPECT_EQ(read_int_key(gpu, key), read_int_key(cpu, key)) << at << " " << key;
                EXPECT_EQ(count_list_entries(gpu, "CausalEdges"),
                          count_list_entries(cpu, "CausalEdges")) << at;
                EXPECT_EQ(genesis_records(gpu), static_cast<int64_t>(c.init.size())) << at;
                EXPECT_EQ(genesis_records(cpu), static_cast<int64_t>(c.init.size())) << at;
                EXPECT_EQ(count_assoc_entries(gpu, "States"), read_int_key(gpu, "NumStates")) << at;
            }
        }
    }
    worker_stop(w);
}

// Under Full states a state record's Step is its class's on the GPU as on the host: the class's
// explore depth under quotient exploration, its least step under full capture. Two rules whose
// classes are reached at several depths, 4 steps (the case of WxfSerializationPin.
// QuotientStateStepIsTheClassShortestDepth): 6 GPU runs of each route give one (CanonicalHash,
// Step) set, equal to the host's, and NumEvents under CanonicalizeEvents Automatic, whose
// identity reads the event's Step, equals the host's on both routes.
TEST(GpuBinaryGate, StateStepIsTheClassStepOnBothDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    auto job = [](bool ecso, const char* events) {
        wxf::Writer ww;
        ww.write_header();
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        ww.write_varint(4);
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string("InitialStates"));
        ww.write(StateList{{{3, 1}, {3, 2, 1}, {3, 2}, {1, 1}, {3, 3}}});
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string("Rules"));
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        ww.write_varint(2);
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string("r0"));
        ww.write_function("Rule", 2);
        ww.write(EdgeList{{4, 3}, {1, 2}});
        ww.write(EdgeList{{5, 5}, {2, 3}});
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string("r1"));
        ww.write_function("Rule", 2);
        ww.write(EdgeList{{3, 2, 3}, {2, 2}});
        ww.write(EdgeList{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}});
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string("Steps"));
        ww.write(int64_t{4});
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string("Options"));
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        ww.write_varint(5);
        put_str_option(ww, "CanonicalizeStates", "Full");
        put_str_option(ww, "CanonicalizeEvents", events);
        put_str_option(ww, "IncludeCanonicalHashes", "True");
        put_str_option(ww, "ExploreFromCanonicalStatesOnly", ecso ? "True" : "False");
        put_str_list_option(ww, "RequestedData", {"States", "NumEvents"});
        return ww.release_data();
    };
    auto hash_steps = [](const std::vector<uint8_t>& out) {
        std::set<std::pair<int64_t, int64_t>> hash_step;
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "States") { vp.skip_value(); return; }
            vp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& valp) {
                kp.skip_value();
                int64_t h = 0, step = -1;
                valp.read_association([&](const std::string& fk, wxf::Parser& fvp) {
                    if (fk == "CanonicalHash") h = fvp.read<int64_t>();
                    else if (fk == "Step") step = fvp.read<int64_t>();
                    else fvp.skip_value();
                });
                hash_step.insert({h, step});
            });
        });
        return hash_step;
    };
    for (bool ecso : {false, true}) {
        const auto cpu = hash_steps(host(job(ecso, "None")));
        ASSERT_FALSE(cpu.empty());
        std::set<std::set<std::pair<int64_t, int64_t>>> seen;
        for (int rep = 0; rep < 6; ++rep) {
            const auto gpu = worker_call(w, job(ecso, "None"));
            ASSERT_FALSE(gpu.empty()) << "quotient " << ecso;
            seen.insert(hash_steps(gpu));
        }
        EXPECT_EQ(seen.size(), 1u) << "quotient " << ecso;
        EXPECT_EQ(*seen.begin(), cpu) << "quotient " << ecso;
        EXPECT_EQ(read_int_key(worker_call(w, job(ecso, "Automatic")), "NumEvents"),
                  read_int_key(host(job(ecso, "Automatic")), "NumEvents")) << "quotient " << ecso;
    }
    worker_stop(w);
}

// A cap past 32 bits is no cap on either device: the GPU saturates it rather than keeping its low
// bits, so MaxStatesPerStep -> 2^32 + 1 returns what no cap returns.
TEST(GpuBinaryGate, ACapPast32BitsIsNoCap) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    for (const char* key : {"MaxStatesPerStep", "MaxSuccessorStatesPerParent", "MatchesPerStateRule"}) {
        auto none = [](wxf::Writer& ww) { put_str_list_option(ww, "RequestedData", {"NumStates"}); };
        auto huge = [key](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"NumStates"});
            ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            ww.write(std::string(key));
            ww.write(static_cast<int64_t>((int64_t{1} << 32) + 1));
        };
        const auto ref = worker_call(w, branch_job(3, "Evolve", 0, none, 1));
        const auto got = worker_call(w, branch_job(3, "Evolve", 0, huge, 2));
        ASSERT_FALSE(got.empty()) << key;
        EXPECT_EQ(read_int_key(got, "NumStates"), read_int_key(ref, "NumStates")) << key;
    }
    worker_stop(w);
}

// The worker's fd 1 is stderr once it serves: the CUDA runtime writes device printf (the
// persistent kernel's progress and stall lines) to the process's standard output, and the
// replies go through a duplicate of the fd the parent gave, so those lines cannot enter a frame.
TEST(GpuBinaryGate, OnlyRepliesReachTheReplyStream) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    auto none = [](wxf::Writer& ww) { put_str_list_option(ww, "RequestedData", {"NumStates"}); };
    ASSERT_FALSE(worker_call(w, branch_job(2, "Evolve", 0, none, 1)).empty());
    auto target = [&](int fd) {
        char buf[4096];
        const std::string link = "/proc/" + std::to_string(w.pid) + "/fd/" + std::to_string(fd);
        const ssize_t n = ::readlink(link.c_str(), buf, sizeof(buf) - 1);
        return n < 0 ? std::string() : std::string(buf, static_cast<size_t>(n));
    };
    const std::string out = target(1), err = target(2);
    ASSERT_FALSE(err.empty());
    EXPECT_EQ(out, err) << "fd 1 of the worker is " << out << ", where device printf lands";
    EXPECT_EQ(out.find(w.out_path), std::string::npos);
    worker_stop(w);
}

// "States" lists every state outside Full on both devices, and under event identity the causal and
// branchial lists name canonical events on both: rule {{4},{2}} -> {} from {{1},{1},{1}} at two steps
// under CanonicalizeStates Automatic has 7 states of 2 classes; rule {{1},{2}} -> {} from
// {{1},{2},{3}} at one step under CanonicalizeEvents Full has 6 applications of one event, so its
// 15 branchial pairs join that event to itself.
TEST(GpuBinaryGate, StateListsAndEventEndpointsAgreeAcrossDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    {
        auto opts = [](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"States", "NumStates"});
            put_str_option(ww, "CanonicalizeStates", "Automatic");
        };
        const auto job = session_envelope({{{1}, {1}, {1}}}, {{4}, {2}}, {}, 2, "Evolve", 0, true,
                                          {}, opts, 2, false);
        const auto cpu = host(job);
        const auto gpu = worker_call(w, job);
        ASSERT_FALSE(gpu.empty());
        EXPECT_EQ(count_assoc_entries(cpu, "States"), 7);
        EXPECT_EQ(count_assoc_entries(gpu, "States"), count_assoc_entries(cpu, "States"));
        EXPECT_EQ(read_int_key(gpu, "NumStates"), read_int_key(cpu, "NumStates"));
    }
    {
        auto opts = [](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"BranchialEdges", "NumEvents"});
            put_str_option(ww, "CanonicalizeEvents", "Full");
        };
        const auto job = session_envelope({{{1}, {2}, {3}}}, {{1}, {2}}, {}, 1, "Evolve", 0, true,
                                          {}, opts, 2, false);
        const auto cpu = host(job);
        const auto gpu = worker_call(w, job);
        ASSERT_FALSE(gpu.empty());
        EXPECT_EQ(read_int_key(gpu, "NumEvents"), 1);
        // The ids the list names: one event, so one id, on each device.
        auto endpoint_ids = [](const std::vector<uint8_t>& out) {
            std::set<int64_t> ids;
            size_t records = 0;
            wxf::Parser parser(out);
            parser.skip_header();
            parser.read_association([&](const std::string& k, wxf::Parser& vp) {
                if (k != "BranchialEdges") { vp.skip_value(); return; }
                vp.read_function([&](const std::string&, size_t n, wxf::Parser& args) {
                    for (size_t i = 0; i < n; ++i, ++records)
                        args.read_association([&](const std::string& f, wxf::Parser& v) {
                            if (f == "From" || f == "To") ids.insert(v.read<int64_t>());
                            else v.skip_value();
                        });
                });
            });
            return std::make_pair(records, ids.size());
        };
        EXPECT_EQ(endpoint_ids(cpu), std::make_pair(size_t{15}, size_t{1}));
        EXPECT_EQ(endpoint_ids(gpu), std::make_pair(size_t{15}, size_t{1}));
    }
    worker_stop(w);
}

// Inputs both devices refuse alike: an edge past MAX_ARITY in a rule or in an initial state is an
// error frame naming the arity (the GPU ran it and returned an empty result with a device-memory
// warning), and an ExplorationProbability that is not a finite number is skipped with an
// OptionSkipped warning, so both devices run the default (the GPU kept NaN and gave 2 states
// where the CPU gave 10).
TEST(GpuBinaryGate, MalformedInputsAreRefusedAlikeOnBothDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    auto counts = [](wxf::Writer& ww) { put_str_list_option(ww, "RequestedData", {"NumStates"}); };
    Edge wide;
    for (int64_t v = 1; v <= 17; ++v) wide.push_back(v);
    const auto wide_rule = session_envelope({{{1, 2}}}, {wide}, {{1, 2}}, 1, "Evolve", 0, true, {},
                                            counts, 1, false);
    const auto wide_init = session_envelope({{wide}}, {{1, 2}}, {{1, 2}, {2, 3}}, 1, "Evolve", 0,
                                            true, {}, counts, 1, false);
    for (const auto* job : {&wide_rule, &wide_init}) {
        EXPECT_TRUE(host(*job).empty());
        EXPECT_FALSE(host.w.last_error.empty());
        EXPECT_TRUE(worker_call(w, *job).empty());
        EXPECT_NE(w.last_error.find("arity"), std::string::npos) << w.last_error;
    }
    auto nan_opts = [](wxf::Writer& ww) {
        put_str_list_option(ww, "RequestedData", {"NumStates"});
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string("ExplorationProbability"));
        ww.write(std::nan(""));
    };
    const auto nan_job = branch_job(3, "Evolve", 0, nan_opts, 2);
    const auto cpu = host(nan_job);
    const auto gpu = worker_call(w, nan_job);
    ASSERT_FALSE(gpu.empty());
    EXPECT_TRUE(reply_mentions(cpu, "OptionSkipped"));
    EXPECT_TRUE(reply_mentions(gpu, "OptionSkipped"));
    EXPECT_EQ(read_int_key(gpu, "NumStates"), read_int_key(cpu, "NumStates"));
    worker_stop(w);
}

// With no visible device (CUDA_VISIBLE_DEVICES empty) the GPU worker answers every job with an
// error frame "no usable GPU: no CUDA-capable device is detected" and stays alive, and the
// one-shot mode exits with kNoUsableDeviceExit and the same line on stderr. Before the device
// probe the worker died at its first CUDA call: "match kernel attributes: no CUDA-capable device
// is detected".
TEST(GpuBinaryGate, NoUsableDeviceIsAStructuredRefusal) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    const std::string expected =
        std::string(kNoUsableDevicePrefix) + "no CUDA-capable device is detected";
    auto counts = [](wxf::Writer& ww) { put_str_list_option(ww, "RequestedData", {"NumStates"}); };
    const auto job = build_input_with_op(2, "Evolve", 0, true, {}, counts, 1);

    WorkerPipes w;
    ASSERT_TRUE(worker_start(w, gpu_binary_path(), 0, ""));
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(worker_call(w, job).empty());
        EXPECT_EQ(w.last_error, expected) << "job " << i;
    }
    worker_stop(w);

    int in_pipe[2], err_pipe[2];
    ASSERT_EQ(::pipe(in_pipe), 0);
    ASSERT_EQ(::pipe(err_pipe), 0);
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        ::dup2(in_pipe[0], 0);
        ::dup2(err_pipe[1], 2);
        ::close(in_pipe[1]);
        ::close(err_pipe[0]);
        ::setenv("CUDA_VISIBLE_DEVICES", "", 1);
        const std::string exe = gpu_binary_path();
        ::execl(exe.c_str(), exe.c_str(), (char*)nullptr);
        ::_exit(127);
    }
    ::close(in_pipe[0]);
    ::close(err_pipe[1]);
    EXPECT_EQ(::write(in_pipe[1], job.data(), job.size()), static_cast<ssize_t>(job.size()));
    ::close(in_pipe[1]);
    std::string err;
    char buf[512];
    for (ssize_t r; (r = ::read(err_pipe[0], buf, sizeof(buf))) > 0;) err.append(buf, r);
    ::close(err_pipe[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), kNoUsableDeviceExit);
    EXPECT_NE(err.find(expected), std::string::npos) << err;
}

// "Device" in RequestedData names the device that ran the job: <|"Type" -> "GPU", "Name" -> ...,
// "ComputeCapability" -> "8.9"|> from the GPU binary, <|"Type" -> "CPU"|> from the CPU binary.
TEST(GpuBinaryGate, TheDeviceComponentNamesTheDevice) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    auto device = [](wxf::Writer& ww) {
        put_str_list_option(ww, "RequestedData", {"NumStates", "Device"});
    };
    const auto job = build_input_with_op(2, "Evolve", 0, true, {}, device, 1);
    WorkerPipes w;
    ASSERT_TRUE(worker_start(w, gpu_binary_path()));
    const auto gpu = worker_call(w, job);
    worker_stop(w);
    if (gpu.empty()) GTEST_SKIP() << "no usable device: " << w.last_error;
    EXPECT_TRUE(reply_mentions(gpu, "Device"));
    EXPECT_TRUE(reply_mentions(gpu, "GPU"));
    EXPECT_TRUE(reply_mentions(gpu, "ComputeCapability"));

    if (!CpuWorker::built()) GTEST_SKIP() << "hg_evolve is not built here";
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    const auto cpu = host(job);
    EXPECT_TRUE(reply_mentions(cpu, "Device"));
    EXPECT_TRUE(reply_mentions(cpu, "CPU"));
    EXPECT_FALSE(reply_mentions(cpu, "ComputeCapability"));
}

// ExploreFromCanonicalStatesOnly needs Full states: under Automatic it is refused with the
// QuotientNeedsFull warning and every state is expanded, on both devices, so the counts are the
// counts without it. Rule {{1,2}} -> {{2,1},{2,1},{1,1,1,1}} from {{1,1},{2,2},{2,2}} at three
// steps gives (18, 75, 48) states, events and causal edges; with the option applied the devices
// gave (18, 31, 19) and the GPU's causal count varied between runs.
TEST(GpuBinaryGate, QuotientExplorationNeedsFullStatesOnBothDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    for (const bool ecso : {false, true}) {
        auto opts = [ecso](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"NumStates", "NumEvents", "NumCausalEdges"});
            put_str_option(ww, "CanonicalizeStates", "Automatic");
            put_str_option(ww, "ExploreFromCanonicalStatesOnly", ecso ? "True" : "False");
        };
        const auto job = session_envelope({{{1, 1}, {2, 2}, {2, 2}}}, {{1, 2}},
                                          {{2, 1}, {2, 1}, {1, 1, 1, 1}}, 3, "Evolve", 0, true,
                                          {}, opts, 3, false);
        for (int run = 0; run < 3; ++run) {
            const auto cpu = host(job);
            const auto gpu = worker_call(w, job);
            ASSERT_FALSE(gpu.empty());
            for (const char* key : {"NumStates", "NumEvents", "NumCausalEdges"})
                EXPECT_EQ(read_int_key(gpu, key), read_int_key(cpu, key)) << key << " ecso " << ecso;
            EXPECT_EQ(read_int_key(gpu, "NumEvents"), 75) << "ecso " << ecso;
            EXPECT_EQ(read_int_key(gpu, "NumCausalEdges"), 48) << "ecso " << ecso;
        }
    }
    worker_stop(w);
}

// "StepStatistics" is the same reply on both devices: under quotient exploration from each
// engine's class multiplicities, and under None, Automatic and Full from its raw states. The
// reply carries the per-vertex distributions and the per-state curvature correlations.
TEST(GpuBinaryGate, StepStatisticsAgreeAcrossDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    const std::pair<const char*, bool> modes[] = {
        {"Full", true}, {"None", false}, {"Automatic", false}, {"Full", false}};
    for (const auto& [canon, quotient] : modes) {
        auto opts = [canon, quotient](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"StepStatistics"});
            put_str_option(ww, "CanonicalizeStates", canon);
            put_str_option(ww, "ExploreFromCanonicalStatesOnly", quotient ? "True" : "False");
        };
        CpuWorker host;
        ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
        const auto cpu = host(branch_job(3, "Evolve", 0, opts, 3));
        const auto gpu = worker_call(w, branch_job(3, "Evolve", 0, opts, 3));
        const auto c = value_bytes(cpu, "StepStatistics"), g = value_bytes(gpu, "StepStatistics");
        ASSERT_FALSE(c.empty()) << canon << " quotient=" << quotient;
        for (const char* k : {"VertexInvariants", "LargestComponentDimension", "OllivierMoranI",
                              "OllivierDegreeCorrelation", "Kurtosis"})
            EXPECT_NE(std::search(c.begin(), c.end(), k, k + std::strlen(k)), c.end()) << k;
        EXPECT_EQ(c, g) << canon << " quotient=" << quotient;
    }
    // "StepStatisticsBranchial" -> All under each state canonicalization, both weightings.
    for (const char* canon : {"None", "Automatic", "Full"})
        for (const char* weighting : {"States", "Classes"}) {
            auto opts = [canon, weighting](wxf::Writer& ww) {
                put_str_list_option(ww, "RequestedData", {"StepStatistics"});
                put_str_option(ww, "CanonicalizeStates", canon);
                put_str_list_option(ww, "StepStatisticsBranchial", {"Graph", "Overlap"});
                put_str_option(ww, "StepStatisticsWeighting", weighting);
            };
            CpuWorker host;
            ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
            const auto cpu = host(branch_job(4, "Evolve", 0, opts, 4));
            const auto gpu = worker_call(w, branch_job(4, "Evolve", 0, opts, 4));
            const auto c = value_bytes(cpu, "StepStatistics");
            ASSERT_FALSE(c.empty()) << canon << " " << weighting;
            const bool full = std::string(canon) == "Full";
            EXPECT_TRUE(reply_mentions(c, full ? "BranchialDegree" : "OverlapByBranchialDistance"))
                << canon;
            const auto rc = branchial_record_bytes(cpu), rg = branchial_record_bytes(gpu);
            ASSERT_EQ(rc.size(), rg.size()) << canon << " " << weighting;
            for (size_t i = 0; i < rc.size(); ++i) {
                ASSERT_EQ(rc[i].size(), rg[i].size()) << canon << " step " << i;
                for (size_t j = 0; j < rc[i].size(); ++j)
                    EXPECT_TRUE(rc[i][j] == rg[i][j])
                        << canon << " " << weighting << " step " << i << " " << rc[i][j].first;
            }
            EXPECT_EQ(c, value_bytes(gpu, "StepStatistics")) << canon << " " << weighting;
        }
    worker_stop(w);
}

// A step of 2048 states or more has its overlap rows counted on the device
// (hg_gpu::branchial_rows_device), and the reply is still the CPU's bytes: rule
// {{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}} from {{1,2},{1,3}} at 6 steps, whose step 6 has
// 14792 states, with "StepStatisticsBranchial" All under both weightings.
TEST(GpuBinaryGate, DeviceBranchialRowsGiveTheCpuStepStatistics) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    for (const char* weighting : {"States", "Classes"}) {
        auto opts = [weighting](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"StepStatistics"});
            put_str_list_option(ww, "StepStatisticsBranchial", {"Graph", "Overlap"});
            put_str_option(ww, "StepStatisticsWeighting", weighting);
        };
        const auto job = session_envelope({{{1, 2}, {1, 3}}}, {{1, 2}, {1, 3}},
                                          {{1, 2}, {1, 4}, {2, 4}, {3, 4}}, 6, "Evolve", 0, true,
                                          {}, opts, 3, false);
        const auto cpu = host(job);
        const auto gpu = worker_call(w, job);
        ASSERT_FALSE(gpu.empty()) << w.last_error;
        const auto c = value_bytes(cpu, "StepStatistics");
        ASSERT_FALSE(c.empty()) << weighting;
        EXPECT_TRUE(reply_mentions(c, "OverlapByBranchialDistance")) << weighting;
        EXPECT_EQ(c, value_bytes(gpu, "StepStatistics")) << weighting;
    }
    worker_stop(w);
}

// The sampling and cap options keep the same transitions on both devices, through one worker
// that has run other jobs. First the GPUEvolution tutorial's calls in its order, sent with every
// option of hgJobOptions: the last, the branching rule from two loops at 5 steps with
// TransitionRate 0.25 and RandomSeed 7, is {11, 10, 9, 2} on the CPU. Then each sampling and cap
// option under CanonicalizeStates None, Automatic, Full and Full with quotient exploration, at
// three seeds. The four counts agree on every job, and quotient exploration gives the counts of
// full capture.
TEST(GpuBinaryGate, SamplingAndCapsKeepTheSameTransitionsOnBothDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    const char* counts[] = {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"};
    auto agree = [&](const std::vector<uint8_t>& job, const std::string& at) {
        const auto cpu = host(job);
        const auto gpu = worker_call(w, job);
        EXPECT_FALSE(cpu.empty()) << at;
        EXPECT_FALSE(gpu.empty()) << at << ": " << w.last_error;
        for (const char* k : counts)
            EXPECT_EQ(read_int_key(gpu, k), read_int_key(cpu, k)) << at << " " << k;
        return cpu;
    };
    auto sym = [](wxf::Writer& ww, const char* k, const char* s) {
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string(k));
        ww.write_symbol(s);
    };
    auto key = [](wxf::Writer& ww, const char* k) {
        ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        ww.write(std::string(k));
    };
    const StateList seed = {{{1, 1}, {1, 1}}};
    const EdgeList lhs = {{1, 2}, {2, 3}};
    const EdgeList rhs = {{1, 3}, {3, 4}, {1, 4}, {2, 4}};

    struct Call { StateList init; EdgeList l, r; int64_t steps; const char* mode; const char* ecso;
                  double rate; };
    const StateList physics_init = {{{1, 2}, {1, 3}}};
    const EdgeList physics_l = {{1, 2}, {1, 3}}, physics_r = {{1, 2}, {1, 4}, {2, 4}, {3, 4}};
    const Call calls[] = {
        {{{{1, 2}}}, {{1, 2}}, {{1, 3}, {3, 2}}, 5, "None", "False", 1.0},
        {{{{1, 2}}}, {{1, 2}}, {{1, 3}, {3, 2}}, 5, "Full", "False", 1.0},
        {physics_init, physics_l, physics_r, 3, "Full", "False", 1.0},
        {physics_init, physics_l, physics_r, 4, "Full", "False", 1.0},
        {physics_init, physics_l, physics_r, 4, "Full", "True", 1.0},
        {seed, lhs, rhs, 5, "None", "False", 0.25},
    };
    std::vector<uint8_t> last;
    for (const Call& c : calls) {
        last = agree(build_input(c.init, c.l, c.r, c.steps, [&](wxf::Writer& ww) {
            sym(ww, "CanonicalizeStates", c.mode);
            sym(ww, "CanonicalizeEvents", "None");
            sym(ww, "CausalTransitiveReduction", "True");
            key(ww, "MaxSuccessorStatesPerParent"); ww.write(int64_t{0});
            key(ww, "MaxStatesPerStep"); ww.write(int64_t{0});
            key(ww, "ExplorationProbability"); ww.write(1.0);
            key(ww, "TransitionRate"); ww.write(c.rate);
            key(ww, "RuleWeights"); ww.write_function("List", 0);
            key(ww, "RandomSeed"); ww.write(int64_t{7});
            sym(ww, "ExploreFromCanonicalStatesOnly", c.ecso);
            sym(ww, "ShowProgress", "False");
            sym(ww, "ShowGenesisEvents", "False");
            key(ww, "BranchialStep"); ww.write(int64_t{-1});
            sym(ww, "EdgeDeduplication", "True");
            sym(ww, "IncludeCanonicalHashes", "False");
            put_str_option(ww, "StepStatisticsWeighting", "States");
            key(ww, "StepStatisticsBranchial"); ww.write_function("List", 0);
            put_str_list_option(ww, "RequestedData", {counts[0], counts[1], counts[2], counts[3]});
            key(ww, "GraphProperties"); ww.write_function("List", 0);
            sym(ww, "UniformRandom", "False");
            key(ww, "MatchesPerStep"); ww.write(int64_t{0});
            key(ww, "MatchesPerStateRule"); ww.write(int64_t{0});
        }, 22), "tutorial call " + std::to_string(&c - calls));
    }
    EXPECT_EQ(read_int_key(last, "NumStates"), 11);
    EXPECT_EQ(read_int_key(last, "NumEvents"), 10);
    EXPECT_EQ(read_int_key(last, "NumCausalEdges"), 9);
    EXPECT_EQ(read_int_key(last, "NumBranchialEdges"), 2);

    struct Option { const char* key; double real; int64_t integer; bool list; };
    const Option options[] = {
        {"TransitionRate", 0.25, 0, false},
        {"TransitionRate", 0.5, 0, false},
        {"ExplorationProbability", 0.5, 0, false},
        {"RuleWeights", 0.4, 0, true},
        {"MaxSuccessorStatesPerParent", 0, 2, false},
        {"MaxStatesPerStep", 0, 3, false},
        {"MatchesPerStateRule", 0, 1, false},
    };
    struct Mode { const char* states; const char* ecso; };
    const Mode modes[] = {{"Full", "True"}, {"None", "False"}, {"Automatic", "False"},
                          {"Full", "False"}};
    // Quotient exploration (modes[0]) and full capture (modes[3]) under Full states give the same
    // counts (docs/SPEC.md §5.4); the devices agree with each other on each.
    auto quotient_matches_full = [&](const std::vector<std::vector<uint8_t>>& by_mode,
                                     const std::string& at) {
        for (const char* k : counts)
            EXPECT_EQ(read_int_key(by_mode[0], k), read_int_key(by_mode[3], k))
                << at << " " << k << ": quotient exploration against full capture";
    };
    for (const Option& o : options) {
        for (int64_t rs : {int64_t{7}, int64_t{1}, int64_t{12345}}) {
            std::vector<std::vector<uint8_t>> by_mode;
            std::string at;
            for (const Mode& m : modes) {
                at = std::string(o.key) + " " +
                    (o.integer != 0 ? std::to_string(o.integer) : std::to_string(o.real)) +
                    " CanonicalizeStates " + m.states + " ExploreFromCanonicalStatesOnly " +
                    m.ecso + " RandomSeed " + std::to_string(rs);
                by_mode.push_back(agree(build_input(seed, lhs, rhs, 5, [&](wxf::Writer& ww) {
                    put_str_list_option(ww, "RequestedData",
                                        {counts[0], counts[1], counts[2], counts[3]});
                    sym(ww, "CanonicalizeStates", m.states);
                    sym(ww, "ExploreFromCanonicalStatesOnly", m.ecso);
                    key(ww, o.key);
                    if (o.list) ww.write(std::vector<double>{o.real});
                    else if (o.integer != 0) ww.write(o.integer);
                    else ww.write(o.real);
                    key(ww, "RandomSeed"); ww.write(rs);
                }, 5), at));
            }
            quotient_matches_full(by_mode, at);
        }
    }
    // "MatchesPerStep" with "UniformRandom": a step's N lowest-ranked transitions.
    for (int64_t rs : {int64_t{7}, int64_t{1}, int64_t{12345}}) {
        for (const Mode& m : modes) {
            const std::string at = std::string("MatchesPerStep 2 CanonicalizeStates ") + m.states +
                                   " ExploreFromCanonicalStatesOnly " + m.ecso + " RandomSeed " +
                                   std::to_string(rs);
            agree(build_input(seed, lhs, rhs, 5, [&](wxf::Writer& ww) {
                put_str_list_option(ww, "RequestedData",
                                    {counts[0], counts[1], counts[2], counts[3]});
                sym(ww, "CanonicalizeStates", m.states);
                sym(ww, "ExploreFromCanonicalStatesOnly", m.ecso);
                sym(ww, "UniformRandom", "True");
                key(ww, "MatchesPerStep"); ww.write(int64_t{2});
                key(ww, "RandomSeed"); ww.write(rs);
            }, 6), at);
        }
    }
    // A rule whose join order differs from its authored order ({2,1} at position 2 is bound
    // before {2} at position 1), on a state of duplicate edges, with an empty right-hand side.
    // A transition's key reads the matched edges' ranks in authored order on both devices.
    const StateList dup_init = {{{1}, {1, 1}, {1}, {1, 1}, {1, 1}, {1, 1}}};
    const EdgeList dup_lhs = {{2, 1}, {2}, {2, 1}};
    const EdgeList dup_rhs = {};
    const Option dup_options[] = {
        {"TransitionRate", 0.25, 0, false},
        {"MaxSuccessorStatesPerParent", 0, 4, false},
        {"MatchesPerStateRule", 0, 4, false},
    };
    for (const Option& o : dup_options) {
        for (int64_t rs : {int64_t{7}, int64_t{1}, int64_t{718832434}}) {
            std::vector<std::vector<uint8_t>> by_mode;
            std::string at;
            for (const Mode& m : modes) {
                at = std::string("authored order ") + o.key + " " +
                    (o.integer != 0 ? std::to_string(o.integer) : std::to_string(o.real)) +
                    " CanonicalizeStates " + m.states + " ExploreFromCanonicalStatesOnly " +
                    m.ecso + " RandomSeed " + std::to_string(rs);
                by_mode.push_back(agree(build_input(dup_init, dup_lhs, dup_rhs, 1, [&](wxf::Writer& ww) {
                    put_str_list_option(ww, "RequestedData",
                                        {counts[0], counts[1], counts[2], counts[3]});
                    sym(ww, "CanonicalizeStates", m.states);
                    sym(ww, "ExploreFromCanonicalStatesOnly", m.ecso);
                    key(ww, o.key);
                    if (o.integer != 0) ww.write(o.integer);
                    else ww.write(o.real);
                    key(ww, "RandomSeed"); ww.write(rs);
                }, 5), at));
            }
            quotient_matches_full(by_mode, at);
        }
    }
    worker_stop(w);
}

// The per-state caps hold however many matches a state has. Rule {{3},{5}} -> {} on n copies of
// {1} has n(n-1) matches in one state: 552 at n = 24 and 1560 at n = 40, past the 512 a capping
// block ranks in shared memory (hg_gpu::kDrainCapBuffer). Under MatchesPerStateRule 4,
// MaxSuccessorStatesPerParent 4 and both, with None and Full states, 1 step, the device gives the
// host's counts (worker_fuzz seed 4293: the device kept all 552).
TEST(GpuBinaryGate, PerStateCapsHoldPastTheSharedMemoryBuffer) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    const char* counts[] = {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"};
    struct Caps { int64_t per_rule, per_parent; };
    for (int n : {24, 40}) {
        StateList init(1);
        for (int i = 0; i < n; ++i) init[0].push_back({1});
        for (const char* mode : {"None", "Full"}) {
            for (Caps c : {Caps{4, 0}, Caps{0, 4}, Caps{4, 4}}) {
                const auto job = build_input(init, {{3}, {5}}, {}, 1, [&](wxf::Writer& ww) {
                    put_str_option(ww, "CanonicalizeStates", mode);
                    ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    ww.write(std::string("CanonicalizeEvents"));
                    ww.write_symbol("None");
                    ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    ww.write(std::string("MatchesPerStateRule"));
                    ww.write(c.per_rule);
                    ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    ww.write(std::string("MaxSuccessorStatesPerParent"));
                    ww.write(c.per_parent);
                    put_str_list_option(ww, "RequestedData",
                                        {counts[0], counts[1], counts[2], counts[3]});
                }, 5);
                const std::string at = "n " + std::to_string(n) + " " + mode + " caps " +
                                       std::to_string(c.per_rule) + "/" +
                                       std::to_string(c.per_parent);
                const auto cpu = host(job);
                const auto gpu = worker_call(w, job);
                ASSERT_FALSE(cpu.empty()) << at;
                ASSERT_FALSE(gpu.empty()) << at << ": " << w.last_error;
                for (const char* k : counts)
                    EXPECT_EQ(read_int_key(gpu, k), read_int_key(cpu, k)) << at << " " << k;
            }
        }
    }
    worker_stop(w);
}

// A job that fails in the GPU worker leaves the worker as a fresh one: the evolver drops its
// engine and the device is reset (PersistentEvolver::release_after_failure), and the next job
// gives the counts it gave before the failure. The failure is a word that is not a verb.
TEST(GpuBinaryGate, AFailedJobLeavesTheWorkerFresh) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    const char* counts[] = {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"};
    const auto job = build_input_with_op(4, "Evolve", 0, true, {}, [&](wxf::Writer& ww) {
        put_str_list_option(ww, "RequestedData", {counts[0], counts[1], counts[2], counts[3]});
    }, 1);
    const auto before = worker_call(w, job);
    ASSERT_FALSE(before.empty()) << w.last_error;
    EXPECT_GT(read_int_key(before, "NumStates"), 1);
    EXPECT_TRUE(worker_call(w, build_input_with_op(4, "Reticulate")).empty());
    EXPECT_NE(w.last_error.find("not a verb"), std::string::npos) << w.last_error;
    for (int i = 0; i < 2; ++i) {
        const auto after = worker_call(w, job);
        ASSERT_FALSE(after.empty()) << w.last_error;
        for (const char* k : counts) EXPECT_EQ(read_int_key(after, k), read_int_key(before, k)) << k;
    }
    worker_stop(w);
}

// Both devices report the same warnings: quotient exploration or Automatic event identity
// without Full state canonicalization, and an option value the parser skips.
TEST(GpuBinaryGate, WarningsAgreeAcrossDevices) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    const std::vector<std::pair<std::string, std::function<void(wxf::Writer&)>>> cases = {
        {"ExploreFromCanonicalStatesOnly", [](wxf::Writer& ww) {
             put_str_option(ww, "ExploreFromCanonicalStatesOnly", "True"); }},
        {"CanonicalizeEvents Automatic", [](wxf::Writer& ww) {
             put_str_option(ww, "CanonicalizeEvents", "Automatic"); }},
        {"CanonicalizeEvents {InputState, 42}", [](wxf::Writer& ww) {
             ww.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
             ww.write(std::string("CanonicalizeEvents"));
             ww.write_function("List", 2);
             ww.write(std::string("InputState"));
             ww.write(static_cast<int64_t>(42)); }},
    };
    for (const auto& [name, option] : cases) {
        auto opts = [&option](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {"NumStates", "NumEvents"});
            option(ww);
        };
        CpuWorker host;
        ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
        const auto cpu = host(branch_job(3, "Evolve", 0, opts, 2));
        const auto gpu = worker_call(w, branch_job(3, "Evolve", 0, opts, 2));
        ASSERT_FALSE(gpu.empty()) << name;
        const auto c = value_bytes(cpu, "Warnings"), g = value_bytes(gpu, "Warnings");
        EXPECT_FALSE(c.empty()) << name << ": the CPU reports no warning";
        EXPECT_EQ(c, g) << name;
        EXPECT_EQ(read_int_key(gpu, "NumStates"), read_int_key(cpu, "NumStates")) << name;
    }
    worker_stop(w);
}

// On the GPU, "StepStatistics" asked of a session opened for "NumStates" under quotient
// exploration is what one evolution asking for it gives on the CPU.
TEST(GpuBinaryGate, SessionStepStatisticsAgreeWithOneEvolve) {
    {
        std::ifstream probe(gpu_binary_path(), std::ios::binary);
        if (!probe) GTEST_SKIP() << "hg_evolve_gpu is not built here";
    }
    WorkerPipes w;
    if (!worker_start(w, gpu_binary_path())) {
        worker_stop(w);
        GTEST_SKIP() << "could not start hg_evolve_gpu --serve";
    }
    auto opts = [](const std::string& prop) {
        return [prop](wxf::Writer& ww) {
            put_str_list_option(ww, "RequestedData", {prop});
            put_str_option(ww, "CanonicalizeStates", "Full");
            put_str_option(ww, "ExploreFromCanonicalStatesOnly", "True");
        };
    };
    CpuWorker host;
    ASSERT_TRUE(host.ok) << "could not start hg_evolve --serve";
    const auto direct = host(branch_job(3, "Evolve", 0, opts("StepStatistics"), 3));
    const auto opened = worker_call(w, branch_job(0, "Open", 0, opts("NumStates"), 3));
    const int64_t handle = read_int_key(opened, "Session");
    ASSERT_NE(handle, 0);
    worker_call(w, branch_job(3, "Step", handle, opts("NumStates"), 3));
    const auto queried = worker_call(w, branch_job(0, "Query", handle, opts("StepStatistics"), 3));
    worker_call(w, branch_job(0, "Close", handle, opts("NumStates"), 3));
    const auto want = value_bytes(direct, "StepStatistics");
    ASSERT_FALSE(want.empty());
    EXPECT_EQ(value_bytes(queried, "StepStatistics"), want);
    worker_stop(w);
}
#endif  // _WIN32

// A record stores the arities and slot degrees as runs of equal values, so its size does not grow
// with the state's edges and vertices when their values repeat. A 12 x 12 directed torus (288
// binary edges, 144 vertices of slot degree 4): one arity run and one degree run, 16 bytes past
// the distributions and the ball, and the sequences read back whole.
TEST(StateStatistics, RecordStoresSequencesAsRuns) {
    std::vector<std::vector<uint32_t>> torus;
    for (uint32_t i = 0; i < 12; ++i)
        for (uint32_t j = 0; j < 12; ++j) {
            torus.push_back({i * 12 + j, ((i + 1) % 12) * 12 + j});
            torus.push_back({i * 12 + j, i * 12 + (j + 1) % 12});
        }
    std::vector<uint64_t> storage;
    const hgcommon::StateInvariantRecord& rec = *hg::stats::invariant_record(torus, storage);
    EXPECT_EQ(rec.num_edges, 288u);
    EXPECT_EQ(rec.num_vertices, 144u);
    EXPECT_EQ(rec.num_arity_runs, 1u);
    EXPECT_EQ(rec.num_degree_runs, 1u);
    EXPECT_EQ(storage.size() * 8,
              hgcommon::StateInvariantRecord::bytes(1, 1, rec.num_ball, rec.num_distributions,
                                                    rec.num_geodesic));
    const auto inv = hg::stats::state_invariants(torus);
    EXPECT_EQ(inv.arities, std::vector<int64_t>(288, 2));
    EXPECT_EQ(inv.degree_sequence, std::vector<int64_t>(144, 4));
}

// The per-state invariants on states whose values are worked by hand.
TEST(StateStatistics, InvariantsOfSmallStates) {
    using hg::stats::state_invariants;
    // A triangle: its incidence graph is a 6-cycle.
    const auto tri = state_invariants({{1, 2}, {2, 3}, {3, 1}});
    EXPECT_EQ(tri.vertex_count, 3);
    EXPECT_EQ(tri.edge_count, 3);
    EXPECT_EQ(tri.arities, (std::vector<int64_t>{2, 2, 2}));
    EXPECT_EQ(tri.degree_sequence, (std::vector<int64_t>{2, 2, 2}));
    EXPECT_EQ(tri.max_degree, 2);
    EXPECT_DOUBLE_EQ(tri.mean_degree, 2.0);
    EXPECT_EQ(tri.two_section_edge_count, 3);
    EXPECT_EQ(tri.components, 1);
    EXPECT_EQ(tri.cycle_rank, 1);
    EXPECT_EQ(tri.incidence_cycle_rank, 1);
    EXPECT_EQ(tri.incidence_diameter, 3);
    EXPECT_DOUBLE_EQ(tri.incidence_mean_distance, 1.8);
    EXPECT_DOUBLE_EQ(tri.largest_component_fraction, 1.0);

    // A self-loop: one vertex, two slots, one incidence.
    const auto loop = state_invariants({{1, 1}});
    EXPECT_EQ(loop.degree_sequence, (std::vector<int64_t>{2}));
    EXPECT_EQ(loop.two_section_edge_count, 0);
    EXPECT_EQ(loop.cycle_rank, 0);
    EXPECT_EQ(loop.incidence_cycle_rank, 0);
    EXPECT_EQ(loop.incidence_diameter, 1);
    EXPECT_DOUBLE_EQ(loop.incidence_mean_distance, 1.0);

    // A two-edge path and a separate edge: the path's component is a 5-node path.
    const auto two = state_invariants({{1, 2}, {2, 3}, {4, 5}});
    EXPECT_EQ(two.components, 2);
    EXPECT_EQ(two.incidence_diameter, 4);
    EXPECT_DOUBLE_EQ(two.incidence_mean_distance, 2.0);
    EXPECT_DOUBLE_EQ(two.largest_component_fraction, 0.6);

    // Two components of four nodes: a ternary edge (diameter 2) and an edge with a self-loop
    // (a 4-node path, diameter 3). The path is the largest by the tie rule, in either order.
    for (const auto& st : {std::vector<std::vector<uint32_t>>{{1, 2, 3}, {4, 5}, {5, 5}},
                           std::vector<std::vector<uint32_t>>{{1, 2}, {2, 2}, {3, 4, 5}}}) {
        const auto tie = state_invariants(st);
        EXPECT_EQ(tie.components, 2);
        EXPECT_EQ(tie.incidence_diameter, 3);
        EXPECT_NEAR(tie.incidence_mean_distance, 10.0 / 6.0, 1e-12);
        EXPECT_DOUBLE_EQ(tie.largest_component_fraction, 0.4);
    }

    // No edges: every invariant is 0.
    const auto none = state_invariants({});
    EXPECT_EQ(none.vertex_count, 0);
    EXPECT_EQ(none.components, 0);
}

// A weighted summary is the summary of the population it stands for.
TEST(StateStatistics, WeightedSummary) {
    const auto odd = hg::stats::summarise({{1.0, 2}, {3.0, 1}}, 1.0);   // 1, 1, 3
    EXPECT_EQ(odd.n, 3u);
    EXPECT_DOUBLE_EQ(odd.mean, 5.0 / 3.0);
    EXPECT_NEAR(odd.standard_deviation, std::sqrt(4.0 / 3.0), 1e-12);
    EXPECT_DOUBLE_EQ(odd.median, 1.0);
    EXPECT_DOUBLE_EQ(odd.min, 1.0);
    EXPECT_DOUBLE_EQ(odd.max, 3.0);
    EXPECT_EQ(odd.histogram, (std::map<double, uint64_t>{{1.0, 2}, {3.0, 1}}));
    const auto even = hg::stats::summarise({{3.0, 1}, {1.0, 1}}, 1.0);  // 1, 3
    EXPECT_DOUBLE_EQ(even.median, 2.0);
    const auto rounded = hg::stats::summarise({{1.234, 1}, {1.231, 1}}, 0.01);
    ASSERT_EQ(rounded.histogram.size(), 1u);
    EXPECT_NEAR(rounded.histogram.begin()->first, 1.23, 1e-12);
}

// Under "Delivery" -> "Delta" a BranchialGraph at its default step, the final one, is delivered
// whole on every verb: the final step moves with each Step, so the graph is not append-only and
// an increment merged into the earlier graph kept the edges of steps that are no longer final.
// The StatesGraph beside it is an increment.
TEST(Session, ADeltaBranchialGraphAtTheFinalStepIsDeliveredWhole) {
    HostBridge host;
    auto opts = [](wxf::Writer& w) {
        put_str_list_option(w, "GraphProperties", {"BranchialGraph", "StatesGraph"});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("BranchialStep"));
        w.write(int64_t{-1});
    };
    // The IsDelta marker and the edge count of one property's GraphData.
    auto graph_of = [](const std::vector<uint8_t>& out, const std::string& prop) {
        std::pair<int64_t, int64_t> v{-1, -1};
        wxf::Parser parser(out);
        parser.skip_header();
        parser.read_association([&](const std::string& k, wxf::Parser& vp) {
            if (k != "GraphData") { vp.skip_value(); return; }
            vp.read_association([&](const std::string& name, wxf::Parser& gp) {
                if (name != prop) { gp.skip_value(); return; }
                gp.read_association([&](const std::string& field, wxf::Parser& fp) {
                    if (field == "IsDelta") {
                        v.first = fp.read<int64_t>();
                    } else if (field == "Edges") {
                        fp.read_function([&](const std::string&, size_t n, wxf::Parser& ep) {
                            for (size_t i = 0; i < n; ++i) ep.skip_value();
                            v.second = static_cast<int64_t>(n);
                        });
                    } else {
                        fp.skip_value();
                    }
                });
            });
        });
        return v;
    };
    const auto opened = run_rewriting_core(branch_job(1, "Open", 0, opts, 2), host);
    const int64_t h = read_int_key(opened, "Session");
    ASSERT_GT(h, 0);
    run_rewriting_core(branch_job(1, "Step", h, opts, 2, true), host);
    const auto step = run_rewriting_core(branch_job(1, "Step", h, opts, 2, true), host);
    const auto full = run_rewriting_core(branch_job(0, "Query", h, opts, 2), host);
    EXPECT_EQ(graph_of(step, "BranchialGraph").first, 0);
    EXPECT_EQ(graph_of(step, "StatesGraph").first, 1);
    EXPECT_FALSE(reply_mentions(step, "OptionSkipped"));
    EXPECT_EQ(graph_of(step, "BranchialGraph").second, graph_of(full, "BranchialGraph").second)
        << "the delivered BranchialGraph is not the whole graph at the final step";
    EXPECT_GT(graph_of(full, "BranchialGraph").second, 0);
    run_rewriting_core(branch_job(0, "Close", h, opts, 2), host);
}

namespace {

// One property's GraphData from a reply: IsDelta, each vertex record by its key's bytes, and the
// set of edge records' bytes.
struct DeliveredGraph {
    int64_t is_delta = -1;
    std::map<std::vector<uint8_t>, std::vector<uint8_t>> vertex_data;
    std::set<std::vector<uint8_t>> vertices, edges;
};
DeliveredGraph delivered_graph(const std::vector<uint8_t>& out, const std::string& prop) {
    DeliveredGraph g;
    auto element_bytes = [](wxf::Parser& p) {
        const size_t at = p.position();
        p.skip_value();
        return std::vector<uint8_t>(p.data() + at, p.data() + p.position());
    };
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "GraphData") { vp.skip_value(); return; }
        vp.read_association([&](const std::string& name, wxf::Parser& gp) {
            if (name != prop) { gp.skip_value(); return; }
            gp.read_association([&](const std::string& field, wxf::Parser& fp) {
                if (field == "IsDelta") {
                    g.is_delta = fp.read<int64_t>();
                } else if (field == "Vertices" || field == "Edges") {
                    fp.read_function([&](const std::string&, size_t n, wxf::Parser& ep) {
                        for (size_t i = 0; i < n; ++i)
                            (field == "Vertices" ? g.vertices : g.edges).insert(element_bytes(ep));
                    });
                } else if (field == "VertexData") {
                    fp.read_association_generic([&](wxf::Parser& kp, wxf::Parser& dp) {
                        auto key = element_bytes(kp);
                        g.vertex_data[std::move(key)] = element_bytes(dp);
                    });
                } else {
                    fp.skip_value();
                }
            });
        });
    });
    return g;
}

}  // namespace

// A Step keeps the graph the session served: each CausalGraph edge and event record of a Step's
// reply is in the next Step's, under quotient exploration and in full capture with Full states.
// The replay mints event ids from per-worker blocks; a block left part-used by one Step gave the
// next Step's events ids below other workers' earlier ones, which renumbered the dense event ids
// (Hypergraph::for_each_reconstructed_event). On {{1,2},{1,3}} -> {{1,2},{1,3},{2,3}} from
// {{1,2},{1,3}} at the default worker count, the fourth Step lost 56 of the third's 72 edges and
// changed 62 event records; at one worker none.
TEST(Session, AStepKeepsTheGraphItServed) {
    for (const char* explore : {"True", "False"}) {
        HostBridge host;
        auto opts = [&](wxf::Writer& w) {
            put_str_list_option(w, "GraphProperties", {"CausalGraph"});
            put_str_option(w, "CanonicalizeStates", "Full");
            put_str_option(w, "ExploreFromCanonicalStatesOnly", explore);
        };
        auto job = [&](int64_t steps, const char* op, int64_t h) {
            return session_envelope(kBranchSeed, kBranchLhs, kBranchRhs, steps, op, h,
                                    std::string(op) == "Open", {}, opts, 3, false);
        };
        const auto opened = run_rewriting_core(job(1, "Open", 0), host);
        const int64_t h = read_int_key(opened, "Session");
        ASSERT_GT(h, 0);
        DeliveredGraph prev = delivered_graph(opened, "CausalGraph");
        for (int step = 1; step <= 4; ++step) {
            const auto out = run_rewriting_core(job(1, "Step", h), host);
            const DeliveredGraph g = delivered_graph(out, "CausalGraph");
            size_t gone = 0, changed = 0;
            for (const auto& e : prev.edges) gone += g.edges.count(e) == 0;
            for (const auto& [k, v] : prev.vertex_data) {
                const auto it = g.vertex_data.find(k);
                changed += it == g.vertex_data.end() || it->second != v;
            }
            EXPECT_EQ(gone, 0u) << "explore " << explore << " step " << step;
            EXPECT_EQ(changed, 0u) << "explore " << explore << " step " << step;
            EXPECT_GE(g.edges.size(), prev.edges.size()) << "explore " << explore;
            prev = g;
        }
        EXPECT_GT(prev.edges.size(), 100u) << "explore " << explore;
        run_rewriting_core(job(0, "Close", h), host);
    }
}

// "Delivery" -> "Delta" under quotient exploration and in full capture: after the first delivery
// each Step's reply is an increment (IsDelta 1, no OptionSkipped), and the increments merged into
// the first delivery are the whole graph a full Query then gives, vertex records included.
TEST(Session, ADeltaIsAnIncrement) {
    for (const char* explore : {"True", "False"}) {
        SCOPED_TRACE(std::string("ExploreFromCanonicalStatesOnly ") + explore);
        HostBridge host;
        const std::vector<std::string> props = {"EvolutionCausalGraph", "CausalGraph",
                                                "StatesGraph", "CausalGraphStructure"};
        auto opts = [&](wxf::Writer& w) {
            put_str_list_option(w, "GraphProperties", props);
            put_str_option(w, "CanonicalizeStates", "Full");
            put_str_option(w, "ExploreFromCanonicalStatesOnly", explore);
        };
        auto job = [&](int64_t steps, const char* op, int64_t h, bool delta) {
            return session_envelope(kBranchSeed, kBranchLhs, kBranchRhs, steps, op, h,
                                    std::string(op) == "Open", {}, opts, 3, delta);
        };
        const auto opened = run_rewriting_core(job(1, "Open", 0, true), host);
        const int64_t h = read_int_key(opened, "Session");
        ASSERT_GT(h, 0);
        std::map<std::string, DeliveredGraph> merged;
        for (const auto& p : props) merged[p] = delivered_graph(opened, p);
        for (int step = 0; step < 4; ++step) {
            const auto out = run_rewriting_core(job(1, "Step", h, true), host);
            EXPECT_FALSE(reply_mentions(out, "OptionSkipped")) << "step " << step;
            for (const auto& p : props) {
                DeliveredGraph d = delivered_graph(out, p);
                // The Open's reply was not a session delivery, so the first Step's is whole.
                EXPECT_EQ(d.is_delta, step == 0 ? 0 : 1) << p << " step " << step;
                DeliveredGraph& m = merged[p];
                if (d.is_delta != 1) m = DeliveredGraph{};
                m.vertices.insert(d.vertices.begin(), d.vertices.end());
                m.edges.insert(d.edges.begin(), d.edges.end());
                for (auto& [k, v] : d.vertex_data) m.vertex_data[k] = v;
            }
        }
        const auto full = run_rewriting_core(job(0, "Query", h, false), host);
        auto missing = [](const auto& from, const auto& in) {
            size_t n = 0;
            for (const auto& x : from) n += in.count(x) == 0;
            return n;
        };
        for (const auto& p : props) {
            const DeliveredGraph f = delivered_graph(full, p);
            const DeliveredGraph& m = merged[p];
            EXPECT_EQ(f.is_delta, 0) << p;
            EXPECT_GT(f.edges.size(), 0u) << p;
            EXPECT_EQ(missing(m.vertices, f.vertices), 0u) << p << ": delivered, not in the graph";
            EXPECT_EQ(missing(f.vertices, m.vertices), 0u) << p << ": in the graph, not delivered";
            EXPECT_EQ(missing(m.edges, f.edges), 0u) << p << ": delivered, not in the graph";
            EXPECT_EQ(missing(f.edges, m.edges), 0u) << p << ": in the graph, not delivered";
            size_t records = 0;
            for (const auto& [k, v] : f.vertex_data) {
                const auto it = m.vertex_data.find(k);
                records += it == m.vertex_data.end() || it->second != v;
            }
            EXPECT_EQ(records, 0u) << p << ": vertex records that differ from the graph's";
        }
        run_rewriting_core(job(0, "Close", h, false), host);
    }
}

// A quotient or full-capture session with "ShowGenesisEvents" -> True, stepped three times with
// "Delivery" -> "Delta": each Step's delivery is an increment (no vertex or edge the caller already
// holds is sent again), and the merged deliveries equal a full Query's vertices, edges and vertex
// data. A genesis event and the genesis state keep their ids across Steps.
TEST(Session, ADeltaWithGenesisEvents) {
    for (const char* explore : {"True", "False"}) {
        SCOPED_TRACE(std::string("ExploreFromCanonicalStatesOnly ") + explore);
        HostBridge host;
        const std::vector<std::string> props = {"CausalGraph", "EvolutionCausalGraph"};
        auto opts = [&](wxf::Writer& w) {
            put_str_list_option(w, "GraphProperties", props);
            put_str_option(w, "CanonicalizeStates", "Full");
            put_str_option(w, "ExploreFromCanonicalStatesOnly", explore);
            put_str_option(w, "ShowGenesisEvents", "True");
        };
        auto job = [&](int64_t steps, const char* op, int64_t h, bool delta) {
            return session_envelope(kBranchSeed, kBranchLhs, kBranchRhs, steps, op, h,
                                    std::string(op) == "Open", {}, opts, 4, delta);
        };
        const auto opened = run_rewriting_core(job(1, "Open", 0, true), host);
        const int64_t h = read_int_key(opened, "Session");
        ASSERT_GT(h, 0);
        std::map<std::string, DeliveredGraph> merged;
        // The first Step's delivery stands alone (the Open's graph is not recorded by the
        // session's cursor); every later one is an increment on what came before.
        for (int step = 0; step < 3; ++step) {
            const auto out = run_rewriting_core(job(1, "Step", h, true), host);
            EXPECT_FALSE(reply_mentions(out, "OptionSkipped")) << "step " << step;
            for (const auto& p : props) {
                DeliveredGraph d = delivered_graph(out, p);
                if (step > 0) EXPECT_EQ(d.is_delta, 1) << p << " step " << step;
                DeliveredGraph& m = merged[p];
                if (d.is_delta != 1) m = DeliveredGraph{};
                for (const auto& v : d.vertices)
                    EXPECT_EQ(m.vertices.count(v), 0u) << p << " step " << step << " resent a vertex";
                for (const auto& e : d.edges)
                    EXPECT_EQ(m.edges.count(e), 0u) << p << " step " << step << " resent an edge";
                EXPECT_FALSE(d.vertices.empty() && d.edges.empty()) << p << " step " << step;
                m.vertices.insert(d.vertices.begin(), d.vertices.end());
                m.edges.insert(d.edges.begin(), d.edges.end());
                for (auto& [k, v] : d.vertex_data) m.vertex_data[k] = v;
            }
        }
        const auto full = run_rewriting_core(job(0, "Query", h, false), host);
        for (const auto& p : props) {
            const DeliveredGraph f = delivered_graph(full, p);
            const DeliveredGraph& m = merged[p];
            EXPECT_EQ(m.vertices, f.vertices) << p;
            EXPECT_EQ(m.edges, f.edges) << p;
            EXPECT_EQ(m.vertex_data, f.vertex_data) << p;
        }
        run_rewriting_core(job(0, "Close", h, false), host);
    }
}

// The branchial graph keeps every branchial pair as an edge, so with all steps shown its edge
// count is NumBranchialEdges. Under CanonicalizeStates -> Full sibling pairs reach the same two
// classes, and a repeated pair in the same order was sent once while one in the other order was
// kept, so the count depended on event order.
TEST(WxfSerializationPin, BranchialGraphKeepsEveryPair) {
    HostBridge host;
    auto in = build_input(kBranchSeed, kBranchLhs, kBranchRhs, 3, [](wxf::Writer& w) {
        put_str_list_option(w, "GraphProperties", {"BranchialGraph"});
        put_str_option(w, "CanonicalizeStates", "Full");
        put_str_list_option(w, "RequestedData", {"NumBranchialEdges"});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("BranchialStep"));
        w.write(int64_t{0});
    }, 4);
    const auto out = run_rewriting_core(in, host);
    ASSERT_FALSE(out.empty());
    const int64_t pairs = read_int_key(out, "NumBranchialEdges");
    ASSERT_GT(pairs, 0);
    EXPECT_EQ(graph_edge_count(out), pairs);
}

// A held verb is served under its session's settings on either device: a Query asking for
// another state canonicalization reports the session's states, and a Step asking for another
// one and another transitive reduction reports what a Query under the session's settings does.
// The device served both under the request's settings.
TEST(Session, AHeldVerbIsServedUnderTheSessionsSettings) {
    HostBridge host;
    auto with = [](const char* canon, bool tr) {
        return [canon, tr](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"States", "NumStates", "NumCausalEdges"});
            put_str_option(w, "CanonicalizeStates", canon);
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("CausalTransitiveReduction"));
            w.write_symbol(tr ? "True" : "False");
        };
    };
    const auto opened = run_rewriting_core(branch_job(3, "Open", 0, with("None", true), 3), host);
    const int64_t h = read_int_key(opened, "Session");
    ASSERT_GT(h, 0);
    const auto same = run_rewriting_core(branch_job(0, "Query", h, with("None", true), 3), host);
    const auto other = run_rewriting_core(branch_job(0, "Query", h, with("Full", true), 3), host);
    EXPECT_EQ(read_int_key(other, "NumStates"), read_int_key(same, "NumStates"));
    EXPECT_EQ(count_assoc_entries(other, "States"), count_assoc_entries(same, "States"));
    EXPECT_TRUE(other == same) << "a Query asking for Full served a different reply";
    const auto stepped = run_rewriting_core(branch_job(1, "Step", h, with("Full", false), 3), host);
    const auto after = run_rewriting_core(branch_job(0, "Query", h, with("None", true), 3), host);
    EXPECT_EQ(read_int_key(stepped, "NumStates"), read_int_key(after, "NumStates"));
    EXPECT_EQ(read_int_key(stepped, "NumCausalEdges"), read_int_key(after, "NumCausalEdges"));
    run_rewriting_core(branch_job(0, "Close", h, with("None", true), 3), host);
}

namespace {

const wxf::WXFValue* assoc_at(const wxf::WXFValue& v, const std::string& key) {
    const auto* a = std::get_if<wxf::WXFValueAssociation>(&v.data);
    if (!a) return nullptr;
    for (const auto& [k, x] : *a) {
        const auto* s = std::get_if<std::string>(&k.data);
        if (s && *s == key) return &x;
    }
    return nullptr;
}

double number_at(const wxf::WXFValue& v, const std::string& key) {
    const wxf::WXFValue* x = assoc_at(v, key);
    if (!x) return std::nan("");
    if (const auto* d = std::get_if<double>(&x->data)) return *d;
    if (const auto* i = std::get_if<int64_t>(&x->data)) return static_cast<double>(*i);
    return std::nan("");
}

bool bytes_contain(const std::vector<uint8_t>& bytes, const std::string& s) {
    return std::search(bytes.begin(), bytes.end(), s.begin(), s.end()) != bytes.end();
}

// Each step's branchial keys through hg::stats::BranchialStepWork, the overlap rows split into at
// most `parts` parts run on this thread in reverse order. Step 0's vertices are S0.
std::map<uint32_t, wxf::WXFValueAssociation> branchial_keys(
    const std::map<uint32_t, hg::stats::BranchialStep>& steps, uint32_t which, size_t parts = 4) {
    std::vector<uint32_t> initial;
    const bool have_initial = steps.count(0) != 0;
    if (have_initial) initial = hg::stats::initial_vertices(steps.at(0));
    std::map<uint32_t, wxf::WXFValueAssociation> out;
    for (const auto& [step, s] : steps) {
        hg::stats::BranchialStepWork work(s, which, have_initial ? &initial : nullptr);
        work.prepare(parts);
        for (size_t p = work.parts(); p-- > 0;) work.rows(p);
        out[step] = work.finish();
    }
    return out;
}

// The "StepStatistics" reply of a population given as points, built by
// hg::stats::StepStatisticsJobs with every job run on this thread.
wxf::WXFValue pooled_steps(
    const std::vector<hg::stats::StepPoint>& points,
    const std::unordered_map<uint64_t, const hgcommon::StateInvariantRecord*>& recs,
    bool by_class) {
    std::map<uint32_t, hg::stats::StepInput> inputs;
    for (const auto& p : points) {
        const auto r = recs.find(p.class_hash);
        hg::stats::add_class(inputs[p.step], p.class_hash, p.weight,
                             r == recs.end() ? nullptr : r->second);
    }
    hg::stats::StepStatisticsJobs jobs(by_class, 0, 1, [](std::function<void()> f) { f(); });
    for (auto& [step, in] : inputs) jobs.submit(step, std::move(in));
    return jobs.reply();
}

}  // namespace

// The per-step branchial metrics of one step worked by hand: states 10 and 11 joined (the pair
// listed both ways), 12 alone (its pair with itself adds nothing). Vertex sets {0,1,2,5},
// {0,1,2,6} and {7}: overlaps 3/5, 0, 0; vertices 0, 1, 2 held twice, 5, 6, 7 once.
TEST(StateStatistics, BranchialStepMetricsOfASmallStep) {
    hg::stats::BranchialStep s;
    s.nodes = {10, 11, 12};
    s.pairs = {{10, 11}, {11, 10}, {12, 12}};
    s.vertex_sets = {{0, 1, 2, 5}, {0, 1, 2, 6}, {7}};
    const auto out = branchial_keys(
        {{1u, s}}, hg::stats::kBranchialGraph | hg::stats::kBranchialOverlap |
                       hg::stats::kBranchialMeasures);
    ASSERT_EQ(out.size(), 1u);
    const wxf::WXFValue rec(out.at(1));
    ASSERT_NE(assoc_at(rec, "BranchialDegree"), nullptr);
    EXPECT_EQ(number_at(*assoc_at(rec, "BranchialDegree"), "N"), 3);
    EXPECT_DOUBLE_EQ(number_at(*assoc_at(rec, "BranchialDegree"), "Mean"), 2.0 / 3.0);
    EXPECT_EQ(number_at(*assoc_at(rec, "BranchialDistance"), "N"), 1);
    EXPECT_EQ(number_at(*assoc_at(rec, "BranchialDistance"), "Max"), 1);
    EXPECT_EQ(number_at(rec, "BranchialComponents"), 2);
    EXPECT_DOUBLE_EQ(number_at(rec, "BranchialDimension"), 1.0);   // K2: log 2 / log 2
    // K2: the lazy walks put half on each end (curvature 1), the non-lazy walks swap the ends
    // (0); no state has eccentricity 3, so no volume-growth key.
    EXPECT_DOUBLE_EQ(number_at(rec, "BranchialOllivierRicciCurvature"), 1.0);
    EXPECT_DOUBLE_EQ(number_at(rec, "BranchialNonLazyOllivierRicciCurvature"), 0.0);
    EXPECT_EQ(assoc_at(rec, "BranchialSphereDimension"), nullptr);
    EXPECT_EQ(number_at(*assoc_at(rec, "StateOverlap"), "N"), 3);
    EXPECT_DOUBLE_EQ(number_at(*assoc_at(rec, "StateOverlap"), "Mean"), 0.2);
    EXPECT_EQ(number_at(*assoc_at(rec, "VertexSharpness"), "N"), 6);
    EXPECT_DOUBLE_EQ(number_at(*assoc_at(rec, "VertexSharpness"), "Mean"), 0.75);
    EXPECT_DOUBLE_EQ(number_at(*assoc_at(rec, "BranchEntropy"), "Mean"), 0.5);
}

// A step whose branchial graph is the cycle of 8 states: every shell up to the antipode has two
// states, so the sphere probe reads dimension 1 and scalar curvature 0, and both Ollivier-Ricci
// curvatures of a long cycle are 0.
TEST(StateStatistics, BranchialCycleMeasures) {
    hg::stats::BranchialStep s;
    for (int64_t i = 0; i < 8; ++i) {
        s.nodes.push_back(20 + i);
        s.pairs.push_back({20 + i, 20 + (i + 1) % 8});
        s.vertex_sets.push_back({static_cast<uint32_t>(i)});
    }
    const auto out =
        branchial_keys({{1u, s}}, hg::stats::kBranchialGraph | hg::stats::kBranchialMeasures);
    ASSERT_EQ(out.size(), 1u);
    const wxf::WXFValue rec(out.at(1));
    // Without kBranchialMeasures the step has the graph keys only.
    const wxf::WXFValue plain(branchial_keys({{1u, s}}, hg::stats::kBranchialGraph).at(1));
    EXPECT_NE(assoc_at(plain, "BranchialDimension"), nullptr);
    EXPECT_EQ(assoc_at(plain, "BranchialSphereDimension"), nullptr);
    EXPECT_EQ(assoc_at(plain, "BranchialOllivierRicciCurvature"), nullptr);
    EXPECT_NEAR(number_at(rec, "BranchialSphereDimension"), 1.0, 1e-12);
    EXPECT_NEAR(number_at(rec, "BranchialSphereScalarCurvature"), 0.0, 1e-12);
    EXPECT_NE(assoc_at(rec, "BranchialRiemannianBallDimension"), nullptr);
    EXPECT_DOUBLE_EQ(number_at(rec, "BranchialOllivierRicciCurvature"), 0.0);
    EXPECT_DOUBLE_EQ(number_at(rec, "BranchialNonLazyOllivierRicciCurvature"), 0.0);
}

// "StepStatisticsMeasures": a step's record has a family's keys only when the family is asked
// for. The record of a state computed with every family gives, under the base set, the 25
// "Invariants" and 3 "VertexInvariants" of the base set and no "EdgeInvariants" or
// "LogGeodesicCount"; under "Particles" the 4 particle entries and its 2 distributions more.
TEST(StateStatistics, MeasuresFamiliesSelectTheKeys) {
    std::vector<uint64_t> storage;
    const hgcommon::StateInvariantRecord* rec =
        hg::stats::invariant_record({{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 1}}, storage);
    hg::stats::StepInput in;
    hg::stats::add_class(in, 7, 1, rec);
    auto keys = [](const wxf::WXFValue& a) {
        std::vector<std::string> k;
        for (const auto& [key, v] : std::get<wxf::WXFValueAssociation>(a.data))
            k.push_back(std::get<std::string>(key.data));
        return k;
    };
    const wxf::WXFValue base(hg::stats::step_record(1, in, false));
    EXPECT_EQ(keys(*assoc_at(base, "Invariants")).size(), 25u);
    EXPECT_EQ(keys(*assoc_at(base, "VertexInvariants")).size(), 3u);
    EXPECT_EQ(assoc_at(base, "EdgeInvariants"), nullptr);
    EXPECT_EQ(assoc_at(base, "LogGeodesicCount"), nullptr);
    const wxf::WXFValue particles(hg::stats::step_record(
        1, in, false, hgcommon::SG_MEASURES_BASE | hg::stats::measure_bits("Particles")));
    EXPECT_EQ(keys(*assoc_at(particles, "Invariants")).size(), 29u);
    EXPECT_EQ(keys(*assoc_at(particles, "VertexInvariants")).size(), 5u);
    EXPECT_NE(assoc_at(*assoc_at(particles, "Invariants"), "TopologicalCharge"), nullptr);
    const wxf::WXFValue all(hg::stats::step_record(1, in, false, hgcommon::SG_MEASURES_ALL));
    EXPECT_EQ(keys(*assoc_at(all, "Invariants")).size(), 42u);
    EXPECT_NE(assoc_at(all, "EdgeInvariants"), nullptr);
    EXPECT_NE(assoc_at(all, "LogGeodesicCount"), nullptr);
    EXPECT_EQ(hg::stats::measure_bits("Branchial"), hg::stats::kMeasureBranchial);
    EXPECT_EQ(hg::stats::measure_bits("Curvature"), 0u);
}

// The pairwise, initial-state and edge metrics of the same step worked by hand. Step 0 is one
// state with vertices {0, 1} (S0). Step 1: A = {0,1,2,5}, B = {0,1,2,6}, C = {7}, U = {0,1,2,5,6,7};
// edges {100,101}, {100,102}, {103}.
//   cosine: A,B 3/sqrt(16) = 0.75; A,C and B,C 0.
//   MI(A;B): (n11, n10, n01, n00) = (3, 1, 1, 1) over 6, 3/2 log2 3 - 7/3.
//   MI(A;C) = MI(B;C): (0, 4, 1, 1) over 6, 2/3 + log2 3 - 5/6 log2 5.
//   initial: U ∪ S0 has 6 elements; MI(A;S0) = MI(B;S0): (2, 2, 0, 2), log2 3 - 4/3.
//   edges: 100 held twice, 101..103 once: EdgeSharpness mean (1/2 + 3)/4, EdgeBranchEntropy 1/4.
//   OverlapByBranchialDistance: only A-B joined, at distance 1, Jaccard 3/5.
TEST(StateStatistics, PairwiseInitialAndEdgeMetricsOfASmallStep) {
    hg::stats::BranchialStep s0, s1;
    s0.nodes = {1};
    s0.vertex_sets = {{0, 1}};
    s0.edge_sets = {{50}};
    s1.nodes = {10, 11, 12};
    s1.pairs = {{10, 11}};
    s1.vertex_sets = {{0, 1, 2, 5}, {0, 1, 2, 6}, {7}};
    s1.edge_sets = {{100, 101}, {100, 102}, {103}};
    const auto out = branchial_keys(
        {{0u, s0}, {1u, s1}}, hg::stats::kBranchialGraph | hg::stats::kBranchialOverlap);
    const wxf::WXFValue rec(out.at(1));
    // One cell p log2(p / (pa pb)) of a 2x2 table over u elements.
    auto cell = [](double n, double na, double nb, double u) {
        return n == 0 ? 0.0 : n / u * std::log2(n * u / (na * nb));
    };
    auto mi = [&](double both, double a, double b, double u) {
        return cell(both, a, b, u) + cell(a - both, a, u - b, u) + cell(b - both, u - a, b, u) +
               cell(u - a - b + both, u - a, u - b, u);
    };
    const double ab = 1.5 * std::log2(3.0) - 7.0 / 3.0;
    EXPECT_NEAR(mi(3, 4, 4, 6), ab, 1e-15);
    const double ac = mi(0, 4, 1, 6);
    EXPECT_NEAR(ac, 2.0 / 3.0 + std::log2(3.0) - 5.0 / 6.0 * std::log2(5.0), 1e-15);
    const auto* cos = assoc_at(rec, "StateCosineSimilarity");
    ASSERT_NE(cos, nullptr);
    EXPECT_EQ(number_at(*cos, "N"), 3);
    EXPECT_DOUBLE_EQ(number_at(*cos, "Mean"), 0.25);
    EXPECT_DOUBLE_EQ(number_at(*cos, "Max"), 0.75);
    const auto* inf = assoc_at(rec, "StateMutualInformation");
    ASSERT_NE(inf, nullptr);
    EXPECT_EQ(number_at(*inf, "N"), 3);
    EXPECT_NEAR(number_at(*inf, "Mean"), (ab + 2 * ac) / 3, 1e-14);
    EXPECT_NEAR(number_at(*inf, "Min"), ab, 1e-14);
    const auto* ini = assoc_at(rec, "InitialStateMutualInformation");
    ASSERT_NE(ini, nullptr);
    EXPECT_EQ(number_at(*ini, "N"), 3);
    const double a0 = std::log2(3.0) - 4.0 / 3.0;
    EXPECT_NEAR(mi(2, 4, 2, 6), a0, 1e-15);
    EXPECT_NEAR(number_at(*ini, "Mean"), (2 * a0 + mi(0, 1, 2, 6)) / 3, 1e-14);
    const auto* es = assoc_at(rec, "EdgeSharpness");
    ASSERT_NE(es, nullptr);
    EXPECT_EQ(number_at(*es, "N"), 4);
    EXPECT_DOUBLE_EQ(number_at(*es, "Mean"), 0.875);
    EXPECT_DOUBLE_EQ(number_at(*assoc_at(rec, "EdgeBranchEntropy"), "Mean"), 0.25);
    const auto* by = assoc_at(rec, "OverlapByBranchialDistance");
    ASSERT_NE(by, nullptr);
    const auto& per = std::get<wxf::WXFValueAssociation>(by->data);
    ASSERT_EQ(per.size(), 1u);
    EXPECT_EQ(std::get<int64_t>(per[0].first.data), 1);
    EXPECT_EQ(number_at(per[0].second, "N"), 1);
    EXPECT_DOUBLE_EQ(number_at(per[0].second, "Mean"), 0.6);
    // Step 0 against itself: one state, so no pair, and S0 carries no information about itself.
    const wxf::WXFValue rec0(out.at(0));
    EXPECT_EQ(number_at(*assoc_at(rec0, "StateCosineSimilarity"), "N"), 0);
    EXPECT_DOUBLE_EQ(number_at(*assoc_at(rec0, "InitialStateMutualInformation"), "Mean"), 0.0);
}

// A path of three states 20 - 21 - 22 with vertex sets {0,1}, {1,2}, {2,3}: Jaccard 1/3 at
// distance 1 (twice) and 0 at distance 2, a pair that shares no vertex. Without "Graph" the key
// is absent; without "Overlap" so are the overlap keys.
TEST(StateStatistics, OverlapByBranchialDistanceCountsDisjointPairsOfAComponent) {
    hg::stats::BranchialStep s;
    s.nodes = {20, 21, 22};
    s.pairs = {{20, 21}, {22, 21}};
    s.vertex_sets = {{0, 1}, {1, 2}, {2, 3}};
    s.edge_sets = {{0}, {1}, {2}};
    const auto both = branchial_keys(
        {{1u, s}}, hg::stats::kBranchialGraph | hg::stats::kBranchialOverlap);
    const wxf::WXFValue rec(both.at(1));
    const auto& per =
        std::get<wxf::WXFValueAssociation>(assoc_at(rec, "OverlapByBranchialDistance")->data);
    ASSERT_EQ(per.size(), 2u);
    EXPECT_EQ(std::get<int64_t>(per[0].first.data), 1);
    EXPECT_EQ(number_at(per[0].second, "N"), 2);
    EXPECT_DOUBLE_EQ(number_at(per[0].second, "Mean"), 1.0 / 3.0);
    EXPECT_EQ(std::get<int64_t>(per[1].first.data), 2);
    EXPECT_EQ(number_at(per[1].second, "N"), 1);
    EXPECT_DOUBLE_EQ(number_at(per[1].second, "Mean"), 0.0);
    // StateOverlap over all three pairs is unchanged by the distance pass: (1/3 + 1/3 + 0) / 3.
    EXPECT_EQ(number_at(*assoc_at(rec, "StateOverlap"), "N"), 3);
    EXPECT_DOUBLE_EQ(number_at(*assoc_at(rec, "StateOverlap"), "Mean"), 2.0 / 9.0);
    const wxf::WXFValue overlap_only(
        branchial_keys({{1u, s}}, hg::stats::kBranchialOverlap).at(1));
    EXPECT_EQ(assoc_at(overlap_only, "OverlapByBranchialDistance"), nullptr);
    EXPECT_NE(assoc_at(overlap_only, "StateMutualInformation"), nullptr);
    const wxf::WXFValue graph_only(
        branchial_keys({{1u, s}}, hg::stats::kBranchialGraph).at(1));
    EXPECT_EQ(assoc_at(graph_only, "OverlapByBranchialDistance"), nullptr);
    EXPECT_EQ(assoc_at(graph_only, "EdgeSharpness"), nullptr);
}

// The branchial values are functions of counts and read the same from any row split: a step of
// 300 states gives the same record with its rows in 1, 2, 3 or 5 parts.
TEST(StateStatistics, BranchialMetricsDoNotDependOnTheRowSplit) {
    hg::stats::BranchialStep s;
    for (uint32_t i = 0; i < 300; ++i) {
        s.nodes.push_back(i);
        s.vertex_sets.push_back({i % 7, 7 + i % 11, 18 + i % 13, 31 + i});
        s.edge_sets.push_back({i % 5, 5 + i % 9, 14 + i});
        if (i) s.pairs.push_back({i - 1, i});
    }
    const uint32_t all = hg::stats::kBranchialGraph | hg::stats::kBranchialOverlap;
    auto bytes = [&](wxf::WXFValue& v, size_t parts) {
        v = wxf::WXFValue(branchial_keys({{1u, s}}, all, parts).at(1));
        wxf::Writer w;
        w.write(v);
        return w.release_data();
    };
    wxf::WXFValue first, again;
    const auto expected = bytes(first, 1);
    for (size_t parts : {2, 3, 5}) EXPECT_EQ(bytes(again, parts), expected) << parts;
    EXPECT_EQ(number_at(*assoc_at(first, "StateMutualInformation"), "N"), 300 * 299 / 2);
}

namespace {

// A "StateMatrices" value (WXF bytes) read back: its "States" and each matrix, in key order, and
// whether each was a SparseArray.
struct ReadMatrices {
    std::vector<int64_t> states;
    std::vector<std::pair<std::string, std::vector<std::vector<double>>>> m;
    std::vector<bool> sparse;
    const std::vector<std::vector<double>>* at(const std::string& name) const {
        for (const auto& [k, v] : m)
            if (k == name) return &v;
        return nullptr;
    }
};
ReadMatrices read_matrices(const wxf::WXFValue& v) {
    ReadMatrices r;
    wxf::Parser p(std::get<wxf::Encoded>(v.data).bytes);
    p.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k == "States") {
            r.states = vp.read<std::vector<int64_t>>();
            return;
        }
        bool sparse = false;
        r.m.push_back({k, read_real_matrix(vp, &sparse)});
        r.sparse.push_back(sparse);
    });
    return r;
}

uint32_t all_matrices() { return hg::stats::kBranchialMatrices; }

}  // namespace

// The state matrices of the step of PairwiseInitialAndEdgeMetricsOfASmallStep worked by hand.
// States 10, 11, 12: A = {0,1,2,5}, B = {0,1,2,6}, C = {7}, U = 6 vertices; edges A {100,101},
// B {100,102}, C {103}, 4 edges.
//   StateOverlap: A,B 3/5; A,C and B,C 0; 1 on the diagonal.
//   StateCosineSimilarity: A,B 3/4; 0 off it; 1 on the diagonal.
//   StateMutualInformation: A,B (3,1,1,1) over 6; A,C (0,4,1,1) and B,C the same; A,A is the
//     entropy of 4 of 6, C,C of 1 of 6.
//   StateEdgeOverlap: A,B 1/3; StateEdgeCosineSimilarity: A,B 1/2;
//   StateEdgeMutualInformation: A,B (1,1,1,1) over 4, which is 0; A,C (0,2,1,1) over 4.
TEST(StateStatistics, StateMatricesOfASmallStep) {
    hg::stats::BranchialStep s;
    s.nodes = {12, 10, 11};
    s.pairs = {{10, 11}};
    s.vertex_sets = {{7}, {0, 1, 2, 5}, {0, 1, 2, 6}};
    s.edge_sets = {{103}, {100, 101}, {100, 102}};
    const auto out = branchial_keys({{1u, s}}, all_matrices() | hg::stats::kBranchialOverlap);
    const wxf::WXFValue rec(out.at(1));
    ASSERT_NE(assoc_at(rec, "StateMatrices"), nullptr);
    const ReadMatrices m = read_matrices(*assoc_at(rec, "StateMatrices"));
    // The states ascending by id, whatever order the step listed them in.
    EXPECT_EQ(m.states, (std::vector<int64_t>{10, 11, 12}));
    ASSERT_EQ(m.m.size(), 6u);
    auto cell = [](double n, double na, double nb, double u) {
        return n == 0 ? 0.0 : n / u * std::log2(n * u / (na * nb));
    };
    auto mi = [&](double both, double a, double b, double u) {
        return cell(both, a, b, u) + cell(a - both, a, u - b, u) + cell(b - both, u - a, b, u) +
               cell(u - a - b + both, u - a, u - b, u);
    };
    const std::vector<std::pair<std::string, std::vector<double>>> want = {
        {"StateOverlap", {1, 0.6, 0, 0.6, 1, 0, 0, 0, 1}},
        {"StateCosineSimilarity", {1, 0.75, 0, 0.75, 1, 0, 0, 0, 1}},
        {"StateMutualInformation",
         {mi(4, 4, 4, 6), mi(3, 4, 4, 6), mi(0, 4, 1, 6), mi(3, 4, 4, 6), mi(4, 4, 4, 6),
          mi(0, 4, 1, 6), mi(0, 4, 1, 6), mi(0, 4, 1, 6), mi(1, 1, 1, 6)}},
        {"StateEdgeOverlap", {1, 1.0 / 3, 0, 1.0 / 3, 1, 0, 0, 0, 1}},
        {"StateEdgeCosineSimilarity", {1, 0.5, 0, 0.5, 1, 0, 0, 0, 1}},
        {"StateEdgeMutualInformation",
         {mi(2, 2, 2, 4), 0, mi(0, 2, 1, 4), 0, mi(2, 2, 2, 4), mi(0, 2, 1, 4), mi(0, 2, 1, 4),
          mi(0, 2, 1, 4), mi(1, 1, 1, 4)}},
    };
    EXPECT_NEAR(mi(3, 4, 4, 6), 1.5 * std::log2(3.0) - 7.0 / 3.0, 1e-15);
    EXPECT_NEAR(mi(4, 4, 4, 6), std::log2(3.0) - 2.0 / 3.0, 1e-15);   // H(2/3)
    for (size_t n = 0; n < want.size(); ++n) {
        const auto& [name, values] = want[n];
        EXPECT_EQ(m.m[n].first, name);
        const auto* got = m.at(name);
        ASSERT_NE(got, nullptr) << name;
        ASSERT_EQ(got->size(), 3u) << name;
        for (size_t a = 0; a < 3; ++a) {
            ASSERT_EQ((*got)[a].size(), 3u) << name;
            for (size_t b = 0; b < 3; ++b) {
                EXPECT_NEAR((*got)[a][b], values[a * 3 + b], 1e-15) << name << a << b;
                EXPECT_EQ((*got)[a][b], (*got)[b][a]) << name;
            }
        }
    }
    // One of the 3 pairs shares a vertex and one shares an edge, under half: the overlap and the
    // cosine of both levels are SparseArrays, the mutual information packed arrays.
    EXPECT_EQ(m.sparse, (std::vector<bool>{true, true, false, true, true, false}));
    // The off-diagonal cells are the values the summaries read: their mean is the summary's.
    for (const char* name : {"StateOverlap", "StateCosineSimilarity", "StateMutualInformation"}) {
        const auto& g = *m.at(name);
        EXPECT_NEAR((g[0][1] + g[0][2] + g[1][2]) / 3, number_at(*assoc_at(rec, name), "Mean"),
                    1e-15)
            << name;
    }
    // Only the matrices asked for, after "States".
    const wxf::WXFValue one(branchial_keys(
        {{1u, s}}, hg::stats::kMatrixStateEdgeMutualInformation).at(1));
    const ReadMatrices m1 = read_matrices(*assoc_at(one, "StateMatrices"));
    ASSERT_EQ(m1.m.size(), 1u);
    EXPECT_EQ(m1.m[0].first, "StateEdgeMutualInformation");
    EXPECT_EQ(assoc_at(one, "StateOverlap"), nullptr);
}

// The overlap and the cosine of a level are SparseArrays when fewer than half of the step's pairs
// share an element (hg::stats::sparse_matrices). Of the 6 pairs of 4 states, 2 sharing gives
// SparseArrays and 3 packed arrays; each level decides for itself.
TEST(StateStatistics, StateMatricesAreSparseBelowHalfThePairs) {
    for (const bool three : {false, true}) {
        hg::stats::BranchialStep s;
        s.nodes = {1, 2, 3, 4};
        // Vertices: (1,2) share 0 and (3,4) share 1; with `three`, (1,3) also share 9.
        s.vertex_sets = {{0, 9}, {0}, {1, three ? 9u : 8u}, {1}};
        // Edges: only (1,2) share one.
        s.edge_sets = {{5}, {5}, {6}, {7}};
        const wxf::WXFValue rec(branchial_keys({{1u, s}}, all_matrices()).at(1));
        const ReadMatrices m = read_matrices(*assoc_at(rec, "StateMatrices"));
        EXPECT_EQ(m.sparse, (std::vector<bool>{!three, !three, false, true, true, false}))
            << three;
        // Either form reads back as the cells of the matrix.
        const auto& overlap = *m.at("StateOverlap");
        EXPECT_EQ(overlap[0][1], 0.5);
        EXPECT_EQ(overlap[0][2], three ? 1.0 / 3 : 0.0);
        EXPECT_EQ(overlap[1][3], 0.0);
        EXPECT_EQ(overlap[3][3], 1.0);
        EXPECT_EQ(m.at("StateEdgeCosineSimilarity")->at(1).at(0), 1.0);
    }
    hgcommon::BoMatrices none;
    EXPECT_FALSE(hg::stats::sparse_matrices(none));
}

// A step of one state has 1 x 1 matrices: the overlap and the cosine 1, the mutual information
// 0 (the state is U), over vertices and over edges, with and without the overlap keys. The step
// has no pair, so its triangle of shared counts is empty.
TEST(StateStatistics, StateMatricesOfOneState) {
    hg::stats::BranchialStep s;
    s.nodes = {4};
    s.vertex_sets = {{0, 1, 2}};
    s.edge_sets = {{7, 8}};
    for (const uint32_t which : {all_matrices(), all_matrices() | hg::stats::kBranchialOverlap}) {
        const wxf::WXFValue rec(branchial_keys({{1u, s}}, which).at(1));
        ASSERT_NE(assoc_at(rec, "StateMatrices"), nullptr) << which;
        const ReadMatrices m = read_matrices(*assoc_at(rec, "StateMatrices"));
        EXPECT_EQ(m.states, (std::vector<int64_t>{4}));
        ASSERT_EQ(m.m.size(), 6u);
        for (size_t i = 0; i < 6; ++i)
            EXPECT_EQ(m.m[i].second, (std::vector<std::vector<double>>{{i % 3 == 2 ? 0.0 : 1.0}}))
                << m.m[i].first << " which " << which;
    }
}

// "StepStatisticsMatrixLimit" is in bytes (hg::stats::matrix_bytes): a step of 3 states with all
// six matrices needs 6 x 16 x 9 + 2 x 2 x 6 = 888 bytes, so it has no "StateMatrices" under a
// limit of 887 and reports its state count, and has them under 888. 0 is Automatic, a quarter of
// the available memory.
TEST(StateStatistics, StateMatricesRespectTheLimit) {
    hg::stats::BranchialStep s;
    s.nodes = {10, 11, 12};
    s.vertex_sets = {{0, 1}, {1, 2}, {2, 3}};
    s.edge_sets = {{0}, {1}, {2}};
    EXPECT_EQ(hg::stats::matrix_bytes(3, all_matrices()), 888u);
    EXPECT_EQ(hg::stats::matrix_bytes(3, hg::stats::kMatrixStateEdgeOverlap), 144u + 12u);
    EXPECT_EQ(hg::stats::matrix_bytes(1u << 29, all_matrices()), UINT64_MAX);
    for (const uint64_t limit : {887u, 888u, 0u}) {
        hg::stats::BranchialStepWork work(s, all_matrices(), nullptr, limit);
        work.prepare(2);
        for (size_t p = 0; p < work.parts(); ++p) work.rows(p);
        const wxf::WXFValue rec(work.finish());
        EXPECT_EQ(assoc_at(rec, "StateMatrices") != nullptr, limit != 887) << limit;
        EXPECT_EQ(work.matrices_over_limit(), limit != 887 ? 0u : 3u) << limit;
    }
    EXPECT_EQ(hg::stats::matrix_limit_bytes(0), hgcommon::host_topology().available_bytes / 4);
    EXPECT_EQ(hg::stats::matrix_limit_bytes(5), 5u);
    EXPECT_EQ(hg::stats::matrix_limit_warning(4, 1500, all_matrices(), 1000),
              "StepStatisticsMatrices: the StateMatrices of step 4 (1500 states) need 224994000 "
              "bytes, above StepStatisticsMatrixLimit 1000; they are not returned");
    hg::stats::StepStatisticsJobs jobs(false, all_matrices(), 1,
                                       [](std::function<void()> f) { f(); });
    jobs.set_matrix_limit(887);
    hg::stats::StepInput in;
    hg::stats::add_class(in, 1, 3, nullptr);
    in.branchial = s;
    jobs.submit(5, std::move(in));
    jobs.reply();
    EXPECT_EQ(jobs.matrices_over_limit(),
              (std::vector<std::pair<uint32_t, uint32_t>>{{5u, 3u}}));
}

// The state matrices read the same from any row split, with and without the overlap keys.
TEST(StateStatistics, StateMatricesDoNotDependOnTheRowSplit) {
    hg::stats::BranchialStep s;
    for (uint32_t i = 0; i < 300; ++i) {
        s.nodes.push_back(i);
        s.vertex_sets.push_back({i % 7, 7 + i % 11, 18 + i % 13, 31 + i});
        s.edge_sets.push_back({i % 5, 5 + i % 9, 14 + i});
        if (i) s.pairs.push_back({i - 1, i});
    }
    for (const uint32_t which :
         {all_matrices(), all_matrices() | hg::stats::kBranchialGraph | hg::stats::kBranchialOverlap}) {
        auto bytes = [&](size_t parts) {
            wxf::Writer w;
            w.write(wxf::WXFValue(branchial_keys({{1u, s}}, which, parts).at(1)));
            return w.release_data();
        };
        const auto expected = bytes(1);
        EXPECT_TRUE(bytes_contain(expected, "StateEdgeMutualInformation"));
        EXPECT_TRUE(bytes_contain(expected, "SparseArray"));
        for (size_t parts : {2, 3, 5}) EXPECT_EQ(bytes(parts), expected) << parts;
    }
}

// Writer::write_packed_reals gives a WXF PackedArray of Real64 (0xC1 0x23), rank 2, which reads
// back as the nested lists, -0.0 as 0.0; a zero dimension is written as lists.
TEST(StateStatistics, PackedRealsAreAPackedArray) {
    const double values[4] = {1.0, -0.0, 0.5, 2.0};
    wxf::Writer w;
    w.write_packed_reals({2, 2}, values);
    const auto bytes = w.release_data();
    ASSERT_GE(bytes.size(), 4u);
    EXPECT_EQ(bytes[0], 0xC1);
    EXPECT_EQ(bytes[1], 0x23);
    EXPECT_EQ(bytes[2], 2);
    wxf::Parser p(bytes);
    const auto back = p.read<std::vector<std::vector<double>>>();
    EXPECT_EQ(back, (std::vector<std::vector<double>>{{1.0, 0.0}, {0.5, 2.0}}));
    EXPECT_FALSE(std::signbit(back[0][1]));
    wxf::Writer e;
    e.write_packed_reals({0, 3}, nullptr);
    wxf::Parser q(e.data());
    EXPECT_TRUE(q.read<std::vector<std::vector<double>>>().empty());
}

#ifdef HG_GPU_BACKEND
// The overlap rows counted on the device (hg_gpu::branchial_rows_device) give the record the
// host's rows give, on steps of 300 and 5000 states with a branchial path and with the graph
// metrics off. With 5000 states some vertices are held by 2500 states, so a row's holder walk
// spans many warps' worth of lanes.
TEST(StateStatistics, DeviceRowsGiveTheHostRecord) {
    for (const uint32_t k : {300u, 5000u})
        for (const uint32_t which : {uint32_t{hg::stats::kBranchialOverlap},
                                     uint32_t{hg::stats::kBranchialGraph | hg::stats::kBranchialOverlap}}) {
            hg::stats::BranchialStep s;
            for (uint32_t i = 0; i < k; ++i) {
                s.nodes.push_back(i);
                s.vertex_sets.push_back({i % 2, 2 + i % 7, 9 + i % 11, 20 + i % 13, 40 + i});
                s.edge_sets.push_back({i % 5, 5 + i % 9, 14 + i});
                if (i && i % 17) s.pairs.push_back({i - 1, i});
            }
            auto bytes = [](const wxf::WXFValueAssociation& rec) {
                wxf::Writer w;
                w.write(wxf::WXFValue(rec));
                return w.release_data();
            };
            hg::stats::BranchialStepWork host(s, which, nullptr);
            host.prepare(4);
            for (size_t p = 0; p < host.parts(); ++p) host.rows(p);
            const auto want = bytes(host.finish());
            hg::stats::BranchialStepWork device(s, which, nullptr);
            device.prepare(4);
            const bool ran = hg_gpu::branchial_rows_device(
                device.rows_view(),
                [&device](uint32_t d, uint32_t sa, uint32_t sb, uint32_t sh, uint64_t n) {
                    device.add_counts(d, sa, sb, sh, n);
                });
            if (!ran) GTEST_SKIP() << "no usable device";
            EXPECT_EQ(bytes(device.finish()), want) << "k " << k << " which " << which;
        }
}

// The state matrices' shared counts made on the device (the edge rows with nothing counted, then
// the vertex rows with the overlap keys) give the host's record byte for byte, at 300 and 5000
// states. At 5000 states one matrix of each level keeps the reply at 400 MB.
TEST(StateStatistics, DeviceRowsGiveTheHostStateMatrices) {
    for (const uint32_t k : {300u, 5000u}) {
        const uint32_t matrices =
            k > 1000 ? hg::stats::kMatrixStateMutualInformation | hg::stats::kMatrixStateEdgeOverlap
                     : hg::stats::kBranchialMatrices;
        for (const uint32_t which : {matrices, matrices | hg::stats::kBranchialGraph |
                                                   hg::stats::kBranchialOverlap}) {
            hg::stats::BranchialStep s;
            for (uint32_t i = 0; i < k; ++i) {
                s.nodes.push_back(i);
                s.vertex_sets.push_back({i % 2, 2 + i % 7, 9 + i % 11, 20 + i % 13, 40 + i});
                s.edge_sets.push_back({i % 5, 5 + i % 9, 14 + i});
                if (i && i % 17) s.pairs.push_back({i - 1, i});
            }
            auto bytes = [](const wxf::WXFValueAssociation& rec) {
                wxf::Writer w;
                w.write(wxf::WXFValue(rec));
                return w.release_data();
            };
            const uint64_t limit = hg::stats::matrix_bytes(k, which);
            hg::stats::BranchialStepWork host(s, which, nullptr, limit);
            host.prepare(4);
            for (size_t p = 0; p < host.parts(); ++p) host.rows(p);
            const auto want = bytes(host.finish());
            hg::stats::BranchialStepWork device(s, which, nullptr, limit);
            device.prepare(4);
            ASSERT_NE(device.vertex_matrices(), nullptr);
            ASSERT_NE(device.edge_matrices(), nullptr);
            bool ran = hg_gpu::branchial_rows_device(device.edge_rows_view(), nullptr,
                                                     device.edge_matrices());
            if (!ran) GTEST_SKIP() << "no usable device";
            ran = hg_gpu::branchial_rows_device(
                device.rows_view(),
                [&device](uint32_t d, uint32_t sa, uint32_t sb, uint32_t sh, uint64_t n) {
                    device.add_counts(d, sa, sb, sh, n);
                },
                device.vertex_matrices());
            ASSERT_TRUE(ran);
            EXPECT_EQ(bytes(device.finish()), want) << "k " << k << " which " << which;
            // The edge sets share an element on under half of the pairs: SparseArrays.
            EXPECT_TRUE(bytes_contain(want, "SparseArray")) << k;
        }
    }
}
#endif

// "StepStatistics" under full capture, from hgffi::StepStatisticsCollector on rule
// {{x,y},{x,z}} -> {{x,y},{x,w},{y,w},{z,w}} from {{0,1},{0,2}} at 5 steps: the reply's bytes at
// `threads` workers, and the number of steps closed after the run. `online` closes steps as the
// engine reports their depths; otherwise every step is closed after the run. `recurrent` adds
// {{x,y},{y,z}} -> {{x,z}}, so a class can recur at a later step. Without `record_in_run` the
// engine records no class record and the collector computes each one. `matrix_limit` is
// "StepStatisticsMatrixLimit" in bytes, 0 for Automatic.
std::pair<std::vector<uint8_t>, uint32_t> collected_step_statistics(
    size_t threads, uint32_t which, bool by_class, bool online, size_t step_cap,
    bool recurrent = false, bool record_in_run = true,
    std::vector<std::pair<uint32_t, uint32_t>>* over_limit = nullptr,
    uint64_t matrix_limit = 0) {
    hypergraph::Hypergraph hg;
    hypergraph::ParallelEvolutionEngine engine(&hg, threads);
    hgcommon::RecordSet rs;
    rs.causal = false;
    rs.branchial = (which & hg::stats::kBranchialGraph) != 0;
    rs.state_events = false;
    rs.state_invariants = record_in_run;
    hg.set_record_set(rs);
    engine.set_max_states_per_step(step_cap);
    engine.add_rule(hypergraph::make_rule(0)
                        .lhs({0, 1})
                        .lhs({0, 2})
                        .rhs({0, 1})
                        .rhs({0, 3})
                        .rhs({1, 3})
                        .rhs({2, 3})
                        .build());
    if (recurrent)
        engine.add_rule(hypergraph::make_rule(1).lhs({0, 1}).lhs({1, 2}).rhs({0, 2}).build());
    hgffi::StepStatisticsCollector collector(hg, engine, which, by_class, 5);
    collector.set_matrix_limit(matrix_limit);
    if (online) collector.attach(step_cap != 0);
    engine.evolve(std::vector<std::vector<hypergraph::VertexId>>{{0u, 1u}, {0u, 2u}}, 5);
    const wxf::WXFValue reply = collector.finish();
    if (over_limit) *over_limit = collector.matrices_over_limit();
    wxf::Writer w;
    w.write(reply);
    return {w.release_data(), collector.closed_at_finish()};
}

// The reply is the same bytes at 1, 2, 8 and 16 threads under each weighting and each
// "StepStatisticsBranchial" setting, and the same whether the steps are closed during the run or
// after it. During the run every step is closed by the engine's depth reports, so the reply
// builds no record; under "MaxStatesPerStep" the last step's states are selected on the depth
// past the budget, which no task reaches, so that step is closed after the run.
TEST(StateStatistics, CollectedStepStatisticsDoNotDependOnTheThreadCount) {
    const uint32_t graph = hg::stats::kBranchialGraph, overlap = hg::stats::kBranchialOverlap;
    for (const size_t cap : {size_t{0}, size_t{4}})
        for (const uint32_t which : {0u, graph, overlap, graph | overlap})
            for (const bool by_class : {false, true}) {
                const auto after = collected_step_statistics(4, which, by_class, false, cap);
                ASSERT_FALSE(after.first.empty());
                EXPECT_EQ(after.second, 6u) << which;
                for (const size_t threads : {1, 2, 8, 16}) {
                    const auto during = collected_step_statistics(threads, which, by_class, true, cap);
                    EXPECT_EQ(during.first, after.first)
                        << "threads " << threads << " which " << which << " by_class " << by_class
                        << " cap " << cap;
                    EXPECT_LE(during.second, cap ? 1u : 0u) << "threads " << threads;
                }
            }
    // A class recurring at a later step: its record can be claimed by a worker at the later
    // step before the earlier step closes, and the step's job then computes it
    // (Hypergraph::state_invariants_of). That computed record gives the same reply as the
    // record its claiming worker writes.
    for (const bool by_class : {false, true}) {
        const auto after = collected_step_statistics(4, graph | overlap, by_class, false, 0, true);
        EXPECT_EQ(
            collected_step_statistics(4, graph | overlap, by_class, false, 0, true, false).first,
            after.first);
        for (const size_t threads : {1, 2, 8, 16})
            for (int run = 0; run < 4; ++run) {
                const auto during =
                    collected_step_statistics(threads, graph | overlap, by_class, true, 0, true);
                EXPECT_EQ(during.first, after.first) << "threads " << threads;
                EXPECT_EQ(during.second, 0u) << "threads " << threads;
            }
    }
}


// The state matrices at 1, 2, 8 and 16 threads, under full capture with the overlap and graph
// keys, closed during the run and after it. The states are numbered by the run, so the matrices
// are compared up to a renumbering of the states. Every step has its six matrices except the
// steps reported over a limit of the bytes of 1000 states.
TEST(StateStatistics, CollectedStateMatricesDoNotDependOnTheThreadCount) {
    const uint32_t which = hg::stats::kBranchialMatrices | hg::stats::kBranchialGraph |
                           hg::stats::kBranchialOverlap;
    const uint64_t limit = hg::stats::matrix_bytes(1000, which);
    for (const size_t cap : {size_t{0}, size_t{4}}) {
        std::vector<std::pair<uint32_t, uint32_t>> over;
        const auto after =
            collected_step_statistics(4, which, false, false, cap, false, true, &over, limit);
        const auto want = matrix_rows(after.first);
        ASSERT_EQ(want.size(), 6u);
        for (uint32_t step = 0; step < want.size(); ++step) {
            const bool over_limit = std::any_of(over.begin(), over.end(), [&](const auto& o) {
                return o.first == step && o.second > 1000;
            });
            EXPECT_EQ(want[step].size(), over_limit ? 0u : 6u) << "step " << step << " cap " << cap;
        }
        EXPECT_EQ(over.size(), cap ? 0u : 1u);
        for (const size_t threads : {1, 2, 8, 16}) {
            const auto during = collected_step_statistics(threads, which, false, true, cap, false,
                                                          true, nullptr, limit);
            EXPECT_EQ(matrix_rows(during.first), want) << "threads " << threads << " cap " << cap;
        }
    }
}

// The per-vertex curvature k_v of a state worked by hand. The path 1-2-3-4: k is 1/2, 1/4, 1/4,
// 1/2 (VertexInvariantsPoolTheVerticesOfAStep). The cycle 1-2-3-4-1: 1/2 at every vertex. A
// vertex with no neighbour (7 in a unary edge, 8 in a self-loop) has no entry.
TEST(StateStatistics, VertexCurvatureOfAPathAndACycle) {
    using E = std::vector<std::pair<int64_t, std::vector<uint32_t>>>;
    const E path = {{0, {1, 2}}, {1, {2, 3}}, {2, {3, 4}}, {3, {7}}, {4, {8, 8}}};
    const E cycle = {{0, {1, 2}}, {1, {2, 3}}, {2, {3, 4}}, {3, {4, 1}}};
    EXPECT_EQ(hg::stats::vertex_curvature(path),
              (hg::stats::VertexCurvature{{1, 0.5}, {2, 0.25}, {3, 0.25}, {4, 0.5}}));
    EXPECT_EQ(hg::stats::vertex_curvature(cycle),
              (hg::stats::VertexCurvature{{1, 0.5}, {2, 0.5}, {3, 0.5}, {4, 0.5}}));
    EXPECT_TRUE(hg::stats::vertex_curvature(E{{0, {5}}}).empty());
    EXPECT_TRUE(hg::stats::vertex_curvature(E{}).empty());
}

#ifdef HG_GPU_BACKEND
// hg_gpu::vertex_curvature_device gives stats::vertex_curvature's values bit for bit: the path
// and cycle above, a state with no edges, and 40 pseudo-random states of 1 to 400 edges of
// arity 1 to 3 over up to 150 vertices, self-loops included, the larger ones past the device's
// first scratch slice.
TEST(StateStatistics, DeviceVertexCurvatureEqualsTheHost) {
    using E = std::vector<std::pair<int64_t, std::vector<uint32_t>>>;
    std::vector<E> states = {
        {{0, {1, 2}}, {1, {2, 3}}, {2, {3, 4}}, {3, {7}}, {4, {8, 8}}},
        {{0, {1, 2}}, {1, {2, 3}}, {2, {3, 4}}, {3, {4, 1}}},
        {}};
    uint64_t x = 0x9E3779B97F4A7C15ull;
    auto next = [&x](uint32_t n) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        return static_cast<uint32_t>(x % n);
    };
    for (uint32_t s = 0; s < 40; ++s) {
        const uint32_t m = 1 + next(s < 30 ? 60 : 400);
        const uint32_t n = 2 + next(150);
        E edges;
        for (uint32_t e = 0; e < m; ++e) {
            std::vector<uint32_t> v(1 + next(3));
            for (auto& u : v) u = 10 + next(n);
            edges.emplace_back(e, std::move(v));
        }
        states.push_back(std::move(edges));
    }
    std::vector<uint32_t> state_edges{0}, edge_verts{0}, verts, vertex_count;
    std::vector<std::vector<uint32_t>> ids(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        for (const auto& e : states[i]) {
            verts.insert(verts.end(), e.second.begin(), e.second.end());
            edge_verts.push_back(static_cast<uint32_t>(verts.size()));
            ids[i].insert(ids[i].end(), e.second.begin(), e.second.end());
        }
        state_edges.push_back(static_cast<uint32_t>(edge_verts.size() - 1));
        std::sort(ids[i].begin(), ids[i].end());
        ids[i].erase(std::unique(ids[i].begin(), ids[i].end()), ids[i].end());
        vertex_count.push_back(static_cast<uint32_t>(ids[i].size()));
    }
    std::vector<double> k;
    std::vector<uint64_t> k_off;
    std::vector<uint8_t> done;
    if (!hg_gpu::vertex_curvature_device(state_edges, edge_verts, verts, vertex_count, k, k_off,
                                         done))
        GTEST_SKIP() << "no usable device";
    for (size_t i = 0; i < states.size(); ++i) {
        ASSERT_TRUE(done[i]) << "state " << i;
        hg::stats::VertexCurvature got;
        for (size_t j = 0; j < ids[i].size(); ++j)
            if (!std::isnan(k[k_off[i] + j])) got.emplace_back(ids[i][j], k[k_off[i] + j]);
        const hg::stats::VertexCurvature want = hg::stats::vertex_curvature(states[i]);
        ASSERT_EQ(got.size(), want.size()) << "state " << i;
        for (size_t j = 0; j < want.size(); ++j) {
            EXPECT_EQ(got[j].first, want[j].first) << "state " << i;
            EXPECT_EQ(std::memcmp(&got[j].second, &want[j].second, sizeof(double)), 0)
                << "state " << i << " vertex " << want[j].first << ": " << got[j].second
                << " against " << want[j].second;
        }
    }
}
#endif

// hgmarshal::parallel_sort on host jobs gives std::sort's order: 300,007 pairs with repeats, and
// a vector below the parallel threshold.
TEST(GraphMarshal, ParallelSortEqualsTheSerialSort) {
    uint64_t x = 0x9E3779B97F4A7C15ull;
    std::vector<std::pair<int64_t, uint32_t>> v(300007);
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        e = {static_cast<int64_t>(x % 50000) - 25000, static_cast<uint32_t>(x >> 40) % 7};
    }
    auto want = v;
    std::sort(want.begin(), want.end());
    hgmarshal::HostJobs jobs(4);
    hgmarshal::parallel_sort(v, jobs.parallel_for());
    EXPECT_EQ(v, want);
    std::vector<std::pair<int64_t, int64_t>> small = {{3, 1}, {1, 2}, {3, 0}, {1, 2}};
    hgmarshal::parallel_sort(small, jobs.parallel_for());
    EXPECT_EQ(small, (std::vector<std::pair<int64_t, int64_t>>{{1, 2}, {1, 2}, {3, 0}, {3, 1}}));
}

// hgmarshal::branchial_steps on host jobs gives the BranchialSteps it gives on the calling
// thread: 3000 states over 6 steps, with contents and with the branchial pairs.
TEST(StateStatistics, BranchialStepsInJobsEqualTheSerialOnes) {
    constexpr uint32_t kStates = 3000;
    uint64_t x = 0x2545F4914F6CDD1Dull;
    auto next = [&x](uint32_t n) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        return static_cast<uint32_t>(x % n);
    };
    std::vector<std::vector<uint32_t>> contents(kStates);   // arity, vertices, arity, ...
    std::vector<uint32_t> step(kStates), output(2 * kStates);
    for (uint32_t s = 0; s < kStates; ++s) {
        step[s] = next(6);
        for (uint32_t e = next(8); e > 0; --e) {
            const uint32_t k = 1 + next(3);
            contents[s].push_back(k);
            for (uint32_t i = 0; i < k; ++i) contents[s].push_back(next(400));
        }
    }
    for (auto& o : output) o = next(kStates);
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    for (uint32_t i = 0; i + 1 < output.size(); i += 2) pairs.emplace_back(i, i + 1);
    const uint32_t which = hg::stats::kBranchialGraph | hg::stats::kBranchialOverlap;
    auto run = [&](const hgmarshal::ParallelFor& par) {
        return hgmarshal::branchial_steps(
            which, kStates, [](uint32_t s) { return s % 11 != 3; },
            [](uint32_t s) { return static_cast<int64_t>(s / 2); },
            [&](uint32_t s) { return step[s]; },
            [&](uint32_t s, std::vector<uint32_t>& out) {
                out.insert(out.end(), contents[s].begin(), contents[s].end());
            },
            pairs, [&](uint32_t e) { return output[e]; }, 6, par);
    };
    const auto serial = run(hgmarshal::serial_for);
    hgmarshal::HostJobs jobs(4);
    const auto parallel = run(jobs.parallel_for());
    ASSERT_EQ(serial.size(), parallel.size());
    for (const auto& [s, b] : serial) {
        const hg::stats::BranchialStep& p = parallel.at(s);
        EXPECT_EQ(b.nodes, p.nodes) << "step " << s;
        EXPECT_EQ(b.pairs, p.pairs) << "step " << s;
        EXPECT_EQ(b.vertex_sets, p.vertex_sets) << "step " << s;
        EXPECT_EQ(b.edge_sets, p.edge_sets) << "step " << s;
        EXPECT_FALSE(b.nodes.empty());
    }
}

// "StepStatisticsWeighting" -> "Classes" counts a class once; "States" counts its raw states.
TEST(StateStatistics, WeightingByClassesCountsEachClassOnce) {
    const std::vector<hg::stats::StepPoint> points = {{0, 1, 3}, {0, 2, 1}};
    std::vector<uint64_t> s1, s2;
    const std::unordered_map<uint64_t, const hgcommon::StateInvariantRecord*> edges = {
        {1, hg::stats::invariant_record({{1, 2}}, s1)},
        {2, hg::stats::invariant_record({{1, 2}, {2, 3}}, s2)}};
    auto vertex_count = [&](bool by_class) {
        const wxf::WXFValue steps = pooled_steps(
            points, edges, by_class);
        const auto& first = std::get<wxf::WXFValueList>(steps.data).at(0);
        EXPECT_EQ(number_at(first, "RawStates"), 4);
        return *assoc_at(*assoc_at(first, "Invariants"), "VertexCount");
    };
    const wxf::WXFValue states = vertex_count(false), classes = vertex_count(true);
    EXPECT_EQ(number_at(states, "N"), 4);
    EXPECT_DOUBLE_EQ(number_at(states, "Mean"), (3 * 2 + 3) / 4.0);
    EXPECT_EQ(number_at(classes, "N"), 2);
    EXPECT_DOUBLE_EQ(number_at(classes, "Mean"), 2.5);
}

// "VertexInvariants" pools the vertices of every state at a step: P4 (3 raw states) and the
// cycle C4 (1). P4's k are 1/2, 1/4, 1/4, 1/2 and C4's are all 1/2 (each C4 edge moves 1/4 one
// step at each end, W1 = 1/2).
// Under "States" N = 3 * 4 + 4 = 16; under "Classes" 8. The per-state values reach "Invariants".
TEST(StateStatistics, VertexInvariantsPoolTheVerticesOfAStep) {
    const std::vector<hg::stats::StepPoint> points = {{0, 1, 3}, {0, 2, 1}};
    std::vector<uint64_t> s1, s2;
    const std::unordered_map<uint64_t, const hgcommon::StateInvariantRecord*> recs = {
        {1, hg::stats::invariant_record({{1, 2}, {2, 3}, {3, 4}}, s1)},
        {2, hg::stats::invariant_record({{1, 2}, {2, 3}, {3, 4}, {4, 1}}, s2)}};
    auto step = [&](bool by_class) {
        const wxf::WXFValue steps = pooled_steps(
            points, recs, by_class);
        return std::get<wxf::WXFValueList>(steps.data).at(0);
    };
    const wxf::WXFValue states = step(false), classes = step(true);
    const wxf::WXFValue& oll = *assoc_at(*assoc_at(states, "VertexInvariants"),
                                         "OllivierRicciCurvature");
    EXPECT_EQ(number_at(oll, "N"), 16);
    EXPECT_DOUBLE_EQ(number_at(oll, "Mean"), (3 * 1.5 + 2.0) / 16);
    EXPECT_DOUBLE_EQ(number_at(oll, "Min"), 0.25);
    EXPECT_DOUBLE_EQ(number_at(oll, "Max"), 0.5);
    for (const char* k : {"Median", "Q1", "Q3", "P10", "P90", "Skewness", "Kurtosis", "Histogram"})
        EXPECT_NE(assoc_at(oll, k), nullptr) << k;
    // The 6 values 1/4 lie in bin [0.25, 0.34375): P10 at t = 1.6 is 0.25 + 1.6/6 of the bin.
    EXPECT_NEAR(number_at(oll, "P10"), 0.25 + 1.6 / 6 * 0.09375, 1e-12);
    // Population moments of 6 x 1/4 and 10 x 1/2: mean 13/32.
    const double mean = 13.0 / 32;
    double m2 = 0, m3 = 0, m4 = 0;
    for (auto [x, c] : {std::pair{0.25, 6}, std::pair{0.5, 10}}) {
        m2 += c * std::pow(x - mean, 2) / 16;
        m3 += c * std::pow(x - mean, 3) / 16;
        m4 += c * std::pow(x - mean, 4) / 16;
    }
    EXPECT_NEAR(number_at(oll, "Skewness"), m3 / std::pow(m2, 1.5), 1e-9);
    EXPECT_NEAR(number_at(oll, "Kurtosis"), m4 / (m2 * m2), 1e-9);
    EXPECT_NEAR(number_at(oll, "StandardDeviation"), std::sqrt(m2 * 16 / 15), 1e-12);
    EXPECT_EQ(number_at(*assoc_at(*assoc_at(classes, "VertexInvariants"),
                                  "OllivierRicciCurvature"), "N"), 8);
    EXPECT_EQ(number_at(*assoc_at(*assoc_at(states, "VertexInvariants"), "LocalDimension"), "N"),
              16);
    const wxf::WXFValue& invs = *assoc_at(states, "Invariants");
    EXPECT_EQ(number_at(*assoc_at(invs, "LargestComponentDimension"), "N"), 4);
    EXPECT_EQ(number_at(*assoc_at(invs, "LocalDimensionMax"), "N"), 4);
    EXPECT_EQ(number_at(*assoc_at(invs, "LocalDimensionStandardDeviation"), "N"), 4);
    // C4's curvature is constant: only P4 has a Moran's I and a degree correlation.
    EXPECT_EQ(number_at(*assoc_at(invs, "OllivierMoranI"), "N"), 3);
    EXPECT_DOUBLE_EQ(number_at(*assoc_at(invs, "OllivierMoranI"), "Mean"), -1.0 / 3.0);
    EXPECT_EQ(number_at(*assoc_at(invs, "OllivierDegreeCorrelation"), "N"), 3);
}

// The moments of a pooled distribution whose spread is small against its mean. Rule
// {{x,y},{y,z}} -> {{x,z},{z,w},{x,w},{y,w}} from {{0,0},{0,0}}, 4 steps, every raw state: step
// 4's k_v are 5/12 (90 vertices), 7/16 (180) and 31/72 (180), mean 0.4306 and standard deviation
// 0.0076 (reference/verify_state_statistics.wls, CorpusLoops). The exact kurtosis is 5/2;
// from double power sums the engine gave 2.4999999967.
TEST(StateStatistics, VertexInvariantMomentsOfANarrowDistribution) {
    hypergraph::Hypergraph hg;
    hypergraph::ParallelEvolutionEngine engine(&hg, 4);
    hgcommon::RecordSet rs;
    rs.causal = false;
    rs.branchial = false;
    rs.state_events = false;
    rs.state_invariants = true;
    hg.set_record_set(rs);
    engine.add_rule(hypergraph::make_rule(0)
                        .lhs({0, 1})
                        .lhs({1, 2})
                        .rhs({0, 2})
                        .rhs({2, 3})
                        .rhs({0, 3})
                        .rhs({1, 3})
                        .build());
    hgffi::StepStatisticsCollector collector(hg, engine, 0, false, 4);
    engine.evolve(std::vector<std::vector<hypergraph::VertexId>>{{0u, 0u}, {0u, 0u}}, 4);
    const wxf::WXFValue reply = collector.finish();
    const wxf::WXFValue* oll = nullptr;
    for (const auto& step : std::get<wxf::WXFValueList>(reply.data))
        if (number_at(step, "Step") == 4)
            oll = assoc_at(*assoc_at(step, "VertexInvariants"), "OllivierRicciCurvature");
    ASSERT_NE(oll, nullptr);
    ASSERT_EQ(number_at(*oll, "N"), 450);
    // Two-pass central moments of the exact values.
    const std::pair<double, int> values[3] = {{5.0 / 12, 90}, {7.0 / 16, 180}, {31.0 / 72, 180}};
    double mean = 0, m2 = 0, m3 = 0, m4 = 0;
    for (const auto& [x, c] : values) mean += c * x / 450;
    for (const auto& [x, c] : values) {
        m2 += c * std::pow(x - mean, 2) / 450;
        m3 += c * std::pow(x - mean, 3) / 450;
        m4 += c * std::pow(x - mean, 4) / 450;
    }
    EXPECT_NEAR(m4 / (m2 * m2), 2.5, 1e-13);
    EXPECT_NEAR(number_at(*oll, "Mean"), mean, 1e-15);
    EXPECT_NEAR(number_at(*oll, "StandardDeviation"), std::sqrt(m2 * 450 / 449), 1e-15);
    EXPECT_NEAR(number_at(*oll, "Skewness"), m3 / std::pow(m2, 1.5), 1e-12);
    EXPECT_NEAR(number_at(*oll, "Kurtosis"), 2.5, 1e-12);
}

// "StepStatisticsBranchial" -> "Graph" under "ExploreFromCanonicalStatesOnly" gives the keys full
// capture gives under "CanonicalizeStates" -> Full, step by step, on every oracle corpus case at
// its oracle depth. Both runs are asked for the overlap keys too, which Full leaves out on both
// routes with the same warning.
TEST(StateStatistics, BranchialGraphUnderQuotientEqualsFullCapture) {
    auto input = [](const oracle::Case& c, bool quotient) {
        wxf::Writer w;
        w.write_header();
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(4);
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("InitialStates"));
        EdgeList init;
        for (const auto& e : c.init) {
            Edge edge;
            for (const auto v : e) edge.push_back(static_cast<int64_t>(v) + 1);
            init.push_back(std::move(edge));
        }
        w.write(StateList{init});
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Rules"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(c.rules.size());
        auto side = [](const auto* edges, uint8_t n) {
            EdgeList out;
            for (uint8_t i = 0; i < n; ++i) {
                Edge edge;
                for (uint8_t k = 0; k < edges[i].arity; ++k)
                    edge.push_back(static_cast<int64_t>(edges[i].vars[k]) + 1);
                out.push_back(std::move(edge));
            }
            return out;
        };
        for (size_t i = 0; i < c.rules.size(); ++i) {
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write("r" + std::to_string(i));
            w.write_function("Rule", 2);
            w.write(side(c.rules[i].lhs, c.rules[i].num_lhs_edges));
            w.write(side(c.rules[i].rhs, c.rules[i].num_rhs_edges));
        }
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Steps"));
        w.write(static_cast<int64_t>(c.oracle_steps));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("Options"));
        w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w.write_varint(4);
        put_str_list_option(w, "RequestedData", {"StepStatistics"});
        put_str_option(w, "CanonicalizeStates", "Full");
        put_str_list_option(w, "StepStatisticsBranchial", {"Graph", "Overlap"});
        put_str_option(w, "ExploreFromCanonicalStatesOnly", quotient ? "True" : "False");
        return w.release_data();
    };
    size_t steps_with_edges = 0;
    for (const auto& c : oracle::corpus()) {
        HostBridge host;
        const auto full = run_rewriting_core(input(c, false), host);
        const auto quotient = run_rewriting_core(input(c, true), host);
        const auto rows = branchial_record_bytes(full);
        ASSERT_FALSE(rows.empty()) << c.name;
        for (const auto& row : rows) {
            ASSERT_GE(row.size(), 3u) << c.name;
            EXPECT_EQ(row[0].first, "BranchialDegree") << c.name;
        }
        EXPECT_EQ(branchial_record_bytes(quotient), rows) << c.name;
        EXPECT_FALSE(reply_mentions(quotient, "not computed under ExploreFromCanonicalStatesOnly"))
            << c.name;
        EXPECT_TRUE(reply_mentions(quotient, "overlap metrics need CanonicalizeStates")) << c.name;
        for (const auto& row : rows)
            for (const auto& [k, v] : row)
                if (k == "BranchialDimension") ++steps_with_edges;
    }
    // Some step of the corpus has a branchial component of two or more states.
    EXPECT_GT(steps_with_edges, 0u);

    // A Query of a quotient session that does not repeat "ExploreFromCanonicalStatesOnly" is
    // served under the session's exploration: every single-rule corpus case, opened and stepped
    // to its oracle depth.
    auto opts = [](bool with_explore) {
        return [with_explore](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"StepStatistics"});
            put_str_option(w, "CanonicalizeStates", "Full");
            put_str_list_option(w, "StepStatisticsBranchial", {"Graph"});
            if (with_explore) put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
        };
    };
    auto side = [](const auto* edges, uint8_t n) {
        EdgeList out;
        for (uint8_t i = 0; i < n; ++i) {
            Edge edge;
            for (uint8_t k = 0; k < edges[i].arity; ++k)
                edge.push_back(static_cast<int64_t>(edges[i].vars[k]) + 1);
            out.push_back(std::move(edge));
        }
        return out;
    };
    for (const auto& c : oracle::corpus()) {
        if (c.rules.size() != 1) continue;
        EdgeList init;
        for (const auto& e : c.init) {
            Edge edge;
            for (const auto v : e) edge.push_back(static_cast<int64_t>(v) + 1);
            init.push_back(std::move(edge));
        }
        const EdgeList lhs = side(c.rules[0].lhs, c.rules[0].num_lhs_edges);
        const EdgeList rhs = side(c.rules[0].rhs, c.rules[0].num_rhs_edges);
        auto job = [&](int64_t steps, const std::string& op, int64_t h, bool explore) {
            return session_envelope(StateList{init}, lhs, rhs, steps, op, h,
                                    op == "Evolve" || op == "Open", {}, opts(explore),
                                    explore ? 4 : 3, false);
        };
        HostBridge host;
        const auto direct = run_rewriting_core(job(c.oracle_steps, "Evolve", 0, true), host);
        const int64_t handle = read_int_key(run_rewriting_core(job(0, "Open", 0, true), host),
                                            "Session");
        ASSERT_NE(handle, 0) << c.name;
        run_rewriting_core(job(c.oracle_steps, "Step", handle, true), host);
        const auto queried = run_rewriting_core(job(0, "Query", handle, false), host);
        run_rewriting_core(job(0, "Close", handle, false), host);
        const auto want = branchial_record_bytes(direct);
        ASSERT_FALSE(want.empty()) << c.name;
        EXPECT_EQ(branchial_record_bytes(queried), want) << c.name;
    }
}

// Through the FFI: the options are read, the branchial keys appear under None and Automatic, and
// under Full the overlap keys are left out with a warning. The branchial keys count each state
// once, so "StepStatisticsWeighting" leaves them unchanged.
TEST(StateStatistics, BranchialOptionThroughTheFfi) {
    auto run = [&](const char* canon, const char* weighting) {
        HostBridge host;
        return run_rewriting_core(build_input({{{1, 2}, {1, 3}}}, kBranchLhs, kBranchRhs, 2,
            [&](wxf::Writer& w) {
                put_str_list_option(w, "RequestedData", {"StepStatistics"});
                put_str_option(w, "CanonicalizeStates", canon);
                put_str_list_option(w, "StepStatisticsBranchial", {"Graph", "Overlap"});
                put_str_option(w, "StepStatisticsWeighting", weighting);
            }, 4), host);
    };
    for (const char* canon : {"None", "Automatic"}) {
        const auto classes = run(canon, "Classes");
        const auto stats = value_bytes(classes, "StepStatistics");
        for (const auto& k : kBranchialKeys)
            if (k != "BranchialDimension") EXPECT_TRUE(bytes_contain(stats, k)) << k;
        for (const char* k : {"WolframHausdorffDimension", "BallGrowthDimension"})
            EXPECT_TRUE(bytes_contain(stats, k)) << k;
        const auto rows = branchial_record_bytes(classes);
        ASSERT_EQ(rows.size(), 3u) << canon;
        EXPECT_EQ(rows, branchial_record_bytes(run(canon, "States"))) << canon;
    }
    const auto full = run("Full", "Classes");
    EXPECT_TRUE(bytes_contain(value_bytes(full, "StepStatistics"), "BranchialDegree"));
    for (const char* k : {"StateOverlap", "StateMutualInformation", "EdgeSharpness",
                          "OverlapByBranchialDistance"})
        EXPECT_FALSE(bytes_contain(value_bytes(full, "StepStatistics"), k)) << k;
    EXPECT_TRUE(bytes_contain(value_bytes(full, "Warnings"), "StepStatisticsBranchial"));
}

// Histogram keys round to the nearest multiple with halves to even, as the reference's
// Round[x, 0.01] does: 1.125 (MeanDegree 9/8) is 112.5 hundredths exactly and keys as 1.12.
TEST(StateStatistics, HistogramKeysRoundHalvesToEven) {
    const auto s = hg::stats::summarise({{1.125, 1}}, 0.01);
    ASSERT_EQ(s.histogram.size(), 1u);
    EXPECT_NEAR(s.histogram.begin()->first, 1.12, 1e-12);
    // 0.125 and 0.375 are halves of 0.25: Round gives 0 and 0.5.
    const auto q = hg::stats::summarise({{0.125, 1}, {0.375, 1}}, 0.25);
    ASSERT_EQ(q.histogram.size(), 2u);
    EXPECT_EQ(q.histogram.begin()->first, 0.0);
    EXPECT_EQ(q.histogram.rbegin()->first, 0.5);
}

// A steered Step from an id expands every frontier state that id stands for. Under
// CanonicalizeStates -> Full without quotient exploration several raw states of one class sit on
// the frontier under one id; only the first was expanded, and the id stayed on the frontier.
TEST(Session, ASteeredStepExpandsEveryStateItsIdStandsFor) {
    HostBridge host;
    auto opts = [](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData", {"NumStates"});
        put_str_option(w, "CanonicalizeStates", "Full");
    };
    const auto opened = run_rewriting_core(branch_job(1, "Open", 0, opts, 2), host);
    const int64_t h = read_int_key(opened, "Session");
    ASSERT_GT(h, 0);
    const std::vector<int64_t> frontier = read_int_list_key(opened, "Frontier");
    ASSERT_FALSE(frontier.empty());
    const int64_t id = frontier.front();
    const auto stepped = run_rewriting_core(
        session_envelope(kBranchSeed, kBranchLhs, kBranchRhs, 1, "Step", h, false, {id}, opts, 2,
                         false), host);
    const std::vector<int64_t> after = read_int_list_key(stepped, "Frontier");
    EXPECT_EQ(std::count(after.begin(), after.end(), id), 0)
        << "state " << id << " is still on the frontier after a Step from it";
    run_rewriting_core(branch_job(0, "Close", h, opts, 2), host);
}

// =============================================================================
// Input limits and option values, checked in the shared parse for both devices
// =============================================================================
namespace {

// The Context string of every entry of the reply's Warnings list.
std::vector<std::string> warning_contexts(const std::vector<uint8_t>& out) {
    std::vector<std::string> ctx;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "Warnings") { vp.skip_value(); return; }
        vp.read_function([&](const std::string&, size_t count, wxf::Parser& ep) {
            for (size_t i = 0; i < count; ++i)
                ep.read_association([&](const std::string& wk, wxf::Parser& wp) {
                    if (wk == "Context") ctx.push_back(wp.read<std::string>());
                    else wp.skip_value();
                });
        });
    });
    return ctx;
}

// kSeed under kLhs -> kRhs for 2 steps, asking for NumStates, with `put` writing one more option.
std::vector<uint8_t> job_with_option(const std::function<void(wxf::Writer&)>& put) {
    return build_input(kSeed, kLhs, kRhs, 2, [&](wxf::Writer& w) {
        put_str_list_option(w, "RequestedData", {"NumStates"});
        put(w);
    }, 2);
}

std::string run_error(const std::vector<uint8_t>& job) {
    HostBridge host;
    try {
        run_rewriting_core(job, host);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

}  // namespace

// Input both devices answered differently (F15): the CPU gave 0 states for an empty initial state
// and the GPU 1; an edge above arity 16 was an error on the CPU and an empty result on the GPU.
// Each is refused with an error before a device is chosen.
TEST(FfiInput, InvalidInitialStatesAndRulesAreRefused) {
    const auto none = [](wxf::Writer&) {};
    struct Case { StateList init; EdgeList lhs, rhs; const char* says; };
    const EdgeList a17 = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}};
    const StateList init17 = {a17};
    const std::vector<Case> cases = {
        {{}, kLhs, kRhs, "InitialStates is empty"},
        {{{}}, kLhs, kRhs, "initial state 0 has no edges"},
        {{{{}}}, kLhs, kRhs, "initial state 0 edge 0 has arity 0"},
        {{{{1, 2}}, {}}, kLhs, kRhs, "initial state 1 has no edges"},
        {init17, kLhs, kRhs, "initial state 0 edge 0 has arity 17"},
        {kSeed, a17, kRhs, "rule 0 LHS edge 0 has arity 17"},
        {kSeed, kLhs, {{1, 2}, {}}, "rule 0 RHS edge 1 has arity 0"},
        {kSeed, {}, kRhs, "rule 0 has an empty left-hand side"},
    };
    for (const Case& c : cases) {
        const std::string err = run_error(build_input(c.init, c.lhs, c.rhs, 1, none, 0));
        EXPECT_NE(err.find(c.says), std::string::npos) << "expected '" << c.says << "', got '" << err << "'";
    }
    EXPECT_EQ(run_error(build_input(kSeed, kLhs, kRhs, 1, none, 0)), "");
}

// A negative cap is skipped with a warning; it was cast to 2^64-1 and acted as no cap silently.
TEST(FfiInput, ANegativeCapIsSkippedWithAWarning) {
    HostBridge host;
    const int64_t plain = read_int_key(run_rewriting_core(job_with_option([](wxf::Writer& w) {
        w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w.write(std::string("RandomSeed"));
        w.write(int64_t{0});
    }), host), "NumStates");
    for (const char* cap : {"MaxStatesPerStep", "MaxSuccessorStatesPerParent", "MatchesPerStateRule",
                            "MatchesPerStep"}) {
        const auto out = run_rewriting_core(job_with_option([&](wxf::Writer& w) {
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string(cap));
            w.write(int64_t{-1});
        }), host);
        EXPECT_EQ(read_int_key(out, "NumStates"), plain) << cap;
        const auto ctx = warning_contexts(out);
        ASSERT_EQ(ctx.size(), 1u) << cap;
        EXPECT_NE(ctx[0].find(std::string("option '") + cap + "' ignored: a cap is a non-negative integer"),
                  std::string::npos) << ctx[0];
    }
}

// NaN or infinity for a probability, rate or weight is skipped with a warning in the shared parse.
// NaN ExplorationProbability gave (10,9,8,0) on the CPU and (2,1,0,0) on the GPU.
TEST(FfiInput, ANonFiniteProbabilityRateOrWeightIsSkippedWithAWarning) {
    HostBridge host;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const char* key : {"ExplorationProbability", "TransitionRate", "RuleWeights"}) {
        for (double v : {nan, inf, -inf}) {
            const auto out = run_rewriting_core(job_with_option([&](wxf::Writer& w) {
                w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                w.write(std::string(key));
                if (std::string(key) == "RuleWeights") w.write(std::vector<double>{v});
                else w.write(v);
            }), host);
            const auto ctx = warning_contexts(out);
            ASSERT_EQ(ctx.size(), 1u) << key << " " << v;
            EXPECT_NE(ctx[0].find("not a finite real number"), std::string::npos) << ctx[0];
        }
    }
}

// An option key that is not valid UTF-8 is quoted in the warning with its bad bytes as \xNN.
TEST(FfiInput, AWarningQuotesACorruptOptionKeyAsValidUtf8) {
    HostBridge host;
    const std::string corrupt = std::string("Re") + char(0xFE) + "ues" + char(0xB3) + "edData";
    const auto out = run_rewriting_core(job_with_option([&](wxf::Writer& w) {
        put_str_option(w, corrupt.c_str(), "True");
    }), host);
    const auto ctx = warning_contexts(out);
    ASSERT_EQ(ctx.size(), 1u);
    EXPECT_NE(ctx[0].find("option 'Re\\xFEues\\xB3edData'"), std::string::npos) << ctx[0];
    EXPECT_EQ(hgmarshal::valid_utf8("caf\xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80"),
              "caf\xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80");
    EXPECT_EQ(hgmarshal::valid_utf8("\xC0\x80\xED\xA0\x80\xF4\x90\x80\x80\xE2\x82"),
              "\\xC0\\x80\\xED\\xA0\\x80\\xF4\\x90\\x80\\x80\\xE2\\x82");
}

// QuotientNeedsFull names the option the job set: CanonicalizeEvents -> Automatic alone does not
// mention ExploreFromCanonicalStatesOnly, and the reverse.
TEST(FfiInput, QuotientNeedsFullNamesTheOptionTheJobSet) {
    HostBridge host;
    const auto events_only = warning_contexts(run_rewriting_core(job_with_option([](wxf::Writer& w) {
        put_str_option(w, "CanonicalizeEvents", "Automatic");
    }), host));
    ASSERT_EQ(events_only.size(), 1u);
    EXPECT_NE(events_only[0].find("\"CanonicalizeEvents\" -> Automatic needs"), std::string::npos);
    EXPECT_EQ(events_only[0].find("ExploreFromCanonicalStatesOnly"), std::string::npos);

    const auto ecso_only = warning_contexts(run_rewriting_core(job_with_option([](wxf::Writer& w) {
        put_str_option(w, "ExploreFromCanonicalStatesOnly", "True");
    }), host));
    ASSERT_EQ(ecso_only.size(), 1u);
    EXPECT_NE(ecso_only[0].find("\"ExploreFromCanonicalStatesOnly\" -> True needs"), std::string::npos);
    EXPECT_EQ(ecso_only[0].find("CanonicalizeEvents"), std::string::npos);

    const auto full = warning_contexts(run_rewriting_core(job_with_option([](wxf::Writer& w) {
        put_str_option(w, "CanonicalizeStates", "Full");
    }), host));
    EXPECT_TRUE(full.empty());
}

// ExploreFromCanonicalStatesOnly is not applied with MaxStatesPerStep: the reply warns
// (QuotientStepCap) and its counts are the counts of full capture.
TEST(FfiInput, QuotientExplorationIsNotAppliedWithAStepCap) {
    HostBridge host;
    auto job = [](bool ecso) {
        return build_input(kSeed, kLhs, kRhs, 3, [&](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"NumStates", "NumEvents", "NumCausalEdges"});
            put_str_option(w, "CanonicalizeStates", "Full");
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("MaxStatesPerStep"));
            w.write(int64_t{2});
            put_str_option(w, "ExploreFromCanonicalStatesOnly", ecso ? "True" : "False");
        }, 4);
    };
    const auto quotient = run_rewriting_core(job(true), host);
    const auto full = run_rewriting_core(job(false), host);
    const auto ctx = warning_contexts(quotient);
    ASSERT_EQ(ctx.size(), 1u);
    EXPECT_NE(ctx[0].find("does not apply with \"MaxStatesPerStep\""), std::string::npos);
    EXPECT_TRUE(warning_contexts(full).empty());
    for (const char* k : {"NumStates", "NumEvents", "NumCausalEdges"})
        EXPECT_EQ(read_int_key(quotient, k), read_int_key(full, k)) << k;
}

// A counts-only request under CanonicalizeStates -> Full counts the isomorphism classes on either
// device. The GPU reply read every state as empty when no state contents were requested (their
// edge counts were not read back), so every state fell into one class and NumStates was 1.
TEST(WxfSerializationPin, ACountsOnlyFullRunCountsTheClasses) {
    HostBridge host;
    auto run = [&](std::vector<std::string> requested) {
        return read_int_key(run_rewriting_core(build_input(kBranchSeed, kBranchLhs, kBranchRhs, 3,
            [&](wxf::Writer& w) {
                put_str_list_option(w, "RequestedData", requested);
                put_str_option(w, "CanonicalizeStates", "Full");
            }, 2), host), "NumStates");
    };
    const int64_t counts_only = run({"NumStates"});
    const int64_t with_states = run({"NumStates", "States"});
    EXPECT_GT(with_states, 1);
    EXPECT_EQ(counts_only, with_states);
}

// "Threads" and the "Tuning" association reach the engine that runs the job: the "Device" reply
// reports the worker count and, on the GPU, the memory cap. An unknown tuning name is skipped
// with an OptionSkipped warning and the other entries still apply.
TEST(WxfSerializationPin, TuningOverridesReachTheEngine) {
    HostBridge host;
    const auto out = run_rewriting_core(build_input(kSeed, kLhs, kRhs, 3,
        [&](wxf::Writer& w) {
            put_str_list_option(w, "RequestedData", {"Device", "NumStates"});
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("Threads"));
            w.write(int64_t{3});
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("Tuning"));
            w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
            w.write_varint(6);
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("GPUL2Persist"));
            w.write(int64_t{0});
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("GPUBlocks"));
            w.write(int64_t{256});
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("GPUBacklog"));
            w.write(int64_t{12});
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("CausalSetCapacity"));
            w.write(int64_t{2048});
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("NoSuchName"));
            w.write(int64_t{7});
            w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w.write(std::string("GPUMemoryLimit"));
            w.write(int64_t{3} << 30);
        }, 3), host);
    std::map<std::string, int64_t> device;
    wxf::Parser parser(out);
    parser.skip_header();
    parser.read_association([&](const std::string& k, wxf::Parser& vp) {
        if (k != "Device") { vp.skip_value(); return; }
        vp.read_association([&](const std::string& f, wxf::Parser& fp) {
            if (f == "Threads" || f == "MemoryCap" || f == "CausalSetCapacity" || f == "Blocks" ||
                f == "Backlog" || f == "L2Persist")
                device[f] = fp.read<int64_t>();
            else fp.skip_value();
        });
    });
    EXPECT_EQ(device["Threads"], 3);
#ifdef HG_GPU_BACKEND
    EXPECT_EQ(device["MemoryCap"], int64_t{3} << 30);
    EXPECT_EQ(device["Blocks"], 256);
    EXPECT_EQ(device["Backlog"], 12);
    ASSERT_TRUE(device.count("L2Persist"));
    EXPECT_EQ(device["L2Persist"], 0);
#else
    EXPECT_EQ(device["CausalSetCapacity"], 2048);
#endif
    EXPECT_TRUE(reply_mentions(out, "NoSuchName"));
    EXPECT_GT(read_int_key(out, "NumStates"), 0);
    // The overrides change no count.
    const auto plain = run_rewriting_core(build_input(kSeed, kLhs, kRhs, 3,
        [&](wxf::Writer& w) { put_str_list_option(w, "RequestedData", {"NumStates", "NumEvents"}); },
        1), host);
    EXPECT_EQ(read_int_key(out, "NumStates"), read_int_key(plain, "NumStates"));
}

// =============================================================================
// The streamed graph marshaller (hgmarshal::build_graph_data)
// =============================================================================

namespace {

namespace hgm = HG_NAMESPACE::marshal;

// A multiway of `n` states and `m` events with fixed contents: event e goes from state e % n to
// state (e * 7 + 1) % n, every event e > 0 is caused by e - 1, every third by e - 3 as well, and
// every seventh by e - 1 twice; events 2k and 2k + 1 are branchial. Event 5 has no payload, as a
// device event missing from the reply.
struct SyntheticGraphSource {
    uint32_t n, m;
    uint32_t num_states() const { return n; }
    bool state_valid(uint32_t sid) const { return sid % 11 != 10; }
    int64_t effective_state_id(uint32_t sid) const { return sid / 2; }
    uint32_t state_step(uint32_t sid) const { return sid % 5; }
    void state_fields(uint32_t sid, hgm::GraphStateFields& f) const {
        f.id = sid;
        f.canonical_id = sid / 2;
        f.step = sid % 5;
    }
    void state_edges(uint32_t sid, std::vector<std::pair<int64_t, std::vector<uint32_t>>>& out) const {
        for (uint32_t k = 0; k < sid % 4; ++k) out.push_back({k, {sid, k, sid + k * 300}});
    }
    uint32_t num_raw_events() const { return m; }
    bool is_valid_event(uint32_t eid) const { return eid % 13 != 12; }
    int64_t effective_event_id(uint32_t eid) const { return eid; }
    uint32_t event_input_state(uint32_t eid) const { return eid % n; }
    uint32_t event_output_state(uint32_t eid) const { return (eid * 7 + 1) % n; }
    bool event_fields(uint32_t eid, hgm::GraphEventFields& f) const {
        if (eid == 5) return false;
        f.reconstructed = eid % 17 == 3;
        f.id = eid;
        f.canonical_id = eid;
        f.rule = eid % 3;
        f.input = event_input_state(eid);
        f.output = event_output_state(eid);
        f.consumed = {static_cast<int64_t>(eid), static_cast<int64_t>(eid) + 1};
        f.produced = {static_cast<int64_t>(eid) + 2};
        f.input_raw = event_input_state(eid);
        f.output_raw = event_output_state(eid);
        return true;
    }
    std::vector<std::pair<uint32_t, uint32_t>> causal_event_pairs() const {
        std::vector<std::pair<uint32_t, uint32_t>> out;
        for (uint32_t e = 1; e < m; ++e) {
            out.push_back({e - 1, e});
            if (e % 3 == 0) out.push_back({e - 3, e});
            if (e % 7 == 0) out.push_back({e - 1, e});
        }
        return out;
    }
    std::vector<std::pair<uint32_t, uint32_t>> branchial_event_pairs() const {
        std::vector<std::pair<uint32_t, uint32_t>> out;
        for (uint32_t e = 0; e + 1 < m; e += 2) out.push_back({e, e + 1});
        return out;
    }
};

const std::vector<std::string> kAllGraphs = {
    "StatesGraph", "CausalGraph", "BranchialGraph", "EvolutionGraph", "EvolutionCausalGraph",
    "EvolutionBranchialGraph", "EvolutionCausalBranchialGraph", "StatesGraphStructure",
    "CausalGraphStructure", "BranchialGraphStructure", "EvolutionCausalBranchialGraphStructure"};

std::vector<uint8_t> graph_bytes(const SyntheticGraphSource& src, const hgm::ParallelFor& par,
                                 bool dedup) {
    hgm::GraphOptions opts;
    opts.edge_deduplication = dedup;
    wxf::Writer w;
    w.write(hgm::build_graph_data(src, kAllGraphs, opts, nullptr, par));
    return w.release_data();
}

enum class GraphTestJob { Chunk };

// A ParallelFor on a job system of `threads` workers, the calling thread running body(0).
struct ThreadsFor {
    job_system::JobSystem<GraphTestJob> system;
    explicit ThreadsFor(size_t threads) : system(threads) { system.start(); }
    hgm::ParallelFor get() {
        return [this](size_t n, const std::function<void(size_t)>& body) {
            for (size_t i = 1; i < n; ++i)
                system.submit(job_system::make_job<GraphTestJob>([&body, i] { body(i); },
                                                                 GraphTestJob::Chunk));
            if (n) body(0);
            system.wait_for_completion();
        };
    }
};

// Two states and two events of SyntheticGraphSource: event 0 (0 -> 1) and event 5 (1 -> 0),
// which has no payload.
struct TwoStateSource : SyntheticGraphSource {
    TwoStateSource() : SyntheticGraphSource{2, 6} {}
    bool state_valid(uint32_t) const { return true; }
    int64_t effective_state_id(uint32_t sid) const { return sid; }
    bool is_valid_event(uint32_t eid) const { return eid == 0 || eid == 5; }
};

}  // namespace

TEST(WxfEncoded, WritesItsBytesVerbatimAtItsPosition) {
    wxf::WXFValueAssociation inner;
    inner.push_back({wxf::WXFValue("A"), wxf::WXFValue(int64_t{300})});
    inner.push_back({wxf::WXFValue("B"), wxf::WXFValue(wxf::WXFValueList{wxf::WXFValue(1.5)})});
    wxf::Writer enc;
    enc.write(wxf::WXFValue(inner));
    wxf::WXFValueAssociation direct, spliced;
    direct.push_back({wxf::WXFValue("x"), wxf::WXFValue(int64_t{1})});
    spliced = direct;
    direct.push_back({wxf::WXFValue("y"), wxf::WXFValue(inner)});
    spliced.push_back({wxf::WXFValue("y"), wxf::WXFValue(wxf::Encoded{enc.release_data()})});
    wxf::Writer a, b;
    a.write(wxf::WXFValue(direct));
    b.write(wxf::WXFValue(spliced));
    EXPECT_EQ(a.data(), b.data());
}

// A StatesGraph of two states and two events, against the value tree spelled out: the state
// payloads, one event payload, and the edge of the event with no payload, which has no Data key.
TEST(GraphMarshalStream, StatesGraphMatchesItsValueTree) {
    const TwoStateSource two;
    hgm::GraphOptions opts;
    wxf::Writer got;
    got.write(hgm::build_graph_data(two, {"StatesGraph"}, opts));

    using V = wxf::WXFValue;
    using L = wxf::WXFValueList;
    using A = wxf::WXFValueAssociation;
    auto edges_of = [&](uint32_t sid) {
        L l;
        std::vector<std::pair<int64_t, std::vector<uint32_t>>> e;
        two.state_edges(sid, e);
        for (const auto& [first, vs] : e) {
            L one{V(first)};
            for (uint32_t v : vs) one.push_back(V(static_cast<int64_t>(v)));
            l.push_back(V(one));
        }
        return V(l);
    };
    auto state = [&](uint32_t sid) {
        return V(A{{V("Id"), V(int64_t(sid))}, {V("CanonicalId"), V(int64_t(sid / 2))},
                   {V("Step"), V(int64_t(sid % 5))}, {V("Edges"), edges_of(sid)},
                   {V("IsInitial"), V(int64_t(sid % 5 == 0))}});
    };
    const V event0(A{{V("Id"), V(int64_t{0})}, {V("CanonicalId"), V(int64_t{0})},
                     {V("RuleIndex"), V(int64_t{0})}, {V("InputState"), V(int64_t{0})},
                     {V("OutputState"), V(int64_t{1})},
                     {V("ConsumedEdges"), V(L{V(int64_t{0}), V(int64_t{1})})},
                     {V("ProducedEdges"), V(L{V(int64_t{2})})},
                     {V("InputStateEdges"), edges_of(0)}, {V("OutputStateEdges"), edges_of(1)}});
    const L edges{
        V(A{{V("From"), V(int64_t{0})}, {V("To"), V(int64_t{1})}, {V("Type"), V("Directed")},
            {V("Data"), event0}}),
        V(A{{V("From"), V(int64_t{1})}, {V("To"), V(int64_t{0})}, {V("Type"), V("Directed")}})};
    const A graph{{V("IsDelta"), V(int64_t{0})},
                  {V("Vertices"), V(L{V(int64_t{0}), V(int64_t{1})})},
                  {V("Edges"), V(edges)},
                  {V("VertexData"), V(A{{V(int64_t{0}), state(0)}, {V(int64_t{1}), state(1)}})}};
    wxf::Writer want;
    want.write(V(A{{V("StatesGraph"), V(graph)}}));
    EXPECT_EQ(got.data(), want.data());
}

// Every graph over 2,000 states and 3,000 events (several chunks each), at 1, 2, 8 and 16
// workers and with the chunks run in reverse order, equals the serial build byte for byte.
TEST(GraphMarshalStream, BytesIndependentOfThreadsAndChunkOrder) {
    const SyntheticGraphSource src{2000, 3000};
    for (bool dedup : {true, false}) {
        const std::vector<uint8_t> serial = graph_bytes(src, hgm::serial_for, dedup);
        ASSERT_GT(serial.size(), 100000u);
        const std::vector<uint8_t> reversed = graph_bytes(
            src,
            [](size_t n, const std::function<void(size_t)>& body) {
                for (size_t i = n; i-- > 0;) body(i);
            },
            dedup);
        EXPECT_EQ(serial, reversed);
        for (size_t threads : {size_t{1}, size_t{2}, size_t{8}, size_t{16}}) {
            ThreadsFor pool(threads);
            EXPECT_EQ(serial, graph_bytes(src, pool.get(), dedup)) << threads << " threads";
        }
    }
}

// =============================================================================
// The "StepStatistics" pool (hgcommon/step_pool_core.hpp)
// =============================================================================

namespace {

// `n` classes over two steps: random small states (1 to 7 edges of arity 1 to 3 over up to 6
// vertices), their records in one word pool, and random weights up to 50.
struct PoolFixture {
    std::vector<uint64_t> pool;
    struct Class {
        uint32_t step;
        uint64_t hash;
        uint64_t word;
        uint64_t weight;
    };
    std::vector<Class> classes;

    explicit PoolFixture(size_t n) {
        std::mt19937_64 rng(20261014);
        std::vector<uint64_t> storage;
        for (size_t i = 0; i < n; ++i) {
            std::vector<std::vector<uint32_t>> edges(1 + rng() % 7);
            for (auto& e : edges) {
                e.resize(1 + rng() % 3);
                for (auto& v : e) v = static_cast<uint32_t>(rng() % 6);
            }
            hg::stats::invariant_record(edges, storage);
            const uint64_t word = pool.size();
            pool.insert(pool.end(), storage.begin(), storage.end());
            classes.push_back({static_cast<uint32_t>(i % 2), 1000 + i, word, 1 + rng() % 50});
        }
    }
    const hgcommon::StateInvariantRecord* record(const Class& c) const {
        return reinterpret_cast<const hgcommon::StateInvariantRecord*>(pool.data() + c.word);
    }
    std::map<uint32_t, hg::stats::StepInput> inputs() const {
        std::map<uint32_t, hg::stats::StepInput> in;
        for (const Class& c : classes) hg::stats::add_class(in[c.step], c.hash, c.weight, record(c));
        return in;
    }
};

std::vector<uint8_t> value_bytes(const wxf::WXFValue& v) {
    wxf::Writer w;
    w.write(v);
    return w.release_data();
}

enum class PoolTestJob { Job };

}  // namespace

// A step's record from its pool made by one part on this thread equals the record made by
// StepStatisticsJobs, which pools 256 classes per job and merges one slot per job, at 1, 2, 8
// and 16 workers, under both weightings.
TEST(StateStatistics, PoolDoesNotDependOnTheClassSplitOrTheThreadCount) {
    const PoolFixture fx(1500);
    for (bool by_class : {false, true}) {
        wxf::WXFValueList serial;
        for (auto& [step, in] : fx.inputs())
            serial.push_back(wxf::WXFValue(hg::stats::step_record(step, in, by_class)));
        const std::vector<uint8_t> want = value_bytes(wxf::WXFValue(serial));
        for (size_t threads : {size_t{1}, size_t{2}, size_t{8}, size_t{16}}) {
            job_system::JobSystem<PoolTestJob> system(threads);
            system.start();
            hg::stats::StepStatisticsJobs jobs(by_class, 0, threads, [&](std::function<void()> f) {
                system.submit(job_system::make_job<PoolTestJob>(std::move(f), PoolTestJob::Job));
            });
            for (auto& [step, in] : fx.inputs()) jobs.submit(step, std::move(in));
            system.wait_for_completion();
            EXPECT_EQ(want, value_bytes(jobs.reply())) << threads << " threads, by_class " << by_class;
        }
    }
}

// The order of a slot's values does not reach the record: two classes added in either order, or
// one class carrying both weights, give one summary.
TEST(StateStatistics, PoolSummariesReadValuesInKeyOrder) {
    std::vector<uint64_t> a, b;
    const auto* ra = hg::stats::invariant_record({{1, 2}, {2, 3}, {3, 1}}, a);
    const auto* rb = hg::stats::invariant_record({{1, 2}, {2, 3}, {3, 4}, {4, 5}}, b);
    auto record = [](std::vector<std::pair<const hgcommon::StateInvariantRecord*, uint64_t>> cls) {
        hg::stats::StepInput in;
        uint64_t hash = 1;
        for (auto [r, w] : cls) hg::stats::add_class(in, hash++, w, r);
        return value_bytes(wxf::WXFValue(
            *assoc_at(wxf::WXFValue(hg::stats::step_record(0, in, false)), "Invariants")));
    };
    EXPECT_EQ(record({{ra, 3}, {rb, 5}}), record({{rb, 5}, {ra, 3}}));
    EXPECT_EQ(record({{ra, 1}, {ra, 2}, {rb, 5}}), record({{ra, 3}, {rb, 5}}));
}

#ifdef HG_GPU_BACKEND
// The device's pool (hg_gpu::step_pool_device) of every class equals the host's
// (stats::pool_classes), entry for entry and total for total, and the record finished from it is
// the host's bytes.
TEST(StateStatistics, DevicePoolEqualsTheHostPool) {
    const PoolFixture fx(3000);
    for (bool by_class : {false, true}) {
        std::vector<hg_gpu::StepPoolPoint> points;
        for (const auto& c : fx.classes)
            points.push_back({c.step, c.word, by_class ? 1 : c.weight, c.weight});
        points.push_back({1, UINT64_MAX, by_class ? 1u : 4u, 4});   // the state with no edges
        std::vector<std::vector<hgcommon::SpEntry>> entries;
        std::vector<hg_gpu::StepPoolTotals> totals;
        ASSERT_TRUE(hg_gpu::step_pool_device(fx.pool, points, 2, entries, totals));
        auto inputs = fx.inputs();
        hg::stats::add_class(inputs[1], 1, 4, nullptr);
        for (auto& [step, in] : inputs) {
            std::vector<const hg::stats::StepClass*> cls;
            for (const auto& [h, c] : in.classes) cls.push_back(&c);
            hg::stats::PoolPart host;
            hg::stats::pool_classes(cls.data(), cls.size(), by_class, true, host);
            ASSERT_EQ(host.entries.size(), entries[step].size()) << "step " << step;
            for (size_t i = 0; i < host.entries.size(); ++i) {
                EXPECT_EQ(host.entries[i].slot, entries[step][i].slot);
                EXPECT_EQ(host.entries[i].key, entries[step][i].key);
                EXPECT_EQ(host.entries[i].weight, entries[step][i].weight);
            }
            for (uint32_t k = 0; k < hgcommon::SG_DISTS; ++k) {
                EXPECT_EQ(host.dist[k].n, totals[step][k].n);
                EXPECT_EQ(host.dist[k].min_key, totals[step][k].min_key);
                EXPECT_EQ(host.dist[k].max_key, totals[step][k].max_key);
                for (uint32_t b = 0; b < hgcommon::SG_DIST_BINS; ++b)
                    EXPECT_EQ(host.dist[k].bins[b], totals[step][k].bins[b]);
            }
            const std::vector<uint8_t> want =
                value_bytes(wxf::WXFValue(hg::stats::step_record(step, in, by_class)));
            in.device_part = std::make_unique<hg::stats::PoolPart>();
            in.device_part->entries = entries[step];
            for (uint32_t k = 0; k < hgcommon::SG_DISTS; ++k) in.device_part->dist[k] = totals[step][k];
            EXPECT_EQ(want, value_bytes(wxf::WXFValue(hg::stats::step_record(step, in, by_class))));
        }
    }
}
#endif

// hgmarshal::record_list, which writes the CausalEdges and BranchialEdges lists, equals the value
// tree of the same records, and gives the same bytes at 1, 2, 8 and 16 workers.
TEST(GraphMarshalStream, RecordListMatchesItsValueTreeAtEveryThreadCount) {
    std::vector<std::array<int64_t, 4>> rows;
    for (int64_t i = 0; i < 5000; ++i) rows.push_back({i, i * 7 - 3, i << 20, -i});
    wxf::WXFValueList tree;
    for (const auto& r : rows)
        tree.push_back(wxf::WXFValue(wxf::WXFValueAssociation{
            {wxf::WXFValue("From"), wxf::WXFValue(r[0])},
            {wxf::WXFValue("To"), wxf::WXFValue(r[1])},
            {wxf::WXFValue("RawFrom"), wxf::WXFValue(r[2])},
            {wxf::WXFValue("RawTo"), wxf::WXFValue(r[3])}}));
    const std::vector<uint8_t> want = value_bytes(wxf::WXFValue(tree));
    EXPECT_EQ(want, value_bytes(hgm::record_list(rows, {"From", "To", "RawFrom", "RawTo"},
                                                 hgm::serial_for)));
    for (size_t threads : {size_t{1}, size_t{2}, size_t{8}, size_t{16}}) {
        ThreadsFor pool(threads);
        EXPECT_EQ(want, value_bytes(hgm::record_list(rows, {"From", "To", "RawFrom", "RawTo"},
                                                     pool.get())))
            << threads << " threads";
    }
}
