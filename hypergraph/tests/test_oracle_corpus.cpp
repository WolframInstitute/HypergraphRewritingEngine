// Correctness gate across the rule-type space: every workload in the shared corpus
// (single/mixed arity, varied edge counts and connectivity, productive/idempotent/
// reductive, self-loop, disconnected LHS, multi-rule) must match the INDEPENDENT
// brute-force isomorphism oracle exactly, in Full mode, and must do so identically
// at 1 and 4 threads (determinism). This is the "all rule types checked against the
// oracle" guarantee that every optimization must keep passing.
#include <gtest/gtest.h>

#include "reference/oracle_corpus.hpp"

using namespace hypergraph;

namespace {

// The causal relation the run SERVES, as a multiset of endpoint-pair keys.
//
// Which mechanism produced it is the run's business, not the gate's: on the reconstruction
// route hg.causal_graph() holds what full capture left behind, and reading it measures the
// wrong mechanism -- silently, and only on the routings where the two differ.
//
// Endpoints are the schedule-stable content triple where the reconstruction serves, and the
// canonical endpoint states otherwise, because a raw event id means nothing across runs.
inline std::multiset<uint64_t> served_causal_pairs(Hypergraph& hg) {
    auto mix = [](uint64_t h, uint64_t v) { h ^= v; h *= 1099511628211ULL; return h; };
    std::multiset<uint64_t> out;
    if (hg.quotient_reconstruction()) {
        hg.for_each_reconstructed_causal_as(
            hg.causal_graph().transitive_reduction_enabled(),
            [&](uint32_t e) { return hg.reconstructed_raw_triple(e); },
            [&](uint64_t p, uint64_t c) { out.insert(mix(mix(0, p), c)); });
        return out;
    }
    auto esig = [&](EventId e) {
        const Event& x = hg.get_event(e);
        uint64_t h = 1469598103934665603ULL;
        h = mix(h, x.input_state == INVALID_ID ? 0 : hg.get_or_compute_canonical_hash(x.input_state));
        h = mix(h, x.output_state == INVALID_ID ? 0 : hg.get_or_compute_canonical_hash(x.output_state));
        return mix(h, x.rule_index);
    };
    for (const auto& ce : hg.causal_graph().get_causal_edges()) {
        if (ce.producer == INVALID_ID || ce.consumer == INVALID_ID) continue;
        out.insert(esig(ce.producer) * 31 + esig(ce.consumer));
    }
    return out;
}

}  // namespace

TEST(OracleCorpus, EveryRuleTypeMatchesBruteForce) {
    for (const auto& c : oracle::corpus()) {
        bool all_small = true;
        size_t brute = oracle::brute_force_iso_count(c.rules, c.init, c.oracle_steps, &all_small);
        ASSERT_TRUE(all_small)
            << c.name << " (" << c.type << "): state exceeded brute-force size bound at oracle depth";
        size_t full1 = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 1);
        EXPECT_EQ(full1, brute)
            << c.name << " (" << c.type << "): Full-mode count != brute-force oracle";
    }
}

TEST(OracleCorpus, DeterministicAcrossThreadCounts) {
    for (const auto& c : oracle::corpus()) {
        size_t t1 = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 1);
        size_t t4 = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 4);
        size_t t8 = oracle::engine_full_count(c.rules, c.init, c.oracle_steps, 8);
        EXPECT_EQ(t1, t4) << c.name << ": canonical count differs between 1 and 4 threads";
        EXPECT_EQ(t1, t8) << c.name << ": canonical count differs between 1 and 8 threads";
    }
}

// The causal + branchial graph invariants (edge counts, event-pair count) are
// properties of the multiway system, independent of thread scheduling — so they must
// be identical at every thread count, run after run. This is the gate that guards the
// causal/closure/transitive-reduction redesign: any change to that code must keep these
// counts deterministic across 1/4/8/16 threads. Run deeper (more events => more race
// surface) and repeat to shake out synchronization flakiness.
TEST(OracleCorpus, CausalBranchialCountsDeterministicAcrossThreads) {
    for (const auto& c : oracle::corpus()) {
        oracle::Counts ref = oracle::engine_counts(c.rules, c.init, c.measure_steps, 1);
        for (int rep = 0; rep < 3; ++rep) {
            for (unsigned t : {4u, 8u, 16u}) {
                oracle::Counts got = oracle::engine_counts(c.rules, c.init, c.measure_steps, t);
                EXPECT_EQ(got.canonical_states, ref.canonical_states)
                    << c.name << ": canonical_states differ @" << t << " threads (rep " << rep << ")";
                EXPECT_EQ(got.events, ref.events)
                    << c.name << ": events differ @" << t << " threads (rep " << rep << ")";
                EXPECT_EQ(got.causal_edges, ref.causal_edges)
                    << c.name << ": causal_edges differ @" << t << " threads (rep " << rep << ")";
                EXPECT_EQ(got.causal_event_pairs, ref.causal_event_pairs)
                    << c.name << ": causal_event_pairs differ @" << t << " threads (rep " << rep << ")";
                EXPECT_EQ(got.branchial_edges, ref.branchial_edges)
                    << c.name << ": branchial_edges differ @" << t << " threads (rep " << rep << ")";
            }
        }
    }
}

// An artifact the run was not asked to record is not built -- and everything else is untouched.
//
// The requested properties used to gate SERIALIZATION only: the causal and branchial relations
// were built in full and dropped at the output. Gating them at source is only correct if what
// survives is IDENTICAL, so this compares the surviving artifacts as SETS against the all-on
// run rather than as counts: two runs can agree on how many causal pairs there are and disagree
// about which.
TEST(OracleCorpus, RecordSetSkipsOnlyWhatItWasNotAskedFor) {
    struct Fp {
        size_t states = 0, events = 0, causal_pairs = 0, branchial = 0, state_event_entries = 0;
        std::multiset<uint64_t> state_hashes, causal, branchial_set, state_events;
    };

    auto run = [](const oracle::Case& c, RecordSet rs) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        hg.set_record_set(rs);
        ParallelEvolutionEngine engine(&hg, 4);
        engine.set_transitive_reduction(true);
        for (const auto& r : c.rules) engine.add_rule(r);
        engine.evolve(c.init, c.measure_steps);

        Fp f;
        f.states = hg.num_canonical_states();
        f.events = hg.num_events();
        f.causal_pairs = hg.observable_num_causal_pairs(
            hg.causal_graph().transitive_reduction_enabled());
        f.branchial = hg.observable_num_branchial();
        for (uint32_t s = 0; s < hg.num_states(); ++s)
            if (hg.get_state(s).id != INVALID_ID)
                f.state_hashes.insert(hg.get_or_compute_canonical_hash(s));
        auto esig = [&](EventId e) {
            const Event& x = hg.get_event(e);
            uint64_t h = 1469598103934665603ULL;
            auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
            mix(x.input_state == INVALID_ID ? 0 : hg.get_or_compute_canonical_hash(x.input_state));
            mix(x.output_state == INVALID_ID ? 0 : hg.get_or_compute_canonical_hash(x.output_state));
            mix(x.rule_index);
            return h;
        };
        f.causal = served_causal_pairs(hg);
        // The per-state event list, as (input state, event) content pairs.
        hg.causal_graph().for_each_state_events([&](StateId in, auto* list) {
            list->for_each([&](EventId e) {
                ++f.state_event_entries;
                f.state_events.insert(
                    (in == INVALID_ID ? 0 : hg.get_or_compute_canonical_hash(in)) * 31 + esig(e));
            });
        });
        for (const auto& b : hg.causal_graph().get_branchial_edges()) {
            if (b.event1 == INVALID_ID || b.event2 == INVALID_ID) continue;
            const uint64_t a = esig(b.event1), d = esig(b.event2);
            f.branchial_set.insert(a < d ? a * 31 + d : d * 31 + a);
        }
        return f;
    };

    bool any_causal = false, any_branchial = false, any_state_events = false;
    for (const auto& c : oracle::corpus()) {
        const Fp all = run(c, RecordSet{true, true, true});
        if (all.causal_pairs) any_causal = true;
        if (all.branchial) any_branchial = true;
        if (all.state_event_entries) any_state_events = true;

        // Causal off: no causal relation, and everything else unchanged.
        const Fp no_c = run(c, RecordSet{false, true, true});
        EXPECT_EQ(no_c.causal_pairs, 0u) << c.name << ": causal was recorded when unrequested";
        EXPECT_EQ(no_c.states, all.states) << c.name << ": dropping causal moved the state count";
        EXPECT_EQ(no_c.events, all.events) << c.name << ": dropping causal moved the event count";
        EXPECT_EQ(no_c.state_hashes, all.state_hashes) << c.name << ": dropping causal moved the states";
        EXPECT_EQ(no_c.branchial_set, all.branchial_set)
            << c.name << ": dropping causal changed the branchial relation";
        EXPECT_EQ(no_c.state_events, all.state_events)
            << c.name << ": dropping causal changed the per-state event list";

        // Branchial off: likewise.
        const Fp no_b = run(c, RecordSet{true, false, true});
        EXPECT_EQ(no_b.branchial, 0u) << c.name << ": branchial was recorded when unrequested";
        EXPECT_EQ(no_b.states, all.states) << c.name << ": dropping branchial moved the state count";
        EXPECT_EQ(no_b.events, all.events) << c.name << ": dropping branchial moved the event count";
        EXPECT_EQ(no_b.state_hashes, all.state_hashes) << c.name << ": dropping branchial moved the states";
        EXPECT_EQ(no_b.causal, all.causal)
            << c.name << ": dropping branchial changed the causal relation";
        // The two branchial artifacts are independent: the pair relation can go without
        // taking the per-state event list with it, which is what an all-siblings view reads.
        EXPECT_EQ(no_b.state_events, all.state_events)
            << c.name << ": dropping the branchial PAIRS also dropped the per-state event list";

        const Fp no_se = run(c, RecordSet{true, true, false});
        EXPECT_EQ(no_se.state_event_entries, 0u)
            << c.name << ": the per-state event list was recorded when unrequested";
        EXPECT_EQ(no_se.branchial_set, all.branchial_set)
            << c.name << ": dropping the per-state event list changed the branchial relation";
        EXPECT_EQ(no_se.causal, all.causal)
            << c.name << ": dropping the per-state event list changed the causal relation";

        // Neither: the evolution itself is untouched.
        const Fp none = run(c, RecordSet{false, false, false});
        EXPECT_EQ(none.causal_pairs, 0u) << c.name;
        EXPECT_EQ(none.branchial, 0u) << c.name;
        EXPECT_EQ(none.state_hashes, all.state_hashes)
            << c.name << ": recording nothing moved the states";
        EXPECT_EQ(none.events, all.events) << c.name << ": recording nothing moved the event count";
    }

    // Without these the equalities above are satisfied by a corpus that produces neither
    // relation, so the gate would pass on an engine that never records either.
    EXPECT_TRUE(any_causal) << "no corpus workload produced a causal relation";
    EXPECT_TRUE(any_branchial) << "no corpus workload produced a branchial relation";
    EXPECT_TRUE(any_state_events) << "no corpus workload produced a per-state event list";
}

// Serial execution produces the same graph as the threaded engine, on every corpus workload.
//
// Serial is not "one worker": no thread is spawned and every job runs inline on the calling
// thread. That is a different execution path through the job system -- the injector FIFO rather
// than the work-stealing deques -- so it is compared against the threaded engine rather than
// assumed equivalent to it. The comparison is on the graph, not on ids: state and event ids are
// handed out in arrival order, and the two paths arrive in different orders.
TEST(OracleCorpus, SerialExecutionMatchesTheThreadedEngine) {
    using Mode = ParallelEvolutionEngine::ExecutionMode;

    struct Fp {
        size_t states = 0, events = 0, causal = 0, branchial = 0;
        std::multiset<uint64_t> state_hashes, causal_pairs;
    };
    auto run = [](const oracle::Case& c, Mode mode, size_t threads) {
        Hypergraph hg;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        ParallelEvolutionEngine e(&hg, threads, mode);
        e.set_transitive_reduction(true);
        for (const auto& r : c.rules) e.add_rule(r);
        e.evolve(c.init, c.measure_steps);

        Fp f;
        f.states = hg.num_canonical_states();
        f.events = hg.num_events();
        f.causal = hg.observable_num_causal_pairs(
            hg.causal_graph().transitive_reduction_enabled());
        f.branchial = hg.observable_num_branchial();
        for (uint32_t s = 0; s < hg.num_states(); ++s)
            if (hg.get_state(s).id != INVALID_ID)
                f.state_hashes.insert(hg.get_or_compute_canonical_hash(s));
        f.causal_pairs = served_causal_pairs(hg);
        return f;
    };

    for (const auto& c : oracle::corpus()) {
        const Fp threaded = run(c, Mode::Parallel, 4);
        const Fp serial   = run(c, Mode::Serial, 0);

        EXPECT_EQ(serial.states, threaded.states) << c.name << ": serial found a different state count";
        EXPECT_EQ(serial.events, threaded.events) << c.name << ": serial found a different event count";
        EXPECT_EQ(serial.causal, threaded.causal) << c.name << ": serial built a different causal relation size";
        EXPECT_EQ(serial.branchial, threaded.branchial) << c.name << ": serial built a different branchial size";
        EXPECT_EQ(serial.state_hashes, threaded.state_hashes) << c.name << ": serial explored different states";
        EXPECT_EQ(serial.causal_pairs, threaded.causal_pairs) << c.name << ": serial built a different causal relation";
    }
}

// A serial engine spawns no thread, which is the property a target without threads needs. One
// worker is a different thing and the two must not be conflated.
TEST(OracleCorpus, SerialSpawnsNoWorkerAndOneWorkerIsNotSerial) {
    using Mode = ParallelEvolutionEngine::ExecutionMode;
    Hypergraph hs, hp;
    ParallelEvolutionEngine serial(&hs, 0, Mode::Serial);
    ParallelEvolutionEngine one_worker(&hp, 1, Mode::Parallel);

    EXPECT_TRUE(serial.is_serial());
    EXPECT_EQ(serial.num_threads(), 1u) << "serial runs on the caller's thread, so it reports one";
    EXPECT_FALSE(one_worker.is_serial()) << "num_threads = 1 is a worker thread, not serial";
}

// Continuing a run gives the same graph as asking for the total in the first place.
//
// evolve_more resumes from the frontier the budget stopped at. What must hold is that the split
// run and the whole run are the same evolution -- compared on the graph, since state and event
// ids are handed out in arrival order and the two paths arrive differently.
TEST(OracleCorpus, ContinuingARunMatchesRunningItInOneCall) {
    struct Fp {
        size_t states = 0, events = 0, causal = 0, branchial = 0;
        std::multiset<uint64_t> state_hashes, causal_pairs;
    };
    auto fingerprint = [](Hypergraph& hg) {
        Fp f;
        f.states = hg.num_canonical_states();
        f.events = hg.num_events();
        f.causal = hg.observable_num_causal_pairs(
            hg.causal_graph().transitive_reduction_enabled());
        f.branchial = hg.observable_num_branchial();
        for (uint32_t s = 0; s < hg.num_states(); ++s)
            if (hg.get_state(s).id != INVALID_ID)
                f.state_hashes.insert(hg.get_or_compute_canonical_hash(s));
        f.causal_pairs = served_causal_pairs(hg);
        return f;
    };

    size_t continued_anything = 0;
    for (const auto& c : oracle::corpus()) {
        if (c.measure_steps < 2) continue;
        const size_t first = c.measure_steps - 1;

        Hypergraph whole;
        whole.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        {
            ParallelEvolutionEngine e(&whole, 4);
            e.set_transitive_reduction(true);
            for (const auto& r : c.rules) e.add_rule(r);
            e.evolve(c.init, c.measure_steps);
        }

        Hypergraph split;
        split.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
        {
            ParallelEvolutionEngine e(&split, 4);
            e.set_transitive_reduction(true);
            for (const auto& r : c.rules) e.add_rule(r);
            e.set_continuable(true);
            e.evolve(c.init, first);
            const size_t after_first = split.num_canonical_states();
            e.evolve_more(c.measure_steps - first);
            if (split.num_canonical_states() > after_first) ++continued_anything;
        }

        const Fp a = fingerprint(whole), b = fingerprint(split);
        std::printf("[cont %-18s] whole s=%zu e=%zu c=%zu b=%zu | split s=%zu e=%zu c=%zu b=%zu\n",
                    c.name, a.states, a.events, a.causal, a.branchial,
                    b.states, b.events, b.causal, b.branchial);
        EXPECT_EQ(b.states, a.states) << c.name << ": continuing found a different state count";
        EXPECT_EQ(b.events, a.events) << c.name << ": continuing found a different event count";
        EXPECT_EQ(b.causal, a.causal) << c.name << ": continuing built a different causal size";
        EXPECT_EQ(b.branchial, a.branchial) << c.name << ": continuing built a different branchial size";
        EXPECT_EQ(b.state_hashes, a.state_hashes) << c.name << ": continuing explored different states";
        EXPECT_EQ(b.causal_pairs, a.causal_pairs) << c.name << ": continuing built a different causal relation";
    }

    // Without this the equalities hold for a corpus where the last step adds nothing, which
    // would pass on an evolve_more that did nothing at all.
    EXPECT_GT(continued_anything, 0u)
        << "no workload grew when continued, so the continuation was never exercised";
}

// A run that was not made continuable says so, rather than returning the graph it already had.
//
// The frontier costs arena on every run, so it is off by default; the failure mode that
// creates is a continuation that silently does nothing and returns a graph that reads as
// converged. That is a programmer error, and the engine's policy is that those are raised.
TEST(OracleCorpus, ContinuingARunThatWasNotMadeContinuableIsAnError) {
    // The corpus is returned BY VALUE; binding a reference to .front() of the temporary does
    // not extend its lifetime, and the dangling case read as an engine with no rules.
    const auto corpus = oracle::corpus();
    const auto& c = corpus.front();
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, 2);
    for (const auto& r : c.rules) e.add_rule(r);
    EXPECT_FALSE(e.continuable()) << "recording the frontier must be opt-in";
    e.evolve(c.init, 2);

    EXPECT_THROW(e.evolve_more(1), std::runtime_error)
        << "evolve_more returned quietly on a run with no frontier, so the caller gets the graph "
        << "it already had and nothing says the continuation did not happen";
}
