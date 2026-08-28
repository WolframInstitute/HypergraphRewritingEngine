#include <gtest/gtest.h>
#include <vector>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <algorithm>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

// =============================================================================
// Canonical determinism gate.
//
// The engine's SEMANTIC output must be a schedule-independent function of
// (rules, initial state, options) -- identical across runs, thread counts, and
// RNG seeds. We fingerprint the output *canonically* (iso-invariant): states by
// their canonical hash, causal/branchial edges as sorted pairs of iso-invariant
// event signatures. This factors out benign id/order churn and detects only
// genuine structural non-determinism.
//
// The matrix includes LOOP-FORMING / recurrence rulesets under quotient exploration:
// a canonical state that recurs across depths has its attribution resolved among
// several producers, which a purely growing rule never exercises, and it is the case
// in which attribution non-determinism shows.
// =============================================================================


namespace {

uint64_t fnv(uint64_t h, uint64_t x) { h ^= x; h *= 1099511628211ULL; return h; }

struct Fingerprint {
    uint64_t states = 0, causal = 0, branchial = 0;
    long num_states = 0, num_events = 0, num_causal = 0, num_branchial = 0;
    long claims = 0, drops = 0, align_fail = 0, badcorr = 0;
    long not_rep = 0, visits = 0;
    long matches = 0, instances = 0, unique = 0;
    uint64_t shape = 0;
    std::vector<uint32_t> shape_v;
    long branchial_stored = 0;
    long stored_before_walk = 0;
    long branchial_pairs = 0;
    long late_submits = 0;
    long dropped_children = 0;
    long invalid_matches = 0;
    std::string warnings;
};

Fingerprint fingerprint(hg::engine::Hypergraph& g) {
    auto canon = [&](hg::engine::StateId s) -> uint64_t {
        return s == hg::engine::INVALID_ID ? 0 : g.get_or_compute_canonical_hash(s);
    };
    auto esig = [&](hg::engine::EventId e) -> uint64_t {
        const hg::engine::Event& x = g.get_event(e);
        return fnv(fnv(fnv(1469598103934665603ULL, canon(x.input_state)),
                       canon(x.output_state)), x.rule_index);
    };

    Fingerprint fp;
    std::vector<uint64_t> sh;
    for (uint32_t s = 0; s < g.num_published_states(); ++s)
        if (g.get_state(s).id != hg::engine::INVALID_ID) sh.push_back(canon(s));
    std::sort(sh.begin(), sh.end());
    fp.states = 1469598103934665603ULL; for (uint64_t v : sh) fp.states = fnv(fp.states, v);
    fp.num_states = static_cast<long>(sh.size());

    // Under quotient the causal relation is reconstructed rather than explored, so it is read
    // from the reconstruction -- which is also what the engine reports. Reading the materialised
    // causal graph here instead would fingerprint an empty set and pass vacuously.
    std::vector<uint64_t> ce;
    if (g.quotient_reconstruction()) {
        // Endpoints keyed on the SCHEDULE-STABLE content triple, not the run identity.
        //
        // hypergraph.hpp says it directly: a run-identity signature's slot components are
        // labels relative to the class frame that run pinned, and on a symmetric class two
        // runs legitimately pin different members of the labelling coset. Fingerprinting the
        // relation under it compares labels and calls a structural identity a difference.
        // Measured: keyed on the run identity this failed 2 of 20 runs on mixed1, always the
        // causal fingerprint alone while states, events and branchial held.
        g.for_each_reconstructed_causal_as(
            /*reduced=*/true,
            [&](uint32_t e) { return g.reconstructed_raw_triple(e); },
            [&](uint64_t p, uint64_t c) { ce.push_back(fnv(fnv(0, p), c)); });
    } else {
        for (const auto& c : g.causal_graph().get_causal_edges()) {
            if (c.producer == hg::engine::INVALID_ID || c.consumer == hg::engine::INVALID_ID) continue;
            ce.push_back(fnv(fnv(0, esig(c.producer)), esig(c.consumer)));
        }
    }
    std::sort(ce.begin(), ce.end());
    fp.causal = 1469598103934665603ULL; for (uint64_t v : ce) fp.causal = fnv(fp.causal, v);
    fp.num_causal = static_cast<long>(ce.size());

    // Branchial, on whichever side the run serves it -- the same split the causal component
    // above makes, and under the same schedule-stable endpoint identity.
    // WAS THE WALK RACING PUSHES STILL IN FLIGHT? num_branchial_edges_ is incremented inside
    // add_branchial_edge, so reading it either side of the walk separates the two explanations
    // for enumerated < stored:
    //   before == after, walk short  -> the edges exist and are unreachable (structural)
    //   after  >  before             -> evolve() returned with pushes outstanding (quiescence)
    // Without this the shortfall is real either way and the cause is a guess.
#ifdef HG_LIST_SEQ_DEBUG
    // WHERE DOES THE CHAIN BREAK? Each node carries the order its CAS won, so a complete chain
    // yields a contiguous run. Report how many stamps the walk reaches, the highest stamp
    // pushed, and the largest single jump between consecutive reached stamps -- one big jump is
    // a chain that lost a contiguous tail, many small ones is repeated single-node loss.
    if (!g.quotient_reconstruction()) {
        std::vector<uint64_t> seqs;
        g.causal_graph().debug_branchial_seqs([&](uint64_t q) { seqs.push_back(q); });
        const uint64_t pushed = g.causal_graph().debug_branchial_pushed();
        uint64_t maxgap = 0, gaps = 0;
        for (size_t i = 1; i < seqs.size(); ++i) {
            const uint64_t d = seqs[i - 1] - seqs[i];   // newest-first, so this is >= 1
            if (d > 1) { ++gaps; if (d - 1 > maxgap) maxgap = d - 1; }
        }
        std::fprintf(stderr,
            "[chain] pushed=%llu reached=%zu missing=%llu gaps=%llu largest_gap=%llu top=%llu\n",
            (unsigned long long)pushed, seqs.size(),
            (unsigned long long)(pushed - seqs.size()), (unsigned long long)gaps,
            (unsigned long long)maxgap,
            seqs.empty() ? 0ull : (unsigned long long)seqs.front());
    }
#endif

    const long stored_before = g.quotient_reconstruction()
        ? 0L : static_cast<long>(g.causal_graph().num_branchial_edges());

    std::vector<uint64_t> be;
    if (g.quotient_reconstruction()) {
        g.for_each_reconstructed_branchial_as(
            [&](uint32_t e) { return g.reconstructed_raw_triple(e); },
            [&](uint64_t a, uint64_t d) {
                be.push_back(a < d ? fnv(fnv(0, a), d) : fnv(fnv(0, d), a));
            });
    } else {
        for (const auto& b : g.causal_graph().get_branchial_edges()) {
            uint64_t a = esig(b.event1), d = esig(b.event2);
            if (a > d) std::swap(a, d);
            be.push_back(fnv(fnv(0, a), d));
        }
    }
    std::sort(be.begin(), be.end());
    fp.branchial = 1469598103934665603ULL; for (uint64_t v : be) fp.branchial = fnv(fp.branchial, v);
    fp.num_branchial = static_cast<long>(be.size());

    // EXACTLY-ONCE, checked on THIS run rather than inferred from a disagreement between runs.
    //
    // add_branchial_edge is reached only when the pair dedup reports a winning claim, so the
    // edge count must equal the number of claimed pairs. The two are maintained by different
    // mechanisms -- a map's occupancy against a counter the winner increments -- so equality
    // tests the map's exactly-once contract instead of restating it.
    //
    // The spread across runs cannot do this job: it reports that two runs disagreed, not which
    // one was wrong. Observed once at 8 threads, 1 of 24 runs, 30064 edges against 30063 with
    // states, events and causal identical -- a duplicate pair, not a lost or extra event. This
    // fires on the run that produced it, with that run's thread count and seed.
    fp.branchial_pairs = g.quotient_reconstruction()
        ? static_cast<long>(g.num_reconstructed_branchial())
        : static_cast<long>(g.causal_graph().num_branchial_pairs_claimed());

    // WHICH SIDE OF THE HANDOFF LOST IT. num_branchial_edges_ is incremented inside
    // add_branchial_edge, so it counts edges STORED; be.size() counts edges ENUMERATED; and
    // branchial_pairs counts distinct keys CLAIMED. Three counters over one quantity split the
    // failure instead of merely reporting it:
    //   stored == claimed, enumerated < stored  -> the edge exists and the walk misses it
    //   stored <  claimed                       -> a winning claim produced no edge
    // Without this the assertion says an edge is missing and cannot say where.
    fp.branchial_stored = g.quotient_reconstruction()
        ? fp.branchial_pairs
        : static_cast<long>(g.causal_graph().num_branchial_edges());
    fp.stored_before_walk = g.quotient_reconstruction() ? fp.branchial_pairs : stored_before;

    fp.num_events = static_cast<long>(g.observable_num_events());
    fp.claims     = static_cast<long>(g.applied_claims());
    fp.drops      = static_cast<long>(g.capture_dropped_no_orbits());
    fp.align_fail = static_cast<long>(g.num_alignment_failures());
    fp.badcorr    = static_cast<long>(g.num_bad_correspondences());
    // THE THREE THE ENGINE ALREADY COUNTED AND NOTHING READ. A firing whose EVENT count moved
    // by one needs to say where the extra application came from, and these separate the
    // candidates: not_rep says whether a different raw state won its class's expansion, visits
    // says how many times an (instance, match) pair was reached, and the shape is the sorted
    // multiset of per-instance application counts -- so a new instance and an existing instance
    // gaining one are distinguishable rather than both reading as "one more event".
    fp.not_rep    = static_cast<long>(g.capture_skipped_not_representative());
    fp.visits     = static_cast<long>(g.applied_visits());
    fp.matches    = static_cast<long>(g.captured_matches());
    fp.instances  = static_cast<long>(g.reconstruction_instances());
    fp.unique     = static_cast<long>(g.applied_unique());
    fp.shape      = g.applied_shape_fingerprint();
    fp.shape_v    = g.applied_shape();
    return fp;
}

Fingerprint run(const std::vector<hg::engine::RewriteRule>& rules,
                const std::vector<std::vector<hg::engine::VertexId>>& init,
                bool quotient, int threads, uint64_t seed, int steps) {
    hg::engine::Hypergraph g;
    g.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::Full);
    hg::engine::ParallelEvolutionEngine e(&g, threads);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(quotient);
    e.set_random_seed(seed);
    for (const auto& r : rules) e.add_rule(r);
    e.evolve(init, steps);
    Fingerprint fp = fingerprint(g);
    // READ BEFORE THE ENGINE GOES OUT OF SCOPE. This is the precondition the quiescence
    // predicate rests on, not a property of the hypergraph, so it comes from the engine.
    fp.late_submits = static_cast<long>(e.late_submits());
    fp.dropped_children = static_cast<long>(e.dropped_fresh_children());
    fp.invalid_matches  = static_cast<long>(g.invalid_matches());
    // A CAPACITY OVERFLOW RETURNS A PARTIAL RESULT AND SAYS SO, by design -- errors are for
    // programmer mistakes, not for a run that outgrew a container. So a truncated run looks
    // exactly like a short one and differs only here, and nothing was reading it.
    for (const std::string& w : e.warnings()) {
        if (!fp.warnings.empty()) fp.warnings += "; ";
        fp.warnings += w;
    }
    return fp;
}

struct Workload {
    const char* name;
    std::vector<hg::engine::RewriteRule> rules;
    std::vector<std::vector<hg::engine::VertexId>> init;
    int steps;
};

std::vector<Workload> workloads() {
    std::vector<Workload> w;
    w.push_back({"WPP",
        {hg::engine::make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
        {{0,1},{0,2}}, 6});
    w.push_back({"mixed1",
        {hg::engine::make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
         hg::engine::make_rule(1).lhs({0,1}).rhs({1,0}).build(),
         hg::engine::make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()},
        {{0,1}}, 6});
    w.push_back({"mixed2",
        {hg::engine::make_rule(0).lhs({0,1}).rhs({1,0}).build(),
         hg::engine::make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()},
        {{0,1}}, 6});
    return w;
}

// Collect the distinct value of each fingerprint component over runs × threads × seeds.
//
// Each divergent fingerprint keeps the configuration that produced it and the counts that came
// with it, because this gate fails about once in thirty invocations and has resisted being
// reproduced on demand -- 1560 targeted runs across both suspect workloads, every thread count
// and both seeds, all deterministic. A gate that can only say "these differed" spends the rare
// firing telling us nothing we did not already know. Saying WHICH configurations differed, and
// whether the COUNTS moved with the hashes, splits the two faults a bare mismatch conflates:
// equal counts with differing hashes is canonicalization, differing counts is exploration.
struct Variant {
    uint64_t fingerprint;
    std::string config;     // first configuration observed to produce it
    long ns, ne, nc, nb;
    // WHERE THE APPLICATIONS WENT, carried so a firing names the mechanism instead of only the
    // symptom. Under quotient every relation is built over (instance, match) applications:
    //   claims - ne   applications that won their claim and were dropped by the width check
    //   drops         captures lost because an endpoint's orbits were not visible yet
    //   align/corr    captures lost aligning a raw state onto its class frame
    // Each of these is a silent drop that changes every relation while leaving the STATE set
    // alone -- the observed shape -- and each was invisible until it was counted.
    long claims, drops, align_fail, badcorr;
    long not_rep, visits;
    long matches, instances, unique;
    uint64_t shape;
    std::vector<uint32_t> shape_v;
};
struct Spread {
    std::set<uint64_t> states, causal, branchial;
    std::set<long> ns, ne, nc, nb;
    std::map<uint64_t, Variant> states_v, causal_v, branchial_v;
    // How many of the `runs` configurations produced each fingerprint. A bare count of DISTINCT
    // fingerprints cannot separate "one run in twenty-four went wrong" from "the runs split evenly",
    // and those two demand different investigations: the first is a race that fires rarely, the
    // second is a configuration axis (thread count, seed) changing the answer every time.
    std::map<uint64_t, int> states_n, causal_n, branchial_n;
    int runs = 0;
};

std::string describe(const Spread& s, const std::map<uint64_t, Variant>& v,
                     const std::map<uint64_t, int>& n, const char* what) {
    if (v.size() < 2) return {};
    std::string out = std::string("\n  ") + what + " took " + std::to_string(v.size()) +
                      " distinct values over " + std::to_string(s.runs) + " runs:";
    for (const auto& [fp, var] : v) {
        const auto it = n.find(fp);
        const int c = it == n.end() ? 0 : it->second;
        out += "\n    " + std::to_string(c) + "/" + std::to_string(s.runs) + " runs  " + var.config +
               "  states=" + std::to_string(var.ns) + " events=" + std::to_string(var.ne) +
               " causal=" + std::to_string(var.nc) + " branchial=" + std::to_string(var.nb) +
               "  [claims=" + std::to_string(var.claims) +
               " width_dropped=" + std::to_string(var.claims - var.ne) +
               " no_orbits=" + std::to_string(var.drops) +
               " not_rep=" + std::to_string(var.not_rep) +
               " visits=" + std::to_string(var.visits) +
               " matches=" + std::to_string(var.matches) +
               " instances=" + std::to_string(var.instances) +
               " claims_minus_unique=" + std::to_string(var.claims - var.unique) +
               " align_fail=" + std::to_string(var.align_fail) +
               " badcorr=" + std::to_string(var.badcorr) +
               " applied_shape=" + std::to_string(var.shape) + "]";
    }
    // THREE SHAPES, NOT TWO, and reading only the STATE count cannot tell them apart. This
    // concluded "counts AGREE -> CANONICALIZATION" on a firing whose event count differed by one
    // (10632 against 10633) purely because both runs had 241 states -- pointing the next reader
    // at hashing when the evidence said an extra application had been claimed.
    // WHICH instance moved. The scalars above cannot separate an instance APPEARING from an
    // existing one gaining an application, and those are different defects: the first is a
    // duplicate instance record, the second is a pair applied twice under two identities.
    if (v.size() == 2) {
        auto it = v.begin();
        std::vector<uint32_t> a = it->second.shape_v;
        std::vector<uint32_t> b = (++it)->second.shape_v;
        if (!a.empty() && !b.empty()) {
            out += "\n  applied shape: " + std::to_string(a.size()) +
                   " applying instances vs " + std::to_string(b.size()) + " applying instances";
            if (a.size() == b.size()) {
                out += " -- SAME COUNT, so one of them gained an application; matches= and\n        instances= above say whether a record was added to draw it from:";
                for (size_t i = 0, shown = 0; i < a.size() && shown < 6; ++i)
                    if (a[i] != b[i]) {
                        out += " [" + std::to_string(i) + "] " + std::to_string(a[i]) + "->" +
                               std::to_string(b[i]);
                        ++shown;
                    }
            } else {
                out += " -- COUNT DIFFERS, so an instance published its first\n        application in one run and none in the other";
            }
        }
    }
    if (s.ns.size() != 1) {
        out += "\n  STATE counts differ -> EXPLORATION: a state exists in one run and not "
               "another";
    } else if (s.ne.size() != 1 || s.nc.size() != 1 || s.nb.size() != 1) {
        out += "\n  states agree but EVENT/CAUSAL/BRANCHIAL counts differ -> RECONSTRUCTION "
               "COMPLETENESS: the same canonical evolution yielded a different number of raw "
               "applications, so an (instance, match) pair was claimed in one run and not the "
               "other. Read `claims` above: it tracks the event count, so an extra event is an "
               "extra claim rather than a lost dedup downstream";
    } else {
        out += "\n  every count agrees and only the fingerprint moved -> ATTRIBUTION: the same "
               "applications, related differently";
    }
    return out;
}

Spread spread(const Workload& w, bool quotient) {
    Spread s;
    for (uint64_t seed : {uint64_t(0xABCDEF), uint64_t(0)})   // fixed then random
        for (int rep = 0; rep < 4; ++rep)
            for (int th : {1, 2, 8, 16, 32}) {
                Fingerprint f = run(w.rules, w.init, quotient, th, seed, w.steps);
                EXPECT_EQ(f.stored_before_walk, f.branchial_stored)
                    << w.name << ": pushes were IN FLIGHT during the walk at threads=" << th
                    << " rep=" << rep << " -- " << f.stored_before_walk << " edges stored before "
                    << "the walk and " << f.branchial_stored << " after. evolve() returned while "
                       "workers were still pushing, so any shortfall is quiescence, not the list.";
                EXPECT_EQ(f.branchial_stored, f.branchial_pairs)
                    << w.name << ": claim/store split at threads=" << th << " rep=" << rep
                    << " -- " << f.branchial_pairs << " distinct keys claimed but "
                    << f.branchial_stored << " edges stored. add_branchial_edge runs on every "
                       "winning claim, so a shortfall here is a claim that won and produced no "
                       "edge.";
                EXPECT_EQ(f.num_branchial, f.branchial_pairs)
                    << w.name << ": branchial dedup admitted a duplicate at threads=" << th
                    << " seed=" << (seed ? "fixed" : "random") << " rep=" << rep
                    << " -- " << f.num_branchial << " edges from "
                    << f.branchial_pairs << " claimed pairs. add_branchial_edge runs only on a "
                       "winning claim, so the two cannot differ unless one key was claimed twice.";
                // THE SILENT DROPS ARE ASSERTED, not merely reported. Each of the four
                // counters below removes an application from the reconstruction while leaving
                // the STATE set intact, so the relations come out short and the shape of the
                // run does not change. Comparing fingerprints across configurations cannot see
                // that: a drop that happens on EVERY run makes every run agree, and the spread
                // is one value. The counters were carried only into the failure diagnostic,
                // which fires after a disagreement has already been found -- so the case where
                // the loss is deterministic had no gate at all.
                //
                // Zero on every quotient run of all three workloads at 1, 2, 8, 16 and 32
                // threads, both seeds, four repetitions.
                if (quotient) {
                    EXPECT_EQ(f.drops, 0)
                        << w.name << " at threads=" << th << ": " << f.drops << " capture(s) "
                        << "dropped because an endpoint's orbits were not visible yet. The "
                           "match is then absent from the class frame, so the replay never "
                           "applies it to any instance and every causal and branchial pair it "
                           "would have produced is missing.";
                    EXPECT_EQ(f.align_fail, 0)
                        << w.name << " at threads=" << th << ": " << f.align_fail
                        << " capture(s) lost aligning a raw state onto its class frame.";
                    EXPECT_EQ(f.badcorr, 0)
                        << w.name << " at threads=" << th << ": " << f.badcorr
                        << " capture(s) lost to a bad vertex correspondence.";
                    // ONE WIN PER KEY, CHECKED ON EVERY RUN. The claim tally counts inserts
                    // that reported a win; applied_unique() walks the keys those wins are
                    // supposed to name. A shortfall is one (instance, match) pair replayed
                    // twice, and the claim is the only thing standing between the two paths
                    // into qc_apply -- qc_add_instance iterating a class's captured matches,
                    // and qc_capture_expansion replaying a new match against the instances
                    // already standing. Comparing runs finds it too, but only as a fingerprint
                    // that differs; this names it where it happens.
                    EXPECT_EQ(f.claims, f.unique)
                        << w.name << " at threads=" << th << ": " << f.claims
                        << " applications won their claim but the applied set holds only "
                        << f.unique << " keys, so " << (f.claims - f.unique)
                        << " (instance, match) pair(s) were claimed twice and replayed twice.";
                    EXPECT_EQ(f.claims, f.num_events)
                        << w.name << " at threads=" << th << ": " << f.claims
                        << " applications won their claim but " << f.num_events << " events "
                        << "exist. Every claim mints an event unless the width check rejects "
                           "it, so the difference is captures whose recorded class width "
                           "disagreed with the instance they were replayed against.";
                }
                // THE ONE THING QUIESCENCE CANNOT DEFEND AGAINST, asserted on every
                // configuration rather than inferred from a shortfall. A job that submits a child
                // AFTER it has been booked complete leaves a window in which the counters agree
                // and every queue is empty while work is still owed -- so evolve() can return
                // early, and the run comes back short with no warning and no other symptom.
                // verification/tla/Quiescence.tla reports exactly that as MCQuiescenceLateSubmit;
                // this is the same precondition checked against the running engine.
                EXPECT_EQ(f.warnings, "")
                    << w.name << " at threads=" << th << " rep=" << rep << ": the run reported "
                    << "a warning, so what it returned is a PARTIAL result and any shortfall "
                    << "against another configuration is that, not non-determinism -- "
                    << f.warnings;
                // AN EVENT THAT NEVER HAPPENED, and the only symptom is a shorter run. A match
                // naming an edge its input state does not hold is refused by Rewriter::apply and
                // returns an empty result, which the caller reads as "produced nothing". Every
                // match is either matched against the state it is applied to or forwarded from a
                // parent that kept it alive, so there is no legitimate way to reach it.
                EXPECT_EQ(f.invalid_matches, 0)
                    << w.name << " at threads=" << th << " rep=" << rep << ": "
                    << f.invalid_matches << " match(es) named an edge their input state does not "
                       "hold and were dropped without being applied.";
                // A SUBTREE THAT WAS NEVER EXPLORED, and the only symptom is a shorter run.
                // Every rewrite creates a NEW raw state, so the set that decides whether to
                // match it cannot already hold that id; if it says otherwise the child and
                // everything below it is dropped silently.
                EXPECT_EQ(f.dropped_children, 0)
                    << w.name << " at threads=" << th << " rep=" << rep << ": "
                    << f.dropped_children << " freshly-created state(s) were reported as already "
                       "matched, so their subtrees were never explored.";
                EXPECT_EQ(f.late_submits, 0)
                    << w.name << " at threads=" << th << " rep=" << rep << ": "
                    << f.late_submits << " submit(s) came from a worker that was not inside a "
                       "job, so quiescence could be declared with a child still owed.";
                s.states.insert(f.states); s.causal.insert(f.causal); s.branchial.insert(f.branchial);
                s.ns.insert(f.num_states); s.ne.insert(f.num_events);
                s.nc.insert(f.num_causal); s.nb.insert(f.num_branchial);

                const std::string cfg = "threads=" + std::to_string(th) +
                                        " seed=" + (seed ? "fixed" : "random") +
                                        " rep=" + std::to_string(rep);
                const Variant var{0, cfg, f.num_states, f.num_events,
                                  f.num_causal, f.num_branchial,
                                  f.claims, f.drops, f.align_fail, f.badcorr,
                                  f.not_rep, f.visits, f.matches, f.instances, f.unique,
                                  f.shape, f.shape_v};
                s.states_v.emplace(f.states, var);
                s.causal_v.emplace(f.causal, var);
                s.branchial_v.emplace(f.branchial, var);
                ++s.states_n[f.states];
                ++s.causal_n[f.causal];
                ++s.branchial_n[f.branchial];
                ++s.runs;
            }
    return s;
}

}  // namespace

// Without quotient the entire semantic output is a pure function of the input.
TEST(CausalDeterminism, NonQuotientFullyDeterministic) {
    for (const auto& w : workloads()) {
        Spread s = spread(w, /*quotient=*/false);
        EXPECT_EQ(s.states.size(), 1u)    << w.name << ": state set non-deterministic"
                                          << describe(s, s.states_v, s.states_n, "state fingerprint");
        EXPECT_EQ(s.causal.size(), 1u)    << w.name << ": causal graph non-deterministic"
                                          << describe(s, s.causal_v, s.causal_n, "causal fingerprint");
        EXPECT_EQ(s.branchial.size(), 1u) << w.name << ": branchial graph non-deterministic"
                                          << describe(s, s.branchial_v, s.branchial_n, "branchial fingerprint");
    }
}

// Under quotient, states / events / branchial are already deterministic; only causal
// attribution is not (the first-writer-wins single producer per canonical edge).
TEST(CausalDeterminism, QuotientStatesEventsBranchialDeterministic) {
    for (const auto& w : workloads()) {
        Spread s = spread(w, /*quotient=*/true);
        // describe() on every one of them. Without it a firing reports only how MANY distinct
        // values there were, which names neither the configuration that differed nor the counts
        // it produced -- and this test fires about once in twenty suite runs, so the report it
        // leaves is the whole evidence.
        EXPECT_EQ(s.states.size(), 1u)    << w.name << ": state set non-deterministic under quotient"
                                          << describe(s, s.states_v, s.states_n, "state fingerprint");
        EXPECT_EQ(s.branchial.size(), 1u) << w.name << ": branchial non-deterministic under quotient"
                                          << describe(s, s.branchial_v, s.branchial_n, "branchial fingerprint");
        EXPECT_EQ(s.ne.size(), 1u)        << w.name << ": event count non-deterministic under quotient"
                                          << describe(s, s.branchial_v, s.branchial_n, "branchial fingerprint");
        EXPECT_EQ(s.nb.size(), 1u)        << w.name << ": branchial count non-deterministic under quotient"
                                          << describe(s, s.branchial_v, s.branchial_n, "branchial fingerprint");
    }
}

// Quotient causal attribution must be order-independent. The run() harness requests TR on,
// but guard_quotient_transitive_reduction() downgrades it to TR-off, so what this verifies
// under quotient is TR-OFF causal determinism; TR-on causal determinism is covered by
// NonQuotientFullyDeterministic.
//
// The guard is a stopgap, not a statement about what is reconstructable. It is there because
// quotient emits causal edges between CANONICAL event ids, whose assignment is
// schedule-dependent, so the reduction tag computed against them is not stable. The
// per-instance reconstruction does retain the raw wiring -- it reproduces raw events, causal
// pairs and branchial edges exactly against full capture across the matrix probe -- so the
// reduced view is reachable from it; nothing here proves otherwise.
TEST(CausalDeterminism, QuotientCausalAttribution) {
    for (const auto& w : workloads()) {
        Spread s = spread(w, /*quotient=*/true);
        // describe() ON THE FIRING, as the sibling gate above already does. A bare "took 2
        // distinct values" cannot separate a race that fired once in a hundred from an axis that
        // changes the answer every time, and cannot say whether the COUNT moved with the
        // fingerprint -- which is the difference between attribution landing elsewhere and
        // attribution going missing. This gate fires rarely enough that the report it leaves is
        // the whole of the evidence.
        EXPECT_EQ(s.causal.size(), 1u)
            << w.name << ": causal attribution non-deterministic under quotient"
            << describe(s, s.causal_v, s.causal_n, "causal fingerprint");
    }
}

// Branchial siblings of one instance must be counted exactly once, whatever order the
// two matches of a pair are applied in.
//
// The pairing used to elect a reporter by match id ("count it if other.id < m.id"), which
// silently assumed id order matched the order matches become visible in the expansion
// list. It does not -- ids come from a global counter while the list is appended
// concurrently -- so a lower-id match could reach the list after a higher-id match had
// already scanned: the higher one never saw it, the lower one dismissed the higher as not
// below it, and the pair was lost by BOTH sides. It reproduced as a branchial count short
// by 2 on roughly one matrix run in four, never on events or causal pairs.
//
// Electing the reporter by application order instead is necessary but not sufficient:
// both sides can observe the other's application claim (claim a, claim b, scan a, scan b),
// so the pair itself has to be claimed. This drives the two smallest configurations that
// exhibited the loss, at the thread count that produced it.
TEST(CausalDeterminism, QuotientBranchialCountedExactlyOnce) {
    // iters is sized from the measured loss rate WITHOUT the fix: dup+dedup/selfloop lost a
    // pair about once in 700 evolutions (observed at iterations 137 and 1310 on two runs), so
    // 4000 catches a regression with high probability in ~2 s. The fan case races far less and
    // is kept short -- it is here because it was one of the two configurations observed to
    // fail, not because it carries the detection.
    struct Case { const char* name; int iters; std::vector<hg::engine::RewriteRule> rules;
                  std::vector<std::vector<hg::engine::VertexId>> init; };
    // dup+dedup: {{x,y}} -> {{x,y}},{{x,y}} together with {{x,y}},{{x,y}} -> {{x,y}}.
    auto dup_dedup = [] {
        return std::vector<hg::engine::RewriteRule>{
            hg::engine::make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build(),
            hg::engine::make_rule(1).lhs({0,1}).lhs({0,1}).rhs({0,1}).build()};
    };
    const std::vector<Case> cases = {
        {"dup+dedup/selfloop", 4000, dup_dedup(), {{0,0}}},
        {"dup+dedup/fan",       400, dup_dedup(), {{0,1},{0,2}}},
    };

    for (const auto& c : cases) {
        size_t expected = 0;
        {   // full capture, single-threaded: the reference the reconstruction must match
            hg::engine::Hypergraph hg;
            hg.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::Full);
            hg::engine::ParallelEvolutionEngine e(&hg, 1);
            e.set_transitive_reduction(false);
            e.set_explore_from_canonical_states_only(false);
            for (const auto& r : c.rules) e.add_rule(r);
            auto in = c.init; e.evolve(in, 3);
            expected = hg.causal_graph().num_branchial_edges();
        }
        ASSERT_GT(expected, 0u) << c.name << ": no branchial edges to compare";

        for (int iter = 0; iter < c.iters; ++iter) {
            hg::engine::Hypergraph hg;
            hg.set_state_canonicalization_mode(hg::engine::StateCanonicalizationMode::Full);
            hg::engine::ParallelEvolutionEngine e(&hg, 8);
            e.set_transitive_reduction(false);
            e.set_explore_from_canonical_states_only(true);
            hg.set_quotient_reconstruction(true);
            for (const auto& r : c.rules) e.add_rule(r);
            auto in = c.init; e.evolve(in, 3);
            ASSERT_EQ(hg.num_reconstructed_branchial(), expected)
                << c.name << ": branchial count differs from full capture on iteration " << iter;
        }
    }
}
