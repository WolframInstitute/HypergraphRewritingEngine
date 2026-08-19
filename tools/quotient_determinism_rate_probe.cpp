// tools/quotient_determinism_rate_probe.cpp
//
// Measure the FAILURE RATE of quotient causal determinism, rather than answering pass/fail.
//
// CausalDeterminism.QuotientCausalAttribution failed exactly once, during a full-suite run:
// workload WPP, two distinct causal fingerprints where there must be one. It then did not
// reproduce in 90 standalone runs of the test or 6 full-suite runs. A gate that answers
// pass/fail cannot tell "fixed" from "did not fire this time", and acting on one sample per
// arm is precisely how an innocent change got convicted earlier in this work.
//
// So: run the sweep many times, report the rate, and make each failure self-diagnosing by
// dumping the engine's own alignment counters alongside the differing fingerprints. Those
// counters (frame disagreements, alignment failures, bad correspondences) are already
// maintained; nothing was reading them.
//
// The single observation came from inside the full suite, so scheduling appears to matter --
// hence --load, which runs background threads to keep the machine busy the way a suite does.
//
// Usage: quotient_determinism_rate_probe [iterations] [--load N] [--workload NAME]

#include "hypergraph/parallel_evolution.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <fstream>

using namespace hypergraph;
using Rules = std::vector<RewriteRule>;
using Init = std::vector<std::vector<VertexId>>;

static uint64_t fnv(uint64_t h, uint64_t x) { h ^= x; h *= 1099511628211ULL; return h; }

struct Workload { const char* name; Rules rules; Init init; int steps; };

static std::vector<Workload> workloads() {
    return {
        {"WPP",
         {make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build()},
         {{0,1},{0,2}}, 6},
        {"mixed1",
         {make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
          make_rule(1).lhs({0,1}).rhs({1,0}).build(),
          make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build()},
         {{0,1}}, 6},
        {"mixed2",
         {make_rule(0).lhs({0,1}).rhs({1,0}).build(),
          make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build()},
         {{0,1}}, 6},
    };
}

// One run's causal fingerprint, plus the engine counters that would explain a disagreement.
struct Sample {
    uint64_t causal_fp;
    uint64_t branchial_fp;
    uint64_t states_fp;
    size_t   branchial_pairs, canonical_states, instances;
    // THE SAME RELATION, COUNTED TWO WAYS. branchial_pairs is what the enumeration yields;
    // branchial_counted is what the replay incremented as it emitted. Neither engine stores the
    // pairs any more, so these are two independent routes to one number and a disagreement says
    // WHICH side is wrong -- the replay emitting a pair twice, or the walk that re-derives them
    // reaching a different set.
    size_t   branchial_counted;
    // APPLICATIONS. Every relation the quotient route serves is built over these: a causal pair
    // is a producer and a consumer of one slot, a branchial pair two applications of one
    // instance sharing one. The (instance, match) rendezvous is what decides that an application
    // happens at all -- each side publishes and then scans for the other -- and a pair missed by
    // BOTH scans is an application that never runs. That would leave the canonical state and
    // event counts untouched, since identity is over classes, while changing both relations:
    // exactly the shape the suite reports.
    size_t   applications;
    size_t   claims;          // distinct (instance, match) keys in the replay's claim set
    size_t   cap_no_orbits;   // captures dropped because an endpoint's orbits were not visible
    size_t   cap_not_rep;     // captures skipped because another raw state represents the class
    uint64_t shape_fp;        // sorted multiset of per-instance application counts
    size_t   frame_disagree, align_fail, align_badcorr;
    size_t   events, causal_pairs;
    // The relation the run does NOT serve on this route, carried alongside so the baseline line
    // shows both magnitudes: which of the two a fingerprint covers is the whole question.
    size_t   full_capture_pairs;
    int      threads;
    uint64_t seed;
};

// When non-empty, run_once writes its sorted relations here: one file per call, suffixed by a
// counter, so two runs of the same configuration can be diffed pair by pair.
static std::string g_dump_prefix;
static int g_dump_seq = 0;

static Sample run_once(const Workload& w, int threads, uint64_t seed) {
    Hypergraph g;
    g.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&g, threads);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(true);
    e.set_random_seed(seed);
    for (const auto& r : w.rules) e.add_rule(r);
    Init in = w.init;
    e.evolve(in, w.steps);

    // The relation the run SERVES. Quotient exploration routes the reconstruction, whose pairs
    // live in the qc_ structures; fingerprinting CausalGraph there would hash whatever full
    // capture happened to leave behind rather than the answer a caller receives.
    //
    // Endpoints are the schedule-stable content triple hash(input class, output class, rule),
    // which is what for_each_reconstructed_causal_as's contract requires of a cross-thread
    // comparison: the run identity's slot components are labels relative to the class frame THIS
    // run pinned, and two runs may legitimately pin different members of the labelling coset.
    std::vector<uint64_t> ce;
    if (g.quotient_reconstruction()) {
        g.for_each_reconstructed_causal_as(
            /*reduced=*/true,
            [&](uint32_t ev) { return g.reconstructed_raw_triple(ev); },
            [&](uint64_t p, uint64_t c) { ce.push_back(fnv(fnv(0, p), c)); });
    } else {
        auto canon = [&](StateId s) -> uint64_t {
            return s == INVALID_ID ? 0 : g.get_or_compute_canonical_hash(s);
        };
        auto esig = [&](EventId ev) -> uint64_t {
            const Event& x = g.get_event(ev);
            return fnv(fnv(fnv(0, canon(x.input_state)), canon(x.output_state)), x.rule_index);
        };
        for (const auto& c : g.causal_graph().get_causal_edges()) {
            if (c.producer == INVALID_ID || c.consumer == INVALID_ID) continue;
            ce.push_back(fnv(fnv(0, esig(c.producer)), esig(c.consumer)));
        }
    }
    // Branchial under the same schedule-stable endpoint identity, kept as its OWN fingerprint:
    // a disagreement has to name which relation moved, which one combined hash cannot.
    std::vector<uint64_t> be;
    if (g.quotient_reconstruction()) {
        g.for_each_reconstructed_branchial_as(
            [&](uint32_t ev) { return g.reconstructed_raw_triple(ev); },
            [&](uint64_t a, uint64_t b) {
                be.push_back(a < b ? fnv(fnv(0, a), b) : fnv(fnv(0, b), a));
            });
    } else {
        for (const auto& x : g.causal_graph().get_branchial_edges()) {
            if (x.event1 == INVALID_ID || x.event2 == INVALID_ID) continue;
            const uint64_t a = g.get_event(x.event1).signature;
            const uint64_t b = g.get_event(x.event2).signature;
            be.push_back(a < b ? fnv(fnv(0, a), b) : fnv(fnv(0, b), a));
        }
    }

    // The canonical STATE set, the gate's first column. Isomorphism hashes, sorted, so it is
    // comparable across runs that number states differently.
    std::vector<uint64_t> st;
    for (StateId s = 0; s < g.num_published_states(); ++s) st.push_back(g.get_or_compute_canonical_hash(s));
    std::sort(st.begin(), st.end());
    st.erase(std::unique(st.begin(), st.end()), st.end());

    auto hash_all = [](std::vector<uint64_t>& v) {
        std::sort(v.begin(), v.end());
        uint64_t h = 1469598103934665603ULL;
        for (uint64_t x : v) h = fnv(h, x);
        return h;
    };
    const uint64_t fp = hash_all(ce);
    const uint64_t bfp = hash_all(be);
    const uint64_t sfp = hash_all(st);

    if (!g_dump_prefix.empty()) {
        char path[512];
        std::snprintf(path, sizeof path, "%s.%s.th%d.seed%llx.%d", g_dump_prefix.c_str(), w.name,
                      threads, (unsigned long long)seed, g_dump_seq++);
        // The raw EVENT multiset, so a relation difference can be told apart from a difference
        // in the applications the relation is built over.
        std::vector<uint64_t> ev;
        g.for_each_reconstructed_raw_triple([&](uint64_t t) { ev.push_back(t); });
        std::sort(ev.begin(), ev.end());

        std::ofstream f(path);
        for (uint64_t x : ev) f << "E " << std::hex << x << "\n";
        for (uint64_t x : st) f << "S " << std::hex << x << "\n";
        for (uint64_t x : ce) f << "C " << std::hex << x << "\n";
        for (uint64_t x : be) f << "B " << std::hex << x << "\n";
    }

    return { fp, bfp, sfp, be.size(), st.size(), g.num_reconstructed_instances(),
             g.num_reconstructed_branchial(), g.applied_scans(), g.applied_claims(),
             g.capture_dropped_no_orbits(), g.capture_skipped_not_representative(),
             g.applied_shape_fingerprint(),
             g.num_frame_alignment_disagreements(), g.num_alignment_failures(),
             g.num_bad_correspondences(), g.observable_num_events(), ce.size(),
             g.causal_graph().num_causal_event_pairs(), threads, seed };
}

int main(int argc, char** argv) {
    int iterations = 200;
    int load_threads = 0;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) g_dump_prefix = argv[++i];
        else if (!std::strcmp(argv[i], "--load") && i + 1 < argc) load_threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--workload") && i + 1 < argc) only = argv[++i];
        else iterations = std::atoi(argv[i]);
    }

    // Background load: the one observed failure came from inside a full suite run, so the
    // machine being busy is part of the conditions being reproduced, not noise to remove.
    std::atomic<bool> stop{false};
    std::vector<std::thread> load;
    for (int i = 0; i < load_threads; ++i)
        load.emplace_back([&stop] { volatile uint64_t x = 1; while (!stop) x = x * 6364136223846793005ULL + 1; });

    std::printf("quotient causal determinism, %d iterations of the full sweep"
                " (threads {1,2,8,16,32} x seeds {fixed, random}), load=%d\n",
                iterations, load_threads);

    size_t total = 0, failed = 0;
    for (const auto& w : workloads()) {
        if (!only.empty() && only != w.name) continue;
        // What is being fingerprinted, before any verdict about it. A fingerprint over an empty
        // relation agrees with itself forever, so the magnitude is part of the reading.
        {
            Sample b = run_once(w, 1, 0xABCDEF);
            std::printf("  %-8s baseline: states=%zu instances=%zu events=%zu causal=%zu"
                        " branchial=%zu (full_capture_causal=%zu)\n", w.name,
                        b.canonical_states, b.instances, b.events, b.causal_pairs,
                        b.branchial_pairs, b.full_capture_pairs);
            // IS THE SUSPECTED MECHANISM EVEN ACTIVE? The attribution race is the frame claim:
            // a class's frame is taken first-writer-wins and every other member aligns onto it
            // up to an automorphism. If these are all zero the alignment freedom is never
            // exercised on this workload and the cause is elsewhere, which is a different
            // search. Reported always, not only on a disagreement, because a zero here is the
            // informative reading.
            std::printf("  %-8s alignment: frame_disagreements=%zu failures=%zu"
                        " bad_correspondences=%zu\n", w.name,
                        b.frame_disagree, b.align_fail, b.align_badcorr);
            // THE TWO ROUTES TO ONE NUMBER, reported every run. The replay counts each pair as
            // it emits it; the walk re-derives them from the applications. Equal is the
            // invariant; unequal names which side to look at.
            // n^2/2^65 with n the DISTINCT claim count: the probability that two distinct
            // (instance, match) pairs mix to one key and the second application is dropped.
            {
                const double n = static_cast<double>(b.claims);
                std::printf("  %-8s claims: distinct=%zu  applications=%zu  "
                            "P(collision)=%.3g per run\n",
                            w.name, b.claims, b.applications,
                            n * n / 36893488147419103232.0);
            }
            std::printf("  %-8s capture: dropped_no_orbits=%zu skipped_not_rep=%zu\n",
                        w.name, b.cap_no_orbits, b.cap_not_rep);
            std::printf("  %-8s branchial: enumerated=%zu counted=%zu %s\n", w.name,
                        b.branchial_pairs, b.branchial_counted,
                        b.branchial_pairs == b.branchial_counted ? "agree" : "DISAGREE");
            if (b.causal_pairs == 0 || b.branchial_pairs == 0 || b.canonical_states == 0)
                std::printf("  %-8s A FINGERPRINTED RELATION IS EMPTY -- it agrees with itself"
                            " forever and constrains nothing\n", w.name);
        }
        size_t w_fail = 0;
        for (int it = 0; it < iterations; ++it) {
            std::set<uint64_t> fps, bfps, sfps;
            // THE FIRST-ORDER QUANTITY. Under quotient both relations are derived from the
            // replay's APPLICATIONS: a causal pair is a producer and a consumer of one slot, a
            // branchial pair is two applications of one instance sharing one. If the instance
            // set varies between runs then both relations vary with it, and a fingerprint
            // spread is the symptom rather than the fault. Nothing was watching this.
            std::set<size_t> inst_counts, event_counts, app_counts;
            std::set<uint64_t> shapes;
            std::vector<Sample> samples;
            for (uint64_t seed : {uint64_t(0xABCDEF), uint64_t(0)})
                for (int th : {1, 2, 8, 16, 32}) {
                    Sample s = run_once(w, th, seed);
                    // THE INVARIANT, CHECKED EVERY RUN rather than only on the baseline: the
                    // replay counts each branchial pair as it emits it, the walk re-derives them
                    // from the applications, and neither engine stores the pairs. A run where
                    // these differ has already told us which side is wrong, and it is worth more
                    // than the fingerprint spread that would otherwise be the only symptom.
                    if (s.branchial_pairs != s.branchial_counted) {
                        std::printf("  %s SPLIT at threads=%d seed=%s: enumerated %zu,"
                                    " counted %zu (%+lld)\n",
                                    w.name, th, seed ? "fixed" : "random",
                                    s.branchial_pairs, s.branchial_counted,
                                    (long long)s.branchial_pairs - (long long)s.branchial_counted);
                    }
                    if (s.cap_no_orbits != 0)
                        std::printf("  %s DROPPED CAPTURE at threads=%d seed=%s: %zu match(es) "
                                    "lost because an endpoint's orbits were not visible yet\n",
                                    w.name, th, seed ? "fixed" : "random", s.cap_no_orbits);
                    shapes.insert(s.shape_fp);
                    inst_counts.insert(s.instances);
                    app_counts.insert(s.applications);
                    event_counts.insert(s.events);
                    fps.insert(s.causal_fp);
                    bfps.insert(s.branchial_fp);
                    sfps.insert(s.states_fp);
                    samples.push_back(s);
                }
            ++total;
            if (shapes.size() > 1) {
                std::printf("  %s iteration %d: THE APPLIED LISTS ARE SHAPED DIFFERENTLY across "
                            "%zu runs -- same applications, distributed over instances "
                            "differently, which is what makes the pair count move while the "
                            "event count does not\n", w.name, it, shapes.size());
            }
            if (app_counts.size() > 1) {
                std::printf("  %s iteration %d: APPLICATIONS VARY -- {", w.name, it);
                for (size_t c : app_counts) std::printf(" %zu", c);
                std::printf(" }  (the relations are built over these, so they vary with them)\n");
            }
            if (inst_counts.size() > 1 || event_counts.size() > 1) {
                std::printf("  %s iteration %d: THE REPLAY ITSELF VARIES -- instance counts {",
                            w.name, it);
                for (size_t c : inst_counts) std::printf(" %zu", c);
                std::printf(" }, event counts {");
                for (size_t c : event_counts) std::printf(" %zu", c);
                std::printf(" }\n");
            }
            if (fps.size() > 1 || bfps.size() > 1 || sfps.size() > 1) {
                ++failed; ++w_fail;
                std::printf("  %s iteration %d: distinct fingerprints -- states %zu,"
                            " causal %zu, branchial %zu\n",
                            w.name, it, sfps.size(), fps.size(), bfps.size());
                for (const auto& s : samples)
                    std::printf("      th=%-2d seed=%-8llx  states=%016llx/%-5zu"
                                " causal=%016llx/%-6zu branchial=%016llx/%-6zu events=%-6zu"
                                " inst=%-6zu frame_disagree=%zu align_fail=%zu bad_corr=%zu\n",
                                s.threads, (unsigned long long)s.seed,
                                (unsigned long long)s.states_fp, s.canonical_states,
                                (unsigned long long)s.causal_fp, s.causal_pairs,
                                (unsigned long long)s.branchial_fp, s.branchial_pairs,
                                s.events, s.instances, s.frame_disagree, s.align_fail,
                                s.align_badcorr);
            }
        }
        std::printf("  %-8s %zu/%d iterations disagreed (%.2f%%)\n",
                    w.name, w_fail, iterations, 100.0 * double(w_fail) / double(iterations));
    }

    stop = true;
    for (auto& t : load) t.join();
    std::printf("\nTOTAL: %zu/%zu sweeps disagreed (%.3f%%)\n",
                failed, total, total ? 100.0 * double(failed) / double(total) : 0.0);
    return failed ? 1 : 0;
}
