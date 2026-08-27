// Which connected join order examines the fewest candidates, and how far the shipped
// order is from it.
//
// WHY A NEW PROBE. tools/ has cpu_rank_order_probe and gpu_rank_order_probe; both are about
// CANONICAL EDGE RANK, a different order entirely. Nothing measures the JOIN order, and the
// join order is chosen by RewriteRule::compute_match_order, whose score is static rule
// structure -- repeated variables worth 100, connected neighbours worth 1 -- and never reads a
// single index bucket. Whether that greedy static choice is the counting optimum among
// connected orders has never been measured.
//
// COUNTS, NOT TIMINGS. The quantity reported is the number of CANDIDATES the join examines:
// every edge the enumerator hands to the recursion, at every depth, before injectivity and
// binding filter it. That is the work the order controls, it is deterministic, and it does not
// move with machine load -- which matters because a wall-clock join-order comparison on a
// contended box measures the box.
//
// NO RULE IS REIMPLEMENTED. The recursion is hgcommon::join_dfs and the enumeration is the
// matcher's own HostJoinContext::for_each_candidate, which delegates to generate_candidates.
// This file supplies a Ctx that FORWARDS to HostJoinContext and increments a counter on the way
// through, plus the driver that swaps match_order and enumerates the connected permutations.
// The order under test is written into RewriteRule::match_order, which is the same field the
// engine reads, so an order that wins here is an order the engine can be made to use.

#include "hypergraph/index.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/rule_analysis.hpp"
#include "hypergraph/pattern_matcher.hpp"
#include "hypergraph/types.hpp"
#include "hgcommon/join_core.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace HG_NAMESPACE;
using namespace HG_NAMESPACE::engine;

namespace {

struct ProbeEdge {
    VertexId vertices[MAX_ARITY];
    uint8_t  arity;
};

// A state: the edges, the two indices built over them, and the membership bitset the join
// filters against. Built with the engine's own PatternMatchingIndex, so bucket contents are
// exactly what the engine would see.
struct ProbeState {
    std::vector<ProbeEdge>       edges;
    ConcurrentHeterogeneousArena arena;
    PatternMatchingIndex         index;
    SparseBitset                 edge_set;

    void build() {
        for (EdgeId e = 0; e < edges.size(); ++e) {
            index.add_edge(e, edges[e].vertices, edges[e].arity, arena);
            edge_set.set(static_cast<EdgeId>(e), arena);
        }
    }
};

// Counting wrapper. Every accessor forwards; for_each_candidate forwards and counts. The
// counter is incremented once per candidate the enumerator yields, which is the number of
// edges the order causes the join to look at.
template <typename Inner>
struct CountingContext {
    using JoinState = typename Inner::JoinState;
    using Candidate = typename Inner::Candidate;

    const Inner* in;
    uint64_t*    n;

    uint8_t        num_lhs_edges()          const { return in->num_lhs_edges(); }
    uint8_t        order_at(uint8_t k)      const { return in->order_at(k); }
    const uint8_t* pattern_vars(uint8_t p)  const { return in->pattern_vars(p); }
    uint8_t        pattern_arity(uint8_t p) const { return in->pattern_arity(p); }
    Candidate      candidate_of(EdgeId e)   const { return in->candidate_of(e); }
    EdgeId         candidate_id(const Candidate& c)   const { return in->candidate_id(c); }
    const VertexId* edge_vertices(const Candidate& c) const { return in->edge_vertices(c); }
    uint8_t         edge_arity(const Candidate& c)    const { return in->edge_arity(c); }
    bool            usable(EdgeId e)        const { return in->usable(e); }
    bool            aborted()               const { return in->aborted(); }

    template <typename F>
    void for_each_candidate(uint8_t p, const JoinState& st, F&& f) const {
        in->for_each_candidate(p, st, [&](const auto& c) { ++*n; f(c); });
    }
};

struct RunResult {
    uint64_t candidates = 0;
    uint64_t matches    = 0;
    // The matches themselves, as (edge id at pattern position 0, at position 1, ...). Filled
    // only when `collect` is set: this is what a disagreement between two orders is diffed on.
    std::vector<std::array<EdgeId, MAX_PATTERN_EDGES>> set;
    std::vector<std::array<EdgeId, MAX_PATTERN_EDGES>> distinct;
};

// Run the real join over `state` for `rule` as its match_order currently stands.
RunResult run_order(const RewriteRule& rule, ProbeState& state, bool collect = false) {
    auto get_edge = [&state](EdgeId e) -> const ProbeEdge& { return state.edges[e]; };
    auto get_sig  = [&state](EdgeId e) {
        return EdgeSignature::from_edge(state.edges[e].vertices, state.edges[e].arity);
    };

    RunResult r;
    PatternMatchingContext<decltype(get_edge), decltype(get_sig)> mc(
        &rule, /*rule_index=*/0, /*state_id=*/0, &state.edge_set,
        &state.index.signature_index(), &state.index.inverted_index(),
        get_edge, get_sig,
        [](uint16_t, const EdgeId*, uint8_t, const VariableBinding&, StateId) {});

    HostJoinContext<decltype(get_edge), decltype(get_sig)> host{&mc};
    CountingContext<decltype(host)> ctx{&host, &r.candidates};

    typename decltype(host)::JoinState st;
    st.reset();
    hgcommon::join_dfs(ctx, st, [&](const auto& s) {
        ++r.matches;
        if (collect) {
            std::array<EdgeId, MAX_PATTERN_EDGES> m;
            m.fill(EdgeId(~0u));
            for (uint8_t d = 0; d < s.depth; ++d) m[s.pattern[d]] = s.matched[d];
            r.set.push_back(m);
        }
    });
    return r;
}

// The same count taken through the engine's own public entry point, so a disagreement between
// two orders cannot be blamed on this file's driver. find_matches builds the context and calls
// scan_pattern itself; nothing here is involved except the rule whose match_order was set.
uint64_t run_via_public(const RewriteRule& rule, ProbeState& state) {
    auto get_edge = [&state](EdgeId e) -> const ProbeEdge& { return state.edges[e]; };
    auto get_sig  = [&state](EdgeId e) {
        return EdgeSignature::from_edge(state.edges[e].vertices, state.edges[e].arity);
    };
    uint64_t n = 0;
    find_matches(rule, /*rule_index=*/0, /*state_id=*/0, state.edge_set,
                 state.index.signature_index(), state.index.inverted_index(),
                 get_edge, get_sig,
                 [&](uint16_t, const EdgeId*, uint8_t, const VariableBinding&, StateId) { ++n; });
    return n;
}

// Every permutation of [0, k) in which each position after the first shares a variable with the
// prefix. Disconnected orders are excluded rather than measured: a disconnected step is a
// cartesian product, which is an asymptotic loss and not a candidate for the comparison.
std::vector<std::vector<uint8_t>> connected_orders(const RewriteRule& rule) {
    const uint8_t k = rule.num_lhs_edges;
    std::vector<uint8_t> perm(k);
    std::iota(perm.begin(), perm.end(), uint8_t{0});
    std::vector<std::vector<uint8_t>> out;
    do {
        uint32_t bound = rule.lhs[perm[0]].var_mask();
        bool ok = true;
        for (uint8_t i = 1; i < k && ok; ++i) {
            if ((rule.lhs[perm[i]].var_mask() & bound) == 0) ok = false;
            bound |= rule.lhs[perm[i]].var_mask();
        }
        if (ok) out.push_back(perm);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return out;
}

// A structured state of the shape rewriting actually produces: a chain of binary edges with a
// controlled amount of branching, plus optional ternary edges tying triples together.
void make_state(ProbeState& s, uint32_t n_edges, uint32_t n_vertices, uint32_t seed,
                bool ternary) {
    std::mt19937_64 rng(seed);
    for (uint32_t i = 0; i < n_edges; ++i) {
        ProbeEdge e{};
        e.arity = ternary && (i % 3 == 0) ? 3 : 2;
        for (uint8_t a = 0; a < e.arity; ++a)
            e.vertices[a] = static_cast<VertexId>(rng() % n_vertices);
        s.edges.push_back(e);
    }
    s.build();
}

struct RuleSpec {
    const char*                                 name;
    std::vector<std::vector<uint8_t>>           lhs;
};

const std::vector<RuleSpec>& rule_corpus() {
    static const std::vector<RuleSpec> rules = {
        {"path2",     {{0, 1}, {1, 2}}},
        {"path3",     {{0, 1}, {1, 2}, {2, 3}}},
        {"triangle",  {{0, 1}, {1, 2}, {2, 0}}},
        {"star3",     {{0, 1}, {0, 2}, {0, 3}}},
        {"fork",      {{0, 1}, {1, 2}, {1, 3}}},
        {"selfloop",  {{0, 0}, {0, 1}}},
        {"ternary",   {{0, 1, 2}, {2, 3}}},
        {"tri_tern",  {{0, 1, 2}, {2, 3}, {3, 0}}},
    };
    return rules;
}

// DISCONNECTED LEFT-HAND SIDES, which the corpus above deliberately excludes because no
// connected order exists for them. They are measured differently: not one order against another,
// but the shipped join against what the SAME join costs when each component is enumerated once
// and the results composed.
//
// The sizes matter and are the point of the set. A component of ONE edge is enumerated by a
// single scan whose every candidate survives, so the product is the output and there is nothing
// to hoist; a component of TWO OR MORE edges has a join of its own, and the shipped schedule
// re-runs that join once per partial match of the components before it.
const std::vector<RuleSpec>& disc_corpus() {
    static const std::vector<RuleSpec> rules = {
        {"1+1",       {{0, 1}, {2, 3}}},
        {"1+1+1",     {{0, 1}, {2, 3}, {4, 5}}},
        {"2+1",       {{0, 1}, {1, 2}, {3, 4}}},
        {"2+2",       {{0, 1}, {1, 2}, {3, 4}, {4, 5}}},
        {"3+1",       {{0, 1}, {1, 2}, {2, 3}, {4, 5}}},
        {"tri+1",     {{0, 1}, {1, 2}, {2, 0}, {3, 4}}},
    };
    return rules;
}

// The candidate count of one component, taken by running the REAL join over a rule holding only
// that component's edges. Its match count comes back too: the product of the per-component match
// counts is how many times the composition would fire, which is the term no factoring removes.
struct CompCost {
    uint64_t candidates = 0;
    uint64_t matches    = 0;
};

CompCost component_cost(const RuleSpec& spec, const uint8_t* comp, uint8_t id,
                        ProbeState& state) {
    RuleBuilder b(0);
    bool any = false;
    for (uint8_t e = 0; e < spec.lhs.size(); ++e)
        if (comp[e] == id) { b.lhs(spec.lhs[e]); any = true; }
    if (!any) return {};
    b.rhs(spec.lhs.front());
    RewriteRule sub = b.build();          // build() re-runs compute_match_order for the subset
    const RunResult r = run_order(sub, state);
    return {r.candidates, r.matches};
}

}  // namespace

int main(int argc, char** argv) {
    const uint32_t n_edges    = argc > 1 ? std::stoul(argv[1]) : 400;
    const uint32_t n_vertices = argc > 2 ? std::stoul(argv[2]) : 120;
    const uint32_t n_seeds    = argc > 3 ? std::stoul(argv[3]) : 5;

    if (argc > 4 && std::string(argv[4]) == "disc") {
        // WHAT FACTORING A DISCONNECTED LEFT-HAND SIDE IS WORTH, in candidates rather than in
        // seconds. `shipped` is what the join examines today. `factored` is the same join run
        // once per component and its results composed: the components share no variable, so
        // their match sets are independent and their product IS the match set, minus the pairs
        // that would take the same data edge twice.
        std::printf("disconnected LHS: candidates examined   edges=%u vertices=%u seeds=%u\n",
                    n_edges, n_vertices, n_seeds);
        std::printf("%-8s %5s %10s %12s %12s %12s %7s  %s\n", "rule", "comps", "matches",
                    "shipped", "enum", "enum+prod", "ratio", "per-component matches");
        for (const RuleSpec& spec : disc_corpus()) {
            RuleBuilder b(0);
            for (const auto& e : spec.lhs) b.lhs(e);
            b.rhs(spec.lhs.front());
            RewriteRule rule = b.build();

            uint8_t comp[MAX_PATTERN_EDGES];
            const uint8_t ncomp = lhs_components(rule, comp);

            uint64_t shipped = 0, factored = 0, matches = 0, product = 0;
            std::vector<uint64_t> per_comp(ncomp, 0);
            for (uint32_t seed = 0; seed < n_seeds; ++seed) {
                ProbeState state;
                make_state(state, n_edges, n_vertices, seed + 1,
                           /*ternary=*/spec.lhs.front().size() == 3);
                const RunResult full = run_order(rule, state);
                shipped += full.candidates;
                matches += full.matches;

                uint64_t prod = 1;
                for (uint8_t c = 0; c < ncomp; ++c) {
                    const CompCost cc = component_cost(spec, comp, c, state);
                    factored += cc.candidates;
                    prod *= cc.matches;
                    if (seed == 0) per_comp[c] = cc.matches;
                }
                product += prod;
            }
            // THE COMPARISON THAT DECIDES IT IS shipped AGAINST enum+prod, NOT AGAINST enum.
            // Factoring hoists each component's enumeration out of the loop, and `enum` is what
            // that costs; but the composition still has to walk the product, because every
            // element of it is a match the caller is owed. So the factored plan pays
            // enum + product, and the product is the OUTPUT -- a term no plan can be under.
            // Reporting `enum` alone would credit factoring with removing the answer.
            //
            // The product counts pairs that take the same data edge twice; the join rejects
            // those, so it is an upper bound on the match set and the gap is the collisions.
            const uint64_t total = factored + product;
            std::printf("%-8s %5u %10llu %12llu %12llu %12llu %6.2fx  ",
                        spec.name, ncomp, (unsigned long long)matches,
                        (unsigned long long)shipped, (unsigned long long)factored,
                        (unsigned long long)total,
                        total ? double(shipped) / double(total) : 0.0);
            for (uint8_t c = 0; c < ncomp; ++c)
                std::printf("%s%llu", c ? " x " : "", (unsigned long long)per_comp[c]);
            std::printf("   product=%llu\n", (unsigned long long)product);
        }
        return 0;
    }

    if (argc > 4 && std::string(argv[4]) == "diff") {
        // MINIMISATION MODE. One rule, one small state, every connected order: print the
        // matches one order finds and another does not. A join order decides how the set is
        // found, never what it is, so any output here is a defect on one of the two sides.
        for (const RuleSpec& spec : rule_corpus()) {
            RuleBuilder b(0);
            for (const auto& e : spec.lhs) b.lhs(e);
            b.rhs(spec.lhs.front());
            RewriteRule rule = b.build();
            const auto orders = connected_orders(rule);
            if (orders.size() < 2) continue;

            ProbeState state;
            make_state(state, n_edges, n_vertices, 1, spec.lhs.front().size() == 3);

            std::vector<RunResult> res;
            for (const auto& ord : orders) {
                RewriteRule r = rule;
                for (uint8_t i = 0; i < r.num_lhs_edges; ++i) r.match_order[i] = ord[i];
                RunResult rr = run_order(r, state, /*collect=*/true);
                std::sort(rr.set.begin(), rr.set.end());
                // DISTINCT matches, kept beside the raw emission count. A join order cannot
                // change the SET; it can change how many times a member of it is emitted, and
                // those are different defects with different causes.
                rr.distinct = rr.set;
                rr.distinct.erase(std::unique(rr.distinct.begin(), rr.distinct.end()),
                                  rr.distinct.end());
                res.push_back(std::move(rr));
            }
            for (size_t i = 1; i < res.size(); ++i) {
                if (res[i].matches == res[0].matches) continue;
                std::printf("rule %s: emitted %llu vs %llu, DISTINCT %zu vs %zu\n",
                            spec.name, (unsigned long long)res[0].matches,
                            (unsigned long long)res[i].matches,
                            res[0].distinct.size(), res[i].distinct.size());
                RewriteRule r0 = rule, ri = rule;
                for (uint8_t q = 0; q < rule.num_lhs_edges; ++q) {
                    r0.match_order[q] = orders[0][q];
                    ri.match_order[q] = orders[i][q];
                }
                std::printf("rule %s: order[0] finds %llu, order[%zu] finds %llu"
                            "   (via find_matches: %llu vs %llu)\n",
                            spec.name, (unsigned long long)res[0].matches, i,
                            (unsigned long long)res[i].matches,
                            (unsigned long long)run_via_public(r0, state),
                            (unsigned long long)run_via_public(ri, state));
                std::vector<std::array<EdgeId, MAX_PATTERN_EDGES>> only0, onlyi;
                std::set_difference(res[0].set.begin(), res[0].set.end(),
                                    res[i].set.begin(), res[i].set.end(),
                                    std::back_inserter(only0));
                std::set_difference(res[i].set.begin(), res[i].set.end(),
                                    res[0].set.begin(), res[0].set.end(),
                                    std::back_inserter(onlyi));
                auto dump = [&](const char* tag, const auto& v) {
                    for (size_t k = 0; k < v.size() && k < 8; ++k) {
                        std::printf("  %s:", tag);
                        for (uint8_t p = 0; p < rule.num_lhs_edges; ++p) {
                            const EdgeId id = v[k][p];
                            std::printf(" p%u=e%u(", p, id);
                            for (uint8_t a = 0; a < state.edges[id].arity; ++a)
                                std::printf("%s%u", a ? "," : "", state.edges[id].vertices[a]);
                            std::printf(")");
                        }
                        std::printf("\n");
                    }
                    if (v.size() > 8) std::printf("  %s: ... %zu more\n", tag, v.size() - 8);
                };
                dump("only in order0", only0);
                dump("only in orderN", onlyi);
                return 3;
            }
        }
        std::printf("every connected order agrees on the match set.\n");
        return 0;
    }

    std::printf("join order by CANDIDATE COUNT   edges=%u vertices=%u seeds=%u\n",
                n_edges, n_vertices, n_seeds);
    std::printf("%-10s %7s %7s %12s %12s %8s  %s\n",
                "rule", "orders", "matches", "shipped", "best", "ratio", "verdict");

    uint32_t worse = 0, total = 0;

    for (const RuleSpec& spec : rule_corpus()) {
        RuleBuilder b(0);
        for (const auto& e : spec.lhs) b.lhs(e);
        b.rhs(spec.lhs.front());          // the RHS does not affect matching; one edge suffices
        RewriteRule rule = b.build();     // build() runs compute_match_order

        std::vector<uint8_t> shipped(rule.match_order,
                                     rule.match_order + rule.num_lhs_edges);
        const auto orders = connected_orders(rule);

        uint64_t shipped_total = 0, best_total = 0, matches_total = 0;
        std::vector<uint8_t> best_order = shipped;

        for (uint32_t seed = 0; seed < n_seeds; ++seed) {
            ProbeState state;
            make_state(state, n_edges, n_vertices, seed + 1,
                       /*ternary=*/spec.lhs.front().size() == 3);

            uint64_t best_here = UINT64_MAX;
            uint64_t match_ref  = UINT64_MAX;
            std::vector<uint8_t> best_here_order;
            for (const auto& ord : orders) {
                RewriteRule r = rule;
                for (uint8_t i = 0; i < r.num_lhs_edges; ++i) r.match_order[i] = ord[i];
                const RunResult res = run_order(r, state);

                // GROUND TRUTH. The join order decides how the match set is FOUND, never what it
                // is: every connected order must emit the same number of matches on the same
                // state. A disagreement means the counting driver is not running the join the
                // engine runs, and every candidate count it reports is worthless.
                if (match_ref == UINT64_MAX) match_ref = res.matches;
                else if (res.matches != match_ref) {
                    std::printf("INSTRUMENT BROKEN: rule %s seed %u: order yields %llu matches, "
                                "another yields %llu. Counts not reported.\n",
                                spec.name, seed, (unsigned long long)res.matches,
                                (unsigned long long)match_ref);
                    return 2;
                }

                if (ord == shipped) { shipped_total += res.candidates; matches_total += res.matches; }
                if (res.candidates < best_here) { best_here = res.candidates; best_here_order = ord; }
            }
            best_total += best_here;
            best_order = best_here_order;
        }

        const double ratio = best_total ? double(shipped_total) / double(best_total) : 1.0;
        ++total;
        const bool is_worse = ratio > 1.01;
        if (is_worse) ++worse;

        std::string verdict = is_worse ? "shipped order loses by " + std::to_string(ratio) + "x"
                                       : "shipped order is the counting optimum";
        std::printf("%-10s %7zu %7llu %12llu %12llu %8.3f  %s\n",
                    spec.name, orders.size(),
                    (unsigned long long)matches_total,
                    (unsigned long long)shipped_total,
                    (unsigned long long)best_total, ratio, verdict.c_str());
        std::fflush(stdout);
    }

    std::printf("\n%u of %u rules: the shipped order is NOT the counting optimum.\n", worse, total);
    return worse == 0 ? 0 : 1;
}
