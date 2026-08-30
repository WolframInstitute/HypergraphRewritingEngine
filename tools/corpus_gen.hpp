#pragma once
#include "hgcommon/core.hpp"

#include <cstdint>
#include <string>
#include <vector>

// A SYSTEMATIC rule/initial-state family, shared by bench_cpu_evolve and bench_gpu_evolve.
//
// WHY THIS EXISTS. Every performance claim about the two engines rested on eight hand-picked
// shapes at depth 5-7: one rule each but for one, and one deliberately automorphic initial state.
// That is a smoke test. The properties that actually decide cost -- automorphism group size,
// connectivity, arity mix, how many rules interact -- were never varied on purpose, and
// canonicalization, which those properties drive, is 51-77% of device cycles. A claim about rule
// space needs a corpus drawn FROM that space rather than from whichever shapes were convenient.
//
// ONE GENERATOR, TWO CALLERS. Both benches include this header, so a CPU row and a GPU row name
// the same workload by construction. A second copy would drift on the first edit, and comparing
// two engines on two different definitions of "the same rule" measures the definitions.
//
// The family is indexed rather than random: workload k is the same workload on every machine, in
// every run, in both engines, so a result is quotable and a regression is locatable. There is no
// seed to lose.

namespace corpus {

// A rule is edges of vertex-tuples on each side. Vertices are small integers; the engine treats
// them as pattern variables on the left and as variables-or-fresh on the right.
struct Rule {
    std::vector<std::vector<uint32_t>> lhs, rhs;
};

struct Workload {
    std::string name;
    std::vector<Rule> rules;
    std::vector<std::vector<uint32_t>> init;
};

// The axes. Each is a property that changes what the matcher or the canonicalizer must do, and
// they are crossed rather than sampled so coverage is stated rather than hoped for.
enum class Shape : uint32_t {
    Path = 0,     // acyclic chain: the ordinary case, join order is easy
    Cycle,        // cyclic: no acyclic join order exists, worst case for a binary join plan
    Star,         // one hub: high automorphism among the leaves, makes IR descend
    Disconnected, // two components sharing no variable: cartesian product in the matcher
    Repeated,     // a variable repeated within one edge ({x,x}): the seed edge cannot be matched
                  // by scanning distinct positions, and the matcher takes a separate branch for it
    Count
};

inline const char* shape_name(Shape s) {
    switch (s) {
        case Shape::Path:         return "path";
        case Shape::Cycle:        return "cycle";
        case Shape::Star:         return "star";
        case Shape::Disconnected: return "disc";
        case Shape::Repeated:     return "rep";
        default:                  return "?";
    }
}

// The arity axis. A number is that width for every edge; ARITY_MIXED gives edge `i` width
// 2 + (i % 3), so one rule carries edges of three different widths.
//
// MIXED IS ITS OWN CASE, not a convenience. Both the matcher and the canonicalizer key on edge
// width -- the candidate scan indexes by arity and the refinement signature counts incidences per
// width -- so a corpus where every edge in a rule is the same width exercises neither on the input
// that separates them.
constexpr uint32_t ARITY_MIXED = 0;

inline uint32_t edge_arity(uint32_t ar, uint32_t i) {
    return ar == ARITY_MIXED ? 2 + (i % 3) : ar;
}

inline std::string arity_name(uint32_t ar) {
    return ar == ARITY_MIXED ? std::string("m") : std::to_string(ar);
}

// Left-hand side of `n` edges in the given shape, over edges of width `edge_arity(ar, i)`.
inline std::vector<std::vector<uint32_t>> make_lhs(Shape s, uint32_t n, uint32_t ar) {
    std::vector<std::vector<uint32_t>> e;
    if (n == 0) return e;
    switch (s) {
        case Shape::Path:
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<uint32_t> t;
                const uint32_t w = edge_arity(ar, i);
                for (uint32_t a = 0; a < w; ++a) t.push_back(i + a);
                e.push_back(std::move(t));
            }
            break;
        case Shape::Cycle:
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<uint32_t> t;
                const uint32_t w = edge_arity(ar, i);
                for (uint32_t a = 0; a < w; ++a) t.push_back((i + a) % n);
                e.push_back(std::move(t));
            }
            break;
        case Shape::Star: {
            // Every edge touches vertex 0. The leaves are interchangeable, which is precisely the
            // automorphism that makes individualization-refinement branch. A running leaf counter
            // keeps the leaves disjoint when the edges differ in width.
            uint32_t leaf = 1;
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<uint32_t> t{0};
                const uint32_t w = edge_arity(ar, i);
                for (uint32_t a = 1; a < w; ++a) t.push_back(leaf++);
                e.push_back(std::move(t));
            }
            break;
        }
        case Shape::Disconnected:
            // n SINGLETON components: `i * ar` already gives every edge its own variables, so
            // disc-lNaK is N components of one edge each and the bases below only space the
            // numbering. The matcher cannot join them and must take a product, which is
            // quadratic per extra component -- kept small deliberately.
            //
            // A component of ONE edge is enumerated by a single scan whose candidates all
            // survive, so the product IS the match set and the shipped join is already within
            // 0.25% of it (tools/join_order_counts.cpp, `disc` mode). The shape with slack is a
            // component of two or more edges, whose internal join can fail; that shape is
            // measured there rather than generated here, because putting it in this corpus
            // would move every disc-* row in the paper to buy a 1.29x on candidates.
            {
                uint32_t next = 0;
                for (uint32_t i = 0; i < n; ++i) {
                    std::vector<uint32_t> t;
                    const uint32_t base = (i < (n + 1) / 2) ? 0u : 100u;
                    const uint32_t w = edge_arity(ar, i);
                    for (uint32_t a = 0; a < w; ++a) t.push_back(base + next + a);
                    next += w;
                    e.push_back(std::move(t));
                }
            }
            break;
        case Shape::Repeated:
            // Positions 0 and 1 are the SAME variable, so every left edge constrains a vertex
            // against itself. Width 3 and above chains the trailing position onto the next edge;
            // at width 2 the edge is a self-loop and shares nothing, which is the shape.
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<uint32_t> t{i, i};
                const uint32_t w = edge_arity(ar, i);
                for (uint32_t a = 2; a < w; ++a) t.push_back(i + a - 1);
                e.push_back(std::move(t));
            }
            break;
        default: break;
    }
    return e;
}

// Right-hand side: keep EVERY left edge, then add `grow` FRESH COPIES OF THE LEFT SIDE, each on
// its own fresh vertices and joined to a bound vertex by one link edge.
//
// GROWTH HAS TO CREATE THE SHAPE THE LEFT SIDE MATCHES, or the workload does not branch, and a
// workload that does not branch measures the engine's per-call floor rather than its concurrency.
// A pendant edge onto a fresh vertex fails that on both counts: it adds a TREE, so a left side
// wanting a cycle or a repeated vertex finds nothing new to match at all; and where it does still
// match -- a star hub, a path end -- every child is isomorphic to every other, canonical dedup
// collapses them to one, and the state count rises by exactly one per step. Copying the whole
// left side instead gives `grow` new match sites per application for EVERY shape, which is the
// branching condition, and it is one rule rather than one per shape.
//
// Keeping every left edge is the other half. Dropping any of them makes a rule with two or three
// left edges net-destructive, and a net-destructive rule cannot grow whatever the right side adds.
//
// The link edge is what keeps the state one component. Without it each application would strew
// `grow` unconnected copies, which turns every shape into the disconnected one and collapses that
// axis into the others.
inline std::vector<std::vector<uint32_t>> make_rhs(
        const std::vector<std::vector<uint32_t>>& lhs, uint32_t grow, uint32_t ar) {
    std::vector<std::vector<uint32_t>> e = lhs;      // non-destructive: the left side survives
    if (lhs.empty()) return e;

    // The left side's distinct vertices in first-seen order. A copy renames them as a block, which
    // preserves WHICH POSITIONS SHARE A VARIABLE -- for the repeated shape that sharing is the
    // shape, and renaming per edge would destroy it.
    std::vector<uint32_t> src;
    for (const auto& edge : lhs)
        for (uint32_t v : edge) {
            bool have = false;
            for (uint32_t u : src) if (u == v) { have = true; break; }
            if (!have) src.push_back(v);
        }

    auto src_index = [&](uint32_t v) {
        for (size_t i = 0; i < src.size(); ++i) if (src[i] == v) return i;
        return src.size();
    };

    // A copy costs one fresh variable per left-side vertex, and the engine binds variables into a
    // MAX_VARS-entry array, so copies are added EDGE BY EDGE while they fit. The bound binds only
    // where the left side is disconnected at the widest arity -- disc-l3a4 wants 36 -- and a
    // partial copy is still a new match site there precisely because the components are
    // independent: a disconnected left side matches any combination of them, so adding edges to
    // one component adds combinations. Every connected left side in the corpus copies whole.
    size_t vars = src.size();
    uint32_t fresh = 200;
    for (uint32_t g = 0; g < grow; ++g) {
        std::vector<uint32_t> mapped(src.size(), 0u);
        std::vector<bool> have(src.size(), false);
        std::vector<std::vector<uint32_t>> copy;
        for (const auto& edge : lhs) {
            std::vector<size_t> wanted;
            for (uint32_t v : edge) {
                const size_t i = src_index(v);
                if (have[i]) continue;
                bool dup = false;
                for (size_t j : wanted) if (j == i) { dup = true; break; }
                if (!dup) wanted.push_back(i);
            }
            if (vars + wanted.size() > hgcommon::MAX_VARS) break;
            for (size_t i : wanted) { mapped[i] = fresh++; have[i] = true; ++vars; }
            std::vector<uint32_t> t;
            for (uint32_t v : edge) t.push_back(mapped[src_index(v)]);
            copy.push_back(std::move(t));
        }
        if (copy.empty()) break;

        // Anchor each copy on a DIFFERENT bound vertex, so growth spreads over the match rather
        // than piling onto one vertex. The link edge carries the workload's own arity and fills
        // its trailing positions from the copy, so an arity-4 workload gains no binary edge and
        // the link costs no variable.
        const auto& anchor = lhs[g % lhs.size()];
        std::vector<uint32_t> link{anchor[g % anchor.size()], copy.front().front()};
        for (uint32_t a = 2; a < edge_arity(ar, g); ++a)
            link.push_back(copy[(a - 1) % copy.size()].front());
        e.push_back(std::move(link));
        for (auto& t : copy) e.push_back(std::move(t));
    }
    return e;
}

// Renumber a rule's variables to a dense 0..n-1 range, preserving identity across both sides.
// The shape builders above use spaced bases (100, 200) so the two components of a disconnected
// left side and the fresh right-side vertices cannot collide; those bases are convenient to
// generate and exceed the engine's MAX_VARS, so they are compacted here rather than there.
inline void compact_vars(Rule& r) {
    std::vector<uint32_t> seen;
    auto idx = [&](uint32_t v) -> uint32_t {
        for (uint32_t i = 0; i < seen.size(); ++i) if (seen[i] == v) return i;
        seen.push_back(v);
        return static_cast<uint32_t>(seen.size() - 1);
    };
    for (auto& e : r.lhs) for (auto& v : e) v = idx(v);   // left first, so left vars are lowest
    for (auto& e : r.rhs) for (auto& v : e) v = idx(v);
}

// Initial state for a left side of `nl` edges, varied on the axes that drive canonicalization
// rather than on size alone.
//
// One size larger than the left side is what gives the left side somewhere to match: a path
// embeds in a longer path, a star in a wider star, n singleton components in n+1, n self-loops
// in n+1. A CYCLE IS THE EXCEPTION AND HAS TO BE STATED, because an n-cycle embeds in no cycle
// but one of length exactly n -- an (n+1)-cycle contains no n-cycle. The cycle initial state is
// therefore the left side's own length, which is the smallest state it can match at all.
inline std::vector<std::vector<uint32_t>> make_init(Shape s, uint32_t nl, uint32_t ar) {
    const uint32_t n = (s == Shape::Cycle) ? nl : nl + 1;
    auto e = make_lhs(s, n, ar);
    Rule tmp; tmp.lhs = e; compact_vars(tmp);   // dense vertex ids, same renumbering rule
    return tmp.lhs;
}

// THE NAMED WORKLOADS, one table for both benches, exactly as the generated family already is:
// a CPU row and a GPU row name the same workload by construction, and a row added here appears
// in both engines' corpora at once. The single-rule rows are the hand-picked shapes the
// differential and exactness gates grew up on; the combination rows interleave rules drawn from
// them, because rule INTERACTION is a workload property none of the single rows exercises: two
// rules compete for the same edges, a rule can mint the structure another consumes, and the
// event/branchial structure couples across rules.
inline std::vector<Workload> named_workloads() {
    const Rule wpp_r      {{{0,1},{0,2}},       {{0,1},{0,3},{1,3},{2,3}}};
    const Rule binary_r   {{{0,1}},             {{0,2},{2,1}}};
    const Rule wolfram_r  {{{0,1},{1,2}},       {{0,1},{1,3},{3,2},{2,0}}};
    const Rule triangle_r {{{0,1},{1,2},{2,0}}, {{0,1},{1,2},{2,3},{3,0}}};
    const Rule arity3_r   {{{0,1,2}},           {{0,1,2},{2,3}}};
    const Rule grow_r     {{{0,1},{1,2}},       {{0,1},{1,3},{3,2}}};
    const Rule contract_r {{{0,1},{1,2}},       {{0,2}}};
    return {
        // The deep/narrow default: two-edge left side, growing right side, two-edge initial.
        {"wpp",       {wpp_r},      {{0,1},{0,2}}},
        // Single-edge left side: every edge in the state is a candidate, so the matcher floods.
        {"binary",    {binary_r},   {{0,1}}},
        // Wolfram 2->4, the shape most of the published models use.
        {"wolfram24", {wolfram_r},  {{0,1},{1,2}}},
        // Cyclic left side: three edges, no acyclic join order, worst case for the matcher.
        {"triangle",  {triangle_r}, {{0,1},{1,2},{2,0}}},
        // Mixed arity on both sides.
        {"arity3",    {arity3_r},   {{0,1,2}}},
        // Two rules over the same state: queue traffic per state doubles and the two compete.
        {"multirule", {grow_r, binary_r}, {{0,1},{1,2}}},
        // Automorphic initial state: the canonicalizer cannot stop at depth one.
        {"cycle4",    {grow_r},     {{0,1},{1,2},{2,3},{3,0}}},
        // Several roots, so the frontier starts wide instead of narrow.
        {"multiroot", {grow_r},     {{0,1},{1,2},{3,4},{4,5},{6,7},{7,8}}},
        // TWO COMPONENTS OF TWO EDGES EACH, which the generated corpus does not build: its
        // Disconnected shape numbers every edge's variables apart, so disc-lNa2 is N components
        // of ONE edge and each component's match set is "every edge of this arity". A component
        // of one edge costs one scan to enumerate, so the product is the output and the join is
        // already output-optimal on it. A component of TWO edges has a join of its own, and the
        // schedule re-runs that join once per partial match of the components before it.
        {"disc2x2",   {Rule{{{0,1},{1,2},{3,4},{4,5}},
                            {{0,1},{1,2},{3,4},{4,5},{2,6}}}},
                      {{0,1},{1,2},{3,4},{4,5}}},
        // Growth against reduction against flooding: the grower deepens, the contraction erases
        // what it grew, and the single-edge splitter matches everything either produces.
        {"growshrink3", {wpp_r, contract_r, binary_r}, {{0,1},{0,2}}},
        // The 2->4 rule mints 3-cycles; the triangle rule consumes exactly those. The second
        // rule's match set exists only through the first rule's output.
        {"wolftri",   {wolfram_r, triangle_r}, {{0,1},{1,2}}},
        // Mixed arity across rules rather than within one: the arity-3 rewriter and the
        // arity-2 splitter each see only their own edges of a state holding both.
        {"arimix",    {arity3_r, binary_r}, {{0,1,2},{0,1}}},
        // Four rules of four characters over a state holding every arity they need.
        {"allfour",   {wpp_r, wolfram_r, binary_r, arity3_r}, {{0,1},{0,2},{3,4,5}}},
    };
}

// The corpus. Crossed over shape x lhs size x arity x growth x rule count, which is coverage that
// can be stated: every combination below is present exactly once.
inline std::vector<Workload> corpus() {
    std::vector<Workload> out;
    for (uint32_t sh = 0; sh < static_cast<uint32_t>(Shape::Count); ++sh) {
        const Shape s = static_cast<Shape>(sh);
        for (uint32_t nl : {1u, 2u, 3u}) {
            for (uint32_t ar : {2u, 3u, 4u, ARITY_MIXED}) {
                for (uint32_t grow : {1u, 2u}) {
                    for (uint32_t nrules : {1u, 2u}) {
                        Workload w;
                        w.name = std::string(shape_name(s)) + "-l" + std::to_string(nl) +
                                 "a" + arity_name(ar) + "g" + std::to_string(grow) +
                                 "r" + std::to_string(nrules);
                        for (uint32_t r = 0; r < nrules; ++r) {
                            Rule rule;
                            // The second rule of a pair is deliberately a DIFFERENT shape, so the
                            // set exercises rules that interact rather than two of a kind.
                            const Shape rs = (r == 0) ? s
                                           : static_cast<Shape>((sh + 1) %
                                                 static_cast<uint32_t>(Shape::Count));
                            rule.lhs = make_lhs(rs, nl, ar);
                            rule.rhs = make_rhs(rule.lhs, grow, ar);
                            compact_vars(rule);
                            if (!rule.lhs.empty()) w.rules.push_back(std::move(rule));
                        }
                        w.init = make_init(s, nl, ar);
                        if (!w.rules.empty() && !w.init.empty()) out.push_back(std::move(w));
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace corpus
