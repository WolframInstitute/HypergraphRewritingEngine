#pragma once
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

// Right-hand side: keep EVERY left edge, then add `grow` fresh edges that attach to vertices the
// left side already binds.
//
// Both halves matter and an earlier version got both wrong. Keeping only the FIRST left edge made
// any rule with two or three left edges net-destructive, so it could not grow: of 96 workloads
// generated that way, 84 were flat and the twelve that grew reached twenty-eight states at depth
// six. And attaching growth to fresh vertices only would extend the state without creating new
// places for the left side to match, which grows the state linearly rather than branching. The
// shape that branches -- `wpp`, the one workload in the hand-picked set that reaches thousands of
// states -- rewrites two connected edges into four that share its vertices, so each application
// creates several new match sites. That is what is reproduced here: every new edge touches an
// existing bound vertex as well as a fresh one.
inline std::vector<std::vector<uint32_t>> make_rhs(
        Shape s, const std::vector<std::vector<uint32_t>>& lhs, uint32_t grow, uint32_t ar) {
    std::vector<std::vector<uint32_t>> e = lhs;      // non-destructive: the left side survives
    if (lhs.empty()) return e;
    uint32_t fresh = 200;
    if (s == Shape::Repeated) {
        // A repeated-variable left side only matches a vertex that repeats, so growth has to
        // create that shape or the workload is flat. Each step adds a link to a fresh vertex
        // and the repeat on it, giving one new match site per application -- the branching
        // condition the rest of this file exists to satisfy.
        for (uint32_t g = 0; g < grow; ++g) {
            const auto& anchor = lhs[g % lhs.size()];
            const uint32_t v = fresh++;
            const uint32_t w = edge_arity(ar, g);
            std::vector<uint32_t> link{anchor[0], v};
            for (uint32_t a = 2; a < w; ++a) link.push_back(fresh++);
            e.push_back(std::move(link));
            std::vector<uint32_t> rep{v, v};
            for (uint32_t a = 2; a < w; ++a) rep.push_back(fresh++);
            e.push_back(std::move(rep));
        }
        return e;
    }
    if (s == Shape::Cycle) {
        // A CYCLE LEFT SIDE IS RIGID: it matches a cycle of its own length and nothing else, so
        // the pendant growth below -- which adds trees -- creates no new match site and the
        // workload fires once and stops. Each step therefore adds a WHOLE FRESH CYCLE of the left
        // side's length, joined to a bound vertex by one link edge so the state stays connected.
        // One new cycle per application is the branching condition, and it is the same argument
        // the repeated case above makes for its own shape.
        const uint32_t len = static_cast<uint32_t>(lhs.size());
        for (uint32_t g = 0; g < grow; ++g) {
            const auto& anchor = lhs[g % lhs.size()];
            std::vector<uint32_t> ring;
            for (uint32_t i = 0; i < len; ++i) ring.push_back(fresh++);
            std::vector<uint32_t> link{anchor[g % anchor.size()], ring[0]};
            for (uint32_t a = 2; a < edge_arity(ar, g); ++a) link.push_back(fresh++);
            e.push_back(std::move(link));
            for (uint32_t i = 0; i < len; ++i) {
                std::vector<uint32_t> t;
                const uint32_t w = edge_arity(ar, i);
                for (uint32_t a = 0; a < w; ++a) t.push_back(ring[(i + a) % len]);
                e.push_back(std::move(t));
            }
        }
        return e;
    }
    for (uint32_t g = 0; g < grow; ++g) {
        std::vector<uint32_t> t;
        // Anchor on a DIFFERENT bound vertex per added edge, so growth spreads over the match
        // rather than piling onto one vertex.
        const auto& anchor = lhs[g % lhs.size()];
        t.push_back(anchor[g % anchor.size()]);
        const uint32_t w = edge_arity(ar, g);
        for (uint32_t a = 1; a < w; ++a) t.push_back(fresh++);
        e.push_back(std::move(t));
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
                            rule.rhs = make_rhs(rs, rule.lhs, grow, ar);
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
