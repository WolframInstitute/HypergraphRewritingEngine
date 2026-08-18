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
    Count
};

inline const char* shape_name(Shape s) {
    switch (s) {
        case Shape::Path:         return "path";
        case Shape::Cycle:        return "cycle";
        case Shape::Star:         return "star";
        case Shape::Disconnected: return "disc";
        default:                  return "?";
    }
}

// Left-hand side of `n` edges in the given shape, over arity-`ar` edges.
inline std::vector<std::vector<uint32_t>> make_lhs(Shape s, uint32_t n, uint32_t ar) {
    std::vector<std::vector<uint32_t>> e;
    if (n == 0) return e;
    switch (s) {
        case Shape::Path:
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<uint32_t> t;
                for (uint32_t a = 0; a < ar; ++a) t.push_back(i + a);
                e.push_back(std::move(t));
            }
            break;
        case Shape::Cycle:
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<uint32_t> t;
                for (uint32_t a = 0; a < ar; ++a) t.push_back((i + a) % n);
                e.push_back(std::move(t));
            }
            break;
        case Shape::Star:
            // Every edge touches vertex 0. The leaves are interchangeable, which is precisely the
            // automorphism that makes individualization-refinement branch.
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<uint32_t> t{0};
                for (uint32_t a = 1; a < ar; ++a) t.push_back(1 + i * (ar - 1) + (a - 1));
                e.push_back(std::move(t));
            }
            break;
        case Shape::Disconnected:
            // Two components with disjoint variables: the matcher cannot join them and must take
            // a product. Kept small deliberately -- this is quadratic per extra component.
            for (uint32_t i = 0; i < n; ++i) {
                std::vector<uint32_t> t;
                const uint32_t base = (i < (n + 1) / 2) ? 0u : 100u;
                for (uint32_t a = 0; a < ar; ++a) t.push_back(base + i * ar + a);
                e.push_back(std::move(t));
            }
            break;
        default: break;
    }
    return e;
}

// Right-hand side: keep the left's first edge (so the rule is not purely destructive) and add
// `grow` fresh edges hanging off a new vertex. `grow` is what makes the state set expand, and
// setting it below the left's size makes the rule reductive instead.
inline std::vector<std::vector<uint32_t>> make_rhs(
        const std::vector<std::vector<uint32_t>>& lhs, uint32_t grow, uint32_t ar) {
    std::vector<std::vector<uint32_t>> e;
    if (!lhs.empty()) e.push_back(lhs.front());
    uint32_t fresh = 200;
    for (uint32_t g = 0; g < grow; ++g) {
        std::vector<uint32_t> t;
        t.push_back(lhs.empty() ? 0u : lhs.front().front());
        for (uint32_t a = 1; a < ar; ++a) t.push_back(fresh++);
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

// Initial states, varied on the axes that drive canonicalization rather than on size alone.
inline std::vector<std::vector<uint32_t>> make_init(Shape s, uint32_t n, uint32_t ar) {
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
            for (uint32_t ar : {2u, 3u}) {
                for (uint32_t grow : {1u, 2u}) {
                    for (uint32_t nrules : {1u, 2u}) {
                        Workload w;
                        w.name = std::string(shape_name(s)) + "-l" + std::to_string(nl) +
                                 "a" + std::to_string(ar) + "g" + std::to_string(grow) +
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
                        w.init = make_init(s, nl + 1, ar);
                        if (!w.rules.empty() && !w.init.empty()) out.push_back(std::move(w));
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace corpus
