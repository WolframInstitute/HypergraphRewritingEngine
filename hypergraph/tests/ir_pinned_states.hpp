#pragma once
// The states whose canonical hash, edge ranks and edge orbits are pinned in
// test_ir_canonicalization.cpp (IRPinned). Generated deterministically: disjoint copies of a small
// piece, stars of arity 2 and 3, cycles, complete graphs and mixed copies, each with its edges in a
// shuffled order and its vertices relabelled, so the IR search individualises symmetric cells.

#include <cstdint>
#include <vector>

namespace ir_pinned {

using State = std::vector<std::vector<uint32_t>>;

// A 64-bit linear congruential generator, so the states do not depend on the standard library.
struct Lcg {
    uint64_t x;
    uint32_t next() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(x >> 33);
    }
};

inline std::vector<State> states() {
    std::vector<State> out;
    Lcg r{12345};
    for (int t = 0; t < 48; ++t) {
        State g;
        const int kind = t % 6;
        if (kind == 0) {            // copies of a random piece of up to 3 edges
            const uint32_t copies = 3 + r.next() % 20, m = 1 + r.next() % 3;
            State piece;
            for (uint32_t i = 0; i < m; ++i) {
                std::vector<uint32_t> e(1 + r.next() % 3);
                for (auto& v : e) v = r.next() % 4;
                piece.push_back(e);
            }
            for (uint32_t c = 0; c < copies; ++c)
                for (const auto& e : piece) {
                    std::vector<uint32_t> x(e);
                    for (auto& v : x) v += 4 * c;
                    g.push_back(x);
                }
        } else if (kind == 1) {     // a star, arity 2 or 3
            const uint32_t leaves = 3 + r.next() % 30, ar = 2 + r.next() % 2;
            for (uint32_t i = 0; i < leaves; ++i) {
                std::vector<uint32_t> e{0, 1 + i};
                if (ar == 3) e.push_back(1 + (i + 1) % leaves);
                g.push_back(e);
            }
        } else if (kind == 2) {     // a cycle
            const uint32_t n = 3 + r.next() % 30;
            for (uint32_t i = 0; i < n; ++i) g.push_back({i, (i + 1) % n});
        } else if (kind == 3) {     // a complete graph, one direction
            const uint32_t k = 3 + r.next() % 6;
            for (uint32_t a = 0; a < k; ++a)
                for (uint32_t b = a + 1; b < k; ++b) g.push_back({a, b});
        } else if (kind == 4) {     // copies of a path and of a triangle
            const uint32_t c1 = 2 + r.next() % 15, c2 = 2 + r.next() % 15;
            uint32_t base = 0;
            for (uint32_t c = 0; c < c1; ++c, base += 3) { g.push_back({base, base + 1}); g.push_back({base + 1, base + 2}); }
            for (uint32_t c = 0; c < c2; ++c, base += 3) {
                g.push_back({base, base + 1}); g.push_back({base + 1, base + 2}); g.push_back({base + 2, base});
            }
        } else {                    // copies of a 3-vertex hyperedge with a back edge
            const uint32_t copies = 2 + r.next() % 25;
            for (uint32_t c = 0; c < copies; ++c) {
                g.push_back({3 * c, 3 * c + 1, 3 * c + 2});
                g.push_back({3 * c + 2, 3 * c});
            }
        }
        // Shuffle the edges and relabel the vertices.
        for (size_t i = g.size(); i > 1; --i) std::swap(g[i - 1], g[r.next() % i]);
        uint32_t nv = 0;
        for (const auto& e : g)
            for (auto v : e) nv = v + 1 > nv ? v + 1 : nv;
        std::vector<uint32_t> perm(nv);
        for (uint32_t v = 0; v < nv; ++v) perm[v] = v;
        for (size_t i = nv; i > 1; --i) std::swap(perm[i - 1], perm[r.next() % i]);
        for (auto& e : g)
            for (auto& v : e) v = perm[v];
        out.push_back(g);
    }
    return out;
}

// FNV-1a over a list of values, to pin a rank or orbit vector in one number.
inline uint64_t digest(const std::vector<uint32_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t x : v)
        for (int i = 0; i < 4; ++i) { h ^= (x >> (8 * i)) & 0xFF; h *= 1099511628211ull; }
    return h;
}

}  // namespace ir_pinned
