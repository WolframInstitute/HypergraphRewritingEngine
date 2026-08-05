// Gate for hgcommon/ir_core.hpp: the shared flat core must produce the SAME canonical hash as
// the host IRCanonicalizer, on every state, bit for bit -- and must be isomorphism-invariant.
//
// Two properties, because either alone is insufficient. Equality alone would be satisfied by
// two implementations that are wrong in the same way; invariance alone would be satisfied by a
// correct implementation that silently changed every stored hash. Both together pin the new
// core to the old one AND to the definition.
//
// The corpus deliberately over-weights the cases the search actually exercises: cycles and
// regular graphs (large automorphism groups, so the individualization search runs deep),
// duplicate edges, repeated vertices within an edge, and mixed arity.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#include "hypergraph/ir_canonicalization.hpp"
#include "hgcommon/ir_core.hpp"

using hypergraph::VertexId;
using Edges = std::vector<std::vector<VertexId>>;

// Flatten to the core's convention: local indices are the rank of each vertex in the
// sorted-unique vertex set, which is exactly how the host canonicalizer indexes them.
struct Flat {
    std::vector<uint8_t> ea;
    std::vector<uint32_t> eoff, ev;
    uint32_t n_verts = 0, total_occ = 0;
};

static Flat flatten(const Edges& edges) {
    Flat f;
    std::vector<VertexId> verts;
    for (const auto& e : edges) for (VertexId v : e) verts.push_back(v);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    f.n_verts = static_cast<uint32_t>(verts.size());
    for (const auto& e : edges) {
        f.eoff.push_back(static_cast<uint32_t>(f.ev.size()));
        f.ea.push_back(static_cast<uint8_t>(e.size()));
        for (VertexId v : e) {
            const uint32_t idx = static_cast<uint32_t>(
                std::lower_bound(verts.begin(), verts.end(), v) - verts.begin());
            f.ev.push_back(idx);
        }
    }
    f.total_occ = static_cast<uint32_t>(f.ev.size());
    return f;
}

// uint64 buffer so the core's 8-byte views are aligned.
static std::vector<uint64_t> g_scratch;

static uint64_t core_hash(const Edges& edges, uint32_t* status_out) {
    if (edges.empty()) { *status_out = hgcommon::IR_EMPTY; return 0; }
    Flat f = flatten(edges);
    const uint32_t depth = hgcommon::IR_MAX_DEPTH_DEFAULT;
    const uint64_t words = hgcommon::ir_scratch_words(
        f.n_verts, static_cast<uint32_t>(f.ea.size()), f.total_occ, depth);
    if (g_scratch.size() < (words + 1) / 2 + 8) g_scratch.assign((words + 1) / 2 + 8, 0);
    auto r = hgcommon::ir_canonical_hash(
        f.ea.data(), f.eoff.data(), f.ev.data(),
        static_cast<uint32_t>(f.ea.size()), f.n_verts, f.total_occ,
        reinterpret_cast<uint32_t*>(g_scratch.data()), depth);
    *status_out = r.status;
    return r.hash;
}

// The core's CANONICAL FORM: arity-prefixed edges in canonical order, under the winning
// labelling. Empty when the core reported anything but IR_OK.
static std::vector<uint32_t> core_form(const Edges& edges, uint32_t* status_out) {
    std::vector<uint32_t> form;
    if (edges.empty()) { *status_out = hgcommon::IR_EMPTY; return form; }
    Flat f = flatten(edges);
    const uint32_t n_e = static_cast<uint32_t>(f.ea.size());
    const uint32_t depth = hgcommon::IR_MAX_DEPTH_DEFAULT;
    const uint64_t words = hgcommon::ir_scratch_words(f.n_verts, n_e, f.total_occ, depth);
    if (g_scratch.size() < (words + 1) / 2 + 8) g_scratch.assign((words + 1) / 2 + 8, 0);
    form.assign(hgcommon::ir_canonical_form_words(n_e, f.total_occ), 0);
    auto r = hgcommon::ir_canonical_hash(
        f.ea.data(), f.eoff.data(), f.ev.data(), n_e, f.n_verts, f.total_occ,
        reinterpret_cast<uint32_t*>(g_scratch.data()), depth,
        nullptr, hgcommon::IR_HOST_GENERATORS, nullptr, nullptr, form.data());
    *status_out = r.status;
    if (r.status != hgcommon::IR_OK) form.clear();
    return form;
}

// The same shape from the host implementation, so the two can be compared word for word.
static std::vector<uint32_t> host_form(const Edges& edges) {
    std::vector<uint32_t> form;
    hypergraph::IRCanonicalizer ir;
    auto res = ir.canonicalize_edges(edges);
    for (const auto& e : res.canonical_form.edges) {
        form.push_back(static_cast<uint32_t>(e.size()));
        for (auto v : e) form.push_back(static_cast<uint32_t>(v));
    }
    return form;
}

static uint64_t host_hash(const Edges& edges) {
    hypergraph::IRCanonicalizer ir;
    return ir.compute_canonical_hash(edges);
}

// Relabel vertices by a random permutation and shuffle the edge order: an isomorphic copy.
static Edges permuted(const Edges& edges, std::mt19937_64& rng) {
    std::vector<VertexId> verts;
    for (const auto& e : edges) for (VertexId v : e) verts.push_back(v);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    std::vector<VertexId> images(verts.size());
    std::iota(images.begin(), images.end(), VertexId{1000});
    std::shuffle(images.begin(), images.end(), rng);
    Edges out;
    for (const auto& e : edges) {
        std::vector<VertexId> m;
        for (VertexId v : e) {
            const size_t i = std::lower_bound(verts.begin(), verts.end(), v) - verts.begin();
            m.push_back(images[i]);
        }
        out.push_back(std::move(m));
    }
    std::shuffle(out.begin(), out.end(), rng);
    return out;
}

// ---- corpus ------------------------------------------------------------------------------
static void add_cycle(std::vector<Edges>& c, uint32_t k) {
    Edges e;
    for (uint32_t i = 0; i < k; ++i) e.push_back({i, (i + 1) % k});
    c.push_back(std::move(e));
}
static void add_complete(std::vector<Edges>& c, uint32_t k) {
    Edges e;
    for (uint32_t i = 0; i < k; ++i) for (uint32_t j = i + 1; j < k; ++j) e.push_back({i, j});
    c.push_back(std::move(e));
}
static void add_star(std::vector<Edges>& c, uint32_t k) {
    Edges e;
    for (uint32_t i = 1; i <= k; ++i) e.push_back({0, i});
    c.push_back(std::move(e));
}
static void add_hypercycle(std::vector<Edges>& c, uint32_t k, uint32_t arity) {
    Edges e;
    for (uint32_t i = 0; i < k; ++i) {
        std::vector<VertexId> t;
        for (uint32_t a = 0; a < arity; ++a) t.push_back((i + a) % k);
        e.push_back(std::move(t));
    }
    c.push_back(std::move(e));
}

int main(int argc, char** argv) {
    const int random_cases = argc > 1 ? atoi(argv[1]) : 4000;

    std::vector<Edges> corpus;
    for (uint32_t k = 3; k <= 14; ++k) { add_cycle(corpus, k); add_star(corpus, k); }
    for (uint32_t k = 3; k <= 8; ++k) add_complete(corpus, k);
    for (uint32_t k = 4; k <= 12; ++k) for (uint32_t a = 2; a <= 4; ++a) add_hypercycle(corpus, k, a);
    // Repeated vertices inside an edge, duplicate edges, and self-loops.
    corpus.push_back({{1, 1}, {1, 2}, {2, 2}});
    corpus.push_back({{1, 2, 1}, {2, 1, 2}});
    corpus.push_back({{1, 2}, {1, 2}, {1, 2}});
    corpus.push_back({{5, 5, 5}});
    corpus.push_back({{1, 2, 3}, {3, 2, 1}});
    corpus.push_back({{1}, {2}, {3}, {1, 2, 3}});

    std::mt19937_64 rng(12345);
    for (int i = 0; i < random_cases; ++i) {
        const uint32_t nv = 2 + (rng() % 14);
        const uint32_t ne = 1 + (rng() % 16);
        Edges e;
        for (uint32_t j = 0; j < ne; ++j) {
            const uint32_t arity = 1 + (rng() % 4);
            std::vector<VertexId> t;
            for (uint32_t a = 0; a < arity; ++a) t.push_back(static_cast<VertexId>(rng() % nv));
            e.push_back(std::move(t));
        }
        corpus.push_back(std::move(e));
    }

    size_t mismatch = 0, noniso = 0, need_depth = 0, form_mismatch = 0;
    for (size_t i = 0; i < corpus.size(); ++i) {
        uint32_t st = 0;
        const uint64_t hc = core_hash(corpus[i], &st);
        const uint64_t hh = host_hash(corpus[i]);
        if (st == hgcommon::IR_NEED_DEPTH) { ++need_depth; continue; }
        if (hc != hh) {
            if (mismatch < 10)
                printf("  MISMATCH case %zu: core=%016llx host=%016llx  (%zu edges)\n",
                       i, (unsigned long long)hc, (unsigned long long)hh, corpus[i].size());
            ++mismatch;
        }
        // The CANONICAL FORM, not just the hash. Equal hashes would be satisfied by two
        // implementations agreeing on an identity while disagreeing about the representative
        // they name, and the FFI serializes the representative.
        {
            uint32_t stf = 0;
            const std::vector<uint32_t> fc = core_form(corpus[i], &stf);
            if (stf == hgcommon::IR_OK) {
                const std::vector<uint32_t> fh = host_form(corpus[i]);
                if (fc != fh) {
                    if (form_mismatch < 10)
                        printf("  FORM MISMATCH case %zu: %zu core words vs %zu host words\n",
                               i, fc.size(), fh.size());
                    ++form_mismatch;
                }
            }
        }

        // Isomorphism invariance of the core itself.
        for (int t = 0; t < 3; ++t) {
            uint32_t st2 = 0;
            const uint64_t hp = core_hash(permuted(corpus[i], rng), &st2);
            if (st2 == hgcommon::IR_NEED_DEPTH) continue;
            if (hp != hc) {
                if (noniso < 10)
                    printf("  NON-INVARIANT case %zu: %016llx vs permuted %016llx\n",
                           i, (unsigned long long)hc, (unsigned long long)hp);
                ++noniso;
                break;
            }
        }
    }

    printf("\n%zu states | hash mismatches vs host: %zu | FORM mismatches vs host: %zu"
           " | isomorphism violations: %zu | depth-limited: %zu\n",
           corpus.size(), mismatch, form_mismatch, noniso, need_depth);
    return (mismatch || noniso || form_mismatch) ? 1 : 0;
}
