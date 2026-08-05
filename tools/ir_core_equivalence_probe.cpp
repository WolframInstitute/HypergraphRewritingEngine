// Gate for hgcommon/ir_core.hpp, on three levels.
//
// 1. AGAINST THE DEFINITION. On every state small enough to brute-force, the labelling the core
//    emits must produce the form the core emits, and the edge orbits must equal the orbits of
//    the FULL automorphism group, enumerated by trying every permutation. This is the only
//    check that pins the algorithm rather than relating it to a sibling, and it shares no code
//    with the thing it checks. It is what catches a truncated generator table -- which agrees
//    on the hash and yields orbits that are too fine -- and it found the orbit NUMBERING
//    reading off the union-find root, which is an artifact of the union order, rather than off
//    the smallest member class, which is canonical.
//
// 2. AGAINST THE HOST ADAPTER. hypergraph::IRCanonicalizer reaches the same core through
//    ir_core_call, which fixes the vertex numbering and escalates depth and generators. The
//    hash, the canonical form and the three per-edge arrays must agree with a direct call, on
//    every state -- that is what pins the adapter's conventions.
//
// 3. INVARIANCE UNDER RELABELLING. The hash may not move when a state is relabelled and its
//    edges shuffled, and the ORBIT ARRAY may not move when a state is relabelled (edge order
//    held, so the arrays compare position by position). Both hold at every size, including
//    those the brute force cannot reach. The orbit one is the property the quotient depends on:
//    a state reached by two parents arrives under two labellings, and its per-instance
//    reconstruction only meets if the orbit id it slots on is the same one both times.
//
// The corpus deliberately over-weights the cases the search actually exercises: cycles and
// regular graphs (large automorphism groups, so the individualization search runs deep),
// duplicate edges, repeated vertices within an edge, and mixed arity.

#include <cstdio>
#include <cstdint>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#include "hypergraph/arena.hpp"
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
// labelling. Empty when the core reported anything but IR_OK. `label_out`, when non-null,
// additionally receives that labelling (local vertex index -> canonical label).
static std::vector<uint32_t> core_form(const Edges& edges, uint32_t* status_out,
                                       std::vector<uint32_t>* label_out = nullptr) {
    std::vector<uint32_t> form;
    if (edges.empty()) { *status_out = hgcommon::IR_EMPTY; return form; }
    Flat f = flatten(edges);
    const uint32_t n_e = static_cast<uint32_t>(f.ea.size());
    const uint32_t depth = hgcommon::IR_MAX_DEPTH_DEFAULT;
    const uint64_t words = hgcommon::ir_scratch_words(f.n_verts, n_e, f.total_occ, depth);
    if (g_scratch.size() < (words + 1) / 2 + 8) g_scratch.assign((words + 1) / 2 + 8, 0);
    form.assign(hgcommon::ir_canonical_form_words(n_e, f.total_occ), 0);
    if (label_out) label_out->assign(f.n_verts, 0);
    auto r = hgcommon::ir_canonical_hash(
        f.ea.data(), f.eoff.data(), f.ev.data(), n_e, f.n_verts, f.total_occ,
        reinterpret_cast<uint32_t*>(g_scratch.data()), depth,
        nullptr, hgcommon::IR_HOST_GENERATORS, nullptr, nullptr, form.data(),
        label_out ? label_out->data() : nullptr);
    *status_out = r.status;
    if (r.status != hgcommon::IR_OK) { form.clear(); if (label_out) label_out->clear(); }
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

// The three per-edge arrays the hash and form do NOT pin: canonical RANK, content CLASS and
// automorphism ORBIT. Event identity keys on rank and class; the quotient reconstruction keys
// instance identity on orbit. An implementation can agree on the canonical hash and the
// canonical form and still disagree here -- measured, when a thin adapter over the core passed
// this probe on hash and form and lost 18 tests in all_tests, one of them reporting an event
// count that depended on the worker count. That is the hole this closes.
struct Arrays {
    std::vector<uint32_t> rank, klass, orbit;
    bool ok = false;
};

static Arrays core_arrays(const Edges& edges) {
    Arrays a;
    if (edges.empty()) return a;
    Flat f = flatten(edges);
    const uint32_t n_e = static_cast<uint32_t>(f.ea.size());
    const uint32_t depth = hgcommon::IR_MAX_DEPTH_DEFAULT;
    const uint64_t words = hgcommon::ir_scratch_words(f.n_verts, n_e, f.total_occ, depth);
    if (g_scratch.size() < (words + 1) / 2 + 8) g_scratch.assign((words + 1) / 2 + 8, 0);
    a.rank.assign(n_e, 0); a.klass.assign(n_e, 0); a.orbit.assign(n_e, 0);
    auto r = hgcommon::ir_canonical_hash(
        f.ea.data(), f.eoff.data(), f.ev.data(), n_e, f.n_verts, f.total_occ,
        reinterpret_cast<uint32_t*>(g_scratch.data()), depth,
        a.rank.data(), hgcommon::IR_HOST_GENERATORS, a.orbit.data(), a.klass.data());
    a.ok = (r.status == hgcommon::IR_OK);
    return a;
}

static Arrays host_arrays(const Edges& edges) {
    Arrays a;
    if (edges.empty()) return a;
    hypergraph::IRCanonicalizer ir;
    // The rank entry point takes the scratch-allocated edge list only, so the corpus case is
    // copied into one. The other two have heap overloads.
    auto mk = hypergraph::worker_scratch().mark();
    {
        hypergraph::SVec<hypergraph::SVec<VertexId>> sv;
        for (const auto& e : edges) sv.emplace_back(e.begin(), e.end());
        ir.compute_canonical_hash_with_edge_rank(sv, a.rank);
    }
    hypergraph::worker_scratch().release(mk);
    ir.compute_canonical_hash_with_edge_map(edges, a.klass);
    ir.compute_canonical_hash_with_edge_orbits(edges, a.orbit);
    a.ok = true;
    return a;
}

static uint64_t host_hash(const Edges& edges) {
    hypergraph::IRCanonicalizer ir;
    return ir.compute_canonical_hash(edges);
}

// ---- the definition ----------------------------------------------------------------------
// Brute force over all n! vertex bijections. 7! = 5040 permutations times a sort of the edge
// list is affordable per state; 8! is not, over a corpus this size. Every state at or under
// this many vertices is checked, and the count is reported so the bound is visible.
static constexpr uint32_t kDefinitionMaxVerts = 7;

// WHAT THE CANONICAL FORM IS, AND WHAT IT IS NOT. The core keeps the smallest form among the
// leaves of its individualization-refinement tree, and refinement only reaches labellings that
// respect the structural cell order -- so the form is NOT the smallest over all n! bijections.
// On the directed 3-cycle the smallest form over all bijections is [2 0 1 | 2 1 2 | 2 2 0] and
// the core emits [2 0 2 | 2 1 0 | 2 2 1], which is the same graph relabelled the other way
// round. Both are canonical forms of the class; only one of them is minimal, and minimality is
// not what the engine needs.
//
// What it needs is that the form be a COMPLETE ISOMORPHISM INVARIANT, and that decomposes into
// two properties this checks separately:
//
//   emitted labelling produces emitted form  =>  the form is a relabelling of THIS state, so
//                                                two states with one form are isomorphic
//   relabelling the input does not move it   =>  two isomorphic states have one form
//
// The second is the isomorphism-invariance pass over the whole corpus. The first is checked
// here, and needs no enumeration -- it is one application of the labelling the core emitted.
// The enumeration is for the automorphism group, which pins the ORBITS.
struct Definition {
    std::vector<uint32_t> label_form;  // the core's own labelling, applied to the input
    std::vector<uint32_t> orbit;       // edge orbits under the FULL automorphism group
    bool checked = false;
};

// The flat form of `edges` (already in local 0..n-1 indices) under vertex relabelling `p`:
// each edge relabelled, the edge list sorted by content, then written as [arity, v...] runs.
// This is the encoding ir_core.hpp compares, so the two forms compare word for word.
static std::vector<uint32_t> form_under(const std::vector<std::vector<uint32_t>>& local,
                                        const std::vector<uint32_t>& p) {
    std::vector<std::vector<uint32_t>> mapped;
    mapped.reserve(local.size());
    for (const auto& e : local) {
        std::vector<uint32_t> m;
        m.reserve(e.size());
        for (uint32_t v : e) m.push_back(p[v]);
        mapped.push_back(std::move(m));
    }
    std::sort(mapped.begin(), mapped.end());
    std::vector<uint32_t> form;
    for (const auto& m : mapped) {
        form.push_back(static_cast<uint32_t>(m.size()));
        for (uint32_t v : m) form.push_back(v);
    }
    return form;
}

// `core_label` is the labelling the core emitted for this state (local vertex index -> canonical
// label). It is applied to the input to produce label_form, and used to number the orbits.
static Definition definition_of(const Edges& edges, const std::vector<uint32_t>& core_label) {
    Definition d;
    if (edges.empty()) return d;
    Flat f = flatten(edges);
    if (f.n_verts == 0 || f.n_verts > kDefinitionMaxVerts) return d;
    if (core_label.size() != f.n_verts) return d;
    // A labelling that is not a bijection would make every downstream comparison meaningless,
    // so it is rejected rather than used.
    {
        std::vector<uint32_t> seen = core_label;
        std::sort(seen.begin(), seen.end());
        for (uint32_t i = 0; i < f.n_verts; ++i) if (seen[i] != i) return d;
    }

    std::vector<std::vector<uint32_t>> local(edges.size());
    for (size_t ei = 0; ei < edges.size(); ++ei)
        local[ei].assign(f.ev.begin() + f.eoff[ei],
                         f.ev.begin() + f.eoff[ei] + f.ea[ei]);

    d.label_form = form_under(local, core_label);

    // A permutation is an automorphism exactly when it leaves the edge MULTISET fixed, which is
    // the identity permutation's form.
    std::vector<uint32_t> p(f.n_verts);
    std::iota(p.begin(), p.end(), 0u);
    const std::vector<uint32_t> identity_form = form_under(local, p);
    std::vector<std::vector<uint32_t>> automorphisms;
    do {
        if (form_under(local, p) == identity_form) automorphisms.push_back(p);
    } while (std::next_permutation(p.begin(), p.end()));

    // Orbits of the distinct edge tuples under the group, by union-find. Edges with equal
    // content are in one orbit by the identity, so working over distinct tuples loses nothing.
    std::vector<std::vector<uint32_t>> tuples = local;
    std::sort(tuples.begin(), tuples.end());
    tuples.erase(std::unique(tuples.begin(), tuples.end()), tuples.end());
    auto tuple_index = [&](const std::vector<uint32_t>& t) {
        return static_cast<uint32_t>(
            std::lower_bound(tuples.begin(), tuples.end(), t) - tuples.begin());
    };
    std::vector<uint32_t> uf(tuples.size());
    std::iota(uf.begin(), uf.end(), 0u);
    auto find = [&](uint32_t x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    for (const auto& a : automorphisms) {
        for (uint32_t t = 0; t < tuples.size(); ++t) {
            std::vector<uint32_t> img;
            img.reserve(tuples[t].size());
            for (uint32_t v : tuples[t]) img.push_back(a[v]);
            const uint32_t x = find(t), y = find(tuple_index(img));
            if (x != y) uf[x] = y;
        }
    }

    // Numbered by the smallest CANONICAL content in the orbit, canonical meaning under the
    // labelling the core emitted -- which is the numbering rule the engine's callers rely on.
    std::map<uint32_t, std::vector<uint32_t>> smallest_canon;   // root -> min canonical tuple
    for (uint32_t t = 0; t < tuples.size(); ++t) {
        std::vector<uint32_t> canon;
        canon.reserve(tuples[t].size());
        for (uint32_t v : tuples[t]) canon.push_back(core_label[v]);
        const uint32_t r = find(t);
        auto it = smallest_canon.find(r);
        if (it == smallest_canon.end()) smallest_canon.emplace(r, std::move(canon));
        else if (canon < it->second) it->second = std::move(canon);
    }
    std::vector<std::pair<std::vector<uint32_t>, uint32_t>> by_content;
    for (const auto& kv : smallest_canon) by_content.emplace_back(kv.second, kv.first);
    std::sort(by_content.begin(), by_content.end());
    std::map<uint32_t, uint32_t> root_to_orbit;
    for (uint32_t i = 0; i < by_content.size(); ++i) root_to_orbit[by_content[i].second] = i;

    d.orbit.resize(edges.size());
    for (size_t ei = 0; ei < edges.size(); ++ei)
        d.orbit[ei] = root_to_orbit[find(tuple_index(local[ei]))];
    d.checked = true;
    return d;
}

// Relabel vertices by a random permutation, keeping the EDGE ORDER. The per-edge arrays are
// then comparable position by position, which is what makes them checkable for invariance:
// a state reached by two parents arrives under two labellings, and the quotient's per-instance
// reconstruction only meets if the orbit id it slots on is the same one both times.
static Edges vertex_permuted(const Edges& edges, std::mt19937_64& rng) {
    std::vector<VertexId> verts;
    for (const auto& e : edges) for (VertexId v : e) verts.push_back(v);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    std::vector<VertexId> images(verts.size());
    std::iota(images.begin(), images.end(), VertexId{1000});
    std::shuffle(images.begin(), images.end(), rng);
    Edges out;
    out.reserve(edges.size());
    for (const auto& e : edges) {
        std::vector<VertexId> m;
        m.reserve(e.size());
        for (VertexId v : e) {
            const size_t i = std::lower_bound(verts.begin(), verts.end(), v) - verts.begin();
            m.push_back(images[i]);
        }
        out.push_back(std::move(m));
    }
    return out;
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
    size_t rank_mismatch = 0, class_mismatch = 0, orbit_mismatch = 0;
    size_t def_checked = 0, def_form_mismatch = 0, def_orbit_mismatch = 0;
    size_t orbit_noninvariant = 0;
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

        // The three per-edge arrays, which the hash and the form both leave free.
        {
            const Arrays ac = core_arrays(corpus[i]);
            if (ac.ok) {
                const Arrays ah = host_arrays(corpus[i]);
                if (ac.rank != ah.rank) {
                    if (rank_mismatch < 5) printf("  RANK MISMATCH case %zu\n", i);
                    ++rank_mismatch;
                }
                if (ac.klass != ah.klass) {
                    if (class_mismatch < 5) printf("  CLASS MISMATCH case %zu\n", i);
                    ++class_mismatch;
                }
                if (ac.orbit != ah.orbit) {
                    if (orbit_mismatch < 5) printf("  ORBIT MISMATCH case %zu\n", i);
                    ++orbit_mismatch;
                }
            }
        }

        // Against the definition, on the states small enough to enumerate.
        {
            uint32_t stf = 0;
            std::vector<uint32_t> label;
            const std::vector<uint32_t> fc = core_form(corpus[i], &stf, &label);
            if (stf == hgcommon::IR_OK) {
                const Definition d = definition_of(corpus[i], label);
                if (d.checked) {
                    ++def_checked;
                    if (fc != d.label_form) {
                        if (def_form_mismatch < 10)
                            printf("  DEFINITION FORM MISMATCH case %zu: the emitted labelling"
                                   " does not produce the emitted form\n", i);
                        ++def_form_mismatch;
                    }
                    const Arrays ac = core_arrays(corpus[i]);
                    if (ac.ok && ac.orbit != d.orbit) {
                        if (def_orbit_mismatch < 10)
                            printf("  DEFINITION ORBIT MISMATCH case %zu\n", i);
                        ++def_orbit_mismatch;
                    }
                }
            }
        }

        // Invariance of the ORBIT ARRAY, at every size. The hash being invariant says nothing
        // about the orbit numbering, and the numbering is what the quotient slots on.
        {
            const Arrays ac = core_arrays(corpus[i]);
            if (ac.ok) {
                for (int t = 0; t < 3; ++t) {
                    const Arrays ap = core_arrays(vertex_permuted(corpus[i], rng));
                    if (ap.ok && ap.orbit != ac.orbit) {
                        if (orbit_noninvariant < 10)
                            printf("  ORBIT NOT RELABELLING-INVARIANT case %zu\n", i);
                        ++orbit_noninvariant;
                        break;
                    }
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

    printf("\n%zu states | hash: %zu | form: %zu | rank: %zu | class: %zu | orbit: %zu"
           " | isomorphism violations: %zu | depth-limited: %zu\n",
           corpus.size(), mismatch, form_mismatch, rank_mismatch, class_mismatch,
           orbit_mismatch, noniso, need_depth);
    printf("vs the definition (brute force, <=%u vertices): %zu of %zu states"
           " | form: %zu | orbit: %zu\n",
           kDefinitionMaxVerts, def_checked, corpus.size(), def_form_mismatch,
           def_orbit_mismatch);
    printf("orbit array not relabelling-invariant: %zu\n", orbit_noninvariant);
    return (mismatch || noniso || form_mismatch || rank_mismatch || class_mismatch
            || orbit_mismatch || def_form_mismatch || def_orbit_mismatch
            || orbit_noninvariant) ? 1 : 0;
}
