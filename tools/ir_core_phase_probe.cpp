// Where the exact canonical hash spends its time, per phase, on real states.
//
// The pipeline is re-driven here from the core's own public pieces rather than through
// ir_canonical_hash, so each phase can be timed in isolation and each is fed exactly the
// input the driver would give it. Phases are timed in separate passes over the whole state
// population rather than summed per state, so the timer's own cost is amortised.
//
// This answers what to attack next: an incremental scheme is only worth building against a
// phase that dominates, and only for the part of that phase a sibling actually shares.

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hgcommon/ir_core.hpp"

using namespace hypergraph;
using Edges = std::vector<std::vector<VertexId>>;

static std::vector<Edges> collect_states(int steps, int workload = 0) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, 1);

    std::vector<std::vector<VertexId>> init;
    if (workload == 0) {
        engine.add_rule(make_rule(0).lhs({0,1}).lhs({0,2})
                            .rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build());
        init = {{0u, 1u}, {0u, 2u}};
    } else {
        // Edge subdivision on a cycle. A subdivided cycle is a cycle, so every state carries a
        // dihedral automorphism group that grows with it -- the case where the initial
        // refinement CANNOT be discrete, because every vertex has the same degree and the same
        // neighbourhood signature. Without a workload of this shape the search counters are
        // measured only on rigid states, where they are all 1 by construction.
        engine.add_rule(make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build());
        init = {{0u, 1u}, {1u, 2u}, {2u, 0u}};
    }
    engine.evolve(init, steps);

    std::vector<Edges> out;
    for (StateId s = 0; s < hg.num_states(); ++s) {
        Edges e;
        hg.get_state(s).edges.for_each([&](EdgeId eid) {
            const Edge& ed = hg.get_edge(eid);
            e.emplace_back(ed.vertices, ed.vertices + ed.arity);
        });
        if (!e.empty()) out.push_back(std::move(e));
    }
    return out;
}

struct Flat {
    std::vector<uint8_t> ea;
    std::vector<uint32_t> eoff, ev;
    uint32_t n_verts = 0, total_occ = 0;
    uint32_t n_edges() const { return static_cast<uint32_t>(ea.size()); }
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
        for (VertexId v : e)
            f.ev.push_back(static_cast<uint32_t>(
                std::lower_bound(verts.begin(), verts.end(), v) - verts.begin()));
    }
    f.total_occ = static_cast<uint32_t>(f.ev.size());
    return f;
}

// One scratch layout matching the core's own sub-allocation order, so the phases can be
// driven individually with the buffers they expect.
struct Views {
    std::vector<uint64_t> store;
    uint32_t *occ_off, *occ_edge, *occ_pos, *cursor, *inc_edges, *edge_epoch, *form_order;
    uint32_t *touched, *on_touched, *torder, *sig_off, *sig_cnt, *gstart, *form;
    uint64_t *worklist, *sig_buf;
    hgcommon::IrPartition pi;
    std::vector<uint32_t> part, labeling;

    void setup(const Flat& f) {
        const uint32_t n = f.n_verts, e = f.n_edges(), occ = f.total_occ;
        const size_t need = size_t(n + 1) + occ + occ + n + e + e + e + n + n + 2 * n
                          + (n + 1) + n + (n + 1) + (occ + e) + 64;
        store.assign(need, 0);
        uint32_t* p = reinterpret_cast<uint32_t*>(store.data());
        auto take = [&](size_t k) { uint32_t* q = p; p += k; return q; };
        occ_off = take(n + 1); occ_edge = take(occ); occ_pos = take(occ); cursor = take(n);
        inc_edges = take(e); edge_epoch = take(e); form_order = take(e);
        touched = take(n); on_touched = take(n); torder = take(2 * n);
        sig_off = take(n + 1); sig_cnt = take(n); gstart = take(n + 1);
        form = take(occ + e);

        wl.assign(hgcommon::ir_bitset_words(n) + 4, 0);
        sig.assign(occ + 4, 0);
        worklist = wl.data(); sig_buf = sig.data();

        part.assign(size_t(5) * n + 8, 0);
        labeling.assign(n, 0);
        pi.lab = part.data(); pi.pos = part.data() + n; pi.cell_of = part.data() + 2 * n;
        pi.cstart = part.data() + 3 * n; pi.clen = part.data() + 4 * n;
        pi.n = n; pi.ncells = 0;
    }
    std::vector<uint64_t> wl, sig;
};

// How the SEARCH cost is spread across states, at a given automorphism-generator budget.
//
// The phase timings above are population totals, which is what phase attribution needs and is
// exactly what hides this: a mean over states says nothing about whether one state costs a
// thousand times another. Both engines run one thread per state start to finish
// (gpu/src/ir_canon.cu:264 is a grid-stride loop over states), so a kernel finishes when its
// WORST state finishes. Splitting a single state across threads is worth building only if a
// minority of states carries the work -- which is a question about the tail, not the mean.
//
// Counted, not timed: leaves and individualization nodes are exact and depend only on the state,
// the depth limit and the generator budget, so the numbers are the DEVICE's too. The generator
// budget is the one input the two engines differ on (IR_HOST_GENERATORS 512 against
// IR_DEVICE_GENERATORS 32), and a smaller table prunes fewer symmetric branches, so it is passed
// in rather than assumed.
static void work_distribution_at(const std::vector<Flat>& flat, uint32_t generators,
                                 const char* label) {
    std::vector<uint32_t> leaves;
    leaves.reserve(flat.size());
    uint64_t total_leaves = 0, total_nodes = 0;
    size_t never_searched = 0;
    uint32_t deepest = 0;

    std::vector<uint32_t> scratch;
    for (const auto& f : flat) {
        const uint64_t need = hgcommon::ir_scratch_words(
            f.n_verts, f.n_edges(), f.total_occ, hgcommon::IR_MAX_DEPTH_DEFAULT, generators);
        if (scratch.size() < need) scratch.assign(need, 0);

        hgcommon::IrWork w{};
        hgcommon::ir_canonical_hash(f.ea.data(), f.eoff.data(), f.ev.data(),
                                    f.n_edges(), f.n_verts, f.total_occ,
                                    scratch.data(), hgcommon::IR_MAX_DEPTH_DEFAULT,
                                    nullptr, generators,
                                    nullptr, nullptr, nullptr, nullptr, &w);
        leaves.push_back(w.leaves);
        total_leaves += w.leaves;
        total_nodes  += w.nodes;
        if (!w.searched) ++never_searched;
        if (w.max_depth > deepest) deepest = w.max_depth;
    }

    std::sort(leaves.begin(), leaves.end());
    const size_t n = leaves.size();
    auto pct = [&](double p) { return leaves[std::min(n - 1, size_t(p * n))]; };

    // The decisive figure: the share of all search that the heaviest 1% of states accounts for.
    // Intra-state parallelism can only ever attack that share.
    const size_t top1_from = n - std::max<size_t>(1, n / 100);
    uint64_t top1_leaves = 0;
    for (size_t i = top1_from; i < n; ++i) top1_leaves += leaves[i];

    printf("\n  --- per-state search cost, %s (%u generators) ---\n", label, generators);
    printf("  never searched (initial refinement already discrete): %zu/%zu  (%.1f%%)\n",
           never_searched, n, 100.0 * double(never_searched) / double(n));
    printf("  leaves per state:  p50 %u   p90 %u   p99 %u   max %u\n",
           pct(0.50), pct(0.90), pct(0.99), leaves.back());
    printf("  totals: %llu leaves, %llu nodes, deepest individualization %u\n",
           (unsigned long long)total_leaves, (unsigned long long)total_nodes, deepest);
    printf("  heaviest 1%% of states (%zu of %zu) carry %.1f%% of all leaves\n",
           n - top1_from, n, total_leaves ? 100.0 * double(top1_leaves) / double(total_leaves) : 0.0);
}

static void work_distribution(const std::vector<Flat>& flat) {
    if (flat.empty()) return;
    work_distribution_at(flat, hgcommon::IR_HOST_GENERATORS, "host budget");
    work_distribution_at(flat, hgcommon::IR_DEVICE_GENERATORS, "device budget");
}

// The counters against a state whose answer is known independently.
//
// A 6-cycle's automorphism group is dihedral of order 12, and its initial refinement puts all
// six vertices in one cell -- every vertex has the same degree and the same neighbourhood
// signature -- so the search MUST individualize and MUST reach more than one leaf. A population
// whose states are all rigid exercises none of that, and counters that are never exercised are
// not evidence. If this case reports no search, the numbers above mean nothing.
static hgcommon::IrWork canonicalize_cycle(uint32_t len, uint32_t generators) {
    Edges cycle;
    for (uint32_t i = 0; i < len; ++i)
        cycle.push_back({VertexId(i), VertexId((i + 1) % len)});
    const Flat f = flatten(cycle);

    std::vector<uint32_t> scratch(hgcommon::ir_scratch_words(
        f.n_verts, f.n_edges(), f.total_occ, hgcommon::IR_MAX_DEPTH_DEFAULT, generators), 0);
    hgcommon::IrWork w{};
    hgcommon::ir_canonical_hash(f.ea.data(), f.eoff.data(), f.ev.data(),
                                f.n_edges(), f.n_verts, f.total_occ,
                                scratch.data(), hgcommon::IR_MAX_DEPTH_DEFAULT,
                                nullptr, generators,
                                nullptr, nullptr, nullptr, nullptr, &w);
    return w;
}

static void ground_truth() {
    // A 6-cycle's automorphism group is dihedral of order 12, and its initial refinement puts
    // all six vertices in one cell -- every vertex has the same degree and the same
    // neighbourhood signature -- so the search MUST individualize and MUST reach more than one
    // leaf. Counters that are never exercised are not evidence.
    const hgcommon::IrWork g = canonicalize_cycle(6, hgcommon::IR_HOST_GENERATORS);
    const bool ok = g.searched == 1 && g.leaves >= 2 && g.nodes >= 1 && g.max_depth >= 1;
    printf("\n  ground truth, 6-cycle (Aut = D6, order 12): searched=%u leaves=%u nodes=%u "
           "depth=%u  -> %s\n", g.searched, g.leaves, g.nodes, g.max_depth,
           ok ? "counters respond" : "COUNTERS ARE DEAD -- numbers above are meaningless");

    // Does the search GROW with the state? A population sampled at one evolution depth cannot
    // say, because its states are all about one size. C_n is the worst case symmetry can
    // present -- one cell containing every vertex, Aut of order 2n -- so if the search stays
    // bounded here it stays bounded, and the answer does not depend on how far a run evolved.
    printf("  cycle size sweep (worst case for symmetry), host budget then device budget:\n");
    for (uint32_t len : {6u, 12u, 24u, 48u, 96u, 192u, 384u}) {
        const hgcommon::IrWork h = canonicalize_cycle(len, hgcommon::IR_HOST_GENERATORS);
        const hgcommon::IrWork d = canonicalize_cycle(len, hgcommon::IR_DEVICE_GENERATORS);
        printf("    C_%-4u  host leaves=%-5u nodes=%-5u depth=%-3u | device leaves=%-5u "
               "nodes=%-5u depth=%u\n",
               len, h.leaves, h.nodes, h.max_depth, d.leaves, d.nodes, d.max_depth);
    }
}

int main(int argc, char** argv) {
    const int steps = argc > 1 ? atoi(argv[1]) : 6;
    const int reps  = argc > 2 ? atoi(argv[2]) : 7;

    auto states = collect_states(steps);
    std::vector<Flat> flat;
    for (const auto& s : states) flat.push_back(flatten(s));
    std::vector<Views> views(flat.size());
    for (size_t i = 0; i < flat.size(); ++i) views[i].setup(flat[i]);

    size_t total_edges = 0, total_verts = 0;
    for (const auto& f : flat) { total_edges += f.n_edges(); total_verts += f.n_verts; }
    printf("states=%zu  mean edges=%.1f  mean verts=%.1f\n", flat.size(),
           double(total_edges) / flat.size(), double(total_verts) / flat.size());

    uint64_t sink = 0;
    auto bench = [&](const char* name, auto&& body) {
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            body();
            best = std::min(best, std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
        }
        printf("  %-26s %7.2f ms\n", name, best);
        return best;
    };

    // Marshalling is what an incremental scheme would have to beat as well: it is redone per
    // state today, and a sibling shares most of its vertex set.
    const double t_flat = bench("flatten (sort+unique)", [&]{
        for (const auto& s : states) { Flat f = flatten(s); sink += f.n_verts; }
    });
    const double t_occ = bench("occurrence CSR", [&]{
        for (size_t i = 0; i < flat.size(); ++i) {
            auto& f = flat[i]; auto& v = views[i];
            hgcommon::ir_build_occurrences(f.ea.data(), f.eoff.data(), f.ev.data(),
                                           f.n_edges(), f.n_verts,
                                           v.occ_off, v.occ_edge, v.occ_pos, v.cursor);
        }
    });
    const double t_init = bench("initial partition", [&]{
        for (size_t i = 0; i < flat.size(); ++i) {
            auto& f = flat[i]; auto& v = views[i];
            hgcommon::ir_initial_partition(f.ea.data(), v.occ_off, v.occ_edge, v.occ_pos,
                                           f.n_verts, v.pi, v.sig_buf, v.torder);
            sink += v.pi.ncells;
        }
    });
    const double t_ref = bench("refine", [&]{
        for (size_t i = 0; i < flat.size(); ++i) {
            auto& f = flat[i]; auto& v = views[i];
            // Re-seed the partition so every repetition refines the same input.
            hgcommon::ir_initial_partition(f.ea.data(), v.occ_off, v.occ_edge, v.occ_pos,
                                           f.n_verts, v.pi, v.sig_buf, v.torder);
            hgcommon::ir_refine(f.ea.data(), f.eoff.data(), f.ev.data(), f.n_edges(),
                                v.occ_off, v.occ_edge, v.pi,
                                v.worklist, v.inc_edges, v.edge_epoch,
                                v.touched, v.on_touched, v.torder,
                                v.sig_off, v.sig_cnt, v.gstart, v.sig_buf);
            sink += v.pi.ncells;
        }
    });
    const double t_form = bench("form + hash", [&]{
        for (size_t i = 0; i < flat.size(); ++i) {
            auto& f = flat[i]; auto& v = views[i];
            for (uint32_t x = 0; x < f.n_verts; ++x) v.labeling[x] = v.pi.cell_of[x];
            hgcommon::ir_build_form(f.ea.data(), f.eoff.data(), f.ev.data(), f.n_edges(),
                                    v.labeling.data(), v.form, v.form_order);
            sink += hgcommon::ir_hash_form(v.form, f.n_edges(), f.n_verts);
        }
    });

    // refine's timing includes a re-seed, so the refinement proper is the difference.
    const double refine_only = t_ref - t_init;
    const double total = t_flat + t_occ + t_init + refine_only + t_form;
    printf("\n  refinement proper          %7.2f ms  (refine pass minus its re-seed)\n", refine_only);
    printf("  ---------------------------------------\n");
    printf("  %-26s %7.2f ms\n", "sum of phases", total);
    printf("\n  share: flatten %.0f%%  occCSR %.0f%%  initial %.0f%%  refine %.0f%%  form+hash %.0f%%\n",
           100 * t_flat / total, 100 * t_occ / total, 100 * t_init / total,
           100 * refine_only / total, 100 * t_form / total);
    printf("(sink %llu)\n", (unsigned long long)sink);

    ground_truth();

    printf("\n================ workload 0: branching rule (states are rigid) ================");
    work_distribution(flat);

    // The same measurement where symmetry is the norm rather than the exception. One rule's
    // population answers for one rule; the decision this feeds is about the engine.
    auto sym_states = collect_states(steps, 1);
    std::vector<Flat> sym_flat;
    sym_flat.reserve(sym_states.size());
    for (const auto& s : sym_states) sym_flat.push_back(flatten(s));
    printf("\n================ workload 1: cycle subdivision (Aut grows with the state) ======");
    work_distribution(sym_flat);
    return 0;
}
