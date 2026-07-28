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

static std::vector<Edges> collect_states(int steps) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, 1);
    engine.add_rule(make_rule(0).lhs({0,1}).lhs({0,2})
                        .rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build());
    std::vector<std::vector<VertexId>> init = {{0u, 1u}, {0u, 2u}};
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
    return 0;
}
