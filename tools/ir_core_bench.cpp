// Times the two exact-canonicalization implementations against each other on REAL states.
//
// End-to-end evolution timings cannot resolve this change: the largest configuration drifts
// over 10% between runs on this host, which is several times the effect. So the states come
// from an actual evolution and are then canonicalized in a tight loop, which removes the
// scheduler, the matcher and the allocator from the measurement and leaves the thing that
// changed. Reported as the minimum over repetitions -- the minimum rejects host load, the mean
// absorbs it.

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hgcommon/ir_core.hpp"

using namespace hypergraph;
using Edges = std::vector<std::vector<VertexId>>;

static std::vector<Edges> collect_states(int steps) {
    Hypergraph hg;
    hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine engine(&hg, 1);

    // The Wolfram rule the rest of the profiling tools drive, so the state population here
    // is the same one the engine actually canonicalizes.
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

int main(int argc, char** argv) {
    const int steps = argc > 1 ? atoi(argv[1]) : 6;
    const int reps  = argc > 2 ? atoi(argv[2]) : 7;

    auto states = collect_states(steps);
    size_t total_edges = 0, max_edges = 0;
    for (const auto& s : states) { total_edges += s.size(); max_edges = std::max(max_edges, s.size()); }
    printf("states=%zu  edges: total=%zu mean=%.1f max=%zu\n",
           states.size(), total_edges,
           states.empty() ? 0.0 : double(total_edges) / states.size(), max_edges);

    // Pre-flatten so the shared core is timed on canonicalization, not on marshalling; the
    // host path takes the edge lists it already wants.
    std::vector<Flat> flat;
    flat.reserve(states.size());
    for (const auto& s : states) flat.push_back(flatten(s));

    std::vector<uint64_t> scratch;
    uint64_t sink = 0;

    auto time_host = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        IRCanonicalizer ir;
        for (const auto& s : states) sink += ir.compute_canonical_hash(s);
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    };
    auto time_core = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        for (const auto& f : flat) {
            for (uint32_t depth : {1u, 8u, hgcommon::IR_MAX_DEPTH_DEFAULT}) {
                const uint64_t words = hgcommon::ir_scratch_words(
                    f.n_verts, static_cast<uint32_t>(f.ea.size()), f.total_occ, depth);
                if (scratch.size() < (words + 3) / 2) scratch.resize((words + 3) / 2);
                auto r = hgcommon::ir_canonical_hash(
                    f.ea.data(), f.eoff.data(), f.ev.data(),
                    static_cast<uint32_t>(f.ea.size()), f.n_verts, f.total_occ,
                    reinterpret_cast<uint32_t*>(scratch.data()), depth);
                if (r.status != hgcommon::IR_NEED_DEPTH) { sink += r.hash; break; }
            }
        }
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    };

    time_host(); time_core();  // warm

    double best_host = 1e30, best_core = 1e30;
    for (int i = 0; i < reps; ++i) {
        best_host = std::min(best_host, time_host());
        best_core = std::min(best_core, time_core());
    }

    printf("host IRCanonicalizer : %8.2f ms\n", best_host);
    printf("shared ir_core       : %8.2f ms   (%+.1f%%)\n",
           best_core, (best_core - best_host) * 100.0 / best_host);
    printf("(sink %llu)\n", (unsigned long long)sink);
    return 0;
}
