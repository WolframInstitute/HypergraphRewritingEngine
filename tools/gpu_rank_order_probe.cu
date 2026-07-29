// Why do the device's canonical edge RANKS move between block counts on an automorphic state,
// when the CPU's do not?
//
// Measured before this probe: on a directed 4-cycle the persistent scheduler's produced-edge
// rank component gives 15 event signatures at 3 blocks and 15 at 17, with different values; on a
// rigid initial state every key component is identical. The CPU, over 72 runs per cell across
// worker counts {1,2,3,5,8,16}, never moved on either automorphic corpus case.
//
// Two candidate mechanisms, and they call for different fixes:
//
//   (A) PRESENTATION ORDER. The canonicalizer is handed a state's edges in CSR slice order,
//       which is the order the rewrites appended them. If that order differs between runs, the
//       individualization search visits leaves in a different order; on an automorphic state
//       several leaves reach the identical canonical form, the FIRST minimal one wins, and the
//       winner's edge order is the rank assignment. Same form, different ranks.
//
//   (B) RANK ASSIGNMENT. The presentation order is stable and something in how best_order is
//       captured or scattered is not.
//
// This distinguishes them. For each state it prints the canonical hash, the edge tuples in the
// order the canonicalizer will see them, and the rank each slot received. Keyed by canonical
// hash so the same state can be found across two runs whose raw ids differ by construction.
//
// Reading it: if two runs disagree on the TUPLE SEQUENCE for a shared canonical hash, it is (A).
// If they agree on the tuples and disagree on the RANKS, it is (B).

#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/rewrite.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace hg_gpu;

namespace {

struct StateShot {
    std::string tuples;   // edge tuples in presentation order
    std::string ranks;    // rank per slot, same order
};

// One run at a given block count, returned keyed by canonical hash.
std::map<uint64_t, std::vector<StateShot>> run_once(
        const std::vector<std::vector<VertexId>>& init,
        const RewriteRule& rule, uint32_t steps, uint32_t blocks) {
    EvolveInput in;
    in.rules = {rule};
    in.initial_state = init;
    in.num_steps = steps;
    in.canonicalization = CanonicalizationMode::Full;

    EngineConfig cfg = config_from_input(in);
    EngineState engine(cfg);
    upload_initial_state(engine, init);
    engine.ensure_edge_ranks();

    std::vector<DeviceRule> rules = {make_device_rule(rule)};
    Pool<MatchRecord> matches(cfg.max_states * 8u);
    matches.reset();
    DeviceArena arena(64ull << 20);

    run_persistent_evolve(engine, rules, /*roots=*/{0u}, steps, matches, arena,
                          /*dedup=*/true, 0xFFFFFFFFu, 0,
                          CanonicalizationMode::Full,
                          hgcommon::EVENT_SIG_AUTOMATIC, blocks);

    const uint32_t ns = engine.num_states_host();
    std::vector<uint64_t> hashes(ns);
    cudaMemcpy(hashes.data(), engine.device().state_canonical_hash,
               sizeof(uint64_t) * ns, cudaMemcpyDeviceToHost);

    // The rank array is parallel to state_edge_ids, so both are read whole and sliced per state.
    std::vector<uint32_t> ranks(cfg.max_state_edge_total);
    cudaMemcpy(ranks.data(), engine.device().state_edge_rank,
               sizeof(uint32_t) * cfg.max_state_edge_total, cudaMemcpyDeviceToHost);
    std::vector<StateEdgeSlice> slices(ns);
    cudaMemcpy(slices.data(), engine.device().state_edge_slices,
               sizeof(StateEdgeSlice) * ns, cudaMemcpyDeviceToHost);

    std::map<uint64_t, std::vector<StateShot>> out;
    for (uint32_t sid = 0; sid < ns; ++sid) {
        const StateEdgeSlice& sl = slices[sid];
        if (sl.count == 0) continue;
        StateShot shot;
        std::vector<EdgeId> eids = engine.state_edges_host(sid);

        // Vertices renumbered by FIRST APPEARANCE in presentation order -- which is exactly
        // what flatten_state does, so this prints the canonicalizer's own view of the state.
        // Printing raw ids instead would flag every run as different for a reason that is not
        // the question: fresh vertices come from an atomic high-water bump, so two runs build
        // the same hypergraph under a different numbering by construction.
        std::map<VertexId, uint32_t> local;
        for (uint32_t k = 0; k < eids.size(); ++k) {
            std::vector<VertexId> vs = engine.edge_vertices_host(eids[k]);
            shot.tuples += "(";
            for (size_t i = 0; i < vs.size(); ++i) {
                auto [it, fresh] = local.emplace(vs[i], static_cast<uint32_t>(local.size()));
                (void)fresh;
                if (i) shot.tuples += ",";
                shot.tuples += std::to_string(it->second);
            }
            shot.tuples += ")";
            const uint32_t r = ranks[sl.offset + k];
            shot.ranks += (r == UINT32_MAX ? std::string("-") : std::to_string(r)) + " ";
        }
        out[hashes[sid]].push_back(shot);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const uint32_t blocks_a = argc > 1 ? std::atoi(argv[1]) : 3;
    const uint32_t blocks_b = argc > 2 ? std::atoi(argv[2]) : 17;
    const uint32_t steps    = argc > 3 ? std::atoi(argv[3]) : 3;

    RewriteRule r;
    r.lhs = {{0, 1}, {1, 2}};
    r.rhs = {{0, 1}, {1, 3}, {3, 2}};
    r.num_lhs_vars = 3;
    r.num_rhs_vars = 4;
    const std::vector<std::vector<VertexId>> init = {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}};

    auto a = run_once(init, r, steps, blocks_a);
    auto b = run_once(init, r, steps, blocks_b);

    std::printf("# directed 4-cycle, %u steps, %u blocks vs %u blocks\n",
                steps, blocks_a, blocks_b);
    std::printf("# states: %zu at %u blocks, %zu at %u blocks\n",
                a.size(), blocks_a, b.size(), blocks_b);

    size_t shared = 0, tuple_diff = 0, rank_diff = 0;
    for (const auto& [hash, shots_a] : a) {
        auto it = b.find(hash);
        if (it == b.end()) continue;
        ++shared;
        const auto& shots_b = it->second;
        // A canonical hash can name several raw states; compared as the set of presentations so
        // the comparison does not depend on which raw id came first.
        std::vector<std::string> ta, tb, ra, rb;
        for (const auto& s : shots_a) { ta.push_back(s.tuples); ra.push_back(s.ranks); }
        for (const auto& s : shots_b) { tb.push_back(s.tuples); rb.push_back(s.ranks); }
        std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
        std::sort(ra.begin(), ra.end()); std::sort(rb.begin(), rb.end());

        const bool tuples_differ = (ta != tb);
        const bool ranks_differ  = (ra != rb);
        if (tuples_differ) ++tuple_diff;
        if (ranks_differ)  ++rank_diff;

        if (tuples_differ || ranks_differ) {
            std::printf("\nhash %llu  %s%s\n", (unsigned long long)hash,
                        tuples_differ ? "TUPLE-ORDER-DIFFERS " : "",
                        ranks_differ  ? "RANKS-DIFFER" : "");
            for (size_t i = 0; i < shots_a.size(); ++i)
                std::printf("  %u: %s   ranks: %s\n", blocks_a,
                            shots_a[i].tuples.c_str(), shots_a[i].ranks.c_str());
            for (size_t i = 0; i < shots_b.size(); ++i)
                std::printf("  %u: %s   ranks: %s\n", blocks_b,
                            shots_b[i].tuples.c_str(), shots_b[i].ranks.c_str());
        }
    }

    std::printf("\n# %zu canonical states in both runs; %zu differ in presentation order, "
                "%zu differ in ranks\n", shared, tuple_diff, rank_diff);
    if (tuple_diff)      std::printf("# => (A) PRESENTATION ORDER is not stable\n");
    else if (rank_diff)  std::printf("# => (B) RANK ASSIGNMENT is not stable at fixed order\n");
    else                 std::printf("# => per-state presentation and ranks both stable\n");

    // The per-state view can be stable while the EVENT view is not: a canonical class holds
    // several raw states with different presentations, and which one an event is attached to
    // follows the race. Ranks are read out of that raw state, so the event's ranks move even
    // though every state's own ranks are fixed. Compared here as multisets keyed by the
    // endpoint hashes, which the earlier per-component measurement already showed are stable.
    auto events_of = [&](uint32_t blocks) {
        std::vector<std::string> rows;
        EvolveInput in;
        in.rules = {r};
        in.initial_state = init;
        in.num_steps = steps;
        in.canonicalization = CanonicalizationMode::Full;

        EngineConfig cfg = config_from_input(in);
        EngineState engine(cfg);
        upload_initial_state(engine, init);
        engine.ensure_edge_ranks();

        std::vector<DeviceRule> rules = {make_device_rule(r)};
        Pool<MatchRecord> matches(cfg.max_states * 8u);
        matches.reset();
        DeviceArena arena(64ull << 20);
        run_persistent_evolve(engine, rules, /*roots=*/{0u}, steps, matches, arena,
                              /*dedup=*/true, 0xFFFFFFFFu, 0,
                              CanonicalizationMode::Full,
                              hgcommon::EVENT_SIG_AUTOMATIC, blocks);

        const uint32_t ne = engine.num_events_host();
        std::vector<DeviceEvent> events(ne);
        cudaMemcpy(events.data(), engine.device().event_pool.data,
                   sizeof(DeviceEvent) * ne, cudaMemcpyDeviceToHost);
        const uint32_t ns = engine.num_states_host();
        std::vector<uint64_t> exact(ns);
        cudaMemcpy(exact.data(), engine.device().state_exact_hash,
                   sizeof(uint64_t) * ns, cudaMemcpyDeviceToHost);
        std::vector<uint32_t> rk(cfg.max_state_edge_total);
        cudaMemcpy(rk.data(), engine.device().state_edge_rank,
                   sizeof(uint32_t) * cfg.max_state_edge_total, cudaMemcpyDeviceToHost);
        std::vector<StateEdgeSlice> sl(ns);
        cudaMemcpy(sl.data(), engine.device().state_edge_slices,
                   sizeof(StateEdgeSlice) * ns, cudaMemcpyDeviceToHost);

        auto rank_of = [&](StateId s, EdgeId e) -> std::string {
            if (s >= ns) return "?";
            std::vector<EdgeId> ids = engine.state_edges_host(s);
            for (uint32_t k = 0; k < ids.size(); ++k)
                if (ids[k] == e) {
                    uint32_t v = rk[sl[s].offset + k];
                    return v == UINT32_MAX ? "-" : std::to_string(v);
                }
            return "x";
        };

        for (const auto& ev : events) {
            if (ev.id == INVALID_ID) continue;
            std::string row = std::to_string(ev.input_state < ns ? exact[ev.input_state] : 0)
                            + "->" + std::to_string(ev.output_state < ns ? exact[ev.output_state] : 0)
                            + " step" + std::to_string(ev.step) + " cons[";
            for (uint8_t i = 0; i < ev.num_consumed; ++i)
                row += rank_of(ev.input_state, ev.consumed_edges[i]) + " ";
            row += "] prod[";
            for (uint8_t i = 0; i < ev.num_produced; ++i)
                row += rank_of(ev.output_state, ev.produced_edges[i]) + " ";
            row += "]";
            rows.push_back(row);
        }
        std::sort(rows.begin(), rows.end());
        return rows;
    };

    const auto ea = events_of(blocks_a);
    const auto eb = events_of(blocks_b);
    std::printf("\n# events: %zu at %u blocks, %zu at %u blocks\n",
                ea.size(), blocks_a, eb.size(), blocks_b);
    if (ea == eb) {
        std::printf("# event rank rows IDENTICAL as multisets\n");
    } else {
        std::printf("# event rank rows DIFFER -- rows only in one run:\n");
        for (const auto& row : ea)
            if (!std::binary_search(eb.begin(), eb.end(), row))
                std::printf("  %u only: %s\n", blocks_a, row.c_str());
        for (const auto& row : eb)
            if (!std::binary_search(ea.begin(), ea.end(), row))
                std::printf("  %u only: %s\n", blocks_b, row.c_str());
    }
    return 0;
}
