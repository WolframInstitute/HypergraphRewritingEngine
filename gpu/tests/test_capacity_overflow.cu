// A run that outgrows its pools must return PARTIAL WORK WITH A WARNING, and must not read or
// write outside an allocation while doing it.
//
// Overflow is not an error condition here, it is a contract: the engine reports what it managed
// and says so, rather than throwing. That makes it easy for the reporting to be present while the
// memory safety is not, because a run that quietly walks off a pool still returns a plausible
// partial answer and a warning. The two halves are asserted separately, and the memory half is
// only meaningful under compute-sanitizer:
//
//     compute-sanitizer --tool memcheck --error-exitcode 9 ./hg_gpu_tests \\
//         --gtest_filter='CapacityOverflow.*'
//
// Engine(cfg) is used directly rather than evolve(), because evolve() answers an overflow by
// doubling the config and re-running -- which is the right behaviour for a caller and the wrong
// one for a test that needs the overflow to actually happen.
//
// THE HAZARD THIS GUARDS. Pool::claim() bumps its counter BEFORE reporting exhaustion, so under
// overflow the counter runs past capacity and is not a count of valid entries. Every consumer
// therefore has to ask size(), which clamps. A consumer that read the counter raw would iterate
// off the end of the allocation, and only a sanitizer run over a genuinely overflowing evolution
// can show that it does not.

#include <gtest/gtest.h>

#include "hg_gpu/evolve.hpp"
#include "hg_gpu/persistent.hpp"
#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/initial_upload.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

hg_gpu::RewriteRule growth_rule() {
    hg_gpu::RewriteRule r;
    r.lhs = {{0, 1}};
    r.rhs = {{0, 1}, {1, 2}};
    r.num_lhs_vars = 2;
    r.num_rhs_vars = 3;
    return r;
}

hg_gpu::EvolveInput growing_input(uint32_t steps) {
    hg_gpu::EvolveInput in;
    in.rules = {growth_rule()};
    in.initial_state = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    in.num_steps = steps;
    in.canonicalization = hg_gpu::CanonicalizationMode::Full;
    return in;
}

}  // namespace

// Pools far too small for the evolution requested. The run must come back, say so, and stay
// inside its allocations.
TEST(CapacityOverflow, PartialResultWithWarningAndNoOutOfBounds) {
    const hg_gpu::EvolveInput in = growing_input(6);

    // Start from what the auto-tuner would pick, then starve the pools that this rule grows.
    hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
    cfg.max_edges            = 512;
    cfg.max_states           = 64;
    cfg.max_state_edge_total = 2048;
    cfg.max_events           = 64;

    hg_gpu::Engine engine(cfg);
    hg_gpu::EvolveResult res = engine.run(in);

    EXPECT_FALSE(res.warnings.empty())
        << "the run outgrew its pools and reported nothing, so a caller cannot tell a truncated "
        << "evolution from a complete one";

    // Partial, not empty: the contract is that the work already done is returned.
    EXPECT_GT(res.states.size(), 0u) << "overflow discarded work that had already been computed";

    // Whatever came back must be internally consistent -- an event may not name a state that is
    // not in the result, which is what a truncation that forgot to clip its outputs would produce.
    std::vector<bool> present(cfg.max_states + 1, false);
    for (const auto& s : res.states)
        if (s.id < present.size()) present[s.id] = true;
    for (const auto& e : res.events) {
        if (e.input_state != hg_gpu::INVALID_ID && e.input_state < present.size())
            EXPECT_TRUE(present[e.input_state])
                << "event " << e.id << " names input state " << e.input_state
                << ", which the truncated result does not contain";
        if (e.output_state != hg_gpu::INVALID_ID && e.output_state < present.size())
            EXPECT_TRUE(present[e.output_state])
                << "event " << e.id << " names output state " << e.output_state
                << ", which the truncated result does not contain";
    }
}

// The same starvation on the persistent scheduler, which cannot grow-and-retry at all: one
// launch, no host in the loop, so returning partial work is its ONLY option rather than a
// fallback.
TEST(CapacityOverflow, PersistentSchedulerAlsoReturnsPartialWork) {
    hg_gpu::EvolveInput in = growing_input(6);

    hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
    cfg.max_edges            = 512;
    cfg.max_states           = 64;
    cfg.max_state_edge_total = 2048;
    cfg.max_events           = 64;

    hg_gpu::Engine engine(cfg);
    hg_gpu::EvolveResult res = engine.run(in);

    EXPECT_FALSE(res.warnings.empty())
        << "the persistent scheduler truncated silently, and it has no retry to fall back on";
    EXPECT_GT(res.states.size(), 0u);
}

// WHERE an overflowing run stops is NOT deterministic, and that is a property to state rather
// than a defect to chase.
//
// Under overflow the workers race for the last slots in a pool and whoever claims one keeps it.
// Making the truncation point reproducible would mean ordering those claims, which is exactly the
// barrier the lock-free design exists to avoid -- so the run stops wherever the schedule left it.
// Measured: the same starved configuration returned 49 states on one run and 55 on another when
// other tests had run first, and returned a stable count when run alone.
//
// What IS guaranteed, and what a caller can build on, is that whatever comes back is a valid
// partial answer: warned about, non-empty, and internally consistent. That is asserted above.
// This test pins the weaker claim so the stronger one is not assumed by a later reader: the run
// stays WITHIN its configured capacity, however far it happened to get.
TEST(CapacityOverflow, TruncationStaysWithinCapacityEvenThoughItsPointIsNotFixed) {
    const hg_gpu::EvolveInput in = growing_input(6);
    hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
    cfg.max_edges            = 512;
    cfg.max_states           = 64;
    cfg.max_state_edge_total = 2048;
    cfg.max_events           = 64;

    for (int rep = 0; rep < 3; ++rep) {
        hg_gpu::Engine engine(cfg);
        const hg_gpu::EvolveResult res = engine.run(in);
        EXPECT_LE(res.states.size(), static_cast<size_t>(cfg.max_states))
            << "an overflowing run returned more states than its pool could hold";
        EXPECT_LE(res.events.size(), static_cast<size_t>(cfg.max_events))
            << "an overflowing run returned more events than its pool could hold";
        EXPECT_GT(res.states.size(), 0u);
        EXPECT_FALSE(res.warnings.empty());
    }
}

// A capacity overflow marks a partial result; the two kinds that describe a complete run do not,
// and carry the names the CPU reports for the same conditions. The kernel reports only a partial
// warning as HGEvolve::overflow.
TEST(CapacityOverflow, OnlyCapacityKindsMarkAPartialResult) {
    EXPECT_TRUE(hg_gpu::error_kind_is_partial(hg_gpu::ErrorKind::kQcNodes));
    EXPECT_TRUE(hg_gpu::error_kind_is_partial(hg_gpu::ErrorKind::kEventPoolFull));
    EXPECT_FALSE(hg_gpu::error_kind_is_partial(hg_gpu::ErrorKind::kEventSigRawFallback));
    EXPECT_FALSE(hg_gpu::error_kind_is_partial(hg_gpu::ErrorKind::kDrainCapBufferFull));
    EXPECT_STREQ(hg_gpu::error_kind_name(hg_gpu::ErrorKind::kEventSigRawFallback),
                 "EventSigRawFallback");
}

// An engine that does not fit in the free device memory stops before its first kernel and
// returns with kDeviceOutOfMemory: under WDDM an oversubscribed allocation pages to system memory
// and the run would otherwise continue about 100x slower.
TEST(CapacityOverflow, AnEngineThatDoesNotFitStopsAndReports) {
    size_t free_b = 0, total_b = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
    const size_t leave = total_b / 128;
    ASSERT_GT(free_b, leave);
    void* hog = nullptr;
    ASSERT_EQ(cudaMalloc(&hog, free_b - leave), cudaSuccess);
    const hg_gpu::EvolveResult r = hg_gpu::evolve(growing_input(3));
    cudaFree(hog);
    bool oom = false;
    for (const auto& w : r.warnings) oom = oom || w.kind == hg_gpu::ErrorKind::kDeviceOutOfMemory;
    EXPECT_TRUE(oom);
}

// A full claim map is retried with larger claim maps. Every claim map is sized from max_states or
// max_events (persistent.cu reuse_map, SessionState), so the retry for kCanonicalMapFull must grow
// both; a knob that sized nothing was doubled here instead, and the retry ran at the same size.
TEST(CapacityOverflow, AFullClaimMapIsRetriedLarger) {
    hg_gpu::EngineConfig cfg;
    const uint32_t states = cfg.max_states, events = cfg.max_events;
    ASSERT_TRUE(hg_gpu::grow_config_for(cfg, hg_gpu::ErrorKind::kCanonicalMapFull));
    EXPECT_EQ(cfg.max_states, 2 * states);
    EXPECT_EQ(cfg.max_events, 2 * events);
}

// estimated_device_bytes covers what an engine allocates, building it and running a quotient
// replay, in total and per replay group. Grow-and-retry stops on the estimate, so an estimate
// below the allocation lets it build an engine that does not fit.
TEST(CapacityOverflow, TheEstimateCoversTheAllocation) {
    hg_gpu::EvolveInput in = growing_input(3);
    in.explore_from_canonical_states_only = true;
    in.record = hgcommon::RecordSet{true, true, true};
    auto allocated = [&](const hg_gpu::EngineConfig& cfg) -> int64_t {
        cudaDeviceSynchronize();
        size_t free0 = 0, total = 0, free1 = 0;
        EXPECT_EQ(cudaMemGetInfo(&free0, &total), cudaSuccess);
        hg_gpu::Engine engine(cfg);
        const hg_gpu::EvolveResult r = engine.run(in);
        EXPECT_TRUE(r.warnings.empty());
        cudaDeviceSynchronize();
        EXPECT_EQ(cudaMemGetInfo(&free1, &total), cudaSuccess);
        return static_cast<int64_t>(free0) - static_cast<int64_t>(free1);
    };
    hg_gpu::EngineConfig base = hg_gpu::config_from_input(in);
    base.qe_class_entries = base.qe_instance_entries = base.qe_event_entries =
        base.qe_pair_entries = 1u << 16;
    base.qe_word_entries = 1u << 20;
    // The first engine of the process also reserves the device stack for every resident
    // thread, and the reservation stays for the process; it is added back below.
    (void)allocated(base);
    size_t stack = 0;
    ASSERT_EQ(cudaDeviceGetLimit(&stack, cudaLimitStackSize), cudaSuccess);
    const int64_t reserved = static_cast<int64_t>(stack * hg_gpu::device_resident_threads());
    const int64_t base_raw = allocated(base);
    const int64_t base_bytes = base_raw + reserved;
    const int64_t base_est = static_cast<int64_t>(hg_gpu::estimated_device_bytes(base));
    std::printf("base: allocated %lld MB (stack %lld MB), estimated %lld MB\n",
                (long long)(base_bytes >> 20), (long long)(reserved >> 20),
                (long long)(base_est >> 20));
    EXPECT_LE(base_bytes, base_est);

    // Every sized field. The replay groups are given explicitly above, so growing max_events or
    // max_states moves only the engine's own tables.
    struct Group { const char* name; uint32_t hg_gpu::EngineConfig::*field; };
#define HG_FIELD(f) {#f, &hg_gpu::EngineConfig::f}
    const Group groups[] = {HG_FIELD(qe_class_entries), HG_FIELD(qe_instance_entries),
                            HG_FIELD(qe_event_entries), HG_FIELD(qe_pair_entries),
                            HG_FIELD(qe_word_entries), HG_FIELD(max_edges),
                            HG_FIELD(max_vertices), HG_FIELD(max_vertex_slots),
                            HG_FIELD(max_states), HG_FIELD(max_state_edge_total),
                            HG_FIELD(inverted_pool), HG_FIELD(sig_index_pool),
                            HG_FIELD(canonical_form_words), HG_FIELD(match_dedup_slots),
                            HG_FIELD(event_canon_slots), HG_FIELD(max_events),
                            HG_FIELD(max_causal_edges), HG_FIELD(max_branchial_edges),
                            HG_FIELD(causal_triple_slots), HG_FIELD(causal_pair_slots),
                            HG_FIELD(branchial_pair_slots), HG_FIELD(edge_consumer_nodes),
                            HG_FIELD(branchial_index_nodes), HG_FIELD(tr_preds_nodes)};
#undef HG_FIELD
    for (const Group& g : groups) {
        hg_gpu::EngineConfig big = base;
        // 2^22 to 2^28 entries more: enough that cudaMalloc's 2 MB page rounding is under 1 B
        // per entry, and few enough that no array passes 4 GiB, past which the driver takes
        // more device memory than the allocation (measured: three 4,311,744,512 B arrays took
        // 16.3 B per entry of free memory for 12 B allocated).
        const uint64_t grow = std::min<uint64_t>(
            std::max<uint64_t>(uint64_t(base.*g.field) * 7u, 1u << 22), 1u << 28);
        big.*g.field = static_cast<uint32_t>((base.*g.field) + grow);
        const int64_t d_bytes = allocated(big) - base_raw;
        const int64_t d_est = static_cast<int64_t>(hg_gpu::estimated_device_bytes(big)) - base_est;
        const double per = double(d_bytes) / double((big.*g.field) - (base.*g.field));
        const double per_est = double(d_est) / double((big.*g.field) - (base.*g.field));
        std::printf("%-22s per entry: allocated %.1f B, estimated %.1f B\n", g.name, per, per_est);
        EXPECT_LE(d_bytes, d_est) << g.name;
    }
}

// A replay group's overflow doubles that group from its resolved size and leaves the others.
TEST(CapacityOverflow, AReplayGroupGrowsAlone) {
    hg_gpu::EngineConfig cfg;
    const hg_gpu::QeEntries before = hg_gpu::qe_entries(cfg);
    ASSERT_TRUE(hg_gpu::grow_config_for(cfg, hg_gpu::ErrorKind::kQeInstancesFull));
    const hg_gpu::QeEntries after = hg_gpu::qe_entries(cfg);
    EXPECT_EQ(after.instances, 2 * before.instances);
    EXPECT_EQ(after.classes, before.classes);
    EXPECT_EQ(after.events, before.events);
    EXPECT_EQ(after.pairs, before.pairs);
    EXPECT_EQ(after.words, before.words);
    ASSERT_TRUE(hg_gpu::grow_config_for(cfg, hg_gpu::ErrorKind::kQeWordsFull));
    EXPECT_EQ(hg_gpu::qe_entries(cfg).words, 2 * before.words);
}

namespace {

bool has_kind(const std::vector<hg_gpu::OverflowWarning>& w, hg_gpu::ErrorKind k) {
    return std::any_of(w.begin(), w.end(), [k](const auto& x) { return x.kind == k; });
}

}  // namespace

// Every record a partial result returns was written by the run that returns it, or is marked: a
// state slot a failed rewrite claimed has id INVALID_ID; every other non-root state is the output
// of exactly one event, every event names states in the result, every state holds edges, and
// every edge record with vertices belongs to a state. Each case starves one pool
// so the run fails a claim after others succeeded, with a capacity the two-edge claims straddle.
// Each engine first completes a smaller run, so a slot the overflowing run claimed and left
// unwritten still holds that run's record.
TEST(CapacityOverflow, APartialResultHoldsNoUnwrittenRecords) {
    struct Starve { const char* name; uint32_t hg_gpu::EngineConfig::*field; uint32_t size; };
    const Starve cases[] = {{"states", &hg_gpu::EngineConfig::max_states, 61},
                            {"events", &hg_gpu::EngineConfig::max_events, 61},
                            {"edges", &hg_gpu::EngineConfig::max_edges, 129},
                            {"vertex slots", &hg_gpu::EngineConfig::max_vertex_slots, 259},
                            {"state edge slots", &hg_gpu::EngineConfig::max_state_edge_total, 501}};
    for (const Starve& c : cases) {
        const hg_gpu::EvolveInput small = growing_input(2);   // 25 states, 52 edges
        const hg_gpu::EvolveInput big = growing_input(3);     // 145 states, 292 edges
        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(big);
        cfg.*c.field = c.size;
        hg_gpu::Engine engine(cfg);
        const hg_gpu::EvolveResult first = engine.run(small);
        ASSERT_TRUE(first.warnings.empty()) << c.name << ": the first run must complete";
        const hg_gpu::EvolveResult r = engine.run(big);
        ASSERT_FALSE(r.warnings.empty()) << c.name << ": the second run must overflow";

        const size_t n = r.states.size();
        std::vector<uint32_t> produced_by(n, 0);
        for (const auto& e : r.events) {
            ASSERT_NE(e.id, hg_gpu::INVALID_ID) << c.name;
            ASSERT_LT(e.input_state, n) << c.name << ": event " << e.id;
            ASSERT_LT(e.output_state, n) << c.name << ": event " << e.id;
            ++produced_by[e.output_state];
        }
        EXPECT_EQ(produced_by[0], 0u) << c.name << ": the root is some event's output";
        std::vector<char> referenced(r.edge_records.size(), 0);
        for (size_t s = 0; s < n; ++s) {
            // A state slot a failed rewrite claimed reads back with id INVALID_ID.
            if (r.states[s].id == hg_gpu::INVALID_ID) {
                EXPECT_EQ(produced_by[s], 0u) << c.name << ": marked state " << s;
                continue;
            }
            EXPECT_EQ(r.states[s].id, s) << c.name;
            if (s != 0)
                EXPECT_EQ(produced_by[s], 1u) << c.name << ": state " << s << " is the output of "
                                              << produced_by[s] << " events";
            EXPECT_GT(r.states[s].num_edges, 0u) << c.name << ": state " << s << " holds no edges";
            for (const auto eid : r.edge_ids(r.states[s]))
                if (eid < referenced.size()) referenced[eid] = 1;
        }
        for (size_t eid = 0; eid < r.edge_records.size(); ++eid)
            if (!r.edge_vertices(static_cast<hg_gpu::EdgeId>(eid)).empty())
                EXPECT_TRUE(referenced[eid]) << c.name << ": edge " << eid
                                             << " has vertices and belongs to no state";
    }
}

// A full relation dedup map is a capacity overflow the run reports, which names the map so the
// retry grows it. Each map is given 16 slots on a run that records more relations than that: the
// raw-event route (state and event identity None), with a second rule whose matches share consumed edges
// with the first's, so sibling events are branchial pairs.
TEST(CapacityOverflow, AFullRelationMapIsReported) {
    struct Map { const char* name; uint32_t hg_gpu::EngineConfig::*field; hg_gpu::ErrorKind kind; };
    const Map maps[] = {
        {"causal triples", &hg_gpu::EngineConfig::causal_triple_slots,
         hg_gpu::ErrorKind::kCausalTripleMapFull},
        {"causal pairs", &hg_gpu::EngineConfig::causal_pair_slots,
         hg_gpu::ErrorKind::kCausalPairMapFull},
        {"branchial pairs", &hg_gpu::EngineConfig::branchial_pair_slots,
         hg_gpu::ErrorKind::kBranchialMapFull}};
    for (const Map& m : maps) {
        hg_gpu::EvolveInput in = growing_input(3);
        hg_gpu::RewriteRule path;
        path.lhs = {{0, 1}, {1, 2}};
        path.rhs = {{0, 1}, {1, 2}, {1, 3}};
        path.num_lhs_vars = 3;
        path.num_rhs_vars = 4;
        in.rules.push_back(path);
        in.canonicalization = hg_gpu::CanonicalizationMode::None;
        in.event_canonicalization = hg_gpu::EventCanonicalizationMode::None;
        in.transitive_reduction = true;
        in.record = hgcommon::RecordSet{true, true, true};
        hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
        cfg.*m.field = 16;
        hg_gpu::Engine engine(cfg);
        const hg_gpu::EvolveResult r = engine.run(in);
        EXPECT_TRUE(has_kind(r.warnings, m.kind)) << m.name << ": the map filled and the run "
                                                  << "reported " << r.warnings.size()
                                                  << " warnings, none naming it";
        hg_gpu::EngineConfig grown = cfg;
        ASSERT_TRUE(hg_gpu::grow_config_for(grown, m.kind)) << m.name;
        EXPECT_EQ(grown.*m.field, 32u) << m.name;
    }
}

// The pool sizes are computed in 64 bits and clamped. A 32,768-edge root at three steps asks for
// 2^30 edges and four times that many vertex slots, past 32 bits.
TEST(CapacityOverflow, ConfigSizingDoesNotWrap) {
    hg_gpu::EvolveInput in = growing_input(3);
    in.initial_state.clear();
    for (hg_gpu::VertexId v = 0; v < 32768; ++v) in.initial_state.push_back({v, v + 1});
    const hg_gpu::EngineConfig cfg = hg_gpu::config_from_input(in);
    EXPECT_GE(cfg.max_edges, 32768u);
    EXPECT_GE(cfg.max_vertex_slots, cfg.max_edges);
    EXPECT_GE(cfg.inverted_pool, cfg.max_edges);
    EXPECT_GE(cfg.sig_index_pool, cfg.max_edges);
    EXPECT_GE(cfg.edge_consumer_nodes, cfg.max_edges);
    EXPECT_GE(cfg.max_states, 1u << 17);
}

namespace {
__global__ void k_arena_claims(hg_gpu::DeviceArena::View a, uint64_t first, uint64_t second,
                               uint32_t* got) {
    got[0] = a.claim(first) != nullptr ? 1u : 0u;
    got[1] = a.claim(second) != nullptr ? 1u : 0u;
}
}  // namespace

// A refused arena claim takes nothing: a later claim that fits is granted.
TEST(CapacityOverflow, ARefusedArenaClaimLeavesTheArenaUsable) {
    hg_gpu::DeviceArena arena(1024);
    arena.reset();
    uint32_t* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, sizeof(uint32_t) * 2), cudaSuccess);
    k_arena_claims<<<1, 1>>>(arena.view(), 4096, 16, d);
    uint32_t got[2] = {9, 9};
    ASSERT_EQ(cudaMemcpy(got, d, sizeof(got), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d);
    EXPECT_EQ(got[0], 0u) << "a claim of 4096 words from 1024 was granted";
    EXPECT_EQ(got[1], 1u) << "a claim of 16 words was refused after a refused oversized one";
    EXPECT_EQ(arena.used_words_host(), 16u);
}

// An index insert the initial upload cannot place is a capacity overflow the run reports.
TEST(CapacityOverflow, AFullIndexAtUploadIsReported) {
    hg_gpu::EngineConfig cfg;
    cfg.sig_index_pool = 4;
    cfg.inverted_pool = 4;
    hg_gpu::EngineState engine(cfg);
    engine.set_maintain_indices(true);
    std::vector<std::vector<hg_gpu::VertexId>> edges;
    for (hg_gpu::VertexId v = 0; v < 16; ++v) edges.push_back({v, v + 1});
    hg_gpu::upload_initial_state(engine, edges);
    std::vector<hg_gpu::OverflowWarning> w;
    engine.collect_warnings_into(w, "upload");
    EXPECT_TRUE(has_kind(w, hg_gpu::ErrorKind::kSigIndexNodes));
    EXPECT_TRUE(has_kind(w, hg_gpu::ErrorKind::kInvIndexNodes));
}

// The reachability search masks into its visited table, so the table is a power of two at any
// tr_scratch_scale.
TEST(CapacityOverflow, TheReachabilityTableIsAPowerOfTwo) {
    for (uint32_t scale : {1u, 3u, 5u, 6u}) {
        hg_gpu::EngineConfig cfg;
        cfg.tr_scratch_scale = scale;
        hg_gpu::EngineState engine(cfg);
        const uint32_t v = engine.device().tr_scratch_visited;
        EXPECT_EQ(v & (v - 1u), 0u) << "scale " << scale << ": visited table of " << v;
        EXPECT_GE(v, hg_gpu::EngineState::kTrScratchVisited * scale) << "scale " << scale;
    }
}

