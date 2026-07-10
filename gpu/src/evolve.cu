#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/wl_hash.hpp"
#include "hg_gpu/evolve.hpp"
#include "hg_gpu/hash_table.hpp"
#include "hg_gpu/initial_upload.hpp"
#include "hg_gpu/ir_canon.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/rewrite.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace hg_gpu {

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hg_gpu::evolve ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

}  // namespace (close anon — config_from_input has external linkage)

EngineConfig config_from_input(const EvolveInput& in) {
    EngineConfig cfg;
    size_t n_init   = in.initial_state.size();
    uint32_t steps  = in.num_steps;

    // Estimate growth per step. A typical Wolfram-style rule produces 2–4
    // new edges per match; matches grow ~linearly with edge count; states
    // grow by a branching factor.
    uint32_t growth = 1u;
    for (uint32_t s = 0; s < steps && growth < 32u; ++s) growth *= 4u;

    // Raw (pre-dedup) state production in a single step can blow past
    // canonical final-step counts by 10× due to within-step branching
    // before dedup collapses isomorphic states. CSR per-state edge lists
    // (Stream 2) replaced the old O(max_states × max_edges) bitset, so
    // sizing is now linear in total edge-slots, not quadratic — we can
    // afford much larger max_states and max_edges without the bitset
    // memory explosion that used to force aggressive downsizing.
    uint32_t expected_edges  = std::max<uint32_t>(1u << 20, static_cast<uint32_t>(n_init) * growth * 512u);
    uint32_t expected_states = std::max<uint32_t>(1u << 17, static_cast<uint32_t>(n_init) * growth * 32u);

    cfg.max_edges              = expected_edges;
    cfg.max_states             = expected_states;
    cfg.max_vertex_slots       = expected_edges * 4u;
    // Total edge-ID slots across all states' CSR rows. Each rewrite
    // consumes parent.count + rhs slots; assume average state size ~
    // max(n_init, 64) and room for ~16 edges per state on average.
    uint32_t avg_state_edges   = std::max<uint32_t>(64u,
                                 static_cast<uint32_t>(n_init) + growth * 16u);
    cfg.max_state_edge_total   = static_cast<uint32_t>(
        std::min<uint64_t>(
            static_cast<uint64_t>(expected_states) * avg_state_edges,
            1ull << 30));  // ≤ 4 GB × sizeof(EdgeId)=4 → ≤ 1G slots
    // Each event allocates ≤ kMaxVars fresh vertices, so vertex IDs bound
    // by n_init-vertices + events × kMaxVars. Be generous.
    cfg.max_vertices           = std::max<uint32_t>(expected_edges,
                                 static_cast<uint32_t>(n_init) * 4u + expected_states * 4u);
    cfg.sig_index_buckets      = 1024;
    cfg.sig_index_pool         = expected_edges * 2u;
    cfg.inverted_pool          = expected_edges * 4u;

    if (in.slice_scan_max_edges) cfg.slice_scan_max_edges = in.slice_scan_max_edges;
    if (in.max_blocks_per_launch) cfg.max_blocks_per_launch = in.max_blocks_per_launch;

    uint32_t expected_events   = expected_states;
    cfg.max_events             = expected_events;
    cfg.max_causal_edges       = expected_events * 8u;
    cfg.max_branchial_edges    = expected_events * 8u;
    cfg.causal_triple_slots    = expected_events * 16u;
    cfg.causal_pair_slots      = expected_events * 8u;
    cfg.branchial_pair_slots   = expected_events * 16u;
    cfg.edge_consumer_nodes    = expected_edges * 4u;
    cfg.branchial_index_buckets = 1u << 20;
    cfg.branchial_index_nodes   = expected_events * 4u;
    cfg.tr_desc_nodes          = expected_events * 16u;
    cfg.tr_anc_nodes           = expected_events * 16u;
    cfg.tr_desc_slots          = expected_events * 16u;
    cfg.tr_anc_slots           = expected_events * 16u;
    return cfg;
}

namespace {  // re-open anon namespace for kernel + helper definitions

// Device-side dedup kernel: for each new state in [lo, hi), look up its WL
// hash in canonical_state_map. First-writer wins → that state's id is
// appended to out_ids via an atomic counter. Duplicates are silently
// dropped. Reuses the same ConcurrentMap primitive already in EngineState
// but allocated transiently here.
using DedupMap = ConcurrentMap<uint64_t, uint32_t>;

// splitmix64 — deterministic, header-quality scalar hash. Used to derive
// a per-(seed, step, sid) coin-flip value for stochastic exploration
// pruning. Cheap (~1 ns) and avoids needing a curand state per thread.
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Seed the initial roots. Every root's hash is inserted into the canonical map so
// that later children isomorphic to a root deduplicate against it. All roots are
// appended to the frontier, unless quotient_roots is set, in which case only the
// first of each canonical class is (isomorphic roots collapse). Default (false)
// matches the reference MultiwaySystem semantics: provided roots are distinct
// entry points even when isomorphic.
__global__ void k_seed_roots(uint32_t num_roots, const uint64_t* hashes,
                             DedupMap::DeviceView map, bool quotient_roots,
                             StateId* out_ids, uint32_t* out_count, uint32_t out_cap) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_roots) return;
    uint64_t h = hashes[tid]; if (h == 0) h = 1;
    auto r = map.insert_if_absent(h, tid);
    if (quotient_roots && !r.inserted) return;
    uint32_t pos = atomicAdd(out_count, 1u);
    if (pos < out_cap) out_ids[pos] = tid;
}

// `dedup` selects the exploration semantics. True: only the first state of each
// canonical hash enters the frontier (explore_from_canonical_states_only). False:
// every new state is explored, so `map` is unused and must not be consulted --
// deduplicating against a scratch map would silently drop states on collision.
__global__ void k_dedup_and_append(uint32_t lo, uint32_t hi,
                                   const uint64_t* hashes,
                                   DedupMap::DeviceView map,
                                   bool dedup,
                                   StateId* out_ids, uint32_t* out_count,
                                   uint32_t out_cap,
                                   uint32_t explore_threshold_u32,
                                   uint64_t explore_seed,
                                   uint32_t step) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = hi - lo;
    if (tid >= n) return;
    StateId sid = lo + tid;
    if (dedup) {
        uint64_t h = hashes[tid];
        auto r = map.insert_if_absent(h == 0 ? 1 : h, sid);
        if (!r.inserted) return;
    }

    // Stochastic-exploration coin flip. UINT32_MAX == "always explore"
    // (the threshold encoding for probability 1.0); skip the hash work
    // entirely on that fast path so the existing all-deterministic
    // workloads pay zero overhead.
    if (explore_threshold_u32 != 0xFFFFFFFFu) {
        if (explore_threshold_u32 == 0u) return;  // probability 0.0
        uint64_t mix = splitmix64(explore_seed
                                  ^ (static_cast<uint64_t>(step) << 32)
                                  ^ static_cast<uint64_t>(sid));
        uint32_t draw = static_cast<uint32_t>(mix);
        if (draw >= explore_threshold_u32) return;
    }

    uint32_t pos = atomicAdd(out_count, 1u);
    if (pos < out_cap) out_ids[pos] = sid;
}

}  // namespace

// ---------------------------------------------------------------------------
// Engine::Impl
//
// Holds every device-side resource that's reusable across run() calls. The
// per-input data (initial state, rules, frontier seeding) is uploaded /
// reset on each run() — pools, indices, and lock-free lists are NOT
// reallocated.
//
// On overflow, the underlying error channel throws via
// engine_state.throw_on_errors with a specific pool name; caller can
// destruct the Engine and construct a new one with a larger config.
// (Auto-grow-on-overflow is Stream 5.)
// ---------------------------------------------------------------------------
struct Engine::Impl {
    explicit Impl(EngineConfig cfg)
        : cfg_(cfg)
        , state_(cfg)
        , matches_(cfg.max_states * 8u)
        , canonical_map_(cfg.max_states * 4u)
    {
        cudaDeviceSetLimit(cudaLimitStackSize, 32u * 1024u);
        check(cudaMalloc(&d_frontier_,      sizeof(StateId) * cfg.max_states), "d_frontier");
        check(cudaMalloc(&d_next_frontier_, sizeof(StateId) * cfg.max_states), "d_next_frontier");
        check(cudaMalloc(&d_next_count_,    sizeof(uint32_t)),                 "d_next_count");
        check(cudaMalloc(&d_state_hashes_,  sizeof(uint64_t) * cfg.max_states),"d_state_hashes");
    }

    ~Impl() {
        if (d_rules_)         cudaFree(d_rules_);
        if (d_frontier_)      cudaFree(d_frontier_);
        if (d_next_frontier_) cudaFree(d_next_frontier_);
        if (d_next_count_)    cudaFree(d_next_count_);
        if (d_state_hashes_)  cudaFree(d_state_hashes_);
    }

    void reset() {
        state_.clear();
        matches_.reset();
        canonical_map_.clear();
    }

    EvolveResult run(const EvolveInput& in);

    EngineConfig                       cfg_;
    EngineState                        state_;
    Pool<MatchRecord>                  matches_;
    DedupMap                           canonical_map_;
    StateId*                           d_frontier_       = nullptr;
    StateId*                           d_next_frontier_  = nullptr;
    uint32_t*                          d_next_count_     = nullptr;
    uint64_t*                          d_state_hashes_   = nullptr;
    DeviceRule*                        d_rules_          = nullptr;
    uint32_t                           d_rules_capacity_ = 0;
};

Engine::Engine(EngineConfig cfg) : impl_(new Impl(cfg)) {}
Engine::~Engine() { delete impl_; }
void Engine::reset() { impl_->reset(); }
const EngineConfig& Engine::config() const { return impl_->cfg_; }
EvolveResult Engine::run(const EvolveInput& in) { return impl_->run(in); }

namespace {
// Fill each state's dedup key with a unique value so NONE of them merge —
// the GPU equivalent of the CPU's None mode (map_key = state id, no iso-dedup).
__global__ void k_fill_unique_keys(uint32_t lo, uint32_t hi, uint64_t* out) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = hi - lo;
    if (tid >= n) return;
    out[tid] = static_cast<uint64_t>(lo + tid) + 1u;  // unique, nonzero
}

// Per-state dedup keys by canonicalization mode, mirroring the CPU's
// create_or_get_canonical_state map_key: None -> unique (tree mode, no dedup),
// Full -> exact IR canonical hash; Automatic -> content-ordered (non-iso) hash.
void compute_state_dedup_keys(EngineState& engine, uint32_t lo, uint32_t hi,
                              uint64_t* out, CanonicalizationMode mode) {
    if (hi <= lo) return;
    if (mode == CanonicalizationMode::None) {
        uint32_t n = hi - lo;
        int b = 64, g = static_cast<int>((n + b - 1) / b);
        k_fill_unique_keys<<<g, b>>>(lo, hi, out);
        cudaDeviceSynchronize();
        return;
    }
    if (mode == CanonicalizationMode::Automatic) {
        compute_state_content_hashes_range(engine, lo, hi, out);
        return;
    }
    compute_state_ir_hashes_range(engine, lo, hi, out);
}
}  // namespace

EvolveResult Engine::Impl::run(const EvolveInput& in) {
    // Reset device state from any prior run().
    reset();

    EvolveResult out;
    if (in.rules.empty() && in.num_steps == 0 && in.initial_state.empty()) {
        return out;
    }

    auto t_total_start = std::chrono::steady_clock::now();
    auto t_init_start = std::chrono::steady_clock::now();
    EngineState& engine = state_;
    engine.set_tr_enabled(in.transitive_reduction);
    double t_init = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_init_start).count();

    // Lazy index maintenance: skip index inserts until some state exceeds the
    // slice-scan threshold (the match kernels never read the indices below it).
    // Normalize to a list of initial states (plural takes precedence).
    const std::vector<std::vector<std::vector<VertexId>>> roots =
        !in.initial_states.empty()
            ? in.initial_states
            : std::vector<std::vector<std::vector<VertexId>>>{in.initial_state};
    size_t max_root_edges = 0;
    for (const auto& r : roots) max_root_edges = std::max(max_root_edges, r.size());
    engine.set_maintain_indices(max_root_edges > engine.config_slice_scan_max_edges());
    const uint32_t num_roots = upload_initial_states(engine, roots);

    // Upload rules. Resize the device-side rules buffer if this run has more
    // rules than any prior run. Re-upload every run so the caller can pass a
    // different rule set per call without surprises.
    std::vector<DeviceRule> rules;
    rules.reserve(in.rules.size());
    for (const auto& r : in.rules) rules.push_back(make_device_rule(r));
    const uint32_t num_rules = static_cast<uint32_t>(rules.size());

    if (num_rules > d_rules_capacity_) {
        if (d_rules_) cudaFree(d_rules_);
        d_rules_ = nullptr;
        d_rules_capacity_ = 0;
        if (num_rules > 0) {
            check(cudaMalloc(&d_rules_, sizeof(DeviceRule) * num_rules), "d_rules alloc");
            d_rules_capacity_ = num_rules;
        }
    }
    if (num_rules > 0) {
        check(cudaMemcpy(d_rules_, rules.data(), sizeof(DeviceRule) * num_rules,
                         cudaMemcpyHostToDevice), "d_rules copy");
    }
    DeviceRule* d_rules = d_rules_;

    const EngineConfig& cfg = engine.config();
    Pool<MatchRecord>& matches = matches_;

    // Per-run aliases for the persistent device buffers.
    StateId*  d_frontier      = d_frontier_;
    StateId*  d_next_frontier = d_next_frontier_;
    uint32_t* d_next_count    = d_next_count_;
    uint64_t* d_state_hashes  = d_state_hashes_;

    // Canonical-state dedup map: hash → first-seen StateId. Cleared in reset().
    DedupMap& canonical_map = canonical_map_;

    // Resolve exploration-probability parameters once per run.
    //   threshold == UINT32_MAX → fast path: always explore (zero overhead).
    //   threshold == 0          → never expand any new state.
    //   else                    → admit with probability ≈ threshold / 2^32.
    float clamped_p = in.exploration_probability;
    if (!(clamped_p > 0.0f)) clamped_p = 0.0f;
    if (clamped_p > 1.0f)    clamped_p = 1.0f;
    uint32_t explore_threshold_u32;
    if (clamped_p >= 1.0f) {
        explore_threshold_u32 = 0xFFFFFFFFu;
    } else if (clamped_p <= 0.0f) {
        explore_threshold_u32 = 0u;
    } else {
        explore_threshold_u32 = static_cast<uint32_t>(
            static_cast<double>(clamped_p) * 4294967296.0);
    }
    uint64_t resolved_seed = in.exploration_seed;
    if (resolved_seed == 0 && clamped_p < 1.0f) {
        std::random_device rd;
        resolved_seed = (static_cast<uint64_t>(rd()) << 32) | rd();
        if (resolved_seed == 0) resolved_seed = 0xA5A5A5A5A5A5A5A5ull;
    }

    // Seed the frontier with the initial (root) states 0..num_roots-1. Roots are
    // never coin-flipped — they always enter the frontier so step 0 has work.
    compute_state_dedup_keys(engine, 0, num_roots, d_state_hashes, in.canonicalization);
    if (in.explore_from_canonical_states_only) {
        uint32_t zero32c = 0;
        check(cudaMemcpy(d_next_count, &zero32c, sizeof(uint32_t), cudaMemcpyHostToDevice),
              "seed count");
        // Insert every root into the canonical map (children dedup against roots)
        // and seed the frontier; roots collapse only when quotient_initial_states.
        int b = 64, g = static_cast<int>((num_roots + b - 1) / b);
        k_seed_roots<<<g, b>>>(num_roots, d_state_hashes, canonical_map.view(),
                               in.quotient_initial_states,
                               d_frontier, d_next_count, cfg.max_states);
        check(cudaDeviceSynchronize(), "seed roots sync");
    } else {
        std::vector<StateId> root_ids(num_roots);
        for (uint32_t i = 0; i < num_roots; ++i) root_ids[i] = i;
        check(cudaMemcpy(d_frontier, root_ids.data(), sizeof(StateId) * num_roots,
                         cudaMemcpyHostToDevice),
              "seed frontier");
    }
    // Initial frontier size: unique roots after dedup (explore-from-canonical), or
    // all roots (full multiway).
    uint32_t frontier_count = 0;
    if (num_roots > 0) {
        if (in.explore_from_canonical_states_only) {
            check(cudaMemcpy(&frontier_count, d_next_count, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost), "read seed count");
        } else {
            frontier_count = num_roots;
        }
    }

    const bool dbg = std::getenv("HG_GPU_DBG_TIME") != nullptr;
    double t_match = 0, t_rewrite = 0, t_hash = 0, t_dedup = 0;

    // Cache the running state_count on host so we only D2H once per step
    // (instead of twice via num_states_host around the rewrite call).
    uint32_t state_count_host = engine.num_states_host();

    // Tag warnings with a step-aware context, e.g. "match kernel step 3".
    // Reused across all four phases per step.
    char ctx_buf[64];

    for (uint32_t step = 0; step < in.num_steps && frontier_count > 0; ++step) {
        auto t0 = std::chrono::steady_clock::now();
        if (dbg) std::fprintf(stderr, "[step %u] frontier=%u state_count=%u\n",
                              step, frontier_count, state_count_host);
        // (1) Match all (frontier, rule) pairs in one kernel.
        matches.reset();
        run_match_kernel_batch_nosync(engine, d_rules, num_rules,
                                      d_frontier, frontier_count, matches);
        std::snprintf(ctx_buf, sizeof(ctx_buf), "match kernel step %u", step);
        engine.collect_warnings_into(out.warnings, ctx_buf);
        auto t1 = std::chrono::steady_clock::now();
        t_match += std::chrono::duration<double, std::milli>(t1 - t0).count();

        uint32_t nm = matches.size_host();
        if (nm == 0) break;

        // (2) Rewrite all matches in one kernel. The rewrite kernel uses
        // a CAS-loop on state_count so it never bumps past max_states;
        // any overflow (state/event/edge/vertex pool) is recorded as a
        // warning. The kernel still runs on whatever budget remains —
        // partial work is fine, the result still self-consistent.
        uint32_t state_before = state_count_host;
        run_rewrite_kernel_with_nosync(engine, d_rules, matches, nm, step + 1);
        std::snprintf(ctx_buf, sizeof(ctx_buf), "rewrite kernel step %u", step);
        engine.collect_warnings_into(out.warnings, ctx_buf);
        uint32_t state_after = engine.num_states_host();
        state_count_host = state_after;
        auto t2 = std::chrono::steady_clock::now();
        t_rewrite += std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (state_after <= state_before) break;

        // (3) WL-hash only the new states. Writes into d_state_hashes[lo..hi).
        compute_state_dedup_keys(engine, state_before, state_after,
                                 d_state_hashes + state_before, in.canonicalization);
        auto t3 = std::chrono::steady_clock::now();
        t_hash += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // (4) Dedup & build next frontier on device.
        uint32_t zero32 = 0;
        check(cudaMemcpy(d_next_count, &zero32, sizeof(uint32_t), cudaMemcpyHostToDevice),
              "reset next_count");
        uint32_t n_new = state_after - state_before;
        int block = 128;
        int grid  = (int)((n_new + block - 1) / block);

        k_dedup_and_append<<<grid, block>>>(state_before, state_after,
                                            d_state_hashes + state_before,
                                            canonical_map.view(),
                                            in.explore_from_canonical_states_only,
                                            d_next_frontier, d_next_count,
                                            cfg.max_states,
                                            explore_threshold_u32,
                                            resolved_seed, step);
        check(cudaDeviceSynchronize(), "dedup sync");

        check(cudaMemcpy(&frontier_count, d_next_count, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost), "read next_count");

        // First state above the slice-scan threshold: build the indices it will
        // be matched through, then maintain them incrementally.
        if (!engine.maintain_indices() && engine.needs_indices_host()) {
            rebuild_indices(engine, engine.num_edges_host());
            engine.set_maintain_indices(true);
        }
        auto t4 = std::chrono::steady_clock::now();
        t_dedup += std::chrono::duration<double, std::milli>(t4 - t3).count();

        // Swap frontier buffers.
        std::swap(d_frontier, d_next_frontier);
    }

    auto t_readback_start = std::chrono::steady_clock::now();

    // Readback — hashes were persisted across steps, no re-hashing needed.
    uint32_t total_states = engine.num_states_host();
    std::vector<uint64_t> h_hashes(total_states);
    if (total_states > 0) {
        check(cudaMemcpy(h_hashes.data(), d_state_hashes, sizeof(uint64_t) * total_states,
                         cudaMemcpyDeviceToHost), "final hashes d2h");
    }
    double t_readback_hashes = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_readback_start).count();

    auto t_readback_states_start = std::chrono::steady_clock::now();
    auto all_edges = engine.all_state_edges_host();
    out.states.reserve(all_edges.size());
    for (uint32_t s = 0; s < all_edges.size(); ++s) {
        CanonicalState cs;
        cs.id             = s;
        cs.canonical_hash = (s < h_hashes.size()) ? h_hashes[s] : 0;
        cs.edges          = std::move(all_edges[s]);
        out.states.push_back(std::move(cs));
    }

    double t_readback_states = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_readback_states_start).count();

    auto t_readback_evcb_start = std::chrono::steady_clock::now();
    auto d_events = engine.events_host();
    out.events.reserve(d_events.size());
    for (const auto& de : d_events) {
        Event e;
        e.id            = de.id;
        e.canonical_id  = de.canonical_id;
        e.input_state   = de.input_state;
        e.output_state  = de.output_state;
        e.rule          = de.rule;
        e.step          = de.step;
        for (uint8_t i = 0; i < de.num_consumed; ++i) e.consumed_edges.push_back(de.consumed_edges[i]);
        for (uint8_t i = 0; i < de.num_produced; ++i) e.produced_edges.push_back(de.produced_edges[i]);
        out.events.push_back(std::move(e));
    }

    auto d_causal = engine.causal_edges_host();
    out.causal_edges.reserve(d_causal.size());
    for (const auto& c : d_causal) out.causal_edges.push_back(CausalEdge{c.from, c.to});
    auto d_branch = engine.branchial_edges_host();
    out.branchial_edges.reserve(d_branch.size());
    for (const auto& b : d_branch) out.branchial_edges.push_back(BranchialEdge{b.a, b.b});

    double t_readback_evcb = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_readback_evcb_start).count();

    // Note: Impl-owned device buffers (d_rules_, d_frontier_, etc) are NOT
    // freed here — they live for the Engine's lifetime and are reused on
    // subsequent run() calls. Impl's destructor handles cleanup.

    double t_total = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_total_start).count();

    if (dbg) {
        std::fprintf(stderr,
            "[evolve dbg] total=%.2f init=%.2f match=%.2f rewrite=%.2f "
            "hash=%.2f dedup=%.2f readback{hashes=%.2f states=%.2f ev/c/b=%.2f} (ms)\n",
            t_total, t_init, t_match, t_rewrite, t_hash, t_dedup,
            t_readback_hashes, t_readback_states, t_readback_evcb);
    }

    return out;
}

// Map an ErrorKind (the pool that overflowed) to the EngineConfig field(s)
// that govern its capacity, and double them. Some kinds map to multiple
// fields (e.g. kVertexPoolFull involves both max_vertex_slots and max_vertices).
// Returns true if growth was applied; false for kinds that have no
// retryable config (kScratchOverflow is a kernel-internal limit and can't
// be grown by reconfiguring pools).
bool grow_config_for(EngineConfig& cfg, ErrorKind kind) {
    auto dbl = [](uint32_t& f) { f = (f >= (1u << 31)) ? f : (f * 2u); };
    switch (kind) {
        case ErrorKind::kEdgePoolFull:
            dbl(cfg.max_edges);
            dbl(cfg.sig_index_pool);
            dbl(cfg.inverted_pool);
            dbl(cfg.edge_consumer_nodes);
            return true;
        case ErrorKind::kStatePoolFull:
            dbl(cfg.max_states);
            dbl(cfg.max_state_edge_total);
            dbl(cfg.canonical_map_slots);
            return true;
        case ErrorKind::kEventPoolFull:
            dbl(cfg.max_events);
            dbl(cfg.tr_desc_nodes);
            dbl(cfg.tr_anc_nodes);
            return true;
        case ErrorKind::kVertexPoolFull:
            dbl(cfg.max_vertex_slots);
            dbl(cfg.max_vertices);
            dbl(cfg.inverted_pool);
            return true;
        case ErrorKind::kCausalPoolFull:
            dbl(cfg.max_causal_edges);
            return true;
        case ErrorKind::kBranchialPoolFull:
            dbl(cfg.max_branchial_edges);
            return true;
        case ErrorKind::kMatchPoolFull:
            // Match pool sized as cfg.max_states * 8 in Engine::Impl ctor;
            // bumping max_states grows both. Edge growth also helps because
            // each match uses bounded RHS edges.
            dbl(cfg.max_states);
            return true;
        case ErrorKind::kCausalTripleMapFull: dbl(cfg.causal_triple_slots);  return true;
        case ErrorKind::kCausalPairMapFull:   dbl(cfg.causal_pair_slots);    return true;
        case ErrorKind::kBranchialMapFull:    dbl(cfg.branchial_pair_slots); return true;
        case ErrorKind::kDescSetFull:         dbl(cfg.tr_desc_slots);        return true;
        case ErrorKind::kAncSetFull:          dbl(cfg.tr_anc_slots);         return true;
        case ErrorKind::kEdgeConsumerNodes:   dbl(cfg.edge_consumer_nodes);  return true;
        case ErrorKind::kBranchialIndexNodes: dbl(cfg.branchial_index_nodes); return true;
        case ErrorKind::kDescListNodes:       dbl(cfg.tr_desc_nodes);        return true;
        case ErrorKind::kAncListNodes:        dbl(cfg.tr_anc_nodes);         return true;
        case ErrorKind::kSigIndexNodes:       dbl(cfg.sig_index_pool);       return true;
        case ErrorKind::kInvIndexNodes:       dbl(cfg.inverted_pool);        return true;
        case ErrorKind::kFrontierCapFull:     dbl(cfg.max_states);           return true;
        case ErrorKind::kScratchOverflow:
            // Kernel-internal local-memory bound; not config-controlled.
            // Cannot retry — caller must accept the soft accuracy
            // degradation (1-WL fallback) or upgrade to global-memory IR.
            return false;
        default: return false;
    }
}

// One-shot wrapper. Builds an Engine sized for `in`, runs it. If the
// kernel reports any retryable overflow warnings, doubles the relevant
// EngineConfig field(s) and re-runs from scratch — up to kMaxRetries
// times (64× capacity growth ceiling). Each retry destructs and
// reconstructs the Engine (the pools have to be re-allocated at the new
// sizes; preserving in-flight state across reallocs is more engineering
// for negligible benefit on the cold-start path).
//
// The returned result accumulates warnings across all attempts — the
// caller sees the cumulative trail, not just the last attempt's warnings.
// If the final attempt still produces warnings (because we hit the retry
// ceiling, or because some warnings are non-retryable like
// kScratchOverflow), the partial result is still returned with all
// warnings attached. The caller decides whether the partial result is
// good enough.
//
// For repeated runs of the same workload prefer `Engine(cfg).run(in)`
// directly with a config you've already validated, so the retry loop
// only fires on the first call.
// Log the EngineConfig that worked (after grow-and-retry) so the user can
// pre-size on subsequent calls. Only the fields that were grown beyond
// their initial value are printed — keeps the message focused. Format is
// stable so callers can grep it; M9 (auto-tune cache) will eventually
// persist this same set of values to disk.
static void log_winning_config(const EngineConfig& initial,
                               const EngineConfig& winning) {
#define LOG_FIELD(field) \
    if (winning.field != initial.field) { \
        std::fprintf(stderr, "  %s: %u → %u\n", #field, initial.field, winning.field); \
    }
    std::fprintf(stderr,
        "hg_gpu::evolve: succeeded after grow-and-retry; pass these to "
        "Engine(cfg) directly to skip the retry loop next time:\n");
    LOG_FIELD(max_edges);
    LOG_FIELD(max_vertices);
    LOG_FIELD(max_vertex_slots);
    LOG_FIELD(max_states);
    LOG_FIELD(max_state_edge_total);
    LOG_FIELD(sig_index_pool);
    LOG_FIELD(inverted_pool);
    LOG_FIELD(canonical_map_slots);
    LOG_FIELD(max_events);
    LOG_FIELD(max_causal_edges);
    LOG_FIELD(max_branchial_edges);
    LOG_FIELD(causal_triple_slots);
    LOG_FIELD(causal_pair_slots);
    LOG_FIELD(branchial_pair_slots);
    LOG_FIELD(edge_consumer_nodes);
    LOG_FIELD(branchial_index_buckets);
    LOG_FIELD(branchial_index_nodes);
    LOG_FIELD(tr_desc_nodes);
    LOG_FIELD(tr_anc_nodes);
    LOG_FIELD(tr_desc_slots);
    LOG_FIELD(tr_anc_slots);
#undef LOG_FIELD
}

// Scale a config's growable capacity fields down proportionally so its estimated
// footprint fits within `cap` bytes, leaving a floor under each so a minimal run
// is still possible. Bucket counts (power-of-two) and fixed control words are
// left alone. A run under the shrunk config that needs more will overflow and
// return a partial result, which is the intended "constrain to memory" behaviour.
void fit_config_to_cap(EngineConfig& cfg, uint64_t cap) {
    uint64_t est = estimated_device_bytes(cfg);
    if (cap == 0 || est <= cap) return;
    double r = static_cast<double>(cap) / static_cast<double>(est);
    auto sc = [&](uint32_t& f, uint32_t floor) {
        uint64_t v = static_cast<uint64_t>(static_cast<double>(f) * r);
        f = static_cast<uint32_t>(v < floor ? floor : v);
    };
    sc(cfg.max_edges, 1u<<12);            sc(cfg.max_vertices, 1u<<12);
    sc(cfg.max_vertex_slots, 1u<<14);     sc(cfg.max_states, 1u<<10);
    sc(cfg.max_state_edge_total, 1u<<16); sc(cfg.sig_index_pool, 1u<<12);
    sc(cfg.inverted_pool, 1u<<12);        sc(cfg.max_events, 1u<<10);
    sc(cfg.max_causal_edges, 1u<<12);     sc(cfg.max_branchial_edges, 1u<<12);
    sc(cfg.causal_triple_slots, 1u<<12);  sc(cfg.causal_pair_slots, 1u<<12);
    sc(cfg.branchial_pair_slots, 1u<<12); sc(cfg.edge_consumer_nodes, 1u<<12);
    sc(cfg.branchial_index_nodes, 1u<<12);sc(cfg.tr_desc_nodes, 1u<<12);
    sc(cfg.tr_anc_nodes, 1u<<12);         sc(cfg.tr_desc_slots, 1u<<12);
    sc(cfg.tr_anc_slots, 1u<<12);         sc(cfg.canonical_map_slots, 1u<<12);
}

uint64_t estimated_device_bytes(const EngineConfig& cfg) {
    // Sum the pools EngineState allocates. Element sizes: Edge 24; DeviceEvent
    // ~160; DeviceCausal/Branchial edge 12; StateEdgeSlice 8; a LockFreeList node
    // is sizeof(value)+4 rounded up; a ConcurrentMap slot is sizeof(K)+sizeof(V).
    // A 4-byte id is the unit for most index/id pools. Approximate — a 15%
    // headroom covers the small frontier/hash scratch buffers and allocation
    // granularity, so the estimate errs high (refusing borderline growth).
    auto u64 = [](uint32_t v) { return static_cast<uint64_t>(v); };
    uint64_t b = 0;
    b += u64(cfg.max_vertex_slots)    * 4;          // vertex_pool
    b += u64(cfg.max_edges)           * 24;         // edge_pool (Edge)
    b += u64(cfg.max_edges)           * 4;          // edge_producer
    b += u64(cfg.max_states)          * 8;          // state_edge_slices
    b += u64(cfg.max_state_edge_total)* 4;          // state_edge_ids
    b += u64(cfg.sig_index_buckets)   * 4 + u64(cfg.sig_index_pool) * 8;   // signature index
    b += u64(cfg.max_vertices)        * 4 + u64(cfg.inverted_pool)  * 8;   // vertex inverted index
    b += u64(cfg.max_events)          * 160;        // event_pool (DeviceEvent)
    b += u64(cfg.max_causal_edges)    * 12;         // causal_edge_pool
    b += u64(cfg.max_branchial_edges) * 12;         // branchial_edge_pool
    b += u64(cfg.max_edges)           * 4 + u64(cfg.edge_consumer_nodes)   * 8;   // edge_consumers
    b += u64(cfg.branchial_index_buckets) * 4 + u64(cfg.branchial_index_nodes) * 16; // branchial index
    b += u64(cfg.causal_triple_slots) * 12;
    b += u64(cfg.causal_pair_slots)   * 12;
    b += u64(cfg.branchial_pair_slots)* 12;
    b += u64(cfg.max_events)          * 8  + u64(cfg.tr_desc_nodes) * 8;   // desc_list
    b += u64(cfg.max_events)          * 8  + u64(cfg.tr_anc_nodes)  * 8;   // anc_list
    b += u64(cfg.tr_desc_slots)       * 12;
    b += u64(cfg.tr_anc_slots)        * 12;
    b += u64(cfg.canonical_map_slots) * 12;         // canonical dedup map
    b += u64(cfg.match_dedup_slots)   * 12 + u64(cfg.event_canon_slots) * 12;
    b += u64(cfg.max_states)          * 8 * 76;     // matches pool (max_states*8 records ~76B)
    b += u64(cfg.max_states)          * 16;         // d_frontier + d_next_frontier + d_state_hashes
    return b + b / 6;   // ~17% headroom
}

EvolveResult evolve(const EvolveInput& in) {
    constexpr int kMaxRetries = 6;  // up to 64× capacity growth
    EngineConfig initial_cfg = config_from_input(in);
    EngineConfig cfg = initial_cfg;
    std::vector<OverflowWarning> trail;

    // Resolve the device-memory ceiling: explicit request, else 90% of total VRAM.
    uint64_t mem_cap = in.max_device_memory_bytes;
    if (mem_cap == 0) {
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
            mem_cap = static_cast<uint64_t>(static_cast<double>(totalB) * 0.90);
        }
        cudaGetLastError();  // clear any sticky status from the query
    }
    // Shrink the initial config to the ceiling if config_from_input over-provisioned
    // past it; the loop below then never grows back over the cap.
    if (mem_cap != 0 && estimated_device_bytes(cfg) > mem_cap) {
        fit_config_to_cap(cfg, mem_cap);
        initial_cfg = cfg;
        std::fprintf(stderr,
            "hg_gpu::evolve: initial config exceeded the memory cap (%llu MB) — "
            "scaled pools down to ~%llu MB; result may be partial.\n",
            (unsigned long long)(mem_cap >> 20),
            (unsigned long long)(estimated_device_bytes(cfg) >> 20));
    }

    // Best partial result seen so far: an attempt that overflowed still returns
    // whatever it computed, and if the next, larger engine no longer fits in
    // device memory that partial is what the caller gets, never an exception.
    EvolveResult best;

    for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
        EvolveResult result;
        try {
            Engine engine(cfg);
            result = engine.run(in);
        } catch (const std::exception& e) {
            trail.push_back(OverflowWarning{
                ErrorKind::kDeviceOutOfMemory, 1u,
                std::string("attempt ") + std::to_string(attempt + 1) + ": " + e.what()});
            std::fprintf(stderr,
                "hg_gpu::evolve: engine at the grown size no longer fits in device "
                "memory (%s) — returning the last completed attempt's partial result.\n",
                e.what());
            best.warnings = std::move(trail);
            return best;
        }

        // No overflow this attempt: success — return with the cumulative
        // trail (which is empty on the first-attempt-clean path).
        if (result.warnings.empty() && trail.empty()) {
            return result;
        }
        if (result.warnings.empty()) {
            // Clean run after one or more grow-and-retry rounds. Surface
            // the winning config to the operator and return with the
            // accumulated trail attached.
            log_winning_config(initial_cfg, cfg);
            result.warnings = std::move(trail);
            return result;
        }

        // Grow the config for any retryable warnings observed THIS
        // attempt. grow_config_for is idempotent under repeats and
        // doubling spuriously is conservative, so we just sweep every
        // warning's kind. If no warning is retryable (only
        // kScratchOverflow, say), we can't make progress — return the
        // partial result with the cumulative trail.
        bool any_retryable = false;
        ErrorKind first_grew = ErrorKind::kCount;
        for (const auto& w : result.warnings) {
            if (grow_config_for(cfg, w.kind)) {
                if (!any_retryable) first_grew = w.kind;
                any_retryable = true;
            }
        }

        // Cumulative trail across attempts; user can see what was hit
        // and how many times across the retries.
        for (auto& w : result.warnings) trail.push_back(std::move(w));

        if (!any_retryable || attempt == kMaxRetries) {
            result.warnings = std::move(trail);
            return result;
        }

        // Would the grown config exceed the memory ceiling? If so, stop here and
        // return the best partial rather than pushing toward a real device OOM.
        if (mem_cap != 0 && estimated_device_bytes(cfg) > mem_cap) {
            trail.push_back(OverflowWarning{
                ErrorKind::kDeviceOutOfMemory,
                static_cast<uint32_t>(estimated_device_bytes(cfg) >> 20),
                "grown config (~" + std::to_string(estimated_device_bytes(cfg) >> 20) +
                " MB) would exceed the device memory cap (" +
                std::to_string(mem_cap >> 20) + " MB); returning partial result"});
            std::fprintf(stderr,
                "hg_gpu::evolve: next config (~%llu MB) would exceed the memory cap "
                "(%llu MB) — returning the partial result.\n",
                (unsigned long long)(estimated_device_bytes(cfg) >> 20),
                (unsigned long long)(mem_cap >> 20));
            result.warnings = std::move(trail);
            return result;
        }

        best = std::move(result);

        std::fprintf(stderr,
            "hg_gpu::evolve: overflow on %s — growing relevant config and "
            "retrying (attempt %d/%d).\n",
            error_kind_name(first_grew), attempt + 2, kMaxRetries + 1);
    }
    // Unreachable: loop returns on every path.
    return EvolveResult{};
}

}  // namespace hg_gpu
