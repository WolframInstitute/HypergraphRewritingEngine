#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/persistent.hpp"   // default_persistent_grid
#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/vertex_inverted_index.hpp"

// The HOST bodies of EngineState and of the three device-side containers it owns.
//
// engine_state.hpp is included by sixteen translation units -- every kernel file, every GPU test,
// and six other headers -- and every one of them was compiling all of this: the constructor's
// forty-odd cudaMallocs, clear(), device(), the readback helpers. None of it is device code and
// none of it runs per item; it runs once per engine, per run, or per readback.
//
// What stays in engine_state.hpp is DeviceState and the DeviceView structs, which are what the
// kernels actually use, plus the constexpr stack-size constants a launch reads.

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

// =============================================================================
// DeviceArena
// =============================================================================

DeviceArena::DeviceArena(uint64_t capacity_words) : capacity_(capacity_words) {
    HG_CUDA_CHECK(cudaMalloc(&base_, capacity_ * sizeof(uint32_t)), "arena alloc");
    HG_CUDA_CHECK(cudaMalloc(&cursor_, sizeof(uint64_t)), "arena cursor alloc");
    reset();
}

DeviceArena::~DeviceArena() {
    if (base_)   cudaFree(base_);
    if (cursor_) cudaFree(cursor_);
}

void DeviceArena::reset(ClearBatch* batch) {
    if (batch) batch->add(cursor_, sizeof(uint64_t), 0);
    else HG_CUDA_CHECK(cudaMemset(cursor_, 0, sizeof(uint64_t)), "arena cursor clear");
}

DeviceArena::View DeviceArena::view() { return View{base_, cursor_, capacity_}; }

uint64_t DeviceArena::capacity_words() const { return capacity_; }

uint64_t DeviceArena::used_words_host() const {
    uint64_t v = 0;
    cudaMemcpy(&v, cursor_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return v;
}

// =============================================================================
// VertexInvertedIndex
// =============================================================================

VertexInvertedIndex::VertexInvertedIndex(uint32_t max_vertices, uint32_t pool_capacity)
    : list_(max_vertices, pool_capacity) {}

VertexInvertedIndex::DeviceView VertexInvertedIndex::view() const {
    return DeviceView{list_.view()};
}

uint32_t VertexInvertedIndex::max_vertices() const { return list_.num_keys(); }
uint32_t VertexInvertedIndex::used() const { return list_.pool_used_host(); }

void VertexInvertedIndex::clear(uint32_t used_vertices, ClearBatch* batch) {
    list_.clear(used_vertices, batch);
}

// =============================================================================
// ClearBatch
// =============================================================================

// blockIdx.y is the region; the x blocks stride over its words.
__global__ void k_clear_regions(const ClearBatch::Set set) {
    if (blockIdx.y >= set.n) return;
    const ClearRegion r = set.r[blockIdx.y];
    uint32_t* p = static_cast<uint32_t*>(r.ptr);
    const uint64_t words = r.bytes / 4u;
    for (uint64_t w = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; w < words;
         w += uint64_t(gridDim.x) * blockDim.x)
        p[w] = r.word;
}

void ClearBatch::add(void* ptr, uint64_t bytes, uint8_t fill) {
    if (bytes == 0) return;
    if ((reinterpret_cast<uintptr_t>(ptr) | bytes) & 3u) {
        HG_CUDA_CHECK(cudaMemset(ptr, fill, bytes), "ClearBatch unaligned region");
        return;
    }
    if (set_.n == kMaxRegions) flush();
    set_.r[set_.n++] = ClearRegion{ptr, bytes, 0x01010101u * fill};
}

void ClearBatch::flush() {
    if (set_.n == 0) return;
    uint64_t most = 0;
    for (uint32_t i = 0; i < set_.n; ++i) most = std::max<uint64_t>(most, set_.r[i].bytes / 4u);
    constexpr uint32_t block = 256;
    const uint32_t x = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>((most + block - 1) / block, 1), 1024));
    k_clear_regions<<<dim3(x, set_.n), block>>>(set_);
    HG_CUDA_CHECK(cudaGetLastError(), "ClearBatch launch");
    set_.n = 0;
}

// =============================================================================
// EngineState
// =============================================================================


// The block exists before any pool member's constructor runs -- block_ is declared first --
// so the pools can be constructed straight onto its slots.
EngineState::CounterBlock::CounterBlock(uint32_t slots) {
    HG_CUDA_CHECK(cudaMalloc(&p, sizeof(uint32_t) * slots), "EngineState counter block alloc");
    // Slots 4 and 5 are bound to their pointers on first use of the feature that owns them.
    // Zeroing the whole block here is what makes such a slot read as zero when its feature
    // never runs.
    HG_CUDA_CHECK(cudaMemset(p, 0, sizeof(uint32_t) * slots), "EngineState counter block init");
}
EngineState::CounterBlock::~CounterBlock() { if (p) cudaFree(p); }

EngineState::EngineState(EngineConfig cfg): cfg_(cfg)
        , vertex_pool_(cfg.max_vertex_slots, block_.p + 7)
        , edge_pool_(cfg.max_edges, block_.p + 6)
        , vertex_inverted_index_(cfg.max_vertices, cfg.inverted_pool)
        , event_pool_(cfg.max_events, block_.p + 8)
        , causal_edge_pool_(cfg.max_causal_edges, block_.p + 9)
        , branchial_edge_pool_(cfg.max_branchial_edges, block_.p + 10)
        , edge_consumers_(cfg.max_edges, cfg.edge_consumer_nodes)
        , branchial_index_(cfg.branchial_index_buckets, cfg.branchial_index_nodes)
        , causal_triple_dedup_(cfg.causal_triple_slots)
        , causal_pair_dedup_(cfg.causal_pair_slots)
        , branchial_pair_dedup_(cfg.branchial_pair_slots)
        , preds_list_(cfg.max_events, cfg.tr_preds_nodes) {
        // The stack the engine's kernels need (device_stack_bytes), set here so it holds for
        // every entry point and is reserved before any pool is allocated.
        // Checked, and then READ BACK: a driver may clamp the request rather than refuse it.
        const size_t stack = device_stack_bytes();
        HG_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack), "set device stack size");
        size_t actual_stack = 0;
        HG_CUDA_CHECK(cudaDeviceGetLimit(&actual_stack, cudaLimitStackSize), "read device stack size");
        if (actual_stack < stack) {
            throw std::runtime_error(
                "EngineState: device stack is " + std::to_string(actual_stack) +
                " bytes after requesting " + std::to_string(stack) +
                "; a kernel would overflow it and report an illegal memory access");
        }
        slice_scan_max_edges_ = cfg.slice_scan_max_edges;
        HG_CUDA_CHECK(cudaMalloc(&state_edge_slices_,
              sizeof(StateEdgeSlice) * cfg_.max_states),
              "EngineState state_edge_slices alloc");
        HG_CUDA_CHECK(cudaMalloc(&state_edge_ids_,
              sizeof(EdgeId) * cfg_.max_state_edge_total),
              "EngineState state_edge_ids alloc");
        // state_edge_ids_counter_ is bumped before the capacity check and before the vertex
        // reservations that can still fail, so slots below the counter can be reserved and never
        // written, and the state-edge readback copies everything below it. One memset here gives
        // those slots a defined value; clear() leaves this array alone on the per-run path
        // because a slot is only ever read through a slice that was written with it.
        HG_CUDA_CHECK(cudaMemset(state_edge_ids_, 0,
              sizeof(EdgeId) * cfg_.max_state_edge_total),
              "EngineState state_edge_ids init");
        // ONE ALLOCATION FOR EVERY SCALAR COUNTER THE HOST READS BACK.
        //
        // Each of these is four bytes, and read on its own each costs a `cudaMemcpy` API call.
        // The transfer is instant; the CALL is not. Measured over one steady-state evolution of
        // `multirule`: 42 cudaMemcpy calls totalling 1.884 ms against a 4.74 ms window -- 39.8%
        // of the call -- at a median of 23.5 us each, 27 of them moving eight bytes or fewer.
        // Reading four bytes six times costs six times 23.5 us; reading twenty-four bytes once
        // costs 23.5 us. Contiguity is what makes the second possible, so these live in one
        // block and the individual pointers are offsets into it. The five pools' counters are
        // slots 6..10, taken in the member-initializer list above.
        //
        // Device code is unaffected: it still writes through the same typed pointers.
        counter_block_          = block_.p;
        state_edge_ids_counter_ = counter_block_ + 0;
        state_count_            = counter_block_ + 1;
        HG_CUDA_CHECK(cudaMalloc(&state_canonical_hash_, sizeof(uint64_t) * cfg_.max_states),
              "EngineState state_canonical_hash alloc");
        HG_CUDA_CHECK(cudaMalloc(&state_exact_hash_, sizeof(uint64_t) * cfg_.max_states),
              "EngineState state_exact_hash alloc");
        needs_indices_          = counter_block_ + 2;
        vertex_high_water_      = counter_block_ + 3;
        HG_CUDA_CHECK(cudaMalloc(&edge_producer_,     sizeof(EventId) * cfg_.max_edges),
              "EngineState edge_producer alloc");
        HG_CUDA_CHECK(cudaMalloc(&event_consumed_,
              sizeof(EdgeId) * uint64_t(cfg_.max_events) * event_consumed_stride(cfg_)),
              "EngineState event_consumed alloc");
        // One reachability slice per persistent block, taken by whichever thread needs one
        // (rewrite.cu is_reachable_preds). The visited table is masked, so the scale is rounded
        // up to a power of two.
        cfg_.tr_scratch_scale = tr_scratch_scale_of(cfg_);
        tr_scratch_slots_ = default_persistent_grid();
        HG_CUDA_CHECK(cudaMalloc(&tr_scratch_, sizeof(uint32_t) * tr_scratch_slots_ *
                                                   (kTrScratchStack + kTrScratchVisited) *
                                                   cfg_.tr_scratch_scale),
                      "EngineState tr_scratch alloc");
        HG_CUDA_CHECK(cudaMalloc(&tr_scratch_busy_, sizeof(uint32_t) * tr_scratch_slots_),
                      "EngineState tr_scratch busy alloc");
        HG_CUDA_CHECK(cudaMemset(tr_scratch_busy_, 0, sizeof(uint32_t) * tr_scratch_slots_),
                      "EngineState tr_scratch busy init");
        if (cfg_.survivor_scratch)
            HG_CUDA_CHECK(cudaMalloc(&survivor_scratch_, sizeof(uint64_t) * tr_scratch_slots_ *
                                                            cfg_.survivor_scratch),
                          "EngineState survivor_scratch alloc");
        clear();
    }

EngineState::~EngineState() {
        if (pinned_)                 cudaFreeHost(pinned_);
        if (state_edge_slices_)      cudaFree(state_edge_slices_);
        if (state_edge_ids_)         cudaFree(state_edge_ids_);
        if (rule_weights_dev_)       cudaFree(rule_weights_dev_);
        if (states_per_step_)        cudaFree(states_per_step_);
        // The scalar counters and the pool counters are slices of block_, which frees itself;
        // nothing here is freed individually.
        if (launch_scratch_.rules)        cudaFree(launch_scratch_.rules);
        if (launch_scratch_.states)       cudaFree(launch_scratch_.states);
        if (launch_scratch_.cursor)       cudaFree(launch_scratch_.cursor);
        if (launch_scratch_.phase_cycles) cudaFree(launch_scratch_.phase_cycles);
        if (state_canonical_hash_)   cudaFree(state_canonical_hash_);
        if (state_exact_hash_)       cudaFree(state_exact_hash_);
        if (state_edge_rank_)        cudaFree(state_edge_rank_);
        if (tr_scratch_)             cudaFree(tr_scratch_);
        if (tr_scratch_busy_)        cudaFree(tr_scratch_busy_);
        if (survivor_scratch_)       cudaFree(survivor_scratch_);
        if (state_edge_orbit_)       cudaFree(state_edge_orbit_);
        if (state_num_orbits_)       cudaFree(state_num_orbits_);
        if (state_invariant_at_)     cudaFree(state_invariant_at_);
        if (invariant_pool_)         cudaFree(invariant_pool_);
        if (invariant_pool_used_)    cudaFree(invariant_pool_used_);
        if (keyed_token_sum_)        cudaFree(keyed_token_sum_);
        if (keyed_first_new_edge_)   cudaFree(keyed_first_new_edge_);
        if (keyed_follow_head_)      cudaFree(keyed_follow_head_);
        if (keyed_follow_next_)      cudaFree(keyed_follow_next_);
        if (edge_producer_)          cudaFree(edge_producer_);
        if (event_consumed_)         cudaFree(event_consumed_);
    }

void EngineState::ensure_edge_ranks() {
        if (state_edge_rank_) return;
        HG_CUDA_CHECK(cudaMalloc(&state_edge_rank_, sizeof(uint32_t) * cfg_.max_state_edge_total),
              "EngineState state_edge_rank alloc");
        event_sig_fallbacks_ = counter_block_ + 4;
        HG_CUDA_CHECK(cudaMemset(state_edge_rank_, 0xFF,
              sizeof(uint32_t) * cfg_.max_state_edge_total),
              "EngineState init state_edge_rank");
        HG_CUDA_CHECK(cudaMemset(event_sig_fallbacks_, 0, sizeof(uint32_t)),
              "EngineState init event_sig_raw_fallbacks");
    }

void EngineState::ensure_edge_orbits() {
        if (state_edge_orbit_) return;
        HG_CUDA_CHECK(cudaMalloc(&state_edge_orbit_, sizeof(uint32_t) * cfg_.max_state_edge_total),
              "EngineState state_edge_orbit alloc");
        HG_CUDA_CHECK(cudaMalloc(&state_num_orbits_, sizeof(uint32_t) * cfg_.max_states),
              "EngineState state_num_orbits alloc");
        HG_CUDA_CHECK(cudaMemset(state_edge_orbit_, 0xFF,
              sizeof(uint32_t) * cfg_.max_state_edge_total),
              "EngineState init state_edge_orbit");
        HG_CUDA_CHECK(cudaMemset(state_num_orbits_, 0, sizeof(uint32_t) * cfg_.max_states),
              "EngineState init state_num_orbits");
    }

void EngineState::ensure_state_invariants() {
        if (state_invariant_at_) return;
        HG_CUDA_CHECK(cudaMalloc(&state_invariant_at_, sizeof(uint32_t) * cfg_.max_states),
              "EngineState state_invariant_at alloc");
        HG_CUDA_CHECK(cudaMalloc(&invariant_pool_, 8 * invariant_pool_words()),
              "EngineState invariant_pool alloc");
        HG_CUDA_CHECK(cudaMalloc(&invariant_pool_used_, sizeof(unsigned long long)),
              "EngineState invariant_pool_used alloc");
        HG_CUDA_CHECK(cudaMemset(state_invariant_at_, 0xFF, sizeof(uint32_t) * cfg_.max_states),
              "EngineState init state_invariant_at");
        HG_CUDA_CHECK(cudaMemset(invariant_pool_used_, 0, sizeof(unsigned long long)),
              "EngineState init invariant_pool_used");
    }

uint64_t EngineState::invariant_pool_used() const {
        if (!invariant_pool_used_) return 0;
        unsigned long long used = 0;
        HG_CUDA_CHECK(cudaMemcpy(&used, invariant_pool_used_, sizeof(used), cudaMemcpyDeviceToHost),
              "EngineState read invariant_pool_used");
        return used < invariant_pool_words() ? used : invariant_pool_words();
    }

void EngineState::ensure_keyed() {
        if (keyed_token_sum_) return;
        HG_CUDA_CHECK(cudaMalloc(&keyed_token_sum_, sizeof(uint64_t) * cfg_.max_states),
              "EngineState keyed token sum alloc");
        HG_CUDA_CHECK(cudaMalloc(&keyed_follow_head_, sizeof(uint32_t) * cfg_.max_states),
              "EngineState keyed follow head alloc");
        // Every head starts FOLLOW_EMPTY (all ones).
        HG_CUDA_CHECK(cudaMemset(keyed_follow_head_, 0xFF, sizeof(uint32_t) * cfg_.max_states),
              "EngineState keyed follow head init");
        HG_CUDA_CHECK(cudaMalloc(&keyed_follow_next_, sizeof(uint32_t) * cfg_.max_events),
              "EngineState keyed follow next alloc");
        HG_CUDA_CHECK(cudaMalloc(&keyed_first_new_edge_, sizeof(uint32_t) * cfg_.max_states),
              "EngineState keyed first new edge alloc");
    }

void EngineState::ensure_event_identity() {
        if (canonical_event_count_) return;
        canonical_event_count_ = counter_block_ + 5;
        HG_CUDA_CHECK(cudaMemset(canonical_event_count_, 0, sizeof(uint32_t)),
              "EngineState init canonical_event_count");
    }

uint32_t EngineState::canonical_event_count() const {
        if (!canonical_event_count_) return 0;
        uint32_t n = 0;
        HG_CUDA_CHECK(cudaMemcpy(&n, canonical_event_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "EngineState read canonical_event_count");
        return n;
    }

size_t EngineState::device_stack_bytes() {
    static size_t cached = 0;
    if (cached) return cached;
    size_t need = std::max(persistent_kernels_stack_bytes(), match_kernels_stack_bytes());
    cached = std::max<size_t>(need, 1024u);   // the driver's default
    return cached;
}

DeviceArena& EngineState::ir_arena(uint64_t needed_words) {
        if (!ir_arena_ || ir_arena_->capacity_words() < needed_words) {
            ir_arena_ = std::make_unique<DeviceArena>(needed_words);
        }
        ir_arena_->reset();
        return *ir_arena_;
    }

uint32_t EngineState::event_sig_raw_fallbacks() const {
        if (!event_sig_fallbacks_) return 0;
        uint32_t n = 0;
        HG_CUDA_CHECK(cudaMemcpy(&n, event_sig_fallbacks_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "EngineState read event_sig_raw_fallbacks");
        return n;
    }

void EngineState::set_sampling(double transition_rate, const double* weights, uint32_t num_weights,
                      uint64_t seed, double exploration_probability, uint32_t max_states_per_step,
                      uint32_t max_successor_states_per_parent, uint32_t matches_per_state_rule,
                      uint32_t num_steps, uint32_t num_rules) {
        transition_rate_     = transition_rate;
        num_rules_           = num_rules;
        sampling_seed_       = seed;
        exploration_probability_ = exploration_probability;
        max_states_per_step_ = max_states_per_step;
        max_succ_per_parent_ = max_successor_states_per_parent;
        matches_per_state_rule_ = matches_per_state_rule;

        if (weights && num_weights) {
            if (num_rule_weights_ < num_weights) {
                if (rule_weights_dev_) cudaFree(rule_weights_dev_);
                HG_CUDA_CHECK(cudaMalloc(&rule_weights_dev_, sizeof(double) * num_weights),
                              "EngineState rule_weights alloc");
            }
            HG_CUDA_CHECK(cudaMemcpy(rule_weights_dev_, weights, sizeof(double) * num_weights,
                                     cudaMemcpyHostToDevice), "EngineState rule_weights h2d");
            num_rule_weights_ = num_weights;
        } else {
            num_rule_weights_ = 0;
        }

        if (max_states_per_step_) {
            const uint32_t slots = num_steps + 2u;
            if (states_per_step_slots_ < slots) {
                if (states_per_step_) cudaFree(states_per_step_);
                HG_CUDA_CHECK(cudaMalloc(&states_per_step_, sizeof(uint32_t) * slots),
                              "EngineState states_per_step alloc");
                states_per_step_slots_ = slots;
            }
            HG_CUDA_CHECK(cudaMemset(states_per_step_, 0, sizeof(uint32_t) * states_per_step_slots_),
                          "EngineState states_per_step clear");
        }
    }

DeviceState EngineState::device() const {
        DeviceState d;
        d.vertex_pool             = vertex_pool_.view();
        d.edge_pool               = edge_pool_.view();
        d.state_edge_slices       = state_edge_slices_;
        d.state_edge_ids          = state_edge_ids_;
        d.state_edge_ids_counter  = state_edge_ids_counter_;
        d.state_edge_ids_capacity = cfg_.max_state_edge_total;
        d.max_states              = cfg_.max_states;
        d.ir_generators           = cfg_.ir_generators;
        d.ir_depth                = cfg_.ir_depth;
        d.max_edge_arity          = cfg_.max_edge_arity ? cfg_.max_edge_arity : kMaxArity;
        d.canonical_key_mask      = cfg_.canonical_key_mask;
        d.event_key_mask          = cfg_.event_key_mask;
        d.replay_id_limit         = cfg_.replay_id_limit;
        d.state_count             = state_count_;
        d.state_canonical_hash    = state_canonical_hash_;
        d.state_exact_hash        = state_exact_hash_;
        d.state_edge_rank         = state_edge_rank_;
        d.transition_rate                  = transition_rate_;
        d.rule_weights                     = rule_weights_dev_;
        d.num_rule_weights                 = num_rule_weights_;
        d.sampling_seed                    = sampling_seed_;
        d.exploration_probability          = exploration_probability_;
        d.max_states_per_step              = max_states_per_step_;
        d.step_pending                     = states_per_step_;
        d.step_slots                       = states_per_step_slots_;
        d.max_successor_states_per_parent  = max_succ_per_parent_;
        d.matches_per_state_rule           = matches_per_state_rule_;
        d.num_rules                        = num_rules_;
        // The orbit arrays stay allocated after a quotient run; a later run of this engine on
        // another route sees null, as a fresh engine does (transition_key_device keys on orbits
        // whenever the pointer is set).
        d.state_edge_orbit        = quotient_causal_ ? state_edge_orbit_ : nullptr;
        d.state_num_orbits        = quotient_causal_ ? state_num_orbits_ : nullptr;
        d.keyed                   = KeyedView{};
        d.keyed.state_token_sum   = keyed_token_sum_;
        d.keyed.state_first_new_edge = keyed_first_new_edge_;
        d.keyed.follow_head       = keyed_follow_head_;
        d.keyed.follow_next       = keyed_follow_next_;
        d.event_sig_raw_fallbacks = event_sig_fallbacks_;
        d.canonical_event_count   = canonical_event_count_;
        d.vertex_high_water       = vertex_high_water_;
        d.vertex_inverted_index   = vertex_inverted_index_.view();
        d.event_pool              = event_pool_.view();
        d.causal_edge_pool        = causal_edge_pool_.view();
        d.branchial_edge_pool     = branchial_edge_pool_.view();
        d.edge_producer           = edge_producer_;
        d.event_consumed          = event_consumed_;
        d.event_consumed_stride   = event_consumed_stride(cfg_);
        d.edge_consumers          = edge_consumers_.view();
        d.branchial_index         = branchial_index_.view();
        d.causal_triple_dedup     = causal_triple_dedup_.view();
        d.causal_pair_dedup       = causal_pair_dedup_.view();
        d.branchial_pair_dedup    = branchial_pair_dedup_.view();
        d.preds_list              = preds_list_.view();
        d.tr_enabled              = tr_enabled_;
        d.tr_scratch              = tr_scratch_;
        d.tr_scratch_busy         = tr_scratch_busy_;
        d.tr_scratch_stack        = kTrScratchStack * cfg_.tr_scratch_scale;
        d.tr_scratch_visited      = kTrScratchVisited * cfg_.tr_scratch_scale;
        d.tr_scratch_slots        = tr_scratch_slots_;
        d.survivor_scratch        = survivor_scratch_;
        d.survivor_scratch_cap    = cfg_.survivor_scratch;
        d.survivor_scratch_slots  = survivor_scratch_ ? tr_scratch_slots_ : 0u;
        d.quotient_causal         = quotient_causal_;
        d.slice_scan_max_edges    = slice_scan_max_edges_;
        d.maintain_indices        = maintain_indices_ ? 1u : 0u;
        d.record_causal           = record_.causal ? 1u : 0u;
        d.record_branchial        = record_.branchial ? 1u : 0u;
        d.record_invariants       = (record_.state_invariants && state_invariant_at_) ? 1u : 0u;
        d.state_invariant_at      = state_invariant_at_;
        d.invariant_pool          = invariant_pool_;
        d.invariant_pool_used     = invariant_pool_used_;
        d.invariant_pool_words    = invariant_pool_ ? invariant_pool_words() : 0;
        d.needs_indices           = needs_indices_;
        d.errors                  = errors_.view();
        return d;
    }

void EngineState::collect_warnings_into(std::vector<OverflowWarning>& out,
                               const char* context) {
        errors_.collect_warnings_into(out, context);
    }

void EngineState::report_event_sig_fallbacks(std::vector<OverflowWarning>& out, const char* context) const {
        report_event_sig_fallbacks(out, context, event_sig_raw_fallbacks());
    }

void EngineState::report_event_sig_fallbacks(std::vector<OverflowWarning>& out, const char* context,
                                             uint32_t fallbacks) {
        if (fallbacks) out.push_back(OverflowWarning{ErrorKind::kEventSigRawFallback, fallbacks, context});
    }

void EngineState::throw_on_errors(const char* context) const {
        errors_.throw_if_any(context);
    }

void EngineState::clear_errors() { errors_.clear(); }

void EngineState::set_record_set(hgcommon::RecordSet r) { record_ = r; }

hgcommon::RecordSet EngineState::record_set() const { return record_; }

void EngineState::set_tr_enabled(bool enabled) { tr_enabled_ = enabled; }

void EngineState::set_quotient_causal(bool enabled) { quotient_causal_ = enabled; }

bool EngineState::quotient_causal() const { return quotient_causal_; }

uint32_t EngineState::config_slice_scan_max_edges() const { return slice_scan_max_edges_; }

void EngineState::set_maintain_indices(bool on) { maintain_indices_ = on; }

bool EngineState::maintain_indices() const { return maintain_indices_; }

bool EngineState::needs_indices_host() const {
        uint32_t v = 0;
        HG_CUDA_CHECK(cudaMemcpy(&v, needs_indices_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "EngineState needs_indices read");
        return v != 0;
    }

void EngineState::clear() {
        // CLEAR WHAT THE LAST RUN DIRTIED, NOT WHAT THE CONFIG RESERVED.
        //
        // The per-edge-slot arrays are sized from the workload ESTIMATE -- config_from_input
        // reserves max_state_edge_total slots -- while a run writes only as many as it produced.
        // Clearing the reservation made every call pay for the estimate: nsys on a depth-3 run
        // producing THIRTEEN states measured 9.8 GB of cudaMemset across 981 operations, the
        // largest single one 538 MB, which is exactly 4 bytes x max_state_edge_total. That is
        // the fixed floor a small run cannot get under, and it is why sizing the pools generously
        // to avoid grow-and-retry made a depth-7 run slower rather than faster.
        //
        // At this point state_edge_ids_counter_ still holds the PREVIOUS run's final value, so it
        // names exactly the prefix that can be dirty. Slots above it were never written and still
        // carry the fill from construction, which zeroes the whole reservation once.
        // The same argument for the head arrays of the lists whose key is a dense id. Each
        // counter still holds the previous run's value here, so it names the prefix that can be
        // dirty; heads above it were never written and still carry the fill from construction.
        // Read them all before anything below resets them, in the counter block's one transfer.
        const CounterSnapshot prev = counters_snapshot_host();
        const uint32_t dirty_vertices = prev.vertex_high;
        const uint32_t dirty_edges_lf = prev.edges;
        const uint32_t dirty_events   = prev.events;
        const uint32_t dirty_edge_slots =
            prev.state_edge_ids <= cfg_.max_state_edge_total ? prev.state_edge_ids
                                                             : cfg_.max_state_edge_total;

        // The per-state and per-edge arrays the same way: a run writes state ids below its state
        // counter and edge ids below its edge counter. The first clear, from the constructor,
        // covers them in full.
        const uint32_t dirty_states =
            !cleared_once_ || prev.states > cfg_.max_states ? cfg_.max_states : prev.states;
        const uint32_t dirty_edges =
            !cleared_once_ || prev.edges > cfg_.max_edges ? cfg_.max_edges : prev.edges;

        // Every region below is cleared by one kernel launch (ClearBatch).
        ClearBatch batch;
        // Every counter, the pools' included (slots 6..10), restarts at zero.
        batch.add(counter_block_, sizeof(uint32_t) * kCounterSlots, 0);
        batch.add(state_edge_slices_, sizeof(StateEdgeSlice) * dirty_states, 0);
        // 0 means "not yet computed", which is why the empty state has its own reserved hash
        // rather than 0 -- see EMPTY_STATE_CANONICAL_HASH.
        batch.add(state_canonical_hash_, sizeof(uint64_t) * dirty_states, 0);
        batch.add(state_exact_hash_, sizeof(uint64_t) * dirty_states, 0);
        // UINT32_MAX, not 0: 0 is a valid rank (the canonically first edge), so a zeroed array
        // would read as "every edge ranks first" instead of "no ranks yet".
        if (state_edge_rank_) batch.add(state_edge_rank_, sizeof(uint32_t) * dirty_edge_slots, 0xFF);
        if (state_edge_orbit_) {
            batch.add(state_edge_orbit_, sizeof(uint32_t) * dirty_edge_slots, 0xFF);
            batch.add(state_num_orbits_, sizeof(uint32_t) * dirty_states, 0);
        }
        if (state_invariant_at_) {
            batch.add(state_invariant_at_, sizeof(uint32_t) * dirty_states, 0xFF);
            batch.add(invariant_pool_used_, sizeof(unsigned long long), 0);
        }
        // edge_producer init to INVALID_ID (0xFF bytes).
        batch.add(edge_producer_, sizeof(EventId) * dirty_edges, 0xFF);
        // A run closes the follower stack of every keyed state it hashes; FOLLOW_EMPTY (0xFF
        // bytes) again for the next run's states.
        if (keyed_follow_head_)
            batch.add(keyed_follow_head_, sizeof(uint32_t) * dirty_states, 0xFF);
        vertex_inverted_index_.clear(dirty_vertices, &batch);
        edge_consumers_.clear(dirty_edges_lf, &batch);
        branchial_index_.clear(0xFFFFFFFFu, &batch);
        // The relation dedup maps are written only on the way to a claim on their pool
        // (try_add_causal_edge inserts the triple, then claims; the pair after a claim;
        // try_add_branchial_edge inserts, then claims), and a pool counter only grows, so a
        // previous run whose counter reads zero left the map as its last clear did. Together
        // they are 60 MB of memset at the default config.
        if (prev.causal) {
            causal_triple_dedup_.clear(&batch);
            causal_pair_dedup_.clear(&batch);
        }
        if (prev.branchial) branchial_pair_dedup_.clear(&batch);
        preds_list_.clear(dirty_events, &batch);
        errors_.clear(&batch);
        batch.flush();
        cleared_once_ = true;
    }

const EngineConfig& EngineState::config() const { return cfg_; }

uint32_t EngineState::num_edges_host() const { return edge_pool_.size_host(); }

uint32_t EngineState::num_states_host() const {
        uint32_t v = 0;
        cudaMemcpy(&v, state_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return v;
    }

uint32_t EngineState::vertex_high_water_host() const {
        uint32_t v = 0;
        cudaMemcpy(&v, vertex_high_water_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return v;
    }

Edge EngineState::edge_at_host(EdgeId eid) const {
        Edge e{};
        cudaMemcpy(&e, edge_pool_view_data() + eid, sizeof(Edge), cudaMemcpyDeviceToHost);
        return e;
    }

std::vector<VertexId> EngineState::edge_vertices_host(EdgeId eid) const {
        Edge e = edge_at_host(eid);
        std::vector<VertexId> out(e.arity);
        cudaMemcpy(out.data(), vertex_pool_view_data() + e.vertex_offset,
                   sizeof(VertexId) * e.arity, cudaMemcpyDeviceToHost);
        return out;
    }

void EngineState::add_state_edges(ReadbackBatch& batch, const CounterSnapshot& snap,
                                  std::vector<StateEdgeSlice>& slices, EvolveResult& out) const {
        if (snap.states == 0) return;
        batch.add(out.edge_records, edge_pool_.view().data, snap.edges);
        batch.add(out.vertex_pool, vertex_pool_.view().data, snap.vertex_slots);
        batch.add(slices, static_cast<const StateEdgeSlice*>(state_edge_slices_), snap.states);
        batch.add(out.state_edge_ids, static_cast<const EdgeId*>(state_edge_ids_),
                  snap.state_edge_ids);
    }

void EngineState::ReadbackBatch::finish() {
        size_t total = 0;
        for (const Region& r : regions_) total += (r.bytes + 15u) & ~size_t(15);
        if (total == 0) { regions_.clear(); return; }
        if (engine_.pinned_bytes_ < total) {
            if (engine_.pinned_) cudaFreeHost(engine_.pinned_);
            engine_.pinned_ = nullptr;
            engine_.pinned_bytes_ = 0;
            HG_CUDA_CHECK(cudaMallocHost(&engine_.pinned_, total), "readback staging alloc");
            engine_.pinned_bytes_ = total;
        }
        char* base = static_cast<char*>(engine_.pinned_);
        size_t off = 0;
        for (const Region& r : regions_) {
            HG_CUDA_CHECK(cudaMemcpyAsync(base + off, r.device, r.bytes, cudaMemcpyDeviceToHost, 0),
                          "readback copy");
            off += (r.bytes + 15u) & ~size_t(15);
        }
        HG_CUDA_CHECK(cudaStreamSynchronize(0), "readback sync");
        off = 0;
        for (const Region& r : regions_) {
            if (r.fill) r.fill(r.host, base + off, r.bytes);
            else std::memcpy(r.host, base + off, r.bytes);
            off += (r.bytes + 15u) & ~size_t(15);
        }
        regions_.clear();
    }

std::vector<EdgeId> EngineState::state_edges_host(StateId sid) const {
        StateEdgeSlice sl{0, 0};
        cudaMemcpy(&sl, state_edge_slices_ + sid, sizeof(StateEdgeSlice),
                   cudaMemcpyDeviceToHost);
        std::vector<EdgeId> out(sl.count);
        if (sl.count > 0) {
            cudaMemcpy(out.data(), state_edge_ids_ + sl.offset,
                       sizeof(EdgeId) * sl.count, cudaMemcpyDeviceToHost);
        }
        return out;
    }

Edge*     EngineState::edge_pool_view_data()    const { return edge_pool_.view().data; }

VertexId* EngineState::vertex_pool_view_data()  const { return vertex_pool_.view().data; }

// Grow-only: an array whose capacity fits is reused as-is, one that does not is freed and
// reallocated to the request. The fixed buffers are allocated on first use and live with the
// engine. Callers run serially against one engine, so there is no aliasing to guard.
EngineState::LaunchScratch& EngineState::launch_scratch(uint32_t num_rules,
                                                        uint32_t num_states) const {
        LaunchScratch& s = launch_scratch_;
        if (s.rules_cap < num_rules) {
            if (s.rules) cudaFree(s.rules);
            HG_CUDA_CHECK(cudaMalloc(&s.rules, sizeof(DeviceRule) * num_rules),
                  "launch scratch rules");
            s.rules_cap = num_rules;
        }
        if (s.states_cap < num_states) {
            if (s.states) cudaFree(s.states);
            HG_CUDA_CHECK(cudaMalloc(&s.states, sizeof(StateId) * num_states),
                  "launch scratch states");
            s.states_cap = num_states;
        }
        if (!s.cursor) {
            HG_CUDA_CHECK(cudaMalloc(&s.cursor, sizeof(uint32_t) * 2), "launch scratch cursor");
            HG_CUDA_CHECK(cudaMalloc(&s.phase_cycles, sizeof(unsigned long long) * 16),
                  "launch scratch phases");
        }
        return s;
    }

EngineState::CounterSnapshot EngineState::counters_snapshot_host() const {
        // The pools' counters LIVE in slots 6..10 -- the pools were constructed onto the
        // block -- so this one transfer carries every scalar with no staging. A staging
        // kernel was measured costlier than the copies it replaced on this platform.
        uint32_t raw[kCounterSlots] = {};
        HG_CUDA_CHECK(cudaMemcpy(raw, counter_block_, sizeof(raw), cudaMemcpyDeviceToHost),
              "EngineState counter block d2h");
        return snapshot_from(raw);
    }

EngineState::CounterSnapshot EngineState::snapshot_from(const uint32_t* raw) {
        CounterSnapshot c;
        c.state_edge_ids = raw[0]; c.states        = raw[1]; c.needs_indices = raw[2];
        c.vertex_high    = raw[3]; c.sig_fallbacks = raw[4]; c.canonical_ev  = raw[5];
        c.edges          = raw[6]; c.vertex_slots  = raw[7]; c.events        = raw[8];
        c.causal         = raw[9]; c.branchial     = raw[10];
        return c;
    }

uint32_t EngineState::num_events_host()          const { return event_pool_.size_host(); }

uint32_t EngineState::num_causal_edges_host()    const { return causal_edge_pool_.size_host(); }

uint32_t EngineState::num_branchial_edges_host() const { return branchial_edge_pool_.size_host(); }

std::vector<DeviceCausalEdge> EngineState::causal_edges_host() const {
        return causal_edges_host(num_causal_edges_host());
    }

std::vector<DeviceBranchialEdge> EngineState::branchial_edges_host() const {
        return branchial_edges_host(num_branchial_edges_host());
    }

void EngineState::add_events(ReadbackBatch& batch, uint32_t n,
                             std::vector<DeviceEvent>& events,
                             std::vector<EdgeId>& consumed) const {
        batch.add(events, static_cast<const DeviceEvent*>(event_pool_.view().data), n);
        batch.add(consumed, static_cast<const EdgeId*>(event_consumed_),
                  size_t(n) * event_consumed_stride(cfg_));
    }

void EngineState::add_causal_edges(ReadbackBatch& batch, uint32_t n,
                                   std::vector<DeviceCausalEdge>& out) const {
        batch.add(out, static_cast<const DeviceCausalEdge*>(causal_edge_pool_.view().data), n);
    }

void EngineState::add_branchial_edges(ReadbackBatch& batch, uint32_t n,
                                      std::vector<DeviceBranchialEdge>& out) const {
        batch.add(out, static_cast<const DeviceBranchialEdge*>(branchial_edge_pool_.view().data),
                  n);
    }

std::vector<DeviceCausalEdge> EngineState::causal_edges_host(uint32_t n) const {
        ReadbackBatch batch(*this);
        std::vector<DeviceCausalEdge> out;
        add_causal_edges(batch, n, out);
        batch.finish();
        return out;
    }

std::vector<DeviceBranchialEdge> EngineState::branchial_edges_host(uint32_t n) const {
        ReadbackBatch batch(*this);
        std::vector<DeviceBranchialEdge> out;
        add_branchial_edges(batch, n, out);
        batch.finish();
        return out;
    }

}  // namespace gpu
}  // namespace HG_NAMESPACE
