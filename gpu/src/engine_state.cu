#include "hg_gpu/engine_state.hpp"
#include "hg_gpu/device_arena.hpp"
#include "hg_gpu/signature_index.hpp"
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

void DeviceArena::reset() {
    HG_CUDA_CHECK(cudaMemset(cursor_, 0, sizeof(uint64_t)), "arena cursor clear");
}

DeviceArena::View DeviceArena::view() { return View{base_, cursor_, capacity_}; }

uint64_t DeviceArena::capacity_words() const { return capacity_; }

uint64_t DeviceArena::used_words_host() const {
    uint64_t v = 0;
    cudaMemcpy(&v, cursor_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return v;
}

// =============================================================================
// SignatureIndex
// =============================================================================

SignatureIndex::SignatureIndex(uint32_t num_buckets_pow2, uint32_t max_edges)
    : list_(num_buckets_pow2, max_edges),
      mask_(num_buckets_pow2 - 1)
{
    if ((num_buckets_pow2 & mask_) != 0 || num_buckets_pow2 == 0) {
        throw std::invalid_argument("SignatureIndex num_buckets must be a power of two ≥ 1");
    }
}

SignatureIndex::DeviceView SignatureIndex::view() const {
    return DeviceView{list_.view(), mask_};
}

uint32_t SignatureIndex::num_buckets() const { return list_.num_keys(); }
uint32_t SignatureIndex::used() const { return list_.pool_used_host(); }

void SignatureIndex::clear() { list_.clear(); }

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

void VertexInvertedIndex::clear(uint32_t used_vertices) { list_.clear(used_vertices); }

// =============================================================================
// EngineState
// =============================================================================


EngineState::EngineState(EngineConfig cfg): cfg_(cfg)
        , vertex_pool_(cfg.max_vertex_slots)
        , edge_pool_(cfg.max_edges)
        , signature_index_(cfg.sig_index_buckets, cfg.sig_index_pool)
        , vertex_inverted_index_(cfg.max_vertices, cfg.inverted_pool)
        , event_pool_(cfg.max_events)
        , causal_edge_pool_(cfg.max_causal_edges)
        , branchial_edge_pool_(cfg.max_branchial_edges)
        , edge_consumers_(cfg.max_edges, cfg.edge_consumer_nodes)
        , branchial_index_(cfg.branchial_index_buckets, cfg.branchial_index_nodes)
        , causal_triple_dedup_(cfg.causal_triple_slots)
        , causal_pair_dedup_(cfg.causal_pair_slots)
        , branchial_pair_dedup_(cfg.branchial_pair_slots)
        , preds_list_(cfg.max_events, cfg.tr_preds_nodes) {
        // Every kernel that runs against an EngineState needs more per-thread stack than the
        // 1 KB default: match_state_rule's DFS recurses to the LHS edge count, apply_one_match
        // holds several kMaxPatternEdges arrays, and a scheduler that calls both from one
        // kernel carries the sum. Raising it here rather than in one scheduler's constructor
        // is what makes it hold for every entry point -- a scheduler that missed it would fail
        // as a stack overflow reported as an illegal memory access.
        // Checked, and then READ BACK. A driver may clamp the request rather than refuse it, so a
        // successful return does not mean the stack is the size that was asked for -- and the
        // failure mode either way is a stack overflow surfacing as an illegal memory access,
        // which reads like a pointer bug and is diagnosed as one.
        // ASK FOR LESS RATHER THAN FAIL. The driver reserves this per-thread size across every
        // thread the device can hold resident -- not across the launch grid -- so on a small or
        // busy device even this fixed request can exceed what is available, and
        // cudaDeviceSetLimit returns out-of-memory. Throwing there would turn a run that could
        // have proceeded into no run at all. Halving until it is accepted reaches the largest
        // stack the device will actually grant.
        //
        // THE REQUEST NO LONGER GROWS WITH THE RUN. It once did -- 32 KB plus 8,704 bytes per
        // step, capped at 256 KB -- because the reconstruction recursed once per depth. An
        // 80-step configuration then asked for the whole 256 KB cap and the memory budget
        // exceeded what the pools could give back, since fit_config_to_cap scales pools and the
        // stack is not a pool. Depth rides a worklist now, so this is a constant and a deep run
        // costs the device exactly what a shallow one does.
        // BOUNDED BY A CONSTANT, and no larger than the run can use. The DP recurses at most
        // kDpNestLevels deep and at most as deep as the evolution, so a short run asks for the
        // shorter of the two: this is <= what the old depth-scaled request asked at EVERY depth,
        // which is what keeps a shallow run from paying for a budget it cannot reach. Measured
        // at two steps, where the request would otherwise have grown: 4.736 ms median against
        // 4.672 for the old sizing, so asking for the full budget there cost 1.4%.
        const uint32_t nest = cfg.reconstruction_max_depth + 1u < kDpNestLevels
                                  ? cfg.reconstruction_max_depth + 1u
                                  : kDpNestLevels;
        size_t want_stack = kDeviceStackFloorBytes + nest * kDpBytesPerNestLevel;
        cudaError_t st_rc = cudaDeviceSetLimit(cudaLimitStackSize, want_stack);
        while (st_rc != cudaSuccess && want_stack > kDeviceStackFloorBytes) {
            cudaGetLastError();                      // clear the sticky error before retrying
            want_stack = want_stack / 2 > kDeviceStackFloorBytes ? want_stack / 2
                                                                 : kDeviceStackFloorBytes;
            st_rc = cudaDeviceSetLimit(cudaLimitStackSize, want_stack);
        }
        HG_CUDA_CHECK(st_rc, "set device stack size");
        size_t actual_stack = 0;
        HG_CUDA_CHECK(cudaDeviceGetLimit(&actual_stack, cudaLimitStackSize), "read device stack size");
        if (actual_stack < kDeviceStackFloorBytes) {
            throw std::runtime_error(
                "EngineState: device stack is " + std::to_string(actual_stack) +
                " bytes after requesting " + std::to_string(want_stack) +
                "; match_state_rule's DFS would overflow it and report an illegal memory access");
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
        // written, and all_state_edges_host() copies everything below it. One memset here gives
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
        // costs 23.5 us. Contiguity is what makes the second possible, so these six live in one
        // block and the individual pointers are offsets into it.
        //
        // Device code is unaffected: it still writes through the same typed pointers.
        HG_CUDA_CHECK(cudaMalloc(&counter_block_, sizeof(uint32_t) * kCounterSlots),
              "EngineState counter block alloc");
        // counters_snapshot_host() transfers all six slots at once, while slots 4 and 5 are bound
        // to their pointers on first use of the feature that owns them. Zeroing the whole block
        // here is what makes a slot read as zero when its feature never runs.
        HG_CUDA_CHECK(cudaMemset(counter_block_, 0, sizeof(uint32_t) * kCounterSlots),
              "EngineState counter block init");
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
        clear();
    }

EngineState::~EngineState() {
        if (state_edge_slices_)      cudaFree(state_edge_slices_);
        if (state_edge_ids_)         cudaFree(state_edge_ids_);
        if (rule_weights_dev_)       cudaFree(rule_weights_dev_);
        if (states_per_step_)        cudaFree(states_per_step_);
        if (successors_per_parent_)  cudaFree(successors_per_parent_);
        // The six scalar counters are slices of counter_block_, not separate allocations, so
        // the block is freed once here and none of them is freed individually.
        if (counter_block_)          cudaFree(counter_block_);
        if (state_canonical_hash_)   cudaFree(state_canonical_hash_);
        if (state_exact_hash_)       cudaFree(state_exact_hash_);
        if (state_edge_rank_)        cudaFree(state_edge_rank_);
        if (state_edge_orbit_)       cudaFree(state_edge_orbit_);
        if (state_num_orbits_)       cudaFree(state_num_orbits_);
        if (edge_producer_)          cudaFree(edge_producer_);
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
                      uint64_t seed, uint32_t max_states_per_step,
                      uint32_t max_successor_states_per_parent, uint32_t matches_per_state_rule,
                      uint32_t num_steps) {
        transition_rate_     = transition_rate;
        sampling_seed_       = seed;
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
        if (max_succ_per_parent_) {
            if (!successors_per_parent_) {
                HG_CUDA_CHECK(cudaMalloc(&successors_per_parent_,
                                         sizeof(uint32_t) * cfg_.max_states),
                              "EngineState successors_per_parent alloc");
            }
            HG_CUDA_CHECK(cudaMemset(successors_per_parent_, 0,
                                     sizeof(uint32_t) * cfg_.max_states),
                          "EngineState successors_per_parent clear");
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
        d.state_count             = state_count_;
        d.state_canonical_hash    = state_canonical_hash_;
        d.state_exact_hash        = state_exact_hash_;
        d.state_edge_rank         = state_edge_rank_;
        d.transition_rate                  = transition_rate_;
        d.rule_weights                     = rule_weights_dev_;
        d.num_rule_weights                 = num_rule_weights_;
        d.sampling_seed                    = sampling_seed_;
        d.max_states_per_step              = max_states_per_step_;
        d.states_per_step                  = states_per_step_;
        d.max_states_per_step_slots        = states_per_step_slots_;
        d.max_successor_states_per_parent  = max_succ_per_parent_;
        d.matches_per_state_rule           = matches_per_state_rule_;
        d.successors_per_parent            = successors_per_parent_;
        d.state_edge_orbit        = state_edge_orbit_;
        d.state_num_orbits        = state_num_orbits_;
        d.event_sig_raw_fallbacks = event_sig_fallbacks_;
        d.canonical_event_count   = canonical_event_count_;
        d.vertex_high_water       = vertex_high_water_;
        d.signature_index         = signature_index_.view();
        d.vertex_inverted_index   = vertex_inverted_index_.view();
        d.event_pool              = event_pool_.view();
        d.causal_edge_pool        = causal_edge_pool_.view();
        d.branchial_edge_pool     = branchial_edge_pool_.view();
        d.edge_producer           = edge_producer_;
        d.edge_consumers          = edge_consumers_.view();
        d.branchial_index         = branchial_index_.view();
        d.causal_triple_dedup     = causal_triple_dedup_.view();
        d.causal_pair_dedup       = causal_pair_dedup_.view();
        d.branchial_pair_dedup    = branchial_pair_dedup_.view();
        d.preds_list              = preds_list_.view();
        d.tr_enabled              = tr_enabled_;
        d.quotient_causal         = quotient_causal_;
        d.slice_scan_max_edges    = slice_scan_max_edges_;
        d.maintain_indices        = maintain_indices_ ? 1u : 0u;
        d.record_causal           = record_.causal ? 1u : 0u;
        d.record_branchial        = record_.branchial ? 1u : 0u;
        d.needs_indices           = needs_indices_;
        d.errors                  = errors_.view();
        return d;
    }

void EngineState::collect_warnings_into(std::vector<OverflowWarning>& out,
                               const char* context) {
        errors_.collect_warnings_into(out, context);
    }

void EngineState::report_event_sig_fallbacks(std::vector<OverflowWarning>& out, const char* context) const {
        if (const uint32_t n = event_sig_raw_fallbacks()) {
            out.push_back(OverflowWarning{ErrorKind::kEventSigRawFallback, n, context});
        }
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
        // Read them all before anything below resets them.
        const uint32_t dirty_vertices = vertex_high_water_host();
        const uint32_t dirty_edges_lf = edge_pool_.size_host();
        const uint32_t dirty_events   = event_pool_.size_host();

        uint32_t dirty_edge_slots = cfg_.max_state_edge_total;
        if (state_edge_ids_counter_) {
            uint32_t n = 0;
            if (cudaMemcpy(&n, state_edge_ids_counter_, sizeof(uint32_t),
                           cudaMemcpyDeviceToHost) == cudaSuccess && n <= cfg_.max_state_edge_total)
                dirty_edge_slots = n;
        }

        HG_CUDA_CHECK(cudaMemset(state_edge_slices_, 0,
              sizeof(StateEdgeSlice) * cfg_.max_states),
              "EngineState clear state_edge_slices");
        HG_CUDA_CHECK(cudaMemset(state_edge_ids_counter_, 0, sizeof(uint32_t)),
              "EngineState clear state_edge_ids_counter");
        HG_CUDA_CHECK(cudaMemset(state_count_,       0, sizeof(uint32_t)), "EngineState clear state_count");
        // 0 means "not yet computed", which is why the empty state has its own reserved hash
        // rather than 0 -- see EMPTY_STATE_CANONICAL_HASH.
        HG_CUDA_CHECK(cudaMemset(state_canonical_hash_, 0, sizeof(uint64_t) * cfg_.max_states),
              "EngineState clear state_canonical_hash");
        HG_CUDA_CHECK(cudaMemset(state_exact_hash_, 0, sizeof(uint64_t) * cfg_.max_states),
              "EngineState clear state_exact_hash");
        if (state_edge_rank_) {
            // UINT32_MAX, not 0: 0 is a valid rank (the canonically first edge), so a zeroed
            // array would read as "every edge ranks first" instead of "no ranks yet".
            HG_CUDA_CHECK(cudaMemset(state_edge_rank_, 0xFF,
                  sizeof(uint32_t) * dirty_edge_slots),
                  "EngineState clear state_edge_rank");
        }
        if (state_edge_orbit_) {
            HG_CUDA_CHECK(cudaMemset(state_edge_orbit_, 0xFF,
                  sizeof(uint32_t) * dirty_edge_slots),
                  "EngineState clear state_edge_orbit");
            HG_CUDA_CHECK(cudaMemset(state_num_orbits_, 0, sizeof(uint32_t) * cfg_.max_states),
                  "EngineState clear state_num_orbits");
        }
        if (event_sig_fallbacks_) {
            HG_CUDA_CHECK(cudaMemset(event_sig_fallbacks_, 0, sizeof(uint32_t)),
                  "EngineState clear event_sig_raw_fallbacks");
        }
        if (canonical_event_count_) {
            HG_CUDA_CHECK(cudaMemset(canonical_event_count_, 0, sizeof(uint32_t)),
                  "EngineState clear canonical_event_count");
        }
        HG_CUDA_CHECK(cudaMemset(needs_indices_,     0, sizeof(uint32_t)), "EngineState clear needs_indices");
        HG_CUDA_CHECK(cudaMemset(vertex_high_water_, 0, sizeof(uint32_t)), "EngineState clear vertex_high_water");
        // edge_producer init to INVALID_ID (0xFF bytes).
        HG_CUDA_CHECK(cudaMemset(edge_producer_, 0xFF, sizeof(EventId) * cfg_.max_edges),
              "EngineState clear edge_producer");
        vertex_pool_.reset();
        edge_pool_.reset();
        signature_index_.clear();
        vertex_inverted_index_.clear(dirty_vertices);
        event_pool_.reset();
        causal_edge_pool_.reset();
        branchial_edge_pool_.reset();
        edge_consumers_.clear(dirty_edges_lf);
        branchial_index_.clear();
        causal_triple_dedup_.clear();
        causal_pair_dedup_.clear();
        branchial_pair_dedup_.clear();
        preds_list_.clear(dirty_events);
        errors_.clear();
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

std::vector<std::vector<std::vector<VertexId>>> EngineState::all_state_edges_host(
            std::vector<std::vector<EdgeId>>* out_edge_ids ,
            std::vector<std::vector<VertexId>>* out_global_edges) const {
        uint32_t n_states = num_states_host();
        std::vector<std::vector<std::vector<VertexId>>> out(n_states);
        if (out_edge_ids) out_edge_ids->assign(n_states, {});
        if (out_global_edges) out_global_edges->clear();
        if (n_states == 0) return out;

        uint32_t n_edges      = edge_pool_.size_host();
        uint32_t n_vert_slots = vertex_pool_.size_host();
        uint32_t n_id_slots   = 0;
        cudaMemcpy(&n_id_slots, state_edge_ids_counter_, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        std::vector<Edge>           edges(n_edges);
        std::vector<VertexId>       verts(n_vert_slots);
        std::vector<StateEdgeSlice> slices(n_states);
        std::vector<EdgeId>         ids(n_id_slots);

        if (n_edges > 0) {
            cudaMemcpy(edges.data(), edge_pool_.view().data,
                       sizeof(Edge) * n_edges, cudaMemcpyDeviceToHost);
        }
        if (n_vert_slots > 0) {
            cudaMemcpy(verts.data(), vertex_pool_.view().data,
                       sizeof(VertexId) * n_vert_slots, cudaMemcpyDeviceToHost);
        }
        cudaMemcpy(slices.data(), state_edge_slices_,
                   sizeof(StateEdgeSlice) * n_states, cudaMemcpyDeviceToHost);
        if (n_id_slots > 0) {
            cudaMemcpy(ids.data(), state_edge_ids_,
                       sizeof(EdgeId) * n_id_slots, cudaMemcpyDeviceToHost);
        }

        if (out_global_edges) {
            out_global_edges->assign(n_edges, {});
            for (uint32_t eid = 0; eid < n_edges; ++eid) {
                const Edge& e = edges[eid];
                if (static_cast<size_t>(e.vertex_offset) + e.arity > verts.size()) continue;
                std::vector<VertexId> vs(e.arity);
                for (uint8_t i = 0; i < e.arity; ++i) vs[i] = verts[e.vertex_offset + i];
                (*out_global_edges)[eid] = std::move(vs);
            }
        }

        for (uint32_t s = 0; s < n_states; ++s) {
            const StateEdgeSlice& sl = slices[s];
            if (static_cast<size_t>(sl.offset) + sl.count > ids.size()) continue;
            for (uint32_t k = 0; k < sl.count; ++k) {
                EdgeId eid = ids[sl.offset + k];
                if (eid >= n_edges) continue;
                if (out_edge_ids) (*out_edge_ids)[s].push_back(eid);
                const Edge& e = edges[eid];
                std::vector<VertexId> vs(e.arity);
                for (uint8_t i = 0; i < e.arity; ++i) {
                    vs[i] = verts[e.vertex_offset + i];
                }
                out[s].push_back(std::move(vs));
            }
        }
        return out;
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

EngineState::CounterSnapshot EngineState::counters_snapshot_host() const {
        uint32_t raw[kCounterSlots] = {};
        HG_CUDA_CHECK(cudaMemcpy(raw, counter_block_, sizeof(raw), cudaMemcpyDeviceToHost),
              "EngineState counter block d2h");
        CounterSnapshot c;
        c.state_edge_ids = raw[0]; c.states        = raw[1]; c.needs_indices = raw[2];
        c.vertex_high    = raw[3]; c.sig_fallbacks = raw[4]; c.canonical_ev  = raw[5];
        return c;
    }

uint32_t EngineState::num_events_host()          const { return event_pool_.size_host(); }

uint32_t EngineState::num_causal_edges_host()    const { return causal_edge_pool_.size_host(); }

uint32_t EngineState::num_branchial_edges_host() const { return branchial_edge_pool_.size_host(); }

std::vector<DeviceEvent> EngineState::events_host() const {
        uint32_t n = num_events_host();
        std::vector<DeviceEvent> out(n);
        if (n > 0) cudaMemcpy(out.data(), event_pool_.view().data,
                              sizeof(DeviceEvent) * n, cudaMemcpyDeviceToHost);
        return out;
    }

std::vector<DeviceCausalEdge> EngineState::causal_edges_host() const {
        uint32_t n = num_causal_edges_host();
        std::vector<DeviceCausalEdge> out(n);
        if (n > 0) cudaMemcpy(out.data(), causal_edge_pool_.view().data,
                              sizeof(DeviceCausalEdge) * n, cudaMemcpyDeviceToHost);
        return out;
    }

std::vector<DeviceBranchialEdge> EngineState::branchial_edges_host() const {
        uint32_t n = num_branchial_edges_host();
        std::vector<DeviceBranchialEdge> out(n);
        if (n > 0) cudaMemcpy(out.data(), branchial_edge_pool_.view().data,
                              sizeof(DeviceBranchialEdge) * n, cudaMemcpyDeviceToHost);
        return out;
    }

}  // namespace gpu
}  // namespace HG_NAMESPACE
