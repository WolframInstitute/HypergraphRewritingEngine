#include "hg_gpu/quotient_causal.hpp"
#include "hg_gpu/quotient_expansion.hpp"

// The HOST bodies of the two quotient state objects: allocation, clear, the counter readbacks
// and the view() calls that hand a device-side struct to a kernel. The DP and the replay
// themselves are __device__ and stay in their headers, as does everything the shared
// hgcommon cores reach.

#include <utility>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// =============================================================================
// QcState
// =============================================================================

QcState::QcState(bool on, uint32_t max_events): transitions_(on ? max_events : 1u),
          trans_from_(on ? (1u << 16) : 1u, on ? max_events : 1u),
          seen_(on ? max_events * 2u : 8u),
          dsup_(on ? (1u << 18) : 1u, on ? max_events * 8u : 8u),
          dsup_seen_(on ? max_events * 16u : 8u),
          reached_(on ? (1u << 20) : 8u),
          arr_cap_(on ? max_events * 16u : 1u),
          on_(on) {
        HG_CUDA_CHECK(cudaMalloc(&arr_, sizeof(uint32_t) * arr_cap_), "QcState arr alloc");
        HG_CUDA_CHECK(cudaMalloc(&cursor_, sizeof(uint32_t)), "QcState cursor alloc");
        clear();
    }

QcState::~QcState() {
    if (work_items_) cudaFree(work_items_);
        if (arr_)    cudaFree(arr_);
        if (cursor_) cudaFree(cursor_);
    }

bool QcState::enabled() const { return on_; }

void QcState::set_record_causal(bool on) { record_causal_ = on; }

bool QcState::record_causal() const { return record_causal_; }

void QcState::clear() {
        seen_.clear();
        dsup_seen_.clear();
        reached_.clear();
        trans_from_.clear();
        dsup_.clear();
        transitions_.reset();
        HG_CUDA_CHECK(cudaMemset(cursor_, 0, sizeof(uint32_t)), "QcState cursor clear");
    }

// The DP descends through this rather than through the call stack, so its size is what bounds
// cascade depth. Same shape and same reasoning as QeState::ensure_work.
void QcState::ensure_work(uint32_t slices, uint32_t max_steps) {
    if (!on_) return;
    const uint32_t cap = max_steps * 64u < 256u ? 256u : max_steps * 64u;
    if (work_items_ && work_slices_ >= slices && work_cap_ >= cap) return;
    if (work_items_) { cudaFree(work_items_); work_items_ = nullptr; }
    work_slices_ = slices > work_slices_ ? slices : work_slices_;
    work_cap_    = cap > work_cap_ ? cap : work_cap_;
    HG_CUDA_CHECK(cudaMalloc(&work_items_, sizeof(QcWorkItem) * work_slices_ * work_cap_),
                  "QcState cascade stacks alloc");
}

QcView QcState::view(uint32_t max_steps) {
        QcView q{};
        q.transitions      = transitions_.view();
        q.trans_from       = trans_from_.view();
        q.seen_transitions = seen_.view();
        q.arr_words        = arr_;
        q.arr_cursor       = cursor_;
        q.arr_capacity     = arr_cap_;
        q.dsup             = dsup_.view();
        q.dsup_seen        = dsup_seen_.view();
        q.reached          = reached_.view();
        q.max_steps        = max_steps;
        q.work_items   = work_items_;
        q.work_cap     = work_cap_;
        q.work_slices  = work_slices_;
        q.enabled          = on_ ? 1u : 0u;
        q.record_causal    = record_causal_ ? 1u : 0u;
        return q;
    }

// =============================================================================
// QeState
// =============================================================================

QeState::QeState(bool on, uint32_t max_events): matches_(on ? max_events : 1u),
          by_from_(on ? (1u << 16) : 1u, on ? max_events : 1u),
          instances_(on ? max_events : 1u),
          by_key_(on ? (1u << 16) : 1u, on ? max_events : 1u),
          rep_(on ? max_events : 8u),
          applied_(on ? max_events * 4u : 8u),
          canon_seen_(on ? max_events * 2u : 8u),
          causal_pairs_(on ? max_events * 4u : 8u),
          inst_applied_(on ? (1u << 16) : 1u, on ? max_events * 2u : 1u),
          frame_(on ? max_events * 2u : 8u),
          arr_cap_(on ? max_events * 16u : 1u),
          on_(on) {
        HG_CUDA_CHECK(cudaMalloc(&arr_, sizeof(uint32_t) * arr_cap_), "QeState arr alloc");
        // ELEVEN SCALARS IN ONE BLOCK, so the host reads them in ONE transfer.
        //
        // Each of these was its own cudaMalloc and each accessor its own synchronous cudaMemcpy
        // of four bytes. A synchronous copy of a scalar costs about 24 microseconds on this host
        // whatever its size, and the result path reads ten of them per evolve call. Laid out
        // contiguously, counters_host() fetches the lot in one copy; the individual accessors
        // remain for ad-hoc use and now index the block.
        HG_CUDA_CHECK(cudaMalloc(&counters_, sizeof(uint32_t) * kNumCounters),
              "QeState counters alloc");
        cursor_            = counters_ + 0;
        next_id_           = counters_ + 1;
        inst_next_id_      = counters_ + 2;
        next_raw_event_    = counters_ + 3;
        align_moved_       = counters_ + 4;
        align_fail_        = counters_ + 5;
        num_canon_         = counters_ + 6;
        num_causal_pairs_  = counters_ + 7;
        num_causal_edges_  = counters_ + 8;
        num_branchial_     = counters_ + 9;
        event_sig_capacity_ = on ? max_events : 1u;
        HG_CUDA_CHECK(cudaMalloc(&event_sig_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event sig alloc");
        HG_CUDA_CHECK(cudaMalloc(&event_runsig_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event runsig alloc");
        clear();
    }

QeState::~QeState() {
    if (work_items_) cudaFree(work_items_);
        if (arr_)     cudaFree(arr_);
        if (counters_) cudaFree(counters_);
        if (event_sig_) cudaFree(event_sig_);
        if (event_runsig_) cudaFree(event_runsig_);
    }

bool QeState::enabled() const { return on_; }

void QeState::clear() {
        frame_.clear();
        by_from_.clear();
        matches_.reset();
        by_key_.clear();
        instances_.reset();
        rep_.clear();
        applied_.clear();
        canon_seen_.clear();
        causal_pairs_.clear();
        inst_applied_.clear();
        HG_CUDA_CHECK(cudaMemset(inst_next_id_, 0, sizeof(uint32_t)), "QeState inst id clear");
        HG_CUDA_CHECK(cudaMemset(next_raw_event_, 0, sizeof(uint32_t)), "QeState raw ev clear");
        HG_CUDA_CHECK(cudaMemset(align_moved_, 0, sizeof(uint32_t)), "QeState align moved clear");
        HG_CUDA_CHECK(cudaMemset(align_fail_, 0, sizeof(uint32_t)), "QeState align fail clear");
        HG_CUDA_CHECK(cudaMemset(num_canon_, 0, sizeof(uint32_t)), "QeState canon clear");
        HG_CUDA_CHECK(cudaMemset(num_causal_pairs_, 0, sizeof(uint32_t)), "QeState c-pairs clear");
        HG_CUDA_CHECK(cudaMemset(num_causal_edges_, 0, sizeof(uint32_t)), "QeState c-edges clear");
        HG_CUDA_CHECK(cudaMemset(num_branchial_, 0, sizeof(uint32_t)), "QeState branchial clear");
        HG_CUDA_CHECK(cudaMemset(event_sig_, 0, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event sig clear");
        HG_CUDA_CHECK(cudaMemset(event_runsig_, 0, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event runsig clear");
        HG_CUDA_CHECK(cudaMemset(cursor_, 0, sizeof(uint32_t)), "QeState cursor clear");
        HG_CUDA_CHECK(cudaMemset(next_id_, 0, sizeof(uint32_t)), "QeState next_id clear");
    }

QeState::Counters QeState::counters_host() const {
        uint32_t v[kNumCounters] = {};
        HG_CUDA_CHECK(cudaMemcpy(v, counters_, sizeof(v), cudaMemcpyDeviceToHost),
              "QeState counters read");
        return Counters{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]};

    }

uint32_t QeState::num_matches_host() { return matches_.size_host(); }

uint32_t QeState::num_raw_events_host() { return read_counter(next_raw_event_, "QeState raw event read"); }

uint32_t QeState::num_causal_pairs_host() { return read_counter(num_causal_pairs_, "QeState c-pairs read"); }

uint32_t QeState::num_causal_edges_host() { return read_counter(num_causal_edges_, "QeState c-edges read"); }

uint32_t QeState::num_branchial_host() { return read_counter(num_branchial_, "QeState branchial read"); }

void QeState::reconstructed_pairs_host(std::vector<std::pair<uint64_t, uint64_t>>& causal,
                                  std::vector<std::pair<uint64_t, uint64_t>>& causal_reduced,
                                  std::vector<std::pair<uint64_t, uint64_t>>& branchial,
                                  bool want_branchial,
                                  std::vector<uint64_t>* event_signature,
                                  std::vector<std::pair<uint32_t, uint32_t>>* causal_raw,
                                  std::vector<std::pair<uint32_t, uint32_t>>* causal_raw_reduced,
                                  std::vector<std::pair<uint32_t, uint32_t>>* branchial_raw) {
        causal.clear();
        causal_reduced.clear();
        branchial.clear();
        if (causal_raw) causal_raw->clear();
        if (causal_raw_reduced) causal_raw_reduced->clear();
        if (branchial_raw) branchial_raw->clear();
        const uint32_t n = num_raw_events_host();
        if (n == 0) return;
        std::vector<uint64_t> sigs(event_sig_capacity_);
        HG_CUDA_CHECK(cudaMemcpy(sigs.data(), event_sig_,
                                 sizeof(uint64_t) * event_sig_capacity_, cudaMemcpyDeviceToHost),
                      "QeState event sig read");
        auto sig_of = [&](uint32_t e) -> uint64_t {
            return e < sigs.size() ? sigs[e] : 0ull;
        };
        // Handed back whole, so a caller can identify an EVENT the same way the relations
        // identify their endpoints rather than by a second convention.
        if (event_signature) {
            // The RUN identity, not the content triple: observable_num_events counts distinct
            // values of THIS, so a graph grouped by it has the vertex set the count describes.
            std::vector<uint64_t> rsigs(event_sig_capacity_);
            if (event_sig_capacity_)
                HG_CUDA_CHECK(cudaMemcpy(rsigs.data(), event_runsig_,
                                         sizeof(uint64_t) * event_sig_capacity_,
                                         cudaMemcpyDeviceToHost), "QeState event runsig read");
            event_signature->assign(rsigs.begin(),
                                    rsigs.begin() + std::min<size_t>(rsigs.size(), n));
        }
        auto drain = [&](DedupMap& m, std::vector<std::pair<uint64_t, uint64_t>>& out) {
            std::vector<uint64_t> keys;
            m.copy_keys_to_host(keys);
            out.reserve(keys.size());
            for (uint64_t k : keys) {
                const hgcommon::IdPair p = hgcommon::id_pair_from_key(k);
                out.emplace_back(sig_of(p.a), sig_of(p.b));
                if (causal_raw)
                    causal_raw->emplace_back(static_cast<uint32_t>(p.a),
                                             static_cast<uint32_t>(p.b));
            }
        };
        drain(causal_pairs_, causal);

        // THE REDUCED VIEW, from the same stored relation and the same rule the host engine
        // uses. It is computed here rather than on the device for two reasons: which pairs
        // survive is a property of the FINISHED relation, and a device that tagged each pair as
        // it landed would answer against whatever the warps had produced so far; and the
        // reduction runs over event IDS, whose order carries the reachability prune, while
        // these vectors carry signatures and two events may share one.
        std::vector<uint64_t> ckeys;
        causal_pairs_.copy_keys_to_host(ckeys);
        hgcommon::tr_reduce(
            [&](auto&& add) {
                for (uint64_t k : ckeys) {
                    const hgcommon::IdPair p = hgcommon::id_pair_from_key(k);
                    add(static_cast<uint32_t>(p.a), static_cast<uint32_t>(p.b));
                }
            },
            [&](uint32_t a, uint32_t b) {
                causal_reduced.emplace_back(sig_of(a), sig_of(b));
                if (causal_raw_reduced) causal_raw_reduced->emplace_back(a, b);
            },
            // A producer wrote the slot its consumer reads, so its application minted the
            // lower id: ids increase along every causal edge of this relation.
            /*ids_topological=*/true);

        if (!want_branchial) return;


        // BRANCHIAL, DERIVED FROM THE APPLICATIONS rather than stored as pairs. A branchial pair
        // is two applications of ONE instance sharing a consumed slot, so the applications are
        // the relation in the form the replay generates it, and the pair list is an expansion of
        // them -- 970,584 against 133,218,996 on the host's disc-l3a2g2r2 depth 3. Storing that
        // expansion on the device is what its 2^22 map ceiling was, and truncating it returned a
        // partial relation with a warning rather than an answer.
        //
        // Order does not matter here, only the SET, so the host groups by instance and takes
        // each unordered pair once. The device's own counter is incremented per emission under
        // the strictly-earlier scan rule, so it and this enumeration are two routes to one
        // number and disagreeing is a defect either can catch.
        std::vector<LockFreeList<QeAppliedMatch>::Node> nodes;
        inst_applied_.copy_nodes_to_host(nodes);
        std::vector<uint32_t> slots(arr_cap_);
        if (arr_cap_)
            HG_CUDA_CHECK(cudaMemcpy(slots.data(), arr_, sizeof(uint32_t) * arr_cap_,
                                     cudaMemcpyDeviceToHost), "QeState arr read");

        std::unordered_map<uint32_t, std::vector<const QeAppliedMatch*>> by_instance;
        for (const auto& nd : nodes) by_instance[nd.value.instance].push_back(&nd.value);
        for (const auto& kv : by_instance) {
            const auto& v = kv.second;
            for (size_t i = 0; i < v.size(); ++i) {
                for (size_t j = i + 1; j < v.size(); ++j) {
                    const QeAppliedMatch& a = *v[i];
                    const QeAppliedMatch& b = *v[j];
                    if (a.event == b.event) continue;
                    bool overlaps = false;
                    for (uint32_t x = 0; x < a.num_consumed && !overlaps; ++x) {
                        const uint32_t ax = a.consumed_offset + x;
                        if (ax >= slots.size()) break;
                        for (uint32_t y = 0; y < b.num_consumed; ++y) {
                            const uint32_t by = b.consumed_offset + y;
                            if (by >= slots.size()) break;
                            if (slots[ax] == slots[by]) { overlaps = true; break; }
                        }
                    }
                    if (!overlaps) continue;
                    const uint32_t lo = a.event < b.event ? a.event : b.event;
                    const uint32_t hi = a.event < b.event ? b.event : a.event;
                    branchial.emplace_back(sig_of(lo), sig_of(hi));
                    if (branchial_raw) branchial_raw->emplace_back(lo, hi);
                }
            }
        }

    }

uint32_t QeState::num_canon_events_host() { return read_counter(num_canon_, "QeState canon read"); }

uint32_t QeState::num_aligned_host() { return read_counter(align_moved_, "QeState align moved read"); }

uint32_t QeState::num_align_failures_host() { return read_counter(align_fail_, "QeState align fail read"); }

uint32_t QeState::num_instances_host() { return instances_.size_host(); }

// The replay descends through this rather than through the call stack, so its size is what
// bounds reconstruction depth. Sized from the run: `slices` drivers, each holding enough items
// for a depth-first walk of `max_steps` levels with room for the siblings a level fans out to.
//
// 64 items per level is a bound on FAN-OUT, not on depth -- the stack holds the matches of each
// level that have not been descended into yet. A workload wider than that reports a capacity
// overflow and returns partial work, which is the same contract every other pool here has, and
// unlike the per-thread stack it can be raised without costing every resident thread.
void QeState::ensure_work(uint32_t slices, uint32_t max_steps) {
    if (!on_) return;
    const uint32_t cap = max_steps * 64u < 256u ? 256u : max_steps * 64u;
    if (work_items_ && work_slices_ >= slices && work_cap_ >= cap) return;
    if (work_items_) { cudaFree(work_items_); work_items_ = nullptr; }
    work_slices_ = slices > work_slices_ ? slices : work_slices_;
    work_cap_    = cap > work_cap_ ? cap : work_cap_;
    const size_t bytes = sizeof(QeWorkItem) * work_slices_ * work_cap_;
    HG_CUDA_CHECK(cudaMalloc(&work_items_, bytes), "QeState descent stacks alloc");
}

QeView QeState::view(uint32_t max_steps, EventSignatureKeys keys,
                bool replay) {
        QeView q{};
        q.matches      = matches_.view();
        q.by_from      = by_from_.view();
        q.instances      = instances_.view();
        q.by_key         = by_key_.view();
        q.inst_next_id   = inst_next_id_;
        q.rep            = rep_.view();
        q.applied        = applied_.view();
        q.align_moved    = align_moved_;
        q.canon_seen     = canon_seen_.view();
        q.num_canon      = num_canon_;
        q.event_sig        = event_sig_;
        q.event_runsig     = event_runsig_;
        q.event_sig_capacity = event_sig_capacity_;
        q.inst_applied     = inst_applied_.view();
        q.num_branchial    = num_branchial_;
        q.causal_pairs   = causal_pairs_.view();
        q.num_causal_pairs = num_causal_pairs_;
        q.num_causal_edges = num_causal_edges_;
        q.keys           = keys;
        q.align_fail     = align_fail_;
        q.next_raw_event = next_raw_event_;
        q.frame        = frame_.view();
        q.arr_words    = arr_;
        q.arr_cursor   = cursor_;
        q.arr_capacity = arr_cap_;
        q.next_id      = next_id_;
        q.max_steps    = max_steps;
        q.work_items   = work_items_;
        q.work_cap     = work_cap_;
        q.work_slices  = work_slices_;
        q.enabled      = on_ ? 1u : 0u;
        q.replay       = (on_ && replay) ? 1u : 0u;
        return q;
    }

uint32_t QeState::read_counter(const uint32_t* p, const char* what) {
        uint32_t v = 0;
        HG_CUDA_CHECK(cudaMemcpy(&v, p, sizeof(uint32_t), cudaMemcpyDeviceToHost), what);
        return v;
    }

}  // namespace gpu
}  // namespace HG_NAMESPACE
