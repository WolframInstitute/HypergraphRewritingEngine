#include "hg_gpu/quotient_causal.hpp"
#include "hg_gpu/quotient_expansion.hpp"

// The HOST bodies of the two quotient state objects: allocation, clear, the counter readbacks
// and the view() calls that hand a device-side struct to a kernel. The DP and the replay
// themselves are __device__ and stay in their headers, as does everything the shared
// hgcommon cores reach.

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

// =============================================================================
// QeState
// =============================================================================

// Buckets for the replay's keyed lists, grown with the event budget. A walk visits every node in
// its bucket, other keys' included, so a fixed count makes each walk proportional to the pool:
// at 2^16 buckets allfour's captures walked tens of foreign instances per depth. Capped at 2^20
// because the heads are cleared every run.
static uint32_t qe_list_buckets(uint32_t max_events) {
    uint32_t n = 1u << 16;
    while (n < max_events && n < (1u << 20)) n <<= 1;
    return n;
}

// Each table is sized from its group in `n` (EngineConfig::qe_class_entries and the four after
// it), and its overflow reports that group's kind.
// One thread per multiplicity point, or per class with replay instances; each sums its terms
// locally and adds once.
__global__ void k_qe_count_branchial(const __grid_constant__ DeviceState ds, const QeView qe,
                                     uint32_t multiplicity) {
    const uint32_t steps = qe.max_steps;
    unsigned long long local = 0;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    if (multiplicity) {
        const uint32_t n = qe.qm_cursor[0] < qe.qm_capacity ? qe.qm_cursor[0] : qe.qm_capacity;
        for (uint32_t p = tid; p < n; p += stride) {
            if (qe.qm_point_depth[p] >= steps) continue;
            const auto r = qe.rep.lookup(qe.qm_point_class[p]);
            if (!r.found || r.value - 1u >= qe.class_nmatch_cap) continue;
            const uint64_t b = qe.class_pairs[r.value - 1u];
            if (b) local = hgcommon::qm_branchial_add(local, qe.qm_mass[p], b);
        }
    } else {
        // A class's representative is a raw state, so its index is a state id.
        for (uint32_t c = tid; c < qe.class_nmatch_cap; c += stride) {
            const uint64_t b = qe.class_pairs[c];
            if (!b) continue;
            const uint64_t h = ds.state_canonical_hash[c];
            for (uint32_t d = 0; d < steps; ++d) {
                uint64_t count = 0;
                qe_for_each_instance(qe, h, d, [&](const DeviceQcInstance&) { ++count; });
                local = hgcommon::qm_branchial_add(local, count, b);
            }
        }
    }
    if (local) qe_qm_add(&qe.qm_counts[1], local);
}

void qe_count_branchial(const DeviceState& ds, const QeView& qe, bool multiplicity) {
    HG_CUDA_CHECK(cudaMemsetAsync(qe.qm_counts + 1, 0, sizeof(unsigned long long)),
                  "QeState branchial count clear");
    k_qe_count_branchial<<<1024, 128>>>(ds, qe, multiplicity ? 1u : 0u);
    HG_CUDA_CHECK(cudaGetLastError(), "QeState branchial count launch");
}

QeState::QeState(bool on, const QeEntries& n): matches_(on ? n.classes : 1u),
          by_from_(on ? qe_list_buckets(n.classes) : 1u, on ? n.classes : 1u),
          instances_(on ? n.instances : 1u),
          blocked_(on ? n.instances : 1u),
          // An application is attempted once from each side of the rendezvous at most.
          tasks_(on ? n.events * 2u : 1u),
          by_key_(on ? qe_list_buckets(n.instances) : 1u, on ? n.instances : 1u),
          rep_(on ? n.classes : 8u),
          applied_(on ? n.pairs * 4u : 8u),
          canon_seen_(on ? n.classes * 2u : 8u),
          causal_pairs_(on ? n.pairs * 4u : 8u),
          qm_points_(on ? n.classes * 2u : 8u),
          qm_consumed_(on ? n.classes * 2u : 8u),
          qm_overlaps_(on ? n.classes * 2u : 8u),
          qm_capacity_(on ? n.classes : 1u),
          inst_applied_(on ? qe_list_buckets(n.events) : 1u, on ? n.events * 2u : 1u),
          frame_(on ? n.classes * 2u : 8u),
          arr_cap_(on ? n.words : 1u),
          on_(on) {
        HG_CUDA_CHECK(cudaMalloc(&arr_, sizeof(uint32_t) * arr_cap_), "QeState arr alloc");
        // THE SCALARS IN ONE BLOCK, so the host reads them in ONE transfer.
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
        // counters_ + 9 is unused.
        num_reduced_pairs_ = counters_ + 12;
        // counters_ + 10 and + 11: the multiplicity point and consumed-cell cursors.
        HG_CUDA_CHECK(cudaMalloc(&qm_words_, sizeof(unsigned long long) * (2ull * qm_capacity_ + 2u)),
                      "QeState multiplicity alloc");
        HG_CUDA_CHECK(cudaMalloc(&qm_queued_, sizeof(uint32_t) * qm_capacity_),
                      "QeState multiplicity queue flags alloc");
        HG_CUDA_CHECK(cudaMalloc(&qm_point_class_, sizeof(unsigned long long) * qm_capacity_),
                      "QeState multiplicity point class alloc");
        class_nmatch_cap_ = on ? n.states : 1u;
        HG_CUDA_CHECK(cudaMalloc(&class_nmatch_, sizeof(uint32_t) * class_nmatch_cap_),
                      "QeState class match counts alloc");
        HG_CUDA_CHECK(cudaMemset(class_nmatch_, 0, sizeof(uint32_t) * class_nmatch_cap_),
                      "QeState class match counts init");
        HG_CUDA_CHECK(cudaMalloc(&class_pairs_, sizeof(unsigned long long) * class_nmatch_cap_),
                      "QeState class pairs alloc");
        HG_CUDA_CHECK(cudaMalloc(&qm_point_depth_, sizeof(uint32_t) * qm_capacity_),
                      "QeState multiplicity point depth alloc");
        event_sig_capacity_ = on ? n.events : 1u;
        HG_CUDA_CHECK(cudaMalloc(&event_sig_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event sig alloc");
        HG_CUDA_CHECK(cudaMalloc(&event_runsig_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event runsig alloc");
        HG_CUDA_CHECK(cudaMalloc(&event_kept_,
                                 sizeof(uint32_t) * kQeKeptStride * size_t(event_sig_capacity_)),
                      "QeState event kept alloc");
        clear();
    }

QeState::~QeState() {
    if (work_items_) cudaFree(work_items_);
    if (lane_reach_) cudaFree(lane_reach_);
        if (arr_)     cudaFree(arr_);
        if (counters_) cudaFree(counters_);
        if (event_sig_) cudaFree(event_sig_);
        if (event_runsig_) cudaFree(event_runsig_);
        if (event_kept_) cudaFree(event_kept_);
        if (class_nmatch_) cudaFree(class_nmatch_);
        if (class_pairs_) cudaFree(class_pairs_);
        if (event_from_class_) cudaFree(event_from_class_);
        if (event_to_class_) cudaFree(event_to_class_);
        if (event_rule_) cudaFree(event_rule_);
        if (qm_words_) cudaFree(qm_words_);
        if (qm_queued_) cudaFree(qm_queued_);
        if (qm_point_class_) cudaFree(qm_point_class_);
        if (qm_point_depth_) cudaFree(qm_point_depth_);
    }

bool QeState::enabled() const { return on_; }

void QeState::clear() {
        // The per-event arrays are written below the raw event counter, so the previous run's
        // count bounds what this clear has to cover; the first clear, from the constructor,
        // covers them in full.
        uint32_t prev[kNumCounters] = {};
        HG_CUDA_CHECK(cudaMemcpy(prev, counters_, sizeof(prev), cudaMemcpyDeviceToHost),
                      "QeState counters read for clear");
        const uint32_t events = !cleared_once_ || prev[3] > event_sig_capacity_
                                    ? event_sig_capacity_ : prev[3];

        // Every region below is cleared by one kernel launch (ClearBatch).
        ClearBatch batch;
        frame_.clear(&batch);
        by_from_.clear(0xFFFFFFFFu, &batch);
        matches_.reset(&batch);
        by_key_.clear(0xFFFFFFFFu, &batch);
        instances_.reset(&batch);
        blocked_.reset(&batch);
        // Published flags start clear; reset_and_clear zeroes the prefix the last run wrote.
        tasks_.reset_and_clear(&batch);
        // Every scalar counter and cursor restarts at zero.
        batch.add(counters_, sizeof(uint32_t) * kNumCounters, 0);
        // The slices it names are in the arena, which restarts with this run.
        if (lane_reach_) batch.add(lane_reach_, sizeof(uint32_t) * lane_reach_slots_, 0);
        rep_.clear(&batch);
        applied_.clear(&batch);
        canon_seen_.clear(&batch);
        causal_pairs_.clear(&batch);
        qm_points_.clear(&batch);
        qm_consumed_.clear(&batch);
        qm_overlaps_.clear(&batch);
        // The masses, cells and flags are zeroed by whoever claims them; only the counts restart.
        batch.add(qm_words_ + 2ull * qm_capacity_, sizeof(unsigned long long) * 2u, 0);
        inst_applied_.clear(0xFFFFFFFFu, &batch);
        batch.add(class_nmatch_, sizeof(uint32_t) * class_nmatch_cap_, 0);
        batch.add(class_pairs_, sizeof(unsigned long long) * class_nmatch_cap_, 0);
        batch.add(event_kept_, sizeof(uint32_t) * kQeKeptStride * size_t(events), 0);
        batch.add(event_sig_, sizeof(uint64_t) * events, 0);
        batch.add(event_runsig_, sizeof(uint64_t) * events, 0);
        batch.flush();
        cleared_once_ = true;
    }

QeState::Counters QeState::counters_host(bool multiplicity) const {
        uint32_t v[kNumCounters] = {};
        HG_CUDA_CHECK(cudaMemcpy(v, counters_, sizeof(v), cudaMemcpyDeviceToHost),
              "QeState counters read");
        unsigned long long q[2] = {};
        if (multiplicity)
            HG_CUDA_CHECK(cudaMemcpy(q, qm_words_ + 2ull * qm_capacity_, sizeof(q),
                                     cudaMemcpyDeviceToHost), "QeState multiplicity counts read");
        return counters_from(v, q);
    }

QeState::Counters QeState::counters_from(const uint32_t* v, const unsigned long long* q) {
        return Counters{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[12],
                        q[0], q[1]};

    }

uint32_t QeState::num_matches_host() { return matches_.size_host(); }

void QeState::class_multiplicities_host(std::vector<ClassMultiplicity>& points,
                                        std::vector<ClassRuleMatches>& matches) {
    points.clear();
    matches.clear();
    uint32_t cursor = 0;
    HG_CUDA_CHECK(cudaMemcpy(&cursor, counters_ + 10, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "QeState multiplicity cursor read");
    const uint32_t n = cursor < qm_capacity_ ? cursor : qm_capacity_;
    std::vector<unsigned long long> mass(n), cls(n);
    std::vector<uint32_t> depth(n);
    if (n) {
        HG_CUDA_CHECK(cudaMemcpy(mass.data(), qm_words_, sizeof(unsigned long long) * n,
                                 cudaMemcpyDeviceToHost), "QeState multiplicity mass read");
        HG_CUDA_CHECK(cudaMemcpy(cls.data(), qm_point_class_, sizeof(unsigned long long) * n,
                                 cudaMemcpyDeviceToHost), "QeState multiplicity class read");
        HG_CUDA_CHECK(cudaMemcpy(depth.data(), qm_point_depth_, sizeof(uint32_t) * n,
                                 cudaMemcpyDeviceToHost), "QeState multiplicity depth read");
    }
    // An index claimed by a thread that lost the map insert was never credited, so it holds 0.
    for (uint32_t i = 0; i < n; ++i)
        if (mass[i]) points.push_back({cls[i], depth[i], mass[i]});

    std::vector<LockFreeList<QeMatchRef>::Node> refs;
    by_from_.copy_nodes_to_host(refs);
    std::vector<DeviceSlotMatch> recs;
    matches_.copy_to_host(recs);
    std::map<std::pair<uint64_t, uint32_t>, uint64_t> count;
    for (const auto& r : refs)
        if (r.value.record < recs.size()) ++count[{r.value.from_hash, recs[r.value.record].rule}];
    for (const auto& [k, c] : count) matches.push_back({k.first, k.second, c});
}

uint32_t QeState::num_raw_events_host() {
    const uint32_t n = read_counter(next_raw_event_, "QeState raw event read");
    return n < id_limit_ ? n : id_limit_;
}

uint32_t QeState::num_causal_pairs_host() { return read_counter(num_causal_pairs_, "QeState c-pairs read"); }
uint32_t QeState::num_reduced_pairs_host() { return read_counter(num_reduced_pairs_, "QeState reduced read"); }

uint32_t QeState::num_causal_edges_host() { return read_counter(num_causal_edges_, "QeState c-edges read"); }


void QeState::event_signature_host(std::vector<uint64_t>& event_signature, uint32_t raw_events) {
        const uint32_t written = std::min(raw_events, event_sig_capacity_);
        event_signature.resize(written);
        if (written)
            HG_CUDA_CHECK(cudaMemcpy(event_signature.data(), event_runsig_,
                                     sizeof(uint64_t) * written, cudaMemcpyDeviceToHost),
                          "QeState event runsig read");
    }

void QeState::reconstructed_pairs_host(std::vector<std::pair<uint64_t, uint64_t>>& causal,
                                  std::vector<std::pair<uint64_t, uint64_t>>& causal_reduced,
                                  std::vector<std::pair<uint64_t, uint64_t>>& branchial,
                                  bool want_branchial,
                                  uint32_t raw_events,
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
        const uint32_t n = raw_events;
        if (n == 0) return;
        // The events written, not the reservation: ids at or above n were never minted.
        const uint32_t written = std::min(n, event_sig_capacity_);
        std::vector<uint64_t> sigs(written);
        if (written)
            HG_CUDA_CHECK(cudaMemcpy(sigs.data(), event_sig_, sizeof(uint64_t) * written,
                                     cudaMemcpyDeviceToHost), "QeState event sig read");
        auto sig_of = [&](uint32_t e) -> uint64_t {
            return e < sigs.size() ? sigs[e] : 0ull;
        };
        // Handed back whole, so a caller can identify an EVENT the same way the relations
        // identify their endpoints rather than by a second convention.
        if (event_signature) {
            // The RUN identity, not the content triple: observable_num_events counts distinct
            // values of THIS, so a graph grouped by it has the vertex set the count describes.
            event_signature->resize(written);
            if (written)
                HG_CUDA_CHECK(cudaMemcpy(event_signature->data(), event_runsig_,
                                         sizeof(uint64_t) * written,
                                         cudaMemcpyDeviceToHost), "QeState event runsig read");
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

        // THE REDUCED VIEW, as the replay kept it: each event's kept producers, decided online in
        // qr_apply by the rule the host engine uses (hgcommon::redundant_producers).
        {
            const uint32_t m = written;
            std::vector<uint32_t> kept(size_t(kQeKeptStride) * m);
            if (m)
                HG_CUDA_CHECK(cudaMemcpy(kept.data(), event_kept_, sizeof(uint32_t) * kept.size(),
                                         cudaMemcpyDeviceToHost), "QeState event kept read");
            std::vector<uint32_t> spill;
            auto spill_word = [&](uint32_t off) -> uint32_t {
                if (spill.empty()) {
                    uint32_t used = 0;
                    HG_CUDA_CHECK(cudaMemcpy(&used, cursor_, sizeof(uint32_t),
                                             cudaMemcpyDeviceToHost), "QeState cursor read");
                    // An allocation that overflowed advanced the cursor past the arena and
                    // wrote nothing; the words that exist end at the capacity.
                    used = std::min(used, arr_cap_);
                    spill.resize(std::max<uint32_t>(used, 1u));
                    HG_CUDA_CHECK(cudaMemcpy(spill.data(), arr_, sizeof(uint32_t) * used,
                                             cudaMemcpyDeviceToHost), "QeState arr read");
                }
                return off < spill.size() ? spill[off] : 0u;
            };
            for (uint32_t c = 0; c < m; ++c) {
                const uint32_t* k = kept.data() + size_t(kQeKeptStride) * c;
                for (uint32_t i = 0; i < k[0]; ++i) {
                    const uint32_t a2 = i < 3u ? k[1 + i] : spill_word(k[4] + (i - 3u));
                    causal_reduced.emplace_back(sig_of(a2), sig_of(c));
                    if (causal_raw_reduced) causal_raw_reduced->emplace_back(a2, c);
                }
            }
        }

        if (!want_branchial) return;


        // BRANCHIAL, DERIVED FROM THE APPLICATIONS rather than stored as pairs. A branchial pair
        // is two applications of ONE instance sharing a consumed slot, so the applications are
        // the relation in the form the replay generates it, and the pair list is an expansion of
        // them -- 970,584 against 133,218,996 on the host's disc-l3a2g2r2 depth 3. Storing that
        // expansion on the device is what its 2^22 map ceiling was, and truncating it returned a
        // partial relation with a warning rather than an answer.
        //
        // Grouped by instance and paired by hgcommon::qr_instance_branchial_pairs, the function
        // the host engine's readback calls.
        std::vector<LockFreeList<QeAppliedMatch>::Node> nodes;
        inst_applied_.copy_nodes_to_host(nodes);
        // The arena prefix the run filled, not its capacity.
        uint32_t used = 0;
        HG_CUDA_CHECK(cudaMemcpy(&used, cursor_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                      "QeState cursor read");
        used = std::min(used, arr_cap_);
        std::vector<uint32_t> slots(used);
        if (used)
            HG_CUDA_CHECK(cudaMemcpy(slots.data(), arr_, sizeof(uint32_t) * used,
                                     cudaMemcpyDeviceToHost), "QeState arr read");

        // An application bound to the copied arena; consumed slots past the copy are not read.
        struct App {
            uint32_t event, num_consumed;
            const uint32_t* s;
            uint32_t consumed(uint32_t j) const { return s[j]; }
        };
        std::unordered_map<uint32_t, std::vector<App>> by_instance;
        for (const auto& nd : nodes) {
            const QeAppliedMatch& a = nd.value;
            const uint32_t off = a.consumed_offset < used ? a.consumed_offset : used;
            const uint32_t nc = std::min(a.num_consumed, used - off);
            by_instance[a.instance].push_back(App{a.event, nc, slots.data() + off});
        }
        std::vector<std::pair<uint32_t, uint32_t>> entries;
        for (const auto& kv : by_instance)
            hgcommon::qr_instance_branchial_pairs(
                kv.second.data(), static_cast<uint32_t>(kv.second.size()), entries,
                [&](uint32_t lo, uint32_t hi) {
                    branchial.emplace_back(sig_of(lo), sig_of(hi));
                    if (branchial_raw) branchial_raw->emplace_back(lo, hi);
                });

    }

uint32_t QeState::num_canon_events_host() { return read_counter(num_canon_, "QeState canon read"); }

uint32_t QeState::num_aligned_host() { return read_counter(align_moved_, "QeState align moved read"); }

uint32_t QeState::num_align_failures_host() { return read_counter(align_fail_, "QeState align fail read"); }

uint32_t QeState::num_instances_host() { return instances_.size_host(); }

// The multiplicity cascade's queues. Sized from the run: `slices` drivers, each holding 64
// items per level of `max_steps`, at least 256. A workload wider than that reports a capacity
// overflow (kQeWorkOverflow); grow-and-retry doubles `scale` (EngineConfig::descent_work_scale)
// and runs again.
void QeState::ensure_work(uint32_t slices, uint32_t max_steps, uint32_t scale) {
    if (!on_) return;
    // In 64 bits: a slice holds at most 2^32 - 1 items, its index is 32 bits on the device.
    const uint64_t per_level = uint64_t{max_steps} * 64u;
    const uint64_t want = (per_level < 256u ? 256u : per_level) * uint64_t{scale};
    if (want > UINT32_MAX)
        throw std::length_error("multiplicity queues of " + std::to_string(want) +
                                " items per driver are past 2^32");
    const uint32_t cap = static_cast<uint32_t>(want);
    if (work_items_ && work_slices_ >= slices && work_cap_ >= cap) return;
    if (work_items_) { cudaFree(work_items_); work_items_ = nullptr; }
    work_slices_ = slices > work_slices_ ? slices : work_slices_;
    work_cap_    = cap > work_cap_ ? cap : work_cap_;
    const size_t bytes = sizeof(QeWorkItem) * size_t{work_slices_} * size_t{work_cap_};
    HG_CUDA_CHECK(cudaMalloc(&work_items_, bytes), "QeState multiplicity queues alloc");
}

void QeState::ensure_lanes(uint32_t lanes) {
    if (!on_ || lanes <= lane_reach_slots_) return;
    if (lane_reach_) cudaFree(lane_reach_);
    lane_reach_ = nullptr;
    lane_reach_slots_ = 0;
    HG_CUDA_CHECK(cudaMalloc(&lane_reach_, sizeof(uint32_t) * lanes), "QeState lane reach alloc");
    HG_CUDA_CHECK(cudaMemset(lane_reach_, 0, sizeof(uint32_t) * lanes), "QeState lane reach init");
    lane_reach_slots_ = lanes;
}

void QeState::ensure_event_content() {
        if (event_from_class_ || !on_) return;
        HG_CUDA_CHECK(cudaMalloc(&event_from_class_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event from-class alloc");
        HG_CUDA_CHECK(cudaMalloc(&event_to_class_, sizeof(uint64_t) * event_sig_capacity_),
                      "QeState event to-class alloc");
        HG_CUDA_CHECK(cudaMalloc(&event_rule_, sizeof(uint32_t) * event_sig_capacity_),
                      "QeState event rule alloc");
}

void QeState::reconstructed_event_content_host(std::vector<uint64_t>& from_class,
                                               std::vector<uint64_t>& to_class,
                                               std::vector<uint32_t>& rule) {
        from_class.clear();
        to_class.clear();
        rule.clear();
        if (!event_from_class_) return;
        const uint32_t n = std::min(num_raw_events_host(), event_sig_capacity_);
        from_class.resize(n);
        to_class.resize(n);
        rule.resize(n);
        if (n == 0) return;
        HG_CUDA_CHECK(cudaMemcpy(from_class.data(), event_from_class_, sizeof(uint64_t) * n,
                                 cudaMemcpyDeviceToHost), "QeState event from-class read");
        HG_CUDA_CHECK(cudaMemcpy(to_class.data(), event_to_class_, sizeof(uint64_t) * n,
                                 cudaMemcpyDeviceToHost), "QeState event to-class read");
        HG_CUDA_CHECK(cudaMemcpy(rule.data(), event_rule_, sizeof(uint32_t) * n,
                                 cudaMemcpyDeviceToHost), "QeState event rule read");
}

namespace {

// hgcommon::qr_producer_of's face over the copied records. A record whose match or
// parent lies past the copy, or whose slot arrays lie past the filled arena, ends the
// walk at "no producer".
struct GenesisCtx {
    const DeviceQcInstance* inst;
    size_t n_inst;
    const DeviceSlotMatch* matches;
    size_t n_matches;
    const uint32_t* words;
    size_t n_words;
    __host__ __device__ bool readable(const DeviceSlotMatch& m) const {
        const uint64_t end = uint64_t{m.arr_offset} + m.num_consumed + m.num_produced +
                             2ull * m.num_survivors + m.to_slots;
        return end <= n_words;
    }
    __host__ __device__ bool lineage_root(const DeviceQcInstance* n) const {
        return n->parent == kQeNoParent || n->parent >= n_inst ||
               n->via >= n_matches || !readable(matches[n->via]);
    }
    __host__ __device__ uint32_t lineage_source(const DeviceQcInstance* n,
                                                uint32_t slot) const {
        const QeMatchView m(matches[n->via], words);
        return slot < m.to_slots ? m.child_source(slot) : hgcommon::QR_SOURCE_NONE;
    }
    __host__ __device__ uint32_t lineage_event(const DeviceQcInstance* n) const {
        return n->event;
    }
    __host__ __device__ const DeviceQcInstance* lineage_parent(
            const DeviceQcInstance* n) const {
        return inst + n->parent;
    }
};

}  // namespace

void QeState::reconstructed_genesis_pairs_host(bool reduced,
                                               std::vector<std::pair<uint32_t, uint32_t>>& out) {
        out.clear();
        if (!on_) return;
        std::vector<DeviceQcInstance> inst;
        instances_.copy_to_host(inst);
        if (inst.empty()) return;
        std::vector<DeviceSlotMatch> matches;
        matches_.copy_to_host(matches);
        // The arena prefix the run filled, not its capacity.
        uint32_t used = 0;
        HG_CUDA_CHECK(cudaMemcpy(&used, cursor_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                      "QeState cursor read");
        used = std::min(used, arr_cap_);
        std::vector<uint32_t> words(std::max<uint32_t>(used, 1u), 0u);
        if (used)
            HG_CUDA_CHECK(cudaMemcpy(words.data(), arr_, sizeof(uint32_t) * used,
                                     cudaMemcpyDeviceToHost), "QeState arr read");

        const GenesisCtx c{inst.data(), inst.size(), matches.data(), matches.size(),
                           words.data(), words.size()};

        for (const DeviceQcInstance& n : inst) {
            if (c.lineage_root(&n)) continue;
            const DeviceQcInstance* root = hgcommon::qr_lineage_root(c, &n);
            if (root->parent != kQeNoParent) continue;   // a lineage cut short by the copy
            const QeMatchView m(matches[n.via], words.data());
            const DeviceQcInstance* parent = inst.data() + n.parent;
            if (hgcommon::qr_genesis_paired(c, parent, m, reduced))
                out.emplace_back(root->event, n.event);
        }
}

QeView QeState::view(uint32_t max_steps, EventSignatureKeys keys,
                bool replay, bool multiplicity, bool event_content) {
        QeView q{};
        q.matches      = matches_.view();
        q.by_from      = by_from_.view();
        q.instances      = instances_.view();
        q.blocked        = blocked_.view();
        q.by_key         = by_key_.view();
        q.inst_next_id   = inst_next_id_;
        q.rep            = rep_.view();
        q.applied        = applied_.view();
        q.class_nmatch     = class_nmatch_;
        q.class_nmatch_cap = class_nmatch_cap_;
        q.class_pairs      = class_pairs_;
        q.align_moved    = align_moved_;
        q.canon_seen     = canon_seen_.view();
        q.num_canon      = num_canon_;
        q.event_sig        = event_sig_;
        q.event_runsig     = event_runsig_;
        q.event_sig_capacity = event_sig_capacity_;
        q.event_from_class = event_content ? event_from_class_ : nullptr;
        q.event_to_class   = event_content ? event_to_class_ : nullptr;
        q.event_rule       = event_content ? event_rule_ : nullptr;
        q.inst_applied     = inst_applied_.view();
        q.causal_pairs   = causal_pairs_.view();
        q.num_causal_pairs = num_causal_pairs_;
        q.num_causal_edges = num_causal_edges_;
        q.event_kept       = event_kept_;
        q.num_reduced_pairs = num_reduced_pairs_;
        q.keys           = keys;
        q.align_fail     = align_fail_;
        q.next_raw_event = next_raw_event_;
        q.frame        = frame_.view();
        q.arr_words    = arr_;
        q.arr_cursor   = cursor_;
        q.arr_capacity = arr_cap_;
        q.next_id      = next_id_;
        q.max_steps    = max_steps;
        q.tasks.items      = tasks_.view();
        q.tasks.cursor     = counters_ + 13;
        q.tasks.done       = counters_ + 14;
        q.lane_reach       = lane_reach_;
        q.lane_reach_slots = lane_reach_slots_;
        q.work_items   = work_items_;
        q.work_cap     = work_cap_;
        q.work_slices  = work_slices_;
        q.enabled      = on_ ? 1u : 0u;
        q.replay       = (on_ && replay) ? 1u : 0u;
        q.multiplicity = (on_ && multiplicity) ? 1u : 0u;
        q.qm_point_class    = qm_point_class_;
        q.qm_point_depth    = qm_point_depth_;
        q.qm_points         = qm_points_.view();
        q.qm_consumed       = qm_consumed_.view();
        q.qm_overlaps       = qm_overlaps_.view();
        q.qm_mass           = qm_words_;
        q.qm_consumed_cells = qm_words_ + qm_capacity_;
        q.qm_counts         = qm_words_ + 2ull * qm_capacity_;
        q.qm_queued         = qm_queued_;
        q.qm_cursor         = counters_ + 10;
        q.qm_capacity       = qm_capacity_;
        return q;
    }

uint32_t QeState::read_counter(const uint32_t* p, const char* what) {
        uint32_t v = 0;
        HG_CUDA_CHECK(cudaMemcpy(&v, p, sizeof(uint32_t), cudaMemcpyDeviceToHost), what);
        return v;
    }

}  // namespace gpu
}  // namespace HG_NAMESPACE
