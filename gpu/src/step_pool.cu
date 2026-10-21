#include "hg_gpu/step_pool.hpp"
#include "hg_gpu/cuda_check.hpp"

// The device side of docs/DEVICE_DESIGN.md 4.8's pooling: one thread per class point runs
// hgcommon::sp_add_class, writing its entries at the point's offset (a prefix sum of the entry
// counts the host takes with the same function) and adding its distribution totals with
// saturating atomics. The entries are sorted by key and then, stably, by (step, slot), and equal
// (step, slot, key) entries are merged with a saturating sum. Every step is the same multiset
// whichever thread wrote which entry, so the merged entries are a function of the points.

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

namespace {

// A merged entry's (step << 32 | slot, key).
struct PoolKey {
    uint64_t step_slot;
    uint64_t key;
    __host__ __device__ bool operator==(const PoolKey& o) const {
        return step_slot == o.step_slot && key == o.key;
    }
};

struct SatAdd {
    __device__ uint64_t operator()(uint64_t a, uint64_t b) const { return hgcommon::qm_sat_add(a, b); }
};

__device__ inline void sat_add(uint64_t* p, uint64_t v) {
    auto* q = reinterpret_cast<unsigned long long*>(p);
    unsigned long long seen = *q, assumed;
    do {
        assumed = seen;
        seen = atomicCAS(q, assumed, static_cast<unsigned long long>(hgcommon::qm_sat_add(assumed, v)));
    } while (seen != assumed);
}

struct DeviceSink {
    uint64_t step_slot_base;   // step << 32
    uint64_t* step_slot;
    uint64_t* key;
    uint64_t* weight;
    uint64_t at;
    hgcommon::SpDistTotals* totals;   // this step's SG_DISTS
    __device__ void entry(uint32_t slot, uint64_t k, uint64_t w) {
        step_slot[at] = step_slot_base | slot;
        key[at] = k;
        weight[at] = w;
        ++at;
    }
    __device__ void dist(uint32_t k, uint64_t n, uint64_t min_key, uint64_t max_key) {
        sat_add(&totals[k].n, n);
        atomicMin(reinterpret_cast<unsigned long long*>(&totals[k].min_key),
                  static_cast<unsigned long long>(min_key));
        atomicMax(reinterpret_cast<unsigned long long*>(&totals[k].max_key),
                  static_cast<unsigned long long>(max_key));
    }
    __device__ void bin(uint32_t k, uint32_t b, uint64_t n) { sat_add(&totals[k].bins[b], n); }
};

struct CountSink {
    uint64_t n = 0;
    HG_HD void entry(uint32_t, uint64_t, uint64_t) { ++n; }
    HG_HD void dist(uint32_t, uint64_t, uint64_t, uint64_t) {}
    HG_HD void bin(uint32_t, uint32_t, uint64_t) {}
};

__global__ void k_step_pool_emit(const uint64_t* pool, uint64_t none_word,
                                 const StepPoolPoint* points, const uint64_t* offset, uint64_t n,
                                 uint64_t* step_slot, uint64_t* key, uint64_t* weight,
                                 hgcommon::SpDistTotals* totals) {
    const uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const StepPoolPoint p = points[i];
    const uint64_t word = p.record_word == UINT64_MAX ? none_word : p.record_word;
    const auto& r = *reinterpret_cast<const hgcommon::StateInvariantRecord*>(pool + word);
    DeviceSink sink{static_cast<uint64_t>(p.step) << 32, step_slot, key, weight, offset[i],
                    totals + static_cast<size_t>(p.step) * hgcommon::SG_DISTS};
    hgcommon::sp_add_class(r, p.weight, p.raw, sink);
}

__global__ void k_gather(const uint64_t* step_slot, const uint64_t* key, const uint64_t* weight,
                         const uint32_t* perm1, const uint32_t* perm2, uint64_t n,
                         PoolKey* keys_out, uint64_t* weight_out) {
    const uint64_t j = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (j >= n) return;
    const uint32_t i = perm1[perm2[j]];
    keys_out[j] = PoolKey{step_slot[i], key[i]};
    weight_out[j] = weight[i];
}

__global__ void k_gather_u64(const uint64_t* in, const uint32_t* perm, uint64_t n, uint64_t* out) {
    const uint64_t j = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (j < n) out[j] = in[perm[j]];
}

constexpr uint32_t kThreads = 256;

uint32_t blocks(uint64_t n) { return static_cast<uint32_t>((n + kThreads - 1) / kThreads); }

}  // namespace

bool step_pool_device(const std::vector<uint64_t>& record_pool,
                      const std::vector<StepPoolPoint>& points, uint32_t steps,
                      std::vector<std::vector<hgcommon::SpEntry>>& entries,
                      std::vector<StepPoolTotals>& totals) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        cudaGetLastError();
        return false;
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    // The pool with a record of zeros after it, which stands for the state with no edges.
    const uint64_t none_word = record_pool.size();
    const uint64_t none_words = (sizeof(hgcommon::StateInvariantRecord) + 7) / 8;
    std::vector<uint64_t> host_pool(record_pool);
    host_pool.resize(none_word + none_words, 0);
    {
        const hgcommon::StateInvariantRecord none{};
        std::memcpy(host_pool.data() + none_word, &none, sizeof none);
    }

    // Each point's entry count, by the same function the device runs.
    std::vector<uint64_t> offset(points.size() + 1, 0);
    for (size_t i = 0; i < points.size(); ++i) {
        const StepPoolPoint& p = points[i];
        if (p.step >= steps) return false;
        const uint64_t word = p.record_word == UINT64_MAX ? none_word : p.record_word;
        if (word + none_words > host_pool.size()) return false;
        CountSink c;
        hgcommon::sp_add_class(
            *reinterpret_cast<const hgcommon::StateInvariantRecord*>(host_pool.data() + word),
            p.weight, p.raw, c);
        offset[i + 1] = offset[i] + c.n;
    }
    const uint64_t n = offset.back();
    if (n == 0) {
        entries.assign(steps, {});
        totals.assign(steps, StepPoolTotals{});
        return true;
    }
    if (n >= static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) return false;

    // Sizes of the CUB passes' temporary storage.
    size_t sort_bytes = 0, reduce_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, static_cast<const uint64_t*>(nullptr),
                                    static_cast<uint64_t*>(nullptr),
                                    static_cast<const uint32_t*>(nullptr),
                                    static_cast<uint32_t*>(nullptr), static_cast<int>(n));
    cub::DeviceReduce::ReduceByKey(nullptr, reduce_bytes, static_cast<const PoolKey*>(nullptr),
                                   static_cast<PoolKey*>(nullptr),
                                   static_cast<const uint64_t*>(nullptr),
                                   static_cast<uint64_t*>(nullptr), static_cast<int*>(nullptr),
                                   SatAdd{}, static_cast<int>(n));
    const uint64_t need = 8 * host_pool.size() + sizeof(StepPoolPoint) * points.size() +
                          8 * offset.size() + sizeof(hgcommon::SpDistTotals) * steps * hgcommon::SG_DISTS +
                          n * (8 * 3 + 8 * 2 + 4 * 4 + sizeof(PoolKey) * 2 + 8 * 2) +
                          std::max(sort_bytes, reduce_bytes) + 64;
    if (need > free_bytes / 4) return false;

    std::vector<void*> owned;
    struct Free {
        std::vector<void*>& p;
        ~Free() { for (void* x : p) cudaFree(x); }
    } free_all{owned};
    auto alloc = [&](size_t bytes) {
        void* d = nullptr;
        HG_CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(bytes, 1)), "cudaMalloc");
        owned.push_back(d);
        return d;
    };
    auto* d_pool = static_cast<uint64_t*>(alloc(8 * host_pool.size()));
    auto* d_points = static_cast<StepPoolPoint*>(alloc(sizeof(StepPoolPoint) * points.size()));
    auto* d_offset = static_cast<uint64_t*>(alloc(8 * offset.size()));
    auto* d_totals = static_cast<hgcommon::SpDistTotals*>(
        alloc(sizeof(hgcommon::SpDistTotals) * steps * hgcommon::SG_DISTS));
    auto* d_step_slot = static_cast<uint64_t*>(alloc(8 * n));
    auto* d_key = static_cast<uint64_t*>(alloc(8 * n));
    auto* d_weight = static_cast<uint64_t*>(alloc(8 * n));
    auto* d_sorted_a = static_cast<uint64_t*>(alloc(8 * n));
    auto* d_sorted_b = static_cast<uint64_t*>(alloc(8 * n));
    auto* d_iota = static_cast<uint32_t*>(alloc(4 * n));
    auto* d_perm1 = static_cast<uint32_t*>(alloc(4 * n));
    auto* d_perm2 = static_cast<uint32_t*>(alloc(4 * n));
    auto* d_keys = static_cast<PoolKey*>(alloc(sizeof(PoolKey) * n));
    auto* d_unique = static_cast<PoolKey*>(alloc(sizeof(PoolKey) * n));
    auto* d_weight_sorted = static_cast<uint64_t*>(alloc(8 * n));
    auto* d_merged = static_cast<uint64_t*>(alloc(8 * n));
    auto* d_runs = static_cast<int*>(alloc(sizeof(int)));
    void* d_temp = alloc(std::max(sort_bytes, reduce_bytes));

    std::vector<hgcommon::SpDistTotals> init(static_cast<size_t>(steps) * hgcommon::SG_DISTS);
    std::vector<uint32_t> iota(n);
    std::iota(iota.begin(), iota.end(), 0u);
    HG_CUDA_CHECK(cudaMemcpy(d_pool, host_pool.data(), 8 * host_pool.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
    HG_CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(StepPoolPoint) * points.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
    HG_CUDA_CHECK(cudaMemcpy(d_offset, offset.data(), 8 * offset.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
    HG_CUDA_CHECK(cudaMemcpy(d_totals, init.data(), sizeof(hgcommon::SpDistTotals) * init.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
    HG_CUDA_CHECK(cudaMemcpy(d_iota, iota.data(), 4 * n, cudaMemcpyHostToDevice), "cudaMemcpy");

    k_step_pool_emit<<<blocks(points.size()), kThreads>>>(d_pool, none_word, d_points, d_offset,
                                                         points.size(), d_step_slot, d_key,
                                                         d_weight, d_totals);
    HG_CUDA_CHECK(cudaGetLastError(), "k_step_pool_emit");

    // By key, then stably by (step, slot): the order (step, slot, key).
    size_t bytes = sort_bytes;
    HG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp, bytes, d_key, d_sorted_a, d_iota, d_perm1,
                                                  static_cast<int>(n)),
                  "SortPairs");
    k_gather_u64<<<blocks(n), kThreads>>>(d_step_slot, d_perm1, n, d_sorted_b);
    HG_CUDA_CHECK(cudaGetLastError(), "k_gather_u64");
    bytes = sort_bytes;
    HG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp, bytes, d_sorted_b, d_sorted_a, d_iota,
                                                  d_perm2, static_cast<int>(n)),
                  "SortPairs");
    k_gather<<<blocks(n), kThreads>>>(d_step_slot, d_key, d_weight, d_perm1, d_perm2, n, d_keys,
                                      d_weight_sorted);
    HG_CUDA_CHECK(cudaGetLastError(), "k_gather");
    bytes = reduce_bytes;
    HG_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(d_temp, bytes, d_keys, d_unique, d_weight_sorted,
                                                 d_merged, d_runs, SatAdd{}, static_cast<int>(n)),
                  "ReduceByKey");

    int runs = 0;
    HG_CUDA_CHECK(cudaMemcpy(&runs, d_runs, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    std::vector<PoolKey> keys(static_cast<size_t>(runs));
    std::vector<uint64_t> merged(static_cast<size_t>(runs));
    std::vector<hgcommon::SpDistTotals> got(init.size());
    if (runs) {
        HG_CUDA_CHECK(cudaMemcpy(keys.data(), d_unique, sizeof(PoolKey) * runs, cudaMemcpyDeviceToHost), "cudaMemcpy");
        HG_CUDA_CHECK(cudaMemcpy(merged.data(), d_merged, 8 * runs, cudaMemcpyDeviceToHost), "cudaMemcpy");
    }
    HG_CUDA_CHECK(cudaMemcpy(got.data(), d_totals, sizeof(hgcommon::SpDistTotals) * got.size(), cudaMemcpyDeviceToHost), "cudaMemcpy");

    entries.assign(steps, {});
    totals.assign(steps, StepPoolTotals{});
    for (int r = 0; r < runs; ++r)
        entries[keys[r].step_slot >> 32].push_back(
            hgcommon::SpEntry{static_cast<uint32_t>(keys[r].step_slot), keys[r].key, merged[r]});
    for (uint32_t s = 0; s < steps; ++s)
        for (uint32_t k = 0; k < hgcommon::SG_DISTS; ++k)
            totals[s][k] = got[static_cast<size_t>(s) * hgcommon::SG_DISTS + k];
    return true;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
