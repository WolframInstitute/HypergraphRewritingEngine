#include "hg_gpu/branchial_rows.hpp"
#include "hg_gpu/cuda_check.hpp"
#include "hg_gpu/ir_canon.hpp"

// The device side of docs/DEVICE_DESIGN.md 4.8: one warp per row of one step's overlap pass,
// running hgcommon::br_row with the warp's lanes (IrTile). A warp's scratch is its own slice of
// four arrays sized by the step (count and touched by states, dist and queue by graph nodes), and
// its row's pairs are counted per key in a table in shared memory. At the end of each row the
// warp appends the table's entries to a global record buffer; the host adds them up per key,
// which is the host route's merge, so the counts do not depend on which warp ran which row. For
// the state matrices each pair's shared count goes to its own cell (hgcommon::br_pair_index),
// written by the one lane that emits the pair, and after the row the warp's lanes fill the row's
// matrix cells (hgcommon::bo_matrix_row), which the host copies back.

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

namespace {

constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kTallySlots = 512;   // per warp, a power of two
constexpr unsigned long long kEmptyKey = ~0ull;

// One row's count of one key, read back by the host.
struct BrRecord {
    unsigned long long key;   // distance << 48 | |B| << 24 | shared
    uint32_t size_a;
    uint32_t n;
};

// The key of a pair, or kEmptyKey when a field does not fit (then the pair is a record of its
// own with the fields in `key` replaced by the caller's).
__device__ inline bool br_key(uint32_t distance, uint32_t size_b, uint32_t shared,
                              unsigned long long& key) {
    if (distance >= (1u << 16) || size_b >= (1u << 24) || shared >= (1u << 24)) return false;
    key = (static_cast<unsigned long long>(distance) << 48) |
          (static_cast<unsigned long long>(size_b) << 24) | shared;
    return true;
}

struct DeviceRowCtx {
    uint32_t* count;
    uint32_t* touched;
    int32_t* dist;
    uint32_t* queue;
    uint32_t* n_touched;
    uint32_t* n_queue;
    unsigned long long* keys;   // this warp's tally, shared memory
    uint32_t* counts;
    BrRecord* out;
    uint32_t* out_n;
    uint32_t out_cap;
    uint32_t* wide;             // pairs whose fields do not fit a key: counted, not recorded
    uint32_t* cells;            // the shared counts by hgcommon::br_pair_index, or null
    bool tally;                 // whether the pairs are counted per key
    uint32_t k;
    uint32_t row;
    uint32_t size_a;

    template <class T> __device__ T ld(const T* p) const { return __ldcg(p); }
    template <class T> __device__ void st(T* p, T v) const { __stcg(p, v); }
    __device__ uint32_t add_count(uint32_t b) { return atomicAdd(&count[b], 1u); }
    __device__ bool claim_dist(uint32_t x, int32_t d) { return atomicCAS(&dist[x], -1, d) == -1; }
    __device__ uint32_t push(uint32_t* counter) { return atomicAdd(counter, 1u); }
    __device__ void emit(uint32_t distance, uint32_t b, uint32_t size_b, uint32_t shared) {
        if (cells && distance == 0) cells[hgcommon::br_pair_index(k, row, b)] = shared;
        if (!tally) return;
        unsigned long long key;
        if (!br_key(distance, size_b, shared, key)) {
            atomicAdd(wide, 1u);
            return;
        }
        uint32_t i = static_cast<uint32_t>((key ^ (key >> 29)) * 0xBF58476D1CE4E5B9ull >> 40) &
                     (kTallySlots - 1);
        for (uint32_t probe = 0; probe < kTallySlots; ++probe, i = (i + 1) & (kTallySlots - 1)) {
            const unsigned long long seen = atomicCAS(&keys[i], kEmptyKey, key);
            if (seen == kEmptyKey || seen == key) {
                atomicAdd(&counts[i], 1u);
                return;
            }
        }
        // The table is full: the pair is a record of its own.
        const uint32_t at = atomicAdd(out_n, 1u);
        if (at < out_cap) out[at] = BrRecord{key, size_a, 1u};
    }
};

__global__ void k_branchial_rows(hgcommon::BrStep s, uint32_t row_end, uint32_t* next_row,
                                 uint32_t* count, uint32_t* touched, int32_t* dist,
                                 uint32_t* queue, uint32_t* counters, BrRecord* out,
                                 uint32_t* out_n, uint32_t out_cap, uint32_t* wide,
                                 uint32_t* cells, bool tally, hgcommon::BoMatrices mat) {
    __shared__ unsigned long long keys[kWarpsPerBlock][kTallySlots];
    __shared__ uint32_t counts[kWarpsPerBlock][kTallySlots];
    const uint32_t lane = threadIdx.x & 31u, wib = threadIdx.x >> 5;
    const uint64_t warp = static_cast<uint64_t>(blockIdx.x) * kWarpsPerBlock + wib;
    for (uint32_t i = lane; i < kTallySlots; i += 32) {
        keys[wib][i] = kEmptyKey;
        counts[wib][i] = 0;
    }
    __syncwarp();
    DeviceRowCtx c{count + warp * s.k,
                   touched + warp * s.k,
                   dist + warp * s.n,
                   queue + warp * s.n,
                   counters + warp * 2,
                   counters + warp * 2 + 1,
                   keys[wib],
                   counts[wib],
                   out,
                   out_n,
                   out_cap,
                   wide,
                   cells,
                   tally,
                   s.k,
                   0,
                   0};
    hg_gpu::IrTile par;
    for (;;) {
        uint32_t a = 0;
        if (lane == 0) a = atomicAdd(next_row, 1u);
        a = __shfl_sync(0xffffffffu, a, 0);
        if (a >= row_end) break;
        c.size_a = s.size[a];
        c.row = a;
        hgcommon::br_row(c, s, a, par);
        __syncwarp();
        if (mat.k) {
            hgcommon::bo_matrix_row(mat, a, lane, 32);
            __syncwarp();
        }
        for (uint32_t i = lane; i < kTallySlots; i += 32) {
            const unsigned long long key = keys[wib][i];
            if (key == kEmptyKey) continue;
            const uint32_t at = atomicAdd(out_n, 1u);
            if (at < out_cap) out[at] = BrRecord{key, c.size_a, counts[wib][i]};
            keys[wib][i] = kEmptyKey;
            counts[wib][i] = 0;
        }
        __syncwarp();
    }
}

template <class T>
T* device_copy(const T* host, size_t n, std::vector<void*>& owned) {
    T* d = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    owned.push_back(d);
    if (n) HG_CUDA_CHECK(cudaMemcpy(d, host, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return d;
}

}  // namespace

bool branchial_rows_device(
    const hgcommon::BrStep& s,
    const std::function<void(uint32_t, uint32_t, uint32_t, uint32_t, uint64_t)>& add,
    hgcommon::BoMatrices* matrices) {
    const bool tally = static_cast<bool>(add);
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
    cudaDeviceProp prop{};
    HG_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    // The step's arrays, then each warp's scratch and the record buffer from a quarter of what
    // is free.
    const uint64_t step_bytes =
        4ull * (s.k + (s.k + 1) + s.row_off[s.k] + (s.vertices + 1) + s.hold_off[s.vertices]) +
        (s.to_graph ? 4ull * ((s.n + 1) + s.adj_off[s.n] + s.k + s.n) : 0ull);
    const uint64_t warp_bytes = 4ull * (2ull * s.k + 2ull * s.n + 2);
    const uint32_t out_cap = tally ? 1u << 22 : 1u;
    const uint64_t cells = matrices ? uint64_t{s.k} * (s.k ? s.k - 1 : 0) / 2 : 0;
    uint32_t asked = 0;
    if (matrices)
        for (double* m : matrices->m) asked += m != nullptr;
    const uint64_t cell_bytes = 4 * cells + 8ull * asked * s.k * s.k;
    const uint64_t budget = free_bytes / 4;
    const uint64_t fixed = step_bytes + out_cap * sizeof(BrRecord) + cell_bytes;
    if (fixed + warp_bytes * 32 > budget) return false;
    const uint64_t fit = (budget - fixed) / warp_bytes;
    const uint64_t want = static_cast<uint64_t>(prop.multiProcessorCount) * 16;
    const uint32_t warps = static_cast<uint32_t>(
        std::min<uint64_t>({fit, want, (s.k + kWarpsPerBlock - 1) / kWarpsPerBlock * kWarpsPerBlock}) /
        kWarpsPerBlock * kWarpsPerBlock);
    if (warps == 0) return false;

    std::vector<void*> owned;
    struct Free {
        std::vector<void*>& p;
        ~Free() { for (void* x : p) cudaFree(x); }
    } free_all{owned};
    hgcommon::BrStep d = s;
    d.size = device_copy(s.size, s.k, owned);
    d.row_off = device_copy(s.row_off, s.k + 1, owned);
    d.row_vert = device_copy(s.row_vert, s.row_off[s.k], owned);
    d.hold_off = device_copy(s.hold_off, s.vertices + 1, owned);
    d.hold = device_copy(s.hold, s.hold_off[s.vertices], owned);
    if (s.to_graph) {
        d.adj_off = device_copy(s.adj_off, s.n + 1, owned);
        d.adj = device_copy(s.adj, s.adj_off[s.n], owned);
        d.to_graph = device_copy(s.to_graph, s.k, owned);
        d.from_graph = device_copy(s.from_graph, s.n, owned);
    }
    uint32_t *count = nullptr, *touched = nullptr, *queue = nullptr, *counters = nullptr,
             *misc = nullptr;
    int32_t* dist = nullptr;
    BrRecord* out = nullptr;
    const size_t wk = static_cast<size_t>(warps) * s.k, wn = static_cast<size_t>(warps) * s.n;
    HG_CUDA_CHECK(cudaMalloc(&count, std::max<size_t>(wk, 1) * 4), "cudaMalloc");
    owned.push_back(count);
    HG_CUDA_CHECK(cudaMalloc(&touched, std::max<size_t>(wk, 1) * 4), "cudaMalloc");
    owned.push_back(touched);
    HG_CUDA_CHECK(cudaMalloc(&dist, std::max<size_t>(wn, 1) * 4), "cudaMalloc");
    owned.push_back(dist);
    HG_CUDA_CHECK(cudaMalloc(&queue, std::max<size_t>(wn, 1) * 4), "cudaMalloc");
    owned.push_back(queue);
    HG_CUDA_CHECK(cudaMalloc(&counters, static_cast<size_t>(warps) * 2 * 4), "cudaMalloc");
    owned.push_back(counters);
    HG_CUDA_CHECK(cudaMalloc(&misc, 3 * 4), "cudaMalloc");   // next_row, out_n, wide
    owned.push_back(misc);
    HG_CUDA_CHECK(cudaMalloc(&out, static_cast<size_t>(out_cap) * sizeof(BrRecord)), "cudaMalloc");
    owned.push_back(out);
    uint32_t* d_cells = nullptr;
    hgcommon::BoMatrices dm;
    if (matrices) {
        HG_CUDA_CHECK(cudaMalloc(&d_cells, std::max<uint64_t>(cells, 1) * 4), "cudaMalloc");
        owned.push_back(d_cells);
        HG_CUDA_CHECK(cudaMemset(d_cells, 0, std::max<uint64_t>(cells, 1) * 4), "cudaMemset");
        dm.k = s.k;
        dm.size = d.size;
        dm.shared = d_cells;
        dm.universe = matrices->universe;
        for (uint32_t kind = 0; kind < 3; ++kind) {
            if (!matrices->m[kind]) continue;
            HG_CUDA_CHECK(cudaMalloc(&dm.m[kind], 8ull * s.k * s.k), "cudaMalloc");
            owned.push_back(dm.m[kind]);
        }
    }
    HG_CUDA_CHECK(cudaMemset(count, 0, std::max<size_t>(wk, 1) * 4), "cudaMemset");
    HG_CUDA_CHECK(cudaMemset(dist, 0xFF, std::max<size_t>(wn, 1) * 4), "cudaMemset");
    HG_CUDA_CHECK(cudaMemset(counters, 0, static_cast<size_t>(warps) * 2 * 4), "cudaMemset");

    // Rows in batches whose records fit the buffer: a batch that overflows is run again in
    // halves. Its rows leave their scratch cleared either way.
    std::vector<BrRecord> records;
    std::map<std::pair<unsigned long long, uint32_t>, uint64_t> summed;   // (key, |A|) -> pairs
    uint64_t wide_total = 0;
    uint32_t batch = tally ? std::max<uint32_t>(1u, out_cap / 64) : std::max<uint32_t>(1u, s.k);
    for (uint32_t row = 0; row < s.k;) {
        const uint32_t end = static_cast<uint32_t>(std::min<uint64_t>(s.k, uint64_t{row} + batch));
        const uint32_t start[3] = {row, 0, 0};
        HG_CUDA_CHECK(cudaMemcpy(misc, start, sizeof(start), cudaMemcpyHostToDevice), "cudaMemcpy");
        k_branchial_rows<<<warps / kWarpsPerBlock, 32 * kWarpsPerBlock>>>(
            d, end, misc, count, touched, dist, queue, counters, out, misc + 1, out_cap, misc + 2,
            d_cells, tally, dm);
        HG_CUDA_CHECK(cudaGetLastError(), "k_branchial_rows");
        uint32_t got[3];
        HG_CUDA_CHECK(cudaMemcpy(got, misc, sizeof(got), cudaMemcpyDeviceToHost), "cudaMemcpy");
        if (got[1] > out_cap) {
            if (batch == 1) return false;
            batch /= 2;
            continue;
        }
        records.resize(got[1]);
        if (got[1])
            HG_CUDA_CHECK(cudaMemcpy(records.data(), out, got[1] * sizeof(BrRecord),
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy");
        for (const BrRecord& r : records) {
            summed[{r.key, r.size_a}] += r.n;
        }
        wide_total += got[2];
        row = end;
    }
    // A pair whose fields do not fit a key (a distance of 2^16 or more, or a state of 2^24
    // vertices) is not representable here; such a step runs on the host.
    if (wide_total) return false;
    if (matrices)
        for (uint32_t kind = 0; kind < 3; ++kind)
            if (matrices->m[kind])
                HG_CUDA_CHECK(cudaMemcpy(matrices->m[kind], dm.m[kind], 8ull * s.k * s.k,
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy");
    for (const auto& [ks, n] : summed) {
        const unsigned long long key = ks.first;
        add(static_cast<uint32_t>(key >> 48), ks.second,
            static_cast<uint32_t>(key >> 24) & 0xFFFFFFu, static_cast<uint32_t>(key) & 0xFFFFFFu,
            n);
    }
    return true;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
