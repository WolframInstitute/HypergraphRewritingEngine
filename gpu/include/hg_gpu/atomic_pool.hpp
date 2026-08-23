#pragma once
#include <vector>
#include "hgcommon/namespace.hpp"

#include "hg_gpu/types.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

// Host-managed pre-allocated device array plus a device atomic counter for
// claiming indices. Append-only during a kernel run; reset between runs.
//
// claim() returns kInvalid when the pool is exhausted. The host sizes the pool from
// EngineConfig before launch, so this fires when the workload outgrew that estimate;
// the device records the overflow and the host retries at a larger size.
//
// The counter is bumped BEFORE exhaustion is reported, so under overflow it keeps rising past
// capacity and is NOT a count of valid entries. Ask size() for that; reading the counter raw is
// how a caller ends up iterating past the allocation.
template <typename T>
class Pool {
public:
    static constexpr uint32_t kInvalid = INVALID_ID;

    struct DeviceView {
        T*        data;
        uint32_t* counter;
        uint32_t  capacity;

        __device__ uint32_t claim() {
            uint32_t idx = atomicAdd(counter, 1u);
            return (idx < capacity) ? idx : kInvalid;
        }

        __device__ uint32_t claim_n(uint32_t n) {
            uint32_t idx = atomicAdd(counter, n);
            return ((uint64_t)idx + n <= capacity) ? idx : kInvalid;
        }

        // How many entries are VALID. Not *counter: claim() bumps the counter unconditionally
        // and only then reports exhaustion, so on overflow the counter runs PAST capacity and a
        // caller that read it raw would walk off the allocation. Every "how many are there"
        // question goes through this.
        __device__ uint32_t size() const {
            const uint32_t c = counter ? *counter : 0u;
            return c < capacity ? c : capacity;
        }

        // Unchecked, and deliberately so on the hot path -- callers index with a slot they
        // claimed or with an id they have already tested against size(). Nothing here can
        // establish validity that the caller does not already know.
        __device__ T&       at(uint32_t idx)       { return data[idx]; }
        __device__ const T& at(uint32_t idx) const { return data[idx]; }
    };

    explicit Pool(uint32_t capacity) : capacity_(capacity) {
        HG_CUDA_CHECK(cudaMalloc(&data_, sizeof(T) * capacity_), "Pool data alloc");
        HG_CUDA_CHECK(cudaMalloc(&counter_, sizeof(uint32_t)),   "Pool counter alloc");
        owns_counter_ = true;
        init_storage();
    }

    // The counter lives in caller-owned device memory -- a slot of EngineState's counter
    // block -- so a host snapshot of that block reads this pool's live count with no per-pool
    // transfer and no staging. The caller owns the slot's lifetime.
    Pool(uint32_t capacity, uint32_t* external_counter) : capacity_(capacity) {
        HG_CUDA_CHECK(cudaMalloc(&data_, sizeof(T) * capacity_), "Pool data alloc");
        counter_ = external_counter;
        owns_counter_ = false;
        init_storage();
    }

private:
    void init_storage() {
        // The counter is a reservation high-water mark, so it can stand ahead of the writes: a
        // thread that claims slots and then fails a later reservation returns without filling
        // them. Host readbacks are bounded by the counter and therefore copy those slots. Zeroing
        // the storage once here is what makes such a slot a defined value rather than whatever
        // the allocator held, at one memset per pool for the life of the engine -- reset() is on
        // the per-run path and clears only the counter.
        HG_CUDA_CHECK(cudaMemset(data_, 0, sizeof(T) * capacity_), "Pool data init");
        reset();
    }

public:
    ~Pool() {
        if (data_)                     cudaFree(data_);
        if (counter_ && owns_counter_) cudaFree(counter_);
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    Pool(Pool&& o) noexcept
        : data_(o.data_), counter_(o.counter_), capacity_(o.capacity_),
          owns_counter_(o.owns_counter_) {
        o.data_ = nullptr; o.counter_ = nullptr; o.capacity_ = 0;
    }

    DeviceView view() const {
        return DeviceView{data_, counter_, capacity_};
    }

    uint32_t capacity() const { return capacity_; }

    uint32_t size_host() const {
        uint32_t v = 0;
        HG_CUDA_CHECK(cudaMemcpy(&v, counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "Pool size_host copy");
        return v;
    }

    // The VALID entries, copied to the host. Clamped to capacity for the same reason size()
    // is: claim() bumps the counter unconditionally and reports exhaustion afterwards, so an
    // overflowed run leaves the counter past the allocation.
    void copy_to_host(std::vector<T>& out) const { copy_to_host(out, size_host()); }

    // The same copy with the count already in hand -- a caller holding a counter snapshot
    // spends no extra cudaMemcpy call re-reading it.
    void copy_to_host(std::vector<T>& out, uint32_t n) const {
        const uint32_t valid = n < capacity_ ? n : capacity_;
        out.resize(valid);
        if (!valid) return;
        HG_CUDA_CHECK(cudaMemcpy(out.data(), data_, sizeof(T) * valid, cudaMemcpyDeviceToHost),
              "Pool copy_to_host");
    }

    void reset() {
        HG_CUDA_CHECK(cudaMemset(counter_, 0, sizeof(uint32_t)), "Pool reset");
    }

    // Zero the payload as well as the counter. Needed by a consumer that reads records
    // concurrently with their producers and therefore relies on a per-record publication flag:
    // the flag has to start clear, and reset() alone leaves the previous run's bytes in place.
    // O(capacity), so it belongs at run setup rather than between steps.
    void reset_and_clear() {
        reset();
        HG_CUDA_CHECK(cudaMemset(data_, 0, sizeof(T) * capacity_), "Pool clear data");
    }

    T*        data_    = nullptr;
    uint32_t* counter_ = nullptr;
    uint32_t  capacity_ = 0;
    // False when the counter is a slot of caller-owned memory (EngineState's counter block).
    bool      owns_counter_ = true;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE