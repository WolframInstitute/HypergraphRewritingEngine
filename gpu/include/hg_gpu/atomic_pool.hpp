#pragma once

#include "hg_gpu/types.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace hg_gpu {

// Host-managed pre-allocated device array plus a device atomic counter for
// claiming indices. Append-only during a kernel run; reset between runs.
//
// claim() returns kInvalid when the pool is exhausted; callers are expected to
// have sized the pool via the auto-tuner so this only fires under genuine
// overflow that should abort the evolution.
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

        __device__ T&       at(uint32_t idx)       { return data[idx]; }
        __device__ const T& at(uint32_t idx) const { return data[idx]; }
    };

    explicit Pool(uint32_t capacity) : capacity_(capacity) {
        check(cudaMalloc(&data_, sizeof(T) * capacity_), "Pool data alloc");
        check(cudaMalloc(&counter_, sizeof(uint32_t)),   "Pool counter alloc");
        reset();
    }

    ~Pool() {
        if (data_)    cudaFree(data_);
        if (counter_) cudaFree(counter_);
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    Pool(Pool&& o) noexcept
        : data_(o.data_), counter_(o.counter_), capacity_(o.capacity_) {
        o.data_ = nullptr; o.counter_ = nullptr; o.capacity_ = 0;
    }

    DeviceView view() const {
        return DeviceView{data_, counter_, capacity_};
    }

    uint32_t capacity() const { return capacity_; }

    uint32_t size_host() const {
        uint32_t v = 0;
        check(cudaMemcpy(&v, counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "Pool size_host copy");
        return v;
    }

    void reset() {
        check(cudaMemset(counter_, 0, sizeof(uint32_t)), "Pool reset");
    }

private:
    static void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hg_gpu::Pool ") + what + ": " +
                                     cudaGetErrorString(err));
        }
    }

    T*        data_    = nullptr;
    uint32_t* counter_ = nullptr;
    uint32_t  capacity_ = 0;
};

}  // namespace hg_gpu
