#pragma once

#include "hg_gpu/overflow.hpp"   // ErrorKind / error_kind_name / OverflowWarning

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hg_gpu {

// Device-side error channel for kernel-observed capacity overflows.
//
// Every kernel that claims capacity-bounded resources (pools, LockFreeLists,
// ConcurrentMaps) records overflow reasons here instead of silently
// early-returning on partial work. After each kernel sync, the host calls
// `collect_warnings_into(...)` to drain the counters into an
// OverflowWarning list attached to EvolveResult. The kernels keep running
// on the partial budget — the caller decides whether the partial result
// is acceptable or to retry with bigger pools (see free `evolve()`).
//
// (`throw_if_any(...)` is retained for unit tests / asserts that want the
// old "fail fast" behaviour, but the production evolve path now uses
// collect_warnings_into instead.)

struct DeviceErrors {
    static constexpr uint32_t kMaxKinds = static_cast<uint32_t>(ErrorKind::kCount);

    struct DeviceView {
        uint32_t* counters;  // [kMaxKinds]

        __device__ void record(ErrorKind k) {
            uint32_t idx = static_cast<uint32_t>(k);
            if (idx < kMaxKinds) {
                atomicAdd(&counters[idx], 1u);
            }
        }
    };

    DeviceErrors() {
        cudaError_t err = cudaMalloc(&counters_, sizeof(uint32_t) * kMaxKinds);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("DeviceErrors alloc: ") +
                                     cudaGetErrorString(err));
        }
        clear();
    }

    ~DeviceErrors() {
        if (counters_) cudaFree(counters_);
    }

    DeviceErrors(const DeviceErrors&)            = delete;
    DeviceErrors& operator=(const DeviceErrors&) = delete;

    DeviceView view() const { return DeviceView{counters_}; }

    void clear() {
        cudaMemset(counters_, 0, sizeof(uint32_t) * kMaxKinds);
    }

    // Drain the device counters into `out` as OverflowWarning entries
    // tagged with `context`, then clear the counters so the next kernel
    // sync starts from zero. Non-throwing — capacity overflows are
    // warnings, not errors. Genuine driver failures (cudaMemcpy fails)
    // still throw std::runtime_error since those indicate a programmer
    // problem, not a runtime resource limit.
    void collect_warnings_into(std::vector<OverflowWarning>& out,
                               const char* context) {
        uint32_t host[kMaxKinds] = {};
        cudaError_t err = cudaMemcpy(host, counters_,
                                     sizeof(uint32_t) * kMaxKinds,
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("DeviceErrors d2h: ") +
                                     cudaGetErrorString(err));
        }
        bool any = false;
        for (uint32_t i = 0; i < kMaxKinds; ++i) {
            if (host[i] == 0) continue;
            out.push_back(OverflowWarning{
                static_cast<ErrorKind>(i),
                host[i],
                std::string(context),
            });
            any = true;
        }
        if (any) {
            cudaMemset(counters_, 0, sizeof(uint32_t) * kMaxKinds);
        }
    }

    // Typed exception carrying the specific ErrorKind that overflowed.
    // The host-side `evolve()` wrapper catches this and grows the
    // corresponding EngineConfig field before retrying. Inherits from
    // std::runtime_error so user-code that catches the latter still works.
    struct PoolOverflow : public std::runtime_error {
        PoolOverflow(ErrorKind k, uint32_t cnt, const std::string& full_msg)
            : std::runtime_error(full_msg), kind(k), count(cnt) {}
        ErrorKind kind;
        uint32_t  count;   // how many times the kernel observed this overflow
    };

    // Pull counters back to host. Blocking sync; call after a kernel you want
    // to audit. Throws PoolOverflow (subclass of std::runtime_error) if any
    // counter is non-zero, naming the FIRST overflowing kind so the host
    // retry loop can grow the corresponding EngineConfig field. The full
    // multi-line message lists ALL overflowing kinds for diagnosis.
    void throw_if_any(const char* context) const {
        uint32_t host[kMaxKinds] = {};
        cudaError_t err = cudaMemcpy(host, counters_,
                                     sizeof(uint32_t) * kMaxKinds,
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("DeviceErrors d2h: ") +
                                     cudaGetErrorString(err));
        }
        std::string msg;
        ErrorKind first_kind = ErrorKind::kCount;  // sentinel
        uint32_t  first_count = 0;
        for (uint32_t i = 0; i < kMaxKinds; ++i) {
            if (host[i] == 0) continue;
            if (first_kind == ErrorKind::kCount) {
                first_kind = static_cast<ErrorKind>(i);
                first_count = host[i];
            }
            if (!msg.empty()) msg += "; ";
            msg += error_kind_name(static_cast<ErrorKind>(i));
            msg += " overflowed ";
            msg += std::to_string(host[i]);
            msg += " times";
        }
        if (first_kind != ErrorKind::kCount) {
            throw PoolOverflow(first_kind, first_count,
                std::string("hg_gpu capacity overflow during ") + context +
                ": " + msg + ". Raise the corresponding EngineConfig field.");
        }
    }

private:
    uint32_t* counters_ = nullptr;
};

}  // namespace hg_gpu
