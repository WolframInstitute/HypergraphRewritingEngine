#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/overflow.hpp"   // ErrorKind / error_kind_name / OverflowWarning

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace HG_NAMESPACE {
namespace gpu {

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

        // AFFORDABLE ONCE, RUINOUS IN BULK -- the same shape ConcurrentMap's `saturated` latch
        // has. Under saturation this is reached by every thread of every block on every inner
        // loop iteration (kQcNodes fires on each failed claim), and they all atomicAdd one
        // 4-byte address, which serialises the whole device precisely when it is already
        // running degraded. A kind that has been recorded needs no further recording: the
        // counter says WHICH capacity ran out, and the retry path doubles that field regardless
        // of the number. So the first observers pay the atomic and the rest read and leave.
        //
        // The load races the add, so a bounded handful of threads may add before any of them
        // observes a non-zero counter. That bound is the concurrency, not the iteration count,
        // which is the entire difference.
        __device__ void record(ErrorKind k) {
            uint32_t idx = static_cast<uint32_t>(k);
            if (idx >= kMaxKinds) return;
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> c(counters[idx]);
            if (c.load(cuda::memory_order_relaxed) != 0) return;
            c.fetch_add(1u, cuda::memory_order_relaxed);
        }
    };

    DeviceErrors();

    ~DeviceErrors();

    DeviceErrors(const DeviceErrors&)            = delete;
    DeviceErrors& operator=(const DeviceErrors&) = delete;

    DeviceView view() const;

    void clear();

    // Drain the device counters into `out` as OverflowWarning entries
    // tagged with `context`, then clear the counters so the next kernel
    // sync starts from zero. Non-throwing — capacity overflows are
    // warnings, not errors. Genuine driver failures (cudaMemcpy fails)
    // still throw std::runtime_error since those indicate a programmer
    // problem, not a runtime resource limit.
    void collect_warnings_into(std::vector<OverflowWarning>& out, const char* context);

    // Typed exception carrying the specific ErrorKind that overflowed.
    // The host-side `evolve()` wrapper catches this and grows the
    // corresponding EngineConfig field before retrying. Inherits from
    // std::runtime_error so user-code that catches the latter still works.
    struct PoolOverflow : public std::runtime_error {
        PoolOverflow(ErrorKind k, uint32_t cnt, const std::string& full_msg);
        ErrorKind kind;
        uint32_t  count;   // how many times the kernel observed this overflow
    };

    // Pull counters back to host. Blocking sync; call after a kernel you want
    // to audit. Throws PoolOverflow (subclass of std::runtime_error) if any
    // counter is non-zero, naming the FIRST overflowing kind so the host
    // retry loop can grow the corresponding EngineConfig field. The full
    // multi-line message lists ALL overflowing kinds for diagnosis.
    void throw_if_any(const char* context) const;

private:
    uint32_t* counters_ = nullptr;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE