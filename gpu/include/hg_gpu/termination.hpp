#pragma once

#include "hg_gpu/types.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace hg_gpu {

// Termination detector for the streaming-persistent-kernel model.
//
// Each queue role (match, chs_build, rewrite, dedup, index, event) maintains
// two monotonic atomic counters: pushed (work added) and completed (work
// drained). A small detector observes the counters; when all pairs equal AND
// the equality persists for a stable observation interval (i.e. nothing was
// pushed during the observation window), it sets should_exit. All persistent
// worker kernels check should_exit at the top of their outer loop.
//
// Counters are uint64_t and never overflow during a real evolution (would
// require ≥ 2^64 enqueues; not a practical concern).
//
// The "stable observation" requirement is what prevents premature exit: an
// instantaneous (pushed == completed) snapshot can occur mid-evolution when
// every in-flight task happens to have just finished and not yet emitted
// follow-ups. We require equality across two consecutive snapshots separated
// by a backoff sleep before signalling exit.
//
// Memory-ordering audit:
//   mark_pushed / mark_completed:  fetch_add(release)
//   snapshot_quiescent:            per-counter load(acquire)
//   signal_exit:                   store(release) on should_exit
//   exit_requested:                load(acquire) on should_exit
//   The release fetch_add on producer/consumer counters ensures that any
//   in-flight work (and the payload it manipulates in downstream queues) is
//   visible to the detector when it observes the counter. Similarly, when a
//   worker sees should_exit via acquire, it observes all state the detector
//   saw before signalling (necessary for clean shutdown).
class TerminationDetector {
public:
    static constexpr uint32_t kMaxRoles = 8;

    struct DeviceView {
        uint64_t* pushed;       // [kMaxRoles]
        uint64_t* completed;    // [kMaxRoles]
        uint8_t*  should_exit;  // single boolean
        uint32_t  num_roles;

        __device__ void mark_pushed(uint32_t role, uint64_t n = 1) const {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(pushed[role]);
            ref.fetch_add(n, cuda::memory_order_release);
        }

        __device__ void mark_completed(uint32_t role, uint64_t n = 1) const {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(completed[role]);
            ref.fetch_add(n, cuda::memory_order_release);
        }

        __device__ bool exit_requested() const {
            cuda::atomic_ref<uint8_t, cuda::thread_scope_device> ref(*should_exit);
            return ref.load(cuda::memory_order_acquire) != 0;
        }

        // Detector-local: snapshot current counters into local buffers and
        // return true iff all roles have pushed == completed. Caller should
        // require two consecutive trues separated by a backoff sleep before
        // signalling exit.
        __device__ bool snapshot_quiescent(uint64_t* p_out, uint64_t* c_out) const {
            bool all_equal = true;
            for (uint32_t r = 0; r < num_roles; ++r) {
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pref(pushed[r]);
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> cref(completed[r]);
                uint64_t p = pref.load(cuda::memory_order_acquire);
                uint64_t c = cref.load(cuda::memory_order_acquire);
                p_out[r] = p;
                c_out[r] = c;
                if (p != c) all_equal = false;
            }
            return all_equal;
        }

        __device__ void signal_exit() const {
            cuda::atomic_ref<uint8_t, cuda::thread_scope_device> ref(*should_exit);
            ref.store(1, cuda::memory_order_release);
        }
    };

    explicit TerminationDetector(uint32_t num_roles) : num_roles_(num_roles) {
        if (num_roles_ == 0 || num_roles_ > kMaxRoles) {
            throw std::invalid_argument("TerminationDetector num_roles out of range");
        }
        check(cudaMalloc(&pushed_,      sizeof(uint64_t) * kMaxRoles), "TD pushed alloc");
        check(cudaMalloc(&completed_,   sizeof(uint64_t) * kMaxRoles), "TD completed alloc");
        check(cudaMalloc(&should_exit_, sizeof(uint8_t)),              "TD should_exit alloc");
        clear();
    }

    ~TerminationDetector() {
        if (pushed_)      cudaFree(pushed_);
        if (completed_)   cudaFree(completed_);
        if (should_exit_) cudaFree(should_exit_);
    }

    TerminationDetector(const TerminationDetector&)            = delete;
    TerminationDetector& operator=(const TerminationDetector&) = delete;

    DeviceView view() {
        return DeviceView{pushed_, completed_, should_exit_, num_roles_};
    }

    void clear() {
        check(cudaMemset(pushed_,      0, sizeof(uint64_t) * kMaxRoles), "TD clear pushed");
        check(cudaMemset(completed_,   0, sizeof(uint64_t) * kMaxRoles), "TD clear completed");
        check(cudaMemset(should_exit_, 0, sizeof(uint8_t)),              "TD clear should_exit");
    }

    bool exit_requested_host() const {
        uint8_t v = 0;
        cudaMemcpy(&v, should_exit_, sizeof(uint8_t), cudaMemcpyDeviceToHost);
        return v != 0;
    }

    uint32_t num_roles() const { return num_roles_; }

private:
    static void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hg_gpu::TerminationDetector ") + what + ": " +
                                     cudaGetErrorString(err));
        }
    }

    uint64_t* pushed_      = nullptr;
    uint64_t* completed_   = nullptr;
    uint8_t*  should_exit_ = nullptr;
    uint32_t  num_roles_   = 0;
};

}  // namespace hg_gpu
