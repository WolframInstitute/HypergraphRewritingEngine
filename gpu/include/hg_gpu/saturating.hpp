#pragma once
#include "hgcommon/namespace.hpp"
// The device's saturating counter add: the quotient replay's multiplicity counts
// (quotient_expansion.hpp, quotient.cu) and the step pool's distribution totals (step_pool.cu).

#include "hgcommon/quotient_multiplicity_core.hpp"

#include <cuda/atomic>

#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

// counter += delta, saturating at hgcommon::QM_SATURATED (hgcommon::qm_sat_add), by
// compare-exchange (verification/gpumc/saturating_totals.cpp).
__device__ __forceinline__ void saturating_add(unsigned long long* counter, uint64_t delta) {
    unsigned long long old = cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(
                                 *counter).load(cuda::memory_order_relaxed);
    for (;;) {
        const unsigned long long next = hgcommon::qm_sat_add(old, delta);
        const unsigned long long seen = atomicCAS(counter, old, next);
        if (seen == old) return;
        old = seen;
    }
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
