#pragma once
#include "hgcommon/namespace.hpp"

// Pinning a thread to a logical CPU, where the platform offers it.
//
// WHY THE ENGINE WANTS THIS. Consumer CPUs are no longer homogeneous: Intel pairs P-cores with
// E-cores, AMD pairs Zen cores with dense Zen-c cores and puts the large cache on one CCD of
// two, and every big.LITTLE part has the same shape. A worker pool sized to
// hardware_concurrency() and placed by the OS therefore runs on cores of DIFFERENT speeds, and
// two things follow. A speedup curve measured that way has no honest denominator, because the
// nth thread is not the same resource as the first -- measured here, one E-core does the same
// work in 30.370 ms that a P-core does in 18.042 ms, so it is 0.59 of a core. And a run's timing
// depends on which cores the scheduler happened to choose, which is why a single-threaded
// baseline moved between 2878 ms pinned to a P-core and 3267 ms unpinned.
//
// WHAT THIS IS NOT. It is not a scheduler and does not decide anything: the caller names the
// CPUs and the OS is left alone otherwise. Nothing pins by default.
//
// PORTABILITY IS NOT UNIFORM, and the enum says so rather than pretending. Linux and Windows
// both expose thread affinity; macOS deliberately does not -- it offers QoS classes as a hint
// and no way to bind a thread to a core -- so affinity_supported() is false there and
// pin_this_thread_to_cpu() reports failure rather than silently doing nothing.
//
// WSL is a further case the enum cannot see: the syscall exists and succeeds, but the topology
// it acts on is the hypervisor's flattened view (a 14900K reports as 16 cores x 2 threads, which
// is neither its 24 physical cores nor its 32 logical ones) and vCPU-to-core placement belongs to
// the host. Pinning there binds to a virtual CPU and says nothing about which physical core runs
// it.

#include <cstddef>

namespace HG_NAMESPACE {
namespace common {

enum class AffinityBackend {
    SchedSetAffinity,       // Linux: pthread_setaffinity_np
    SetThreadAffinityMask,  // Windows: one processor group, so CPU indices below 64
    None                    // macOS and anything else: no thread-to-core binding exists
};

constexpr AffinityBackend affinity_backend() {
#if defined(__linux__)
    return AffinityBackend::SchedSetAffinity;
#elif defined(_WIN32)
    return AffinityBackend::SetThreadAffinityMask;
#else
    return AffinityBackend::None;
#endif
}

constexpr bool affinity_supported() {
    return affinity_backend() != AffinityBackend::None;
}

// Bind the CALLING thread to `cpu`, a logical CPU index. Returns whether the binding took
// effect, so a caller that needs a homogeneous core set can fail loudly rather than report a
// scaling curve it did not actually measure.
bool pin_this_thread_to_cpu(unsigned cpu);

}  // namespace common
}  // namespace HG_NAMESPACE
