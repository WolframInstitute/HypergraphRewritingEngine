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
#include <vector>

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

// THE FASTEST CLASS OF CORE ON THIS MACHINE, as logical CPU indices, or EMPTY when the
// question has no answer here.
//
// Pinning needs a set to pin TO, and until now the caller had to write one by hand -- which
// means knowing the machine, and getting it wrong silently produces the mixed-speed curve the
// pinning exists to prevent. This answers it from the operating system:
//
//   Windows  GetLogicalProcessorInformationEx(RelationProcessorCore) reports an
//            EfficiencyClass per core; the highest value is the performance class. This is
//            what makes a Windows speedup curve honest on a P/E part.
//   Linux    /sys/devices/cpu_core/cpus lists the P-cores on an Intel hybrid part; on ARM
//            big.LITTLE the same question is cpu_capacity per CPU, highest value wins.
//
// EMPTY means "no distinction available", NOT "no fast cores": a homogeneous machine, macOS,
// or a hypervisor that flattens the topology all answer empty, and the caller should then use
// every CPU rather than none. WSL is the flattening case that matters here -- it presents a
// 14900K as 16 uniform SMT cores, so this returns empty there and says nothing false.
//
// One entry per PHYSICAL core of that class (the first hardware thread of each), because a
// speedup denominator counts cores, not siblings that share one.
std::vector<unsigned> performance_cpus();

// WHICH CPUS SHARE A LAST-LEVEL CACHE, as one dense domain id per entry of `cpus`, in the same
// order. Two entries are equal exactly when those two CPUs share an LLC.
//
// WHY THE ENGINE WANTS THIS. A core count says nothing about how expensive it is for two
// workers to share data. On a part with one LLC across every core, two threads touching the
// same line settle it on-die. On a chiplet part they may not share any cache at all, and the
// line moves over an off-die fabric instead. The difference is not a detail: on an EPYC 9174F,
// which gives 16 cores EIGHT separate L3 instances -- two cores each -- the same two-thread
// workload takes 1519 ms when both threads share an L3 and 1852 ms when they do not, a 21%
// penalty for identical work, decided entirely by placement.
//
// A work-stealing pool can act on that: a steal moves a job to a thief that will then touch the
// data the victim just touched, so drawing the victim from the thief's own domain keeps the
// transfer inside a cache both ends share. That is the one use here, and it is a preference
// rather than a restriction -- a thief that finds nothing near still steals from anywhere,
// because leaving a core idle costs more than a distant line.
//
//   Linux    the deepest /sys/devices/system/cpu/cpuN/cache/indexK that reports an `id`; the
//            kernel already numbers each cache instance, so equal ids mean one cache.
//   Windows  GetLogicalProcessorInformationEx(RelationCache), taking the deepest level and
//            grouping by the GroupMask of each cache instance.
//
// EMPTY means the topology could not be read -- no sysfs, no cache relationship, a hypervisor
// that does not present one. A caller must then treat every CPU as one domain rather than
// invent a grouping, since a wrong grouping steers steals at data that is not there.
std::vector<unsigned> cache_domains_of(const std::vector<unsigned>& cpus);

// THE CPUS THIS THREAD IS ALLOWED TO RUN ON, as logical CPU indices, or EMPTY when the
// operating system does not say.
//
// A process under taskset, a cpuset cgroup or a container is confined to a subset of the
// machine, and a thread can still bind itself outside that subset: on Linux the taskset mask is
// inherited, not enforced, so a worker pinned by index to a CPU the caller was not given runs
// there anyway. Every default placement therefore intersects the machine's CPUs with this set,
// and a caller that named CPUs explicitly is trusted to have meant them.
//
//   Linux    sched_getaffinity of the calling thread; a worker inherits its creator's mask.
//   Windows  GetProcessAffinityMask, group 0 (the same reach as SetThreadAffinityMask).
//
// EMPTY means "unknown", never "none": macOS has no mask to read, and the caller then uses
// every CPU.
std::vector<unsigned> allowed_cpus();

}  // namespace common
}  // namespace HG_NAMESPACE
