#include "hgcommon/affinity.hpp"

// The affinity body, and the platform headers it calls into. affinity.hpp names only the backend
// SELECTION, so a translation unit that pins parses <cstddef> rather than windows.h or sched.h.
//
// This lives under job_system for the reason park.cpp does: job_system is the only consumer and
// the only target that could hold a body, and hgcommon is headers with no library of its own.

#if defined(__linux__)
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#  include <pthread.h>
#  include <sched.h>
#elif defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX   // windows.h's min/max macros poison std::numeric_limits<T>::max()
#  endif
#  include <windows.h>
#endif

namespace HG_NAMESPACE {
namespace common {

bool pin_this_thread_to_cpu([[maybe_unused]] unsigned cpu) {
#if defined(__linux__)
    // CPU_SETSIZE bounds what cpu_set_t can express; a request past it is refused rather than
    // wrapped, because a silently-different CPU is the failure this function exists to expose.
    if (cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#elif defined(_WIN32)
    // SetThreadAffinityMask takes a KAFFINITY, which addresses one processor GROUP -- at most 64
    // logical CPUs. Machines wider than that need SetThreadGroupAffinity and a group index, so a
    // request past 64 is refused here rather than aliased onto the wrong processor.
    if (cpu >= 64) return false;
    const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << cpu;
    return ::SetThreadAffinityMask(::GetCurrentThread(), mask) != 0;
#else
    // macOS has no thread-to-core binding to call. Reporting failure is the whole contract:
    // a caller that needs a homogeneous core set learns it did not get one.
    return false;
#endif
}

}  // namespace common
}  // namespace HG_NAMESPACE
