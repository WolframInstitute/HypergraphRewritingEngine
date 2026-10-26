#include "hgcommon/memory_limit.hpp"
#include "hgcommon/affinity.hpp"   // host_topology: physical memory

#include <atomic>

#if defined(HG_VERIFICATION)
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

namespace HG_NAMESPACE {
namespace common {

uint64_t resolved_memory_limit(const TuningOverrides& job, TuneSource* source) {
#if defined(HG_VERIFICATION)
    (void)job;
    if (source) *source = TuneSource::Derived;
    return kNoMemoryLimit;
#else
    const uint64_t physical = host_topology().total_bytes;
    const int64_t v = tuned(job, Tune::MemoryLimit, [&] {
        const uint64_t d = derived_memory_limit(physical);
        return static_cast<int64_t>(d > uint64_t(INT64_MAX) ? uint64_t(INT64_MAX) : d);
    }, source);
    const uint64_t bytes = v == INT64_MAX ? kNoMemoryLimit : static_cast<uint64_t>(v);
    return clamp_memory_limit(bytes, physical);
#endif
}

const char* memory_limit_backend_name(MemoryLimitBackend b) {
    switch (b) {
        case MemoryLimitBackend::JobObject:  return "Job Object";
        case MemoryLimitBackend::RlimitData: return "RLIMIT_DATA";
        case MemoryLimitBackend::None:       return "none";
    }
    return "none";
}

namespace {
// The installed limit; 0 before install_process_memory_limit succeeds.
std::atomic<uint64_t> g_installed{0};
#if !defined(HG_VERIFICATION) && defined(_WIN32)
HANDLE g_job = nullptr;   // held open for the life of the process

bool apply_job_limit(HANDLE job, uint64_t bytes) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY;
    const uint64_t cap = bytes > uint64_t(SIZE_MAX) ? uint64_t(SIZE_MAX) : bytes;
    info.ProcessMemoryLimit = static_cast<SIZE_T>(cap);
    info.JobMemoryLimit = static_cast<SIZE_T>(cap);
    return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info,
                                   sizeof info) != 0;
}
#elif !defined(HG_VERIFICATION) && defined(__linux__)
bool apply_rlimit_data(uint64_t bytes) {
    rlimit r{};
    if (getrlimit(RLIMIT_DATA, &r) != 0) return false;
    const rlim_t want = bytes >= uint64_t(RLIM_INFINITY) ? RLIM_INFINITY : static_cast<rlim_t>(bytes);
    r.rlim_cur = r.rlim_max != RLIM_INFINITY && want > r.rlim_max ? r.rlim_max : want;
    return setrlimit(RLIMIT_DATA, &r) == 0;
}
#endif
}  // namespace

bool install_process_memory_limit(uint64_t bytes) {
#if defined(HG_VERIFICATION)
    (void)bytes;
    return false;
#elif defined(_WIN32)
    if (g_job) return set_process_memory_limit(bytes);
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (!job) return false;
    if (!apply_job_limit(job, bytes) || !AssignProcessToJobObject(job, GetCurrentProcess())) {
        CloseHandle(job);
        return false;
    }
    g_job = job;
    g_installed.store(bytes, std::memory_order_relaxed);
    return true;
#elif defined(__linux__)
    rlimit r{};
    if (getrlimit(RLIMIT_DATA, &r) != 0) return false;
    const uint64_t physical = host_topology().total_bytes;
    const uint64_t ceiling = physical > bytes ? physical : bytes;
    if (ceiling != kNoMemoryLimit && (r.rlim_max == RLIM_INFINITY || ceiling < r.rlim_max)) {
        r.rlim_max = static_cast<rlim_t>(ceiling);
        if (r.rlim_cur == RLIM_INFINITY || r.rlim_cur > r.rlim_max) r.rlim_cur = r.rlim_max;
        if (setrlimit(RLIMIT_DATA, &r) != 0) return false;
    }
    if (!apply_rlimit_data(bytes)) return false;
    g_installed.store(bytes, std::memory_order_relaxed);
    return true;
#else
    (void)bytes;
    return false;
#endif
}

bool set_process_memory_limit(uint64_t bytes) {
    if (g_installed.load(std::memory_order_relaxed) == 0) return false;
#if defined(HG_VERIFICATION)
    (void)bytes;
    return false;
#elif defined(_WIN32)
    if (!apply_job_limit(g_job, bytes)) return false;
#elif defined(__linux__)
    if (!apply_rlimit_data(bytes)) return false;
#else
    (void)bytes;
    return false;
#endif
    g_installed.store(bytes, std::memory_order_relaxed);
    return true;
}

uint64_t process_memory_limit() { return g_installed.load(std::memory_order_relaxed); }

}  // namespace common
}  // namespace HG_NAMESPACE
