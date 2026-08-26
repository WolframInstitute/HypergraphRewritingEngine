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
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#elif defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX   // windows.h's min/max macros poison std::numeric_limits<T>::max()
#  endif
#  include <windows.h>
#  include <memory>
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

#if defined(__linux__)
namespace {
// Reads a sysfs CPU list ("0-7", "0,2,4", "0-3,8-11") into logical CPU indices. Returns false
// when the file is absent, which is the ordinary answer on a machine with no such distinction
// and must not be confused with an empty list that the file actually contained.
bool read_cpu_list(const char* path, std::vector<unsigned>& out) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char buf[4096] = {0};
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    if (n == 0) return false;
    for (char* tok = std::strtok(buf, ",\n"); tok; tok = std::strtok(nullptr, ",\n")) {
        char* dash = std::strchr(tok, '-');
        const unsigned lo = static_cast<unsigned>(std::strtoul(tok, nullptr, 10));
        const unsigned hi = dash ? static_cast<unsigned>(std::strtoul(dash + 1, nullptr, 10)) : lo;
        for (unsigned c = lo; c <= hi && c < CPU_SETSIZE; ++c) out.push_back(c);
    }
    return true;
}

// The first hardware thread of each physical core among `cpus`, read from sysfs. A speedup
// denominator counts cores; two siblings of one core are not two of anything.
std::vector<unsigned> first_thread_per_core(const std::vector<unsigned>& cpus) {
    std::vector<unsigned> out;
    std::vector<unsigned> seen_core_ids;
    for (unsigned c : cpus) {
        char path[128];
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%u/topology/core_id", c);
        std::FILE* f = std::fopen(path, "r");
        if (!f) { out.push_back(c); continue; }   // no topology: take the CPU as its own core
        unsigned long core_id = 0;
        const int got = std::fscanf(f, "%lu", &core_id);
        std::fclose(f);
        if (got != 1) { out.push_back(c); continue; }
        bool dup = false;
        for (unsigned s : seen_core_ids) if (s == core_id) { dup = true; break; }
        if (dup) continue;
        seen_core_ids.push_back(static_cast<unsigned>(core_id));
        out.push_back(c);
    }
    return out;
}

// The kernel's own id for the DEEPEST cache instance `cpu` reports, packed above its level so
// that two levels which both number their instances from zero cannot be read as one cache.
// False when the CPU exposes no cache index carrying an id, which is the answer under a
// hypervisor that presents no cache topology.
bool deepest_cache_key(unsigned cpu, unsigned long long& key) {
    bool found = false;
    unsigned long best_level = 0;
    for (unsigned idx = 0; idx < 16; ++idx) {
        char path[160];
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%u/cache/index%u/level", cpu, idx);
        std::FILE* f = std::fopen(path, "r");
        if (!f) break;                       // the indices are contiguous; a gap ends the list
        unsigned long level = 0;
        const int got = std::fscanf(f, "%lu", &level);
        std::fclose(f);
        if (got != 1) continue;
        if (found && level <= best_level) continue;
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%u/cache/index%u/id", cpu, idx);
        f = std::fopen(path, "r");
        if (!f) continue;                    // an index with no id cannot name an instance
        unsigned long id = 0;
        const int got_id = std::fscanf(f, "%lu", &id);
        std::fclose(f);
        if (got_id != 1) continue;
        best_level = level;
        key = (static_cast<unsigned long long>(level) << 32) | id;
        found = true;
    }
    return found;
}
}  // namespace
#endif

namespace {
// Dense 0..k-1 ids for a list of opaque cache keys, preserving "equal key means equal domain".
// One entry per worker CPU, and the distinct keys number the caches on the machine, so the
// inner scan is over a handful of entries.
std::vector<unsigned> densify_keys(const std::vector<unsigned long long>& keys) {
    std::vector<unsigned long long> distinct;
    std::vector<unsigned> out;
    out.reserve(keys.size());
    for (unsigned long long k : keys) {
        size_t d = 0;
        for (; d < distinct.size(); ++d) if (distinct[d] == k) break;
        if (d == distinct.size()) distinct.push_back(k);
        out.push_back(static_cast<unsigned>(d));
    }
    return out;
}
}  // namespace

#if defined(_WIN32)
namespace {
// EfficiencyClass is the byte at offset 1 of PROCESSOR_RELATIONSHIP, immediately after Flags.
// Windows 10 added it INSIDE the struct's reserved area, so the layout never changed: the
// Windows SDK names the field, while mingw-w64's winnt.h of this vintage still declares the
// same byte as the head of Reserved[21]. Reading it positionally is ABI-identical on both and
// COMPILES on both, where the field name compiles on only one -- which the cross build caught.
static_assert(sizeof(PROCESSOR_RELATIONSHIP) >= 2, "PROCESSOR_RELATIONSHIP smaller than Flags+EfficiencyClass");
inline BYTE efficiency_class_of(const PROCESSOR_RELATIONSHIP* p) {
    return reinterpret_cast<const BYTE*>(p)[1];
}
}  // namespace
#endif

std::vector<unsigned> performance_cpus() {
    std::vector<unsigned> out;
#if defined(__linux__)
    // Intel hybrid: the kernel publishes the two core types as separate PMUs, and the P-core
    // list is exactly the set a homogeneous measurement wants.
    std::vector<unsigned> cpus;
    if (read_cpu_list("/sys/devices/cpu_core/cpus", cpus) && !cpus.empty()) {
        return first_thread_per_core(cpus);
    }
    // ARM big.LITTLE: no such PMU split, but each CPU carries a capacity and the largest value
    // is the big cluster. A machine where every capacity is equal is homogeneous, and the
    // empty return says so.
    unsigned long best = 0;
    std::vector<unsigned> best_cpus;
    bool any_capacity = false, differing = false;
    for (unsigned c = 0; c < 4096; ++c) {
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cpu_capacity", c);
        std::FILE* f = std::fopen(path, "r");
        if (!f) { if (c > 0) break; else continue; }
        unsigned long cap = 0;
        const int got = std::fscanf(f, "%lu", &cap);
        std::fclose(f);
        if (got != 1) continue;
        any_capacity = true;
        if (cap > best) { if (best != 0) differing = true; best = cap; best_cpus.clear(); }
        else if (cap < best) { differing = true; continue; }
        if (cap == best) best_cpus.push_back(c);
    }
    if (any_capacity && differing) return first_thread_per_core(best_cpus);
    return out;                                   // homogeneous, or nothing to read
#elif defined(_WIN32)
    // EfficiencyClass is per physical core and only meaningful RELATIVE to the others on the
    // machine: the highest value present is the performance class. A part with one class is
    // homogeneous and answers empty.
    DWORD bytes = 0;
    ::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes == 0) return out;
    std::unique_ptr<unsigned char[]> buf(new unsigned char[bytes]);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get());
    if (!::GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bytes)) return out;

    BYTE best_class = 0;
    bool differing = false, first = true;
    for (DWORD off = 0; off < bytes; ) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get() + off);
        if (e->Relationship == RelationProcessorCore) {
            const BYTE ec = efficiency_class_of(&e->Processor);
            if (first) { best_class = ec; first = false; }
            else if (ec != best_class) { differing = true; if (ec > best_class) best_class = ec; }
        }
        off += e->Size;
    }
    if (!differing) return out;                   // one class: homogeneous, nothing to choose

    for (DWORD off = 0; off < bytes; ) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get() + off);
        if (e->Relationship == RelationProcessorCore &&
            efficiency_class_of(&e->Processor) == best_class && e->Processor.GroupCount > 0) {
            // One entry per physical core: the LOWEST set bit of the core's mask is its first
            // hardware thread, and SetThreadAffinityMask addresses group 0 only, so a core in
            // another group is skipped rather than aliased onto the wrong processor.
            const GROUP_AFFINITY& ga = e->Processor.GroupMask[0];
            if (ga.Group == 0) {
                for (unsigned bit = 0; bit < 64; ++bit) {
                    if (ga.Mask & (static_cast<KAFFINITY>(1) << bit)) { out.push_back(bit); break; }
                }
            }
        }
        off += e->Size;
    }
    return out;
#else
    return out;                                   // macOS: no binding, so no set to name
#endif
}

std::vector<unsigned> cache_domains_of([[maybe_unused]] const std::vector<unsigned>& cpus) {
    std::vector<unsigned> out;
    if (cpus.empty()) return out;
#if defined(__linux__)
    std::vector<unsigned long long> keys;
    keys.reserve(cpus.size());
    for (unsigned c : cpus) {
        unsigned long long key = 0;
        if (!deepest_cache_key(c, key)) return {};   // one unreadable CPU voids the whole answer
        keys.push_back(key);
    }
    return densify_keys(keys);
#elif defined(_WIN32)
    // The deepest level present is the last-level cache. Each RelationCache entry at that level
    // is one instance, and its GroupMask names the CPUs that share it.
    DWORD bytes = 0;
    ::GetLogicalProcessorInformationEx(RelationCache, nullptr, &bytes);
    if (bytes == 0) return out;
    std::unique_ptr<unsigned char[]> buf(new unsigned char[bytes]);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get());
    if (!::GetLogicalProcessorInformationEx(RelationCache, info, &bytes)) return out;

    BYTE deepest = 0;
    for (DWORD off = 0; off < bytes; ) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get() + off);
        if (e->Relationship == RelationCache && e->Cache.Level > deepest) deepest = e->Cache.Level;
        off += e->Size;
    }
    if (deepest == 0) return out;

    std::vector<unsigned long long> keys(cpus.size(), 0);
    unsigned long long instance = 1;              // 0 stays free to mean "matched no instance"
    for (DWORD off = 0; off < bytes; ) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get() + off);
        if (e->Relationship == RelationCache && e->Cache.Level == deepest) {
            // SetThreadAffinityMask addresses group 0, so a cache instance in another group
            // describes CPUs this process cannot pin to and is skipped rather than aliased.
            const GROUP_AFFINITY& ga = e->Cache.GroupMask;
            if (ga.Group == 0) {
                for (size_t i = 0; i < cpus.size(); ++i) {
                    if (cpus[i] < 64 && (ga.Mask & (static_cast<KAFFINITY>(1) << cpus[i])))
                        keys[i] = instance;
                }
            }
            ++instance;
        }
        off += e->Size;
    }
    for (unsigned long long k : keys) if (k == 0) return {};
    return densify_keys(keys);
#else
    return out;                                   // macOS: no binding, so nothing to group
#endif
}

}  // namespace common
}  // namespace HG_NAMESPACE
