#include "hgcommon/affinity.hpp"

// The affinity body, and the platform headers it calls into. affinity.hpp names only the backend
// SELECTION, so a translation unit that pins parses <cstddef> rather than windows.h or sched.h.
//
// This lives under job_system for the reason park.cpp does: job_system is the only consumer and
// the only target that could hold a body, and hgcommon is headers with no library of its own.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>

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
#  include <memory>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#  include <cpuid.h>
#  define HG_HAVE_CPUID 1
#elif (defined(_M_X64) || defined(_M_IX86)) && defined(_MSC_VER)
#  include <intrin.h>
#  define HG_HAVE_CPUID 1
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

bool exempt_this_thread_from_power_throttling() {
#if defined(_WIN32)
    // The thread form of PROCESS_POWER_THROTTLING_STATE has the same three fields; MinGW's headers
    // name the information class but not the thread struct, so the process struct is passed.
    PROCESS_POWER_THROTTLING_STATE state{};
    state.Version = 1;        // THREAD_POWER_THROTTLING_CURRENT_VERSION
    state.ControlMask = 1;    // THREAD_POWER_THROTTLING_EXECUTION_SPEED
    state.StateMask = 0;      // the control bit cleared: not throttled
    return ::SetThreadInformation(::GetCurrentThread(), ThreadPowerThrottling, &state,
                                  sizeof(state)) != 0;
#else
    return false;
#endif
}

namespace {
// Dense 0..k-1 ids for a list of opaque keys, preserving "equal key means equal id". The distinct
// keys number the cores or caches on the machine, so the inner scan is over a handful of entries.
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

#if defined(HG_HAVE_CPUID)
void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#  if defined(_MSC_VER)
    int v[4];
    __cpuidex(v, static_cast<int>(leaf), static_cast<int>(sub));
    for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(v[i]);
#  else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#  endif
}

// The 48-byte brand string of CPUID leaves 0x80000002-4, leading spaces removed.
std::string cpuid_brand() {
    unsigned r[4];
    cpuid(0x80000000u, 0, r);
    if (r[0] < 0x80000004u) return {};
    char s[49] = {0};
    for (unsigned i = 0; i < 3; ++i) {
        cpuid(0x80000002u + i, 0, r);
        std::memcpy(s + 16 * i, r, 16);
    }
    const char* p = s;
    while (*p == ' ') ++p;
    return p;
}

// CPUID.07H:EDX[15], set on a part with more than one core type.
bool cpuid_hybrid() {
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] < 7) return false;
    cpuid(7, 0, r);
    return (r[3] >> 15) & 1u;
}
#endif
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

std::vector<unsigned> allowed_cpus() {
    std::vector<unsigned> out;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return out;
    for (unsigned c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set)) out.push_back(c);
#elif defined(_WIN32)
    DWORD_PTR process = 0, system = 0;
    if (!::GetProcessAffinityMask(::GetCurrentProcess(), &process, &system)) return out;
    for (unsigned c = 0; c < 64; ++c)
        if (process & (static_cast<DWORD_PTR>(1) << c)) out.push_back(c);
#endif
    return out;
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

// The first whitespace-delimited token of a one-line file.
bool read_token(const char* path, char* out, size_t cap) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return false;
    const bool ok = std::fgets(out, static_cast<int>(cap), f) != nullptr;
    std::fclose(f);
    if (!ok) return false;
    for (char* p = out; *p; ++p) if (*p == '\n' || *p == ' ') { *p = 0; break; }
    return out[0] != 0;
}

bool read_ulong(const char* path, unsigned long long& v) {
    char buf[64];
    if (!read_token(path, buf, sizeof buf)) return false;
    char* end = nullptr;
    v = std::strtoull(buf, &end, 10);
    return end != buf;
}

// sysfs cache sizes read "2048K" or "36864K"; a bare number is bytes.
bool read_size(const char* path, unsigned long long& bytes) {
    char buf[64];
    if (!read_token(path, buf, sizeof buf)) return false;
    char* end = nullptr;
    bytes = std::strtoull(buf, &end, 10);
    if (end == buf) return false;
    if (*end == 'K') bytes <<= 10;
    else if (*end == 'M') bytes <<= 20;
    else if (*end == 'G') bytes <<= 30;
    return true;
}

// /proc/meminfo "Name:   123 kB", in bytes.
bool meminfo_bytes(const char* name, unsigned long long& bytes) {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return false;
    char line[256];
    const size_t len = std::strlen(name);
    bool found = false;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, name, len) == 0 && line[len] == ':') {
            bytes = std::strtoull(line + len + 1, nullptr, 10) << 10;
            found = true;
            break;
        }
    }
    std::fclose(f);
    return found;
}

#if defined(HG_HAVE_CPUID)
// CPUID.1AH:EAX[31:24] of logical CPU `cpu`, read by moving the calling thread there and back:
// 0x40 is a performance core, 0x20 an efficiency core, 0 when the leaf is absent or masked.
unsigned cpuid_core_type_on(unsigned cpu) {
    cpu_set_t old, one;
    if (sched_getaffinity(0, sizeof old, &old) != 0) return 0;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    if (sched_setaffinity(0, sizeof one, &one) != 0) return 0;
    unsigned r[4];
    cpuid(0, 0, r);
    unsigned type = 0;
    if (r[0] >= 0x1A) { cpuid(0x1A, 0, r); type = r[0] >> 24; }
    sched_setaffinity(0, sizeof old, &old);
    return type;
}
#endif

void assign_linux_core_classes(HostTopology& t);

void read_linux(HostTopology& t) {
    std::vector<unsigned> online;
    if (!read_cpu_list("/sys/devices/system/cpu/online", online) || online.empty())
        for (unsigned c = 0; c < std::thread::hardware_concurrency(); ++c) online.push_back(c);

    const std::vector<unsigned> allowed = allowed_cpus();
    std::vector<unsigned long long> core_keys, llc_keys;
    std::vector<unsigned long long> llc_size_of_cpu, l2_of_cpu;
    bool cache_known = true;
    for (unsigned c : online) {
        HostCpu e;
        e.cpu = c;
        e.allowed = allowed.empty() || std::find(allowed.begin(), allowed.end(), c) != allowed.end();
        char path[160];
        unsigned long long pkg = 0, core = c;
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%u/topology/physical_package_id", c);
        read_ulong(path, pkg);
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%u/topology/core_id", c);
        if (!read_ulong(path, core)) core = c | (1ull << 31);   // no topology: its own core
        core_keys.push_back((pkg << 32) | core);

        // The deepest cache index that reports an `id` is the last-level cache; the kernel numbers
        // each cache instance, so equal ids at one level mean one cache.
        unsigned long long best_level = 0, key = 0, llc_size = 0, l2 = 0;
        bool found = false;
        for (unsigned idx = 0; idx < 16; ++idx) {
            unsigned long long level = 0, id = 0, size = 0;
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%u/cache/index%u/level", c, idx);
            if (!read_ulong(path, level)) break;   // the indices are contiguous
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%u/cache/index%u/size", c, idx);
            const bool has_size = read_size(path, size);
            char type[32] = {0};
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%u/cache/index%u/type", c, idx);
            read_token(path, type, sizeof type);
            if (level == 2 && has_size && std::strcmp(type, "Instruction") != 0) l2 = size;
            std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%u/cache/index%u/id", c, idx);
            if (!read_ulong(path, id)) continue;
            if (found && level <= best_level) continue;
            best_level = level;
            key = (level << 32) | id;
            llc_size = has_size ? size : 0;
            found = true;
        }
        if (!found) cache_known = false;
        llc_keys.push_back(key);
        llc_size_of_cpu.push_back(llc_size);
        l2_of_cpu.push_back(l2);
        t.cpus.push_back(e);
    }

    const std::vector<unsigned> core_ids = densify_keys(core_keys);
    for (size_t i = 0; i < t.cpus.size(); ++i) t.cpus[i].core = core_ids[i];
    if (cache_known) {
        const std::vector<unsigned> llc_ids = densify_keys(llc_keys);
        for (size_t i = 0; i < t.cpus.size(); ++i) {
            t.cpus[i].llc = llc_ids[i];
            if (t.llc_bytes.size() <= llc_ids[i]) t.llc_bytes.resize(llc_ids[i] + 1, 0);
            t.llc_bytes[llc_ids[i]] = llc_size_of_cpu[i];
        }
    }
    t.cache_known = cache_known && !t.cpus.empty();

    assign_linux_core_classes(t);
    for (size_t i = 0; i < t.cpus.size(); ++i)
        if (t.cpus[i].core_class == 0) { t.l2_bytes = l2_of_cpu[i]; break; }
}

// Core classes. Intel hybrid: the kernel publishes the P-cores as the cpu_core PMU. ARM
// big.LITTLE: cpu_capacity per CPU, ranked from the largest. Intel hybrid without the PMU split
// (an older kernel): CPUID leaf 0x1A on each CPU. None of these: one class.
void assign_linux_core_classes(HostTopology& t) {
    std::vector<unsigned> pcores;
    if (read_cpu_list("/sys/devices/cpu_core/cpus", pcores) && !pcores.empty()) {
        for (auto& e : t.cpus)
            e.core_class = std::find(pcores.begin(), pcores.end(), e.cpu) != pcores.end() ? 0u : 1u;
        return;
    }
    std::vector<unsigned long long> caps;
    for (const auto& e : t.cpus) {
        char path[128];
        unsigned long long cap = 0;
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%u/cpu_capacity", e.cpu);
        if (!read_ulong(path, cap)) { caps.clear(); break; }
        caps.push_back(cap);
    }
    if (!caps.empty()) {
        std::vector<unsigned long long> distinct = caps;
        std::sort(distinct.begin(), distinct.end(), std::greater<unsigned long long>());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        if (distinct.size() > 1) {
            for (size_t i = 0; i < t.cpus.size(); ++i)
                t.cpus[i].core_class = static_cast<unsigned>(
                    std::find(distinct.begin(), distinct.end(), caps[i]) - distinct.begin());
            return;
        }
    }
#if defined(HG_HAVE_CPUID) && !defined(HG_VERIFICATION)
    if (cpuid_hybrid()) {
        for (auto& e : t.cpus) e.core_class = cpuid_core_type_on(e.cpu) == 0x20 ? 1u : 0u;
    }
#endif
}

void read_linux_platform(HostTopology& t) {
    unsigned long long v = 0;
    if (meminfo_bytes("MemTotal", v)) t.total_bytes = v;
    if (meminfo_bytes("MemAvailable", v)) t.available_bytes = v;
    if (read_ulong("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", v)) t.huge_page_bytes = v;
    char rel[256] = {0};
    if (std::FILE* f = std::fopen("/proc/sys/kernel/osrelease", "r")) {
        if (std::fgets(rel, sizeof rel, f)) {
            for (char* p = rel; *p; ++p) if (*p >= 'A' && *p <= 'Z') *p = static_cast<char>(*p - 'A' + 'a');
            t.virtualised = std::strstr(rel, "microsoft") != nullptr;
        }
        std::fclose(f);
    }
}
}  // namespace
#endif

#if defined(_WIN32)
namespace {
void read_windows(HostTopology& t) {
    DWORD bytes = 0;
    ::GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes);
    if (bytes == 0) return;
    std::unique_ptr<unsigned char[]> buf(new unsigned char[bytes]);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get());
    if (!::GetLogicalProcessorInformationEx(RelationAll, info, &bytes)) return;

    // Group 0 only: SetThreadAffinityMask addresses group 0, so CPUs in another group are CPUs
    // this process cannot pin to.
    std::vector<int> core_of(64, -1), class_of(64, 0), llc_of(64, -1);
    std::vector<BYTE> classes;
    BYTE deepest = 0;
    int cores = 0;
    for (DWORD off = 0; off < bytes; ) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get() + off);
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount > 0 &&
            e->Processor.GroupMask[0].Group == 0) {
            const BYTE ec = efficiency_class_of(&e->Processor);
            if (std::find(classes.begin(), classes.end(), ec) == classes.end()) classes.push_back(ec);
            for (unsigned bit = 0; bit < 64; ++bit)
                if (e->Processor.GroupMask[0].Mask & (static_cast<KAFFINITY>(1) << bit)) {
                    core_of[bit] = cores;
                    class_of[bit] = ec;
                }
            ++cores;
        } else if (e->Relationship == RelationCache && e->Cache.Level > deepest) {
            deepest = e->Cache.Level;
        }
        off += e->Size;
    }
    // EfficiencyClass is larger for faster cores; core_class counts down from the fastest.
    std::sort(classes.begin(), classes.end(), std::greater<BYTE>());
    int llcs = 0;
    for (DWORD off = 0; off < bytes; ) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.get() + off);
        if (e->Relationship == RelationCache && e->Cache.GroupMask.Group == 0 &&
            e->Cache.Type != CacheInstruction) {
            const KAFFINITY m = e->Cache.GroupMask.Mask;
            if (e->Cache.Level == deepest) {
                for (unsigned bit = 0; bit < 64; ++bit)
                    if (m & (static_cast<KAFFINITY>(1) << bit)) llc_of[bit] = llcs;
                t.llc_bytes.push_back(e->Cache.CacheSize);
                ++llcs;
            }
            if (e->Cache.Level == 2 && t.l2_bytes == 0) t.l2_bytes = e->Cache.CacheSize;
        }
        off += e->Size;
    }
    const std::vector<unsigned> allowed = allowed_cpus();
    bool cache_known = deepest != 0;
    for (unsigned c = 0; c < 64; ++c) {
        if (core_of[c] < 0) continue;
        HostCpu h;
        h.cpu = c;
        h.core = static_cast<unsigned>(core_of[c]);
        h.core_class = static_cast<unsigned>(
            std::find(classes.begin(), classes.end(), static_cast<BYTE>(class_of[c])) - classes.begin());
        if (llc_of[c] < 0) cache_known = false;
        else h.llc = static_cast<unsigned>(llc_of[c]);
        h.allowed = allowed.empty() || std::find(allowed.begin(), allowed.end(), c) != allowed.end();
        t.cpus.push_back(h);
    }
    t.cache_known = cache_known && !t.cpus.empty();
    if (!t.cache_known) {
        t.llc_bytes.clear();
        for (auto& h : t.cpus) h.llc = 0;
    }

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (::GlobalMemoryStatusEx(&ms)) {
        t.total_bytes = ms.ullTotalPhys;
        t.available_bytes = ms.ullAvailPhys;
    }
    t.huge_page_bytes = ::GetLargePageMinimum();
}
}  // namespace
#endif

#if defined(__APPLE__)
namespace {
bool sysctl_u64(const char* name, unsigned long long& v) {
    uint64_t x = 0;
    size_t len = sizeof x;
    if (::sysctlbyname(name, &x, &len, nullptr, 0) != 0) return false;
    v = len == sizeof(uint32_t) ? static_cast<uint32_t>(x) : x;
    return true;
}

// macOS names no CPU index a thread can bind to, so `cpus` stays empty and only the counts,
// caches and memory are filled.
void read_apple(HostTopology& t) {
    unsigned long long v = 0;
    if (sysctl_u64("hw.logicalcpu", v)) t.logical_cpus = static_cast<unsigned>(v);
    if (sysctl_u64("hw.physicalcpu", v)) t.physical_cores = static_cast<unsigned>(v);
    unsigned long long p0 = 0, p1 = 0;
    if (sysctl_u64("hw.perflevel0.physicalcpu", p0)) {
        t.performance_cores = static_cast<unsigned>(p0);
        if (sysctl_u64("hw.perflevel1.physicalcpu", p1) && p1 > 0) t.core_classes = 2;
    } else {
        t.performance_cores = t.physical_cores;
    }
    if (sysctl_u64("hw.perflevel0.l2cachesize", v) || sysctl_u64("hw.l2cachesize", v)) t.l2_bytes = v;
    if (sysctl_u64("hw.l3cachesize", v) && v > 0) t.llc_bytes.push_back(v);
    if (sysctl_u64("hw.memsize", v)) t.total_bytes = v;
    if (t.physical_cores > 0 && t.logical_cpus >= t.physical_cores)
        t.smt_width = t.logical_cpus / t.physical_cores;
    char brand[256] = {0};
    size_t len = sizeof brand;
    if (::sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0) t.model = brand;
}
}  // namespace
#endif

namespace {
HostTopology read_host_topology() {
    HostTopology t;
#if defined(__linux__)
    read_linux(t);
    read_linux_platform(t);
#elif defined(_WIN32)
    read_windows(t);
#elif defined(__APPLE__)
    read_apple(t);
#endif
#if defined(HG_HAVE_CPUID)
    if (t.model.empty()) t.model = cpuid_brand();
#endif
    if (!t.cpus.empty()) {
        t.logical_cpus = static_cast<unsigned>(t.cpus.size());
        unsigned cores = 0, classes = 1;
        std::vector<unsigned> per_core;
        for (const auto& e : t.cpus) {
            cores = std::max(cores, e.core + 1);
            classes = std::max(classes, e.core_class + 1);
        }
        per_core.assign(cores, 0);
        std::vector<bool> fast(cores, false);
        for (auto& e : t.cpus) {
            e.first_thread = per_core[e.core]++ == 0;
            if (e.core_class == 0) fast[e.core] = true;
        }
        t.physical_cores = cores;
        t.core_classes = classes;
        t.smt_width = *std::max_element(per_core.begin(), per_core.end());
        t.performance_cores = static_cast<unsigned>(std::count(fast.begin(), fast.end(), true));
    }
    return t;
}
}  // namespace

int HostTopology::index_of(unsigned cpu) const {
    for (size_t i = 0; i < cpus.size(); ++i)
        if (cpus[i].cpu == cpu) return static_cast<int>(i);
    return -1;
}

const HostTopology& host_topology() {
    static const HostTopology t = read_host_topology();
    return t;
}

// One entry per PHYSICAL core of the fastest class (its first hardware thread), and only on a
// part with more than one class.
std::vector<unsigned> performance_cpus() {
    std::vector<unsigned> out;
    const HostTopology& t = host_topology();
    if (t.core_classes < 2) return out;
    for (const auto& e : t.cpus)
        if (e.core_class == 0 && e.first_thread) out.push_back(e.cpu);
    return out;
}

std::vector<unsigned> cache_domains_of(const std::vector<unsigned>& cpus) {
    const HostTopology& t = host_topology();
    if (cpus.empty() || !t.cache_known) return {};
    std::vector<unsigned long long> keys;
    keys.reserve(cpus.size());
    for (unsigned c : cpus) {
        const int i = t.index_of(c);
        if (i < 0) return {};                          // a CPU the topology does not describe
        keys.push_back(t.cpus[static_cast<size_t>(i)].llc);
    }
    return densify_keys(keys);
}

std::vector<unsigned> default_worker_cpus() {
    if (!affinity_supported()) return {};
    const HostTopology& t = host_topology();
    std::vector<unsigned> cpus = performance_cpus();
    if (cpus.empty())
        for (const auto& e : t.cpus) cpus.push_back(e.cpu);
    std::vector<unsigned> kept;
    for (unsigned c : cpus) {
        const int i = t.index_of(c);
        if (i >= 0 && t.cpus[static_cast<size_t>(i)].allowed) kept.push_back(c);
    }
    if (kept.size() < 2) return {};
    const std::vector<unsigned> dom = cache_domains_of(kept);
    if (dom.size() != kept.size()) return {};
    bool many = false;
    for (size_t i = 1; i < dom.size(); ++i) if (dom[i] != dom[0]) { many = true; break; }
    if (!many) return {};
    std::vector<size_t> idx(kept.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return dom[a] < dom[b]; });
    std::vector<unsigned> packed;
    packed.reserve(kept.size());
    for (size_t i : idx) packed.push_back(kept[i]);
    return packed;
}

}  // namespace common
}  // namespace HG_NAMESPACE
