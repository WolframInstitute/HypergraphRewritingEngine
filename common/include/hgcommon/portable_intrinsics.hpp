#pragma once
#include "hgcommon/namespace.hpp"

// Portable spellings of the compiler intrinsics the engine relies on, so the
// same source builds under GCC/Clang (the cross-compiles we ship today) and
// under MSVC cl.exe (required as nvcc's host compiler for the native Windows
// CUDA build). The GCC/Clang path is bit-for-bit the previous direct __builtin_*
// calls; only the MSVC branch is new. The MSVC branch targets x86/x64 (the
// Windows CUDA host); ARM-MSVC is not a build target.

// EVERY BODY IN THIS FILE IS INLINE ON PURPOSE, and it is the one header in the project where
// that is the answer rather than an omission. Each function compiles to a single instruction, so
// an out-of-line definition would cost a call to save that instruction. The HG_HD ones are
// reached from device code, which has no library to link against at all. And hgcommon is a
// header-only surface shared by the engine and the CUDA port with no target of its own, so there
// is no object file these could go in that both sides already link.

#include <cstdint>

#include "hgcommon/core.hpp"

// MSVC declares _BitScanForward and __popcnt here. Without it they are undefined
// identifiers, and nvcc driving cl.exe reports exactly that -- which is how the native
// Windows CUDA build failed while every GCC/Clang build passed.
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace HG_NAMESPACE {
namespace common {

// A monotonic cycle counter, for attributing time to phases INSIDE a run.
//
// Wall-clock timers are the wrong instrument here twice over: a phase runs for microseconds,
// and every worker is inside one, so the quantity wanted is cycles spent per phase summed over
// workers rather than elapsed time. This is the host twin of the device's clock64() phase
// accounting, and it carries the same caveat -- the counter runs while a thread is stalled or
// descheduled, so a bucket is ELAPSED cycles in that phase, not issued work, and the buckets
// are meaningful as fractions of their sum rather than as absolute time.
//
// rdtsc is invariant on every x86-64 part this engine targets (constant_tsc), so it does not
// vary with frequency scaling. On AArch64 the virtual counter serves the same purpose at a
// lower, fixed frequency. Neither is serialised: an out-of-order core can move the read across
// nearby instructions, which is acceptable for attributing microsecond phases and would not be
// for timing individual instructions.
inline uint64_t cycle_counter() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __rdtsc();
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return 0;   // no counter: every bucket reads zero, which is visibly "not measured"
#endif
}

HG_HD inline int popcount(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#elif defined(_MSC_VER)
    return static_cast<int>(__popcnt(x));
#else
    return __builtin_popcount(x);
#endif
}

inline int popcount64(uint64_t x) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

// Count trailing zeros. Undefined for x == 0, matching __builtin_ctz.
HG_HD inline int ctz(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __ffs(static_cast<int>(x)) - 1;
#elif defined(_MSC_VER)
    unsigned long i;
    _BitScanForward(&i, x);
    return static_cast<int>(i);
#else
    return __builtin_ctz(x);
#endif
}

inline int ctz64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanForward64(&i, x);
    return static_cast<int>(i);
#else
    return __builtin_ctzll(x);
#endif
}

}  // namespace common
}  // namespace HG_NAMESPACE
