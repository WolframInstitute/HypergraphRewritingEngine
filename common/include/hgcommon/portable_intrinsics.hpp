#pragma once

// Portable spellings of the compiler intrinsics the engine relies on, so the
// same source builds under GCC/Clang (the cross-compiles we ship today) and
// under MSVC cl.exe (required as nvcc's host compiler for the native Windows
// CUDA build). The GCC/Clang path is bit-for-bit the previous direct __builtin_*
// calls; only the MSVC branch is new. The MSVC branch targets x86/x64 (the
// Windows CUDA host); ARM-MSVC is not a build target.

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace hgcommon {

inline int popcount(uint32_t x) {
#if defined(_MSC_VER)
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
inline int ctz(uint32_t x) {
#if defined(_MSC_VER)
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

// CPU relaxation hint for bounded spin-wait loops: PAUSE on x86, YIELD on ARM.
inline void cpu_relax() {
#if defined(_MSC_VER)
#if defined(_M_ARM64) || defined(_M_ARM)
    __yield();
#else
    _mm_pause();
#endif
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#endif
}

}  // namespace hgcommon
