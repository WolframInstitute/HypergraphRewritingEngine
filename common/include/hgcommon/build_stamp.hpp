#pragma once
#include "hgcommon/namespace.hpp"
//
// The configuration an artifact was built with, written into the artifact as a byte string.
//
// A release ships binaries for six platforms plus two CUDA executables, and the machine that
// assembles them can EXECUTE at most one platform's worth. So `--version` cannot answer "is this
// artifact current?" for the others: an ARM64 macOS dylib on a Linux box is a file, not a program.
// A literal in .rodata can be read from any of them without running anything, which is why the
// stamp is a scannable string and not a function.
//
// The stamp names more than the commit, because a number taken from a binary is only evidence
// for the configuration that binary was built with: the diagnostic counters (HG_ENGINE_STATS)
// compile the arena's per-worker fast path out and put every allocation through shared
// per-site counters; the per-phase timers (HG_PHASE_TIMING) add a read of the cycle counter to
// every task; a sanitizer changes everything. Every measuring script reads this record from the
// binary it is about to run and refuses one that is not the release configuration, and the
// release sign-off reads it from every shipped artifact.
//
// FORMAT. This is a contract with tools/dev/artifact_stamp_check.py and tools/dev/paper_tables.py,
// which parse it:
//
//     HGBUILDSTAMP/2;commit=<40-hex-or-unknown>;variant=<name>;stats=<0|1>;phase_timing=<0|1>;
//     ndebug=<0|1>;asan=<0|1>;tsan=<0|1>;ubsan=<0|1>;type=<CMAKE_BUILD_TYPE>;
//     compiler=<version string>;flags=<CMAKE_CXX_FLAGS and the config's flags>;:HGBUILDSTAMP
//
// on one line, ';' separating fields (no field contains one), the trailing sentinel bounding the
// match. Every artifact that carries the stamp defines it with external linkage and references it
// from its entry point, because an unreferenced object with internal linkage is exactly what a
// linker is entitled to drop, and a stamp that can be dropped proves nothing.
//
// The commit, the build type and the flags come from CMake through the generated header
// hgcommon/build_commit.hpp (configured into the build directory, so a new commit recompiles
// the stamp-defining translation units and nothing else); the variant from the target
// (HG_BUILD_VARIANT); the rest from the compile-time definitions the options themselves set, so
// the record cannot disagree with the code it sits in.

#if __has_include("hgcommon/build_commit.hpp")
#include "hgcommon/build_commit.hpp"
#endif

#ifndef HG_BUILD_COMMIT
#define HG_BUILD_COMMIT "unknown"
#endif
#ifndef HG_BUILD_VARIANT
#define HG_BUILD_VARIANT "unknown"
#endif
#ifndef HG_BUILD_TYPE
#define HG_BUILD_TYPE "unknown"
#endif
#ifndef HG_BUILD_FLAGS
#define HG_BUILD_FLAGS "unknown"
#endif

#if defined(HG_ENGINE_STATS) && HG_ENGINE_STATS
#define HG_BUILD_STAMP_STATS "1"
#else
#define HG_BUILD_STAMP_STATS "0"
#endif
#if defined(HG_PHASE_TIMING)
#define HG_BUILD_STAMP_PHASE_TIMING "1"
#else
#define HG_BUILD_STAMP_PHASE_TIMING "0"
#endif
#if defined(NDEBUG)
#define HG_BUILD_STAMP_NDEBUG "1"
#else
#define HG_BUILD_STAMP_NDEBUG "0"
#endif
#if defined(HG_BUILD_ASAN) || defined(__SANITIZE_ADDRESS__)
#define HG_BUILD_STAMP_ASAN "1"
#else
#define HG_BUILD_STAMP_ASAN "0"
#endif
#if defined(HG_BUILD_TSAN) || defined(__SANITIZE_THREAD__)
#define HG_BUILD_STAMP_TSAN "1"
#else
#define HG_BUILD_STAMP_TSAN "0"
#endif
#if defined(HG_BUILD_UBSAN)
#define HG_BUILD_STAMP_UBSAN "1"
#else
#define HG_BUILD_STAMP_UBSAN "0"
#endif
#if defined(__clang__)
#define HG_BUILD_STAMP_COMPILER "clang " __clang_version__
#elif defined(__GNUC__) && defined(__VERSION__)
#define HG_BUILD_STAMP_COMPILER "gcc " __VERSION__
#elif defined(_MSC_FULL_VER)
#define HG_BUILD_STAMP_STR2(x) #x
#define HG_BUILD_STAMP_STR(x) HG_BUILD_STAMP_STR2(x)
#define HG_BUILD_STAMP_COMPILER "MSVC " HG_BUILD_STAMP_STR(_MSC_FULL_VER)
#else
#define HG_BUILD_STAMP_COMPILER "unknown"
#endif

#define HG_BUILD_STAMP_LITERAL                                                              \
    "HGBUILDSTAMP/2;commit=" HG_BUILD_COMMIT ";variant=" HG_BUILD_VARIANT                   \
    ";stats=" HG_BUILD_STAMP_STATS ";phase_timing=" HG_BUILD_STAMP_PHASE_TIMING             \
    ";ndebug=" HG_BUILD_STAMP_NDEBUG ";asan=" HG_BUILD_STAMP_ASAN ";tsan=" HG_BUILD_STAMP_TSAN \
    ";ubsan=" HG_BUILD_STAMP_UBSAN ";type=" HG_BUILD_TYPE ";compiler=" HG_BUILD_STAMP_COMPILER \
    ";flags=" HG_BUILD_FLAGS ";:HGBUILDSTAMP"
