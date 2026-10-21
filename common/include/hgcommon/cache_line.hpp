#pragma once
#include "hgcommon/namespace.hpp"

// H16 (docs/TUNING.md): the span two threads' hot data are kept apart by. Every alignas that
// separates one thread's writes from another's reads this constant. 128 bytes on Apple aarch64,
// whose L1 and L2 lines are 128 bytes; 64 bytes on x86-64 and other aarch64 parts.
// HG_DESTRUCTIVE_LINE overrides it at compile time.

#include <cstddef>

#ifndef HG_DESTRUCTIVE_LINE
#if defined(__APPLE__) && defined(__aarch64__)
#define HG_DESTRUCTIVE_LINE 128
#else
#define HG_DESTRUCTIVE_LINE 64
#endif
#endif

namespace HG_NAMESPACE {
namespace common {

inline constexpr std::size_t kDestructiveLine = HG_DESTRUCTIVE_LINE;
static_assert((kDestructiveLine & (kDestructiveLine - 1)) == 0, "a cache line is a power of two");

}  // namespace common
}  // namespace HG_NAMESPACE
