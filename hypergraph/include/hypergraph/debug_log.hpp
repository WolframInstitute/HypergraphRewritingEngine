#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdio>
#include <atomic>

namespace HG_NAMESPACE {
namespace engine {
namespace debug {

// Callback function type for debug output routing
// The callback receives a formatted string (no newline at end)
using DebugCallback = void (*)(const char* message);

// Global debug callback - set by FFI layer to route to the Wolfram Language
// When null, DEBUG_LOG uses printf (for standalone C++ usage)
inline std::atomic<DebugCallback> g_debug_callback{nullptr};

// Set the debug callback (called from FFI layer)
void set_debug_callback(DebugCallback cb);

// Clear the debug callback
void clear_debug_callback();

} // namespace debug
}  // namespace engine
}  // namespace HG_NAMESPACE
// Debug logging macro - routes to callback if set, otherwise printf
//
// The formatter is DECLARED here and defined in hypergraph/src/debug_log.cpp, so <cstdarg>,
// <sstream> and <thread> are named by that one translation unit and by nothing else. With debug
// output off, DEBUG_LOG is a no-op and nothing names debug_output at all. Both matter: libstdc++'s
// iostream and threading machinery would otherwise reach every translation unit that includes this
// header for a function none of them call, and a tool that models pthreads itself cannot satisfy
// gthr-default.h -- see verification/genmc/README.md.
#ifdef ENABLE_DEBUG_OUTPUT

namespace HG_NAMESPACE {
namespace engine {
namespace debug {

// Internal: format and output debug message. Body in hypergraph/src/debug_log.cpp, which is
// where <cstdarg>, <sstream> and <thread> now go with it.
void debug_output(const char* fmt, ...);

} // namespace debug
}  // namespace engine
}  // namespace HG_NAMESPACE

    #define DEBUG_LOG(fmt, ...) ::hypergraph::debug::debug_output(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_LOG(fmt, ...) ((void)0)
#endif

