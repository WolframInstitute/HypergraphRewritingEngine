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
inline void set_debug_callback(DebugCallback cb) {
    g_debug_callback.store(cb, std::memory_order_release);
}

// Clear the debug callback
inline void clear_debug_callback() {
    g_debug_callback.store(nullptr, std::memory_order_release);
}

} // namespace debug
}  // namespace engine
}  // namespace HG_NAMESPACE
// Debug logging macro - routes to callback if set, otherwise printf
//
// The formatter and the three headers it needs are compiled only when debug output is on. With it
// off DEBUG_LOG is a no-op and nothing names debug_output, so <sstream> and <thread> would drag
// libstdc++'s iostream and threading machinery into every translation unit that includes this
// header for a function none of them call. Keeping them inside the guard also lets a translation
// unit be compiled by a tool that models pthreads itself and cannot satisfy gthr-default.h --
// see verification/genmc/README.md.
#ifdef ENABLE_DEBUG_OUTPUT

#include <cstdarg>
#include <sstream>
#include <thread>

namespace HG_NAMESPACE {
namespace engine {
namespace debug {

// Internal: format and output debug message
inline void debug_output(const char* fmt, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    // Add thread ID prefix
    std::ostringstream oss;
    oss << std::this_thread::get_id();

    char full_message[1100];
    snprintf(full_message, sizeof(full_message), "[DEBUG][T%s] %s", oss.str().c_str(), buffer);

    DebugCallback cb = g_debug_callback.load(std::memory_order_acquire);
    if (cb) {
        cb(full_message);
    } else {
        printf("%s\n", full_message);
        fflush(stdout);
    }
}

} // namespace debug
}  // namespace engine
}  // namespace HG_NAMESPACE
    #define DEBUG_LOG(fmt, ...) ::hypergraph::debug::debug_output(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_LOG(fmt, ...) ((void)0)
#endif

