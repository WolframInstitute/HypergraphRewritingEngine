#ifndef HYPERGRAPH_DEBUG_LOG_HPP
#define HYPERGRAPH_DEBUG_LOG_HPP

#include <cstdio>
#include <thread>
#include <sstream>
#include <cstdarg>
#include <atomic>

namespace hypergraph {
namespace debug {

// Callback function type for debug output routing
// The callback receives a formatted string (no newline at end)
using DebugCallback = void (*)(const char* message);

// Global debug callback - set by FFI layer to route to Mathematica
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
} // namespace hypergraph

// Debug logging macro - routes to callback if set, otherwise printf
#ifdef ENABLE_DEBUG_OUTPUT
    #define DEBUG_LOG(fmt, ...) ::hypergraph::debug::debug_output(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_LOG(fmt, ...) ((void)0)
#endif

#endif // HYPERGRAPH_DEBUG_LOG_HPP
