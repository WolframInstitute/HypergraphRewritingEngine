#include "hypergraph/debug_log.hpp"

// The debug channel's bodies. debug_log.hpp is included by parallel_evolution.hpp,
// rewriter.hpp and concurrent_map.hpp, so it reaches almost every engine translation unit;
// this is the only one that names <cstdarg>, <sstream> or <thread> for it.

#ifdef ENABLE_DEBUG_OUTPUT
#include <cstdarg>
#include <sstream>
#include <thread>
#endif

namespace HG_NAMESPACE {
namespace engine {
namespace debug {

void set_debug_callback(DebugCallback cb) {
    g_debug_callback.store(cb, std::memory_order_release);
}

void clear_debug_callback() {
    g_debug_callback.store(nullptr, std::memory_order_release);
}

#ifdef ENABLE_DEBUG_OUTPUT

void debug_output(const char* fmt, ...) {
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

#endif  // ENABLE_DEBUG_OUTPUT

} // namespace debug
}  // namespace engine
}  // namespace HG_NAMESPACE
