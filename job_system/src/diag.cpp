#include "hgcommon/diag.hpp"
#include "hgcommon/utf8.hpp"

#include <atomic>
#include <cstdarg>
#include <cstdio>

namespace HG_NAMESPACE {
namespace common {

namespace {
std::atomic<DiagSink> g_sink{nullptr};
}  // namespace

void set_diag_sink(DiagSink sink) { g_sink.store(sink, std::memory_order_release); }

void diag_printf(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    const int n = std::vsnprintf(buffer, sizeof buffer, format, args);
    va_end(args);
    if (n <= 0) return;
    // A message cut to the buffer ends at a character boundary.
    const std::size_t length = static_cast<std::size_t>(n) < sizeof buffer
                                   ? static_cast<std::size_t>(n)
                                   : utf8_complete_prefix(buffer, sizeof buffer - 1);
    if (DiagSink sink = g_sink.load(std::memory_order_acquire)) {
        sink(buffer, length);
    } else {
        std::fwrite(buffer, 1, length, stderr);
        std::fflush(stderr);
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
