#pragma once
#include "hgcommon/namespace.hpp"

// THE ENGINE'S DIAGNOSTIC CHANNEL. Host code that reports something outside a job's reply (the
// placement of the workers, a GPU phase breakdown under HG_GPU_DBG_TIME, an arena failure) calls
// diag_printf instead of writing to stderr. Without a sink the text goes to stderr, as it did; the
// hg_evolve worker installs a sink that puts each message in the calling thread's ring
// (paclet_source/frame_writer.hpp), from which the worker's one writer thread sends it to the
// paclet as a log frame. The sink is called on the reporting thread and must not block it.

#include <cstddef>

namespace HG_NAMESPACE {
namespace common {

using DiagSink = void (*)(const char* text, std::size_t length);

// Installs `sink` for every later message, or restores stderr with nullptr.
void set_diag_sink(DiagSink sink);

// Formats a message (at most 1023 bytes; longer text is cut) and hands it to the sink.
void diag_printf(const char* format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

}  // namespace common
}  // namespace HG_NAMESPACE
