// The engine's diagnostic channel (hgcommon/diag.hpp). DEBUG_LOG (hypergraph/debug_log.hpp) writes
// through it, so a host's sink receives the engine's trace lines.
#ifndef ENABLE_DEBUG_OUTPUT
#define ENABLE_DEBUG_OUTPUT
#endif
#include "hypergraph/debug_log.hpp"

#include <gtest/gtest.h>

#include <string>

#include "hgcommon/diag.hpp"

namespace {
std::string g_received;
void collect(const char* text, std::size_t length) { g_received.append(text, length); }
}  // namespace

TEST(DiagChannel, DebugLogReachesTheSink) {
    g_received.clear();
    hgcommon::set_diag_sink(collect);
    DEBUG_LOG("rewrite %d of %s", 3, "x");
    hgcommon::set_diag_sink(nullptr);
    EXPECT_EQ(g_received, "[DEBUG] rewrite 3 of x\n");
}
