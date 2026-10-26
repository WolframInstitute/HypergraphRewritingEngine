// The engine's diagnostic channel (hgcommon/diag.hpp). DEBUG_LOG (hypergraph/debug_log.hpp) writes
// through it, so a host's sink receives the engine's trace lines; a message cut to the 1023-byte
// buffer ends at a UTF-8 character boundary.
#ifndef ENABLE_DEBUG_OUTPUT
#define ENABLE_DEBUG_OUTPUT
#endif
#include "hypergraph/debug_log.hpp"

#include <gtest/gtest.h>

#include <string>

#include "hgcommon/diag.hpp"
#include "hgcommon/utf8.hpp"

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

// 1021 ASCII bytes and then a 3-byte character: the buffer holds 1023 bytes, so the character is
// left out whole.
TEST(DiagChannel, ACutMessageEndsAtACharacterBoundary) {
    const std::string text = std::string(1021, 'a') + "\xE2\x82\xAC" + "tail";
    g_received.clear();
    hgcommon::set_diag_sink(collect);
    hgcommon::diag_printf("%s", text.c_str());
    hgcommon::set_diag_sink(nullptr);
    EXPECT_EQ(g_received, std::string(1021, 'a'));
    EXPECT_EQ(hgcommon::utf8_complete_prefix(g_received.data(), g_received.size()), g_received.size());
}
