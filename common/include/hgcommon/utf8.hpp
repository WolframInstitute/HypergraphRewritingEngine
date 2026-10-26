#pragma once
#include "hgcommon/namespace.hpp"

// UTF-8 TEXT CUT TO A LENGTH. utf8_complete_prefix(s, n) is the largest m <= n for which s[0, m)
// does not end inside a multi-byte character: a lead byte among the last four whose character
// runs past n is left out, with its continuation bytes. Bytes that are not UTF-8 count as complete
// characters. Callers: diag_printf's 1023-byte message (job_system/src/diag.cpp), the worker's
// 1000-byte log slots and its output capture (paclet_source/frame_writer.cpp).

#include <cstddef>

namespace HG_NAMESPACE {
namespace common {

// The byte count of the character that starts with `lead`; 1 for a byte that starts none.
inline std::size_t utf8_sequence_length(unsigned char lead) {
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;
}

inline std::size_t utf8_complete_prefix(const char* s, std::size_t n) {
    std::size_t back = 0;
    while (back < 3 && back < n && (static_cast<unsigned char>(s[n - 1 - back]) & 0xC0) == 0x80)
        ++back;
    if (back == n) return n;
    const std::size_t lead = n - 1 - back;
    return lead + utf8_sequence_length(static_cast<unsigned char>(s[lead])) > n ? lead : n;
}

}  // namespace common
}  // namespace HG_NAMESPACE
