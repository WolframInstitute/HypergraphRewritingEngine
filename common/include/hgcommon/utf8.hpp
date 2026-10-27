#pragma once
#include "hgcommon/namespace.hpp"

// THE ENGINE'S UTF-8 RULE, strict (RFC 3629, table 3-7): a lead byte C2-DF starts 2 bytes, E0-EF 3
// and F0-F4 4; the second byte after E0 is A0-BF, after ED 80-9F, after F0 90-BF, after F4 80-8F;
// every other continuation byte is 80-BF. Overlong forms, surrogates, code points above U+10FFFF
// and stray continuation bytes are not characters.
//   utf8_char_length       the length of the character at a position, or 0
//   utf8_complete_prefix   the longest prefix that does not end inside a character
//   utf8_escaped           the text with each byte that is not part of a character written \xHH
// Callers: diag_printf's 1023-byte cut (job_system/src/diag.cpp); the worker's log slots, output
// capture and frames (paclet_source/frame_writer.cpp); the strings of a reply's warnings and device
// name (paclet_source/paclet_support.cpp, hg_gpu_backend.cpp).

#include <cstddef>
#include <string>

namespace HG_NAMESPACE {
namespace common {

// The length of the character starting at s[0] when s holds n bytes, or 0 when s[0] starts none.
// `matched`, when given, receives how many leading bytes conform: equal to n with a result of 0
// means the bytes are the start of a character that the n bytes cut short.
inline std::size_t utf8_char_length(const char* s, std::size_t n, std::size_t* matched = nullptr) {
    const auto u = [&](std::size_t i) { return static_cast<unsigned char>(s[i]); };
    if (matched) *matched = 0;
    if (n == 0) return 0;
    const unsigned char c = u(0);
    std::size_t len = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c < 0x80) len = 1;
    else if (c >= 0xC2 && c <= 0xDF) len = 2;
    else if (c >= 0xE0 && c <= 0xEF) {
        len = 3;
        if (c == 0xE0) lo = 0xA0;
        if (c == 0xED) hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        len = 4;
        if (c == 0xF0) lo = 0x90;
        if (c == 0xF4) hi = 0x8F;
    }
    if (len == 0) return 0;
    for (std::size_t k = 1; k < len; ++k) {
        if (k >= n) {
            if (matched) *matched = k;
            return 0;
        }
        const unsigned char b = u(k);
        if (k == 1 ? (b < lo || b > hi) : (b & 0xC0) != 0x80) {
            if (matched) *matched = k;
            return 0;
        }
    }
    if (matched) *matched = len;
    return len;
}

// The largest m <= n for which s[0, m) does not end inside a character: a character among the last
// three bytes that the n bytes cut short is left out. Bytes that start no character count as
// whole.
inline std::size_t utf8_complete_prefix(const char* s, std::size_t n) {
    for (std::size_t back = 1; back <= 3 && back <= n; ++back) {
        const std::size_t p = n - back;
        std::size_t matched = 0;
        if (utf8_char_length(s + p, back, &matched) == 0 && matched == back) return p;
    }
    return n;
}

// Calls emit(text, length) with s's characters and, for each byte that is not part of a
// character, the four bytes \xHH. Returns the total length emitted.
template <class Emit>
std::size_t utf8_escaped(const char* s, std::size_t n, Emit&& emit) {
    static const char hex[] = "0123456789ABCDEF";
    std::size_t total = 0, run = 0, i = 0;
    while (i < n) {
        const std::size_t len = utf8_char_length(s + i, n - i);
        if (len) {
            run += len;
            i += len;
            continue;
        }
        if (run) emit(s + i - run, run);
        total += run;
        run = 0;
        const unsigned char c = static_cast<unsigned char>(s[i]);
        const char esc[4] = {'\\', 'x', hex[c >> 4], hex[c & 0xF]};
        emit(esc, 4);
        total += 4;
        ++i;
    }
    if (run) emit(s + n - run, run);
    return total + run;
}

// The length utf8_escaped emits for s.
inline std::size_t utf8_escaped_length(const char* s, std::size_t n) {
    return utf8_escaped(s, n, [](const char*, std::size_t) {});
}

inline std::string utf8_escaped(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    utf8_escaped(s.data(), s.size(), [&](const char* p, std::size_t k) { out.append(p, k); });
    return out;
}

}  // namespace common
}  // namespace HG_NAMESPACE
