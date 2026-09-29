#pragma once
#include "hgcommon/namespace.hpp"
//
// A STORED IR CANONICAL FORM, the identity that exact state deduplication compares on a hash hit.
//
// The form is ir_canonical_hash's out_canonical_form: for each edge in canonical order, its arity
// followed by its canonically labelled vertices, ir_canonical_form_words(n_edges, total_occ)
// words. Two states have equal forms if and only if they are isomorphic; the 64-bit canonical
// hash is a digest of the form and can collide.
//
// A record is a 12-byte header (the representative state id, word count, element width)
// followed by the words, each stored in `width` bytes: 1 when every word is below 256, 2 when
// below 65536, else 4. On the Wolfram rule at depth 6 a state has 14 edges of arity 2, so its
// record is 12 + 42 bytes.

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

struct CanonicalFormRecord {
    uint32_t state;   // the class's representative state
    uint32_t words;
    uint32_t width;   // bytes per stored word: 1, 2 or 4
};

HG_HD inline uint32_t canonical_form_width(const uint32_t* form, uint32_t words) {
    uint32_t mx = 0;
    for (uint32_t i = 0; i < words; ++i) mx = form[i] > mx ? form[i] : mx;
    return mx < 256u ? 1u : (mx < 65536u ? 2u : 4u);
}

HG_HD inline uint64_t canonical_form_record_bytes(uint32_t words, uint32_t width) {
    return sizeof(CanonicalFormRecord) + uint64_t(words) * width;
}

// Writes the record for `form` into `out`, which holds canonical_form_record_bytes(words, width)
// bytes and is 4-byte aligned.
HG_HD inline void canonical_form_encode(uint32_t state, const uint32_t* form, uint32_t words,
                                        uint32_t width, CanonicalFormRecord* out) {
    out->state = state;
    out->words = words;
    out->width = width;
    unsigned char* data = reinterpret_cast<unsigned char*>(out + 1);
    if (width == 1u) {
        for (uint32_t i = 0; i < words; ++i) data[i] = static_cast<unsigned char>(form[i]);
    } else if (width == 2u) {
        uint16_t* d = reinterpret_cast<uint16_t*>(data);
        for (uint32_t i = 0; i < words; ++i) d[i] = static_cast<uint16_t>(form[i]);
    } else {
        uint32_t* d = reinterpret_cast<uint32_t*>(data);
        for (uint32_t i = 0; i < words; ++i) d[i] = form[i];
    }
}

// True when `rec` stores exactly `form`. Compared at the record's width: a form word too large
// for that width differs from the stored word, so the new form's width is not needed. The loops
// OR the differences with no early exit because on a hash hit the forms are almost always
// equal and every word is read regardless.
HG_HD inline bool canonical_form_equals(const CanonicalFormRecord* rec, const uint32_t* form,
                                        uint32_t words) {
    if (rec->words != words) return false;
    const unsigned char* data = reinterpret_cast<const unsigned char*>(rec + 1);
    uint32_t diff = 0;
    if (rec->width == 1u) {
        for (uint32_t i = 0; i < words; ++i) diff |= uint32_t(data[i]) ^ form[i];
    } else if (rec->width == 2u) {
        const uint16_t* d = reinterpret_cast<const uint16_t*>(data);
        for (uint32_t i = 0; i < words; ++i) diff |= uint32_t(d[i]) ^ form[i];
    } else {
        const uint32_t* d = reinterpret_cast<const uint32_t*>(data);
        for (uint32_t i = 0; i < words; ++i) diff |= d[i] ^ form[i];
    }
    return diff == 0;
}

}  // namespace common
}  // namespace HG_NAMESPACE
