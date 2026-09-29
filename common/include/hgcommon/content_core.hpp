#pragma once
#include "hgcommon/namespace.hpp"
// Shared CPU/GPU content-ordered state identity -- what Automatic deduplicates states by.
//
// The host walks a SparseBitset and the device walks an edge slice with a liveness filter, so
// the ITERATION differs and cannot be shared. Everything else must be: the basis, the mixing,
// the order the pieces enter, and the separator between edges. The constants live here and the
// callers only drive them.
//
// The edge COUNT is hashed first so that a state and a strict sub-state of it cannot collide by
// the sub-state's edges being a prefix of the other's. The separator closes each edge so that
// {{1,2},{3}} and {{1},{2,3}} differ despite the same vertex sequence.
//
// Deliberately NOT isomorphism-invariant: Automatic identifies states by CONTENT, so a
// relabelling is a different state. That is the whole difference from the Full path.
#include <cstdint>
#include "hgcommon/core.hpp"  // HG_HD, FNV_OFFSET, fnv_hash, mix64

namespace HG_NAMESPACE {
namespace common {

struct ContentHasher {
    uint64_t h;

    HG_HD explicit ContentHasher(uint32_t edge_count)
        : h(fnv_hash(FNV_OFFSET, mix64(static_cast<uint64_t>(edge_count)))) {}

    HG_HD void edge_begin(uint32_t arity) {
        h = fnv_hash(h, mix64(static_cast<uint64_t>(arity)));
    }
    HG_HD void vertex(uint64_t v) { h = fnv_hash(h, mix64(v)); }
    HG_HD void edge_end() { h = fnv_hash(h, 0xDEADBEEFCAFEBABEull); }

    HG_HD uint64_t value() const { return h; }
};

// The words ContentHasher digests, in its order: the edge count, then per edge the arity and
// the vertices. Automatic identity is the equality of these words. Driven like ContentHasher;
// each word goes to `emit`.
template <class Emit>
struct ContentWords {
    Emit emit;

    HG_HD ContentWords(uint32_t edge_count, Emit e) : emit(e) { emit(edge_count); }

    HG_HD void edge_begin(uint32_t arity) { emit(arity); }
    HG_HD void vertex(uint64_t v) { emit(static_cast<uint32_t>(v)); }
    HG_HD void edge_end() {}
};

}  // namespace common
}  // namespace HG_NAMESPACE
