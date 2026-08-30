#pragma once
#include "hgcommon/namespace.hpp"
#include "hgcommon/portable_intrinsics.hpp"
#include <cstdint>
//
// THE CLAIM RULE OF A TAGGED FREE-LIST (Treiber stack, 16-bit tag over a 48-bit node ref).
//
// A popper reads the head, dereferences the CANDIDATE's link, and only then attempts the CAS
// that makes the candidate its own. The link read is therefore SPECULATIVE: a rival may win
// the node and rewrite its link -- or hand it to an owner who does -- before the loser's CAS
// fails and discards the value. The tag is what makes the stale value harmless (the CAS
// cannot succeed against a moved head); the link accessors' atomicity is what makes the read
// DEFINED. Both halves are required: with a plain link access the algorithm still "works" on
// x86 and is a data race in the memory model -- ThreadSanitizer reported exactly that mix
// against the arena's block pool (CI tsan leg, 30/08), which is why the rule lives here where
// verification/genmc/block_pool_exactly_once.cpp can compile the decision without the mmap
// and cudaMalloc machinery around its production caller.
//
// Node refs are nonzero 48-bit values; 0 is the empty list. Ops supplies the storage half:
//   uint64_t head_load()                              ACQUIRE
//   bool     head_cas(uint64_t& expected, uint64_t)   ACQ_REL, may fail spuriously
//   uint64_t link_load(uint64_t node)                 RELAXED -- the speculative read
//   void     link_store(uint64_t node, uint64_t v)    RELAXED
//
namespace HG_NAMESPACE {
namespace common {

inline constexpr uint64_t POOL_PTR_MASK = (uint64_t(1) << 48) - 1;

template <class Ops>
HG_HD inline void pool_core_push(Ops& ops, uint64_t node) {
    uint64_t old = ops.head_load();
    uint64_t next;
    do {
        ops.link_store(node, old & POOL_PTR_MASK);
        next = (node & POOL_PTR_MASK) | (((old >> 48) + 1) << 48);
    } while (!ops.head_cas(old, next));
}

// The claimed node, or 0 when the list was observed empty.
template <class Ops>
HG_HD inline uint64_t pool_core_pop(Ops& ops) {
    uint64_t old = ops.head_load();
    while (uint64_t node = old & POOL_PTR_MASK) {
        const uint64_t next =
            (ops.link_load(node) & POOL_PTR_MASK) | (((old >> 48) + 1) << 48);
        if (ops.head_cas(old, next)) return node;
    }
    return 0;
}

}  // namespace common
}  // namespace HG_NAMESPACE
