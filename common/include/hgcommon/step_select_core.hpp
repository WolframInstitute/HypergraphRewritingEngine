#pragma once
#include "hgcommon/namespace.hpp"
// THE "MatchesPerStep" SELECTION OVER A STEP'S BLOCKS, one body for host and device. A block is
// one state's transitions in key order (an order-statistic set, hgcommon/key_order_core.hpp)
// with the state's group. The step's order is (group, key): the blocks in group order, and the
// blocks of one group merged by key. The kept transitions are those at the positions
// hgcommon::step_positions drew, each with every transition of equal (group, key).
//
// `blocks` is in group order and provides
//   uint32_t count();                                 the blocks
//   uint64_t group(uint32_t i); uint64_t size(uint32_t i);
//   V select(uint32_t i, uint64_t p, uint64_t* lo, uint64_t* hi);
//                                                     block i's entry at p, and the positions
//                                                     [lo, hi) of its key
//   Run merged(uint32_t g, uint32_t h);               blocks g .. h - 1 (one group) merged by key:
//                                                     Run::key(q), block(q), value(q)
// `pos[0..n)` is ascending, below the blocks' total. keep(block, value) receives each kept
// transition once, in step order.

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

template <class Blocks, class Keep>
HG_HD inline void step_select_blocks(Blocks& blocks, const uint64_t* pos, uint32_t n,
                                     Keep&& keep) {
    uint32_t next = 0;   // the first position not yet taken
    uint64_t offset = 0;
    const uint32_t nb = blocks.count();
    for (uint32_t g = 0; g < nb;) {
        uint32_t h = g + 1u;
        while (h < nb && blocks.group(h) == blocks.group(g)) ++h;
        uint64_t size = 0;
        for (uint32_t i = g; i < h; ++i) size += blocks.size(i);
        if (h == g + 1u) {
            uint64_t taken_hi = 0;
            for (; next < n && pos[next] < offset + size; ++next) {
                const uint64_t p = pos[next] - offset;
                if (p < taken_hi) continue;   // in the tie run already kept
                uint64_t lo = 0, hi = 0;
                blocks.select(g, p, &lo, &hi);
                for (uint64_t q = lo; q < hi; ++q) {
                    uint64_t l2 = 0, h2 = 0;
                    keep(g, blocks.select(g, q, &l2, &h2));
                }
                taken_hi = hi;
            }
        } else if (next < n && pos[next] < offset + size) {
            // Equal groups: the blocks' entries merged by key.
            auto run = blocks.merged(g, h);
            uint64_t taken_hi = 0;
            for (; next < n && pos[next] < offset + size; ++next) {
                const uint64_t p = pos[next] - offset;
                if (p < taken_hi) continue;
                uint64_t lo = p, hi = p + 1u;
                while (lo > 0u && run.key(lo - 1u) == run.key(p)) --lo;
                while (hi < size && run.key(hi) == run.key(p)) ++hi;
                for (uint64_t q = lo; q < hi; ++q) keep(run.block(q), run.value(q));
                taken_hi = hi;
            }
        }
        offset += size;
        g = h;
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
