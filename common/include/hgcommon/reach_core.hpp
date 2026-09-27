#pragma once
#include "hgcommon/namespace.hpp"
// THE BACKWARD REACHABILITY SEARCH of the online transitive reduction, one body for the host and
// the device, for full capture and for the quotient replay.
//
// Does p reach `consumer` through the kept (reduced) predecessor adjacency? A pair (p, c) is
// redundant exactly when it does through c's other kept producers, and the reduced adjacency
// preserves reachability, so the answer is exact over it. With `topological`, ids increase along
// every edge: a node numbered at or below p can neither be p nor have p as an ancestor, so the
// search visits only ids above p and stops at once when p >= consumer.
//
// A Ctx supplies the storage and nothing else:
//   void reset();                           empty the stack and the visited set
//   bool visit(uint32_t x);                 insert-if-absent; true when x was not yet visited
//   void push(uint32_t x);
//   bool pop(uint32_t& x);                  false when the stack is empty
//   template <class F> void for_each_pred(uint32_t x, F&& f);   f(q) per kept predecessor of x
// A Ctx with bounded storage records its own overflow. A visit or push it cannot hold must answer
// as a node not explored, which can only keep a pair the reduction would drop.

#include <cstdint>

#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

template <class Ctx>
HG_HD bool reach_backward(Ctx& c, uint32_t p, uint32_t consumer, bool topological) {
    if (p == consumer) return true;
    if (topological && p >= consumer) return false;
    c.reset();
    c.visit(consumer);
    c.push(consumer);
    uint32_t x = 0;
    while (c.pop(x)) {
        bool found = false;
        c.for_each_pred(x, [&](uint32_t q) {
            if (found) return;
            if (q == p) { found = true; return; }
            if ((!topological || q > p) && c.visit(q)) c.push(q);
        });
        if (found) return true;
    }
    return false;
}

// WHICH OF ONE CONSUMER'S PRODUCERS ARE REDUNDANT, decided together. A pair (p, c) is redundant
// exactly when p reaches c through another producer of c, which is when p is a proper ancestor
// of another producer. So one backward search from every producer's predecessors marks them all:
// bit i of the result is set when producers[i] is reached. `producers` are distinct, at most 32.
// With `topological` the search visits only ids at or above the smallest producer, and it stops
// once every producer but the largest is found (no producer lies above the largest).
template <class Ctx>
HG_HD uint32_t redundant_producers(Ctx& c, const uint32_t* producers, uint32_t n,
                                   bool topological) {
    if (n < 2) return 0;
    uint32_t floor = producers[0];
    for (uint32_t i = 1; i < n; ++i)
        if (producers[i] < floor) floor = producers[i];
    const uint32_t findable = topological ? n - 1 : n;
    uint32_t mask = 0, found = 0;
    c.reset();
    auto admit = [&](uint32_t q) {
        if (topological && q < floor) return;
        if (!c.visit(q)) return;
        for (uint32_t i = 0; i < n; ++i)
            if (producers[i] == q) { mask |= (1u << i); ++found; break; }
        c.push(q);
    };
    for (uint32_t i = 0; i < n && found < findable; ++i) c.for_each_pred(producers[i], admit);
    uint32_t x = 0;
    while (found < findable && c.pop(x)) c.for_each_pred(x, admit);
    return mask;
}

// Storage for the searches above in caller-provided fixed arrays, for a caller with no
// allocator. The visited table is open-addressed over a power-of-two capacity and holds id + 1,
// so zero is empty. When either array fills, `overflow` is set and the node is treated as not
// explored, which can only keep a pair the reduction would drop; the caller retries with larger
// arrays or reports the overflow.
template <class ForEachPred>
struct BoundedReachCtx {
    ForEachPred preds;
    uint32_t* stack;
    uint32_t  stack_cap;
    uint32_t* table;
    uint32_t  table_cap;
    uint32_t  sp = 0;
    bool      overflow = false;

    HG_HD BoundedReachCtx(ForEachPred p, uint32_t* st, uint32_t st_cap, uint32_t* tab,
                          uint32_t tab_cap)
        : preds(p), stack(st), stack_cap(st_cap), table(tab), table_cap(tab_cap) {}
    HG_HD void reset() {
        sp = 0;
        overflow = false;
        for (uint32_t i = 0; i < table_cap; ++i) table[i] = 0;
    }
    HG_HD bool visit(uint32_t x) {
        uint32_t slot = (x * 2654435761u) & (table_cap - 1u);
        for (uint32_t probe = 0; probe < table_cap; ++probe) {
            const uint32_t held = table[slot];
            if (held == x + 1u) return false;
            if (held == 0u) { table[slot] = x + 1u; return true; }
            slot = (slot + 1u) & (table_cap - 1u);
        }
        overflow = true;
        return false;
    }
    HG_HD void push(uint32_t x) {
        if (sp < stack_cap) stack[sp++] = x;
        else overflow = true;
    }
    HG_HD bool pop(uint32_t& x) {
        if (sp == 0) return false;
        x = stack[--sp];
        return true;
    }
    template <class F> HG_HD void for_each_pred(uint32_t x, F&& f) { preds(x, f); }
};

}  // namespace common
}  // namespace HG_NAMESPACE
