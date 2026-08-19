#pragma once
#include "hgcommon/namespace.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace HG_NAMESPACE {
namespace common {

// THE TRANSITIVE REDUCTION OF A STORED RELATION, one body for every caller.
//
// A finite DAG has exactly one transitive reduction, so the answer is a function of the relation
// alone. That uniqueness is the entire reason this is computed from the stored pair set rather
// than decided as each pair lands: a tag evaluated against the pairs seen SO FAR keeps (p,c) when
// it arrives before the longer path that bypasses it and drops it when it arrives after, and an
// append-only adjacency never retracts the difference. Two schedules then serve two relations.
//
// `enumerate` calls its argument with each (producer, consumer) of the stored relation, in any
// order and with duplicates permitted. `emit` receives each surviving pair, sorted.
//
// ONE SEARCH PER SOURCE, not per pair. (p,c) is redundant exactly when c is reachable from p in
// two or more steps, and that set is shared by every edge leaving p, so it is computed once and
// every target of p is then decided by a lookup.
//
// `ids_topological` says ids increase along every edge, which lets the search stop at the
// largest target of p: a path to any target cannot pass through a node numbered above it. It is
// false where ids are assigned first-writer-wins, and there the search runs unpruned.
//
// IDS ARE INDICES HERE, so they must be dense: the adjacency and the membership marks are arrays
// sized by the largest id seen, not hash tables. Both callers pass event ids, which come from a
// counter. A caller passing sparse ids -- hashes, say -- would allocate proportional to the
// largest value rather than to the number of nodes.
template <class Enumerate, class Emit>
void tr_reduce(Enumerate&& enumerate, Emit&& emit, bool ids_topological = false) {
    std::vector<std::pair<uint32_t, uint32_t>> all;
    enumerate([&](uint32_t p, uint32_t c) { all.emplace_back(p, c); });
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());

    if (all.empty()) return;

    // `all` is sorted by (source, target), so every source's targets are already contiguous and
    // the adjacency is an index into it rather than a container: succ_begin[p] is where p's
    // targets start and succ_begin[p+1] is where they end. The walk below then reaches a node's
    // successors with two loads and no lookup.
    uint32_t max_id = 0;
    for (const auto& e : all) {
        max_id = std::max(max_id, e.first);
        max_id = std::max(max_id, e.second);
    }
    std::vector<uint32_t> succ_begin(static_cast<size_t>(max_id) + 2, 0);
    for (const auto& e : all) ++succ_begin[e.first + 1];
    for (size_t k = 1; k < succ_begin.size(); ++k) succ_begin[k] += succ_begin[k - 1];

    // Membership by generation stamp rather than by a set that is emptied per source. Clearing a
    // hash set rewrites its whole bucket array, so the cost is (sources x buckets) however few
    // nodes each search actually reaches, and the searches here are small and numerous.
    // Incrementing a counter costs one add, and a stale entry is stale precisely because it
    // carries an earlier stamp.
    std::vector<uint32_t> seen(static_cast<size_t>(max_id) + 1, 0);
    uint32_t stamp = 0;
    std::vector<uint32_t> stack;

    for (size_t i = 0; i < all.size();) {
        const uint32_t p = all[i].first;
        const size_t j = succ_begin[p + 1];                   // [i, j) are p's targets, sorted
        const uint32_t cmax = all[j - 1].second;

        ++stamp;                                              // empties `seen` in one add
        auto admit = [&](uint32_t x) {
            if (ids_topological && x > cmax) return;          // beyond every target of p
            if (seen[x] != stamp) { seen[x] = stamp; stack.push_back(x); }
        };

        // Seed at distance two: the direct edges themselves are what is being judged.
        stack.clear();
        for (size_t a = succ_begin[p]; a < succ_begin[p + 1]; ++a) {
            const uint32_t w = all[a].second;
            for (size_t b = succ_begin[w]; b < succ_begin[w + 1]; ++b) admit(all[b].second);
        }
        while (!stack.empty()) {
            const uint32_t x = stack.back();
            stack.pop_back();
            for (size_t b = succ_begin[x]; b < succ_begin[x + 1]; ++b) admit(all[b].second);
        }

        for (size_t k = i; k < j; ++k)
            if (seen[all[k].second] != stamp) emit(p, all[k].second);
        i = j;
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
