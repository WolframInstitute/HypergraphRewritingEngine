#pragma once
#include "hgcommon/namespace.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
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
template <class Enumerate, class Emit>
void tr_reduce(Enumerate&& enumerate, Emit&& emit, bool ids_topological = false) {
    std::vector<std::pair<uint32_t, uint32_t>> all;
    enumerate([&](uint32_t p, uint32_t c) { all.emplace_back(p, c); });
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());

    std::unordered_map<uint32_t, std::vector<uint32_t>> succ;
    for (const auto& e : all) succ[e.first].push_back(e.second);

    std::unordered_set<uint32_t> reached;
    std::vector<uint32_t> stack;
    for (size_t i = 0; i < all.size();) {
        const uint32_t p = all[i].first;
        size_t j = i;
        while (j < all.size() && all[j].first == p) ++j;      // [i, j) are p's targets, sorted

        const uint32_t cmax = all[j - 1].second;
        auto admit = [&](uint32_t x) {
            if (ids_topological && x > cmax) return;          // beyond every target of p
            if (reached.insert(x).second) stack.push_back(x);
        };

        // Seed at distance two: the direct edges themselves are what is being judged.
        reached.clear();
        stack.clear();
        auto sp = succ.find(p);
        if (sp != succ.end()) {
            for (uint32_t w : sp->second) {
                auto sw = succ.find(w);
                if (sw != succ.end()) for (uint32_t x : sw->second) admit(x);
            }
        }
        while (!stack.empty()) {
            const uint32_t x = stack.back();
            stack.pop_back();
            auto sx = succ.find(x);
            if (sx == succ.end()) continue;
            for (uint32_t y : sx->second) admit(y);
        }

        for (size_t k = i; k < j; ++k)
            if (!reached.count(all[k].second)) emit(p, all[k].second);
        i = j;
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
