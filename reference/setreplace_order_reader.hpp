#pragma once
// The recorded SetReplace cases of reference/setreplace_order_cases.hpp, decoded, and the
// comparison up to a renaming of vertices. Read by hypergraph/tests/test_event_ordering.cpp (the
// host engine) and gpu/tests/test_gpu_vs_cpu_differential.cpp (the device engine).

#include "hgcommon/sampling_core.hpp"
#include "reference/setreplace_order_cases.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace setreplace_order {

using Edges = std::vector<std::vector<uint32_t>>;

struct Recorded {
    std::string name;
    std::vector<std::pair<Edges, Edges>> rules;
    Edges init;
    uint32_t order = 0;   // hgcommon::event_order_compare
    uint32_t max_events = 0;
    Edges events;   // per event: the rule from 1, the inputs, then the outputs
    Edges expressions;
    std::vector<std::pair<uint32_t, uint32_t>> causal;
};

// The cases of kStream, in the layout setreplace_order_cases.hpp describes.
inline std::vector<Recorded> read_cases() {
    std::vector<Recorded> out;
    const uint32_t* p = kStream;
    auto edges = [&p]() {
        Edges l(*p++);
        for (auto& e : l) {
            e.assign(p + 1, p + 1 + *p);
            p += 1 + e.size();
        }
        return l;
    };
    for (uint32_t c = 0; c < kCases; ++c) {
        Recorded k;
        k.name = kNames[c];
        for (const char* const* o = kOrderings[c]; *o != nullptr; ++o)
            k.order = hgcommon::event_order_append(k.order, hgcommon::event_order_code(*o));
        k.max_events = *p++;
        k.rules.resize(*p++);
        for (auto& r : k.rules) {
            r.first = edges();
            r.second = edges();
        }
        k.init = edges();
        const uint32_t n = *p++;
        for (uint32_t i = 0; i < n; ++i) {
            std::vector<uint32_t> ev{*p++};
            const uint32_t ins = *p++;
            ev.insert(ev.end(), p, p + ins);
            p += ins;
            const uint32_t outs = *p++;
            ev.insert(ev.end(), p, p + outs);
            p += outs;
            k.events.push_back(ev);
        }
        k.expressions = edges();
        const uint32_t pairs = *p++;
        for (uint32_t i = 0; i < pairs; ++i, p += 2) k.causal.emplace_back(p[0], p[1]);
        out.push_back(std::move(k));
    }
    return out;
}

// Equal after a one-to-one renaming of vertices.
inline bool same_up_to_relabel(const Edges& a, const Edges& b) {
    if (a.size() != b.size()) return false;
    std::map<uint32_t, uint32_t> fwd, back;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j) {
            const auto f = fwd.emplace(a[i][j], b[i][j]).first;
            const auto g = back.emplace(b[i][j], a[i][j]).first;
            if (f->second != b[i][j] || g->second != a[i][j]) return false;
        }
    }
    return true;
}

}  // namespace setreplace_order
