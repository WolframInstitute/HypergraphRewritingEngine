// The bodies behind the paclet's support headers.
//
// These headers are parsed by every paclet translation unit -- the LibraryLink library, both
// standalone binaries and the test binaries -- so a body written inline is recompiled once per
// target per header. One .cpp serves them all rather than one per header, because each target
// names its sources explicitly and a file added here has to be added to five lists.

#include "session.hpp"
#include "cpu_engine_holder.hpp"
#include "delivery_cursor.hpp"
#include "graph_marshal.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hgcommon/content_core.hpp"
#include "state_statistics.hpp"
#include "hgcommon/quotient_multiplicity_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <functional>
#include <type_traits>

namespace HG_NAMESPACE {
namespace ffi {

// =============================================================================
// EngineHolder
// =============================================================================

EngineHolder::~EngineHolder() = default;

DeliveryCursor& EngineHolder::delivery_cursor() { return delivery_cursor_; }

// =============================================================================
// SessionError / SessionSlot
// =============================================================================

SessionError::SessionError(const std::string& what) : std::runtime_error(what) {}

bool SessionSlot::is_live() const { return state_ == SessionState::Live; }

SessionState SessionSlot::state() const { return state_; }

uint64_t SessionSlot::handle() const { return handle_; }

std::string SessionSlot::already_live_message(uint64_t live_handle) {
    return "Open: a session is already live (" + std::to_string(live_handle) +
           "); this build serves one session at a time";
}

std::vector<size_t> steered_entries(const std::vector<int64_t>& entry_ids,
                                    const std::vector<int64_t>& wanted) {
    std::vector<size_t> out;
    for (int64_t want : wanted) {
        const size_t before = out.size();
        for (size_t i = 0; i < entry_ids.size(); ++i)
            if (entry_ids[i] == want) out.push_back(i);
        if (out.size() == before)
            throw std::runtime_error(
                "Step: state " + std::to_string(want) + " is not on this session's frontier, so "
                "there is nothing to continue from it. The frontier is reported as \"Frontier\" "
                "in every session reply.");
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

uint64_t SessionSlot::mint_handle() {
    // 0 is reserved, and handles are never reused.
    static uint64_t next = [] {
        std::random_device rd;
        const uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ rd() ^
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        return (hgcommon::mix64(seed) >> 2) | 1;
    }();
    return next++;
}

uint64_t SessionSlot::open(std::unique_ptr<EngineHolder> holder) {
    if (!holder) throw SessionError("Open: no engine holder");
    if (state_ == SessionState::Live)
        throw SessionError(already_live_message(handle_));
    holder_ = std::move(holder);
    handle_ = mint_handle();
    state_ = SessionState::Live;
    return handle_;
}

EngineHolder& SessionSlot::engine(uint64_t handle) {
    require(handle);
    return *holder_;
}

void SessionSlot::invalidate() {
    if (state_ != SessionState::Live) return;
    holder_.reset();
    state_ = SessionState::Invalidated;
}

void SessionSlot::close(uint64_t handle) {
    require(handle);
    holder_.reset();
    handle_ = kNoSession;
    state_ = SessionState::None;
}

void SessionSlot::require(uint64_t handle) const {
    if (handle == kNoSession) throw SessionError("no session handle given");
    if (handle != handle_)
        throw SessionError("session " + std::to_string(handle) + " is not this worker's live "
                           "session");
    if (state_ == SessionState::Invalidated)
        throw SessionError("session " + std::to_string(handle) + " was invalidated: the run "
                           "overflowed and its engine was discarded, so the exploration it "
                           "held is gone. Open a new session");
    if (state_ != SessionState::Live)
        throw SessionError("session " + std::to_string(handle) + " is closed");
}


// =============================================================================
// DeliveryCursor
// =============================================================================

bool DeliveryCursor::take_vertex(const std::string& property, int64_t id, uint32_t revision) {
    auto& sent = by_property_[property].vertex_revision;
    auto it = sent.find(id);
    if (it != sent.end() && it->second == revision) return false;
    sent[id] = revision;
    return true;
}

bool DeliveryCursor::take_edge(const std::string& property, int64_t from, int64_t to,
                               uint32_t type, uint32_t index) {
    return by_property_[property].edges.emplace(from, to, type, index).second;
}

bool DeliveryCursor::delivered_before(const std::string& property) const {
    return by_property_.find(property) != by_property_.end();
}

void DeliveryCursor::reset() { by_property_.clear(); }

// =============================================================================
// CpuEngineHolder
// =============================================================================

CpuEngineHolder::CpuEngineHolder(bool continuable, unsigned threads)
    : engine_(&hg_, threads ? threads : 1u) {
    engine_.set_continuable(continuable);
}

hypergraph::Hypergraph& CpuEngineHolder::hypergraph() { return hg_; }

hypergraph::ParallelEvolutionEngine& CpuEngineHolder::engine() { return engine_; }

const hypergraph::ParallelEvolutionEngine& CpuEngineHolder::engine() const { return engine_; }

void CpuEngineHolder::extend(int steps, const std::vector<hgcommon::StateId>& only_from) {
    if (steps <= 0) return;
    if (only_from.empty()) {
        engine_.evolve_more(static_cast<std::size_t>(steps));
        return;
    }
    const std::unordered_set<hgcommon::StateId> sel(only_from.begin(), only_from.end());
    engine_.evolve_more(static_cast<std::size_t>(steps), &sel);
}

std::vector<hgcommon::StateId> CpuEngineHolder::frontier() const {
    std::vector<hgcommon::StateId> out;
    for (const auto& [state, step] : engine_.frontier()) {
        (void)step;
        out.push_back(state);
    }
    return out;
}

}  // namespace ffi

namespace marshal {

// =============================================================================
// graph_marshal
// =============================================================================

std::vector<uint8_t> session_ack(uint64_t handle) {
    wxf::Writer w;
    w.write_header();
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(1);
    w.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
    w.write(std::string("Session"));
    w.write(static_cast<int64_t>(handle));
    return w.release_data();
}

uint32_t branchial_target_step(int branchial_step, int steps, bool& filter_by_step) {
    filter_by_step = (branchial_step != 0);
    if (!filter_by_step) return 0;
    if (branchial_step > 0) return static_cast<uint32_t>(branchial_step);
    return static_cast<uint32_t>(steps + 1 + branchial_step);
}

void push_branchial_state_edges(wxf::WXFValueAssociation& result,
                                const BranchialStateEdgeSet& set) {
    wxf::WXFValueList edges;
    for (const auto& e : set.edges) {
        wxf::WXFValueAssociation ed;
        ed.push_back({wxf::WXFValue("From"), wxf::WXFValue(e.first)});
        ed.push_back({wxf::WXFValue("To"), wxf::WXFValue(e.second)});
        edges.push_back(wxf::WXFValue(ed));
    }
    result.push_back({wxf::WXFValue("BranchialStateEdges"), wxf::WXFValue(edges)});

    wxf::WXFValueList verts;
    for (int64_t v : set.vertices) verts.push_back(wxf::WXFValue(v));
    result.push_back({wxf::WXFValue("BranchialStateVertices"), wxf::WXFValue(verts)});
}

void state_record_edges(bool full,
                        std::vector<std::pair<int64_t, std::vector<uint32_t>>>& edges) {
    if (!full || edges.empty()) return;
    std::vector<std::vector<hypergraph::VertexId>> contents;
    contents.reserve(edges.size());
    for (const auto& e : edges) contents.emplace_back(e.second.begin(), e.second.end());
    hypergraph::IRCanonicalizer ir;
    const auto canon = ir.canonicalize_edges(contents);
    edges.clear();
    int64_t idx = 0;
    for (const auto& ce : canon.canonical_form.edges)
        edges.emplace_back(idx++, std::vector<uint32_t>(ce.begin(), ce.end()));
}

GraphPropertyNeeds graph_property_needs(const std::string& graph_property) {
    const bool is_causal    = graph_property.rfind("Causal", 0) == 0;
    const bool is_branchial = graph_property.rfind("Branchial", 0) == 0;
    const bool is_evolution = graph_property.find("Evolution") != std::string::npos;
    const bool is_states    = graph_property.rfind("States", 0) == 0;
    return GraphPropertyNeeds{
        is_causal    || (is_evolution && graph_property.find("Causal") != std::string::npos),
        is_branchial || (is_evolution && graph_property.find("Branchial") != std::string::npos),
        is_states    || is_evolution};
}

std::unordered_map<uint64_t, int64_t> lowest_id_by_content(const std::vector<int64_t>& ids,
                                                           const std::vector<uint64_t>& hashes) {
    std::unordered_map<uint64_t, int64_t> out;
    out.reserve(ids.size());
    for (size_t i = 0; i < ids.size() && i < hashes.size(); ++i) {
        auto [it, fresh] = out.emplace(hashes[i], ids[i]);
        if (!fresh && ids[i] < it->second) it->second = ids[i];
    }
    return out;
}

uint64_t content_hash_of(std::vector<std::pair<int64_t, std::vector<uint32_t>>> edges) {
    std::sort(edges.begin(), edges.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    hgcommon::ContentHasher ch(static_cast<uint32_t>(edges.size()));
    for (const auto& e : edges) {
        ch.edge_begin(static_cast<uint32_t>(e.second.size()));
        for (uint32_t v : e.second) ch.vertex(static_cast<uint64_t>(v));
        ch.edge_end();
    }
    return ch.value();
}

std::string valid_utf8(const std::string& s) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size());
    const size_t n = s.size();
    auto byte = [&](size_t i) { return static_cast<unsigned char>(s[i]); };
    auto cont = [&](size_t i) { return i < n && (byte(i) & 0xC0) == 0x80; };
    for (size_t i = 0; i < n;) {
        const unsigned char c = byte(i);
        // Length of the well-formed sequence starting at i (RFC 3629 table 3-7), or 0.
        size_t len = 0;
        if (c < 0x80) {
            len = 1;
        } else if (c >= 0xC2 && c <= 0xDF) {
            len = cont(i + 1) ? 2 : 0;
        } else if (c >= 0xE0 && c <= 0xEF) {
            const unsigned char lo = c == 0xE0 ? 0xA0 : 0x80, hi = c == 0xED ? 0x9F : 0xBF;
            len = (i + 1 < n && byte(i + 1) >= lo && byte(i + 1) <= hi && cont(i + 2)) ? 3 : 0;
        } else if (c >= 0xF0 && c <= 0xF4) {
            const unsigned char lo = c == 0xF0 ? 0x90 : 0x80, hi = c == 0xF4 ? 0x8F : 0xBF;
            len = (i + 1 < n && byte(i + 1) >= lo && byte(i + 1) <= hi && cont(i + 2) &&
                   cont(i + 3)) ? 4 : 0;
        }
        if (len == 0) {
            out += "\\x";
            out += hex[c >> 4];
            out += hex[c & 0xF];
            ++i;
        } else {
            out.append(s, i, len);
            i += len;
        }
    }
    return out;
}

wxf::WXFValue warning_record(const std::string& kind, int64_t count, const std::string& context,
                             bool partial) {
    wxf::WXFValueAssociation wa;
    wa.push_back({wxf::WXFValue("Kind"), wxf::WXFValue(kind)});
    wa.push_back({wxf::WXFValue("Count"), wxf::WXFValue(count)});
    wa.push_back({wxf::WXFValue("Context"), wxf::WXFValue(valid_utf8(context))});
    wa.push_back({wxf::WXFValue("Partial"), wxf::WXFValue(static_cast<int64_t>(partial ? 1 : 0))});
    return wxf::WXFValue(wa);
}

GraphPropertyNeeds graph_property_needs(const std::vector<std::string>& properties) {
    GraphPropertyNeeds n;
    for (const std::string& p : properties) {
        const GraphPropertyNeeds one = graph_property_needs(p);
        n.causal    = n.causal    || one.causal;
        n.branchial = n.branchial || one.branchial;
        n.events    = n.events    || one.events;
    }
    return n;
}

}  // namespace marshal

namespace stats {

// =============================================================================
// state_invariants / summarise
// =============================================================================

const hgcommon::StateInvariantRecord* invariant_record(
    const std::vector<std::vector<uint32_t>>& edges, std::vector<uint64_t>& storage) {
    const uint32_t m = static_cast<uint32_t>(edges.size());
    std::vector<uint32_t> off(m + 1, 0), verts;
    for (uint32_t i = 0; i < m; ++i) {
        off[i] = static_cast<uint32_t>(verts.size());
        verts.insert(verts.end(), edges[i].begin(), edges[i].end());
    }
    off[m] = static_cast<uint32_t>(verts.size());
    const uint32_t slots = off[m];
    std::vector<uint32_t> scratch(hgcommon::si_scratch_words(slots, m, 1));
    storage.assign((hgcommon::StateInvariantRecord::bytes(m, slots) + 7) / 8, 0);
    auto* rec = new (storage.data()) hgcommon::StateInvariantRecord{};
    auto* words = reinterpret_cast<uint32_t*>(rec + 1);
    hgcommon::state_invariants(off.data(), verts.data(), m, scratch.data(), rec->v, words,
                               words + m);
    rec->num_edges = m;
    rec->num_vertices = static_cast<uint32_t>(rec->v.vertex_count);
    return rec;
}

StateInvariants state_invariants(const std::vector<std::vector<uint32_t>>& edges) {
    std::vector<uint64_t> storage;
    const hgcommon::StateInvariantRecord& rec = *invariant_record(edges, storage);
    const hgcommon::StateInvariantValues& v = rec.v;
    StateInvariants r;
    r.vertex_count = v.vertex_count;
    r.edge_count = v.edge_count;
    r.arities.assign(rec.arities(), rec.arities() + rec.num_edges);
    r.degree_sequence.assign(rec.degrees(), rec.degrees() + rec.num_vertices);
    r.max_degree = v.max_degree;
    r.mean_degree = v.mean_degree;
    r.two_section_edge_count = v.two_section_edge_count;
    r.components = v.components;
    r.cycle_rank = v.cycle_rank;
    r.incidence_cycle_rank = v.incidence_cycle_rank;
    r.incidence_diameter = v.incidence_diameter;
    r.incidence_mean_distance = v.incidence_mean_distance;
    r.largest_component_fraction = v.largest_component_fraction;
    return r;
}

Summary summarise(const std::vector<std::pair<double, uint64_t>>& value_weight, double round) {
    Summary s;
    std::vector<std::pair<double, uint64_t>> vw;
    for (const auto& p : value_weight)
        if (p.second > 0) vw.push_back(p);
    if (vw.empty()) return s;
    std::sort(vw.begin(), vw.end());
    long double n = 0, sum = 0;
    for (const auto& [v, w] : vw) {
        s.n = hgcommon::qm_sat_add(s.n, w);
        n += static_cast<long double>(w);
        sum += static_cast<long double>(v) * static_cast<long double>(w);
    }
    const long double mean = sum / n;
    long double sq = 0;
    for (const auto& [v, w] : vw)
        sq += static_cast<long double>(w) * (v - mean) * (v - mean);
    s.mean = static_cast<double>(mean);
    s.standard_deviation = n > 1 ? static_cast<double>(std::sqrt(sq / (n - 1))) : 0.0;
    s.min = vw.front().first;
    s.max = vw.back().first;
    // The values at 0-based positions floor((N-1)/2) and floor(N/2) of the sorted population.
    const long double lo_at = std::floor((n - 1) / 2), hi_at = std::floor(n / 2);
    long double seen = 0;
    double lo = vw.front().first, hi = vw.front().first;
    bool lo_set = false;
    for (const auto& [v, w] : vw) {
        if (!lo_set && seen + w > lo_at) { lo = v; lo_set = true; }
        if (seen + w > hi_at) { hi = v; break; }
        seen += w;
    }
    s.median = (lo + hi) / 2;
    for (const auto& [v, w] : vw) {
        // Halves to even (nearbyint in the default rounding mode), as Round[x, round] does.
        const double key = round == 1.0 ? v : std::nearbyint(v / round) * round;
        s.histogram[key] = hgcommon::qm_sat_add(s.histogram[key], w);
    }
    return s;
}

namespace {

wxf::WXFValue summary_value(const Summary& s, bool integral) {
    auto num = [&](double v) {
        return integral ? wxf::WXFValue(static_cast<int64_t>(std::llround(v))) : wxf::WXFValue(v);
    };
    wxf::WXFValueAssociation a;
    a.push_back({wxf::WXFValue("N"), wxf::WXFValue(static_cast<int64_t>(s.n))});
    if (s.n == 0) return wxf::WXFValue(a);
    a.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(s.mean)});
    a.push_back({wxf::WXFValue("StandardDeviation"), wxf::WXFValue(s.standard_deviation)});
    a.push_back({wxf::WXFValue("Min"), num(s.min)});
    a.push_back({wxf::WXFValue("Max"), num(s.max)});
    a.push_back({wxf::WXFValue("Median"), wxf::WXFValue(s.median)});
    wxf::WXFValueAssociation h;
    for (const auto& [v, w] : s.histogram) h.push_back({num(v), wxf::WXFValue(static_cast<int64_t>(w))});
    a.push_back({wxf::WXFValue("Histogram"), wxf::WXFValue(h)});
    return wxf::WXFValue(a);
}

template <class Key>
wxf::WXFValue count_association(const std::map<Key, uint64_t>& m) {
    wxf::WXFValueAssociation a;
    for (const auto& [k, w] : m) {
        if constexpr (std::is_same_v<Key, std::vector<int64_t>>) {
            wxf::WXFValueList l;
            for (int64_t x : k) l.push_back(wxf::WXFValue(x));
            a.push_back({wxf::WXFValue(l), wxf::WXFValue(static_cast<int64_t>(w))});
        } else {
            a.push_back({wxf::WXFValue(static_cast<int64_t>(k)), wxf::WXFValue(static_cast<int64_t>(w))});
        }
    }
    return wxf::WXFValue(a);
}

}  // namespace

void events_from_multiplicities(
    const std::vector<StepPoint>& points,
    const std::unordered_map<uint64_t, std::map<int64_t, uint64_t>>& matches_by_rule,
    uint32_t steps, std::map<uint32_t, uint64_t>& events,
    std::map<uint32_t, std::map<int64_t, uint64_t>>& rule_counts) {
    for (const auto& p : points) {
        if (p.step >= steps) continue;
        auto it = matches_by_rule.find(p.class_hash);
        if (it == matches_by_rule.end()) continue;
        for (const auto& [rule, k] : it->second) {
            const uint64_t w = hgcommon::qm_sat_mul(p.weight, k);
            auto& r = rule_counts[p.step + 1][rule];
            r = hgcommon::qm_sat_add(r, w);
            auto& e = events[p.step + 1];
            e = hgcommon::qm_sat_add(e, w);
        }
    }
}

wxf::WXFValue step_statistics(
    const std::vector<StepPoint>& points,
    const std::unordered_map<uint64_t, const hgcommon::StateInvariantRecord*>& class_invariants,
    const std::map<uint32_t, uint64_t>& events,
    const std::map<uint32_t, std::map<int64_t, uint64_t>>& rule_counts) {
    std::map<uint32_t, std::map<uint64_t, uint64_t>> by_step;
    for (const auto& p : points)
        if (p.weight) {
            auto& w = by_step[p.step][p.class_hash];
            w = hgcommon::qm_sat_add(w, p.weight);
        }
    static const hgcommon::StateInvariantRecord kNone{};
    auto invariants = [&](uint64_t h) -> const hgcommon::StateInvariantRecord& {
        auto it = class_invariants.find(h);
        return it != class_invariants.end() && it->second ? *it->second : kNone;
    };

    wxf::WXFValueList steps;
    for (const auto& [step, classes] : by_step) {
        uint64_t raw = 0, max_mult = 0;
        long double raw_ld = 0;
        std::map<int64_t, uint64_t> mult_hist;
        for (const auto& [h, w] : classes) {
            raw = hgcommon::qm_sat_add(raw, w);
            raw_ld += static_cast<long double>(w);
            max_mult = std::max(max_mult, w);
            ++mult_hist[static_cast<int64_t>(w)];
        }
        long double entropy = 0;
        for (const auto& [h, w] : classes) {
            const long double p = static_cast<long double>(w) / raw_ld;
            entropy -= p * std::log2(p);
        }
        const size_t nclasses = classes.size();

        std::vector<std::pair<double, uint64_t>> vertex_count, edge_count, max_degree, mean_degree,
            two_section, components, cycle_rank, inc_cycle_rank, inc_diameter, inc_mean_distance,
            largest_fraction;
        std::map<int64_t, uint64_t> arity_hist, degree_hist;
        std::map<std::vector<int64_t>, uint64_t> arity_sig_hist, degree_seq_hist;
        for (const auto& [h, w] : classes) {
            const hgcommon::StateInvariantRecord& r = invariants(h);
            const hgcommon::StateInvariantValues& s = r.v;
            vertex_count.push_back({double(s.vertex_count), w});
            edge_count.push_back({double(s.edge_count), w});
            max_degree.push_back({double(s.max_degree), w});
            mean_degree.push_back({s.mean_degree, w});
            two_section.push_back({double(s.two_section_edge_count), w});
            components.push_back({double(s.components), w});
            cycle_rank.push_back({double(s.cycle_rank), w});
            inc_cycle_rank.push_back({double(s.incidence_cycle_rank), w});
            inc_diameter.push_back({double(s.incidence_diameter), w});
            inc_mean_distance.push_back({s.incidence_mean_distance, w});
            largest_fraction.push_back({s.largest_component_fraction, w});
            const std::vector<int64_t> arities(r.arities(), r.arities() + r.num_edges);
            const std::vector<int64_t> degrees(r.degrees(), r.degrees() + r.num_vertices);
            for (int64_t a : arities) arity_hist[a] = hgcommon::qm_sat_add(arity_hist[a], w);
            for (int64_t d : degrees) degree_hist[d] = hgcommon::qm_sat_add(degree_hist[d], w);
            arity_sig_hist[arities] = hgcommon::qm_sat_add(arity_sig_hist[arities], w);
            degree_seq_hist[degrees] = hgcommon::qm_sat_add(degree_seq_hist[degrees], w);
        }

        wxf::WXFValueAssociation invs;
        auto put = [&](const char* k, const std::vector<std::pair<double, uint64_t>>& v,
                       double round, bool integral) {
            invs.push_back({wxf::WXFValue(k), summary_value(summarise(v, round), integral)});
        };
        put("VertexCount", vertex_count, 1.0, true);
        put("EdgeCount", edge_count, 1.0, true);
        put("MaxDegree", max_degree, 1.0, true);
        put("MeanDegree", mean_degree, 0.01, false);
        put("TwoSectionEdgeCount", two_section, 1.0, true);
        put("Components", components, 1.0, true);
        put("CycleRank", cycle_rank, 1.0, true);
        put("IncidenceCycleRank", inc_cycle_rank, 1.0, true);
        put("IncidenceDiameter", inc_diameter, 1.0, true);
        put("IncidenceMeanDistance", inc_mean_distance, 0.01, false);
        put("LargestComponentFraction", largest_fraction, 0.01, false);

        wxf::WXFValueAssociation rec;
        auto i64 = [](uint64_t v) { return wxf::WXFValue(static_cast<int64_t>(v)); };
        rec.push_back({wxf::WXFValue("Step"), wxf::WXFValue(static_cast<int64_t>(step))});
        rec.push_back({wxf::WXFValue("RawStates"), i64(raw)});
        rec.push_back({wxf::WXFValue("Classes"), i64(nclasses)});
        rec.push_back({wxf::WXFValue("Redundancy"),
                       wxf::WXFValue(static_cast<double>(raw_ld / nclasses))});
        rec.push_back({wxf::WXFValue("MaxMultiplicity"), i64(max_mult)});
        rec.push_back({wxf::WXFValue("MultiplicityHistogram"), count_association(mult_hist)});
        rec.push_back({wxf::WXFValue("ClassEntropyBits"), wxf::WXFValue(static_cast<double>(entropy))});
        rec.push_back({wxf::WXFValue("ClassEntropyNormalized"),
                       wxf::WXFValue(nclasses == 1 ? 1.0
                                     : static_cast<double>(entropy / std::log2(static_cast<long double>(nclasses))))});
        auto ev = events.find(step);
        rec.push_back({wxf::WXFValue("Events"), i64(ev == events.end() ? 0 : ev->second)});
        auto rc = rule_counts.find(step);
        rec.push_back({wxf::WXFValue("RuleCounts"),
                       count_association(rc == rule_counts.end() ? std::map<int64_t, uint64_t>{}
                                                                 : rc->second)});
        rec.push_back({wxf::WXFValue("Invariants"), wxf::WXFValue(invs)});
        rec.push_back({wxf::WXFValue("ArityHistogram"), count_association(arity_hist)});
        rec.push_back({wxf::WXFValue("AritySignatureHistogram"), count_association(arity_sig_hist)});
        rec.push_back({wxf::WXFValue("DegreeHistogram"), count_association(degree_hist)});
        rec.push_back({wxf::WXFValue("DegreeSequenceHistogram"), count_association(degree_seq_hist)});
        steps.push_back(wxf::WXFValue(rec));
    }
    return wxf::WXFValue(steps);
}

}  // namespace stats

}  // namespace HG_NAMESPACE
