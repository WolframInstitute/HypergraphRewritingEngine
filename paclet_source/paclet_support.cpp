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
#include "hgcommon/quotient_causal_core.hpp"
#include "hgcommon/branchial_overlap_core.hpp"
#include "hgcommon/branchial_rows_core.hpp"
#include "hgcommon/ir_core.hpp"
#include "job_system/job_system.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>
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

// =============================================================================
// StepStatisticsCollector
// =============================================================================

StepStatisticsCollector::StepStatisticsCollector(hypergraph::Hypergraph& hg,
                                                 hypergraph::ParallelEvolutionEngine& engine,
                                                 uint32_t which, bool by_class, int steps)
    : hg_(hg), engine_(engine), which_(which), steps_(steps),
      jobs_(by_class, which, engine.num_threads(),
            [&engine](std::function<void()> f) { engine.submit_side_job(std::move(f)); }) {}

StepStatisticsCollector::~StepStatisticsCollector() {
    if (attached_) {
        engine_.set_on_depth_complete(nullptr);
        hg_.set_index_states_by_step(false);
    }
}

void StepStatisticsCollector::attach(bool step_cap) {
    attached_ = true;
    indexed_ = true;
    hg_.set_index_states_by_step(true);
    engine_.set_on_depth_complete([this, step_cap](uint32_t depth) {
        if (inactive_) return;
        // A run whose classes are weighted by the multiplicity count has its weights only when
        // the run ends; the engine decides that before its first task.
        if (hg_.quotient_multiplicity()) {
            inactive_ = true;
            indexed_ = false;
            hg_.set_index_states_by_step(false);
            return;
        }
        // The report runs inside a task's completion, where an exception cannot propagate;
        // finish() rethrows it.
        try {
            close_through(step_cap ? depth - 1 : depth);
        } catch (...) {
            if (!failed_) failed_ = std::current_exception();
        }
    });
}

wxf::WXFValue StepStatisticsCollector::finish(std::map<uint32_t, stats::StepInput>* weighted) {
    if (attached_) engine_.set_on_depth_complete(nullptr);
    if (failed_) std::rethrow_exception(failed_);
    weighted_ = weighted;
    if (!indexed_) {
        const uint32_t n_pub = hg_.num_published_states();
        for (uint32_t sid = 0; sid < n_pub; ++sid) {
            const hypergraph::State& st = hg_.get_state(sid);
            if (st.id == hypergraph::INVALID_ID) continue;
            if (st.step >= by_step_.size()) by_step_.resize(st.step + 1);
            by_step_[st.step].push_back(sid);
        }
    }
    // A step with no state ends the run's steps: every state at s + 1 comes from one at s.
    uint32_t end = next_;
    while (!states_at(end).empty()) ++end;
    if (end > next_) {
        closed_at_finish_ = end - next_;
        close_through(end - 1);
    }
    engine_.wait_side_jobs();
    return jobs_.reply();
}

// Closes steps next_..last, from one thread at a time: the depth join's reporter, or the caller
// of finish once the run is over.
void StepStatisticsCollector::close_through(uint32_t last) {
    if (last < next_ || last == UINT32_MAX) return;
    if (which_ & stats::kBranchialGraph) take_pairs();
    for (; next_ <= last; ++next_) {
        const uint32_t s = next_;
        stats::StepStatisticsJobs::Step* slot = jobs_.reserve(s);
        std::vector<std::pair<int64_t, int64_t>> pairs;
        if (auto it = pairs_.find(s); it != pairs_.end()) {
            pairs = std::move(it->second);
            pairs_.erase(it);
        }
        // Step 0 is started here: every later step reads its vertices.
        if (s == 0) {
            jobs_.start(slot, gather(0, std::move(pairs)));
            continue;
        }
        engine_.submit_side_job([this, slot, s, p = std::move(pairs)]() mutable {
            jobs_.start(slot, gather(s, std::move(p)));
        });
    }
}

// The branchial state pairs recorded since the last call, by step.
void StepStatisticsCollector::take_pairs() {
    std::vector<std::pair<uint32_t, uint32_t>> events;
    hg_.causal_graph().for_each_branchial_edge_since(
        cursor_, [&](const hypergraph::BranchialEdge& be) {
            if (hg_.is_genesis_event(be.event1) || hg_.is_genesis_event(be.event2)) return;
            events.emplace_back(be.event1, be.event2);
        });
    marshal::branchial_step_pairs(
        events,
        [&](uint32_t eid) { return static_cast<uint32_t>(hg_.get_event(eid).output_state); },
        [&](uint32_t sid) { return static_cast<int64_t>(hg_.get_canonical_state(sid)); },
        [&](uint32_t sid) { return hg_.get_state(sid).step; }, steps_,
        [&](uint32_t step, int64_t a, int64_t b) { pairs_[step].emplace_back(a, b); });
}

std::vector<uint32_t> StepStatisticsCollector::states_at(uint32_t step) const {
    std::vector<uint32_t> sids;
    if (indexed_) {
        hg_.for_each_state_at_step(step, [&](hypergraph::StateId sid) { sids.push_back(sid); });
        std::sort(sids.begin(), sids.end());
    } else if (step < by_step_.size()) {
        sids = by_step_[step];
    }
    // The genesis events' input state stands before the initial states and is not one of them.
    if (step == 0)
        sids.erase(std::remove_if(sids.begin(), sids.end(),
                                  [&](uint32_t sid) { return hg_.is_genesis_state(sid); }),
                   sids.end());
    return sids;
}

// Step `step`'s classes, with each class's record, and the events into the step, each the
// creating event of one of its raw states; or, with weighted inputs from finish, the step's entry
// there (read by this step's job only). Then the step's branchial input, the states in id order.
stats::StepInput StepStatisticsCollector::gather(uint32_t step,
                                                 std::vector<std::pair<int64_t, int64_t>> pairs) {
    stats::StepInput in;
    if (weighted_) {
        if (auto it = weighted_->find(step); it != weighted_->end()) in = std::move(it->second);
        in.branchial = {};
    }
    marshal::ListNumbers edge_number;
    std::vector<uint32_t> flat;
    auto contents = [&](uint32_t sid, std::vector<uint32_t>& vs) {
        hg_.get_state(sid).edges.for_each([&](hypergraph::EdgeId eid) {
            const auto& e = hg_.get_edge(eid);
            vs.push_back(e.arity);
            vs.insert(vs.end(), e.vertices, e.vertices + e.arity);
        });
    };
    for (uint32_t sid : states_at(step)) {
        const hypergraph::State& st = hg_.get_state(sid);
        if (st.id == hypergraph::INVALID_ID) continue;
        if (!weighted_) {
            const uint64_t h = hg_.get_or_compute_canonical_hash(sid);
            stats::add_class(in, h, 1, hg_.state_invariants_of(sid, h));
            const hypergraph::EventId e = st.parent_event;
            if (e != hypergraph::INVALID_ID && !hg_.is_genesis_event(e)) {
                const hypergraph::Event& ev = hg_.get_event(e);
                if (ev.id != hypergraph::INVALID_ID && ev.output_state == sid) {
                    ++in.events;
                    ++in.rule_counts[static_cast<int64_t>(ev.rule_index)];
                }
            }
        }
        if (which_)
            marshal::branchial_step_state(which_, sid,
                                          static_cast<int64_t>(hg_.get_canonical_state(sid)),
                                          contents, edge_number, flat, in.branchial);
    }
    in.branchial.pairs = std::move(pairs);
    return in;
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
    // The geometry reads the state in its IR canonical labelling (state_record).
    std::vector<uint32_t> goff(1, 0), gverts;
    if (m) {
        const std::vector<std::vector<hypergraph::VertexId>> in(edges.begin(), edges.end());
        for (const auto& e : hypergraph::IRCanonicalizer{}.canonicalize_edges(in)
                                 .canonical_form.edges) {
            gverts.insert(gverts.end(), e.begin(), e.end());
            goff.push_back(static_cast<uint32_t>(gverts.size()));
        }
    }
    const uint32_t gm = static_cast<uint32_t>(goff.size() - 1);
    uint64_t bytes = hgcommon::si_record_bytes_hint(slots, m, 1);
    std::vector<uint64_t> scratch;
    hgcommon::SiResult r;
    for (;;) {
        scratch.assign((bytes + 7) / 8, 0);
        uint64_t needed = 0;
        if (hgcommon::state_record(off.data(), verts.data(), m, goff.data(), gverts.data(), gm,
                                   reinterpret_cast<unsigned char*>(scratch.data()), bytes,
                                   needed, r))
            break;
        bytes = needed > bytes ? needed : 2 * bytes;
    }
    storage.assign(r.bytes() / 8, 0);
    return hgcommon::si_record_write(storage.data(), r);
}

StateInvariants state_invariants(const std::vector<std::vector<uint32_t>>& edges) {
    std::vector<uint64_t> storage;
    const hgcommon::StateInvariantRecord& rec = *invariant_record(edges, storage);
    const hgcommon::StateInvariantValues& v = rec.v;
    StateInvariants r;
    r.vertex_count = v.vertex_count;
    r.edge_count = v.edge_count;
    rec.for_each_arity([&](uint32_t a) { r.arities.push_back(a); });
    rec.for_each_degree([&](uint32_t d) { r.degree_sequence.push_back(d); });
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
    // vw is ascending and the rounding is monotonic, so the keys arrive ascending and each new key
    // goes at the end of the map.
    for (const auto& [v, w] : vw) {
        // Halves to even (nearbyint in the default rounding mode), as Round[x, round] does.
        const double key = round == 1.0 ? v : std::nearbyint(v / round) * round;
        if (!s.histogram.empty() && std::prev(s.histogram.end())->first == key)
            std::prev(s.histogram.end())->second =
                hgcommon::qm_sat_add(std::prev(s.histogram.end())->second, w);
        else
            s.histogram.emplace_hint(s.histogram.end(), key, w);
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

// One step's per-vertex distribution: the classes' hgcommon::SgDistribution summaries, each
// weighted by its class's weight. Pooling adds the counts, power sums and histogram bins.
struct PooledDistribution {
    uint32_t which = 0;
    uint64_t n = 0;
    long double weight = 0, sum[4] = {0, 0, 0, 0};
    double min = 0.0, max = 0.0;
    long double bins[hgcommon::SG_DIST_BINS] = {};

    void add(const hgcommon::SgDistribution& d, uint64_t w) {
        if (d.count == 0 || w == 0) return;
        if (n == 0 || d.min < min) min = d.min;
        if (n == 0 || d.max > max) max = d.max;
        n = hgcommon::qm_sat_add(n, hgcommon::qm_sat_mul(d.count, w));
        const long double lw = static_cast<long double>(w);
        weight += lw * d.count;
        for (int k = 0; k < 4; ++k) sum[k] += lw * d.sum[k];
        for (uint32_t b = 0; b < hgcommon::SG_DIST_BINS; ++b) bins[b] += lw * d.bins[b];
    }

    // The histogram quantile at q: the bin holding the q-th fraction of the weight, linear inside
    // the bin between its edges clipped to [min, max].
    double quantile(long double q) const {
        double lo = 0.0, hi = 0.0;
        hgcommon::sg_dist_range(which, lo, hi);
        const double width = (hi - lo) / hgcommon::SG_DIST_BINS;
        const long double t = q * weight;
        long double seen = 0;
        for (uint32_t b = 0; b < hgcommon::SG_DIST_BINS; ++b) {
            if (bins[b] == 0 || seen + bins[b] < t) { seen += bins[b]; continue; }
            const double a = b == 0 ? min : std::max(lo + b * width, min);
            const double z = b + 1 == hgcommon::SG_DIST_BINS ? max : std::min(lo + (b + 1) * width, max);
            const double v = static_cast<double>(a + (t - seen) / bins[b] * (z - a));
            return std::min(std::max(v, min), max);
        }
        return max;
    }
};

wxf::WXFValue pooled_value(const PooledDistribution& p) {
    wxf::WXFValueAssociation a;
    auto put = [&](const char* k, double v) { a.push_back({wxf::WXFValue(k), wxf::WXFValue(v)}); };
    a.push_back({wxf::WXFValue("N"), wxf::WXFValue(static_cast<int64_t>(p.n))});
    if (p.n == 0) return wxf::WXFValue(a);
    const long double N = p.weight, mean = p.sum[0] / N;
    const long double e2 = p.sum[1] / N, e3 = p.sum[2] / N, e4 = p.sum[3] / N;
    long double m2 = 0, m3 = 0, m4 = 0;
    if (p.min != p.max) {
        m2 = e2 - mean * mean;
        m3 = e3 - 3 * mean * e2 + 2 * mean * mean * mean;
        m4 = e4 - 4 * mean * e3 + 6 * mean * mean * e2 - 3 * mean * mean * mean * mean;
        if (m2 < 0) m2 = 0;
    }
    put("Mean", static_cast<double>(mean));
    put("StandardDeviation", N > 1 ? static_cast<double>(std::sqrt(m2 * N / (N - 1))) : 0.0);
    put("Min", p.min);
    put("Max", p.max);
    put("Median", p.quantile(0.5L));
    put("Q1", p.quantile(0.25L));
    put("Q3", p.quantile(0.75L));
    put("P10", p.quantile(0.1L));
    put("P90", p.quantile(0.9L));
    if (m2 > 0) {
        put("Skewness", static_cast<double>(m3 / (m2 * std::sqrt(m2))));
        put("Kurtosis", static_cast<double>(m4 / (m2 * m2)));
    }
    double lo = 0.0, hi = 0.0;
    hgcommon::sg_dist_range(p.which, lo, hi);
    const double width = (hi - lo) / hgcommon::SG_DIST_BINS;
    wxf::WXFValueAssociation h;
    for (uint32_t b = 0; b < hgcommon::SG_DIST_BINS; ++b)
        if (p.bins[b] > 0)
            h.push_back({wxf::WXFValue(lo + b * width),
                         wxf::WXFValue(static_cast<int64_t>(std::min<long double>(
                             p.bins[b], static_cast<long double>(INT64_MAX))))});
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

// A state's hyperedges as the EdgeList hgcommon::sg_state_geometry reads.
struct NestedEdges {
    const std::vector<std::vector<uint32_t>>* e;
    uint32_t count() const { return static_cast<uint32_t>(e->size()); }
    uint32_t arity(uint32_t i) const { return static_cast<uint32_t>((*e)[i].size()); }
    uint32_t at(uint32_t i, uint32_t k) const { return (*e)[i][k]; }
};

// sg_state_geometry with a buffer that grows to what the call reports it needs.
template <class EdgeList>
hgcommon::SgGeometry geometry_with_scratch(const EdgeList& el, std::vector<double>& ball,
                                           uint32_t want) {
    static thread_local std::vector<unsigned char> scratch(1 << 16);
    hgcommon::SgGeometry g;
    size_t needed = 0;
    while (!hgcommon::sg_state_geometry(el, scratch.data(), scratch.size(), g, ball.data(),
                                        static_cast<uint32_t>(ball.size()), needed, want))
        scratch.resize(std::max(needed, scratch.size() * 2));
    return g;
}

}  // namespace

StateGeometry state_geometry(const std::vector<std::vector<uint32_t>>& edges) {
    StateGeometry r;
    size_t slots = 0;
    for (const auto& e : edges) slots += e.size();
    r.ball_dimension.assign(slots + 1, 0.0);   // R < number of vertices <= slots
    r.values = geometry_with_scratch(NestedEdges{&edges}, r.ball_dimension, ~0u);
    r.ball_dimension.resize(r.values.ball_radii);
    return r;
}

namespace {

// A branchial graph's nodes and edges as an EdgeList: one binary edge per pair, and a unary
// edge per node so that a node with no pair is a vertex.
struct BranchialEdges {
    const std::vector<std::pair<uint32_t, uint32_t>>* pairs;
    uint32_t nodes;
    uint32_t count() const { return static_cast<uint32_t>(pairs->size()) + nodes; }
    uint32_t arity(uint32_t i) const { return i < pairs->size() ? 2u : 1u; }
    uint32_t at(uint32_t i, uint32_t k) const {
        if (i >= pairs->size()) return i - static_cast<uint32_t>(pairs->size());
        return k == 0 ? (*pairs)[i].first : (*pairs)[i].second;
    }
};

void put_summary(wxf::WXFValueAssociation& rec, const char* key,
                 const std::vector<std::pair<double, uint64_t>>& values, double round,
                 bool integral) {
    rec.push_back({wxf::WXFValue(key), summary_value(summarise(values, round), integral)});
}

// The branchial graph of one step: the step's states (`ids`, by effective id, ascending), joined
// by the step's pairs (a pair of one state with itself adds nothing, and a repeated pair one
// edge).
struct BranchialGraph {
    std::vector<int64_t> ids;
    std::vector<std::pair<uint32_t, uint32_t>> edges;   // x < y, ascending
    std::vector<std::vector<uint32_t>> adj;
};

BranchialGraph branchial_graph(const BranchialStep& s) {
    BranchialGraph g;
    std::vector<int64_t>& ids = g.ids;
    ids = s.nodes;
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const uint32_t n = static_cast<uint32_t>(ids.size());
    auto index = [&](int64_t id) {
        return static_cast<uint32_t>(std::lower_bound(ids.begin(), ids.end(), id) - ids.begin());
    };
    std::vector<std::pair<uint32_t, uint32_t>>& edges = g.edges;
    for (const auto& [a, b] : s.pairs) {
        if (a == b) continue;
        uint32_t x = index(a), y = index(b);
        if (x >= n || y >= n || ids[x] != a || ids[y] != b) continue;
        if (x > y) std::swap(x, y);
        edges.emplace_back(x, y);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    g.adj.assign(n, {});
    for (const auto& [x, y] : edges) { g.adj[x].push_back(y); g.adj[y].push_back(x); }
    return g;
}

// The branchial graph metrics of one step.
void branchial_graph_metrics(const BranchialGraph& graph, wxf::WXFValueAssociation& rec) {
    const uint32_t n = static_cast<uint32_t>(graph.ids.size());
    const auto& edges = graph.edges;
    const auto& adj = graph.adj;

    std::vector<std::pair<double, uint64_t>> degree, distance;
    for (uint32_t v = 0; v < n; ++v) degree.push_back({double(adj[v].size()), 1});
    std::vector<int64_t> comp(n, -1), dist(n, -1);
    std::vector<std::vector<uint32_t>> members;
    std::vector<uint32_t> queue;
    std::map<int64_t, uint64_t> distance_counts;
    for (uint32_t v = 0; v < n; ++v) {
        if (comp[v] < 0) {
            const int64_t c = static_cast<int64_t>(members.size());
            members.emplace_back();
            queue.assign(1, v);
            comp[v] = c;
            for (size_t i = 0; i < queue.size(); ++i) {
                members[c].push_back(queue[i]);
                for (uint32_t w : adj[queue[i]])
                    if (comp[w] < 0) { comp[w] = c; queue.push_back(w); }
            }
        }
        // Distances to the later states of the same component: each unordered pair once.
        queue.assign(1, v);
        dist[v] = 0;
        for (size_t i = 0; i < queue.size(); ++i)
            for (uint32_t w : adj[queue[i]])
                if (dist[w] < 0) { dist[w] = dist[queue[i]] + 1; queue.push_back(w); }
        for (uint32_t w : queue) {
            if (w > v) ++distance_counts[dist[w]];
            dist[w] = -1;
        }
    }
    for (const auto& [d, k] : distance_counts) distance.push_back({double(d), k});
    put_summary(rec, "BranchialDegree", degree, 1.0, true);
    put_summary(rec, "BranchialDistance", distance, 1.0, true);
    rec.push_back({wxf::WXFValue("BranchialComponents"),
                   wxf::WXFValue(static_cast<int64_t>(members.size()))});

    // The dimension of the largest component: most states, then the greatest dimension.
    size_t most = 0;
    for (const auto& c : members) most = std::max(most, c.size());
    bool have = false;
    double best = 0.0;
    std::vector<double> ball(most + 1);
    for (const auto& c : members) {
        if (c.size() != most || most < 2) continue;
        std::vector<uint32_t> local(n, 0);
        for (uint32_t i = 0; i < c.size(); ++i) local[c[i]] = i;
        // The component in its IR canonical labelling, each edge in both directions so the
        // labelling is one of the undirected graph: the dimension's floating-point sums follow
        // the vertex order, and the states' ids are a function of the schedule.
        std::vector<std::vector<hypergraph::VertexId>> both;
        for (const auto& [x, y] : edges)
            if (comp[x] == comp[c[0]]) {
                both.push_back({local[x], local[y]});
                both.push_back({local[y], local[x]});
            }
        std::vector<std::pair<uint32_t, uint32_t>> sub;
        for (const auto& e : hypergraph::IRCanonicalizer{}.canonicalize_edges(both)
                                 .canonical_form.edges)
            if (e[0] < e[1]) sub.emplace_back(e[0], e[1]);
        const hgcommon::SgGeometry g = geometry_with_scratch(
            BranchialEdges{&sub, static_cast<uint32_t>(c.size())}, ball, hgcommon::SG_HAUSDORFF);
        if ((g.defined & hgcommon::SG_HAUSDORFF) && (!have || g.hausdorff_dimension > best)) {
            best = g.hausdorff_dimension;
            have = true;
        }
    }
    if (have) rec.push_back({wxf::WXFValue("BranchialDimension"), wxf::WXFValue(best)});
}

// The summaries of 1/k and log2 k over the ids held by k states, one value per id, from the
// number of ids held by k states, by k.
void put_multiplicity(wxf::WXFValueAssociation& rec, const char* sharpness_key,
                      const char* entropy_key, const std::map<uint64_t, uint64_t>& by_k) {
    std::vector<std::pair<double, uint64_t>> sharpness, entropy;
    for (const auto& [k, ids] : by_k) {
        sharpness.push_back({hgcommon::bo_sharpness(k), ids});
        entropy.push_back({hgcommon::bo_branch_entropy(k), ids});
    }
    put_summary(rec, sharpness_key, sharpness, 0.01, false);
    put_summary(rec, entropy_key, entropy, 0.01, false);
}

// The counts of a pair of states: the smaller and the larger vertex count, the shared vertices,
// and the pair's branchial distance. Distance 0 counts every pair that shares a vertex; a
// distance d > 0 counts every pair of one branchial component at distance d, sharing or not, so a
// pair can be counted under both.
struct PairCounts {
    uint32_t distance, lo, hi, both;
    bool operator==(const PairCounts& o) const {
        return distance == o.distance && lo == o.lo && hi == o.hi && both == o.both;
    }
    bool operator<(const PairCounts& o) const {
        if (distance != o.distance) return distance < o.distance;
        if (lo != o.lo) return lo < o.lo;
        if (hi != o.hi) return hi < o.hi;
        return both < o.both;
    }
};
struct PairCountsHash {
    size_t operator()(const PairCounts& k) const {
        uint64_t h = (uint64_t{k.lo} << 32 | k.hi) * 0x9E3779B97F4A7C15ull;
        h ^= (uint64_t{k.both} << 32 | k.distance) + 0x632BE59BD9B4E019ull + (h << 6) + (h >> 2);
        return static_cast<size_t>(h * 0xBF58476D1CE4E5B9ull);
    }
};

// One row's pair counts, keyed by distance << 48 | |B| << 24 | shared, in an open-addressed
// table cleared slot by slot after the row: a pair costs one probe of a table sized to the row's
// distinct keys, and each key reaches the thread's PairCounts map once per row.
struct RowTally {
    static constexpr uint64_t kEmpty = ~uint64_t{0};
    std::vector<uint64_t> keys = std::vector<uint64_t>(64, kEmpty);
    std::vector<uint64_t> counts = std::vector<uint64_t>(64, 0);
    std::vector<uint32_t> used;

    static bool fits(uint32_t distance, uint32_t size, uint32_t both) {
        return distance < (1u << 16) && size < (1u << 24) && both < (1u << 24);
    }
    static uint64_t key(uint32_t distance, uint32_t size, uint32_t both) {
        return uint64_t{distance} << 48 | uint64_t{size} << 24 | both;
    }
    static size_t slot_hash(uint64_t k) {
        k ^= k >> 29;
        k *= 0xBF58476D1CE4E5B9ull;
        return static_cast<size_t>(k ^ (k >> 32));
    }
    size_t find(uint64_t k) const {
        const size_t mask = keys.size() - 1;
        size_t i = slot_hash(k) & mask;
        while (keys[i] != kEmpty && keys[i] != k) i = (i + 1) & mask;
        return i;
    }
    void add(uint64_t k) {
        if ((used.size() + 1) * 2 > keys.size()) {
            std::vector<uint64_t> old_keys(keys.size() * 2, kEmpty), old_counts(keys.size() * 2, 0);
            old_keys.swap(keys);
            old_counts.swap(counts);
            for (uint32_t& u : used) {
                const size_t j = find(old_keys[u]);
                keys[j] = old_keys[u];
                counts[j] = old_counts[u];
                u = static_cast<uint32_t>(j);
            }
        }
        const size_t i = find(k);
        if (keys[i] == kEmpty) {
            keys[i] = k;
            used.push_back(static_cast<uint32_t>(i));
        }
        ++counts[i];
    }
    // Calls f(distance, size, both, pairs) for each key and clears the table.
    template <class F>
    void drain(F&& f) {
        for (uint32_t i : used) {
            const uint64_t k = keys[i];
            f(static_cast<uint32_t>(k >> 48), static_cast<uint32_t>(k >> 24) & 0xFFFFFFu,
              static_cast<uint32_t>(k) & 0xFFFFFFu, counts[i]);
            keys[i] = kEmpty;
            counts[i] = 0;
        }
        used.clear();
    }
};

}  // namespace

// The overlap metrics of one step, in parts. Each state is its set of the engine's vertex ids, and
// a vertex is shared by the states that inherited it; an edge is its list of vertex ids, numbered
// in BranchialStep::edge_sets. `graph`, when the branchial graph metrics are computed too, gives
// each pair's branchial distance for "OverlapByBranchialDistance". `initial`, the vertices of step
// 0's states, gives "InitialStateMutualInformation".
//
// Each state a is counted against the states b > a that share a vertex with it: a count per b in
// a dense array, the rows split into parts, each part with its own array and counts. With the
// branchial graph, a breadth-first search from a also gives every b > a in a's component, sharing
// or not, with its distance. A pair's values are functions of its PairCounts, so each part counts
// the pairs per PairCounts and the counts are merged in key order: the order the rows land in does
// not reach the reply. The search costs O(component) per row, the cost of the "BranchialDistance"
// pass.
struct BranchialStepWork::State {
    BranchialStep step;
    uint32_t which;
    const std::vector<uint32_t>* initial;
    BranchialGraph graph;
    wxf::WXFValueAssociation graph_keys;
    // The overlap pass.
    std::vector<std::vector<uint32_t>> sets, edge_sets;
    std::vector<int64_t> ids;
    std::map<uint32_t, std::vector<uint32_t>> holders;   // vertex -> the states holding it
    // The step in compressed rows (hgcommon::BrStep): vertices numbered densely in `holders`
    // order.
    std::vector<uint32_t> size, row_off, row_vert, hold_off, hold, adj_off, adj, to_graph,
        from_graph;
    hgcommon::BrStep view;
    std::vector<std::unordered_map<PairCounts, uint64_t, PairCountsHash>> part;
    std::atomic<uint32_t> next_row{0};

    const BranchialGraph* graph_or_null() const {
        return (which & kBranchialGraph) ? &graph : nullptr;
    }
};

BranchialStepWork::BranchialStepWork(BranchialStep step, uint32_t which,
                                     const std::vector<uint32_t>* initial)
    : s_(new State) {
    s_->step = std::move(step);
    s_->which = which;
    s_->initial = initial;
}

BranchialStepWork::~BranchialStepWork() = default;

void BranchialStepWork::prepare(size_t max_parts) {
    State& w = *s_;
    if (w.which & kBranchialGraph) {
        w.graph = branchial_graph(w.step);
        branchial_graph_metrics(w.graph, w.graph_keys);
    }
    if (!(w.which & kBranchialOverlap)) return;
    const BranchialStep& s = w.step;
    // Each state once, by effective id.
    std::map<int64_t, size_t> by_id;
    for (size_t i = 0; i < s.nodes.size() && i < s.vertex_sets.size(); ++i)
        by_id.emplace(s.nodes[i], i);
    for (const auto& [id, i] : by_id) {
        std::vector<uint32_t> v = s.vertex_sets[i];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        w.sets.push_back(std::move(v));
        std::vector<uint32_t> e;
        if (i < s.edge_sets.size()) e = s.edge_sets[i];
        std::sort(e.begin(), e.end());
        e.erase(std::unique(e.begin(), e.end()), e.end());
        w.edge_sets.push_back(std::move(e));
        w.ids.push_back(id);
    }
    const uint32_t k = static_cast<uint32_t>(w.sets.size());
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t v : w.sets[i]) w.holders[v].push_back(i);
    std::unordered_map<uint32_t, uint32_t> dense;
    dense.reserve(w.holders.size());
    w.hold_off.push_back(0);
    for (const auto& [v, hs] : w.holders) {
        dense.emplace(v, static_cast<uint32_t>(dense.size()));
        w.hold.insert(w.hold.end(), hs.begin(), hs.end());
        w.hold_off.push_back(static_cast<uint32_t>(w.hold.size()));
    }
    w.row_off.push_back(0);
    for (uint32_t a = 0; a < k; ++a) {
        w.size.push_back(static_cast<uint32_t>(w.sets[a].size()));
        for (uint32_t v : w.sets[a]) w.row_vert.push_back(dense.at(v));
        w.row_off.push_back(static_cast<uint32_t>(w.row_vert.size()));
    }
    hgcommon::BrStep& view = w.view;
    view.k = k;
    view.size = w.size.data();
    view.row_off = w.row_off.data();
    view.row_vert = w.row_vert.data();
    view.hold_off = w.hold_off.data();
    view.hold = w.hold.data();
    view.vertices = static_cast<uint32_t>(w.holders.size());
    // A state's node in the branchial graph, and back.
    if (const BranchialGraph* graph = w.graph_or_null()) {
        const uint32_t n = static_cast<uint32_t>(graph->ids.size());
        w.from_graph.assign(n, hgcommon::BR_NONE);
        for (uint32_t i = 0; i < k; ++i) {
            const auto it = std::lower_bound(graph->ids.begin(), graph->ids.end(), w.ids[i]);
            const uint32_t x = it != graph->ids.end() && *it == w.ids[i]
                                   ? static_cast<uint32_t>(it - graph->ids.begin())
                                   : hgcommon::BR_NONE;
            w.to_graph.push_back(x);
            if (x != hgcommon::BR_NONE) w.from_graph[x] = i;
        }
        w.adj_off.push_back(0);
        for (uint32_t x = 0; x < n; ++x) {
            w.adj.insert(w.adj.end(), graph->adj[x].begin(), graph->adj[x].end());
            w.adj_off.push_back(static_cast<uint32_t>(w.adj.size()));
        }
        view.n = n;
        view.adj_off = w.adj_off.data();
        view.adj = w.adj.data();
        view.to_graph = w.to_graph.data();
        view.from_graph = w.from_graph.data();
    }
    w.part.resize(std::min<size_t>(k / 64 + 1, std::max<size_t>(1, max_parts)));
}

size_t BranchialStepWork::parts() const { return s_->part.size(); }

const hgcommon::BrStep& BranchialStepWork::rows_view() const { return s_->view; }

void BranchialStepWork::add_counts(uint32_t distance, uint32_t size_a, uint32_t size_b,
                                   uint32_t shared, uint64_t n) {
    s_->part.at(0)[PairCounts{distance, std::min(size_a, size_b), std::max(size_a, size_b),
                              shared}] += n;
}

namespace {

// hgcommon::br_row's scratch on one host worker: plain memory and a RowTally per row.
struct HostRowCtx {
    std::vector<uint32_t> count_v, touched_v, queue_v;
    std::vector<int32_t> dist_v;
    uint32_t* count;
    uint32_t* touched;
    int32_t* dist;
    uint32_t* queue;
    uint32_t touched_n = 0, queue_n = 0;
    uint32_t* n_touched = &touched_n;
    uint32_t* n_queue = &queue_n;
    uint32_t size_a = 0;
    RowTally tally;
    std::unordered_map<PairCounts, uint64_t, PairCountsHash>* counted;

    HostRowCtx(const hgcommon::BrStep& s, std::unordered_map<PairCounts, uint64_t, PairCountsHash>& c)
        : count_v(s.k, 0), touched_v(s.k), queue_v(s.n), dist_v(s.n, -1), counted(&c) {
        count = count_v.data();
        touched = touched_v.data();
        dist = dist_v.data();
        queue = queue_v.data();
    }
    template <class T> T ld(const T* p) const { return *p; }
    template <class T> void st(T* p, T v) const { *p = v; }
    uint32_t add_count(uint32_t b) { return count[b]++; }
    bool claim_dist(uint32_t x, int32_t d) {
        if (dist[x] >= 0) return false;
        dist[x] = d;
        return true;
    }
    uint32_t push(uint32_t* counter) { return (*counter)++; }
    void emit(uint32_t distance, uint32_t size_b, uint32_t shared) {
        if (RowTally::fits(distance, size_b, shared))
            tally.add(RowTally::key(distance, size_b, shared));
        else
            ++(*counted)[PairCounts{distance, std::min(size_a, size_b), std::max(size_a, size_b),
                                    shared}];
    }
};

}  // namespace

void BranchialStepWork::rows(size_t t) {
    State& w = *s_;
    const hgcommon::BrStep& s = w.view;
    auto& counted = w.part[t];
    HostRowCtx c(s, counted);
    for (uint32_t a; (a = w.next_row.fetch_add(1, std::memory_order_relaxed)) < s.k;) {
        c.size_a = s.size[a];
        hgcommon::br_row(c, s, a, hgcommon::IrSerial{});
        c.tally.drain([&](uint32_t distance, uint32_t size_b, uint32_t both, uint64_t n) {
            counted[PairCounts{distance, std::min(c.size_a, size_b), std::max(c.size_a, size_b),
                               both}] += n;
        });
    }
}

wxf::WXFValueAssociation BranchialStepWork::finish() {
    State& w = *s_;
    wxf::WXFValueAssociation rec = std::move(w.graph_keys);
    if (!(w.which & kBranchialOverlap)) return rec;
    const BranchialGraph* graph = w.graph_or_null();
    const size_t k = w.sets.size();
    std::map<PairCounts, uint64_t> counted;
    for (const auto& p : w.part)
        for (const auto& [key, n] : p) counted[key] += n;

    // The pairs that share no vertex: per (lo, hi), every pair of states of those sizes less the
    // ones counted at distance 0.
    std::map<uint32_t, uint64_t> by_size;
    for (const auto& v : w.sets) ++by_size[static_cast<uint32_t>(v.size())];
    std::map<std::pair<uint32_t, uint32_t>, uint64_t> rest;
    for (auto i = by_size.begin(); i != by_size.end(); ++i) {
        rest[{i->first, i->first}] = i->second * (i->second - 1) / 2;
        for (auto j = std::next(i); j != by_size.end(); ++j)
            rest[{i->first, j->first}] = i->second * j->second;
    }
    for (const auto& [key, n] : counted)
        if (key.distance == 0) rest[{key.lo, key.hi}] -= n;

    const uint64_t universe = w.holders.size();
    std::map<uint64_t, uint64_t> ratios;   // (shared << 32 | union) -> pairs, shared > 0
    std::map<double, uint64_t> cosine, information;
    std::map<uint32_t, std::map<uint64_t, uint64_t>> by_distance;   // the same, per distance
    std::map<uint32_t, uint64_t> at_distance;                       // every pair, per distance
    uint64_t shared_pairs = 0;
    for (const auto& [key, n] : counted) {
        const uint64_t uni = uint64_t{key.lo} + key.hi - key.both;
        if (key.distance) {
            at_distance[key.distance] += n;
            if (key.both) by_distance[key.distance][uint64_t{key.both} << 32 | uni] += n;
            continue;
        }
        ratios[uint64_t{key.both} << 32 | uni] += n;
        shared_pairs += n;
        cosine[hgcommon::bo_cosine(key.both, key.lo, key.hi)] += n;
        information[hgcommon::bo_mutual_information(key.both, key.lo, key.hi, universe)] += n;
    }
    for (const auto& [sizes, n] : rest) {
        if (!n) continue;
        cosine[0.0] += n;
        information[hgcommon::bo_mutual_information(0, sizes.first, sizes.second, universe)] += n;
    }
    // The Jaccard values of (shared << 32 | union) counts, with `zero` pairs sharing nothing.
    auto jaccard_values = [](const std::map<uint64_t, uint64_t>& r, uint64_t zero) {
        std::vector<std::pair<double, uint64_t>> out;
        for (const auto& [key, n] : r)
            out.push_back({hgcommon::bo_jaccard_of_union(key >> 32, key & 0xFFFFFFFFu), n});
        if (zero) out.push_back({0.0, zero});
        return out;
    };
    auto values = [](const std::map<double, uint64_t>& m) {
        return std::vector<std::pair<double, uint64_t>>(m.begin(), m.end());
    };
    const uint64_t pairs = static_cast<uint64_t>(k) * (k - (k ? 1 : 0)) / 2;
    put_summary(rec, "StateOverlap", jaccard_values(ratios, pairs - shared_pairs), 0.01, false);
    put_summary(rec, "StateCosineSimilarity", values(cosine), 0.01, false);
    put_summary(rec, "StateMutualInformation", values(information), 0.01, false);
    if (w.initial) {
        // U is the step's vertices and S0's; each state once against S0.
        const std::vector<uint32_t>& initial = *w.initial;
        uint64_t u = universe;
        for (uint32_t v : initial) u += w.holders.count(v) ? 0 : 1;
        std::map<double, uint64_t> mi;
        for (const auto& v : w.sets) {
            uint64_t both = 0;
            for (uint32_t x : v) both += std::binary_search(initial.begin(), initial.end(), x);
            ++mi[hgcommon::bo_mutual_information(both, v.size(), initial.size(), u)];
        }
        put_summary(rec, "InitialStateMutualInformation", values(mi), 0.01, false);
    }
    std::map<uint64_t, uint64_t> by_k;
    for (const auto& [v, hs] : w.holders) ++by_k[hs.size()];
    put_multiplicity(rec, "VertexSharpness", "BranchEntropy", by_k);
    // Edge numbers are dense over the run: one count per number.
    uint32_t edge_end = 0;
    for (const auto& e : w.edge_sets)
        if (!e.empty()) edge_end = std::max(edge_end, e.back() + 1);
    std::vector<uint32_t> edge_holders(edge_end, 0);
    for (const auto& e : w.edge_sets)
        for (uint32_t x : e) ++edge_holders[x];
    by_k.clear();
    for (uint32_t k_e : edge_holders)
        if (k_e) ++by_k[k_e];
    put_multiplicity(rec, "EdgeSharpness", "EdgeBranchEntropy", by_k);
    if (graph) {
        wxf::WXFValueAssociation per;
        for (const auto& [d, n] : at_distance) {
            const auto it = by_distance.find(d);
            uint64_t shared = 0;
            if (it != by_distance.end())
                for (const auto& [key, m] : it->second) shared += m;
            static const std::map<uint64_t, uint64_t> none;
            per.push_back({wxf::WXFValue(static_cast<int64_t>(d)),
                           summary_value(summarise(jaccard_values(it == by_distance.end()
                                                                      ? none
                                                                      : it->second,
                                                                  n - shared),
                                                   0.01),
                                         false)});
        }
        rec.push_back({wxf::WXFValue("OverlapByBranchialDistance"), wxf::WXFValue(per)});
    }
    return rec;
}

std::vector<uint32_t> initial_vertices(const BranchialStep& step0) {
    std::vector<uint32_t> initial;
    for (const auto& v : step0.vertex_sets) initial.insert(initial.end(), v.begin(), v.end());
    std::sort(initial.begin(), initial.end());
    initial.erase(std::unique(initial.begin(), initial.end()), initial.end());
    return initial;
}

void events_from_multiplicities(
    const std::vector<StepPoint>& points,
    const std::unordered_map<uint64_t, std::map<int64_t, uint64_t>>& matches_by_rule,
    bool by_depth, uint32_t steps, std::map<uint32_t, uint64_t>& events,
    std::map<uint32_t, std::map<int64_t, uint64_t>>& rule_counts) {
    for (const auto& p : points) {
        if (p.step >= steps) continue;
        auto it = matches_by_rule.find(hgcommon::qc_expansion_key(p.class_hash, p.step, by_depth));
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

void add_class(StepInput& in, uint64_t hash, uint64_t weight,
               const hgcommon::StateInvariantRecord* record) {
    if (!weight) return;
    StepClass& c = in.classes[hash];
    c.weight = hgcommon::qm_sat_add(c.weight, weight);
    if (!c.record) c.record = record;
}

wxf::WXFValueAssociation step_record(uint32_t step, const StepInput& in, bool weight_by_classes) {
    static const hgcommon::StateInvariantRecord kNone{};
    const auto& classes = in.classes;
    uint64_t raw = 0, max_mult = 0;
    long double raw_ld = 0;
    std::map<int64_t, uint64_t> mult_hist;
    for (const auto& [h, c] : classes) {
        raw = hgcommon::qm_sat_add(raw, c.weight);
        raw_ld += static_cast<long double>(c.weight);
        max_mult = std::max(max_mult, c.weight);
        ++mult_hist[static_cast<int64_t>(c.weight)];
    }
    long double entropy = 0;
    for (const auto& [h, c] : classes) {
        const long double p = static_cast<long double>(c.weight) / raw_ld;
        entropy -= p * std::log2(p);
    }
    const size_t nclasses = classes.size();

    std::vector<std::pair<double, uint64_t>> vertex_count, edge_count, max_degree, mean_degree,
        two_section, components, cycle_rank, inc_cycle_rank, inc_diameter, inc_mean_distance,
        largest_fraction;
    std::map<int64_t, uint64_t> arity_hist, degree_hist;
    std::map<std::vector<int64_t>, uint64_t> arity_sig_hist, degree_seq_hist;
    std::vector<std::pair<double, uint64_t>> radius, eccentricity, hausdorff, ricci, ollivier,
        degree_entropy, local_entropy, mutual_information, fisher;
    std::vector<std::pair<double, uint64_t>> largest_dimension, local_dimension_max,
        local_dimension_sd, moran, degree_correlation;
    PooledDistribution pooled[hgcommon::SG_DISTS];
    for (uint32_t k = 0; k < hgcommon::SG_DISTS; ++k) pooled[k].which = k;
    std::map<uint32_t, std::vector<std::pair<double, uint64_t>>> ball_growth;
    for (const auto& [h, c] : classes) {
        const uint64_t w = weight_by_classes ? 1 : c.weight;
        const hgcommon::StateInvariantRecord& r = c.record ? *c.record : kNone;
        const hgcommon::StateInvariantValues& s = r.v;
        const hgcommon::SgGeometry& g = r.g;
        if (g.defined & hgcommon::SG_RADIUS) {
            radius.push_back({double(g.radius), w});
            eccentricity.push_back({g.mean_eccentricity, w});
        }
        if (g.defined & hgcommon::SG_HAUSDORFF) hausdorff.push_back({g.hausdorff_dimension, w});
        if (g.defined & hgcommon::SG_RICCI) ricci.push_back({g.ricci_scalar, w});
        if (g.defined & hgcommon::SG_OLLIVIER) ollivier.push_back({g.ollivier_ricci, w});
        if (g.defined & hgcommon::SG_DEGREE_ENTROPY) {
            degree_entropy.push_back({g.degree_entropy, w});
            local_entropy.push_back({g.local_entropy, w});
        }
        if (g.defined & hgcommon::SG_MUTUAL_INFORMATION)
            mutual_information.push_back({g.mutual_information, w});
        if (g.defined & hgcommon::SG_FISHER) fisher.push_back({g.fisher_information, w});
        if (g.defined & hgcommon::SG_LARGEST_DIMENSION) {
            largest_dimension.push_back({g.largest_dimension, w});
            local_dimension_max.push_back({g.local_dimension_max, w});
            local_dimension_sd.push_back({g.local_dimension_sd, w});
        }
        if (g.defined & hgcommon::SG_OLLIVIER_MORAN) moran.push_back({g.ollivier_moran_i, w});
        if (g.defined & hgcommon::SG_OLLIVIER_DEGREE)
            degree_correlation.push_back({g.ollivier_degree_correlation, w});
        for (uint32_t k = 0; k < r.num_distributions && k < hgcommon::SG_DISTS; ++k)
            pooled[k].add(r.distributions()[k], w);
        for (uint32_t b = 0; b < r.num_ball; ++b)
            ball_growth[b + 1].push_back({r.ball()[b], w});
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
        std::vector<int64_t> arities, degrees;
        arities.reserve(r.num_edges);
        degrees.reserve(r.num_vertices);
        r.for_each_arity([&](uint32_t a) { arities.push_back(a); });
        r.for_each_degree([&](uint32_t d) { degrees.push_back(d); });
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
    put("GraphRadius", radius, 1.0, true);
    put("MeanEccentricity", eccentricity, 0.01, false);
    put("WolframHausdorffDimension", hausdorff, 0.01, false);
    put("WolframRicciCurvatureScalar", ricci, 0.01, false);
    put("OllivierRicciCurvature", ollivier, 0.01, false);
    put("DegreeEntropy", degree_entropy, 0.01, false);
    put("LocalEntropy", local_entropy, 0.01, false);
    put("MutualInformation", mutual_information, 0.01, false);
    put("FisherInformation", fisher, 0.01, false);
    put("LargestComponentDimension", largest_dimension, 0.01, false);
    put("LocalDimensionMax", local_dimension_max, 0.01, false);
    put("LocalDimensionStandardDeviation", local_dimension_sd, 0.01, false);
    put("OllivierMoranI", moran, 0.01, false);
    put("OllivierDegreeCorrelation", degree_correlation, 0.01, false);
    wxf::WXFValueAssociation vertex_invs;
    vertex_invs.push_back({wxf::WXFValue("LocalDimension"),
                           pooled_value(pooled[hgcommon::SG_DIST_LOCAL_DIMENSION])});
    vertex_invs.push_back({wxf::WXFValue("OllivierRicciCurvature"),
                           pooled_value(pooled[hgcommon::SG_DIST_OLLIVIER])});
    vertex_invs.push_back({wxf::WXFValue("WolframRicciCurvatureScalar"),
                           pooled_value(pooled[hgcommon::SG_DIST_RICCI])});
    wxf::WXFValueAssociation balls;
    for (const auto& [r, v] : ball_growth)
        balls.push_back({wxf::WXFValue(static_cast<int64_t>(r)),
                         summary_value(summarise(v, 0.01), false)});

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
    rec.push_back({wxf::WXFValue("Events"), i64(in.events)});
    rec.push_back({wxf::WXFValue("RuleCounts"), count_association(in.rule_counts)});
    rec.push_back({wxf::WXFValue("Invariants"), wxf::WXFValue(invs)});
    rec.push_back({wxf::WXFValue("VertexInvariants"), wxf::WXFValue(vertex_invs)});
    rec.push_back({wxf::WXFValue("ArityHistogram"), count_association(arity_hist)});
    rec.push_back({wxf::WXFValue("AritySignatureHistogram"), count_association(arity_sig_hist)});
    rec.push_back({wxf::WXFValue("DegreeHistogram"), count_association(degree_hist)});
    rec.push_back({wxf::WXFValue("DegreeSequenceHistogram"), count_association(degree_seq_hist)});
    rec.push_back({wxf::WXFValue("BallGrowthDimension"), wxf::WXFValue(std::move(balls))});
    return rec;
}

// One submitted step: its input until the pooling job has read it, the pooled keys, and the
// branchial work until its last part has written the branchial keys.
struct StepStatisticsJobs::Step {
    uint32_t step = 0;
    bool populated = false;
    StepInput in;
    wxf::WXFValueAssociation pooled;
    std::unique_ptr<BranchialStepWork> work;
    wxf::WXFValueAssociation branchial;
    std::atomic<size_t> remaining{0};
};

StepStatisticsJobs::StepStatisticsJobs(bool weight_by_classes, uint32_t which, size_t workers,
                                       Submit submit, DeviceRows device_rows,
                                       uint32_t device_min_states)
    : weight_by_classes_(weight_by_classes), which_(which), workers_(workers ? workers : 1),
      submit_(std::move(submit)), device_rows_(std::move(device_rows)),
      device_min_states_(device_min_states) {}

StepStatisticsJobs::~StepStatisticsJobs() = default;

StepStatisticsJobs::Step* StepStatisticsJobs::reserve(uint32_t step) {
    auto [it, fresh] = steps_.emplace(step, nullptr);
    if (!fresh) return nullptr;
    it->second = std::make_unique<Step>();
    it->second->step = step;
    return it->second.get();
}

void StepStatisticsJobs::start(Step* s, StepInput in) {
    if ((which_ & kBranchialOverlap) && s->step == 0) {
        initial_ = initial_vertices(in.branchial);
        have_initial_ = true;
    }
    // A step without a population has no record, and its branchial keys have nowhere to go.
    s->populated = !in.classes.empty();
    if (!s->populated) return;
    BranchialStep branchial = std::move(in.branchial);
    s->in = std::move(in);
    submit_([this, s] {
        s->pooled = step_record(s->step, s->in, weight_by_classes_);
        s->in = StepInput{};
    });
    if (!which_) return;
    s->work = std::make_unique<BranchialStepWork>(std::move(branchial), which_,
                                                  have_initial_ ? &initial_ : nullptr);
    submit_([this, s] {
        s->work->prepare(workers_);
        const size_t n = s->work->parts();
        auto run_part = [s](size_t p) {
            s->work->rows(p);
            if (s->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                s->branchial = s->work->finish();
                s->work.reset();
            }
        };
        // No overlap rows, or rows counted off the host.
        if (n == 0 || (device_rows_ && s->work->rows_view().k >= device_min_states_ &&
                       device_rows_(*s->work))) {
            s->branchial = s->work->finish();
            s->work.reset();
            return;
        }
        s->remaining.store(n, std::memory_order_relaxed);
        for (size_t p = 1; p < n; ++p) submit_([run_part, p] { run_part(p); });
        run_part(0);
    });
}

namespace {
enum class HostStatsJob { Step };
}  // namespace

wxf::WXFValue step_statistics_on_host(std::map<uint32_t, StepInput> inputs,
                                      bool weight_by_classes, uint32_t which,
                                      StepStatisticsJobs::DeviceRows device_rows,
                                      uint32_t device_min_states) {
    job_system::JobSystem<HostStatsJob> system(std::thread::hardware_concurrency());
    system.start();
    StepStatisticsJobs jobs(
        weight_by_classes, which, system.get_num_workers(),
        [&system](std::function<void()> f) {
            system.submit(job_system::make_job<HostStatsJob>(std::move(f), HostStatsJob::Step));
        },
        std::move(device_rows), device_min_states);
    for (auto& [step, in] : inputs) jobs.submit(step, std::move(in));
    system.wait_for_completion();
    if (system.has_error())
        throw std::runtime_error(std::string("StepStatistics failed in a worker: ") +
                                 system.get_error_message());
    std::atomic_thread_fence(std::memory_order_acquire);
    return jobs.reply();
}

wxf::WXFValue StepStatisticsJobs::reply() {
    wxf::WXFValueList out;
    for (auto& [step, s] : steps_) {
        if (!s->populated) continue;
        wxf::WXFValueAssociation rec = std::move(s->pooled);
        for (auto& kv : s->branchial) rec.push_back(std::move(kv));
        out.push_back(wxf::WXFValue(std::move(rec)));
    }
    return wxf::WXFValue(std::move(out));
}

}  // namespace stats

}  // namespace HG_NAMESPACE
