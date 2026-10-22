#pragma once
#include "hgcommon/namespace.hpp"
//
// Shared GraphData marshaller for HGEvolve's *Graph properties.
//
// The Wolfram-side builds every graph property (StatesGraph, CausalGraph,
// BranchialGraph, Evolution*Graph and their Structure variants) from the
// "GraphData" association this produces. Both the CPU FFI and the GPU backend
// marshal their results through build_graph_data(), so CPU and GPU emit
// byte-identical GraphData for every property and every canonicalization mode.
//
// build_graph_data() is a template over a Source that exposes the evolved
// multiway as effective (canonicalization-collapsed) ids plus the per-vertex
// tooltip data. The CPU adapter reads the engine; the GPU adapter reads the
// device result. The Source concept:
//
//   uint32_t num_states() const;                 // scan bound for raw state ids
//
//   THE SCAN BOUNDS BELOW MUST BE PUBLICATION COUNTS, NOT CLAIM COUNTS. Ids are handed out by an
//   atomic increment before the object exists -- that is what makes them cheap, and why they are
//   not repeatable across runs -- so an id can be claimed and never emplaced, leaving a claim
//   counter permanently above what exists. This marshaller scans [0, bound) and calls
//   state_valid/is_valid_event, and on the host those dereference; handing it a claim counter
//   makes it reach an index holding no element, which throws out of the marshaller and returns
//   the caller an error instead of a graph. An adapter supplies the bound, so an adapter that
//   supplies the wrong one breaks a path that looks correct here.
//   bool     state_valid(uint32_t sid) const;
//   int64_t  effective_state_id(uint32_t sid) const;   // canonical/content/raw per mode
//   uint32_t state_step(uint32_t sid) const;
//   // A styled graph's state payload (GraphStateFields) and the state's edges, each as its
//   // first number and its vertices. Called from several threads at once.
//   void state_fields(uint32_t sid, GraphStateFields& f) const;
//   void state_edges(uint32_t sid,
//                    std::vector<std::pair<int64_t, std::vector<uint32_t>>>& out) const;
//
//   uint32_t num_raw_events() const;             // scan bound for raw event ids
//   bool     is_valid_event(uint32_t eid) const; // false for genesis unless shown
//   int64_t  effective_event_id(uint32_t eid) const;
//   uint32_t event_input_state(uint32_t eid) const;
//   uint32_t event_output_state(uint32_t eid) const;
//   // A styled graph's event payload (GraphEventFields); false when there is none, which
//   // writes the empty association. Called from several threads at once.
//   bool event_fields(uint32_t eid, GraphEventFields& f) const;
//
//   // (producer, consumer) raw event-id pairs, genesis already filtered, with
//   // per-shared-hyperedge multiplicity preserved (edge_deduplication uses it).
//   std::vector<std::pair<uint32_t, uint32_t>> causal_event_pairs() const;
//   // (event1, event2) raw branchial event-id pairs, genesis already filtered.
//   std::vector<std::pair<uint32_t, uint32_t>> branchial_event_pairs() const;

//
// DELTA DELIVERY. build_graph_data takes an optional DeliveryCursor: with one, it emits only the
// vertices and edges that session has not already been sent, and records what it emits. Without
// one -- every `Evolve`, and any session asking for a full delivery -- it emits everything.
// There is ONE walk either way: a cursor that has never seen a property answers "not sent" to
// every question, so the full delivery is the delta against an empty record rather than a second
// code path that agrees until one of them is edited.
//
// WHY IT MATTERS HERE: a Step re-serialised the whole accumulated graph, and
// tools/dev/session_step_cost.wls measured that as ~100% of a Step's wall time (24.8 MB and
// 1.9 s at depth 7, against a Query that explores nothing and costs the same).

#include "delivery_cursor.hpp"
#include "hgcommon/core.hpp"
#include "hgcommon/explore_depth_core.hpp"
#include "state_statistics.hpp"
#include "wxf.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <algorithm>
#include <array>
#include <string_view>

namespace HG_NAMESPACE {
namespace marshal {

// The CanonicalHash a reply reports for a state, on both devices: EMPTY_STATE_CANONICAL_HASH for
// the empty state in every mode, otherwise `hash`, the state's IR canonical hash.
inline uint64_t reported_state_hash(bool empty, uint64_t hash) {
    return empty ? hgcommon::EMPTY_STATE_CANONICAL_HASH : hash;
}

// The Step a reply reports for a state, on both devices: outside Full states the state's own
// step. Under Full the class's: its explore depth under quotient exploration (the shortest depth
// from an initial state; hgcommon::kExploreNoDepth when not known), otherwise the least step of
// any raw state of the class (UINT32_MAX when not known), otherwise the state's own step.
inline uint32_t reported_state_step(bool full, uint32_t class_explore_depth,
                                    uint32_t class_min_step, uint32_t own_step) {
    if (!full) return own_step;
    if (class_explore_depth != hgcommon::kExploreNoDepth) return class_explore_depth;
    return class_min_step != UINT32_MAX ? class_min_step : own_step;
}

// The reply for a verb that returns a session handle and no evolution.
//
// ONE BODY, BOTH DEVICES, for the same reason build_graph_data() is one body: a caller cannot
// tell which device answered and must not be able to. Two copies of this were byte-identical
// and separately named, so nothing compared them and either could have drifted alone.
std::vector<uint8_t> session_ack(uint64_t handle);

// Edge-kind tags for the delivery cursor's keys. They distinguish edges that share endpoints
// but not meaning -- a causal edge and a branchial edge between the same two events are two
// edges, and a record that conflated them would drop the second from every later delta.
inline constexpr uint32_t kEdgeDirected       = 1;   // states graph
inline constexpr uint32_t kEdgeCausal         = 2;
inline constexpr uint32_t kEdgeBranchialState = 3;   // branchial graph, state endpoints
inline constexpr uint32_t kEdgeBranchialEvent = 4;   // evolution graph, event endpoints
inline constexpr uint32_t kEdgeStateEvent     = 5;
inline constexpr uint32_t kEdgeEventState     = 6;

// An event vertex's payload never changes after it is minted, so it has one revision forever.
inline constexpr uint32_t kEventRevision = 0;

// The evolution graph tags states and events separately but the cursor records one id space per
// property, so state ids are biased out of the event range. Far above any event count and
// exactly representable, so the sum is the id it looks like rather than a rounded one.
inline constexpr int64_t kStateVertexBias = 1ll << 40;

struct GraphOptions {
    bool edge_deduplication = true;   // one causal edge per (producer, consumer) pair
    int  branchial_step = 0;          // 0 = all; >0 = 1-based step; <0 = from end (-1 = final)
    int  steps = 0;                   // total evolution steps (for the from-end branchial_step)
};

// Resolve branchial_step into an absolute target step (0 means "no filter").
uint32_t branchial_target_step(int branchial_step, int steps, bool& filter_by_step);

// THE STATE-ENDPOINT PROJECTION OF THE BRANCHIAL RELATION, which "BranchialStateEdges" and
// "BranchialStateEdgesAllSiblings" deliver as raw lists rather than as a GraphData.
//
// Two rules, not one, and they differ in more than their source: the pair form takes the
// branchial relation the engine computed and reads each event's output state, while the sibling
// form ignores overlap entirely and pairs every two events leaving the same input state. They
// also filter by step differently -- the pair form tests the FIRST event's output state only,
// the sibling form tests both -- which is behaviour the reference defines, so it is preserved
// exactly rather than tidied into agreement.
//
// Neither multiset is deduplicated: multiplicity is meaning here, the same as for the branchial
// event edges, and the reference preserves it.
//
// The TRAVERSAL is left to the caller and only the DECISION lives here. Which order an engine
// visits its events in is a storage question -- a per-state list on the host, an id-ordered
// array on the device -- and forcing one on both would change the host's output for no reason.
struct BranchialStateEdgeSet {
    std::vector<std::pair<int64_t, int64_t>> edges;   // (From, To), multiplicity preserved
    std::vector<int64_t> vertices;                    // unique endpoints, ascending
};

// From branchial EVENT pairs. `pairs` is (event1, event2) with genesis already filtered, which
// is exactly what Source::branchial_event_pairs() returns on both engines.
template <class OutputStateOf, class CanonicalOf, class StepOf>
inline BranchialStateEdgeSet branchial_state_edges_from_pairs(
        const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
        OutputStateOf&& output_state_of, CanonicalOf&& canonical_of, StepOf&& step_of,
        const GraphOptions& opts) {
    BranchialStateEdgeSet out;
    bool filter_by_step = false;
    const uint32_t target_step = branchial_target_step(opts.branchial_step, opts.steps,
                                                       filter_by_step);
    std::set<int64_t> unique_states;
    for (const auto& pr : pairs) {
        const uint32_t out1 = output_state_of(pr.first);
        // The FIRST event's output state alone decides the step filter. Both events of a
        // branchial pair were produced at the same step, so testing the second adds nothing.
        if (filter_by_step && step_of(out1) != target_step) continue;
        const int64_t state1 = canonical_of(out1);
        const int64_t state2 = canonical_of(output_state_of(pr.second));
        unique_states.insert(state1);
        unique_states.insert(state2);
        out.edges.emplace_back(state1, state2);
    }
    out.vertices.assign(unique_states.begin(), unique_states.end());
    return out;
}

// Numbers for lists of vertex ids, by first appearance: an open-addressed table of numbers over a
// pool holding each list as its length and then its ids.
class ListNumbers {
public:
    uint32_t number(const uint32_t* list, uint32_t length) {
        if ((offset_of_.size() + 1) * 2 > slots_.size()) grow();
        size_t i = hash(list, length) & (slots_.size() - 1);
        for (; slots_[i]; i = (i + 1) & (slots_.size() - 1)) {
            const uint32_t at = offset_of_[slots_[i] - 1];
            if (pool_[at] == length && std::equal(list, list + length, pool_.begin() + at + 1))
                return slots_[i] - 1;
        }
        const uint32_t n = static_cast<uint32_t>(offset_of_.size());
        offset_of_.push_back(static_cast<uint32_t>(pool_.size()));
        pool_.push_back(length);
        pool_.insert(pool_.end(), list, list + length);
        slots_[i] = n + 1;
        return n;
    }

private:
    static size_t hash(const uint32_t* list, uint32_t length) {
        uint64_t h = 0x9E3779B97F4A7C15ull ^ length;
        for (uint32_t k = 0; k < length; ++k) h = (h ^ list[k]) * 0x100000001B3ull;
        return static_cast<size_t>(h ^ (h >> 29));
    }
    void grow() {
        slots_.assign(std::max<size_t>(64, slots_.size() * 2), 0);
        for (uint32_t n = 0; n < offset_of_.size(); ++n) {
            const uint32_t at = offset_of_[n];
            size_t i = hash(pool_.data() + at + 1, pool_[at]) & (slots_.size() - 1);
            while (slots_[i]) i = (i + 1) & (slots_.size() - 1);
            slots_[i] = n + 1;
        }
    }
    std::vector<uint32_t> pool_, offset_of_, slots_;
};

// Runs body(i) for every i < n, on any threads and in any order, and returns after every call
// has returned.
using ParallelFor = std::function<void(size_t n, const std::function<void(size_t)>& body)>;

inline void serial_for(size_t n, const std::function<void(size_t)>& body) {
    for (size_t i = 0; i < n; ++i) body(i);
}

// "StepStatisticsBranchial": one state of a step (stats::BranchialStepWork), for both engines,
// appended to `b`: its id `effective`, and with kBranchialContents its vertex ids and edge
// numbers from `contents_of(sid, out)`, which appends each of the state's edges as its arity and
// then its vertex ids. An edge is its ordered list of vertex ids, numbered by `edge_number` by
// first appearance: the edge metrics read which states hold the same list, so two members of an
// Automatic class, which hold the same lists under different edge ids, give one edge set. The
// metrics read only which states hold the same number, so the numbering may be per step or per
// run. `flat` is the caller's scratch.
template <class ContentsOf>
void branchial_step_state(uint32_t which, uint32_t sid, int64_t effective,
                          ContentsOf&& contents_of, ListNumbers& edge_number,
                          std::vector<uint32_t>& flat, stats::BranchialStep& b) {
    b.nodes.push_back(effective);
    if (!(which & stats::kBranchialContents)) return;
    flat.clear();
    contents_of(sid, flat);
    std::vector<uint32_t> vs, es;
    vs.reserve(flat.size());
    for (size_t i = 0; i < flat.size(); i += 1 + flat[i]) {
        vs.insert(vs.end(), flat.begin() + static_cast<std::ptrdiff_t>(i + 1),
                  flat.begin() + static_cast<std::ptrdiff_t>(i + 1 + flat[i]));
        es.push_back(edge_number.number(flat.data() + i + 1, flat[i]));
    }
    b.vertex_sets.push_back(std::move(vs));
    b.edge_sets.push_back(std::move(es));
}

// The branchial state pairs of branchial event pairs (genesis filtered), projected through
// `output_state_of` by branchial_state_edges_from_pairs with every step's pairs kept:
// f(step, a, b) for each, under the step of its first event's output state.
template <class Effective, class StepOf, class OutputStateOf, class F>
void branchial_step_pairs(const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                          OutputStateOf&& output_state_of, Effective&& effective,
                          StepOf&& step_of, int steps, F&& f) {
    // Each endpoint carries its state's step above bit 40, so one unfiltered projection gives
    // every step's pairs.
    constexpr int kStepShift = 40;
    GraphOptions all_steps;
    all_steps.branchial_step = 0;
    all_steps.steps = steps;
    const auto set = branchial_state_edges_from_pairs(
        pairs, output_state_of,
        [&](uint32_t sid) {
            return (static_cast<int64_t>(step_of(sid)) << kStepShift) | effective(sid);
        },
        step_of, all_steps);
    const int64_t mask = (int64_t{1} << kStepShift) - 1;
    for (const auto& [a, b] : set.edges) f(static_cast<uint32_t>(a >> kStepShift), a & mask, b & mask);
}

// Every step's BranchialStep: the states are the ids below `num_states` that `valid` admits,
// each at `step_of(sid)` (branchial_step_state), and with kBranchialGraph the pairs of `pairs`
// (branchial_step_pairs). The states are grouped by step on the calling thread; `par` then runs
// one job per step, each numbering its own step's edges, and one job for the pairs. The four
// functions are called from those jobs.
template <class Valid, class Effective, class StepOf, class ContentsOf, class OutputStateOf>
std::map<uint32_t, stats::BranchialStep> branchial_steps(
        uint32_t which, uint32_t num_states, Valid&& valid, Effective&& effective,
        StepOf&& step_of, ContentsOf&& contents_of,
        const std::vector<std::pair<uint32_t, uint32_t>>& pairs, OutputStateOf&& output_state_of,
        int steps, const ParallelFor& par = serial_for) {
    std::map<uint32_t, std::vector<uint32_t>> by_step;
    for (uint32_t sid = 0; sid < num_states; ++sid)
        if (valid(sid)) by_step[step_of(sid)].push_back(sid);
    std::map<uint32_t, stats::BranchialStep> out;
    std::vector<std::pair<const std::vector<uint32_t>*, stats::BranchialStep*>> step_jobs;
    for (const auto& [step, sids] : by_step) step_jobs.emplace_back(&sids, &out[step]);
    const bool with_pairs = (which & stats::kBranchialGraph) != 0;
    std::vector<std::tuple<uint32_t, int64_t, int64_t>> step_pairs;
    par(step_jobs.size() + (with_pairs ? 1 : 0), [&](size_t i) {
        if (i == step_jobs.size()) {
            branchial_step_pairs(pairs, output_state_of, effective, step_of, steps,
                                 [&](uint32_t step, int64_t a, int64_t b) {
                                     step_pairs.emplace_back(step, a, b);
                                 });
            return;
        }
        ListNumbers edge_number;
        std::vector<uint32_t> flat;
        stats::BranchialStep& b = *step_jobs[i].second;
        for (uint32_t sid : *step_jobs[i].first)
            branchial_step_state(which, sid, effective(sid), contents_of, edge_number, flat, b);
    });
    for (const auto& [step, a, b] : step_pairs) out[step].pairs.emplace_back(a, b);
    return out;
}

// From SIBLING groups: every two events leaving one input state, whether or not they overlap.
// `for_each_group` calls its argument once per input state with that state's event ids, genesis
// already excluded -- the caller enumerates, because the order it enumerates in is its own.
template <class ForEachGroup, class OutputStateOf, class CanonicalOf, class StepOf>
inline BranchialStateEdgeSet branchial_state_edges_all_siblings(
        ForEachGroup&& for_each_group,
        OutputStateOf&& output_state_of, CanonicalOf&& canonical_of, StepOf&& step_of,
        const GraphOptions& opts) {
    BranchialStateEdgeSet out;
    bool filter_by_step = false;
    const uint32_t target_step = branchial_target_step(opts.branchial_step, opts.steps,
                                                       filter_by_step);
    std::set<int64_t> unique_states;
    for_each_group([&](const std::vector<uint32_t>& events) {
        for (size_t i = 0; i < events.size(); ++i) {
            const uint32_t out1 = output_state_of(events[i]);
            if (filter_by_step && step_of(out1) != target_step) continue;
            const int64_t state1 = canonical_of(out1);
            for (size_t j = i + 1; j < events.size(); ++j) {
                const uint32_t out2 = output_state_of(events[j]);
                // Both endpoints are tested here, unlike the pair form: siblings of one input
                // state are not required to have been produced at the same step.
                if (filter_by_step && step_of(out2) != target_step) continue;
                const int64_t state2 = canonical_of(out2);
                unique_states.insert(state1);
                unique_states.insert(state2);
                out.edges.emplace_back(state1, state2);
            }
        }
    });
    out.vertices.assign(unique_states.begin(), unique_states.end());
    return out;
}

// The two keys both forms deliver under, so the reply shape is spelled once.
void push_branchial_state_edges(wxf::WXFValueAssociation& result,
                                const BranchialStateEdgeSet& set);

// THE "States" AND "Events" RECORDS, one schema for both engines, written into a wxf::Writer.
// A sink writes one association: begin(n), then n fields.
class StreamRecordSink {
public:
    explicit StreamRecordSink(wxf::Writer& w) : w_(w) {}
    void begin(std::size_t fields) {
        w_.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w_.write_varint(fields);
    }
    void i64(const char* key, int64_t v) { key_(key); w_.write(v); }
    void i64_list(const char* key, const std::vector<int64_t>& v) {
        key_(key);
        w_.write_function("List", v.size());
        for (int64_t x : v) w_.write(x);
    }
    void edges(const char* key,
               const std::vector<std::pair<int64_t, std::vector<uint32_t>>>& e) {
        key_(key);
        w_.write_function("List", e.size());
        for (const auto& [id, verts] : e) {
            w_.write_function("List", verts.size() + 1);
            w_.write(id);
            for (uint32_t v : verts) w_.write(static_cast<int64_t>(v));
        }
    }
    void vertex_reals(const char* key, const std::vector<std::pair<uint32_t, double>>& kv) {
        key_(key);
        w_.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        w_.write_varint(kv.size());
        for (const auto& [v, x] : kv) {
            w_.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            w_.write(static_cast<int64_t>(v));
            w_.write(x);
        }
    }
private:
    void key_(const char* k) {
        w_.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
        w_.write(std::string(k));
    }
    wxf::Writer& w_;
};

struct StateRecordIds {
    int64_t id, canonical_id, content_id, step;
    bool    with_hash;
    int64_t hash;
    bool    with_curvature = false;   // "IncludeVertexCurvature"
    // The curvature already computed for the record's edges (the GPU engine computes it on the
    // device); null computes it here.
    const stats::VertexCurvature* curvature = nullptr;
};

// A "States" record. `edges` is the state's edges as (edge id, vertices). Under Full the record
// carries the state's IR canonical form, its edges numbered from 0 in canonical order; otherwise
// the edges as stored, under the edge ids an event's ConsumedEdges and ProducedEdges name.
// IsInitial is the 0/1 integer a WXFValue(bool) stores. With r.with_curvature the record has
// "VertexCurvature" (stats::vertex_curvature of the edges the record lists, or r.curvature).
void state_record_edges(bool full,
                        std::vector<std::pair<int64_t, std::vector<uint32_t>>>& edges);
template <class Sink>
void write_state_record(Sink& s, const StateRecordIds& r, bool full,
                        std::vector<std::pair<int64_t, std::vector<uint32_t>>> edges) {
    state_record_edges(full, edges);
    s.begin((r.with_hash ? 7u : 6u) + (r.with_curvature ? 1u : 0u));
    s.i64("Id", r.id);
    s.i64("CanonicalId", r.canonical_id);
    s.i64("ContentStateId", r.content_id);
    s.i64("Step", r.step);
    s.edges("Edges", edges);
    s.i64("IsInitial", r.step == 0 ? 1 : 0);
    if (r.with_hash) s.i64("CanonicalHash", r.hash);
    if (r.with_curvature)
        s.vertex_reals("VertexCurvature",
                       r.curvature ? *r.curvature : stats::vertex_curvature(edges));
}

// The fields of a state vertex's payload in a styled graph. The payload is <|Id, CanonicalId,
// Step, Edges, IsInitial|>, with IsInitial 1 at step 0.
struct GraphStateFields {
    int64_t id = 0, canonical_id = 0, step = 0;
};

// The fields of an event's payload in a styled graph. A materialised event's payload is <|Id,
// CanonicalId, RuleIndex, InputState, OutputState, ConsumedEdges, ProducedEdges,
// InputStateEdges, OutputStateEdges|>, the last two the edges of states `input_raw` and
// `output_raw`. A reconstructed application's is <|Id, RuleIndex, InputState, OutputState|>: the
// reconstruction materialises no edge lists.
struct GraphEventFields {
    bool reconstructed = false;
    int64_t id = 0, canonical_id = 0, rule = 0, input = 0, output = 0;
    std::vector<int64_t> consumed, produced;
    uint32_t input_raw = 0, output_raw = 0;
};

// Sorts `v` ascending. Above kParallelSortMin elements, kParallelSortRuns runs are sorted by
// `par` and merged pairwise in rounds, each round's merges also by `par`. The run count does not
// depend on the threads, and the element type's order is total, so the result is std::sort's.
constexpr size_t kParallelSortMin = size_t{1} << 16;
constexpr size_t kParallelSortRuns = 16;
template <class T>
void parallel_sort(std::vector<T>& v, const ParallelFor& par) {
    const size_t n = v.size();
    if (n < kParallelSortMin) {
        std::sort(v.begin(), v.end());
        return;
    }
    std::vector<size_t> bound(kParallelSortRuns + 1);
    for (size_t r = 0; r <= kParallelSortRuns; ++r) bound[r] = n * r / kParallelSortRuns;
    par(kParallelSortRuns, [&](size_t r) {
        std::sort(v.begin() + static_cast<std::ptrdiff_t>(bound[r]),
                  v.begin() + static_cast<std::ptrdiff_t>(bound[r + 1]));
    });
    std::vector<T> buf(n);
    std::vector<T>* from = &v;
    std::vector<T>* to = &buf;
    for (size_t width = 1; width < kParallelSortRuns; width *= 2) {
        const size_t merges = (kParallelSortRuns + 2 * width - 1) / (2 * width);
        par(merges, [&](size_t m) {
            const size_t lo = bound[std::min(kParallelSortRuns, 2 * width * m)];
            const size_t mid = bound[std::min(kParallelSortRuns, 2 * width * m + width)];
            const size_t hi = bound[std::min(kParallelSortRuns, 2 * width * (m + 1))];
            std::merge(from->begin() + static_cast<std::ptrdiff_t>(lo),
                       from->begin() + static_cast<std::ptrdiff_t>(mid),
                       from->begin() + static_cast<std::ptrdiff_t>(mid),
                       from->begin() + static_cast<std::ptrdiff_t>(hi),
                       to->begin() + static_cast<std::ptrdiff_t>(lo));
        });
        std::swap(from, to);
    }
    if (from != &v) v.swap(*from);
}

// Appends item(w, i) for every i < n to `out` in order of i, the items written by `par` in
// chunks of 256, each chunk into its own buffer.
inline void write_items(wxf::Writer& out, size_t n, const ParallelFor& par,
                        const std::function<void(wxf::Writer&, size_t)>& item) {
    constexpr size_t kItems = 256;
    std::vector<wxf::Writer> chunk((n + kItems - 1) / kItems);
    par(chunk.size(), [&](size_t c) {
        const size_t end = std::min(n, (c + 1) * kItems);
        for (size_t i = c * kItems; i < end; ++i) item(chunk[c], i);
    });
    for (const wxf::Writer& w : chunk) out.append(w.data());
}

// A list of records, row i as <|keys[0] -> rows[i][0], ...|>, written by `par` (write_items).
template <size_t N>
wxf::WXFValue record_list(const std::vector<std::array<int64_t, N>>& rows,
                          const std::array<const char*, N>& keys, const ParallelFor& par) {
    wxf::Writer w;
    w.write_function("List", rows.size());
    write_items(w, rows.size(), par, [&](wxf::Writer& out, size_t i) {
        out.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        out.write_varint(N);
        for (size_t k = 0; k < N; ++k) {
            out.write_key(keys[k]);
            out.write(rows[i][k]);
        }
    });
    return wxf::WXFValue(wxf::Encoded{w.release_data()});
}

// A job system of `threads` workers (0 is hgcommon::default_worker_count), for a reply whose run
// had no host engine (the GPU backend). Started on construction, stopped on destruction.
class HostJobs {
public:
    explicit HostJobs(unsigned threads = 0);
    ~HostJobs();
    HostJobs(const HostJobs&) = delete;
    HostJobs& operator=(const HostJobs&) = delete;
    size_t workers() const;
    void submit(std::function<void()> fn);
    // Returns after every submitted job has run; throws when a job threw.
    void wait();
    // Runs on the workers and the calling thread, which runs body(0).
    ParallelFor parallel_for();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct EventRecordIds {
    int64_t id, canonical_id, rule, input, output, canonical_input, canonical_output;
};

// An "Events" record: seven fields, and the two edge lists unless `minimal` (EventsMinimal).
template <class Sink>
void write_event_record(Sink& s, const EventRecordIds& r, bool minimal,
                        const std::vector<int64_t>& consumed, const std::vector<int64_t>& produced) {
    s.begin(minimal ? 7u : 9u);
    s.i64("Id", r.id);
    s.i64("CanonicalId", r.canonical_id);
    s.i64("RuleIndex", r.rule);
    s.i64("InputState", r.input);
    s.i64("OutputState", r.output);
    s.i64("CanonicalInputState", r.canonical_input);
    s.i64("CanonicalOutputState", r.canonical_output);
    if (minimal) return;
    s.i64_list("ConsumedEdges", consumed);
    s.i64_list("ProducedEdges", produced);
}

// Which relations a graph property is built FROM, decided by its name.
//
// Shared with whoever must decide what the run records: a property whose relation was never
// recorded is served as an empty graph, and a second copy of this test is how the two come to
// disagree about one name.
struct GraphPropertyNeeds {
    bool causal = false;
    bool branchial = false;
    // The events themselves: a states graph has one edge per event, and an evolution graph has
    // a vertex per event. Under quotient exploration they are the reconstruction's.
    bool events = false;
};

GraphPropertyNeeds graph_property_needs(const std::string& graph_property);

// The union over a list of properties.
GraphPropertyNeeds graph_property_needs(const std::vector<std::string>& properties);

// THE "ContentStateId" RULE, for both engines: among the states a reply lists, the lowest id of
// those with the same content hash. `ids` and `hashes` are parallel.
std::unordered_map<uint64_t, int64_t> lowest_id_by_content(const std::vector<int64_t>& ids,
                                                           const std::vector<uint64_t>& hashes);

// hgcommon::ContentHasher over a state's edges taken in edge-id order: the content hash the host
// computes from its stored state (Hypergraph::get_state_content_hash).
uint64_t content_hash_of(std::vector<std::pair<int64_t, std::vector<uint32_t>>> edges);

// One entry of the reply's "Warnings" list, for both engines. "Partial" is 1 for a result the
// engine cut short, which the kernel reports as HGEvolve::overflow, and 0 for every other kind,
// reported as HGEvolve::warn.
wxf::WXFValue warning_record(const std::string& kind, int64_t count, const std::string& context,
                             bool partial);

// `s` with every byte that is not part of a well-formed UTF-8 sequence written as the four
// characters \xNN. Warning contexts and error messages quote names from the request (an
// option key, an Op), which arrive as raw bytes; the WL side decodes the reply as UTF-8.
std::string valid_utf8(const std::string& s);

// Whether a request reads the raw events and branchial pairs only through NumEvents and
// NumBranchialEdges. Under quotient exploration those then come from class multiplicities
// (RecordSet::raw_counts_only). Both engines' jobs carry these fields under these names.
template <typename Job>
bool reads_raw_counts_only(const Job& job, const GraphPropertyNeeds& g) {
    return !(job.include_events || job.include_events_minimal || job.include_branchial_edges ||
             job.include_branchial_state_edges || job.include_branchial_state_edges_all_siblings ||
             g.causal || g.branchial || g.events);
}

// What a run records, from what the request reads, for both engines. The raw events are on for
// the event records and counts, the branchial state views (which read each event's endpoints)
// and graphs over events; progress reports read the relations. A session records everything,
// since a later query may read what its Open did not name.
template <typename Job>
hgcommon::RecordSet record_set_for(const Job& job, const GraphPropertyNeeds& g) {
    hgcommon::RecordSet r;
    // The "StepStatisticsBranchial" steps of a full-capture run read the events and their
    // branchial pairs; under quotient exploration they read the class expansions.
    const bool step_branchial = job.include_step_statistics &&
                                job.step_statistics_branchial != 0 &&
                                !job.explore_from_canonical_states_only;
    r.causal = job.include_causal_edges || job.include_num_causal_edges || g.causal ||
               job.show_progress;
    r.branchial = job.include_branchial_edges || job.include_num_branchial_edges ||
                  job.include_branchial_state_edges || g.branchial || job.show_progress ||
                  step_branchial;
    r.state_events = job.include_branchial_state_edges_all_siblings;
    r.raw_events = job.include_events || job.include_events_minimal || job.include_num_events ||
                   job.include_branchial_state_edges ||
                   job.include_branchial_state_edges_all_siblings || g.events || job.show_progress ||
                   step_branchial;
    r.raw_counts_only = reads_raw_counts_only(job, g);
    r.multiplicities = job.include_step_statistics;
    r.state_invariants = job.include_step_statistics;
    r.relation_lists = job.include_causal_edges || job.include_branchial_edges || g.causal ||
                       g.branchial;
    r.event_lists = job.include_events || job.include_events_minimal || g.events;
    // On the quotient route the explored events' own branchial pairs are read by the branchial
    // state view and by the "StepStatisticsBranchial" steps of a full-capture run (Full states
    // with Automatic events); the other branchial outputs read the replay.
    r.explored_branchial = job.include_branchial_state_edges ||
                           step_branchial;
    if (job.session_op == "Open")
        r.causal = r.branchial = r.state_events = r.raw_events = r.multiplicities =
            r.state_invariants = r.relation_lists = r.event_lists = r.explored_branchial = true;
    return r;
}

namespace graph_detail {

// A graph vertex or edge endpoint: the integer `id` (tag 0), or the list {"S", id} or {"E", id}.
struct End {
    char tag = 0;
    int64_t id = 0;
};

inline void write_end(wxf::Writer& w, const End& e) {
    if (e.tag) {
        w.write_function("List", 2);
        w.write_string(std::string_view(&e.tag, 1));
    }
    w.write(e.id);
}

enum class EdgeType : uint8_t { Directed, Causal, Branchial, StateEvent, EventState };

inline std::string_view edge_type_name(EdgeType t) {
    switch (t) {
        case EdgeType::Directed:   return "Directed";
        case EdgeType::Causal:     return "Causal";
        case EdgeType::Branchial:  return "Branchial";
        case EdgeType::StateEvent: return "StateEvent";
        case EdgeType::EventState: return "EventState";
    }
    return "";
}

// An edge's "Data": an event's payload; <|EventId -> a|>; <|ProducerEvent -> a, ConsumerEvent
// -> b|> with EdgeIndex -> c when `indexed`; <|State1 -> a, State2 -> b|>; <|Event1 -> a, Event2
// -> b|>.
enum class EdgeData : uint8_t { EventPayload, EventId, Causal, States, Events };

struct Edge {
    End from, to;
    EdgeType type = EdgeType::Directed;
    EdgeData data = EdgeData::EventId;
    bool indexed = false;
    int64_t a = 0, b = 0, c = 0;
    uint32_t payload = 0;   // EventPayload: the index of the event among the payload events
};

struct Vertex {
    End key;
    bool event = false;
    uint32_t raw = 0;
    uint32_t payload = 0;   // an event vertex of a styled graph: its index among the payload events
};

struct SentKey {
    int64_t from, to;
    uint32_t type, index;
    bool operator==(const SentKey& o) const {
        return from == o.from && to == o.to && type == o.type && index == o.index;
    }
};

struct SentKeyHash {
    size_t operator()(const SentKey& k) const {
        uint64_t h = static_cast<uint64_t>(k.from) * 0x9E3779B97F4A7C15ull;
        h ^= static_cast<uint64_t>(k.to) + 0x632BE59BD9B4E019ull + (h << 6) + (h >> 2);
        h ^= (static_cast<uint64_t>(k.type) << 32 | k.index) + 0x94D049BB133111EBull + (h << 6) +
             (h >> 2);
        return static_cast<size_t>(h ^ (h >> 31));
    }
};

// Items per job when a graph's vertices and edges are written in parallel.
inline constexpr size_t kChunk = 256;

inline size_t chunks(size_t n) { return (n + kChunk - 1) / kChunk; }

// The edge lists of the states a graph's payloads name, each written once as WXF bytes: a List
// of {first, v1, v2, ...} as Source::state_edges gives them.
class EdgeMemo {
public:
    // Writes the lists of `sids` not written before. Slots are assigned on the calling thread;
    // the lists are written by `par`.
    template <class Source>
    void add(const Source& src, std::vector<uint32_t> sids, const ParallelFor& par) {
        std::sort(sids.begin(), sids.end());
        sids.erase(std::unique(sids.begin(), sids.end()), sids.end());
        std::vector<uint32_t> fresh;
        for (uint32_t sid : sids)
            if (slot_of_.emplace(sid, static_cast<uint32_t>(bytes_.size() + fresh.size())).second)
                fresh.push_back(sid);
        if (fresh.empty()) return;
        const size_t base = bytes_.size();
        bytes_.resize(base + fresh.size());
        par(chunks(fresh.size()), [&](size_t c) {
            std::vector<std::pair<int64_t, std::vector<uint32_t>>> edges;
            const size_t end = std::min(fresh.size(), (c + 1) * kChunk);
            for (size_t i = c * kChunk; i < end; ++i) {
                edges.clear();
                src.state_edges(fresh[i], edges);
                wxf::Writer w;
                w.write_function("List", edges.size());
                for (const auto& [first, verts] : edges) {
                    w.write_function("List", verts.size() + 1);
                    w.write(first);
                    for (uint32_t v : verts) w.write(static_cast<int64_t>(v));
                }
                bytes_[base + i] = w.release_data();
            }
        });
    }
    const std::vector<uint8_t>& of(uint32_t sid) const { return bytes_[slot_of_.at(sid)]; }

private:
    std::unordered_map<uint32_t, uint32_t> slot_of_;
    std::vector<std::vector<uint8_t>> bytes_;
};

inline void write_association_header(wxf::Writer& w, size_t n) {
    w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
    w.write_varint(n);
}

inline void write_state_payload(wxf::Writer& w, const GraphStateFields& f,
                                const std::vector<uint8_t>& edges) {
    write_association_header(w, 5);
    w.write_key("Id");          w.write(f.id);
    w.write_key("CanonicalId"); w.write(f.canonical_id);
    w.write_key("Step");        w.write(f.step);
    w.write_key("Edges");       w.append(edges);
    w.write_key("IsInitial");   w.write(static_cast<int64_t>(f.step == 0));
}

inline void write_int_list(wxf::Writer& w, const std::vector<int64_t>& v) {
    w.write_function("List", v.size());
    for (int64_t x : v) w.write(x);
}

// An event's payload, or the empty association when the source has none for it.
inline void write_event_payload(wxf::Writer& w, const GraphEventFields& f, bool present,
                                const EdgeMemo& memo) {
    if (!present) { write_association_header(w, 0); return; }
    if (f.reconstructed) {
        write_association_header(w, 4);
        w.write_key("Id");          w.write(f.id);
        w.write_key("RuleIndex");   w.write(f.rule);
        w.write_key("InputState");  w.write(f.input);
        w.write_key("OutputState"); w.write(f.output);
        return;
    }
    write_association_header(w, 9);
    w.write_key("Id");               w.write(f.id);
    w.write_key("CanonicalId");      w.write(f.canonical_id);
    w.write_key("RuleIndex");        w.write(f.rule);
    w.write_key("InputState");       w.write(f.input);
    w.write_key("OutputState");      w.write(f.output);
    w.write_key("ConsumedEdges");    write_int_list(w, f.consumed);
    w.write_key("ProducedEdges");    write_int_list(w, f.produced);
    w.write_key("InputStateEdges");  w.append(memo.of(f.input_raw));
    w.write_key("OutputStateEdges"); w.append(memo.of(f.output_raw));
}

}  // namespace graph_detail

// Build the "GraphData" association: property name -> <|IsDelta, Vertices, Edges, VertexData|>,
// written as WXF bytes and returned as one wxf::Encoded value.
//
// Each property is built in three passes. The vertex and edge lists, and the cursor's record,
// are chosen on the calling thread. The payload fields and the payload states' edge lists are
// then computed by `par`, and the vertices and edges are written by `par` in chunks of
// graph_detail::kChunk, each chunk into its own buffer. The buffers are joined in chunk order, so
// the bytes do not depend on which thread wrote which chunk.
template <typename Source>
wxf::WXFValue build_graph_data(const Source& src,
                               const std::vector<std::string>& graph_properties,
                               const GraphOptions& opts,
                               ffi::DeliveryCursor* cursor = nullptr,
                               const ParallelFor& par = serial_for) {
    using namespace graph_detail;
    wxf::Writer out;
    write_association_header(out, graph_properties.size());
    EdgeMemo memo;

    for (const std::string& graph_property : graph_properties) {
        std::vector<Vertex> vertices;
        std::vector<Edge> edges;
        // The events whose payload a styled graph writes: event vertices and states-graph edges.
        std::vector<uint32_t> payload_events;

        const bool is_states    = graph_property.rfind("States", 0) == 0;
        const bool is_causal    = graph_property.rfind("Causal", 0) == 0;
        const bool is_branchial = graph_property.rfind("Branchial", 0) == 0;
        const bool is_evolution = graph_property.find("Evolution") != std::string::npos;
        const GraphPropertyNeeds needs = graph_property_needs(graph_property);
        const bool has_causal    = is_evolution && needs.causal;
        const bool has_branchial = is_evolution && needs.branchial;
        // Structure variants carry ids and steps and no hypergraph contents: a state carries Id
        // and Step, an event Id and its endpoint state ids (InputState marks an event vertex),
        // and a states-graph edge the event id its tooltip names.
        const bool is_structure = graph_property.find("Structure") != std::string::npos;
        // A branchial graph read at a step counted from the end (a negative BranchialStep, the
        // default for BranchialGraph) shows a later step after each Step, so it does not only
        // grow: it is delivered whole, with IsDelta 0, every time.
        ffi::DeliveryCursor* const cur =
            opts.branchial_step < 0 && (is_branchial || has_branchial) ? nullptr : cursor;
        // Read before the walk records anything: after it, every property has been delivered.
        const bool delivered_before =
            cur != nullptr && cur->delivered_before(graph_property);

        auto payload_of = [&](uint32_t raw_eid) {
            payload_events.push_back(raw_eid);
            return static_cast<uint32_t>(payload_events.size() - 1);
        };
        auto add_state_vertex = [&](End key, uint32_t raw) {
            vertices.push_back(Vertex{key, false, raw, 0});
        };
        auto add_event_vertex = [&](End key, uint32_t raw) {
            vertices.push_back(Vertex{key, true, raw, is_structure ? 0u : payload_of(raw)});
        };

        // A vertex is emitted when the cursor has not seen it at this REVISION. A state's
        // revision is its step, which try_lower_explore_depth may lower after the state was
        // delivered -- the one field of a delivered vertex that changes. An event's payload is
        // immutable, so its revision is constant.
        auto send_vertex = [&](int64_t id, uint32_t revision) {
            return cur == nullptr || cur->take_vertex(graph_property, id, revision);
        };
        // One edge per (from, to, type, index): its two endpoints, its type, and -- for the
        // un-deduplicated causal case, where N edges share a pair -- its index among them. A
        // one-shot build and a session's cursor both record the exact tuple, so raw events that
        // share an event identity share one edge on either path.
        std::unordered_set<SentKey, SentKeyHash> sent_edges;
        auto send_edge = [&](int64_t from, int64_t to, uint32_t type_tag, uint32_t index) {
            if (cur != nullptr) return cur->take_edge(graph_property, from, to, type_tag, index);
            return sent_edges.insert(SentKey{from, to, type_tag, index}).second;
        };
        // An edge whose tuple is distinct from every other this property emits by construction
        // (the causal and branchial edges below: each (from, to) group indexes its own edges)
        // is recorded only by a cursor.
        auto send_unique_edge = [&](int64_t from, int64_t to, uint32_t type_tag, uint32_t index) {
            return cur == nullptr || cur->take_edge(graph_property, from, to, type_tag, index);
        };
        // The source's id bounds, read once.
        const uint32_t num_states = src.num_states();
        const uint32_t num_events = src.num_raw_events();

        // The lowest raw id of each effective id, in effective-id order.
        auto lowest_by_effective = [&](std::vector<std::pair<int64_t, uint32_t>>& eff_raw) {
            parallel_sort(eff_raw, par);
            eff_raw.erase(std::unique(eff_raw.begin(), eff_raw.end(),
                                      [](const auto& x, const auto& y) { return x.first == y.first; }),
                          eff_raw.end());
        };

        // Causal edges: one per (producer, consumer) pair in pair order; with
        // edge_deduplication off, one per shared hyperedge, the duplicates told apart by
        // EdgeIndex so Graph[] does not merge them.
        auto add_causal_edges = [&]() {
            std::vector<std::pair<int64_t, int64_t>> pairs;
            for (const auto& [producer, consumer] : src.causal_event_pairs())
                pairs.emplace_back(src.effective_event_id(producer), src.effective_event_id(consumer));
            parallel_sort(pairs, par);
            for (size_t i = 0; i < pairs.size();) {
                size_t j = i;
                while (j < pairs.size() && pairs[j] == pairs[i]) ++j;
                const auto [from, to] = pairs[i];
                const size_t num_edges = opts.edge_deduplication ? 1 : j - i;
                for (size_t k = 0; k < num_edges; ++k) {
                    if (!send_unique_edge(from, to, kEdgeCausal, static_cast<uint32_t>(k))) continue;
                    Edge e;
                    e.from = End{'E', from};
                    e.to = End{'E', to};
                    e.type = EdgeType::Causal;
                    e.data = EdgeData::Causal;
                    e.indexed = num_edges > 1;
                    e.a = from;
                    e.b = to;
                    e.c = static_cast<int64_t>(k);
                    edges.push_back(e);
                }
                i = j;
            }
        };

        if (is_states) {
            std::vector<std::pair<int64_t, uint32_t>> state_verts;
            for (uint32_t sid = 0; sid < num_states; ++sid)
                if (src.state_valid(sid)) state_verts.emplace_back(src.effective_state_id(sid), sid);
            lowest_by_effective(state_verts);
            for (const auto& [eff, raw] : state_verts)
                if (send_vertex(eff, src.state_step(raw))) add_state_vertex(End{0, eff}, raw);
            for (uint32_t eid = 0; eid < num_events; ++eid) {
                if (!src.is_valid_event(eid)) continue;
                const int64_t in  = src.effective_state_id(src.event_input_state(eid));
                const int64_t out_ = src.effective_state_id(src.event_output_state(eid));
                // Keyed by the EVENT as well as the endpoints: two distinct events can join the
                // same pair of states, and a states-graph edge carries the event id its tooltip
                // names, so collapsing them would drop an edge a full delivery emits.
                const int64_t eff_eid = src.effective_event_id(eid);
                if (!send_edge(in, out_, kEdgeDirected, static_cast<uint32_t>(eff_eid))) continue;
                Edge e;
                e.from = End{0, in};
                e.to = End{0, out_};
                e.type = EdgeType::Directed;
                if (is_structure) {
                    e.data = EdgeData::EventId;
                    e.a = eff_eid;
                } else {
                    e.data = EdgeData::EventPayload;
                    e.payload = payload_of(eid);
                }
                edges.push_back(e);
            }
        }
        else if (is_causal) {
            std::vector<std::pair<int64_t, uint32_t>> event_verts;
            for (uint32_t eid = 0; eid < num_events; ++eid)
                if (src.is_valid_event(eid)) event_verts.emplace_back(src.effective_event_id(eid), eid);
            lowest_by_effective(event_verts);
            for (const auto& [eff, raw] : event_verts)
                if (send_vertex(eff, kEventRevision)) add_event_vertex(End{'E', eff}, raw);
            add_causal_edges();
        }
        else if (is_branchial) {
            bool filter_by_step = false;
            const uint32_t target_step =
                branchial_target_step(opts.branchial_step, opts.steps, filter_by_step);
            std::map<int64_t, uint32_t> state_verts;
            const auto pairs = src.branchial_event_pairs();
            for (const auto& [e1, e2] : pairs) {
                if (filter_by_step && src.state_step(src.event_output_state(e1)) != target_step) continue;
                const int64_t s1 = src.effective_state_id(src.event_output_state(e1));
                const int64_t s2 = src.effective_state_id(src.event_output_state(e2));
                state_verts.emplace(s1, src.event_output_state(e1));
                state_verts.emplace(s2, src.event_output_state(e2));
            }
            for (const auto& [eff, raw] : state_verts)
                if (send_vertex(eff, src.state_step(raw))) add_state_vertex(End{0, eff}, raw);
            // Every pair is an edge (multiplicity is meaning, see above): its index is the
            // number of earlier pairs between the same two states in the same order, which a
            // session's cursor sees unchanged as the pair list grows.
            std::map<std::pair<int64_t, int64_t>, uint32_t> occurrence;
            for (const auto& [e1, e2] : pairs) {
                if (filter_by_step && src.state_step(src.event_output_state(e1)) != target_step) continue;
                const int64_t s1 = src.effective_state_id(src.event_output_state(e1));
                const int64_t s2 = src.effective_state_id(src.event_output_state(e2));
                if (!send_unique_edge(s1, s2, kEdgeBranchialState, occurrence[{s1, s2}]++))
                    continue;
                Edge e;
                e.from = End{0, s1};
                e.to = End{0, s2};
                e.type = EdgeType::Branchial;
                e.data = EdgeData::States;
                e.a = s1;
                e.b = s2;
                edges.push_back(e);
            }
        }
        else if (is_evolution) {
            std::map<int64_t, uint32_t> raw_states;
            std::vector<std::pair<int64_t, uint32_t>> event_verts;
            for (uint32_t eid = 0; eid < num_events; ++eid) {
                if (!src.is_valid_event(eid)) continue;
                raw_states.emplace(src.effective_state_id(src.event_input_state(eid)),
                                   src.event_input_state(eid));
                raw_states.emplace(src.effective_state_id(src.event_output_state(eid)),
                                   src.event_output_state(eid));
                event_verts.emplace_back(src.effective_event_id(eid), eid);
            }
            lowest_by_effective(event_verts);
            // The two vertex kinds share one id space here (both are tagged), so a state and an
            // event with the same number must not collide in the record: states are offset.
            for (const auto& [sid, raw] : raw_states)
                if (send_vertex(kStateVertexBias + sid, src.state_step(raw)))
                    add_state_vertex(End{'S', sid}, raw);
            for (const auto& [eff, raw] : event_verts)
                if (send_vertex(eff, kEventRevision)) add_event_vertex(End{'E', eff}, raw);
            for (uint32_t eid = 0; eid < num_events; ++eid) {
                if (!src.is_valid_event(eid)) continue;
                const int64_t eff_eid = src.effective_event_id(eid);
                const int64_t in_id  = src.effective_state_id(src.event_input_state(eid));
                const int64_t out_id = src.effective_state_id(src.event_output_state(eid));
                Edge e;
                e.data = EdgeData::EventId;
                e.a = eff_eid;
                if (send_edge(in_id, eff_eid, kEdgeStateEvent, 0)) {
                    e.from = End{'S', in_id};
                    e.to = End{'E', eff_eid};
                    e.type = EdgeType::StateEvent;
                    edges.push_back(e);
                }
                if (send_edge(eff_eid, out_id, kEdgeEventState, 0)) {
                    e.from = End{'E', eff_eid};
                    e.to = End{'S', out_id};
                    e.type = EdgeType::EventState;
                    edges.push_back(e);
                }
            }
            if (has_causal) add_causal_edges();
            if (has_branchial) {
                bool filter_by_step = false;
                const uint32_t target_step =
                    branchial_target_step(opts.branchial_step, opts.steps, filter_by_step);
                // One edge per unordered pair of event vertices, lower id first, in pair order.
                std::vector<std::pair<int64_t, int64_t>> pairs;
                for (const auto& [e1, e2] : src.branchial_event_pairs()) {
                    if (filter_by_step && src.state_step(src.event_output_state(e1)) != target_step) continue;
                    const int64_t a = src.effective_event_id(e1);
                    const int64_t b = src.effective_event_id(e2);
                    pairs.emplace_back(std::min(a, b), std::max(a, b));
                }
                parallel_sort(pairs, par);
                pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
                for (const auto& [from, to] : pairs) {
                    if (!send_unique_edge(from, to, kEdgeBranchialEvent, 0)) continue;
                    Edge e;
                    e.from = End{'E', from};
                    e.to = End{'E', to};
                    e.type = EdgeType::Branchial;
                    e.data = EdgeData::Events;
                    e.a = from;
                    e.b = to;
                    edges.push_back(e);
                }
            }
        }

        // The payload fields, then the edge lists of the states they name.
        std::vector<GraphStateFields> state_fields;
        std::vector<GraphEventFields> event_fields;
        std::vector<uint8_t> event_present;
        if (!is_structure) {
            state_fields.resize(vertices.size());
            event_fields.resize(payload_events.size());
            event_present.assign(payload_events.size(), 0);
            const size_t vc = chunks(vertices.size());
            par(vc + chunks(payload_events.size()), [&](size_t t) {
                if (t < vc) {
                    const size_t end = std::min(vertices.size(), (t + 1) * kChunk);
                    for (size_t i = t * kChunk; i < end; ++i)
                        if (!vertices[i].event) src.state_fields(vertices[i].raw, state_fields[i]);
                    return;
                }
                t -= vc;
                const size_t end = std::min(payload_events.size(), (t + 1) * kChunk);
                for (size_t i = t * kChunk; i < end; ++i)
                    event_present[i] = src.event_fields(payload_events[i], event_fields[i]) ? 1 : 0;
            });
            std::vector<uint32_t> named;
            for (const Vertex& v : vertices)
                if (!v.event) named.push_back(v.raw);
            for (size_t i = 0; i < event_fields.size(); ++i)
                if (event_present[i] && !event_fields[i].reconstructed) {
                    named.push_back(event_fields[i].input_raw);
                    named.push_back(event_fields[i].output_raw);
                }
            memo.add(src, std::move(named), par);
        }

        const size_t vc = chunks(vertices.size()), ec = chunks(edges.size());
        std::vector<wxf::Writer> vertex_list(vc), vertex_data(vc), edge_list(ec);
        par(vc + ec, [&](size_t t) {
            if (t < vc) {
                wxf::Writer& vl = vertex_list[t];
                wxf::Writer& vd = vertex_data[t];
                const size_t end = std::min(vertices.size(), (t + 1) * kChunk);
                for (size_t i = t * kChunk; i < end; ++i) {
                    const Vertex& v = vertices[i];
                    write_end(vl, v.key);
                    vd.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    write_end(vd, v.key);
                    if (!is_structure) {
                        if (v.event)
                            write_event_payload(vd, event_fields[v.payload],
                                                event_present[v.payload] != 0, memo);
                        else
                            write_state_payload(vd, state_fields[i], memo.of(v.raw));
                    } else if (v.event) {
                        write_association_header(vd, 3);
                        vd.write_key("Id");          vd.write(src.effective_event_id(v.raw));
                        vd.write_key("InputState");
                        vd.write(src.effective_state_id(src.event_input_state(v.raw)));
                        vd.write_key("OutputState");
                        vd.write(src.effective_state_id(src.event_output_state(v.raw)));
                    } else {
                        write_association_header(vd, 2);
                        vd.write_key("Id");   vd.write(src.effective_state_id(v.raw));
                        vd.write_key("Step"); vd.write(static_cast<int64_t>(src.state_step(v.raw)));
                    }
                }
                return;
            }
            t -= vc;
            wxf::Writer& w = edge_list[t];
            const size_t end = std::min(edges.size(), (t + 1) * kChunk);
            for (size_t i = t * kChunk; i < end; ++i) {
                const Edge& e = edges[i];
                w.write_byte(static_cast<uint8_t>(wxf::Token::Association));
                const size_t count_at = w.size();
                w.write_varint(4);
                w.write_key("From"); write_end(w, e.from);
                w.write_key("To");   write_end(w, e.to);
                w.write_key("Type"); w.write_string(edge_type_name(e.type));
                const size_t data_at = w.size();
                w.write_key("Data");
                switch (e.data) {
                    case EdgeData::EventPayload: {
                        const size_t payload_at = w.size();
                        write_event_payload(w, event_fields[e.payload],
                                            event_present[e.payload] != 0, memo);
                        // An empty payload is no Data key.
                        if (w.size() - payload_at == 2 && w.data()[payload_at + 1] == 0) {
                            w.truncate(data_at);
                            w.set_byte(count_at, 3);
                        }
                        break;
                    }
                    case EdgeData::EventId:
                        write_association_header(w, 1);
                        w.write_key("EventId"); w.write(e.a);
                        break;
                    case EdgeData::Causal:
                        write_association_header(w, e.indexed ? 3 : 2);
                        w.write_key("ProducerEvent"); w.write(e.a);
                        w.write_key("ConsumerEvent"); w.write(e.b);
                        if (e.indexed) { w.write_key("EdgeIndex"); w.write(e.c); }
                        break;
                    case EdgeData::States:
                        write_association_header(w, 2);
                        w.write_key("State1"); w.write(e.a);
                        w.write_key("State2"); w.write(e.b);
                        break;
                    case EdgeData::Events:
                        write_association_header(w, 2);
                        w.write_key("Event1"); w.write(e.a);
                        w.write_key("Event2"); w.write(e.b);
                        break;
                }
            }
        });

        // WHETHER THIS IS A WHOLE GRAPH OR AN INCREMENT, stated rather than inferred: an
        // increment that contains every vertex and a full delivery have the same content, and so
        // do an empty delta and an empty system. The FIRST delta for a property is marked 0,
        // because a cursor that has not seen the property emits everything. As an integer, this
        // wire's boolean convention (`IsInitial` on a state is written the same way).
        out.write_key(graph_property);
        write_association_header(out, 4);
        out.write_key("IsDelta");
        out.write(static_cast<int64_t>(cur != nullptr && delivered_before));
        out.write_key("Vertices");
        out.write_function("List", vertices.size());
        for (const wxf::Writer& w : vertex_list) out.append(w.data());
        out.write_key("Edges");
        out.write_function("List", edges.size());
        for (const wxf::Writer& w : edge_list) out.append(w.data());
        out.write_key("VertexData");
        write_association_header(out, vertices.size());
        for (const wxf::Writer& w : vertex_data) out.append(w.data());
    }

    return wxf::WXFValue(wxf::Encoded{out.release_data()});
}

}  // namespace marshal
}  // namespace HG_NAMESPACE