#pragma once
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
//   bool     state_valid(uint32_t sid) const;
//   int64_t  effective_state_id(uint32_t sid) const;   // canonical/content/raw per mode
//   uint32_t state_step(uint32_t sid) const;
//   wxf::WXFValueAssociation serialize_state_data(uint32_t sid) const;
//
//   uint32_t num_raw_events() const;             // scan bound for raw event ids
//   bool     is_valid_event(uint32_t eid) const; // false for genesis unless shown
//   int64_t  effective_event_id(uint32_t eid) const;
//   uint32_t event_input_state(uint32_t eid) const;
//   uint32_t event_output_state(uint32_t eid) const;
//   wxf::WXFValueAssociation serialize_event_data(uint32_t eid) const;
//
//   // (producer, consumer) raw event-id pairs, genesis already filtered, with
//   // per-shared-hyperedge multiplicity preserved (edge_deduplication uses it).
//   std::vector<std::pair<uint32_t, uint32_t>> causal_event_pairs() const;
//   // (event1, event2) raw branchial event-id pairs, genesis already filtered.
//   std::vector<std::pair<uint32_t, uint32_t>> branchial_event_pairs() const;

#include "wxf.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace hgmarshal {

struct GraphOptions {
    bool edge_deduplication = true;   // one causal edge per (producer, consumer) pair
    int  branchial_step = 0;          // 0 = all; >0 = 1-based step; <0 = from end (-1 = final)
    int  steps = 0;                   // total evolution steps (for the from-end branchial_step)
};

// Resolve branchial_step into an absolute target step (0 means "no filter").
inline uint32_t branchial_target_step(int branchial_step, int steps, bool& filter_by_step) {
    filter_by_step = (branchial_step != 0);
    if (!filter_by_step) return 0;
    if (branchial_step > 0) return static_cast<uint32_t>(branchial_step);
    return static_cast<uint32_t>(steps + 1 + branchial_step);
}

// Build the "GraphData" association: property name -> <|Vertices, Edges, VertexData|>.
template <typename Source>
wxf::WXFValue build_graph_data(const Source& src,
                               const std::vector<std::string>& graph_properties,
                               const GraphOptions& opts) {
    wxf::WXFValueAssociation all_graph_data;

    for (const std::string& graph_property : graph_properties) {
        wxf::WXFValueList vertices;
        wxf::WXFValueList edges;
        wxf::WXFValueAssociation vertex_data;

        const bool is_states    = graph_property.rfind("States", 0) == 0;
        const bool is_causal    = graph_property.rfind("Causal", 0) == 0;
        const bool is_branchial = graph_property.rfind("Branchial", 0) == 0;
        const bool is_evolution = graph_property.find("Evolution") != std::string::npos;
        const bool has_causal    = is_evolution && graph_property.find("Causal") != std::string::npos;
        const bool has_branchial = is_evolution && graph_property.find("Branchial") != std::string::npos;

        auto add_graph_edge = [&](wxf::WXFValue from, wxf::WXFValue to, const std::string& type,
                                  wxf::WXFValueAssociation data = {}) {
            wxf::WXFValueAssociation edge;
            edge.push_back({wxf::WXFValue("From"), from});
            edge.push_back({wxf::WXFValue("To"), to});
            edge.push_back({wxf::WXFValue("Type"), wxf::WXFValue(type)});
            if (!data.empty()) edge.push_back({wxf::WXFValue("Data"), wxf::WXFValue(data)});
            edges.push_back(wxf::WXFValue(edge));
        };

        // Causal edges: dedup to unique (producer, consumer) pairs; with
        // edge_deduplication off, keep one edge per shared hyperedge (multiplicity).
        // A unique EdgeIndex on the duplicates stops Graph[] from collapsing them.
        auto add_causal_edges = [&]() {
            std::map<std::pair<int64_t, int64_t>, size_t> pair_counts;
            for (const auto& [producer, consumer] : src.causal_event_pairs()) {
                int64_t from = src.effective_event_id(producer);
                int64_t to   = src.effective_event_id(consumer);
                pair_counts[{from, to}]++;
            }
            for (const auto& [pair, count] : pair_counts) {
                wxf::WXFValueList from_tag = {wxf::WXFValue("E"), wxf::WXFValue(pair.first)};
                wxf::WXFValueList to_tag   = {wxf::WXFValue("E"), wxf::WXFValue(pair.second)};
                size_t num_edges = opts.edge_deduplication ? 1 : count;
                for (size_t k = 0; k < num_edges; ++k) {
                    wxf::WXFValueAssociation causal_data;
                    causal_data.push_back({wxf::WXFValue("ProducerEvent"), wxf::WXFValue(pair.first)});
                    causal_data.push_back({wxf::WXFValue("ConsumerEvent"), wxf::WXFValue(pair.second)});
                    if (num_edges > 1)
                        causal_data.push_back({wxf::WXFValue("EdgeIndex"), wxf::WXFValue(static_cast<int64_t>(k))});
                    add_graph_edge(wxf::WXFValue(from_tag), wxf::WXFValue(to_tag), "Causal", causal_data);
                }
            }
        };

        if (is_states) {
            std::map<int64_t, uint32_t> state_verts;
            for (uint32_t sid = 0; sid < src.num_states(); ++sid) {
                if (!src.state_valid(sid)) continue;
                int64_t eff = src.effective_state_id(sid);
                if (!state_verts.count(eff)) state_verts[eff] = sid;
            }
            for (auto& [eff, raw] : state_verts) {
                vertices.push_back(wxf::WXFValue(eff));
                vertex_data.push_back({wxf::WXFValue(eff), wxf::WXFValue(src.serialize_state_data(raw))});
            }
            for (uint32_t eid = 0; eid < src.num_raw_events(); ++eid) {
                if (!src.is_valid_event(eid)) continue;
                add_graph_edge(wxf::WXFValue(src.effective_state_id(src.event_input_state(eid))),
                               wxf::WXFValue(src.effective_state_id(src.event_output_state(eid))),
                               "Directed", src.serialize_event_data(eid));
            }
        }
        else if (is_causal) {
            std::map<int64_t, uint32_t> event_verts;
            for (uint32_t eid = 0; eid < src.num_raw_events(); ++eid) {
                if (!src.is_valid_event(eid)) continue;
                int64_t eff = src.effective_event_id(eid);
                if (!event_verts.count(eff)) event_verts[eff] = eid;
            }
            for (auto& [eff, raw] : event_verts) {
                wxf::WXFValueList tag = {wxf::WXFValue("E"), wxf::WXFValue(eff)};
                vertices.push_back(wxf::WXFValue(tag));
                vertex_data.push_back({wxf::WXFValue(tag), wxf::WXFValue(src.serialize_event_data(raw))});
            }
            add_causal_edges();
        }
        else if (is_branchial) {
            bool filter_by_step = false;
            uint32_t target_step = branchial_target_step(opts.branchial_step, opts.steps, filter_by_step);

            std::map<int64_t, uint32_t> state_verts;
            auto pairs = src.branchial_event_pairs();
            for (const auto& [e1, e2] : pairs) {
                if (filter_by_step && src.state_step(src.event_output_state(e1)) != target_step) continue;
                int64_t s1 = src.effective_state_id(src.event_output_state(e1));
                int64_t s2 = src.effective_state_id(src.event_output_state(e2));
                if (!state_verts.count(s1)) state_verts[s1] = src.event_output_state(e1);
                if (!state_verts.count(s2)) state_verts[s2] = src.event_output_state(e2);
            }
            for (auto& [eff, raw] : state_verts) {
                vertices.push_back(wxf::WXFValue(eff));
                vertex_data.push_back({wxf::WXFValue(eff), wxf::WXFValue(src.serialize_state_data(raw))});
            }
            for (const auto& [e1, e2] : pairs) {
                if (filter_by_step && src.state_step(src.event_output_state(e1)) != target_step) continue;
                int64_t s1 = src.effective_state_id(src.event_output_state(e1));
                int64_t s2 = src.effective_state_id(src.event_output_state(e2));
                wxf::WXFValueAssociation branchial_data;
                branchial_data.push_back({wxf::WXFValue("State1"), wxf::WXFValue(s1)});
                branchial_data.push_back({wxf::WXFValue("State2"), wxf::WXFValue(s2)});
                add_graph_edge(wxf::WXFValue(s1), wxf::WXFValue(s2), "Branchial", branchial_data);
            }
        }
        else if (is_evolution) {
            std::set<int64_t> state_ids;
            std::map<int64_t, uint32_t> raw_states;
            std::map<int64_t, uint32_t> event_verts;
            for (uint32_t eid = 0; eid < src.num_raw_events(); ++eid) {
                if (!src.is_valid_event(eid)) continue;
                int64_t in_id  = src.effective_state_id(src.event_input_state(eid));
                int64_t out_id = src.effective_state_id(src.event_output_state(eid));
                if (!raw_states.count(in_id))  raw_states[in_id]  = src.event_input_state(eid);
                if (!raw_states.count(out_id)) raw_states[out_id] = src.event_output_state(eid);
                state_ids.insert(in_id);
                state_ids.insert(out_id);
                int64_t eff = src.effective_event_id(eid);
                if (!event_verts.count(eff)) event_verts[eff] = eid;
            }
            for (int64_t sid : state_ids) {
                wxf::WXFValueList tag = {wxf::WXFValue("S"), wxf::WXFValue(sid)};
                vertices.push_back(wxf::WXFValue(tag));
                vertex_data.push_back({wxf::WXFValue(tag), wxf::WXFValue(src.serialize_state_data(raw_states[sid]))});
            }
            for (auto& [eff, raw] : event_verts) {
                wxf::WXFValueList tag = {wxf::WXFValue("E"), wxf::WXFValue(eff)};
                vertices.push_back(wxf::WXFValue(tag));
                vertex_data.push_back({wxf::WXFValue(tag), wxf::WXFValue(src.serialize_event_data(raw))});
            }
            for (uint32_t eid = 0; eid < src.num_raw_events(); ++eid) {
                if (!src.is_valid_event(eid)) continue;
                int64_t eff_eid = src.effective_event_id(eid);
                wxf::WXFValueList s_in  = {wxf::WXFValue("S"), wxf::WXFValue(src.effective_state_id(src.event_input_state(eid)))};
                wxf::WXFValueList s_out = {wxf::WXFValue("S"), wxf::WXFValue(src.effective_state_id(src.event_output_state(eid)))};
                wxf::WXFValueList e_tag = {wxf::WXFValue("E"), wxf::WXFValue(eff_eid)};
                wxf::WXFValueAssociation edge_data;
                edge_data.push_back({wxf::WXFValue("EventId"), wxf::WXFValue(eff_eid)});
                add_graph_edge(wxf::WXFValue(s_in), wxf::WXFValue(e_tag), "StateEvent", edge_data);
                add_graph_edge(wxf::WXFValue(e_tag), wxf::WXFValue(s_out), "EventState", edge_data);
            }
            if (has_causal) add_causal_edges();
            if (has_branchial) {
                bool filter_by_step = false;
                uint32_t target_step = branchial_target_step(opts.branchial_step, opts.steps, filter_by_step);
                for (const auto& [e1, e2] : src.branchial_event_pairs()) {
                    if (filter_by_step && src.state_step(src.event_output_state(e1)) != target_step) continue;
                    int64_t from = src.effective_event_id(e1);
                    int64_t to   = src.effective_event_id(e2);
                    wxf::WXFValueList from_tag = {wxf::WXFValue("E"), wxf::WXFValue(from)};
                    wxf::WXFValueList to_tag   = {wxf::WXFValue("E"), wxf::WXFValue(to)};
                    wxf::WXFValueAssociation branchial_data;
                    branchial_data.push_back({wxf::WXFValue("Event1"), wxf::WXFValue(from)});
                    branchial_data.push_back({wxf::WXFValue("Event2"), wxf::WXFValue(to)});
                    add_graph_edge(wxf::WXFValue(from_tag), wxf::WXFValue(to_tag), "Branchial", branchial_data);
                }
            }
        }

        wxf::WXFValueAssociation graph_data;
        graph_data.push_back({wxf::WXFValue("Vertices"), wxf::WXFValue(vertices)});
        graph_data.push_back({wxf::WXFValue("Edges"), wxf::WXFValue(edges)});
        graph_data.push_back({wxf::WXFValue("VertexData"), wxf::WXFValue(vertex_data)});
        all_graph_data.push_back({wxf::WXFValue(graph_property), wxf::WXFValue(graph_data)});
    }

    return wxf::WXFValue(all_graph_data);
}

}  // namespace hgmarshal
