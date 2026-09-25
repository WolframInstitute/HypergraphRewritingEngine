#pragma once
#include "hgcommon/namespace.hpp"
// Per-state hypergraph invariants and their per-step summaries over a multiway population, for
// both engines. A state is its edge list, each edge the list of its vertices. The definitions
// are those of the multiway-statistics probes (docs/research/multiway-statistics/probes/
// common.wls, mwsCheapInvariants and mwsGraphInvariants), which the differential check
// reference/verify_state_statistics.wls compares against.
//
// Every invariant is isomorphism-invariant, so a step's population is summarised from one
// representative per class weighted by the class's multiplicity, and no raw state is built.

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "wxf.hpp"

namespace HG_NAMESPACE {
namespace stats {

struct StateInvariants {
    int64_t vertex_count = 0;
    int64_t edge_count = 0;
    std::vector<int64_t> arities;            // ascending
    std::vector<int64_t> degree_sequence;    // slot degree, descending
    int64_t max_degree = 0;
    double  mean_degree = 0.0;               // slots / vertex_count
    int64_t two_section_edge_count = 0;      // distinct unordered vertex pairs sharing an edge
    // The incidence bipartite graph: vertex nodes, edge nodes, one link per distinct
    // (edge, vertex) incidence.
    int64_t components = 0;
    int64_t cycle_rank = 0;                  // two_section_edge_count - vertex_count + components
    int64_t incidence_cycle_rank = 0;        // incidences - (vertex_count + edge_count) + components
    // The largest component: most nodes, then the greatest diameter, then the greatest mean
    // distance, then the most vertices.
    int64_t incidence_diameter = 0;          // of the largest component
    double  incidence_mean_distance = 0.0;   // mean over that component's node pairs
    double  largest_component_fraction = 0.0;  // that component's vertices / vertex_count
};

StateInvariants state_invariants(const std::vector<std::vector<uint32_t>>& edges);

// A summary of a weighted population of numbers: every value stands for `weight` raw states.
// Mean, sample standard deviation (N - 1), min, max and median over the N = sum of weights
// values; the histogram maps each value, rounded to `round` when `round` is not 1, to the total
// weight carrying it. Weights and N saturate at 2^63 - 1.
struct Summary {
    uint64_t n = 0;
    double mean = 0.0, standard_deviation = 0.0, min = 0.0, max = 0.0, median = 0.0;
    std::map<double, uint64_t> histogram;
};

Summary summarise(const std::vector<std::pair<double, uint64_t>>& value_weight, double round);

// One (step, class) point of a population: `weight` raw states of the class at the step.
struct StepPoint {
    uint32_t step;
    uint64_t class_hash;
    uint64_t weight;
};

// The raw events into each step of a quotient run, from its multiplicities: a class at depth d
// with m raw states and M_r matches of rule r gives m * M_r events of rule r into step d + 1.
// `matches_by_rule` maps a class to its captured matches per rule; no event leaves step `steps`.
void events_from_multiplicities(
    const std::vector<StepPoint>& points,
    const std::unordered_map<uint64_t, std::map<int64_t, uint64_t>>& matches_by_rule,
    uint32_t steps, std::map<uint32_t, uint64_t>& events,
    std::map<uint32_t, std::map<int64_t, uint64_t>>& rule_counts);

// The "StepStatistics" reply: one association per step, in step order, with the keys of the
// probes' per-step record that a class and its multiplicity determine. `class_edges` holds each
// class's representative; `events` and `rule_counts` the raw events whose output state is at a
// step, in total and per rule index.
wxf::WXFValue step_statistics(
    const std::vector<StepPoint>& points,
    const std::unordered_map<uint64_t, std::vector<std::vector<uint32_t>>>& class_edges,
    const std::map<uint32_t, uint64_t>& events,
    const std::map<uint32_t, std::map<int64_t, uint64_t>>& rule_counts);

}  // namespace stats
}  // namespace HG_NAMESPACE
