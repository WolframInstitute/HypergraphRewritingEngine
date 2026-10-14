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
// The invariants are hgcommon::state_invariants (state_invariants_core.hpp), computed by each
// engine as it creates a class; step_record reads one record per class.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "hgcommon/state_invariants_core.hpp"
#include "hgcommon/state_geometry_core.hpp"
#include "hgcommon/branchial_rows_core.hpp"
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

// The record of one state given as its edge list, computed on the calling thread and held in
// `storage`.
const hgcommon::StateInvariantRecord* invariant_record(
    const std::vector<std::vector<uint32_t>>& edges, std::vector<uint64_t>& storage);

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
// `matches_by_rule` maps an expansion key (hgcommon::qc_expansion_key of the class and depth,
// `by_depth` under MaxStatesPerStep) to its captured matches per rule; no event leaves step
// `steps`.
void events_from_multiplicities(
    const std::vector<StepPoint>& points,
    const std::unordered_map<uint64_t, std::map<int64_t, uint64_t>>& matches_by_rule,
    bool by_depth, uint32_t steps, std::map<uint32_t, uint64_t>& events,
    std::map<uint32_t, std::map<int64_t, uint64_t>>& rule_counts);

// The geometry of one state (hgcommon/state_geometry_core.hpp): the values and the ball-growth
// dimension at r = 1..R.
struct StateGeometry {
    hgcommon::SgGeometry values;
    std::vector<double> ball_dimension;
};

StateGeometry state_geometry(const std::vector<std::vector<uint32_t>>& edges);

// One step's states and branchial pairs, for the per-step branchial metrics. `nodes` are the
// step's states by effective id; `pairs` the branchial state pairs whose first event's output
// state is at the step; for the overlap metrics, `vertex_sets[i]` the engine's vertex ids of
// nodes[i] and `edge_sets[i]` its edges, one number per distinct list of vertex ids in the run.
struct BranchialStep {
    std::vector<int64_t> nodes;
    std::vector<std::pair<int64_t, int64_t>> pairs;
    std::vector<std::vector<uint32_t>> vertex_sets;
    std::vector<std::vector<uint32_t>> edge_sets;
};

// Which per-step branchial metrics a request asks for: "StepStatisticsBranchial" sets
// kBranchialGraph and kBranchialOverlap, "StepStatisticsMatrices" the kMatrix* bits, one per
// state x state matrix of "StateMatrices".
enum : uint32_t {
    kBranchialGraph = 1u,
    kBranchialOverlap = 2u,
    kMatrixStateOverlap = 1u << 2,
    kMatrixStateCosineSimilarity = 1u << 3,
    kMatrixStateMutualInformation = 1u << 4,
    kMatrixStateEdgeOverlap = 1u << 5,
    kMatrixStateEdgeCosineSimilarity = 1u << 6,
    kMatrixStateEdgeMutualInformation = 1u << 7,
    kMatrixVertex = kMatrixStateOverlap | kMatrixStateCosineSimilarity |
                    kMatrixStateMutualInformation,
    kMatrixEdge = kMatrixStateEdgeOverlap | kMatrixStateEdgeCosineSimilarity |
                  kMatrixStateEdgeMutualInformation,
    kBranchialMatrices = kMatrixVertex | kMatrixEdge,
    // The bits that read each state's vertices and edges.
    kBranchialContents = kBranchialOverlap | kBranchialMatrices,
};

// The "StepStatisticsMatrices" names, in kMatrix* bit order from kMatrixStateOverlap.
constexpr const char* kMatrixNames[6] = {
    "StateOverlap",     "StateCosineSimilarity",     "StateMutualInformation",
    "StateEdgeOverlap", "StateEdgeCosineSimilarity", "StateEdgeMutualInformation"};

// The kMatrix* bit of a "StepStatisticsMatrices" name; 0 for any other string.
uint32_t matrix_bit(const std::string& name);

// "StepStatisticsMatrixLimit": the most states a step may have for its "StateMatrices".
constexpr uint32_t kMatrixLimitDefault = 1000;

// The warning for a step whose `states` states are above `limit`.
std::string matrix_limit_warning(uint32_t step, uint32_t states, uint32_t limit);

// The vertices of step 0's states (S0), sorted and distinct, which
// "InitialStateMutualInformation" reads at every step.
std::vector<uint32_t> initial_vertices(const BranchialStep& step0);

// The per-step keys of the branchial metrics of one step, from its BranchialStep: with
// kBranchialGraph "BranchialDegree", "BranchialDistance", "BranchialComponents" and
// "BranchialDimension" (absent when undefined); with kBranchialOverlap "StateOverlap",
// "StateCosineSimilarity", "StateMutualInformation", "InitialStateMutualInformation" (when
// `initial` is given), "VertexSharpness", "BranchEntropy", "EdgeSharpness" and
// "EdgeBranchEntropy"; with both, "OverlapByBranchialDistance"; with kMatrix* bits
// "StateMatrices", when the step has at most `matrix_limit` states.
//
// Built in three calls so the overlap rows can run on several workers: prepare(max_parts), then
// rows(p) for every p < parts(), concurrently and in any order, then finish(). The row counts
// are merged in key order, so the keys do not depend on which part ran which row.
class BranchialStepWork {
public:
    BranchialStepWork(BranchialStep step, uint32_t which, const std::vector<uint32_t>* initial,
                      uint32_t matrix_limit = kMatrixLimitDefault);
    ~BranchialStepWork();
    BranchialStepWork(const BranchialStepWork&) = delete;
    BranchialStepWork& operator=(const BranchialStepWork&) = delete;

    // The graph metrics, and the overlap pass's index; the rows are split into at most
    // `max_parts` parts, one per 64 states.
    void prepare(size_t max_parts);
    size_t parts() const;
    void rows(size_t part);
    // The overlap pass's index in compressed rows, from prepare until finish, for row counts
    // made elsewhere (hgcommon::br_row on the device), which add_counts adds in place of rows():
    // `n` pairs under (distance, |A|, |B|, shared). One thread at a time.
    const hgcommon::BrStep& rows_view() const;
    void add_counts(uint32_t distance, uint32_t size_a, uint32_t size_b, uint32_t shared,
                    uint64_t n);
    // Whether rows_view()'s rows are wanted at all: the overlap keys or a vertex-level matrix.
    bool wants_vertex_rows() const;
    // The shared counts of the state matrices, k x k, row a holding a's pairs with every b > a:
    // the vertex rows' shared vertices and the edge rows' shared edges. Null when not asked for
    // or over the limit. Rows made elsewhere write the cells their pairs emit at distance 0.
    uint32_t* vertex_shared();
    uint32_t* edge_shared();
    // The rows over the states' edge numbers, for edge_shared(); valid when edge_shared() is not
    // null.
    const hgcommon::BrStep& edge_rows_view() const;
    wxf::WXFValueAssociation finish();
    // The step's states when its matrices were asked for and it has more than the limit, else 0.
    uint32_t matrices_over_limit() const;

private:
    // The kBranchialOverlap keys.
    void overlap_keys(wxf::WXFValueAssociation& rec);
    struct State;
    std::unique_ptr<State> s_;
};

// One class of a step's population: `weight` raw states (saturating at 2^63 - 1) and the class's
// record, null reading as the state with no edges.
struct StepClass {
    uint64_t weight = 0;
    const hgcommon::StateInvariantRecord* record = nullptr;
};

// One step's input to its record: its classes by canonical hash, the raw events whose output
// state is at the step, in total and per rule index, and its BranchialStep.
struct StepInput {
    std::map<uint64_t, StepClass> classes;
    uint64_t events = 0;
    std::map<int64_t, uint64_t> rule_counts;
    BranchialStep branchial;
};

// Adds `weight` raw states of class `hash` with record `record` to `in`.
void add_class(StepInput& in, uint64_t hash, uint64_t weight,
               const hgcommon::StateInvariantRecord* record);

// The keys of the probes' per-step record that the step's classes and their weights determine,
// in reply order. `weight_by_classes`: each class counts once ("StepStatisticsWeighting" ->
// "Classes"); otherwise it stands for its raw states ("States").
wxf::WXFValueAssociation step_record(uint32_t step, const StepInput& in, bool weight_by_classes);

// THE "StepStatistics" REPLY, built by jobs. submit(step, input) hands a step's record to the
// caller's job system: one job pools the classes (step_record), one builds the branchial graph
// and prepares the overlap pass, and the overlap rows run as up to `workers` jobs, the last of
// which writes the step's branchial keys. reply() concatenates the finished records.
//
// A step is reserved by one thread at a time, at most once, and then started from any thread with
// its input, so the input can be gathered in a job. With the overlap metrics, step 0's start
// returns before any other step's start begins (every step reads step 0's vertices). reply is
// called after every started job has finished.
class StepStatisticsJobs {
public:
    using Submit = std::function<void(std::function<void()>)>;
    // The row counts of a prepared step made off the host (BranchialStepWork::add_counts): true
    // when every pair was added, false when nothing was and the rows run as jobs.
    using DeviceRows = std::function<bool(BranchialStepWork&)>;
    struct Step;
    StepStatisticsJobs(bool weight_by_classes, uint32_t which, size_t workers, Submit submit,
                       DeviceRows device_rows = nullptr, uint32_t device_min_states = 0);
    ~StepStatisticsJobs();
    StepStatisticsJobs(const StepStatisticsJobs&) = delete;
    StepStatisticsJobs& operator=(const StepStatisticsJobs&) = delete;

    // Null when `step` was reserved before.
    Step* reserve(uint32_t step);
    void start(Step* slot, StepInput in);
    void submit(uint32_t step, StepInput in) {
        if (Step* s = reserve(step)) start(s, std::move(in));
    }
    // One association per started step with a population, in step order.
    wxf::WXFValue reply();
    // "StepStatisticsMatrixLimit"; set before the first step starts.
    void set_matrix_limit(uint32_t limit) { matrix_limit_ = limit; }
    // After reply: (step, states) for each step whose matrices were over the limit.
    std::vector<std::pair<uint32_t, uint32_t>> matrices_over_limit() const;

private:
    bool weight_by_classes_;
    uint32_t which_;
    size_t workers_;
    Submit submit_;
    DeviceRows device_rows_;
    uint32_t device_min_states_;
    uint32_t matrix_limit_ = kMatrixLimitDefault;
    std::vector<uint32_t> initial_;
    bool have_initial_ = false;
    std::map<uint32_t, std::unique_ptr<Step>> steps_;
};

// The reply of `inputs`, built by StepStatisticsJobs on a job system of the host's hardware
// threads, for a reply whose run had no host engine (the GPU backend). A step of at least
// `device_min_states` states has its overlap rows counted by `device_rows` when it can.
// `over_limit`, when given, receives StepStatisticsJobs::matrices_over_limit.
wxf::WXFValue step_statistics_on_host(std::map<uint32_t, StepInput> inputs,
                                      bool weight_by_classes, uint32_t which,
                                      StepStatisticsJobs::DeviceRows device_rows = nullptr,
                                      uint32_t device_min_states = 0,
                                      uint32_t matrix_limit = kMatrixLimitDefault,
                                      std::vector<std::pair<uint32_t, uint32_t>>* over_limit =
                                          nullptr);

}  // namespace stats
}  // namespace HG_NAMESPACE
