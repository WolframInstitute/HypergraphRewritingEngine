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
#include <string>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "hgcommon/equilibrium_core.hpp"
#include "hgcommon/state_invariants_core.hpp"
#include "hgcommon/state_geometry_core.hpp"
#include "hgcommon/branchial_rows_core.hpp"
#include "hgcommon/step_pool_core.hpp"
#include "hgcommon/state_cost_core.hpp"
#include "wxf.hpp"

namespace HG_NAMESPACE {
namespace marshal {
class HostJobs;   // graph_marshal.hpp
}  // namespace marshal
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
// `storage`, its geometry on the graph rule `graph` (hgcommon::SG_GRAPH_*).
const hgcommon::StateInvariantRecord* invariant_record(
    const std::vector<std::vector<uint32_t>>& edges, std::vector<uint64_t>& storage,
    uint32_t graph = hgcommon::SG_GRAPH_CONSECUTIVE);

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

struct StepInput;

// The output-class pairs of each expansion's branchial pairs (hgcommon::qm_expansion_state_pairs),
// by expansion key, each pair once.
using ExpansionStatePairs = std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>>;

// Sorts `pairs` with each pair's smaller class first and removes repeats.
void unique_state_pairs(std::vector<std::pair<uint64_t, uint64_t>>& pairs);

// "StepStatisticsBranchial" -> "Graph" for a quotient run, from its multiplicities: each step's
// BranchialStep has the step's classes as its states, each named by its canonical hash, and as
// its pairs the output classes of the branchial pairs of the raw states one step earlier, read
// from `state_pairs` by expansion key (as events_from_multiplicities reads its matches). This is
// full capture's branchial graph of the step with each state named by its class.
void branchial_from_multiplicities(const std::vector<StepPoint>& points,
                                   const ExpansionStatePairs& state_pairs, bool by_depth,
                                   uint32_t steps, std::map<uint32_t, StepInput>& inputs);

// The geometry of one state (hgcommon/state_geometry_core.hpp): the values and the ball-growth
// dimension at r = 1..R.
struct StateGeometry {
    hgcommon::SgGeometry values;
    std::vector<double> ball_dimension;
};

StateGeometry state_geometry(const std::vector<std::vector<uint32_t>>& edges,
                             uint32_t graph = hgcommon::SG_GRAPH_CONSECUTIVE);

// (vertex, k_v) for each vertex of a state's graph with a neighbour, ascending, from the state's
// edges as a "States" record lists them ((edge id, vertices)): k_v is the mean Ollivier-Ricci
// curvature of v's edges (hgcommon/state_geometry_core.hpp), the value "VertexInvariants" pools
// as "OllivierRicciCurvature", on the graph rule `graph` (hgcommon::SG_GRAPH_*).
using VertexCurvature = std::vector<std::pair<uint32_t, double>>;
VertexCurvature vertex_curvature(const std::vector<std::pair<int64_t, std::vector<uint32_t>>>& edges,
                                 uint32_t graph = hgcommon::SG_GRAPH_CONSECUTIVE);

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
    // With kBranchialGraph: the volume growth and curvatures of the largest component
    // ("StepStatisticsMeasures" "Branchial").
    kBranchialMeasures = 1u << 8,
};

// "StepStatisticsMeasures": the family of each name as hgcommon::sg_geometry_of `want` bits, and
// kMeasureBranchial for "Branchial"; 0 for any other name.
constexpr uint32_t kMeasureBranchial = 1u << 30;
// "Equilibrium": each step's stability scores over the steps before it (hgcommon/equilibrium_core.hpp).
constexpr uint32_t kMeasureEquilibrium = 1u << 29;

// The "StepStatisticsMeasures" names and their bits.
struct MeasureName {
    const char* name;
    uint32_t bits;
};
inline constexpr MeasureName kMeasureNames[] = {
    {"VolumeGrowth", hgcommon::SG_VOLUME_GROWTH},
    {"Geodesics", hgcommon::SG_GEODESIC},
    {"EdgeCurvature", hgcommon::SG_NONLAZY_OLLIVIER},
    {"Combinatorial", hgcommon::SG_COMBINATORIAL},
    {"Particles", hgcommon::SG_PARTICLE},
    {"DimensionGradient", hgcommon::SG_DIMENSION_GRADIENT},
    {"Topology", hgcommon::SG_TOPOLOGY},
    {"MetricTensor", hgcommon::SG_METRIC_TENSOR},
    {"Holonomy", hgcommon::SG_HOLONOMY},
    {"BranchAlignment", hgcommon::SG_BRANCH_ALIGNMENT},
    {"Equilibrium", kMeasureEquilibrium},
    {"GeodesicAnalysis", hgcommon::SG_GEODESIC_ANALYSIS},
    {"Branchial", kMeasureBranchial},
};
uint32_t measure_bits(const std::string& name);

// The "StepStatisticsMatrices" names, in kMatrix* bit order from kMatrixStateOverlap.
constexpr const char* kMatrixNames[6] = {
    "StateOverlap",     "StateCosineSimilarity",     "StateMutualInformation",
    "StateEdgeOverlap", "StateEdgeCosineSimilarity", "StateEdgeMutualInformation"};

// The kMatrix* bit of a "StepStatisticsMatrices" name; 0 for any other string.
uint32_t matrix_bit(const std::string& name);

// "StateCost": the "Invariants" slot (hgcommon::sp_invariant) of a statistic name; -1 for any
// other string.
int invariant_slot(const std::string& name);

// "StateCost": the program (hgcommon::StateCost) of a postfix expression. `code` holds one token
// per operation: a statistic name (invariant_slot), "Const" (the next value of `constants`),
// "Plus", "Subtract", "Times", "Divide", "Min", "Max", "Minus", "Abs", "Sqrt", "Log", "Exp" or
// "Power" (raises the top to the next value of `constants`). Empty `code` is no cost. Throws
// std::runtime_error naming the defect.
hgcommon::StateCost compile_state_cost(const std::vector<std::string>& code,
                                       const std::vector<double>& constants);

// Whether the overlap and cosine matrices of one level (hgcommon::BoMatrices of a step's k
// states) go out as a SparseArray: fewer than half of the k (k - 1) / 2 state pairs share an
// element, read from the overlap's nonzero cells above the diagonal (the cosine's when the
// overlap is not asked for; the two are nonzero on the same pairs). Below half, the SparseArray
// takes under 6 k² + O(k) bytes in WXF against 8 k² for the packed array (a real and a column
// index of at most 4 bytes per nonzero), and about the packed array's bytes or fewer in the
// kernel (16 per nonzero against 8 per cell). Measured on Wolfram 15.0 at k = 1000: at half the
// pairs the SparseArray is 0.63 of the packed array in WXF and 1.002 of it by ByteCount; at a
// tenth, 0.13 and 0.20. False with neither matrix or with fewer than 2 states. The mutual
// information is nonzero for disjoint states and is always a packed array.
bool sparse_matrices(const hgcommon::BoMatrices& x);

// The bytes the matrices `which` of a step of k states take while they are built: per matrix its
// k x k reals and their dense WXF copy, 16 k^2; per level (vertex, edge) its k (k - 1) / 2
// shared counts, 2 k (k - 1). UINT64_MAX above 2^28 states.
uint64_t matrix_bytes(uint32_t k, uint32_t which);

// "StepStatisticsMatrixLimit" as given (bytes; 0 is Automatic) to bytes: Automatic is a quarter
// of the host's available memory at the process's probe (hgcommon::host_topology()), the
// fraction branchial_rows_device takes of the free device memory.
uint64_t matrix_limit_bytes(uint64_t option);

// The warning for a step of `states` states whose matrices `which` need more than `limit` bytes.
std::string matrix_limit_warning(uint32_t step, uint32_t states, uint32_t which, uint64_t limit);

// The vertices of step 0's states (S0), sorted and distinct, which
// "InitialStateMutualInformation" reads at every step.
std::vector<uint32_t> initial_vertices(const BranchialStep& step0);

// The per-step keys of the branchial metrics of one step, from its BranchialStep: with
// kBranchialGraph "BranchialDegree", "BranchialDistance", "BranchialComponents" and
// "BranchialDimension" (absent when undefined); with kBranchialOverlap "StateOverlap",
// "StateCosineSimilarity", "StateMutualInformation", "InitialStateMutualInformation" (when
// `initial` is given), "VertexSharpness", "BranchEntropy", "EdgeSharpness" and
// "EdgeBranchEntropy"; with both, "OverlapByBranchialDistance"; with kMatrix* bits
// "StateMatrices", when they need at most `matrix_limit` bytes (matrix_bytes; 0 is Automatic,
// matrix_limit_bytes).
//
// Built in three calls so the overlap rows can run on several workers: prepare(max_parts), then
// rows(p) for every p < parts(), concurrently and in any order, then finish(). The row counts
// are merged in key order, so the keys do not depend on which part ran which row.
class BranchialStepWork {
public:
    BranchialStepWork(BranchialStep step, uint32_t which, const std::vector<uint32_t>* initial,
                      uint64_t matrix_limit = 0);
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
    // The state matrices of the vertex rows and of the edge rows, each row's cells written by
    // hgcommon::bo_matrix_row after the row. Null when not asked for or over the limit. Rows made
    // elsewhere fill the asked-for arrays (m[kind]) in place of rows().
    hgcommon::BoMatrices* vertex_matrices();
    hgcommon::BoMatrices* edge_matrices();
    // The rows over the states' edge numbers, for edge_matrices(); valid when that is not null.
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

using PoolEntry = hgcommon::SpEntry;

// A part of a step's pool, made by one job or by the device: its entries, sorted by (slot, key)
// with equal keys merged once sort_part has run; its distribution totals; and the histograms of
// arity lists and degree sequences, keyed by the list.
struct PoolPart {
    std::vector<PoolEntry> entries;
    hgcommon::SpDistTotals dist[hgcommon::SG_DISTS];
    std::map<std::vector<int64_t>, uint64_t> arity_signatures, degree_sequences;
};

// Sorts `part`'s entries by (slot, key) and merges equal keys.
void sort_part(PoolPart& part);

// One step's input to its record: its classes by canonical hash, the raw events whose output
// state is at the step, in total and per rule index, and its BranchialStep. `device_part`, when
// set, holds the pool entries and totals of every class, made on the device; the classes then
// give the signature histograms only.
struct StepInput {
    std::map<uint64_t, StepClass> classes;
    uint64_t events = 0;
    std::map<int64_t, uint64_t> rule_counts;
    BranchialStep branchial;
    std::unique_ptr<PoolPart> device_part;
};

// Adds `weight` raw states of class `hash` with record `record` to `in`.
void add_class(StepInput& in, uint64_t hash, uint64_t weight,
               const hgcommon::StateInvariantRecord* record);

// Adds classes[0..n) to `part` and sorts it (sort_part); with `entries` false, only the
// signature histograms. Under `weight_by_classes` ("StepStatisticsWeighting" -> "Classes") each
// class counts once; otherwise it stands for its raw states ("States").
void pool_classes(const StepClass* const* classes, size_t n, bool weight_by_classes, bool entries,
                  PoolPart& part);

// The keys of the probes' per-step record that a step's pool determines, in reply order, built
// in three calls so the merge can run on several workers: run(t) for every t < tasks(), in any
// order and concurrently, then record(). Each task merges one slot's entries over the parts and
// summarises them in key order, so the record does not depend on how the classes were split.
class StepPoolFinish {
public:
    // `measures`: the requested families (hgcommon::SG_* want bits); a family's keys are in the
    // record only when requested.
    StepPoolFinish(uint32_t step, std::vector<PoolPart> parts, uint64_t events,
                   std::map<int64_t, uint64_t> rule_counts,
                   uint32_t measures = hgcommon::SG_MEASURES_BASE);
    ~StepPoolFinish();
    StepPoolFinish(const StepPoolFinish&) = delete;
    StepPoolFinish& operator=(const StepPoolFinish&) = delete;
    size_t tasks() const;
    void run(size_t task);
    wxf::WXFValueAssociation record();
    // The step's means of the equilibrium series (hgcommon::EQ_DIMENSION, EQ_DEGREE_ENTROPY,
    // EQ_SIZE), 0 for an invariant no state has; call after every task, before record.
    void equilibrium_series(double* out) const;

private:
    struct State;
    std::unique_ptr<State> s_;
};

// The record of `in` on the calling thread: one part, every task, the record.
wxf::WXFValueAssociation step_record(uint32_t step, const StepInput& in, bool weight_by_classes,
                                     uint32_t measures = hgcommon::SG_MEASURES_BASE);

// THE "StepStatistics" REPLY, built by jobs. submit(step, input) hands a step's record to the
// caller's job system: the step's classes are pooled by one job per 256 classes (pool_classes),
// the pool is merged and summarised by one job per slot (StepPoolFinish), one job builds the
// branchial graph and prepares the overlap pass, and the overlap rows run as up to `workers`
// jobs, the last of which writes the step's branchial keys. reply() concatenates the finished
// records.
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
    // "StepStatisticsMatrixLimit" in bytes, 0 for Automatic; set before the first step starts.
    void set_matrix_limit(uint64_t limit) { matrix_limit_ = limit; }
    // "StepStatisticsMeasures" as hgcommon::SG_* want bits; set before the first step starts.
    void set_measures(uint32_t measures) { measures_ = measures; }
    // After reply: (step, states) for each step whose matrices were over the limit.
    std::vector<std::pair<uint32_t, uint32_t>> matrices_over_limit() const;

private:
    bool weight_by_classes_;
    uint32_t which_;
    size_t workers_;
    Submit submit_;
    DeviceRows device_rows_;
    uint32_t device_min_states_;
    uint64_t matrix_limit_ = 0;
    uint32_t measures_ = hgcommon::SG_MEASURES_BASE;
    std::vector<uint32_t> initial_;
    bool have_initial_ = false;
    std::map<uint32_t, std::unique_ptr<Step>> steps_;
    Step* last_reserved_ = nullptr;

    void pool(Step* s);
    // Marks `s` closed and computes the equilibrium scores of every closed step whose
    // predecessor has its scores, from `s` on.
    void close_equilibrium(Step* s);
};

// The reply of `inputs`, built by StepStatisticsJobs on `host`, for a reply whose run had no host
// engine (the GPU backend). A step of at least `device_min_states` states has its overlap rows
// counted by `device_rows` when it can. `over_limit`, when given, receives
// StepStatisticsJobs::matrices_over_limit.
wxf::WXFValue step_statistics_on_host(marshal::HostJobs& host, std::map<uint32_t, StepInput> inputs,
                                      bool weight_by_classes, uint32_t which,
                                      StepStatisticsJobs::DeviceRows device_rows = nullptr,
                                      uint32_t device_min_states = 0,
                                      uint64_t matrix_limit = 0,
                                      std::vector<std::pair<uint32_t, uint32_t>>* over_limit =
                                          nullptr,
                                      uint32_t measures = hgcommon::SG_MEASURES_BASE);

}  // namespace stats
}  // namespace HG_NAMESPACE
