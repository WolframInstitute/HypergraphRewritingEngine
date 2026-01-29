#include "WolframLibrary.h"
#include "WolframNumericArrayLibrary.h"
#ifdef HAVE_WSTP
#include "wstp.h"
#endif
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <cstring>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>

// Include unified engine headers
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hypergraph/debug_log.hpp"
#include "job_system/job_system.hpp"

// Include comprehensive WXF library
#include "wxf.hpp"

// Include blackhole analysis (without layout dependency)
#include "blackhole/hausdorff_analysis.hpp"
#include "blackhole/bh_types.hpp"
#include "blackhole/geodesic_analysis.hpp"
#include "blackhole/particle_detection.hpp"
#include "blackhole/curvature_analysis.hpp"
#include "blackhole/entropy_analysis.hpp"
#include "blackhole/branchial_analysis.hpp"
#include "blackhole/equilibrium_analysis.hpp"
#include "blackhole/bh_initial_condition.hpp"
#include "blackhole/branch_alignment.hpp"

using namespace hypergraph;

// =============================================================================
// Lock-Free Debug Message Queue (MPSC - Multiple Producer, Single Consumer)
// =============================================================================
// Worker threads push messages (lock-free), main thread drains via atomic swap.

namespace {

struct DebugNode {
    char message[256];  // Fixed-size to avoid allocation in hot path
    DebugNode* next;
};

class LockFreeDebugQueue {
    std::atomic<DebugNode*> head_{nullptr};
    std::atomic<size_t> count_{0};
    static constexpr size_t MAX_MESSAGES = 10000;

public:
    // Push message (lock-free, called from worker threads)
    void push(const char* msg) {
        // Limit queue size to prevent runaway memory use
        if (count_.load(std::memory_order_relaxed) >= MAX_MESSAGES) {
            return;  // Drop message
        }

        DebugNode* node = new DebugNode();
        strncpy(node->message, msg, sizeof(node->message) - 1);
        node->message[sizeof(node->message) - 1] = '\0';

        // Lock-free prepend to list
        DebugNode* old_head = head_.load(std::memory_order_relaxed);
        do {
            node->next = old_head;
        } while (!head_.compare_exchange_weak(old_head, node,
                    std::memory_order_release, std::memory_order_relaxed));

        count_.fetch_add(1, std::memory_order_relaxed);
    }

    // Drain all messages (called from main thread only)
    // Returns messages in reverse order (oldest first after reversal)
    std::vector<std::string> drain() {
        // Atomically swap head with nullptr
        DebugNode* list = head_.exchange(nullptr, std::memory_order_acquire);
        count_.store(0, std::memory_order_relaxed);

        // Collect and reverse (list is newest-first, we want oldest-first)
        std::vector<std::string> messages;
        while (list) {
            messages.emplace_back(list->message);
            DebugNode* next = list->next;
            delete list;
            list = next;
        }
        std::reverse(messages.begin(), messages.end());
        return messages;
    }

    // Clear without returning (for cleanup)
    void clear() {
        DebugNode* list = head_.exchange(nullptr, std::memory_order_relaxed);
        count_.store(0, std::memory_order_relaxed);
        while (list) {
            DebugNode* next = list->next;
            delete list;
            list = next;
        }
    }
};

// Global instance
LockFreeDebugQueue g_debug_queue;

}  // anonymous namespace

// WXF Helper Functions using comprehensive wxf library
namespace ffi_helpers {
    // Parse rules association using wxf library
    // Returns vector of pairs to preserve rule order (unordered_map doesn't preserve order!)
    // IMPORTANT: Rules must use numeric vertex IDs, not symbolic patterns!
    // Valid:   {{0, 1}, {1, 2}} -> {{0, 1}, {1, 2}, {2, 3}}
    // Invalid: {{x, y}, {y, z}} -> {{x, y}, {y, z}, {z, w}}
    std::vector<std::pair<std::string, std::vector<std::vector<std::vector<int64_t>>>>>
    read_rules_association(wxf::Parser& parser) {
        std::vector<std::pair<std::string, std::vector<std::vector<std::vector<int64_t>>>>> rules;

        parser.read_association([&](const std::string& rule_name, wxf::Parser& rule_parser) {
            // Each rule value should be a function Rule[lhs, rhs]
            std::vector<std::vector<std::vector<int64_t>>> rule_parts;

            rule_parser.read_function([&](const std::string& head, size_t count, wxf::Parser& args_parser) {
                if (head != "Rule" || count != 2) {
                    throw std::runtime_error("Expected Rule[lhs, rhs]");
                }

                // Read LHS and RHS using recursive templates
                // Catch type errors and provide helpful message about symbolic vs numeric rules
                try {
                    auto lhs = args_parser.read<std::vector<std::vector<int64_t>>>();
                    auto rhs = args_parser.read<std::vector<std::vector<int64_t>>>();
                    rule_parts = {lhs, rhs};
                } catch (const wxf::TypeError& e) {
                    throw std::runtime_error(
                        "Rules must use numeric vertex IDs, not symbolic patterns. "
                        "Use {{0, 1}, {1, 2}} -> {{0, 1}, {1, 2}, {2, 3}} instead of "
                        "{{x, y}, {y, z}} -> {{x, y}, {y, z}, {z, w}}. "
                        "Original error: " + std::string(e.what()));
                }
            });

            rules.push_back({rule_name, rule_parts});
        });

        return rules;
    }
}

// Error handling
static void handle_error(WolframLibraryData libData, const char* message) {
    if (libData && libData->Message) {
        libData->Message(message);
    }
}

#ifdef HAVE_WSTP
// Mutex for WSTP calls (not thread-safe)
static std::mutex wstp_mutex;

// Progress reporting via WSTP - sends Print[] to frontend
static void print_to_frontend(WolframLibraryData libData, const std::string& message) {
    if (!libData) return;

    std::lock_guard<std::mutex> lock(wstp_mutex);
    WSLINK link = libData->getWSLINK(libData);
    if (!link) return;

    WSPutFunction(link, "EvaluatePacket", 1);
    WSPutFunction(link, "Print", 1);
    WSPutString(link, message.c_str());
    libData->processWSLINK(link);

    int pkt = WSNextPacket(link);
    if (pkt == RETURNPKT) {
        WSNewPacket(link);
    }
}
#endif

/**
 * Perform multiway rewriting evolution
 * Input: WXF binary data as 1D byte tensor containing:
 *   Association[
 *     "InitialEdges" -> {{vertices...}, ...},
 *     "Rules" -> <"Rule1" -> {{lhs edges}, {rhs edges}}, ...>,
 *     "Steps" -> integer,
 *     "Options" -> Association[...,
 *       "HashStrategy" -> "iUT"|"UT"|"WL",
 *       "IRVerification" -> True|False,
 *       "ReturnCanonicalStates" -> True|False,
 *       ...]
 *   ]
 *
 * Output: WXF Association with States, Events, CausalEdges, BranchialEdges
 * When ReturnCanonicalStates is True, each state includes "CanonicalEdges"
 * with IR-canonicalized (relabeled, sorted) edge list.
 */
EXTERN_C DLLEXPORT int performRewriting(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 1) {
            handle_error(libData, "performRewriting expects 1 argument: WXF ByteArray data");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Get WXF data as ByteArray (MNumericArray)
        MNumericArray wxf_array = MArgument_getMNumericArray(argv[0]);

        // Get array properties
        mint rank = libData->numericarrayLibraryFunctions->MNumericArray_getRank(wxf_array);
        if (rank != 1) {
            handle_error(libData, "WXF ByteArray must be 1-dimensional");
            return LIBRARY_FUNCTION_ERROR;
        }

        const mint* dims = libData->numericarrayLibraryFunctions->MNumericArray_getDimensions(wxf_array);
        mint wxf_size = dims[0];

        // Get raw byte data
        void* raw_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(wxf_array);
        const uint8_t* wxf_byte_data = static_cast<const uint8_t*>(raw_data);

        // Convert to vector
        std::vector<uint8_t> wxf_bytes(wxf_byte_data, wxf_byte_data + wxf_size);

        // Parse WXF input
        wxf::Parser parser(wxf_bytes);
        parser.skip_header();

        std::vector<std::vector<std::vector<int64_t>>> initial_states_raw;
        std::vector<std::pair<std::string, std::vector<std::vector<std::vector<int64_t>>>>> parsed_rules_raw;
        int steps = 1;

        // Option values
        hypergraph::StateCanonicalizationMode state_canon_mode = hypergraph::StateCanonicalizationMode::None;  // Default: tree mode
        hypergraph::EventSignatureKeys event_signature_keys = hypergraph::EVENT_SIG_NONE;  // Default: no event canonicalization
        bool show_genesis_events = false;
        bool show_progress = false;
        bool causal_transitive_reduction = true;
        size_t max_successor_states_per_parent = 0;
        size_t max_states_per_step = 0;
        double exploration_probability = 1.0;
        bool explore_from_canonical_states_only = false;  // Exploration deduplication
        std::string hash_strategy = "iUT";  // Default: IncrementalUniquenessTree
        bool ir_verification = false;  // IR-based exact edge correspondence and collision verification
        bool return_canonical_states = false;  // Compute canonical forms on demand via IR
        bool uniform_random = false;  // Use uniform random match selection (reservoir sampling)
        size_t matches_per_step = 0;  // Matches per step in uniform random mode (0 = all)

        // Dimension analysis options - compute Hausdorff dimension in C++ instead of WL round-trips
        bool compute_dimensions = false;
        int dim_min_radius = 1;
        int dim_max_radius = 5;
        bool dimension_per_vertex = false;    // Include per-vertex dimension data in PerState
        bool dimension_timestep_aggregation = false;  // Include PerTimestep aggregation section

        // Geodesic analysis options - trace test particles through the graph
        bool compute_geodesics = false;
        std::vector<int64_t> geodesic_sources;  // Empty = auto-select
        int geodesic_max_steps = 50;
        int geodesic_bundle_width = 5;
        bool geodesic_follow_gradient = false;
        float geodesic_dimension_percentile = 0.9f;

        // Particle detection options - detect topological defects (Robertson-Seymour)
        bool detect_particles = false;
        bool detect_k5_minors = true;
        bool detect_k33_minors = true;
        bool detect_dimension_spikes = true;
        bool detect_high_degree = true;
        float dimension_spike_threshold = 1.5f;
        float degree_percentile = 0.95f;
        bool compute_topological_charge = false;
        float charge_radius = 3.0f;
        bool charge_per_vertex = false;    // Include per-vertex charge data in PerState
        bool charge_timestep_aggregation = false;  // Include PerTimestep aggregation section

        // Curvature analysis options - Ollivier-Ricci, Wolfram-Ricci, and dimension gradient
        bool compute_curvature = false;
        std::string curvature_method = "All";  // "OllivierRicci", "WolframRicci", "DimensionGradient", "Both" (OR+DG), "All"
        bool curvature_ollivier_ricci = true;
        bool curvature_wolfram_ricci = true;
        bool curvature_wolfram_scalar = true;
        bool curvature_dimension_gradient = true;
        float curvature_ricci_alpha = 0.5f;  // Laziness parameter for Ollivier-Ricci
        int curvature_gradient_radius = 2;
        bool curvature_per_vertex = false;    // Include per-vertex curvature data in PerState
        bool curvature_timestep_aggregation = false;  // Include PerTimestep aggregation section

        // Entropy analysis options - graph entropy and information measures
        bool compute_entropy = false;
        bool entropy_local = true;
        bool entropy_mutual_info = true;
        bool entropy_fisher_info = true;
        int entropy_neighborhood_radius = 2;
        bool entropy_timestep_aggregation = false;  // Include PerTimestep aggregation section

        // Hilbert space analysis options - state bitvector inner products
        bool compute_hilbert_space = false;
        int hilbert_step = -1;  // Which step to analyze (-1 = all steps)
        std::string hilbert_scope = "Global";  // "Global", "PerTimestep", or "Both"

        // Branchial analysis options - distribution sharpness and branch entropy
        bool compute_branchial = false;
        std::string branchial_scope = "Global";  // "Global", "PerTimestep", or "Both"
        bool branchial_per_vertex = false;  // Include per-vertex sharpness data

        // Multispace analysis options - vertex/edge probabilities across branches
        bool compute_multispace = false;
        std::string multispace_scope = "Global";  // "Global", "PerTimestep", or "Both"

        // Equilibrium analysis options - track macroscopic property stability
        bool compute_equilibrium = false;
        int equilibrium_window = 20;        // Sliding window for stability computation
        float equilibrium_threshold = 0.95f; // Threshold for equilibrium detection

        // Topology configuration (for initial condition generation)
        std::string topology_type = "Flat";

        // Initial condition configuration
        std::string initial_condition_type = "Edges";  // "Edges", "Grid", or "Sprinkling"

        // Grid configuration (for regular grid initial conditions)
        int grid_width = 10;
        int grid_height = 10;

        // Sprinkling configuration (for Minkowski causal set initial conditions)
        int sprinkling_density = 500;        // Number of spacetime points
        float sprinkling_time_extent = 10.0f;    // Time dimension extent
        float sprinkling_spatial_extent = 10.0f; // Spatial dimension extent

        // Data selection flags - which components to include in output
        // By default all are included for backward compatibility
        bool include_states = true;
        bool include_events = true;
        bool include_events_minimal = false;  // Minimal event data: Id, InputState, OutputState only
        bool include_causal_edges = true;
        bool include_branchial_edges = true;       // Event-to-event (for Evolution*Branchial)
        bool include_branchial_state_edges = false; // State-to-state (for BranchialGraph) - overlap-based
        bool include_branchial_state_edges_all_siblings = false; // State-to-state all siblings (no overlap check)
        int branchial_step = 0;  // 0=All steps, positive=1-based step, negative=from end (-1=final)
        bool edge_deduplication = true;  // True: one edge per (from,to) pair; False: N edges for N shared hypergraph edges
        bool include_num_states = true;
        bool include_num_events = true;
        bool include_num_causal_edges = true;
        bool include_num_branchial_edges = true;
        bool include_global_edges = false;      // All edges created during evolution
        bool include_state_bitvectors = false;  // State edge sets as lists of edge IDs

        // GraphProperties option for graph-ready data output (list of properties)
        std::vector<std::string> graph_properties;  // e.g., {"StatesGraph", "CausalGraphStructure"}
        std::string canonicalize_states_mode = "None";  // Track actual mode string for effective ID computation

        // Parse main association
        parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
            if (key == "InitialStates") {
                initial_states_raw = value_parser.read<std::vector<std::vector<std::vector<int64_t>>>>();
            }
            else if (key == "InitialEdges") {
                // Legacy single-state format
                auto edges_data = value_parser.read<std::vector<std::vector<int64_t>>>();
                initial_states_raw.push_back(edges_data);
            }
            else if (key == "Rules") {
                parsed_rules_raw = ffi_helpers::read_rules_association(value_parser);
            }
            else if (key == "Steps") {
                steps = value_parser.read<int>();
            }
            else if (key == "Options") {
                value_parser.read_association([&](const std::string& option_key, wxf::Parser& option_parser) {
                    try {
                        if (option_key == "MaxSuccessorStatesPerParent") {
                            max_successor_states_per_parent = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "MaxStatesPerStep") {
                            max_states_per_step = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "MatchesPerStep") {
                            matches_per_step = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "DimensionMinRadius") {
                            dim_min_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "DimensionMaxRadius") {
                            dim_max_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "DimensionPerVertex") {
                            std::string symbol = option_parser.read<std::string>();
                            dimension_per_vertex = (symbol == "True");
                        } else if (option_key == "DimensionTimestepAggregation") {
                            std::string symbol = option_parser.read<std::string>();
                            dimension_timestep_aggregation = (symbol == "True");
                        } else if (option_key == "GeodesicSources") {
                            // List of vertex IDs for geodesic tracing ({-1} = auto-select)
                            auto sources = option_parser.read<std::vector<int64_t>>();
                            // Filter out -1 sentinel (means auto-select)
                            for (int64_t s : sources) {
                                if (s >= 0) geodesic_sources.push_back(s);
                            }
                        } else if (option_key == "GeodesicMaxSteps") {
                            geodesic_max_steps = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "GeodesicBundleWidth") {
                            geodesic_bundle_width = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "GeodesicDimensionPercentile") {
                            geodesic_dimension_percentile = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "DimensionSpikeThreshold") {
                            dimension_spike_threshold = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "DegreePercentile") {
                            degree_percentile = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "ChargeRadius") {
                            charge_radius = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "ChargePerVertex") {
                            std::string symbol = option_parser.read<std::string>();
                            charge_per_vertex = (symbol == "True");
                        } else if (option_key == "ChargeTimestepAggregation") {
                            std::string symbol = option_parser.read<std::string>();
                            charge_timestep_aggregation = (symbol == "True");
                        } else if (option_key == "CurvatureRicciAlpha") {
                            curvature_ricci_alpha = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "CurvatureGradientRadius") {
                            curvature_gradient_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "EntropyNeighborhoodRadius") {
                            entropy_neighborhood_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "EntropyTimestepAggregation") {
                            std::string symbol = option_parser.read<std::string>();
                            entropy_timestep_aggregation = (symbol == "True");
                        } else if (option_key == "HilbertStep") {
                            // Which step to analyze for Hilbert space (-1 = all steps)
                            hilbert_step = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "HilbertScope") {
                            hilbert_scope = option_parser.read<std::string>();
                        } else if (option_key == "BranchialScope") {
                            branchial_scope = option_parser.read<std::string>();
                        } else if (option_key == "BranchialPerVertex") {
                            std::string symbol = option_parser.read<std::string>();
                            branchial_per_vertex = (symbol == "True");
                        } else if (option_key == "MultispaceScope") {
                            multispace_scope = option_parser.read<std::string>();
                        } else if (option_key == "EquilibriumWindow") {
                            // Sliding window size for equilibrium stability computation
                            equilibrium_window = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "EquilibriumThreshold") {
                            // Threshold for equilibrium detection (0-1)
                            equilibrium_threshold = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "Topology") {
                            topology_type = option_parser.read<std::string>();
                        } else if (option_key == "InitialCondition") {
                            initial_condition_type = option_parser.read<std::string>();
                        } else if (option_key == "GridWidth") {
                            grid_width = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "GridHeight") {
                            grid_height = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "SprinklingDensity") {
                            sprinkling_density = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "SprinklingTimeExtent") {
                            sprinkling_time_extent = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "SprinklingSpatialExtent") {
                            sprinkling_spatial_extent = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "ExplorationProbability") {
                            exploration_probability = option_parser.read<double>();
                        } else if (option_key == "HashStrategy") {
                            hash_strategy = option_parser.read<std::string>();
                        } else if (option_key == "IRVerification") {
                            std::string symbol = option_parser.read<std::string>();
                            ir_verification = (symbol == "True");
                        } else if (option_key == "ReturnCanonicalStates") {
                            std::string symbol = option_parser.read<std::string>();
                            return_canonical_states = (symbol == "True");
                        } else if (option_key == "CurvatureMethod") {
                            curvature_method = option_parser.read<std::string>();
                        } else if (option_key == "CurvaturePerVertex") {
                            std::string symbol = option_parser.read<std::string>();
                            curvature_per_vertex = (symbol == "True");
                        } else if (option_key == "CurvatureTimestepAggregation") {
                            std::string symbol = option_parser.read<std::string>();
                            curvature_timestep_aggregation = (symbol == "True");
                        } else if (option_key == "BranchialStep") {
                            // 0=All, positive=1-based step index, negative=from end (-1=final)
                            branchial_step = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "EdgeDeduplication") {
                            std::string symbol = option_parser.read<std::string>();
                            edge_deduplication = (symbol == "True");
                        } else if (option_key == "RequestedData") {
                            // Parse list of data component names
                            // When specified, only include requested components
                            include_states = false;
                            include_events = false;
                            include_events_minimal = false;
                            include_causal_edges = false;
                            include_branchial_edges = false;
                            include_branchial_state_edges = false;

                            // Reset all include flags when RequestedData is specified
                            include_num_states = false;
                            include_num_events = false;
                            include_num_causal_edges = false;
                            include_num_branchial_edges = false;

                            auto components = option_parser.read<std::vector<std::string>>();
                            for (const auto& comp : components) {
                                if (comp == "States") include_states = true;
                                else if (comp == "Events") include_events = true;
                                else if (comp == "EventsMinimal") include_events_minimal = true;
                                else if (comp == "CausalEdges") include_causal_edges = true;
                                else if (comp == "BranchialEdges") include_branchial_edges = true;
                                else if (comp == "BranchialStateEdges") include_branchial_state_edges = true;
                                else if (comp == "BranchialStateEdgesAllSiblings") include_branchial_state_edges_all_siblings = true;
                                else if (comp == "NumStates") include_num_states = true;
                                else if (comp == "NumEvents") include_num_events = true;
                                else if (comp == "NumCausalEdges") include_num_causal_edges = true;
                                else if (comp == "NumBranchialEdges") include_num_branchial_edges = true;
                                else if (comp == "GlobalEdges") include_global_edges = true;
                                else if (comp == "StateBitvectors") include_state_bitvectors = true;
                            }
                        } else if (option_key == "CanonicalizeEvents") {
                            // Can be: None, Full, Automatic (symbols), or {"InputState", "OutputState", ...} (list)
                            try {
                                // Try to read as list first
                                auto keys = option_parser.read<std::vector<std::string>>();
                                event_signature_keys = hypergraph::EVENT_SIG_NONE;
                                for (const auto& key : keys) {
                                    if (key == "InputState") event_signature_keys |= hypergraph::EventKey_InputState;
                                    else if (key == "OutputState") event_signature_keys |= hypergraph::EventKey_OutputState;
                                    else if (key == "Step") event_signature_keys |= hypergraph::EventKey_Step;
                                    else if (key == "Rule") event_signature_keys |= hypergraph::EventKey_Rule;
                                    else if (key == "ConsumedEdges") event_signature_keys |= hypergraph::EventKey_ConsumedEdges;
                                    else if (key == "ProducedEdges") event_signature_keys |= hypergraph::EventKey_ProducedEdges;
                                }
                            } catch (...) {
                                // Read as symbol
                                std::string symbol = option_parser.read<std::string>();
                                if (symbol == "None") {
                                    event_signature_keys = hypergraph::EVENT_SIG_NONE;
                                } else if (symbol == "Full") {
                                    event_signature_keys = hypergraph::EVENT_SIG_FULL;
                                } else if (symbol == "Automatic") {
                                    event_signature_keys = hypergraph::EVENT_SIG_AUTOMATIC;
                                }
                                // else keep default (None)
                            }
                        } else if (option_key == "CanonicalizeStates") {
                            // Can be: None, Automatic, Full symbols (or legacy True/False)
                            // NOTE: Only Full mode does evolution-time deduplication.
                            // Automatic mode does NOT do evolution-time deduplication to match
                            // reference behavior (MultiwaySystem). Instead, Automatic only affects
                            // display-time grouping via ContentStateId computed in the FFI.
                            std::string symbol = option_parser.read<std::string>();
                            if (symbol == "None" || symbol == "False") {
                                state_canon_mode = hypergraph::StateCanonicalizationMode::None;
                                canonicalize_states_mode = "None";
                            } else if (symbol == "Automatic") {
                                // Automatic behaves like None for evolution (no deduplication)
                                // ContentStateId is computed separately for display-time grouping
                                state_canon_mode = hypergraph::StateCanonicalizationMode::None;
                                canonicalize_states_mode = "Automatic";
                            } else if (symbol == "Full" || symbol == "True") {
                                state_canon_mode = hypergraph::StateCanonicalizationMode::Full;
                                canonicalize_states_mode = "Full";
                            }
                        } else if (option_key == "GraphProperties") {
                            // Graph properties for graph-ready data output (list)
                            graph_properties = option_parser.read<std::vector<std::string>>();
                        } else {
                            std::string symbol = option_parser.read<std::string>();
                            bool value = (symbol == "True");

                            if (option_key == "ShowGenesisEvents") {
                                show_genesis_events = value;
                            } else if (option_key == "ShowProgress") {
                                show_progress = value;
                            } else if (option_key == "CausalTransitiveReduction") {
                                causal_transitive_reduction = value;
                            } else if (option_key == "ExploreFromCanonicalStatesOnly") {
                                // Exploration deduplication: only explore from canonical states
                                // Requires CanonicalizeStates -> Full to have any effect
                                explore_from_canonical_states_only = value;
                            } else if (option_key == "UniformRandom") {
                                // Use uniform random evolution mode (reservoir sampling)
                                uniform_random = value;
                            } else if (option_key == "DimensionAnalysis") {
                                // Compute Hausdorff dimension for all states in C++
                                compute_dimensions = value;
                            } else if (option_key == "GeodesicAnalysis") {
                                // Trace geodesic paths through the graph
                                compute_geodesics = value;
                            } else if (option_key == "TopologicalAnalysis") {
                                // Detect topological defects (Robertson-Seymour)
                                detect_particles = value;
                            } else if (option_key == "TopologicalCharge") {
                                // Compute per-vertex topological charge
                                compute_topological_charge = value;
                            } else if (option_key == "DetectK5Minors") {
                                detect_k5_minors = value;
                            } else if (option_key == "DetectK33Minors") {
                                detect_k33_minors = value;
                            } else if (option_key == "DetectDimensionSpikes") {
                                detect_dimension_spikes = value;
                            } else if (option_key == "DetectHighDegree") {
                                detect_high_degree = value;
                            } else if (option_key == "GeodesicFollowGradient") {
                                geodesic_follow_gradient = value;
                            } else if (option_key == "CurvatureAnalysis") {
                                // Compute curvature analysis
                                compute_curvature = value;
                            } else if (option_key == "CurvatureOllivierRicci") {
                                curvature_ollivier_ricci = value;
                            } else if (option_key == "CurvatureWolframRicci") {
                                curvature_wolfram_ricci = value;
                            } else if (option_key == "CurvatureWolframScalar") {
                                curvature_wolfram_scalar = value;
                            } else if (option_key == "CurvatureDimensionGradient") {
                                curvature_dimension_gradient = value;
                            } else if (option_key == "EntropyAnalysis") {
                                // Compute graph entropy and information measures
                                compute_entropy = value;
                            } else if (option_key == "EntropyLocal") {
                                entropy_local = value;
                            } else if (option_key == "EntropyMutualInfo") {
                                entropy_mutual_info = value;
                            } else if (option_key == "EntropyFisherInfo") {
                                entropy_fisher_info = value;
                            } else if (option_key == "HilbertSpaceAnalysis") {
                                // Compute Hilbert space analysis (state bitvector inner products)
                                compute_hilbert_space = value;
                            } else if (option_key == "BranchialAnalysis") {
                                // Compute branchial analysis (distribution sharpness, branch entropy)
                                compute_branchial = value;
                            } else if (option_key == "MultispaceAnalysis") {
                                // Compute multispace analysis (vertex/edge probabilities)
                                compute_multispace = value;
                            } else if (option_key == "EquilibriumAnalysis") {
                                // Compute equilibrium analysis (macroscopic stability)
                                compute_equilibrium = value;
                            }
                        }
                    } catch (...) {
                        option_parser.skip_value();
                    }
                });

                // Handle numeric options that may have been parsed as integers
                // (these need special handling since they come after the bool options)
            }
            else {
                value_parser.skip_value();
            }
        });

        // Generate sprinkling initial condition if requested
        if (initial_condition_type == "Sprinkling") {
            namespace bh = viz::blackhole;

            // Generate Minkowski sprinkling
            bh::SprinklingConfig sconfig;
            sconfig.spatial_dim = 2;
            sconfig.time_extent = sprinkling_time_extent;
            sconfig.spatial_extent = sprinkling_spatial_extent;
            sconfig.transitivity_reduction = true;
            sconfig.seed = 0;  // Could add option for this

            auto sprinkling = bh::generate_minkowski_sprinkling(
                static_cast<uint32_t>(sprinkling_density), sconfig);

            // Convert sprinkling edges to initial_states_raw format
            // Each causal edge becomes a hyperedge {from, to}
            std::vector<std::vector<int64_t>> sprinkling_edges;
            for (const auto& edge : sprinkling.causal_edges) {
                std::vector<int64_t> edge_vec;
                edge_vec.push_back(static_cast<int64_t>(edge.v1));
                edge_vec.push_back(static_cast<int64_t>(edge.v2));
                sprinkling_edges.push_back(edge_vec);
            }

            // Replace any provided initial states with sprinkling
            initial_states_raw.clear();
            initial_states_raw.push_back(sprinkling_edges);

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolve: Generated Minkowski sprinkling with "
                    << sprinkling.points.size() << " points, "
                    << sprinkling_edges.size() << " edges";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // Generate grid initial condition if requested
        if (initial_condition_type == "Grid") {
            namespace bh = viz::blackhole;

            // Create a minimal BHConfig (no black holes, just a box)
            bh::BHConfig config;
            config.mass1 = 0.0f;
            config.mass2 = 0.0f;
            config.separation = 0.0f;
            config.box_x = {-10.0f, 10.0f};
            config.box_y = {-10.0f, 10.0f};

            auto grid = bh::generate_solid_grid(grid_width, grid_height, config);

            // Convert grid edges to initial_states_raw format
            std::vector<std::vector<int64_t>> grid_edges;
            for (const auto& edge : grid.edges) {
                std::vector<int64_t> edge_vec;
                edge_vec.push_back(static_cast<int64_t>(edge.v1));
                edge_vec.push_back(static_cast<int64_t>(edge.v2));
                grid_edges.push_back(edge_vec);
            }

            // Replace any provided initial states with grid
            initial_states_raw.clear();
            initial_states_raw.push_back(grid_edges);

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolve: Generated " << grid_width << "x" << grid_height
                    << " grid with " << grid.vertex_count() << " vertices, "
                    << grid_edges.size() << " edges";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        if (initial_states_raw.empty()) {
            handle_error(libData, "No initial states provided");
            return LIBRARY_FUNCTION_ERROR;
        }

        if (parsed_rules_raw.empty()) {
            handle_error(libData, "No valid rules found");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Create hypergraph
        hypergraph::Hypergraph hg;

        // Set hash strategy
        if (hash_strategy == "WL") {
            hg.set_hash_strategy(hypergraph::HashStrategy::WL);
        } else if (hash_strategy == "UT") {
            hg.set_hash_strategy(hypergraph::HashStrategy::UniquenessTree);
        } else {
            hg.set_hash_strategy(hypergraph::HashStrategy::IncrementalUniquenessTree);
        }

        // Configure IR verification layer
        hg.set_ir_verification(ir_verification);
        hg.set_return_canonical_states(return_canonical_states);

        // Configure event canonicalization
        hg.set_event_signature_keys(event_signature_keys);

        // Configure state canonicalization mode
        hg.set_state_canonicalization_mode(state_canon_mode);

        // Create parallel evolution engine
        hypergraph::ParallelEvolutionEngine engine(&hg, std::thread::hardware_concurrency());

        // Configure engine options
        engine.set_max_steps(static_cast<size_t>(steps));
        engine.set_transitive_reduction(causal_transitive_reduction);
        engine.set_exploration_probability(exploration_probability);
        engine.set_max_successor_states_per_parent(max_successor_states_per_parent);
        engine.set_max_states_per_step(max_states_per_step);
        engine.set_genesis_events(show_genesis_events);
        engine.set_explore_from_canonical_states_only(explore_from_canonical_states_only);

        // Convert rules to unified format
        uint16_t rule_index = 0;
        for (const auto& [rule_name, rule_data] : parsed_rules_raw) {
            if (rule_data.size() != 2) continue;

            hypergraph::RewriteRule rule;
            rule.index = rule_index++;

            // Track max variable seen for variable counting
            uint8_t max_lhs_var = 0;
            uint8_t max_rhs_var = 0;

            // Parse LHS edges
            rule.num_lhs_edges = 0;
            for (const auto& edge : rule_data[0]) {
                if (rule.num_lhs_edges >= hypergraph::MAX_PATTERN_EDGES) break;
                hypergraph::PatternEdge& pe = rule.lhs[rule.num_lhs_edges];
                pe.arity = 0;
                for (int64_t v : edge) {
                    if (v >= 0 && pe.arity < hypergraph::MAX_ARITY) {
                        pe.vars[pe.arity++] = static_cast<uint8_t>(v);
                        if (v > max_lhs_var) max_lhs_var = static_cast<uint8_t>(v);
                    }
                }
                if (pe.arity > 0) {
                    rule.num_lhs_edges++;
                }
            }

            // Parse RHS edges
            rule.num_rhs_edges = 0;
            for (const auto& edge : rule_data[1]) {
                if (rule.num_rhs_edges >= hypergraph::MAX_PATTERN_EDGES) break;
                hypergraph::PatternEdge& pe = rule.rhs[rule.num_rhs_edges];
                pe.arity = 0;
                for (int64_t v : edge) {
                    if (v >= 0 && pe.arity < hypergraph::MAX_ARITY) {
                        pe.vars[pe.arity++] = static_cast<uint8_t>(v);
                        if (v > max_rhs_var) max_rhs_var = static_cast<uint8_t>(v);
                    }
                }
                if (pe.arity > 0) {
                    rule.num_rhs_edges++;
                }
            }

            rule.num_lhs_vars = max_lhs_var + 1;
            rule.num_rhs_vars = max_rhs_var + 1;
            rule.num_new_vars = (max_rhs_var > max_lhs_var) ? (max_rhs_var - max_lhs_var) : 0;

            if (rule.num_lhs_edges > 0 && rule.num_rhs_edges > 0) {
                engine.add_rule(rule);
            }
        }

        // Convert all initial states to vectors of edges
        // Multiple initial states are supported for exploring the full multiway system
        // CRITICAL: Each initial state gets CANONICAL vertex numbering (starting from 0)
        // This ensures isomorphic initial states like {{0,0},{0,0}} and {{1,1},{1,1}}
        // get the SAME internal representation and thus the SAME canonical hash.
        // The engine handles multiplicity - if the same canonical state appears multiple
        // times, it spawns MATCH tasks for each instance.
        std::vector<std::vector<std::vector<hypergraph::VertexId>>> initial_states;

        for (const auto& state_raw : initial_states_raw) {
            // Create a per-state vertex mapping: input_vertex -> canonical_vertex
            // Always start from 0 for canonical form
            std::unordered_map<int64_t, hypergraph::VertexId> vertex_map;
            hypergraph::VertexId next_vertex = 0;

            std::vector<std::vector<hypergraph::VertexId>> state_edges;
            for (const auto& edge : state_raw) {
                std::vector<hypergraph::VertexId> edge_vertices;
                for (int64_t v : edge) {
                    if (v >= 0) {
                        // Map this input vertex to a canonical vertex ID
                        auto it = vertex_map.find(v);
                        if (it == vertex_map.end()) {
                            vertex_map[v] = next_vertex;
                            edge_vertices.push_back(next_vertex);
                            next_vertex++;
                        } else {
                            edge_vertices.push_back(it->second);
                        }
                    }
                }
                if (!edge_vertices.empty()) {
                    state_edges.push_back(edge_vertices);
                }
            }
            if (!state_edges.empty()) {
                initial_states.push_back(std::move(state_edges));
            }
        }

        // Run evolution with progress reporting from main thread
        // NOTE: All WSTP calls happen on the main thread to avoid thread-safety issues
        // The abort_check callback runs every ~50ms, which is frequent enough for progress
#ifdef HAVE_WSTP
        auto evolution_start = std::chrono::steady_clock::now();
        auto last_progress_report = evolution_start;
        size_t last_states = 0, last_events = 0;
        size_t last_causal = 0, last_branchial = 0;

        if (show_progress) {
            print_to_frontend(libData, "HGEvolve: Starting evolution...");
        }

        // Set up debug callback to route DEBUG_LOG to our queue
        // Messages are drained and printed from the main thread in the abort callback
        debug::set_debug_callback([](const char* msg) {
            g_debug_queue.push(msg);
        });
#endif

        // Run evolution with abort checking - allows user to cancel via Mathematica's Abort[]
        // Progress is reported from the abort callback (runs on main thread every ~50ms)
#ifdef HAVE_WSTP
        bool was_aborted = false;

        if (uniform_random) {
            // Use uniform random mode (reservoir sampling) - same as blackhole_viz --uniform-random
            if (show_progress) {
                print_to_frontend(libData, "Using uniform random mode (reservoir sampling)");
            }
            if (!initial_states.empty()) {
                engine.evolve_uniform_random(
                    initial_states[0],  // Single initial state edges
                    static_cast<size_t>(steps),
                    matches_per_step
                );
            }
        } else {
            // Standard parallel evolution
            was_aborted = engine.evolve_with_abort(
            initial_states,
            static_cast<size_t>(steps),
            [&]() {
                bool should_abort = libData->AbortQ();

                // Drain debug messages and print them (from main thread)
                if (show_progress) {
                    auto debug_messages = g_debug_queue.drain();
                    for (const auto& msg : debug_messages) {
                        print_to_frontend(libData, msg);
                    }
                }

                // Print progress every ~500ms (callback runs every ~50ms)
                if (show_progress && !should_abort) {
                    auto now = std::chrono::steady_clock::now();
                    auto since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress_report).count();

                    if (since_last >= 500) {
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - evolution_start).count();

                        size_t cur_states = hg.num_canonical_states();
                        size_t cur_events = hg.num_events();
                        size_t cur_causal = hg.num_causal_event_pairs();  // v1 semantics
                        size_t cur_branchial = hg.num_branchial_edges();

                        // Calculate rates
                        double state_rate = (since_last > 0) ? (cur_states - last_states) * 1000.0 / since_last : 0;
                        double event_rate = (since_last > 0) ? (cur_events - last_events) * 1000.0 / since_last : 0;
                        double causal_rate = (since_last > 0) ? (cur_causal - last_causal) * 1000.0 / since_last : 0;
                        double branchial_rate = (since_last > 0) ? (cur_branchial - last_branchial) * 1000.0 / since_last : 0;

                        size_t pending = engine.pending_jobs();
                        size_t executing = engine.executing_jobs();

                        std::ostringstream oss;
                        oss << "[" << elapsed_ms << "ms] "
                            << "States: " << cur_states << " (" << static_cast<int>(state_rate) << "/s), "
                            << "Events: " << cur_events << " (" << static_cast<int>(event_rate) << "/s), "
                            << "Causal: " << cur_causal << " (" << static_cast<int>(causal_rate) << "/s), "
                            << "Branchial: " << cur_branchial << " (" << static_cast<int>(branchial_rate) << "/s)"
                            << " | Jobs: " << pending << " pending, " << executing << " executing";

                        print_to_frontend(libData, oss.str());

                        last_progress_report = now;
                        last_states = cur_states;
                        last_events = cur_events;
                        last_causal = cur_causal;
                        last_branchial = cur_branchial;
                    }
                }

                return should_abort;
            }
        );
        }  // end else (standard evolution)
#else
        bool was_aborted = false;
        if (uniform_random) {
            if (!initial_states.empty()) {
                engine.evolve_uniform_random(
                    initial_states[0],
                    static_cast<size_t>(steps),
                    matches_per_step
                );
            }
        } else {
            was_aborted = engine.evolve_with_abort(
                initial_states,
                static_cast<size_t>(steps),
                [libData]() { return libData->AbortQ(); }
            );
        }
#endif

#ifdef HAVE_WSTP
        // Clear debug callback and drain any remaining messages
        debug::clear_debug_callback();
        if (show_progress) {
            auto final_debug_messages = g_debug_queue.drain();
            for (const auto& msg : final_debug_messages) {
                print_to_frontend(libData, msg);
            }
        } else {
            g_debug_queue.clear();  // Discard if not showing progress
        }

        if (show_progress) {
            auto evolution_end = std::chrono::steady_clock::now();
            auto evolution_ms = std::chrono::duration_cast<std::chrono::milliseconds>(evolution_end - evolution_start).count();
            std::ostringstream oss;
            if (was_aborted) {
                oss << "HGEvolve: ABORTED after " << evolution_ms << "ms. ";
            } else {
                oss << "HGEvolve: Evolution complete in " << evolution_ms << "ms. ";
            }
            oss << "States: " << hg.num_canonical_states() << ", "
                << "Events: " << hg.num_events() << ", "
                << "Causal: " << hg.num_causal_event_pairs() << ", "  // v1 semantics
                << "Branchial: " << hg.num_branchial_edges();
            print_to_frontend(libData, oss.str());
            if (!was_aborted) {
                print_to_frontend(libData, "HGEvolve: Starting serialization...");
            }
        }
#else
        (void)was_aborted;  // Suppress unused variable warning when WSTP not available
#endif

        // =========================================================================
        // Dimension Analysis (if requested)
        // =========================================================================
        // Compute Hausdorff dimension for all states in C++ - avoids O(N) FFI round-trips
        namespace bh = viz::blackhole;

        std::unordered_map<uint32_t, bh::DimensionStats> state_dimension_stats;
        std::unordered_map<uint32_t, std::vector<float>> state_vertex_dimensions;
        std::unordered_map<uint32_t, std::vector<bh::VertexId>> state_vertex_ids;  // Vertex IDs for dimension lookup
        std::unordered_map<uint32_t, uint32_t> state_to_step;  // State ID -> timestep for aggregation
        float global_dim_min = std::numeric_limits<float>::max();
        float global_dim_max = std::numeric_limits<float>::lowest();

        if (compute_dimensions) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolve: Computing dimension analysis...");
            }
#endif
            // Create job system for parallel dimension computation
            job_system::JobSystem<int> dim_js(std::thread::hardware_concurrency());
            dim_js.start();

            uint32_t num_states = hg.num_states();

            // Mutex for thread-safe updates to shared maps
            std::mutex dim_mutex;

            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Build edges for SimpleGraph (only binary edges)
                std::vector<bh::Edge> edges;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    if (edge.arity == 2) {
                        edges.push_back({edge.vertices[0], edge.vertices[1]});
                    }
                });

                if (edges.size() >= 2) {
                    // Submit dimension analysis as a job
                    dim_js.submit_function([&, sid, edges = std::move(edges)]() {
                        bh::SimpleGraph graph;
                        graph.build_from_edges(edges);

                        bh::DimensionConfig config;
                        config.min_radius = dim_min_radius;
                        config.max_radius = dim_max_radius;

                        auto per_vertex = bh::estimate_all_dimensions(graph, config);
                        auto stats = bh::compute_dimension_stats(per_vertex);
                        auto vertices = graph.vertices();
                        uint32_t step = hg.get_state(sid).step;

                        // Thread-safe update of shared maps
                        std::lock_guard<std::mutex> lock(dim_mutex);
                        state_dimension_stats[sid] = stats;
                        state_vertex_dimensions[sid] = std::move(per_vertex);
                        state_vertex_ids[sid] = std::move(vertices);
                        state_to_step[sid] = step;

                        if (stats.count > 0) {
                            global_dim_min = std::min(global_dim_min, stats.mean);
                            global_dim_max = std::max(global_dim_max, stats.mean);
                        }
                    }, 0);
                }
            }

            // Wait for all dimension jobs to complete
            dim_js.wait_for_completion();
            dim_js.shutdown();

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolve: Dimension analysis complete. Analyzed "
                    << state_dimension_stats.size() << " states";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // ==========================================================================
        // Geodesic Analysis - trace test particles through graph
        // ==========================================================================
        // Per-state geodesic results: state_id -> vector of paths (each path is vector of vertex IDs)
        std::unordered_map<uint32_t, std::vector<std::vector<bh::VertexId>>> state_geodesic_paths;
        std::unordered_map<uint32_t, std::vector<std::vector<float>>> state_geodesic_proper_times;
        std::unordered_map<uint32_t, std::vector<std::vector<float>>> state_geodesic_local_dimensions;
        std::unordered_map<uint32_t, float> state_geodesic_bundle_spread;
        // Lensing metrics per state: state_id -> vector of LensingMetrics (one per path)
        std::unordered_map<uint32_t, std::vector<bh::LensingMetrics>> state_geodesic_lensing;
        std::unordered_map<uint32_t, float> state_geodesic_mean_deflection;
        std::unordered_map<uint32_t, bh::VertexId> state_geodesic_lensing_center;

        if (compute_geodesics) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolve: Computing geodesic analysis...");
            }
#endif
            uint32_t num_states = hg.num_states();

            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Build SimpleGraph
                std::vector<bh::Edge> edges;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    if (edge.arity == 2) {
                        edges.push_back({edge.vertices[0], edge.vertices[1]});
                    }
                });

                if (edges.size() >= 2) {
                    bh::SimpleGraph graph;
                    graph.build_from_edges(edges);

                    // Configure geodesic tracing
                    bh::GeodesicConfig geo_config;
                    geo_config.max_steps = geodesic_max_steps;
                    geo_config.bundle_width = geodesic_bundle_width;
                    geo_config.follow_dimension_ascent = geodesic_follow_gradient;
                    geo_config.direction = geodesic_follow_gradient
                        ? bh::GeodesicDirection::DimensionGradient
                        : bh::GeodesicDirection::Random;

                    // Get geodesic sources
                    std::vector<bh::VertexId> sources;
                    for (int64_t src : geodesic_sources) {
                        sources.push_back(static_cast<bh::VertexId>(src));
                    }

                    // Use dimension data if available for auto-selection and gradient following
                    const std::vector<float>* dims = nullptr;
                    auto it = state_vertex_dimensions.find(sid);
                    if (it != state_vertex_dimensions.end()) {
                        dims = &it->second;
                    }

                    // Run geodesic analysis
                    auto geo_result = bh::analyze_geodesics(graph, sources, geo_config, dims);

                    // Store results
                    std::vector<std::vector<bh::VertexId>> paths;
                    std::vector<std::vector<float>> proper_times;
                    std::vector<std::vector<float>> local_dims;
                    for (const auto& path : geo_result.paths) {
                        paths.push_back(path.vertices);
                        proper_times.push_back(path.proper_time);
                        local_dims.push_back(path.local_dimension);
                    }
                    state_geodesic_paths[sid] = std::move(paths);
                    state_geodesic_proper_times[sid] = std::move(proper_times);
                    state_geodesic_local_dimensions[sid] = std::move(local_dims);
                    state_geodesic_bundle_spread[sid] = geo_result.mean_spread;

                    // Store lensing metrics
                    if (!geo_result.lensing.empty()) {
                        state_geodesic_lensing[sid] = geo_result.lensing;
                        state_geodesic_mean_deflection[sid] = geo_result.mean_deflection;
                        state_geodesic_lensing_center[sid] = geo_result.lensing_center;
                    }
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolve: Geodesic analysis complete. Analyzed "
                    << state_geodesic_paths.size() << " states";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // ==========================================================================
        // Particle Detection - detect topological defects (Robertson-Seymour)
        // ==========================================================================
        // Per-state defect results
        struct FFIDetectedDefect {
            int type;  // 0=None, 1=K5, 2=K33, 3=HighDegree, 4=DimSpike
            std::vector<bh::VertexId> core_vertices;
            float charge;
            float centroid_x, centroid_y;
            float local_dimension;
            int confidence;
        };
        std::unordered_map<uint32_t, std::vector<FFIDetectedDefect>> state_defects;
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_charges;
        std::unordered_map<uint32_t, uint32_t> topological_state_to_step;  // State ID -> timestep for aggregation

        if (detect_particles) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolve: Detecting topological defects...");
            }
#endif
            uint32_t num_states = hg.num_states();

            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Build SimpleGraph
                std::vector<bh::Edge> edges;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    if (edge.arity == 2) {
                        edges.push_back({edge.vertices[0], edge.vertices[1]});
                    }
                });

                if (edges.size() >= 2) {
                    bh::SimpleGraph graph;
                    graph.build_from_edges(edges);

                    // Configure particle detection
                    bh::ParticleDetectionConfig particle_config;
                    particle_config.detect_k5 = detect_k5_minors;
                    particle_config.detect_k33 = detect_k33_minors;
                    particle_config.use_dimension_spikes = detect_dimension_spikes;
                    particle_config.dimension_spike_threshold = dimension_spike_threshold;
                    particle_config.use_high_degree = detect_high_degree;
                    particle_config.degree_threshold_percentile = degree_percentile;
                    particle_config.compute_charges = compute_topological_charge;
                    particle_config.charge_radius = charge_radius;

                    // Get dimension data if available
                    const std::vector<float>* dims = nullptr;
                    auto it = state_vertex_dimensions.find(sid);
                    if (it != state_vertex_dimensions.end()) {
                        dims = &it->second;
                    }

                    // Run particle analysis
                    auto particle_result = bh::analyze_particles(graph, particle_config, dims, nullptr);

                    // Convert defects
                    std::vector<FFIDetectedDefect> defects;
                    for (const auto& defect : particle_result.defects) {
                        FFIDetectedDefect fd;
                        fd.type = static_cast<int>(defect.type);
                        fd.core_vertices = defect.core_vertices;
                        fd.charge = defect.charge;
                        fd.centroid_x = defect.centroid.x;
                        fd.centroid_y = defect.centroid.y;
                        fd.local_dimension = defect.local_dimension;
                        fd.confidence = defect.detection_confidence;
                        defects.push_back(std::move(fd));
                    }
                    state_defects[sid] = std::move(defects);
                    state_charges[sid] = particle_result.charge_map;
                    topological_state_to_step[sid] = state.step;  // Store timestep for aggregation
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolve: Particle detection complete. Analyzed "
                    << state_defects.size() << " states";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // ==========================================================================
        // Curvature Analysis - Ollivier-Ricci, Wolfram-Ricci, and dimension gradient
        // ==========================================================================

        // Apply CurvatureMethod setting to individual flags
        if (curvature_method == "OllivierRicci") {
            curvature_ollivier_ricci = true;
            curvature_wolfram_ricci = false;
            curvature_dimension_gradient = false;
        } else if (curvature_method == "WolframRicci") {
            curvature_ollivier_ricci = false;
            curvature_wolfram_ricci = true;
            curvature_dimension_gradient = false;
        } else if (curvature_method == "DimensionGradient") {
            curvature_ollivier_ricci = false;
            curvature_wolfram_ricci = false;
            curvature_dimension_gradient = true;
        } else if (curvature_method == "Both") {
            // Backward compatible: Ollivier-Ricci + DimensionGradient
            curvature_ollivier_ricci = true;
            curvature_wolfram_ricci = false;
            curvature_dimension_gradient = true;
        } else if (curvature_method == "All") {
            curvature_ollivier_ricci = true;
            curvature_wolfram_ricci = true;
            curvature_dimension_gradient = true;
        }
        // Individual flags (CurvatureOllivierRicci, etc.) can still override

        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_ollivier_ricci;
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_wolfram_ricci;
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_wolfram_scalar;
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_dimension_gradient;
        std::unordered_map<uint32_t, float> state_mean_curvature;
        std::unordered_map<uint32_t, uint32_t> curvature_state_to_step;  // State ID -> timestep for aggregation

        if (compute_curvature) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolve: Computing curvature analysis...");
            }
#endif
            // Create job system for parallel curvature computation
            job_system::JobSystem<int> curv_js(std::thread::hardware_concurrency());
            curv_js.start();

            uint32_t num_states = hg.num_states();

            // Mutex for thread-safe updates to shared maps
            std::mutex curv_mutex;

            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Build edges for SimpleGraph
                std::vector<bh::Edge> edges;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    if (edge.arity == 2) {
                        edges.push_back({edge.vertices[0], edge.vertices[1]});
                    }
                });

                if (edges.size() >= 2) {
                    // Copy dimension data if available (for thread safety)
                    std::vector<float> dims_copy;
                    auto it = state_vertex_dimensions.find(sid);
                    if (it != state_vertex_dimensions.end()) {
                        dims_copy = it->second;
                    }
                    uint32_t step = state.step;

                    // Submit curvature analysis as a job
                    curv_js.submit_function([&, sid, step, edges = std::move(edges),
                                            dims = std::move(dims_copy)]() {
                        bh::SimpleGraph graph;
                        graph.build_from_edges(edges);

                        // Configure curvature analysis
                        bh::CurvatureConfig curv_config;
                        curv_config.compute_ollivier_ricci = curvature_ollivier_ricci;
                        curv_config.compute_wolfram_ricci = curvature_wolfram_ricci;
                        curv_config.compute_wolfram_scalar = curvature_wolfram_scalar;
                        curv_config.compute_dimension_gradient = curvature_dimension_gradient;
                        curv_config.ricci_alpha = curvature_ricci_alpha;
                        curv_config.gradient_radius = curvature_gradient_radius;

                        // Get dimension data if available
                        const std::vector<float>* dims_ptr = dims.empty() ? nullptr : &dims;

                        // Run curvature analysis
                        auto curv_result = bh::analyze_curvature(graph, curv_config, dims_ptr);

                        // Thread-safe update of shared maps
                        std::lock_guard<std::mutex> lock(curv_mutex);

                        if (curvature_ollivier_ricci) {
                            state_ollivier_ricci[sid] = std::move(curv_result.ollivier_ricci_map);
                        }
                        if (curvature_wolfram_ricci) {
                            state_wolfram_ricci[sid] = std::move(curv_result.wolfram_ricci_map);
                        }
                        if (curvature_wolfram_scalar) {
                            state_wolfram_scalar[sid] = std::move(curv_result.wolfram_scalar_map);
                        }
                        if (curvature_dimension_gradient) {
                            state_dimension_gradient[sid] = std::move(curv_result.dimension_gradient_map);
                        }
                        // Use first available mean curvature
                        if (curvature_ollivier_ricci && curv_result.mean_ollivier_ricci != 0.0f) {
                            state_mean_curvature[sid] = curv_result.mean_ollivier_ricci;
                        } else if (curvature_wolfram_ricci && curv_result.mean_wolfram_ricci != 0.0f) {
                            state_mean_curvature[sid] = curv_result.mean_wolfram_ricci;
                        } else if (curvature_wolfram_scalar && curv_result.mean_wolfram_scalar != 0.0f) {
                            state_mean_curvature[sid] = curv_result.mean_wolfram_scalar;
                        } else if (curvature_dimension_gradient && curv_result.mean_dimension_gradient != 0.0f) {
                            state_mean_curvature[sid] = curv_result.mean_dimension_gradient;
                        } else {
                            state_mean_curvature[sid] = 0.0f;
                        }
                        curvature_state_to_step[sid] = step;
                    }, 0);
                }
            }

            curv_js.wait_for_completion();

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolve: Curvature analysis complete. Analyzed "
                    << state_ollivier_ricci.size() << " states";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // ==========================================================================
        // Entropy Analysis - graph entropy and information measures
        // ==========================================================================
        std::unordered_map<uint32_t, float> state_degree_entropy;
        std::unordered_map<uint32_t, float> state_graph_entropy;
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_local_entropy;
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_mutual_info;
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_fisher_info;
        std::unordered_map<uint32_t, uint32_t> entropy_state_to_step;  // State ID -> timestep for aggregation

        if (compute_entropy) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolve: Computing entropy analysis...");
            }
#endif
            // Create job system for parallel entropy computation
            job_system::JobSystem<int> ent_js(std::thread::hardware_concurrency());
            ent_js.start();

            uint32_t num_states = hg.num_states();

            // Mutex for thread-safe updates to shared maps
            std::mutex ent_mutex;

            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Build SimpleGraph edges
                std::vector<bh::Edge> edges;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    if (edge.arity == 2) {
                        edges.push_back({edge.vertices[0], edge.vertices[1]});
                    }
                });

                if (edges.size() >= 2) {
                    // Copy dimension data for thread safety
                    std::vector<float> dims_copy;
                    auto it = state_vertex_dimensions.find(sid);
                    if (it != state_vertex_dimensions.end()) {
                        dims_copy = it->second;
                    }
                    uint32_t step = state.step;

                    // Submit entropy analysis as a job
                    ent_js.submit_function([&, sid, step, edges = std::move(edges),
                                           dims = std::move(dims_copy)]() {
                        bh::SimpleGraph graph;
                        graph.build_from_edges(edges);

                        // Configure entropy analysis
                        bh::EntropyConfig ent_config;
                        ent_config.compute_local_entropy = entropy_local;
                        ent_config.compute_mutual_info = entropy_mutual_info;
                        ent_config.compute_fisher_info = entropy_fisher_info;
                        ent_config.neighborhood_radius = entropy_neighborhood_radius;

                        // Get dimension data if available (for Fisher info)
                        const std::vector<float>* dims_ptr = dims.empty() ? nullptr : &dims;

                        // Run entropy analysis
                        auto ent_result = bh::analyze_entropy(graph, ent_config, dims_ptr);

                        // Thread-safe update of shared maps
                        std::lock_guard<std::mutex> lock(ent_mutex);
                        state_degree_entropy[sid] = ent_result.degree_entropy;
                        state_graph_entropy[sid] = ent_result.graph_entropy;
                        if (entropy_local) {
                            state_local_entropy[sid] = std::move(ent_result.local_entropy_map);
                        }
                        if (entropy_mutual_info) {
                            state_mutual_info[sid] = std::move(ent_result.mutual_info_map);
                        }
                        if (entropy_fisher_info) {
                            state_fisher_info[sid] = std::move(ent_result.fisher_info_map);
                        }
                        entropy_state_to_step[sid] = step;
                    }, 0);
                }
            }

            ent_js.wait_for_completion();

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolve: Entropy analysis complete. Analyzed "
                    << state_degree_entropy.size() << " states";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // ==========================================================================
        // Branchial-based Analyses (Hilbert, Branchial, Multispace)
        // Build shared BranchState structures once if any of these are needed
        // ==========================================================================
        bh::HilbertSpaceAnalysis hilbert_result;
        std::map<uint32_t, bh::HilbertSpaceAnalysis> hilbert_per_timestep;  // For per-timestep scope
        bh::BranchialAnalysisResult branchial_result;
        std::map<uint32_t, bh::BranchialAnalysisResult> branchial_per_timestep;  // For per-timestep scope
        bh::EquilibriumAnalysisResult equilibrium_result;
        bool has_hilbert_data = false;
        bool has_branchial_data = false;
        bool has_multispace_data = false;
        bool has_equilibrium_data = false;

        // Shared vertex/edge probability data for multispace (global)
        std::unordered_map<bh::VertexId, float> multispace_vertex_probs;
        std::map<std::pair<uint32_t, uint32_t>, float> multispace_edge_probs;
        float multispace_mean_vertex_prob = 0.0f;
        float multispace_mean_edge_prob = 0.0f;
        float multispace_total_entropy = 0.0f;

        // Per-timestep multispace data
        struct MultispaceTimestepData {
            std::unordered_map<bh::VertexId, float> vertex_probs;
            std::map<std::pair<uint32_t, uint32_t>, float> edge_probs;
            float mean_vertex_prob = 0.0f;
            float mean_edge_prob = 0.0f;
            float total_entropy = 0.0f;
        };
        std::map<uint32_t, MultispaceTimestepData> multispace_per_timestep;

        if (compute_hilbert_space || compute_branchial || compute_multispace || compute_equilibrium) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolve: Building branchial structures...");
            }
#endif
            // Build BranchState structures from the hypergraph
            std::vector<bh::BranchState> branch_states;
            uint32_t num_states = hg.num_states();

            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                bh::BranchState bs;
                bs.state_id = sid;
                bs.branch_id = 0;  // TODO: Get actual branch ID from multiway structure
                bs.step = state.step;

                // Collect all unique vertices in this state
                std::unordered_set<bh::VertexId> vertex_set;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    for (size_t i = 0; i < edge.arity; ++i) {
                        vertex_set.insert(edge.vertices[i]);
                    }
                });
                bs.vertices.assign(vertex_set.begin(), vertex_set.end());

                // Collect edges
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    if (edge.arity == 2) {
                        bs.edges.push_back({edge.vertices[0], edge.vertices[1]});
                    }
                });

                branch_states.push_back(std::move(bs));
            }

            if (!branch_states.empty()) {
                // Build branchial graph
                bh::BranchialConfig config;
                auto branchial_graph = bh::build_branchial_graph(branch_states, config);

                // Hilbert Space Analysis
                if (compute_hilbert_space) {
#ifdef HAVE_WSTP
                    if (show_progress) {
                        print_to_frontend(libData, "HGEvolve: Computing Hilbert space analysis...");
                    }
#endif
                    // Create job system for parallel Hilbert space computation
                    job_system::JobSystem<int> hilbert_js(std::thread::hardware_concurrency());
                    hilbert_js.start();

                    if (hilbert_step >= 0) {
                        hilbert_result = bh::analyze_hilbert_space_parallel(branchial_graph, &hilbert_js, static_cast<uint32_t>(hilbert_step));
                    } else {
                        hilbert_result = bh::analyze_hilbert_space_full_parallel(branchial_graph, &hilbert_js);
                    }

                    hilbert_js.wait_for_completion();
                    has_hilbert_data = (hilbert_result.num_states > 0);

                    // Compute edge-level MI directly from SparseBitsets (dot product of edge membership)
                    if (has_hilbert_data && hilbert_result.num_states > 1) {
                        size_t n = hilbert_result.num_states;
                        hilbert_result.edge_mutual_information_matrix.resize(n, std::vector<float>(n, 0.0f));

                        // Build edge universe (all edges across all states being analyzed)
                        std::unordered_set<hypergraph::EdgeId> edge_universe;
                        for (uint32_t sid : hilbert_result.state_indices) {
                            const hypergraph::State& state = hg.get_state(sid);
                            if (state.id != hypergraph::INVALID_ID) {
                                state.edges.for_each([&](hypergraph::EdgeId eid) {
                                    edge_universe.insert(eid);
                                });
                            }
                        }

                        float universe_size = static_cast<float>(edge_universe.size());
                        if (universe_size > 0) {
                            float sum_edge_mi = 0.0f;
                            int edge_mi_count = 0;

                            for (size_t i = 0; i < n; ++i) {
                                const hypergraph::State& state_a = hg.get_state(hilbert_result.state_indices[i]);
                                if (state_a.id == hypergraph::INVALID_ID) continue;

                                for (size_t j = i; j < n; ++j) {
                                    const hypergraph::State& state_b = hg.get_state(hilbert_result.state_indices[j]);
                                    if (state_b.id == hypergraph::INVALID_ID) continue;

                                    // Count joint occurrences over edge universe
                                    int n00 = 0, n01 = 0, n10 = 0, n11 = 0;
                                    for (hypergraph::EdgeId eid : edge_universe) {
                                        bool in_a = state_a.edges.contains(eid);
                                        bool in_b = state_b.edges.contains(eid);
                                        if (!in_a && !in_b) ++n00;
                                        else if (!in_a && in_b) ++n01;
                                        else if (in_a && !in_b) ++n10;
                                        else ++n11;
                                    }

                                    // Compute MI
                                    float p00 = n00 / universe_size, p01 = n01 / universe_size;
                                    float p10 = n10 / universe_size, p11 = n11 / universe_size;
                                    float p_a0 = p00 + p01, p_a1 = p10 + p11;
                                    float p_b0 = p00 + p10, p_b1 = p01 + p11;

                                    float mi = 0.0f;
                                    auto add_term = [&](float pj, float px, float py) {
                                        if (pj > 0 && px > 0 && py > 0) mi += pj * std::log2(pj / (px * py));
                                    };
                                    add_term(p00, p_a0, p_b0);
                                    add_term(p01, p_a0, p_b1);
                                    add_term(p10, p_a1, p_b0);
                                    add_term(p11, p_a1, p_b1);
                                    mi = std::max(0.0f, mi);

                                    hilbert_result.edge_mutual_information_matrix[i][j] = mi;
                                    hilbert_result.edge_mutual_information_matrix[j][i] = mi;

                                    if (i != j) {
                                        sum_edge_mi += mi;
                                        ++edge_mi_count;
                                        if (mi > hilbert_result.max_edge_mutual_information) {
                                            hilbert_result.max_edge_mutual_information = mi;
                                        }
                                    }
                                }
                            }

                            if (edge_mi_count > 0) {
                                hilbert_result.mean_edge_mutual_information = sum_edge_mi / edge_mi_count;
                            }
                        }
                    }

                    // Per-timestep Hilbert analysis (if scope is "PerTimestep" or "Both")
                    bool include_per_timestep_hilbert = (hilbert_scope == "PerTimestep" || hilbert_scope == "Both");
                    if (include_per_timestep_hilbert) {
#ifdef HAVE_WSTP
                        if (show_progress) {
                            print_to_frontend(libData, "HGEvolve: Computing per-timestep Hilbert space analysis...");
                        }
#endif
                        // Create job system for parallel per-timestep Hilbert computation
                        job_system::JobSystem<int> step_hilbert_js(std::thread::hardware_concurrency());
                        step_hilbert_js.start();

                        // Get unique timesteps from branchial graph
                        for (const auto& [step, state_ids] : branchial_graph.step_to_states) {
                            if (state_ids.empty()) continue;

                            // Analyze Hilbert space for this timestep (using parallel version)
                            auto step_result = bh::analyze_hilbert_space_parallel(branchial_graph, &step_hilbert_js, step);
                            if (step_result.num_states == 0) continue;

                            // Compute edge MI for this timestep's states
                            if (step_result.num_states > 1) {
                                size_t n = step_result.num_states;
                                step_result.edge_mutual_information_matrix.resize(n, std::vector<float>(n, 0.0f));

                                // Build edge universe for states at this timestep
                                std::unordered_set<hypergraph::EdgeId> step_edge_universe;
                                for (uint32_t sid : step_result.state_indices) {
                                    const hypergraph::State& state = hg.get_state(sid);
                                    if (state.id != hypergraph::INVALID_ID) {
                                        state.edges.for_each([&](hypergraph::EdgeId eid) {
                                            step_edge_universe.insert(eid);
                                        });
                                    }
                                }

                                float universe_size = static_cast<float>(step_edge_universe.size());
                                if (universe_size > 0) {
                                    float sum_edge_mi = 0.0f;
                                    int edge_mi_count = 0;

                                    for (size_t i = 0; i < n; ++i) {
                                        const hypergraph::State& state_a = hg.get_state(step_result.state_indices[i]);
                                        if (state_a.id == hypergraph::INVALID_ID) continue;

                                        for (size_t j = i; j < n; ++j) {
                                            const hypergraph::State& state_b = hg.get_state(step_result.state_indices[j]);
                                            if (state_b.id == hypergraph::INVALID_ID) continue;

                                            // Count joint occurrences over edge universe
                                            int n00 = 0, n01 = 0, n10 = 0, n11 = 0;
                                            for (hypergraph::EdgeId eid : step_edge_universe) {
                                                bool in_a = state_a.edges.contains(eid);
                                                bool in_b = state_b.edges.contains(eid);
                                                if (!in_a && !in_b) ++n00;
                                                else if (!in_a && in_b) ++n01;
                                                else if (in_a && !in_b) ++n10;
                                                else ++n11;
                                            }

                                            // Compute MI
                                            float p00 = n00 / universe_size, p01 = n01 / universe_size;
                                            float p10 = n10 / universe_size, p11 = n11 / universe_size;
                                            float p_a0 = p00 + p01, p_a1 = p10 + p11;
                                            float p_b0 = p00 + p10, p_b1 = p01 + p11;

                                            float mi = 0.0f;
                                            auto add_term = [&](float pj, float px, float py) {
                                                if (pj > 0 && px > 0 && py > 0) mi += pj * std::log2(pj / (px * py));
                                            };
                                            add_term(p00, p_a0, p_b0);
                                            add_term(p01, p_a0, p_b1);
                                            add_term(p10, p_a1, p_b0);
                                            add_term(p11, p_a1, p_b1);
                                            mi = std::max(0.0f, mi);

                                            step_result.edge_mutual_information_matrix[i][j] = mi;
                                            step_result.edge_mutual_information_matrix[j][i] = mi;

                                            if (i != j) {
                                                sum_edge_mi += mi;
                                                ++edge_mi_count;
                                                if (mi > step_result.max_edge_mutual_information) {
                                                    step_result.max_edge_mutual_information = mi;
                                                }
                                            }
                                        }
                                    }

                                    if (edge_mi_count > 0) {
                                        step_result.mean_edge_mutual_information = sum_edge_mi / edge_mi_count;
                                    }
                                }
                            }

                            hilbert_per_timestep[step] = std::move(step_result);
                        }

                        step_hilbert_js.wait_for_completion();
                    }
                }

                // Branchial Analysis (distribution sharpness, branch entropy)
                if (compute_branchial) {
#ifdef HAVE_WSTP
                    if (show_progress) {
                        print_to_frontend(libData, "HGEvolve: Computing branchial analysis...");
                    }
#endif
                    // Create job system for parallel branchial computation
                    job_system::JobSystem<int> branchial_js(std::thread::hardware_concurrency());
                    branchial_js.start();

                    branchial_result = bh::analyze_branchial_parallel(branch_states, &branchial_js);
                    has_branchial_data = (branchial_result.num_unique_vertices > 0);

                    // Per-timestep branchial analysis (if scope is "PerTimestep" or "Both")
                    bool include_per_timestep_branchial = (branchial_scope == "PerTimestep" || branchial_scope == "Both");
                    if (include_per_timestep_branchial) {
#ifdef HAVE_WSTP
                        if (show_progress) {
                            print_to_frontend(libData, "HGEvolve: Computing per-timestep branchial analysis...");
                        }
#endif
                        // Group branch_states by timestep
                        std::unordered_map<uint32_t, std::vector<bh::BranchState>> states_by_step;
                        for (const auto& bs : branch_states) {
                            states_by_step[bs.step].push_back(bs);
                        }

                        // Analyze each timestep separately (reuse same job system)
                        for (const auto& [step, step_states] : states_by_step) {
                            if (step_states.empty()) continue;
                            auto step_result = bh::analyze_branchial_parallel(step_states, &branchial_js);
                            if (step_result.num_unique_vertices > 0) {
                                branchial_per_timestep[step] = std::move(step_result);
                            }
                        }
                    }

                    branchial_js.wait_for_completion();
                }

                // Multispace Analysis (vertex/edge probabilities across branches)
                if (compute_multispace) {
#ifdef HAVE_WSTP
                    if (show_progress) {
                        print_to_frontend(libData, "HGEvolve: Computing multispace analysis...");
                    }
#endif
                    // Compute vertex probabilities (how often each vertex appears across branches)
                    std::unordered_map<bh::VertexId, int> vertex_counts;
                    std::map<std::pair<uint32_t, uint32_t>, int> edge_counts;
                    int total_branches = static_cast<int>(branch_states.size());

                    for (const auto& bs : branch_states) {
                        for (bh::VertexId v : bs.vertices) {
                            vertex_counts[v]++;
                        }
                        for (const auto& e : bs.edges) {
                            auto key = std::make_pair(std::min(e.v1, e.v2), std::max(e.v1, e.v2));
                            edge_counts[key]++;
                        }
                    }

                    // Convert counts to probabilities
                    float vertex_entropy_sum = 0.0f;
                    for (const auto& [v, count] : vertex_counts) {
                        float prob = static_cast<float>(count) / total_branches;
                        multispace_vertex_probs[v] = prob;
                        multispace_mean_vertex_prob += prob;
                        // Entropy contribution: -p * log(p)
                        if (prob > 0 && prob < 1) {
                            vertex_entropy_sum -= prob * std::log2(prob);
                        }
                    }
                    if (!vertex_counts.empty()) {
                        multispace_mean_vertex_prob /= vertex_counts.size();
                    }

                    for (const auto& [e, count] : edge_counts) {
                        float prob = static_cast<float>(count) / total_branches;
                        multispace_edge_probs[e] = prob;
                        multispace_mean_edge_prob += prob;
                    }
                    if (!edge_counts.empty()) {
                        multispace_mean_edge_prob /= edge_counts.size();
                    }

                    multispace_total_entropy = vertex_entropy_sum;
                    has_multispace_data = !multispace_vertex_probs.empty();

                    // Per-timestep multispace analysis (if scope is "PerTimestep" or "Both")
                    bool include_per_timestep_multispace = (multispace_scope == "PerTimestep" || multispace_scope == "Both");
                    if (include_per_timestep_multispace) {
#ifdef HAVE_WSTP
                        if (show_progress) {
                            print_to_frontend(libData, "HGEvolve: Computing per-timestep multispace analysis...");
                        }
#endif
                        // Group branch_states by timestep
                        std::unordered_map<uint32_t, std::vector<const bh::BranchState*>> states_by_step;
                        for (const auto& bs : branch_states) {
                            states_by_step[bs.step].push_back(&bs);
                        }

                        // Compute multispace for each timestep separately
                        for (const auto& [step, step_states] : states_by_step) {
                            if (step_states.empty()) continue;

                            MultispaceTimestepData step_data;
                            std::unordered_map<bh::VertexId, int> step_vertex_counts;
                            std::map<std::pair<uint32_t, uint32_t>, int> step_edge_counts;
                            int step_total = static_cast<int>(step_states.size());

                            for (const auto* bs : step_states) {
                                for (bh::VertexId v : bs->vertices) {
                                    step_vertex_counts[v]++;
                                }
                                for (const auto& e : bs->edges) {
                                    auto key = std::make_pair(std::min(e.v1, e.v2), std::max(e.v1, e.v2));
                                    step_edge_counts[key]++;
                                }
                            }

                            // Convert counts to probabilities
                            float step_entropy_sum = 0.0f;
                            for (const auto& [v, count] : step_vertex_counts) {
                                float prob = static_cast<float>(count) / step_total;
                                step_data.vertex_probs[v] = prob;
                                step_data.mean_vertex_prob += prob;
                                if (prob > 0 && prob < 1) {
                                    step_entropy_sum -= prob * std::log2(prob);
                                }
                            }
                            if (!step_vertex_counts.empty()) {
                                step_data.mean_vertex_prob /= step_vertex_counts.size();
                            }

                            for (const auto& [e, count] : step_edge_counts) {
                                float prob = static_cast<float>(count) / step_total;
                                step_data.edge_probs[e] = prob;
                                step_data.mean_edge_prob += prob;
                            }
                            if (!step_edge_counts.empty()) {
                                step_data.mean_edge_prob /= step_edge_counts.size();
                            }

                            step_data.total_entropy = step_entropy_sum;
                            multispace_per_timestep[step] = std::move(step_data);
                        }
                    }
                }
            }

                // Equilibrium Analysis
                if (compute_equilibrium) {
#ifdef HAVE_WSTP
                    if (show_progress) {
                        print_to_frontend(libData, "HGEvolve: Computing equilibrium analysis...");
                    }
#endif
                    bh::EquilibriumConfig eq_config;
                    eq_config.stability_window = equilibrium_window;
                    eq_config.equilibrium_threshold = equilibrium_threshold;
                    eq_config.compute_branchial_metrics = true;
                    eq_config.compute_hilbert_metrics = true;

                    // Build timestep aggregations from the hypergraph states
                    // Group states by step and aggregate their data
                    std::map<uint32_t, std::vector<const hypergraph::State*>> states_by_step;
                    for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
                        const hypergraph::State& state = hg.get_state(sid);
                        if (state.id != hypergraph::INVALID_ID) {
                            states_by_step[state.step].push_back(&state);
                        }
                    }

                    // For each step, compute metrics
                    for (const auto& [step, states_at_step] : states_by_step) {
                        bh::TimestepMetrics metrics;
                        metrics.step = step;
                        metrics.state_count = states_at_step.size();

                        // Collect all vertices and edges at this step
                        std::unordered_set<uint32_t> all_vertices;
                        std::set<std::pair<uint32_t, uint32_t>> all_edges;

                        for (const auto* state : states_at_step) {
                            state->edges.for_each([&](hypergraph::EdgeId eid) {
                                const hypergraph::Edge& edge = hg.get_edge(eid);
                                for (size_t i = 0; i < edge.arity; ++i) {
                                    all_vertices.insert(edge.vertices[i]);
                                }
                                if (edge.arity >= 2) {
                                    uint32_t v1 = edge.vertices[0], v2 = edge.vertices[1];
                                    if (v1 > v2) std::swap(v1, v2);
                                    all_edges.insert(std::make_pair(v1, v2));
                                }
                            });
                        }

                        metrics.vertex_count = all_vertices.size();
                        metrics.edge_count = all_edges.size();

                        // Compute dimension statistics if we have state_dimension_stats
                        if (!state_dimension_stats.empty()) {
                            std::vector<float> dims;
                            for (const auto* state : states_at_step) {
                                auto it = state_dimension_stats.find(state->id);
                                if (it != state_dimension_stats.end()) {
                                    dims.push_back(it->second.mean);
                                }
                            }
                            if (!dims.empty()) {
                                float sum = 0;
                                for (float d : dims) sum += d;
                                metrics.mean_dimension = sum / dims.size();

                                float var_sum = 0;
                                for (float d : dims) {
                                    float diff = d - metrics.mean_dimension;
                                    var_sum += diff * diff;
                                }
                                metrics.dimension_variance = var_sum / dims.size();
                            }
                        }

                        // Build SimpleGraph for degree entropy
                        std::vector<bh::VertexId> verts(all_vertices.begin(), all_vertices.end());
                        std::vector<bh::Edge> edges;
                        for (const auto& e : all_edges) {
                            edges.push_back({e.first, e.second});
                        }
                        bh::SimpleGraph graph;
                        graph.build(verts, edges);
                        metrics.degree_entropy = bh::compute_degree_entropy(graph);

                        if (graph.vertex_count() > 0) {
                            float deg_sum = 0;
                            for (auto v : graph.vertices()) {
                                deg_sum += graph.neighbors(v).size();
                            }
                            metrics.mean_degree = deg_sum / graph.vertex_count();
                        }

                        equilibrium_result.history.push_back(metrics);
                    }

                    // Compute stability scores
                    bh::compute_stability_scores(equilibrium_result, eq_config.stability_window);
                    bh::detect_equilibrium(equilibrium_result, eq_config.equilibrium_threshold,
                                          eq_config.min_steps_for_equilibrium);
                    has_equilibrium_data = !equilibrium_result.history.empty();
                }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolve: Branchial analyses complete.";
                if (has_hilbert_data) {
                    oss << " Hilbert: " << hilbert_result.num_states << " states.";
                }
                if (has_branchial_data) {
                    oss << " Branchial: " << branchial_result.num_unique_vertices << " vertices.";
                }
                if (has_multispace_data) {
                    oss << " Multispace: " << multispace_vertex_probs.size() << " vertices.";
                }
                if (has_equilibrium_data) {
                    oss << " Equilibrium: " << equilibrium_result.history.size() << " steps, "
                        << (equilibrium_result.is_equilibrated ? "EQUILIBRATED" : "not equilibrated") << ".";
                }
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // Build WXF output - only include requested data components
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::WXFValueAssociation full_result;

        // States -> Association[state_id -> state_data]
        // Send ALL states (not just canonical) - WL uses CanonicalId/ContentStateId for vertex merging
        // Each state includes: Id, CanonicalId, ContentStateId, Step, Edges, IsInitial
        // - CanonicalId: isomorphism-based (for Full mode) - isomorphic states share ID
        // - ContentStateId: content-based (for Automatic mode) - same-content states share ID
        // This matches reference behavior where canonicalization is applied at display time
        if (include_states) {
            wxf::WXFValueAssociation states_assoc;
            uint32_t num_states = hg.num_states();

            // Single pass: compute content hash for each state and build mapping
            // Uses the library's get_state_content_hash which is the SAME function
            // used during evolution for Automatic state deduplication, ensuring consistency.
            std::unordered_map<uint64_t, hypergraph::StateId> content_hash_to_id;
            std::vector<uint64_t> state_content_hashes(num_states, 0);
            content_hash_to_id.reserve(num_states);

            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Use the library's content hash function (same as evolution-time deduplication)
                // This ensures FFI ContentStateId matches the grouping done during evolution
                uint64_t hash = hg.get_state_content_hash(sid);

                state_content_hashes[sid] = hash;
                if (content_hash_to_id.find(hash) == content_hash_to_id.end()) {
                    content_hash_to_id[hash] = sid;
                }
            }

            // Export states with both canonical IDs
            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Get canonical state ID (isomorphism-based)
                hypergraph::StateId canonical_id = hg.get_canonical_state(sid);

                // Get content state ID (content-based) - from cached hash
                hypergraph::StateId content_id = content_hash_to_id[state_content_hashes[sid]];

                // Build edge list: each edge is {edge_id, v1, v2, ...}
                // When ReturnCanonicalStates is enabled, edges are IR-canonicalized
                // (vertices relabeled to 0..n-1, edges sorted) with sequential edge IDs
                wxf::WXFValueList edge_list;
                if (hg.return_canonical_states()) {
                    std::vector<std::vector<hypergraph::VertexId>> edge_vecs;
                    state.edges.for_each([&](hypergraph::EdgeId eid) {
                        const hypergraph::Edge& e = hg.get_edge(eid);
                        edge_vecs.emplace_back(e.vertices, e.vertices + e.arity);
                    });

                    if (!edge_vecs.empty()) {
                        hypergraph::IRCanonicalizer ir;
                        auto canon_result = ir.canonicalize_edges(edge_vecs);
                        int64_t edge_idx = 0;
                        for (const auto& canon_edge : canon_result.canonical_form.edges) {
                            wxf::WXFValueList edge_data;
                            edge_data.push_back(wxf::WXFValue(edge_idx++));
                            for (auto v : canon_edge) {
                                edge_data.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                            }
                            edge_list.push_back(wxf::WXFValue(edge_data));
                        }
                    }
                } else {
                    state.edges.for_each([&](hypergraph::EdgeId eid) {
                        const hypergraph::Edge& edge = hg.get_edge(eid);
                        wxf::WXFValueList edge_data;
                        edge_data.push_back(wxf::WXFValue(static_cast<int64_t>(eid)));
                        for (uint8_t i = 0; i < edge.arity; ++i) {
                            edge_data.push_back(wxf::WXFValue(static_cast<int64_t>(edge.vertices[i])));
                        }
                        edge_list.push_back(wxf::WXFValue(edge_data));
                    });
                }

                wxf::WXFValueAssociation state_assoc;
                state_assoc.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(sid))});
                state_assoc.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(static_cast<int64_t>(canonical_id))});
                state_assoc.push_back({wxf::WXFValue("ContentStateId"), wxf::WXFValue(static_cast<int64_t>(content_id))});
                state_assoc.push_back({wxf::WXFValue("Step"), wxf::WXFValue(static_cast<int64_t>(state.step))});
                state_assoc.push_back({wxf::WXFValue("Edges"), wxf::WXFValue(edge_list)});
                state_assoc.push_back({wxf::WXFValue("IsInitial"), wxf::WXFValue(state.step == 0)});

                states_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(sid)), wxf::WXFValue(state_assoc)});
            }
            full_result.push_back({wxf::WXFValue("States"), wxf::WXFValue(states_assoc)});
        }

        // Events -> Association[event_id -> event_data]
        // Only canonical events are sent (for graph vertices)
        // State IDs are mapped through get_canonical_state() so edges connect canonical states
        if (include_events) {
            // Send ALL events (not just canonical) - WL uses CanonicalId for vertex merging
            // This preserves event multiplicity: multiple events with same canonical ID
            // map to one vertex, but their edges to different output states are preserved.
            wxf::WXFValueAssociation events_assoc;
            uint32_t num_raw_events = hg.num_raw_events();
            for (uint32_t eid = 0; eid < num_raw_events; ++eid) {
                const hypergraph::Event& event = hg.get_event(eid);
                if (event.id == hypergraph::INVALID_ID) continue;
                // Skip genesis events if ShowGenesisEvents is false
                if (!show_genesis_events && hg.is_genesis_event(eid)) continue;

                // Send BOTH raw and canonical state IDs - WL chooses which to use per graph type
                // Raw IDs are for edge connectivity to actual states
                // Canonical IDs are for when state canonicalization is enabled (merging isomorphic states)
                int64_t raw_input_state_id = static_cast<int64_t>(event.input_state);
                int64_t raw_output_state_id = static_cast<int64_t>(event.output_state);
                int64_t canonical_input_state_id = static_cast<int64_t>(hg.get_canonical_state(event.input_state));
                int64_t canonical_output_state_id = static_cast<int64_t>(hg.get_canonical_state(event.output_state));

                // Canonical event ID: for canonical events use own ID, for duplicates use the canonical's ID
                int64_t canonical_event_id = event.is_canonical()
                    ? static_cast<int64_t>(eid)
                    : static_cast<int64_t>(event.canonical_event_id);

                wxf::WXFValueAssociation event_data;
                event_data.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(eid))});
                event_data.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(canonical_event_id)});
                event_data.push_back({wxf::WXFValue("RuleIndex"), wxf::WXFValue(static_cast<int64_t>(event.rule_index))});
                // Raw state IDs (always included)
                event_data.push_back({wxf::WXFValue("InputState"), wxf::WXFValue(raw_input_state_id)});
                event_data.push_back({wxf::WXFValue("OutputState"), wxf::WXFValue(raw_output_state_id)});
                // Canonical state IDs (for when state canonicalization is enabled)
                event_data.push_back({wxf::WXFValue("CanonicalInputState"), wxf::WXFValue(canonical_input_state_id)});
                event_data.push_back({wxf::WXFValue("CanonicalOutputState"), wxf::WXFValue(canonical_output_state_id)});

                // Consumed/produced edges
                wxf::WXFValueList consumed_list, produced_list;
                for (uint8_t i = 0; i < event.num_consumed; ++i) {
                    consumed_list.push_back(wxf::WXFValue(static_cast<int64_t>(event.consumed_edges[i])));
                }
                for (uint8_t i = 0; i < event.num_produced; ++i) {
                    produced_list.push_back(wxf::WXFValue(static_cast<int64_t>(event.produced_edges[i])));
                }
                event_data.push_back({wxf::WXFValue("ConsumedEdges"), wxf::WXFValue(consumed_list)});
                event_data.push_back({wxf::WXFValue("ProducedEdges"), wxf::WXFValue(produced_list)});

                events_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(eid)), wxf::WXFValue(event_data)});
            }
            full_result.push_back({wxf::WXFValue("Events"), wxf::WXFValue(events_assoc)});
        }

        // EventsMinimal -> Association[event_id -> {Id, CanonicalId, RuleIndex, InputState, OutputState, CanonicalInputState, CanonicalOutputState}]
        // Reduced event data for graph structure variants that don't need full event details
        // Send ALL events - WL uses CanonicalId for vertex merging, RuleIndex for Event=Automatic grouping
        if (include_events_minimal && !include_events) {
            wxf::WXFValueAssociation events_assoc;
            uint32_t num_raw_events = hg.num_raw_events();
            for (uint32_t eid = 0; eid < num_raw_events; ++eid) {
                const hypergraph::Event& event = hg.get_event(eid);
                if (event.id == hypergraph::INVALID_ID) continue;
                if (!show_genesis_events && hg.is_genesis_event(eid)) continue;

                // Send BOTH raw and canonical state IDs - WL chooses which to use per graph type
                int64_t raw_input_state_id = static_cast<int64_t>(event.input_state);
                int64_t raw_output_state_id = static_cast<int64_t>(event.output_state);
                int64_t canonical_input_state_id = static_cast<int64_t>(hg.get_canonical_state(event.input_state));
                int64_t canonical_output_state_id = static_cast<int64_t>(hg.get_canonical_state(event.output_state));

                // Canonical event ID: for canonical events use own ID, for duplicates use the canonical's ID
                int64_t canonical_event_id = event.is_canonical()
                    ? static_cast<int64_t>(eid)
                    : static_cast<int64_t>(event.canonical_event_id);

                wxf::WXFValueAssociation event_data;
                event_data.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(eid))});
                event_data.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(canonical_event_id)});
                event_data.push_back({wxf::WXFValue("RuleIndex"), wxf::WXFValue(static_cast<int64_t>(event.rule_index))});
                // Raw state IDs (always included)
                event_data.push_back({wxf::WXFValue("InputState"), wxf::WXFValue(raw_input_state_id)});
                event_data.push_back({wxf::WXFValue("OutputState"), wxf::WXFValue(raw_output_state_id)});
                // Canonical state IDs (for when state canonicalization is enabled)
                event_data.push_back({wxf::WXFValue("CanonicalInputState"), wxf::WXFValue(canonical_input_state_id)});
                event_data.push_back({wxf::WXFValue("CanonicalOutputState"), wxf::WXFValue(canonical_output_state_id)});

                events_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(eid)), wxf::WXFValue(event_data)});
            }
            full_result.push_back({wxf::WXFValue("Events"), wxf::WXFValue(events_assoc)});
        }

        // CausalEdges -> List of {From -> canonical_event_id, To -> canonical_event_id}
        // Endpoints are mapped to canonical event IDs for graph structure
        // Deduplicated by RAW (producer, consumer) pairs to remove internal duplication
        // but preserving all unique raw relationships (which may share canonical endpoints)
        if (include_causal_edges) {
            wxf::WXFValueList causal_edges;
            auto causal_edge_vec = hg.causal_graph().get_causal_edges();

            // Deduplicate by RAW event pairs (not canonical) - this removes internal doubling
            // while preserving edges that happen to have the same canonical endpoints
            auto pair_hash = [](const std::pair<hypergraph::EventId, hypergraph::EventId>& p) {
                return std::hash<uint64_t>{}((static_cast<uint64_t>(p.first) << 32) | p.second);
            };
            std::unordered_set<std::pair<hypergraph::EventId, hypergraph::EventId>, decltype(pair_hash)> seen_raw_pairs(
                causal_edge_vec.size(), pair_hash);

            for (const auto& edge : causal_edge_vec) {
                // Skip edges involving genesis events if ShowGenesisEvents is false
                if (!show_genesis_events &&
                    (hg.is_genesis_event(edge.producer) || hg.is_genesis_event(edge.consumer))) {
                    continue;
                }

                // Deduplicate by RAW event pair
                auto raw_pair = std::make_pair(edge.producer, edge.consumer);
                if (seen_raw_pairs.count(raw_pair)) continue;
                seen_raw_pairs.insert(raw_pair);

                // Map to canonical event IDs for output (also include raw for flexibility)
                hypergraph::EventId canonical_from = hg.get_canonical_event(edge.producer);
                hypergraph::EventId canonical_to = hg.get_canonical_event(edge.consumer);

                wxf::WXFValueAssociation edge_data;
                edge_data.push_back({wxf::WXFValue("From"), wxf::WXFValue(static_cast<int64_t>(canonical_from))});
                edge_data.push_back({wxf::WXFValue("To"), wxf::WXFValue(static_cast<int64_t>(canonical_to))});
                // Also include raw event IDs for when event canonicalization is disabled
                edge_data.push_back({wxf::WXFValue("RawFrom"), wxf::WXFValue(static_cast<int64_t>(edge.producer))});
                edge_data.push_back({wxf::WXFValue("RawTo"), wxf::WXFValue(static_cast<int64_t>(edge.consumer))});
                causal_edges.push_back(wxf::WXFValue(edge_data));
            }
            full_result.push_back({wxf::WXFValue("CausalEdges"), wxf::WXFValue(causal_edges)});
        }

        // BranchialEdges -> List of {From -> canonical_event_id, To -> canonical_event_id}
        // For Evolution*Branchial graphs where event vertices need canonical IDs
        // NO deduplication - multiplicity matters for branchial edges
        if (include_branchial_edges) {
            wxf::WXFValueList branchial_edges;
            auto branchial_edge_vec = hg.causal_graph().get_branchial_edges();
            for (const auto& edge : branchial_edge_vec) {
                // Skip edges involving genesis events if ShowGenesisEvents is false
                if (!show_genesis_events &&
                    (hg.is_genesis_event(edge.event1) || hg.is_genesis_event(edge.event2))) {
                    continue;
                }
                // Map to canonical event IDs
                hypergraph::EventId canonical_from = hg.get_canonical_event(edge.event1);
                hypergraph::EventId canonical_to = hg.get_canonical_event(edge.event2);

                wxf::WXFValueAssociation edge_data;
                edge_data.push_back({wxf::WXFValue("From"), wxf::WXFValue(static_cast<int64_t>(canonical_from))});
                edge_data.push_back({wxf::WXFValue("To"), wxf::WXFValue(static_cast<int64_t>(canonical_to))});
                branchial_edges.push_back(wxf::WXFValue(edge_data));
            }
            full_result.push_back({wxf::WXFValue("BranchialEdges"), wxf::WXFValue(branchial_edges)});
        }

        // BranchialStateEdges -> List of {From -> canonical_state_id, To -> canonical_state_id}
        // BranchialStateVertices -> List of unique state IDs that appear in branchial edges
        // For BranchialGraph where state vertices are the output states of events
        // NOTE: Do NOT deduplicate by canonical state pair - reference preserves edge multiplicity
        if (include_branchial_state_edges) {
            wxf::WXFValueList branchial_state_edges;
            std::set<hypergraph::StateId> unique_states;
            auto branchial_edge_vec = hg.causal_graph().get_branchial_edges();

            // Compute target step for filtering (0 = all, positive = 1-based, negative = from end)
            uint32_t target_step = 0;
            bool filter_by_step = (branchial_step != 0);
            if (filter_by_step) {
                if (branchial_step > 0) {
                    // 1-based indexing: 1 = step 1, 2 = step 2, etc.
                    target_step = static_cast<uint32_t>(branchial_step);
                } else {
                    // Negative from end: -1 = final step (steps), -2 = steps-1, etc.
                    target_step = static_cast<uint32_t>(steps + 1 + branchial_step);
                }
            }

            for (const auto& edge : branchial_edge_vec) {
                // Skip edges involving genesis events if ShowGenesisEvents is false
                if (!show_genesis_events &&
                    (hg.is_genesis_event(edge.event1) || hg.is_genesis_event(edge.event2))) {
                    continue;
                }
                // Get the output states of the events, mapped to canonical state IDs
                const hypergraph::Event& event1 = hg.get_event(edge.event1);
                const hypergraph::Event& event2 = hg.get_event(edge.event2);

                // Filter by step if specified (branchial edges are between events at the same step)
                if (filter_by_step) {
                    const hypergraph::State& output_state = hg.get_state(event1.output_state);
                    if (output_state.step != target_step) {
                        continue;
                    }
                }

                hypergraph::StateId state1 = hg.get_canonical_state(event1.output_state);
                hypergraph::StateId state2 = hg.get_canonical_state(event2.output_state);

                // Track unique states for vertices
                unique_states.insert(state1);
                unique_states.insert(state2);

                // No deduplication - preserve edge multiplicity like reference
                wxf::WXFValueAssociation edge_data;
                edge_data.push_back({wxf::WXFValue("From"), wxf::WXFValue(static_cast<int64_t>(state1))});
                edge_data.push_back({wxf::WXFValue("To"), wxf::WXFValue(static_cast<int64_t>(state2))});
                branchial_state_edges.push_back(wxf::WXFValue(edge_data));
            }
            full_result.push_back({wxf::WXFValue("BranchialStateEdges"), wxf::WXFValue(branchial_state_edges)});

            // Send unique state vertices
            wxf::WXFValueList state_vertices;
            for (hypergraph::StateId sid : unique_states) {
                state_vertices.push_back(wxf::WXFValue(static_cast<int64_t>(sid)));
            }
            full_result.push_back({wxf::WXFValue("BranchialStateVertices"), wxf::WXFValue(state_vertices)});
        }

        // BranchialStateEdgesAllSiblings: ALL pairs of output states from same input state
        // This matches reference BranchialGraph behavior (no overlap check, all siblings)
        if (include_branchial_state_edges_all_siblings) {
            wxf::WXFValueList branchial_state_edges;
            std::set<hypergraph::StateId> unique_states;

            // Compute target step for filtering (0 = all, positive = 1-based, negative = from end)
            uint32_t target_step = 0;
            bool filter_by_step = (branchial_step != 0);
            if (filter_by_step) {
                if (branchial_step > 0) {
                    target_step = static_cast<uint32_t>(branchial_step);
                } else {
                    target_step = static_cast<uint32_t>(steps + 1 + branchial_step);
                }
            }

            // Iterate over all input states and their events
            hg.causal_graph().for_each_state_events([&]([[maybe_unused]] hypergraph::StateId input_state, auto* event_list) {
                // Collect all events from this input state
                std::vector<hypergraph::EventId> events;
                event_list->for_each([&](hypergraph::EventId eid) {
                    // Skip genesis events if not showing them
                    if (!show_genesis_events && hg.is_genesis_event(eid)) {
                        return;
                    }
                    events.push_back(eid);
                });

                // Create all pairs of output states (C(n,2) pairs)
                for (size_t i = 0; i < events.size(); ++i) {
                    const hypergraph::Event& event1 = hg.get_event(events[i]);

                    // Filter by step if specified
                    if (filter_by_step) {
                        const hypergraph::State& output_state = hg.get_state(event1.output_state);
                        if (output_state.step != target_step) {
                            continue;
                        }
                    }

                    hypergraph::StateId state1 = hg.get_canonical_state(event1.output_state);

                    for (size_t j = i + 1; j < events.size(); ++j) {
                        const hypergraph::Event& event2 = hg.get_event(events[j]);

                        // Filter event2 by step too
                        if (filter_by_step) {
                            const hypergraph::State& output_state2 = hg.get_state(event2.output_state);
                            if (output_state2.step != target_step) {
                                continue;
                            }
                        }

                        hypergraph::StateId state2 = hg.get_canonical_state(event2.output_state);

                        // Track unique states
                        unique_states.insert(state1);
                        unique_states.insert(state2);

                        // Add edge (no deduplication - preserve multiplicity like reference)
                        wxf::WXFValueAssociation edge_data;
                        edge_data.push_back({wxf::WXFValue("From"), wxf::WXFValue(static_cast<int64_t>(state1))});
                        edge_data.push_back({wxf::WXFValue("To"), wxf::WXFValue(static_cast<int64_t>(state2))});
                        branchial_state_edges.push_back(wxf::WXFValue(edge_data));
                    }
                }
            });

            full_result.push_back({wxf::WXFValue("BranchialStateEdges"), wxf::WXFValue(branchial_state_edges)});

            // Send unique state vertices
            wxf::WXFValueList state_vertices;
            for (hypergraph::StateId sid : unique_states) {
                state_vertices.push_back(wxf::WXFValue(static_cast<int64_t>(sid)));
            }
            full_result.push_back({wxf::WXFValue("BranchialStateVertices"), wxf::WXFValue(state_vertices)});
        }

        // ========================================================================
        // GraphData - Graph-ready data for direct Graph[] construction in WL
        // ========================================================================
        if (!graph_properties.empty()) {
            // Compute content hashes for Automatic mode (if not already computed)
            uint32_t num_states = hg.num_states();
            std::unordered_map<uint64_t, hypergraph::StateId> gd_content_hash_to_id;
            std::vector<uint64_t> gd_state_content_hashes(num_states, 0);

            if (canonicalize_states_mode == "Automatic") {
                gd_content_hash_to_id.reserve(num_states);
                for (uint32_t sid = 0; sid < num_states; ++sid) {
                    const hypergraph::State& state = hg.get_state(sid);
                    if (state.id == hypergraph::INVALID_ID) continue;
                    uint64_t hash = hg.get_state_content_hash(sid);
                    gd_state_content_hashes[sid] = hash;
                    if (gd_content_hash_to_id.find(hash) == gd_content_hash_to_id.end()) {
                        gd_content_hash_to_id[hash] = sid;
                    }
                }
            }

            // Helper: Get effective state ID based on canonicalization mode
            auto get_effective_state_id = [&](hypergraph::StateId sid) -> int64_t {
                if (canonicalize_states_mode == "Full")
                    return static_cast<int64_t>(hg.get_canonical_state(sid));
                if (canonicalize_states_mode == "Automatic")
                    return static_cast<int64_t>(gd_content_hash_to_id[gd_state_content_hashes[sid]]);
                return static_cast<int64_t>(sid);
            };

            // Helper: Get effective event ID based on event canonicalization
            // Note: EVENT_SIG_FULL uses InputState/OutputState which require canonical state IDs.
            // When CanonicalizeStates=None, we must use raw event IDs because canonical_event_id
            // was computed using canonical state IDs during evolution.
            auto get_effective_event_id = [&](hypergraph::EventId eid) -> int64_t {
                if (canonicalize_states_mode == "None" || event_signature_keys == hypergraph::EVENT_SIG_NONE)
                    return static_cast<int64_t>(eid);
                const hypergraph::Event& e = hg.get_event(eid);
                return e.is_canonical() ? static_cast<int64_t>(eid)
                                        : static_cast<int64_t>(e.canonical_event_id);
            };

            // Helper: Serialize state edges as list of {edgeId, v1, v2, ...}
            // When ReturnCanonicalStates is enabled, emits IR-canonicalized edges
            auto serialize_state_edges = [&](hypergraph::StateId sid) -> wxf::WXFValueList {
                wxf::WXFValueList edge_list;
                if (hg.return_canonical_states()) {
                    std::vector<std::vector<hypergraph::VertexId>> edge_vecs;
                    hg.get_state(sid).edges.for_each([&](hypergraph::EdgeId eid) {
                        const hypergraph::Edge& edge = hg.get_edge(eid);
                        edge_vecs.emplace_back(edge.vertices, edge.vertices + edge.arity);
                    });
                    if (!edge_vecs.empty()) {
                        hypergraph::IRCanonicalizer ir;
                        auto canon_result = ir.canonicalize_edges(edge_vecs);
                        int64_t edge_idx = 0;
                        for (const auto& canon_edge : canon_result.canonical_form.edges) {
                            wxf::WXFValueList e;
                            e.push_back(wxf::WXFValue(edge_idx++));
                            for (auto v : canon_edge)
                                e.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                            edge_list.push_back(wxf::WXFValue(e));
                        }
                    }
                } else {
                    hg.get_state(sid).edges.for_each([&](hypergraph::EdgeId eid) {
                        const hypergraph::Edge& edge = hg.get_edge(eid);
                        wxf::WXFValueList e;
                        e.push_back(wxf::WXFValue(static_cast<int64_t>(eid)));
                        for (uint8_t i = 0; i < edge.arity; ++i)
                            e.push_back(wxf::WXFValue(static_cast<int64_t>(edge.vertices[i])));
                        edge_list.push_back(wxf::WXFValue(e));
                    });
                }
                return edge_list;
            };

            // Helper: Serialize state data for tooltips
            auto serialize_state_data = [&](hypergraph::StateId sid) -> wxf::WXFValueAssociation {
                const hypergraph::State& state = hg.get_state(sid);
                wxf::WXFValueAssociation d;
                d.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(sid))});
                d.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(static_cast<int64_t>(hg.get_canonical_state(sid)))});
                d.push_back({wxf::WXFValue("Step"), wxf::WXFValue(static_cast<int64_t>(state.step))});
                d.push_back({wxf::WXFValue("Edges"), wxf::WXFValue(serialize_state_edges(sid))});
                d.push_back({wxf::WXFValue("IsInitial"), wxf::WXFValue(state.step == 0)});
                return d;
            };

            // Helper: Serialize event data for tooltips
            auto serialize_event_data = [&](hypergraph::EventId eid) -> wxf::WXFValueAssociation {
                const hypergraph::Event& e = hg.get_event(eid);
                wxf::WXFValueAssociation d;
                d.push_back({wxf::WXFValue("Id"), wxf::WXFValue(static_cast<int64_t>(eid))});
                d.push_back({wxf::WXFValue("CanonicalId"), wxf::WXFValue(get_effective_event_id(eid))});
                d.push_back({wxf::WXFValue("RuleIndex"), wxf::WXFValue(static_cast<int64_t>(e.rule_index))});
                d.push_back({wxf::WXFValue("InputState"), wxf::WXFValue(static_cast<int64_t>(e.input_state))});
                d.push_back({wxf::WXFValue("OutputState"), wxf::WXFValue(static_cast<int64_t>(e.output_state))});
                // Consumed/produced edges
                wxf::WXFValueList consumed, produced;
                for (uint8_t i = 0; i < e.num_consumed; ++i)
                    consumed.push_back(wxf::WXFValue(static_cast<int64_t>(e.consumed_edges[i])));
                for (uint8_t i = 0; i < e.num_produced; ++i)
                    produced.push_back(wxf::WXFValue(static_cast<int64_t>(e.produced_edges[i])));
                d.push_back({wxf::WXFValue("ConsumedEdges"), wxf::WXFValue(consumed)});
                d.push_back({wxf::WXFValue("ProducedEdges"), wxf::WXFValue(produced)});
                // For styled rendering: include input/output state edges
                d.push_back({wxf::WXFValue("InputStateEdges"), wxf::WXFValue(serialize_state_edges(e.input_state))});
                d.push_back({wxf::WXFValue("OutputStateEdges"), wxf::WXFValue(serialize_state_edges(e.output_state))});
                return d;
            };

            // Helper: Check if event should be included
            auto is_valid_event = [&](hypergraph::EventId eid) -> bool {
                const hypergraph::Event& e = hg.get_event(eid);
                if (e.id == hypergraph::INVALID_ID) return false;
                if (!show_genesis_events && hg.is_genesis_event(eid)) return false;
                return true;
            };

            // Build GraphData for each requested graph property
            wxf::WXFValueAssociation all_graph_data;

            for (const std::string& graph_property : graph_properties) {
                wxf::WXFValueList vertices;
                wxf::WXFValueList edges;
                wxf::WXFValueAssociation vertex_data;

                // Parse property name
                bool is_states = graph_property.rfind("States", 0) == 0;
                bool is_causal = graph_property.rfind("Causal", 0) == 0;
                bool is_branchial = graph_property.rfind("Branchial", 0) == 0;
                bool is_evolution = graph_property.find("Evolution") != std::string::npos;
                bool has_causal = is_evolution && graph_property.find("Causal") != std::string::npos;
                bool has_branchial = is_evolution && graph_property.find("Branchial") != std::string::npos;

                // Helper to add edge to edges list
                auto add_graph_edge = [&](wxf::WXFValue from, wxf::WXFValue to, const std::string& type,
                                          wxf::WXFValueAssociation data = {}) {
                    wxf::WXFValueAssociation edge;
                    edge.push_back({wxf::WXFValue("From"), from});
                    edge.push_back({wxf::WXFValue("To"), to});
                    edge.push_back({wxf::WXFValue("Type"), wxf::WXFValue(type)});
                    if (!data.empty()) edge.push_back({wxf::WXFValue("Data"), wxf::WXFValue(data)});
                    edges.push_back(wxf::WXFValue(edge));
                };

            // Helper to add causal edges with proper deduplication
            // EdgeDeduplication=True: one edge per raw (producer, consumer) pair
            // EdgeDeduplication=False: N edges for N shared hyperedges
            // Note: Our causal graph stores one CausalEdge per (producer, consumer, hyperedge) triple
            // IMPORTANT: Each edge must have unique data or Mathematica's Graph[] will deduplicate!
            auto add_causal_edges = [&]() {
                // First pass: count CausalEdges per raw (producer, consumer) pair
                std::map<std::pair<hypergraph::EventId, hypergraph::EventId>, size_t> pair_counts;
                for (const auto& ce : hg.causal_graph().get_causal_edges()) {
                    if (!show_genesis_events && (hg.is_genesis_event(ce.producer) || hg.is_genesis_event(ce.consumer))) continue;
                    pair_counts[{ce.producer, ce.consumer}]++;
                }

                // Second pass: emit edges (with unique index to prevent Graph[] deduplication)
                for (const auto& [pair, count] : pair_counts) {
                    int64_t from = get_effective_event_id(pair.first);
                    int64_t to = get_effective_event_id(pair.second);
                    wxf::WXFValueList from_tag = {wxf::WXFValue("E"), wxf::WXFValue(from)};
                    wxf::WXFValueList to_tag = {wxf::WXFValue("E"), wxf::WXFValue(to)};

                    size_t num_edges = edge_deduplication ? 1 : count;
                    for (size_t k = 0; k < num_edges; ++k) {
                        wxf::WXFValueAssociation causal_data;
                        causal_data.push_back({wxf::WXFValue("ProducerEvent"), wxf::WXFValue(from)});
                        causal_data.push_back({wxf::WXFValue("ConsumerEvent"), wxf::WXFValue(to)});
                        // Add unique index to prevent Mathematica Graph[] deduplication
                        if (num_edges > 1) {
                            causal_data.push_back({wxf::WXFValue("EdgeIndex"), wxf::WXFValue(static_cast<int64_t>(k))});
                        }
                        add_graph_edge(wxf::WXFValue(from_tag), wxf::WXFValue(to_tag), "Causal", causal_data);
                    }
                }
            };

            if (is_states) {
                // === STATES GRAPH ===
                // Vertices: states (deduplicated by effective ID)
                std::map<int64_t, hypergraph::StateId> state_verts;
                for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
                    if (hg.get_state(sid).id == hypergraph::INVALID_ID) continue;
                    int64_t eff = get_effective_state_id(sid);
                    if (!state_verts.count(eff)) state_verts[eff] = sid;
                }
                for (auto& [eff, raw] : state_verts) {
                    vertices.push_back(wxf::WXFValue(eff));
                    vertex_data.push_back({wxf::WXFValue(eff), wxf::WXFValue(serialize_state_data(raw))});
                }
                // Edges: events (state → state)
                for (uint32_t eid = 0; eid < hg.num_raw_events(); ++eid) {
                    if (!is_valid_event(eid)) continue;
                    const hypergraph::Event& e = hg.get_event(eid);
                    add_graph_edge(wxf::WXFValue(get_effective_state_id(e.input_state)),
                                   wxf::WXFValue(get_effective_state_id(e.output_state)),
                                   "Directed", serialize_event_data(eid));
                }
            }
            else if (is_causal) {
                // === CAUSAL GRAPH ===
                // Vertices: events (deduplicated, tagged)
                std::map<int64_t, hypergraph::EventId> event_verts;
                for (uint32_t eid = 0; eid < hg.num_raw_events(); ++eid) {
                    if (!is_valid_event(eid)) continue;
                    int64_t eff = get_effective_event_id(eid);
                    if (!event_verts.count(eff)) event_verts[eff] = eid;
                }
                for (auto& [eff, raw] : event_verts) {
                    wxf::WXFValueList tag = {wxf::WXFValue("E"), wxf::WXFValue(eff)};
                    vertices.push_back(wxf::WXFValue(tag));
                    vertex_data.push_back({wxf::WXFValue(tag), wxf::WXFValue(serialize_event_data(raw))});
                }
                // Add causal edges using shared helper
                add_causal_edges();
            }
            else if (is_branchial) {
                // === BRANCHIAL GRAPH ===
                // Vertices: states involved in branchial edges (output states of branchial event pairs)
                std::set<hypergraph::StateId> state_set;
                std::map<int64_t, hypergraph::StateId> state_verts;
                auto branchial_edge_vec = hg.causal_graph().get_branchial_edges();

                // Compute target step for filtering (0 = all, positive = 1-based, negative = from end)
                uint32_t target_step = 0;
                bool filter_by_step = (branchial_step != 0);
                if (filter_by_step) {
                    if (branchial_step > 0) {
                        target_step = static_cast<uint32_t>(branchial_step);
                    } else {
                        target_step = static_cast<uint32_t>(steps + 1 + branchial_step);
                    }
                }

                // First pass: collect all states
                for (const auto& edge : branchial_edge_vec) {
                    if (!show_genesis_events &&
                        (hg.is_genesis_event(edge.event1) || hg.is_genesis_event(edge.event2))) continue;
                    const hypergraph::Event& event1 = hg.get_event(edge.event1);
                    const hypergraph::Event& event2 = hg.get_event(edge.event2);

                    // Filter by step if specified
                    if (filter_by_step) {
                        const hypergraph::State& output_state = hg.get_state(event1.output_state);
                        if (output_state.step != target_step) continue;
                    }

                    // Use effective state IDs based on canonicalization mode
                    int64_t state1 = get_effective_state_id(event1.output_state);
                    int64_t state2 = get_effective_state_id(event2.output_state);
                    if (!state_verts.count(state1))
                        state_verts[state1] = event1.output_state;
                    if (!state_verts.count(state2))
                        state_verts[state2] = event2.output_state;
                }

                // Add vertices with data
                for (auto& [eff, raw] : state_verts) {
                    vertices.push_back(wxf::WXFValue(eff));
                    vertex_data.push_back({wxf::WXFValue(eff), wxf::WXFValue(serialize_state_data(raw))});
                }

                // Edges: branchial state edges (no deduplication - preserve multiplicity)
                for (const auto& edge : branchial_edge_vec) {
                    if (!show_genesis_events &&
                        (hg.is_genesis_event(edge.event1) || hg.is_genesis_event(edge.event2))) continue;
                    const hypergraph::Event& event1 = hg.get_event(edge.event1);
                    const hypergraph::Event& event2 = hg.get_event(edge.event2);

                    // Filter by step if specified
                    if (filter_by_step) {
                        const hypergraph::State& output_state = hg.get_state(event1.output_state);
                        if (output_state.step != target_step) continue;
                    }

                    // Use effective state IDs based on canonicalization mode
                    int64_t state1 = get_effective_state_id(event1.output_state);
                    int64_t state2 = get_effective_state_id(event2.output_state);
                    // Tooltip data: effective state IDs (matches graph vertices)
                    wxf::WXFValueAssociation branchial_data;
                    branchial_data.push_back({wxf::WXFValue("State1"), wxf::WXFValue(state1)});
                    branchial_data.push_back({wxf::WXFValue("State2"), wxf::WXFValue(state2)});
                    add_graph_edge(wxf::WXFValue(state1), wxf::WXFValue(state2), "Branchial", branchial_data);
                }
            }
            else if (is_evolution) {
                // === EVOLUTION GRAPH ===
                // Vertices: states (tagged {"S", id}) + events (tagged {"E", id})
                std::set<int64_t> state_ids;
                std::map<int64_t, hypergraph::StateId> raw_states;
                std::map<int64_t, hypergraph::EventId> event_verts;

                for (uint32_t eid = 0; eid < hg.num_raw_events(); ++eid) {
                    if (!is_valid_event(eid)) continue;
                    const hypergraph::Event& e = hg.get_event(eid);
                    // Track states using effective IDs for proper deduplication
                    int64_t in_id = get_effective_state_id(e.input_state);
                    int64_t out_id = get_effective_state_id(e.output_state);
                    if (!raw_states.count(in_id)) raw_states[in_id] = e.input_state;
                    if (!raw_states.count(out_id)) raw_states[out_id] = e.output_state;
                    state_ids.insert(in_id);
                    state_ids.insert(out_id);
                    int64_t eff = get_effective_event_id(eid);
                    if (!event_verts.count(eff)) event_verts[eff] = eid;
                }

                // Add state vertices
                for (int64_t sid : state_ids) {
                    wxf::WXFValueList tag = {wxf::WXFValue("S"), wxf::WXFValue(sid)};
                    vertices.push_back(wxf::WXFValue(tag));
                    vertex_data.push_back({wxf::WXFValue(tag), wxf::WXFValue(serialize_state_data(raw_states[sid]))});
                }
                // Add event vertices
                for (auto& [eff, raw] : event_verts) {
                    wxf::WXFValueList tag = {wxf::WXFValue("E"), wxf::WXFValue(eff)};
                    vertices.push_back(wxf::WXFValue(tag));
                    vertex_data.push_back({wxf::WXFValue(tag), wxf::WXFValue(serialize_event_data(raw))});
                }

                // Edges: state↔event from each event (use effective IDs for deduplication)
                for (uint32_t eid = 0; eid < hg.num_raw_events(); ++eid) {
                    if (!is_valid_event(eid)) continue;
                    const hypergraph::Event& e = hg.get_event(eid);
                    int64_t eff_eid = get_effective_event_id(eid);
                    wxf::WXFValueList s_in = {wxf::WXFValue("S"), wxf::WXFValue(get_effective_state_id(e.input_state))};
                    wxf::WXFValueList s_out = {wxf::WXFValue("S"), wxf::WXFValue(get_effective_state_id(e.output_state))};
                    wxf::WXFValueList e_tag = {wxf::WXFValue("E"), wxf::WXFValue(eff_eid)};
                    // Tooltip data: effective event ID (matches graph vertices)
                    wxf::WXFValueAssociation edge_data;
                    edge_data.push_back({wxf::WXFValue("EventId"), wxf::WXFValue(eff_eid)});
                    add_graph_edge(wxf::WXFValue(s_in), wxf::WXFValue(e_tag), "StateEvent", edge_data);
                    add_graph_edge(wxf::WXFValue(e_tag), wxf::WXFValue(s_out), "EventState", edge_data);
                }

                // Optional causal edges using shared helper
                if (has_causal) {
                    add_causal_edges();
                }

                // Optional branchial edges
                if (has_branchial) {
                    // Compute target step for filtering (same logic as BranchialGraph)
                    uint32_t target_step = 0;
                    bool filter_by_step = (branchial_step != 0);
                    if (filter_by_step) {
                        if (branchial_step > 0) {
                            target_step = static_cast<uint32_t>(branchial_step);
                        } else {
                            target_step = static_cast<uint32_t>(steps + 1 + branchial_step);
                        }
                    }

                    for (const auto& be : hg.causal_graph().get_branchial_edges()) {
                        if (!show_genesis_events && (hg.is_genesis_event(be.event1) || hg.is_genesis_event(be.event2))) continue;

                        // Filter by step if specified
                        if (filter_by_step) {
                            const hypergraph::Event& event1 = hg.get_event(be.event1);
                            const hypergraph::State& output_state = hg.get_state(event1.output_state);
                            if (output_state.step != target_step) continue;
                        }

                        int64_t from = get_effective_event_id(be.event1);
                        int64_t to = get_effective_event_id(be.event2);
                        wxf::WXFValueList from_tag = {wxf::WXFValue("E"), wxf::WXFValue(from)};
                        wxf::WXFValueList to_tag = {wxf::WXFValue("E"), wxf::WXFValue(to)};
                        // Tooltip data: effective event IDs (matches graph vertices)
                        wxf::WXFValueAssociation branchial_data;
                        branchial_data.push_back({wxf::WXFValue("Event1"), wxf::WXFValue(from)});
                        branchial_data.push_back({wxf::WXFValue("Event2"), wxf::WXFValue(to)});
                        add_graph_edge(wxf::WXFValue(from_tag), wxf::WXFValue(to_tag), "Branchial", branchial_data);
                    }
                }
            }

                // Build GraphData association for this property
                wxf::WXFValueAssociation graph_data;
                graph_data.push_back({wxf::WXFValue("Vertices"), wxf::WXFValue(vertices)});
                graph_data.push_back({wxf::WXFValue("Edges"), wxf::WXFValue(edges)});
                graph_data.push_back({wxf::WXFValue("VertexData"), wxf::WXFValue(vertex_data)});
                all_graph_data.push_back({wxf::WXFValue(graph_property), wxf::WXFValue(graph_data)});
            }  // end for each graph_property

            // Add keyed GraphData to result
            full_result.push_back({wxf::WXFValue("GraphData"), wxf::WXFValue(all_graph_data)});
        }

        // Only include counts when requested
        if (include_num_states) {
            full_result.push_back({wxf::WXFValue("NumStates"), wxf::WXFValue(static_cast<int64_t>(hg.num_canonical_states()))});
        }
        if (include_num_events) {
            full_result.push_back({wxf::WXFValue("NumEvents"), wxf::WXFValue(static_cast<int64_t>(engine.num_events()))});
        }
        if (include_num_causal_edges) {
            // Count unique (producer, consumer) event pairs for v1 semantics
            // When show_genesis_events is false, we must filter out pairs involving genesis events
            // to match reference behavior ("IncludeInitialEvent" -> False)
            int64_t causal_count;
            if (show_genesis_events) {
                // Include all pairs
                causal_count = static_cast<int64_t>(hg.num_causal_event_pairs());
            } else {
                // Filter out genesis event pairs - must iterate and count
                auto causal_edge_vec = hg.causal_graph().get_causal_edges();
                auto pair_hash = [](const std::pair<hypergraph::EventId, hypergraph::EventId>& p) {
                    return std::hash<uint64_t>{}((static_cast<uint64_t>(p.first) << 32) | p.second);
                };
                std::unordered_set<std::pair<hypergraph::EventId, hypergraph::EventId>, decltype(pair_hash)> seen_pairs(
                    0, pair_hash);
                seen_pairs.reserve(causal_edge_vec.size());

                for (const auto& edge : causal_edge_vec) {
                    // Skip edges involving genesis events
                    if (hg.is_genesis_event(edge.producer) || hg.is_genesis_event(edge.consumer)) {
                        continue;
                    }
                    seen_pairs.insert({edge.producer, edge.consumer});
                }
                causal_count = static_cast<int64_t>(seen_pairs.size());
            }
            full_result.push_back({wxf::WXFValue("NumCausalEdges"), wxf::WXFValue(causal_count)});
        }
        if (include_num_branchial_edges) {
            full_result.push_back({wxf::WXFValue("NumBranchialEdges"), wxf::WXFValue(static_cast<int64_t>(hg.num_branchial_edges()))});
        }

        // GlobalEdges -> List of all edges created during evolution
        // Each edge is {edge_id, v1, v2, ...}
        if (include_global_edges) {
            wxf::WXFValueList global_edges;
            uint32_t num_edges = hg.num_edges();
            for (uint32_t eid = 0; eid < num_edges; ++eid) {
                const hypergraph::Edge& edge = hg.get_edge(eid);
                if (edge.id == hypergraph::INVALID_ID) continue;

                wxf::WXFValueList edge_data;
                edge_data.push_back(wxf::WXFValue(static_cast<int64_t>(eid)));
                for (uint8_t i = 0; i < edge.arity; ++i) {
                    edge_data.push_back(wxf::WXFValue(static_cast<int64_t>(edge.vertices[i])));
                }
                global_edges.push_back(wxf::WXFValue(edge_data));
            }
            full_result.push_back(std::make_pair(wxf::WXFValue("GlobalEdges"), wxf::WXFValue(global_edges)));
        }

        // StateBitvectors -> Association[state_id -> List of edge IDs present in that state]
        // Represents each state's edge set (the bitvector) as a list of edge indices
        if (include_state_bitvectors) {
            wxf::WXFValueAssociation state_bitvectors;
            uint32_t num_states = hg.num_states();
            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Convert SparseBitset to list of edge IDs
                wxf::WXFValueList edge_ids;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    edge_ids.push_back(wxf::WXFValue(static_cast<int64_t>(eid)));
                });

                state_bitvectors.push_back(std::make_pair(
                    wxf::WXFValue(static_cast<int64_t>(sid)),
                    wxf::WXFValue(edge_ids)
                ));
            }
            full_result.push_back(std::make_pair(wxf::WXFValue("StateBitvectors"), wxf::WXFValue(state_bitvectors)));
        }

        // DimensionData -> Association["PerState" -> {...}, "PerTimestep" -> {...}, "Global" -> {...}]
        if (compute_dimensions && !state_dimension_stats.empty()) {
            wxf::WXFValueAssociation dim_data;

            // Per-state stats: state_id -> {Mean, Min, Max, StdDev, Vertices -> <|v->d|> (if enabled)}
            wxf::WXFValueAssociation per_state;
            for (const auto& [sid, stats] : state_dimension_stats) {
                wxf::WXFValueAssociation stats_assoc;
                stats_assoc.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(static_cast<double>(stats.mean))});
                stats_assoc.push_back({wxf::WXFValue("Min"), wxf::WXFValue(static_cast<double>(stats.min))});
                stats_assoc.push_back({wxf::WXFValue("Max"), wxf::WXFValue(static_cast<double>(stats.max))});
                stats_assoc.push_back({wxf::WXFValue("StdDev"), wxf::WXFValue(static_cast<double>(stats.stddev))});

                // Add per-vertex dimensions only if dimension_per_vertex is enabled
                if (dimension_per_vertex) {
                    auto vids_it = state_vertex_ids.find(sid);
                    auto dims_it = state_vertex_dimensions.find(sid);
                    if (vids_it != state_vertex_ids.end() && dims_it != state_vertex_dimensions.end()) {
                        const auto& vids = vids_it->second;
                        const auto& dims = dims_it->second;
                        if (vids.size() == dims.size()) {
                            wxf::WXFValueAssociation per_vertex;
                            for (size_t i = 0; i < vids.size(); ++i) {
                                if (dims[i] > 0) {
                                    per_vertex.push_back({wxf::WXFValue(static_cast<int64_t>(vids[i])),
                                                         wxf::WXFValue(static_cast<double>(dims[i]))});
                                }
                            }
                            stats_assoc.push_back({wxf::WXFValue("PerVertex"), wxf::WXFValue(per_vertex)});
                        }
                    }
                }

                per_state.push_back({wxf::WXFValue(static_cast<int64_t>(sid)), wxf::WXFValue(stats_assoc)});
            }
            dim_data.push_back({wxf::WXFValue("PerState"), wxf::WXFValue(per_state)});

            // PerTimestep aggregation (if enabled)
            if (dimension_timestep_aggregation) {
                // Group states by timestep
                std::map<uint32_t, std::vector<uint32_t>> step_to_states;
                for (const auto& [sid, step] : state_to_step) {
                    step_to_states[step].push_back(sid);
                }

                wxf::WXFValueAssociation per_timestep;
                for (const auto& [step, state_ids] : step_to_states) {
                    wxf::WXFValueAssociation step_data;

                    // Compute mean/variance of state means at this timestep
                    double sum = 0.0, sum_sq = 0.0;
                    size_t count = 0;
                    for (uint32_t sid : state_ids) {
                        auto it = state_dimension_stats.find(sid);
                        if (it != state_dimension_stats.end() && it->second.count > 0) {
                            sum += it->second.mean;
                            sum_sq += it->second.mean * it->second.mean;
                            ++count;
                        }
                    }
                    if (count > 0) {
                        double mean = sum / count;
                        double variance = (count > 1) ? (sum_sq / count - mean * mean) : 0.0;
                        step_data.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(mean)});
                        step_data.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(variance)});
                        step_data.push_back({wxf::WXFValue("StateCount"), wxf::WXFValue(static_cast<int64_t>(count))});

                        // Per-vertex aggregation across states at this timestep (if enabled)
                        if (dimension_per_vertex) {
                            std::unordered_map<bh::VertexId, std::vector<float>> vertex_values;
                            for (uint32_t sid : state_ids) {
                                auto vids_it = state_vertex_ids.find(sid);
                                auto dims_it = state_vertex_dimensions.find(sid);
                                if (vids_it != state_vertex_ids.end() && dims_it != state_vertex_dimensions.end()) {
                                    const auto& vids = vids_it->second;
                                    const auto& dims = dims_it->second;
                                    for (size_t i = 0; i < vids.size() && i < dims.size(); ++i) {
                                        if (dims[i] > 0) {
                                            vertex_values[vids[i]].push_back(dims[i]);
                                        }
                                    }
                                }
                            }

                            wxf::WXFValueAssociation vertices;
                            for (const auto& [vid, values] : vertex_values) {
                                double v_sum = 0.0, v_sum_sq = 0.0;
                                for (float v : values) {
                                    v_sum += v;
                                    v_sum_sq += v * v;
                                }
                                double v_mean = v_sum / values.size();
                                double v_var = (values.size() > 1) ? (v_sum_sq / values.size() - v_mean * v_mean) : 0.0;

                                wxf::WXFValueAssociation v_data;
                                v_data.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(v_mean)});
                                v_data.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(v_var)});
                                v_data.push_back({wxf::WXFValue("Count"), wxf::WXFValue(static_cast<int64_t>(values.size()))});

                                vertices.push_back({wxf::WXFValue(static_cast<int64_t>(vid)), wxf::WXFValue(v_data)});
                            }
                            step_data.push_back({wxf::WXFValue("Vertices"), wxf::WXFValue(vertices)});
                        }
                    }

                    per_timestep.push_back({wxf::WXFValue(static_cast<int64_t>(step)), wxf::WXFValue(step_data)});
                }
                dim_data.push_back({wxf::WXFValue("PerTimestep"), wxf::WXFValue(per_timestep)});
            }

            // Global section (always present)
            {
                wxf::WXFValueAssociation global;

                // Compute global statistics across all states
                double sum = 0.0, sum_sq = 0.0;
                float g_min = std::numeric_limits<float>::max();
                float g_max = std::numeric_limits<float>::lowest();
                size_t count = 0;
                for (const auto& [sid, stats] : state_dimension_stats) {
                    if (stats.count > 0) {
                        sum += stats.mean;
                        sum_sq += stats.mean * stats.mean;
                        g_min = std::min(g_min, stats.min);
                        g_max = std::max(g_max, stats.max);
                        ++count;
                    }
                }

                if (count > 0) {
                    double mean = sum / count;
                    double variance = (count > 1) ? (sum_sq / count - mean * mean) : 0.0;
                    global.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(mean)});
                    global.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(variance)});
                    global.push_back({wxf::WXFValue("Min"), wxf::WXFValue(static_cast<double>(g_min))});
                    global.push_back({wxf::WXFValue("Max"), wxf::WXFValue(static_cast<double>(g_max))});
                } else {
                    global.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(0.0)});
                    global.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(0.0)});
                    global.push_back({wxf::WXFValue("Min"), wxf::WXFValue(0.0)});
                    global.push_back({wxf::WXFValue("Max"), wxf::WXFValue(3.0)});
                }

                // Range for color normalization
                wxf::WXFValueList range;
                if (global_dim_min <= global_dim_max) {
                    range.push_back(wxf::WXFValue(static_cast<double>(global_dim_min)));
                    range.push_back(wxf::WXFValue(static_cast<double>(global_dim_max)));
                } else {
                    range.push_back(wxf::WXFValue(0.0));
                    range.push_back(wxf::WXFValue(3.0));
                }
                global.push_back({wxf::WXFValue("Range"), wxf::WXFValue(range)});

                dim_data.push_back({wxf::WXFValue("Global"), wxf::WXFValue(global)});
            }

            full_result.push_back({wxf::WXFValue("DimensionData"), wxf::WXFValue(dim_data)});
        }

        // GeodesicData -> Association["PerState" -> {...}]
        // Each state: state_id -> {"Paths" -> {...}, "ProperTimes" -> {...}, "BundleSpread" -> float}
        if (compute_geodesics && !state_geodesic_paths.empty()) {
            wxf::WXFValueAssociation geodesic_data;

            // Per-state paths
            wxf::WXFValueAssociation per_state;
            for (const auto& [sid, paths] : state_geodesic_paths) {
                wxf::WXFValueAssociation state_data;

                // Paths: list of lists of vertex IDs
                wxf::WXFValueList path_list;
                for (const auto& path : paths) {
                    wxf::WXFValueList vertices;
                    for (bh::VertexId v : path) {
                        vertices.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                    }
                    path_list.push_back(wxf::WXFValue(vertices));
                }
                state_data.push_back({wxf::WXFValue("Paths"), wxf::WXFValue(path_list)});

                // Proper times
                auto pt_it = state_geodesic_proper_times.find(sid);
                if (pt_it != state_geodesic_proper_times.end()) {
                    wxf::WXFValueList pt_list;
                    for (const auto& times : pt_it->second) {
                        wxf::WXFValueList time_vals;
                        for (float t : times) {
                            time_vals.push_back(wxf::WXFValue(static_cast<double>(t)));
                        }
                        pt_list.push_back(wxf::WXFValue(time_vals));
                    }
                    state_data.push_back({wxf::WXFValue("ProperTimes"), wxf::WXFValue(pt_list)});
                }

                // Local dimensions along each path
                auto ld_it = state_geodesic_local_dimensions.find(sid);
                if (ld_it != state_geodesic_local_dimensions.end()) {
                    wxf::WXFValueList ld_list;
                    for (const auto& dims : ld_it->second) {
                        wxf::WXFValueList dim_vals;
                        for (float d : dims) {
                            dim_vals.push_back(wxf::WXFValue(static_cast<double>(d)));
                        }
                        ld_list.push_back(wxf::WXFValue(dim_vals));
                    }
                    state_data.push_back({wxf::WXFValue("LocalDimensions"), wxf::WXFValue(ld_list)});
                }

                // Bundle spread
                auto spread_it = state_geodesic_bundle_spread.find(sid);
                if (spread_it != state_geodesic_bundle_spread.end()) {
                    state_data.push_back({wxf::WXFValue("BundleSpread"),
                                         wxf::WXFValue(static_cast<double>(spread_it->second))});
                }

                // Lensing metrics per path
                auto lensing_it = state_geodesic_lensing.find(sid);
                if (lensing_it != state_geodesic_lensing.end() && !lensing_it->second.empty()) {
                    wxf::WXFValueList lensing_list;
                    for (const auto& lm : lensing_it->second) {
                        wxf::WXFValueAssociation lm_assoc;
                        lm_assoc.push_back({wxf::WXFValue("DeflectionAngle"),
                                           wxf::WXFValue(static_cast<double>(lm.deflection_angle))});
                        lm_assoc.push_back({wxf::WXFValue("ImpactParameter"),
                                           wxf::WXFValue(static_cast<double>(lm.impact_parameter))});
                        lm_assoc.push_back({wxf::WXFValue("ExpectedDeflection"),
                                           wxf::WXFValue(static_cast<double>(lm.expected_deflection))});
                        lm_assoc.push_back({wxf::WXFValue("DeflectionRatio"),
                                           wxf::WXFValue(static_cast<double>(lm.deflection_ratio))});
                        lm_assoc.push_back({wxf::WXFValue("ClosestVertex"),
                                           wxf::WXFValue(static_cast<int64_t>(lm.closest_vertex))});
                        lm_assoc.push_back({wxf::WXFValue("ClosestDimension"),
                                           wxf::WXFValue(static_cast<double>(lm.closest_dimension))});
                        lm_assoc.push_back({wxf::WXFValue("PassesNearCenter"),
                                           wxf::WXFValue(lm.passes_near_center ? "True" : "False")});
                        lensing_list.push_back(wxf::WXFValue(lm_assoc));
                    }
                    state_data.push_back({wxf::WXFValue("Lensing"), wxf::WXFValue(lensing_list)});

                    // Mean deflection and lensing center for this state
                    auto md_it = state_geodesic_mean_deflection.find(sid);
                    if (md_it != state_geodesic_mean_deflection.end()) {
                        state_data.push_back({wxf::WXFValue("MeanDeflection"),
                                             wxf::WXFValue(static_cast<double>(md_it->second))});
                    }
                    auto lc_it = state_geodesic_lensing_center.find(sid);
                    if (lc_it != state_geodesic_lensing_center.end()) {
                        state_data.push_back({wxf::WXFValue("LensingCenter"),
                                             wxf::WXFValue(static_cast<int64_t>(lc_it->second))});
                    }
                }

                per_state.push_back({wxf::WXFValue(static_cast<int64_t>(sid)), wxf::WXFValue(state_data)});
            }
            geodesic_data.push_back({wxf::WXFValue("PerState"), wxf::WXFValue(per_state)});

            full_result.push_back({wxf::WXFValue("GeodesicData"), wxf::WXFValue(geodesic_data)});
        }

        // TopologicalData -> Association["PerState" -> {...}, "PerTimestep" -> {...}, "Global" -> {...}]
        // Each state: state_id -> {"Defects" -> [...], "Charges" -> <|vertex -> charge|> (if enabled)}
        if (detect_particles && !state_defects.empty()) {
            wxf::WXFValueAssociation topo_data;

            // Per-state defects
            wxf::WXFValueAssociation per_state;
            double global_charge_sum = 0.0, global_charge_sum_sq = 0.0;
            size_t global_charge_count = 0;

            for (const auto& [sid, defects] : state_defects) {
                wxf::WXFValueAssociation state_data;

                // Defects list
                wxf::WXFValueList defect_list;
                for (const auto& defect : defects) {
                    wxf::WXFValueAssociation def_assoc;

                    // Type as string
                    const char* type_names[] = {"None", "K5", "K33", "HighDegree", "DimensionSpike", "Unknown"};
                    int type_idx = defect.type >= 0 && defect.type <= 5 ? defect.type : 5;
                    def_assoc.push_back({wxf::WXFValue("Type"), wxf::WXFValue(type_names[type_idx])});

                    // Core vertices
                    wxf::WXFValueList core_verts;
                    for (bh::VertexId v : defect.core_vertices) {
                        core_verts.push_back(wxf::WXFValue(static_cast<int64_t>(v)));
                    }
                    def_assoc.push_back({wxf::WXFValue("CoreVertices"), wxf::WXFValue(core_verts)});

                    def_assoc.push_back({wxf::WXFValue("Charge"), wxf::WXFValue(static_cast<double>(defect.charge))});
                    def_assoc.push_back({wxf::WXFValue("CentroidX"), wxf::WXFValue(static_cast<double>(defect.centroid_x))});
                    def_assoc.push_back({wxf::WXFValue("CentroidY"), wxf::WXFValue(static_cast<double>(defect.centroid_y))});
                    def_assoc.push_back({wxf::WXFValue("LocalDimension"), wxf::WXFValue(static_cast<double>(defect.local_dimension))});
                    def_assoc.push_back({wxf::WXFValue("Confidence"), wxf::WXFValue(static_cast<int64_t>(defect.confidence))});

                    defect_list.push_back(wxf::WXFValue(def_assoc));
                }
                state_data.push_back({wxf::WXFValue("Defects"), wxf::WXFValue(defect_list)});

                // Compute mean charge for state
                auto charge_it = state_charges.find(sid);
                if (charge_it != state_charges.end() && !charge_it->second.empty()) {
                    double state_charge_sum = 0.0;
                    for (const auto& [v, charge] : charge_it->second) {
                        state_charge_sum += charge;
                        global_charge_sum += charge;
                        global_charge_sum_sq += charge * charge;
                        ++global_charge_count;
                    }
                    double state_mean_charge = state_charge_sum / charge_it->second.size();
                    state_data.push_back({wxf::WXFValue("MeanCharge"), wxf::WXFValue(state_mean_charge)});

                    // Per-vertex charges only if charge_per_vertex is enabled
                    if (charge_per_vertex) {
                        wxf::WXFValueAssociation charges_assoc;
                        for (const auto& [v, charge] : charge_it->second) {
                            charges_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(v)),
                                                    wxf::WXFValue(static_cast<double>(charge))});
                        }
                        state_data.push_back({wxf::WXFValue("Vertices"), wxf::WXFValue(charges_assoc)});
                    }
                }

                per_state.push_back({wxf::WXFValue(static_cast<int64_t>(sid)), wxf::WXFValue(state_data)});
            }
            topo_data.push_back({wxf::WXFValue("PerState"), wxf::WXFValue(per_state)});

            // PerTimestep aggregation (if enabled)
            if (charge_timestep_aggregation) {
                // Group states by timestep
                std::map<uint32_t, std::vector<uint32_t>> step_to_states;
                for (const auto& [sid, step] : topological_state_to_step) {
                    step_to_states[step].push_back(sid);
                }

                wxf::WXFValueAssociation per_timestep;
                for (const auto& [step, state_ids] : step_to_states) {
                    wxf::WXFValueAssociation step_data;

                    // Aggregate charges at this timestep
                    double sum = 0.0, sum_sq = 0.0;
                    size_t count = 0;
                    std::unordered_map<bh::VertexId, std::vector<float>> vertex_charges;

                    for (uint32_t sid : state_ids) {
                        auto charge_it = state_charges.find(sid);
                        if (charge_it != state_charges.end()) {
                            for (const auto& [v, charge] : charge_it->second) {
                                sum += charge;
                                sum_sq += charge * charge;
                                ++count;
                                if (charge_per_vertex) {
                                    vertex_charges[v].push_back(charge);
                                }
                            }
                        }
                    }

                    if (count > 0) {
                        double mean = sum / count;
                        double variance = (count > 1) ? (sum_sq / count - mean * mean) : 0.0;
                        step_data.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(mean)});
                        step_data.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(variance)});
                        step_data.push_back({wxf::WXFValue("StateCount"), wxf::WXFValue(static_cast<int64_t>(state_ids.size()))});

                        // Per-vertex aggregation (if enabled)
                        if (charge_per_vertex && !vertex_charges.empty()) {
                            wxf::WXFValueAssociation vertices;
                            for (const auto& [vid, values] : vertex_charges) {
                                double v_sum = 0.0, v_sum_sq = 0.0;
                                for (float v : values) {
                                    v_sum += v;
                                    v_sum_sq += v * v;
                                }
                                double v_mean = v_sum / values.size();
                                double v_var = (values.size() > 1) ? (v_sum_sq / values.size() - v_mean * v_mean) : 0.0;

                                wxf::WXFValueAssociation v_data;
                                v_data.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(v_mean)});
                                v_data.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(v_var)});
                                v_data.push_back({wxf::WXFValue("Count"), wxf::WXFValue(static_cast<int64_t>(values.size()))});

                                vertices.push_back({wxf::WXFValue(static_cast<int64_t>(vid)), wxf::WXFValue(v_data)});
                            }
                            step_data.push_back({wxf::WXFValue("Vertices"), wxf::WXFValue(vertices)});
                        }
                    }

                    per_timestep.push_back({wxf::WXFValue(static_cast<int64_t>(step)), wxf::WXFValue(step_data)});
                }
                topo_data.push_back({wxf::WXFValue("PerTimestep"), wxf::WXFValue(per_timestep)});
            }

            // Global section (always present)
            {
                wxf::WXFValueAssociation global;

                size_t total_defects = 0;
                for (const auto& [sid, defects] : state_defects) {
                    total_defects += defects.size();
                }
                global.push_back({wxf::WXFValue("TotalDefects"), wxf::WXFValue(static_cast<int64_t>(total_defects))});

                if (global_charge_count > 0) {
                    double mean = global_charge_sum / global_charge_count;
                    double variance = (global_charge_count > 1) ?
                        (global_charge_sum_sq / global_charge_count - mean * mean) : 0.0;
                    global.push_back({wxf::WXFValue("MeanCharge"), wxf::WXFValue(mean)});
                    global.push_back({wxf::WXFValue("VarianceCharge"), wxf::WXFValue(variance)});
                } else {
                    global.push_back({wxf::WXFValue("MeanCharge"), wxf::WXFValue(0.0)});
                    global.push_back({wxf::WXFValue("VarianceCharge"), wxf::WXFValue(0.0)});
                }

                topo_data.push_back({wxf::WXFValue("Global"), wxf::WXFValue(global)});
            }

            full_result.push_back({wxf::WXFValue("TopologicalData"), wxf::WXFValue(topo_data)});
        }

        // CurvatureData -> Association["PerState" -> {...}, "PerTimestep" -> {...}, "Global" -> {...}]
        // Each state: state_id -> {"MeanCurvature" -> float, "Vertices" -> {...} (if enabled)}
        if (compute_curvature && (!state_ollivier_ricci.empty() || !state_wolfram_ricci.empty() || !state_wolfram_scalar.empty() || !state_dimension_gradient.empty())) {
            wxf::WXFValueAssociation curv_data;

            // Per-state curvatures
            wxf::WXFValueAssociation per_state;
            double global_curv_sum = 0.0, global_curv_sum_sq = 0.0;
            size_t global_curv_count = 0;

            for (const auto& [sid, mean_curv] : state_mean_curvature) {
                wxf::WXFValueAssociation state_data;

                // Mean curvature (always present)
                state_data.push_back({wxf::WXFValue("MeanCurvature"),
                                     wxf::WXFValue(static_cast<double>(mean_curv))});
                global_curv_sum += mean_curv;
                global_curv_sum_sq += mean_curv * mean_curv;
                ++global_curv_count;

                // Per-vertex curvatures only if curvature_per_vertex is enabled
                if (curvature_per_vertex) {
                    // Ollivier-Ricci per-vertex
                    auto or_it = state_ollivier_ricci.find(sid);
                    if (or_it != state_ollivier_ricci.end() && !or_it->second.empty()) {
                        wxf::WXFValueAssociation or_assoc;
                        for (const auto& [v, curv] : or_it->second) {
                            or_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(v)),
                                               wxf::WXFValue(static_cast<double>(curv))});
                        }
                        state_data.push_back({wxf::WXFValue("OllivierRicci"), wxf::WXFValue(or_assoc)});
                    }

                    // Wolfram-Ricci per-vertex (geodesic tube volume method)
                    auto wr_it = state_wolfram_ricci.find(sid);
                    if (wr_it != state_wolfram_ricci.end() && !wr_it->second.empty()) {
                        wxf::WXFValueAssociation wr_assoc;
                        for (const auto& [v, curv] : wr_it->second) {
                            wr_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(v)),
                                               wxf::WXFValue(static_cast<double>(curv))});
                        }
                        state_data.push_back({wxf::WXFValue("WolframRicci"), wxf::WXFValue(wr_assoc)});
                    }

                    // Wolfram-Scalar per-vertex (ball volume method)
                    auto ws_it = state_wolfram_scalar.find(sid);
                    if (ws_it != state_wolfram_scalar.end() && !ws_it->second.empty()) {
                        wxf::WXFValueAssociation ws_assoc;
                        for (const auto& [v, curv] : ws_it->second) {
                            ws_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(v)),
                                               wxf::WXFValue(static_cast<double>(curv))});
                        }
                        state_data.push_back({wxf::WXFValue("WolframScalar"), wxf::WXFValue(ws_assoc)});
                    }

                    // Dimension gradient per-vertex
                    auto dg_it = state_dimension_gradient.find(sid);
                    if (dg_it != state_dimension_gradient.end() && !dg_it->second.empty()) {
                        wxf::WXFValueAssociation dg_assoc;
                        for (const auto& [v, curv] : dg_it->second) {
                            dg_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(v)),
                                               wxf::WXFValue(static_cast<double>(curv))});
                        }
                        state_data.push_back({wxf::WXFValue("DimensionGradient"), wxf::WXFValue(dg_assoc)});
                    }
                }

                per_state.push_back({wxf::WXFValue(static_cast<int64_t>(sid)), wxf::WXFValue(state_data)});
            }
            curv_data.push_back({wxf::WXFValue("PerState"), wxf::WXFValue(per_state)});

            // PerTimestep aggregation (if enabled)
            if (curvature_timestep_aggregation) {
                // Group states by timestep
                std::map<uint32_t, std::vector<uint32_t>> step_to_states;
                for (const auto& [sid, step] : curvature_state_to_step) {
                    step_to_states[step].push_back(sid);
                }

                wxf::WXFValueAssociation per_timestep;
                for (const auto& [step, state_ids] : step_to_states) {
                    wxf::WXFValueAssociation step_data;

                    // Aggregate mean curvatures at this timestep
                    double sum = 0.0, sum_sq = 0.0;
                    size_t count = 0;
                    for (uint32_t sid : state_ids) {
                        auto it = state_mean_curvature.find(sid);
                        if (it != state_mean_curvature.end()) {
                            sum += it->second;
                            sum_sq += it->second * it->second;
                            ++count;
                        }
                    }

                    if (count > 0) {
                        double mean = sum / count;
                        double variance = (count > 1) ? (sum_sq / count - mean * mean) : 0.0;
                        step_data.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(mean)});
                        step_data.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(variance)});
                        step_data.push_back({wxf::WXFValue("StateCount"), wxf::WXFValue(static_cast<int64_t>(count))});
                    }

                    per_timestep.push_back({wxf::WXFValue(static_cast<int64_t>(step)), wxf::WXFValue(step_data)});
                }
                curv_data.push_back({wxf::WXFValue("PerTimestep"), wxf::WXFValue(per_timestep)});
            }

            // Global section (always present)
            {
                wxf::WXFValueAssociation global;

                if (global_curv_count > 0) {
                    double mean = global_curv_sum / global_curv_count;
                    double variance = (global_curv_count > 1) ?
                        (global_curv_sum_sq / global_curv_count - mean * mean) : 0.0;
                    global.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(mean)});
                    global.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(variance)});
                } else {
                    global.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(0.0)});
                    global.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(0.0)});
                }

                curv_data.push_back({wxf::WXFValue("Global"), wxf::WXFValue(global)});
            }

            full_result.push_back({wxf::WXFValue("CurvatureData"), wxf::WXFValue(curv_data)});
        }

        // EntropyData -> Association["PerState" -> {...}, "PerTimestep" -> {...}, "Global" -> {...}]
        // Each state: state_id -> {"DegreeEntropy" -> float, "GraphEntropy" -> float, ...}
        if (compute_entropy && !state_degree_entropy.empty()) {
            wxf::WXFValueAssociation ent_data;

            // Per-state entropy
            wxf::WXFValueAssociation per_state;
            double global_deg_ent_sum = 0.0, global_deg_ent_sum_sq = 0.0;
            double global_graph_ent_sum = 0.0, global_graph_ent_sum_sq = 0.0;
            size_t global_count = 0;

            for (const auto& [sid, deg_ent] : state_degree_entropy) {
                wxf::WXFValueAssociation state_data;

                state_data.push_back({wxf::WXFValue("DegreeEntropy"),
                                     wxf::WXFValue(static_cast<double>(deg_ent))});
                global_deg_ent_sum += deg_ent;
                global_deg_ent_sum_sq += deg_ent * deg_ent;

                auto ge_it = state_graph_entropy.find(sid);
                if (ge_it != state_graph_entropy.end()) {
                    state_data.push_back({wxf::WXFValue("GraphEntropy"),
                                         wxf::WXFValue(static_cast<double>(ge_it->second))});
                    global_graph_ent_sum += ge_it->second;
                    global_graph_ent_sum_sq += ge_it->second * ge_it->second;
                }
                ++global_count;

                // Local entropy per-vertex (always included - no per-vertex control for entropy)
                auto le_it = state_local_entropy.find(sid);
                if (le_it != state_local_entropy.end() && !le_it->second.empty()) {
                    wxf::WXFValueAssociation le_assoc;
                    for (const auto& [v, ent] : le_it->second) {
                        le_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(v)),
                                           wxf::WXFValue(static_cast<double>(ent))});
                    }
                    state_data.push_back({wxf::WXFValue("LocalEntropy"), wxf::WXFValue(le_assoc)});
                }

                // Mutual info per-vertex
                auto mi_it = state_mutual_info.find(sid);
                if (mi_it != state_mutual_info.end() && !mi_it->second.empty()) {
                    wxf::WXFValueAssociation mi_assoc;
                    for (const auto& [v, mi] : mi_it->second) {
                        mi_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(v)),
                                           wxf::WXFValue(static_cast<double>(mi))});
                    }
                    state_data.push_back({wxf::WXFValue("MutualInfo"), wxf::WXFValue(mi_assoc)});
                }

                // Fisher info per-vertex
                auto fi_it = state_fisher_info.find(sid);
                if (fi_it != state_fisher_info.end() && !fi_it->second.empty()) {
                    wxf::WXFValueAssociation fi_assoc;
                    for (const auto& [v, fi] : fi_it->second) {
                        fi_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(v)),
                                           wxf::WXFValue(static_cast<double>(fi))});
                    }
                    state_data.push_back({wxf::WXFValue("FisherInfo"), wxf::WXFValue(fi_assoc)});
                }

                per_state.push_back({wxf::WXFValue(static_cast<int64_t>(sid)), wxf::WXFValue(state_data)});
            }
            ent_data.push_back({wxf::WXFValue("PerState"), wxf::WXFValue(per_state)});

            // PerTimestep aggregation (if enabled)
            if (entropy_timestep_aggregation) {
                // Group states by timestep
                std::map<uint32_t, std::vector<uint32_t>> step_to_states;
                for (const auto& [sid, step] : entropy_state_to_step) {
                    step_to_states[step].push_back(sid);
                }

                wxf::WXFValueAssociation per_timestep;
                for (const auto& [step, state_ids] : step_to_states) {
                    wxf::WXFValueAssociation step_data;

                    // Aggregate degree entropy at this timestep
                    double deg_sum = 0.0, deg_sum_sq = 0.0;
                    double graph_sum = 0.0, graph_sum_sq = 0.0;
                    size_t count = 0;
                    for (uint32_t sid : state_ids) {
                        auto deg_it = state_degree_entropy.find(sid);
                        if (deg_it != state_degree_entropy.end()) {
                            deg_sum += deg_it->second;
                            deg_sum_sq += deg_it->second * deg_it->second;
                            ++count;
                        }
                        auto graph_it = state_graph_entropy.find(sid);
                        if (graph_it != state_graph_entropy.end()) {
                            graph_sum += graph_it->second;
                            graph_sum_sq += graph_it->second * graph_it->second;
                        }
                    }

                    if (count > 0) {
                        double deg_mean = deg_sum / count;
                        double deg_variance = (count > 1) ? (deg_sum_sq / count - deg_mean * deg_mean) : 0.0;
                        step_data.push_back({wxf::WXFValue("DegreeEntropyMean"), wxf::WXFValue(deg_mean)});
                        step_data.push_back({wxf::WXFValue("DegreeEntropyVariance"), wxf::WXFValue(deg_variance)});

                        double graph_mean = graph_sum / count;
                        double graph_variance = (count > 1) ? (graph_sum_sq / count - graph_mean * graph_mean) : 0.0;
                        step_data.push_back({wxf::WXFValue("GraphEntropyMean"), wxf::WXFValue(graph_mean)});
                        step_data.push_back({wxf::WXFValue("GraphEntropyVariance"), wxf::WXFValue(graph_variance)});

                        step_data.push_back({wxf::WXFValue("StateCount"), wxf::WXFValue(static_cast<int64_t>(count))});
                    }

                    per_timestep.push_back({wxf::WXFValue(static_cast<int64_t>(step)), wxf::WXFValue(step_data)});
                }
                ent_data.push_back({wxf::WXFValue("PerTimestep"), wxf::WXFValue(per_timestep)});
            }

            // Global section (always present)
            {
                wxf::WXFValueAssociation global;

                if (global_count > 0) {
                    double deg_mean = global_deg_ent_sum / global_count;
                    double deg_variance = (global_count > 1) ?
                        (global_deg_ent_sum_sq / global_count - deg_mean * deg_mean) : 0.0;
                    global.push_back({wxf::WXFValue("DegreeEntropyMean"), wxf::WXFValue(deg_mean)});
                    global.push_back({wxf::WXFValue("DegreeEntropyVariance"), wxf::WXFValue(deg_variance)});

                    double graph_mean = global_graph_ent_sum / global_count;
                    double graph_variance = (global_count > 1) ?
                        (global_graph_ent_sum_sq / global_count - graph_mean * graph_mean) : 0.0;
                    global.push_back({wxf::WXFValue("GraphEntropyMean"), wxf::WXFValue(graph_mean)});
                    global.push_back({wxf::WXFValue("GraphEntropyVariance"), wxf::WXFValue(graph_variance)});
                } else {
                    global.push_back({wxf::WXFValue("DegreeEntropyMean"), wxf::WXFValue(0.0)});
                    global.push_back({wxf::WXFValue("DegreeEntropyVariance"), wxf::WXFValue(0.0)});
                    global.push_back({wxf::WXFValue("GraphEntropyMean"), wxf::WXFValue(0.0)});
                    global.push_back({wxf::WXFValue("GraphEntropyVariance"), wxf::WXFValue(0.0)});
                }

                ent_data.push_back({wxf::WXFValue("Global"), wxf::WXFValue(global)});
            }

            full_result.push_back({wxf::WXFValue("EntropyData"), wxf::WXFValue(ent_data)});
        }

        // HilbertSpaceData -> Association["Global" -> {...}, "PerTimestep" -> {...}]
        // Scope controls: "Global" (default), "PerTimestep", or "Both"
        if (has_hilbert_data) {
            wxf::WXFValueAssociation hilbert_data;

            // Global section (always present with summary stats; matrices if scope includes Global)
            {
                wxf::WXFValueAssociation global;

                // Statistics (always present)
                global.push_back({wxf::WXFValue("NumStates"),
                                       wxf::WXFValue(static_cast<int64_t>(hilbert_result.num_states))});
                global.push_back({wxf::WXFValue("NumVertices"),
                                       wxf::WXFValue(static_cast<int64_t>(hilbert_result.num_vertices))});
                global.push_back({wxf::WXFValue("MeanInnerProduct"),
                                       wxf::WXFValue(static_cast<double>(hilbert_result.mean_inner_product))});
                global.push_back({wxf::WXFValue("MaxInnerProduct"),
                                       wxf::WXFValue(static_cast<double>(hilbert_result.max_inner_product))});
                global.push_back({wxf::WXFValue("MeanVertexProbability"),
                                       wxf::WXFValue(static_cast<double>(hilbert_result.mean_vertex_probability))});
                global.push_back({wxf::WXFValue("VertexProbabilityEntropy"),
                                       wxf::WXFValue(static_cast<double>(hilbert_result.vertex_probability_entropy))});
                global.push_back({wxf::WXFValue("MeanMutualInformation"),
                                       wxf::WXFValue(static_cast<double>(hilbert_result.mean_mutual_information))});
                global.push_back({wxf::WXFValue("MaxMutualInformation"),
                                       wxf::WXFValue(static_cast<double>(hilbert_result.max_mutual_information))});
                global.push_back({wxf::WXFValue("MeanEdgeMutualInformation"),
                                       wxf::WXFValue(static_cast<double>(hilbert_result.mean_edge_mutual_information))});
                global.push_back({wxf::WXFValue("MaxEdgeMutualInformation"),
                                       wxf::WXFValue(static_cast<double>(hilbert_result.max_edge_mutual_information))});

                // Full matrices only if scope is "Global" or "Both" (not "PerTimestep" only)
                bool include_global_matrices = (hilbert_scope == "Global" || hilbert_scope == "Both");
                if (include_global_matrices) {
                    // Vertex probabilities: vertex_id -> probability
                    wxf::WXFValueAssociation vertex_probs;
                    for (const auto& [vid, prob] : hilbert_result.vertex_probabilities) {
                        vertex_probs.push_back({wxf::WXFValue(static_cast<int64_t>(vid)),
                                               wxf::WXFValue(static_cast<double>(prob))});
                    }
                    global.push_back({wxf::WXFValue("VertexProbabilities"), wxf::WXFValue(vertex_probs)});

                    // Inner product matrix (as list of lists for efficient transfer)
                    wxf::WXFValueList ip_matrix;
                    for (const auto& row : hilbert_result.inner_product_matrix) {
                        wxf::WXFValueList row_list;
                        for (float val : row) {
                            row_list.push_back(wxf::WXFValue(static_cast<double>(val)));
                        }
                        ip_matrix.push_back(wxf::WXFValue(row_list));
                    }
                    global.push_back({wxf::WXFValue("InnerProductMatrix"), wxf::WXFValue(ip_matrix)});

                    // Mutual information matrix (as list of lists)
                    wxf::WXFValueList mi_matrix;
                    for (const auto& row : hilbert_result.mutual_information_matrix) {
                        wxf::WXFValueList row_list;
                        for (float val : row) {
                            row_list.push_back(wxf::WXFValue(static_cast<double>(val)));
                        }
                        mi_matrix.push_back(wxf::WXFValue(row_list));
                    }
                    global.push_back({wxf::WXFValue("MutualInformationMatrix"), wxf::WXFValue(mi_matrix)});

                    // Edge-level mutual information matrix (as list of lists)
                    wxf::WXFValueList edge_mi_matrix;
                    for (const auto& row : hilbert_result.edge_mutual_information_matrix) {
                        wxf::WXFValueList row_list;
                        for (float val : row) {
                            row_list.push_back(wxf::WXFValue(static_cast<double>(val)));
                        }
                        edge_mi_matrix.push_back(wxf::WXFValue(row_list));
                    }
                    global.push_back({wxf::WXFValue("EdgeMutualInformationMatrix"), wxf::WXFValue(edge_mi_matrix)});

                    // State indices (for mapping matrix rows/columns to state IDs)
                    wxf::WXFValueList state_ids;
                    for (uint32_t sid : hilbert_result.state_indices) {
                        state_ids.push_back(wxf::WXFValue(static_cast<int64_t>(sid)));
                    }
                    global.push_back({wxf::WXFValue("StateIndices"), wxf::WXFValue(state_ids)});
                }

                hilbert_data.push_back({wxf::WXFValue("Global"), wxf::WXFValue(global)});
            }

            // PerTimestep section (if scope is "PerTimestep" or "Both")
            bool include_per_timestep = (hilbert_scope == "PerTimestep" || hilbert_scope == "Both");
            if (include_per_timestep && !hilbert_per_timestep.empty()) {
                wxf::WXFValueAssociation per_timestep;

                for (const auto& [step, step_result] : hilbert_per_timestep) {
                    wxf::WXFValueAssociation step_data;

                    // Statistics for this timestep
                    step_data.push_back({wxf::WXFValue("NumStates"),
                                        wxf::WXFValue(static_cast<int64_t>(step_result.num_states))});
                    step_data.push_back({wxf::WXFValue("NumVertices"),
                                        wxf::WXFValue(static_cast<int64_t>(step_result.num_vertices))});
                    step_data.push_back({wxf::WXFValue("MeanInnerProduct"),
                                        wxf::WXFValue(static_cast<double>(step_result.mean_inner_product))});
                    step_data.push_back({wxf::WXFValue("MaxInnerProduct"),
                                        wxf::WXFValue(static_cast<double>(step_result.max_inner_product))});
                    step_data.push_back({wxf::WXFValue("MeanVertexProbability"),
                                        wxf::WXFValue(static_cast<double>(step_result.mean_vertex_probability))});
                    step_data.push_back({wxf::WXFValue("VertexProbabilityEntropy"),
                                        wxf::WXFValue(static_cast<double>(step_result.vertex_probability_entropy))});
                    step_data.push_back({wxf::WXFValue("MeanMutualInformation"),
                                        wxf::WXFValue(static_cast<double>(step_result.mean_mutual_information))});
                    step_data.push_back({wxf::WXFValue("MaxMutualInformation"),
                                        wxf::WXFValue(static_cast<double>(step_result.max_mutual_information))});
                    step_data.push_back({wxf::WXFValue("MeanEdgeMutualInformation"),
                                        wxf::WXFValue(static_cast<double>(step_result.mean_edge_mutual_information))});
                    step_data.push_back({wxf::WXFValue("MaxEdgeMutualInformation"),
                                        wxf::WXFValue(static_cast<double>(step_result.max_edge_mutual_information))});

                    // Vertex probabilities
                    wxf::WXFValueAssociation vertex_probs;
                    for (const auto& [vid, prob] : step_result.vertex_probabilities) {
                        vertex_probs.push_back({wxf::WXFValue(static_cast<int64_t>(vid)),
                                               wxf::WXFValue(static_cast<double>(prob))});
                    }
                    step_data.push_back({wxf::WXFValue("VertexProbabilities"), wxf::WXFValue(vertex_probs)});

                    // Inner product matrix
                    wxf::WXFValueList ip_matrix;
                    for (const auto& row : step_result.inner_product_matrix) {
                        wxf::WXFValueList row_list;
                        for (float val : row) {
                            row_list.push_back(wxf::WXFValue(static_cast<double>(val)));
                        }
                        ip_matrix.push_back(wxf::WXFValue(row_list));
                    }
                    step_data.push_back({wxf::WXFValue("InnerProductMatrix"), wxf::WXFValue(ip_matrix)});

                    // Mutual information matrix
                    wxf::WXFValueList mi_matrix;
                    for (const auto& row : step_result.mutual_information_matrix) {
                        wxf::WXFValueList row_list;
                        for (float val : row) {
                            row_list.push_back(wxf::WXFValue(static_cast<double>(val)));
                        }
                        mi_matrix.push_back(wxf::WXFValue(row_list));
                    }
                    step_data.push_back({wxf::WXFValue("MutualInformationMatrix"), wxf::WXFValue(mi_matrix)});

                    // Edge-level mutual information matrix
                    wxf::WXFValueList edge_mi_matrix;
                    for (const auto& row : step_result.edge_mutual_information_matrix) {
                        wxf::WXFValueList row_list;
                        for (float val : row) {
                            row_list.push_back(wxf::WXFValue(static_cast<double>(val)));
                        }
                        edge_mi_matrix.push_back(wxf::WXFValue(row_list));
                    }
                    step_data.push_back({wxf::WXFValue("EdgeMutualInformationMatrix"), wxf::WXFValue(edge_mi_matrix)});

                    // State indices
                    wxf::WXFValueList state_ids;
                    for (uint32_t sid : step_result.state_indices) {
                        state_ids.push_back(wxf::WXFValue(static_cast<int64_t>(sid)));
                    }
                    step_data.push_back({wxf::WXFValue("StateIndices"), wxf::WXFValue(state_ids)});

                    per_timestep.push_back({wxf::WXFValue(static_cast<int64_t>(step)), wxf::WXFValue(step_data)});
                }

                hilbert_data.push_back({wxf::WXFValue("PerTimestep"), wxf::WXFValue(per_timestep)});
            }

            full_result.push_back({wxf::WXFValue("HilbertSpaceData"), wxf::WXFValue(hilbert_data)});
        }

        // BranchialData -> Association["Global" -> {...}, "PerTimestep" -> {...}]
        // Scope controls: "Global" (default), "PerTimestep", or "Both"
        if (has_branchial_data) {
            wxf::WXFValueAssociation branchial_data;

            // Global section (always present with summary stats; per-vertex if branchial_per_vertex and scope includes Global)
            bool include_global_branchial = (branchial_scope == "Global" || branchial_scope == "Both");
            {
                wxf::WXFValueAssociation global;

                // Statistics (always present)
                global.push_back({wxf::WXFValue("NumUniqueVertices"),
                                 wxf::WXFValue(static_cast<int64_t>(branchial_result.num_unique_vertices))});
                global.push_back({wxf::WXFValue("MeanSharpness"),
                                 wxf::WXFValue(static_cast<double>(branchial_result.mean_sharpness))});
                global.push_back({wxf::WXFValue("MeanBranchEntropy"),
                                 wxf::WXFValue(static_cast<double>(branchial_result.mean_branch_entropy))});
                global.push_back({wxf::WXFValue("MaxBranchesPerVertex"),
                                 wxf::WXFValue(static_cast<int64_t>(branchial_result.max_branches_per_vertex))});

                // Per-vertex data only if branchial_per_vertex is true and scope includes Global
                if (branchial_per_vertex && include_global_branchial) {
                    // Per-vertex sharpness
                    wxf::WXFValueAssociation vertex_sharpness;
                    for (const auto& [vid, sharpness] : branchial_result.vertex_sharpness) {
                        vertex_sharpness.push_back({wxf::WXFValue(static_cast<int64_t>(vid)),
                                                   wxf::WXFValue(static_cast<double>(sharpness))});
                    }
                    global.push_back({wxf::WXFValue("VertexSharpness"), wxf::WXFValue(vertex_sharpness)});

                    // Per-vertex entropy
                    wxf::WXFValueAssociation vertex_entropy;
                    for (const auto& [vid, entropy] : branchial_result.vertex_entropy) {
                        vertex_entropy.push_back({wxf::WXFValue(static_cast<int64_t>(vid)),
                                                 wxf::WXFValue(static_cast<double>(entropy))});
                    }
                    global.push_back({wxf::WXFValue("VertexEntropy"), wxf::WXFValue(vertex_entropy)});
                }

                // Delocalized vertices (always included - just a list, not expensive)
                wxf::WXFValueList delocalized;
                for (const auto& [vid, sharpness] : branchial_result.vertex_sharpness) {
                    if (sharpness < 1.0f) {
                        delocalized.push_back(wxf::WXFValue(static_cast<int64_t>(vid)));
                    }
                }
                global.push_back({wxf::WXFValue("DelocalizedVertices"), wxf::WXFValue(delocalized)});

                branchial_data.push_back({wxf::WXFValue("Global"), wxf::WXFValue(global)});
            }

            // PerTimestep section (if scope is "PerTimestep" or "Both")
            bool include_per_timestep_branchial = (branchial_scope == "PerTimestep" || branchial_scope == "Both");
            if (include_per_timestep_branchial && !branchial_per_timestep.empty()) {
                wxf::WXFValueAssociation per_timestep;

                for (const auto& [step, step_result] : branchial_per_timestep) {
                    wxf::WXFValueAssociation step_data;

                    // Statistics for this timestep
                    step_data.push_back({wxf::WXFValue("NumUniqueVertices"),
                                        wxf::WXFValue(static_cast<int64_t>(step_result.num_unique_vertices))});
                    step_data.push_back({wxf::WXFValue("MeanSharpness"),
                                        wxf::WXFValue(static_cast<double>(step_result.mean_sharpness))});
                    step_data.push_back({wxf::WXFValue("MeanBranchEntropy"),
                                        wxf::WXFValue(static_cast<double>(step_result.mean_branch_entropy))});
                    step_data.push_back({wxf::WXFValue("MaxBranchesPerVertex"),
                                        wxf::WXFValue(static_cast<int64_t>(step_result.max_branches_per_vertex))});

                    // Per-vertex data if requested
                    if (branchial_per_vertex) {
                        wxf::WXFValueAssociation vertex_sharpness;
                        for (const auto& [vid, sharpness] : step_result.vertex_sharpness) {
                            vertex_sharpness.push_back({wxf::WXFValue(static_cast<int64_t>(vid)),
                                                       wxf::WXFValue(static_cast<double>(sharpness))});
                        }
                        step_data.push_back({wxf::WXFValue("VertexSharpness"), wxf::WXFValue(vertex_sharpness)});

                        wxf::WXFValueAssociation vertex_entropy;
                        for (const auto& [vid, entropy] : step_result.vertex_entropy) {
                            vertex_entropy.push_back({wxf::WXFValue(static_cast<int64_t>(vid)),
                                                     wxf::WXFValue(static_cast<double>(entropy))});
                        }
                        step_data.push_back({wxf::WXFValue("VertexEntropy"), wxf::WXFValue(vertex_entropy)});
                    }

                    // Delocalized vertices for this timestep
                    wxf::WXFValueList delocalized;
                    for (const auto& [vid, sharpness] : step_result.vertex_sharpness) {
                        if (sharpness < 1.0f) {
                            delocalized.push_back(wxf::WXFValue(static_cast<int64_t>(vid)));
                        }
                    }
                    step_data.push_back({wxf::WXFValue("DelocalizedVertices"), wxf::WXFValue(delocalized)});

                    per_timestep.push_back({wxf::WXFValue(static_cast<int64_t>(step)), wxf::WXFValue(step_data)});
                }

                branchial_data.push_back({wxf::WXFValue("PerTimestep"), wxf::WXFValue(per_timestep)});
            }

            full_result.push_back({wxf::WXFValue("BranchialData"), wxf::WXFValue(branchial_data)});
        }

        // MultispaceData -> Association["Global" -> {...}, "PerTimestep" -> {...}]
        // Scope controls: "Global" (default), "PerTimestep", or "Both"
        if (has_multispace_data) {
            wxf::WXFValueAssociation multispace_data;

            // Global section (always present with summary stats; full data if scope includes Global)
            bool include_global_multispace = (multispace_scope == "Global" || multispace_scope == "Both");
            {
                wxf::WXFValueAssociation global;

                // Statistics (always present)
                global.push_back({wxf::WXFValue("NumVertices"),
                                 wxf::WXFValue(static_cast<int64_t>(multispace_vertex_probs.size()))});
                global.push_back({wxf::WXFValue("NumEdges"),
                                 wxf::WXFValue(static_cast<int64_t>(multispace_edge_probs.size()))});
                global.push_back({wxf::WXFValue("MeanVertexProbability"),
                                 wxf::WXFValue(static_cast<double>(multispace_mean_vertex_prob))});
                global.push_back({wxf::WXFValue("MeanEdgeProbability"),
                                 wxf::WXFValue(static_cast<double>(multispace_mean_edge_prob))});
                global.push_back({wxf::WXFValue("TotalEntropy"),
                                 wxf::WXFValue(static_cast<double>(multispace_total_entropy))});

                // Full probability data only if scope includes Global
                if (include_global_multispace) {
                    // Per-vertex probabilities
                    wxf::WXFValueAssociation vertex_probs_assoc;
                    for (const auto& [vid, prob] : multispace_vertex_probs) {
                        vertex_probs_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(vid)),
                                                     wxf::WXFValue(static_cast<double>(prob))});
                    }
                    global.push_back({wxf::WXFValue("VertexProbabilities"), wxf::WXFValue(vertex_probs_assoc)});

                    // Per-edge probabilities (as list of {{v1, v2}, prob})
                    wxf::WXFValueList edge_probs_list;
                    for (const auto& [e, prob] : multispace_edge_probs) {
                        wxf::WXFValueList edge_entry;
                        wxf::WXFValueList edge_pair;
                        edge_pair.push_back(wxf::WXFValue(static_cast<int64_t>(e.first)));
                        edge_pair.push_back(wxf::WXFValue(static_cast<int64_t>(e.second)));
                        edge_entry.push_back(wxf::WXFValue(edge_pair));
                        edge_entry.push_back(wxf::WXFValue(static_cast<double>(prob)));
                        edge_probs_list.push_back(wxf::WXFValue(edge_entry));
                    }
                    global.push_back({wxf::WXFValue("EdgeProbabilities"), wxf::WXFValue(edge_probs_list)});
                }

                multispace_data.push_back({wxf::WXFValue("Global"), wxf::WXFValue(global)});
            }

            // PerTimestep section (if scope is "PerTimestep" or "Both")
            bool include_per_timestep_multispace = (multispace_scope == "PerTimestep" || multispace_scope == "Both");
            if (include_per_timestep_multispace && !multispace_per_timestep.empty()) {
                wxf::WXFValueAssociation per_timestep;

                for (const auto& [step, step_data] : multispace_per_timestep) {
                    wxf::WXFValueAssociation step_assoc;

                    // Statistics for this timestep
                    step_assoc.push_back({wxf::WXFValue("NumVertices"),
                                         wxf::WXFValue(static_cast<int64_t>(step_data.vertex_probs.size()))});
                    step_assoc.push_back({wxf::WXFValue("NumEdges"),
                                         wxf::WXFValue(static_cast<int64_t>(step_data.edge_probs.size()))});
                    step_assoc.push_back({wxf::WXFValue("MeanVertexProbability"),
                                         wxf::WXFValue(static_cast<double>(step_data.mean_vertex_prob))});
                    step_assoc.push_back({wxf::WXFValue("MeanEdgeProbability"),
                                         wxf::WXFValue(static_cast<double>(step_data.mean_edge_prob))});
                    step_assoc.push_back({wxf::WXFValue("TotalEntropy"),
                                         wxf::WXFValue(static_cast<double>(step_data.total_entropy))});

                    // Per-vertex probabilities
                    wxf::WXFValueAssociation vertex_probs_assoc;
                    for (const auto& [vid, prob] : step_data.vertex_probs) {
                        vertex_probs_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(vid)),
                                                     wxf::WXFValue(static_cast<double>(prob))});
                    }
                    step_assoc.push_back({wxf::WXFValue("VertexProbabilities"), wxf::WXFValue(vertex_probs_assoc)});

                    // Per-edge probabilities
                    wxf::WXFValueList edge_probs_list;
                    for (const auto& [e, prob] : step_data.edge_probs) {
                        wxf::WXFValueList edge_entry;
                        wxf::WXFValueList edge_pair;
                        edge_pair.push_back(wxf::WXFValue(static_cast<int64_t>(e.first)));
                        edge_pair.push_back(wxf::WXFValue(static_cast<int64_t>(e.second)));
                        edge_entry.push_back(wxf::WXFValue(edge_pair));
                        edge_entry.push_back(wxf::WXFValue(static_cast<double>(prob)));
                        edge_probs_list.push_back(wxf::WXFValue(edge_entry));
                    }
                    step_assoc.push_back({wxf::WXFValue("EdgeProbabilities"), wxf::WXFValue(edge_probs_list)});

                    per_timestep.push_back({wxf::WXFValue(static_cast<int64_t>(step)), wxf::WXFValue(step_assoc)});
                }

                multispace_data.push_back({wxf::WXFValue("PerTimestep"), wxf::WXFValue(per_timestep)});
            }

            full_result.push_back({wxf::WXFValue("MultispaceData"), wxf::WXFValue(multispace_data)});
        }

        // EquilibriumData -> Association with stability scores and per-timestep metrics
        if (has_equilibrium_data) {
            wxf::WXFValueAssociation eq_data;

            // Overall stability scores
            eq_data.push_back(std::make_pair(wxf::WXFValue("DimensionStability"),
                              wxf::WXFValue(static_cast<double>(equilibrium_result.dimension_stability))));
            eq_data.push_back(std::make_pair(wxf::WXFValue("DegreeEntropyStability"),
                              wxf::WXFValue(static_cast<double>(equilibrium_result.degree_entropy_stability))));
            eq_data.push_back(std::make_pair(wxf::WXFValue("SharpnessStability"),
                              wxf::WXFValue(static_cast<double>(equilibrium_result.sharpness_stability))));
            eq_data.push_back(std::make_pair(wxf::WXFValue("SizeStability"),
                              wxf::WXFValue(static_cast<double>(equilibrium_result.size_stability))));
            eq_data.push_back(std::make_pair(wxf::WXFValue("OverallStability"),
                              wxf::WXFValue(static_cast<double>(equilibrium_result.overall_stability))));

            // Equilibrium detection
            eq_data.push_back(std::make_pair(wxf::WXFValue("IsEquilibrated"),
                              wxf::WXFValue(equilibrium_result.is_equilibrated)));
            eq_data.push_back(std::make_pair(wxf::WXFValue("EquilibrationStep"),
                              wxf::WXFValue(static_cast<int64_t>(equilibrium_result.equilibration_step))));
            eq_data.push_back(std::make_pair(wxf::WXFValue("EquilibriumThreshold"),
                              wxf::WXFValue(static_cast<double>(equilibrium_result.equilibrium_threshold))));

            // Trend analysis
            eq_data.push_back(std::make_pair(wxf::WXFValue("DimensionTrend"),
                              wxf::WXFValue(static_cast<double>(equilibrium_result.dimension_trend))));
            eq_data.push_back(std::make_pair(wxf::WXFValue("SizeTrend"),
                              wxf::WXFValue(static_cast<double>(equilibrium_result.size_trend))));

            // Per-timestep metrics history
            wxf::WXFValueList history_list;
            for (const auto& m : equilibrium_result.history) {
                wxf::WXFValueAssociation step_data;
                step_data.push_back(std::make_pair(wxf::WXFValue("Step"), wxf::WXFValue(static_cast<int64_t>(m.step))));
                step_data.push_back(std::make_pair(wxf::WXFValue("MeanDimension"), wxf::WXFValue(static_cast<double>(m.mean_dimension))));
                step_data.push_back(std::make_pair(wxf::WXFValue("DimensionVariance"), wxf::WXFValue(static_cast<double>(m.dimension_variance))));
                step_data.push_back(std::make_pair(wxf::WXFValue("DegreeEntropy"), wxf::WXFValue(static_cast<double>(m.degree_entropy))));
                step_data.push_back(std::make_pair(wxf::WXFValue("MeanDegree"), wxf::WXFValue(static_cast<double>(m.mean_degree))));
                step_data.push_back(std::make_pair(wxf::WXFValue("MeanSharpness"), wxf::WXFValue(static_cast<double>(m.mean_sharpness))));
                step_data.push_back(std::make_pair(wxf::WXFValue("VertexCount"), wxf::WXFValue(static_cast<int64_t>(m.vertex_count))));
                step_data.push_back(std::make_pair(wxf::WXFValue("EdgeCount"), wxf::WXFValue(static_cast<int64_t>(m.edge_count))));
                step_data.push_back(std::make_pair(wxf::WXFValue("StateCount"), wxf::WXFValue(static_cast<int64_t>(m.state_count))));
                step_data.push_back(std::make_pair(wxf::WXFValue("MeanInnerProduct"), wxf::WXFValue(static_cast<double>(m.mean_inner_product))));
                step_data.push_back(std::make_pair(wxf::WXFValue("MeanMutualInformation"), wxf::WXFValue(static_cast<double>(m.mean_mutual_information))));
                step_data.push_back(std::make_pair(wxf::WXFValue("VertexProbabilityEntropy"), wxf::WXFValue(static_cast<double>(m.vertex_probability_entropy))));
                history_list.push_back(wxf::WXFValue(step_data));
            }
            eq_data.push_back(std::make_pair(wxf::WXFValue("History"), wxf::WXFValue(history_list)));

            full_result.push_back(std::make_pair(wxf::WXFValue("EquilibriumData"), wxf::WXFValue(eq_data)));
        }

        // Topology -> String (for metadata)
        if (!topology_type.empty() && topology_type != "Flat") {
            full_result.push_back(std::make_pair(wxf::WXFValue("Topology"), wxf::WXFValue(topology_type)));
        }

        // Write final association
        wxf_writer.write(wxf::WXFValue(full_result));
        const auto& wxf_data = wxf_writer.data();

        // Create output ByteArray
        mint wxf_dims[1] = {static_cast<mint>(wxf_data.size())};
        MNumericArray result_array;
        int err = libData->numericarrayLibraryFunctions->MNumericArray_new(MNumericArray_Type_UBit8, 1, wxf_dims, &result_array);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }

        // Copy byte data
        void* result_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(result_array);
        uint8_t* byte_result_data = static_cast<uint8_t*>(result_data);
        std::memcpy(byte_result_data, wxf_data.data(), wxf_data.size());

#ifdef HAVE_WSTP
        if (show_progress) {
            print_to_frontend(libData, "HGEvolve: Serialization complete.");
        }
#endif

        MArgument_setMNumericArray(res, result_array);
        return LIBRARY_NO_ERROR;

    } catch (const wxf::TypeError& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "WXF TypeError: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    } catch (const std::exception& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "HGEvolve error: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Perform Hausdorff dimension analysis on a graph
 * Input: WXF binary data containing:
 *   Association[
 *     "Edges" -> {{v1, v2}, ...},
 *     "Options" -> Association[
 *       "Formula" -> "LinearRegression" | "DiscreteDerivative",
 *       "SaturationThreshold" -> 0.5,
 *       "MinRadius" -> 1,
 *       "MaxRadius" -> 5,
 *       "NumAnchors" -> 6,
 *       "AnchorSeparation" -> 3
 *     ]
 *   ]
 *
 * Output: WXF Association with per-vertex dimensions, stats, anchors, coords
 */
EXTERN_C DLLEXPORT int performHausdorffAnalysis(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    // Use namespace alias to avoid conflicts with hypergraph namespace
    namespace bh = viz::blackhole;

    try {
        if (argc != 1) {
            handle_error(libData, "performHausdorffAnalysis expects 1 argument: WXF ByteArray data");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Get WXF data as ByteArray (MNumericArray)
        MNumericArray wxf_array = MArgument_getMNumericArray(argv[0]);

        mint rank = libData->numericarrayLibraryFunctions->MNumericArray_getRank(wxf_array);
        if (rank != 1) {
            handle_error(libData, "WXF ByteArray must be 1-dimensional");
            return LIBRARY_FUNCTION_ERROR;
        }

        const mint* dims = libData->numericarrayLibraryFunctions->MNumericArray_getDimensions(wxf_array);
        mint wxf_size = dims[0];

        void* wxf_raw_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(wxf_array);
        const uint8_t* wxf_byte_data = static_cast<const uint8_t*>(wxf_raw_data);

        // Convert to vector
        std::vector<uint8_t> wxf_bytes(wxf_byte_data, wxf_byte_data + wxf_size);

        // Parse WXF input
        wxf::Parser parser(wxf_bytes);
        parser.skip_header();

        // Parse options
        std::vector<std::vector<int64_t>> edges_raw;
        bh::DimensionFormula formula = bh::DimensionFormula::LinearRegression;
        float saturation_threshold = 0.5f;
        int min_radius = 1;
        int max_radius = 5;
        int num_anchors = 6;
        int anchor_separation = 3;
        bh::AggregationMethod aggregation = bh::AggregationMethod::Mean;
        bool directed = false;

        parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
            if (key == "Edges") {
                // Read edges as list of pairs: {{v1, v2}, {v3, v4}, ...}
                edges_raw = value_parser.read<std::vector<std::vector<int64_t>>>();
            }
            else if (key == "Options") {
                value_parser.read_association([&](const std::string& opt_key, wxf::Parser& opt_parser) {
                    if (opt_key == "Formula") {
                        std::string formula_str = opt_parser.read<std::string>();
                        if (formula_str == "DiscreteDerivative") {
                            formula = bh::DimensionFormula::DiscreteDerivative;
                        }
                    }
                    else if (opt_key == "SaturationThreshold") {
                        saturation_threshold = static_cast<float>(opt_parser.read<double>());
                    }
                    else if (opt_key == "MinRadius") {
                        min_radius = static_cast<int>(opt_parser.read<int64_t>());
                    }
                    else if (opt_key == "MaxRadius") {
                        max_radius = static_cast<int>(opt_parser.read<int64_t>());
                    }
                    else if (opt_key == "NumAnchors") {
                        num_anchors = static_cast<int>(opt_parser.read<int64_t>());
                    }
                    else if (opt_key == "AnchorSeparation") {
                        anchor_separation = static_cast<int>(opt_parser.read<int64_t>());
                    }
                    else if (opt_key == "Aggregation") {
                        std::string agg_str = opt_parser.read<std::string>();
                        if (agg_str == "Min") aggregation = bh::AggregationMethod::Min;
                        else if (agg_str == "Max") aggregation = bh::AggregationMethod::Max;
                    }
                    else if (opt_key == "Directed") {
                        // Read boolean as WXF symbol "True" or "False"
                        std::string bool_str = opt_parser.read<std::string>();
                        directed = (bool_str == "True");
                    }
                    else {
                        opt_parser.skip_value();
                    }
                });
            }
            else {
                value_parser.skip_value();
            }
        });

        if (edges_raw.empty()) {
            handle_error(libData, "No edges provided");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Build SimpleGraph from edges
        std::vector<bh::Edge> edge_list;
        edge_list.reserve(edges_raw.size());
        for (const auto& edge : edges_raw) {
            if (edge.size() >= 2) {
                edge_list.push_back(bh::Edge(
                    static_cast<bh::VertexId>(edge[0]),
                    static_cast<bh::VertexId>(edge[1])
                ));
            }
        }

        bh::SimpleGraph graph;
        graph.build_from_edges(edge_list);

        // Select anchors
        std::vector<bh::VertexId> candidates = graph.vertices();
        std::vector<bh::VertexId> anchors = bh::select_anchors(graph, candidates, num_anchors, anchor_separation);

        // Compute geodesic coordinates
        auto geodesic_coords = bh::compute_geodesic_coordinates(graph, anchors);

        // Configure dimension estimation
        bh::DimensionConfig config;
        config.formula = formula;
        config.saturation_threshold = saturation_threshold;
        config.min_radius = min_radius;
        config.max_radius = max_radius;
        config.aggregation = aggregation;
        config.directed = directed;

        // Create job system for parallel computation
        job_system::JobSystem<int> js(std::thread::hardware_concurrency());
        js.start();

        // Compute dimensions for all vertices (parallel)
        std::vector<float> dimensions = bh::estimate_all_dimensions_parallel(graph, &js, config);

        js.shutdown();

        // Compute stats
        bh::DimensionStats stats = bh::compute_dimension_stats(dimensions);

        // Build WXF output
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::WXFValueAssociation result;

        // PerVertex -> Association[vertex_id -> dimension]
        wxf::WXFValueAssociation per_vertex_assoc;
        const auto& vertices = graph.vertices();
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (dimensions[i] > 0) {
                per_vertex_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(vertices[i])),
                                            wxf::WXFValue(static_cast<double>(dimensions[i]))});
            }
        }
        result.push_back({wxf::WXFValue("PerVertex"), wxf::WXFValue(per_vertex_assoc)});

        // GeodesicCoords -> Association[vertex_id -> {d1, d2, ...}]
        wxf::WXFValueAssociation coords_assoc;
        for (const auto& [vid, coords] : geodesic_coords) {
            wxf::WXFValueList coord_list;
            for (int d : coords) {
                coord_list.push_back(wxf::WXFValue(static_cast<int64_t>(d)));
            }
            coords_assoc.push_back({wxf::WXFValue(static_cast<int64_t>(vid)), wxf::WXFValue(coord_list)});
        }
        result.push_back({wxf::WXFValue("GeodesicCoords"), wxf::WXFValue(coords_assoc)});

        // Anchors -> {anchor1, anchor2, ...}
        wxf::WXFValueList anchors_list;
        for (bh::VertexId a : anchors) {
            anchors_list.push_back(wxf::WXFValue(static_cast<int64_t>(a)));
        }
        result.push_back({wxf::WXFValue("Anchors"), wxf::WXFValue(anchors_list)});

        // Stats -> Association
        wxf::WXFValueAssociation stats_assoc;
        stats_assoc.push_back({wxf::WXFValue("Mean"), wxf::WXFValue(static_cast<double>(stats.mean))});
        stats_assoc.push_back({wxf::WXFValue("Min"), wxf::WXFValue(static_cast<double>(stats.min))});
        stats_assoc.push_back({wxf::WXFValue("Max"), wxf::WXFValue(static_cast<double>(stats.max))});
        stats_assoc.push_back({wxf::WXFValue("Variance"), wxf::WXFValue(static_cast<double>(stats.variance))});
        stats_assoc.push_back({wxf::WXFValue("StdDev"), wxf::WXFValue(static_cast<double>(stats.stddev))});
        stats_assoc.push_back({wxf::WXFValue("Count"), wxf::WXFValue(static_cast<int64_t>(stats.count))});
        result.push_back({wxf::WXFValue("Stats"), wxf::WXFValue(stats_assoc)});

        // Config -> Association (echo back the config used)
        wxf::WXFValueAssociation config_assoc;
        config_assoc.push_back({wxf::WXFValue("Formula"),
            wxf::WXFValue(formula == bh::DimensionFormula::LinearRegression ? "LinearRegression" : "DiscreteDerivative")});
        config_assoc.push_back({wxf::WXFValue("SaturationThreshold"), wxf::WXFValue(static_cast<double>(saturation_threshold))});
        config_assoc.push_back({wxf::WXFValue("MinRadius"), wxf::WXFValue(static_cast<int64_t>(min_radius))});
        config_assoc.push_back({wxf::WXFValue("MaxRadius"), wxf::WXFValue(static_cast<int64_t>(max_radius))});
        config_assoc.push_back({wxf::WXFValue("NumAnchors"), wxf::WXFValue(static_cast<int64_t>(num_anchors))});
        config_assoc.push_back({wxf::WXFValue("AnchorSeparation"), wxf::WXFValue(static_cast<int64_t>(anchor_separation))});
        result.push_back({wxf::WXFValue("Config"), wxf::WXFValue(config_assoc)});

        // Write output
        wxf_writer.write(wxf::WXFValue(result));
        const auto& wxf_data = wxf_writer.data();

        // Create output ByteArray
        mint wxf_dims[1] = {static_cast<mint>(wxf_data.size())};
        MNumericArray result_array;
        int err = libData->numericarrayLibraryFunctions->MNumericArray_new(MNumericArray_Type_UBit8, 1, wxf_dims, &result_array);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }

        void* result_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(result_array);
        std::memcpy(result_data, wxf_data.data(), wxf_data.size());

        MArgument_setMNumericArray(res, result_array);
        return LIBRARY_NO_ERROR;

    } catch (const wxf::TypeError& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "WXF TypeError in HausdorffAnalysis: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    } catch (const std::exception& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "Exception in HausdorffAnalysis: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    }
}

/*
 * performBranchAlignment - Align branch curvature using PCA on spectral embedding
 *
 * Input: WXF Association with:
 *   "Edges" -> {{v1, v2}, {v3, v4}, ...}
 *   "Curvature" -> Association[vertex_id -> curvature_value]
 *   "Options" -> Association[
 *     "NumIterations" -> 100  // Power iteration iterations for spectral embedding
 *   ]
 *
 * Output: WXF Association with:
 *   "Vertices" -> {v1, v2, ...}  // In canonical order (sorted by PC1)
 *   "PC1" -> {pc1_1, pc1_2, ...}
 *   "PC2" -> {pc2_1, pc2_2, ...}
 *   "PC3" -> {pc3_1, pc3_2, ...}
 *   "Curvature" -> {c1, c2, ...}  // In canonical order
 *   "Rank" -> {r1, r2, ...}  // Normalized [0, 1]
 *   "Eigenvalues" -> {e1, e2, e3}
 *   "Centroid" -> {cx, cy, cz}
 *   "Stats" -> Association["NumVertices" -> n, "CurvatureMin" -> min, "CurvatureMax" -> max, "CurvatureMean" -> mean]
 */
EXTERN_C DLLEXPORT int performBranchAlignment(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    namespace bh = viz::blackhole;

    try {
        if (argc != 1) {
            handle_error(libData, "performBranchAlignment expects 1 argument: WXF ByteArray data");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Get WXF data as ByteArray (MNumericArray)
        MNumericArray wxf_array = MArgument_getMNumericArray(argv[0]);

        mint rank = libData->numericarrayLibraryFunctions->MNumericArray_getRank(wxf_array);
        if (rank != 1) {
            handle_error(libData, "WXF ByteArray must be 1-dimensional");
            return LIBRARY_FUNCTION_ERROR;
        }

        const mint* dims = libData->numericarrayLibraryFunctions->MNumericArray_getDimensions(wxf_array);
        mint wxf_size = dims[0];

        void* wxf_raw_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(wxf_array);
        const uint8_t* wxf_byte_data = static_cast<const uint8_t*>(wxf_raw_data);

        // Convert to vector
        std::vector<uint8_t> wxf_bytes(wxf_byte_data, wxf_byte_data + wxf_size);

        // Parse WXF input
        wxf::Parser parser(wxf_bytes);
        parser.skip_header();

        // Parse options
        std::vector<std::vector<int64_t>> edges_raw;
        std::unordered_map<bh::VertexId, float> curvature;

        parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
            if (key == "Edges") {
                edges_raw = value_parser.read<std::vector<std::vector<int64_t>>>();
            }
            else if (key == "Curvature") {
                value_parser.read_association([&](const std::string& curv_key, wxf::Parser& curv_parser) {
                    // Key is vertex ID as string, value is curvature
                    int64_t vid = std::stoll(curv_key);
                    double curv_val = curv_parser.read<double>();
                    curvature[static_cast<bh::VertexId>(vid)] = static_cast<float>(curv_val);
                });
            }
            else if (key == "Options") {
                value_parser.read_association([&](const std::string& /* opt_key */, wxf::Parser& opt_parser) {
                    opt_parser.skip_value();
                });
            }
            else {
                value_parser.skip_value();
            }
        });

        if (edges_raw.empty()) {
            handle_error(libData, "No edges provided");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Build SimpleGraph from edges
        std::vector<bh::Edge> edge_list;
        edge_list.reserve(edges_raw.size());
        for (const auto& edge : edges_raw) {
            if (edge.size() >= 2) {
                edge_list.push_back(bh::Edge(
                    static_cast<bh::VertexId>(edge[0]),
                    static_cast<bh::VertexId>(edge[1])
                ));
            }
        }

        bh::SimpleGraph graph;
        graph.build_from_edges(edge_list);

        // Perform branch alignment
        bh::BranchAlignmentResult alignment = bh::align_branch(graph, curvature);

        if (!alignment.valid) {
            handle_error(libData, alignment.error.c_str());
            return LIBRARY_FUNCTION_ERROR;
        }

        // Build WXF output
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::WXFValueAssociation result;

        // Vertices -> {v1, v2, ...}
        wxf::WXFValueList vertices_list;
        for (bh::VertexId vid : alignment.vertices) {
            vertices_list.push_back(wxf::WXFValue(static_cast<int64_t>(vid)));
        }
        result.push_back({wxf::WXFValue("Vertices"), wxf::WXFValue(vertices_list)});

        // PC1, PC2, PC3 -> {f1, f2, ...}
        wxf::WXFValueList pc1_list, pc2_list, pc3_list;
        for (size_t i = 0; i < alignment.num_vertices; ++i) {
            pc1_list.push_back(wxf::WXFValue(static_cast<double>(alignment.pc1[i])));
            pc2_list.push_back(wxf::WXFValue(static_cast<double>(alignment.pc2[i])));
            pc3_list.push_back(wxf::WXFValue(static_cast<double>(alignment.pc3[i])));
        }
        result.push_back({wxf::WXFValue("PC1"), wxf::WXFValue(pc1_list)});
        result.push_back({wxf::WXFValue("PC2"), wxf::WXFValue(pc2_list)});
        result.push_back({wxf::WXFValue("PC3"), wxf::WXFValue(pc3_list)});

        // Curvature -> {c1, c2, ...}
        wxf::WXFValueList curvature_list;
        for (size_t i = 0; i < alignment.num_vertices; ++i) {
            curvature_list.push_back(wxf::WXFValue(static_cast<double>(alignment.curvature[i])));
        }
        result.push_back({wxf::WXFValue("Curvature"), wxf::WXFValue(curvature_list)});

        // Rank -> {r1, r2, ...}
        wxf::WXFValueList rank_list;
        for (size_t i = 0; i < alignment.num_vertices; ++i) {
            rank_list.push_back(wxf::WXFValue(static_cast<double>(alignment.rank[i])));
        }
        result.push_back({wxf::WXFValue("Rank"), wxf::WXFValue(rank_list)});

        // Eigenvalues -> {e1, e2, e3}
        wxf::WXFValueList evals_list;
        for (int i = 0; i < 3; ++i) {
            evals_list.push_back(wxf::WXFValue(static_cast<double>(alignment.eigenvalues[i])));
        }
        result.push_back({wxf::WXFValue("Eigenvalues"), wxf::WXFValue(evals_list)});

        // Centroid -> {cx, cy, cz}
        wxf::WXFValueList centroid_list;
        centroid_list.push_back(wxf::WXFValue(static_cast<double>(alignment.centroid.x)));
        centroid_list.push_back(wxf::WXFValue(static_cast<double>(alignment.centroid.y)));
        centroid_list.push_back(wxf::WXFValue(static_cast<double>(alignment.centroid.z)));
        result.push_back({wxf::WXFValue("Centroid"), wxf::WXFValue(centroid_list)});

        // Stats -> Association
        wxf::WXFValueAssociation stats_assoc;
        stats_assoc.push_back({wxf::WXFValue("NumVertices"), wxf::WXFValue(static_cast<int64_t>(alignment.num_vertices))});
        stats_assoc.push_back({wxf::WXFValue("CurvatureMin"), wxf::WXFValue(static_cast<double>(alignment.curvature_min))});
        stats_assoc.push_back({wxf::WXFValue("CurvatureMax"), wxf::WXFValue(static_cast<double>(alignment.curvature_max))});
        stats_assoc.push_back({wxf::WXFValue("CurvatureMean"), wxf::WXFValue(static_cast<double>(alignment.curvature_mean))});
        result.push_back({wxf::WXFValue("Stats"), wxf::WXFValue(stats_assoc)});

        // Write output
        wxf_writer.write(wxf::WXFValue(result));
        const auto& wxf_data = wxf_writer.data();

        // Create output ByteArray
        mint wxf_dims[1] = {static_cast<mint>(wxf_data.size())};
        MNumericArray result_array;
        int err = libData->numericarrayLibraryFunctions->MNumericArray_new(MNumericArray_Type_UBit8, 1, wxf_dims, &result_array);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }

        void* result_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(result_array);
        std::memcpy(result_data, wxf_data.data(), wxf_data.size());

        MArgument_setMNumericArray(res, result_array);
        return LIBRARY_NO_ERROR;

    } catch (const wxf::TypeError& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "WXF TypeError in BranchAlignment: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    } catch (const std::exception& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "Exception in BranchAlignment: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    }
}

/*
 * performBranchAlignmentBatch - Batch alignment on full evolution result
 *
 * Input: WXF Association with:
 *   "States" -> Association[stateId -> {"Edges" -> ..., "Curvature" -> ...}]
 *   "StateToStep" -> Association[stateId -> step]
 *   "Options" -> Association[...]  (optional)
 *
 * Output: WXF Association with:
 *   "PerState" -> Association[stateId -> alignment result]
 *   "PerTimestep" -> Association[step -> aggregated alignment data]
 *   "GlobalBounds" -> {"PC1Min" -> ..., "PC1Max" -> ..., "CurvatureAbsMax" -> ...}
 */
EXTERN_C DLLEXPORT int performBranchAlignmentBatch(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    namespace bh = viz::blackhole;

    try {
        if (argc != 1) {
            handle_error(libData, "performBranchAlignmentBatch expects 1 argument: WXF ByteArray data");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Get WXF data as ByteArray
        MNumericArray wxf_array = MArgument_getMNumericArray(argv[0]);

        mint rank = libData->numericarrayLibraryFunctions->MNumericArray_getRank(wxf_array);
        if (rank != 1) {
            handle_error(libData, "WXF ByteArray must be 1-dimensional");
            return LIBRARY_FUNCTION_ERROR;
        }

        const mint* dims = libData->numericarrayLibraryFunctions->MNumericArray_getDimensions(wxf_array);
        mint wxf_size = dims[0];

        void* wxf_raw_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(wxf_array);
        const uint8_t* wxf_byte_data = static_cast<const uint8_t*>(wxf_raw_data);

        std::vector<uint8_t> wxf_bytes(wxf_byte_data, wxf_byte_data + wxf_size);

        // Parse WXF input
        wxf::Parser parser(wxf_bytes);
        parser.skip_header();

        // Data structures for batch processing
        std::map<uint32_t, std::vector<std::vector<int64_t>>> state_edges;
        std::map<uint32_t, std::unordered_map<bh::VertexId, float>> state_curvatures;
        std::map<uint32_t, uint32_t> state_to_step;

        // Alignment options with defaults
        bool canonical_orientation = true;
        bool scale_normalization = true;

        parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
            if (key == "States") {
                value_parser.read_association([&](const std::string& state_key, wxf::Parser& state_parser) {
                    uint32_t state_id = static_cast<uint32_t>(std::stoll(state_key));

                    state_parser.read_association([&](const std::string& field_key, wxf::Parser& field_parser) {
                        if (field_key == "Edges") {
                            state_edges[state_id] = field_parser.read<std::vector<std::vector<int64_t>>>();
                        }
                        else if (field_key == "Curvature") {
                            field_parser.read_association([&](const std::string& curv_key, wxf::Parser& curv_parser) {
                                int64_t vid = std::stoll(curv_key);
                                double curv_val = curv_parser.read<double>();
                                state_curvatures[state_id][static_cast<bh::VertexId>(vid)] = static_cast<float>(curv_val);
                            });
                        }
                        else {
                            field_parser.skip_value();
                        }
                    });
                });
            }
            else if (key == "StateToStep") {
                value_parser.read_association([&](const std::string& state_key, wxf::Parser& step_parser) {
                    uint32_t state_id = static_cast<uint32_t>(std::stoll(state_key));
                    int64_t step = step_parser.read<int64_t>();
                    state_to_step[state_id] = static_cast<uint32_t>(step);
                });
            }
            else if (key == "Options") {
                value_parser.read_association([&](const std::string& opt_key, wxf::Parser& opt_parser) {
                    if (opt_key == "CanonicalOrientation") {
                        canonical_orientation = opt_parser.read<bool>();
                    }
                    else if (opt_key == "ScaleNormalization") {
                        scale_normalization = opt_parser.read<bool>();
                    }
                    else {
                        opt_parser.skip_value();
                    }
                });
            }
            else {
                value_parser.skip_value();
            }
        });

        if (state_edges.empty()) {
            handle_error(libData, "No states provided");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Build graphs and curvature vectors in parallel
        std::vector<bh::SimpleGraph> graphs;
        std::vector<std::unordered_map<bh::VertexId, float>> curvatures;
        std::vector<uint32_t> state_ids;

        for (const auto& [state_id, edges_raw] : state_edges) {
            state_ids.push_back(state_id);

            std::vector<bh::Edge> edge_list;
            edge_list.reserve(edges_raw.size());
            for (const auto& edge : edges_raw) {
                if (edge.size() >= 2) {
                    edge_list.push_back(bh::Edge(
                        static_cast<bh::VertexId>(edge[0]),
                        static_cast<bh::VertexId>(edge[1])
                    ));
                }
            }

            bh::SimpleGraph graph;
            graph.build_from_edges(edge_list);
            graphs.push_back(std::move(graph));

            auto curv_it = state_curvatures.find(state_id);
            if (curv_it != state_curvatures.end()) {
                curvatures.push_back(curv_it->second);
            } else {
                curvatures.push_back({});
            }
        }

        // Group states by timestep FIRST
        std::map<uint32_t, std::vector<size_t>> step_to_indices;
        for (size_t i = 0; i < state_ids.size(); ++i) {
            auto it = state_to_step.find(state_ids[i]);
            uint32_t step = (it != state_to_step.end()) ? it->second : 0;
            step_to_indices[step].push_back(i);
        }

        // Use GLOBAL PCA per timestep (ensures consistent axes within each timestep)
        // Cross-timestep alignment uses reference frame from previous timestep
        job_system::JobSystem<int> js(std::thread::hardware_concurrency());
        js.start();

        std::map<uint32_t, bh::AlignmentAggregation> per_timestep;
        bh::AlignmentReferenceFrame reference_frame;  // Propagates between timesteps

        for (const auto& [step, indices] : step_to_indices) {
            // Collect graphs and curvatures for this timestep only
            std::vector<bh::SimpleGraph> step_graphs;
            std::vector<std::unordered_map<bh::VertexId, float>> step_curvatures;
            std::vector<bh::StateId> step_state_ids;

            for (size_t i : indices) {
                step_graphs.push_back(graphs[i]);
                step_curvatures.push_back(curvatures[i]);
                step_state_ids.push_back(state_ids[i]);
            }

            // Per-branch PCA alignment with canonical orientation
            // Pass reference from previous timestep for temporal smoothing
            bh::AlignmentReferenceFrame new_frame;
            per_timestep[step] = bh::align_branches_per_branch(
                step_graphs, step_curvatures, step_state_ids, &js,
                reference_frame.valid ? &reference_frame : nullptr,
                &new_frame, canonical_orientation, scale_normalization);
            reference_frame = new_frame;
        }

        js.shutdown();

        // Compute global bounds
        float global_pc1_min = std::numeric_limits<float>::max();
        float global_pc1_max = std::numeric_limits<float>::lowest();
        float global_pc2_min = std::numeric_limits<float>::max();
        float global_pc2_max = std::numeric_limits<float>::lowest();
        float global_pc3_min = std::numeric_limits<float>::max();
        float global_pc3_max = std::numeric_limits<float>::lowest();
        float global_curv_min = std::numeric_limits<float>::max();
        float global_curv_max = std::numeric_limits<float>::lowest();

        for (const auto& [step, agg] : per_timestep) {
            if (agg.total_points > 0) {
                global_pc1_min = std::min(global_pc1_min, agg.pc1_min);
                global_pc1_max = std::max(global_pc1_max, agg.pc1_max);
                global_pc2_min = std::min(global_pc2_min, agg.pc2_min);
                global_pc2_max = std::max(global_pc2_max, agg.pc2_max);
                global_pc3_min = std::min(global_pc3_min, agg.pc3_min);
                global_pc3_max = std::max(global_pc3_max, agg.pc3_max);
                global_curv_min = std::min(global_curv_min, agg.curvature_min);
                global_curv_max = std::max(global_curv_max, agg.curvature_max);
            }
        }

        float curv_abs_max = std::max(std::abs(global_curv_min), std::abs(global_curv_max));

        // Build WXF output
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::WXFValueAssociation result;

        // PerState -> Association[stateId -> alignment]
        // Extract per-state data from global alignments (points are in globally-aligned coordinates)
        wxf::WXFValueAssociation per_state_assoc;
        for (const auto& [step, agg] : per_timestep) {
            if (agg.total_points == 0) continue;

            // Group points by state_id within this timestep's aggregation
            std::map<bh::StateId, std::vector<size_t>> state_point_indices;
            for (size_t j = 0; j < agg.total_points; ++j) {
                state_point_indices[agg.state_id[j]].push_back(j);
            }

            for (const auto& [sid, indices] : state_point_indices) {
                if (indices.empty()) continue;

                wxf::WXFValueAssociation state_result;

                // Vertices
                wxf::WXFValueList vertices_list;
                for (size_t idx : indices) {
                    vertices_list.push_back(wxf::WXFValue(static_cast<int64_t>(agg.all_vertices[idx])));
                }
                state_result.push_back({wxf::WXFValue("Vertices"), wxf::WXFValue(vertices_list)});

                // PC1, PC2, PC3 (now in globally-aligned coordinates!)
                wxf::WXFValueList pc1_list, pc2_list, pc3_list;
                for (size_t idx : indices) {
                    pc1_list.push_back(wxf::WXFValue(static_cast<double>(agg.all_pc1[idx])));
                    pc2_list.push_back(wxf::WXFValue(static_cast<double>(agg.all_pc2[idx])));
                    pc3_list.push_back(wxf::WXFValue(static_cast<double>(agg.all_pc3[idx])));
                }
                state_result.push_back({wxf::WXFValue("PC1"), wxf::WXFValue(pc1_list)});
                state_result.push_back({wxf::WXFValue("PC2"), wxf::WXFValue(pc2_list)});
                state_result.push_back({wxf::WXFValue("PC3"), wxf::WXFValue(pc3_list)});

                // Curvature and Rank
                wxf::WXFValueList curv_list, rank_list;
                float curv_min = std::numeric_limits<float>::max();
                float curv_max = std::numeric_limits<float>::lowest();
                float curv_sum = 0;
                for (size_t idx : indices) {
                    float c = agg.all_curvature[idx];
                    curv_list.push_back(wxf::WXFValue(static_cast<double>(c)));
                    rank_list.push_back(wxf::WXFValue(static_cast<double>(agg.all_rank[idx])));
                    curv_min = std::min(curv_min, c);
                    curv_max = std::max(curv_max, c);
                    curv_sum += c;
                }
                state_result.push_back({wxf::WXFValue("Curvature"), wxf::WXFValue(curv_list)});
                state_result.push_back({wxf::WXFValue("Rank"), wxf::WXFValue(rank_list)});

                // Stats
                wxf::WXFValueAssociation stats_assoc;
                stats_assoc.push_back({wxf::WXFValue("NumVertices"), wxf::WXFValue(static_cast<int64_t>(indices.size()))});
                stats_assoc.push_back({wxf::WXFValue("CurvatureMin"), wxf::WXFValue(static_cast<double>(curv_min))});
                stats_assoc.push_back({wxf::WXFValue("CurvatureMax"), wxf::WXFValue(static_cast<double>(curv_max))});
                stats_assoc.push_back({wxf::WXFValue("CurvatureMean"), wxf::WXFValue(static_cast<double>(curv_sum / indices.size()))});
                state_result.push_back({wxf::WXFValue("Stats"), wxf::WXFValue(stats_assoc)});

                per_state_assoc.push_back({
                    wxf::WXFValue(std::to_string(sid)),
                    wxf::WXFValue(state_result)
                });
            }
        }
        result.push_back({wxf::WXFValue("PerState"), wxf::WXFValue(per_state_assoc)});

        // PerTimestep -> Association[step -> aggregated data]
        wxf::WXFValueAssociation per_timestep_assoc;
        for (const auto& [step, agg] : per_timestep) {
            if (agg.total_points == 0) continue;

            wxf::WXFValueAssociation step_result;

            // All PC values
            wxf::WXFValueList all_pc1, all_pc2, all_pc3, all_curv, all_rank, branch_ids, all_verts, all_states;
            for (size_t j = 0; j < agg.total_points; ++j) {
                all_pc1.push_back(wxf::WXFValue(static_cast<double>(agg.all_pc1[j])));
                all_pc2.push_back(wxf::WXFValue(static_cast<double>(agg.all_pc2[j])));
                all_pc3.push_back(wxf::WXFValue(static_cast<double>(agg.all_pc3[j])));
                all_curv.push_back(wxf::WXFValue(static_cast<double>(agg.all_curvature[j])));
                all_rank.push_back(wxf::WXFValue(static_cast<double>(agg.all_rank[j])));
                branch_ids.push_back(wxf::WXFValue(static_cast<int64_t>(agg.branch_id[j])));
                all_verts.push_back(wxf::WXFValue(static_cast<int64_t>(agg.all_vertices[j])));
                all_states.push_back(wxf::WXFValue(static_cast<int64_t>(agg.state_id[j])));
            }
            step_result.push_back({wxf::WXFValue("PC1"), wxf::WXFValue(all_pc1)});
            step_result.push_back({wxf::WXFValue("PC2"), wxf::WXFValue(all_pc2)});
            step_result.push_back({wxf::WXFValue("PC3"), wxf::WXFValue(all_pc3)});
            step_result.push_back({wxf::WXFValue("Curvature"), wxf::WXFValue(all_curv)});
            step_result.push_back({wxf::WXFValue("Rank"), wxf::WXFValue(all_rank)});
            step_result.push_back({wxf::WXFValue("BranchId"), wxf::WXFValue(branch_ids)});
            step_result.push_back({wxf::WXFValue("Vertices"), wxf::WXFValue(all_verts)});
            step_result.push_back({wxf::WXFValue("StateId"), wxf::WXFValue(all_states)});

            // Statistics
            wxf::WXFValueAssociation bounds_assoc;
            bounds_assoc.push_back({wxf::WXFValue("TotalPoints"), wxf::WXFValue(static_cast<int64_t>(agg.total_points))});
            bounds_assoc.push_back({wxf::WXFValue("NumBranches"), wxf::WXFValue(static_cast<int64_t>(agg.num_branches))});
            bounds_assoc.push_back({wxf::WXFValue("PC1Min"), wxf::WXFValue(static_cast<double>(agg.pc1_min))});
            bounds_assoc.push_back({wxf::WXFValue("PC1Max"), wxf::WXFValue(static_cast<double>(agg.pc1_max))});
            bounds_assoc.push_back({wxf::WXFValue("PC2Min"), wxf::WXFValue(static_cast<double>(agg.pc2_min))});
            bounds_assoc.push_back({wxf::WXFValue("PC2Max"), wxf::WXFValue(static_cast<double>(agg.pc2_max))});
            bounds_assoc.push_back({wxf::WXFValue("CurvatureMin"), wxf::WXFValue(static_cast<double>(agg.curvature_min))});
            bounds_assoc.push_back({wxf::WXFValue("CurvatureMax"), wxf::WXFValue(static_cast<double>(agg.curvature_max))});
            step_result.push_back({wxf::WXFValue("Bounds"), wxf::WXFValue(bounds_assoc)});

            per_timestep_assoc.push_back({
                wxf::WXFValue(static_cast<int64_t>(step)),
                wxf::WXFValue(step_result)
            });
        }
        result.push_back({wxf::WXFValue("PerTimestep"), wxf::WXFValue(per_timestep_assoc)});

        // GlobalBounds
        wxf::WXFValueAssociation global_bounds;
        global_bounds.push_back({wxf::WXFValue("PC1Min"), wxf::WXFValue(static_cast<double>(global_pc1_min))});
        global_bounds.push_back({wxf::WXFValue("PC1Max"), wxf::WXFValue(static_cast<double>(global_pc1_max))});
        global_bounds.push_back({wxf::WXFValue("PC2Min"), wxf::WXFValue(static_cast<double>(global_pc2_min))});
        global_bounds.push_back({wxf::WXFValue("PC2Max"), wxf::WXFValue(static_cast<double>(global_pc2_max))});
        global_bounds.push_back({wxf::WXFValue("PC3Min"), wxf::WXFValue(static_cast<double>(global_pc3_min))});
        global_bounds.push_back({wxf::WXFValue("PC3Max"), wxf::WXFValue(static_cast<double>(global_pc3_max))});
        global_bounds.push_back({wxf::WXFValue("CurvatureMin"), wxf::WXFValue(static_cast<double>(global_curv_min))});
        global_bounds.push_back({wxf::WXFValue("CurvatureMax"), wxf::WXFValue(static_cast<double>(global_curv_max))});
        global_bounds.push_back({wxf::WXFValue("CurvatureAbsMax"), wxf::WXFValue(static_cast<double>(curv_abs_max))});
        result.push_back({wxf::WXFValue("GlobalBounds"), wxf::WXFValue(global_bounds)});

        // Write output
        wxf_writer.write(wxf::WXFValue(result));
        const auto& wxf_data = wxf_writer.data();

        // Create output ByteArray
        mint wxf_dims[1] = {static_cast<mint>(wxf_data.size())};
        MNumericArray result_array;
        int err = libData->numericarrayLibraryFunctions->MNumericArray_new(MNumericArray_Type_UBit8, 1, wxf_dims, &result_array);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }

        void* result_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(result_array);
        std::memcpy(result_data, wxf_data.data(), wxf_data.size());

        MArgument_setMNumericArray(res, result_array);
        return LIBRARY_NO_ERROR;

    } catch (const wxf::TypeError& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "WXF TypeError in BranchAlignmentBatch: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    } catch (const std::exception& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "Exception in BranchAlignmentBatch: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    }
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData /* libData */) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData /* libData */) {
}