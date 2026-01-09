#include "WolframLibrary.h"
#include "WolframNumericArrayLibrary.h"
#ifdef HAVE_WSTP
#include "wstp.h"
#endif
#include <vector>
#include <set>
#include <unordered_set>
#include <cstring>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>

// Include unified engine headers
#include "hypergraph/unified_hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/debug_log.hpp"

// Include comprehensive WXF library
#include "wxf.hpp"

// Include blackhole analysis (without layout dependency)
#include "blackhole/hausdorff_analysis.hpp"
#include "blackhole/bh_types.hpp"
#include "blackhole/geodesic_analysis.hpp"
#include "blackhole/particle_detection.hpp"
#include "blackhole/curvature_analysis.hpp"
#include "blackhole/entropy_analysis.hpp"
#include "blackhole/rotation_analysis.hpp"
#include "blackhole/branchial_analysis.hpp"
#include "blackhole/bh_initial_condition.hpp"

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

// V1 performRewriting function removed - use performRewritingV2 (unified engine) instead

// Deprecated V1 function stub - returns error directing to V2
EXTERN_C DLLEXPORT int performRewriting(WolframLibraryData libData, mint /* argc */, MArgument* /* argv */, MArgument /* res */) {
    handle_error(libData, "performRewriting (V1) has been removed. Use performRewritingV2 (unified engine) instead.");
    return LIBRARY_FUNCTION_ERROR;
}

/**
 * Perform multiway rewriting evolution using unified V2 engine
 * Input: WXF binary data as 1D byte tensor containing:
 *   Association[
 *     "InitialEdges" -> {{vertices...}, ...},
 *     "Rules" -> <"Rule1" -> {{lhs edges}, {rhs edges}}, ...>,
 *     "Steps" -> integer,
 *     "Options" -> Association[..., "HashStrategy" -> "iUT"|"UT"|"WL", ...]
 *   ]
 *
 * Output: WXF Association with States, Events, CausalEdges, BranchialEdges
 */
EXTERN_C DLLEXPORT int performRewritingV2(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 1) {
            handle_error(libData, "performRewritingV2 expects 1 argument: WXF ByteArray data");
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
        bool uniform_random = false;  // Use uniform random match selection (reservoir sampling)
        size_t matches_per_step = 0;  // Matches per step in uniform random mode (0 = all)

        // Dimension analysis options - compute Hausdorff dimension in C++ instead of WL round-trips
        bool compute_dimensions = false;
        int dim_min_radius = 1;
        int dim_max_radius = 5;

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

        // Curvature analysis options - Ollivier-Ricci and dimension gradient curvature
        bool compute_curvature = false;
        bool curvature_ollivier_ricci = true;
        bool curvature_dimension_gradient = true;
        float curvature_ricci_alpha = 0.5f;  // Laziness parameter for Ollivier-Ricci
        int curvature_gradient_radius = 2;

        // Entropy analysis options - graph entropy and information measures
        bool compute_entropy = false;
        bool entropy_local = true;
        bool entropy_mutual_info = true;
        bool entropy_fisher_info = true;
        int entropy_neighborhood_radius = 2;

        // Rotation curve analysis options - orbital velocity vs radius
        bool compute_rotation_curve = false;
        int rotation_min_radius = 2;
        int rotation_max_radius = 20;
        int rotation_orbits_per_radius = 4;

        // Hilbert space analysis options - state bitvector inner products
        bool compute_hilbert_space = false;
        int hilbert_step = -1;  // Which step to analyze (-1 = all steps)

        // Branchial analysis options - distribution sharpness and branch entropy
        bool compute_branchial = false;

        // Multispace analysis options - vertex/edge probabilities across branches
        bool compute_multispace = false;

        // Topology configuration (for initial condition generation)
        std::string topology_type = "Flat";

        // Sprinkling configuration (for Minkowski causal set initial conditions)
        std::string initial_condition_type = "Edges";  // "Edges" (from InitialEdges) or "Sprinkling"
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
                        } else if (option_key == "CurvatureRicciAlpha") {
                            curvature_ricci_alpha = static_cast<float>(option_parser.read<double>());
                        } else if (option_key == "CurvatureGradientRadius") {
                            curvature_gradient_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "EntropyNeighborhoodRadius") {
                            entropy_neighborhood_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "RotationMinRadius") {
                            rotation_min_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "RotationMaxRadius") {
                            rotation_max_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "RotationOrbitsPerRadius") {
                            rotation_orbits_per_radius = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "HilbertStep") {
                            // Which step to analyze for Hilbert space (-1 = all steps)
                            hilbert_step = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "Topology") {
                            topology_type = option_parser.read<std::string>();
                        } else if (option_key == "InitialCondition") {
                            initial_condition_type = option_parser.read<std::string>();
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
                                // Compute Ollivier-Ricci and dimension gradient curvature
                                compute_curvature = value;
                            } else if (option_key == "CurvatureOllivierRicci") {
                                curvature_ollivier_ricci = value;
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
                            } else if (option_key == "RotationCurveAnalysis") {
                                // Compute rotation curve (orbital velocity vs radius)
                                compute_rotation_curve = value;
                            } else if (option_key == "HilbertSpaceAnalysis") {
                                // Compute Hilbert space analysis (state bitvector inner products)
                                compute_hilbert_space = value;
                            } else if (option_key == "BranchialAnalysis") {
                                // Compute branchial analysis (distribution sharpness, branch entropy)
                                compute_branchial = value;
                            } else if (option_key == "MultispaceAnalysis") {
                                // Compute multispace analysis (vertex/edge probabilities)
                                compute_multispace = value;
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
                oss << "HGEvolveV2: Generated Minkowski sprinkling with "
                    << sprinkling.points.size() << " points, "
                    << sprinkling_edges.size() << " edges";
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

        // Create unified hypergraph
        hypergraph::UnifiedHypergraph hg;

        // Set hash strategy
        if (hash_strategy == "WL") {
            hg.set_hash_strategy(hypergraph::HashStrategy::WL);
        } else if (hash_strategy == "UT") {
            hg.set_hash_strategy(hypergraph::HashStrategy::UniquenessTree);
        } else {
            hg.set_hash_strategy(hypergraph::HashStrategy::IncrementalUniquenessTree);
        }

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
        // V2 now supports multiple initial states for exploring the full multiway system
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
            print_to_frontend(libData, "HGEvolveV2: Starting evolution...");
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
                oss << "HGEvolveV2: ABORTED after " << evolution_ms << "ms. ";
            } else {
                oss << "HGEvolveV2: Evolution complete in " << evolution_ms << "ms. ";
            }
            oss << "States: " << hg.num_canonical_states() << ", "
                << "Events: " << hg.num_events() << ", "
                << "Causal: " << hg.num_causal_event_pairs() << ", "  // v1 semantics
                << "Branchial: " << hg.num_branchial_edges();
            print_to_frontend(libData, oss.str());
            if (!was_aborted) {
                print_to_frontend(libData, "HGEvolveV2: Starting serialization...");
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
        float global_dim_min = std::numeric_limits<float>::max();
        float global_dim_max = std::numeric_limits<float>::lowest();

        if (compute_dimensions) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolveV2: Computing dimension analysis...");
            }
#endif
            uint32_t num_states = hg.num_states();

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
                    bh::SimpleGraph graph;
                    graph.build_from_edges(edges);

                    bh::DimensionConfig config;
                    config.min_radius = dim_min_radius;
                    config.max_radius = dim_max_radius;

                    auto per_vertex = bh::estimate_all_dimensions(graph, config);
                    auto stats = bh::compute_dimension_stats(per_vertex);

                    state_dimension_stats[sid] = stats;
                    state_vertex_dimensions[sid] = std::move(per_vertex);

                    // Track global range
                    if (stats.count > 0) {
                        global_dim_min = std::min(global_dim_min, stats.mean);
                        global_dim_max = std::max(global_dim_max, stats.mean);
                    }
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolveV2: Dimension analysis complete. Analyzed "
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
        std::unordered_map<uint32_t, float> state_geodesic_bundle_spread;

        if (compute_geodesics) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolveV2: Computing geodesic analysis...");
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
                    for (const auto& path : geo_result.paths) {
                        paths.push_back(path.vertices);
                        proper_times.push_back(path.proper_time);
                    }
                    state_geodesic_paths[sid] = std::move(paths);
                    state_geodesic_proper_times[sid] = std::move(proper_times);
                    state_geodesic_bundle_spread[sid] = geo_result.mean_spread;
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolveV2: Geodesic analysis complete. Analyzed "
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

        if (detect_particles) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolveV2: Detecting topological defects...");
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
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolveV2: Particle detection complete. Analyzed "
                    << state_defects.size() << " states";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // ==========================================================================
        // Curvature Analysis - Ollivier-Ricci and dimension gradient curvature
        // ==========================================================================
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_ollivier_ricci;
        std::unordered_map<uint32_t, std::unordered_map<bh::VertexId, float>> state_dimension_gradient;
        std::unordered_map<uint32_t, float> state_mean_curvature;

        if (compute_curvature) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolveV2: Computing curvature analysis...");
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

                    // Configure curvature analysis
                    bh::CurvatureConfig curv_config;
                    curv_config.compute_ollivier_ricci = curvature_ollivier_ricci;
                    curv_config.compute_dimension_gradient = curvature_dimension_gradient;
                    curv_config.ricci_alpha = curvature_ricci_alpha;
                    curv_config.gradient_radius = curvature_gradient_radius;

                    // Get dimension data if available
                    const std::vector<float>* dims = nullptr;
                    auto it = state_vertex_dimensions.find(sid);
                    if (it != state_vertex_dimensions.end()) {
                        dims = &it->second;
                    }

                    // Run curvature analysis
                    auto curv_result = bh::analyze_curvature(graph, curv_config, dims);

                    // Store results
                    if (curvature_ollivier_ricci) {
                        state_ollivier_ricci[sid] = curv_result.ollivier_ricci_map;
                    }
                    if (curvature_dimension_gradient) {
                        state_dimension_gradient[sid] = curv_result.dimension_gradient_map;
                    }
                    state_mean_curvature[sid] = curv_result.mean_ollivier_ricci;
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolveV2: Curvature analysis complete. Analyzed "
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

        if (compute_entropy) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolveV2: Computing entropy analysis...");
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

                    // Configure entropy analysis
                    bh::EntropyConfig ent_config;
                    ent_config.compute_local_entropy = entropy_local;
                    ent_config.compute_mutual_info = entropy_mutual_info;
                    ent_config.compute_fisher_info = entropy_fisher_info;
                    ent_config.neighborhood_radius = entropy_neighborhood_radius;

                    // Get dimension data if available (for Fisher info)
                    const std::vector<float>* dims = nullptr;
                    auto it = state_vertex_dimensions.find(sid);
                    if (it != state_vertex_dimensions.end()) {
                        dims = &it->second;
                    }

                    // Run entropy analysis
                    auto ent_result = bh::analyze_entropy(graph, ent_config, dims);

                    // Store results
                    state_degree_entropy[sid] = ent_result.degree_entropy;
                    state_graph_entropy[sid] = ent_result.graph_entropy;
                    if (entropy_local) {
                        state_local_entropy[sid] = ent_result.local_entropy_map;
                    }
                    if (entropy_mutual_info) {
                        state_mutual_info[sid] = ent_result.mutual_info_map;
                    }
                    if (entropy_fisher_info) {
                        state_fisher_info[sid] = ent_result.fisher_info_map;
                    }
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolveV2: Entropy analysis complete. Analyzed "
                    << state_degree_entropy.size() << " states";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // ==========================================================================
        // Rotation Curve Analysis - orbital velocity vs radius
        // ==========================================================================
        std::unordered_map<uint32_t, bh::RotationCurveResult> state_rotation_curves;

        if (compute_rotation_curve) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolveV2: Computing rotation curve analysis...");
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

                    // Configure rotation analysis
                    bh::RotationConfig rot_config;
                    rot_config.min_radius = rotation_min_radius;
                    rot_config.max_radius = rotation_max_radius;
                    rot_config.orbits_per_radius = rotation_orbits_per_radius;
                    rot_config.compute_power_law_fit = true;
                    rot_config.detect_flat_rotation = true;

                    // Get dimension data if available
                    const std::vector<float>* dims = nullptr;
                    auto it = state_vertex_dimensions.find(sid);
                    if (it != state_vertex_dimensions.end()) {
                        dims = &it->second;
                    }

                    // Run rotation curve analysis
                    auto rot_result = bh::analyze_rotation_curve(graph, rot_config, dims);
                    state_rotation_curves[sid] = std::move(rot_result);
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolveV2: Rotation curve analysis complete. Analyzed "
                    << state_rotation_curves.size() << " states";
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // ==========================================================================
        // Branchial-based Analyses (Hilbert, Branchial, Multispace)
        // Build shared BranchState structures once if any of these are needed
        // ==========================================================================
        bh::HilbertSpaceAnalysis hilbert_result;
        bh::BranchialAnalysisResult branchial_result;
        bool has_hilbert_data = false;
        bool has_branchial_data = false;
        bool has_multispace_data = false;

        // Shared vertex/edge probability data for multispace
        std::unordered_map<bh::VertexId, float> multispace_vertex_probs;
        std::map<std::pair<uint32_t, uint32_t>, float> multispace_edge_probs;
        float multispace_mean_vertex_prob = 0.0f;
        float multispace_mean_edge_prob = 0.0f;
        float multispace_total_entropy = 0.0f;

        if (compute_hilbert_space || compute_branchial || compute_multispace) {
#ifdef HAVE_WSTP
            if (show_progress) {
                print_to_frontend(libData, "HGEvolveV2: Building branchial structures...");
            }
#endif
            // Build BranchState structures from the unified hypergraph
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
                        print_to_frontend(libData, "HGEvolveV2: Computing Hilbert space analysis...");
                    }
#endif
                    if (hilbert_step >= 0) {
                        hilbert_result = bh::analyze_hilbert_space(branchial_graph, static_cast<uint32_t>(hilbert_step));
                    } else {
                        hilbert_result = bh::analyze_hilbert_space_full(branchial_graph);
                    }
                    has_hilbert_data = (hilbert_result.num_states > 0);
                }

                // Branchial Analysis (distribution sharpness, branch entropy)
                if (compute_branchial) {
#ifdef HAVE_WSTP
                    if (show_progress) {
                        print_to_frontend(libData, "HGEvolveV2: Computing branchial analysis...");
                    }
#endif
                    branchial_result = bh::analyze_branchial(branch_states);
                    has_branchial_data = (branchial_result.num_unique_vertices > 0);
                }

                // Multispace Analysis (vertex/edge probabilities across branches)
                if (compute_multispace) {
#ifdef HAVE_WSTP
                    if (show_progress) {
                        print_to_frontend(libData, "HGEvolveV2: Computing multispace analysis...");
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
                }
            }

#ifdef HAVE_WSTP
            if (show_progress) {
                std::ostringstream oss;
                oss << "HGEvolveV2: Branchial analyses complete.";
                if (has_hilbert_data) {
                    oss << " Hilbert: " << hilbert_result.num_states << " states.";
                }
                if (has_branchial_data) {
                    oss << " Branchial: " << branchial_result.num_unique_vertices << " vertices.";
                }
                if (has_multispace_data) {
                    oss << " Multispace: " << multispace_vertex_probs.size() << " vertices.";
                }
                print_to_frontend(libData, oss.str());
            }
#endif
        }

        // Build WXF output - only include requested data components
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::ValueAssociation full_result;

        // States -> Association[state_id -> state_data]
        // Send ALL states (not just canonical) - WL uses CanonicalId/ContentStateId for vertex merging
        // Each state includes: Id, CanonicalId, ContentStateId, Step, Edges, IsInitial
        // - CanonicalId: isomorphism-based (for Full mode) - isomorphic states share ID
        // - ContentStateId: content-based (for Automatic mode) - same-content states share ID
        // This matches reference behavior where canonicalization is applied at display time
        if (include_states) {
            wxf::ValueAssociation states_assoc;
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
                wxf::ValueList edge_list;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    wxf::ValueList edge_data;
                    edge_data.push_back(wxf::Value(static_cast<int64_t>(eid)));
                    for (uint8_t i = 0; i < edge.arity; ++i) {
                        edge_data.push_back(wxf::Value(static_cast<int64_t>(edge.vertices[i])));
                    }
                    edge_list.push_back(wxf::Value(edge_data));
                });

                wxf::ValueAssociation state_assoc;
                state_assoc.push_back({wxf::Value("Id"), wxf::Value(static_cast<int64_t>(sid))});
                state_assoc.push_back({wxf::Value("CanonicalId"), wxf::Value(static_cast<int64_t>(canonical_id))});
                state_assoc.push_back({wxf::Value("ContentStateId"), wxf::Value(static_cast<int64_t>(content_id))});
                state_assoc.push_back({wxf::Value("Step"), wxf::Value(static_cast<int64_t>(state.step))});
                state_assoc.push_back({wxf::Value("Edges"), wxf::Value(edge_list)});
                state_assoc.push_back({wxf::Value("IsInitial"), wxf::Value(state.step == 0)});

                states_assoc.push_back({wxf::Value(static_cast<int64_t>(sid)), wxf::Value(state_assoc)});
            }
            full_result.push_back({wxf::Value("States"), wxf::Value(states_assoc)});
        }

        // Events -> Association[event_id -> event_data]
        // Only canonical events are sent (for graph vertices)
        // State IDs are mapped through get_canonical_state() so edges connect canonical states
        if (include_events) {
            // Send ALL events (not just canonical) - WL uses CanonicalId for vertex merging
            // This preserves event multiplicity: multiple events with same canonical ID
            // map to one vertex, but their edges to different output states are preserved.
            wxf::ValueAssociation events_assoc;
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

                wxf::ValueAssociation event_data;
                event_data.push_back({wxf::Value("Id"), wxf::Value(static_cast<int64_t>(eid))});
                event_data.push_back({wxf::Value("CanonicalId"), wxf::Value(canonical_event_id)});
                event_data.push_back({wxf::Value("RuleIndex"), wxf::Value(static_cast<int64_t>(event.rule_index))});
                // Raw state IDs (always included)
                event_data.push_back({wxf::Value("InputState"), wxf::Value(raw_input_state_id)});
                event_data.push_back({wxf::Value("OutputState"), wxf::Value(raw_output_state_id)});
                // Canonical state IDs (for when state canonicalization is enabled)
                event_data.push_back({wxf::Value("CanonicalInputState"), wxf::Value(canonical_input_state_id)});
                event_data.push_back({wxf::Value("CanonicalOutputState"), wxf::Value(canonical_output_state_id)});

                // Consumed/produced edges
                wxf::ValueList consumed_list, produced_list;
                for (uint8_t i = 0; i < event.num_consumed; ++i) {
                    consumed_list.push_back(wxf::Value(static_cast<int64_t>(event.consumed_edges[i])));
                }
                for (uint8_t i = 0; i < event.num_produced; ++i) {
                    produced_list.push_back(wxf::Value(static_cast<int64_t>(event.produced_edges[i])));
                }
                event_data.push_back({wxf::Value("ConsumedEdges"), wxf::Value(consumed_list)});
                event_data.push_back({wxf::Value("ProducedEdges"), wxf::Value(produced_list)});

                events_assoc.push_back({wxf::Value(static_cast<int64_t>(eid)), wxf::Value(event_data)});
            }
            full_result.push_back({wxf::Value("Events"), wxf::Value(events_assoc)});
        }

        // EventsMinimal -> Association[event_id -> {Id, CanonicalId, RuleIndex, InputState, OutputState, CanonicalInputState, CanonicalOutputState}]
        // Reduced event data for graph structure variants that don't need full event details
        // Send ALL events - WL uses CanonicalId for vertex merging, RuleIndex for Event=Automatic grouping
        if (include_events_minimal && !include_events) {
            wxf::ValueAssociation events_assoc;
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

                wxf::ValueAssociation event_data;
                event_data.push_back({wxf::Value("Id"), wxf::Value(static_cast<int64_t>(eid))});
                event_data.push_back({wxf::Value("CanonicalId"), wxf::Value(canonical_event_id)});
                event_data.push_back({wxf::Value("RuleIndex"), wxf::Value(static_cast<int64_t>(event.rule_index))});
                // Raw state IDs (always included)
                event_data.push_back({wxf::Value("InputState"), wxf::Value(raw_input_state_id)});
                event_data.push_back({wxf::Value("OutputState"), wxf::Value(raw_output_state_id)});
                // Canonical state IDs (for when state canonicalization is enabled)
                event_data.push_back({wxf::Value("CanonicalInputState"), wxf::Value(canonical_input_state_id)});
                event_data.push_back({wxf::Value("CanonicalOutputState"), wxf::Value(canonical_output_state_id)});

                events_assoc.push_back({wxf::Value(static_cast<int64_t>(eid)), wxf::Value(event_data)});
            }
            full_result.push_back({wxf::Value("Events"), wxf::Value(events_assoc)});
        }

        // CausalEdges -> List of {From -> canonical_event_id, To -> canonical_event_id}
        // Endpoints are mapped to canonical event IDs for graph structure
        // Deduplicated by RAW (producer, consumer) pairs to remove internal duplication
        // but preserving all unique raw relationships (which may share canonical endpoints)
        if (include_causal_edges) {
            wxf::ValueList causal_edges;
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

                wxf::ValueAssociation edge_data;
                edge_data.push_back({wxf::Value("From"), wxf::Value(static_cast<int64_t>(canonical_from))});
                edge_data.push_back({wxf::Value("To"), wxf::Value(static_cast<int64_t>(canonical_to))});
                // Also include raw event IDs for when event canonicalization is disabled
                edge_data.push_back({wxf::Value("RawFrom"), wxf::Value(static_cast<int64_t>(edge.producer))});
                edge_data.push_back({wxf::Value("RawTo"), wxf::Value(static_cast<int64_t>(edge.consumer))});
                causal_edges.push_back(wxf::Value(edge_data));
            }
            full_result.push_back({wxf::Value("CausalEdges"), wxf::Value(causal_edges)});
        }

        // BranchialEdges -> List of {From -> canonical_event_id, To -> canonical_event_id}
        // For Evolution*Branchial graphs where event vertices need canonical IDs
        // NO deduplication - multiplicity matters for branchial edges
        if (include_branchial_edges) {
            wxf::ValueList branchial_edges;
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

                wxf::ValueAssociation edge_data;
                edge_data.push_back({wxf::Value("From"), wxf::Value(static_cast<int64_t>(canonical_from))});
                edge_data.push_back({wxf::Value("To"), wxf::Value(static_cast<int64_t>(canonical_to))});
                branchial_edges.push_back(wxf::Value(edge_data));
            }
            full_result.push_back({wxf::Value("BranchialEdges"), wxf::Value(branchial_edges)});
        }

        // BranchialStateEdges -> List of {From -> canonical_state_id, To -> canonical_state_id}
        // BranchialStateVertices -> List of unique state IDs that appear in branchial edges
        // For BranchialGraph where state vertices are the output states of events
        // NOTE: Do NOT deduplicate by canonical state pair - reference preserves edge multiplicity
        if (include_branchial_state_edges) {
            wxf::ValueList branchial_state_edges;
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
                wxf::ValueAssociation edge_data;
                edge_data.push_back({wxf::Value("From"), wxf::Value(static_cast<int64_t>(state1))});
                edge_data.push_back({wxf::Value("To"), wxf::Value(static_cast<int64_t>(state2))});
                branchial_state_edges.push_back(wxf::Value(edge_data));
            }
            full_result.push_back({wxf::Value("BranchialStateEdges"), wxf::Value(branchial_state_edges)});

            // Send unique state vertices
            wxf::ValueList state_vertices;
            for (hypergraph::StateId sid : unique_states) {
                state_vertices.push_back(wxf::Value(static_cast<int64_t>(sid)));
            }
            full_result.push_back({wxf::Value("BranchialStateVertices"), wxf::Value(state_vertices)});
        }

        // BranchialStateEdgesAllSiblings: ALL pairs of output states from same input state
        // This matches reference BranchialGraph behavior (no overlap check, all siblings)
        if (include_branchial_state_edges_all_siblings) {
            wxf::ValueList branchial_state_edges;
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
                        wxf::ValueAssociation edge_data;
                        edge_data.push_back({wxf::Value("From"), wxf::Value(static_cast<int64_t>(state1))});
                        edge_data.push_back({wxf::Value("To"), wxf::Value(static_cast<int64_t>(state2))});
                        branchial_state_edges.push_back(wxf::Value(edge_data));
                    }
                }
            });

            full_result.push_back({wxf::Value("BranchialStateEdges"), wxf::Value(branchial_state_edges)});

            // Send unique state vertices
            wxf::ValueList state_vertices;
            for (hypergraph::StateId sid : unique_states) {
                state_vertices.push_back(wxf::Value(static_cast<int64_t>(sid)));
            }
            full_result.push_back({wxf::Value("BranchialStateVertices"), wxf::Value(state_vertices)});
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
            auto serialize_state_edges = [&](hypergraph::StateId sid) -> wxf::ValueList {
                wxf::ValueList edge_list;
                hg.get_state(sid).edges.for_each([&](hypergraph::EdgeId eid) {
                    const hypergraph::Edge& edge = hg.get_edge(eid);
                    wxf::ValueList e;
                    e.push_back(wxf::Value(static_cast<int64_t>(eid)));
                    for (uint8_t i = 0; i < edge.arity; ++i)
                        e.push_back(wxf::Value(static_cast<int64_t>(edge.vertices[i])));
                    edge_list.push_back(wxf::Value(e));
                });
                return edge_list;
            };

            // Helper: Serialize state data for tooltips
            auto serialize_state_data = [&](hypergraph::StateId sid) -> wxf::ValueAssociation {
                const hypergraph::State& state = hg.get_state(sid);
                wxf::ValueAssociation d;
                d.push_back({wxf::Value("Id"), wxf::Value(static_cast<int64_t>(sid))});
                d.push_back({wxf::Value("CanonicalId"), wxf::Value(static_cast<int64_t>(hg.get_canonical_state(sid)))});
                d.push_back({wxf::Value("Step"), wxf::Value(static_cast<int64_t>(state.step))});
                d.push_back({wxf::Value("Edges"), wxf::Value(serialize_state_edges(sid))});
                d.push_back({wxf::Value("IsInitial"), wxf::Value(state.step == 0)});
                return d;
            };

            // Helper: Serialize event data for tooltips
            auto serialize_event_data = [&](hypergraph::EventId eid) -> wxf::ValueAssociation {
                const hypergraph::Event& e = hg.get_event(eid);
                wxf::ValueAssociation d;
                d.push_back({wxf::Value("Id"), wxf::Value(static_cast<int64_t>(eid))});
                d.push_back({wxf::Value("CanonicalId"), wxf::Value(get_effective_event_id(eid))});
                d.push_back({wxf::Value("RuleIndex"), wxf::Value(static_cast<int64_t>(e.rule_index))});
                d.push_back({wxf::Value("InputState"), wxf::Value(static_cast<int64_t>(e.input_state))});
                d.push_back({wxf::Value("OutputState"), wxf::Value(static_cast<int64_t>(e.output_state))});
                // Consumed/produced edges
                wxf::ValueList consumed, produced;
                for (uint8_t i = 0; i < e.num_consumed; ++i)
                    consumed.push_back(wxf::Value(static_cast<int64_t>(e.consumed_edges[i])));
                for (uint8_t i = 0; i < e.num_produced; ++i)
                    produced.push_back(wxf::Value(static_cast<int64_t>(e.produced_edges[i])));
                d.push_back({wxf::Value("ConsumedEdges"), wxf::Value(consumed)});
                d.push_back({wxf::Value("ProducedEdges"), wxf::Value(produced)});
                // For styled rendering: include input/output state edges
                d.push_back({wxf::Value("InputStateEdges"), wxf::Value(serialize_state_edges(e.input_state))});
                d.push_back({wxf::Value("OutputStateEdges"), wxf::Value(serialize_state_edges(e.output_state))});
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
            wxf::ValueAssociation all_graph_data;

            for (const std::string& graph_property : graph_properties) {
                wxf::ValueList vertices;
                wxf::ValueList edges;
                wxf::ValueAssociation vertex_data;

                // Parse property name
                bool is_states = graph_property.rfind("States", 0) == 0;
                bool is_causal = graph_property.rfind("Causal", 0) == 0;
                bool is_branchial = graph_property.rfind("Branchial", 0) == 0;
                bool is_evolution = graph_property.find("Evolution") != std::string::npos;
                bool has_causal = is_evolution && graph_property.find("Causal") != std::string::npos;
                bool has_branchial = is_evolution && graph_property.find("Branchial") != std::string::npos;

                // Helper to add edge to edges list
                auto add_graph_edge = [&](wxf::Value from, wxf::Value to, const std::string& type,
                                          wxf::ValueAssociation data = {}) {
                    wxf::ValueAssociation edge;
                    edge.push_back({wxf::Value("From"), from});
                    edge.push_back({wxf::Value("To"), to});
                    edge.push_back({wxf::Value("Type"), wxf::Value(type)});
                    if (!data.empty()) edge.push_back({wxf::Value("Data"), wxf::Value(data)});
                    edges.push_back(wxf::Value(edge));
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
                    wxf::ValueList from_tag = {wxf::Value("E"), wxf::Value(from)};
                    wxf::ValueList to_tag = {wxf::Value("E"), wxf::Value(to)};

                    size_t num_edges = edge_deduplication ? 1 : count;
                    for (size_t k = 0; k < num_edges; ++k) {
                        wxf::ValueAssociation causal_data;
                        causal_data.push_back({wxf::Value("ProducerEvent"), wxf::Value(from)});
                        causal_data.push_back({wxf::Value("ConsumerEvent"), wxf::Value(to)});
                        // Add unique index to prevent Mathematica Graph[] deduplication
                        if (num_edges > 1) {
                            causal_data.push_back({wxf::Value("EdgeIndex"), wxf::Value(static_cast<int64_t>(k))});
                        }
                        add_graph_edge(wxf::Value(from_tag), wxf::Value(to_tag), "Causal", causal_data);
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
                    vertices.push_back(wxf::Value(eff));
                    vertex_data.push_back({wxf::Value(eff), wxf::Value(serialize_state_data(raw))});
                }
                // Edges: events (state → state)
                for (uint32_t eid = 0; eid < hg.num_raw_events(); ++eid) {
                    if (!is_valid_event(eid)) continue;
                    const hypergraph::Event& e = hg.get_event(eid);
                    add_graph_edge(wxf::Value(get_effective_state_id(e.input_state)),
                                   wxf::Value(get_effective_state_id(e.output_state)),
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
                    wxf::ValueList tag = {wxf::Value("E"), wxf::Value(eff)};
                    vertices.push_back(wxf::Value(tag));
                    vertex_data.push_back({wxf::Value(tag), wxf::Value(serialize_event_data(raw))});
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
                    vertices.push_back(wxf::Value(eff));
                    vertex_data.push_back({wxf::Value(eff), wxf::Value(serialize_state_data(raw))});
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
                    wxf::ValueAssociation branchial_data;
                    branchial_data.push_back({wxf::Value("State1"), wxf::Value(state1)});
                    branchial_data.push_back({wxf::Value("State2"), wxf::Value(state2)});
                    add_graph_edge(wxf::Value(state1), wxf::Value(state2), "Branchial", branchial_data);
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
                    wxf::ValueList tag = {wxf::Value("S"), wxf::Value(sid)};
                    vertices.push_back(wxf::Value(tag));
                    vertex_data.push_back({wxf::Value(tag), wxf::Value(serialize_state_data(raw_states[sid]))});
                }
                // Add event vertices
                for (auto& [eff, raw] : event_verts) {
                    wxf::ValueList tag = {wxf::Value("E"), wxf::Value(eff)};
                    vertices.push_back(wxf::Value(tag));
                    vertex_data.push_back({wxf::Value(tag), wxf::Value(serialize_event_data(raw))});
                }

                // Edges: state↔event from each event (use effective IDs for deduplication)
                for (uint32_t eid = 0; eid < hg.num_raw_events(); ++eid) {
                    if (!is_valid_event(eid)) continue;
                    const hypergraph::Event& e = hg.get_event(eid);
                    int64_t eff_eid = get_effective_event_id(eid);
                    wxf::ValueList s_in = {wxf::Value("S"), wxf::Value(get_effective_state_id(e.input_state))};
                    wxf::ValueList s_out = {wxf::Value("S"), wxf::Value(get_effective_state_id(e.output_state))};
                    wxf::ValueList e_tag = {wxf::Value("E"), wxf::Value(eff_eid)};
                    // Tooltip data: effective event ID (matches graph vertices)
                    wxf::ValueAssociation edge_data;
                    edge_data.push_back({wxf::Value("EventId"), wxf::Value(eff_eid)});
                    add_graph_edge(wxf::Value(s_in), wxf::Value(e_tag), "StateEvent", edge_data);
                    add_graph_edge(wxf::Value(e_tag), wxf::Value(s_out), "EventState", edge_data);
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
                        wxf::ValueList from_tag = {wxf::Value("E"), wxf::Value(from)};
                        wxf::ValueList to_tag = {wxf::Value("E"), wxf::Value(to)};
                        // Tooltip data: effective event IDs (matches graph vertices)
                        wxf::ValueAssociation branchial_data;
                        branchial_data.push_back({wxf::Value("Event1"), wxf::Value(from)});
                        branchial_data.push_back({wxf::Value("Event2"), wxf::Value(to)});
                        add_graph_edge(wxf::Value(from_tag), wxf::Value(to_tag), "Branchial", branchial_data);
                    }
                }
            }

                // Build GraphData association for this property
                wxf::ValueAssociation graph_data;
                graph_data.push_back({wxf::Value("Vertices"), wxf::Value(vertices)});
                graph_data.push_back({wxf::Value("Edges"), wxf::Value(edges)});
                graph_data.push_back({wxf::Value("VertexData"), wxf::Value(vertex_data)});
                all_graph_data.push_back({wxf::Value(graph_property), wxf::Value(graph_data)});
            }  // end for each graph_property

            // Add keyed GraphData to result
            full_result.push_back({wxf::Value("GraphData"), wxf::Value(all_graph_data)});
        }

        // Only include counts when requested
        if (include_num_states) {
            full_result.push_back({wxf::Value("NumStates"), wxf::Value(static_cast<int64_t>(hg.num_canonical_states()))});
        }
        if (include_num_events) {
            full_result.push_back({wxf::Value("NumEvents"), wxf::Value(static_cast<int64_t>(engine.num_events()))});
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
            full_result.push_back({wxf::Value("NumCausalEdges"), wxf::Value(causal_count)});
        }
        if (include_num_branchial_edges) {
            full_result.push_back({wxf::Value("NumBranchialEdges"), wxf::Value(static_cast<int64_t>(hg.num_branchial_edges()))});
        }

        // GlobalEdges -> List of all edges created during evolution
        // Each edge is {edge_id, v1, v2, ...}
        if (include_global_edges) {
            wxf::ValueList global_edges;
            uint32_t num_edges = hg.num_edges();
            for (uint32_t eid = 0; eid < num_edges; ++eid) {
                const hypergraph::Edge& edge = hg.get_edge(eid);
                if (edge.id == hypergraph::INVALID_ID) continue;

                wxf::ValueList edge_data;
                edge_data.push_back(wxf::Value(static_cast<int64_t>(eid)));
                for (uint8_t i = 0; i < edge.arity; ++i) {
                    edge_data.push_back(wxf::Value(static_cast<int64_t>(edge.vertices[i])));
                }
                global_edges.push_back(wxf::Value(edge_data));
            }
            full_result.push_back(std::make_pair(wxf::Value("GlobalEdges"), wxf::Value(global_edges)));
        }

        // StateBitvectors -> Association[state_id -> List of edge IDs present in that state]
        // Represents each state's edge set (the bitvector) as a list of edge indices
        if (include_state_bitvectors) {
            wxf::ValueAssociation state_bitvectors;
            uint32_t num_states = hg.num_states();
            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;

                // Convert SparseBitset to list of edge IDs
                wxf::ValueList edge_ids;
                state.edges.for_each([&](hypergraph::EdgeId eid) {
                    edge_ids.push_back(wxf::Value(static_cast<int64_t>(eid)));
                });

                state_bitvectors.push_back(std::make_pair(
                    wxf::Value(static_cast<int64_t>(sid)),
                    wxf::Value(edge_ids)
                ));
            }
            full_result.push_back(std::make_pair(wxf::Value("StateBitvectors"), wxf::Value(state_bitvectors)));
        }

        // DimensionData -> Association["PerState" -> {...}, "GlobalRange" -> {min, max}]
        if (compute_dimensions && !state_dimension_stats.empty()) {
            wxf::ValueAssociation dim_data;

            // Per-state stats: state_id -> {Mean, Min, Max, StdDev}
            wxf::ValueAssociation per_state;
            for (const auto& [sid, stats] : state_dimension_stats) {
                wxf::ValueAssociation stats_assoc;
                stats_assoc.push_back({wxf::Value("Mean"), wxf::Value(static_cast<double>(stats.mean))});
                stats_assoc.push_back({wxf::Value("Min"), wxf::Value(static_cast<double>(stats.min))});
                stats_assoc.push_back({wxf::Value("Max"), wxf::Value(static_cast<double>(stats.max))});
                stats_assoc.push_back({wxf::Value("StdDev"), wxf::Value(static_cast<double>(stats.stddev))});
                per_state.push_back({wxf::Value(static_cast<int64_t>(sid)), wxf::Value(stats_assoc)});
            }
            dim_data.push_back({wxf::Value("PerState"), wxf::Value(per_state)});

            // Global range for color normalization
            wxf::ValueList range;
            if (global_dim_min <= global_dim_max) {
                range.push_back(wxf::Value(static_cast<double>(global_dim_min)));
                range.push_back(wxf::Value(static_cast<double>(global_dim_max)));
            } else {
                // No valid data - use defaults
                range.push_back(wxf::Value(0.0));
                range.push_back(wxf::Value(3.0));
            }
            dim_data.push_back({wxf::Value("GlobalRange"), wxf::Value(range)});

            full_result.push_back({wxf::Value("DimensionData"), wxf::Value(dim_data)});
        }

        // GeodesicData -> Association["PerState" -> {...}]
        // Each state: state_id -> {"Paths" -> {...}, "ProperTimes" -> {...}, "BundleSpread" -> float}
        if (compute_geodesics && !state_geodesic_paths.empty()) {
            wxf::ValueAssociation geodesic_data;

            // Per-state paths
            wxf::ValueAssociation per_state;
            for (const auto& [sid, paths] : state_geodesic_paths) {
                wxf::ValueAssociation state_data;

                // Paths: list of lists of vertex IDs
                wxf::ValueList path_list;
                for (const auto& path : paths) {
                    wxf::ValueList vertices;
                    for (bh::VertexId v : path) {
                        vertices.push_back(wxf::Value(static_cast<int64_t>(v)));
                    }
                    path_list.push_back(wxf::Value(vertices));
                }
                state_data.push_back({wxf::Value("Paths"), wxf::Value(path_list)});

                // Proper times
                auto pt_it = state_geodesic_proper_times.find(sid);
                if (pt_it != state_geodesic_proper_times.end()) {
                    wxf::ValueList pt_list;
                    for (const auto& times : pt_it->second) {
                        wxf::ValueList time_vals;
                        for (float t : times) {
                            time_vals.push_back(wxf::Value(static_cast<double>(t)));
                        }
                        pt_list.push_back(wxf::Value(time_vals));
                    }
                    state_data.push_back({wxf::Value("ProperTimes"), wxf::Value(pt_list)});
                }

                // Bundle spread
                auto spread_it = state_geodesic_bundle_spread.find(sid);
                if (spread_it != state_geodesic_bundle_spread.end()) {
                    state_data.push_back({wxf::Value("BundleSpread"),
                                         wxf::Value(static_cast<double>(spread_it->second))});
                }

                per_state.push_back({wxf::Value(static_cast<int64_t>(sid)), wxf::Value(state_data)});
            }
            geodesic_data.push_back({wxf::Value("PerState"), wxf::Value(per_state)});

            full_result.push_back({wxf::Value("GeodesicData"), wxf::Value(geodesic_data)});
        }

        // TopologicalData -> Association["PerState" -> {...}]
        // Each state: state_id -> {"Defects" -> [...], "Charges" -> <|vertex -> charge|>}
        if (detect_particles && !state_defects.empty()) {
            wxf::ValueAssociation topo_data;

            // Per-state defects
            wxf::ValueAssociation per_state;
            for (const auto& [sid, defects] : state_defects) {
                wxf::ValueAssociation state_data;

                // Defects list
                wxf::ValueList defect_list;
                for (const auto& defect : defects) {
                    wxf::ValueAssociation def_assoc;

                    // Type as string
                    const char* type_names[] = {"None", "K5", "K33", "HighDegree", "DimensionSpike", "Unknown"};
                    int type_idx = defect.type >= 0 && defect.type <= 5 ? defect.type : 5;
                    def_assoc.push_back({wxf::Value("Type"), wxf::Value(type_names[type_idx])});

                    // Core vertices
                    wxf::ValueList core_verts;
                    for (bh::VertexId v : defect.core_vertices) {
                        core_verts.push_back(wxf::Value(static_cast<int64_t>(v)));
                    }
                    def_assoc.push_back({wxf::Value("CoreVertices"), wxf::Value(core_verts)});

                    def_assoc.push_back({wxf::Value("Charge"), wxf::Value(static_cast<double>(defect.charge))});
                    def_assoc.push_back({wxf::Value("CentroidX"), wxf::Value(static_cast<double>(defect.centroid_x))});
                    def_assoc.push_back({wxf::Value("CentroidY"), wxf::Value(static_cast<double>(defect.centroid_y))});
                    def_assoc.push_back({wxf::Value("LocalDimension"), wxf::Value(static_cast<double>(defect.local_dimension))});
                    def_assoc.push_back({wxf::Value("Confidence"), wxf::Value(static_cast<int64_t>(defect.confidence))});

                    defect_list.push_back(wxf::Value(def_assoc));
                }
                state_data.push_back({wxf::Value("Defects"), wxf::Value(defect_list)});

                // Vertex charges (if computed)
                auto charge_it = state_charges.find(sid);
                if (charge_it != state_charges.end() && !charge_it->second.empty()) {
                    wxf::ValueAssociation charges_assoc;
                    for (const auto& [v, charge] : charge_it->second) {
                        charges_assoc.push_back({wxf::Value(static_cast<int64_t>(v)),
                                                wxf::Value(static_cast<double>(charge))});
                    }
                    state_data.push_back({wxf::Value("Charges"), wxf::Value(charges_assoc)});
                }

                per_state.push_back({wxf::Value(static_cast<int64_t>(sid)), wxf::Value(state_data)});
            }
            topo_data.push_back({wxf::Value("PerState"), wxf::Value(per_state)});

            full_result.push_back({wxf::Value("TopologicalData"), wxf::Value(topo_data)});
        }

        // CurvatureData -> Association["PerState" -> {...}]
        // Each state: state_id -> {"OllivierRicci" -> {...}, "DimensionGradient" -> {...}, "MeanCurvature" -> float}
        if (compute_curvature && (!state_ollivier_ricci.empty() || !state_dimension_gradient.empty())) {
            wxf::ValueAssociation curv_data;

            // Per-state curvatures
            wxf::ValueAssociation per_state;
            for (const auto& [sid, curv_map] : state_mean_curvature) {
                wxf::ValueAssociation state_data;

                // Ollivier-Ricci per-vertex
                auto or_it = state_ollivier_ricci.find(sid);
                if (or_it != state_ollivier_ricci.end() && !or_it->second.empty()) {
                    wxf::ValueAssociation or_assoc;
                    for (const auto& [v, curv] : or_it->second) {
                        or_assoc.push_back({wxf::Value(static_cast<int64_t>(v)),
                                           wxf::Value(static_cast<double>(curv))});
                    }
                    state_data.push_back({wxf::Value("OllivierRicci"), wxf::Value(or_assoc)});
                }

                // Dimension gradient per-vertex
                auto dg_it = state_dimension_gradient.find(sid);
                if (dg_it != state_dimension_gradient.end() && !dg_it->second.empty()) {
                    wxf::ValueAssociation dg_assoc;
                    for (const auto& [v, curv] : dg_it->second) {
                        dg_assoc.push_back({wxf::Value(static_cast<int64_t>(v)),
                                           wxf::Value(static_cast<double>(curv))});
                    }
                    state_data.push_back({wxf::Value("DimensionGradient"), wxf::Value(dg_assoc)});
                }

                // Mean curvature
                state_data.push_back({wxf::Value("MeanCurvature"),
                                     wxf::Value(static_cast<double>(curv_map))});

                per_state.push_back({wxf::Value(static_cast<int64_t>(sid)), wxf::Value(state_data)});
            }
            curv_data.push_back({wxf::Value("PerState"), wxf::Value(per_state)});

            full_result.push_back({wxf::Value("CurvatureData"), wxf::Value(curv_data)});
        }

        // EntropyData -> Association["PerState" -> {...}]
        // Each state: state_id -> {"DegreeEntropy" -> float, "GraphEntropy" -> float, ...}
        if (compute_entropy && !state_degree_entropy.empty()) {
            wxf::ValueAssociation ent_data;

            // Per-state entropy
            wxf::ValueAssociation per_state;
            for (const auto& [sid, deg_ent] : state_degree_entropy) {
                wxf::ValueAssociation state_data;

                state_data.push_back({wxf::Value("DegreeEntropy"),
                                     wxf::Value(static_cast<double>(deg_ent))});

                auto ge_it = state_graph_entropy.find(sid);
                if (ge_it != state_graph_entropy.end()) {
                    state_data.push_back({wxf::Value("GraphEntropy"),
                                         wxf::Value(static_cast<double>(ge_it->second))});
                }

                // Local entropy per-vertex
                auto le_it = state_local_entropy.find(sid);
                if (le_it != state_local_entropy.end() && !le_it->second.empty()) {
                    wxf::ValueAssociation le_assoc;
                    for (const auto& [v, ent] : le_it->second) {
                        le_assoc.push_back({wxf::Value(static_cast<int64_t>(v)),
                                           wxf::Value(static_cast<double>(ent))});
                    }
                    state_data.push_back({wxf::Value("LocalEntropy"), wxf::Value(le_assoc)});
                }

                // Mutual info per-vertex
                auto mi_it = state_mutual_info.find(sid);
                if (mi_it != state_mutual_info.end() && !mi_it->second.empty()) {
                    wxf::ValueAssociation mi_assoc;
                    for (const auto& [v, mi] : mi_it->second) {
                        mi_assoc.push_back({wxf::Value(static_cast<int64_t>(v)),
                                           wxf::Value(static_cast<double>(mi))});
                    }
                    state_data.push_back({wxf::Value("MutualInfo"), wxf::Value(mi_assoc)});
                }

                // Fisher info per-vertex
                auto fi_it = state_fisher_info.find(sid);
                if (fi_it != state_fisher_info.end() && !fi_it->second.empty()) {
                    wxf::ValueAssociation fi_assoc;
                    for (const auto& [v, fi] : fi_it->second) {
                        fi_assoc.push_back({wxf::Value(static_cast<int64_t>(v)),
                                           wxf::Value(static_cast<double>(fi))});
                    }
                    state_data.push_back({wxf::Value("FisherInfo"), wxf::Value(fi_assoc)});
                }

                per_state.push_back({wxf::Value(static_cast<int64_t>(sid)), wxf::Value(state_data)});
            }
            ent_data.push_back({wxf::Value("PerState"), wxf::Value(per_state)});

            full_result.push_back({wxf::Value("EntropyData"), wxf::Value(ent_data)});
        }

        // RotationData -> Association["PerState" -> {...}]
        // Each state: state_id -> {"Curve" -> [...], "PowerLawExponent" -> float, ...}
        if (compute_rotation_curve && !state_rotation_curves.empty()) {
            wxf::ValueAssociation rot_data;

            // Per-state rotation curves
            wxf::ValueAssociation per_state;
            for (const auto& [sid, rot_result] : state_rotation_curves) {
                wxf::ValueAssociation state_data;

                // Center vertex
                state_data.push_back({wxf::Value("Center"),
                                     wxf::Value(static_cast<int64_t>(rot_result.center))});

                // Power law fit parameters
                state_data.push_back({wxf::Value("PowerLawExponent"),
                                     wxf::Value(static_cast<double>(rot_result.power_law_exponent))});
                state_data.push_back({wxf::Value("FitResidual"),
                                     wxf::Value(static_cast<double>(rot_result.fit_residual))});

                // Flat rotation detection
                state_data.push_back({wxf::Value("HasFlatRotation"),
                                     wxf::Value(rot_result.has_flat_rotation)});
                state_data.push_back({wxf::Value("FlatRegionStart"),
                                     wxf::Value(static_cast<double>(rot_result.flat_region_start))});
                state_data.push_back({wxf::Value("FlatnessScore"),
                                     wxf::Value(static_cast<double>(rot_result.flatness_score))});

                // Curve data points
                wxf::ValueList curve_list;
                for (const auto& pt : rot_result.curve) {
                    wxf::ValueAssociation pt_assoc;
                    pt_assoc.push_back({wxf::Value("Radius"),
                                       wxf::Value(static_cast<int64_t>(pt.radius))});
                    pt_assoc.push_back({wxf::Value("OrbitalVelocity"),
                                       wxf::Value(static_cast<double>(pt.orbital_velocity))});
                    pt_assoc.push_back({wxf::Value("ExpectedVelocity"),
                                       wxf::Value(static_cast<double>(pt.expected_velocity))});
                    pt_assoc.push_back({wxf::Value("Deviation"),
                                       wxf::Value(static_cast<double>(pt.deviation))});
                    curve_list.push_back(wxf::Value(pt_assoc));
                }
                state_data.push_back({wxf::Value("Curve"), wxf::Value(curve_list)});

                per_state.push_back({wxf::Value(static_cast<int64_t>(sid)), wxf::Value(state_data)});
            }
            rot_data.push_back({wxf::Value("PerState"), wxf::Value(per_state)});

            full_result.push_back({wxf::Value("RotationData"), wxf::Value(rot_data)});
        }

        // HilbertSpaceData -> Association with inner products and vertex probabilities
        if (has_hilbert_data) {
            wxf::ValueAssociation hilbert_data;

            // Statistics
            hilbert_data.push_back({wxf::Value("NumStates"),
                                   wxf::Value(static_cast<int64_t>(hilbert_result.num_states))});
            hilbert_data.push_back({wxf::Value("NumVertices"),
                                   wxf::Value(static_cast<int64_t>(hilbert_result.num_vertices))});
            hilbert_data.push_back({wxf::Value("MeanInnerProduct"),
                                   wxf::Value(static_cast<double>(hilbert_result.mean_inner_product))});
            hilbert_data.push_back({wxf::Value("MaxInnerProduct"),
                                   wxf::Value(static_cast<double>(hilbert_result.max_inner_product))});
            hilbert_data.push_back({wxf::Value("MeanVertexProbability"),
                                   wxf::Value(static_cast<double>(hilbert_result.mean_vertex_probability))});
            hilbert_data.push_back({wxf::Value("VertexProbabilityEntropy"),
                                   wxf::Value(static_cast<double>(hilbert_result.vertex_probability_entropy))});

            // Vertex probabilities: vertex_id -> probability
            wxf::ValueAssociation vertex_probs;
            for (const auto& [vid, prob] : hilbert_result.vertex_probabilities) {
                vertex_probs.push_back({wxf::Value(static_cast<int64_t>(vid)),
                                       wxf::Value(static_cast<double>(prob))});
            }
            hilbert_data.push_back({wxf::Value("VertexProbabilities"), wxf::Value(vertex_probs)});

            // Inner product matrix (as list of lists for efficient transfer)
            wxf::ValueList ip_matrix;
            for (const auto& row : hilbert_result.inner_product_matrix) {
                wxf::ValueList row_list;
                for (float val : row) {
                    row_list.push_back(wxf::Value(static_cast<double>(val)));
                }
                ip_matrix.push_back(wxf::Value(row_list));
            }
            hilbert_data.push_back({wxf::Value("InnerProductMatrix"), wxf::Value(ip_matrix)});

            // State indices (for mapping matrix rows/columns to state IDs)
            wxf::ValueList state_ids;
            for (uint32_t sid : hilbert_result.state_indices) {
                state_ids.push_back(wxf::Value(static_cast<int64_t>(sid)));
            }
            hilbert_data.push_back({wxf::Value("StateIndices"), wxf::Value(state_ids)});

            full_result.push_back({wxf::Value("HilbertSpaceData"), wxf::Value(hilbert_data)});
        }

        // BranchialData -> Association with distribution sharpness and branch entropy
        if (has_branchial_data) {
            wxf::ValueAssociation branchial_data;

            // Statistics
            branchial_data.push_back(std::make_pair(wxf::Value("NumUniqueVertices"),
                                     wxf::Value(static_cast<int64_t>(branchial_result.num_unique_vertices))));
            branchial_data.push_back(std::make_pair(wxf::Value("MeanSharpness"),
                                     wxf::Value(static_cast<double>(branchial_result.mean_sharpness))));
            branchial_data.push_back(std::make_pair(wxf::Value("MeanBranchEntropy"),
                                     wxf::Value(static_cast<double>(branchial_result.mean_branch_entropy))));
            branchial_data.push_back(std::make_pair(wxf::Value("MaxBranchesPerVertex"),
                                     wxf::Value(static_cast<int64_t>(branchial_result.max_branches_per_vertex))));

            // Per-vertex sharpness
            wxf::ValueAssociation vertex_sharpness;
            for (const auto& [vid, sharpness] : branchial_result.vertex_sharpness) {
                vertex_sharpness.push_back(std::make_pair(wxf::Value(static_cast<int64_t>(vid)),
                                           wxf::Value(static_cast<double>(sharpness))));
            }
            branchial_data.push_back(std::make_pair(wxf::Value("VertexSharpness"), wxf::Value(vertex_sharpness)));

            // Delocalized vertices (vertices with sharpness < 1.0, appearing in multiple branches)
            wxf::ValueList delocalized;
            for (const auto& [vid, sharpness] : branchial_result.vertex_sharpness) {
                if (sharpness < 1.0f) {
                    delocalized.push_back(wxf::Value(static_cast<int64_t>(vid)));
                }
            }
            branchial_data.push_back(std::make_pair(wxf::Value("DelocalizedVertices"), wxf::Value(delocalized)));

            full_result.push_back(std::make_pair(wxf::Value("BranchialData"), wxf::Value(branchial_data)));
        }

        // MultispaceData -> Association with vertex/edge probabilities across branches
        if (has_multispace_data) {
            wxf::ValueAssociation multispace_data;

            // Statistics
            multispace_data.push_back(std::make_pair(wxf::Value("NumVertices"),
                                      wxf::Value(static_cast<int64_t>(multispace_vertex_probs.size()))));
            multispace_data.push_back(std::make_pair(wxf::Value("NumEdges"),
                                      wxf::Value(static_cast<int64_t>(multispace_edge_probs.size()))));
            multispace_data.push_back(std::make_pair(wxf::Value("MeanVertexProbability"),
                                      wxf::Value(static_cast<double>(multispace_mean_vertex_prob))));
            multispace_data.push_back(std::make_pair(wxf::Value("MeanEdgeProbability"),
                                      wxf::Value(static_cast<double>(multispace_mean_edge_prob))));
            multispace_data.push_back(std::make_pair(wxf::Value("TotalEntropy"),
                                      wxf::Value(static_cast<double>(multispace_total_entropy))));

            // Per-vertex probabilities
            wxf::ValueAssociation vertex_probs_assoc;
            for (const auto& [vid, prob] : multispace_vertex_probs) {
                vertex_probs_assoc.push_back(std::make_pair(wxf::Value(static_cast<int64_t>(vid)),
                                             wxf::Value(static_cast<double>(prob))));
            }
            multispace_data.push_back(std::make_pair(wxf::Value("VertexProbabilities"), wxf::Value(vertex_probs_assoc)));

            // Per-edge probabilities (as list of {{v1, v2}, prob})
            wxf::ValueList edge_probs_list;
            for (const auto& [e, prob] : multispace_edge_probs) {
                wxf::ValueList edge_entry;
                wxf::ValueList edge_pair;
                edge_pair.push_back(wxf::Value(static_cast<int64_t>(e.first)));
                edge_pair.push_back(wxf::Value(static_cast<int64_t>(e.second)));
                edge_entry.push_back(wxf::Value(edge_pair));
                edge_entry.push_back(wxf::Value(static_cast<double>(prob)));
                edge_probs_list.push_back(wxf::Value(edge_entry));
            }
            multispace_data.push_back(std::make_pair(wxf::Value("EdgeProbabilities"), wxf::Value(edge_probs_list)));

            full_result.push_back(std::make_pair(wxf::Value("MultispaceData"), wxf::Value(multispace_data)));
        }

        // Topology -> String (for metadata)
        if (!topology_type.empty() && topology_type != "Flat") {
            full_result.push_back({wxf::Value("Topology"), wxf::Value(topology_type)});
        }

        // Write final association
        wxf_writer.write(wxf::Value(full_result));
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
            print_to_frontend(libData, "HGEvolveV2: Serialization complete.");
        }
#endif

        MArgument_setMNumericArray(res, result_array);
        return LIBRARY_NO_ERROR;

    } catch (const wxf::TypeError& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "WXF TypeError in V2: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    } catch (const std::exception& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "Exception in V2: %.200s", e.what());
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

        // Compute dimensions for all vertices
        std::vector<float> dimensions = bh::estimate_all_dimensions(graph, config);

        // Compute stats
        bh::DimensionStats stats = bh::compute_dimension_stats(dimensions);

        // Build WXF output
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::ValueAssociation result;

        // PerVertex -> Association[vertex_id -> dimension]
        wxf::ValueAssociation per_vertex_assoc;
        const auto& vertices = graph.vertices();
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (dimensions[i] > 0) {
                per_vertex_assoc.push_back({wxf::Value(static_cast<int64_t>(vertices[i])),
                                            wxf::Value(static_cast<double>(dimensions[i]))});
            }
        }
        result.push_back({wxf::Value("PerVertex"), wxf::Value(per_vertex_assoc)});

        // GeodesicCoords -> Association[vertex_id -> {d1, d2, ...}]
        wxf::ValueAssociation coords_assoc;
        for (const auto& [vid, coords] : geodesic_coords) {
            wxf::ValueList coord_list;
            for (int d : coords) {
                coord_list.push_back(wxf::Value(static_cast<int64_t>(d)));
            }
            coords_assoc.push_back({wxf::Value(static_cast<int64_t>(vid)), wxf::Value(coord_list)});
        }
        result.push_back({wxf::Value("GeodesicCoords"), wxf::Value(coords_assoc)});

        // Anchors -> {anchor1, anchor2, ...}
        wxf::ValueList anchors_list;
        for (bh::VertexId a : anchors) {
            anchors_list.push_back(wxf::Value(static_cast<int64_t>(a)));
        }
        result.push_back({wxf::Value("Anchors"), wxf::Value(anchors_list)});

        // Stats -> Association
        wxf::ValueAssociation stats_assoc;
        stats_assoc.push_back({wxf::Value("Mean"), wxf::Value(static_cast<double>(stats.mean))});
        stats_assoc.push_back({wxf::Value("Min"), wxf::Value(static_cast<double>(stats.min))});
        stats_assoc.push_back({wxf::Value("Max"), wxf::Value(static_cast<double>(stats.max))});
        stats_assoc.push_back({wxf::Value("Variance"), wxf::Value(static_cast<double>(stats.variance))});
        stats_assoc.push_back({wxf::Value("StdDev"), wxf::Value(static_cast<double>(stats.stddev))});
        stats_assoc.push_back({wxf::Value("Count"), wxf::Value(static_cast<int64_t>(stats.count))});
        result.push_back({wxf::Value("Stats"), wxf::Value(stats_assoc)});

        // Config -> Association (echo back the config used)
        wxf::ValueAssociation config_assoc;
        config_assoc.push_back({wxf::Value("Formula"),
            wxf::Value(formula == bh::DimensionFormula::LinearRegression ? "LinearRegression" : "DiscreteDerivative")});
        config_assoc.push_back({wxf::Value("SaturationThreshold"), wxf::Value(static_cast<double>(saturation_threshold))});
        config_assoc.push_back({wxf::Value("MinRadius"), wxf::Value(static_cast<int64_t>(min_radius))});
        config_assoc.push_back({wxf::Value("MaxRadius"), wxf::Value(static_cast<int64_t>(max_radius))});
        config_assoc.push_back({wxf::Value("NumAnchors"), wxf::Value(static_cast<int64_t>(num_anchors))});
        config_assoc.push_back({wxf::Value("AnchorSeparation"), wxf::Value(static_cast<int64_t>(anchor_separation))});
        result.push_back({wxf::Value("Config"), wxf::Value(config_assoc)});

        // Write output
        wxf_writer.write(wxf::Value(result));
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

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData /* libData */) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData /* libData */) {
}