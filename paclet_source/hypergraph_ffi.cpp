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

// Include basic hypergraph headers
#include "hypergraph/wolfram_evolution.hpp"
#include "hypergraph/rewriting.hpp"
#include "hypergraph/wolfram_states.hpp"

// Include unified engine headers (V2)
#include "hypergraph/unified/unified_hypergraph.hpp"
#include "hypergraph/unified/parallel_evolution.hpp"
#include "hypergraph/unified/pattern.hpp"
#include "hypergraph/debug_log.hpp"

// Include comprehensive WXF library
#include "wxf.hpp"

// Include blackhole analysis (without layout dependency)
#include "blackhole/hausdorff_analysis.hpp"
#include "blackhole/bh_types.hpp"

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
                auto lhs = args_parser.read<std::vector<std::vector<int64_t>>>();
                auto rhs = args_parser.read<std::vector<std::vector<int64_t>>>();

                rule_parts = {lhs, rhs};
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
 * Perform multiway rewriting evolution using WXF input/output
 * Input: WXF binary data as 1D byte tensor containing:
 *   Association[
 *     "InitialEdges" -> {{vertices...}, ...},
 *     "Rules" -> <"Rule1" -> {{lhs edges}, {rhs edges}}, ...>,
 *     "Steps" -> integer
 *   ]
 *
 * Output: WXF Association with States, Events, CausalEdges, BranchialEdges
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

        // Parse WXF input using comprehensive wxf library
        wxf::Parser parser(wxf_bytes);
        parser.skip_header();

        std::vector<std::vector<std::vector<GlobalVertexId>>> initial_states;
        std::vector<RewritingRule> parsed_rules;
        int steps = 1;

        // Default option values
        bool canonicalize_states = true;
        bool canonicalize_events = true;
        bool deduplicate_events = false;  // Simple graph mode - return existing event for duplicates
        bool causal_transitive_reduction = true;
        bool early_termination = false;
        bool full_capture = true;  // Enable to store states for visualization
        bool full_capture_non_canonicalised = false;  // Store all states, not just canonical ones
        size_t max_successor_states_per_parent = 0;  // 0 = unlimited
        size_t max_states_per_step = 0;  // 0 = unlimited
        double exploration_probability = 1.0;  // 1.0 = always explore

        // Parse main association using comprehensive wxf library
        parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
            if (key == "InitialStates") {
                // New format: list of states (3D array)
                auto states_data = value_parser.read<std::vector<std::vector<std::vector<int64_t>>>>();
                for (const auto& state_edges : states_data) {
                    std::vector<std::vector<GlobalVertexId>> state;
                    for (const auto& edge : state_edges) {
                        std::vector<GlobalVertexId> edge_vertices;
                        for (int64_t v : edge) {
                            if (v >= 0) {
                                edge_vertices.push_back(static_cast<GlobalVertexId>(v));
                            }
                        }
                        if (!edge_vertices.empty()) {
                            state.push_back(edge_vertices);
                        }
                    }
                    if (!state.empty()) {
                        initial_states.push_back(state);
                    }
                }
            }
            else if (key == "InitialEdges") {
                // Legacy format: single state (2D array) - wrap as single state in states list
                auto edges_data = value_parser.read<std::vector<std::vector<int64_t>>>();
                std::vector<std::vector<GlobalVertexId>> single_state;
                for (const auto& edge : edges_data) {
                    std::vector<GlobalVertexId> edge_vertices;
                    for (int64_t v : edge) {
                        if (v >= 0) {
                            edge_vertices.push_back(static_cast<GlobalVertexId>(v));
                        }
                    }
                    if (!edge_vertices.empty()) {
                        single_state.push_back(edge_vertices);
                    }
                }
                if (!single_state.empty()) {
                    initial_states.push_back(single_state);
                }
            }
            else if (key == "Rules") {
                auto rules = ffi_helpers::read_rules_association(value_parser);
                for (const auto& [rule_name, rule_data] : rules) {
                    if (rule_data.size() != 2) continue; // Should be {lhs, rhs}

                    PatternHypergraph lhs, rhs;

                    // Parse LHS
                    for (const auto& edge : rule_data[0]) {
                        std::vector<PatternVertex> edge_vertices;
                        for (int64_t v : edge) {
                            if (v >= 0) {
                                edge_vertices.push_back(PatternVertex::variable(static_cast<VertexId>(v)));
                            }
                        }
                        if (!edge_vertices.empty()) {
                            lhs.add_edge(edge_vertices);
                        }
                    }

                    // Parse RHS
                    for (const auto& edge : rule_data[1]) {
                        std::vector<PatternVertex> edge_vertices;
                        for (int64_t v : edge) {
                            if (v >= 0) {
                                edge_vertices.push_back(PatternVertex::variable(static_cast<VertexId>(v)));
                            }
                        }
                        if (!edge_vertices.empty()) {
                            rhs.add_edge(edge_vertices);
                        }
                    }

                    if (lhs.num_edges() > 0 && rhs.num_edges() > 0) {
                        parsed_rules.emplace_back(lhs, rhs);
                    }
                }
            }
            else if (key == "Steps") {
                steps = value_parser.read<int>();
            }
            else if (key == "Options") {
                // Parse options association using wxf library
                value_parser.read_association([&](const std::string& option_key, wxf::Parser& option_parser) {
                    // CRITICAL: Must ALWAYS consume the value, even if we don't recognize the option
                    // Otherwise parser position gets misaligned!
                    try {
                        // Check if this is a numeric option first
                        if (option_key == "MaxSuccessorStatesPerParent") {
                            max_successor_states_per_parent = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "MaxStatesPerStep") {
                            max_states_per_step = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "ExplorationProbability") {
                            exploration_probability = option_parser.read<double>();
                        } else {
                            // Try to read as string (symbol) - this handles both String and Symbol tokens now
                            std::string symbol = option_parser.read<std::string>();
                            bool value = (symbol == "True");

                            if (option_key == "CanonicalizeStates") {
                                canonicalize_states = value;
                            } else if (option_key == "CanonicalizeEvents") {
                                canonicalize_events = value;
                            } else if (option_key == "DeduplicateEvents") {
                                deduplicate_events = value;
                            } else if (option_key == "CausalTransitiveReduction") {
                                causal_transitive_reduction = value;
                            } else if (option_key == "EarlyTermination") {
                                early_termination = value;
                            } else if (option_key == "FullCapture") {
                                full_capture = value;
                            }
                            // If we successfully read but don't recognize the option, that's fine
                            // The value was consumed
                        }
                    } catch (...) {
                        // If reading as string/number failed, skip whatever the value is
                        // This handles options like "AspectRatio" -> Automatic (not a string/symbol we can read)
                        option_parser.skip_value();
                    }
                });
            }
            else {
                // Unknown key - must skip its value to maintain parser position
                value_parser.skip_value();
            }
        });

        if (initial_states.empty()) {
            handle_error(libData, "No initial states provided");
            return LIBRARY_FUNCTION_ERROR;
        }

        if (parsed_rules.empty()) {
            handle_error(libData, "No valid rules found");
            return LIBRARY_FUNCTION_ERROR;
        }

        // When canonicalization is disabled, we need to store all states (not just canonical ones)
        if (!canonicalize_states) {
            full_capture_non_canonicalised = true;
        }


        // Run evolution with parsed options
        // TODO: Automatic mode event canonicalization (full_event_canonicalization=false) is temporarily disabled
        WolframEvolution evolution(static_cast<std::size_t>(steps), std::thread::hardware_concurrency(),
                                   canonicalize_states, full_capture, canonicalize_events, deduplicate_events,
                                   causal_transitive_reduction, early_termination, full_capture_non_canonicalised,
                                   max_successor_states_per_parent, max_states_per_step, exploration_probability,
                                   /*full_event_canonicalization=*/true);

        for (const auto& rule : parsed_rules) {
            evolution.add_rule(rule);
        }

        evolution.evolve(initial_states);

        // Get results from multiway graph
        const auto& multiway_graph = evolution.get_multiway_graph();
        auto all_states = multiway_graph.get_all_states();
        auto all_events = multiway_graph.get_all_events();
        auto event_edges = multiway_graph.get_event_edges();


        // Create WXF output using comprehensive wxf library
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        // Build main association

        // Get initial state IDs to mark them in the output
        const auto& initial_state_ids = multiway_graph.get_initial_state_ids();
        std::unordered_set<std::size_t> initial_ids_set;
        for (const auto& id : initial_state_ids) {
            initial_ids_set.insert(id.value);
        }

        // States -> Association[state_id -> state_data]
        // Send ALL states by their raw ID (not canonical) so events can reference them correctly
        std::unordered_map<int64_t, std::pair<std::vector<std::vector<int64_t>>, bool>> states_map;
        for (const auto& state : all_states) {
            StateID state_id = state->id();  // Use raw ID, not canonical

            std::vector<std::vector<int64_t>> state_edges;
            for (const auto& edge : state->edges()) {
                std::vector<int64_t> edge_data = {static_cast<int64_t>(edge.global_id)};
                for (const auto& vertex : edge.global_vertices) {
                    edge_data.push_back(static_cast<int64_t>(vertex));
                }
                state_edges.push_back(edge_data);
            }

            bool is_initial = initial_ids_set.find(state_id.value) != initial_ids_set.end();
            states_map[static_cast<int64_t>(state_id.value)] = {state_edges, is_initial};
        }

        // Build full result using Value type for heterogeneous association
        wxf::ValueAssociation full_result;

        // States -> Association[state_id -> Association["Edges" -> edges, "IsInitialState" -> bool]]
        wxf::ValueAssociation states_assoc;
        for (const auto& [state_id, state_data] : states_map) {
            const auto& [edges, is_initial] = state_data;

            wxf::ValueList edge_list;
            for (const auto& edge : edges) {
                wxf::ValueList edge_data;
                for (int64_t v : edge) {
                    edge_data.push_back(wxf::Value(v));
                }
                edge_list.push_back(wxf::Value(edge_data));
            }

            // Create state association with edges and IsInitialState flag
            wxf::ValueAssociation state_assoc;
            state_assoc.push_back({wxf::Value("Edges"), wxf::Value(edge_list)});
            state_assoc.push_back({wxf::Value("IsInitialState"), wxf::Value(is_initial ? "True" : "False")});

            states_assoc.push_back({wxf::Value(state_id), wxf::Value(state_assoc)});
        }
        full_result.push_back({wxf::Value("States"), wxf::Value(states_assoc)});

        // Events -> Association of canonical events with multiplicity
        // Build event lookup and count multiplicity in one pass
        std::unordered_map<std::size_t, int> event_multiplicity;
        std::unordered_map<std::size_t, const WolframEvent*> event_lookup;

        for (const auto& event : all_events) {
            event_lookup[event.event_id] = &event;

            if (event.canonical_event_id.has_value()) {
                // This is a duplicate - increment canonical event's multiplicity
                event_multiplicity[event.canonical_event_id.value()]++;
            } else {
                // This is a canonical event - initialize to 1 if not seen
                if (event_multiplicity.find(event.event_id) == event_multiplicity.end()) {
                    event_multiplicity[event.event_id] = 1;
                }
            }
        }

        wxf::ValueAssociation events_assoc;
        for (const auto& event : all_events) {
            // Skip duplicate events - only send canonical representatives
            if (event.canonical_event_id.has_value()) {
                continue;
            }

            wxf::ValueAssociation event_data;
            event_data.push_back({wxf::Value("EventId"), wxf::Value(static_cast<int64_t>(event.event_id))});
            event_data.push_back({wxf::Value("RuleIndex"), wxf::Value(static_cast<int64_t>(event.rule_index))});
            event_data.push_back({wxf::Value("Multiplicity"), wxf::Value(static_cast<int64_t>(event_multiplicity[event.event_id]))});

            // Send RAW state IDs for consumed/produced edge matching
            event_data.push_back({wxf::Value("InputStateId"), wxf::Value(static_cast<int64_t>(event.input_state_id.value))});
            event_data.push_back({wxf::Value("OutputStateId"), wxf::Value(static_cast<int64_t>(event.output_state_id.value))});

            // Also send canonical IDs for graph structure (linking isomorphic states)
            if (event.has_canonical_input_state_id()) {
                event_data.push_back({wxf::Value("CanonicalInputStateId"), wxf::Value(static_cast<int64_t>(event.canonical_input_state_id.value))});
            }
            if (event.has_canonical_output_state_id()) {
                event_data.push_back({wxf::Value("CanonicalOutputStateId"), wxf::Value(static_cast<int64_t>(event.canonical_output_state_id.value))});
            }

            // Add consumed/produced edges for graph highlighting
            wxf::ValueList consumed_list, produced_list;
            for (auto edge_id : event.consumed_edges) {
                consumed_list.push_back(wxf::Value(static_cast<int64_t>(edge_id)));
            }
            for (auto edge_id : event.produced_edges) {
                produced_list.push_back(wxf::Value(static_cast<int64_t>(edge_id)));
            }
            event_data.push_back({wxf::Value("ConsumedEdges"), wxf::Value(consumed_list)});
            event_data.push_back({wxf::Value("ProducedEdges"), wxf::Value(produced_list)});

            events_assoc.push_back({wxf::Value(static_cast<int64_t>(event.event_id)), wxf::Value(event_data)});
        }
        full_result.push_back({wxf::Value("Events"), wxf::Value(events_assoc)});

        // CausalEdges and BranchialEdges with multiplicity
        // Remap to canonical IDs and count multiplicity
        std::map<std::pair<std::size_t, std::size_t>, int> causal_multiplicity;
        std::map<std::pair<std::size_t, std::size_t>, int> branchial_multiplicity;

        for (const auto& edge : event_edges) {
            // Remap from_event to canonical
            std::size_t from_canonical = edge.from_event;
            auto from_it = event_lookup.find(edge.from_event);
            if (from_it != event_lookup.end() && from_it->second->canonical_event_id.has_value()) {
                from_canonical = from_it->second->canonical_event_id.value();
            }

            // Remap to_event to canonical
            std::size_t to_canonical = edge.to_event;
            auto to_it = event_lookup.find(edge.to_event);
            if (to_it != event_lookup.end() && to_it->second->canonical_event_id.has_value()) {
                to_canonical = to_it->second->canonical_event_id.value();
            }

            if (edge.type == EventRelationType::CAUSAL) {
                causal_multiplicity[{from_canonical, to_canonical}]++;
            } else if (edge.type == EventRelationType::BRANCHIAL) {
                branchial_multiplicity[{from_canonical, to_canonical}]++;
            }
        }

        // Serialize causal edges with multiplicity
        wxf::ValueList causal_edges;
        for (const auto& [edge_pair, mult] : causal_multiplicity) {
            wxf::ValueAssociation edge_data;
            edge_data.push_back({wxf::Value("From"), wxf::Value(static_cast<int64_t>(edge_pair.first))});
            edge_data.push_back({wxf::Value("To"), wxf::Value(static_cast<int64_t>(edge_pair.second))});
            edge_data.push_back({wxf::Value("Multiplicity"), wxf::Value(static_cast<int64_t>(mult))});
            causal_edges.push_back(wxf::Value(edge_data));
        }

        // Serialize branchial edges with multiplicity
        wxf::ValueList branchial_edges;
        for (const auto& [edge_pair, mult] : branchial_multiplicity) {
            wxf::ValueAssociation edge_data;
            edge_data.push_back({wxf::Value("From"), wxf::Value(static_cast<int64_t>(edge_pair.first))});
            edge_data.push_back({wxf::Value("To"), wxf::Value(static_cast<int64_t>(edge_pair.second))});
            edge_data.push_back({wxf::Value("Multiplicity"), wxf::Value(static_cast<int64_t>(mult))});
            branchial_edges.push_back(wxf::Value(edge_data));
        }

        full_result.push_back({wxf::Value("CausalEdges"), wxf::Value(causal_edges)});
        full_result.push_back({wxf::Value("BranchialEdges"), wxf::Value(branchial_edges)});

        // Counts
        full_result.push_back({wxf::Value("NumStates"), wxf::Value(static_cast<int64_t>(multiway_graph.num_states()))});
        full_result.push_back({wxf::Value("NumEvents"), wxf::Value(static_cast<int64_t>(multiway_graph.num_events()))});
        full_result.push_back({wxf::Value("NumCausalEdges"), wxf::Value(static_cast<int64_t>(multiway_graph.get_causal_edge_count()))});
        full_result.push_back({wxf::Value("NumBranchialEdges"), wxf::Value(static_cast<int64_t>(multiway_graph.get_branchial_edge_count()))});

        // Write final association using Value
        wxf_writer.write(wxf::Value(full_result));
        const auto& wxf_data = wxf_writer.data();

        // Create output ByteArray
        mint wxf_dims[1] = {static_cast<mint>(wxf_data.size())};
        MNumericArray result_array;
        int err = libData->numericarrayLibraryFunctions->MNumericArray_new(MNumericArray_Type_UBit8, 1, wxf_dims, &result_array);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }

        // Copy byte data directly
        void* result_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(result_array);
        uint8_t* byte_result_data = static_cast<uint8_t*>(result_data);
        std::memcpy(byte_result_data, wxf_data.data(), wxf_data.size());

        MArgument_setMNumericArray(res, result_array);
        return LIBRARY_NO_ERROR;

    } catch (const wxf::TypeError& e) {
        char err1[200], err2[200];
        std::string msg = e.what();
        snprintf(err1, sizeof(err1), "WXF TypeError (1/2): %s", msg.substr(0, 150).c_str());
        snprintf(err2, sizeof(err2), "WXF TypeError (2/2): %s", msg.length() > 150 ? msg.substr(150).c_str() : "");
        handle_error(libData, err1);
        if (msg.length() > 150) {
            handle_error(libData, err2);
        }
        return LIBRARY_FUNCTION_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
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
        unified::StateCanonicalizationMode state_canon_mode = unified::StateCanonicalizationMode::None;  // Default: tree mode
        unified::EventSignatureKeys event_signature_keys = unified::EVENT_SIG_NONE;  // Default: no event canonicalization
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
                            }
                        } else if (option_key == "CanonicalizeEvents") {
                            // Can be: None, Full, Automatic (symbols), or {"InputState", "OutputState", ...} (list)
                            try {
                                // Try to read as list first
                                auto keys = option_parser.read<std::vector<std::string>>();
                                event_signature_keys = unified::EVENT_SIG_NONE;
                                for (const auto& key : keys) {
                                    if (key == "InputState") event_signature_keys |= unified::EventKey_InputState;
                                    else if (key == "OutputState") event_signature_keys |= unified::EventKey_OutputState;
                                    else if (key == "Step") event_signature_keys |= unified::EventKey_Step;
                                    else if (key == "Rule") event_signature_keys |= unified::EventKey_Rule;
                                    else if (key == "ConsumedEdges") event_signature_keys |= unified::EventKey_ConsumedEdges;
                                    else if (key == "ProducedEdges") event_signature_keys |= unified::EventKey_ProducedEdges;
                                }
                            } catch (...) {
                                // Read as symbol
                                std::string symbol = option_parser.read<std::string>();
                                if (symbol == "None") {
                                    event_signature_keys = unified::EVENT_SIG_NONE;
                                } else if (symbol == "Full") {
                                    event_signature_keys = unified::EVENT_SIG_FULL;
                                } else if (symbol == "Automatic") {
                                    event_signature_keys = unified::EVENT_SIG_AUTOMATIC;
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
                                state_canon_mode = unified::StateCanonicalizationMode::None;
                                canonicalize_states_mode = "None";
                            } else if (symbol == "Automatic") {
                                // Automatic behaves like None for evolution (no deduplication)
                                // ContentStateId is computed separately for display-time grouping
                                state_canon_mode = unified::StateCanonicalizationMode::None;
                                canonicalize_states_mode = "Automatic";
                            } else if (symbol == "Full" || symbol == "True") {
                                state_canon_mode = unified::StateCanonicalizationMode::Full;
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
                            }
                        }
                    } catch (...) {
                        option_parser.skip_value();
                    }
                });
            }
            else {
                value_parser.skip_value();
            }
        });

        if (initial_states_raw.empty()) {
            handle_error(libData, "No initial states provided");
            return LIBRARY_FUNCTION_ERROR;
        }

        if (parsed_rules_raw.empty()) {
            handle_error(libData, "No valid rules found");
            return LIBRARY_FUNCTION_ERROR;
        }

        // Create unified hypergraph
        unified::UnifiedHypergraph hg;

        // Set hash strategy
        if (hash_strategy == "WL") {
            hg.set_hash_strategy(unified::HashStrategy::WL);
        } else if (hash_strategy == "UT") {
            hg.set_hash_strategy(unified::HashStrategy::UniquenessTree);
        } else {
            hg.set_hash_strategy(unified::HashStrategy::IncrementalUniquenessTree);
        }

        // Configure event canonicalization
        hg.set_event_signature_keys(event_signature_keys);

        // Configure state canonicalization mode
        hg.set_state_canonicalization_mode(state_canon_mode);

        // Create parallel evolution engine
        unified::ParallelEvolutionEngine engine(&hg, std::thread::hardware_concurrency());

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

            unified::RewriteRule rule;
            rule.index = rule_index++;

            // Track max variable seen for variable counting
            uint8_t max_lhs_var = 0;
            uint8_t max_rhs_var = 0;

            // Parse LHS edges
            rule.num_lhs_edges = 0;
            for (const auto& edge : rule_data[0]) {
                if (rule.num_lhs_edges >= unified::MAX_PATTERN_EDGES) break;
                unified::PatternEdge& pe = rule.lhs[rule.num_lhs_edges];
                pe.arity = 0;
                for (int64_t v : edge) {
                    if (v >= 0 && pe.arity < unified::MAX_ARITY) {
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
                if (rule.num_rhs_edges >= unified::MAX_PATTERN_EDGES) break;
                unified::PatternEdge& pe = rule.rhs[rule.num_rhs_edges];
                pe.arity = 0;
                for (int64_t v : edge) {
                    if (v >= 0 && pe.arity < unified::MAX_ARITY) {
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
        std::vector<std::vector<std::vector<unified::VertexId>>> initial_states;

        for (const auto& state_raw : initial_states_raw) {
            // Create a per-state vertex mapping: input_vertex -> canonical_vertex
            // Always start from 0 for canonical form
            std::unordered_map<int64_t, unified::VertexId> vertex_map;
            unified::VertexId next_vertex = 0;

            std::vector<std::vector<unified::VertexId>> state_edges;
            for (const auto& edge : state_raw) {
                std::vector<unified::VertexId> edge_vertices;
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
                const unified::State& state = hg.get_state(sid);
                if (state.id == unified::INVALID_ID) continue;

                // Build edges for SimpleGraph (only binary edges)
                std::vector<bh::Edge> edges;
                state.edges.for_each([&](unified::EdgeId eid) {
                    const unified::Edge& edge = hg.get_edge(eid);
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
            std::unordered_map<uint64_t, unified::StateId> content_hash_to_id;
            std::vector<uint64_t> state_content_hashes(num_states, 0);
            content_hash_to_id.reserve(num_states);

            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const unified::State& state = hg.get_state(sid);
                if (state.id == unified::INVALID_ID) continue;

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
                const unified::State& state = hg.get_state(sid);
                if (state.id == unified::INVALID_ID) continue;

                // Get canonical state ID (isomorphism-based)
                unified::StateId canonical_id = hg.get_canonical_state(sid);

                // Get content state ID (content-based) - from cached hash
                unified::StateId content_id = content_hash_to_id[state_content_hashes[sid]];

                // Build edge list: each edge is {edge_id, v1, v2, ...}
                wxf::ValueList edge_list;
                state.edges.for_each([&](unified::EdgeId eid) {
                    const unified::Edge& edge = hg.get_edge(eid);
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
                const unified::Event& event = hg.get_event(eid);
                if (event.id == unified::INVALID_ID) continue;
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
                const unified::Event& event = hg.get_event(eid);
                if (event.id == unified::INVALID_ID) continue;
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
            auto pair_hash = [](const std::pair<unified::EventId, unified::EventId>& p) {
                return std::hash<uint64_t>{}((static_cast<uint64_t>(p.first) << 32) | p.second);
            };
            std::unordered_set<std::pair<unified::EventId, unified::EventId>, decltype(pair_hash)> seen_raw_pairs(
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
                unified::EventId canonical_from = hg.get_canonical_event(edge.producer);
                unified::EventId canonical_to = hg.get_canonical_event(edge.consumer);

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
                unified::EventId canonical_from = hg.get_canonical_event(edge.event1);
                unified::EventId canonical_to = hg.get_canonical_event(edge.event2);

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
            std::set<unified::StateId> unique_states;
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
                const unified::Event& event1 = hg.get_event(edge.event1);
                const unified::Event& event2 = hg.get_event(edge.event2);

                // Filter by step if specified (branchial edges are between events at the same step)
                if (filter_by_step) {
                    const unified::State& output_state = hg.get_state(event1.output_state);
                    if (output_state.step != target_step) {
                        continue;
                    }
                }

                unified::StateId state1 = hg.get_canonical_state(event1.output_state);
                unified::StateId state2 = hg.get_canonical_state(event2.output_state);

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
            for (unified::StateId sid : unique_states) {
                state_vertices.push_back(wxf::Value(static_cast<int64_t>(sid)));
            }
            full_result.push_back({wxf::Value("BranchialStateVertices"), wxf::Value(state_vertices)});
        }

        // BranchialStateEdgesAllSiblings: ALL pairs of output states from same input state
        // This matches reference BranchialGraph behavior (no overlap check, all siblings)
        if (include_branchial_state_edges_all_siblings) {
            wxf::ValueList branchial_state_edges;
            std::set<unified::StateId> unique_states;

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
            hg.causal_graph().for_each_state_events([&]([[maybe_unused]] unified::StateId input_state, auto* event_list) {
                // Collect all events from this input state
                std::vector<unified::EventId> events;
                event_list->for_each([&](unified::EventId eid) {
                    // Skip genesis events if not showing them
                    if (!show_genesis_events && hg.is_genesis_event(eid)) {
                        return;
                    }
                    events.push_back(eid);
                });

                // Create all pairs of output states (C(n,2) pairs)
                for (size_t i = 0; i < events.size(); ++i) {
                    const unified::Event& event1 = hg.get_event(events[i]);

                    // Filter by step if specified
                    if (filter_by_step) {
                        const unified::State& output_state = hg.get_state(event1.output_state);
                        if (output_state.step != target_step) {
                            continue;
                        }
                    }

                    unified::StateId state1 = hg.get_canonical_state(event1.output_state);

                    for (size_t j = i + 1; j < events.size(); ++j) {
                        const unified::Event& event2 = hg.get_event(events[j]);

                        // Filter event2 by step too
                        if (filter_by_step) {
                            const unified::State& output_state2 = hg.get_state(event2.output_state);
                            if (output_state2.step != target_step) {
                                continue;
                            }
                        }

                        unified::StateId state2 = hg.get_canonical_state(event2.output_state);

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
            for (unified::StateId sid : unique_states) {
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
            std::unordered_map<uint64_t, unified::StateId> gd_content_hash_to_id;
            std::vector<uint64_t> gd_state_content_hashes(num_states, 0);

            if (canonicalize_states_mode == "Automatic") {
                gd_content_hash_to_id.reserve(num_states);
                for (uint32_t sid = 0; sid < num_states; ++sid) {
                    const unified::State& state = hg.get_state(sid);
                    if (state.id == unified::INVALID_ID) continue;
                    uint64_t hash = hg.get_state_content_hash(sid);
                    gd_state_content_hashes[sid] = hash;
                    if (gd_content_hash_to_id.find(hash) == gd_content_hash_to_id.end()) {
                        gd_content_hash_to_id[hash] = sid;
                    }
                }
            }

            // Helper: Get effective state ID based on canonicalization mode
            auto get_effective_state_id = [&](unified::StateId sid) -> int64_t {
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
            auto get_effective_event_id = [&](unified::EventId eid) -> int64_t {
                if (canonicalize_states_mode == "None" || event_signature_keys == unified::EVENT_SIG_NONE)
                    return static_cast<int64_t>(eid);
                const unified::Event& e = hg.get_event(eid);
                return e.is_canonical() ? static_cast<int64_t>(eid)
                                        : static_cast<int64_t>(e.canonical_event_id);
            };

            // Helper: Serialize state edges as list of {edgeId, v1, v2, ...}
            auto serialize_state_edges = [&](unified::StateId sid) -> wxf::ValueList {
                wxf::ValueList edge_list;
                hg.get_state(sid).edges.for_each([&](unified::EdgeId eid) {
                    const unified::Edge& edge = hg.get_edge(eid);
                    wxf::ValueList e;
                    e.push_back(wxf::Value(static_cast<int64_t>(eid)));
                    for (uint8_t i = 0; i < edge.arity; ++i)
                        e.push_back(wxf::Value(static_cast<int64_t>(edge.vertices[i])));
                    edge_list.push_back(wxf::Value(e));
                });
                return edge_list;
            };

            // Helper: Serialize state data for tooltips
            auto serialize_state_data = [&](unified::StateId sid) -> wxf::ValueAssociation {
                const unified::State& state = hg.get_state(sid);
                wxf::ValueAssociation d;
                d.push_back({wxf::Value("Id"), wxf::Value(static_cast<int64_t>(sid))});
                d.push_back({wxf::Value("CanonicalId"), wxf::Value(static_cast<int64_t>(hg.get_canonical_state(sid)))});
                d.push_back({wxf::Value("Step"), wxf::Value(static_cast<int64_t>(state.step))});
                d.push_back({wxf::Value("Edges"), wxf::Value(serialize_state_edges(sid))});
                d.push_back({wxf::Value("IsInitial"), wxf::Value(state.step == 0)});
                return d;
            };

            // Helper: Serialize event data for tooltips
            auto serialize_event_data = [&](unified::EventId eid) -> wxf::ValueAssociation {
                const unified::Event& e = hg.get_event(eid);
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
            auto is_valid_event = [&](unified::EventId eid) -> bool {
                const unified::Event& e = hg.get_event(eid);
                if (e.id == unified::INVALID_ID) return false;
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
                std::map<std::pair<unified::EventId, unified::EventId>, size_t> pair_counts;
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
                std::map<int64_t, unified::StateId> state_verts;
                for (uint32_t sid = 0; sid < hg.num_states(); ++sid) {
                    if (hg.get_state(sid).id == unified::INVALID_ID) continue;
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
                    const unified::Event& e = hg.get_event(eid);
                    add_graph_edge(wxf::Value(get_effective_state_id(e.input_state)),
                                   wxf::Value(get_effective_state_id(e.output_state)),
                                   "Directed", serialize_event_data(eid));
                }
            }
            else if (is_causal) {
                // === CAUSAL GRAPH ===
                // Vertices: events (deduplicated, tagged)
                std::map<int64_t, unified::EventId> event_verts;
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
                std::set<unified::StateId> state_set;
                std::map<int64_t, unified::StateId> state_verts;
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
                    const unified::Event& event1 = hg.get_event(edge.event1);
                    const unified::Event& event2 = hg.get_event(edge.event2);

                    // Filter by step if specified
                    if (filter_by_step) {
                        const unified::State& output_state = hg.get_state(event1.output_state);
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
                    const unified::Event& event1 = hg.get_event(edge.event1);
                    const unified::Event& event2 = hg.get_event(edge.event2);

                    // Filter by step if specified
                    if (filter_by_step) {
                        const unified::State& output_state = hg.get_state(event1.output_state);
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
                std::map<int64_t, unified::StateId> raw_states;
                std::map<int64_t, unified::EventId> event_verts;

                for (uint32_t eid = 0; eid < hg.num_raw_events(); ++eid) {
                    if (!is_valid_event(eid)) continue;
                    const unified::Event& e = hg.get_event(eid);
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
                    const unified::Event& e = hg.get_event(eid);
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
                            const unified::Event& event1 = hg.get_event(be.event1);
                            const unified::State& output_state = hg.get_state(event1.output_state);
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
                auto pair_hash = [](const std::pair<unified::EventId, unified::EventId>& p) {
                    return std::hash<uint64_t>{}((static_cast<uint64_t>(p.first) << 32) | p.second);
                };
                std::unordered_set<std::pair<unified::EventId, unified::EventId>, decltype(pair_hash)> seen_pairs(
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