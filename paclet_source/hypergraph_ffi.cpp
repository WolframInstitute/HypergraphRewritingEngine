#include "WolframLibrary.h"
#include "WolframNumericArrayLibrary.h"
#include <vector>
#include <cstring>
#include <unordered_map>

// Include basic hypergraph headers
#include "hypergraph/wolfram_evolution.hpp"
#include "hypergraph/rewriting.hpp"
#include "hypergraph/wolfram_states.hpp"

// Include unified engine headers (V2)
#include "hypergraph/unified/unified_hypergraph.hpp"
#include "hypergraph/unified/parallel_evolution.hpp"
#include "hypergraph/unified/pattern.hpp"

// Include comprehensive WXF library
#include "wxf.hpp"

using namespace hypergraph;

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
        bool canonicalize_states = true;
        bool canonicalize_events = true;
        bool causal_transitive_reduction = true;
        size_t max_successor_states_per_parent = 0;
        size_t max_states_per_step = 0;
        double exploration_probability = 1.0;
        std::string hash_strategy = "iUT";  // Default: IncrementalUniquenessTree

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
                        } else if (option_key == "ExplorationProbability") {
                            exploration_probability = option_parser.read<double>();
                        } else if (option_key == "HashStrategy") {
                            hash_strategy = option_parser.read<std::string>();
                        } else {
                            std::string symbol = option_parser.read<std::string>();
                            bool value = (symbol == "True");

                            if (option_key == "CanonicalizeStates") {
                                canonicalize_states = value;
                            } else if (option_key == "CanonicalizeEvents") {
                                canonicalize_events = value;
                            } else if (option_key == "CausalTransitiveReduction") {
                                causal_transitive_reduction = value;
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
        if (canonicalize_events) {
            hg.set_event_canonicalization_mode(unified::EventCanonicalizationMode::ByStateAndEdges);
        } else {
            hg.set_event_canonicalization_mode(unified::EventCanonicalizationMode::None);
        }

        // Create parallel evolution engine
        unified::ParallelEvolutionEngine engine(&hg, std::thread::hardware_concurrency());

        // Configure engine options
        engine.set_max_steps(static_cast<size_t>(steps));
        engine.set_transitive_reduction(causal_transitive_reduction);
        engine.set_exploration_probability(exploration_probability);
        engine.set_max_successor_states_per_parent(max_successor_states_per_parent);
        engine.set_max_states_per_step(max_states_per_step);

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

        // Convert first initial state to vector of edges
        // (V2 currently supports single initial state - take first)
        std::vector<std::vector<unified::VertexId>> initial_edges;
        if (!initial_states_raw.empty()) {
            for (const auto& edge : initial_states_raw[0]) {
                std::vector<unified::VertexId> edge_vertices;
                for (int64_t v : edge) {
                    if (v >= 0) {
                        edge_vertices.push_back(static_cast<unified::VertexId>(v));
                    }
                }
                if (!edge_vertices.empty()) {
                    initial_edges.push_back(edge_vertices);
                }
            }
        }

        // Run evolution
        engine.evolve(initial_edges, static_cast<size_t>(steps));

        // Build WXF output
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::ValueAssociation full_result;

        // States -> Association[state_id -> Association["Edges" -> edges, "IsInitialState" -> bool]]
        wxf::ValueAssociation states_assoc;
        uint32_t num_states = hg.num_states();
        for (uint32_t sid = 0; sid < num_states; ++sid) {
            const unified::State& state = hg.get_state(sid);
            if (state.id == unified::INVALID_ID) continue;

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

            bool is_initial = (state.step == 0);

            wxf::ValueAssociation state_assoc;
            state_assoc.push_back({wxf::Value("Edges"), wxf::Value(edge_list)});
            state_assoc.push_back({wxf::Value("IsInitialState"), wxf::Value(is_initial ? "True" : "False")});

            states_assoc.push_back({wxf::Value(static_cast<int64_t>(sid)), wxf::Value(state_assoc)});
        }
        full_result.push_back({wxf::Value("States"), wxf::Value(states_assoc)});

        // Events -> Association[event_id -> event_data]
        wxf::ValueAssociation events_assoc;
        uint32_t num_events = hg.num_events();
        for (uint32_t eid = 0; eid < num_events; ++eid) {
            const unified::Event& event = hg.get_event(eid);
            if (event.id == unified::INVALID_ID) continue;

            wxf::ValueAssociation event_data;
            event_data.push_back({wxf::Value("EventId"), wxf::Value(static_cast<int64_t>(eid))});
            event_data.push_back({wxf::Value("RuleIndex"), wxf::Value(static_cast<int64_t>(event.rule_index))});
            event_data.push_back({wxf::Value("Multiplicity"), wxf::Value(static_cast<int64_t>(1))});
            event_data.push_back({wxf::Value("InputStateId"), wxf::Value(static_cast<int64_t>(event.input_state))});
            event_data.push_back({wxf::Value("OutputStateId"), wxf::Value(static_cast<int64_t>(event.output_state))});

            // Get canonical states
            unified::StateId canonical_input = hg.get_canonical_state(event.input_state);
            unified::StateId canonical_output = hg.get_canonical_state(event.output_state);
            event_data.push_back({wxf::Value("CanonicalInputStateId"), wxf::Value(static_cast<int64_t>(canonical_input))});
            event_data.push_back({wxf::Value("CanonicalOutputStateId"), wxf::Value(static_cast<int64_t>(canonical_output))});

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

        // CausalEdges
        wxf::ValueList causal_edges;
        auto causal_edge_vec = hg.causal_graph().get_causal_edges();
        for (const auto& edge : causal_edge_vec) {
            wxf::ValueAssociation edge_data;
            edge_data.push_back({wxf::Value("From"), wxf::Value(static_cast<int64_t>(edge.producer))});
            edge_data.push_back({wxf::Value("To"), wxf::Value(static_cast<int64_t>(edge.consumer))});
            edge_data.push_back({wxf::Value("Multiplicity"), wxf::Value(static_cast<int64_t>(1))});
            causal_edges.push_back(wxf::Value(edge_data));
        }
        full_result.push_back({wxf::Value("CausalEdges"), wxf::Value(causal_edges)});

        // BranchialEdges
        wxf::ValueList branchial_edges;
        auto branchial_edge_vec = hg.causal_graph().get_branchial_edges();
        for (const auto& edge : branchial_edge_vec) {
            wxf::ValueAssociation edge_data;
            edge_data.push_back({wxf::Value("From"), wxf::Value(static_cast<int64_t>(edge.event1))});
            edge_data.push_back({wxf::Value("To"), wxf::Value(static_cast<int64_t>(edge.event2))});
            edge_data.push_back({wxf::Value("Multiplicity"), wxf::Value(static_cast<int64_t>(1))});
            branchial_edges.push_back(wxf::Value(edge_data));
        }
        full_result.push_back({wxf::Value("BranchialEdges"), wxf::Value(branchial_edges)});

        // Counts
        full_result.push_back({wxf::Value("NumStates"), wxf::Value(static_cast<int64_t>(hg.num_canonical_states()))});
        full_result.push_back({wxf::Value("NumEvents"), wxf::Value(static_cast<int64_t>(engine.num_events()))});
        full_result.push_back({wxf::Value("NumCausalEdges"), wxf::Value(static_cast<int64_t>(hg.num_causal_edges()))});
        full_result.push_back({wxf::Value("NumBranchialEdges"), wxf::Value(static_cast<int64_t>(hg.num_branchial_edges()))});

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

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData /* libData */) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData /* libData */) {
}