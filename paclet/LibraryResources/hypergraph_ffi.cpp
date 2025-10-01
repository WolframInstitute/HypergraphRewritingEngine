#include "WolframLibrary.h"
#include "WolframNumericArrayLibrary.h"
#include <vector>
#include <cstring>
#include <unordered_map>

// Include basic hypergraph headers
#include "hypergraph/wolfram_evolution.hpp"
#include "hypergraph/rewriting.hpp"
#include "hypergraph/wolfram_states.hpp"

// Include comprehensive WXF library
#include "wxf.hpp"

using namespace hypergraph;

// WXF Helper Functions using comprehensive wxf library
namespace ffi_helpers {
    // Parse rules association using wxf library
    std::unordered_map<std::string, std::vector<std::vector<std::vector<int64_t>>>>
    read_rules_association(wxf::Parser& parser) {
        std::unordered_map<std::string, std::vector<std::vector<std::vector<int64_t>>>> rules;

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

            rules[rule_name] = rule_parts;
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

        std::vector<std::vector<GlobalVertexId>> initial_edges;
        std::vector<RewritingRule> parsed_rules;
        int steps = 1;

        // Default option values
        bool canonicalize_states = true;
        bool canonicalize_events = true;
        bool causal_transitive_reduction = true;
        bool early_termination = false;
        bool full_capture = true;  // Enable to store states for visualization
        bool full_capture_non_canonicalised = false;  // Store all states, not just canonical ones

        // Parse main association using comprehensive wxf library
        parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
            if (key == "InitialEdges") {
                auto edges_data = value_parser.read<std::vector<std::vector<int64_t>>>();
                for (const auto& edge : edges_data) {
                    std::vector<GlobalVertexId> edge_vertices;
                    for (int64_t v : edge) {
                        if (v >= 0) {
                            edge_vertices.push_back(static_cast<GlobalVertexId>(v));
                        }
                    }
                    if (!edge_vertices.empty()) {
                        initial_edges.push_back(edge_vertices);
                    }
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
                        // Try to read as string (symbol) - this handles both String and Symbol tokens now
                        std::string symbol = option_parser.read<std::string>();
                        bool value = (symbol == "True");

                        if (option_key == "CanonicalizeStates") {
                            canonicalize_states = value;
                        } else if (option_key == "CanonicalizeEvents") {
                            canonicalize_events = value;
                        } else if (option_key == "CausalTransitiveReduction") {
                            causal_transitive_reduction = value;
                        } else if (option_key == "EarlyTermination") {
                            early_termination = value;
                        } else if (option_key == "FullCapture") {
                            full_capture = value;
                        }
                        // If we successfully read but don't recognize the option, that's fine
                        // The value was consumed
                    } catch (...) {
                        // If reading as string failed, skip whatever the value is
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

        if (initial_edges.empty()) {
            handle_error(libData, "No initial edges provided");
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
        WolframEvolution evolution(static_cast<std::size_t>(steps), std::thread::hardware_concurrency(),
                                   canonicalize_states, full_capture, canonicalize_events,
                                   causal_transitive_reduction, early_termination, full_capture_non_canonicalised);

        for (const auto& rule : parsed_rules) {
            evolution.add_rule(rule);
        }

        evolution.evolve(initial_edges);

        // Get results from multiway graph
        const auto& multiway_graph = evolution.get_multiway_graph();
        auto all_states = multiway_graph.get_all_states();
        auto all_events = multiway_graph.get_all_events();
        auto event_edges = multiway_graph.get_event_edges();


        // Create WXF output using comprehensive wxf library
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        // Build main association

        // States -> Association[state_id -> state_edges]
        std::unordered_map<int64_t, std::vector<std::vector<int64_t>>> states_map;
        for (const auto& state : all_states) {
            StateID state_id_to_use = state->has_canonical_id()
                                     ? state->get_canonical_id()
                                     : state->id();

            std::vector<std::vector<int64_t>> state_edges;
            for (const auto& edge : state->edges()) {
                std::vector<int64_t> edge_data = {static_cast<int64_t>(edge.global_id)};
                for (const auto& vertex : edge.global_vertices) {
                    edge_data.push_back(static_cast<int64_t>(vertex));
                }
                state_edges.push_back(edge_data);
            }
            states_map[static_cast<int64_t>(state_id_to_use.value)] = state_edges;
        }

        // Build full result using Value type for heterogeneous association
        wxf::ValueAssociation full_result;

        // States -> Association[state_id -> state_edges]
        wxf::ValueAssociation states_assoc;
        for (const auto& [state_id, edges] : states_map) {
            wxf::ValueList edge_list;
            for (const auto& edge : edges) {
                wxf::ValueList edge_data;
                for (int64_t v : edge) {
                    edge_data.push_back(wxf::Value(v));
                }
                edge_list.push_back(wxf::Value(edge_data));
            }
            states_assoc.push_back({wxf::Value(state_id), wxf::Value(edge_list)});
        }
        full_result.push_back({wxf::Value("States"), wxf::Value(states_assoc)});

        // Events -> List of event associations
        wxf::ValueList events_list;
        for (const auto& event : all_events) {
            wxf::ValueAssociation event_data;
            event_data.push_back({wxf::Value("EventId"), wxf::Value(static_cast<int64_t>(event.event_id))});
            event_data.push_back({wxf::Value("RuleIndex"), wxf::Value(static_cast<int64_t>(event.rule_index))});

            StateID input_state_id = event.has_canonical_input_state_id()
                                    ? event.canonical_input_state_id
                                    : event.input_state_id;
            event_data.push_back({wxf::Value("InputStateId"), wxf::Value(static_cast<int64_t>(input_state_id.value))});

            StateID output_state_id = event.has_canonical_output_state_id()
                                     ? event.canonical_output_state_id
                                     : event.output_state_id;
            event_data.push_back({wxf::Value("OutputStateId"), wxf::Value(static_cast<int64_t>(output_state_id.value))});

            events_list.push_back(wxf::Value(event_data));
        }
        full_result.push_back({wxf::Value("Events"), wxf::Value(events_list)});

        // CausalEdges and BranchialEdges
        wxf::ValueList causal_edges, branchial_edges;
        for (const auto& edge : event_edges) {
            wxf::ValueList edge_pair;
            edge_pair.push_back(wxf::Value(static_cast<int64_t>(edge.from_event)));
            edge_pair.push_back(wxf::Value(static_cast<int64_t>(edge.to_event)));

            if (edge.type == EventRelationType::CAUSAL) {
                causal_edges.push_back(wxf::Value(edge_pair));
            } else if (edge.type == EventRelationType::BRANCHIAL) {
                branchial_edges.push_back(wxf::Value(edge_pair));
            }
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

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData /* libData */) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData /* libData */) {
}