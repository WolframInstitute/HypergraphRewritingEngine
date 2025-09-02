#include "WolframLibrary.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <unordered_map>

// Include basic hypergraph headers
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/canonicalization.hpp"
#include "hypergraph/wolfram_evolution.hpp"
#include "hypergraph/rewriting.hpp"
#include "hypergraph/wolfram_states.hpp"

using namespace hypergraph;

// WXF Parser class
class WXFParser {
private:
    const uint8_t* data;
    size_t size;
    size_t pos;
    
public:
    WXFParser(const uint8_t* d, size_t s) : data(d), size(s), pos(0) {}
    
    uint8_t read_byte() {
        if (pos >= size) throw std::runtime_error("WXF data underflow");
        return data[pos++];
    }
    
    size_t read_varint() {
        size_t value = 0;
        size_t shift = 0;
        uint8_t byte;
        do {
            byte = read_byte();
            value |= (size_t(byte & 0x7F) << shift);
            shift += 7;
        } while (byte & 0x80);
        return value;
    }
    
    int64_t read_integer() {
        uint8_t type = read_byte();
        switch(type) {
            case 'C': { // 8-bit
                int8_t val = read_byte();
                return val;
            }
            case 'j': { // 16-bit
                int16_t val = 0;
                for(int i = 0; i < 2; i++) {
                    val |= (read_byte() << (i * 8));
                }
                return val;
            }
            case 'i': { // 32-bit
                int32_t val = 0;
                for(int i = 0; i < 4; i++) {
                    val |= (read_byte() << (i * 8));
                }
                return val;
            }
            case 'L': { // 64-bit
                int64_t val = 0;
                for(int i = 0; i < 8; i++) {
                    val |= (static_cast<int64_t>(read_byte()) << (i * 8));
                }
                return val;
            }
            default:
                throw std::runtime_error("Unknown integer type in WXF");
        }
    }
    
    std::string read_string() {
        uint8_t type = read_byte();
        if (type != 'S') throw std::runtime_error("Expected string in WXF");
        size_t len = read_varint();
        std::string str(len, '\0');
        for(size_t i = 0; i < len; i++) {
            str[i] = read_byte();
        }
        return str;
    }
    
    std::vector<std::vector<int64_t>> read_list_of_lists() {
        uint8_t type = read_byte();
        if (type != 'f') throw std::runtime_error("Expected function (List) in WXF");
        
        size_t outer_len = read_varint();
        
        // Skip List symbol
        type = read_byte();
        if (type != 's') throw std::runtime_error("Expected symbol in WXF");
        size_t sym_len = read_varint();
        for(size_t i = 0; i < sym_len; i++) read_byte(); // Skip "List"
        
        std::vector<std::vector<int64_t>> result;
        result.reserve(outer_len);
        
        for(size_t i = 0; i < outer_len; i++) {
            type = read_byte();
            if (type != 'f') throw std::runtime_error("Expected inner function (List) in WXF");
            
            size_t inner_len = read_varint();
            
            // Skip List symbol
            type = read_byte();
            if (type != 's') throw std::runtime_error("Expected symbol in WXF");
            sym_len = read_varint();
            for(size_t j = 0; j < sym_len; j++) read_byte(); // Skip "List"
            
            std::vector<int64_t> inner;
            inner.reserve(inner_len);
            for(size_t j = 0; j < inner_len; j++) {
                inner.push_back(read_integer());
            }
            result.push_back(std::move(inner));
        }
        
        return result;
    }
    
    std::unordered_map<std::string, std::vector<std::vector<std::vector<int64_t>>>> read_rules_association() {
        uint8_t type = read_byte();
        if (type != 'A') throw std::runtime_error("Expected association in WXF");
        
        size_t num_rules = read_varint();
        std::unordered_map<std::string, std::vector<std::vector<std::vector<int64_t>>>> rules;
        
        for(size_t r = 0; r < num_rules; r++) {
            type = read_byte();
            if (type != '-') throw std::runtime_error("Expected rule marker in WXF");
            
            // Read rule name (e.g., "Rule1")
            std::string rule_name = read_string();
            
            // Read rule data (LHS -> RHS)
            type = read_byte();
            if (type != 'f') throw std::runtime_error("Expected Rule function in WXF");
            size_t rule_parts = read_varint();
            if (rule_parts != 2) throw std::runtime_error("Rule must have 2 parts (LHS and RHS)");
            
            // Skip Rule symbol
            type = read_byte();
            if (type != 's') throw std::runtime_error("Expected symbol in WXF");
            size_t sym_len = read_varint();
            for(size_t i = 0; i < sym_len; i++) read_byte(); // Skip "Rule"
            
            // Read LHS
            auto lhs = read_list_of_lists();
            // Read RHS  
            auto rhs = read_list_of_lists();
            
            rules[rule_name] = {lhs, rhs};
        }
        
        return rules;
    }
    
    void skip_header() {
        // Skip WXF header "8:"
        if (read_byte() != '8' || read_byte() != ':') {
            throw std::runtime_error("Invalid WXF header");
        }
    }
};

// WXF Serialization helpers
void add_varint(std::vector<uint8_t>& data, std::size_t value) {
    while (value >= 0x80) {
        data.push_back(0x80 | (value & 0x7F));
        value >>= 7;
    }
    data.push_back(value & 0x7F);
}

void add_string_to_wxf(std::vector<uint8_t>& wxf_data, const std::string& str) {
    wxf_data.push_back('S');
    add_varint(wxf_data, str.size());
    wxf_data.insert(wxf_data.end(), str.begin(), str.end());
}

void add_integer_to_wxf(std::vector<uint8_t>& wxf_data, int64_t value) {
    if (value >= -128 && value <= 127) {
        wxf_data.push_back('C');
        wxf_data.push_back(static_cast<uint8_t>(value & 0xFF));
    } else if (value >= INT16_MIN && value <= INT16_MAX) {
        wxf_data.push_back('j');
        int16_t val16 = static_cast<int16_t>(value);
        wxf_data.push_back(val16 & 0xFF);
        wxf_data.push_back((val16 >> 8) & 0xFF);
    } else if (value >= INT32_MIN && value <= INT32_MAX) {
        wxf_data.push_back('i');
        int32_t val32 = static_cast<int32_t>(value);
        for (int i = 0; i < 4; ++i) {
            wxf_data.push_back(val32 & 0xFF);
            val32 >>= 8;
        }
    } else {
        wxf_data.push_back('L');
        for (int i = 0; i < 8; ++i) {
            wxf_data.push_back(value & 0xFF);
            value >>= 8;
        }
    }
}

template<typename Func>
void add_rule_to_wxf(std::vector<uint8_t>& wxf_data, const std::string& key, Func value_func) {
    wxf_data.push_back('-');
    add_string_to_wxf(wxf_data, key);
    value_func(wxf_data);
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
EXTERN_C DLLEXPORT int performRewritingWXF(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 1) {
            handle_error(libData, "performRewritingWXF expects 1 argument: WXF data");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        // Get WXF data tensor
        MTensor wxf_tensor = MArgument_getMTensor(argv[0]);
        if (libData->MTensor_getRank(wxf_tensor) != 1) {
            handle_error(libData, "WXF data must be a 1D byte tensor");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        const mint* dims = libData->MTensor_getDimensions(wxf_tensor);
        mint wxf_size = dims[0];
        const mint* wxf_mint_data = libData->MTensor_getIntegerData(wxf_tensor);
        
        // Convert mint array to byte array
        std::vector<uint8_t> wxf_bytes(wxf_size);
        for (mint i = 0; i < wxf_size; i++) {
            wxf_bytes[i] = static_cast<uint8_t>(wxf_mint_data[i]);
        }
        
        // Parse WXF input
        WXFParser parser(wxf_bytes.data(), wxf_bytes.size());
        parser.skip_header();
        
        // Read main association
        uint8_t type = parser.read_byte();
        if (type != 'A') {
            handle_error(libData, "Expected Association at top level");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        size_t num_entries = parser.read_varint();
        
        std::vector<std::vector<GlobalVertexId>> initial_edges;
        std::vector<RewritingRule> parsed_rules;
        mint steps = 1;
        
        // Parse association entries
        for (size_t e = 0; e < num_entries; e++) {
            type = parser.read_byte();
            if (type != '-') {
                handle_error(libData, "Expected rule marker in Association");
                return LIBRARY_FUNCTION_ERROR;
            }
            
            std::string key = parser.read_string();
            
            if (key == "InitialEdges") {
                auto edges_data = parser.read_list_of_lists();
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
                auto rules = parser.read_rules_association();
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
                steps = static_cast<mint>(parser.read_integer());
            }
        }
        
        if (initial_edges.empty()) {
            handle_error(libData, "No initial edges provided");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        if (parsed_rules.empty()) {
            handle_error(libData, "No valid rules found");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        // Run evolution
        WolframEvolution evolution(static_cast<std::size_t>(steps), 1, true, true);
        
        for (const auto& rule : parsed_rules) {
            evolution.add_rule(rule);
        }
        
        evolution.evolve(initial_edges);
        
        // Get results
        const auto& multiway_graph = evolution.get_multiway_graph();
        auto all_states = multiway_graph.get_all_states();
        auto all_events = multiway_graph.get_all_events();
        auto event_edges = multiway_graph.get_event_edges();
        
        // Prepare edge vectors
        std::vector<std::pair<std::size_t, std::size_t>> causal_edges;
        std::vector<std::pair<std::size_t, std::size_t>> branchial_edges;
        
        if (event_edges.empty() && all_events.size() > 1) {
            for (const auto& event_a : all_events) {
                for (const auto& event_b : all_events) {
                    if (event_a.event_id != event_b.event_id) {
                        if (event_a.output_state_id == event_b.input_state_id) {
                            causal_edges.push_back(std::make_pair(event_a.event_id, event_b.event_id));
                        }
                        else if (event_a.input_state_id == event_b.input_state_id && event_a.event_id < event_b.event_id) {
                            branchial_edges.push_back(std::make_pair(event_a.event_id, event_b.event_id));
                        }
                    }
                }
            }
        } else {
            for (const auto& edge : event_edges) {
                if (edge.type == EventRelationType::CAUSAL) {
                    causal_edges.push_back(std::make_pair(edge.from_event, edge.to_event));
                } else {
                    branchial_edges.push_back(std::make_pair(edge.from_event, edge.to_event));
                }
            }
        }
        
        // Create WXF output
        std::vector<uint8_t> wxf_data;
        
        // WXF Header
        wxf_data.push_back('8');
        wxf_data.push_back(':');
        
        // Main association
        wxf_data.push_back('A');
        add_varint(wxf_data, 4); // 4 key-value pairs
        
        // "States" -> List of states
        add_rule_to_wxf(wxf_data, "States", [&](std::vector<uint8_t>& d) {
            d.push_back('f');
            add_varint(d, all_states.size());
            d.push_back('s');
            add_varint(d, 4);
            d.insert(d.end(), {'L', 'i', 's', 't'});
            
            for (const auto& state : all_states) {
                const auto& edges = state.edges();
                d.push_back('f');
                add_varint(d, edges.size());
                d.push_back('s');
                add_varint(d, 4);
                d.insert(d.end(), {'L', 'i', 's', 't'});
                
                for (const auto& edge : edges) {
                    d.push_back('f');
                    add_varint(d, edge.global_vertices.size());
                    d.push_back('s');
                    add_varint(d, 4);
                    d.insert(d.end(), {'L', 'i', 's', 't'});
                    
                    for (const auto& vertex : edge.global_vertices) {
                        add_integer_to_wxf(d, static_cast<int64_t>(vertex));
                    }
                }
            }
        });
        
        // "Events" -> List of events
        add_rule_to_wxf(wxf_data, "Events", [&](std::vector<uint8_t>& d) {
            d.push_back('f');
            add_varint(d, all_events.size());
            d.push_back('s');
            add_varint(d, 4);
            d.insert(d.end(), {'L', 'i', 's', 't'});
            
            for (const auto& event : all_events) {
                d.push_back('A');
                add_varint(d, 6);
                
                add_rule_to_wxf(d, "EventId", [&](std::vector<uint8_t>& dd) {
                    add_integer_to_wxf(dd, static_cast<int64_t>(event.event_id));
                });
                add_rule_to_wxf(d, "RuleIndex", [&](std::vector<uint8_t>& dd) {
                    add_integer_to_wxf(dd, static_cast<int64_t>(event.rule_index));
                });
                add_rule_to_wxf(d, "InputStateId", [&](std::vector<uint8_t>& dd) {
                    add_integer_to_wxf(dd, static_cast<int64_t>(event.input_state_id));
                });
                add_rule_to_wxf(d, "OutputStateId", [&](std::vector<uint8_t>& dd) {
                    add_integer_to_wxf(dd, static_cast<int64_t>(event.output_state_id));
                });
                
                add_rule_to_wxf(d, "ConsumedEdges", [&](std::vector<uint8_t>& dd) {
                    dd.push_back('f');
                    add_varint(dd, event.consumed_edges.size());
                    dd.push_back('s');
                    add_varint(dd, 4);
                    dd.insert(dd.end(), {'L', 'i', 's', 't'});
                    for (const auto& edge_id : event.consumed_edges) {
                        add_integer_to_wxf(dd, static_cast<int64_t>(edge_id));
                    }
                });
                
                add_rule_to_wxf(d, "ProducedEdges", [&](std::vector<uint8_t>& dd) {
                    dd.push_back('f');
                    add_varint(dd, event.produced_edges.size());
                    dd.push_back('s');
                    add_varint(dd, 4);
                    dd.insert(dd.end(), {'L', 'i', 's', 't'});
                    for (const auto& edge_id : event.produced_edges) {
                        add_integer_to_wxf(dd, static_cast<int64_t>(edge_id));
                    }
                });
            }
        });
        
        // "CausalEdges" -> List of causal edges
        add_rule_to_wxf(wxf_data, "CausalEdges", [&](std::vector<uint8_t>& d) {
            d.push_back('f');
            add_varint(d, causal_edges.size());
            d.push_back('s');
            add_varint(d, 4);
            d.insert(d.end(), {'L', 'i', 's', 't'});
            
            for (const auto& edge : causal_edges) {
                d.push_back('f');
                add_varint(d, 2);
                d.push_back('s');
                add_varint(d, 4);
                d.insert(d.end(), {'L', 'i', 's', 't'});
                add_integer_to_wxf(d, static_cast<int64_t>(edge.first));
                add_integer_to_wxf(d, static_cast<int64_t>(edge.second));
            }
        });
        
        // "BranchialEdges" -> List of branchial edges
        add_rule_to_wxf(wxf_data, "BranchialEdges", [&](std::vector<uint8_t>& d) {
            d.push_back('f');
            add_varint(d, branchial_edges.size());
            d.push_back('s');
            add_varint(d, 4);
            d.insert(d.end(), {'L', 'i', 's', 't'});
            
            for (const auto& edge : branchial_edges) {
                d.push_back('f');
                add_varint(d, 2);
                d.push_back('s');
                add_varint(d, 4);
                d.insert(d.end(), {'L', 'i', 's', 't'});
                add_integer_to_wxf(d, static_cast<int64_t>(edge.first));
                add_integer_to_wxf(d, static_cast<int64_t>(edge.second));
            }
        });
        
        // Create output tensor
        mint wxf_dims[1] = {static_cast<mint>(wxf_data.size())};
        MTensor result_tensor;
        int err = libData->MTensor_new(MType_Integer, 1, wxf_dims, &result_tensor);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }
        
        mint* result_data = libData->MTensor_getIntegerData(result_tensor);
        for (std::size_t i = 0; i < wxf_data.size(); ++i) {
            result_data[i] = static_cast<mint>(wxf_data[i]);
        }
        
        MArgument_setMTensor(res, result_tensor);
        return LIBRARY_NO_ERROR;
        
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
}