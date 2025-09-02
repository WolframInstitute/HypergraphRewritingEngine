#include "WolframLibrary.h"
#include "WolframSparseLibrary.h"
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>

// Include our hypergraph headers
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/canonicalization.hpp"
#include "hypergraph/pattern_matching.hpp"
#include "hypergraph/rewriting.hpp"
#include "hypergraph/wolfram_states.hpp"
#include "hypergraph/wolfram_evolution.hpp"

using namespace hypergraph;

// Global state management
static std::unordered_map<mint, std::shared_ptr<Hypergraph>> hypergraph_store;
static std::unordered_map<mint, std::shared_ptr<WolframEvolution>> engine_store;
static mint next_id = 1;
static bool parallel_enabled = false;

// Error handling
static void handle_error(WolframLibraryData libData, const char* message) {
    libData->Message(message);
}

/**
 * Create a new hypergraph from an edge list
 * Input: MTensor of edges (2D integer array)
 * Output: DataStore ID
 */
EXTERN_C DLLEXPORT int createHypergraph(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 1) {
            handle_error(libData, "createHypergraph expects 1 argument");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        MTensor edges_tensor = MArgument_getMTensor(argv[0]);
        mint* dimensions = libData->MTensor_getDimensions(edges_tensor);
        mint rank = libData->MTensor_getRank(edges_tensor);
        
        if (rank != 2) {
            handle_error(libData, "Edge tensor must be 2D");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        mint num_edges = dimensions[0];
        mint max_arity = dimensions[1];
        mint* data = libData->MTensor_getIntegerData(edges_tensor);
        
        // Create hypergraph
        auto hg = std::make_shared<Hypergraph>();
        
        for (mint i = 0; i < num_edges; i++) {
            std::vector<VertexId> vertices;
            for (mint j = 0; j < max_arity; j++) {
                mint vertex = data[i * max_arity + j];
                if (vertex >= 0) { // -1 indicates unused slot
                    vertices.push_back(static_cast<VertexId>(vertex));
                }
            }
            if (!vertices.empty()) {
                hg->add_edge(vertices);
            }
        }
        
        // Store hypergraph and return ID
        mint id = next_id++;
        hypergraph_store[id] = hg;
        
        DataStore ds = libData->createDataStore();
        libData->DataStore_addInteger(ds, id);
        MArgument_setDataStore(res, ds);
        
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Add an edge to an existing hypergraph
 * Input: DataStore ID, edge vertices
 * Output: Updated DataStore ID
 */
EXTERN_C DLLEXPORT int addEdgeToHypergraph(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 2) {
            handle_error(libData, "addEdgeToHypergraph expects 2 arguments");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        DataStore ds = MArgument_getDataStore(argv[0]);
        mint hg_id = libData->DataStore_getInteger(ds, 1);
        
        auto it = hypergraph_store.find(hg_id);
        if (it == hypergraph_store.end()) {
            handle_error(libData, "Invalid hypergraph ID");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        MTensor vertices_tensor = MArgument_getMTensor(argv[1]);
        mint length = libData->MTensor_getFlattenedLength(vertices_tensor);
        mint* vertices_data = libData->MTensor_getIntegerData(vertices_tensor);
        
        std::vector<VertexId> vertices;
        for (mint i = 0; i < length; i++) {
            vertices.push_back(static_cast<VertexId>(vertices_data[i]));
        }
        
        // Add edge to hypergraph
        it->second->add_edge(vertices);
        
        // Return the same DataStore
        MArgument_setDataStore(res, ds);
        
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Canonicalize a hypergraph
 * Input: DataStore ID
 * Output: 2D integer tensor representing canonical edges
 */
EXTERN_C DLLEXPORT int canonicalizeHypergraph(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 1) {
            handle_error(libData, "canonicalizeHypergraph expects 1 argument");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        DataStore ds = MArgument_getDataStore(argv[0]);
        mint hg_id = libData->DataStore_getInteger(ds, 1);
        
        auto it = hypergraph_store.find(hg_id);
        if (it == hypergraph_store.end()) {
            handle_error(libData, "Invalid hypergraph ID");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        // Canonicalize hypergraph
        Canonicalizer canonicalizer;
        auto result = canonicalizer.canonicalize(*it->second);
        
        const auto& edges = result.canonicalized_hypergraph.edges();
        if (edges.empty()) {
            // Return empty tensor
            mint dims[2] = {0, 0};
            MTensor output;
            libData->MTensor_new(MType_Integer, 2, dims, &output);
            MArgument_setMTensor(res, output);
            return LIBRARY_NO_ERROR;
        }
        
        // Find maximum arity
        std::size_t max_arity = 0;
        for (const auto& edge : edges) {
            max_arity = std::max(max_arity, edge.vertices().size());
        }
        
        // Create output tensor
        mint dims[2] = {static_cast<mint>(edges.size()), static_cast<mint>(max_arity)};
        MTensor output;
        int err = libData->MTensor_new(MType_Integer, 2, dims, &output);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }
        
        mint* output_data = libData->MTensor_getIntegerData(output);
        
        // Fill output tensor
        for (std::size_t i = 0; i < edges.size(); i++) {
            const auto& vertices = edges[i].vertices();
            for (std::size_t j = 0; j < max_arity; j++) {
                if (j < vertices.size()) {
                    output_data[i * max_arity + j] = static_cast<mint>(vertices[j]);
                } else {
                    output_data[i * max_arity + j] = -1; // Pad with -1
                }
            }
        }
        
        MArgument_setMTensor(res, output);
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Pattern matching in hypergraph
 * Input: DataStore ID (hypergraph), DataStore ID (pattern)
 * Output: 2D integer tensor representing matches
 */
EXTERN_C DLLEXPORT int patternMatchHypergraph(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 2) {
            handle_error(libData, "patternMatchHypergraph expects 2 arguments");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        DataStore hg_ds = MArgument_getDataStore(argv[0]);
        DataStore pattern_ds = MArgument_getDataStore(argv[1]);
        
        mint hg_id = libData->DataStore_getInteger(hg_ds, 1);
        mint pattern_id = libData->DataStore_getInteger(pattern_ds, 1);
        
        auto hg_it = hypergraph_store.find(hg_id);
        auto pattern_it = hypergraph_store.find(pattern_id);
        
        if (hg_it == hypergraph_store.end() || pattern_it == hypergraph_store.end()) {
            handle_error(libData, "Invalid hypergraph ID");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        // Perform pattern matching
        PatternMatcher matcher;
        auto matches = matcher.find_all_matches(*pattern_it->second, *hg_it->second);
        
        if (matches.empty()) {
            // Return empty tensor
            mint dims[2] = {0, 0};
            MTensor output;
            libData->MTensor_new(MType_Integer, 2, dims, &output);
            MArgument_setMTensor(res, output);
            return LIBRARY_NO_ERROR;
        }
        
        // Convert matches to tensor format
        mint dims[2] = {static_cast<mint>(matches.size()), static_cast<mint>(matches[0].assignment.size())};
        MTensor output;
        int err = libData->MTensor_new(MType_Integer, 2, dims, &output);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }
        
        mint* output_data = libData->MTensor_getIntegerData(output);
        
        for (std::size_t i = 0; i < matches.size(); i++) {
            std::size_t j = 0;
            for (const auto& [pattern_vertex, target_vertex] : matches[i].assignment) {
                output_data[i * matches[0].assignment.size() + j] = static_cast<mint>(target_vertex);
                j++;
            }
        }
        
        MArgument_setMTensor(res, output);
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Apply rewriting rule to hypergraph
 * Input: DataStore ID (hypergraph), DataStore ID (LHS pattern), DataStore ID (RHS replacement)
 * Output: DataStore ID (new hypergraph)
 */
EXTERN_C DLLEXPORT int applyRewritingRule(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 3) {
            handle_error(libData, "applyRewritingRule expects 3 arguments");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        DataStore hg_ds = MArgument_getDataStore(argv[0]);
        DataStore lhs_ds = MArgument_getDataStore(argv[1]);
        DataStore rhs_ds = MArgument_getDataStore(argv[2]);
        
        mint hg_id = libData->DataStore_getInteger(hg_ds, 1);
        mint lhs_id = libData->DataStore_getInteger(lhs_ds, 1);
        mint rhs_id = libData->DataStore_getInteger(rhs_ds, 1);
        
        auto hg_it = hypergraph_store.find(hg_id);
        auto lhs_it = hypergraph_store.find(lhs_id);
        auto rhs_it = hypergraph_store.find(rhs_id);
        
        if (hg_it == hypergraph_store.end() || lhs_it == hypergraph_store.end() || rhs_it == hypergraph_store.end()) {
            handle_error(libData, "Invalid hypergraph ID");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        // Create rewriting rule
        RewritingRule rule(*lhs_it->second, *rhs_it->second);
        
        // Apply rule
        RewritingEngine engine;
        auto result_hg = engine.apply_rule(*hg_it->second, rule);
        
        if (!result_hg) {
            handle_error(libData, "Rule application failed");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        // Store result and return ID
        mint new_id = next_id++;
        hypergraph_store[new_id] = std::make_shared<Hypergraph>(*result_hg);
        
        DataStore new_ds = libData->createDataStore();
        libData->DataStore_addInteger(new_ds, new_id);
        MArgument_setDataStore(res, new_ds);
        
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Evolve hypergraph using multiway evolution
 * Input: DataStore ID (initial hypergraph), rule indices, number of steps
 * Output: 3D integer tensor representing evolution states
 */
EXTERN_C DLLEXPORT int evolveMultiway(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 3) {
            handle_error(libData, "evolveMultiway expects 3 arguments");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        DataStore hg_ds = MArgument_getDataStore(argv[0]);
        MTensor rules_tensor = MArgument_getMTensor(argv[1]);
        mint steps = MArgument_getInteger(argv[2]);
        
        mint hg_id = libData->DataStore_getInteger(hg_ds, 1);
        auto hg_it = hypergraph_store.find(hg_id);
        
        if (hg_it == hypergraph_store.end()) {
            handle_error(libData, "Invalid hypergraph ID");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        // Create Wolfram evolution engine
        WolframEvolution engine(steps, parallel_enabled ? 4 : 1);
        
        // Add dummy rules (this would need to be populated with actual rules from the rules tensor)
        // For now, create a simple rule as an example
        Hypergraph lhs, rhs;
        lhs.add_edge({0, 1, 2});
        rhs.add_edge({0, 1});
        rhs.add_edge({1, 2});
        rhs.add_edge({2, 0});
        RewritingRule dummy_rule(lhs, rhs);
        engine.add_rule(dummy_rule);
        
        // Convert initial hypergraph to initial edges
        std::vector<std::vector<GlobalVertexId>> initial_edges;
        for (const auto& edge : hg_it->second->edges()) {
            std::vector<GlobalVertexId> vertices;
            for (VertexId v : edge.vertices()) {
                vertices.push_back(static_cast<GlobalVertexId>(v));
            }
            initial_edges.push_back(vertices);
        }
        
        // Evolve with initial edges
        engine.evolve(initial_edges);
        
        // Get basic statistics (since full state access is private)
        std::size_t num_states = engine.get_multiway_graph().num_states();
        std::size_t num_events = engine.get_multiway_graph().num_events();
        
        // Return tensor format with evolution statistics
        mint dims[3] = {2, 1, 1}; // Just return number of states and events
        MTensor output;
        int err = libData->MTensor_new(MType_Integer, 3, dims, &output);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }
        
        mint* output_data = libData->MTensor_getIntegerData(output);
        output_data[0] = static_cast<mint>(num_states);
        output_data[1] = static_cast<mint>(num_events);
        
        MArgument_setMTensor(res, output);
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Get engine performance statistics
 */
EXTERN_C DLLEXPORT int getEngineStats(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        mint dims[1] = {4};
        MTensor output;
        int err = libData->MTensor_new(MType_Real, 1, dims, &output);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }
        
        double* data = libData->MTensor_getRealData(output);
        data[0] = static_cast<double>(hypergraph_store.size());
        data[1] = static_cast<double>(engine_store.size());
        data[2] = 0.0; // Cache hits - would need actual implementation
        data[3] = 0.0; // Cache misses - would need actual implementation
        
        MArgument_setMTensor(res, output);
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Clear internal caches
 */
EXTERN_C DLLEXPORT int clearEngineCache(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        hypergraph_store.clear();
        engine_store.clear();
        next_id = 1;
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Enable/disable parallel mode
 */
EXTERN_C DLLEXPORT int setParallelMode(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    try {
        if (argc != 1) {
            handle_error(libData, "setParallelMode expects 1 argument");
            return LIBRARY_FUNCTION_ERROR;
        }
        
        mbool enabled = MArgument_getBoolean(argv[0]);
        parallel_enabled = (enabled == True);
        
        return LIBRARY_NO_ERROR;
    } catch (const std::exception& e) {
        handle_error(libData, e.what());
        return LIBRARY_FUNCTION_ERROR;
    }
}

/**
 * Initialize and cleanup functions
 */
EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
    hypergraph_store.clear();
    engine_store.clear();
    next_id = 1;
    parallel_enabled = false;
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
    hypergraph_store.clear();
    engine_store.clear();
}