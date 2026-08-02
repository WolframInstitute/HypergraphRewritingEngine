#ifndef HG_STANDALONE_BINARY
#include "WolframLibrary.h"
#include "WolframNumericArrayLibrary.h"
#endif  // HG_STANDALONE_BINARY
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>

#include "hg_core.hpp"
#include "hg_gpu_backend.hpp"

// Include unified engine headers
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/pattern.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include "hypergraph/debug_log.hpp"
#include "job_system/job_system.hpp"

// Include comprehensive WXF library
#include "wxf.hpp"
#include "graph_marshal.hpp"

using namespace hypergraph;


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

// The core reports progress and checks for abort through the HostBridge; these
// wrappers make an empty callback a no-op / never-abort.
static inline void core_progress(const HostBridge& host, const std::string& message) {
    if (host.progress) host.progress(message);
}

#ifndef HG_STANDALONE_BINARY
// Error handling
static void handle_error(WolframLibraryData libData, const char* message) {
    if (libData && libData->Message) {
        libData->Message(message);
    }
}
#endif  // HG_STANDALONE_BINARY


/**
 * Perform multiway rewriting evolution
 * Input: WXF binary data as 1D byte tensor containing:
 *   Association[
 *     "InitialEdges" -> {{vertices...}, ...},
 *     "Rules" -> <"Rule1" -> {{lhs edges}, {rhs edges}}, ...>,
 *     "Steps" -> integer,
 *     "Options" -> Association[...,
 *       "CanonicalizeStates" -> None|Automatic|Full,
 *       ...]
 *   ]
 *
 * Output: WXF Association with States, Events, CausalEdges, BranchialEdges
 * When CanonicalizeStates is Full, states are deduplicated and edges are
 * IR-canonicalized (vertices relabeled to 0..n-1, edges sorted).
 */
std::vector<uint8_t> run_rewriting_core(const std::vector<uint8_t>& wxf_bytes,
                                        const HostBridge& host) {
    try {
        // Parse WXF input
        wxf::Parser parser(wxf_bytes);
        parser.skip_header();

        std::vector<std::vector<std::vector<int64_t>>> initial_states_raw;
        std::vector<std::pair<std::string, std::vector<std::vector<std::vector<int64_t>>>>> parsed_rules_raw;
        int steps = 1;

        // Warning trail served under the "Warnings" result key, schema shared with the GPU
        // backend (Kind/Count/Context) so the WL formatter handles both backends. Collects
        // option-parse skips, analysis refusals, and the engine's own warnings.
        struct FfiWarning { std::string kind; int64_t count; std::string context; };
        std::vector<FfiWarning> ffi_warnings;

        // Option values
        hypergraph::StateCanonicalizationMode state_canon_mode = hypergraph::StateCanonicalizationMode::None;  // Default: tree mode
        hypergraph::EventSignatureKeys event_signature_keys = hypergraph::EVENT_SIG_NONE;  // Default: no event canonicalization
        bool positional_event_identity = false;  // CanonicalizeEvents -> "Positional"
        bool show_genesis_events = false;
        bool show_progress = false;
        bool causal_transitive_reduction = true;
        size_t max_successor_states_per_parent = 0;
        size_t max_states_per_step = 0;
        double exploration_probability = 1.0;
        bool explore_from_canonical_states_only = false;  // Exploration deduplication
        bool quotient_initial_states = false;             // Collapse isomorphic initial states
        // ir_verification and return_canonical_states are derived from state_canon_mode == Full
        bool uniform_random = false;  // Use uniform random match selection (reservoir sampling)
        size_t matches_per_step = 0;  // Matches per step in uniform random mode (0 = all)

        // Data selection flags - which components to include in output
        // By default all are included for backward compatibility
        bool include_states = true;
        bool include_canonical_hashes = false;  // Emit per-state IR canonical hash (CanonicalHash); stable across runs, for cross-run fusion
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
                // Every budget check in the engine is `step > max_steps_`, and this is cast to
                // size_t downstream -- so a negative value becomes SIZE_MAX and no check ever
                // fires. The run then ends only by exhausting the SegmentedArray ceiling,
                // after multi-GB of arena, as a hard failure.
                if (steps < 0) {
                    throw std::runtime_error("Steps must be non-negative, got " +
                                             std::to_string(steps));
                }
            }
            else if (key == "Options") {
                value_parser.read_association([&](const std::string& option_key, wxf::Parser& option_parser) {
                    // A read that throws mid-value leaves the cursor INSIDE the value, and a
                    // skip from there consumes the wrong tokens -- every later option then
                    // parses from a shifted offset. Recovery seeks back here first.
                    const size_t option_value_start = option_parser.position();
                    try {
                        if (option_key == "MaxSuccessorStatesPerParent") {
                            max_successor_states_per_parent = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "MaxStatesPerStep") {
                            max_states_per_step = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "MatchesPerStep") {
                            matches_per_step = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "ExplorationProbability") {
                            exploration_probability = option_parser.read<double>();
                        } else if (option_key == "BranchialStep") {
                            // 0=All, positive=1-based step index, negative=from end (-1=final)
                            branchial_step = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "EdgeDeduplication") {
                            std::string symbol = option_parser.read<std::string>();
                            edge_deduplication = (symbol == "True");
                        } else if (option_key == "IncludeCanonicalHashes") {
                            std::string symbol = option_parser.read<std::string>();
                            include_canonical_hashes = (symbol == "True");
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
                                // Try to read as list first. A failure mid-list (the header
                                // parses, an element does not) leaves the cursor inside the
                                // list, so the symbol fallback re-aligns to the value start.
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
                                // Read as symbol, from the START of the value.
                                option_parser.seek(option_value_start);
                                std::string symbol = option_parser.read<std::string>();
                                if (symbol == "None") {
                                    event_signature_keys = hypergraph::EVENT_SIG_NONE;
                                } else if (symbol == "Full") {
                                    event_signature_keys = hypergraph::EVENT_SIG_FULL;
                                } else if (symbol == "Automatic") {
                                    event_signature_keys = hypergraph::EVENT_SIG_AUTOMATIC;
                                } else if (symbol == "Positional") {
                                    // Same key set as Automatic; ranks read from each raw
                                    // state's own labelling rather than the class frame.
                                    event_signature_keys = hypergraph::EVENT_SIG_AUTOMATIC;
                                    positional_event_identity = true;
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
                            } else if (option_key == "QuotientInitialStates") {
                                quotient_initial_states = value;
                            } else if (option_key == "ExploreFromCanonicalStatesOnly") {
                                // Exploration deduplication: only explore from canonical states
                                // Requires CanonicalizeStates -> Full to have any effect
                                explore_from_canonical_states_only = value;
                            } else if (option_key == "UniformRandom") {
                                // Use uniform random evolution mode (reservoir sampling)
                                uniform_random = value;
                            }
                        }
                    } catch (const std::exception& e) {
                        // An option's value failed to parse (wrong WXF type, out-of-range
                        // narrowing, etc.). Skip the malformed value and continue — the
                        // alternative is aborting the whole evolve call on a single bad
                        // option, which breaks WL callers that pass forward-compatible
                        // option sets the C++ side doesn't yet know about. The failed read
                        // may have consumed part of the value, so re-align to its start
                        // before skipping, and put the skip on the warning trail (the
                        // progress callback is a no-op under performRewriting).
                        option_parser.seek(option_value_start);
                        option_parser.skip_value();
                        ffi_warnings.push_back(
                            {"OptionSkipped", 1,
                             "option '" + option_key + "' ignored: " + e.what()});
                        core_progress(host,
                            "FFI: skipping malformed option '" + option_key + "': " + e.what());
                    }
                });

                // Handle numeric options that may have been parsed as integers
                // (these need special handling since they come after the bool options)
            }
            else {
                value_parser.skip_value();
            }
        });

        if (parsed_rules_raw.empty()) {
            throw std::runtime_error("No valid rules found");
        }

#ifdef HG_GPU_BACKEND
        // GPU binary: route the parsed job to hg_gpu::evolve and marshal its
        // result into the same WXF output.
        {
            // The per-step caps have no device implementation: EvolveInput carries no
            // max_states_per_step / max_successor_states_per_parent, so a capped run on the
            // GPU returns the UNCAPPED state set while the same call on the CPU returns a
            // capped one. Reported rather than applied -- silently returning a different
            // answer per device is the divergence class the differential suite exists to
            // catch, and a cap the caller asked for and did not get is exactly that.
            if (max_states_per_step > 0) {
                ffi_warnings.push_back(
                    {"OptionSkipped", 1,
                     "'MaxStatesPerStep' has no GPU implementation and was not applied; "
                     "the returned state set is uncapped. Use TargetDevice -> \"CPU\" to "
                     "apply it."});
            }
            if (max_successor_states_per_parent > 0) {
                ffi_warnings.push_back(
                    {"OptionSkipped", 1,
                     "'MaxSuccessorStatesPerParent' has no GPU implementation and was not "
                     "applied; the returned state set is uncapped. Use TargetDevice -> "
                     "\"CPU\" to apply it."});
            }
            if (uniform_random && matches_per_step > 0) {
                ffi_warnings.push_back(
                    {"OptionSkipped", 1,
                     "'MatchesPerStep' maps to the MaxStatesPerStep cap, which has no GPU "
                     "implementation and was not applied."});
            }

            GpuJob job{
                parsed_rules_raw,
                initial_states_raw,
                steps,
                // 0 None, 1 Full, 2 Automatic -- GpuJob::event_canon_mode's own order, which is
                // NOT the state order below and is not the enum order either. Collapsing this to
                // "0 if None else 1" sent code 1 for an AUTOMATIC request, and the backend reads
                // 1 as FULL: the caller silently got a coarser event identity than asked for, and
                // code 2 was never sent at all.
                (event_signature_keys == hypergraph::EVENT_SIG_NONE)      ? GpuJob::EventCanonCode::kNone :
                (event_signature_keys == hypergraph::EVENT_SIG_AUTOMATIC) ? GpuJob::EventCanonCode::kAutomatic
                                                                          : GpuJob::EventCanonCode::kFull,
                // 0 None, 1 Automatic, 2 Full (hg_gpu::CanonicalizationMode order)
                (state_canon_mode == hypergraph::StateCanonicalizationMode::Full)      ? GpuJob::StateCanonCode::kFull :
                (state_canon_mode == hypergraph::StateCanonicalizationMode::Automatic) ? GpuJob::StateCanonCode::kAutomatic
                                                                                       : GpuJob::StateCanonCode::kNone,
                causal_transitive_reduction,
                explore_from_canonical_states_only,
                quotient_initial_states,
                exploration_probability,
                0,  // max_device_memory_bytes: default (90% VRAM) resolved by the GPU engine
                include_states,
                include_events || include_events_minimal,
                include_causal_edges,
                include_branchial_edges,
                include_canonical_hashes,
                graph_properties,
                edge_deduplication,
                branchial_step,
                show_genesis_events,
            };
            if (show_progress) {
                core_progress(host, "HGEvolve: Starting GPU evolution...");
            }
            std::vector<uint8_t> out = run_gpu_evolution(job, host);
            if (show_progress) {
                core_progress(host, "HGEvolve: GPU evolution complete.");
            }
            return out;
        }
#endif

        // Create hypergraph
        hypergraph::Hypergraph hg;

        // The hot-path state hash is always Weisfeiler-Leman; exact IR
        // canonicalization is selected via CanonicalizeStates -> Full.

        // Full canonicalization mode: IR-based dedup, exact edge correspondence, canonical output
        const bool full_canonicalization = (state_canon_mode == hypergraph::StateCanonicalizationMode::Full);

        // Configure event canonicalization
        hg.set_event_signature_keys(event_signature_keys);
        hg.set_positional_event_identity(positional_event_identity);

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
        engine.set_quotient_initial_states(quotient_initial_states);

        // Convert rules to unified format
        uint16_t rule_index = 0;
        for (const auto& [rule_name, rule_data] : parsed_rules_raw) {
            if (rule_data.size() != 2) continue;

            hypergraph::RewriteRule rule;
            rule.index = rule_index++;

            // Track max variable seen for variable counting
            uint8_t max_lhs_var = 0;
            uint8_t max_rhs_var = 0;

            // Parse one side of the rule. Every limit is REPORTED, not absorbed: silently
            // truncating an over-long pattern, dropping an out-of-range variable or skipping a
            // negative id all hand back a DIFFERENT rule than the caller wrote, and the run
            // then succeeds, so nothing downstream can tell that it happened.
            //
            // The variable bound is the one that matters most. A pattern variable is an index
            // into VariableBinding's MAX_VARS-entry array and a bit position in its 32-bit
            // bound_mask, so a variable at or above MAX_VARS writes out of bounds and shifts
            // by more than the width -- memory corruption, not a wrong answer. Above 255 it
            // also wraps through uint8_t, silently merging two distinct variables.
            auto parse_side = [&](const auto& edges, hypergraph::PatternEdge* out,
                                  uint8_t& num_edges, uint8_t& max_var, const char* side) {
                num_edges = 0;
                size_t edge_index = 0;
                for (const auto& edge : edges) {
                    if (num_edges >= hypergraph::MAX_PATTERN_EDGES) {
                        throw std::runtime_error(
                            std::string("rule ") + std::to_string(rule.index) + " " + side +
                            " has more than " + std::to_string(hypergraph::MAX_PATTERN_EDGES) +
                            " edges");
                    }
                    hypergraph::PatternEdge& pe = out[num_edges];
                    pe.arity = 0;
                    for (int64_t v : edge) {
                        if (v < 0) {
                            throw std::runtime_error(
                                std::string("rule ") + std::to_string(rule.index) + " " + side +
                                " edge " + std::to_string(edge_index) +
                                " has a negative pattern variable");
                        }
                        if (v >= static_cast<int64_t>(hypergraph::MAX_VARS)) {
                            throw std::runtime_error(
                                std::string("rule ") + std::to_string(rule.index) + " " + side +
                                " uses pattern variable " + std::to_string(v) +
                                ", but the maximum is " +
                                std::to_string(hypergraph::MAX_VARS - 1));
                        }
                        if (pe.arity >= hypergraph::MAX_ARITY) {
                            throw std::runtime_error(
                                std::string("rule ") + std::to_string(rule.index) + " " + side +
                                " edge " + std::to_string(edge_index) + " has arity above " +
                                std::to_string(hypergraph::MAX_ARITY));
                        }
                        pe.vars[pe.arity++] = static_cast<uint8_t>(v);
                        if (static_cast<uint8_t>(v) > max_var) max_var = static_cast<uint8_t>(v);
                    }
                    if (pe.arity > 0) num_edges++;
                    ++edge_index;
                }
            };

            parse_side(rule_data[0], rule.lhs, rule.num_lhs_edges, max_lhs_var, "LHS");
            parse_side(rule_data[1], rule.rhs, rule.num_rhs_edges, max_rhs_var, "RHS");

            rule.num_lhs_vars = max_lhs_var + 1;
            rule.num_rhs_vars = max_rhs_var + 1;
            rule.num_new_vars = (max_rhs_var > max_lhs_var) ? (max_rhs_var - max_lhs_var) : 0;

            // An EMPTY RHS is a legitimate rule -- {{x,y}} -> {} deletes an edge, and the
            // engine gives the resulting empty state a canonical hash of its own precisely so
            // it works. Only an empty LHS is rejected, since it matches everywhere and would
            // not terminate.
            if (rule.num_lhs_edges == 0) {
                throw std::runtime_error(std::string("rule ") + std::to_string(rule.index) +
                                         " has an empty left-hand side");
            }
            engine.add_rule(rule);
        }


        // Convert all initial states to vectors of edges
        // Multiple initial states are supported for exploring the full multiway system
        // CRITICAL: Each initial state gets CANONICAL vertex numbering (starting from 0)
        // This ensures isomorphic initial states like {{0,0},{0,0}} and {{1,1},{1,1}}
        // get the SAME internal representation and thus the SAME canonical hash.
        // The engine handles multiplicity - if the same canonical state appears multiple
        // times, it spawns MATCH tasks for each instance.
        std::vector<std::vector<std::vector<hypergraph::VertexId>>> initial_states;
        std::unordered_map<int64_t, hypergraph::VertexId> initial_vertex_map;

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
                // GeodesicSources are given in the USER'S labels; the engine sees only the
                // dense renumbering above. Keep the first state's map so the sources can be
                // translated at the geodesic block (initial vertices keep their engine ids
                // through the evolution, so the translation stays valid on evolved states).
                if (initial_states.empty()) initial_vertex_map = vertex_map;
                initial_states.push_back(std::move(state_edges));
            }
        }

        // Run the evolution. Abort is a process kill by the parent, so there is
        // no cooperative abort; progress (when requested) is reported through the
        // host bridge.
        if (show_progress) {
            core_progress(host, "HGEvolve: Starting evolution...");
        }
        auto evolution_start = std::chrono::steady_clock::now();

        // MatchesPerStep is a per-DEPTH count, and a count over a depth cannot be sampled
        // without a barrier. What the step-synchronised path actually did with it was stop
        // applying once that many states existed for the step -- a cap by arrival order, which
        // MaxStatesPerStep already delivers with no barrier at all. So it maps to the cap it
        // always was, and the uniformity it used to claim moves to TransitionRate, which is a
        // rate and needs no depth to be defined over.
        if (uniform_random && matches_per_step > 0) {
            engine.set_max_states_per_step(matches_per_step);
        }
        engine.evolve(initial_states, static_cast<size_t>(steps));

        if (show_progress) {
            auto evolution_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - evolution_start).count();
            std::ostringstream oss;
            oss << "HGEvolve: Evolution complete in " << evolution_ms << "ms. "
                << "States: " << hg.num_canonical_states() << ", "
                << "Events: " << hg.num_events() << ", "
                << "Causal: " << hg.num_causal_event_pairs() << ", "
                << "Branchial: " << hg.num_branchial_edges();
            core_progress(host, oss.str());
            core_progress(host, "HGEvolve: Starting serialization...");
        }

        for (const auto& w : engine.warnings())
            ffi_warnings.push_back({"Engine", 1, w});

        // Build WXF output - only include requested data components
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::WXFValueAssociation full_result;

        // The States and Events sections dominate the result size, so they are
        // streamed straight into this scratch Writer as complete key->value blobs
        // rather than accumulated into an O(states+events) WXFValue tree. They are
        // spliced ahead of full_result at write time (see the tail of this function);
        // an Association is a flat concatenation of entries, so the byte stream is
        // identical to building one combined value tree.
        wxf::Writer sections;
        std::size_t streamed_top_sections = 0;

        // States -> Association[state_id -> state_data]
        // Send ALL states (not just canonical) - WL uses CanonicalId/ContentStateId for vertex merging
        // Each state includes: Id, CanonicalId, ContentStateId, Step, Edges, IsInitial
        // - CanonicalId: isomorphism-based (for Full mode) - isomorphic states share ID
        // - ContentStateId: content-based (for Automatic mode) - same-content states share ID
        // This matches reference behavior where canonicalization is applied at display time
        if (include_states) {
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

            // First pass fixes the emitted state set so the association length is
            // known before streaming. When CanonicalizeStates is Full, emit one state
            // per canonical ID (isomorphism-based deduplication).
            std::vector<uint32_t> emit_sids;
            emit_sids.reserve(num_states);
            std::unordered_set<hypergraph::StateId> emitted_canonical_ids;
            for (uint32_t sid = 0; sid < num_states; ++sid) {
                const hypergraph::State& state = hg.get_state(sid);
                if (state.id == hypergraph::INVALID_ID) continue;
                if (full_canonicalization) {
                    hypergraph::StateId cid = hg.get_canonical_state(sid);
                    if (!emitted_canonical_ids.insert(cid).second) continue;
                }
                emit_sids.push_back(sid);
            }

            // States -> Association[state_id -> state_data], streamed directly.
            sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            sections.write(std::string("States"));
            sections.write_byte(static_cast<uint8_t>(wxf::Token::Association));
            sections.write_varint(emit_sids.size());

            for (uint32_t sid : emit_sids) {
                const hypergraph::State& state = hg.get_state(sid);
                // Canonical state ID (isomorphism-based) and content state ID (from cached hash).
                hypergraph::StateId canonical_id = hg.get_canonical_state(sid);
                hypergraph::StateId content_id = content_hash_to_id[state_content_hashes[sid]];

                // Association key: raw state id.
                sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                sections.write(static_cast<int64_t>(sid));

                // state_data association: Id, CanonicalId, ContentStateId, Step, Edges,
                // IsInitial, and (optionally) CanonicalHash.
                sections.write_byte(static_cast<uint8_t>(wxf::Token::Association));
                sections.write_varint(include_canonical_hashes ? 7u : 6u);

                auto put_i64 = [&](const char* k, int64_t v) {
                    sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    sections.write(std::string(k));
                    sections.write(v);
                };
                put_i64("Id", static_cast<int64_t>(sid));
                put_i64("CanonicalId", static_cast<int64_t>(canonical_id));
                put_i64("ContentStateId", static_cast<int64_t>(content_id));
                put_i64("Step", static_cast<int64_t>(state.step));

                // Edges -> List of {edge_id, v1, v2, ...}. When CanonicalizeStates is
                // Full, edges are IR-canonicalized (vertices 0..n-1, sorted) with
                // sequential edge IDs.
                sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                sections.write(std::string("Edges"));
                if (full_canonicalization) {
                    std::vector<std::vector<hypergraph::VertexId>> edge_vecs;
                    state.edges.for_each([&](hypergraph::EdgeId eid) {
                        const hypergraph::Edge& e = hg.get_edge(eid);
                        edge_vecs.emplace_back(e.vertices, e.vertices + e.arity);
                    });

                    if (!edge_vecs.empty()) {
                        hypergraph::IRCanonicalizer ir;
                        auto canon_result = ir.canonicalize_edges(edge_vecs);
                        const auto& cedges = canon_result.canonical_form.edges;
                        sections.write_function("List", cedges.size());
                        int64_t edge_idx = 0;
                        for (const auto& canon_edge : cedges) {
                            sections.write_function("List", canon_edge.size() + 1);
                            sections.write(edge_idx++);
                            for (auto v : canon_edge) sections.write(static_cast<int64_t>(v));
                        }
                    } else {
                        sections.write_function("List", 0);
                    }
                } else {
                    sections.write_function("List", state.edges.count());
                    state.edges.for_each([&](hypergraph::EdgeId eid) {
                        const hypergraph::Edge& edge = hg.get_edge(eid);
                        sections.write_function("List", static_cast<std::size_t>(edge.arity) + 1);
                        sections.write(static_cast<int64_t>(eid));
                        for (uint8_t i = 0; i < edge.arity; ++i)
                            sections.write(static_cast<int64_t>(edge.vertices[i]));
                    });
                }

                // IsInitial -> boolean, serialized as the 0/1 integer the value tree
                // produces (WXFValue(bool) stores int64).
                sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                sections.write(std::string("IsInitial"));
                sections.write(static_cast<int64_t>(state.step == 0));

                if (include_canonical_hashes) {
                    // The option's contract is the IR canonical hash: identical for isomorphic
                    // states within and across runs, so it can key cross-run fusion. The dedup
                    // map's own key (state.canonical_hash) IS that hash under Full but the WL
                    // hash otherwise -- WL is isomorphism-invariant yet INCOMPLETE, so WL
                    // equality does not imply isomorphism and would silently merge
                    // non-isomorphic states in a fusion. Outside Full the exact hash is
                    // computed here, per serialized state, priced only under the option.
                    // The empty state keeps its dedicated engine hash in every mode.
                    uint64_t exact_hash = state.canonical_hash;
                    if (!full_canonicalization && state.edges.count() > 0) {
                        std::vector<std::vector<hypergraph::VertexId>> hash_edges;
                        state.edges.for_each([&](hypergraph::EdgeId eid) {
                            const hypergraph::Edge& edge = hg.get_edge(eid);
                            hash_edges.emplace_back(edge.vertices, edge.vertices + edge.arity);
                        });
                        hypergraph::IRCanonicalizer ir;
                        exact_hash = ir.compute_canonical_hash(hash_edges);
                    }
                    // Reinterpreted to int64 (bijective, so equality and grouping are
                    // preserved); a hash with the top bit set surfaces as a negative integer.
                    put_i64("CanonicalHash", static_cast<int64_t>(exact_hash));
                }
            }
            ++streamed_top_sections;
        }

        // Events -> Association[event_id -> event_data]
        // Only canonical events are sent (for graph vertices)
        // State IDs are mapped through get_canonical_state() so edges connect canonical states
        if (include_events) {
            // Send ALL events (not just canonical) - WL uses CanonicalId for vertex merging
            // This preserves event multiplicity: multiple events with same canonical ID
            // map to one vertex, but their edges to different output states are preserved.
            uint32_t num_raw_events = hg.num_raw_events();

            // First pass fixes the emitted event set so the association length is
            // known before streaming.
            std::vector<uint32_t> emit_eids;
            emit_eids.reserve(num_raw_events);
            for (uint32_t eid = 0; eid < num_raw_events; ++eid) {
                const hypergraph::Event& event = hg.get_event(eid);
                if (event.id == hypergraph::INVALID_ID) continue;
                if (!show_genesis_events && hg.is_genesis_event(eid)) continue;
                emit_eids.push_back(eid);
            }

            sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            sections.write(std::string("Events"));
            sections.write_byte(static_cast<uint8_t>(wxf::Token::Association));
            sections.write_varint(emit_eids.size());

            for (uint32_t eid : emit_eids) {
                const hypergraph::Event& event = hg.get_event(eid);

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

                // Association key: raw event id.
                sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                sections.write(static_cast<int64_t>(eid));

                sections.write_byte(static_cast<uint8_t>(wxf::Token::Association));
                sections.write_varint(9u);

                auto put_i64 = [&](const char* k, int64_t v) {
                    sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    sections.write(std::string(k));
                    sections.write(v);
                };
                put_i64("Id", static_cast<int64_t>(eid));
                put_i64("CanonicalId", canonical_event_id);
                put_i64("RuleIndex", static_cast<int64_t>(event.rule_index));
                put_i64("InputState", raw_input_state_id);
                put_i64("OutputState", raw_output_state_id);
                put_i64("CanonicalInputState", canonical_input_state_id);
                put_i64("CanonicalOutputState", canonical_output_state_id);

                // Consumed/produced edges as integer lists.
                sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                sections.write(std::string("ConsumedEdges"));
                sections.write_function("List", event.num_consumed);
                for (uint8_t i = 0; i < event.num_consumed; ++i)
                    sections.write(static_cast<int64_t>(event.consumed_edges[i]));

                sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                sections.write(std::string("ProducedEdges"));
                sections.write_function("List", event.num_produced);
                for (uint8_t i = 0; i < event.num_produced; ++i)
                    sections.write(static_cast<int64_t>(event.produced_edges[i]));
            }
            ++streamed_top_sections;
        }

        // EventsMinimal -> Association[event_id -> {Id, CanonicalId, RuleIndex, InputState, OutputState, CanonicalInputState, CanonicalOutputState}]
        // Reduced event data for graph structure variants that don't need full event details
        // Send ALL events - WL uses CanonicalId for vertex merging, RuleIndex for Event=Automatic grouping
        if (include_events_minimal && !include_events) {
            uint32_t num_raw_events = hg.num_raw_events();

            std::vector<uint32_t> emit_eids;
            emit_eids.reserve(num_raw_events);
            for (uint32_t eid = 0; eid < num_raw_events; ++eid) {
                const hypergraph::Event& event = hg.get_event(eid);
                if (event.id == hypergraph::INVALID_ID) continue;
                if (!show_genesis_events && hg.is_genesis_event(eid)) continue;
                emit_eids.push_back(eid);
            }

            sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            sections.write(std::string("Events"));
            sections.write_byte(static_cast<uint8_t>(wxf::Token::Association));
            sections.write_varint(emit_eids.size());

            for (uint32_t eid : emit_eids) {
                const hypergraph::Event& event = hg.get_event(eid);

                // Send BOTH raw and canonical state IDs - WL chooses which to use per graph type
                int64_t raw_input_state_id = static_cast<int64_t>(event.input_state);
                int64_t raw_output_state_id = static_cast<int64_t>(event.output_state);
                int64_t canonical_input_state_id = static_cast<int64_t>(hg.get_canonical_state(event.input_state));
                int64_t canonical_output_state_id = static_cast<int64_t>(hg.get_canonical_state(event.output_state));

                // Canonical event ID: for canonical events use own ID, for duplicates use the canonical's ID
                int64_t canonical_event_id = event.is_canonical()
                    ? static_cast<int64_t>(eid)
                    : static_cast<int64_t>(event.canonical_event_id);

                sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                sections.write(static_cast<int64_t>(eid));

                sections.write_byte(static_cast<uint8_t>(wxf::Token::Association));
                sections.write_varint(7u);

                auto put_i64 = [&](const char* k, int64_t v) {
                    sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                    sections.write(std::string(k));
                    sections.write(v);
                };
                put_i64("Id", static_cast<int64_t>(eid));
                put_i64("CanonicalId", canonical_event_id);
                put_i64("RuleIndex", static_cast<int64_t>(event.rule_index));
                put_i64("InputState", raw_input_state_id);
                put_i64("OutputState", raw_output_state_id);
                put_i64("CanonicalInputState", canonical_input_state_id);
                put_i64("CanonicalOutputState", canonical_output_state_id);
            }
            ++streamed_top_sections;
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
            // When CanonicalizeStates is Full, emits IR-canonicalized edges
            auto serialize_state_edges = [&](hypergraph::StateId sid) -> wxf::WXFValueList {
                wxf::WXFValueList edge_list;
                if (full_canonicalization) {
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

            // Adapt the engine to the shared graph marshaller (graph_marshal.hpp) so the CPU
            // and the GPU backend build byte-identical GraphData. The wrappers reuse the
            // effective-id and serialization lambdas above -- one graph-building code path.
            struct CpuGraphSource {
                const hypergraph::Hypergraph& hg;
                bool show_genesis;
                std::function<int64_t(hypergraph::StateId)> eff_state;
                std::function<int64_t(hypergraph::EventId)> eff_event;
                std::function<bool(hypergraph::EventId)> valid_event;
                std::function<wxf::WXFValueAssociation(hypergraph::StateId)> state_data;
                std::function<wxf::WXFValueAssociation(hypergraph::EventId)> event_data;

                uint32_t num_states() const { return hg.num_states(); }
                bool state_valid(uint32_t sid) const { return hg.get_state(sid).id != hypergraph::INVALID_ID; }
                int64_t effective_state_id(uint32_t sid) const { return eff_state(sid); }
                uint32_t state_step(uint32_t sid) const { return hg.get_state(sid).step; }
                wxf::WXFValueAssociation serialize_state_data(uint32_t sid) const { return state_data(sid); }
                uint32_t num_raw_events() const { return hg.num_raw_events(); }
                bool is_valid_event(uint32_t eid) const { return valid_event(eid); }
                int64_t effective_event_id(uint32_t eid) const { return eff_event(eid); }
                uint32_t event_input_state(uint32_t eid) const { return hg.get_event(eid).input_state; }
                uint32_t event_output_state(uint32_t eid) const { return hg.get_event(eid).output_state; }
                wxf::WXFValueAssociation serialize_event_data(uint32_t eid) const { return event_data(eid); }
                std::vector<std::pair<uint32_t, uint32_t>> causal_event_pairs() const {
                    std::vector<std::pair<uint32_t, uint32_t>> out;
                    for (const auto& ce : hg.causal_graph().get_causal_edges()) {
                        if (!show_genesis && (hg.is_genesis_event(ce.producer) || hg.is_genesis_event(ce.consumer))) continue;
                        out.emplace_back(ce.producer, ce.consumer);
                    }
                    return out;
                }
                std::vector<std::pair<uint32_t, uint32_t>> branchial_event_pairs() const {
                    std::vector<std::pair<uint32_t, uint32_t>> out;
                    for (const auto& be : hg.causal_graph().get_branchial_edges()) {
                        if (!show_genesis && (hg.is_genesis_event(be.event1) || hg.is_genesis_event(be.event2))) continue;
                        out.emplace_back(be.event1, be.event2);
                    }
                    return out;
                }
            };
            CpuGraphSource gsrc{hg, show_genesis_events,
                get_effective_state_id, get_effective_event_id, is_valid_event,
                serialize_state_data, serialize_event_data};
            hgmarshal::GraphOptions gopts;
            gopts.edge_deduplication = edge_deduplication;
            gopts.branchial_step = branchial_step;
            gopts.steps = steps;
            full_result.push_back({wxf::WXFValue("GraphData"),
                                   hgmarshal::build_graph_data(gsrc, graph_properties, gopts)});
        }

        // Only include counts when requested
        if (include_num_states) {
            full_result.push_back({wxf::WXFValue("NumStates"), wxf::WXFValue(static_cast<int64_t>(hg.num_canonical_states()))});
        }
        if (include_num_events) {
            // Under the reconstruction (quotient exploration, or Automatic identity on either
            // path) the observable event count is the reconstruction's -- the authority-anchored
            // identity count the golden matrix pins -- not the materialised dedup count.
            const int64_t n_events = hg.quotient_reconstruction()
                ? static_cast<int64_t>(hg.observable_num_events())
                : static_cast<int64_t>(engine.num_events());
            full_result.push_back({wxf::WXFValue("NumEvents"), wxf::WXFValue(n_events)});
        }
        if (include_num_causal_edges) {
            // Count unique (producer, consumer) event pairs for v1 semantics
            // When show_genesis_events is false, we must filter out pairs involving genesis events
            // to match reference behavior ("IncludeInitialEvent" -> False)
            //
            // Reconstruction branch first: its pairs live in its own store over its own event
            // ids, which never include genesis events (INIT is a sentinel producer, dropped at
            // emission), so the genesis filter below -- which indexes HYPERGRAPH events and
            // would misread a reconstruction id -- must not run on them.
            int64_t causal_count;
            if (hg.quotient_reconstruction()) {
                causal_count = static_cast<int64_t>(hg.observable_num_causal_pairs(
                    hg.causal_graph().transitive_reduction_enabled()));
            } else if (show_genesis_events) {
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
            const int64_t n_branchial = hg.quotient_reconstruction()
                ? static_cast<int64_t>(hg.observable_num_branchial())
                : static_cast<int64_t>(hg.num_branchial_edges());
            full_result.push_back({wxf::WXFValue("NumBranchialEdges"), wxf::WXFValue(n_branchial)});
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

        // Warning trail (engine warnings + analysis refusals), same schema as the GPU backend.
        if (!ffi_warnings.empty()) {
            wxf::WXFValueList warn;
            for (const auto& w : ffi_warnings) {
                wxf::WXFValueAssociation wa;
                wa.push_back({wxf::WXFValue("Kind"), wxf::WXFValue(w.kind)});
                wa.push_back({wxf::WXFValue("Count"), wxf::WXFValue(w.count)});
                wa.push_back({wxf::WXFValue("Context"), wxf::WXFValue(w.context)});
                warn.push_back(wxf::WXFValue(wa));
            }
            full_result.push_back({wxf::WXFValue("Warnings"), wxf::WXFValue(warn)});
        }

        // Write the final top-level association: the streamed States/Events section
        // blobs spliced ahead of the assembled remaining pairs. An Association is a
        // flat concatenation of key->value entries, so emitting the header with the
        // combined count, appending the pre-serialized section bytes, then writing
        // each full_result pair yields the same byte stream as one combined value
        // tree — without materializing the O(states+events) tree.
        wxf_writer.reserve(1024 + sections.size()
                           + 128 * static_cast<std::size_t>(full_result.size()));
        wxf_writer.write_byte(static_cast<uint8_t>(wxf::Token::Association));
        wxf_writer.write_varint(streamed_top_sections + full_result.size());
        wxf_writer.append(sections.data());
        for (const auto& [key, value] : full_result) {
            wxf_writer.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
            wxf_writer.write(key);
            wxf_writer.write(value);
        }
        std::vector<uint8_t> wxf_data = wxf_writer.release_data();

        // Wire-size instrument: per-top-level-key serialized bytes, written to the file
        // HG_FFI_PAYLOAD_FILE names. Each value is re-serialized into a throwaway writer, so
        // the cost exists only when the instrument is on; the streamed States/Events sections
        // are reported as one line (they never pass through full_result).
        if (const char* payload_file = std::getenv("HG_FFI_PAYLOAD_FILE")) {
            if (std::FILE* pf = std::fopen(payload_file, "w")) {
                std::fprintf(pf, "TOTAL\t%zu\n", wxf_data.size());
                std::fprintf(pf, "States+Events(streamed)\t%zu\n", sections.size());
                for (const auto& [key, value] : full_result) {
                    wxf::Writer wv;
                    wv.write(value);
                    const std::string name =
                        key.holds<std::string>() ? key.get<std::string>() : std::string("?");
                    std::fprintf(pf, "%s\t%zu\n", name.c_str(), wv.release_data().size());
                }
                std::fclose(pf);
            }
        }

        if (show_progress) {
            core_progress(host, "HGEvolve: Serialization complete.");
        }

        return wxf_data;

    } catch (const wxf::TypeError& e) {
        throw std::runtime_error(std::string("WXF TypeError: ") + e.what());
    }
}

#ifndef HG_STANDALONE_BINARY

// LibraryLink fallback adapter: unwrap the WXF ByteArray argument, run the core,
// wrap the result ByteArray. The standalone binary is the primary path; this is
// an in-process fallback with no progress or abort (abort is a process kill).
EXTERN_C DLLEXPORT int performRewriting(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    if (argc != 1) {
        handle_error(libData, "performRewriting expects 1 argument: WXF ByteArray data");
        return LIBRARY_FUNCTION_ERROR;
    }
    try {
        MNumericArray wxf_array = MArgument_getMNumericArray(argv[0]);
        mint rank = libData->numericarrayLibraryFunctions->MNumericArray_getRank(wxf_array);
        if (rank != 1) {
            handle_error(libData, "WXF ByteArray must be 1-dimensional");
            return LIBRARY_FUNCTION_ERROR;
        }
        const mint* dims = libData->numericarrayLibraryFunctions->MNumericArray_getDimensions(wxf_array);
        mint wxf_size = dims[0];
        void* raw_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(wxf_array);
        const uint8_t* wxf_byte_data = static_cast<const uint8_t*>(raw_data);
        std::vector<uint8_t> wxf_bytes(wxf_byte_data, wxf_byte_data + wxf_size);

        HostBridge host;

        std::vector<uint8_t> out = run_rewriting_core(wxf_bytes, host);

        mint out_dims[1] = {static_cast<mint>(out.size())};
        MNumericArray result_array;
        int err = libData->numericarrayLibraryFunctions->MNumericArray_new(MNumericArray_Type_UBit8, 1, out_dims, &result_array);
        if (err != LIBRARY_NO_ERROR) {
            return err;
        }
        void* result_data = libData->numericarrayLibraryFunctions->MNumericArray_getData(result_array);
        std::memcpy(result_data, out.data(), out.size());
        MArgument_setMNumericArray(res, result_array);
        return LIBRARY_NO_ERROR;

    } catch (const std::exception& e) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "HGEvolve error: %.200s", e.what());
        handle_error(libData, err_msg);
        return LIBRARY_FUNCTION_ERROR;
    }
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData /* libData */) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData /* libData */) {
}

#endif  // HG_STANDALONE_BINARY