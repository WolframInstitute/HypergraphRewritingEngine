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
#include "ffi_job.hpp"           // ParsedJob -- the envelope, parsed once
#include "cpu_engine_holder.hpp"   // owns the Hypergraph and its engine as one lifetime
#include "build_stamp.hpp"         // the commit this artifact was built from

using namespace hypergraph;

// The build stamp lives HERE because this translation unit is compiled into all three shipped
// artifacts -- the paclet library, hg_evolve and hg_evolve_gpu -- so one definition stamps every
// one of them, and a fourth artifact could not be added without also compiling this file.
namespace HG_NAMESPACE {
namespace ffi {
const char kBuildStamp[] =
    "HGBUILDSTAMP/1 commit=" HG_BUILD_COMMIT " variant=" HG_BUILD_VARIANT " :HGBUILDSTAMP";
}  // namespace ffi
}  // namespace HG_NAMESPACE


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
// The worker's one session (D7). Function-local so it lives exactly as long as the process
// serving jobs -- which is what `--serve` and `--serve-socket` give it. A one-shot invocation
// opens and closes within its single call or not at all, so the same code serves both.
static hgffi::SessionSlot& worker_session() {
    static hgffi::SessionSlot slot;
    return slot;
}

// The envelope, parsed. Split out of run_rewriting_core because it is the one phase whose
// dependency runs one way -- it writes ParsedJob and reads nothing a later phase produces.
// Everything after it in that function reads the same fields, which is why the op boundary
// (#12) had to be drawn by binding references rather than by extracting a second function.
static void parse_job(const std::vector<uint8_t>& wxf_bytes, const HostBridge& host,
                      hgffi::ParsedJob& req) {
        // Parse WXF input
        wxf::Parser parser(wxf_bytes);
        parser.skip_header();

        // Parse main association
        parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
            if (key == "InitialStates") {
                req.initial_states_raw = value_parser.read<std::vector<std::vector<std::vector<int64_t>>>>();
            }
            else if (key == "InitialEdges") {
                // Legacy single-state format
                auto edges_data = value_parser.read<std::vector<std::vector<int64_t>>>();
                req.initial_states_raw.push_back(edges_data);
            }
            else if (key == "Rules") {
                req.parsed_rules_raw = ffi_helpers::read_rules_association(value_parser);
            }
            else if (key == "Steps") {
                req.steps = value_parser.read<int>();
                // Every budget check in the engine is `step > max_steps_`, and this is cast to
                // size_t downstream -- so a negative value becomes SIZE_MAX and no check ever
                // fires. The run then ends only by exhausting the SegmentedArray ceiling,
                // after multi-GB of arena, as a hard failure.
                if (req.steps < 0) {
                    throw std::runtime_error("Steps must be non-negative, got " +
                                             std::to_string(req.steps));
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
                            req.max_successor_states_per_parent = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "MaxStatesPerStep") {
                            req.max_states_per_step = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "MatchesPerStep") {
                            req.matches_per_step = static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "MatchesPerStateRule") {
                            req.matches_per_state_rule =
                                static_cast<size_t>(option_parser.read<int64_t>());
                        } else if (option_key == "RandomSeed") {
                            req.random_seed = static_cast<uint64_t>(option_parser.read<int64_t>());
                        } else if (option_key == "ExplorationProbability") {
                            req.exploration_probability = option_parser.read<double>();
                        } else if (option_key == "TransitionRate") {
                            req.transition_rate = option_parser.read<double>();
                        } else if (option_key == "RuleWeights") {
                            req.rule_weights = option_parser.read<std::vector<double>>();
                        } else if (option_key == "BranchialStep") {
                            // 0=All, positive=1-based step index, negative=from end (-1=final)
                            req.branchial_step = static_cast<int>(option_parser.read<int64_t>());
                        } else if (option_key == "EdgeDeduplication") {
                            std::string symbol = option_parser.read<std::string>();
                            req.edge_deduplication = (symbol == "True");
                        } else if (option_key == "IncludeCanonicalHashes") {
                            std::string symbol = option_parser.read<std::string>();
                            req.include_canonical_hashes = (symbol == "True");
                        } else if (option_key == "RequestedData") {
                            // Parse list of data component names
                            // When specified, only include requested components
                            req.include_states = false;
                            req.include_events = false;
                            req.include_events_minimal = false;
                            req.include_causal_edges = false;
                            req.include_branchial_edges = false;
                            req.include_branchial_state_edges = false;

                            // Reset all include flags when RequestedData is specified
                            req.include_num_states = false;
                            req.include_num_events = false;
                            req.include_num_causal_edges = false;
                            req.include_num_branchial_edges = false;

                            auto components = option_parser.read<std::vector<std::string>>();
                            for (const auto& comp : components) {
                                if (comp == "States") req.include_states = true;
                                else if (comp == "Events") req.include_events = true;
                                else if (comp == "EventsMinimal") req.include_events_minimal = true;
                                else if (comp == "CausalEdges") req.include_causal_edges = true;
                                else if (comp == "BranchialEdges") req.include_branchial_edges = true;
                                else if (comp == "BranchialStateEdges") req.include_branchial_state_edges = true;
                                else if (comp == "BranchialStateEdgesAllSiblings") req.include_branchial_state_edges_all_siblings = true;
                                else if (comp == "NumStates") req.include_num_states = true;
                                else if (comp == "NumEvents") req.include_num_events = true;
                                else if (comp == "NumCausalEdges") req.include_num_causal_edges = true;
                                else if (comp == "NumBranchialEdges") req.include_num_branchial_edges = true;
                                else if (comp == "GlobalEdges") req.include_global_edges = true;
                                else if (comp == "StateBitvectors") req.include_state_bitvectors = true;
                            }
                        } else if (option_key == "CanonicalizeEvents") {
                            // Can be: None, Full, Automatic (symbols), or {"InputState", "OutputState", ...} (list)
                            try {
                                // Try to read as list first. A failure mid-list (the header
                                // parses, an element does not) leaves the cursor inside the
                                // list, so the symbol fallback re-aligns to the value start.
                                auto keys = option_parser.read<std::vector<std::string>>();
                                req.event_signature_keys = hypergraph::EVENT_SIG_NONE;
                                for (const auto& sig_key : keys) {
                                    if (sig_key == "InputState") req.event_signature_keys |= hypergraph::EventKey_InputState;
                                    else if (sig_key == "OutputState") req.event_signature_keys |= hypergraph::EventKey_OutputState;
                                    else if (sig_key == "Step") req.event_signature_keys |= hypergraph::EventKey_Step;
                                    else if (sig_key == "Rule") req.event_signature_keys |= hypergraph::EventKey_Rule;
                                    else if (sig_key == "ConsumedEdges") req.event_signature_keys |= hypergraph::EventKey_ConsumedEdges;
                                    else if (sig_key == "ProducedEdges") req.event_signature_keys |= hypergraph::EventKey_ProducedEdges;
                                }
                            } catch (...) {
                                // Read as symbol, from the START of the value.
                                option_parser.seek(option_value_start);
                                std::string symbol = option_parser.read<std::string>();
                                if (symbol == "None") {
                                    req.event_signature_keys = hypergraph::EVENT_SIG_NONE;
                                } else if (symbol == "Full") {
                                    req.event_signature_keys = hypergraph::EVENT_SIG_FULL;
                                } else if (symbol == "Automatic") {
                                    req.event_signature_keys = hypergraph::EVENT_SIG_AUTOMATIC;
                                } else if (symbol == "Positional") {
                                    // Same key set as Automatic; ranks read from each raw
                                    // state's own labelling rather than the class frame.
                                    req.event_signature_keys = hypergraph::EVENT_SIG_AUTOMATIC;
                                    req.positional_event_identity = true;
                                }
                                // else keep default (None)
                            }
                        } else if (option_key == "CanonicalizeStates") {
                            // None, Automatic or Full (legacy False/True). Each names the identity
                            // the EVOLUTION deduplicates states by, so each is answered by the
                            // engine as it runs: None gives every state its own id, Automatic
                            // deduplicates by content, Full by the exact canonical form. A mode
                            // that only regrouped the finished output would make the identity a
                            // property of how results are read rather than of the run.
                            std::string symbol = option_parser.read<std::string>();
                            if (symbol == "None" || symbol == "False") {
                                req.state_canon_mode = hypergraph::StateCanonicalizationMode::None;
                                req.canonicalize_states_mode = "None";
                            } else if (symbol == "Automatic") {
                                req.state_canon_mode =
                                    hypergraph::StateCanonicalizationMode::Automatic;
                                req.canonicalize_states_mode = "Automatic";
                            } else if (symbol == "Full" || symbol == "True") {
                                req.state_canon_mode = hypergraph::StateCanonicalizationMode::Full;
                                req.canonicalize_states_mode = "Full";
                            }
                        } else if (option_key == "GraphProperties") {
                            // Graph properties for graph-ready data output (list)
                            req.graph_properties = option_parser.read<std::vector<std::string>>();
                        } else {
                            std::string symbol = option_parser.read<std::string>();
                            bool value = (symbol == "True");

                            if (option_key == "ShowGenesisEvents") {
                                req.show_genesis_events = value;
                            } else if (option_key == "ShowProgress") {
                                req.show_progress = value;
                            } else if (option_key == "CausalTransitiveReduction") {
                                req.causal_transitive_reduction = value;
                            } else if (option_key == "QuotientInitialStates") {
                                req.quotient_initial_states = value;
                            } else if (option_key == "ExploreFromCanonicalStatesOnly") {
                                // Exploration deduplication: only explore from canonical states
                                // Requires CanonicalizeStates -> Full to have any effect
                                req.explore_from_canonical_states_only = value;
                            } else if (option_key == "UniformRandom") {
                                // With MatchesPerStep, a per-step cap by arrival order
                                req.uniform_random = value;
                            } else {
                                // AN OPTION THIS BUILD DOES NOT KNOW IS NAMED, NOT DROPPED.
                                // Every branch above matches a specific key, so reaching here
                                // means the key matched none of them and its value happened to
                                // parse as a symbol -- which is what a misspelling looks like.
                                // Silently continuing is how a documented-but-nonexistent option
                                // produced no diagnostic at all: the caller sees a successful
                                // evolution computed without the option they asked for.
                                //
                                // A WARNING, NOT AN ERROR. A malformed value is already reported
                                // this way, and a WL caller may pass a forward-compatible option
                                // set an older engine does not know; refusing the whole evolution
                                // over one unrecognised key would break those callers for no
                                // correctness gain. The warning trail reaches the user through
                                // HGEvolve::warn, so the mistake is visible either way.
                                req.ffi_warnings.push_back(
                                    {"OptionSkipped", 1,
                                     "option '" + option_key + "' is not recognised by this "
                                     "engine build and was ignored; check its spelling against "
                                     "the documented option list"});
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
                        req.ffi_warnings.push_back(
                            {"OptionSkipped", 1,
                             "option '" + option_key + "' ignored: " + e.what()});
                        core_progress(host,
                            "FFI: skipping malformed option '" + option_key + "': " + e.what());
                    }
                });

                // Handle numeric options that may have been parsed as integers
                // (these need special handling since they come after the bool options)
            }
            // The session envelope. A job that names no `Op` is an `Evolve` job, which is the
            // whole of today's protocol -- so an existing caller's bytes take exactly the path
            // they took before, and the compatibility guarantee is a property of the parser
            // rather than of a later branch remembering to preserve it.
            //
            // `Session` is the opaque handle: a per-worker counter, 0 reserved for "no session",
            // so no client can construct one and none is confused with absence.
            else if (key == "Op") {
                req.session_op = value_parser.read<std::string>();
            }
            else if (key == "Session") {
                req.session_handle = static_cast<uint64_t>(value_parser.read<int64_t>());
            }
            // The frontier states a `Step` expands, by effective id. Absent means all of them.
            else if (key == "From") {
                req.session_from = value_parser.read<std::vector<int64_t>>();
            }
            // "Delta" sends only what this session has not been sent; anything else is a full
            // delivery, which also resets the record.
            else if (key == "Delivery") {
                req.delivery_delta = (value_parser.read<std::string>() == "Delta");
            }
            else {
                value_parser.skip_value();
            }
        });
}

#ifdef HG_GPU_BACKEND
// The GPU binary's whole job: translate a ParsedJob into a GpuJob, run hg_gpu::evolve, and
// marshal the result through the same WXF path the CPU uses.
//
// A phase like the parse: it reads the job and produces the reply, and touches no engine
// this file owns. Extracted for that reason -- run_rewriting_core's remaining length is the
// CPU path, and this block was never part of it.
// `req` is not const: the device has no implementation for the per-step caps and appends an
// OptionSkipped warning for each, so the job it was handed carries what it did not apply.
static std::vector<uint8_t> run_gpu_job(hgffi::ParsedJob& req, const HostBridge& host) {
        // Sessions are served on the device too. What used to make that impossible -- the
        // evolver rebuilding its graph from `initial_states` every call -- no longer holds:
        // SessionState carries the identity maps and the budget's frontier across calls, and
        // run_session refuses to rebuild the engine rather than silently continuing against a
        // fresh one. The verb rides on the job and the backend answers it.

        // The per-(state, rule) cap is applied at a state's drain, and the device has no drain
        // to apply it at: EvolveInput carries no matches_per_state_rule, and the GpuJob built
        // below has no field for it. Same class as the two above -- a cap the caller asked for
        // and did not get -- and it is the one the documentation names as the reproducible
        // alternative to them, so a caller steered there by that text is exactly who runs into
        // this on a device job.
        if (req.matches_per_state_rule > 0) {
            req.ffi_warnings.push_back(
                {"OptionSkipped", 1,
                 "'MatchesPerStateRule' is not implemented on the GPU; the result is "
                 "uncapped."});
        }
        // The device has no genesis events to show or hide. On the host a genesis event
        // connects the genesis state to an initial state, and this option decides whether the
        // causal and branchial output carries the pairs those events take part in; the GPU
        // engine never mints one, so the option selects between two identical answers there
        // while selecting between two different ones here.
        if (req.show_genesis_events) {
            req.ffi_warnings.push_back(
                {"OptionSkipped", 1,
                 "'ShowGenesisEvents' has no effect on the GPU; the device creates no "
                 "genesis events."});
        }
        // The device thins states through `exploration_probability` and has no per-transition
        // draw, so it has no spine either. Silently running unthinned would return a FULL
        // evolution where the caller asked for a sample, which reads as a system with that
        // many states rather than as an option that did not apply.
                
        GpuJob job{
            req.parsed_rules_raw,
            req.initial_states_raw,
            req.steps,
            // 0 None, 1 Full, 2 Automatic -- GpuJob::event_canon_mode's own order, which is
            // NOT the state order below and is not the enum order either. Collapsing this to
            // "0 if None else 1" sent code 1 for an AUTOMATIC request, and the backend reads
            // 1 as FULL: the caller silently got a coarser event identity than asked for, and
            // code 2 was never sent at all.
            (req.event_signature_keys == hypergraph::EVENT_SIG_NONE)      ? GpuJob::EventCanonCode::kNone :
            (req.event_signature_keys == hypergraph::EVENT_SIG_AUTOMATIC) ? GpuJob::EventCanonCode::kAutomatic
                                                                      : GpuJob::EventCanonCode::kFull,
            // 0 None, 1 Automatic, 2 Full (hg_gpu::CanonicalizationMode order)
            (req.state_canon_mode == hypergraph::StateCanonicalizationMode::Full)      ? GpuJob::StateCanonCode::kFull :
            (req.state_canon_mode == hypergraph::StateCanonicalizationMode::Automatic) ? GpuJob::StateCanonCode::kAutomatic
                                                                                   : GpuJob::StateCanonCode::kNone,
            req.causal_transitive_reduction,
            req.explore_from_canonical_states_only,
            req.quotient_initial_states,
            req.exploration_probability,
            req.random_seed,
            0,  // max_device_memory_bytes: default (90% VRAM) resolved by the GPU engine
            req.transition_rate,
            req.rule_weights,
            req.max_states_per_step,
            req.max_successor_states_per_parent,
            req.matches_per_state_rule,
            req.include_states,
            req.include_events || req.include_events_minimal,
            req.include_events_minimal && !req.include_events,
            req.include_causal_edges,
            req.include_branchial_edges,
            req.include_canonical_hashes,
            req.include_num_states,
            req.include_num_events,
            req.include_num_causal_edges,
            req.include_num_branchial_edges,
            req.include_branchial_state_edges,
            req.include_branchial_state_edges_all_siblings,
            req.include_global_edges,
            req.include_state_bitvectors,
            req.graph_properties,
            req.edge_deduplication,
            req.branchial_step,
            req.show_genesis_events,
            req.session_op,
            req.session_handle,
            req.session_from,
        };
        if (req.show_progress) {
            core_progress(host, "HGEvolve: Starting GPU evolution...");
        }
        std::vector<uint8_t> out = run_gpu_evolution(job, host);
        if (req.show_progress) {
            core_progress(host, "HGEvolve: GPU evolution complete.");
        }
        return out;
}
#endif  // HG_GPU_BACKEND

// Everything a FRESH run needs before it can be serialized: the identity and recording
// configuration, the rules, the initial states, and the evolution itself.
//
// One-way, like the parse and the GPU job: it reads the parsed job and writes the engine,
// and reads nothing the serialization below produces. A HELD session (Step/Query) skips it
// entirely -- its engine already has all of this, which is what makes the op boundary a
// question of where the engine comes from rather than where serialization begins.
static void configure_and_evolve(hgffi::ParsedJob& req, hypergraph::Hypergraph& hg,
                                 hypergraph::ParallelEvolutionEngine& engine,
                                 const hypergraph::RecordSet& record,
                                 const HostBridge& host) {

    // Configure event canonicalization
    hg.set_event_signature_keys(req.event_signature_keys);
    hg.set_positional_event_identity(req.positional_event_identity);

    // Configure state canonicalization mode
    hg.set_state_canonicalization_mode(req.state_canon_mode);

    hg.set_record_set(record);

    // Configure engine options
    engine.set_max_steps(static_cast<size_t>(req.steps));
    engine.set_transitive_reduction(req.causal_transitive_reduction);
    engine.set_exploration_probability(req.exploration_probability);
    // Per-transition thinning, with the spine that keeps a sparse sample reaching full depth.
    // ExplorationProbability thins states and has no spine, so the two are not interchangeable.
    engine.set_transition_rate(req.transition_rate);
    // Per-rule multipliers on that rate. Composes with it rather than replacing it, so a rate
    // of 1 with one rule weighted to 0 still samples.
    engine.set_rule_weights(req.rule_weights);
    // 0 keeps the engine's default -- a fresh seed per run. Nonzero is what makes the
    // sampling draws reproducible, which is the whole content of the option.
    engine.set_random_seed(req.random_seed);
    engine.set_max_successor_states_per_parent(req.max_successor_states_per_parent);
    engine.set_max_states_per_step(req.max_states_per_step);
    engine.set_matches_per_state_rule(req.matches_per_state_rule);
    engine.set_genesis_events(req.show_genesis_events);
    engine.set_explore_from_canonical_states_only(req.explore_from_canonical_states_only);
    engine.set_quotient_initial_states(req.quotient_initial_states);

    // Convert rules to unified format
    uint16_t rule_index = 0;
    for (const auto& [rule_name, rule_data] : req.parsed_rules_raw) {
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

    for (const auto& state_raw : req.initial_states_raw) {
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
    if (req.show_progress) {
        core_progress(host, "HGEvolve: Starting evolution...");
    }
    auto evolution_start = std::chrono::steady_clock::now();

    // MatchesPerStep is a per-DEPTH count, and a count over a depth cannot be sampled
    // without a barrier. What the step-synchronised path actually did with it was stop
    // applying once that many states existed for the step -- a cap by arrival order, which
    // MaxStatesPerStep already delivers with no barrier at all. So it maps to the cap it
    // always was, and the uniformity it used to claim moves to TransitionRate, which is a
    // rate and needs no depth to be defined over.
    if (req.uniform_random && req.matches_per_step > 0) {
        engine.set_max_states_per_step(req.matches_per_step);
    }
    engine.evolve(initial_states, static_cast<size_t>(req.steps));

    if (req.show_progress) {
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
}

// A SESSION'S IDENTITY IS FIXED WHEN IT OPENS, and these four are read back from the
// engine rather than from the job. They decide what a state and an event ARE, so serving
// a `Query` under the identity its own envelope happened to carry would describe a graph
// built under one convention using another: `Full` at `Open` and nothing at `Query` would
// report exact canonical forms as tree-mode ones, in fields that all still parse.
//
// `Steps` is the accumulated depth, which `evolve_more` raises. A step counted from the
// end is defined against that total, not against the increment a `Step` just asked for.
static void read_back_session_identity(hgffi::ParsedJob& req, const hypergraph::Hypergraph& hg,
                                const hypergraph::ParallelEvolutionEngine& engine) {
    req.state_canon_mode = hg.state_canonicalization_mode();
    req.canonicalize_states_mode =
        (req.state_canon_mode == hypergraph::StateCanonicalizationMode::Full)      ? "Full" :
        (req.state_canon_mode == hypergraph::StateCanonicalizationMode::Automatic) ? "Automatic"
                                                                              : "None";
    req.event_signature_keys = hg.event_signature_keys();
    req.positional_event_identity = hg.positional_event_identity();
    req.show_genesis_events = engine.genesis_events();
}

std::vector<uint8_t> run_rewriting_core(const std::vector<uint8_t>& wxf_bytes,
                                        const HostBridge& host) {
    try {
        hgffi::ParsedJob req;
        parse_job(wxf_bytes, host, req);


        // Close needs nothing else from the job: it names a handle and releases what that handle
        // holds. Answered before the rules are checked, because a Close carries no rules and
        // demanding them would make releasing a session harder than opening one.
        if (req.session_op == "Close") {
            // ANSWERED BEFORE THE RULES ARE CHECKED, on either device: a Close carries no
            // rules, and demanding them would make releasing a session harder than opening
            // one. Which SESSION it releases is the only difference -- in the GPU binary the
            // handle names a device session, and closing a CPU slot this binary never
            // populated would leave the device holding its engine forever. The
            // process-boundary gate caught exactly that: Close was the one verb that errored.
#ifdef HG_GPU_BACKEND
            return run_gpu_job(req, host);
#else
            worker_session().close(req.session_handle);
            return hgmarshal::session_ack(hgffi::SessionSlot::kNoSession);
#endif
        }

        // WHERE THE ENGINE COMES FROM is the whole of the op boundary. `Step` and `Query` answer
        // about an engine the session already holds; `Evolve` and `Open` build one from the job.
        // Everything after the acquisition below -- the content index, the artifact gating and
        // the entire serialization -- reads `hg` and `engine` and cannot tell which produced
        // them, so the four verbs share ONE serializer rather than growing a second one that
        // would be right on the day it was written.
        const bool opening_session = (req.session_op == "Open");
        const bool held_session = (req.session_op == "Step" || req.session_op == "Query");
        if (!req.session_op.empty() && !opening_session && !held_session && req.session_op != "Evolve") {
            throw std::runtime_error(
                "Op '" + req.session_op + "' is not a verb; 'Evolve', 'Open', 'Step', 'Query' and "
                "'Close' are");
        }

        // A held session already has its rules; sending them again would be sending rules the
        // job cannot apply, since the engine's rule set was fixed when it opened.
        if (!held_session && req.parsed_rules_raw.empty()) {
            throw std::runtime_error("No valid rules found");
        }
        if (held_session && !req.parsed_rules_raw.empty()) {
            throw std::runtime_error(
                "Op '" + req.session_op + "' carries rules, but a session's rule set was fixed when "
                "it opened; applying these would answer about a system the session is not "
                "exploring");
        }

#ifdef HG_GPU_BACKEND
        // The GPU binary answers the whole job on the device; nothing below runs.
        return run_gpu_job(req, host);
#endif

        // The graph and its engine, owned together by a holder rather than as two locals.
        // ParallelEvolutionEngine holds a POINTER to its Hypergraph, so the pair has one
        // lifetime and cannot be handed anywhere by value; heap-allocating the holder is what
        // lets a session outlive this call and keep the graph's address stable. `hg` and
        // `engine` below are references to what the holder owns, so everything that reads them
        // reads the same objects it always did.
        //
        // Continuable only for a session. The frontier a continuation resumes from costs about
        // 12.5 MB across the oracle corpus and 3.9% of the arena, and a one-shot job never
        // resumes -- so `Evolve` does not pay for it and `Open` does, which is the only
        // difference between the two paths.
        //
        // `Step` and `Query` take the holder the session already owns and build nothing. The
        // cast is checked rather than assumed: when the device holder arrives (#121) a handle
        // opened on the GPU would otherwise be read here as a CPU one and answered from the
        // wrong engine's memory.
        std::unique_ptr<hgffi::CpuEngineHolder> engine_holder;
        hgffi::CpuEngineHolder* holder = nullptr;
        if (held_session) {
            holder = dynamic_cast<hgffi::CpuEngineHolder*>(
                &worker_session().engine(req.session_handle));
            if (!holder)
                throw std::runtime_error(
                    "session " + std::to_string(req.session_handle) +
                    " is not held by this binary's engine");
        } else {
            engine_holder =
                std::make_unique<hgffi::CpuEngineHolder>(/*continuable=*/opening_session);
            holder = engine_holder.get();
        }
        hypergraph::Hypergraph& hg = holder->hypergraph();
        hypergraph::ParallelEvolutionEngine& engine = holder->engine();

        // D16: a held verb takes its identity from the SESSION, not from its own envelope.
        if (held_session) read_back_session_identity(req, hg, engine);

        // The hot-path state hash is always Weisfeiler-Leman; exact IR
        // canonicalization is selected via CanonicalizeStates -> Full.

        // Full canonicalization mode: IR-based dedup, exact edge correspondence, canonical output
        const bool full_canonicalization = (req.state_canon_mode == hypergraph::StateCanonicalizationMode::Full);

        // What this call must RECORD, derived from what it will return. An artifact nothing asked
        // for is not built at all. Every consumer is named here and nowhere else:
        //
        //   causal       CausalEdges, NumCausalEdges, any graph property built from the causal
        //                relation, and the progress line's pair count
        //   branchial    BranchialEdges, NumBranchialEdges, BranchialStateEdges (which reads the
        //                pair relation), the branchial graph properties, the progress line
        //   state events BranchialStateEdgesAllSiblings alone -- it pairs every two output
        //                states of one input state and never consults the pair relation
        //
        // Which graph properties need which comes from graph_property_needs, the same test the
        // marshaller builds them with, so the two cannot disagree about a name. The serialization
        // reads `gneeds` too, so it is derived once for every op rather than per path.
        const hgmarshal::GraphPropertyNeeds gneeds = hgmarshal::graph_property_needs(req.graph_properties);
        hypergraph::RecordSet record;
        record.causal = req.include_causal_edges || req.include_num_causal_edges || gneeds.causal ||
                        req.show_progress;
        record.branchial = req.include_branchial_edges || req.include_num_branchial_edges ||
                           req.include_branchial_state_edges || gneeds.branchial || req.show_progress;
        record.state_events = req.include_branchial_state_edges_all_siblings;
        // THE RAW UNFOLDING, which under quotient exploration is the reconstruction and the
        // engine's largest single cost -- 99.57% of all cycles on multirule at depth 6, growing
        // 14.6x per depth step while the canonical answer grows 1.17x. It defaults ON in
        // RecordSet so a caller that states nothing keeps the counts it always had; here the
        // caller HAS stated something, so it is derived like the other three rather than left at
        // the default. On for a request the raw set answers: the event records themselves, or a
        // count taken over them.
        //
        // The causal and branchial relations do not need it named: under quotient they are
        // reconstructed too, and record.causal / record.branchial already drive the replay.
        record.raw_events = req.include_events || req.include_events_minimal ||
                            req.include_num_events || req.show_progress;

        // A SESSION RECORDS EVERYTHING, because it exists to be continued and queried in ways
        // its Open cannot know. Deriving its record set from the properties named on the Open
        // call makes the answer to a later Query depend on what the FIRST call happened to ask
        // for: open for "States", ask for the causal graph three steps later, and the relation
        // comes back empty because the evolution that would have built it has already run. A
        // continuation must not depend on the order the caller asked things in.
        //
        // One-shot calls keep the derived set above, which is where the saving is -- the raw
        // unfolding alone is 25x on multirule at depth 6 -- and they have no later query to
        // serve.
        if (opening_session) {
            record.causal = record.branchial = record.state_events = record.raw_events = true;
        }

        if (!held_session) configure_and_evolve(req, hg, engine, record, host);

        // Content identity of every state: the hash the engine deduplicates on under Automatic,
        // and the first state carrying each. TWO output sections need this same map, and it is a
        // pass over every state, so it is built at most once and only if something asks.
        //
        // The hash comes from the library's get_state_content_hash -- the same function the
        // evolution deduplicates with -- so the ContentStateId a caller reads back is the
        // grouping the run actually used, not a second opinion about it.
        struct ContentIndex {
            std::vector<uint64_t> hash_of;                                  // by raw state id
            std::unordered_map<uint64_t, hypergraph::StateId> first_with;
        };
        ContentIndex content_index_storage;
        bool content_index_built = false;
        auto content_index = [&]() -> const ContentIndex& {
            if (!content_index_built) {
                // Sized by the CLAIM count because any live state id indexes this table, but
                // ITERATED to the published count: an id can be claimed and never emplaced, and
                // get_state on such an index throws rather than returning an invalid state.
                const uint32_t n = hg.num_states();
                const uint32_t n_pub = hg.num_published_states();
                content_index_storage.hash_of.assign(n, 0);
                content_index_storage.first_with.reserve(n_pub);
                for (uint32_t sid = 0; sid < n_pub; ++sid) {
                    if (hg.get_state(sid).id == hypergraph::INVALID_ID) continue;
                    const uint64_t h = hg.get_state_content_hash(sid);
                    content_index_storage.hash_of[sid] = h;
                    content_index_storage.first_with.emplace(h, sid);
                }
                content_index_built = true;
            }
            return content_index_storage;
        };

        // WHAT A STATE'S ID IS, in one place. Both the subset Step below and the whole
        // serialisation ask this, and a second body would answer differently the moment one was
        // edited -- the defect this project removed from its canonicalizer, its matcher and its
        // event identity. Under `Full` the engine's canonical id; under `Automatic` the first
        // raw state carrying the same content hash; otherwise the raw id itself.
        auto get_effective_state_id = [&](hypergraph::StateId sid) -> int64_t {
            if (req.canonicalize_states_mode == "Full")
                return static_cast<int64_t>(hg.get_canonical_state(sid));
            if (req.canonicalize_states_mode == "Automatic") {
                const ContentIndex& ci = content_index();
                return static_cast<int64_t>(ci.first_with.at(ci.hash_of[sid]));
            }
            return static_cast<int64_t>(sid);
        };

        // A `Step` carries the exploration further from the frontier the budget stopped it
        // at, keeping every state, event and relation already built and the raw ids that
        // name them. `Query` extends by nothing and reports what the session holds.
        //
        // D14: a throw here leaves an engine whose run latched an error, and the session's
        // accumulated exploration with it. The handle stays addressable so the next verb on
        // it says so, rather than being served from a fresh engine that would satisfy every
        // internal check while having lost the caller's work.
        if (held_session) {
            if (req.show_progress)
                core_progress(host, "HGEvolve: " + req.session_op + " on session " +
                                        std::to_string(req.session_handle) + "...");
            if (req.session_op == "Step") {
                // A steered Step names frontier states by effective id. Resolving them against
                // the frontier -- rather than against every state -- is what makes an id that
                // is not on the frontier an ERROR rather than a silent no-op: a caller steering
                // toward a state the exploration already passed would otherwise get an
                // unexplained empty step.
                std::vector<hgcommon::StateId> only_from;
                if (!req.session_from.empty()) {
                    std::unordered_map<int64_t, hgcommon::StateId> by_effective;
                    for (hgcommon::StateId raw : holder->frontier())
                        by_effective.emplace(get_effective_state_id(raw), raw);
                    for (int64_t want : req.session_from) {
                        auto it = by_effective.find(want);
                        if (it == by_effective.end())
                            throw std::runtime_error(
                                "Step: state " + std::to_string(want) + " is not on this "
                                "session's frontier, so there is nothing to continue from it. "
                                "The frontier is reported as \"Frontier\" in every session "
                                "reply.");
                        only_from.push_back(it->second);
                    }
                }
                try {
                    holder->extend(req.steps, only_from);
                } catch (...) {
                    worker_session().invalidate();
                    throw;
                }
                // The graph grew: the snapshot above named states by the identity they had
                // BEFORE the step, which is right for reading the caller's selection and wrong
                // for everything after it.
                content_index_built = false;
            }
            req.steps = static_cast<int>(engine.max_steps());
        }


        for (const auto& w : engine.warnings())
            req.ffi_warnings.push_back({"Engine", 1, w});

        // Build WXF output - only include requested data components
        wxf::Writer wxf_writer;
        wxf_writer.write_header();

        wxf::WXFValueAssociation full_result;

        // An `Open` retains the engine, and the reply has to carry the handle -- a session the
        // caller cannot name is a leak it cannot close. Taken HERE, before serialization, so the
        // handle is a value in the result rather than something appended after the bytes are
        // built. Moving the unique_ptr does not move the objects: the holder owns them on the
        // heap, so the `hg` and `engine` references below stay valid.
        //
        // A refusal (one session is already live, D7) aborts the job rather than silently
        // returning an unretained result, because a caller that asked for a session and got a
        // plain answer has no way to tell.
        if (opening_session) {
            try {
                const uint64_t h = worker_session().open(std::move(engine_holder));
                full_result.push_back({wxf::WXFValue("Session"),
                                       wxf::WXFValue(static_cast<int64_t>(h))});
            } catch (const hgffi::SessionError& e) {
                throw std::runtime_error(std::string("Open refused: ") + e.what());
            }
        } else if (held_session) {
            // Echoed so every reply that came from a session says which one, and a caller
            // multiplexing over replies never has to match one to a request to find out.
            full_result.push_back({wxf::WXFValue("Session"),
                                   wxf::WXFValue(static_cast<int64_t>(req.session_handle))});
        }

        // THE FRONTIER, in every reply a session gives. A caller cannot steer a continuation
        // toward states it cannot name, and it has no other way to learn which states are
        // still unexpanded: a state's presence in the graph says nothing about whether the
        // budget stopped before or after expanding it. Reported as effective ids, the same
        // identity every other field uses, and deduplicated because several raw states of one
        // canonical class can sit on the frontier together while the caller sees one id.
        if (opening_session || held_session) {
            hgffi::EngineHolder* h = holder;
            std::set<int64_t> seen;
            wxf::WXFValueList frontier;
            for (hgcommon::StateId raw : h->frontier()) {
                const int64_t eff = get_effective_state_id(raw);
                if (seen.insert(eff).second) frontier.push_back(wxf::WXFValue(eff));
            }
            full_result.push_back({wxf::WXFValue("Frontier"), wxf::WXFValue(frontier)});
        }

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
        if (req.include_states) {
            const uint32_t num_states = hg.num_published_states();
            const ContentIndex& ci = content_index();
            const auto& content_hash_to_id = ci.first_with;
            const auto& state_content_hashes = ci.hash_of;

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
                hypergraph::StateId content_id = content_hash_to_id.at(state_content_hashes[sid]);

                // Association key: raw state id.
                sections.write_byte(static_cast<uint8_t>(wxf::Token::Rule));
                sections.write(static_cast<int64_t>(sid));

                // state_data association: Id, CanonicalId, ContentStateId, Step, Edges,
                // IsInitial, and (optionally) CanonicalHash.
                sections.write_byte(static_cast<uint8_t>(wxf::Token::Association));
                sections.write_varint(req.include_canonical_hashes ? 7u : 6u);

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

                if (req.include_canonical_hashes) {
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
        if (req.include_events) {
            // Send ALL events (not just canonical) - WL uses CanonicalId for vertex merging
            // This preserves event multiplicity: multiple events with same canonical ID
            // map to one vertex, but their edges to different output states are preserved.
            uint32_t num_raw_events = hg.num_published_events();

            // First pass fixes the emitted event set so the association length is
            // known before streaming.
            std::vector<uint32_t> emit_eids;
            emit_eids.reserve(num_raw_events);
            for (uint32_t eid = 0; eid < num_raw_events; ++eid) {
                const hypergraph::Event& event = hg.get_event(eid);
                if (event.id == hypergraph::INVALID_ID) continue;
                if (!req.show_genesis_events && hg.is_genesis_event(eid)) continue;
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
        if (req.include_events_minimal && !req.include_events) {
            uint32_t num_raw_events = hg.num_published_events();

            std::vector<uint32_t> emit_eids;
            emit_eids.reserve(num_raw_events);
            for (uint32_t eid = 0; eid < num_raw_events; ++eid) {
                const hypergraph::Event& event = hg.get_event(eid);
                if (event.id == hypergraph::INVALID_ID) continue;
                if (!req.show_genesis_events && hg.is_genesis_event(eid)) continue;
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
        if (req.include_causal_edges) {
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
                if (!req.show_genesis_events &&
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
        if (req.include_branchial_edges) {
            wxf::WXFValueList branchial_edges;
            auto branchial_edge_vec = hg.causal_graph().get_branchial_edges();
            for (const auto& edge : branchial_edge_vec) {
                // Skip edges involving genesis events if ShowGenesisEvents is false
                if (!req.show_genesis_events &&
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

        // BranchialStateEdges / BranchialStateEdgesAllSiblings -> the state-endpoint projection
        // of the branchial relation: {From -> state, To -> state} plus the unique vertices.
        // Both rules live in hgmarshal so the device answers them identically; what stays here
        // is this engine's traversal of its own storage, which is the part that legitimately
        // differs.
        const auto bse_output_state = [&](uint32_t eid) {
            return static_cast<uint32_t>(hg.get_event(eid).output_state);
        };
        const auto bse_canonical = [&](uint32_t sid) {
            return static_cast<int64_t>(hg.get_canonical_state(sid));
        };
        const auto bse_step = [&](uint32_t sid) { return hg.get_state(sid).step; };
        hgmarshal::GraphOptions bse_opts;
        bse_opts.branchial_step = req.branchial_step;
        bse_opts.steps = static_cast<int>(req.steps);

        if (req.include_branchial_state_edges) {
            std::vector<std::pair<uint32_t, uint32_t>> pairs;
            for (const auto& be : hg.causal_graph().get_branchial_edges()) {
                if (!req.show_genesis_events &&
                    (hg.is_genesis_event(be.event1) || hg.is_genesis_event(be.event2))) {
                    continue;
                }
                pairs.emplace_back(be.event1, be.event2);
            }
            hgmarshal::push_branchial_state_edges(
                full_result,
                hgmarshal::branchial_state_edges_from_pairs(
                    pairs, bse_output_state, bse_canonical, bse_step, bse_opts));
        }

        if (req.include_branchial_state_edges_all_siblings) {
            const auto for_each_group = [&](auto&& emit) {
                hg.causal_graph().for_each_state_events(
                    [&]([[maybe_unused]] hypergraph::StateId input_state, auto* event_list) {
                        std::vector<uint32_t> events;
                        event_list->for_each([&](hypergraph::EventId eid) {
                            if (!req.show_genesis_events && hg.is_genesis_event(eid)) return;
                            events.push_back(eid);
                        });
                        emit(events);
                    });
            };
            hgmarshal::push_branchial_state_edges(
                full_result,
                hgmarshal::branchial_state_edges_all_siblings(
                    for_each_group, bse_output_state, bse_canonical, bse_step, bse_opts));
        }

        // ========================================================================
        // GraphData - Graph-ready data for direct Graph[] construction in WL
        // ========================================================================
        if (!req.graph_properties.empty()) {

            // Helper: Get effective state ID based on canonicalization mode
            // Helper: Get effective event ID based on event canonicalization
            // Note: EVENT_SIG_FULL uses InputState/OutputState which require canonical state IDs.
            // When CanonicalizeStates=None, we must use raw event IDs because canonical_event_id
            // was computed using canonical state IDs during evolution.
            auto get_effective_event_id = [&](hypergraph::EventId eid) -> int64_t {
                if (req.canonicalize_states_mode == "None" || req.event_signature_keys == hypergraph::EVENT_SIG_NONE)
                    return static_cast<int64_t>(eid);
                const hypergraph::Event& e = hg.get_event(eid);
                return e.is_canonical() ? static_cast<int64_t>(eid)
                                        : static_cast<int64_t>(e.canonical_event_id);
            };

            // Helper: Serialize state edges as list of {edgeId, v1, v2, ...}
            // When CanonicalizeStates is Full, emits IR-canonicalized edges
            // Memoized by state, because the same state's edges are asked for once as a state
            // and again by EVERY event that enters or leaves it: S + 2E calls over S distinct
            // states. Under Full each call ran a whole IR canonicalization, so an evolution with
            // more events than states paid for the same state's canonical form many times over.
            //
            // The cache holds what the frame is about to carry anyway, and stays empty for the
            // Structure properties, which serialize ids and steps and never ask for edges.
            std::unordered_map<uint32_t, wxf::WXFValueList> state_edges_memo;
            auto build_state_edges = [&](hypergraph::StateId sid) -> wxf::WXFValueList {
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
            auto serialize_state_edges = [&](hypergraph::StateId sid) -> const wxf::WXFValueList& {
                auto it = state_edges_memo.find(sid);
                if (it == state_edges_memo.end())
                    it = state_edges_memo.emplace(sid, build_state_edges(sid)).first;
                return it->second;
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
                if (!req.show_genesis_events && hg.is_genesis_event(eid)) return false;
                return true;
            };

            // Adapt the engine to the shared graph marshaller (graph_marshal.hpp) so the CPU
            // and the GPU backend build byte-identical GraphData. The wrappers reuse the
            // effective-id and serialization lambdas above -- one graph-building code path.
            // Under the reconstruction the events a caller is TOLD about are the replay's, not
            // the materialised ones, and NumEvents already reports those. Built once here so the
            // graph's vertices are the same set: dense id per distinct identity, and the content
            // that describes it.
            struct ReconEvents {
                bool active = false;
                std::unordered_map<uint64_t, int64_t> dense_of_sig;   // identity -> vertex id
                std::unordered_map<int64_t, hypergraph::QcEventContent> content;
                uint32_t raw_count = 0;
            };
            ReconEvents recon;
            if (hg.quotient_reconstruction()) {
                recon.active = true;
                recon.raw_count = static_cast<uint32_t>(hg.num_reconstructed_raw_events());
                hg.for_each_reconstructed_event(
                    [&](uint32_t dense, uint32_t raw, const hypergraph::QcEventContent& c) {
                        const int64_t id = static_cast<int64_t>(dense);
                        recon.dense_of_sig[hg.event_pair_signature(raw)] = id;
                        recon.content[id] = c;
                    });
            }

            struct CpuGraphSource {
                const hypergraph::Hypergraph& hg;
                const ReconEvents& recon;
                bool show_genesis;
                std::function<int64_t(hypergraph::StateId)> eff_state;
                std::function<int64_t(hypergraph::EventId)> eff_event;
                std::function<bool(hypergraph::EventId)> valid_event;
                std::function<wxf::WXFValueAssociation(hypergraph::StateId)> state_data;
                std::function<wxf::WXFValueAssociation(hypergraph::EventId)> event_data;
                // What the CALLER's property list asked for, from graph_property_needs. This is
                // not the same question as what the run recorded: the engine may prove a relation
                // empty from the rules alone and skip the work, and then a requested property is
                // correctly served an empty graph. The guards below are about the name pairing,
                // so they ask the request, which is the thing that pairing decides.
                bool needs_causal;
                bool needs_branchial;

                // THE SCAN BOUND IS WHAT IS PUBLISHED, NOT WHAT WAS CLAIMED. State ids come
                // from an atomic increment taken before the state is constructed, so the claim
                // counter runs ahead and an id claimed but never emplaced leaves it permanently
                // above what exists. state_valid below dereferences, so a scan to the claim
                // counter reaches an index that holds no element and throws out of the
                // marshaller -- the caller receives an error instead of a graph.
                uint32_t num_states() const { return hg.num_published_states(); }
                bool state_valid(uint32_t sid) const { return hg.get_state(sid).id != hypergraph::INVALID_ID; }
                int64_t effective_state_id(uint32_t sid) const { return eff_state(sid); }
                uint32_t state_step(uint32_t sid) const { return hg.get_state(sid).step; }
                wxf::WXFValueAssociation serialize_state_data(uint32_t sid) const { return state_data(sid); }
                // Under the reconstruction an "event id" is one of the replay's applications.
                // The scan bound, the validity test and the identity all follow from that, so
                // the marshaller builds its graph over the reconstruction without knowing.
                uint32_t num_raw_events() const {
                    // Published, not claimed, for the same reason as num_states above. The
                    // reconstruction's own count is a materialised total and is already exact.
                    return recon.active ? recon.raw_count : hg.num_published_events();
                }
                bool is_valid_event(uint32_t eid) const {
                    if (!recon.active) return valid_event(eid);
                    // An application whose identity was not registered stands for no vertex.
                    return recon.dense_of_sig.count(hg.event_pair_signature(eid)) != 0;
                }
                int64_t effective_event_id(uint32_t eid) const {
                    if (!recon.active) return eff_event(eid);
                    auto it = recon.dense_of_sig.find(hg.event_pair_signature(eid));
                    return it == recon.dense_of_sig.end() ? -1 : it->second;
                }
                // The endpoints of a reconstructed event are CLASSES, and a class is pointed at
                // by its frame -- the state whose labelling it is described in. Nothing is
                // materialised for the event itself.
                uint32_t event_input_state(uint32_t eid) const {
                    if (!recon.active) return hg.get_event(eid).input_state;
                    const auto* c = hg.reconstructed_event_content(eid);
                    return c ? hg.class_frame_state(c->from_class) : hypergraph::INVALID_ID;
                }
                uint32_t event_output_state(uint32_t eid) const {
                    if (!recon.active) return hg.get_event(eid).output_state;
                    const auto* c = hg.reconstructed_event_content(eid);
                    return c ? hg.class_frame_state(c->to_class) : hypergraph::INVALID_ID;
                }
                wxf::WXFValueAssociation serialize_event_data(uint32_t eid) const {
                    if (!recon.active) return event_data(eid);
                    // What the reconstruction holds and no more: the identity, the rule, and the
                    // endpoint classes as their frame states. A reconstructed event has no
                    // consumed/produced edge lists -- the replay mints an id and materialises
                    // nothing -- so claiming any would be inventing them.
                    wxf::WXFValueAssociation d;
                    d.push_back({wxf::WXFValue("Id"), wxf::WXFValue(effective_event_id(eid))});
                    const auto* c = hg.reconstructed_event_content(eid);
                    d.push_back({wxf::WXFValue("RuleIndex"),
                                 wxf::WXFValue(static_cast<int64_t>(c ? c->rule : 0))});
                    d.push_back({wxf::WXFValue("InputState"),
                                 wxf::WXFValue(effective_state_id(event_input_state(eid)))});
                    d.push_back({wxf::WXFValue("OutputState"),
                                 wxf::WXFValue(effective_state_id(event_output_state(eid)))});
                    return d;
                }
                std::vector<std::pair<uint32_t, uint32_t>> causal_event_pairs() const {
                    // A property the request derivation did not anticipate would be handed an
                    // EMPTY relation and would serve an empty graph without a word. The name test
                    // that decides what to record lives in graph_marshal.hpp beside the one
                    // that decides what to build, so a miss here is a defect in that pairing.
                    if (!needs_causal) {
                        throw std::runtime_error(
                            "a graph property asked for the causal relation, which this run was "
                            "not asked to record: graph_property_needs missed its name");
                    }
                    std::vector<std::pair<uint32_t, uint32_t>> out;
                    if (recon.active) {
                        // The relation the run SERVES. Its endpoints are the replay's
                        // application ids, which effective_event_id maps to vertices above.
                        hg.for_each_reconstructed_causal_as(
                            hg.causal_graph().transitive_reduction_enabled(),
                            [](uint32_t e) { return e; },
                            [&](uint64_t p, uint64_t c) {
                                out.emplace_back(static_cast<uint32_t>(p), static_cast<uint32_t>(c));
                            });
                        return out;
                    }
                    for (const auto& ce : hg.causal_graph().get_causal_edges()) {
                        if (!show_genesis && (hg.is_genesis_event(ce.producer) || hg.is_genesis_event(ce.consumer))) continue;
                        out.emplace_back(ce.producer, ce.consumer);
                    }
                    return out;
                }
                std::vector<std::pair<uint32_t, uint32_t>> branchial_event_pairs() const {
                    if (!needs_branchial) {
                        throw std::runtime_error(
                            "a graph property asked for the branchial relation, which this run "
                            "was not asked to record: graph_property_needs missed its name");
                    }
                    std::vector<std::pair<uint32_t, uint32_t>> out;
                    if (recon.active) {
                        hg.for_each_reconstructed_branchial_as(
                            [](uint32_t e) { return e; },
                            [&](uint64_t a, uint64_t b) {
                                out.emplace_back(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
                            });
                        return out;
                    }
                    for (const auto& be : hg.causal_graph().get_branchial_edges()) {
                        if (!show_genesis && (hg.is_genesis_event(be.event1) || hg.is_genesis_event(be.event2))) continue;
                        out.emplace_back(be.event1, be.event2);
                    }
                    return out;
                }
            };
            CpuGraphSource gsrc{hg, recon, req.show_genesis_events,
                get_effective_state_id, get_effective_event_id, is_valid_event,
                serialize_state_data, serialize_event_data,
                gneeds.causal, gneeds.branchial};
            hgmarshal::GraphOptions gopts;
            gopts.edge_deduplication = req.edge_deduplication;
            gopts.branchial_step = req.branchial_step;
            gopts.steps = req.steps;
            // DELTA DELIVERY is a session's, because only a session has a record of what it has
            // already been sent. An `Evolve` has no history to be a delta against.
            //
            // REFUSED ON THE QUOTIENT RECONSTRUCTION ROUTE, and reported rather than silently
            // downgraded. There the causal relation is REDUCED ON READ -- the stored base is a
            // set and a DAG's transitive reduction is unique -- so an edge already SERVED can
            // drop out of the reduction once a later path makes it redundant. A delta has no way
            // to say "withdraw that edge", so the caller's merged graph would keep an edge the
            // engine no longer serves. Retraction entries would fix it; until they exist the
            // route delivers in full.
            hgffi::DeliveryCursor* cursor = nullptr;
            if (held_session && req.delivery_delta) {
                if (recon.active) {
                    req.ffi_warnings.push_back(
                        {"OptionSkipped", 1,
                         "Delivery -> \"Delta\" is not served on the quotient reconstruction "
                         "route: its causal relation is reduced on read, so an edge already sent "
                         "can leave the reduction, and a delta cannot withdraw one. This reply "
                         "carries the whole graph."});
                } else {
                    cursor = &holder->delivery_cursor();
                }
            }
            if (held_session && !req.delivery_delta) holder->delivery_cursor().reset();
            full_result.push_back(
                {wxf::WXFValue("GraphData"),
                 hgmarshal::build_graph_data(gsrc, req.graph_properties, gopts, cursor)});
        }

        // Only include counts when requested
        if (req.include_num_states) {
            full_result.push_back({wxf::WXFValue("NumStates"), wxf::WXFValue(static_cast<int64_t>(hg.num_canonical_states()))});
        }
        if (req.include_num_events) {
            // Under the reconstruction (quotient exploration, or Automatic identity on either
            // path) the observable event count is the reconstruction's -- the authority-anchored
            // identity count the golden matrix pins -- not the materialised dedup count.
            const int64_t n_events = hg.quotient_reconstruction()
                ? static_cast<int64_t>(hg.observable_num_events())
                : static_cast<int64_t>(engine.num_events());
            full_result.push_back({wxf::WXFValue("NumEvents"), wxf::WXFValue(n_events)});
        }
        if (req.include_num_causal_edges) {
            // Count unique (producer, consumer) event pairs for v1 semantics
            // When ShowGenesisEvents is false, we must filter out pairs involving genesis events
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
            } else if (req.show_genesis_events) {
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
        if (req.include_num_branchial_edges) {
            const int64_t n_branchial = hg.quotient_reconstruction()
                ? static_cast<int64_t>(hg.observable_num_branchial())
                : static_cast<int64_t>(hg.num_branchial_edges());
            full_result.push_back({wxf::WXFValue("NumBranchialEdges"), wxf::WXFValue(n_branchial)});
        }

        // GlobalEdges -> List of all edges created during evolution
        // Each edge is {edge_id, v1, v2, ...}
        if (req.include_global_edges) {
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
        if (req.include_state_bitvectors) {
            wxf::WXFValueAssociation state_bitvectors;
            uint32_t num_states = hg.num_published_states();
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
        if (!req.ffi_warnings.empty()) {
            wxf::WXFValueList warn;
            for (const auto& w : req.ffi_warnings) {
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

        if (req.show_progress) {
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
    // Reference the stamp from an exported entry point so it reaches the DLL's .rodata. An
    // object nothing refers to is what a linker is entitled to drop, and a stamp that can be
    // dropped is not evidence about the artifact.
    static const char* volatile stamp_anchor = nullptr;
    stamp_anchor = hgffi::kBuildStamp;
    (void)stamp_anchor;
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData /* libData */) {
}

#endif  // HG_STANDALONE_BINARY