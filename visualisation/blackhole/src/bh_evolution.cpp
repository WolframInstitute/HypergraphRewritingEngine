#include <blackhole/bh_evolution.hpp>
#include <sstream>
#include <iostream>
#include <regex>
#include <mutex>
#include <cmath>
#include <set>
#include <unordered_set>
#include <numeric>
#include <limits>

namespace viz::blackhole {

// =============================================================================
// EvolutionRunner Implementation
// =============================================================================

EvolutionRunner::EvolutionRunner(size_t num_threads)
    : num_threads_(num_threads > 0 ? num_threads : std::thread::hardware_concurrency())
{
    // Create analysis job system (separate from evolution's job system)
    analysis_job_system_ = std::make_unique<job_system::JobSystem<AnalysisJobType>>(num_threads_);
}

EvolutionRunner::~EvolutionRunner() {
    if (analysis_job_system_) {
        analysis_job_system_->shutdown();
    }
}

void EvolutionRunner::report_progress(const std::string& stage, float progress) {
    if (progress_callback_) {
        progress_callback_(stage, progress);
    }
}

bool EvolutionRunner::parse_rule(const std::string& rule_string) {
    // Parse Wolfram-style rule: {{a,b},{b,c}} -> {{a,b},{b,c},{c,d},{d,a}}
    // This is a simplified parser that handles the common cases

    if (!engine_) return false;

    // Find the arrow
    auto arrow_pos = rule_string.find("->");
    if (arrow_pos == std::string::npos) {
        std::cerr << "Invalid rule format: missing '->' in " << rule_string << std::endl;
        return false;
    }

    std::string lhs = rule_string.substr(0, arrow_pos);
    std::string rhs = rule_string.substr(arrow_pos + 2);

    // Trim whitespace
    auto trim = [](std::string& s) {
        while (!s.empty() && std::isspace(s.front())) s.erase(s.begin());
        while (!s.empty() && std::isspace(s.back())) s.pop_back();
    };
    trim(lhs);
    trim(rhs);

    // Parse edge lists: {{a,b},{c,d}} -> vector of vectors of variable names
    auto parse_edges = [](const std::string& str) -> std::vector<std::vector<std::string>> {
        std::vector<std::vector<std::string>> result;

        // Simple state machine parser
        int depth = 0;
        std::string current_var;
        std::vector<std::string> current_edge;

        for (char c : str) {
            if (c == '{') {
                depth++;
                if (depth == 2) {
                    current_edge.clear();
                    current_var.clear();
                }
            } else if (c == '}') {
                if (depth == 2) {
                    if (!current_var.empty()) {
                        current_edge.push_back(current_var);
                        current_var.clear();
                    }
                    result.push_back(current_edge);
                }
                depth--;
            } else if (c == ',') {
                if (depth == 2 && !current_var.empty()) {
                    current_edge.push_back(current_var);
                    current_var.clear();
                }
            } else if (!std::isspace(c)) {
                current_var += c;
            }
        }

        return result;
    };

    auto lhs_edges = parse_edges(lhs);
    auto rhs_edges = parse_edges(rhs);

    // Debug output
    std::cout << "  Rule parsing:" << std::endl;
    std::cout << "    LHS: " << lhs_edges.size() << " edges" << std::endl;
    for (size_t i = 0; i < lhs_edges.size(); ++i) {
        std::cout << "      [" << i << "] {";
        for (size_t j = 0; j < lhs_edges[i].size(); ++j) {
            if (j > 0) std::cout << ",";
            std::cout << lhs_edges[i][j];
        }
        std::cout << "}" << std::endl;
    }
    std::cout << "    RHS: " << rhs_edges.size() << " edges" << std::endl;
    for (size_t i = 0; i < rhs_edges.size(); ++i) {
        std::cout << "      [" << i << "] {";
        for (size_t j = 0; j < rhs_edges[i].size(); ++j) {
            if (j > 0) std::cout << ",";
            std::cout << rhs_edges[i][j];
        }
        std::cout << "}" << std::endl;
    }

    if (lhs_edges.empty()) {
        std::cerr << "Invalid rule: empty LHS" << std::endl;
        return false;
    }

    // Build variable name -> index mapping
    std::unordered_map<std::string, uint8_t> var_map;
    uint8_t next_var = 0;

    auto get_var_id = [&](const std::string& name) -> uint8_t {
        auto it = var_map.find(name);
        if (it == var_map.end()) {
            var_map[name] = next_var;
            return next_var++;
        }
        return it->second;
    };

    // Build rule using RuleBuilder
    using namespace hypergraph::unified;
    auto builder = make_rule(0);

    // Add LHS edges
    for (const auto& edge : lhs_edges) {
        std::vector<uint8_t> vars;
        for (const auto& v : edge) {
            vars.push_back(get_var_id(v));
        }
        // Use initializer_list via intermediate
        switch (vars.size()) {
            case 2: builder.lhs({vars[0], vars[1]}); break;
            case 3: builder.lhs({vars[0], vars[1], vars[2]}); break;
            case 4: builder.lhs({vars[0], vars[1], vars[2], vars[3]}); break;
            default: break;  // Unsupported arity
        }
    }

    // Add RHS edges
    for (const auto& edge : rhs_edges) {
        std::vector<uint8_t> vars;
        for (const auto& v : edge) {
            vars.push_back(get_var_id(v));
        }
        switch (vars.size()) {
            case 2: builder.rhs({vars[0], vars[1]}); break;
            case 3: builder.rhs({vars[0], vars[1], vars[2]}); break;
            case 4: builder.rhs({vars[0], vars[1], vars[2], vars[3]}); break;
            default: break;  // Unsupported arity
        }
    }

    // Print variable mapping
    std::cout << "    Variables: ";
    for (const auto& [name, id] : var_map) {
        std::cout << name << "=" << static_cast<int>(id) << " ";
    }
    std::cout << std::endl;

    // Build and add rule
    engine_->add_rule(builder.build());

    return true;
}

std::vector<std::vector<hypergraph::unified::VertexId>> EvolutionRunner::convert_edges(
    const BHInitialCondition& initial
) {
    std::vector<std::vector<hypergraph::unified::VertexId>> result;
    result.reserve(initial.edges.size());

    for (const auto& edge : initial.edges) {
        result.push_back({edge.v1, edge.v2});
    }

    return result;
}

EvolutionResult EvolutionRunner::extract_states() {
    EvolutionResult result;

    if (!hypergraph_) return result;

    uint32_t num_states = hypergraph_->num_states();
    result.total_states = static_cast<int>(num_states);
    result.total_events = static_cast<int>(hypergraph_->num_events());

    if (num_states == 0) return result;

    // Find max step
    uint32_t max_step = 0;
    for (uint32_t sid = 0; sid < num_states; ++sid) {
        const auto& state = hypergraph_->get_state(sid);
        if (state.step > max_step) {
            max_step = state.step;
        }
    }
    result.max_step_reached = static_cast<int>(max_step);

    // Resize output
    result.states_by_step.resize(max_step + 1);
    result.state_data_by_step.resize(max_step + 1);

#ifdef BH_DEBUG_AGGREGATION
    // Debug: count states per step
    std::vector<int> states_per_step(max_step + 1, 0);
    for (uint32_t sid = 0; sid < num_states; ++sid) {
        const auto& state = hypergraph_->get_state(sid);
        states_per_step[state.step]++;
    }
    std::cout << "[DEBUG extract_states] num_states=" << num_states << ", max_step=" << max_step << std::endl;
    for (uint32_t s = 0; s <= max_step; ++s) {
        std::cout << "  step " << s << ": " << states_per_step[s] << " states" << std::endl;
    }
#endif

    // Convert each state to SimpleGraph AND extract StateData with edge IDs
    for (uint32_t sid = 0; sid < num_states; ++sid) {
        if (should_stop_.load(std::memory_order_relaxed)) break;

        const auto& state = hypergraph_->get_state(sid);

        // Extract SimpleGraph for dimension computation
        SimpleGraph graph = state_to_simple_graph(*hypergraph_, sid);

        // Extract StateData with edge IDs for frequency counting
        StateData sd;
        std::unordered_set<VertexId> vertex_set;

        state.edges.for_each([&](hypergraph::unified::EdgeId eid) {
            const auto& edge = hypergraph_->get_edge(eid);

            // For binary edges (arity 2), store directly with edge ID
            if (edge.arity == 2) {
                VertexId v1 = edge.vertices[0];
                VertexId v2 = edge.vertices[1];
                sd.edges.push_back({v1, v2, static_cast<EdgeId>(eid)});

                if (vertex_set.insert(v1).second) sd.vertices.push_back(v1);
                if (vertex_set.insert(v2).second) sd.vertices.push_back(v2);
            } else {
                // For higher-arity edges, store consecutive vertex pairs
                // These get synthetic IDs (INVALID_EDGE_ID) since they don't map 1:1
                for (uint8_t i = 0; i < edge.arity; ++i) {
                    VertexId v = edge.vertices[i];
                    if (vertex_set.insert(v).second) sd.vertices.push_back(v);
                }
                for (uint8_t i = 0; i + 1 < edge.arity; ++i) {
                    sd.edges.push_back({edge.vertices[i], edge.vertices[i + 1], INVALID_EDGE_ID});
                }
            }
        });

        if (graph.vertex_count() > 0) {
            result.states_by_step[state.step].push_back(std::move(graph));
            result.state_data_by_step[state.step].push_back(std::move(sd));
        }
    }

    return result;
}

EvolutionResult EvolutionRunner::run_evolution(
    const BHInitialCondition& initial,
    const EvolutionConfig& config
) {
    should_stop_.store(false);
    report_progress("Initializing", 0.0f);

    // Create fresh hypergraph and engine
    hypergraph_ = std::make_unique<hypergraph::unified::UnifiedHypergraph>();
    engine_ = std::make_unique<hypergraph::unified::ParallelEvolutionEngine>(
        hypergraph_.get(), num_threads_
    );

    // Configure engine
    engine_->set_max_steps(config.max_steps > 0 ? config.max_steps : 100);
    engine_->set_max_states_per_step(config.max_states_per_step);  // 0 = unlimited
    engine_->set_max_successor_states_per_parent(config.max_successors_per_parent);  // 0 = unlimited
    engine_->set_exploration_probability(config.exploration_probability);

    // State canonicalization: deduplicate isomorphic states
    if (config.canonicalize_states) {
        hypergraph_->set_state_canonicalization_mode(
            hypergraph::unified::StateCanonicalizationMode::Full
        );
    } else {
        hypergraph_->set_state_canonicalization_mode(
            hypergraph::unified::StateCanonicalizationMode::None
        );
    }

    // Event canonicalization: deduplicate equivalent events
    if (config.canonicalize_events) {
        hypergraph_->set_event_signature_keys(hypergraph::unified::EVENT_SIG_FULL);
    } else {
        hypergraph_->set_event_signature_keys(hypergraph::unified::EVENT_SIG_NONE);
    }

    // Exploration: only explore from canonical state representatives
    engine_->set_explore_from_canonical_states_only(config.explore_from_canonical_only);

    // Batched matching mode
    engine_->set_batched_matching(config.batched_matching);

    // Early termination: stop pattern matching when reservoir is full (speed vs uniformity tradeoff)
    engine_->set_early_terminate_on_reservoir_full(config.early_terminate_reservoir);

    // Parse and add rule
    if (!parse_rule(config.rule)) {
        std::cerr << "Failed to parse rule: " << config.rule << std::endl;
        return {};
    }

    // Convert initial edges
    auto initial_edges = convert_edges(initial);

    report_progress("Running evolution", 0.1f);

    // Run evolution - use uniform random mode if configured
    if (config.uniform_random) {
        std::cout << "  Using uniform random mode, matches_per_step=" << config.matches_per_step << std::endl;
        engine_->evolve_uniform_random(
            initial_edges,
            config.max_steps,
            config.matches_per_step > 0 ? static_cast<size_t>(config.matches_per_step) : 0
        );
    } else {
        engine_->evolve(initial_edges, config.max_steps);
    }

    // Print evolution stats (only for non-uniform-random mode, as these stats
    // are for the match forwarding system which uniform_random doesn't use)
    if (!config.uniform_random) {
        const auto& stats = engine_->stats();
        std::cout << "\n=== Evolution Stats ===" << std::endl;
        std::cout << "  Total matches found: " << engine_->total_matches() << std::endl;
        std::cout << "  Matches forwarded: " << stats.matches_forwarded.load() << std::endl;
        std::cout << "  Matches invalidated: " << stats.matches_invalidated.load() << std::endl;
        std::cout << "  New matches (delta): " << stats.new_matches_discovered.load() << std::endl;
        std::cout << "  Full pattern matches: " << stats.full_pattern_matches.load() << std::endl;
        std::cout << "  Delta pattern matches: " << stats.delta_pattern_matches.load() << std::endl;
        std::cout << "=======================" << std::endl;
    }

    report_progress("Extracting states", 0.8f);

    // Extract states
    auto result = extract_states();

    // MEMORY OPTIMIZATION: Clear hypergraph and engine after extraction
    // The data has been copied to result, so we don't need the originals anymore
    // This can save hundreds of MB to GB for large evolutions
    size_t num_states = hypergraph_ ? hypergraph_->num_states() : 0;
    size_t num_edges = hypergraph_ ? hypergraph_->num_edges() : 0;
    engine_.reset();
    hypergraph_.reset();
    std::cout << "  Freed hypergraph memory (" << num_states << " states, "
              << num_edges << " edges)" << std::endl;

    report_progress("Evolution complete", 1.0f);

    return result;
}

BHAnalysisResult EvolutionRunner::run_full_analysis(
    const BHInitialCondition& initial,
    const EvolutionConfig& config,
    const AnalysisConfig& analysis_config
) {
    should_stop_.store(false);

    // Run evolution
    auto evolution_result = run_evolution(initial, config);

    if (evolution_result.total_states == 0 || should_stop_.load()) {
        return {};
    }

    // Run parallel analysis
    return analyze_parallel(initial, config, evolution_result, analysis_config);
}

BHAnalysisResult EvolutionRunner::run_analysis(
    const BHInitialCondition& initial,
    const EvolutionConfig& config,
    const std::vector<std::vector<SimpleGraph>>& states_by_step,
    int total_states,
    int total_events,
    int max_step_reached,
    const AnalysisConfig& analysis_config
) {
    should_stop_.store(false);

    // Create EvolutionResult from provided data
    EvolutionResult evolution_result;
    evolution_result.states_by_step = states_by_step;
    evolution_result.total_states = total_states;
    evolution_result.total_events = total_events;
    evolution_result.max_step_reached = max_step_reached;

    if (evolution_result.total_states == 0) {
        return {};
    }

    // Run parallel analysis
    return analyze_parallel(initial, config, evolution_result, analysis_config);
}

BHAnalysisResult EvolutionRunner::analyze_parallel(
    const BHInitialCondition& initial,
    const EvolutionConfig& config,
    EvolutionResult& evolution_result,
    const AnalysisConfig& analysis_config
) {
    report_progress("Starting analysis", 0.0f);

    // Use the run_analysis function from hausdorff_analysis.hpp
    // but with parallel state analysis

    BHAnalysisResult result;
    result.initial = initial;
    result.evolution_config = config;
    result.analysis_max_radius = analysis_config.max_radius;

    if (evolution_result.states_by_step.empty()) {
        return result;
    }

    report_progress("Selecting anchors", 0.05f);

    // Find stable vertices WITHOUT copying all graphs
    std::vector<VertexId> stable = find_stable_vertices_nested(evolution_result.states_by_step);

    // Use step 0 graph for anchor selection (or first non-empty step)
    const SimpleGraph* anchor_graph = nullptr;
    for (const auto& step_graphs : evolution_result.states_by_step) {
        if (!step_graphs.empty()) {
            anchor_graph = &step_graphs[0];
            break;
        }
    }

    if (!anchor_graph || anchor_graph->vertex_count() == 0) {
        std::cerr << "No valid graph for anchor selection" << std::endl;
        return result;
    }

    // Select anchors
    std::vector<VertexId> candidates = stable.empty() ? anchor_graph->vertices() : stable;
    result.anchor_vertices = select_anchors(
        *anchor_graph,
        candidates,
        analysis_config.num_anchors,
        analysis_config.anchor_min_separation
    );

    // Print anchor information
    std::cout << "\n=== Anchor Vertex Selection ===" << std::endl;
    std::cout << "  Stable vertices across all timesteps: " << stable.size() << std::endl;
    std::cout << "  Candidate pool size: " << candidates.size()
              << (stable.empty() ? " (using all vertices - no stable vertices found)" : " (stable vertices)")
              << std::endl;
    std::cout << "  Anchors selected: " << result.anchor_vertices.size()
              << " / " << analysis_config.num_anchors << " requested" << std::endl;

    if (!result.anchor_vertices.empty()) {
        std::cout << "  Anchor IDs: ";
        for (size_t i = 0; i < result.anchor_vertices.size(); ++i) {
            std::cout << result.anchor_vertices[i];
            if (i + 1 < result.anchor_vertices.size()) std::cout << ", ";
        }
        std::cout << std::endl;

        // Compute and print pairwise distances between anchors
        std::cout << "  Pairwise distances:" << std::endl;
        for (size_t i = 0; i < result.anchor_vertices.size(); ++i) {
            for (size_t j = i + 1; j < result.anchor_vertices.size(); ++j) {
                int dist = anchor_graph->distance(result.anchor_vertices[i], result.anchor_vertices[j]);
                std::cout << "    anchor[" << i << "]=" << result.anchor_vertices[i]
                          << " <-> anchor[" << j << "]=" << result.anchor_vertices[j]
                          << " : " << (dist < 0 ? "unreachable" : std::to_string(dist)) << std::endl;
            }
        }
    }

    if (result.anchor_vertices.size() < static_cast<size_t>(analysis_config.num_anchors)) {
        std::cout << "  WARNING: Could not select requested number of anchors!" << std::endl;
        std::cout << "           This may affect dimension estimation quality." << std::endl;
    }
    std::cout << std::endl;

    if (result.anchor_vertices.empty()) {
        std::cerr << "Failed to select anchors" << std::endl;
        return result;
    }

    report_progress("Analyzing states", 0.1f);

    // Start analysis job system for parallel processing
    analysis_job_system_->start();

    // Use lightweight analysis to avoid copying vertices/edges
    size_t num_steps = evolution_result.states_by_step.size();
    std::vector<std::vector<LightweightAnalysis>> analyses_by_step(num_steps);

    // Pre-allocate vectors for each step to match graph count
    for (uint32_t step = 0; step < num_steps; ++step) {
        analyses_by_step[step].resize(evolution_result.states_by_step[step].size());
    }

    // Count total states for progress tracking
    size_t total_states = 0;
    for (const auto& step_graphs : evolution_result.states_by_step) {
        total_states += step_graphs.size();
    }

    std::atomic<size_t> states_analyzed{0};

    // Submit analysis jobs for each state (parallel)
    for (uint32_t step = 0; step < num_steps; ++step) {
        const auto& step_graphs = evolution_result.states_by_step[step];

        for (size_t g_idx = 0; g_idx < step_graphs.size(); ++g_idx) {
            analysis_job_system_->submit_function(
                [this, &evolution_result, &result, &analysis_config,
                 &analyses_by_step, &states_analyzed,
                 step, g_idx, total_states]() {
                    if (should_stop_.load(std::memory_order_relaxed)) return;

                    const auto& graph = evolution_result.states_by_step[step][g_idx];

                    // Run lightweight analysis (no vertex/edge copy)
                    LightweightAnalysis analysis = analyze_state_lightweight(
                        graph,
                        result.anchor_vertices,
                        analysis_config.max_radius
                    );

                    // Store result at pre-allocated index (no mutex needed - each index unique)
                    analyses_by_step[step][g_idx] = std::move(analysis);

                    // Update progress
                    size_t done = states_analyzed.fetch_add(1) + 1;
                    if (done % 10 == 0) {
                        float progress = 0.1f + 0.7f * static_cast<float>(done) / total_states;
                        report_progress("Analyzing states", progress);
                    }
                },
                AnalysisJobType::STATE_ANALYSIS
            );
        }
    }

    // Wait for all state analyses to complete
    analysis_job_system_->wait_for_completion();

    if (should_stop_.load()) {
        analysis_job_system_->shutdown();
        return result;
    }

    report_progress("Aggregating timesteps", 0.85f);

    // Aggregate each timestep in parallel
    result.per_timestep.resize(num_steps);

    for (uint32_t step = 0; step < num_steps; ++step) {
        analysis_job_system_->submit_function(
            [this, &result, &evolution_result, &analyses_by_step, &initial, step]() {
                if (should_stop_.load(std::memory_order_relaxed)) return;

                const auto& step_graphs = evolution_result.states_by_step[step];
                const auto& step_analyses = analyses_by_step[step];

                if (step_graphs.empty()) {
                    result.per_timestep[step].step = step;
                    result.per_timestep[step].num_states = 0;
                    return;
                }

                // Streaming aggregation: uses original graphs, no StateAnalysis copy
                // Pass global anchors for consistent coordinate bucketing across timesteps
                result.per_timestep[step] = aggregate_timestep_streaming(
                    step,
                    step_graphs,
                    step_analyses,
                    initial.vertex_positions,
                    result.anchor_vertices
                );
            },
            AnalysisJobType::TIMESTEP_AGGREGATE
        );
    }

    // Wait for all aggregation jobs to complete
    analysis_job_system_->wait_for_completion();

    if (should_stop_.load()) {
        analysis_job_system_->shutdown();
        return result;
    }

    // Shutdown analysis job system
    analysis_job_system_->shutdown();

    // Collect all dimensions, variances, and global dimensions for global statistics
    std::vector<float> all_dimensions;
    std::vector<float> all_variances;
    std::vector<float> all_global_means;
    std::vector<float> all_global_vars;
    for (uint32_t step = 0; step < num_steps; ++step) {
        for (float d : result.per_timestep[step].mean_dimensions) {
            if (d >= 0) {
                all_dimensions.push_back(d);
            }
        }
        for (float v : result.per_timestep[step].variance_dimensions) {
            if (v >= 0) {
                all_variances.push_back(v);
            }
        }
        for (float g : result.per_timestep[step].global_mean_dimensions) {
            if (g >= 0) {
                all_global_means.push_back(g);
            }
        }
        for (float g : result.per_timestep[step].global_variance_dimensions) {
            if (g >= 0) {
                all_global_vars.push_back(g);
            }
        }
    }

    // Compute local dimension statistics (from per-state bucketing)
    if (!all_dimensions.empty()) {
        result.dim_min = *std::min_element(all_dimensions.begin(), all_dimensions.end());
        result.dim_max = *std::max_element(all_dimensions.begin(), all_dimensions.end());
        result.dim_q05 = quantile(all_dimensions, analysis_config.quantile_low);
        result.dim_q95 = quantile(all_dimensions, analysis_config.quantile_high);
    }

    // Compute variance statistics
    if (!all_variances.empty()) {
        result.var_min = *std::min_element(all_variances.begin(), all_variances.end());
        result.var_max = *std::max_element(all_variances.begin(), all_variances.end());
        result.var_q05 = quantile(all_variances, analysis_config.quantile_low);
        result.var_q95 = quantile(all_variances, analysis_config.quantile_high);
    }

    // Compute global mean dimension statistics (computed on union graph)
    if (!all_global_means.empty()) {
        result.global_dim_min = *std::min_element(all_global_means.begin(), all_global_means.end());
        result.global_dim_max = *std::max_element(all_global_means.begin(), all_global_means.end());
        result.global_dim_q05 = quantile(all_global_means, analysis_config.quantile_low);
        result.global_dim_q95 = quantile(all_global_means, analysis_config.quantile_high);
    }

    // Compute global variance dimension statistics
    if (!all_global_vars.empty()) {
        result.global_var_min = *std::min_element(all_global_vars.begin(), all_global_vars.end());
        result.global_var_max = *std::max_element(all_global_vars.begin(), all_global_vars.end());
        result.global_var_q05 = quantile(all_global_vars, analysis_config.quantile_low);
        result.global_var_q95 = quantile(all_global_vars, analysis_config.quantile_high);
    }

    result.total_steps = evolution_result.max_step_reached;
    result.total_states = evolution_result.total_states;
    result.total_events = evolution_result.total_events;

    // Compute layouts for all timesteps
    std::cout << "  Computing layouts..." << std::endl;
    compute_all_layouts(result.per_timestep, initial.vertex_positions, analysis_config.layout);

    // Compute overall bounding radius from laid-out positions (for camera framing)
    float max_radius = 0.0f;
    for (const auto& ts : result.per_timestep) {
        for (const auto& p : ts.layout_positions) {
            float r = std::sqrt(p.x * p.x + p.y * p.y);
            if (r > max_radius) max_radius = r;
        }
    }
    result.layout_bounding_radius = std::max(1.0f, max_radius);

    // Populate states_per_step for single-state viewing
    // Use the pre-extracted StateData with edge IDs (preserves proper edge identity)
    // MEMORY OPTIMIZATION: Move instead of copy to avoid duplicate storage
    result.states_per_step.resize(num_steps);
    for (uint32_t step = 0; step < num_steps; ++step) {
        if (step < evolution_result.state_data_by_step.size()) {
            // Move pre-extracted StateData with edge IDs (avoids copy)
            result.states_per_step[step] = std::move(evolution_result.state_data_by_step[step]);
        } else {
            // Fallback: reconstruct from SimpleGraph (loses edge IDs)
            const auto& step_graphs = evolution_result.states_by_step[step];
            result.states_per_step[step].reserve(step_graphs.size());
            for (const auto& graph : step_graphs) {
                StateData sd;
                sd.vertices = graph.vertices();
                for (VertexId v : graph.vertices()) {
                    for (VertexId u : graph.neighbors(v)) {
                        if (v < u) {  // Only add each edge once
                            sd.edges.push_back({v, u});
                        }
                    }
                }
                result.states_per_step[step].push_back(std::move(sd));
            }
        }
    }
    std::cout << "  Per-state data: " << result.states_per_step.size() << " timesteps" << std::endl;
    // Debug: show states per step
    for (size_t step = 0; step < std::min(result.states_per_step.size(), size_t(5)); ++step) {
        std::cout << "    step " << step << ": " << result.states_per_step[step].size() << " states" << std::endl;
    }
    if (result.states_per_step.size() > 5) {
        std::cout << "    ..." << std::endl;
    }

    // ==========================================================================
    // Mega-union dimension computation (across ALL timesteps)
    // ==========================================================================
    std::cout << "  Computing mega-union dimension..." << std::endl;
    {
        // MEMORY OPTIMIZATION: Build global vertex/edge sets from per-timestep unions
        // instead of re-iterating all states. This uses already-computed union data.
        std::unordered_set<VertexId> all_vertex_set;
        std::set<std::pair<VertexId, VertexId>> all_edge_set;

        for (const auto& ts : result.per_timestep) {
            for (VertexId v : ts.union_vertices) {
                all_vertex_set.insert(v);
            }
            for (const auto& e : ts.union_edges) {
                VertexId v1 = std::min(e.v1, e.v2);
                VertexId v2 = std::max(e.v1, e.v2);
                all_edge_set.insert({v1, v2});
            }
        }

        // Store in result
        result.all_vertices.assign(all_vertex_set.begin(), all_vertex_set.end());
        std::sort(result.all_vertices.begin(), result.all_vertices.end());

        for (const auto& ep : all_edge_set) {
            result.all_edges.push_back({ep.first, ep.second});
        }

        std::cout << "    Total unique vertices: " << result.all_vertices.size()
                  << ", edges: " << result.all_edges.size() << std::endl;

        // Build mega-union graph
        SimpleGraph mega_union;
        mega_union.build(result.all_vertices, result.all_edges);

        // Compute dimension on mega-union
        constexpr int mega_max_radius = 5;
        auto mega_dims = estimate_all_dimensions(mega_union, mega_max_radius);

        // Get geodesic coordinates on mega-union using stored anchor vertices
        std::vector<VertexId> valid_mega_anchors;
        for (VertexId a : result.anchor_vertices) {
            if (mega_union.has_vertex(a)) {
                valid_mega_anchors.push_back(a);
            }
        }
        auto mega_coords = compute_geodesic_coordinates(mega_union, valid_mega_anchors);
        int mega_num_anchors = static_cast<int>(valid_mega_anchors.size());

        // Bucket by geodesic coordinate
        std::unordered_map<CoordKey, std::vector<float>, CoordKeyHash> mega_coord_buckets;
        for (size_t i = 0; i < result.all_vertices.size(); ++i) {
            VertexId v = result.all_vertices[i];
            if (i >= mega_dims.size()) continue;
            float dim = mega_dims[i];
            if (dim <= 0) continue;

            auto coord_it = mega_coords.find(v);
            if (coord_it == mega_coords.end()) continue;

            CoordKey key;
            key.num_anchors = mega_num_anchors;
            for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
                key.coords[a] = coord_it->second[a];
            }
            mega_coord_buckets[key].push_back(dim);
        }

        // Compute mean per bucket
        std::unordered_map<CoordKey, float, CoordKeyHash> mega_coord_to_dim;
        for (auto& [key, dims] : mega_coord_buckets) {
            if (!dims.empty()) {
                float mean = std::accumulate(dims.begin(), dims.end(), 0.0f) / dims.size();
                mega_coord_to_dim[key] = mean;
            }
        }

        // Assign to vertices and compute min/max
        result.mega_dim_min = std::numeric_limits<float>::max();
        result.mega_dim_max = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < result.all_vertices.size(); ++i) {
            VertexId v = result.all_vertices[i];

            auto coord_it = mega_coords.find(v);
            if (coord_it == mega_coords.end()) continue;

            CoordKey key;
            key.num_anchors = mega_num_anchors;
            for (size_t a = 0; a < coord_it->second.size() && a < MAX_ANCHORS; ++a) {
                key.coords[a] = coord_it->second[a];
            }

            auto dim_it = mega_coord_to_dim.find(key);
            if (dim_it != mega_coord_to_dim.end()) {
                result.mega_dimension[v] = dim_it->second;
                result.mega_dim_min = std::min(result.mega_dim_min, dim_it->second);
                result.mega_dim_max = std::max(result.mega_dim_max, dim_it->second);
            }
        }

        if (result.mega_dim_min > result.mega_dim_max) {
            result.mega_dim_min = 0;
            result.mega_dim_max = 0;
        }

        std::cout << "    Mega-union dimension range: [" << result.mega_dim_min
                  << ", " << result.mega_dim_max << "] for "
                  << result.mega_dimension.size() << " vertices" << std::endl;
    }

    // MEMORY OPTIMIZATION: Clear evolution data no longer needed
    // states_by_step was used for dimension computation and mega-union, now done
    // state_data_by_step was moved to result.states_per_step above
    {
        size_t cleared_bytes = 0;
        for (auto& step_graphs : evolution_result.states_by_step) {
            for (auto& graph : step_graphs) {
                cleared_bytes += graph.vertex_count() * sizeof(VertexId) * 2;  // Rough estimate
            }
            step_graphs.clear();
            step_graphs.shrink_to_fit();
        }
        evolution_result.states_by_step.clear();
        evolution_result.states_by_step.shrink_to_fit();
        evolution_result.state_data_by_step.clear();
        evolution_result.state_data_by_step.shrink_to_fit();
        std::cout << "  Memory freed: ~" << (cleared_bytes / (1024 * 1024)) << " MB (evolution data)" << std::endl;
    }

    report_progress("Analysis complete", 1.0f);

    return result;
}

// =============================================================================
// state_to_simple_graph
// =============================================================================

SimpleGraph state_to_simple_graph(
    const hypergraph::unified::UnifiedHypergraph& hg,
    hypergraph::unified::StateId state_id
) {
    const auto& state = hg.get_state(state_id);

    // Collect vertices and edges
    std::vector<VertexId> vertices;
    std::vector<Edge> edges;
    std::unordered_set<VertexId> vertex_set;

    state.edges.for_each([&](hypergraph::unified::EdgeId eid) {
        const auto& edge = hg.get_edge(eid);

        // For binary edges (arity 2), convert directly
        if (edge.arity == 2) {
            VertexId v1 = edge.vertices[0];
            VertexId v2 = edge.vertices[1];

            edges.push_back({v1, v2});

            if (vertex_set.find(v1) == vertex_set.end()) {
                vertex_set.insert(v1);
                vertices.push_back(v1);
            }
            if (vertex_set.find(v2) == vertex_set.end()) {
                vertex_set.insert(v2);
                vertices.push_back(v2);
            }
        } else {
            // For higher-arity edges, create edges between consecutive vertices
            // This is a simplification - could also do clique
            for (uint8_t i = 0; i < edge.arity; ++i) {
                VertexId v = edge.vertices[i];
                if (vertex_set.find(v) == vertex_set.end()) {
                    vertex_set.insert(v);
                    vertices.push_back(v);
                }
            }

            // Connect consecutive vertices
            for (uint8_t i = 0; i + 1 < edge.arity; ++i) {
                edges.push_back({edge.vertices[i], edge.vertices[i + 1]});
            }
        }
    });

    SimpleGraph graph;
    graph.build(vertices, edges);
    return graph;
}

} // namespace viz::blackhole
