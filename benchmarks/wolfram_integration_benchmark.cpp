// BENCHMARK_CATEGORY: Wolfram Integration

#include "benchmark_framework.hpp"
#include <regex>
#include <sstream>
#include <chrono>

using namespace benchmark;

#if WOLFRAMSCRIPT_AVAILABLE

// =============================================================================
// Wolfram Language Integration Benchmarks
// =============================================================================

/**
 * Get WolframScript path, handling Windows/WSL conversion
 */
std::string getWolframScriptPath() {
    std::string wolfram_exe = WOLFRAMSCRIPT_EXECUTABLE;
#if defined(WSL_ENVIRONMENT) && defined(_WIN32)
    // Convert /mnt/c/... to C:/... for Windows executable
    if (wolfram_exe.find("/mnt/c/") == 0) {
        wolfram_exe = "C:" + wolfram_exe.substr(6);
    }
#endif
    return wolfram_exe;
}

/**
 * Execute Wolfram code and return raw output
 * Common implementation for all Wolfram execution functions
 */
std::string execute_wolfram_code(const std::string& full_code, const char* debug_label) {
    fprintf(stderr, "\n=== Wolfram Code%s ===\n%s\n=== End Code ===\n",
            debug_label ? debug_label : "", full_code.c_str());

    std::string wolfram_path = WOLFRAMSCRIPT_EXECUTABLE;
    std::string cmd;

#if defined(WSL_ENVIRONMENT) && defined(_WIN32)
    // Write code to temp file and use -file with UNC path to avoid quoting issues
    std::string linux_code_file = "/tmp/wolfram_bench_code_" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) + ".wl";
    std::string windows_code_file = "\\\\\\\\wsl.localhost\\\\Ubuntu\\\\tmp\\\\wolfram_bench_code_" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) + ".wl";

    std::ofstream f(linux_code_file);
    f << full_code << std::endl;
    f.close();

    cmd = "bash -c '\"" + wolfram_path + "\" -file \"" + windows_code_file + "\" 2>&1'";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::remove(linux_code_file.c_str());
        throw std::runtime_error("Failed to execute WolframScript");
    }

    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    int exit_code = pclose(pipe);
    std::remove(linux_code_file.c_str());

    fprintf(stderr, "\n=== WolframScript Output (Windows) ===\n%s\n=== End Output ===\n", output.c_str());
    fprintf(stderr, "Exit code: %d\n", exit_code);

    if (exit_code != 0) {
        throw std::runtime_error("WolframScript execution failed");
    }
#else
    // Native Linux - simple approach
    cmd = "\"" + wolfram_path + "\" -code '" + full_code + "' 2>&1";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to execute WolframScript");
    }

    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    int exit_code = pclose(pipe);

    fprintf(stderr, "\n=== WolframScript Output (Linux) ===\n%s\n=== End Output ===\n", output.c_str());
    fprintf(stderr, "Exit code: %d\n", exit_code);

    if (exit_code != 0) {
        throw std::runtime_error("WolframScript execution failed");
    }
#endif

    return output;
}

/**
 * Get paclet path with proper escaping for current platform
 */
std::string get_paclet_path() {
#if defined(WSL_ENVIRONMENT) && defined(_WIN32)
    return std::string("\\\\\\\\wsl.localhost\\\\Ubuntu") + BENCHMARK_SOURCE_DIR + "\\\\paclet";
#else
    return std::string(BENCHMARK_SOURCE_DIR) + "/paclet";
#endif
}

/**
 * Extract last line from Wolfram output
 */
std::string extract_last_line(const std::string& output) {
    std::string trimmed = output;
    size_t last_newline = trimmed.find_last_not_of("\n\r");
    if (last_newline != std::string::npos) {
        trimmed = trimmed.substr(0, last_newline + 1);
    }

    size_t last_line_start = trimmed.find_last_of('\n');
    return (last_line_start != std::string::npos)
        ? trimmed.substr(last_line_start + 1)
        : trimmed;
}

/**
 * Convert Mathematica scientific notation (*^) to standard (e)
 */
std::string convert_mathematica_notation(std::string str) {
    size_t star_pos;
    while ((star_pos = str.find("*^")) != std::string::npos) {
        str.replace(star_pos, 2, "e");
    }
    return str;
}

/**
 * Run Wolfram code and measure timing using AbsoluteTiming
 * Returns timing in microseconds
 */
double run_wolfram_timed(const std::string& hg_code) {
    std::string paclet_path = get_paclet_path();
    std::string full_code =
        "PacletDirectoryLoad[\"" + paclet_path + "\"]; "
        "<< HypergraphRewriting`;\n"
        "timing = AbsoluteTiming[" + hg_code + ";]; "
        "timing[[1]]";

    std::string output = execute_wolfram_code(full_code, "");
    std::string last_line = extract_last_line(output);
    last_line = convert_mathematica_notation(last_line);

    double seconds = std::stod(last_line);
    return seconds * 1000000.0;  // Convert to microseconds
}

/**
 * Run Wolfram code that returns a list of timings
 * Returns vector of timings in microseconds
 * The hg_code should return a Wolfram list {num1, num2, ...}
 */
std::vector<double> run_wolfram_timed_list(const std::string& hg_code) {
    std::string paclet_path = get_paclet_path();
    std::string full_code =
        "PacletDirectoryLoad[\"" + paclet_path + "\"]; "
        "<< HypergraphRewriting`;\n"
        + hg_code;

    std::string output = execute_wolfram_code(full_code, " (List)");
    std::string last_line = extract_last_line(output);

    // Parse Wolfram list format {num1, num2, num3, ...}
    size_t open_brace = last_line.find('{');
    size_t close_brace = last_line.find('}');
    if (open_brace == std::string::npos || close_brace == std::string::npos) {
        throw std::runtime_error("Expected Wolfram list format {...}, got: " + last_line);
    }

    std::string list_content = last_line.substr(open_brace + 1, close_brace - open_brace - 1);
    list_content = convert_mathematica_notation(list_content);

    // Parse comma-separated numbers
    std::vector<double> timings;
    std::istringstream stream(list_content);
    std::string token;
    while (std::getline(stream, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t\n\r");
        size_t end = token.find_last_not_of(" \t\n\r");
        if (start != std::string::npos) {
            token = token.substr(start, end - start + 1);
            double seconds = std::stod(token);
            timings.push_back(seconds * 1000000.0);  // Convert to microseconds
        }
    }

    return timings;
}

// =============================================================================
// Comparative Benchmarks: HGEvolve vs Wolfram/Multicomputation
// =============================================================================

/**
 * Run HGEvolve benchmark and compute speedup vs reference if available
 */
void run_hgevolve_benchmark(const std::string& reference_name, const std::string& rule,
                            const std::string& hg_init, int min_steps, int max_steps) {
    auto& ctx = benchmark::get_context();
    std::string base_benchmark_name = ctx.current_benchmark;

    for (int step = min_steps; step <= max_steps; ++step) {
        // Benchmark HGEvolve with adaptive sampling
        BENCHMARK_PARAM("steps", step);
        BENCHMARK_META("y_scale", "log");
        BENCHMARK_META("legend_label", "HGEvolve");
        BENCHMARK_META("reference_legend_label", "Wolfram/Multicomputation");
        BENCHMARK_CODE([&]() {
            std::string hg_code = "HGEvolve[" + rule + ", " + hg_init +
                                  ", " + std::to_string(step) + ", \"StatesGraphStructure\"]";
            double timing_us = run_wolfram_timed(hg_code);
            BENCHMARK_SUBMIT(timing_us, benchmark::ResultType::TIME);
        });

        // Load reference data and compute speedup
        std::map<std::string, std::string> params = {{"steps", std::to_string(step)}};
        auto ref_timing = benchmark::ReferenceDataLoader::instance().get_reference_timing(
            reference_name, params);

        if (!ref_timing) {
            throw std::runtime_error("Reference timing not found for " + reference_name +
                                   " with steps=" + std::to_string(step) +
                                   ". Reference benchmark may have failed or produced incomplete data.");
        }

        // Get the converged HGEvolve timing from the last result
        auto& registry = benchmark::BenchmarkRegistry::instance();
        double hg_timing_us = registry.get_last_result().avg_us;
        double speedup = *ref_timing / hg_timing_us;

        // Submit speedup as separate benchmark
        BENCHMARK_PARAM("steps", step);
        BENCHMARK_META("y_scale", "log");
        BENCHMARK_META("y_label", "Speedup (x)");
        BENCHMARK_SUBMIT_AS(base_benchmark_name + "_speedup", speedup,
                           benchmark::ResultType::RATIO);
    }
}

/**
 * Run Wolfram/Multicomputation reference benchmark
 */
void run_multicomputation_benchmark(const std::string& rule, const std::string& wm_init,
                                    int min_steps, int max_steps) {
    std::string wm_code = "Module[{warmup}, "
                          "warmup = PacletSymbol[\"Wolfram/Multicomputation\", "
                          "\"Wolfram`Multicomputation`MultiwaySystem\"][{{{1,2}}->{{1,2}}}, {{{1,2}}}]; "
                          "warmup[\"StatesGraphStructure\", 1, \"CanonicalStateFunction\" -> \"CanonicalHypergraph\"]; "
                          "Table[Module[{multi}, "
                          "multi = PacletSymbol[\"Wolfram/Multicomputation\", "
                          "\"Wolfram`Multicomputation`MultiwaySystem\"][" + rule + ", " + wm_init + "]; "
                          "AbsoluteTiming[multi[\"StatesGraphStructure\", step, "
                          "\"CanonicalStateFunction\" -> \"CanonicalHypergraph\"];][[1]]], {step, " +
                          std::to_string(min_steps) + ", " + std::to_string(max_steps) + "}]]";
    std::vector<double> wm_timings = run_wolfram_timed_list(wm_code);

    for (size_t i = 0; i < wm_timings.size(); ++i) {
        int step = min_steps + i;
        BENCHMARK_PARAM("steps", step);
        BENCHMARK_SUBMIT(std::vector<double>{wm_timings[i]});
    }
}

// Regular benchmarks - HGEvolve only, with speedup computed from reference data
BENCHMARK_WITH_REFERENCE(comparative_config1, "HGEvolve benchmark (Config1) with speedup vs Wolfram/Multicomputation", "comparative_config1_reference") {
    run_hgevolve_benchmark(
        "comparative_config1_reference",
        "{{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}}",
        "{{1, 2}, {1, 3}}",
        1, 4
    );
}

BENCHMARK_WITH_REFERENCE(comparative_config2, "HGEvolve benchmark (Config2) with speedup vs Wolfram/Multicomputation", "comparative_config2_reference") {
    run_hgevolve_benchmark(
        "comparative_config2_reference",
        "{{{1, 2}, {2, 3}} -> {{1, 2}, {1, 4}, {2, 4}}}",
        "{{1, 2}, {2, 3}}",
        1, 5
    );
}

BENCHMARK_WITH_REFERENCE(comparative_config3, "HGEvolve benchmark (Config3) with speedup vs Wolfram/Multicomputation", "comparative_config3_reference") {
    run_hgevolve_benchmark(
        "comparative_config3_reference",
        "{{{1, 2, 3}, {5, 1}} -> {{1, 5, 6}, {3, 2}, {3, 5}}}",
        "{{1, 1, 1}, {1, 1}}",
        1, 8
    );
}

/**
 * Run 2D HGEvolve benchmark (graph_edges × steps) and compute speedup vs reference
 */
void run_hgevolve_benchmark_2d(const std::string& reference_name, const std::string& rule,
                                int min_edges, int max_edges, int min_steps, int max_steps) {
    auto& ctx = benchmark::get_context();
    std::string base_benchmark_name = ctx.current_benchmark;

    for (int n_edges = min_edges; n_edges <= max_edges; ++n_edges) {
        for (int step = min_steps; step <= max_steps; ++step) {
            // Generate initial state: Table[{i, i + 1}, {i, graph_edges}]
            std::string hg_init = "{";
            for (int i = 1; i <= n_edges; ++i) {
                if (i > 1) hg_init += ", ";
                hg_init += "{" + std::to_string(i) + ", " + std::to_string(i + 1) + "}";
            }
            hg_init += "}";

            // Benchmark HGEvolve with adaptive sampling
            BENCHMARK_PARAM("graph_edges", n_edges);
            BENCHMARK_PARAM("steps", step);
            BENCHMARK_META("y_scale", "log");
            BENCHMARK_META("legend_label", "HGEvolve");
            BENCHMARK_META("reference_legend_label", "Wolfram/Multicomputation");
            BENCHMARK_CODE([&]() {
                std::string hg_code = "HGEvolve[" + rule + ", " + hg_init +
                                      ", " + std::to_string(step) + ", \"StatesGraphStructure\"]";
                double timing_us = run_wolfram_timed(hg_code);
                BENCHMARK_SUBMIT(timing_us, benchmark::ResultType::TIME);
            });

            // Load reference data and compute speedup
            std::map<std::string, std::string> params = {
                {"graph_edges", std::to_string(n_edges)},
                {"steps", std::to_string(step)}
            };
            auto ref_timing = benchmark::ReferenceDataLoader::instance().get_reference_timing(
                reference_name, params);

            if (!ref_timing) {
                throw std::runtime_error("Reference timing not found for " + reference_name +
                                       " with graph_edges=" + std::to_string(n_edges) +
                                       ", steps=" + std::to_string(step) +
                                       ". Reference benchmark may have failed or produced incomplete data.");
            }

            // Get the converged HGEvolve timing from the last result
            auto& registry = benchmark::BenchmarkRegistry::instance();
            double hg_timing_us = registry.get_last_result().avg_us;
            double speedup = *ref_timing / hg_timing_us;

            // Submit speedup as separate benchmark
            BENCHMARK_PARAM("graph_edges", n_edges);
            BENCHMARK_PARAM("steps", step);
            BENCHMARK_META("z_scale", "log");
            BENCHMARK_META("z_label", "Speedup (x)");
            BENCHMARK_SUBMIT_AS(base_benchmark_name + "_speedup", speedup,
                               benchmark::ResultType::RATIO);
        }
    }
}

/**
 * Run 2D Wolfram/Multicomputation reference benchmark (graph_edges × steps)
 */
void run_multicomputation_benchmark_2d(const std::string& rule,
                                        int min_edges, int max_edges,
                                        int min_steps, int max_steps) {
    for (int n_edges = min_edges; n_edges <= max_edges; ++n_edges) {
        // Generate initial state for Multicomputation: {Table[{i, i + 1}, {i, graph_edges}]}
        std::string wm_init = "{{";
        for (int i = 1; i <= n_edges; ++i) {
            if (i > 1) wm_init += ", ";
            wm_init += "{" + std::to_string(i) + ", " + std::to_string(i + 1) + "}";
        }
        wm_init += "}}";

        // Generate Wolfram code for all steps at this edge count
        std::string wm_code = "Module[{warmup}, "
                              "warmup = PacletSymbol[\"Wolfram/Multicomputation\", "
                              "\"Wolfram`Multicomputation`MultiwaySystem\"][{{{1,2}}->{{1,2}}}, {{{1,2}}}]; "
                              "warmup[\"StatesGraphStructure\", 1, \"CanonicalStateFunction\" -> \"CanonicalHypergraph\"]; "
                              "Table[Module[{multi}, "
                              "multi = PacletSymbol[\"Wolfram/Multicomputation\", "
                              "\"Wolfram`Multicomputation`MultiwaySystem\"][" + rule + ", " + wm_init + "]; "
                              "AbsoluteTiming[multi[\"StatesGraphStructure\", step, "
                              "\"CanonicalStateFunction\" -> \"CanonicalHypergraph\"];][[1]]], {step, " +
                              std::to_string(min_steps) + ", " + std::to_string(max_steps) + "}]]";
        std::vector<double> wm_timings = run_wolfram_timed_list(wm_code);

        for (size_t i = 0; i < wm_timings.size(); ++i) {
            int step = min_steps + i;
            BENCHMARK_PARAM("graph_edges", n_edges);
            BENCHMARK_PARAM("steps", step);
            BENCHMARK_SUBMIT(std::vector<double>{wm_timings[i]});
        }
    }
}

// Reference benchmarks - Wolfram/Multicomputation only
BENCHMARK_REFERENCE(comparative_config1_reference, "Wolfram/Multicomputation reference (Config1)") {
    run_multicomputation_benchmark(
        "{{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}}",
        "{{{1, 2}, {1, 3}}}",
        1, 4
    );
}

BENCHMARK_REFERENCE(comparative_config2_reference, "Wolfram/Multicomputation reference (Config2)") {
    run_multicomputation_benchmark(
        "{{{1, 2}, {2, 3}} -> {{1, 2}, {1, 4}, {2, 4}}}",
        "{{{1, 2}, {2, 3}}}",
        1, 5
    );
}

BENCHMARK_REFERENCE(comparative_config3_reference, "Wolfram/Multicomputation reference (Config3)") {
    run_multicomputation_benchmark(
        "{{{1, 2, 3}, {5, 1}} -> {{1, 5, 6}, {3, 2}, {3, 5}}}",
        "{{{1, 1, 1}, {1, 1}}}",
        1, 8
    );
}

BENCHMARK_REFERENCE(comparative_2d_edges_steps_reference, "Wolfram/Multicomputation reference 2D (Edges × Steps)") {
    run_multicomputation_benchmark_2d(
        "{{{1, 2}, {2, 3}} -> {{1, 2}, {1, 4}, {2, 4}}}",
        2, 6,  // graph_edges: 2-6
        1, 5   // steps: 1-5
    );
}

// 2D comparative benchmark
BENCHMARK_WITH_REFERENCE(comparative_2d_edges_steps, "HGEvolve 2D benchmark (Edges × Steps) with speedup vs Wolfram/Multicomputation", "comparative_2d_edges_steps_reference") {
    run_hgevolve_benchmark_2d(
        "comparative_2d_edges_steps_reference",
        "{{{1, 2}, {2, 3}} -> {{1, 2}, {1, 4}, {2, 4}}}",
        2, 6,  // graph_edges: 2-6
        1, 5   // steps: 1-5
    );
}

#endif // WOLFRAMSCRIPT_AVAILABLE
