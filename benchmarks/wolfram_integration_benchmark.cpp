// BENCHMARK_CATEGORY: Wolfram Integration

#include "benchmark_framework.hpp"
#include <regex>

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
 * Run Wolfram code and measure timing using AbsoluteTiming
 * Returns timing in microseconds
 */
double run_wolfram_timed(const std::string& hg_code) {
#if defined(WSL_ENVIRONMENT) && defined(_WIN32)
    // Use UNC path for paclet directory on Windows
    std::string paclet_path = std::string("\\\\\\\\wsl.localhost\\\\Ubuntu") + BENCHMARK_SOURCE_DIR + "\\\\paclet";
#else
    std::string paclet_path = std::string(BENCHMARK_SOURCE_DIR) + "/paclet";
#endif

    std::string full_code =
        "PacletDirectoryLoad[\"" + paclet_path + "\"]; "
        "<< HypergraphRewriting`;\n"
        "timing = AbsoluteTiming[" + hg_code + ";]; "
        "Print[timing[[1]]]";

    // Debug: print the code we're executing
    fprintf(stderr, "\n=== Wolfram Code ===\n%s\n=== End Code ===\n", full_code.c_str());

    std::string wolfram_path = WOLFRAMSCRIPT_EXECUTABLE;
    std::string cmd;

#if defined(WSL_ENVIRONMENT) && defined(_WIN32)
    // Write code to temp file and use -file with UNC path to avoid quoting issues
    std::string linux_code_file = "/tmp/wolfram_bench_code_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".wl";
    std::string windows_code_file = "\\\\\\\\wsl.localhost\\\\Ubuntu\\\\tmp\\\\wolfram_bench_code_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".wl";

    // Write code to Linux file
    std::ofstream f(linux_code_file);
    f << full_code << std::endl;
    f.close();

    // Execute using bash and capture stdout
    cmd = "bash -c '\"" + wolfram_path + "\" -file \"" + windows_code_file + "\" 2>&1'";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::remove(linux_code_file.c_str());
        throw std::runtime_error("Failed to execute WolframScript");
    }

    // Read output
    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    int exit_code = pclose(pipe);
    std::remove(linux_code_file.c_str());

    // Debug: print full output
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

    // Read output
    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    int exit_code = pclose(pipe);

    // Debug: print full output
    fprintf(stderr, "\n=== WolframScript Output (Linux) ===\n%s\n=== End Output ===\n", output.c_str());
    fprintf(stderr, "Exit code: %d\n", exit_code);

    if (exit_code != 0) {
        throw std::runtime_error("WolframScript execution failed");
    }
#endif

    // Parse the timing value - look for the last line with a number
    size_t last_newline = output.find_last_not_of("\n\r");
    if (last_newline != std::string::npos) {
        output = output.substr(0, last_newline + 1);
    }

    size_t last_line_start = output.find_last_of('\n');
    std::string last_line = (last_line_start != std::string::npos)
        ? output.substr(last_line_start + 1)
        : output;

    // Convert Mathematica scientific notation (*^) to standard (e)
    size_t star_pos = last_line.find("*^");
    if (star_pos != std::string::npos) {
        last_line.replace(star_pos, 2, "e");
    }

    double seconds = std::stod(last_line);
    return seconds * 1000000.0;  // Convert to microseconds
}

BENCHMARK(wolfram_evolution_by_steps) {
    for (int steps : {1, 2, 3, 4, 5}) {
        BENCHMARK_PARAM("steps", steps);

        std::string code = "HGEvolve["
                "{{{1, 2}, {2, 3}} -> {{3, 2}, {2, 1}, {1, 4}}}, "  // Rule
                "{{1, 1}, {1, 1}}, "                                // Initial state
                + std::to_string(steps) + ", "                      // Steps
                "\"StatesGraph\""                                   // Property
            "]";

        std::vector<double> timings;
        for (int i = 0; i < 2; ++i) {
            timings.push_back(run_wolfram_timed(code));
        }

        BENCHMARK_SUBMIT(timings);
    }
}

#endif // WOLFRAMSCRIPT_AVAILABLE
