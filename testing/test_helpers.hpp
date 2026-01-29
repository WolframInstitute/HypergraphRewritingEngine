#pragma once
#include <gtest/gtest.h>
#include <hypergraph/ir_canonicalization.hpp>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <cstdio>

namespace test_utils {

/**
 * Performance measurement utility
 */
class PerfTimer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    PerfTimer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

/**
 * Expectation helpers for hypergraph comparisons using edge vectors
 */
inline void expect_canonical_equal(
    const std::vector<std::vector<hypergraph::VertexId>>& edges1,
    const std::vector<std::vector<hypergraph::VertexId>>& edges2) {
    hypergraph::IRCanonicalizer canonicalizer;
    auto canon1 = canonicalizer.canonicalize_edges(edges1);
    auto canon2 = canonicalizer.canonicalize_edges(edges2);
    EXPECT_EQ(canon1.canonical_form, canon2.canonical_form)
        << "Hypergraphs should have same canonical form: "
        << canon1.canonical_form.to_string() << " vs " << canon2.canonical_form.to_string();
}

inline void expect_canonical_different(
    const std::vector<std::vector<hypergraph::VertexId>>& edges1,
    const std::vector<std::vector<hypergraph::VertexId>>& edges2) {
    hypergraph::IRCanonicalizer canonicalizer;
    auto canon1 = canonicalizer.canonicalize_edges(edges1);
    auto canon2 = canonicalizer.canonicalize_edges(edges2);
    EXPECT_NE(canon1.canonical_form, canon2.canonical_form)
        << "Hypergraphs should have different canonical forms but both are: "
        << canon1.canonical_form.to_string();
}

/**
 * Convert WSL paths to Windows paths when cross-compiling for Windows
 * This ensures Windows executables can find the correct paths
 */
inline std::string getWolframScriptPath() {
#if WOLFRAMSCRIPT_AVAILABLE
    std::string wolfram_exe = WOLFRAMSCRIPT_EXECUTABLE;
#if defined(WSL_ENVIRONMENT) && defined(_WIN32)
    // Convert /mnt/c/... to C:/... for Windows executable
    // Keep forward slashes - Windows accepts them and they avoid escaping issues
    if (wolfram_exe.find("/mnt/c/") == 0) {
        wolfram_exe = "C:" + wolfram_exe.substr(6);
        // Don't convert to backslashes - keep forward slashes!
    }
#endif
    return wolfram_exe;
#else
    throw std::runtime_error("WolframScript not available - cannot get WolframScript path");
#endif
}

/**
 * Execute WolframScript with proper path handling and quoting
 * Uses bash -c to execute WolframScript from WSL cross-compiled environment
 */
inline int executeWolframScript(const std::string& code) {
#if defined(WSL_ENVIRONMENT) && defined(_WIN32)
    // Use bash -c with original /mnt/c path - avoid Windows path conversion
    std::string wolfram_path = WOLFRAMSCRIPT_EXECUTABLE;  // Use original path

    // For complex commands, write to temporary file to avoid quoting issues
    if (code.find("\"") != std::string::npos) {
        // Create temporary file that WolframScript can access via UNC path
        std::string linux_temp_file = "/tmp/wolfram_test_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".wl";
        std::string windows_temp_file = "\\\\\\\\wsl.localhost\\\\Ubuntu\\\\tmp\\\\wolfram_test_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".wl";
        // Write to Linux temp file
        std::ofstream f(linux_temp_file);
        f << code << std::endl;
        f.close();

        // Use Windows UNC path for WolframScript to access
        std::string cmd = "bash -c '\"" + wolfram_path + "\" -file \"" + windows_temp_file + "\"'";
        int result = std::system(cmd.c_str());
        std::remove(linux_temp_file.c_str());
        return result;
    } else {
        // Simple command without quotes - use direct -code
        std::string cmd = "bash -c '\"" + wolfram_path + "\" -code \"" + code + "\"'";
        return std::system(cmd.c_str());
    }
#else
    // Native Linux or Windows - use simple quoting
    std::string wolfram_path = getWolframScriptPath();
    std::string cmd = "\"" + wolfram_path + "\" -code \"" + code + "\"";
    return std::system(cmd.c_str());
#endif
}

} // namespace test_utils
