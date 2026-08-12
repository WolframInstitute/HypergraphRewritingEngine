#pragma once
#include <gtest/gtest.h>
#include <hypergraph/ir_canonicalization.hpp>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
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

// A canonical form written out, for an assertion message that has to say WHICH two forms
// differed. It lives here because these two assertions are its only callers: as a member of
// CanonicalForm it pulled <sstream> into every translation unit that includes hypergraph.hpp,
// for a function none of them call.
inline std::string canonical_form_string(const hypergraph::CanonicalForm& cf) {
    std::ostringstream oss;
    oss << "CanonicalForm(vertices=" << cf.vertex_count << ", edges=[";
    for (std::size_t i = 0; i < cf.edges.size(); ++i) {
        oss << "[";
        for (std::size_t j = 0; j < cf.edges[i].size(); ++j) {
            oss << cf.edges[i][j];
            if (j + 1 < cf.edges[i].size()) oss << ",";
        }
        oss << "]";
        if (i + 1 < cf.edges.size()) oss << ", ";
    }
    oss << "])";
    return oss.str();
}

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
        << canonical_form_string(canon1.canonical_form) << " vs "
        << canonical_form_string(canon2.canonical_form);
}

inline void expect_canonical_different(
    const std::vector<std::vector<hypergraph::VertexId>>& edges1,
    const std::vector<std::vector<hypergraph::VertexId>>& edges2) {
    hypergraph::IRCanonicalizer canonicalizer;
    auto canon1 = canonicalizer.canonicalize_edges(edges1);
    auto canon2 = canonicalizer.canonicalize_edges(edges2);
    EXPECT_NE(canon1.canonical_form, canon2.canonical_form)
        << "Hypergraphs should have different canonical forms but both are: "
        << canonical_form_string(canon1.canonical_form);
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
 * Run WL code and return its combined stdout+stderr.
 *
 * THE ONLY WAY THIS PROJECT CONSULTS WOLFRAM, because the exit status cannot carry the
 * verdict. Two independent failures make it lie, both observed here:
 *
 *   - A Windows wolframscript.exe invoked from WSL exits non-zero on a benign license
 *     error at shutdown, AFTER the script's own Exit[0].
 *   - The WSL interop vsock intermittently times out launching it at all
 *     ("UtilAcceptVsock:251: accept4 failed 110" -- ETIMEDOUT), so the script never runs
 *     and produces no output at all.
 *
 * So a caller asserts on a MARKER the script prints. That also separates the two answers a
 * caller needs to tell apart: Wolfram disagreed (marker says so) versus Wolfram was never
 * consulted (no marker at all). Retrying the second is legitimate; retrying the first buries
 * a regression.
 */
// A path the wolframscript this suite invokes can actually open. `getWolframScriptPath` may name
// a Windows .exe reached across the WSL boundary, and such a process cannot open a POSIX path; on
// a native Linux or native Windows box the path is already right and wslpath does not exist.
// Asking wslpath and falling back is the same rule the temporary script file goes through below,
// so there is one answer to "which side is this path for" rather than two.
inline std::string hostVisiblePath(const std::string& path) {
    std::string cmd = "wslpath -w '" + path + "' 2>/dev/null";
#if defined(_MSC_VER)
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    std::string translated;
    if (pipe) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), pipe)) translated += buf;
#if defined(_MSC_VER)
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    }
    while (!translated.empty() && (translated.back() == '\n' || translated.back() == '\r'))
        translated.pop_back();
    return translated.empty() ? path : translated;
}

// The same path, escaped for embedding in Wolfram Language source. A Windows path is full of
// backslashes and WL reads `\` in a string as the start of an escape, so an unescaped one either
// changes the path or fails to parse.
inline std::string wlStringLiteralBody(const std::string& path) {
    std::string out;
    for (char c : path) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}

inline std::string executeWolframScriptCapture(const std::string& code) {
    std::string wolfram_path = getWolframScriptPath();
    std::string tmp = "/tmp/wolfram_test_" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) + ".wl";
    {
        std::ofstream f(tmp);
        f << code << std::endl;
    }
    // BOUNDED. wolframscript here is a Windows executable reached through the WSL interop
    // socket, and that socket wedges: a single hung child took one suite run from 3.5 minutes to
    // 40, because popen waits forever and nothing above it can interrupt. `timeout` bounds the
    // child, and a killed child returns no output -- which every caller already treats as "no
    // verdict" and retries, the same path a launch failure takes.
    const char* env_to = std::getenv("HG_WOLFRAM_TIMEOUT");
    const std::string secs = (env_to && *env_to) ? env_to : "90";
    std::string cmd = "timeout " + secs + " \"" + wolfram_path + "\" -file \"$(wslpath -w '" + tmp
                    + "' 2>/dev/null || echo '" + tmp + "')\" 2>&1";
    std::string output;
    // MSVC spells the POSIX pipe pair _popen/_pclose.
#if defined(_MSC_VER)
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (pipe) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), pipe)) output += buf;
#if defined(_MSC_VER)
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    }
    std::remove(tmp.c_str());
    return output;
}

} // namespace test_utils
