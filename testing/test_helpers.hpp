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
    PerfTimer();

    double elapsed_ms() const;

    void reset();
};

// A canonical form written out, for an assertion message that has to say WHICH two forms
// differed. It lives here because these two assertions are its only callers: as a member of
// CanonicalForm it pulled <sstream> into every translation unit that includes hypergraph.hpp,
// for a function none of them call.
std::string canonical_form_string(const hypergraph::CanonicalForm& cf);

/**
 * Expectation helpers for hypergraph comparisons using edge vectors
 */
void expect_canonical_equal( const std::vector<std::vector<hypergraph::VertexId>>& edges1, const std::vector<std::vector<hypergraph::VertexId>>& edges2);void expect_canonical_different( const std::vector<std::vector<hypergraph::VertexId>>& edges1, const std::vector<std::vector<hypergraph::VertexId>>& edges2);

/**
 * Convert WSL paths to Windows paths when cross-compiling for Windows
 * This ensures Windows executables can find the correct paths
 */
std::string getWolframScriptPath();

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
std::string hostVisiblePath(const std::string& path);

// The same path, escaped for embedding in Wolfram Language source. A Windows path is full of
// backslashes and WL reads `\` in a string as the start of an escape, so an unescaped one either
// changes the path or fails to parse.
std::string wlStringLiteralBody(const std::string& path);std::string executeWolframScriptCapture(const std::string& code);

} // namespace test_utils
