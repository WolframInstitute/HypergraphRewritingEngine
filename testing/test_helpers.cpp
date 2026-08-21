#include "test_helpers.hpp"

// The bodies behind test_helpers.hpp. The header is included by every test target's sources, so
// each of them was compiling the WolframScript shell-out and the canonical-form comparisons.
// Nothing here is a template.

namespace test_utils {

std::string canonical_form_string(const hypergraph::CanonicalForm& cf) {
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

void expect_canonical_equal(
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

void expect_canonical_different(
    const std::vector<std::vector<hypergraph::VertexId>>& edges1,
    const std::vector<std::vector<hypergraph::VertexId>>& edges2) {
    hypergraph::IRCanonicalizer canonicalizer;
    auto canon1 = canonicalizer.canonicalize_edges(edges1);
    auto canon2 = canonicalizer.canonicalize_edges(edges2);
    EXPECT_NE(canon1.canonical_form, canon2.canonical_form)
        << "Hypergraphs should have different canonical forms but both are: "
        << canonical_form_string(canon1.canonical_form);
}

std::string getWolframScriptPath() {
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

std::string hostVisiblePath(const std::string& path) {
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

std::string wlStringLiteralBody(const std::string& path) {
    std::string out;
    for (char c : path) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}

std::string executeWolframScriptCapture(const std::string& code) {
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


// A wall-clock stopwatch for the tests that report timings.

PerfTimer::PerfTimer(): start_(std::chrono::high_resolution_clock::now()) {}

double PerfTimer::elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

void PerfTimer::reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

}  // namespace test_utils
