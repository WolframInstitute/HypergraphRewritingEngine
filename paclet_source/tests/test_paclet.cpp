#include <gtest/gtest.h>
#include "test_helpers.hpp"

/**
 * Paclet Integration Tests
 *
 * These tests verify that the Wolfram Language paclet works end-to-end,
 * testing the full pipeline: WXF serialization → FFI → hypergraph evolution → WXF deserialization
 */

class PacletTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup if needed
    }

    void TearDown() override {
        // Test cleanup if needed
    }
};

// ============================================================================
// PACLET INTEGRATION TESTS
// ============================================================================

#if BUILD_WOLFRAM_LANGUAGE_PACLET && WOLFRAMSCRIPT_AVAILABLE

// Can wolframscript be reached at all? Every other test in this file is meaningless if not,
// so this is the precondition, checked once.
//
// It asserts on PRINTED OUTPUT, not on the exit status. A Windows wolframscript.exe invoked
// from WSL exits non-zero on a benign license error at shutdown even after Exit[0] --
// executeWolframScriptCapture's own contract says so -- and the WSL interop vsock
// intermittently times out launching it at all ("UtilAcceptVsock:251: accept4 failed 110",
// captured in a WXFTest failure). Reading the exit code turns both into a red suite.
TEST_F(PacletTest, WolframScriptIsReachable) {
    const std::string out = test_utils::executeWolframScriptCapture("Print[6*7]");
    EXPECT_NE(out.find("42"), std::string::npos)
        << "wolframscript did not evaluate a trivial expression, so every check in this file "
        << "that depends on it proves nothing. Its output was:\n" << out;
}

TEST_F(PacletTest, TestPacletBasicFunctionality) {
    // Test that the paclet loads and basic functions work
    std::string paclet_dir;

#if defined(WSL_ENVIRONMENT) && defined(_WIN32)
    // WSL cross-compilation - use UNC path so WolframScript can access it
    paclet_dir = "\\\\\\\\wsl.localhost\\\\Ubuntu\\\\home\\\\fly\\\\my_projects\\\\efficient_rewriting_final\\\\paclet";
#else
    // Native Windows or Linux - use relative path
    paclet_dir = "../paclet";
#endif

    std::string code = "Print[\"Loading paclet from: " + paclet_dir + "\"]; "
                      "PacletDirectoryLoad[\"" + paclet_dir + "\"]; "
                      "Print[\"Loading HypergraphRewriting package...\"]; "
                      "<< \"HypergraphRewriting`\"; "
                      "Print[\"Testing HGEvolve with Debug option...\"]; "
                      "Print[\"About to call HGEvolve...\"]; "
                      "result = HypergraphRewriting`HGEvolve[{{{1, 2}, {2, 3}} -> {{3, 2}, {2, 1}, {1, 4}}}, {{1, 2}, {2, 3}}, 4, \"Debug\"]; "
                      "Print[\"HGEvolve completed\"]; "
                      "Print[\"Result: \", result]; "
                      "Print[\"Result type: \", Head[result]]; "
                      "If[AssociationQ[result], Print[\"PACLET_TEST_OK keys: \", Keys[result]], Print[\"PACLET_TEST_FAIL: \", result]]";

    // Assert on a printed success marker, not the process exit code: a Windows
    // wolframscript.exe invoked from WSL exits with a benign license error at
    // shutdown, so the exit status is unreliable. The marker is printed only after
    // HGEvolve returns a valid Debug association.
    // Retry only a VERDICTLESS run, and never a verdict.
    //
    // The script prints exactly one of two markers, so their joint absence means it never got far
    // enough to evaluate HGEvolve at all -- a Windows wolframscript.exe invoked from WSL
    // intermittently fails to acquire a licence and produces no output. That is the environment,
    // not the paclet, and retrying it is what stops an unrelated flake from reading as a
    // regression.
    //
    // PACLET_TEST_FAIL is a verdict: HGEvolve ran and returned the wrong thing. It is never
    // retried, because retrying a real defect until it passes is how a suite stops meaning
    // anything.
    constexpr int kMaxAttempts = 3;
    std::string out;
    int attempts = 0;
    for (; attempts < kMaxAttempts; ++attempts) {
        out = test_utils::executeWolframScriptCapture(code);
        const bool verdict = out.find("PACLET_TEST_OK") != std::string::npos ||
                             out.find("PACLET_TEST_FAIL") != std::string::npos;
        if (verdict) break;
    }

    EXPECT_NE(out.find("PACLET_TEST_OK"), std::string::npos)
        << "Paclet basic functionality test failed after " << (attempts + 1) << " attempt(s) - "
           "HGEvolve did not return a Debug association. If neither marker appears at all, "
           "wolframscript never evaluated it. WolframScript output:\n" << out;
    EXPECT_EQ(out.find("PACLET_TEST_FAIL"), std::string::npos)
        << "HGEvolve returned a non-association result. WolframScript output:\n" << out;
}

#else

TEST_F(PacletTest, SkipPacletTests) {
    GTEST_SKIP() << "Paclet tests require BUILD_WOLFRAM_LANGUAGE_PACLET and WOLFRAMSCRIPT_AVAILABLE";
}

#endif // BUILD_WOLFRAM_LANGUAGE_PACLET && WOLFRAMSCRIPT_AVAILABLE