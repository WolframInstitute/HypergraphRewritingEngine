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
// A LAUNCH FAILURE IS RETRIED; A WRONG ANSWER IS NOT. The same rule the paclet test below
// follows, and this one needs it for the same reason: the vsock timeout means wolframscript
// never started, so there is no verdict to respect, and one transient reds a suite whose other
// 255 tests passed. Observed: this test failed on `accept4 failed 110` in a run where the paclet
// test immediately after it -- which does retry -- passed.
//
// If wolframscript DOES run and prints something other than 42, that is a verdict and it stands.
TEST_F(PacletTest, WolframScriptIsReachable) {
    constexpr int kMaxAttempts = 3;
    std::string out;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        out = test_utils::executeWolframScriptCapture("Print[6*7]");
        // It ran if it produced anything that is not purely the interop's own complaint.
        const bool never_launched =
            out.empty() || out.find("UtilAcceptVsock") != std::string::npos;
        if (!never_launched) break;
    }
    EXPECT_NE(out.find("42"), std::string::npos)
        << "wolframscript did not evaluate a trivial expression in " << kMaxAttempts
        << " attempts, so every check in this file that depends on it proves nothing. Its last "
           "output was:\n" << out;
}

TEST_F(PacletTest, TestPacletBasicFunctionality) {
    // The paclet is found from where the REPOSITORY is, which CMake bakes in at configure time,
    // translated to whichever side of the WSL boundary the wolframscript being invoked runs on
    // and escaped for the WL string it goes into. A path relative to the working directory
    // resolves against whatever directory the caller happened to be in, so the same binary
    // passed from one and failed from another; a path naming one machine's home directory
    // passes only on that machine.
    const std::string paclet_dir =
        test_utils::wlStringLiteralBody(test_utils::hostVisiblePath(
            std::string(HG_SOURCE_DIR) + "/paclet"));

    std::string code = "Print[\"Loading paclet from: " + paclet_dir + "\"]; "
                      "PacletDirectoryLoad[\"" + paclet_dir + "\"]; "
                      "Print[\"Loading HypergraphRewriting package...\"]; "
                      "<< \"HypergraphRewriting`\"; "
                      // Separate "the package is not loaded" from "HGEvolve returned the wrong
                      // thing". Without this the two are indistinguishable downstream: an
                      // unloaded package leaves HGEvolve undefined, so it returns UNEVALUATED,
                      // which is not an association -- and the run reports the same
                      // PACLET_TEST_FAIL a real defect would, pointing the reader at the engine.
                      "If[!MemberQ[$Packages, \"HypergraphRewriting`\"], "
                      "Print[\"PACLET_LOAD_FAIL: package not on $Packages after loading "
                      + paclet_dir + "\"]; Exit[0]]; "
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
                             out.find("PACLET_TEST_FAIL") != std::string::npos ||
                             out.find("PACLET_LOAD_FAIL") != std::string::npos;
        if (verdict) break;
    }

    // Reported before the functional assertions, because when the package did not load they
    // cannot mean anything and their message would send the reader after the engine instead.
    ASSERT_EQ(out.find("PACLET_LOAD_FAIL"), std::string::npos)
        << "The paclet did not load, so nothing below was exercised. It was looked for at `"
        << paclet_dir << "`, an absolute path from the configured source tree; if that directory "
           "holds no built paclet, build the `paclet` target. WolframScript output:\n"
        << out;

    EXPECT_NE(out.find("PACLET_TEST_OK"), std::string::npos)
        << "Paclet basic functionality test failed after " << (attempts + 1) << " attempt(s) - "
           "HGEvolve did not return a Debug association. If neither marker appears at all, "
           "wolframscript never evaluated it. WolframScript output:\n" << out;
    EXPECT_EQ(out.find("PACLET_TEST_FAIL"), std::string::npos)
        << "HGEvolve returned a non-association result. WolframScript output:\n" << out;
}

// THE ENGINE BINARY IS A ROUTE ON ITS OWN, and this is the only test that says so.
//
// HGEvolve prefers the persistent worker, then a one-shot RunProcess of the engine binary, and
// falls back to the in-process LibraryLink last. Every other test here runs where BOTH are
// present, so a dispatch that quietly requires the library passes all of them -- which is what
// happened: HGEvolve refused every call on a platform shipping hg_evolve and no library, and
// the Linux CI that builds both could not see it.
//
// Block'ing performRewriting to a plain symbol is exactly the state FindLibrary leaves it in
// when the platform has no library, and it does it without removing a file the other tests in
// this process need.
TEST_F(PacletTest, EvolveWorksThroughTheBinaryWhenNoLibraryIsLoaded) {
    const std::string paclet_dir =
        test_utils::wlStringLiteralBody(test_utils::hostVisiblePath(
            std::string(HG_SOURCE_DIR) + "/paclet"));

    const std::string code =
        "PacletDirectoryLoad[\"" + paclet_dir + "\"]; "
        "<< \"HypergraphRewriting`\"; "
        "If[!MemberQ[$Packages, \"HypergraphRewriting`\"], Print[\"NOLIB_LOAD_FAIL\"]; Exit[0]]; "
        // Nothing to prove if this build ships no binary either: then the library IS the only
        // route and refusing without it is correct.
        "If[!HypergraphRewriting`Private`hgCpuBinaryAvailableQ[], Print[\"NOLIB_SKIP: no engine "
        "binary for \", $SystemID]; Exit[0]]; "
        "r = Block[{HypergraphRewriting`Private`performRewriting = Null}, "
        "  HypergraphRewriting`HGEvolve[{{{1, 2}, {2, 3}} -> {{3, 2}, {2, 1}, {1, 4}}}, "
        "                               {{1, 2}, {2, 3}}, 4, \"Debug\"]]; "
        "If[AssociationQ[r] && IntegerQ[r[\"NumStates\"]] && r[\"NumStates\"] > 0, "
        "  Print[\"NOLIB_OK states=\", r[\"NumStates\"]], Print[\"NOLIB_FAIL: \", r]]";

    constexpr int kMaxAttempts = 3;
    std::string out;
    for (int attempts = 0; attempts < kMaxAttempts; ++attempts) {
        out = test_utils::executeWolframScriptCapture(code);
        const bool verdict = out.find("NOLIB_OK") != std::string::npos ||
                             out.find("NOLIB_FAIL") != std::string::npos ||
                             out.find("NOLIB_SKIP") != std::string::npos ||
                             out.find("NOLIB_LOAD_FAIL") != std::string::npos;
        if (verdict) break;
    }

    if (out.find("NOLIB_SKIP") != std::string::npos) {
        GTEST_SKIP() << "this build ships no engine binary for this platform, so the library is "
                        "the only route: " << out;
    }
    ASSERT_EQ(out.find("NOLIB_LOAD_FAIL"), std::string::npos)
        << "the paclet did not load, so nothing below was exercised:\n" << out;
    EXPECT_NE(out.find("NOLIB_OK"), std::string::npos)
        << "HGEvolve did not evolve with the library absent, though an engine binary is present. "
           "The dispatch is requiring a route it documents as the LAST fallback. WolframScript "
           "output:\n" << out;
}

// NO CALL OPENS A SOCKET: Wolfram 15.0's SocketConnect can end the kernel on Windows.
// reference/no_socket_check.wl runs HGEvolve and every session verb on the CPU and on the GPU and
// compares Sockets[] before and after each call.
TEST_F(PacletTest, NoCallOpensASocket) {
    const std::string paclet_dir =
        test_utils::wlStringLiteralBody(test_utils::hostVisiblePath(
            std::string(HG_SOURCE_DIR) + "/paclet"));
    const std::string check_file =
        test_utils::wlStringLiteralBody(test_utils::hostVisiblePath(
            std::string(HG_SOURCE_DIR) + "/reference/no_socket_check.wl"));
    const std::string code =
        "Get[\"" + check_file + "\"]; hgNoSocketCheck[\"" + paclet_dir + "\"]";

    constexpr int kMaxAttempts = 3;
    std::string out;
    for (int attempts = 0; attempts < kMaxAttempts; ++attempts) {
        out = test_utils::executeWolframScriptCapture(code);
        const bool verdict = out.find("NOSOCK_OK") != std::string::npos ||
                             out.find("NOSOCK_FAIL") != std::string::npos ||
                             out.find("NOSOCK_LOAD_FAIL") != std::string::npos;
        if (verdict) break;
    }
    ASSERT_EQ(out.find("NOSOCK_LOAD_FAIL"), std::string::npos)
        << "the paclet did not load, so nothing below was exercised:\n" << out;
    EXPECT_NE(out.find("NOSOCK_OK"), std::string::npos)
        << "a paclet call opened a socket. WolframScript output:\n" << out;
}

// A WORKER THAT DOES NOT ANSWER THE HANDSHAKE IS NOT USED. With no time allowed for the handshake
// (as on a Wolfram version whose process streams do not carry the bytes), hgWorkerStart kills the
// worker, the CPU transport is marked broken, and HGEvolve runs the job in a one-shot RunProcess
// and returns its result.
TEST_F(PacletTest, AWorkerWithoutAHandshakeSendsCallsToRunProcess) {
    const std::string paclet_dir =
        test_utils::wlStringLiteralBody(test_utils::hostVisiblePath(
            std::string(HG_SOURCE_DIR) + "/paclet"));
    const std::string code =
        "PacletDirectoryLoad[\"" + paclet_dir + "\"]; "
        "<< \"HypergraphRewriting`\"; "
        "If[!MemberQ[$Packages, \"HypergraphRewriting`\"], Print[\"SHAKE_LOAD_FAIL\"]; Exit[0]]; "
        "HypergraphRewriting`Private`$hgHandshakeSeconds = 0; "
        "n = HypergraphRewriting`HGEvolve[{{{1, 2}, {2, 3}} -> {{3, 2}, {2, 1}, {1, 4}}}, "
        "  {{1, 2}, {2, 3}}, 4, \"NumStates\", \"Tuning\" -> <|\"MemoryLimit\" -> 2^30|>]; "
        "b = TrueQ[HypergraphRewriting`Private`$hgWorkerBroken[\"CPU\"]]; "
        "w = KeyExistsQ[HypergraphRewriting`Private`$hgWorkerProc, \"CPU\"]; "
        "If[n === 27 && b && !w, Print[\"SHAKE_OK\"], Print[\"SHAKE_FAIL: \", {n, b, w}]]";
    constexpr int kMaxAttempts = 3;
    std::string out;
    for (int attempts = 0; attempts < kMaxAttempts; ++attempts) {
        out = test_utils::executeWolframScriptCapture(code);
        if (out.find("SHAKE_OK") != std::string::npos ||
            out.find("SHAKE_FAIL") != std::string::npos ||
            out.find("SHAKE_LOAD_FAIL") != std::string::npos)
            break;
    }
    ASSERT_EQ(out.find("SHAKE_LOAD_FAIL"), std::string::npos) << out;
    EXPECT_NE(out.find("SHAKE_OK"), std::string::npos) << "WolframScript output:\n" << out;
}

// THE WORKER TRANSPORT: hg_evolve --serve on the process's standard streams, no socket. Two calls
// use one worker; "ShowProgress" prints the worker's progress frames; a worker killed during a
// job gives HGEvolve::workerlost and $Failed, and the next call starts a new worker and returns.
// The killer is a separate process that ends the worker by its PID once the worker has used 0.5 s
// of CPU time, which only the job does, during a call of about three seconds (Wolfram's rule,
// 6 steps, Full states, one thread).
TEST_F(PacletTest, TheWorkerServesOverItsStandardStreams) {
    const std::string paclet_dir =
        test_utils::wlStringLiteralBody(test_utils::hostVisiblePath(
            std::string(HG_SOURCE_DIR) + "/paclet"));
    const std::string code =
        "PacletDirectoryLoad[\"" + paclet_dir + "\"]; "
        "<< \"HypergraphRewriting`\"; "
        "If[!MemberQ[$Packages, \"HypergraphRewriting`\"], Print[\"PIPE_LOAD_FAIL\"]; Exit[0]]; "
        "If[!HypergraphRewriting`Private`hgCpuBinaryAvailableQ[], Print[\"PIPE_SKIP: no engine "
        "binary for \", $SystemID]; Exit[0]]; "
        "gen := Lookup[HypergraphRewriting`Private`$hgWorkerGeneration, \"CPU\", None]; "
        "small := HypergraphRewriting`HGEvolve[{{{1, 2}, {2, 3}} -> {{3, 2}, {2, 1}, {1, 4}}}, "
        "  {{1, 2}, {2, 3}}, 4, \"NumStates\", \"Tuning\" -> <|\"MemoryLimit\" -> 2^30|>]; "
        "n1 = small; g1 = gen; n2 = small; g2 = gen; "
        "HypergraphRewriting`HGEvolve[{{{1, 2}, {2, 3}} -> {{3, 2}, {2, 1}, {1, 4}}}, "
        "  {{1, 2}, {2, 3}}, 2, \"NumStates\", \"ShowProgress\" -> True]; "
        "pid = First[HypergraphRewriting`Private`$hgWorkerProc[\"CPU\"]][\"PID\"]; "
        "StartProcess[If[$OperatingSystem === \"Windows\", "
        "  {\"powershell\", \"-NoProfile\", \"-Command\", "
        "   \"while ((Get-Process -Id \" <> ToString[pid] <> \").CPU -lt 0.5) "
        "    { Start-Sleep -Milliseconds 20 }; Stop-Process -Force -Id \" <> ToString[pid]}, "
        "  {\"sh\", \"-c\", \"while [ $(awk '{print $14}' /proc/\" <> ToString[pid] <> "
        "    \"/stat) -lt 50 ]; do sleep 0.02; done; kill -9 \" <> ToString[pid]}]]; "
        "lost = Check[HypergraphRewriting`HGEvolve[{{x, y}, {x, z}} -> "
        "  {{x, y}, {x, w}, {y, w}, {z, w}}, {{0, 0}, {0, 0}}, 6, \"NumStates\", "
        "  \"CanonicalizeStates\" -> Full, \"Threads\" -> 1, "
        "  \"Tuning\" -> <|\"MemoryLimit\" -> 3 2^30|>], \"WORKERLOST\", HypergraphRewriting`HGEvolve::workerlost]; "
        "n3 = small; g3 = gen; "
        "If[IntegerQ[n1] && n1 === n2 === n3 && g1 === g2 && lost === \"WORKERLOST\" && "
        "   IntegerQ[g3] && g3 > g2, Print[\"PIPE_OK states=\", n1], "
        "  Print[\"PIPE_FAIL: \", {n1, n2, n3, g1, g2, g3, lost}]]";

    constexpr int kMaxAttempts = 3;
    std::string out;
    for (int attempts = 0; attempts < kMaxAttempts; ++attempts) {
        out = test_utils::executeWolframScriptCapture(code);
        const bool verdict = out.find("PIPE_OK") != std::string::npos ||
                             out.find("PIPE_FAIL") != std::string::npos ||
                             out.find("PIPE_SKIP") != std::string::npos ||
                             out.find("PIPE_LOAD_FAIL") != std::string::npos;
        if (verdict) break;
    }
    if (out.find("PIPE_SKIP") != std::string::npos)
        GTEST_SKIP() << "this build ships no engine binary for this platform:\n" << out;
    ASSERT_EQ(out.find("PIPE_LOAD_FAIL"), std::string::npos)
        << "the paclet did not load, so nothing below was exercised:\n" << out;
    EXPECT_NE(out.find("PIPE_OK"), std::string::npos) << "WolframScript output:\n" << out;
    EXPECT_NE(out.find("HGEvolve: Evolution complete"), std::string::npos)
        << "\"ShowProgress\" printed no progress frame. WolframScript output:\n" << out;
}

#else

TEST_F(PacletTest, SkipPacletTests) {
    GTEST_SKIP() << "Paclet tests require BUILD_WOLFRAM_LANGUAGE_PACLET and WOLFRAMSCRIPT_AVAILABLE";
}

#endif // BUILD_WOLFRAM_LANGUAGE_PACLET && WOLFRAMSCRIPT_AVAILABLE