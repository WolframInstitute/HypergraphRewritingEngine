#include <gtest/gtest.h>
#include "../wxf/wxf.hpp"
#include "test_helpers.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

/**
 * Consolidated WXF Testing Suite
 * Tests all WXF functionality with both C++ unit tests and WolframScript integration
 */

class WXFTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Consultations of the Wolfram oracle, and the ones that returned no verdict at all.
    // A RATE, not a pass/fail: an unreachable oracle is not a WXF defect, and reporting it as
    // one is the same conflation this file already removed between "Wolfram disagreed" and
    // "Wolfram was never asked". Reported in TearDownTestSuite and bounded there.
    static int s_consultations;
    static int s_unavailable;

    static void SetUpTestSuite() { s_consultations = 0; s_unavailable = 0; }

    static void TearDownTestSuite() {
#if WOLFRAMSCRIPT_AVAILABLE
        if (s_consultations == 0) return;
        std::printf("# wolfram oracle: %d/%d consultations returned no verdict "
                    "(WSL interop vsock), after %d spaced attempts each\n",
                    s_unavailable, s_consultations, 3);
        // Bounded rather than tolerated. One wedged vsock in a long suite is the machine; a
        // quarter of them is a broken toolchain, and then the round-trips prove nothing.
        EXPECT_LE(s_unavailable * 4, s_consultations)
            << s_unavailable << " of " << s_consultations << " oracle consultations could not "
            << "be made. Above a quarter, these tests are not checking anything -- fix the "
            << "interop (WSL_INTEROP pointing at a live /run/WSL socket) before trusting them.";
#endif
    }

    // Test round-trip with WolframScript using ByteArray
    bool test_wolfram_roundtrip(const std::vector<uint8_t>& wxf_data) {
#if WOLFRAMSCRIPT_AVAILABLE
        // Build ByteArray[{...}] from our WXF data
        std::string byte_array = "ByteArray[{";
        for (size_t i = 0; i < wxf_data.size(); ++i) {
            if (i > 0) byte_array += ",";
            byte_array += std::to_string(static_cast<int>(wxf_data[i]));
        }
        byte_array += "}]";

        // ASSERT ON A PRINTED MARKER, NOT ON THE EXIT CODE.
        //
        // A Windows wolframscript.exe invoked from WSL exits non-zero on a benign license
        // error at shutdown, AFTER the script's own Exit -- executeWolframScriptCapture's
        // own contract says so, and it is why that function exists. Reading the exit status
        // therefore reports a round-trip failure for a run whose round-trip succeeded.
        //
        // Measured before this change: a DIFFERENT WXFTest failed on each full-suite run
        // (String, then VerifyTestInfrastructureDetectsFailures, then Integer64), each
        // passing in isolation -- the signature of a per-invocation shutdown code, not of a
        // serialization defect.
        //
        // The two markers are distinguished so a genuine disagreement with Wolfram is not
        // reported as infrastructure, and infrastructure is not reported as a disagreement.
        std::string code = "cppBytes = " + byte_array + "; "
                          "mathData = BinaryDeserialize[cppBytes]; "
                          "mathBytes = BinarySerialize[mathData]; "
                          "Print[If[mathBytes === cppBytes, "
                          "\"WXF_ROUNDTRIP_OK\", \"WXF_ROUNDTRIP_MISMATCH\"]]";

        // THE ORACLE NOT ANSWERING IS NOT THE ORACLE DISAGREEING, and the two must not be
        // reported as one. Measured cause of the former, captured by this very branch:
        //
        //   <3>WSL (2781446 - ) ERROR: UtilAcceptVsock:251: accept4 failed 110
        //
        // errno 110 is ETIMEDOUT on the WSL interop vsock -- the bridge that launches a
        // Windows .exe from Linux timed out accepting the connection, so wolframscript never
        // started and the round-trip was never performed. The attempt took 16 s and printed
        // nothing else.
        //
        // RETRIED WITH A BACKOFF, and the backoff is the measured part. Retrying ONCE
        // immediately was tried first and is not enough: a run hit the fault and BOTH attempts
        // returned the same accept4/ETIMEDOUT within 38 s total. The failure is therefore not
        // independent per launch -- it is a wedged vsock with a recovery time, so two calls a
        // few milliseconds apart sample the same wedged state. Attempts are spaced instead, and
        // the spacing costs a healthy run nothing because the first attempt returns.
        //
        // The retry covers ONLY the no-verdict case. A MISMATCH is returned immediately and is
        // never retried, because retrying a disagreement is how a real regression gets buried.
        const OracleReply r = consult(code);
        if (!r.verdict) return true;   // unproven, not disproven -- see the counter in consult()
        if (r.out.find("WXF_ROUNDTRIP_MISMATCH") != std::string::npos) {
            ADD_FAILURE() << "Wolfram re-serialized this payload to different bytes";
            return false;
        }
        return true;
#else
        return true; // Skip if not available
#endif
    }

#if WOLFRAMSCRIPT_AVAILABLE
    struct OracleReply {
        std::string out;
        bool verdict;   // one of the two markers arrived, so the oracle actually answered
    };

    // One consultation, retried ONLY while it is verdictless. Every caller goes through here:
    // the retry policy and the unavailability accounting are one rule, and a second copy would
    // drift into asserting on a transport fault the other tolerates -- which is exactly what
    // VerifyTestInfrastructureDetectsFailures did, failing the suite on an accept4/ETIMEDOUT
    // that the round-trips were already counting and bounding.
    static OracleReply consult(const std::string& code) {
        ++s_consultations;
        constexpr int kAttempts = 3;
        std::string out;
        for (int attempt = 0; attempt < kAttempts; ++attempt) {
            if (attempt > 0) {
                std::this_thread::sleep_for(std::chrono::seconds(2 * attempt));
            }
            out = test_utils::executeWolframScriptCapture(code);
            if (out.find("WXF_ROUNDTRIP_OK") != std::string::npos ||
                out.find("WXF_ROUNDTRIP_MISMATCH") != std::string::npos) {
                return {out, true};
            }
        }
        // NOT a failure of the caller. The oracle was never reached, so whatever was asked is
        // neither confirmed nor refuted; it is counted, and the suite-level bound in
        // TearDownTestSuite decides whether that has gone too far.
        ++s_unavailable;
        std::printf("#   oracle unavailable (%d spaced attempts): %s\n",
                    kAttempts, out.empty() ? "(no output)" : out.c_str());
        return {out, false};
    }
#endif
};

int WXFTest::s_consultations = 0;
int WXFTest::s_unavailable   = 0;

// ============================================================================
// BASIC TYPE TESTS - C++ Unit Tests + WolframScript Round-trips
// ============================================================================

TEST_F(WXFTest, Integer8) {
    std::vector<int8_t> test_values = {-128, -1, 0, 1, 127};

    for (int8_t value : test_values) {
        wxf::Writer writer;
        writer.write_header();
        writer.write(value);

        wxf::Parser parser(writer.data());
        parser.skip_header();
        int8_t result = parser.read<int8_t>();
        EXPECT_EQ(value, result);

        EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
    }
}

TEST_F(WXFTest, Integer16) {
    std::vector<int16_t> test_values = {-32768, -1, 0, 1, 32767};

    for (int16_t value : test_values) {
        wxf::Writer writer;
        writer.write_header();
        writer.write(value);

        wxf::Parser parser(writer.data());
        parser.skip_header();
        int16_t result = parser.read<int16_t>();
        EXPECT_EQ(value, result);

        EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
    }
}

TEST_F(WXFTest, Integer32) {
    std::vector<int32_t> test_values = {-2147483648, -1, 0, 1, 2147483647};

    for (int32_t value : test_values) {
        wxf::Writer writer;
        writer.write_header();
        writer.write(value);

        wxf::Parser parser(writer.data());
        parser.skip_header();
        int32_t result = parser.read<int32_t>();
        EXPECT_EQ(value, result);

        EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
    }
}

TEST_F(WXFTest, Integer64) {
    std::vector<int64_t> test_values = {-9223372036854775807LL, -1, 0, 1, 9223372036854775807LL};

    for (int64_t value : test_values) {
        wxf::Writer writer;
        writer.write_header();
        writer.write(value);

        wxf::Parser parser(writer.data());
        parser.skip_header();
        int64_t result = parser.read<int64_t>();
        EXPECT_EQ(value, result);

        EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
    }
}

TEST_F(WXFTest, Real64) {
    std::vector<double> test_values = {
        -1.7976931348623157e+308,
        -1.0, -0.0, 0.0, 1.0,
        3.141592653589793,
        2.718281828459045,
        1.7976931348623157e+308
    };

    for (double value : test_values) {
        wxf::Writer writer;
        writer.write_header();
        writer.write(value);

        wxf::Parser parser(writer.data());
        parser.skip_header();
        double result = parser.read<double>();
        EXPECT_DOUBLE_EQ(value, result);

        EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
    }
}

TEST_F(WXFTest, String) {
    std::vector<std::string> test_values = {
        "",
        "Hello",
        "Hello, World!",
        "Unicode: αβγδε",
        "Special chars: !@#$%^&*()",
        "Newlines:\n\r\t",
        std::string(1000, 'A')
    };

    for (const std::string& value : test_values) {
        wxf::Writer writer;
        writer.write_header();
        writer.write(value);

        wxf::Parser parser(writer.data());
        parser.skip_header();
        std::string result = parser.read<std::string>();
        EXPECT_EQ(value, result);

        EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
    }
}

TEST_F(WXFTest, BinaryString) {
    std::vector<std::vector<uint8_t>> test_values = {
        {},
        {0x00},
        {0xFF},
        {0x00, 0x01, 0x02, 0x03, 0x04, 0x05},
        {0xDE, 0xAD, 0xBE, 0xEF}
    };

    // Add all byte values test
    std::vector<uint8_t> all_bytes(256);
    for (int i = 0; i < 256; ++i) {
        all_bytes[i] = static_cast<uint8_t>(i);
    }
    test_values.push_back(all_bytes);

    for (const auto& value : test_values) {
        wxf::Writer writer;
        writer.write_header();
        writer.write(value);

        wxf::Parser parser(writer.data());
        parser.skip_header();
        auto result = parser.read<std::vector<uint8_t>>();
        EXPECT_EQ(value, result);

        if (!value.empty()) {
            EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
        }
    }
}

// ============================================================================
// NESTED STRUCTURE TESTS
// ============================================================================

TEST_F(WXFTest, ListOfIntegers) {
    std::vector<int64_t> values = {1, 2, 3, 4, 5};

    wxf::Writer writer;
    writer.write_header();
    writer.write(values);

    wxf::Parser parser(writer.data());
    parser.skip_header();
    auto result = parser.read<std::vector<int64_t>>();
    EXPECT_EQ(values, result);

    EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
}

TEST_F(WXFTest, NestedLists) {
    std::vector<std::vector<int64_t>> hypergraph_edges = {
        {1, 2, 3},
        {2, 3, 4, 5},
        {1, 4},
        {},
        {100, 200, 300, 400, 500, 600}
    };

    wxf::Writer writer;
    writer.write_header();
    writer.write(hypergraph_edges);

    wxf::Parser parser(writer.data());
    parser.skip_header();
    auto result = parser.read<std::vector<std::vector<int64_t>>>();
    EXPECT_EQ(hypergraph_edges, result);

    EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
}

TEST_F(WXFTest, ArbitraryNesting) {
    // Test 3-level nesting
    std::vector<std::vector<std::vector<int64_t>>> triple_nested = {
        {{1, 2}, {3, 4}},
        {{5, 6, 7}, {8}, {}},
        {{9, 10, 11, 12}}
    };

    wxf::Writer writer;
    writer.write_header();
    writer.write(triple_nested);

    wxf::Parser parser(writer.data());
    parser.skip_header();
    auto result = parser.read<std::vector<std::vector<std::vector<int64_t>>>>();
    EXPECT_EQ(triple_nested, result);

    // Test vector of strings
    std::vector<std::string> string_list = {"hello", "world", "test", ""};

    wxf::Writer writer2;
    writer2.write_header();
    writer2.write(string_list);

    wxf::Parser parser2(writer2.data());
    parser2.skip_header();
    auto result2 = parser2.read<std::vector<std::string>>();
    EXPECT_EQ(string_list, result2);
}

// ============================================================================
// ASSOCIATION TESTS
// ============================================================================

TEST_F(WXFTest, Association) {
    std::unordered_map<std::string, int64_t> assoc = {
        {"Options", -1},
        {"Steps", 5},
        {"Value", 100},
        {"zero", 0}
    };

    wxf::Writer writer;
    writer.write_header();
    writer.write_association(assoc);

    wxf::Parser parser(writer.data());
    parser.skip_header();

    std::unordered_map<std::string, int64_t> result;
    parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
        result[key] = value_parser.read<int64_t>();
    });

    EXPECT_EQ(assoc, result);
    EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
}

TEST_F(WXFTest, AssociationWithArbitraryKeys) {
    // Test Wolfram Language-generated WXF with complex arbitrary keys
    std::vector<uint8_t> wolfram_language_wxf = {
        56, 58, 65, 1, 45, 65, 2, 45, 102, 3, 115, 4, 76, 105, 115, 116, 67, 1, 67, 2, 102, 1, 115, 8, 71, 108, 111, 98, 97, 108, 96, 102, 102, 1, 115, 8, 71, 108, 111, 98, 97, 108, 96, 103, 115, 8, 71, 108, 111, 98, 97, 108, 96, 104, 65, 1, 45, 67, 1, 67, 2, 45, 67, 2, 67, 3, 67, 45
    };

    wxf::Parser parser(wolfram_language_wxf);
    parser.skip_header();

    int outer_count = 0;
    parser.read_association_generic([&](wxf::Parser& outer_key_parser, wxf::Parser& outer_value_parser) {
        outer_count++;

        int middle_assoc_count = 0;
        outer_key_parser.read_association_generic([&](wxf::Parser& middle_key_parser, wxf::Parser& middle_value_parser) {
            middle_assoc_count++;

            if (middle_assoc_count == 1) {
                std::string list_head;
                std::vector<int64_t> list_values;
                bool has_function = false;

                middle_key_parser.read_function([&](const std::string& h, size_t count, wxf::Parser& p) {
                    list_head = h;
                    EXPECT_EQ("List", h);
                    EXPECT_EQ(3, count);

                    list_values.push_back(p.read<int64_t>());
                    list_values.push_back(p.read<int64_t>());

                    p.read_function([&](const std::string& func_h, size_t func_count, wxf::Parser& func_p) {
                        EXPECT_EQ("Global`f", func_h);
                        EXPECT_EQ(1, func_count);
                        has_function = true;

                        func_p.read_function([&](const std::string& g_h, size_t g_count, wxf::Parser& g_p) {
                            EXPECT_EQ("Global`g", g_h);
                            EXPECT_EQ(1, g_count);

                            std::string h_symbol = g_p.read_symbol();
                            EXPECT_EQ("Global`h", h_symbol);
                        });
                    });
                });

                EXPECT_EQ(2, list_values.size());
                EXPECT_EQ(1, list_values[0]);
                EXPECT_EQ(2, list_values[1]);
                EXPECT_TRUE(has_function);

                int inner_assoc_count = 0;
                middle_value_parser.read_association_generic([&](wxf::Parser& k, wxf::Parser& v) {
                    inner_assoc_count++;
                    int64_t key = k.read<int64_t>();
                    int64_t value = v.read<int64_t>();
                    EXPECT_EQ(1, key);
                    EXPECT_EQ(2, value);
                });
                EXPECT_EQ(1, inner_assoc_count);

            } else if (middle_assoc_count == 2) {
                int64_t key = middle_key_parser.read<int64_t>();
                int64_t value = middle_value_parser.read<int64_t>();
                EXPECT_EQ(2, key);
                EXPECT_EQ(3, value);
            }
        });

        EXPECT_EQ(2, middle_assoc_count);

        int64_t outer_value = outer_value_parser.read<int64_t>();
        EXPECT_EQ(45, outer_value);
    });

    EXPECT_EQ(1, outer_count);
}

// ============================================================================
// FUNCTION TESTS
// ============================================================================

TEST_F(WXFTest, Function) {
    wxf::Writer writer;
    writer.write_header();
    writer.write_function("CustomFunction", 3);
    writer.write(int64_t(1));
    writer.write(int64_t(2));
    writer.write(int64_t(3));

    wxf::Parser parser(writer.data());
    parser.skip_header();

    std::string head;
    size_t arg_count = 0;
    std::vector<int64_t> args;

    parser.read_function([&](const std::string& h, size_t count, wxf::Parser& args_parser) {
        head = h;
        arg_count = count;
        for (size_t i = 0; i < count; ++i) {
            args.push_back(args_parser.read<int64_t>());
        }
    });

    EXPECT_EQ("CustomFunction", head);
    EXPECT_EQ(3u, arg_count);
    EXPECT_EQ(std::vector<int64_t>({1, 2, 3}), args);

    EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

TEST_F(WXFTest, InvalidHeader) {
    std::vector<uint8_t> invalid_wxf = {0xFF, 0xFF, 0x43, 0x00};
    wxf::Parser parser(invalid_wxf);
    EXPECT_THROW(parser.skip_header(), wxf::ParseError);
}

TEST_F(WXFTest, UnexpectedEndOfData) {
    std::vector<uint8_t> truncated_wxf = {'8', ':'};
    wxf::Parser parser(truncated_wxf);
    parser.skip_header();
    EXPECT_THROW(parser.read<int64_t>(), wxf::ParseError);
}

TEST_F(WXFTest, TypeMismatch) {
    wxf::Writer writer;
    writer.write_header();
    writer.write(int64_t(42));

    wxf::Parser parser(writer.data());
    parser.skip_header();

    EXPECT_THROW(parser.read<std::string>(), wxf::TypeError);
}

TEST_F(WXFTest, UnimplementedTypes) {
    std::vector<uint8_t> big_int_data = {'8', ':', 'I', 5, '1', '2', '3', '4', '5'};
    wxf::Parser parser1(big_int_data);
    parser1.skip_header();
    EXPECT_THROW(parser1.read_big_integer(), wxf::ParseError);

    std::vector<uint8_t> big_real_data = {'8', ':', 'R', 5, '3', '.', '1', '4', '1'};
    wxf::Parser parser2(big_real_data);
    parser2.skip_header();
    EXPECT_THROW(parser2.read_big_real(), wxf::ParseError);

    std::vector<uint8_t> delayed_rule_data = {'8', ':', ':'};
    wxf::Parser parser3(delayed_rule_data);
    parser3.skip_header();
    EXPECT_THROW(parser3.read<int64_t>(), wxf::ParseError);

    std::vector<uint8_t> packed_array_data = {'8', ':', 0xC1, 0, 0};
    wxf::Parser parser4(packed_array_data);
    parser4.skip_header();
    EXPECT_THROW(parser4.read<int64_t>(), wxf::ParseError);

    std::vector<uint8_t> numeric_array_data = {'8', ':', 0xC2, 0, 0};
    wxf::Parser parser5(numeric_array_data);
    parser5.skip_header();
    EXPECT_THROW(parser5.read<int64_t>(), wxf::ParseError);
}

// ============================================================================
// NEW FEATURE TESTS
// ============================================================================

TEST_F(WXFTest, ReadStringHandlesBothStringAndSymbol) {
    // Test String token
    wxf::Writer writer1;
    writer1.write_header();
    writer1.write_string("Hello");

    wxf::Parser parser1(writer1.data());
    parser1.skip_header();
    std::string result1 = parser1.read<std::string>();
    EXPECT_EQ("Hello", result1);

    // Test Symbol token
    wxf::Writer writer2;
    writer2.write_header();
    writer2.write_symbol("True");

    wxf::Parser parser2(writer2.data());
    parser2.skip_header();
    std::string result2 = parser2.read<std::string>();
    EXPECT_EQ("True", result2);
}

TEST_F(WXFTest, ValueVariantHeterogeneousNesting) {
    wxf::WXFValueList list;
    list.push_back(wxf::WXFValue(int64_t(42)));
    list.push_back(wxf::WXFValue("string"));
    list.push_back(wxf::WXFValue(3.14));

    wxf::WXFValueAssociation nested_assoc;
    nested_assoc.push_back({wxf::WXFValue("nested_key"), wxf::WXFValue(int64_t(999))});
    list.push_back(wxf::WXFValue(nested_assoc));

    wxf::Writer writer;
    writer.write_header();
    writer.write(wxf::WXFValue(list));

    wxf::Parser parser(writer.data());
    parser.skip_header();

    bool parsed = false;
    try {
        parser.read_function([&](const std::string& head, size_t count, wxf::Parser& p) {
            EXPECT_EQ("List", head);
            EXPECT_EQ(4u, count);
            for (size_t i = 0; i < count; ++i) {
                p.skip_value();
            }
            parsed = true;
        });
    } catch (...) {
        parsed = false;
    }

    EXPECT_TRUE(parsed);
}

TEST_F(WXFTest, SkipValue) {
    wxf::Writer writer;
    writer.write_header();
    writer.write(int64_t(42));
    writer.write(3.14);
    writer.write("string");
    writer.write(std::vector<int64_t>{1, 2, 3});

    wxf::Parser parser(writer.data());
    parser.skip_header();

    parser.skip_value();
    parser.skip_value();
    parser.skip_value();
    parser.skip_value();

    EXPECT_TRUE(parser.at_end());
}

TEST_F(WXFTest, ConvenienceFunctions) {
    int64_t original = 42;

    auto bytes = wxf::serialize(original);
    auto result = wxf::deserialize<int64_t>(bytes);
    EXPECT_EQ(original, result);

    auto result2 = wxf::deserialize<int64_t>(bytes.data(), bytes.size());
    EXPECT_EQ(original, result2);
}

// ============================================================================
// TEST INFRASTRUCTURE VALIDATION
// ============================================================================

// POSITIVE CONTROL for the mechanism every round-trip above depends on.
//
// Those tests pass when wolframscript PRINTS a success marker. A marker check that could
// never print the failure marker would pass silently forever, so this drives both branches
// and a genuinely corrupt payload through the same capture path.
//
// It deliberately does NOT assert on exit codes. A Windows wolframscript.exe invoked from
// WSL exits non-zero on a benign license error at shutdown even after Exit[0], so
// `EXPECT_EQ(0, result)` for a successful script is a coin flip -- this test asserted exactly
// that and was one of the tests failing at random in the full suite.
TEST_F(WXFTest, VerifyTestInfrastructureDetectsFailures) {
#if WOLFRAMSCRIPT_AVAILABLE
    // Each branch goes through consult(), so an oracle that could not be reached is counted
    // and bounded exactly as it is for the round-trips rather than failing here. A control
    // that fails on the transport asserts something about WSL's vsock, not about the markers.
    // The bound in TearDownTestSuite is what notices if the oracle stops being reachable.

    // The success branch prints the OK marker and nothing else.
    const OracleReply ok = consult(
        "Print[If[1 === 1, \"WXF_ROUNDTRIP_OK\", \"WXF_ROUNDTRIP_MISMATCH\"]]");
    if (ok.verdict) {
        EXPECT_NE(ok.out.find("WXF_ROUNDTRIP_OK"), std::string::npos)
            << "the success branch did not print the success marker, so a passing round-trip "
            << "cannot be distinguished from a script that did not run. Output:\n" << ok.out;
        EXPECT_EQ(ok.out.find("WXF_ROUNDTRIP_MISMATCH"), std::string::npos)
            << "the failure marker appeared on the success branch";
    }

    // The failure branch prints the mismatch marker, and NOT the OK marker -- a substring
    // check that matched both would report every mismatch as a pass.
    const OracleReply bad = consult(
        "Print[If[1 === 2, \"WXF_ROUNDTRIP_OK\", \"WXF_ROUNDTRIP_MISMATCH\"]]");
    if (bad.verdict) {
        EXPECT_NE(bad.out.find("WXF_ROUNDTRIP_MISMATCH"), std::string::npos)
            << "the failure branch did not print the failure marker, so a real disagreement "
            << "would read as infrastructure trouble. Output:\n" << bad.out;
        EXPECT_EQ(bad.out.find("WXF_ROUNDTRIP_OK"), std::string::npos)
            << "the success marker appeared on the failure branch";
    }

    // A corrupt payload reaches the same failure branch through BinaryDeserialize rather
    // than through a hand-written condition.
    const OracleReply corrupt = consult(
        "cppBytes = ByteArray[{56, 58, 255}]; "
        "mathData = Quiet@BinaryDeserialize[cppBytes]; "
        "Print[If[FailureQ[mathData], \"WXF_ROUNDTRIP_MISMATCH\", \"WXF_ROUNDTRIP_OK\"]]");
    if (corrupt.verdict) {
        EXPECT_NE(corrupt.out.find("WXF_ROUNDTRIP_MISMATCH"), std::string::npos)
            << "a corrupt payload did not reach the failure branch. Output:\n" << corrupt.out;
    }
#endif
}

// ============================================================================
// WOLFRAM-GENERATED DATA TEST
// ============================================================================

#if WOLFRAMSCRIPT_AVAILABLE
TEST_F(WXFTest, WolframGeneratedData) {
    // Test with hardcoded Wolfram Language-generated WXF data
    std::string mathWxfHex = "383a41042d730b476c6f62616c60696e743843d62d730c476c6f62616c60696e7436346915cd5b072d730f476c6f62616c60706f73697469766543642d730b476c6f62616c607a65726f4300";

    std::vector<uint8_t> mathWxfBytes;
    for (size_t i = 0; i < mathWxfHex.length(); i += 2) {
        std::string byteStr = mathWxfHex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byteStr, nullptr, 16));
        mathWxfBytes.push_back(byte);
    }

    wxf::Parser parser(mathWxfBytes);
    parser.skip_header();

    std::unordered_map<std::string, int64_t> result;
    parser.read_association([&](const std::string& key, wxf::Parser& value_parser) {
        result[key] = value_parser.read<int64_t>();
    });

    EXPECT_EQ(4u, result.size());
}
#endif

// ============================================================================
// VARINT EDGE CASES — pure C++ (no wolframscript), fast
// ============================================================================
// The parser's read_varint is private; we exercise it indirectly through a
// BinaryString frame, whose length prefix is a varint. A legal varint decodes
// and then the parser throws on payload truncation (distinguishable message);
// an illegal varint throws "Varint too large" from inside read_varint itself.

namespace {
std::vector<uint8_t> encode_varint(uint64_t value) {
    std::vector<uint8_t> out;
    while (value >= 0x80) {
        out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<uint8_t>(value & 0x7F));
    return out;
}

// Build a frame that forces read_varint to run via a BinaryString length prefix.
std::vector<uint8_t> make_binarystring_frame(const std::vector<uint8_t>& length_varint) {
    std::vector<uint8_t> frame = {'8', ':', static_cast<uint8_t>(wxf::Token::BinaryString)};
    frame.insert(frame.end(), length_varint.begin(), length_varint.end());
    return frame;
}
} // namespace

TEST_F(WXFTest, Varint_TenByteEncoding_Uint64Max) {
    // UINT64_MAX encodes to the maximal 10-byte varint; the decoder's shift bound
    // (`shift > 63`) must accept all 10 bytes, since the 64th bit lands in byte 10.
    // After successful varint decode the downstream vector<uint8_t> allocation will
    // fail with std::length_error — that's fine; we only want to confirm the
    // varint layer didn't reject the legal 10-byte encoding.
    auto varint = encode_varint(UINT64_MAX);
    ASSERT_EQ(varint.size(), 10u);

    auto frame = make_binarystring_frame(varint);
    wxf::Parser parser(frame);
    parser.skip_header();

    try {
        (void)parser.read<std::vector<uint8_t>>();
        FAIL() << "Expected some exception — frame has no payload";
    } catch (const std::exception& e) {
        std::string msg = e.what();
        EXPECT_EQ(msg.find("Varint too large"), std::string::npos)
            << "Unexpected 'Varint too large' for a legal 10-byte varint: " << msg;
    }
}

TEST_F(WXFTest, Varint_ElevenByteEncoding_Rejected) {
    // 11 bytes with continuation on the first 10 implies shift reaches 70 — UB if
    // we did the shift, so the parser must reject.
    std::vector<uint8_t> bad_varint;
    for (int i = 0; i < 10; ++i) bad_varint.push_back(0xFF);  // continuation set
    bad_varint.push_back(0x01);                                // final byte, no continuation

    auto frame = make_binarystring_frame(bad_varint);
    wxf::Parser parser(frame);
    parser.skip_header();

    try {
        (void)parser.read<std::vector<uint8_t>>();
        FAIL() << "Expected 'Varint too large' ParseError";
    } catch (const wxf::ParseError& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Varint too large"), std::string::npos)
            << "Expected 'Varint too large', got: " << msg;
    }
}

TEST_F(WXFTest, Varint_BoundaryValues_Decode) {
    // Each of these is a legal varint — the decode must not throw "Varint too large".
    const uint64_t kCases[] = {
        0, 1, 127,           // 1-byte
        128, 16383,          // 2-byte
        16384, 2097151,      // 3-byte
        uint64_t(1) << 28,   // 5-byte
        uint64_t(1) << 56,   // 9-byte
        (uint64_t(1) << 63), // 10-byte — this is the value that exercises shift=63
        UINT64_MAX,          // 10-byte max
    };
    for (uint64_t v : kCases) {
        auto varint = encode_varint(v);
        auto frame = make_binarystring_frame(varint);
        wxf::Parser parser(frame);
        parser.skip_header();

        if (v == 0) {
            // Empty BinaryString — should succeed.
            auto bytes = parser.read<std::vector<uint8_t>>();
            EXPECT_TRUE(bytes.empty()) << "v=0 should decode to empty BinaryString";
            continue;
        }

        try {
            (void)parser.read<std::vector<uint8_t>>();
            FAIL() << "Expected payload underrun / oversize error for v=" << v;
        } catch (const std::exception& e) {
            std::string msg = e.what();
            EXPECT_EQ(msg.find("Varint too large"), std::string::npos)
                << "Legal varint v=" << v << " was rejected: " << msg;
        }
    }
}

// WXF is parsed from bytes that arrive from outside the process, so malformed input must be
// rejected as an error -- never as a crash, and never as an allocation the sender chose.
//
// These are written to DISCRIMINATE. A first attempt asserted only "throws a WXFException",
// which BOTH the guarded and the unguarded build satisfy: the unguarded one just throws
// later, on running out of input. A test that passes with the defect still present is worse
// than no test, so each case below is built so removing its guard changes what is observed.

// Depth. skip_value recurses on structured tokens and the INPUT chooses how deep. These
// lists are well-formed and properly terminated, so without the bound they parse fine; the
// discriminator is success-versus-refusal, not which exception comes out.
TEST_F(WXFTest, NestingPastTheSkipDepthLimitIsRefused) {
    // Built with the Writer so the bytes are unquestionably valid WXF -- the point of the
    // test is the DEPTH, and hand-assembled bytes would confound "refused for nesting" with
    // "refused for being malformed".
    auto nested = [](size_t depth) {
        wxf::Writer w;
        w.write_header();
        for (size_t i = 0; i < depth; ++i) w.write_function("List", 1);
        w.write_int8(0);                                       // innermost leaf
        return w.data();
    };

    {   // well inside the bound: must still be accepted
        auto d = nested(wxf::Parser::MAX_SKIP_DEPTH / 4);
        wxf::Parser p(d.data(), d.size());
        p.skip_header();
        EXPECT_NO_THROW(p.skip_value()) << "nesting within the limit must still parse";
    }
    {   // past the bound: refused rather than recursed into
        auto d = nested(wxf::Parser::MAX_SKIP_DEPTH + 16);
        wxf::Parser p(d.data(), d.size());
        p.skip_header();
        EXPECT_THROW(p.skip_value(), wxf::WXFException)
            << "nesting past the limit must be refused -- recursion here is driven by the "
               "input, and unbounded it exhausts the stack";
    }
}

// Container length. Without the bound this reserves on the sender's number and dies with
// std::bad_alloc, which is NOT a WXFException -- so a caller catching the WXF base type
// leaks it. Demanding a WXFException is exactly what separates the two builds.
TEST_F(WXFTest, ImpossibleListLengthIsRejectedBeforeAllocating) {
    // A COMPLETE list header -- token, count, and the List head symbol -- so the parser
    // reaches the element loop. Without the head the read fails on the malformed header
    // instead, never touching the allocation this test is about.
    std::vector<uint8_t> data = {'8', ':', 'f'};
    uint64_t huge = 1ull << 40;                    // ~1e12 elements, from an 18-byte message
    while (huge >= 0x80) { data.push_back(static_cast<uint8_t>((huge & 0x7F) | 0x80)); huge >>= 7; }
    data.push_back(static_cast<uint8_t>(huge));
    data.push_back('s'); data.push_back(4);
    for (char c : std::string("List")) data.push_back(static_cast<uint8_t>(c));

    // Assert the SPECIFIC rejection, not merely "some WXF error". Without the bound this
    // still throws -- the reserve succeeds under overcommit and the read then runs out of
    // input -- so requiring only a WXFException cannot tell the two builds apart. What
    // distinguishes them is being refused BEFORE the count is used to size anything.
    wxf::Parser parser(data.data(), data.size());
    parser.skip_header();
    bool refused_on_length = false;
    try {
        parser.read<std::vector<int64_t>>();
    } catch (const wxf::ParseError& e) {
        refused_on_length =
            std::string(e.what()).find("exceeds the remaining input") != std::string::npos;
    } catch (...) {
    }
    EXPECT_TRUE(refused_on_length)
        << "an element count past the remaining input must be refused on the count itself; "
           "reaching the allocation means an 18-byte message can size a multi-terabyte "
           "reservation";
}
