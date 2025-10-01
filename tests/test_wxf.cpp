#include <gtest/gtest.h>
#include "../wxf/wxf.hpp"
#include "test_helpers.hpp"
#include <cstdio>
#include <cstring>

/**
 * Consolidated WXF Testing Suite
 * Tests all WXF functionality with both C++ unit tests and WolframScript integration
 */

class WXFTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

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

        // Round-trip test
        std::string code = "cppBytes = " + byte_array + "; "
                          "mathData = BinaryDeserialize[cppBytes]; "
                          "mathBytes = BinarySerialize[mathData]; "
                          "If[mathBytes === cppBytes, Exit[0], Exit[1]]";

        int result = test_utils::executeWolframScript(code);
        return result == 0;
#else
        return true; // Skip if not available
#endif
    }
};

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
    // Test Mathematica-generated WXF with complex arbitrary keys
    std::vector<uint8_t> mathematica_wxf = {
        56, 58, 65, 1, 45, 65, 2, 45, 102, 3, 115, 4, 76, 105, 115, 116, 67, 1, 67, 2, 102, 1, 115, 8, 71, 108, 111, 98, 97, 108, 96, 102, 102, 1, 115, 8, 71, 108, 111, 98, 97, 108, 96, 103, 115, 8, 71, 108, 111, 98, 97, 108, 96, 104, 65, 1, 45, 67, 1, 67, 2, 45, 67, 2, 67, 3, 67, 45
    };

    wxf::Parser parser(mathematica_wxf);
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
    wxf::ValueList list;
    list.push_back(wxf::Value(int64_t(42)));
    list.push_back(wxf::Value("string"));
    list.push_back(wxf::Value(3.14));

    wxf::ValueAssociation nested_assoc;
    nested_assoc.push_back({wxf::Value("nested_key"), wxf::Value(int64_t(999))});
    list.push_back(wxf::Value(nested_assoc));

    wxf::Writer writer;
    writer.write_header();
    writer.write(wxf::Value(list));

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

TEST_F(WXFTest, VerifyTestInfrastructureDetectsFailures) {
#if WOLFRAMSCRIPT_AVAILABLE
    // Verify that Exit[1] is detected as failure
    std::string fail_code = "Exit[1]";
    int result = test_utils::executeWolframScript(fail_code);
    EXPECT_NE(0, result) << "Exit[1] should produce non-zero exit code";

    // Verify that Exit[0] is detected as success
    std::string success_code = "Exit[0]";
    result = test_utils::executeWolframScript(success_code);
    EXPECT_EQ(0, result) << "Exit[0] should produce zero exit code";

    // Verify that $Failed in round-trip is detected
    std::string failed_deserialize = "cppBytes = ByteArray[{56, 58, 255}]; "
                                     "mathData = BinaryDeserialize[cppBytes]; "
                                     "If[FailureQ[mathData], Exit[1], Exit[0]]";
    result = test_utils::executeWolframScript(failed_deserialize);
    EXPECT_NE(0, result) << "Failed deserialization should be detected";
#endif
}

// ============================================================================
// WOLFRAM-GENERATED DATA TEST
// ============================================================================

#if WOLFRAMSCRIPT_AVAILABLE
TEST_F(WXFTest, WolframGeneratedData) {
    // Test with hardcoded Mathematica-generated WXF data
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
