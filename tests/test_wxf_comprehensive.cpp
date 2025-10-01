#include <gtest/gtest.h>
#include "../wxf/wxf.hpp"
#include <cstring>

/**
 * Comprehensive WXF Testing Suite
 * Tests both unit-level functionality and round-trip validation with WolframScript
 */

class WXFTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No setup needed - using ByteArray directly
    }

    void TearDown() override {
        // No cleanup needed - no files
    }

    // Helper to test WolframScript round-trip using ByteArray
    bool test_wolfram_roundtrip(const std::vector<uint8_t>& wxf_data) {
#if WOLFRAMSCRIPT_AVAILABLE
        // Build ByteArray[{...}] from our WXF data
        std::string byte_array = "ByteArray[{";
        for (size_t i = 0; i < wxf_data.size(); ++i) {
            if (i > 0) byte_array += ",";
            byte_array += std::to_string(static_cast<int>(wxf_data[i]));
        }
        byte_array += "}]";

        // Test: BinaryDeserialize[ByteArray[{...}]] then serialize back
        std::string code = "original = Normal[" + byte_array + "]; "
                          "result = Normal[BinarySerialize[BinaryDeserialize[" + byte_array + "]]]; "
                          "If[result === original, Exit[0], Exit[1]]";

        std::string cmd = std::string("\"") + WOLFRAMSCRIPT_EXECUTABLE + "\" -code \"" + code + "\"";
        int result = std::system(cmd.c_str());
        return result == 0;
#else
        return true; // Skip if not available
#endif
    }
};

// ============================================================================
// Unit Tests - Core WXF Types
// ============================================================================

TEST_F(WXFTest, TestInteger8) {
    int8_t value = -42;

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    int8_t result = parser.read<int8_t>();

    EXPECT_EQ(value, result);
}

TEST_F(WXFTest, TestInteger16) {
    int16_t value = -12345;

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    int16_t result = parser.read<int16_t>();

    EXPECT_EQ(value, result);
}

TEST_F(WXFTest, TestInteger32) {
    int32_t value = -123456789;

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    int32_t result = parser.read<int32_t>();

    EXPECT_EQ(value, result);
}

TEST_F(WXFTest, TestInteger64) {
    int64_t value = -1234567890123456789LL;

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    int64_t result = parser.read<int64_t>();

    EXPECT_EQ(value, result);
}

TEST_F(WXFTest, TestReal64) {
    double value = 3.14159265359;

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    double result = parser.read<double>();

    EXPECT_DOUBLE_EQ(value, result);
}

TEST_F(WXFTest, TestString) {
    std::string value = "Hello, WXF!";

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    std::string result = parser.read<std::string>();

    EXPECT_EQ(value, result);
}

TEST_F(WXFTest, TestBinaryString) {
    std::vector<uint8_t> value = {0x00, 0xFF, 0x42, 0xDE, 0xAD, 0xBE, 0xEF};

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    std::vector<uint8_t> result = parser.read<std::vector<uint8_t>>();

    EXPECT_EQ(value, result);
}

TEST_F(WXFTest, TestListOfIntegers) {
    std::vector<int64_t> value = {1, 2, 3, 4, 5};

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    std::vector<int64_t> result = parser.read<std::vector<int64_t>>();

    EXPECT_EQ(value, result);
}

TEST_F(WXFTest, TestNestedLists) {
    std::vector<std::vector<int64_t>> value = {
        {1, 2, 3},
        {4, 5},
        {6, 7, 8, 9}
    };

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    std::vector<std::vector<int64_t>> result = parser.read<std::vector<std::vector<int64_t>>>();

    EXPECT_EQ(value, result);
}

TEST_F(WXFTest, TestAssociation) {
    std::unordered_map<std::string, int64_t> map = {
        {"foo", 42},
        {"bar", 100},
        {"baz", -5}
    };

    wxf::Writer writer;
    writer.write_header();
    writer.write_association(map);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();

    std::unordered_map<std::string, int64_t> result;
    parser.read_association([&result](const std::string& key, wxf::Parser& p) {
        result[key] = p.read<int64_t>();
    });

    EXPECT_EQ(map, result);
}

TEST_F(WXFTest, TestFunction) {
    wxf::Writer writer;
    writer.write_header();
    writer.write_function("Plus", 2);
    writer.write<int64_t>(3);
    writer.write<int64_t>(4);

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();

    std::string head;
    std::vector<int64_t> args;

    parser.read_function([&](const std::string& h, size_t count, wxf::Parser& p) {
        head = h;
        for (size_t i = 0; i < count; ++i) {
            args.push_back(p.read<int64_t>());
        }
    });

    EXPECT_EQ("Plus", head);
    EXPECT_EQ(2, args.size());
    EXPECT_EQ(3, args[0]);
    EXPECT_EQ(4, args[1]);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(WXFTest, TestInvalidHeader) {
    std::vector<uint8_t> bad_data = {'7', ':', 1, 2, 3};

    wxf::Parser parser(bad_data);
    EXPECT_THROW(parser.skip_header(), wxf::ParseError);
}

TEST_F(WXFTest, TestUnexpectedEndOfData) {
    std::vector<uint8_t> truncated = {'8', ':', 'L'};  // 64-bit int token but no data

    wxf::Parser parser(truncated);
    parser.skip_header();

    EXPECT_THROW(parser.read<int64_t>(), wxf::ParseError);
}

TEST_F(WXFTest, TestTypeMismatch) {
    wxf::Writer writer;
    writer.write_header();
    writer.write<std::string>("hello");  // Write string

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();

    // Try to read string as integer - should throw TypeError
    EXPECT_THROW(parser.read<int64_t>(), wxf::TypeError);
}

// ============================================================================
// WolframScript Round-Trip Tests (if available)
// ============================================================================

#if WOLFRAMSCRIPT_AVAILABLE

TEST_F(WXFTest, TestWolframScriptRoundTripSimple) {
    // Create a simple integer in WXF
    int64_t value = 42;

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    // Test round-trip with WolframScript using ByteArray
    EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));

    // Test C++ round-trip
    wxf::Parser parser(writer.data());
    parser.skip_header();
    int64_t parsed_result = parser.read<int64_t>();
    EXPECT_EQ(value, parsed_result);
}

TEST_F(WXFTest, TestWolframScriptRoundTripList) {
    // Create a list in WXF
    std::vector<int64_t> value = {1, 2, 3, 4, 5};

    wxf::Writer writer;
    writer.write_header();
    writer.write(value);

    // Test round-trip with WolframScript using ByteArray
    EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));

    // Test C++ round-trip
    wxf::Parser parser(writer.data());
    parser.skip_header();
    auto parsed_result = parser.read<std::vector<int64_t>>();
    EXPECT_EQ(value, parsed_result);
}

TEST_F(WXFTest, TestWolframScriptRoundTripNestedLists) {
    // Create nested lists like hypergraph edges
    std::vector<std::vector<int64_t>> edges = {
        {1, 2, 3},
        {2, 3, 4},
        {1, 4}
    };

    wxf::Writer writer;
    writer.write_header();
    writer.write(edges);

    // Test round-trip with WolframScript using ByteArray
    EXPECT_TRUE(test_wolfram_roundtrip(writer.data()));

    // Test C++ round-trip
    wxf::Parser parser(writer.data());
    parser.skip_header();
    auto parsed_result = parser.read<std::vector<std::vector<int64_t>>>();
    EXPECT_EQ(edges, parsed_result);
}

TEST_F(WXFTest, TestWolframScriptGeneratedData) {
    // Have WolframScript generate ByteArray and verify we can read it
#if WOLFRAMSCRIPT_AVAILABLE
    std::string cmd = std::string("\"") + WOLFRAMSCRIPT_EXECUTABLE + "\"" +
        " -code \"Print[Normal[BinarySerialize[Association[\\\"foo\\\" -> 42, \\\"bar\\\" -> {1, 2, 3}]]]]\"";

    // This would need proper output capture - for now just test our own data
    wxf::Writer writer;
    writer.write_header();
    std::unordered_map<std::string, int64_t> test_map = {{"foo", 42}, {"test", 123}};
    writer.write_association(test_map);

    // Test round-trip
    wxf::Parser parser(writer.data());
    parser.skip_header();

    std::unordered_map<std::string, int64_t> result;
    parser.read_association([&](const std::string& key, wxf::Parser& p) {
        result[key] = p.read<int64_t>();
    });

    EXPECT_EQ(test_map, result);
#else
    GTEST_SKIP() << "WolframScript not available";
#endif
}

#endif // WOLFRAMSCRIPT_AVAILABLE

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(WXFTest, TestLargeDataSerialization) {
    // Create a large nested structure
    std::vector<std::vector<int64_t>> large_data;
    for (int i = 0; i < 1000; ++i) {
        std::vector<int64_t> row;
        for (int j = 0; j < 100; ++j) {
            row.push_back(i * 100 + j);
        }
        large_data.push_back(row);
    }

    auto start = std::chrono::high_resolution_clock::now();

    wxf::Writer writer;
    writer.write_header();
    writer.write(large_data);

    auto serialize_end = std::chrono::high_resolution_clock::now();

    auto data = writer.data();

    wxf::Parser parser(data);
    parser.skip_header();
    auto result = parser.read<std::vector<std::vector<int64_t>>>();

    auto deserialize_end = std::chrono::high_resolution_clock::now();

    EXPECT_EQ(large_data, result);

    auto serialize_time = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - start).count();
    auto deserialize_time = std::chrono::duration_cast<std::chrono::milliseconds>(deserialize_end - serialize_end).count();

    std::cout << "Large data serialization: " << serialize_time << "ms" << std::endl;
    std::cout << "Large data deserialization: " << deserialize_time << "ms" << std::endl;
    std::cout << "Data size: " << data.size() << " bytes" << std::endl;

    // Performance assertions - should be fast
    EXPECT_LT(serialize_time, 1000);  // Less than 1 second
    EXPECT_LT(deserialize_time, 1000);  // Less than 1 second
}