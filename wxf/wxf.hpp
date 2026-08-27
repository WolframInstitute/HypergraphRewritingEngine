#pragma once
#include "hgcommon/namespace.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <complex>
#include <cstdint>
#include <limits>
#include <variant>

/**
 * Wolfram Exchange Format (WXF) Implementation
 * Full specification compliance with template-based generic API
 * Supports bidirectional serialization/deserialization
 */
namespace HG_NAMESPACE {
namespace wxf {

// WXF Token definitions from specification
enum class Token : uint8_t {
    // Atomic types
    String = 'S',           // UTF-8 string
    Symbol = 's',           // Wolfram symbol
    BigInteger = 'I',       // Arbitrary precision integer
    BigReal = 'R',          // Arbitrary precision real

    // Numeric types
    Integer8 = 'C',         // 8-bit signed integer
    Integer16 = 'j',        // 16-bit signed integer
    Integer32 = 'i',        // 32-bit signed integer
    Integer64 = 'L',        // 64-bit signed integer
    Real64 = 'r',           // IEEE 754 double precision

    // Structured types
    Function = 'f',         // Function[head, args...]
    Association = 'A',      // Association[key1->val1, ...]
    Rule = '-',             // Rule marker (key->value)
    DelayedRule = ':',      // Delayed rule marker
    BinaryString = 'B',     // Binary data

    // Packed numeric arrays
    PackedArray = 0xC1,     // '\301' - Packed array
    NumericArray = 0xC2,    // '\302' - Numeric array
};

// Exception types for structured error handling
class WXFException : public std::runtime_error {
public:
    WXFException(const std::string& message, size_t position = 0);

    size_t position() const noexcept;

private:
    size_t position_;
};

class ParseError : public WXFException {
public:
    ParseError(const std::string& message, size_t position = 0);
};

class TypeError : public WXFException {
public:
    TypeError(const std::string& message, size_t position = 0);
};

// Forward declarations
class Parser;
class Writer;

// Heterogeneous value type for arbitrary WXF data structures
// Supports recursive nesting via std::variant
// SUPPRESSED FOR ONE COMPILER, AND SCOPED TO ONE STRUCT. MinGW's GCC reported a spurious
// -Wmaybe-uninitialized inside std::variant's machinery for this type; the warning is a known
// false positive of the variant implementation and not a statement about this code. The push/pop
// pair below bounds it to WXFValue's definition, so nothing a caller writes is covered by it --
// a blanket suppression in a header is how a genuine uninitialized-use warning goes unseen in
// every file that includes it.
//
// CURRENTLY INERT ON EVERY GCC AVAILABLE HERE, checked 2026-08-27 by neutralising the pragma and
// recompiling: host GCC 15.2.0, aarch64-linux-gnu-g++ 13.3.0, and x86_64-w64-mingw32-g++ 13 all
// report ZERO maybe-uninitialized on wxf.cpp, and mingw reports zero on the variant-heavy
// test_wxf.cpp as well. It is kept because those three are not every toolchain this ships to and
// the pragma costs nothing where the warning does not fire; it retires when the same check is run
// across the full platform set with the pragma removed and stays clean.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
struct WXFValue;

using WXFValueList = std::vector<WXFValue>;
using WXFValueAssociation = std::vector<std::pair<WXFValue, WXFValue>>;  // Key-value pairs, keys can be any expression

struct WXFValue {
    std::variant<
        std::monostate,           // Null/empty
        int64_t,                  // Integer
        double,                   // Real
        std::string,              // String or Symbol
        std::vector<uint8_t>,     // BinaryString
        WXFValueList,             // List of values
        WXFValueAssociation       // Association (arbitrary keys and values)
    > data;

    WXFValue() : data(std::monostate{}) {}

    template<typename T,
             typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, WXFValue>>>
    WXFValue(T&& value) : data(std::forward<T>(value)) {}

    template<typename T>
    T& get() { return std::get<T>(data); }

    template<typename T>
    const T& get() const { return std::get<T>(data); }

    template<typename T>
    bool holds() const { return std::holds_alternative<T>(data); }
};
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

/**
 * Type traits for WXF serialization support
 */
template<typename T>
struct is_wxf_serializable : std::false_type {};

// Specializations for supported types
template<> struct is_wxf_serializable<int8_t> : std::true_type {};
template<> struct is_wxf_serializable<int16_t> : std::true_type {};
template<> struct is_wxf_serializable<int32_t> : std::true_type {};
template<> struct is_wxf_serializable<int64_t> : std::true_type {};
template<> struct is_wxf_serializable<double> : std::true_type {};
template<> struct is_wxf_serializable<std::string> : std::true_type {};
template<> struct is_wxf_serializable<std::complex<double>> : std::true_type {};

template<typename T>
struct is_wxf_serializable<std::vector<T>> : is_wxf_serializable<T> {};

template<typename K, typename V>
struct is_wxf_serializable<std::unordered_map<K, V>>
    : std::conjunction<is_wxf_serializable<K>, is_wxf_serializable<V>> {};

// Type traits for generic container detection
template<typename T>
struct is_vector : std::false_type {};

template<typename T>
struct is_vector<std::vector<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

template<typename T>
struct is_map : std::false_type {};

template<typename K, typename V>
struct is_map<std::unordered_map<K, V>> : std::true_type {};

template<typename T>
inline constexpr bool is_map_v = is_map<T>::value;

/**
 * WXF Parser - Deserializes WXF binary data to C++ types
 */
class Parser {
private:
    const uint8_t* data_;
    size_t size_;
    size_t read_position_;

public:
    explicit Parser(const uint8_t* data, size_t size);
    explicit Parser(const std::vector<uint8_t>& data);

    // Core reading methods
    uint8_t read_byte();
    size_t read_varint();
    void skip_header();

    // Type-specific readers
    int8_t read_int8();
    int16_t read_int16();
    int32_t read_int32();
    int64_t read_int64();
    float read_real32();
    double read_real64();
    std::string read_string();
    std::string read_symbol();
    std::string read_big_integer();
    std::string read_big_real();
    std::vector<uint8_t> read_binary_string();
    std::complex<float> read_complex32();
    std::complex<double> read_complex64();

    // Template-based generic reader
    template<typename T>
    T read();

    // Callback-driven structured readers
    using AssociationCallback = std::function<void(const std::string&, Parser&)>;
    using GenericAssociationCallback = std::function<void(Parser&, Parser&)>;
    using FunctionCallback = std::function<void(const std::string&, size_t, Parser&)>;

    void read_association(const AssociationCallback& callback);
    void read_association_generic(const GenericAssociationCallback& callback);
    void read_function(const FunctionCallback& callback);

    // Utility methods
    size_t position() const noexcept;
    // Where this parser's view begins. read_association hands each value a SUB-PARSER over the
    // remaining bytes, so its position() counts from that value's first byte and says nothing
    // about the offset into the original buffer. A caller that wants a value's bytes needs the
    // base too, and computing the offset from position() alone yields the start of the whole
    // stream for every value -- a slice that compares equal for any two outputs.
    const uint8_t* data() const noexcept;
    size_t remaining() const noexcept;
    bool at_end() const noexcept;
    // Restore the cursor to a position previously obtained from position(). A read that
    // throws mid-value leaves the cursor inside that value; error recovery seeks back to
    // the value's start and skip_value()s the whole thing, so the stream stays aligned
    // for every later read. Seeking anywhere else breaks token alignment.
    void seek(size_t pos);
    // Skip over any WXF value (atomic or structured).
    //
    // Structured values recurse, and the input decides how deep -- a Rule token costs ONE
    // BYTE and recurses twice, so a small crafted message can drive the stack past its limit
    // and crash the process. WXF arrives from outside, so the depth is bounded rather than
    // trusted; exceeding it is a malformed message, not a fatal condition.
    static constexpr size_t MAX_SKIP_DEPTH = 512;
    void skip_value();
    void skip_value(size_t depth);

private:
    void ensure_bytes(size_t count);
    Token peek_token();
    // Step over a PackedArray or NumericArray body: element type, rank, the dimensions,
    // then the product of the dimensions times the element size.
    void skip_array();
    // Both big-number tokens have one wire shape; one body reads either.
    std::string read_big_number(Token expected, const char* what);
};

/**
 * WXF Writer - Serializes C++ types to WXF binary data
 */
class Writer {
private:
    std::vector<uint8_t> data_;

public:
    Writer() = default;

    // Core writing methods
    void write_byte(uint8_t value);
    void write_varint(size_t value);
    void write_header();

    // Type-specific writers
    void write_int8(int8_t value);
    void write_int16(int16_t value);
    void write_int32(int32_t value);
    void write_int64(int64_t value);
    void write_real64(double value);
    void write_string(const std::string& value);
    void write_symbol(const std::string& value);
    void write_binary_string(const std::vector<uint8_t>& value);

    // Template-based generic writer
    template<typename T>
    void write(const T& value);

    // Structured writers
    template<typename MapType>
    void write_association(const MapType& map);

    void write_function(const std::string& head, size_t arg_count);

    // Data access
    const std::vector<uint8_t>& data() const noexcept;
    std::vector<uint8_t> release_data() noexcept;
    void clear() noexcept;
    size_t size() const noexcept;

    // Reserve output capacity up front so a large serialization grows the single
    // backing buffer at most once instead of on every doubling. Capacity only; the
    // emitted bytes are unaffected.
    void reserve(std::size_t n);

    // Splice an already-serialized byte run into the stream. Used to compose a
    // top-level association whose element count is known only after its largest
    // sections have been streamed into a scratch Writer: write the association
    // header, append the streamed section bytes, then emit the remaining pairs.
    void append(const std::vector<uint8_t>& bytes);
};

// Template implementations

template<typename T>
T Parser::read() {
    // Check for unimplemented WXF types first and provide helpful error messages
    Token token = peek_token();
    if (token == Token::BigInteger) {
        throw ParseError("BigInteger not implemented - requires arbitrary precision library", read_position_);
    } else if (token == Token::BigReal) {
        throw ParseError("BigReal not implemented - requires arbitrary precision library", read_position_);
    } else if (token == Token::DelayedRule) {
        throw ParseError("DelayedRule not implemented", read_position_);
    } else if (token == Token::PackedArray) {
        throw ParseError("PackedArray not implemented", read_position_);
    } else if (token == Token::NumericArray) {
        throw ParseError("NumericArray not implemented", read_position_);
    }

    if constexpr (std::is_integral_v<T>) {
        // Read any integer type and convert to requested type with safe upcasting
        if (token == Token::Integer8) {
            int8_t value = read_int8();
            if constexpr (sizeof(T) < sizeof(int8_t)) {
                throw TypeError("Cannot narrow int8 to smaller type", read_position_);
            }
            return static_cast<T>(value);
        } else if (token == Token::Integer16) {
            int16_t value = read_int16();
            if constexpr (sizeof(T) < sizeof(int16_t)) {
                throw TypeError("Cannot narrow int16 to smaller type", read_position_);
            }
            return static_cast<T>(value);
        } else if (token == Token::Integer32) {
            int32_t value = read_int32();
            if constexpr (sizeof(T) < sizeof(int32_t)) {
                throw TypeError("Cannot narrow int32 to smaller type", read_position_);
            }
            return static_cast<T>(value);
        } else if (token == Token::Integer64) {
            int64_t value = read_int64();
            if constexpr (sizeof(T) < sizeof(int64_t)) {
                throw TypeError("Cannot narrow int64 to smaller type", read_position_);
            }
            return static_cast<T>(value);
        } else {
            throw TypeError("Expected integer type", read_position_);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        return read_real64();
    } else if constexpr (std::is_same_v<T, std::string>) {
        // Handle both String and Symbol tokens for std::string type
        // This allows reading Wolfram Symbols (True, False, etc.) as strings.
        // `token` is the one peeked at the top of this function: nothing has consumed a byte
        // since, so re-peeking would return the same value under a second name.
        if (token == Token::String) {
            return read_string();
        } else if (token == Token::Symbol) {
            return read_symbol();
        } else {
            throw TypeError("Expected String or Symbol token for std::string", read_position_);
        }
    } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
        return read_binary_string();
    } else if constexpr (is_vector_v<T>) {
        // Generic vector support - recursive template
        if (token != Token::Function) {
            throw TypeError("Expected List function", read_position_);
        }

        read_byte(); // consume 'f'
        size_t len = read_varint();

        // Skip List symbol
        std::string head = read_symbol();
        if (head != "List") {
            throw TypeError("Expected List function, got " + head, read_position_);
        }

        T result;
        // `len` is an unchecked varint from the input, so reserving on it lets a 14-byte
        // message ask for tens of gigabytes. Every element costs at least one more byte on
        // the wire, so a length past what remains in the buffer is provably a malformed
        // message -- reject it instead of allocating for it. read_string, read_symbol and
        // read_binary_string already bound themselves this way via ensure_bytes.
        if (len > remaining()) {
            throw ParseError("WXF list length exceeds the remaining input", read_position_);
        }
        result.reserve(len);

        using ElementType = typename T::value_type;
        for (size_t i = 0; i < len; ++i) {
            result.push_back(read<ElementType>());
        }

        return result;
    } else {
        static_assert(is_wxf_serializable<T>::value, "Type not supported for WXF serialization");
    }
}

template<typename T>
void Writer::write(const T& value) {
    if constexpr (std::is_same_v<T, const char*> || std::is_same_v<T, char*>) {
        // Handle C-style string literals
        write_string(std::string(value));
    } else if constexpr (std::is_array_v<T> && std::is_same_v<std::remove_extent_t<T>, char>) {
        // Handle char array literals like "string"
        write_string(std::string(value));
    } else if constexpr (std::is_integral_v<T>) {
        // Use smallest integer type that can hold the value (like Wolfram does)
        if (value >= std::numeric_limits<int8_t>::min() && value <= std::numeric_limits<int8_t>::max()) {
            write_int8(static_cast<int8_t>(value));
        } else if (value >= std::numeric_limits<int16_t>::min() && value <= std::numeric_limits<int16_t>::max()) {
            write_int16(static_cast<int16_t>(value));
        } else if (value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max()) {
            write_int32(static_cast<int32_t>(value));
        } else {
            write_int64(static_cast<int64_t>(value));
        }
    } else if constexpr (std::is_same_v<T, double>) {
        write_real64(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        write_string(value);
    } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
        write_binary_string(value);
    } else if constexpr (is_vector_v<T>) {
        // Generic vector support - recursive template
        write_function("List", value.size());
        for (const auto& item : value) {
            write(item);  // Recursively serialize each element
        }
    } else if constexpr (is_map_v<T>) {
        // Generic map support - recursive template
        write_association(value);
    } else {
        static_assert(is_wxf_serializable<T>::value, "Type not supported for WXF serialization");
    }
}

template<typename MapType>
void Writer::write_association(const MapType& map) {
    write_byte(static_cast<uint8_t>(Token::Association));
    write_varint(map.size());

    for (const auto& [key, value] : map) {
        write_byte(static_cast<uint8_t>(Token::Rule));
        write(key);
        write(value);
    }
}

// Specialization for WXFValue type - supports heterogeneous nesting. A full specialization is
// a function, not a template, so its body is in wxf.cpp.
template<>
void Writer::write<WXFValue>(const WXFValue& value);

// Convenience functions for common operations
template<typename T>
std::vector<uint8_t> serialize(const T& value) {
    Writer writer;
    writer.write_header();
    writer.write(value);
    return writer.release_data();
}

template<typename T>
T deserialize(const std::vector<uint8_t>& data) {
    Parser parser(data);
    parser.skip_header();
    return parser.read<T>();
}

template<typename T>
T deserialize(const uint8_t* data, size_t size) {
    Parser parser(data, size);
    parser.skip_header();
    return parser.read<T>();
}

}  // namespace wxf
}  // namespace HG_NAMESPACE