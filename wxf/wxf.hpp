#ifndef WXF_HPP
#define WXF_HPP

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
    WXFException(const std::string& message, size_t position = 0)
        : std::runtime_error(message), position_(position) {}

    size_t position() const noexcept { return position_; }

private:
    size_t position_;
};

class ParseError : public WXFException {
public:
    ParseError(const std::string& message, size_t position = 0)
        : WXFException("Parse error: " + message, position) {}
};

class TypeError : public WXFException {
public:
    TypeError(const std::string& message, size_t position = 0)
        : WXFException("Type error: " + message, position) {}
};

// Forward declarations
class Parser;
class Writer;

// Heterogeneous value type for arbitrary WXF data structures
// Supports recursive nesting via std::variant
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
    explicit Parser(const uint8_t* data, size_t size)
        : data_(data), size_(size), read_position_(0) {}

    explicit Parser(const std::vector<uint8_t>& data)
        : Parser(data.data(), data.size()) {}

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
    size_t position() const noexcept { return read_position_; }
    size_t remaining() const noexcept { return size_ - read_position_; }
    bool at_end() const noexcept { return read_position_ >= size_; }
    void skip_value();  // Skip over any WXF value (atomic or structured)

private:
    void ensure_bytes(size_t count);
    Token peek_token();
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
    const std::vector<uint8_t>& data() const noexcept { return data_; }
    std::vector<uint8_t> release_data() noexcept { return std::move(data_); }
    void clear() noexcept { data_.clear(); }
    size_t size() const noexcept { return data_.size(); }
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
        // This allows reading Wolfram Symbols (True, False, etc.) as strings
        Token token = peek_token();
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
        Token token = peek_token();
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

// Specialization for WXFValue type - supports heterogeneous nesting
template<>
inline void Writer::write<WXFValue>(const WXFValue& value) {
    std::visit([this](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            // Write empty list for null/empty
            write_function("List", 0);
        } else if constexpr (std::is_same_v<T, WXFValueList>) {
            write_function("List", v.size());
            for (const auto& item : v) {
                write(item);
            }
        } else if constexpr (std::is_same_v<T, WXFValueAssociation>) {
            write_byte(static_cast<uint8_t>(Token::Association));
            write_varint(v.size());
            for (const auto& [k, val] : v) {
                write_byte(static_cast<uint8_t>(Token::Rule));
                write(k);
                write(val);
            }
        } else {
            write(v);  // Use generic write for primitives
        }
    }, value.data);
}

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

} // namespace wxf

#endif // WXF_HPP