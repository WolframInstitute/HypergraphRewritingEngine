#include "wxf.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace wxf {

// Parser implementation

uint8_t Parser::read_byte() {
    ensure_bytes(1);
    return data_[pos_++];
}

size_t Parser::read_varint() {
    size_t value = 0;
    size_t shift = 0;
    uint8_t byte;

    do {
        byte = read_byte();
        if (shift >= 63) {
            throw ParseError("Varint too large", pos_ - 1);
        }
        value |= (size_t(byte & 0x7F) << shift);
        shift += 7;
    } while (byte & 0x80);

    return value;
}

void Parser::skip_header() {
    // WXF header: "8:" for uncompressed, "C:" for compressed
    uint8_t first = read_byte();
    uint8_t second = read_byte();

    if (first == '8' && second == ':') {
        // Uncompressed format - good to go
        return;
    } else if (first == 'C' && second == ':') {
        throw ParseError("Compressed WXF format not supported", 0);
    } else {
        throw ParseError("Invalid WXF header", 0);
    }
}

int8_t Parser::read_int8() {
    Token token = peek_token();
    if (token != Token::Integer8) {
        throw TypeError("Expected 8-bit integer", pos_);
    }
    read_byte(); // consume token
    return static_cast<int8_t>(read_byte());
}

int16_t Parser::read_int16() {
    Token token = peek_token();
    if (token != Token::Integer16) {
        throw TypeError("Expected 16-bit integer", pos_);
    }
    read_byte(); // consume token

    ensure_bytes(2);
    int16_t value = 0;
    // Little-endian encoding
    for (int i = 0; i < 2; i++) {
        value |= (static_cast<int16_t>(data_[pos_ + i]) << (i * 8));
    }
    pos_ += 2;
    return value;
}

int32_t Parser::read_int32() {
    Token token = peek_token();
    if (token != Token::Integer32) {
        throw TypeError("Expected 32-bit integer", pos_);
    }
    read_byte(); // consume token

    ensure_bytes(4);
    int32_t value = 0;
    // Little-endian encoding
    for (int i = 0; i < 4; i++) {
        value |= (static_cast<int32_t>(data_[pos_ + i]) << (i * 8));
    }
    pos_ += 4;
    return value;
}

int64_t Parser::read_int64() {
    Token token = peek_token();
    if (token != Token::Integer64) {
        throw TypeError("Expected 64-bit integer", pos_);
    }
    read_byte(); // consume token

    ensure_bytes(8);
    int64_t value = 0;
    // Little-endian encoding
    for (int i = 0; i < 8; i++) {
        value |= (static_cast<int64_t>(data_[pos_ + i]) << (i * 8));
    }
    pos_ += 8;
    return value;
}

double Parser::read_real64() {
    Token token = peek_token();
    if (token != Token::Real64) {
        throw TypeError("Expected 64-bit real", pos_);
    }
    read_byte(); // consume token

    ensure_bytes(8);
    double value;
    std::memcpy(&value, &data_[pos_], 8);
    pos_ += 8;

    // Ensure little-endian interpretation on big-endian systems
    // Simple endianness check for C++17 compatibility
    uint16_t endian_test = 1;
    bool is_little_endian = (*reinterpret_cast<uint8_t*>(&endian_test) == 1);

    if (!is_little_endian) {
        uint64_t temp;
        std::memcpy(&temp, &value, 8);
        temp = __builtin_bswap64(temp);
        std::memcpy(&value, &temp, 8);
    }

    return value;
}

std::string Parser::read_string() {
    Token token = peek_token();
    if (token != Token::String) {
        throw TypeError("Expected string", pos_);
    }
    read_byte(); // consume token

    size_t len = read_varint();
    ensure_bytes(len);

    std::string result(len, '\0');
    std::memcpy(result.data(), &data_[pos_], len);
    pos_ += len;

    // Note: WXF supports full UTF-8, skipping validation for now
    // TODO: Implement proper UTF-8 validation if needed

    return result;
}

std::string Parser::read_symbol() {
    Token token = peek_token();
    if (token != Token::Symbol) {
        throw TypeError("Expected symbol", pos_);
    }
    read_byte(); // consume token

    size_t len = read_varint();
    ensure_bytes(len);

    std::string result(len, '\0');
    std::memcpy(result.data(), &data_[pos_], len);
    pos_ += len;

    return result;
}

std::vector<uint8_t> Parser::read_binary_string() {
    Token token = peek_token();
    if (token != Token::BinaryString) {
        throw TypeError("Expected binary string", pos_);
    }
    read_byte(); // consume token

    size_t len = read_varint();
    ensure_bytes(len);

    std::vector<uint8_t> result(len);
    std::memcpy(result.data(), &data_[pos_], len);
    pos_ += len;

    return result;
}

void Parser::read_association(const AssociationCallback& callback) {
    Token token = peek_token();
    if (token != Token::Association) {
        throw TypeError("Expected association", pos_);
    }
    read_byte(); // consume 'A'

    size_t num_entries = read_varint();

    for (size_t i = 0; i < num_entries; i++) {
        Token rule_token = peek_token();
        if (rule_token != Token::Rule) {
            throw TypeError("Expected rule marker in association", pos_);
        }
        read_byte(); // consume '-'

        std::string key = read_string();

        // Create a sub-parser for the value
        Parser value_parser(data_ + pos_, size_ - pos_);
        callback(key, value_parser);

        // Advance our position by how much the callback consumed
        pos_ += value_parser.position();
    }
}

void Parser::read_function(const FunctionCallback& callback) {
    Token token = peek_token();
    if (token != Token::Function) {
        throw TypeError("Expected function", pos_);
    }
    read_byte(); // consume 'f'

    size_t arg_count = read_varint();

    // Read the head (function name)
    std::string head = read_symbol();

    // Create a sub-parser for the arguments
    Parser args_parser(data_ + pos_, size_ - pos_);
    callback(head, arg_count, args_parser);

    // Advance our position by how much the callback consumed
    pos_ += args_parser.position();
}

void Parser::ensure_bytes(size_t count) {
    if (pos_ + count > size_) {
        throw ParseError("Unexpected end of WXF data", pos_);
    }
}

Token Parser::peek_token() {
    ensure_bytes(1);
    return static_cast<Token>(data_[pos_]);
}

// Writer implementation

void Writer::write_byte(uint8_t value) {
    data_.push_back(value);
}

void Writer::write_varint(size_t value) {
    while (value >= 0x80) {
        data_.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    data_.push_back(static_cast<uint8_t>(value & 0x7F));
}

void Writer::write_header() {
    // Write uncompressed WXF header
    data_.push_back('8');
    data_.push_back(':');
}

void Writer::write_int8(int8_t value) {
    write_byte(static_cast<uint8_t>(Token::Integer8));
    write_byte(static_cast<uint8_t>(value));
}

void Writer::write_int16(int16_t value) {
    write_byte(static_cast<uint8_t>(Token::Integer16));

    // Little-endian encoding
    for (int i = 0; i < 2; i++) {
        write_byte(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
    }
}

void Writer::write_int32(int32_t value) {
    write_byte(static_cast<uint8_t>(Token::Integer32));

    // Little-endian encoding
    for (int i = 0; i < 4; i++) {
        write_byte(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
    }
}

void Writer::write_int64(int64_t value) {
    write_byte(static_cast<uint8_t>(Token::Integer64));

    // Little-endian encoding
    for (int i = 0; i < 8; i++) {
        write_byte(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
    }
}

void Writer::write_real64(double value) {
    write_byte(static_cast<uint8_t>(Token::Real64));

    // Normalize -0.0 to 0.0 to match Wolfram behavior
    if (value == 0.0 && std::signbit(value)) {
        value = 0.0;
    }

    // Ensure little-endian encoding
    uint64_t bits;
    std::memcpy(&bits, &value, 8);

    // Simple endianness check for C++17 compatibility
    uint16_t endian_test = 1;
    bool is_little_endian = (*reinterpret_cast<uint8_t*>(&endian_test) == 1);

    if (!is_little_endian) {
        bits = __builtin_bswap64(bits);
    }

    for (int i = 0; i < 8; i++) {
        write_byte(static_cast<uint8_t>((bits >> (i * 8)) & 0xFF));
    }
}

void Writer::write_string(const std::string& value) {
    write_byte(static_cast<uint8_t>(Token::String));
    write_varint(value.size());

    for (char c : value) {
        write_byte(static_cast<uint8_t>(c));
    }
}

void Writer::write_symbol(const std::string& value) {
    write_byte(static_cast<uint8_t>(Token::Symbol));
    write_varint(value.size());

    for (char c : value) {
        write_byte(static_cast<uint8_t>(c));
    }
}

void Writer::write_binary_string(const std::vector<uint8_t>& value) {
    write_byte(static_cast<uint8_t>(Token::BinaryString));
    write_varint(value.size());

    for (uint8_t b : value) {
        write_byte(b);
    }
}

void Writer::write_function(const std::string& head, size_t arg_count) {
    write_byte(static_cast<uint8_t>(Token::Function));
    write_varint(arg_count);
    write_symbol(head);
}

} // namespace wxf