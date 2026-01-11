#include "wxf.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace wxf {

// Parser implementation

uint8_t Parser::read_byte() {
    ensure_bytes(1);
    return data_[read_position_++];
}

size_t Parser::read_varint() {
    size_t value = 0;
    size_t shift = 0;
    uint8_t byte;

    do {
        byte = read_byte();
        if (shift >= 63) {
            throw ParseError("Varint too large", read_position_ - 1);
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
        throw TypeError("Expected 8-bit integer", read_position_);
    }
    read_byte(); // consume token
    return static_cast<int8_t>(read_byte());
}

int16_t Parser::read_int16() {
    Token token = peek_token();
    if (token != Token::Integer16) {
        throw TypeError("Expected 16-bit integer", read_position_);
    }
    read_byte(); // consume token

    ensure_bytes(2);
    int16_t value = 0;
    // Little-endian encoding
    for (int i = 0; i < 2; i++) {
        value |= (static_cast<int16_t>(data_[read_position_ + i]) << (i * 8));
    }
    read_position_ += 2;
    return value;
}

int32_t Parser::read_int32() {
    Token token = peek_token();
    if (token != Token::Integer32) {
        throw TypeError("Expected 32-bit integer", read_position_);
    }
    read_byte(); // consume token

    ensure_bytes(4);
    int32_t value = 0;
    // Little-endian encoding
    for (int i = 0; i < 4; i++) {
        value |= (static_cast<int32_t>(data_[read_position_ + i]) << (i * 8));
    }
    read_position_ += 4;
    return value;
}

int64_t Parser::read_int64() {
    Token token = peek_token();
    if (token != Token::Integer64) {
        throw TypeError("Expected 64-bit integer", read_position_);
    }
    read_byte(); // consume token

    ensure_bytes(8);
    int64_t value = 0;
    // Little-endian encoding
    for (int i = 0; i < 8; i++) {
        value |= (static_cast<int64_t>(data_[read_position_ + i]) << (i * 8));
    }
    read_position_ += 8;
    return value;
}

double Parser::read_real64() {
    Token token = peek_token();
    if (token != Token::Real64) {
        throw TypeError("Expected 64-bit real", read_position_);
    }
    read_byte(); // consume token

    ensure_bytes(8);
    double value;
    std::memcpy(&value, &data_[read_position_], 8);
    read_position_ += 8;

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
        throw TypeError("Expected string", read_position_);
    }
    read_byte(); // consume token

    size_t len = read_varint();
    ensure_bytes(len);

    std::string result(len, '\0');
    std::memcpy(result.data(), &data_[read_position_], len);
    read_position_ += len;

    // Note: WXF supports full UTF-8, skipping validation for now
    // TODO: Implement proper UTF-8 validation if needed

    return result;
}

std::string Parser::read_symbol() {
    Token token = peek_token();
    if (token != Token::Symbol) {
        throw TypeError("Expected symbol", read_position_);
    }
    read_byte(); // consume token

    size_t len = read_varint();
    ensure_bytes(len);

    std::string result(len, '\0');
    std::memcpy(result.data(), &data_[read_position_], len);
    read_position_ += len;

    return result;
}

std::vector<uint8_t> Parser::read_binary_string() {
    Token token = peek_token();
    if (token != Token::BinaryString) {
        throw TypeError("Expected binary string", read_position_);
    }
    read_byte(); // consume token

    size_t len = read_varint();
    ensure_bytes(len);

    std::vector<uint8_t> result(len);
    std::memcpy(result.data(), &data_[read_position_], len);
    read_position_ += len;

    return result;
}

std::string Parser::read_big_integer() {
    Token token = peek_token();
    if (token != Token::BigInteger) {
        throw TypeError("Expected big integer", read_position_);
    }
    throw ParseError("BigInteger not implemented - requires arbitrary precision library", read_position_);
}

std::string Parser::read_big_real() {
    Token token = peek_token();
    if (token != Token::BigReal) {
        throw TypeError("Expected big real", read_position_);
    }
    throw ParseError("BigReal not implemented - requires arbitrary precision library", read_position_);
}

void Parser::read_association(const AssociationCallback& callback) {
    Token token = peek_token();
    if (token != Token::Association) {
        char err[256];
        snprintf(err, sizeof(err), "Expected association at pos %zu, got token 0x%02X", read_position_, static_cast<uint8_t>(token));
        throw TypeError(err, read_position_);
    }
    read_byte(); // consume 'A'

    size_t num_entries = read_varint();

    for (size_t i = 0; i < num_entries; i++) {
        Token rule_token = peek_token();
        if (rule_token != Token::Rule) {
            char err[256];
            snprintf(err, sizeof(err), "Expected rule marker in association at pos %zu (entry %zu/%zu), got 0x%02X",
                     read_position_, i+1, num_entries, static_cast<uint8_t>(rule_token));
            throw TypeError(err, read_position_);
        }
        read_byte(); // consume '-'

        // Read key - can be String, Symbol, or any other type (including integers)
        std::string key;
        Token key_token = peek_token();
        if (key_token == Token::String) {
            key = read_string();
        } else if (key_token == Token::Symbol) {
            key = read_symbol();
        } else if (key_token == Token::Integer8 || key_token == Token::Integer16 ||
                   key_token == Token::Integer32 || key_token == Token::Integer64) {
            // Convert integer keys to strings for uniform interface
            if (key_token == Token::Integer8) {
                key = std::to_string(read_int8());
            } else if (key_token == Token::Integer16) {
                key = std::to_string(read_int16());
            } else if (key_token == Token::Integer32) {
                key = std::to_string(read_int32());
            } else {
                key = std::to_string(read_int64());
            }
        } else {
            throw TypeError("Association key must be String, Symbol, or Integer", read_position_);
        }

        // Create a sub-parser for the value
        Parser value_parser(data_ + read_position_, size_ - read_position_);
        callback(key, value_parser);

        // Advance our position by how much the callback consumed
        read_position_ += value_parser.position();
    }
}

void Parser::skip_value() {
    Token token = peek_token();

    switch (token) {
        case Token::Integer8:
            read_int8();
            break;
        case Token::Integer16:
            read_int16();
            break;
        case Token::Integer32:
            read_int32();
            break;
        case Token::Integer64:
            read_int64();
            break;
        case Token::Real64:
            read_real64();
            break;
        case Token::String:
            read_string();
            break;
        case Token::Symbol:
            read_symbol();
            break;
        case Token::BinaryString:
            read_binary_string();
            break;
        case Token::Function:
            read_function([](const std::string&, size_t arg_count, Parser& p) {
                // Skip all arguments recursively
                for (size_t i = 0; i < arg_count; i++) {
                    p.skip_value();
                }
            });
            break;
        case Token::Association:
            read_association_generic([](Parser& k, Parser& v) {
                // Skip both key and value recursively
                k.skip_value();
                v.skip_value();
            });
            break;
        case Token::Rule:
            read_byte(); // consume '-'
            skip_value(); // skip key
            skip_value(); // skip value
            break;
        default:
            throw TypeError("Cannot skip unsupported or invalid token", read_position_);
    }
}

void Parser::read_association_generic(const GenericAssociationCallback& callback) {
    Token token = peek_token();
    if (token != Token::Association) {
        throw TypeError("Expected association", read_position_);
    }
    read_byte(); // consume 'A'

    size_t num_entries = read_varint();

    for (size_t i = 0; i < num_entries; i++) {
        Token rule_token = peek_token();
        if (rule_token != Token::Rule) {
            char err[256];
            snprintf(err, sizeof(err), "Expected rule marker in association (generic) at pos %zu (entry %zu/%zu), got 0x%02X",
                     read_position_, i+1, num_entries, static_cast<uint8_t>(rule_token));
            throw TypeError(err, read_position_);
        }
        read_byte(); // consume '-'

        // Measure the key length by skipping it in a temporary parser
        Parser key_measurer(data_ + read_position_, size_ - read_position_);
        key_measurer.skip_value();
        size_t key_length = key_measurer.position();

        // Create properly positioned parsers
        Parser key_parser(data_ + read_position_, key_length);
        read_position_ += key_length;

        Parser value_parser(data_ + read_position_, size_ - read_position_);
        callback(key_parser, value_parser);

        // Advance our position by how much the value callback consumed
        read_position_ += value_parser.position();
    }
}

void Parser::read_function(const FunctionCallback& callback) {
    Token token = peek_token();
    if (token != Token::Function) {
        throw TypeError("Expected function", read_position_);
    }
    read_byte(); // consume 'f'

    size_t arg_count = read_varint();

    // Read the head (function name)
    std::string head = read_symbol();

    // Create a sub-parser for the arguments
    Parser args_parser(data_ + read_position_, size_ - read_position_);
    callback(head, arg_count, args_parser);

    // Advance our position by how much the callback consumed
    read_position_ += args_parser.position();
}

void Parser::ensure_bytes(size_t count) {
    if (read_position_ + count > size_) {
        throw ParseError("Unexpected end of WXF data", read_position_);
    }
}

Token Parser::peek_token() {
    ensure_bytes(1);
    return static_cast<Token>(data_[read_position_]);
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