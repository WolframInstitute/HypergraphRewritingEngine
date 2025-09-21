#include "wxf.hpp"
#include <stdexcept>
#include <climits>

namespace wxf {

uint8_t Parser::read_byte() {
    if (pos >= size) throw std::runtime_error("WXF data underflow");
    return data[pos++];
}

size_t Parser::read_varint() {
    size_t value = 0;
    size_t shift = 0;
    uint8_t byte;
    do {
        byte = read_byte();
        value |= (size_t(byte & 0x7F) << shift);
        shift += 7;
    } while (byte & 0x80);
    return value;
}

std::string Parser::read_string() {
    uint8_t type = read_byte();
    if (type != 'S') throw std::runtime_error("Expected string in WXF");
    size_t len = read_varint();
    std::string str(len, '\0');
    for(size_t i = 0; i < len; i++) {
        str[i] = read_byte();
    }
    return str;
}

std::string Parser::read_symbol() {
    uint8_t type = read_byte();
    if (type != 's') throw std::runtime_error("Expected symbol in WXF");
    size_t len = read_varint();
    std::string str(len, '\0');
    for(size_t i = 0; i < len; i++) {
        str[i] = read_byte();
    }
    return str;
}

int64_t Parser::read_integer_impl() {
    uint8_t type = read_byte();
    switch(type) {
        case 'C': { // 8-bit
            int8_t val = read_byte();
            return val;
        }
        case 'j': { // 16-bit
            int16_t val = 0;
            for(int i = 0; i < 2; i++) {
                val |= (read_byte() << (i * 8));
            }
            return val;
        }
        case 'i': { // 32-bit
            int32_t val = 0;
            for(int i = 0; i < 4; i++) {
                val |= (read_byte() << (i * 8));
            }
            return val;
        }
        case 'L': { // 64-bit
            int64_t val = 0;
            for(int i = 0; i < 8; i++) {
                val |= (static_cast<int64_t>(read_byte()) << (i * 8));
            }
            return val;
        }
        default:
            throw std::runtime_error("Unknown integer type in WXF");
    }
}

std::vector<int64_t> Parser::read_list_impl() {
    uint8_t type = read_byte();
    if (type != 'f') throw std::runtime_error("Expected function (List) in WXF");

    size_t len = read_varint();

    // Skip List symbol
    std::string symbol = read_symbol();
    if (symbol != "List") throw std::runtime_error("Expected List symbol in WXF");

    std::vector<int64_t> result;
    result.reserve(len);
    for(size_t i = 0; i < len; i++) {
        result.push_back(read_integer_impl());
    }
    return result;
}

std::vector<std::vector<int64_t>> Parser::read_list_of_lists_impl() {
    uint8_t type = read_byte();
    if (type != 'f') throw std::runtime_error("Expected function (List) in WXF");

    size_t outer_len = read_varint();

    // Skip List symbol
    std::string symbol = read_symbol();
    if (symbol != "List") throw std::runtime_error("Expected List symbol in WXF");

    std::vector<std::vector<int64_t>> result;
    result.reserve(outer_len);

    for(size_t i = 0; i < outer_len; i++) {
        result.push_back(read_list_impl());
    }

    return result;
}

void Parser::skip_header() {
    // Skip WXF header "8:"
    if (read_byte() != '8' || read_byte() != ':') {
        throw std::runtime_error("Invalid WXF header");
    }
}

// Serializer implementation
void Serializer::write_header() {
    data.push_back('8');
    data.push_back(':');
}

void Serializer::write_varint(size_t value) {
    while (value >= 0x80) {
        data.push_back(0x80 | (value & 0x7F));
        value >>= 7;
    }
    data.push_back(value & 0x7F);
}

void Serializer::write_integer(int64_t value) {
    if (value >= -128 && value <= 127) {
        data.push_back('C');
        data.push_back(static_cast<uint8_t>(value & 0xFF));
    } else if (value >= INT16_MIN && value <= INT16_MAX) {
        data.push_back('j');
        int16_t val16 = static_cast<int16_t>(value);
        data.push_back(val16 & 0xFF);
        data.push_back((val16 >> 8) & 0xFF);
    } else if (value >= INT32_MIN && value <= INT32_MAX) {
        data.push_back('i');
        int32_t val32 = static_cast<int32_t>(value);
        for (int i = 0; i < 4; ++i) {
            data.push_back(val32 & 0xFF);
            val32 >>= 8;
        }
    } else {
        data.push_back('L');
        for (int i = 0; i < 8; ++i) {
            data.push_back(value & 0xFF);
            value >>= 8;
        }
    }
}

void Serializer::write_string(const std::string& str) {
    data.push_back('S');
    write_varint(str.size());
    data.insert(data.end(), str.begin(), str.end());
}

void Serializer::write_symbol(const std::string& symbol) {
    data.push_back('s');
    write_varint(symbol.size());
    data.insert(data.end(), symbol.begin(), symbol.end());
}

void Serializer::begin_list(size_t size) {
    data.push_back('f');
    write_varint(size);
    write_symbol("List");
}

void Serializer::begin_association(size_t size) {
    data.push_back('A');
    write_varint(size);
}

void Serializer::write_association_rule(const std::string& key) {
    data.push_back('-');
    write_string(key);
}

} // namespace wxf