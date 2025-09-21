#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cstdint>
#include <type_traits>
#include <stdexcept>

namespace wxf {

class Parser {
private:
    const uint8_t* data;
    size_t size;
    size_t pos;

public:
    Parser(const uint8_t* d, size_t s) : data(d), size(s), pos(0) {}

    uint8_t read_byte();
    size_t read_varint();
    std::string read_string();
    std::string read_symbol();

    // Templated read function that infers return type
    template<typename T>
    T read();

    // Generic association reader
    template<typename KeyValueHandler>
    void read_association(KeyValueHandler handler);

    // Generic function reader
    template<typename FunctionHandler>
    void read_function(FunctionHandler handler);

    void skip_header();

private:
    // Type-specific implementations
    int64_t read_integer_impl();
    std::vector<int64_t> read_list_impl();
    std::vector<std::vector<int64_t>> read_list_of_lists_impl();
};

class Serializer {
private:
    std::vector<uint8_t> data;

public:
    void write_header();
    void write_varint(size_t value);
    void write_integer(int64_t value);
    void write_string(const std::string& str);
    void write_symbol(const std::string& symbol);

    void begin_list(size_t size);
    void begin_association(size_t size);
    void write_association_rule(const std::string& key);

    template<typename ValueWriter>
    void write_association_entry(const std::string& key, ValueWriter writer);

    const std::vector<uint8_t>& get_data() const { return data; }
    std::vector<uint8_t>&& move_data() { return std::move(data); }
};

// Template specializations for Parser::read()
template<>
inline int64_t Parser::read<int64_t>() {
    return read_integer_impl();
}

template<>
inline std::vector<int64_t> Parser::read<std::vector<int64_t>>() {
    return read_list_impl();
}

template<>
inline std::vector<std::vector<int64_t>> Parser::read<std::vector<std::vector<int64_t>>>() {
    return read_list_of_lists_impl();
}

template<>
inline std::string Parser::read<std::string>() {
    return read_string();
}

// Template implementations that must be in header
template<typename KeyValueHandler>
void Parser::read_association(KeyValueHandler handler) {
    uint8_t type = read_byte();
    if (type != 'A') throw std::runtime_error("Expected association in WXF");

    size_t num_entries = read_varint();
    for(size_t i = 0; i < num_entries; i++) {
        type = read_byte();
        if (type != '-') throw std::runtime_error("Expected rule marker in WXF association");

        std::string key = read_string();
        handler(key, *this);
    }
}

template<typename FunctionHandler>
void Parser::read_function(FunctionHandler handler) {
    uint8_t type = read_byte();
    if (type != 'f') throw std::runtime_error("Expected function in WXF");

    size_t arg_count = read_varint();
    std::string function_name = read_symbol();

    handler(function_name, arg_count, *this);
}

template<typename ValueWriter>
void Serializer::write_association_entry(const std::string& key, ValueWriter writer) {
    data.push_back('-');
    write_string(key);
    writer(*this);
}

} // namespace wxf