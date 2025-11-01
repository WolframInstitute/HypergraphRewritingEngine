// BENCHMARK_CATEGORY: WXF Serialization

#include "benchmark_framework.hpp"
#include <wxf.hpp>
#include <vector>
#include <string>

using namespace benchmark;

// Helper to generate nested data structures of varying complexity
std::vector<int> generate_int_list(int size) {
    std::vector<int> data;
    data.reserve(size);
    for (int i = 0; i < size; ++i) {
        data.push_back(i);
    }
    return data;
}

std::vector<std::vector<int>> generate_nested_list(int outer_size, int inner_size) {
    std::vector<std::vector<int>> data;
    data.reserve(outer_size);
    for (int i = 0; i < outer_size; ++i) {
        data.push_back(generate_int_list(inner_size));
    }
    return data;
}

BENCHMARK(wxf_serialize_flat_list, "Measures WXF serialization time for flat integer lists of varying sizes") {
    for (int size : {10, 50, 100, 500, 1000, 5000}) {
        BENCHMARK_PARAM("size", size);

        auto data = generate_int_list(size);

        BENCHMARK_CODE([&]() {
            auto serialized = wxf::serialize(data);
        });
    }
}

BENCHMARK(wxf_deserialize_flat_list, "Measures WXF deserialization time for flat integer lists of varying sizes") {
    for (int size : {10, 50, 100, 500, 1000, 5000}) {
        BENCHMARK_PARAM("size", size);

        auto data = generate_int_list(size);
        auto serialized = wxf::serialize(data);

        BENCHMARK_CODE([&]() {
            auto deserialized = wxf::deserialize<std::vector<int>>(serialized);
        });
    }
}

BENCHMARK(wxf_serialize_nested_list, "Measures WXF serialization time for nested lists (outer_size x inner_size)") {
    for (int outer : {10, 20, 50, 100}) {
        for (int inner : {10, 50, 100}) {
            BENCHMARK_PARAM("outer_size", outer);
            BENCHMARK_PARAM("inner_size", inner);

            auto data = generate_nested_list(outer, inner);

            BENCHMARK_CODE([&]() {
                auto serialized = wxf::serialize(data);
            });
        }
    }
}

BENCHMARK(wxf_deserialize_nested_list, "Measures WXF deserialization time for nested lists (outer_size x inner_size)") {
    for (int outer : {10, 20, 50, 100}) {
        for (int inner : {10, 50, 100}) {
            BENCHMARK_PARAM("outer_size", outer);
            BENCHMARK_PARAM("inner_size", inner);

            auto data = generate_nested_list(outer, inner);
            auto serialized = wxf::serialize(data);

            BENCHMARK_CODE([&]() {
                auto deserialized = wxf::deserialize<std::vector<std::vector<int>>>(serialized);
            });
        }
    }
}

BENCHMARK(wxf_roundtrip, "Measures WXF round-trip (serialize + deserialize) time for various data sizes") {
    for (int size : {10, 50, 100, 500, 1000}) {
        BENCHMARK_PARAM("size", size);

        auto data = generate_int_list(size);

        BENCHMARK_CODE([&]() {
            auto serialized = wxf::serialize(data);
            auto deserialized = wxf::deserialize<std::vector<int>>(serialized);
        });
    }
}
