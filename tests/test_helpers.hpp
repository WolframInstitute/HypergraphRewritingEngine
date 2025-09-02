#pragma once
#include <gtest/gtest.h>
#include <hypergraph/hypergraph.hpp>
#include <hypergraph/canonicalization.hpp>
#include <vector>
#include <chrono>

namespace test_utils {

/**
 * Create a simple hypergraph with given edge structures
 */
inline hypergraph::Hypergraph create_test_hypergraph(const std::vector<std::vector<hypergraph::VertexId>>& edges) {
    hypergraph::Hypergraph hg;
    for (const auto& edge : edges) {
        hg.add_edge(edge);
    }
    return hg;
}

/**
 * Performance measurement utility
 */
class PerfTimer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    PerfTimer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

/**
 * Expectation helpers for hypergraph comparisons
 */
inline void expect_canonical_equal(const hypergraph::Hypergraph& hg1, const hypergraph::Hypergraph& hg2) {
    hypergraph::Canonicalizer canonicalizer;
    auto canon1 = canonicalizer.canonicalize(hg1);
    auto canon2 = canonicalizer.canonicalize(hg2);
    EXPECT_EQ(canon1.canonical_form, canon2.canonical_form) 
        << "Hypergraphs should have same canonical form: " 
        << canon1.canonical_form.to_string() << " vs " << canon2.canonical_form.to_string();
}

inline void expect_canonical_different(const hypergraph::Hypergraph& hg1, const hypergraph::Hypergraph& hg2) {
    hypergraph::Canonicalizer canonicalizer;
    auto canon1 = canonicalizer.canonicalize(hg1);
    auto canon2 = canonicalizer.canonicalize(hg2);
    EXPECT_NE(canon1.canonical_form, canon2.canonical_form)
        << "Hypergraphs should have different canonical forms but both are: " 
        << canon1.canonical_form.to_string();
}

} // namespace test_utils