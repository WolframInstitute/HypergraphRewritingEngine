#include <blackhole/bh_serialization.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace viz::blackhole;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.bhdata>" << std::endl;
        return 1;
    }

    BHAnalysisResult result;
    if (!read_analysis(argv[1], result)) {
        std::cerr << "Failed to read: " << argv[1] << std::endl;
        return 1;
    }

    std::cout << "=== Analysis Summary ===" << std::endl;
    std::cout << "Total steps: " << result.total_steps << std::endl;
    std::cout << "Total states: " << result.total_states << std::endl;
    std::cout << "Anchors: " << result.anchor_vertices.size() << std::endl;
    std::cout << "Global dim range: [" << result.dim_min << ", " << result.dim_max << "]" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Per-Timestep Data ===" << std::endl;
    std::cout << std::setw(6) << "Step"
              << std::setw(8) << "States"
              << std::setw(10) << "Vertices"
              << std::setw(10) << "UnionE"
              << std::setw(10) << "IntersE"
              << std::setw(10) << "PoolMean"
              << std::setw(10) << "PoolMin"
              << std::setw(10) << "PoolMax"
              << std::setw(12) << "DeltaV"
              << std::setw(12) << "DeltaUE"
              << std::setw(12) << "DeltaIE"
              << std::endl;

    size_t prev_verts = 0, prev_union_edges = 0, prev_inter_edges = 0;

    for (size_t i = 0; i < result.per_timestep.size(); ++i) {
        const auto& ts = result.per_timestep[i];

        int delta_v = static_cast<int>(ts.union_vertices.size()) - static_cast<int>(prev_verts);
        int delta_ue = static_cast<int>(ts.union_edges.size()) - static_cast<int>(prev_union_edges);
        int delta_ie = static_cast<int>(ts.intersection_edges.size()) - static_cast<int>(prev_inter_edges);

        std::cout << std::setw(6) << ts.step
                  << std::setw(8) << ts.num_states
                  << std::setw(10) << ts.union_vertices.size()
                  << std::setw(10) << ts.union_edges.size()
                  << std::setw(10) << ts.intersection_edges.size()
                  << std::setw(10) << std::fixed << std::setprecision(3) << ts.pooled_mean
                  << std::setw(10) << ts.pooled_min
                  << std::setw(10) << ts.pooled_max;

        if (i > 0) {
            std::cout << std::setw(12) << (delta_v >= 0 ? "+" : "") << delta_v
                      << std::setw(12) << (delta_ue >= 0 ? "+" : "") << delta_ue
                      << std::setw(12) << (delta_ie >= 0 ? "+" : "") << delta_ie;

            // Flag large discontinuities
            if (std::abs(delta_ie) > 50 || std::abs(delta_ue) > 50) {
                std::cout << " <<<< DISCONTINUITY!";
            }
        }
        std::cout << std::endl;

        prev_verts = ts.union_vertices.size();
        prev_union_edges = ts.union_edges.size();
        prev_inter_edges = ts.intersection_edges.size();
    }

    // Find dimension value changes
    std::cout << std::endl << "=== Dimension Changes ===" << std::endl;
    float prev_pooled_mean = 0;
    for (size_t i = 0; i < result.per_timestep.size(); ++i) {
        const auto& ts = result.per_timestep[i];
        if (i > 0) {
            float delta = ts.pooled_mean - prev_pooled_mean;
            if (std::abs(delta) > 0.3) {  // Threshold for "large" change
                std::cout << "Step " << ts.step << ": pooled_mean changed by "
                          << std::showpos << delta << std::noshowpos << std::endl;
            }
        }
        prev_pooled_mean = ts.pooled_mean;
    }

    return 0;
}
