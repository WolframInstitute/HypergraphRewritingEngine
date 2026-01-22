#pragma once

#include "bh_types.hpp"
#include "hausdorff_analysis.hpp"
#include <string>
#include <fstream>
#include <iostream>

namespace viz::blackhole {

// Forward declaration
struct EvolutionResult;

// =============================================================================
// Binary Serialization Format
// =============================================================================
//
// File format (.bhdata):
// - Header: magic (4 bytes) + version (4 bytes) + flags (4 bytes)
// - BHInitialCondition section
// - EvolutionConfig section
// - Anchor vertices section
// - Per-timestep aggregations section
// - Global stats section
//
// File format (.bhevo):
// - Header: magic (4 bytes) + version (4 bytes)
// - BHInitialCondition section
// - EvolutionConfig section
// - States by step (graphs)
//
// All multi-byte values are little-endian.

constexpr uint32_t BH_MAGIC = 0x48424844;  // "BHRD" in little-endian
constexpr uint32_t BH_VERSION = 10;  // v10: added per-state aggregates for scatter plots

constexpr uint32_t EVO_MAGIC = 0x4F564542;  // "BEVO" in little-endian
constexpr uint32_t EVO_VERSION = 1;

// Section IDs
constexpr uint32_t SECTION_INITIAL = 0x01;
constexpr uint32_t SECTION_EVOLUTION_CONFIG = 0x02;
constexpr uint32_t SECTION_ANCHORS = 0x03;
constexpr uint32_t SECTION_TIMESTEPS = 0x04;
constexpr uint32_t SECTION_STATS = 0x05;
constexpr uint32_t SECTION_STATES = 0x06;
constexpr uint32_t SECTION_PER_STATE = 0x07;       // v7+: per-state vertices/edges
constexpr uint32_t SECTION_MEGA_DIM = 0x08;        // v7+: mega-union dimension (Global mode)
constexpr uint32_t SECTION_CURVATURE = 0x09;       // v8+: curvature analysis results
constexpr uint32_t SECTION_HILBERT = 0x0A;         // v8+: Hilbert space analysis
constexpr uint32_t SECTION_BRANCHIAL = 0x0B;       // v8+: branchial analysis
constexpr uint32_t SECTION_BRANCH_ALIGNMENT = 0x0C; // v8+: branch alignment (shape space)
constexpr uint32_t SECTION_CURV_QUANTILES = 0x0D;   // v9+: global curvature quantiles (for normalization)
constexpr uint32_t SECTION_STATE_AGGREGATES = 0x0E; // v10+: per-state aggregates for scatter plots
constexpr uint32_t SECTION_END = 0xFF;

// =============================================================================
// Evolution Data (for intermediate caching)
// =============================================================================

struct EvolutionData {
    BHInitialCondition initial;
    EvolutionConfig config;
    std::vector<std::vector<SimpleGraph>> states_by_step;
    int total_states = 0;
    int total_events = 0;
    int max_step_reached = 0;
};

// =============================================================================
// Serialization Functions
// =============================================================================

// Write analysis result to binary file (.bhdata)
bool write_analysis(const std::string& path, const BHAnalysisResult& result);

// Read analysis result from binary file (.bhdata)
bool read_analysis(const std::string& path, BHAnalysisResult& result);

// Write evolution data to binary file (.bhevo)
bool write_evolution(const std::string& path, const EvolutionData& data);

// Read evolution data from binary file (.bhevo)
bool read_evolution(const std::string& path, EvolutionData& data);

// =============================================================================
// Low-Level I/O Helpers
// =============================================================================

class BinaryWriter {
public:
    explicit BinaryWriter(const std::string& path);
    ~BinaryWriter();

    bool is_open() const { return file_.is_open(); }

    void write_u32(uint32_t v);
    void write_i32(int32_t v);
    void write_f32(float v);
    void write_string(const std::string& s);
    void write_bytes(const void* data, size_t size);

    template<typename T>
    void write_vector(const std::vector<T>& vec) {
        write_u32(static_cast<uint32_t>(vec.size()));
        if (!vec.empty()) {
            write_bytes(vec.data(), vec.size() * sizeof(T));
        }
    }

    void write_vec2(const Vec2& v);
    void write_vec3(const Vec3& v);
    void write_edge(const Edge& e);

private:
    std::ofstream file_;
};

class BinaryReader {
public:
    explicit BinaryReader(const std::string& path);
    ~BinaryReader();

    bool is_open() const { return file_.is_open(); }
    bool eof() const { return file_.eof(); }

    uint32_t read_u32();
    int32_t read_i32();
    float read_f32();
    std::string read_string();
    void read_bytes(void* data, size_t size);

    template<typename T>
    std::vector<T> read_vector() {
        uint32_t size = read_u32();
        std::vector<T> vec(size);
        if (size > 0) {
            read_bytes(vec.data(), size * sizeof(T));
        }
        return vec;
    }

    Vec2 read_vec2();
    Vec3 read_vec3();
    Edge read_edge();

private:
    std::ifstream file_;
};

} // namespace viz::blackhole
