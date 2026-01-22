#include <blackhole/bh_serialization.hpp>
#include <cstring>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace viz::blackhole {

// =============================================================================
// BinaryWriter Implementation
// =============================================================================

BinaryWriter::BinaryWriter(const std::string& path)
    : file_(path, std::ios::binary)
{
}

BinaryWriter::~BinaryWriter() {
    if (file_.is_open()) {
        file_.close();
    }
}

void BinaryWriter::write_u32(uint32_t v) {
    file_.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void BinaryWriter::write_i32(int32_t v) {
    file_.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void BinaryWriter::write_f32(float v) {
    file_.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void BinaryWriter::write_string(const std::string& s) {
    write_u32(static_cast<uint32_t>(s.size()));
    if (!s.empty()) {
        file_.write(s.data(), s.size());
    }
}

void BinaryWriter::write_bytes(const void* data, size_t size) {
    file_.write(reinterpret_cast<const char*>(data), size);
}

void BinaryWriter::write_vec2(const Vec2& v) {
    write_f32(v.x);
    write_f32(v.y);
}

void BinaryWriter::write_vec3(const Vec3& v) {
    write_f32(v.x);
    write_f32(v.y);
    write_f32(v.z);
}

void BinaryWriter::write_edge(const Edge& e) {
    write_u32(e.v1);
    write_u32(e.v2);
}

// =============================================================================
// BinaryReader Implementation
// =============================================================================

BinaryReader::BinaryReader(const std::string& path)
    : file_(path, std::ios::binary)
{
}

BinaryReader::~BinaryReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

uint32_t BinaryReader::read_u32() {
    uint32_t v;
    file_.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

int32_t BinaryReader::read_i32() {
    int32_t v;
    file_.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

float BinaryReader::read_f32() {
    float v;
    file_.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

std::string BinaryReader::read_string() {
    uint32_t size = read_u32();
    std::string s(size, '\0');
    if (size > 0) {
        file_.read(s.data(), size);
    }
    return s;
}

void BinaryReader::read_bytes(void* data, size_t size) {
    file_.read(reinterpret_cast<char*>(data), size);
}

Vec2 BinaryReader::read_vec2() {
    Vec2 v;
    v.x = read_f32();
    v.y = read_f32();
    return v;
}

Vec3 BinaryReader::read_vec3() {
    Vec3 v;
    v.x = read_f32();
    v.y = read_f32();
    v.z = read_f32();
    return v;
}

Edge BinaryReader::read_edge() {
    Edge e;
    e.v1 = read_u32();
    e.v2 = read_u32();
    return e;
}

// =============================================================================
// High-Level Serialization
// =============================================================================

bool write_analysis(const std::string& path, const BHAnalysisResult& result) {
    BinaryWriter writer(path);
    if (!writer.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return false;
    }

    // Header
    writer.write_u32(BH_MAGIC);
    writer.write_u32(BH_VERSION);
    writer.write_u32(0);  // flags (reserved)

    // Section: Initial Condition
    writer.write_u32(SECTION_INITIAL);
    writer.write_f32(result.initial.config.mass1);
    writer.write_f32(result.initial.config.mass2);
    writer.write_f32(result.initial.config.separation);
    writer.write_f32(result.initial.config.edge_threshold);
    writer.write_f32(result.initial.config.box_x[0]);
    writer.write_f32(result.initial.config.box_x[1]);
    writer.write_f32(result.initial.config.box_y[0]);
    writer.write_f32(result.initial.config.box_y[1]);

    // Vertex positions
    writer.write_u32(static_cast<uint32_t>(result.initial.vertex_positions.size()));
    for (const auto& v : result.initial.vertex_positions) {
        writer.write_vec2(v);
    }

    // Edges
    writer.write_u32(static_cast<uint32_t>(result.initial.edges.size()));
    for (const auto& e : result.initial.edges) {
        writer.write_edge(e);
    }

    // Section: Evolution Config
    writer.write_u32(SECTION_EVOLUTION_CONFIG);
    writer.write_string(result.evolution_config.rule);
    writer.write_i32(result.evolution_config.max_steps);
    writer.write_i32(result.evolution_config.max_states_per_step);
    writer.write_i32(result.evolution_config.max_successors_per_parent);
    writer.write_f32(result.evolution_config.exploration_probability);
    writer.write_u32(result.evolution_config.canonicalize_states ? 1 : 0);

    // Section: Anchors
    writer.write_u32(SECTION_ANCHORS);
    writer.write_u32(static_cast<uint32_t>(result.anchor_vertices.size()));
    for (auto v : result.anchor_vertices) {
        writer.write_u32(v);
    }
    writer.write_i32(result.analysis_max_radius);

    // Section: Timesteps
    writer.write_u32(SECTION_TIMESTEPS);
    writer.write_u32(static_cast<uint32_t>(result.per_timestep.size()));

    for (const auto& ts : result.per_timestep) {
        writer.write_u32(ts.step);
        writer.write_i32(ts.num_states);

        // Union vertices
        writer.write_u32(static_cast<uint32_t>(ts.union_vertices.size()));
        for (auto v : ts.union_vertices) {
            writer.write_u32(v);
        }

        // Union edges
        writer.write_u32(static_cast<uint32_t>(ts.union_edges.size()));
        for (const auto& e : ts.union_edges) {
            writer.write_edge(e);
        }

        // Intersection edges (v2+)
        writer.write_u32(static_cast<uint32_t>(ts.intersection_edges.size()));
        for (const auto& e : ts.intersection_edges) {
            writer.write_edge(e);
        }

        // Frequent edges (v5+) - edges in more than one state
        writer.write_u32(static_cast<uint32_t>(ts.frequent_edges.size()));
        for (const auto& e : ts.frequent_edges) {
            writer.write_edge(e);
        }

        // Vertex positions
        writer.write_u32(static_cast<uint32_t>(ts.vertex_positions.size()));
        for (const auto& p : ts.vertex_positions) {
            writer.write_vec2(p);
        }

        // Layout positions (v6+) - pre-computed force-directed layout
        writer.write_u32(static_cast<uint32_t>(ts.layout_positions.size()));
        for (const auto& p : ts.layout_positions) {
            writer.write_vec2(p);
        }

        // Mean dimensions
        writer.write_u32(static_cast<uint32_t>(ts.mean_dimensions.size()));
        for (float d : ts.mean_dimensions) {
            writer.write_f32(d);
        }

        // Variance dimensions (v3+)
        writer.write_u32(static_cast<uint32_t>(ts.variance_dimensions.size()));
        for (float v : ts.variance_dimensions) {
            writer.write_f32(v);
        }

        // Stats
        writer.write_f32(ts.pooled_mean);
        writer.write_f32(ts.pooled_min);
        writer.write_f32(ts.pooled_max);

        // Variance stats (v3+)
        writer.write_f32(ts.pooled_variance);
        writer.write_f32(ts.var_min);
        writer.write_f32(ts.var_max);

        // Global mean dimensions (v4+)
        writer.write_u32(static_cast<uint32_t>(ts.global_mean_dimensions.size()));
        for (float g : ts.global_mean_dimensions) {
            writer.write_f32(g);
        }

        // Global variance dimensions (v4+)
        writer.write_u32(static_cast<uint32_t>(ts.global_variance_dimensions.size()));
        for (float g : ts.global_variance_dimensions) {
            writer.write_f32(g);
        }

        // Global dimension stats (v4+)
        writer.write_f32(ts.global_mean_pooled);
        writer.write_f32(ts.global_mean_min);
        writer.write_f32(ts.global_mean_max);
        writer.write_f32(ts.global_var_pooled);
        writer.write_f32(ts.global_var_min);
        writer.write_f32(ts.global_var_max);

        // Per-timestep curvature (v9+)
        // Branchial curvature (mean/variance)
        auto write_float_vec = [&writer](const std::vector<float>& v) {
            writer.write_u32(static_cast<uint32_t>(v.size()));
            for (float f : v) writer.write_f32(f);
        };
        write_float_vec(ts.mean_curvature_ollivier);
        write_float_vec(ts.variance_curvature_ollivier);
        write_float_vec(ts.mean_curvature_wolfram_scalar);
        write_float_vec(ts.variance_curvature_wolfram_scalar);
        write_float_vec(ts.mean_curvature_wolfram_ricci);
        write_float_vec(ts.variance_curvature_wolfram_ricci);
        write_float_vec(ts.mean_curvature_dim_gradient);
        write_float_vec(ts.variance_curvature_dim_gradient);

        // Branchial curvature stats
        writer.write_f32(ts.ollivier_mean_min); writer.write_f32(ts.ollivier_mean_max);
        writer.write_f32(ts.ollivier_var_min); writer.write_f32(ts.ollivier_var_max);
        writer.write_f32(ts.wolfram_scalar_mean_min); writer.write_f32(ts.wolfram_scalar_mean_max);
        writer.write_f32(ts.wolfram_scalar_var_min); writer.write_f32(ts.wolfram_scalar_var_max);
        writer.write_f32(ts.wolfram_ricci_mean_min); writer.write_f32(ts.wolfram_ricci_mean_max);
        writer.write_f32(ts.wolfram_ricci_var_min); writer.write_f32(ts.wolfram_ricci_var_max);
        writer.write_f32(ts.dim_gradient_mean_min); writer.write_f32(ts.dim_gradient_mean_max);
        writer.write_f32(ts.dim_gradient_var_min); writer.write_f32(ts.dim_gradient_var_max);

        // Foliation curvature
        write_float_vec(ts.foliation_curvature_ollivier);
        write_float_vec(ts.foliation_curvature_wolfram_scalar);
        write_float_vec(ts.foliation_curvature_wolfram_ricci);
        write_float_vec(ts.foliation_curvature_dim_gradient);

        // Foliation curvature stats
        writer.write_f32(ts.foliation_ollivier_min); writer.write_f32(ts.foliation_ollivier_max);
        writer.write_f32(ts.foliation_wolfram_scalar_min); writer.write_f32(ts.foliation_wolfram_scalar_max);
        writer.write_f32(ts.foliation_wolfram_ricci_min); writer.write_f32(ts.foliation_wolfram_ricci_max);
        writer.write_f32(ts.foliation_dim_gradient_min); writer.write_f32(ts.foliation_dim_gradient_max);
    }

    // Section: Stats
    writer.write_u32(SECTION_STATS);
    writer.write_f32(result.dim_min);
    writer.write_f32(result.dim_max);
    writer.write_f32(result.dim_q05);
    writer.write_f32(result.dim_q95);
    writer.write_i32(result.total_steps);
    writer.write_i32(result.total_states);
    writer.write_i32(result.total_events);

    // Variance stats (v3+)
    writer.write_f32(result.var_min);
    writer.write_f32(result.var_max);
    writer.write_f32(result.var_q05);
    writer.write_f32(result.var_q95);

    // Global mean dimension stats (v4+)
    writer.write_f32(result.global_dim_min);
    writer.write_f32(result.global_dim_max);
    writer.write_f32(result.global_dim_q05);
    writer.write_f32(result.global_dim_q95);

    // Global variance dimension stats (v4+)
    writer.write_f32(result.global_var_min);
    writer.write_f32(result.global_var_max);
    writer.write_f32(result.global_var_q05);
    writer.write_f32(result.global_var_q95);

    // Layout bounding radius (v6+) - for camera framing
    writer.write_f32(result.layout_bounding_radius);

    // Section: Per-State Data (v7+) - for single-state viewing
    writer.write_u32(SECTION_PER_STATE);
    writer.write_u32(static_cast<uint32_t>(result.states_per_step.size()));
    for (const auto& step_states : result.states_per_step) {
        writer.write_u32(static_cast<uint32_t>(step_states.size()));
        for (const auto& state : step_states) {
            // Write vertices
            writer.write_u32(static_cast<uint32_t>(state.vertices.size()));
            for (VertexId v : state.vertices) {
                writer.write_u32(v);
            }
            // Write edges
            writer.write_u32(static_cast<uint32_t>(state.edges.size()));
            for (const auto& e : state.edges) {
                writer.write_edge(e);
            }
        }
    }

    // Section: Mega-union dimension - for Global mode
    writer.write_u32(SECTION_MEGA_DIM);
    writer.write_f32(result.mega_dim_min);
    writer.write_f32(result.mega_dim_max);
    writer.write_u32(static_cast<uint32_t>(result.mega_dimension.size()));
    for (const auto& [vid, dim] : result.mega_dimension) {
        writer.write_u32(vid);
        writer.write_f32(dim);
    }

    // Section: Curvature Analysis (v8+)
    if (result.has_curvature_analysis) {
        writer.write_u32(SECTION_CURVATURE);
        // Ollivier-Ricci map
        writer.write_u32(static_cast<uint32_t>(result.curvature_ollivier_ricci.size()));
        for (const auto& [vid, val] : result.curvature_ollivier_ricci) {
            writer.write_u32(vid);
            writer.write_f32(val);
        }
        writer.write_f32(result.curvature_ollivier_mean);
        writer.write_f32(result.curvature_ollivier_min);
        writer.write_f32(result.curvature_ollivier_max);
        // Dimension gradient map
        writer.write_u32(static_cast<uint32_t>(result.curvature_dimension_gradient.size()));
        for (const auto& [vid, val] : result.curvature_dimension_gradient) {
            writer.write_u32(vid);
            writer.write_f32(val);
        }
        writer.write_f32(result.curvature_dim_grad_mean);
        writer.write_f32(result.curvature_dim_grad_min);
        writer.write_f32(result.curvature_dim_grad_max);
        // Wolfram scalar curvature map (ball volume method)
        writer.write_u32(static_cast<uint32_t>(result.curvature_wolfram_scalar.size()));
        for (const auto& [vid, val] : result.curvature_wolfram_scalar) {
            writer.write_u32(vid);
            writer.write_f32(val);
        }
        writer.write_f32(result.curvature_wolfram_scalar_mean);
        writer.write_f32(result.curvature_wolfram_scalar_min);
        writer.write_f32(result.curvature_wolfram_scalar_max);
        // Wolfram Ricci curvature map (tube volume method, full tensor)
        writer.write_u32(static_cast<uint32_t>(result.curvature_wolfram_ricci.size()));
        for (const auto& [vid, val] : result.curvature_wolfram_ricci) {
            writer.write_u32(vid);
            writer.write_f32(val);
        }
        writer.write_f32(result.curvature_wolfram_ricci_mean);
        writer.write_f32(result.curvature_wolfram_ricci_min);
        writer.write_f32(result.curvature_wolfram_ricci_max);
    }

    // Section: Hilbert Space Analysis (v8+)
    if (result.has_hilbert_analysis) {
        writer.write_u32(SECTION_HILBERT);
        writer.write_u32(static_cast<uint32_t>(result.hilbert_num_states));
        writer.write_u32(static_cast<uint32_t>(result.hilbert_num_vertices));
        writer.write_f32(result.hilbert_mean_inner_product);
        writer.write_f32(result.hilbert_max_inner_product);
        writer.write_f32(result.hilbert_mean_vertex_probability);
        writer.write_f32(result.hilbert_vertex_probability_entropy);
        // Vertex probabilities map
        writer.write_u32(static_cast<uint32_t>(result.hilbert_vertex_probabilities.size()));
        for (const auto& [vid, val] : result.hilbert_vertex_probabilities) {
            writer.write_u32(vid);
            writer.write_f32(val);
        }
    }

    // Section: Branchial Analysis (v8+)
    if (result.has_branchial_analysis) {
        writer.write_u32(SECTION_BRANCHIAL);
        writer.write_f32(result.branchial_mean_sharpness);
        writer.write_f32(result.branchial_mean_entropy);
        // Vertex sharpness map
        writer.write_u32(static_cast<uint32_t>(result.branchial_vertex_sharpness.size()));
        for (const auto& [vid, val] : result.branchial_vertex_sharpness) {
            writer.write_u32(vid);
            writer.write_f32(val);
        }
        // Vertex entropy map
        writer.write_u32(static_cast<uint32_t>(result.branchial_vertex_entropy.size()));
        for (const auto& [vid, val] : result.branchial_vertex_entropy) {
            writer.write_u32(vid);
            writer.write_f32(val);
        }
    }

    // Section: Branch Alignment (v8+) - curvature shape space
    if (result.has_branch_alignment) {
        writer.write_u32(SECTION_BRANCH_ALIGNMENT);
        // Global bounds
        writer.write_f32(result.global_pc1_min);
        writer.write_f32(result.global_pc1_max);
        writer.write_f32(result.global_pc2_min);
        writer.write_f32(result.global_pc2_max);
        writer.write_f32(result.global_pc3_min);
        writer.write_f32(result.global_pc3_max);
        writer.write_f32(result.global_curvature_min);
        writer.write_f32(result.global_curvature_max);
        writer.write_f32(result.curvature_abs_max);

        // Per-timestep alignment (Wolfram-Ricci)
        writer.write_u32(static_cast<uint32_t>(result.alignment_per_timestep.size()));
        for (const auto& pta : result.alignment_per_timestep) {
            writer.write_u32(static_cast<uint32_t>(pta.total_points));
            writer.write_u32(static_cast<uint32_t>(pta.num_branches));
            // PC coordinates
            for (size_t i = 0; i < pta.total_points; ++i) {
                writer.write_f32(pta.all_pc1[i]);
                writer.write_f32(pta.all_pc2[i]);
                writer.write_f32(pta.all_pc3[i]);
                writer.write_f32(pta.all_curvature[i]);
                writer.write_f32(pta.all_rank[i]);
                writer.write_u32(static_cast<uint32_t>(pta.branch_id[i]));
                writer.write_u32(pta.all_vertices[i]);
                writer.write_u32(pta.state_id[i]);
            }
            // Branch sizes
            writer.write_u32(static_cast<uint32_t>(pta.branch_sizes.size()));
            for (size_t sz : pta.branch_sizes) {
                writer.write_u32(static_cast<uint32_t>(sz));
            }
        }

        // Ollivier-Ricci alignment (if available)
        writer.write_u32(result.has_ollivier_alignment ? 1 : 0);
        if (result.has_ollivier_alignment) {
            writer.write_f32(result.ollivier_pc1_min);
            writer.write_f32(result.ollivier_pc1_max);
            writer.write_f32(result.ollivier_pc2_min);
            writer.write_f32(result.ollivier_pc2_max);
            writer.write_f32(result.ollivier_pc3_min);
            writer.write_f32(result.ollivier_pc3_max);
            writer.write_f32(result.ollivier_curvature_min);
            writer.write_f32(result.ollivier_curvature_max);
            writer.write_f32(result.ollivier_curvature_abs_max);

            writer.write_u32(static_cast<uint32_t>(result.alignment_ollivier.size()));
            for (const auto& pta : result.alignment_ollivier) {
                writer.write_u32(static_cast<uint32_t>(pta.total_points));
                writer.write_u32(static_cast<uint32_t>(pta.num_branches));
                for (size_t i = 0; i < pta.total_points; ++i) {
                    writer.write_f32(pta.all_pc1[i]);
                    writer.write_f32(pta.all_pc2[i]);
                    writer.write_f32(pta.all_pc3[i]);
                    writer.write_f32(pta.all_curvature[i]);
                    writer.write_f32(pta.all_rank[i]);
                    writer.write_u32(static_cast<uint32_t>(pta.branch_id[i]));
                    writer.write_u32(pta.all_vertices[i]);
                    writer.write_u32(pta.state_id[i]);
                }
                writer.write_u32(static_cast<uint32_t>(pta.branch_sizes.size()));
                for (size_t sz : pta.branch_sizes) {
                    writer.write_u32(static_cast<uint32_t>(sz));
                }
            }
        }
    }

    // Section: Global Curvature Quantiles (for normalization toggle)
    // Only write if we have computed quantiles (check if any are non-zero)
    bool has_curv_quantiles =
        result.curv_ollivier_mean_q05 != 0 || result.curv_ollivier_mean_q95 != 0 ||
        result.curv_foliation_ollivier_q05 != 0 || result.curv_foliation_ollivier_q95 != 0;
    if (has_curv_quantiles) {
        writer.write_u32(SECTION_CURV_QUANTILES);
        // Branchial mean
        writer.write_f32(result.curv_ollivier_mean_q05);
        writer.write_f32(result.curv_ollivier_mean_q95);
        writer.write_f32(result.curv_wolfram_scalar_mean_q05);
        writer.write_f32(result.curv_wolfram_scalar_mean_q95);
        writer.write_f32(result.curv_wolfram_ricci_mean_q05);
        writer.write_f32(result.curv_wolfram_ricci_mean_q95);
        writer.write_f32(result.curv_dim_gradient_mean_q05);
        writer.write_f32(result.curv_dim_gradient_mean_q95);
        // Branchial variance
        writer.write_f32(result.curv_ollivier_var_q05);
        writer.write_f32(result.curv_ollivier_var_q95);
        writer.write_f32(result.curv_wolfram_scalar_var_q05);
        writer.write_f32(result.curv_wolfram_scalar_var_q95);
        writer.write_f32(result.curv_wolfram_ricci_var_q05);
        writer.write_f32(result.curv_wolfram_ricci_var_q95);
        writer.write_f32(result.curv_dim_gradient_var_q05);
        writer.write_f32(result.curv_dim_gradient_var_q95);
        // Foliation
        writer.write_f32(result.curv_foliation_ollivier_q05);
        writer.write_f32(result.curv_foliation_ollivier_q95);
        writer.write_f32(result.curv_foliation_wolfram_scalar_q05);
        writer.write_f32(result.curv_foliation_wolfram_scalar_q95);
        writer.write_f32(result.curv_foliation_wolfram_ricci_q05);
        writer.write_f32(result.curv_foliation_wolfram_ricci_q95);
        writer.write_f32(result.curv_foliation_dim_gradient_q05);
        writer.write_f32(result.curv_foliation_dim_gradient_q95);
    }

    // Section: Per-state aggregates for scatter plots
    if (!result.state_aggregates.empty()) {
        writer.write_u32(SECTION_STATE_AGGREGATES);
        writer.write_u32(static_cast<uint32_t>(result.state_aggregates.size()));
        for (const auto& agg : result.state_aggregates) {
            writer.write_u32(agg.state_id);
            writer.write_u32(agg.step);
            writer.write_f32(agg.mean_dimension);
            writer.write_f32(agg.mean_ollivier_ricci);
            writer.write_f32(agg.mean_wolfram_scalar);
            writer.write_f32(agg.mean_wolfram_ricci);
            writer.write_f32(agg.mean_dim_gradient);
        }
    }

    // End marker
    writer.write_u32(SECTION_END);

    return true;
}

bool read_analysis(const std::string& path, BHAnalysisResult& result) {
    BinaryReader reader(path);
    if (!reader.is_open()) {
        std::cerr << "Failed to open file for reading: " << path << std::endl;
        return false;
    }

    // Header
    uint32_t magic = reader.read_u32();
    if (magic != BH_MAGIC) {
        std::cerr << "Invalid file magic: expected " << BH_MAGIC << ", got " << magic << std::endl;
        return false;
    }

    uint32_t version = reader.read_u32();
    std::cout << "  File version: " << version << " (current: " << BH_VERSION << ")" << std::endl;
    if (version < 1 || version > BH_VERSION) {
        std::cerr << "Unsupported version: " << version << " (expected 1-" << BH_VERSION << ")" << std::endl;
        return false;
    }

    reader.read_u32();  // flags (ignored)

    // Read sections
    while (!reader.eof()) {
        uint32_t section = reader.read_u32();

        if (section == SECTION_END) {
            break;
        }

        switch (section) {
            case SECTION_INITIAL: {
                result.initial.config.mass1 = reader.read_f32();
                result.initial.config.mass2 = reader.read_f32();
                result.initial.config.separation = reader.read_f32();
                result.initial.config.edge_threshold = reader.read_f32();
                result.initial.config.box_x[0] = reader.read_f32();
                result.initial.config.box_x[1] = reader.read_f32();
                result.initial.config.box_y[0] = reader.read_f32();
                result.initial.config.box_y[1] = reader.read_f32();

                uint32_t n_verts = reader.read_u32();
                result.initial.vertex_positions.resize(n_verts);
                for (uint32_t i = 0; i < n_verts; ++i) {
                    result.initial.vertex_positions[i] = reader.read_vec2();
                }

                uint32_t n_edges = reader.read_u32();
                result.initial.edges.resize(n_edges);
                for (uint32_t i = 0; i < n_edges; ++i) {
                    result.initial.edges[i] = reader.read_edge();
                }
                break;
            }

            case SECTION_EVOLUTION_CONFIG: {
                result.evolution_config.rule = reader.read_string();
                result.evolution_config.max_steps = reader.read_i32();
                result.evolution_config.max_states_per_step = reader.read_i32();
                result.evolution_config.max_successors_per_parent = reader.read_i32();
                result.evolution_config.exploration_probability = reader.read_f32();
                result.evolution_config.canonicalize_states = reader.read_u32() != 0;
                break;
            }

            case SECTION_ANCHORS: {
                uint32_t n_anchors = reader.read_u32();
                result.anchor_vertices.resize(n_anchors);
                for (uint32_t i = 0; i < n_anchors; ++i) {
                    result.anchor_vertices[i] = reader.read_u32();
                }
                result.analysis_max_radius = reader.read_i32();
                break;
            }

            case SECTION_TIMESTEPS: {
                uint32_t n_timesteps = reader.read_u32();
                result.per_timestep.resize(n_timesteps);

                for (uint32_t t = 0; t < n_timesteps; ++t) {
                    auto& ts = result.per_timestep[t];

                    ts.step = reader.read_u32();
                    ts.num_states = reader.read_i32();

                    uint32_t n_union_verts = reader.read_u32();
                    ts.union_vertices.resize(n_union_verts);
                    for (uint32_t i = 0; i < n_union_verts; ++i) {
                        ts.union_vertices[i] = reader.read_u32();
                    }

                    uint32_t n_union_edges = reader.read_u32();
                    ts.union_edges.resize(n_union_edges);
                    for (uint32_t i = 0; i < n_union_edges; ++i) {
                        ts.union_edges[i] = reader.read_edge();
                    }

                    // Intersection edges (v2+)
                    if (version >= 2) {
                        uint32_t n_intersection_edges = reader.read_u32();
                        ts.intersection_edges.resize(n_intersection_edges);
                        for (uint32_t i = 0; i < n_intersection_edges; ++i) {
                            ts.intersection_edges[i] = reader.read_edge();
                        }
                    }

                    // Frequent edges (v5+)
                    if (version >= 5) {
                        uint32_t n_frequent_edges = reader.read_u32();
                        ts.frequent_edges.resize(n_frequent_edges);
                        for (uint32_t i = 0; i < n_frequent_edges; ++i) {
                            ts.frequent_edges[i] = reader.read_edge();
                        }
                    }

                    uint32_t n_positions = reader.read_u32();
                    ts.vertex_positions.resize(n_positions);
                    for (uint32_t i = 0; i < n_positions; ++i) {
                        ts.vertex_positions[i] = reader.read_vec2();
                    }

                    // Layout positions (v6+)
                    if (version >= 6) {
                        uint32_t n_layout_positions = reader.read_u32();
                        ts.layout_positions.resize(n_layout_positions);
                        for (uint32_t i = 0; i < n_layout_positions; ++i) {
                            ts.layout_positions[i] = reader.read_vec2();
                        }
                    }

                    uint32_t n_dims = reader.read_u32();
                    ts.mean_dimensions.resize(n_dims);
                    for (uint32_t i = 0; i < n_dims; ++i) {
                        ts.mean_dimensions[i] = reader.read_f32();
                    }

                    // Variance dimensions (v3+)
                    if (version >= 3) {
                        uint32_t n_var_dims = reader.read_u32();
                        ts.variance_dimensions.resize(n_var_dims);
                        for (uint32_t i = 0; i < n_var_dims; ++i) {
                            ts.variance_dimensions[i] = reader.read_f32();
                        }
                    }

                    ts.pooled_mean = reader.read_f32();
                    ts.pooled_min = reader.read_f32();
                    ts.pooled_max = reader.read_f32();

                    // Variance stats (v3+)
                    if (version >= 3) {
                        ts.pooled_variance = reader.read_f32();
                        ts.var_min = reader.read_f32();
                        ts.var_max = reader.read_f32();
                    }

                    // Global mean dimensions (v4+)
                    if (version >= 4) {
                        uint32_t n_global_means = reader.read_u32();
                        ts.global_mean_dimensions.resize(n_global_means);
                        for (uint32_t i = 0; i < n_global_means; ++i) {
                            ts.global_mean_dimensions[i] = reader.read_f32();
                        }

                        // Global variance dimensions (v4+)
                        uint32_t n_global_vars = reader.read_u32();
                        ts.global_variance_dimensions.resize(n_global_vars);
                        for (uint32_t i = 0; i < n_global_vars; ++i) {
                            ts.global_variance_dimensions[i] = reader.read_f32();
                        }

                        // Global dimension stats
                        ts.global_mean_pooled = reader.read_f32();
                        ts.global_mean_min = reader.read_f32();
                        ts.global_mean_max = reader.read_f32();
                        ts.global_var_pooled = reader.read_f32();
                        ts.global_var_min = reader.read_f32();
                        ts.global_var_max = reader.read_f32();
                    }

                    // Per-timestep curvature (v9+)
                    if (version >= 9) {
                        auto read_float_vec = [&reader](std::vector<float>& v) {
                            uint32_t n = reader.read_u32();
                            v.resize(n);
                            for (uint32_t i = 0; i < n; ++i) v[i] = reader.read_f32();
                        };

                        // Branchial curvature (mean/variance)
                        read_float_vec(ts.mean_curvature_ollivier);
                        read_float_vec(ts.variance_curvature_ollivier);
                        read_float_vec(ts.mean_curvature_wolfram_scalar);
                        read_float_vec(ts.variance_curvature_wolfram_scalar);
                        read_float_vec(ts.mean_curvature_wolfram_ricci);
                        read_float_vec(ts.variance_curvature_wolfram_ricci);
                        read_float_vec(ts.mean_curvature_dim_gradient);
                        read_float_vec(ts.variance_curvature_dim_gradient);

                        // Branchial curvature stats
                        ts.ollivier_mean_min = reader.read_f32(); ts.ollivier_mean_max = reader.read_f32();
                        ts.ollivier_var_min = reader.read_f32(); ts.ollivier_var_max = reader.read_f32();
                        ts.wolfram_scalar_mean_min = reader.read_f32(); ts.wolfram_scalar_mean_max = reader.read_f32();
                        ts.wolfram_scalar_var_min = reader.read_f32(); ts.wolfram_scalar_var_max = reader.read_f32();
                        ts.wolfram_ricci_mean_min = reader.read_f32(); ts.wolfram_ricci_mean_max = reader.read_f32();
                        ts.wolfram_ricci_var_min = reader.read_f32(); ts.wolfram_ricci_var_max = reader.read_f32();
                        ts.dim_gradient_mean_min = reader.read_f32(); ts.dim_gradient_mean_max = reader.read_f32();
                        ts.dim_gradient_var_min = reader.read_f32(); ts.dim_gradient_var_max = reader.read_f32();

                        // Foliation curvature
                        read_float_vec(ts.foliation_curvature_ollivier);
                        read_float_vec(ts.foliation_curvature_wolfram_scalar);
                        read_float_vec(ts.foliation_curvature_wolfram_ricci);
                        read_float_vec(ts.foliation_curvature_dim_gradient);

                        // Foliation curvature stats
                        ts.foliation_ollivier_min = reader.read_f32(); ts.foliation_ollivier_max = reader.read_f32();
                        ts.foliation_wolfram_scalar_min = reader.read_f32(); ts.foliation_wolfram_scalar_max = reader.read_f32();
                        ts.foliation_wolfram_ricci_min = reader.read_f32(); ts.foliation_wolfram_ricci_max = reader.read_f32();
                        ts.foliation_dim_gradient_min = reader.read_f32(); ts.foliation_dim_gradient_max = reader.read_f32();
                    }
                }
                break;
            }

            case SECTION_STATS: {
                result.dim_min = reader.read_f32();
                result.dim_max = reader.read_f32();
                result.dim_q05 = reader.read_f32();
                result.dim_q95 = reader.read_f32();
                result.total_steps = reader.read_i32();
                result.total_states = reader.read_i32();
                result.total_events = reader.read_i32();

                // Variance stats (v3+)
                if (version >= 3) {
                    result.var_min = reader.read_f32();
                    result.var_max = reader.read_f32();
                    result.var_q05 = reader.read_f32();
                    result.var_q95 = reader.read_f32();
                }

                // Global mean dimension stats (v4+)
                if (version >= 4) {
                    result.global_dim_min = reader.read_f32();
                    result.global_dim_max = reader.read_f32();
                    result.global_dim_q05 = reader.read_f32();
                    result.global_dim_q95 = reader.read_f32();

                    // Global variance dimension stats (v4+)
                    result.global_var_min = reader.read_f32();
                    result.global_var_max = reader.read_f32();
                    result.global_var_q05 = reader.read_f32();
                    result.global_var_q95 = reader.read_f32();
                }

                // Layout bounding radius (v6+)
                if (version >= 6) {
                    result.layout_bounding_radius = reader.read_f32();
                }
                break;
            }

            case SECTION_PER_STATE: {
                // Per-state data (v7+) - for single-state viewing
                uint32_t n_steps = reader.read_u32();
                std::cout << "  Loading per-state data: " << n_steps << " timesteps" << std::endl;
                result.states_per_step.resize(n_steps);
                for (uint32_t s = 0; s < n_steps; ++s) {
                    uint32_t n_states = reader.read_u32();
                    result.states_per_step[s].resize(n_states);
                    for (uint32_t st = 0; st < n_states; ++st) {
                        auto& state = result.states_per_step[s][st];
                        // Read vertices
                        uint32_t n_verts = reader.read_u32();
                        state.vertices.resize(n_verts);
                        for (uint32_t i = 0; i < n_verts; ++i) {
                            state.vertices[i] = reader.read_u32();
                        }
                        // Read edges
                        uint32_t n_edges = reader.read_u32();
                        state.edges.resize(n_edges);
                        for (uint32_t i = 0; i < n_edges; ++i) {
                            state.edges[i] = reader.read_edge();
                        }
                    }
                }
                std::cout << "  Per-state data loaded: " << result.states_per_step.size() << " timesteps" << std::endl;
                break;
            }

            case SECTION_MEGA_DIM: {
                // Mega-union dimension - for Global mode
                result.mega_dim_min = reader.read_f32();
                result.mega_dim_max = reader.read_f32();
                uint32_t n_dims = reader.read_u32();
                std::cout << "  Loading mega-union dimension: " << n_dims << " vertices" << std::endl;
                for (uint32_t i = 0; i < n_dims; ++i) {
                    VertexId vid = reader.read_u32();
                    float dim = reader.read_f32();
                    result.mega_dimension[vid] = dim;
                }
                std::cout << "  Mega-union dimension range: [" << result.mega_dim_min
                          << ", " << result.mega_dim_max << "]" << std::endl;
                break;
            }

            case SECTION_CURVATURE: {
                // Curvature Analysis (v8+)
                uint32_t n_ollivier = reader.read_u32();
                for (uint32_t i = 0; i < n_ollivier; ++i) {
                    VertexId vid = reader.read_u32();
                    float val = reader.read_f32();
                    result.curvature_ollivier_ricci[vid] = val;
                }
                result.curvature_ollivier_mean = reader.read_f32();
                result.curvature_ollivier_min = reader.read_f32();
                result.curvature_ollivier_max = reader.read_f32();
                uint32_t n_dimgrad = reader.read_u32();
                for (uint32_t i = 0; i < n_dimgrad; ++i) {
                    VertexId vid = reader.read_u32();
                    float val = reader.read_f32();
                    result.curvature_dimension_gradient[vid] = val;
                }
                result.curvature_dim_grad_mean = reader.read_f32();
                result.curvature_dim_grad_min = reader.read_f32();
                result.curvature_dim_grad_max = reader.read_f32();
                // Wolfram scalar curvature (v9+ extension)
                uint32_t n_wolfram_scalar = 0;
                uint32_t n_wolfram_ricci = 0;
                try {
                    n_wolfram_scalar = reader.read_u32();
                    for (uint32_t i = 0; i < n_wolfram_scalar; ++i) {
                        VertexId vid = reader.read_u32();
                        float val = reader.read_f32();
                        result.curvature_wolfram_scalar[vid] = val;
                    }
                    result.curvature_wolfram_scalar_mean = reader.read_f32();
                    result.curvature_wolfram_scalar_min = reader.read_f32();
                    result.curvature_wolfram_scalar_max = reader.read_f32();
                    // Wolfram Ricci curvature (tube method)
                    n_wolfram_ricci = reader.read_u32();
                    for (uint32_t i = 0; i < n_wolfram_ricci; ++i) {
                        VertexId vid = reader.read_u32();
                        float val = reader.read_f32();
                        result.curvature_wolfram_ricci[vid] = val;
                    }
                    result.curvature_wolfram_ricci_mean = reader.read_f32();
                    result.curvature_wolfram_ricci_min = reader.read_f32();
                    result.curvature_wolfram_ricci_max = reader.read_f32();
                } catch (...) {
                    // Old file format without Wolfram curvature - ignore
                }
                result.has_curvature_analysis = true;
                std::cout << "  Curvature analysis loaded: " << n_ollivier << " Ollivier-Ricci, "
                          << n_dimgrad << " dimension gradient, "
                          << n_wolfram_scalar << " Wolfram scalar, "
                          << n_wolfram_ricci << " Wolfram Ricci values" << std::endl;
                break;
            }

            case SECTION_HILBERT: {
                // Hilbert Space Analysis (v8+)
                result.hilbert_num_states = reader.read_u32();
                result.hilbert_num_vertices = reader.read_u32();
                result.hilbert_mean_inner_product = reader.read_f32();
                result.hilbert_max_inner_product = reader.read_f32();
                result.hilbert_mean_vertex_probability = reader.read_f32();
                result.hilbert_vertex_probability_entropy = reader.read_f32();
                uint32_t n_probs = reader.read_u32();
                for (uint32_t i = 0; i < n_probs; ++i) {
                    VertexId vid = reader.read_u32();
                    float val = reader.read_f32();
                    result.hilbert_vertex_probabilities[vid] = val;
                }
                result.has_hilbert_analysis = true;
                std::cout << "  Hilbert space analysis loaded: " << result.hilbert_num_states
                          << " states, " << result.hilbert_num_vertices << " vertices" << std::endl;
                break;
            }

            case SECTION_BRANCHIAL: {
                // Branchial Analysis (v8+)
                result.branchial_mean_sharpness = reader.read_f32();
                result.branchial_mean_entropy = reader.read_f32();
                uint32_t n_sharpness = reader.read_u32();
                for (uint32_t i = 0; i < n_sharpness; ++i) {
                    VertexId vid = reader.read_u32();
                    float val = reader.read_f32();
                    result.branchial_vertex_sharpness[vid] = val;
                }
                uint32_t n_entropy = reader.read_u32();
                for (uint32_t i = 0; i < n_entropy; ++i) {
                    VertexId vid = reader.read_u32();
                    float val = reader.read_f32();
                    result.branchial_vertex_entropy[vid] = val;
                }
                result.has_branchial_analysis = true;
                std::cout << "  Branchial analysis loaded: " << n_sharpness << " sharpness, "
                          << n_entropy << " entropy values" << std::endl;
                break;
            }

            case SECTION_BRANCH_ALIGNMENT: {
                // Branch Alignment (v8+) - curvature shape space
                result.global_pc1_min = reader.read_f32();
                result.global_pc1_max = reader.read_f32();
                result.global_pc2_min = reader.read_f32();
                result.global_pc2_max = reader.read_f32();
                result.global_pc3_min = reader.read_f32();
                result.global_pc3_max = reader.read_f32();
                result.global_curvature_min = reader.read_f32();
                result.global_curvature_max = reader.read_f32();
                result.curvature_abs_max = reader.read_f32();

                uint32_t n_timesteps = reader.read_u32();
                result.alignment_per_timestep.resize(n_timesteps);
                for (uint32_t t = 0; t < n_timesteps; ++t) {
                    auto& pta = result.alignment_per_timestep[t];
                    pta.total_points = reader.read_u32();
                    pta.num_branches = reader.read_u32();
                    pta.all_pc1.resize(pta.total_points);
                    pta.all_pc2.resize(pta.total_points);
                    pta.all_pc3.resize(pta.total_points);
                    pta.all_curvature.resize(pta.total_points);
                    pta.all_rank.resize(pta.total_points);
                    pta.branch_id.resize(pta.total_points);
                    pta.all_vertices.resize(pta.total_points);
                    pta.state_id.resize(pta.total_points);
                    for (size_t i = 0; i < pta.total_points; ++i) {
                        pta.all_pc1[i] = reader.read_f32();
                        pta.all_pc2[i] = reader.read_f32();
                        pta.all_pc3[i] = reader.read_f32();
                        pta.all_curvature[i] = reader.read_f32();
                        pta.all_rank[i] = reader.read_f32();
                        pta.branch_id[i] = reader.read_u32();
                        pta.all_vertices[i] = reader.read_u32();
                        pta.state_id[i] = reader.read_u32();
                    }
                    uint32_t n_branch_sizes = reader.read_u32();
                    pta.branch_sizes.resize(n_branch_sizes);
                    for (uint32_t i = 0; i < n_branch_sizes; ++i) {
                        pta.branch_sizes[i] = reader.read_u32();
                    }
                }

                result.has_ollivier_alignment = (reader.read_u32() != 0);
                if (result.has_ollivier_alignment) {
                    result.ollivier_pc1_min = reader.read_f32();
                    result.ollivier_pc1_max = reader.read_f32();
                    result.ollivier_pc2_min = reader.read_f32();
                    result.ollivier_pc2_max = reader.read_f32();
                    result.ollivier_pc3_min = reader.read_f32();
                    result.ollivier_pc3_max = reader.read_f32();
                    result.ollivier_curvature_min = reader.read_f32();
                    result.ollivier_curvature_max = reader.read_f32();
                    result.ollivier_curvature_abs_max = reader.read_f32();

                    uint32_t n_ollivier_timesteps = reader.read_u32();
                    result.alignment_ollivier.resize(n_ollivier_timesteps);
                    for (uint32_t t = 0; t < n_ollivier_timesteps; ++t) {
                        auto& pta = result.alignment_ollivier[t];
                        pta.total_points = reader.read_u32();
                        pta.num_branches = reader.read_u32();
                        pta.all_pc1.resize(pta.total_points);
                        pta.all_pc2.resize(pta.total_points);
                        pta.all_pc3.resize(pta.total_points);
                        pta.all_curvature.resize(pta.total_points);
                        pta.all_rank.resize(pta.total_points);
                        pta.branch_id.resize(pta.total_points);
                        pta.all_vertices.resize(pta.total_points);
                        pta.state_id.resize(pta.total_points);
                        for (size_t i = 0; i < pta.total_points; ++i) {
                            pta.all_pc1[i] = reader.read_f32();
                            pta.all_pc2[i] = reader.read_f32();
                            pta.all_pc3[i] = reader.read_f32();
                            pta.all_curvature[i] = reader.read_f32();
                            pta.all_rank[i] = reader.read_f32();
                            pta.branch_id[i] = reader.read_u32();
                            pta.all_vertices[i] = reader.read_u32();
                            pta.state_id[i] = reader.read_u32();
                        }
                        uint32_t n_branch_sizes = reader.read_u32();
                        pta.branch_sizes.resize(n_branch_sizes);
                        for (uint32_t i = 0; i < n_branch_sizes; ++i) {
                            pta.branch_sizes[i] = reader.read_u32();
                        }
                    }
                }

                result.has_branch_alignment = true;
                std::cout << "  Branch alignment loaded: " << n_timesteps << " timesteps"
                          << (result.has_ollivier_alignment ? " (with Ollivier)" : "") << std::endl;
                break;
            }

            case SECTION_CURV_QUANTILES: {
                // Branchial mean
                result.curv_ollivier_mean_q05 = reader.read_f32();
                result.curv_ollivier_mean_q95 = reader.read_f32();
                result.curv_wolfram_scalar_mean_q05 = reader.read_f32();
                result.curv_wolfram_scalar_mean_q95 = reader.read_f32();
                result.curv_wolfram_ricci_mean_q05 = reader.read_f32();
                result.curv_wolfram_ricci_mean_q95 = reader.read_f32();
                result.curv_dim_gradient_mean_q05 = reader.read_f32();
                result.curv_dim_gradient_mean_q95 = reader.read_f32();
                // Branchial variance
                result.curv_ollivier_var_q05 = reader.read_f32();
                result.curv_ollivier_var_q95 = reader.read_f32();
                result.curv_wolfram_scalar_var_q05 = reader.read_f32();
                result.curv_wolfram_scalar_var_q95 = reader.read_f32();
                result.curv_wolfram_ricci_var_q05 = reader.read_f32();
                result.curv_wolfram_ricci_var_q95 = reader.read_f32();
                result.curv_dim_gradient_var_q05 = reader.read_f32();
                result.curv_dim_gradient_var_q95 = reader.read_f32();
                // Foliation
                result.curv_foliation_ollivier_q05 = reader.read_f32();
                result.curv_foliation_ollivier_q95 = reader.read_f32();
                result.curv_foliation_wolfram_scalar_q05 = reader.read_f32();
                result.curv_foliation_wolfram_scalar_q95 = reader.read_f32();
                result.curv_foliation_wolfram_ricci_q05 = reader.read_f32();
                result.curv_foliation_wolfram_ricci_q95 = reader.read_f32();
                result.curv_foliation_dim_gradient_q05 = reader.read_f32();
                result.curv_foliation_dim_gradient_q95 = reader.read_f32();
                std::cout << "  Curvature quantiles loaded" << std::endl;
                break;
            }

            case SECTION_STATE_AGGREGATES: {
                uint32_t n_states = reader.read_u32();
                result.state_aggregates.resize(n_states);
                for (uint32_t i = 0; i < n_states; ++i) {
                    auto& agg = result.state_aggregates[i];
                    agg.state_id = reader.read_u32();
                    agg.step = reader.read_u32();
                    agg.mean_dimension = reader.read_f32();
                    agg.mean_ollivier_ricci = reader.read_f32();
                    agg.mean_wolfram_scalar = reader.read_f32();
                    agg.mean_wolfram_ricci = reader.read_f32();
                    agg.mean_dim_gradient = reader.read_f32();
                }
                std::cout << "  State aggregates loaded: " << n_states << " states" << std::endl;
                break;
            }

            default:
                std::cerr << "Unknown section: " << section << std::endl;
                return false;
        }
    }

    // Post-processing: Compute prefix sums and global vertex/edge union from loaded data
    // This is derived data that doesn't need to be serialized
    {
        std::unordered_set<VertexId> all_vertex_set;
        std::set<std::pair<VertexId, VertexId>> all_edge_set;

        for (size_t t = 0; t < result.per_timestep.size(); ++t) {
            auto& ts = result.per_timestep[t];

            // Build global vertex/edge sets
            for (VertexId v : ts.union_vertices) {
                all_vertex_set.insert(v);
            }
            for (const Edge& e : ts.union_edges) {
                VertexId v1 = std::min(e.v1, e.v2);
                VertexId v2 = std::max(e.v1, e.v2);
                all_edge_set.insert({v1, v2});
            }

            // Copy prefix sums from previous timestep
            if (t > 0) {
                const auto& prev = result.per_timestep[t - 1];
                ts.dim_prefix_sum = prev.dim_prefix_sum;
                ts.dim_prefix_count = prev.dim_prefix_count;
                ts.var_prefix_sum = prev.var_prefix_sum;
                ts.var_prefix_count = prev.var_prefix_count;
                ts.global_dim_prefix_sum = prev.global_dim_prefix_sum;
                ts.global_dim_prefix_count = prev.global_dim_prefix_count;
                ts.global_var_prefix_sum = prev.global_var_prefix_sum;
                ts.global_var_prefix_count = prev.global_var_prefix_count;
            }

            // Add current timestep values to prefix sums
            for (size_t i = 0; i < ts.union_vertices.size(); ++i) {
                VertexId v = ts.union_vertices[i];

                // Local mean dimension
                if (i < ts.mean_dimensions.size()) {
                    float dim = ts.mean_dimensions[i];
                    if (dim >= 0 && std::isfinite(dim)) {
                        ts.dim_prefix_sum[v] += dim;
                        ts.dim_prefix_count[v] += 1;
                    }
                }

                // Local variance
                if (i < ts.variance_dimensions.size()) {
                    float var = ts.variance_dimensions[i];
                    if (var >= 0 && std::isfinite(var)) {
                        ts.var_prefix_sum[v] += var;
                        ts.var_prefix_count[v] += 1;
                    }
                }

                // Global mean dimension
                if (i < ts.global_mean_dimensions.size()) {
                    float gdim = ts.global_mean_dimensions[i];
                    if (gdim >= 0 && std::isfinite(gdim)) {
                        ts.global_dim_prefix_sum[v] += gdim;
                        ts.global_dim_prefix_count[v] += 1;
                    }
                }

                // Global variance
                if (i < ts.global_variance_dimensions.size()) {
                    float gvar = ts.global_variance_dimensions[i];
                    if (gvar >= 0 && std::isfinite(gvar)) {
                        ts.global_var_prefix_sum[v] += gvar;
                        ts.global_var_prefix_count[v] += 1;
                    }
                }
            }
        }

        // Store global vertex/edge union
        result.all_vertices.assign(all_vertex_set.begin(), all_vertex_set.end());
        std::sort(result.all_vertices.begin(), result.all_vertices.end());

        for (const auto& ep : all_edge_set) {
            result.all_edges.push_back({ep.first, ep.second});
        }
    }

    return true;
}

// =============================================================================
// Evolution Data Serialization
// =============================================================================

// Helper to write a SimpleGraph
static void write_simple_graph(BinaryWriter& writer, const SimpleGraph& graph) {
    // Write vertices
    const auto& verts = graph.vertices();
    writer.write_u32(static_cast<uint32_t>(verts.size()));
    for (auto v : verts) {
        writer.write_u32(v);
    }

    // Reconstruct edges from adjacency (each edge appears once)
    std::vector<Edge> edges;
    for (VertexId v : verts) {
        for (VertexId u : graph.neighbors(v)) {
            if (v < u) {  // Only store each edge once
                edges.push_back({v, u});
            }
        }
    }

    writer.write_u32(static_cast<uint32_t>(edges.size()));
    for (const auto& e : edges) {
        writer.write_edge(e);
    }
}

// Helper to read a SimpleGraph
static SimpleGraph read_simple_graph(BinaryReader& reader) {
    uint32_t n_verts = reader.read_u32();
    std::vector<VertexId> verts(n_verts);
    for (uint32_t i = 0; i < n_verts; ++i) {
        verts[i] = reader.read_u32();
    }

    uint32_t n_edges = reader.read_u32();
    std::vector<Edge> edges(n_edges);
    for (uint32_t i = 0; i < n_edges; ++i) {
        edges[i] = reader.read_edge();
    }

    SimpleGraph graph;
    graph.build(verts, edges);
    return graph;
}

bool write_evolution(const std::string& path, const EvolutionData& data) {
    BinaryWriter writer(path);
    if (!writer.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return false;
    }

    // Header
    writer.write_u32(EVO_MAGIC);
    writer.write_u32(EVO_VERSION);

    // Section: Initial Condition
    writer.write_u32(SECTION_INITIAL);
    writer.write_f32(data.initial.config.mass1);
    writer.write_f32(data.initial.config.mass2);
    writer.write_f32(data.initial.config.separation);
    writer.write_f32(data.initial.config.edge_threshold);
    writer.write_f32(data.initial.config.box_x[0]);
    writer.write_f32(data.initial.config.box_x[1]);
    writer.write_f32(data.initial.config.box_y[0]);
    writer.write_f32(data.initial.config.box_y[1]);

    writer.write_u32(static_cast<uint32_t>(data.initial.vertex_positions.size()));
    for (const auto& v : data.initial.vertex_positions) {
        writer.write_vec2(v);
    }

    writer.write_u32(static_cast<uint32_t>(data.initial.edges.size()));
    for (const auto& e : data.initial.edges) {
        writer.write_edge(e);
    }

    // Section: Evolution Config
    writer.write_u32(SECTION_EVOLUTION_CONFIG);
    writer.write_string(data.config.rule);
    writer.write_i32(data.config.max_steps);
    writer.write_i32(data.config.max_states_per_step);
    writer.write_i32(data.config.max_successors_per_parent);
    writer.write_f32(data.config.exploration_probability);
    writer.write_u32(data.config.canonicalize_states ? 1 : 0);

    // Section: States
    writer.write_u32(SECTION_STATES);
    writer.write_i32(data.total_states);
    writer.write_i32(data.total_events);
    writer.write_i32(data.max_step_reached);

    writer.write_u32(static_cast<uint32_t>(data.states_by_step.size()));
    for (const auto& step_graphs : data.states_by_step) {
        writer.write_u32(static_cast<uint32_t>(step_graphs.size()));
        for (const auto& graph : step_graphs) {
            write_simple_graph(writer, graph);
        }
    }

    // End marker
    writer.write_u32(SECTION_END);

    return true;
}

bool read_evolution(const std::string& path, EvolutionData& data) {
    BinaryReader reader(path);
    if (!reader.is_open()) {
        std::cerr << "Failed to open file for reading: " << path << std::endl;
        return false;
    }

    // Header
    uint32_t magic = reader.read_u32();
    if (magic != EVO_MAGIC) {
        std::cerr << "Invalid evolution file magic: expected " << EVO_MAGIC << ", got " << magic << std::endl;
        return false;
    }

    uint32_t version = reader.read_u32();
    if (version != EVO_VERSION) {
        std::cerr << "Unsupported evolution file version: " << version << std::endl;
        return false;
    }

    // Read sections
    while (!reader.eof()) {
        uint32_t section = reader.read_u32();

        if (section == SECTION_END) {
            break;
        }

        switch (section) {
            case SECTION_INITIAL: {
                data.initial.config.mass1 = reader.read_f32();
                data.initial.config.mass2 = reader.read_f32();
                data.initial.config.separation = reader.read_f32();
                data.initial.config.edge_threshold = reader.read_f32();
                data.initial.config.box_x[0] = reader.read_f32();
                data.initial.config.box_x[1] = reader.read_f32();
                data.initial.config.box_y[0] = reader.read_f32();
                data.initial.config.box_y[1] = reader.read_f32();

                uint32_t n_verts = reader.read_u32();
                data.initial.vertex_positions.resize(n_verts);
                for (uint32_t i = 0; i < n_verts; ++i) {
                    data.initial.vertex_positions[i] = reader.read_vec2();
                }

                uint32_t n_edges = reader.read_u32();
                data.initial.edges.resize(n_edges);
                for (uint32_t i = 0; i < n_edges; ++i) {
                    data.initial.edges[i] = reader.read_edge();
                }
                break;
            }

            case SECTION_EVOLUTION_CONFIG: {
                data.config.rule = reader.read_string();
                data.config.max_steps = reader.read_i32();
                data.config.max_states_per_step = reader.read_i32();
                data.config.max_successors_per_parent = reader.read_i32();
                data.config.exploration_probability = reader.read_f32();
                data.config.canonicalize_states = reader.read_u32() != 0;
                break;
            }

            case SECTION_STATES: {
                data.total_states = reader.read_i32();
                data.total_events = reader.read_i32();
                data.max_step_reached = reader.read_i32();

                uint32_t n_steps = reader.read_u32();
                data.states_by_step.resize(n_steps);
                for (uint32_t s = 0; s < n_steps; ++s) {
                    uint32_t n_graphs = reader.read_u32();
                    data.states_by_step[s].resize(n_graphs);
                    for (uint32_t g = 0; g < n_graphs; ++g) {
                        data.states_by_step[s][g] = read_simple_graph(reader);
                    }
                }
                break;
            }

            default:
                std::cerr << "Unknown section in evolution file: " << section << std::endl;
                return false;
        }
    }

    return true;
}

// =============================================================================
// Utility Implementations
// =============================================================================

std::string BHInitialCondition::to_hge_format() const {
    std::string result = "{";
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i > 0) result += ", ";
        result += "{" + std::to_string(edges[i].v1) + ", " + std::to_string(edges[i].v2) + "}";
    }
    result += "}";
    return result;
}

std::string CoordKey::to_string() const {
    std::string s = "{";
    for (int i = 0; i < num_anchors; ++i) {
        if (i > 0) s += ",";
        s += std::to_string(coords[i]);
    }
    s += "}";
    return s;
}

} // namespace viz::blackhole
