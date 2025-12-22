// Abstract layout engine interface
// Implementations: CUDA (NVIDIA), Compute shaders (Vulkan/WebGPU)

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

namespace viz::layout {

// Forward declarations
struct LayoutGraph;
struct LayoutResult;

// Layout engine backend type
enum class LayoutBackend {
    Auto,           // Auto-detect best available
    CUDA,           // NVIDIA CUDA (best performance on NVIDIA)
    VulkanCompute,  // Vulkan compute shaders
    WebGPUCompute,  // WebGPU compute shaders
    CPU             // CPU fallback (single-threaded, for debugging)
};

// Layout algorithm type
enum class LayoutAlgorithm {
    BarnesHut,      // O(n log n) Barnes-Hut approximation
    Direct          // O(n^2) direct computation (for small graphs)
};

// Layout dimensionality
enum class LayoutDimension {
    Layout3D,       // Full 3D layout (octree)
    Layout2D        // 2D layout (quadtree), Z preserved but not used in forces
};

// Graph data for layout (CPU-side, uploaded to GPU)
struct LayoutGraph {
    // Vertex data
    std::vector<float> positions_x;
    std::vector<float> positions_y;
    std::vector<float> positions_z;
    std::vector<float> masses;          // Vertex masses (affects forces)
    std::vector<bool> pinned;           // Pinned vertices don't move
    std::vector<bool> visible;          // Visibility mask for ghost layout (empty = all visible)
                                        // Hidden vertices receive forces but don't influence visible ones

    // Edge data (springs)
    std::vector<uint32_t> edge_sources;
    std::vector<uint32_t> edge_targets;
    std::vector<float> edge_rest_lengths; // Desired edge lengths (0 = compute from positions)
    std::vector<float> edge_strengths;    // Per-edge spring strength multiplier

    uint32_t vertex_count() const { return static_cast<uint32_t>(positions_x.size()); }
    uint32_t edge_count() const { return static_cast<uint32_t>(edge_sources.size()); }

    // Add vertex, returns index
    uint32_t add_vertex(float x, float y, float z, float mass = 1.0f, bool pin = false) {
        uint32_t idx = vertex_count();
        positions_x.push_back(x);
        positions_y.push_back(y);
        positions_z.push_back(z);
        masses.push_back(mass);
        pinned.push_back(pin);
        return idx;
    }

    // Add edge
    void add_edge(uint32_t src, uint32_t dst, float rest_length = 0.0f, float strength = 1.0f) {
        edge_sources.push_back(src);
        edge_targets.push_back(dst);
        edge_rest_lengths.push_back(rest_length);
        edge_strengths.push_back(strength);
    }

    void clear() {
        positions_x.clear();
        positions_y.clear();
        positions_z.clear();
        masses.clear();
        pinned.clear();
        visible.clear();
        edge_sources.clear();
        edge_targets.clear();
        edge_rest_lengths.clear();
        edge_strengths.clear();
    }
};

// Result of layout iteration(s)
struct LayoutResult {
    bool converged;                 // True if layout has settled
    float max_displacement;         // Maximum vertex movement this iteration
    float average_displacement;     // Average vertex movement
    uint32_t iteration_count;       // Total iterations performed
    double compute_time_ms;         // Time spent in GPU compute
};

// Layout parameters (can be changed between iterations)
struct LayoutParams {
    LayoutAlgorithm algorithm = LayoutAlgorithm::BarnesHut;
    LayoutDimension dimension = LayoutDimension::Layout3D;

    // Force parameters
    float spring_constant = 1.0f;
    float repulsion_constant = 1.0f;
    float damping = 0.9f;
    float gravity = 0.0f;           // Pull toward center

    // Barnes-Hut specific
    float theta = 0.5f;             // Multipole acceptance criterion

    // Convergence
    float convergence_threshold = 0.001f;
    uint32_t max_iterations = 1000;

    // Per-iteration limits
    float max_displacement = 1.0f;

    // Edge budget for stochastic spring updates (0 = no limit, process all edges)
    // When edge_count > edge_budget, randomly sample edge_budget edges per iteration
    // This provides O(edge_budget) spring forces instead of O(E)
    uint32_t edge_budget = 0;
};

// Callback for progress during layout
using LayoutProgressCallback = std::function<void(uint32_t iteration, float displacement)>;

// Abstract layout engine interface
class ILayoutEngine {
public:
    virtual ~ILayoutEngine() = default;

    // Get backend type
    virtual LayoutBackend get_backend() const = 0;

    // Upload graph data to GPU (call before iterate)
    virtual void upload_graph(const LayoutGraph& graph) = 0;

    // Download current positions from GPU
    virtual void download_positions(LayoutGraph& graph) = 0;

    // Run one iteration of layout
    virtual LayoutResult iterate(const LayoutParams& params) = 0;

    // Run multiple iterations until convergence or max_iterations
    virtual LayoutResult run_until_converged(const LayoutParams& params,
                                              LayoutProgressCallback progress = nullptr) = 0;

    // Check if graph is currently uploaded
    virtual bool has_graph() const = 0;

    // Get current vertex count
    virtual uint32_t vertex_count() const = 0;

    // Seed positions from another graph (for child state initialization)
    // Maps vertices by index; new vertices get positions from seed_positions
    virtual void seed_from(const LayoutGraph& parent,
                           const std::vector<uint32_t>& vertex_mapping,
                           const std::vector<float>& seed_x,
                           const std::vector<float>& seed_y,
                           const std::vector<float>& seed_z) = 0;
};

// Factory function to create layout engine
std::unique_ptr<ILayoutEngine> create_layout_engine(LayoutBackend backend = LayoutBackend::Auto);

// Check which backends are available
bool is_backend_available(LayoutBackend backend);

// Get string name for backend
const char* backend_name(LayoutBackend backend);

} // namespace viz::layout
