// Evolution Visualisation Application
// Real-time visualization of hypergraph evolution from the unified CPU engine
//
// This app demonstrates:
// 1. Running ParallelEvolutionEngine in a background thread
// 2. Streaming events via MPSC ring buffer to the render thread
// 3. Incrementally building and rendering the state graph

#include <platform/window.hpp>
#include <gal/gal.hpp>
#include <gal/vulkan/vk_loader.hpp>
#include <camera/camera.hpp>
#include <math/types.hpp>
#include <layout/layout_engine.hpp>
#include <scene/hypergraph_data.hpp>
#include <scene/hypergraph_renderer.hpp>
#include <scene/evolution_observer.hpp>
#include <scene/color_palette.hpp>

// Engine headers (only included when HYPERGRAPH_ENABLE_VISUALIZATION is defined)
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
#include <hypergraph/unified/parallel_evolution.hpp>
#include <hypergraph/unified/unified_hypergraph.hpp>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <queue>
#include <algorithm>

using namespace viz;
using namespace viz::scene;

// Layout mode for evolution states
enum class EvolutionLayoutMode {
    FlatLayers,    // States in horizontal lines per generation (original)
    SlabLayers     // States in centered square/rectangle slabs per generation
};

// Vulkan surface creation helper
namespace viz::gal {
    VkSurfaceKHR create_xcb_surface(VkInstance instance, void* connection, void* window);
    VkSurfaceKHR create_win32_surface(VkInstance instance, void* hinstance, void* hwnd);
    VkInstance get_vk_instance(Device* device);
}

// Load SPIR-V shader
std::vector<uint32_t> load_spirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader: " << path << std::endl;
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirv.data()), size);
    return spirv;
}

struct Vertex {
    float x, y, z;
    float r, g, b, a;
};

std::vector<Vertex> create_axes(float length = 5.0f) {
    using namespace viz::scene::colors;
    std::vector<Vertex> vertices;
    // X axis - Red
    vertices.push_back({0, 0, 0, DEBUG_AXIS_X.x, DEBUG_AXIS_X.y, DEBUG_AXIS_X.z, DEBUG_AXIS_X.w});
    vertices.push_back({length, 0, 0, DEBUG_AXIS_X.x, DEBUG_AXIS_X.y, DEBUG_AXIS_X.z, DEBUG_AXIS_X.w});
    // Y axis - Green
    vertices.push_back({0, 0, 0, DEBUG_AXIS_Y.x, DEBUG_AXIS_Y.y, DEBUG_AXIS_Y.z, DEBUG_AXIS_Y.w});
    vertices.push_back({0, length, 0, DEBUG_AXIS_Y.x, DEBUG_AXIS_Y.y, DEBUG_AXIS_Y.z, DEBUG_AXIS_Y.w});
    // Z axis - Blue
    vertices.push_back({0, 0, 0, DEBUG_AXIS_Z.x, DEBUG_AXIS_Z.y, DEBUG_AXIS_Z.z, DEBUG_AXIS_Z.w});
    vertices.push_back({0, 0, length, DEBUG_AXIS_Z.x, DEBUG_AXIS_Z.y, DEBUG_AXIS_Z.z, DEBUG_AXIS_Z.w});
    return vertices;
}

#ifdef HYPERGRAPH_ENABLE_VISUALIZATION

// Rule configurations
enum class RuleConfig {
    BranchingEdge = 0,   // {{x,y}} -> {{y,z}, {z,x}}  (1->2 branching, 2-edges)
    SelfLoopGrowth = 1,  // {{x,y}, {y,z}} -> {{x,y}, {y,z}, {z,w}}  (path extension, 2-edges)
    TriangleGrowth = 2,  // {{x,y,z}} -> {{x,y,w}, {y,z,w}, {z,x,w}}  (3-edge splits into 3)
    TetraGrowth = 3,     // {{x,y,z,w}} -> {{x,y,z,v}, {y,z,w,v}, {z,w,x,v}, {w,x,y,v}} (4-edge splits)
    MixedArity = 4       // Multiple edge arities in initial state and rule
};

const char* rule_config_name(RuleConfig config) {
    switch (config) {
        case RuleConfig::BranchingEdge: return "Branching: {x,y} -> {y,z}, {z,x}";
        case RuleConfig::SelfLoopGrowth: return "Path Extension: {x,y},{y,z} -> {x,y},{y,z},{z,w}";
        case RuleConfig::TriangleGrowth: return "Triangle: {x,y,z} -> {x,y,w}, {y,z,w}, {z,x,w}";
        case RuleConfig::TetraGrowth: return "Tetra: {x,y,z,w} -> 4x {*,*,*,v}";
        case RuleConfig::MixedArity: return "Mixed: {x,y},{x,y,z} -> {y,z},{y,z,w},{w,x,y,z}";
        default: return "Unknown";
    }
}

// Background thread for evolution
class EvolutionRunner {
public:
    EvolutionRunner() = default;
    ~EvolutionRunner() { stop(); }

    // Non-copyable
    EvolutionRunner(const EvolutionRunner&) = delete;
    EvolutionRunner& operator=(const EvolutionRunner&) = delete;

    void start(size_t steps, RuleConfig config) {
        stop();  // Ensure any previous run is fully stopped
        stop_requested_ = false;
        running_ = true;
        complete_ = false;
        worker_ = std::thread([this, steps, config]() { run_evolution(steps, config); });
    }

    void stop() {
        // Request early termination
        stop_requested_ = true;

        // If engine is running, ask it to stop
        {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            if (engine_) {
                engine_->request_stop();
            }
        }

        // Wait for thread to finish
        if (worker_.joinable()) {
            worker_.join();
        }
        running_ = false;
    }

    // Wait for evolution to complete (blocks until done)
    void wait_for_completion() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    bool is_running() const { return running_.load(); }
    bool is_complete() const { return complete_.load(); }

    size_t num_states() const { return num_states_.load(); }
    size_t num_events() const { return num_events_.load(); }

private:
    void run_evolution(size_t steps, RuleConfig config) {
        using namespace hypergraph::unified;

        std::cout << "[Evolution] Starting evolution with " << steps << " steps" << std::endl;
        std::cout << "[Evolution] Rule: " << rule_config_name(config) << std::endl;
        std::cout << "[Evolution] VizEventSink active: " << (viz::VizEventSink::is_active() ? "YES" : "NO") << std::endl;

        // Create hypergraph and engine
        UnifiedHypergraph hg;
        ParallelEvolutionEngine engine(&hg, 4);  // 4 worker threads

        // Make engine accessible for stop requests
        {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            engine_ = &engine;
        }

        // Check if stop was requested before we even started
        if (stop_requested_.load()) {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            engine_ = nullptr;
            return;
        }

        std::vector<std::vector<VertexId>> initial;

        switch (config) {
            case RuleConfig::BranchingEdge: {
                // Rule: {{x, y}} -> {{y, z}, {z, x}}  (branching)
                auto rule = make_rule(0)
                    .lhs({0, 1})
                    .rhs({1, 2})
                    .rhs({2, 0})
                    .build();
                engine.add_rule(rule);
                // Initial state: single edge
                initial = {{0, 1}};
                break;
            }
            case RuleConfig::SelfLoopGrowth: {
                // Rule: {{x,y}, {y,z}} -> {{x,y}, {y,z}, {z,w}}  (path extension)
                // Matches a path of 2 edges and extends it by one
                auto rule = make_rule(0)
                    .lhs({0, 1})      // First edge: x -> y
                    .lhs({1, 2})      // Second edge: y -> z
                    .rhs({0, 1})      // Keep x -> y
                    .rhs({1, 2})      // Keep y -> z
                    .rhs({2, 3})      // Add z -> w (new vertex)
                    .build();
                engine.add_rule(rule);
                // Initial state: two self-loops on vertex 0
                // This forms a path {0,0},{0,0} that can be matched
                initial = {{0, 0}, {0, 0}};
                break;
            }
            case RuleConfig::TriangleGrowth: {
                // Rule: {{x,y,z}} -> {{x,y,w}, {y,z,w}, {z,x,w}}
                // A 3-edge (triangle) splits into 3 new triangles sharing a new vertex
                auto rule = make_rule(0)
                    .lhs({0, 1, 2})       // Match a 3-edge
                    .rhs({0, 1, 3})       // Triangle 1 with new vertex
                    .rhs({1, 2, 3})       // Triangle 2 with new vertex
                    .rhs({2, 0, 3})       // Triangle 3 with new vertex
                    .build();
                engine.add_rule(rule);
                // Initial state: single 3-edge (triangle)
                initial = {{0, 1, 2}};
                break;
            }
            case RuleConfig::TetraGrowth: {
                // Rule: {{x,y,z,w}} -> 4 tetrahedra sharing new vertex
                // A 4-edge splits into 4 new 4-edges
                auto rule = make_rule(0)
                    .lhs({0, 1, 2, 3})        // Match a 4-edge
                    .rhs({0, 1, 2, 4})        // Face 1 with new vertex
                    .rhs({1, 2, 3, 4})        // Face 2 with new vertex
                    .rhs({2, 3, 0, 4})        // Face 3 with new vertex
                    .rhs({3, 0, 1, 4})        // Face 4 with new vertex
                    .build();
                engine.add_rule(rule);
                // Initial state: single 4-edge (tetrahedron)
                initial = {{0, 1, 2, 3}};
                break;
            }
            case RuleConfig::MixedArity: {
                // Rule with mixed arities: {{x,y}, {x,y,z}} -> {{y,z}, {y,z,w}, {w,x,y,z}}
                // Matches a 2-edge and 3-edge sharing vertices, produces 2-edge, 3-edge, 4-edge
                auto rule = make_rule(0)
                    .lhs({0, 1})          // 2-edge
                    .lhs({0, 1, 2})       // 3-edge sharing x,y
                    .rhs({1, 2})          // New 2-edge
                    .rhs({1, 2, 3})       // New 3-edge with fresh vertex
                    .rhs({3, 0, 1, 2})    // New 4-edge
                    .build();
                engine.add_rule(rule);
                // Initial state: 2-edge and 3-edge sharing vertex 0
                initial = {{0, 1}, {0, 1, 2}};
                break;
            }
        }

        // Enable genesis events for initial states - this creates synthetic events
        // that "produce" all initial edges, enabling causal tracking from gen 0
        engine.set_genesis_events(true);

        // Run evolution (this calls job_system_->wait_for_completion() internally)
        engine.evolve(initial, steps);

        // Update stats
        num_states_ = engine.num_states();
        num_events_ = engine.num_events();

        // Clear engine pointer before it goes out of scope
        {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            engine_ = nullptr;
        }

        if (stop_requested_.load()) {
            std::cout << "[Evolution] Stopped early: " << num_states_ << " states, "
                      << num_events_ << " events" << std::endl;
        } else {
            std::cout << "[Evolution] Complete: " << num_states_ << " states, "
                      << num_events_ << " events" << std::endl;
        }

        complete_ = true;
        running_ = false;
    }

    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> complete_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<size_t> num_states_{0};
    std::atomic<size_t> num_events_{0};

    // Protected access to engine for stop requests
    std::mutex engine_mutex_;
    hypergraph::unified::ParallelEvolutionEngine* engine_ = nullptr;
};
#endif

// Layout states in a deterministic tree layout
// Uses parent-child relationships from evolution edges to compute generations
// IMPORTANT: Only renders states that are connected (have incoming edge or are initial)
EvolutionRenderer::EvolutionLayout layout_evolution_simple(const Evolution& evo) {
    EvolutionRenderer::EvolutionLayout layout;

    if (evo.states.empty()) {
        return layout;
    }

    const float spacing = 3.0f;
    const float layer_height = 4.0f;

    const size_t num_states = evo.states.size();

    // Build adjacency: for each state, find its parent via Event edges
    // and track children for deterministic ordering
    std::vector<StateId> parent(num_states, UINT32_MAX);  // UINT32_MAX = no parent
    std::vector<std::vector<StateId>> children(num_states);

    for (const auto& edge : evo.evolution_edges) {
        if (edge.type == EvolutionEdgeType::Event) {
            if (edge.source < num_states && edge.target < num_states) {
                // target is child of source
                if (parent[edge.target] == UINT32_MAX) {
                    parent[edge.target] = edge.source;
                    children[edge.source].push_back(edge.target);
                }
            }
        }
    }

    // Find root(s) - only initial states (not orphaned states waiting for edges)
    std::vector<StateId> roots;
    for (StateId s = 0; s < num_states; ++s) {
        // A state is a root if it's marked as initial
        if (evo.states[s].is_initial) {
            roots.push_back(s);
        }
    }

    // Sort roots and children for deterministic output
    std::sort(roots.begin(), roots.end());
    for (auto& child_list : children) {
        std::sort(child_list.begin(), child_list.end());
    }

    // BFS from roots to compute generations and track which states are reachable
    std::vector<uint32_t> generation(num_states, 0);
    std::vector<bool> visited(num_states, false);
    std::vector<std::vector<StateId>> by_gen;

    std::queue<StateId> queue;
    for (StateId root : roots) {
        queue.push(root);
        visited[root] = true;
    }
    if (!roots.empty()) {
        by_gen.push_back(roots);
    }

    while (!queue.empty()) {
        StateId s = queue.front();
        queue.pop();

        uint32_t child_gen = generation[s] + 1;
        for (StateId child : children[s]) {
            if (!visited[child]) {
                visited[child] = true;
                generation[child] = child_gen;
                while (by_gen.size() <= child_gen) {
                    by_gen.push_back({});
                }
                by_gen[child_gen].push_back(child);
                queue.push(child);
            }
        }
    }

    // Sort each generation for deterministic layout
    for (auto& gen_states : by_gen) {
        std::sort(gen_states.begin(), gen_states.end());
    }

    // Position states - center each generation horizontally
    // Y increases downward (positive Y = down the screen, toward camera)
    layout.state_positions.resize(num_states, {0, 0, 0});
    layout.state_visible = visited;  // Copy visibility from BFS traversal

    for (size_t gen = 0; gen < by_gen.size(); ++gen) {
        const auto& gen_states = by_gen[gen];
        if (gen_states.empty()) continue;

        float y = static_cast<float>(gen) * layer_height;  // Positive Y = down
        float width = static_cast<float>(gen_states.size() - 1) * spacing;
        float start_x = -width / 2.0f;

        for (size_t i = 0; i < gen_states.size(); ++i) {
            StateId s = gen_states[i];
            layout.state_positions[s] = {
                start_x + static_cast<float>(i) * spacing,
                y,
                0.0f
            };
        }
    }

    return layout;
}

// Layout states in centered square/rectangle slabs per generation
// Each generation is arranged in a 2D grid matching the target aspect ratio
// aspect_ratio = width / height (e.g., 16:9 = 1.777)
EvolutionRenderer::EvolutionLayout layout_evolution_slab(const Evolution& evo,
                                                          float aspect_ratio = 16.0f / 9.0f) {
    EvolutionRenderer::EvolutionLayout layout;

    if (evo.states.empty()) {
        return layout;
    }

    const float spacing = 3.0f;        // Spacing between states in XZ plane
    const float layer_height = 5.0f;   // Vertical spacing between generation slabs

    const size_t num_states = evo.states.size();

    // Build parent-child relationships (same as flat layout)
    std::vector<StateId> parent(num_states, UINT32_MAX);
    std::vector<std::vector<StateId>> children(num_states);

    for (const auto& edge : evo.evolution_edges) {
        if (edge.type == EvolutionEdgeType::Event) {
            if (edge.source < num_states && edge.target < num_states) {
                if (parent[edge.target] == UINT32_MAX) {
                    parent[edge.target] = edge.source;
                    children[edge.source].push_back(edge.target);
                }
            }
        }
    }

    // Find initial roots
    std::vector<StateId> roots;
    for (StateId s = 0; s < num_states; ++s) {
        if (evo.states[s].is_initial) {
            roots.push_back(s);
        }
    }
    std::sort(roots.begin(), roots.end());
    for (auto& child_list : children) {
        std::sort(child_list.begin(), child_list.end());
    }

    // BFS to compute generations
    std::vector<uint32_t> generation(num_states, 0);
    std::vector<bool> visited(num_states, false);
    std::vector<std::vector<StateId>> by_gen;

    std::queue<StateId> queue;
    for (StateId root : roots) {
        queue.push(root);
        visited[root] = true;
    }
    if (!roots.empty()) {
        by_gen.push_back(roots);
    }

    while (!queue.empty()) {
        StateId s = queue.front();
        queue.pop();

        uint32_t child_gen = generation[s] + 1;
        for (StateId child : children[s]) {
            if (!visited[child]) {
                visited[child] = true;
                generation[child] = child_gen;
                while (by_gen.size() <= child_gen) {
                    by_gen.push_back({});
                }
                by_gen[child_gen].push_back(child);
                queue.push(child);
            }
        }
    }

    for (auto& gen_states : by_gen) {
        std::sort(gen_states.begin(), gen_states.end());
    }

    // Position states in 2D slabs centered at each generation's Y level
    layout.state_positions.resize(num_states, {0, 0, 0});
    layout.state_visible = visited;

    for (size_t gen = 0; gen < by_gen.size(); ++gen) {
        const auto& gen_states = by_gen[gen];
        if (gen_states.empty()) continue;

        size_t count = gen_states.size();
        float y = static_cast<float>(gen) * layer_height;

        // Compute grid dimensions to best match aspect ratio
        // aspect = cols / rows => cols = rows * aspect
        // count = rows * cols = rows * rows * aspect => rows = sqrt(count / aspect)
        uint32_t rows = static_cast<uint32_t>(std::ceil(std::sqrt(count / aspect_ratio)));
        if (rows < 1) rows = 1;
        uint32_t cols = static_cast<uint32_t>(std::ceil(static_cast<float>(count) / rows));
        if (cols < 1) cols = 1;

        // Center the grid in XZ plane
        float grid_width = static_cast<float>(cols - 1) * spacing;
        float grid_depth = static_cast<float>(rows - 1) * spacing;
        float start_x = -grid_width / 2.0f;
        float start_z = -grid_depth / 2.0f;

        for (size_t i = 0; i < count; ++i) {
            StateId s = gen_states[i];
            uint32_t col = static_cast<uint32_t>(i % cols);
            uint32_t row = static_cast<uint32_t>(i / cols);

            layout.state_positions[s] = {
                start_x + static_cast<float>(col) * spacing,
                y,
                start_z + static_cast<float>(row) * spacing
            };
        }
    }

    return layout;
}

// Unified layout function that dispatches to the appropriate layout algorithm
EvolutionRenderer::EvolutionLayout layout_evolution(const Evolution& evo,
                                                     EvolutionLayoutMode mode,
                                                     float aspect_ratio = 16.0f / 9.0f) {
    switch (mode) {
        case EvolutionLayoutMode::SlabLayers:
            return layout_evolution_slab(evo, aspect_ratio);
        case EvolutionLayoutMode::FlatLayers:
        default:
            return layout_evolution_simple(evo);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Evolution Visualisation" << std::endl;
    std::cout << "=======================" << std::endl;

#ifndef HYPERGRAPH_ENABLE_VISUALIZATION
    std::cout << "WARNING: HYPERGRAPH_ENABLE_VISUALIZATION not defined!" << std::endl;
    std::cout << "Running in mock mode (no real evolution)." << std::endl;
    std::cout << std::endl;
#endif

    std::cout << "Controls:" << std::endl;
    std::cout << "  Left-drag: Orbit camera" << std::endl;
    std::cout << "  Shift+drag: Pan camera" << std::endl;
    std::cout << "  Scroll: Zoom" << std::endl;
    std::cout << "  SPACE: Start/restart evolution with selected rule" << std::endl;
    std::cout << "  1-5: Select rule (see below)" << std::endl;
    std::cout << "  L: Toggle layout mode (Flat Layers / Slab Layers)" << std::endl;
    std::cout << "  M: Toggle MSAA antialiasing (4x, default ON)" << std::endl;
    std::cout << "  A: Toggle debug axis (XYZ lines)" << std::endl;
    std::cout << "  R: Reset visualization" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;
    std::cout << std::endl;
    std::cout << "Rules:" << std::endl;
    std::cout << "  1: {x,y} -> {y,z}, {z,x}           (2-edge branching)" << std::endl;
    std::cout << "  2: {x,y},{y,z} -> +{z,w}           (2-edge path extension, init={{0,0},{0,0}})" << std::endl;
    std::cout << "  3: {x,y,z} -> 3x {*,*,w}           (3-edge triangle split)" << std::endl;
    std::cout << "  4: {x,y,z,w} -> 4x {*,*,*,v}       (4-edge tetra split)" << std::endl;
    std::cout << "  5: {x,y},{x,y,z} -> mixed arities  (2+3 edge -> 2+3+4 edge)" << std::endl;
    std::cout << std::endl;

    // Create window
    platform::WindowDesc window_desc;
    window_desc.title = "Evolution Visualisation";
    window_desc.width = 1920;
    window_desc.height = 1080;

    auto window = platform::Window::create(window_desc);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        return 1;
    }

    // Initialize GAL
    if (!gal::initialize(gal::Backend::Vulkan)) {
        std::cerr << "Failed to initialize GAL" << std::endl;
        return 1;
    }

    gal::DeviceDesc device_desc;
    device_desc.app_name = "EvolutionViz";
    device_desc.enable_validation = true;

    auto device = gal::Device::create(device_desc);
    if (!device) {
        std::cerr << "Failed to create device" << std::endl;
        gal::shutdown();
        return 1;
    }

    std::cout << "Device: " << device->get_info().device_name << std::endl;

    // Create surface
    VkInstance vk_instance = gal::get_vk_instance(device.get());
    VkSurfaceKHR surface = VK_NULL_HANDLE;

#if defined(VIZ_PLATFORM_LINUX)
    surface = gal::create_xcb_surface(vk_instance, window->get_native_display(), window->get_native_window());
#elif defined(VIZ_PLATFORM_WINDOWS)
    surface = gal::create_win32_surface(vk_instance, GetModuleHandle(nullptr), window->get_native_window());
#endif

    if (surface == VK_NULL_HANDLE) {
        std::cerr << "Failed to create surface" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    auto swapchain = device->create_swapchain(
        reinterpret_cast<gal::Handle>(surface),
        window->get_width(), window->get_height());

    if (!swapchain) {
        std::cerr << "Failed to create swapchain" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // Load shaders
    auto vert_spirv = load_spirv("../visualisation/shaders/spirv/basic3d.vert.spv");
    auto frag_spirv = load_spirv("../visualisation/shaders/spirv/basic3d.frag.spv");
    auto instance_cone_vert_spirv = load_spirv("../visualisation/shaders/spirv/instance_cone.vert.spv");
    auto instance_sphere_vert_spirv = load_spirv("../visualisation/shaders/spirv/instance_sphere.vert.spv");

    if (vert_spirv.empty() || frag_spirv.empty()) {
        std::cerr << "Failed to load basic shaders" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // Instance shaders are optional - if not available, instanced rendering won't be used
    bool instanced_shaders_available = !instance_cone_vert_spirv.empty() && !instance_sphere_vert_spirv.empty();
    if (!instanced_shaders_available) {
        std::cerr << "Warning: Instanced shaders not found, instanced rendering disabled" << std::endl;
    }

    gal::ShaderDesc vert_desc;
    vert_desc.stage = gal::ShaderStage::Vertex;
    vert_desc.spirv_code = vert_spirv.data();
    vert_desc.spirv_size = vert_spirv.size() * sizeof(uint32_t);
    auto vertex_shader = device->create_shader(vert_desc);

    gal::ShaderDesc frag_desc;
    frag_desc.stage = gal::ShaderStage::Fragment;
    frag_desc.spirv_code = frag_spirv.data();
    frag_desc.spirv_size = frag_spirv.size() * sizeof(uint32_t);
    auto fragment_shader = device->create_shader(frag_desc);

    if (!vertex_shader || !fragment_shader) {
        std::cerr << "Failed to create shaders" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // Vertex layout
    gal::VertexAttribute attribs[] = {
        {0, gal::Format::RGB32_FLOAT, 0},
        {1, gal::Format::RGBA32_FLOAT, sizeof(float) * 3}
    };

    gal::VertexBufferLayout vertex_layout;
    vertex_layout.stride = sizeof(Vertex);
    vertex_layout.step_mode = gal::VertexStepMode::Vertex;
    vertex_layout.attributes = attribs;
    vertex_layout.attribute_count = 2;

    // Pipelines
    gal::Format color_format = swapchain->get_format();

    gal::BlendState opaque_blend;
    opaque_blend.write_mask = 0xF;
    opaque_blend.blend_enable = false;

    gal::BlendState alpha_blend = gal::BlendState::alpha_blend();

    gal::RenderPipelineDesc pipeline_desc;
    pipeline_desc.vertex_shader = vertex_shader.get();
    pipeline_desc.fragment_shader = fragment_shader.get();
    pipeline_desc.vertex_layouts = &vertex_layout;
    pipeline_desc.vertex_layout_count = 1;
    pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
    pipeline_desc.rasterizer.cull_mode = gal::CullMode::Back;
    pipeline_desc.depth_stencil.depth_test_enable = false;
    pipeline_desc.blend_states = &opaque_blend;
    pipeline_desc.blend_state_count = 1;
    pipeline_desc.color_formats = &color_format;
    pipeline_desc.color_format_count = 1;
    pipeline_desc.push_constant_size = sizeof(math::mat4);

    auto triangle_pipeline = device->create_render_pipeline(pipeline_desc);

    pipeline_desc.topology = gal::PrimitiveTopology::LineList;
    pipeline_desc.rasterizer.cull_mode = gal::CullMode::None;
    auto line_pipeline = device->create_render_pipeline(pipeline_desc);

    pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
    pipeline_desc.blend_states = &alpha_blend;
    auto transparent_pipeline = device->create_render_pipeline(pipeline_desc);

    // Transparent line pipeline (for wireframe with alpha < 1)
    pipeline_desc.topology = gal::PrimitiveTopology::LineList;
    pipeline_desc.blend_states = &alpha_blend;
    auto transparent_line_pipeline = device->create_render_pipeline(pipeline_desc);

    if (!triangle_pipeline || !line_pipeline || !transparent_pipeline || !transparent_line_pipeline) {
        std::cerr << "Failed to create pipelines" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // ========== Instanced Rendering Setup ==========
    std::unique_ptr<gal::Shader> instance_cone_vertex_shader;
    std::unique_ptr<gal::Shader> instance_sphere_vertex_shader;
    std::unique_ptr<gal::RenderPipeline> instanced_cone_pipeline;
    std::unique_ptr<gal::RenderPipeline> instanced_sphere_pipeline;
    bool instanced_cone_available = false;
    bool instanced_sphere_available = false;

    // Generate unit meshes
    UnitConeMesh unit_cone;
    unit_cone.generate(8);
    UnitSphereMesh unit_sphere;
    unit_sphere.generate(12, 8);

    if (instanced_shaders_available) {
        // Create cone shader
        gal::ShaderDesc cone_vert_desc;
        cone_vert_desc.stage = gal::ShaderStage::Vertex;
        cone_vert_desc.spirv_code = instance_cone_vert_spirv.data();
        cone_vert_desc.spirv_size = instance_cone_vert_spirv.size() * sizeof(uint32_t);
        cone_vert_desc.debug_name = "instance_cone_vert";
        instance_cone_vertex_shader = device->create_shader(cone_vert_desc);

        // Create sphere shader
        gal::ShaderDesc sphere_vert_desc;
        sphere_vert_desc.stage = gal::ShaderStage::Vertex;
        sphere_vert_desc.spirv_code = instance_sphere_vert_spirv.data();
        sphere_vert_desc.spirv_size = instance_sphere_vert_spirv.size() * sizeof(uint32_t);
        sphere_vert_desc.debug_name = "instance_sphere_vert";
        instance_sphere_vertex_shader = device->create_shader(sphere_vert_desc);

        if (instance_cone_vertex_shader && instance_sphere_vertex_shader) {
            // Cone vertex layout (binding 0, per-vertex)
            gal::VertexAttribute cone_attribs[] = {
                {0, gal::Format::RGB32_FLOAT, 0},                    // position
                {1, gal::Format::RGB32_FLOAT, sizeof(float) * 3},    // normal
            };
            gal::VertexBufferLayout cone_vertex_layout;
            cone_vertex_layout.stride = sizeof(ConeVertex);
            cone_vertex_layout.step_mode = gal::VertexStepMode::Vertex;
            cone_vertex_layout.attributes = cone_attribs;
            cone_vertex_layout.attribute_count = 2;

            // Cone instance layout (binding 1, per-instance)
            gal::VertexAttribute cone_instance_attribs[] = {
                {2, gal::Format::RGB32_FLOAT, 0},                     // tip position
                {3, gal::Format::RGB32_FLOAT, sizeof(float) * 3},     // direction
                {4, gal::Format::RG32_FLOAT, sizeof(float) * 6},      // length, radius
                {5, gal::Format::RGBA32_FLOAT, sizeof(float) * 8},    // color
            };
            gal::VertexBufferLayout cone_instance_layout;
            cone_instance_layout.stride = sizeof(ConeInstance);
            cone_instance_layout.step_mode = gal::VertexStepMode::Instance;
            cone_instance_layout.attributes = cone_instance_attribs;
            cone_instance_layout.attribute_count = 4;

            gal::VertexBufferLayout cone_layouts[] = {cone_vertex_layout, cone_instance_layout};

            gal::RenderPipelineDesc cone_pipeline_desc;
            cone_pipeline_desc.vertex_shader = instance_cone_vertex_shader.get();
            cone_pipeline_desc.fragment_shader = fragment_shader.get();
            cone_pipeline_desc.vertex_layouts = cone_layouts;
            cone_pipeline_desc.vertex_layout_count = 2;
            cone_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
            cone_pipeline_desc.rasterizer.cull_mode = gal::CullMode::Back;
            cone_pipeline_desc.depth_stencil.depth_test_enable = false;
            cone_pipeline_desc.blend_states = &opaque_blend;
            cone_pipeline_desc.blend_state_count = 1;
            cone_pipeline_desc.color_formats = &color_format;
            cone_pipeline_desc.color_format_count = 1;
            cone_pipeline_desc.push_constant_size = sizeof(math::mat4);
            cone_pipeline_desc.debug_name = "instanced_cone_pipeline";

            instanced_cone_pipeline = device->create_render_pipeline(cone_pipeline_desc);
            instanced_cone_available = instanced_cone_pipeline != nullptr;

            // Sphere vertex layout (binding 0, per-vertex)
            gal::VertexAttribute sphere_attribs[] = {
                {0, gal::Format::RGB32_FLOAT, 0},                    // position
                {1, gal::Format::RGB32_FLOAT, sizeof(float) * 3},    // normal
            };
            gal::VertexBufferLayout sphere_vertex_layout;
            sphere_vertex_layout.stride = sizeof(SphereVertex);
            sphere_vertex_layout.step_mode = gal::VertexStepMode::Vertex;
            sphere_vertex_layout.attributes = sphere_attribs;
            sphere_vertex_layout.attribute_count = 2;

            // Sphere instance layout (binding 1, per-instance)
            gal::VertexAttribute sphere_instance_attribs[] = {
                {2, gal::Format::RGB32_FLOAT, 0},                     // center position
                {3, gal::Format::R32_FLOAT, sizeof(float) * 3},       // radius
                {4, gal::Format::RGBA32_FLOAT, sizeof(float) * 4},    // color
            };
            gal::VertexBufferLayout sphere_instance_layout;
            sphere_instance_layout.stride = sizeof(SphereInstance);
            sphere_instance_layout.step_mode = gal::VertexStepMode::Instance;
            sphere_instance_layout.attributes = sphere_instance_attribs;
            sphere_instance_layout.attribute_count = 3;

            gal::VertexBufferLayout sphere_layouts[] = {sphere_vertex_layout, sphere_instance_layout};

            gal::RenderPipelineDesc sphere_pipeline_desc;
            sphere_pipeline_desc.vertex_shader = instance_sphere_vertex_shader.get();
            sphere_pipeline_desc.fragment_shader = fragment_shader.get();
            sphere_pipeline_desc.vertex_layouts = sphere_layouts;
            sphere_pipeline_desc.vertex_layout_count = 2;
            sphere_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
            sphere_pipeline_desc.rasterizer.cull_mode = gal::CullMode::Back;
            sphere_pipeline_desc.depth_stencil.depth_test_enable = false;
            sphere_pipeline_desc.blend_states = &opaque_blend;
            sphere_pipeline_desc.blend_state_count = 1;
            sphere_pipeline_desc.color_formats = &color_format;
            sphere_pipeline_desc.color_format_count = 1;
            sphere_pipeline_desc.push_constant_size = sizeof(math::mat4);
            sphere_pipeline_desc.debug_name = "instanced_sphere_pipeline";

            instanced_sphere_pipeline = device->create_render_pipeline(sphere_pipeline_desc);
            instanced_sphere_available = instanced_sphere_pipeline != nullptr;

            if (instanced_cone_available && instanced_sphere_available) {
                std::cout << "Instanced rendering enabled (cones + spheres)" << std::endl;
            }
        }
    }

    // Create renderers
    EvolutionRenderer evo_renderer;
    // NOTE: EvolutionObserver contains a large ring buffer (~8MB), must be heap-allocated
    auto observer = std::make_unique<EvolutionObserver>();

    // Create buffers
    size_t buffer_size = 500000 * sizeof(Vertex);
    gal::BufferDesc buffer_desc;
    buffer_desc.size = buffer_size;
    buffer_desc.usage = gal::BufferUsage::Vertex;
    buffer_desc.memory = gal::MemoryLocation::CPU_TO_GPU;

    auto axis_buffer = device->create_buffer(buffer_desc);
    auto triangle_buffer = device->create_buffer(buffer_desc);
    auto transparent_buffer = device->create_buffer(buffer_desc);
    auto line_buffer = device->create_buffer(buffer_desc);
    auto transparent_line_buffer = device->create_buffer(buffer_desc);

    auto axis_verts = create_axes(5.0f);
    if (axis_buffer) {
        axis_buffer->write(axis_verts.data(), axis_verts.size() * sizeof(Vertex));
    }

    // Instanced rendering buffers
    std::unique_ptr<gal::Buffer> cone_mesh_buffer;
    std::unique_ptr<gal::Buffer> cone_instance_buffer;
    std::unique_ptr<gal::Buffer> sphere_mesh_buffer;
    std::unique_ptr<gal::Buffer> sphere_instance_buffer;
    size_t cone_instance_buffer_size = 50000 * sizeof(ConeInstance);
    size_t sphere_instance_buffer_size = 50000 * sizeof(SphereInstance);

    // Instance data vectors
    std::vector<ConeInstance> cone_instances;
    std::vector<SphereInstance> sphere_instances;

    if (instanced_cone_available) {
        gal::BufferDesc mesh_desc;
        mesh_desc.size = unit_cone.byte_size();
        mesh_desc.usage = gal::BufferUsage::Vertex;
        mesh_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        mesh_desc.initial_data = unit_cone.data();
        mesh_desc.debug_name = "unit_cone_mesh";
        cone_mesh_buffer = device->create_buffer(mesh_desc);

        gal::BufferDesc inst_desc;
        inst_desc.size = cone_instance_buffer_size;
        inst_desc.usage = gal::BufferUsage::Vertex;
        inst_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        inst_desc.debug_name = "cone_instances";
        cone_instance_buffer = device->create_buffer(inst_desc);

        if (!cone_mesh_buffer || !cone_instance_buffer) {
            std::cerr << "Warning: Failed to create cone instance buffers" << std::endl;
            instanced_cone_available = false;
        }
    }

    if (instanced_sphere_available) {
        gal::BufferDesc mesh_desc;
        mesh_desc.size = unit_sphere.byte_size();
        mesh_desc.usage = gal::BufferUsage::Vertex;
        mesh_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        mesh_desc.initial_data = unit_sphere.data();
        mesh_desc.debug_name = "unit_sphere_mesh";
        sphere_mesh_buffer = device->create_buffer(mesh_desc);

        gal::BufferDesc inst_desc;
        inst_desc.size = sphere_instance_buffer_size;
        inst_desc.usage = gal::BufferUsage::Vertex;
        inst_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        inst_desc.debug_name = "sphere_instances";
        sphere_instance_buffer = device->create_buffer(inst_desc);

        if (!sphere_mesh_buffer || !sphere_instance_buffer) {
            std::cerr << "Warning: Failed to create sphere instance buffers" << std::endl;
            instanced_sphere_available = false;
        }
    }

    // Camera - looking at positive Y (layout grows downward in positive Y)
    camera::PerspectiveCamera cam;
    cam.set_perspective(60.0f, static_cast<float>(window->get_width()) / window->get_height(), 0.1f, 1000.0f);
    cam.set_target(math::vec3(0, 5, 0));  // Target slightly below origin where evolution grows
    cam.set_distance(25.0f);
    cam.orbit(0.5f, -0.3f);  // Negative pitch to look down at the tree

    camera::CameraController controller(&cam);

    // Sync objects
    auto image_semaphore = device->create_semaphore();
    auto render_semaphore = device->create_semaphore();
    auto fence = device->create_fence(true);

    // Input state
    bool left_mouse_down = false;
    float last_mouse_x = 0, last_mouse_y = 0;
    bool should_resize = false;
    uint32_t new_width = 0, new_height = 0;

#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
    EvolutionRunner evolution_runner;
    bool evolution_started = false;
    RuleConfig current_rule = RuleConfig::BranchingEdge;
    size_t evolution_steps = 5;
#endif
    EvolutionLayoutMode layout_mode = EvolutionLayoutMode::SlabLayers;  // Default to slab layout
    float viewport_aspect = 16.0f / 9.0f;  // Will be updated from actual window dimensions
    bool show_debug_axis = false;  // Hidden by default, toggle with 'A' key

    // MSAA state
    bool msaa_enabled = true;  // MSAA on by default
    uint32_t msaa_samples = 4;  // 4x MSAA
    bool msaa_dirty = true;     // Need to create MSAA resources
    std::unique_ptr<gal::Texture> msaa_texture;
    std::unique_ptr<gal::RenderPipeline> msaa_triangle_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_line_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_transparent_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_transparent_line_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_instanced_cone_pipeline;
    std::unique_ptr<gal::RenderPipeline> msaa_instanced_sphere_pipeline;

    // Helper to create MSAA resources
    auto create_msaa_resources = [&](uint32_t width, uint32_t height) {
        if (!msaa_enabled) {
            msaa_texture.reset();
            msaa_triangle_pipeline.reset();
            msaa_line_pipeline.reset();
            msaa_transparent_pipeline.reset();
            msaa_transparent_line_pipeline.reset();
            msaa_instanced_cone_pipeline.reset();
            msaa_instanced_sphere_pipeline.reset();
            return;
        }

        // Create MSAA color texture
        gal::TextureDesc msaa_desc;
        msaa_desc.size = {width, height, 1};
        msaa_desc.format = swapchain->get_format();
        msaa_desc.usage = gal::TextureUsage::RenderTarget;
        msaa_desc.sample_count = msaa_samples;
        msaa_texture = device->create_texture(msaa_desc);

        if (!msaa_texture) {
            std::cerr << "Failed to create MSAA texture, falling back to no MSAA" << std::endl;
            msaa_enabled = false;
            return;
        }

        // Create MSAA pipelines
        gal::RenderPipelineDesc msaa_pipeline_desc;
        msaa_pipeline_desc.vertex_shader = vertex_shader.get();
        msaa_pipeline_desc.fragment_shader = fragment_shader.get();
        msaa_pipeline_desc.vertex_layouts = &vertex_layout;
        msaa_pipeline_desc.vertex_layout_count = 1;
        msaa_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
        msaa_pipeline_desc.rasterizer.cull_mode = gal::CullMode::Back;
        msaa_pipeline_desc.depth_stencil.depth_test_enable = false;
        msaa_pipeline_desc.blend_states = &opaque_blend;
        msaa_pipeline_desc.blend_state_count = 1;
        msaa_pipeline_desc.color_formats = &color_format;
        msaa_pipeline_desc.color_format_count = 1;
        msaa_pipeline_desc.push_constant_size = sizeof(math::mat4);
        msaa_pipeline_desc.multisample.count = msaa_samples;

        msaa_triangle_pipeline = device->create_render_pipeline(msaa_pipeline_desc);

        msaa_pipeline_desc.topology = gal::PrimitiveTopology::LineList;
        msaa_pipeline_desc.rasterizer.cull_mode = gal::CullMode::None;
        msaa_line_pipeline = device->create_render_pipeline(msaa_pipeline_desc);

        msaa_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
        msaa_pipeline_desc.blend_states = &alpha_blend;
        msaa_transparent_pipeline = device->create_render_pipeline(msaa_pipeline_desc);

        msaa_pipeline_desc.topology = gal::PrimitiveTopology::LineList;
        msaa_transparent_line_pipeline = device->create_render_pipeline(msaa_pipeline_desc);

        if (!msaa_triangle_pipeline || !msaa_line_pipeline ||
            !msaa_transparent_pipeline || !msaa_transparent_line_pipeline) {
            std::cerr << "Failed to create MSAA pipelines, falling back to no MSAA" << std::endl;
            msaa_enabled = false;
            msaa_texture.reset();
            msaa_triangle_pipeline.reset();
            msaa_line_pipeline.reset();
            msaa_transparent_pipeline.reset();
            msaa_transparent_line_pipeline.reset();
            return;
        }

        // Create MSAA instanced pipelines if instanced rendering is available
        if (instanced_cone_available && instance_cone_vertex_shader) {
            // Cone vertex layout (binding 0, per-vertex)
            gal::VertexAttribute cone_attribs[] = {
                {0, gal::Format::RGB32_FLOAT, 0},                    // position
                {1, gal::Format::RGB32_FLOAT, sizeof(float) * 3},    // normal
            };
            gal::VertexBufferLayout cone_vertex_layout;
            cone_vertex_layout.stride = sizeof(ConeVertex);
            cone_vertex_layout.step_mode = gal::VertexStepMode::Vertex;
            cone_vertex_layout.attributes = cone_attribs;
            cone_vertex_layout.attribute_count = 2;

            // Cone instance layout (binding 1, per-instance)
            gal::VertexAttribute cone_instance_attribs[] = {
                {2, gal::Format::RGB32_FLOAT, 0},                     // tip position
                {3, gal::Format::RGB32_FLOAT, sizeof(float) * 3},     // direction
                {4, gal::Format::RG32_FLOAT, sizeof(float) * 6},      // length, radius
                {5, gal::Format::RGBA32_FLOAT, sizeof(float) * 8},    // color
            };
            gal::VertexBufferLayout cone_instance_layout;
            cone_instance_layout.stride = sizeof(ConeInstance);
            cone_instance_layout.step_mode = gal::VertexStepMode::Instance;
            cone_instance_layout.attributes = cone_instance_attribs;
            cone_instance_layout.attribute_count = 4;

            gal::VertexBufferLayout cone_layouts[] = {cone_vertex_layout, cone_instance_layout};

            gal::RenderPipelineDesc cone_pipeline_desc;
            cone_pipeline_desc.vertex_shader = instance_cone_vertex_shader.get();
            cone_pipeline_desc.fragment_shader = fragment_shader.get();
            cone_pipeline_desc.vertex_layouts = cone_layouts;
            cone_pipeline_desc.vertex_layout_count = 2;
            cone_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
            cone_pipeline_desc.rasterizer.cull_mode = gal::CullMode::Back;
            cone_pipeline_desc.depth_stencil.depth_test_enable = false;
            cone_pipeline_desc.blend_states = &opaque_blend;
            cone_pipeline_desc.blend_state_count = 1;
            cone_pipeline_desc.color_formats = &color_format;
            cone_pipeline_desc.color_format_count = 1;
            cone_pipeline_desc.push_constant_size = sizeof(math::mat4);
            cone_pipeline_desc.multisample.count = msaa_samples;
            cone_pipeline_desc.debug_name = "msaa_instanced_cone_pipeline";

            msaa_instanced_cone_pipeline = device->create_render_pipeline(cone_pipeline_desc);
        }

        if (instanced_sphere_available && instance_sphere_vertex_shader) {
            // Sphere vertex layout (binding 0, per-vertex)
            gal::VertexAttribute sphere_attribs[] = {
                {0, gal::Format::RGB32_FLOAT, 0},                    // position
                {1, gal::Format::RGB32_FLOAT, sizeof(float) * 3},    // normal
            };
            gal::VertexBufferLayout sphere_vertex_layout;
            sphere_vertex_layout.stride = sizeof(SphereVertex);
            sphere_vertex_layout.step_mode = gal::VertexStepMode::Vertex;
            sphere_vertex_layout.attributes = sphere_attribs;
            sphere_vertex_layout.attribute_count = 2;

            // Sphere instance layout (binding 1, per-instance)
            gal::VertexAttribute sphere_instance_attribs[] = {
                {2, gal::Format::RGB32_FLOAT, 0},                     // center position
                {3, gal::Format::R32_FLOAT, sizeof(float) * 3},       // radius
                {4, gal::Format::RGBA32_FLOAT, sizeof(float) * 4},    // color
            };
            gal::VertexBufferLayout sphere_instance_layout;
            sphere_instance_layout.stride = sizeof(SphereInstance);
            sphere_instance_layout.step_mode = gal::VertexStepMode::Instance;
            sphere_instance_layout.attributes = sphere_instance_attribs;
            sphere_instance_layout.attribute_count = 3;

            gal::VertexBufferLayout sphere_layouts[] = {sphere_vertex_layout, sphere_instance_layout};

            gal::RenderPipelineDesc sphere_pipeline_desc;
            sphere_pipeline_desc.vertex_shader = instance_sphere_vertex_shader.get();
            sphere_pipeline_desc.fragment_shader = fragment_shader.get();
            sphere_pipeline_desc.vertex_layouts = sphere_layouts;
            sphere_pipeline_desc.vertex_layout_count = 2;
            sphere_pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
            sphere_pipeline_desc.rasterizer.cull_mode = gal::CullMode::Back;
            sphere_pipeline_desc.depth_stencil.depth_test_enable = false;
            sphere_pipeline_desc.blend_states = &opaque_blend;
            sphere_pipeline_desc.blend_state_count = 1;
            sphere_pipeline_desc.color_formats = &color_format;
            sphere_pipeline_desc.color_format_count = 1;
            sphere_pipeline_desc.push_constant_size = sizeof(math::mat4);
            sphere_pipeline_desc.multisample.count = msaa_samples;
            sphere_pipeline_desc.debug_name = "msaa_instanced_sphere_pipeline";

            msaa_instanced_sphere_pipeline = device->create_render_pipeline(sphere_pipeline_desc);
        }

        std::cout << "MSAA " << msaa_samples << "x enabled" << std::endl;
    };

    // Geometry buffers
    std::vector<Vertex> triangle_verts;
    std::vector<Vertex> transparent_verts;
    std::vector<Vertex> line_verts;
    std::vector<Vertex> transparent_line_verts;  // Wireframe with alpha blending
    bool geometry_dirty = true;

    // Logging state (must be outside render loop for lambda access)
    size_t last_state_count = 0;
    bool evolution_complete_reported = false;

    platform::WindowCallbacks callbacks;
    callbacks.on_resize = [&](uint32_t w, uint32_t h) {
        should_resize = true;
        new_width = w;
        new_height = h;
        float aspect = static_cast<float>(w) / h;
        cam.set_aspect_ratio(aspect);
        viewport_aspect = aspect;
    };

    callbacks.on_key = [&](platform::KeyCode key, bool pressed, platform::Modifiers mods) {
        // Track shift state for camera panning (even on release)
        if (key == platform::KeyCode::LeftShift || key == platform::KeyCode::RightShift) {
            controller.set_shift_held(pressed);
        }

        if (!pressed) return;

        if (key == platform::KeyCode::Escape) {
            window->request_close();
        }
#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
        else if (key == platform::KeyCode::Space) {
            if (!evolution_runner.is_running()) {
                // Stop any previous evolution and reset
                evolution_runner.stop();
                observer->reset();
                geometry_dirty = true;

                // Reset static counters for clean logging
                last_state_count = 0;
                evolution_complete_reported = false;

                // Start new evolution with current rule
                evolution_runner.start(evolution_steps, current_rule);
                evolution_started = true;
                std::cout << "\n=== Evolution started (" << evolution_steps << " steps) ===" << std::endl;
                std::cout << "Rule: " << rule_config_name(current_rule) << std::endl;
            }
        }
        else if (key == platform::KeyCode::Num1) {
            current_rule = RuleConfig::BranchingEdge;
            std::cout << "Selected rule 1: " << rule_config_name(current_rule) << std::endl;
            std::cout << "Press SPACE to run evolution" << std::endl;
        }
        else if (key == platform::KeyCode::Num2) {
            current_rule = RuleConfig::SelfLoopGrowth;
            std::cout << "Selected rule 2: " << rule_config_name(current_rule) << std::endl;
            std::cout << "Press SPACE to run evolution" << std::endl;
        }
        else if (key == platform::KeyCode::Num3) {
            current_rule = RuleConfig::TriangleGrowth;
            std::cout << "Selected rule 3: " << rule_config_name(current_rule) << std::endl;
            std::cout << "Press SPACE to run evolution" << std::endl;
        }
        else if (key == platform::KeyCode::Num4) {
            current_rule = RuleConfig::TetraGrowth;
            std::cout << "Selected rule 4: " << rule_config_name(current_rule) << std::endl;
            std::cout << "Press SPACE to run evolution" << std::endl;
        }
        else if (key == platform::KeyCode::Num5) {
            current_rule = RuleConfig::MixedArity;
            std::cout << "Selected rule 5: " << rule_config_name(current_rule) << std::endl;
            std::cout << "Press SPACE to run evolution" << std::endl;
        }
        else if (key == platform::KeyCode::R) {
            // Reset visualization without running evolution
            if (!evolution_runner.is_running()) {
                evolution_runner.stop();
                observer->reset();
                geometry_dirty = true;
                last_state_count = 0;
                evolution_complete_reported = false;
                evolution_started = false;
                std::cout << "Visualization reset. Press SPACE to run evolution." << std::endl;
            }
        }
#endif
        // Keys that work regardless of HYPERGRAPH_ENABLE_VISUALIZATION
        if (key == platform::KeyCode::L) {
            // Toggle layout mode
            if (layout_mode == EvolutionLayoutMode::FlatLayers) {
                layout_mode = EvolutionLayoutMode::SlabLayers;
                std::cout << "Layout mode: Slab Layers (2D grids per generation)" << std::endl;
            } else {
                layout_mode = EvolutionLayoutMode::FlatLayers;
                std::cout << "Layout mode: Flat Layers (horizontal lines per generation)" << std::endl;
            }
            geometry_dirty = true;  // Force re-layout
        }
        else if (key == platform::KeyCode::A) {
            // Toggle debug axis visibility
            show_debug_axis = !show_debug_axis;
            std::cout << "Debug axis: " << (show_debug_axis ? "ON" : "OFF") << std::endl;
        }
        else if (key == platform::KeyCode::M) {
            // Toggle MSAA
            msaa_enabled = !msaa_enabled;
            msaa_dirty = true;  // Will recreate resources on next frame
            std::cout << "MSAA: " << (msaa_enabled ? "ON (4x)" : "OFF") << std::endl;
        }
    };

    callbacks.on_mouse_button = [&](platform::MouseButton button, bool pressed, int x, int y, platform::Modifiers) {
        if (button == platform::MouseButton::Left) {
            left_mouse_down = pressed;
            controller.set_mouse_captured(pressed);
        }
    };

    callbacks.on_mouse_move = [&](int x, int y) {
        float dx = static_cast<float>(x) - last_mouse_x;
        float dy = static_cast<float>(y) - last_mouse_y;
        last_mouse_x = static_cast<float>(x);
        last_mouse_y = static_cast<float>(y);
        if (left_mouse_down) {
            controller.on_mouse_move(dx, dy);
        }
    };

    callbacks.on_scroll = [&](float dx, float dy) {
        controller.on_mouse_scroll(dy);
    };

    window->set_callbacks(callbacks);

    std::cout << "\nStarting render loop..." << std::endl;
    std::cout << "Press SPACE to start evolution" << std::endl;

    // Render loop
    while (window->is_open()) {
        window->poll_events();

        if (should_resize && new_width > 0 && new_height > 0) {
            device->wait_idle();
            swapchain->resize(new_width, new_height);
            msaa_dirty = true;  // Need to recreate MSAA texture at new size
            should_resize = false;
        }

        if (window->is_minimized()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Process events from evolution engine
        size_t events_processed = observer->process_events(100);  // Process up to 100 per frame

        if (events_processed > 0) {
            geometry_dirty = true;
            const Evolution& evo = observer->get_evolution();
            // Only log when state count changes to reduce spam
            if (evo.states.size() != last_state_count) {
                std::cout << "[Frame] States: " << evo.states.size()
                          << ", Events: " << evo.events.size()
                          << ", Edges: " << evo.evolution_edges.size() << std::endl;
                last_state_count = evo.states.size();
            }
        }

        // Check if evolution is complete
        if (observer->is_complete() && !evolution_complete_reported) {
            const Evolution& evo = observer->get_evolution();
            size_t event_edges = 0, causal_edges = 0, branchial_edges = 0;
            for (const auto& e : evo.evolution_edges) {
                if (e.type == EvolutionEdgeType::Event) event_edges++;
                else if (e.type == EvolutionEdgeType::Causal) causal_edges++;
                else if (e.type == EvolutionEdgeType::Branchial) branchial_edges++;
            }
            std::cout << "Evolution complete! Total events: "
                      << observer->total_events_processed() << std::endl;
            std::cout << "  States: " << evo.states.size()
                      << ", Events: " << evo.events.size() << std::endl;
            std::cout << "  Edge breakdown - Event: " << event_edges
                      << ", Causal: " << causal_edges
                      << ", Branchial: " << branchial_edges << std::endl;
            std::cout << "  Pending - Causal: " << observer->pending_causal_count()
                      << ", Branchial: " << observer->pending_branchial_count() << std::endl;
            evolution_complete_reported = true;
        }

        // Reset reported flag when starting new evolution
        if (events_processed > 0 && !observer->is_complete()) {
            evolution_complete_reported = false;
        }

        // Regenerate geometry if needed
        if (geometry_dirty) {
            triangle_verts.clear();
            transparent_verts.clear();
            line_verts.clear();
            transparent_line_verts.clear();
            sphere_instances.clear();
            cone_instances.clear();

            const Evolution& evo = observer->get_evolution();
            if (!evo.states.empty()) {
                auto layout = layout_evolution(evo, layout_mode, viewport_aspect);
                std::cout << "[Render] Generating geometry for " << evo.states.size()
                          << " states with " << layout.state_positions.size() << " positions"
                          << " (mode: " << (layout_mode == EvolutionLayoutMode::SlabLayers ? "Slab" : "Flat")
                          << ")" << std::endl;
                auto geo = evo_renderer.generate_states_graph(evo, layout);
                std::cout << "[Render] Generated: sphere_instances=" << geo.internal_sphere_instances.size()
                          << " cone_instances=" << geo.internal_cone_instances.size()
                          << " internal_lines=" << geo.internal_edge_lines.size()
                          << " internal_bubbles=" << geo.internal_bubbles.size()
                          << " wireframe=" << geo.state_wireframe.size()
                          << " faces=" << geo.state_faces.size()
                          << " event_arrows=" << geo.event_arrows.size()
                          << " event_lines=" << geo.event_lines.size()
                          << " causal_lines=" << geo.causal_lines.size()
                          << " branchial_lines=" << geo.branchial_lines.size() << std::endl;

                // Collect instanced sphere data
                if (instanced_sphere_available) {
                    sphere_instances.insert(sphere_instances.end(),
                        geo.internal_sphere_instances.begin(),
                        geo.internal_sphere_instances.end());
                }

                // Collect instanced cone data (internal arrows)
                if (instanced_cone_available) {
                    cone_instances.insert(cone_instances.end(),
                        geo.internal_cone_instances.begin(),
                        geo.internal_cone_instances.end());
                }

                // Event arrows still use legacy triangles (they're not internal hypergraph arrows)
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.event_arrows.data()),
                    reinterpret_cast<Vertex*>(geo.event_arrows.data() + geo.event_arrows.size()));

                // Helper to route lines to opaque or transparent pipeline based on alpha
                auto add_lines = [&](const std::vector<RenderVertex>& verts) {
                    if (verts.empty()) return;
                    // Check alpha of first vertex to determine transparency
                    bool is_transparent = verts[0].a < 1.0f;
                    auto& target = is_transparent ? transparent_line_verts : line_verts;
                    target.insert(target.end(),
                        reinterpret_cast<const Vertex*>(verts.data()),
                        reinterpret_cast<const Vertex*>(verts.data() + verts.size()));
                };

                // Route each line type to appropriate pipeline based on alpha
                add_lines(geo.state_wireframe);
                add_lines(geo.event_lines);
                add_lines(geo.internal_edge_lines);
                add_lines(geo.causal_lines);
                add_lines(geo.branchial_lines);

                // Transparent (faces, bubbles)
                transparent_verts.insert(transparent_verts.end(),
                    reinterpret_cast<Vertex*>(geo.state_faces.data()),
                    reinterpret_cast<Vertex*>(geo.state_faces.data() + geo.state_faces.size()));
                transparent_verts.insert(transparent_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_bubbles.data()),
                    reinterpret_cast<Vertex*>(geo.internal_bubbles.data() + geo.internal_bubbles.size()));
            }

            // Upload to buffers
            if (triangle_buffer && !triangle_verts.empty()) {
                triangle_buffer->write(triangle_verts.data(),
                    std::min(triangle_verts.size() * sizeof(Vertex), buffer_size));
            }
            if (transparent_buffer && !transparent_verts.empty()) {
                transparent_buffer->write(transparent_verts.data(),
                    std::min(transparent_verts.size() * sizeof(Vertex), buffer_size));
            }
            if (line_buffer && !line_verts.empty()) {
                line_buffer->write(line_verts.data(),
                    std::min(line_verts.size() * sizeof(Vertex), buffer_size));
            }
            if (transparent_line_buffer && !transparent_line_verts.empty()) {
                transparent_line_buffer->write(transparent_line_verts.data(),
                    std::min(transparent_line_verts.size() * sizeof(Vertex), buffer_size));
            }

            // Upload instanced data
            if (instanced_sphere_available && sphere_instance_buffer && !sphere_instances.empty()) {
                size_t data_size = sphere_instances.size() * sizeof(SphereInstance);
                if (data_size <= sphere_instance_buffer_size) {
                    sphere_instance_buffer->write(sphere_instances.data(), data_size);
                }
            }
            if (instanced_cone_available && cone_instance_buffer && !cone_instances.empty()) {
                size_t data_size = cone_instances.size() * sizeof(ConeInstance);
                if (data_size <= cone_instance_buffer_size) {
                    cone_instance_buffer->write(cone_instances.data(), data_size);
                }
            }

            geometry_dirty = false;
        }

        fence->wait();
        fence->reset();

        auto acquire = swapchain->acquire_next_image(image_semaphore.get(), nullptr);
        if (!acquire.success) {
            device->wait_idle();
            swapchain->resize(window->get_width(), window->get_height());
            continue;
        }

        auto* tex = swapchain->get_texture(acquire.image_index);
        uint32_t w = tex->get_size().width;
        uint32_t h = tex->get_size().height;

        // Handle MSAA resource creation/recreation
        if (msaa_dirty) {
            device->wait_idle();
            create_msaa_resources(w, h);
            msaa_dirty = false;
        }

        math::mat4 vp = cam.get_view_projection_matrix();

        auto encoder = device->create_command_encoder();

        // Select pipelines based on MSAA state
        gal::RenderPipeline* active_triangle_pipeline = msaa_enabled && msaa_triangle_pipeline
            ? msaa_triangle_pipeline.get() : triangle_pipeline.get();
        gal::RenderPipeline* active_line_pipeline = msaa_enabled && msaa_line_pipeline
            ? msaa_line_pipeline.get() : line_pipeline.get();
        gal::RenderPipeline* active_transparent_pipeline = msaa_enabled && msaa_transparent_pipeline
            ? msaa_transparent_pipeline.get() : transparent_pipeline.get();
        gal::RenderPipeline* active_transparent_line_pipeline = msaa_enabled && msaa_transparent_line_pipeline
            ? msaa_transparent_line_pipeline.get() : transparent_line_pipeline.get();

        // Render pass
        gal::RenderPassColorAttachment color_att;
        if (msaa_enabled && msaa_texture) {
            // MSAA: render to MSAA texture, resolve to swapchain
            color_att.texture = msaa_texture.get();
            color_att.resolve_texture = tex;
        } else {
            // Non-MSAA: render directly to swapchain
            color_att.texture = tex;
            color_att.resolve_texture = nullptr;
        }
        color_att.load_op = gal::LoadOp::Clear;
        color_att.store_op = gal::StoreOp::Store;
        color_att.clear_color[0] = colors::BACKGROUND.x;
        color_att.clear_color[1] = colors::BACKGROUND.y;
        color_att.clear_color[2] = colors::BACKGROUND.z;
        color_att.clear_color[3] = colors::BACKGROUND.w;

        gal::RenderPassBeginInfo rp_info;
        rp_info.pipeline = active_triangle_pipeline;
        rp_info.color_attachments = &color_att;
        rp_info.color_attachment_count = 1;

        auto rp = encoder->begin_render_pass(rp_info);
        if (rp) {
            rp->set_viewport(0, 0, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f);
            rp->set_scissor(0, 0, w, h);

            // Opaque triangles
            if (!triangle_verts.empty()) {
                rp->set_pipeline(active_triangle_pipeline);
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, triangle_buffer.get());
                rp->draw(static_cast<uint32_t>(triangle_verts.size()), 1, 0, 0);
            }

            // Instanced spheres (opaque)
            if (instanced_sphere_available && !sphere_instances.empty() && sphere_mesh_buffer && sphere_instance_buffer) {
                // Use MSAA pipeline if MSAA is enabled and we have one
                gal::RenderPipeline* sphere_pipe = (msaa_enabled && msaa_instanced_sphere_pipeline)
                    ? msaa_instanced_sphere_pipeline.get() : instanced_sphere_pipeline.get();
                rp->set_pipeline(sphere_pipe);
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, sphere_mesh_buffer.get());
                rp->set_vertex_buffer(1, sphere_instance_buffer.get());
                rp->draw(static_cast<uint32_t>(unit_sphere.vertex_count()),
                         static_cast<uint32_t>(sphere_instances.size()), 0, 0);
            }

            // Instanced cones (opaque)
            if (instanced_cone_available && !cone_instances.empty() && cone_mesh_buffer && cone_instance_buffer) {
                // Use MSAA pipeline if MSAA is enabled and we have one
                gal::RenderPipeline* cone_pipe = (msaa_enabled && msaa_instanced_cone_pipeline)
                    ? msaa_instanced_cone_pipeline.get() : instanced_cone_pipeline.get();
                rp->set_pipeline(cone_pipe);
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, cone_mesh_buffer.get());
                rp->set_vertex_buffer(1, cone_instance_buffer.get());
                rp->draw(static_cast<uint32_t>(unit_cone.vertex_count()),
                         static_cast<uint32_t>(cone_instances.size()), 0, 0);
            }

            // Transparent
            if (!transparent_verts.empty()) {
                rp->set_pipeline(active_transparent_pipeline);
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, transparent_buffer.get());
                rp->draw(static_cast<uint32_t>(transparent_verts.size()), 1, 0, 0);
            }

            // Lines
            rp->set_pipeline(active_line_pipeline);
            rp->push_constants(vp.m, sizeof(math::mat4));

            // Debug axis (conditionally rendered)
            if (show_debug_axis) {
                rp->set_vertex_buffer(0, axis_buffer.get());
                rp->draw(static_cast<uint32_t>(axis_verts.size()), 1, 0, 0);
            }

            if (!line_verts.empty()) {
                rp->set_vertex_buffer(0, line_buffer.get());
                rp->draw(static_cast<uint32_t>(line_verts.size()), 1, 0, 0);
            }

            // Transparent lines (wireframes with alpha blending)
            if (!transparent_line_verts.empty()) {
                rp->set_pipeline(active_transparent_line_pipeline);
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, transparent_line_buffer.get());
                rp->draw(static_cast<uint32_t>(transparent_line_verts.size()), 1, 0, 0);
            }

            rp->end();
        }

        auto cmd = encoder->finish();
        device->submit(cmd.get(), image_semaphore.get(), render_semaphore.get(), fence.get());

        if (!swapchain->present(render_semaphore.get())) {
            device->wait_idle();
            swapchain->resize(window->get_width(), window->get_height());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    std::cout << "\nShutting down..." << std::endl;

#ifdef HYPERGRAPH_ENABLE_VISUALIZATION
    evolution_runner.stop();
#endif

    device->wait_idle();

    // Cleanup
    transparent_line_buffer.reset();
    line_buffer.reset();
    transparent_buffer.reset();
    triangle_buffer.reset();
    axis_buffer.reset();

    // Instanced rendering resources
    sphere_instance_buffer.reset();
    sphere_mesh_buffer.reset();
    cone_instance_buffer.reset();
    cone_mesh_buffer.reset();
    instanced_sphere_pipeline.reset();
    instanced_cone_pipeline.reset();
    instance_sphere_vertex_shader.reset();
    instance_cone_vertex_shader.reset();

    // MSAA resources
    msaa_instanced_sphere_pipeline.reset();
    msaa_instanced_cone_pipeline.reset();
    msaa_transparent_line_pipeline.reset();
    msaa_transparent_pipeline.reset();
    msaa_line_pipeline.reset();
    msaa_triangle_pipeline.reset();
    msaa_texture.reset();

    transparent_line_pipeline.reset();
    transparent_pipeline.reset();
    line_pipeline.reset();
    triangle_pipeline.reset();
    vertex_shader.reset();
    fragment_shader.reset();
    fence.reset();
    render_semaphore.reset();
    image_semaphore.reset();
    swapchain.reset();

    if (surface != VK_NULL_HANDLE) {
        gal::vk::vkDestroySurfaceKHR(vk_instance, surface, nullptr);
    }

    device.reset();
    gal::shutdown();

    return 0;
}
