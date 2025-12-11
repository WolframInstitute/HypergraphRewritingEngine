// Hypergraph rendering test application
// Tests the fundamental rendering elements: vertices, edges, hyperedges, self-loops, bubbles

#include <platform/window.hpp>
#include <gal/gal.hpp>
#include <gal/vulkan/vk_loader.hpp>
#include <camera/camera.hpp>
#include <math/types.hpp>
#include <layout/layout_engine.hpp>
#include <scene/hypergraph_data.hpp>
#include <scene/mock_evolution.hpp>
#include <scene/hypergraph_renderer.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>

using namespace viz;
using namespace viz::scene;

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

// Test modes - each demonstrates different visualization features
enum class TestMode {
    SimpleHypergraph,   // Basic hypergraph with normal edges
    SelfLoops,          // Hypergraph with self-loops
    EvolutionTree,      // States graph (tree structure)
    EvolutionDAG,       // States graph with canonicalization
    LargeHypergraph,    // Stress test
    COUNT
};

const char* test_mode_names[] = {
    "Simple Hypergraph",
    "Self-Loops Test",
    "Evolution Tree",
    "Evolution DAG",
    "Large Hypergraph"
};

// Detailed descriptions of what each test mode shows:
//
// MODE 1 - Simple Hypergraph:
//   - 7 vertices rendered as gray spheres
//   - 3 hyperedges: {0,1,2} (triangle), {2,3} (binary), {3,4,5,6} (chain)
//   - Each hyperedge is "near-line-graph": consecutive vertices connected by gray lines
//   - Translucent blue bubbles surround each hyperedge (convex hull + extrusion)
//   - Force-directed layout spreads vertices apart
//
// MODE 2 - Self-Loops Test:
//   - 4 vertices as spheres
//   - Hyperedges with self-loops: {0,1}, {1,1,2}, {2,2,2,3}, {3,3}
//   - Self-loops rendered with ORANGE arcs via virtual vertices
//   - Virtual vertex placed perpendicular to the edge direction
//   - You should see: normal gray edges + orange triangular arcs for loops
//
// MODE 3 - Evolution Tree:
//   - States graph from mock multiway evolution (depth=3, branching=2)
//   - 15 states rendered as WHITE CUBES (initial state) or GRAY CUBES
//   - 14 event edges as GRAY LINES connecting parent→child states
//   - Tree structure: root at center, children spread outward
//   - No canonicalization, so pure tree (no merging)
//
// MODE 4 - Evolution DAG:
//   - Similar to mode 3, but with ~30% canonicalization rate
//   - Some states map to same canonical representative
//   - Results in DAG structure (multiple paths can merge)
//   - Still 15 states as cubes, but edge structure may differ
//
// MODE 5 - Large Hypergraph:
//   - Stress test: 50 vertices, 80 random hyperedges
//   - Many overlapping spheres and edge lines
//   - Layout may not fully converge (stops at 500 iterations)
//   - Tests rendering performance with dense geometry

// Convert RenderVertex to our vertex format
struct Vertex {
    float x, y, z;
    float r, g, b, a;
};

// Create coordinate axes
std::vector<Vertex> create_axes(float length = 5.0f) {
    std::vector<Vertex> vertices;
    vertices.push_back({0, 0, 0, 1, 0, 0, 1});
    vertices.push_back({length, 0, 0, 1, 0, 0, 1});
    vertices.push_back({0, 0, 0, 0, 1, 0, 1});
    vertices.push_back({0, length, 0, 0, 1, 0, 1});
    vertices.push_back({0, 0, 0, 0, 0, 1, 1});
    vertices.push_back({0, 0, length, 0, 0, 1, 1});
    return vertices;
}

// Generate random layout for a hypergraph
HypergraphLayout generate_random_layout(const Hypergraph& hg, float spread = 3.0f) {
    HypergraphLayout layout;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-spread, spread);

    for (uint32_t i = 0; i < hg.vertex_count; ++i) {
        layout.vertex_positions.push_back({dist(rng), dist(rng), dist(rng)});
    }
    return layout;
}

// Layout hypergraph using force-directed algorithm
HypergraphLayout layout_hypergraph(const Hypergraph& hg) {
    // Convert to layout graph
    layout::LayoutGraph lg;

    // Add vertices - small initial spread to keep things visible
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (uint32_t i = 0; i < hg.vertex_count; ++i) {
        lg.add_vertex(dist(rng), dist(rng), dist(rng), 1.0f, false);
    }

    // Add edges from hyperedges - FULLY CONNECT unique vertices within each hyperedge
    // This keeps hyperedge vertices clustered together during layout
    for (const auto& edge : hg.edges) {
        // Get unique vertices in this hyperedge
        std::vector<VertexId> unique_verts;
        for (auto v : edge.vertices) {
            bool found = false;
            for (auto uv : unique_verts) {
                if (uv == v) { found = true; break; }
            }
            if (!found) unique_verts.push_back(v);
        }

        // Fully connect all unique vertices (clique)
        for (size_t i = 0; i < unique_verts.size(); ++i) {
            for (size_t j = i + 1; j < unique_verts.size(); ++j) {
                lg.add_edge(unique_verts[i], unique_verts[j], 1.5f, 1.0f);
            }
        }
    }

    // Run layout
    auto engine = layout::create_layout_engine(layout::LayoutBackend::Auto);
    if (engine) {
        engine->upload_graph(lg);

        layout::LayoutParams params;
        params.spring_constant = 1.0f;
        params.repulsion_constant = 0.5f;  // Lower repulsion to keep disconnected components closer
        params.damping = 0.9f;
        params.convergence_threshold = 0.01f;
        params.max_iterations = 200;

        engine->run_until_converged(params, [](uint32_t iter, float disp) {
            if (iter % 50 == 0) {
                std::cout << "  Layout iteration " << iter << ", disp=" << disp << std::endl;
            }
        });

        engine->download_positions(lg);
    }

    // Convert to HypergraphLayout
    HypergraphLayout layout;
    for (uint32_t i = 0; i < lg.vertex_count(); ++i) {
        layout.vertex_positions.push_back({
            lg.positions_x[i],
            lg.positions_y[i],
            lg.positions_z[i]
        });
    }
    return layout;
}

// Layout evolution graph (states)
EvolutionRenderer::EvolutionLayout layout_evolution(const Evolution& evo) {
    layout::LayoutGraph lg;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    // Add state vertices
    for (const auto& state : evo.states) {
        lg.add_vertex(dist(rng), dist(rng), dist(rng), 1.0f, state.is_initial);
    }

    // Add edges from events
    for (const auto& edge : evo.evolution_edges) {
        if (edge.type == EvolutionEdgeType::Event) {
            lg.add_edge(edge.source, edge.target, 2.0f, 1.0f);
        }
    }

    // Run layout
    auto engine = layout::create_layout_engine(layout::LayoutBackend::Auto);
    if (engine) {
        engine->upload_graph(lg);

        layout::LayoutParams params;
        params.spring_constant = 0.3f;
        params.repulsion_constant = 3.0f;
        params.damping = 0.9f;
        params.convergence_threshold = 0.01f;

        engine->run_until_converged(params, nullptr);
        engine->download_positions(lg);
    }

    EvolutionRenderer::EvolutionLayout layout;
    for (uint32_t i = 0; i < lg.vertex_count(); ++i) {
        layout.state_positions.push_back({
            lg.positions_x[i],
            lg.positions_y[i],
            lg.positions_z[i]
        });
    }
    return layout;
}

int main(int argc, char* argv[]) {
    // Allow specifying initial mode via command line: ./hypergraph_test [1-5]
    int initial_mode = 0;
    if (argc > 1) {
        initial_mode = std::atoi(argv[1]) - 1;
        if (initial_mode < 0 || initial_mode >= static_cast<int>(TestMode::COUNT)) {
            initial_mode = 0;
        }
    }

    std::cout << "Hypergraph Rendering Test" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Left-drag: Orbit camera" << std::endl;
    std::cout << "  Scroll: Zoom" << std::endl;
    std::cout << "  1-5: Switch test mode" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;
    std::cout << std::endl;

    // Create window
    platform::WindowDesc window_desc;
    window_desc.title = "Hypergraph Rendering Test";
    window_desc.width = 1280;
    window_desc.height = 720;

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
    device_desc.app_name = "HypergraphTest";
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
    auto vert_spirv = load_spirv("../shaders/spirv/basic3d.vert.spv");
    auto frag_spirv = load_spirv("../shaders/spirv/basic3d.frag.spv");

    // Instanced cone shaders (uses basic3d.frag for fragment stage)
    auto instance_cone_vert_spirv = load_spirv("../shaders/spirv/instance_cone.vert.spv");

    // Instanced sphere shaders (uses basic3d.frag for fragment stage)
    auto instance_sphere_vert_spirv = load_spirv("../shaders/spirv/instance_sphere.vert.spv");

    // WBOIT shaders
    auto wboit_vert_spirv = load_spirv("../shaders/spirv/wboit.vert.spv");
    auto wboit_frag_spirv = load_spirv("../shaders/spirv/wboit.frag.spv");
    auto composite_vert_spirv = load_spirv("../shaders/spirv/composite.vert.spv");
    auto composite_frag_spirv = load_spirv("../shaders/spirv/composite.frag.spv");

    if (vert_spirv.empty() || frag_spirv.empty()) {
        std::cerr << "Failed to load basic shaders" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    if (wboit_vert_spirv.empty() || wboit_frag_spirv.empty() ||
        composite_vert_spirv.empty() || composite_frag_spirv.empty()) {
        std::cerr << "Failed to load WBOIT shaders - falling back to basic transparency" << std::endl;
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

    // Create WBOIT shaders
    std::unique_ptr<gal::Shader> wboit_vertex_shader, wboit_fragment_shader;
    std::unique_ptr<gal::Shader> composite_vertex_shader, composite_fragment_shader;
    bool wboit_available = false;

    if (!wboit_vert_spirv.empty() && !wboit_frag_spirv.empty() &&
        !composite_vert_spirv.empty() && !composite_frag_spirv.empty()) {

        gal::ShaderDesc wboit_vert_desc;
        wboit_vert_desc.stage = gal::ShaderStage::Vertex;
        wboit_vert_desc.spirv_code = wboit_vert_spirv.data();
        wboit_vert_desc.spirv_size = wboit_vert_spirv.size() * sizeof(uint32_t);
        wboit_vertex_shader = device->create_shader(wboit_vert_desc);

        gal::ShaderDesc wboit_frag_desc;
        wboit_frag_desc.stage = gal::ShaderStage::Fragment;
        wboit_frag_desc.spirv_code = wboit_frag_spirv.data();
        wboit_frag_desc.spirv_size = wboit_frag_spirv.size() * sizeof(uint32_t);
        wboit_fragment_shader = device->create_shader(wboit_frag_desc);

        gal::ShaderDesc comp_vert_desc;
        comp_vert_desc.stage = gal::ShaderStage::Vertex;
        comp_vert_desc.spirv_code = composite_vert_spirv.data();
        comp_vert_desc.spirv_size = composite_vert_spirv.size() * sizeof(uint32_t);
        composite_vertex_shader = device->create_shader(comp_vert_desc);

        gal::ShaderDesc comp_frag_desc;
        comp_frag_desc.stage = gal::ShaderStage::Fragment;
        comp_frag_desc.spirv_code = composite_frag_spirv.data();
        comp_frag_desc.spirv_size = composite_frag_spirv.size() * sizeof(uint32_t);
        composite_fragment_shader = device->create_shader(comp_frag_desc);

        wboit_available = wboit_vertex_shader && wboit_fragment_shader &&
                          composite_vertex_shader && composite_fragment_shader;

        if (wboit_available) {
            std::cout << "WBOIT (Weighted Blended OIT) enabled" << std::endl;
        }
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

    // Opaque blend state (for spheres, cubes, arrows)
    gal::BlendState opaque_blend;
    opaque_blend.write_mask = 0xF;
    opaque_blend.blend_enable = false;

    // Standard alpha blend state (for bubbles) - not order-independent, but looks better for now
    // Will implement proper OIT later
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

    // Transparent triangle pipeline (for bubbles) - standard alpha blending for now
    pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
    pipeline_desc.rasterizer.cull_mode = gal::CullMode::None;  // Draw both sides of bubbles
    pipeline_desc.blend_states = &alpha_blend;
    auto transparent_pipeline = device->create_render_pipeline(pipeline_desc);

    if (!triangle_pipeline || !line_pipeline || !transparent_pipeline) {
        std::cerr << "Failed to create pipelines" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // ========== Instanced Cone Rendering Setup ==========
    std::unique_ptr<gal::Shader> instance_cone_vertex_shader;
    std::unique_ptr<gal::RenderPipeline> instanced_cone_pipeline;
    bool instanced_cone_available = false;

    // Generate unit cone mesh
    UnitConeMesh unit_cone;
    unit_cone.generate(8);  // 8 segments
    std::cout << "Unit cone mesh: " << unit_cone.vertex_count() << " vertices, "
              << unit_cone.byte_size() << " bytes" << std::endl;

    if (!instance_cone_vert_spirv.empty()) {
        gal::ShaderDesc cone_vert_desc;
        cone_vert_desc.stage = gal::ShaderStage::Vertex;
        cone_vert_desc.spirv_code = instance_cone_vert_spirv.data();
        cone_vert_desc.spirv_size = instance_cone_vert_spirv.size() * sizeof(uint32_t);
        cone_vert_desc.debug_name = "instance_cone_vert";
        instance_cone_vertex_shader = device->create_shader(cone_vert_desc);

        if (instance_cone_vertex_shader) {
            // Vertex layout for unit cone mesh (binding 0, per-vertex)
            // ConeVertex: {float x, y, z, nx, ny, nz}
            gal::VertexAttribute cone_attribs[] = {
                {0, gal::Format::RGB32_FLOAT, 0},                    // position
                {1, gal::Format::RGB32_FLOAT, sizeof(float) * 3},    // normal
            };
            gal::VertexBufferLayout cone_vertex_layout;
            cone_vertex_layout.stride = sizeof(ConeVertex);
            cone_vertex_layout.step_mode = gal::VertexStepMode::Vertex;
            cone_vertex_layout.attributes = cone_attribs;
            cone_vertex_layout.attribute_count = 2;

            // Instance layout (binding 1, per-instance)
            // ConeInstance: {tip(3), dir(3), length, radius, color(4)} = 12 floats
            gal::VertexAttribute instance_attribs[] = {
                {2, gal::Format::RGB32_FLOAT, 0},                     // tip position
                {3, gal::Format::RGB32_FLOAT, sizeof(float) * 3},     // direction
                {4, gal::Format::RG32_FLOAT, sizeof(float) * 6},      // length, radius
                {5, gal::Format::RGBA32_FLOAT, sizeof(float) * 8},    // color
            };
            gal::VertexBufferLayout instance_layout;
            instance_layout.stride = sizeof(ConeInstance);
            instance_layout.step_mode = gal::VertexStepMode::Instance;
            instance_layout.attributes = instance_attribs;
            instance_layout.attribute_count = 4;

            gal::VertexBufferLayout cone_layouts[] = {cone_vertex_layout, instance_layout};

            gal::RenderPipelineDesc cone_pipeline_desc;
            cone_pipeline_desc.vertex_shader = instance_cone_vertex_shader.get();
            cone_pipeline_desc.fragment_shader = fragment_shader.get();  // Reuse basic3d.frag
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

            if (instanced_cone_available) {
                std::cout << "Instanced cone rendering enabled" << std::endl;
            } else {
                std::cerr << "Failed to create instanced cone pipeline" << std::endl;
            }
        }
    }

    // ========== Instanced Sphere Rendering Setup ==========
    std::unique_ptr<gal::Shader> instance_sphere_vertex_shader;
    std::unique_ptr<gal::RenderPipeline> instanced_sphere_pipeline;
    bool instanced_sphere_available = false;

    // Generate unit sphere mesh
    UnitSphereMesh unit_sphere;
    unit_sphere.generate(12, 8);  // 12 segments, 8 rings
    std::cout << "Unit sphere mesh: " << unit_sphere.vertex_count() << " vertices, "
              << unit_sphere.byte_size() << " bytes" << std::endl;

    if (!instance_sphere_vert_spirv.empty()) {
        gal::ShaderDesc sphere_vert_desc;
        sphere_vert_desc.stage = gal::ShaderStage::Vertex;
        sphere_vert_desc.spirv_code = instance_sphere_vert_spirv.data();
        sphere_vert_desc.spirv_size = instance_sphere_vert_spirv.size() * sizeof(uint32_t);
        sphere_vert_desc.debug_name = "instance_sphere_vert";
        instance_sphere_vertex_shader = device->create_shader(sphere_vert_desc);

        if (instance_sphere_vertex_shader) {
            // Vertex layout for unit sphere mesh (binding 0, per-vertex)
            // SphereVertex: {float x, y, z, nx, ny, nz}
            gal::VertexAttribute sphere_attribs[] = {
                {0, gal::Format::RGB32_FLOAT, 0},                    // position
                {1, gal::Format::RGB32_FLOAT, sizeof(float) * 3},    // normal
            };
            gal::VertexBufferLayout sphere_vertex_layout;
            sphere_vertex_layout.stride = sizeof(SphereVertex);
            sphere_vertex_layout.step_mode = gal::VertexStepMode::Vertex;
            sphere_vertex_layout.attributes = sphere_attribs;
            sphere_vertex_layout.attribute_count = 2;

            // Instance layout (binding 1, per-instance)
            // SphereInstance: {center(3), radius, color(4)} = 8 floats
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
            sphere_pipeline_desc.fragment_shader = fragment_shader.get();  // Reuse basic3d.frag
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

            if (instanced_sphere_available) {
                std::cout << "Instanced sphere rendering enabled" << std::endl;
            } else {
                std::cerr << "Failed to create instanced sphere pipeline" << std::endl;
            }
        }
    }

    // ========== WBOIT Setup ==========
    // OIT render target textures - created/resized on demand
    std::unique_ptr<gal::Texture> oit_accum_texture;
    std::unique_ptr<gal::Texture> oit_reveal_texture;
    std::unique_ptr<gal::Sampler> oit_sampler;
    std::unique_ptr<gal::BindGroupLayout> composite_bind_group_layout;
    std::unique_ptr<gal::BindGroup> composite_bind_group;
    std::unique_ptr<gal::RenderPipeline> wboit_pipeline;
    std::unique_ptr<gal::RenderPipeline> wboit_line_pipeline;
    std::unique_ptr<gal::RenderPipeline> composite_pipeline;
    uint32_t oit_width = 0, oit_height = 0;

    // Create bind group layout and sampler upfront (needed for pipeline creation)
    if (wboit_available) {
        gal::SamplerDesc sampler_desc;
        sampler_desc.mag_filter = gal::Filter::Nearest;
        sampler_desc.min_filter = gal::Filter::Nearest;
        sampler_desc.address_u = gal::AddressMode::ClampToEdge;
        sampler_desc.address_v = gal::AddressMode::ClampToEdge;
        oit_sampler = device->create_sampler(sampler_desc);

        if (!oit_sampler) {
            std::cerr << "Failed to create OIT sampler" << std::endl;
            wboit_available = false;
        } else {
            gal::BindGroupLayoutEntry layout_entries[] = {
                {0, gal::ShaderStage::Fragment, gal::BindingType::CombinedTextureSampler, 1},  // accum
                {1, gal::ShaderStage::Fragment, gal::BindingType::CombinedTextureSampler, 1},  // reveal
            };
            gal::BindGroupLayoutDesc layout_desc;
            layout_desc.entries = layout_entries;
            layout_desc.entry_count = 2;
            layout_desc.debug_name = "CompositeBindGroupLayout";
            composite_bind_group_layout = device->create_bind_group_layout(layout_desc);

            if (!composite_bind_group_layout) {
                std::cerr << "Failed to create composite bind group layout" << std::endl;
                wboit_available = false;
            }
        }
    }

    // Function to create/resize OIT resources (textures and bind group)
    auto create_oit_resources = [&](uint32_t width, uint32_t height) {
        if (!wboit_available || (width == oit_width && height == oit_height)) return;

        oit_width = width;
        oit_height = height;

        // Create accumulation texture (RGBA16F)
        gal::TextureDesc accum_desc;
        accum_desc.size = {width, height, 1};
        accum_desc.format = gal::Format::RGBA16_FLOAT;
        accum_desc.usage = gal::TextureUsage::RenderTarget | gal::TextureUsage::Sampled;
        accum_desc.debug_name = "OIT_Accum";
        oit_accum_texture = device->create_texture(accum_desc);

        // Create revealage texture (R8)
        gal::TextureDesc reveal_desc;
        reveal_desc.size = {width, height, 1};
        reveal_desc.format = gal::Format::R8_UNORM;
        reveal_desc.usage = gal::TextureUsage::RenderTarget | gal::TextureUsage::Sampled;
        reveal_desc.debug_name = "OIT_Reveal";
        oit_reveal_texture = device->create_texture(reveal_desc);

        if (!oit_accum_texture || !oit_reveal_texture) {
            std::cerr << "Failed to create OIT textures" << std::endl;
            wboit_available = false;
            return;
        }

        // Create/recreate bind group with new textures
        gal::BindGroupEntry bind_entries[] = {
            {0, nullptr, 0, 0, oit_accum_texture.get(), oit_sampler.get()},
            {1, nullptr, 0, 0, oit_reveal_texture.get(), oit_sampler.get()},
        };
        gal::BindGroupDesc bind_desc;
        bind_desc.layout = composite_bind_group_layout.get();
        bind_desc.entries = bind_entries;
        bind_desc.entry_count = 2;
        bind_desc.debug_name = "CompositeBindGroup";
        composite_bind_group = device->create_bind_group(bind_desc);

        std::cout << "OIT resources created: " << width << "x" << height << std::endl;
    };

    // Create WBOIT and composite pipelines (only if shaders available)
    if (wboit_available) {
        // WBOIT pipeline - renders transparent geometry to accum+reveal textures
        // Uses two color attachments with specific blend modes

        // Blend state for accumulation: additive (ONE, ONE)
        gal::BlendState accum_blend;
        accum_blend.blend_enable = true;
        accum_blend.src_color = gal::BlendFactor::One;
        accum_blend.dst_color = gal::BlendFactor::One;
        accum_blend.color_op = gal::BlendOp::Add;
        accum_blend.src_alpha = gal::BlendFactor::One;
        accum_blend.dst_alpha = gal::BlendFactor::One;
        accum_blend.alpha_op = gal::BlendOp::Add;
        accum_blend.write_mask = 0xF;

        // Blend state for revealage: dst = dst * src = product of (1-alpha)
        // Shader outputs (1-alpha), so multiply dst by src directly
        // result = src * 0 + dst * src = dst * (1-alpha)
        gal::BlendState reveal_blend;
        reveal_blend.blend_enable = true;
        reveal_blend.src_color = gal::BlendFactor::Zero;
        reveal_blend.dst_color = gal::BlendFactor::SrcColor;  // Multiply dst by src
        reveal_blend.color_op = gal::BlendOp::Add;
        reveal_blend.src_alpha = gal::BlendFactor::Zero;
        reveal_blend.dst_alpha = gal::BlendFactor::SrcColor;
        reveal_blend.alpha_op = gal::BlendOp::Add;
        reveal_blend.write_mask = 0x1;  // Only R channel

        gal::BlendState wboit_blend_states[] = {accum_blend, reveal_blend};
        gal::Format wboit_formats[] = {gal::Format::RGBA16_FLOAT, gal::Format::R8_UNORM};

        gal::RenderPipelineDesc wboit_desc;
        wboit_desc.vertex_shader = wboit_vertex_shader.get();
        wboit_desc.fragment_shader = wboit_fragment_shader.get();
        wboit_desc.vertex_layouts = &vertex_layout;
        wboit_desc.vertex_layout_count = 1;
        wboit_desc.topology = gal::PrimitiveTopology::TriangleList;
        wboit_desc.rasterizer.cull_mode = gal::CullMode::None;  // Draw both sides
        wboit_desc.depth_stencil.depth_test_enable = false;  // No depth test for OIT
        wboit_desc.depth_stencil.depth_write_enable = false;
        wboit_desc.blend_states = wboit_blend_states;
        wboit_desc.blend_state_count = 2;
        wboit_desc.color_formats = wboit_formats;
        wboit_desc.color_format_count = 2;
        wboit_desc.push_constant_size = sizeof(math::mat4);
        wboit_desc.render_to_texture = true;  // OIT textures will be sampled
        wboit_desc.debug_name = "WBOIT_Pipeline";

        wboit_pipeline = device->create_render_pipeline(wboit_desc);

        // WBOIT line pipeline for transparent lines
        wboit_desc.topology = gal::PrimitiveTopology::LineList;
        wboit_desc.debug_name = "WBOIT_Line_Pipeline";
        wboit_line_pipeline = device->create_render_pipeline(wboit_desc);

        // Composite pipeline - fullscreen triangle that reads accum+reveal and outputs to swapchain
        gal::BlendState composite_blend = gal::BlendState::alpha_blend();

        const gal::BindGroupLayout* composite_layouts[] = {composite_bind_group_layout.get()};

        gal::RenderPipelineDesc composite_desc;
        composite_desc.vertex_shader = composite_vertex_shader.get();
        composite_desc.fragment_shader = composite_fragment_shader.get();
        composite_desc.vertex_layouts = nullptr;  // No vertex input (fullscreen triangle)
        composite_desc.vertex_layout_count = 0;
        composite_desc.topology = gal::PrimitiveTopology::TriangleList;
        composite_desc.rasterizer.cull_mode = gal::CullMode::None;
        composite_desc.depth_stencil.depth_test_enable = false;
        composite_desc.blend_states = &composite_blend;
        composite_desc.blend_state_count = 1;
        composite_desc.color_formats = &color_format;
        composite_desc.color_format_count = 1;
        composite_desc.bind_group_layouts = composite_layouts;
        composite_desc.bind_group_layout_count = 1;
        composite_desc.debug_name = "Composite_Pipeline";

        composite_pipeline = device->create_render_pipeline(composite_desc);

        if (!wboit_pipeline || !wboit_line_pipeline || !composite_pipeline) {
            std::cerr << "Failed to create WBOIT pipelines - falling back to basic transparency" << std::endl;
            wboit_available = false;
        }
    }

    // Initialize renderers
    HypergraphRenderer hg_renderer;
    EvolutionRenderer evo_renderer;
    MockEvolutionGenerator mock_gen;

    // Current test mode (can be set via command line)
    TestMode current_mode = static_cast<TestMode>(initial_mode);

    // Generated geometry buffers
    std::vector<Vertex> triangle_verts;       // Opaque triangles (spheres, cubes, arrows)
    std::vector<Vertex> transparent_verts;    // Translucent triangles (bubbles)
    std::vector<Vertex> line_verts;
    std::vector<ConeInstance> cone_instances; // Cone instances (for instanced rendering)
    std::vector<SphereInstance> sphere_instances; // Sphere instances (for instanced rendering)
    auto axis_verts = create_axes(5.0f);

    // Function to regenerate geometry for current test mode
    auto regenerate_geometry = [&]() {
        triangle_verts.clear();
        transparent_verts.clear();
        line_verts.clear();
        cone_instances.clear();
        sphere_instances.clear();

        std::cout << "\nGenerating: " << test_mode_names[static_cast<int>(current_mode)] << std::endl;

        switch (current_mode) {
            case TestMode::SimpleHypergraph: {
                // Simple chain of hyperedges with increasing arity
                // Each edge shares ONE vertex with the next edge
                // This clearly demonstrates hyperedge structure
                Hypergraph hg;

                // Chain: 2-edge -> 3-edge -> 4-edge -> 5-edge
                // Shared vertices: 1 connects e0-e1, 3 connects e1-e2, 6 connects e2-e3
                //
                // Vertices: 0,1 | 1,2,3 | 3,4,5,6 | 6,7,8,9,10
                //           ^^^   ^^^^^   ^^^^^^^   ^^^^^^^^^
                //           2-edge 3-edge  4-edge    5-edge

                hg.add_edge({0, 1});             // e0: 2-edge (binary)
                hg.add_edge({1, 2, 3});          // e1: 3-edge (shares v1 with e0)
                hg.add_edge({3, 4, 5, 6});       // e2: 4-edge (shares v3 with e1)
                hg.add_edge({6, 7, 8, 9, 10});   // e3: 5-edge (shares v6 with e2)

                std::cout << "  Laying out connected hypergraph..." << std::endl;
                auto layout = layout_hypergraph(hg);

                // Graph coloring: greedy coloring based on shared vertices
                // Two edges sharing a vertex should have different colors
                std::vector<int> edge_colors(hg.edges.size(), -1);

                // Palette of 8 distinct colors (low alpha for transparency)
                math::vec4 palette[] = {
                    {0.9f, 0.2f, 0.2f, 0.15f},   // 0: Red
                    {0.2f, 0.8f, 0.2f, 0.15f},   // 1: Green
                    {0.2f, 0.4f, 0.9f, 0.15f},   // 2: Blue
                    {0.9f, 0.9f, 0.2f, 0.15f},   // 3: Yellow
                    {0.9f, 0.2f, 0.9f, 0.15f},   // 4: Magenta
                    {0.2f, 0.9f, 0.9f, 0.15f},   // 5: Cyan
                    {1.0f, 0.5f, 0.1f, 0.15f},   // 6: Orange
                    {0.6f, 0.3f, 0.9f, 0.15f},   // 7: Purple
                };
                int num_colors = 8;

                // Greedy graph coloring
                for (size_t e = 0; e < hg.edges.size(); ++e) {
                    // Find colors used by adjacent edges (sharing vertices)
                    std::vector<bool> used(num_colors, false);
                    for (size_t other = 0; other < e; ++other) {
                        if (edge_colors[other] >= 0) {
                            // Check if edges share a vertex
                            bool share = false;
                            for (auto v1 : hg.edges[e].vertices) {
                                for (auto v2 : hg.edges[other].vertices) {
                                    if (v1 == v2) { share = true; break; }
                                }
                                if (share) break;
                            }
                            if (share) {
                                used[edge_colors[other]] = true;
                            }
                        }
                    }
                    // Pick first unused color
                    for (int c = 0; c < num_colors; ++c) {
                        if (!used[c]) {
                            edge_colors[e] = c;
                            break;
                        }
                    }
                    if (edge_colors[e] < 0) edge_colors[e] = 0; // fallback
                }

                // Generate geometry
                HypergraphGeometry geo;

                // Vertex spheres (bright white, larger so they're visible)
                for (VertexId v = 0; v < hg.vertex_count; ++v) {
                    math::vec3 pos = layout.get_vertex_pos(v);
                    hg_renderer.generate_sphere(geo.vertex_triangles, pos, 0.08f,
                                               {1.0f, 1.0f, 1.0f, 1.0f});
                }

                // Generate each hyperedge with its assigned color
                std::cout << "  Edge colors: ";
                for (size_t e = 0; e < hg.edges.size(); ++e) {
                    const auto& edge = hg.edges[e];
                    std::vector<math::vec3> positions;
                    for (auto v : edge.vertices) {
                        positions.push_back(layout.get_vertex_pos(v));
                    }

                    math::vec4 color = palette[edge_colors[e]];

                    // Slightly brighter color for lines (opaque)
                    math::vec4 line_color = {
                        std::min(1.0f, color.x + 0.3f),
                        std::min(1.0f, color.y + 0.3f),
                        std::min(1.0f, color.z + 0.3f),
                        1.0f
                    };

                    // Arrow color (darker, opaque)
                    math::vec4 arrow_color = {
                        color.x * 0.8f,
                        color.y * 0.8f,
                        color.z * 0.8f,
                        1.0f
                    };

                    // Edge lines and arrowheads showing vertex ordering
                    for (size_t i = 0; i + 1 < edge.vertices.size(); ++i) {
                        if (edge.vertices[i] != edge.vertices[i + 1]) {
                            math::vec3 p1 = positions[i];
                            math::vec3 p2 = positions[i + 1];

                            // Line segment
                            hg_renderer.generate_line(geo.edge_lines, p1, p2, line_color);

                            // Arrowhead (cone) pointing toward p2, showing vertex ordering
                            // Tip placed exactly at the target vertex sphere surface
                            math::vec3 dir = p2 - p1;
                            float dist = math::length(dir);
                            float vertex_radius = 0.08f;  // Must match sphere radius used above
                            float arrow_length = 0.12f;
                            float arrow_radius = 0.05f;
                            // TODO: Cone angle should be reduced when vertex has many edges
                            // to prevent overlapping. Calculate max half-angle based on
                            // vertex degree: theta = asin(min(1, 2*pi/(N*k))) for some k
                            // See PLAN.md for details.

                            if (dist > vertex_radius + arrow_length + 0.05f) {
                                // Arrow tip exactly touches target vertex sphere
                                math::vec3 arrow_tip = p2 - math::normalize(dir) * vertex_radius;
                                // Use instanced version to populate both legacy triangles and instances
                                hg_renderer.generate_cone_instanced(geo, arrow_tip, dir,
                                                         arrow_length, arrow_radius, arrow_color);
                            }
                        }
                    }

                    // Bubble
                    std::vector<math::vec3> unique_positions;
                    for (const auto& p : positions) {
                        bool dup = false;
                        for (const auto& up : unique_positions) {
                            if (math::length(p - up) < 0.001f) { dup = true; break; }
                        }
                        if (!dup) unique_positions.push_back(p);
                    }
                    hg_renderer.generate_bubble_for_arity(geo.bubble_triangles, unique_positions, color);

                    std::cout << "e" << e << "=" << edge_colors[e] << " ";
                }
                std::cout << std::endl;

                // Copy to render buffers
                // Opaque: vertex spheres
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.vertex_triangles.data()),
                    reinterpret_cast<Vertex*>(geo.vertex_triangles.data() + geo.vertex_triangles.size()));
                // Arrow cones: only add legacy triangles if NOT using instanced rendering
                if (!instanced_cone_available) {
                    triangle_verts.insert(triangle_verts.end(),
                        reinterpret_cast<Vertex*>(geo.arrow_triangles.data()),
                        reinterpret_cast<Vertex*>(geo.arrow_triangles.data() + geo.arrow_triangles.size()));
                }

                // Transparent: bubbles
                transparent_verts.insert(transparent_verts.end(),
                    reinterpret_cast<Vertex*>(geo.bubble_triangles.data()),
                    reinterpret_cast<Vertex*>(geo.bubble_triangles.data() + geo.bubble_triangles.size()));

                // Lines
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.edge_lines.data()),
                    reinterpret_cast<Vertex*>(geo.edge_lines.data() + geo.edge_lines.size()));

                // Cone instances (for instanced rendering)
                cone_instances.insert(cone_instances.end(),
                    geo.cone_instances.begin(), geo.cone_instances.end());

                // Sphere instances (for instanced rendering)
                sphere_instances.insert(sphere_instances.end(),
                    geo.sphere_instances.begin(), geo.sphere_instances.end());

                std::cout << "  Vertices: " << hg.vertex_count << ", Hyperedges: " << hg.edges.size() << std::endl;
                std::cout << "  Colors: Red=0, Green=1, Blue=2, Yellow=3, Magenta=4, Cyan=5, Orange=6, Purple=7" << std::endl;
                std::cout << "  Arrows: " << geo.arrow_triangles.size() / 3 << ", Bubble tris: " << geo.bubble_triangles.size() / 3 << std::endl;
                std::cout << "  Cone instances: " << geo.cone_instances.size() << ", Sphere instances: " << geo.sphere_instances.size() << std::endl;
                break;
            }

            case TestMode::SelfLoops: {
                auto evo = mock_gen.generate_self_loop_test();
                auto& hg = evo.states[0].hypergraph;

                std::cout << "  Self-loop test hyperedges:" << std::endl;
                for (size_t i = 0; i < hg.edges.size(); ++i) {
                    std::cout << "    Edge " << i << ": {";
                    for (size_t j = 0; j < hg.edges[i].vertices.size(); ++j) {
                        if (j > 0) std::cout << ",";
                        std::cout << hg.edges[i].vertices[j];
                    }
                    std::cout << "}" << std::endl;
                }

                std::cout << "  Laying out self-loop test..." << std::endl;
                auto layout = layout_hypergraph(hg);
                auto geo = hg_renderer.generate(hg, layout);

                // Opaque: spheres + arrows
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.vertex_triangles.data()),
                    reinterpret_cast<Vertex*>(geo.vertex_triangles.data() + geo.vertex_triangles.size()));
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.arrow_triangles.data()),
                    reinterpret_cast<Vertex*>(geo.arrow_triangles.data() + geo.arrow_triangles.size()));

                // DEBUG: Skip bubbles to see the structure clearly
                // Once OIT is implemented, re-enable this
                // transparent_verts.insert(transparent_verts.end(),
                //     reinterpret_cast<Vertex*>(geo.bubble_triangles.data()),
                //     reinterpret_cast<Vertex*>(geo.bubble_triangles.data() + geo.bubble_triangles.size()));

                // Lines (edges + self-loops)
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.edge_lines.data()),
                    reinterpret_cast<Vertex*>(geo.edge_lines.data() + geo.edge_lines.size()));
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.self_loop_lines.data()),
                    reinterpret_cast<Vertex*>(geo.self_loop_lines.data() + geo.self_loop_lines.size()));

                std::cout << "  Vertices: " << hg.vertex_count << std::endl;
                std::cout << "  Vertex spheres: " << geo.vertex_triangles.size() / 3 << " tris" << std::endl;
                std::cout << "  Edge lines: " << geo.edge_lines.size() / 2 << std::endl;
                std::cout << "  Self-loop lines: " << geo.self_loop_lines.size() / 2 << std::endl;
                std::cout << "  Arrow tris: " << geo.arrow_triangles.size() / 3 << std::endl;
                std::cout << "  Bubble tris: " << geo.bubble_triangles.size() / 3 << " (DISABLED for debug)" << std::endl;
                break;
            }

            case TestMode::EvolutionTree: {
                auto evo = mock_gen.generate_tree(3, 2);

                std::cout << "  Laying out evolution tree..." << std::endl;
                auto layout = layout_evolution(evo);
                auto geo = evo_renderer.generate_states_graph(evo, layout);

                // Opaque: internal vertex spheres, internal arrows, and event arrows
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_vertex_spheres.data()),
                    reinterpret_cast<Vertex*>(geo.internal_vertex_spheres.data() + geo.internal_vertex_spheres.size()));
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_arrows.data()),
                    reinterpret_cast<Vertex*>(geo.internal_arrows.data() + geo.internal_arrows.size()));
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.event_arrows.data()),
                    reinterpret_cast<Vertex*>(geo.event_arrows.data() + geo.event_arrows.size()));

                // Lines: cube wireframe + internal edges + event edges
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.state_wireframe.data()),
                    reinterpret_cast<Vertex*>(geo.state_wireframe.data() + geo.state_wireframe.size()));
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_edge_lines.data()),
                    reinterpret_cast<Vertex*>(geo.internal_edge_lines.data() + geo.internal_edge_lines.size()));
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.event_lines.data()),
                    reinterpret_cast<Vertex*>(geo.event_lines.data() + geo.event_lines.size()));
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.branchial_lines.data()),
                    reinterpret_cast<Vertex*>(geo.branchial_lines.data() + geo.branchial_lines.size()));

                // Translucent: cube faces + internal bubbles
                transparent_verts.insert(transparent_verts.end(),
                    reinterpret_cast<Vertex*>(geo.state_faces.data()),
                    reinterpret_cast<Vertex*>(geo.state_faces.data() + geo.state_faces.size()));
                transparent_verts.insert(transparent_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_bubbles.data()),
                    reinterpret_cast<Vertex*>(geo.internal_bubbles.data() + geo.internal_bubbles.size()));

                std::cout << "  States: " << evo.states.size() << ", Events: " << evo.events.size() << std::endl;
                std::cout << "  Internal spheres: " << geo.internal_vertex_spheres.size() / 3
                          << ", Event arrows: " << geo.event_arrows.size() / 3 << std::endl;
                break;
            }

            case TestMode::EvolutionDAG: {
                auto evo = mock_gen.generate_dag(15, 0.3f);

                std::cout << "  Laying out evolution DAG..." << std::endl;
                auto layout = layout_evolution(evo);
                auto geo = evo_renderer.generate_states_graph(evo, layout);

                // Opaque: internal vertex spheres, internal arrows, and event arrows
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_vertex_spheres.data()),
                    reinterpret_cast<Vertex*>(geo.internal_vertex_spheres.data() + geo.internal_vertex_spheres.size()));
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_arrows.data()),
                    reinterpret_cast<Vertex*>(geo.internal_arrows.data() + geo.internal_arrows.size()));
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.event_arrows.data()),
                    reinterpret_cast<Vertex*>(geo.event_arrows.data() + geo.event_arrows.size()));

                // Lines: cube wireframe + internal edges + event edges
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.state_wireframe.data()),
                    reinterpret_cast<Vertex*>(geo.state_wireframe.data() + geo.state_wireframe.size()));
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_edge_lines.data()),
                    reinterpret_cast<Vertex*>(geo.internal_edge_lines.data() + geo.internal_edge_lines.size()));
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.event_lines.data()),
                    reinterpret_cast<Vertex*>(geo.event_lines.data() + geo.event_lines.size()));

                // Translucent: cube faces + internal bubbles
                transparent_verts.insert(transparent_verts.end(),
                    reinterpret_cast<Vertex*>(geo.state_faces.data()),
                    reinterpret_cast<Vertex*>(geo.state_faces.data() + geo.state_faces.size()));
                transparent_verts.insert(transparent_verts.end(),
                    reinterpret_cast<Vertex*>(geo.internal_bubbles.data()),
                    reinterpret_cast<Vertex*>(geo.internal_bubbles.data() + geo.internal_bubbles.size()));

                std::cout << "  States: " << evo.states.size() << " (with canonicalization)" << std::endl;
                std::cout << "  Event arrows: " << geo.event_arrows.size() / 3 << std::endl;
                break;
            }

            case TestMode::LargeHypergraph: {
                auto evo = mock_gen.generate_large_hypergraph(50, 80);
                auto& hg = evo.states[0].hypergraph;

                std::cout << "  Laying out large hypergraph..." << std::endl;
                auto layout = layout_hypergraph(hg);
                auto geo = hg_renderer.generate(hg, layout);

                // Opaque: spheres + arrows
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.vertex_triangles.data()),
                    reinterpret_cast<Vertex*>(geo.vertex_triangles.data() + geo.vertex_triangles.size()));
                triangle_verts.insert(triangle_verts.end(),
                    reinterpret_cast<Vertex*>(geo.arrow_triangles.data()),
                    reinterpret_cast<Vertex*>(geo.arrow_triangles.data() + geo.arrow_triangles.size()));

                // Transparent: bubbles
                transparent_verts.insert(transparent_verts.end(),
                    reinterpret_cast<Vertex*>(geo.bubble_triangles.data()),
                    reinterpret_cast<Vertex*>(geo.bubble_triangles.data() + geo.bubble_triangles.size()));

                // Lines
                line_verts.insert(line_verts.end(),
                    reinterpret_cast<Vertex*>(geo.edge_lines.data()),
                    reinterpret_cast<Vertex*>(geo.edge_lines.data() + geo.edge_lines.size()));

                std::cout << "  Vertices: " << hg.vertex_count << ", Edges: " << hg.edges.size() << std::endl;
                break;
            }

            default:
                break;
        }

        std::cout << "  Opaque tris: " << triangle_verts.size() / 3
                  << ", Transparent tris: " << transparent_verts.size() / 3
                  << ", Lines: " << line_verts.size() / 2 << std::endl;
    };

    // Initial geometry generation
    regenerate_geometry();

    // Dynamic buffer sizes - will grow as needed
    size_t triangle_buffer_size = 200000 * sizeof(Vertex);  // Initial ~5.6MB
    size_t transparent_buffer_size = 200000 * sizeof(Vertex);
    size_t line_buffer_size = 50000 * sizeof(Vertex);       // Initial ~1.4MB

    // Helper to create a vertex buffer
    auto create_vertex_buffer = [&](size_t size) {
        gal::BufferDesc desc;
        desc.size = size;
        desc.usage = gal::BufferUsage::Vertex;
        desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        desc.initial_data = nullptr;
        return device->create_buffer(desc);
    };

    // Helper to ensure buffer is large enough, resize if needed
    // Returns true if buffer was resized (caller should update vertex buffer binding)
    auto ensure_buffer_size = [&](std::unique_ptr<gal::Buffer>& buffer, size_t& current_size,
                                   size_t required_size, const char* name) -> bool {
        if (required_size <= current_size) {
            return false;  // No resize needed
        }

        // Calculate new size with 50% headroom to avoid frequent resizes
        size_t new_size = required_size + required_size / 2;
        new_size = ((new_size + 65535) / 65536) * 65536;  // Round up to 64KB

        std::cout << "Resizing " << name << " buffer: " << current_size / 1024 << "KB -> "
                  << new_size / 1024 << "KB (needed " << required_size / 1024 << "KB)" << std::endl;

        // Wait for GPU to finish using old buffer
        device->wait_idle();

        // Release old buffer (destructor called when unique_ptr reset)
        buffer.reset();

        // Create new larger buffer
        buffer = create_vertex_buffer(new_size);
        if (!buffer) {
            std::cerr << "Failed to create resized " << name << " buffer!" << std::endl;
            return false;
        }

        current_size = new_size;
        return true;
    };

    // Create initial buffers
    auto axis_buffer = create_vertex_buffer(axis_verts.size() * sizeof(Vertex));
    if (axis_buffer && !axis_verts.empty()) {
        axis_buffer->write(axis_verts.data(), axis_verts.size() * sizeof(Vertex));
    }

    auto triangle_buffer = create_vertex_buffer(triangle_buffer_size);
    auto transparent_buffer = create_vertex_buffer(transparent_buffer_size);
    auto line_buffer = create_vertex_buffer(line_buffer_size);

    // Instanced cone buffers
    std::unique_ptr<gal::Buffer> cone_mesh_buffer;
    std::unique_ptr<gal::Buffer> cone_instance_buffer;
    size_t cone_instance_buffer_size = 10000 * sizeof(ConeInstance);  // Initial capacity for 10k cones

    if (instanced_cone_available) {
        // Static unit cone mesh buffer
        gal::BufferDesc cone_mesh_desc;
        cone_mesh_desc.size = unit_cone.byte_size();
        cone_mesh_desc.usage = gal::BufferUsage::Vertex;
        cone_mesh_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        cone_mesh_desc.initial_data = unit_cone.data();
        cone_mesh_desc.debug_name = "unit_cone_mesh";
        cone_mesh_buffer = device->create_buffer(cone_mesh_desc);

        // Dynamic instance buffer
        cone_instance_buffer = create_vertex_buffer(cone_instance_buffer_size);

        if (!cone_mesh_buffer || !cone_instance_buffer) {
            std::cerr << "Failed to create instanced cone buffers, falling back to legacy rendering" << std::endl;
            instanced_cone_available = false;
        } else {
            std::cout << "Cone mesh buffer: " << unit_cone.byte_size() << " bytes" << std::endl;
            std::cout << "Cone instance buffer: " << cone_instance_buffer_size / 1024 << " KB" << std::endl;
        }
    }

    // Instanced sphere buffers
    std::unique_ptr<gal::Buffer> sphere_mesh_buffer;
    std::unique_ptr<gal::Buffer> sphere_instance_buffer;
    size_t sphere_instance_buffer_size = 10000 * sizeof(SphereInstance);  // Initial capacity for 10k spheres

    if (instanced_sphere_available) {
        // Static unit sphere mesh buffer
        gal::BufferDesc sphere_mesh_desc;
        sphere_mesh_desc.size = unit_sphere.byte_size();
        sphere_mesh_desc.usage = gal::BufferUsage::Vertex;
        sphere_mesh_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
        sphere_mesh_desc.initial_data = unit_sphere.data();
        sphere_mesh_desc.debug_name = "unit_sphere_mesh";
        sphere_mesh_buffer = device->create_buffer(sphere_mesh_desc);

        // Dynamic instance buffer
        sphere_instance_buffer = create_vertex_buffer(sphere_instance_buffer_size);

        if (!sphere_mesh_buffer || !sphere_instance_buffer) {
            std::cerr << "Failed to create instanced sphere buffers, falling back to legacy rendering" << std::endl;
            instanced_sphere_available = false;
        } else {
            std::cout << "Sphere mesh buffer: " << unit_sphere.byte_size() << " bytes" << std::endl;
            std::cout << "Sphere instance buffer: " << sphere_instance_buffer_size / 1024 << " KB" << std::endl;
        }
    }

    // Helper to write data to buffer, resizing if needed
    auto write_to_buffer = [&](std::unique_ptr<gal::Buffer>& buffer, size_t& buffer_size,
                                const std::vector<Vertex>& data, const char* name) {
        if (data.empty()) return;

        size_t data_size = data.size() * sizeof(Vertex);
        ensure_buffer_size(buffer, buffer_size, data_size, name);

        if (buffer) {
            buffer->write(data.data(), data_size);
        }
    };

    // Helper to write cone instances to buffer
    auto write_cone_instances = [&]() {
        if (!instanced_cone_available || cone_instances.empty()) return;

        size_t data_size = cone_instances.size() * sizeof(ConeInstance);
        ensure_buffer_size(cone_instance_buffer, cone_instance_buffer_size, data_size, "cone_instance");

        if (cone_instance_buffer) {
            cone_instance_buffer->write(cone_instances.data(), data_size);
        }
    };

    // Helper to write sphere instances to buffer
    auto write_sphere_instances = [&]() {
        if (!instanced_sphere_available || sphere_instances.empty()) return;

        size_t data_size = sphere_instances.size() * sizeof(SphereInstance);
        ensure_buffer_size(sphere_instance_buffer, sphere_instance_buffer_size, data_size, "sphere_instance");

        if (sphere_instance_buffer) {
            sphere_instance_buffer->write(sphere_instances.data(), data_size);
        }
    };

    // Write initial geometry
    write_to_buffer(triangle_buffer, triangle_buffer_size, triangle_verts, "triangle");
    write_to_buffer(transparent_buffer, transparent_buffer_size, transparent_verts, "transparent");
    write_to_buffer(line_buffer, line_buffer_size, line_verts, "line");
    write_cone_instances();
    write_sphere_instances();

    if (!axis_buffer || !triangle_buffer || !transparent_buffer || !line_buffer) {
        std::cerr << "Failed to create buffers" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // Camera
    camera::PerspectiveCamera cam;
    cam.set_perspective(60.0f, static_cast<float>(window->get_width()) / window->get_height(), 0.1f, 1000.0f);
    cam.set_target(math::vec3(0, 0, 0));
    cam.set_distance(12.0f);
    cam.orbit(0.5f, 0.3f);

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
    bool geometry_dirty = false;

    platform::WindowCallbacks callbacks;
    callbacks.on_resize = [&](uint32_t w, uint32_t h) {
        should_resize = true;
        new_width = w;
        new_height = h;
        cam.set_aspect_ratio(static_cast<float>(w) / h);
    };

    callbacks.on_key = [&](platform::KeyCode key, bool pressed, platform::Modifiers mods) {
        if (!pressed) return;

        if (key == platform::KeyCode::Escape) {
            window->request_close();
        } else if (key >= platform::KeyCode::Num1 && key <= platform::KeyCode::Num5) {
            int mode = static_cast<int>(key) - static_cast<int>(platform::KeyCode::Num1);
            if (mode < static_cast<int>(TestMode::COUNT)) {
                current_mode = static_cast<TestMode>(mode);
                geometry_dirty = true;
            }
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

    // Render loop
    while (window->is_open()) {
        window->poll_events();

        if (should_resize && new_width > 0 && new_height > 0) {
            device->wait_idle();
            swapchain->resize(new_width, new_height);
            should_resize = false;
        }

        if (window->is_minimized()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Regenerate geometry if test mode changed
        if (geometry_dirty) {
            device->wait_idle();
            regenerate_geometry();

            // Write geometry to buffers (auto-resizes if needed)
            write_to_buffer(triangle_buffer, triangle_buffer_size, triangle_verts, "triangle");
            write_to_buffer(transparent_buffer, transparent_buffer_size, transparent_verts, "transparent");
            write_to_buffer(line_buffer, line_buffer_size, line_verts, "line");
            write_cone_instances();
            write_sphere_instances();

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

        math::mat4 vp = cam.get_view_projection_matrix();

        // Ensure OIT resources exist for current resolution
        if (wboit_available) {
            create_oit_resources(w, h);
        }

        auto encoder = device->create_command_encoder();

        // ========== PASS 1: Opaque geometry to swapchain ==========
        gal::RenderPassColorAttachment color_att;
        color_att.texture = tex;
        color_att.load_op = gal::LoadOp::Clear;
        color_att.store_op = gal::StoreOp::Store;
        color_att.clear_color[0] = 0.02f;
        color_att.clear_color[1] = 0.02f;
        color_att.clear_color[2] = 0.04f;
        color_att.clear_color[3] = 1.0f;

        gal::RenderPassBeginInfo rp_info;
        rp_info.pipeline = triangle_pipeline.get();
        rp_info.color_attachments = &color_att;
        rp_info.color_attachment_count = 1;

        auto rp = encoder->begin_render_pass(rp_info);
        if (rp) {
            rp->set_viewport(0, 0, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f);
            rp->set_scissor(0, 0, w, h);

            // Check if WBOIT will be used for transparency
            bool will_use_wboit = wboit_available && !transparent_verts.empty() && oit_accum_texture && oit_reveal_texture;

            // 1. Draw opaque triangles (spheres, and arrows if not using instanced rendering)
            if (!triangle_verts.empty()) {
                rp->set_pipeline(triangle_pipeline.get());
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, triangle_buffer.get());
                uint32_t tri_count = static_cast<uint32_t>(triangle_verts.size());
                rp->draw(tri_count, 1, 0, 0);
            }

            // 1b. Draw instanced cones (if available)
            if (instanced_cone_available && !cone_instances.empty()) {
                rp->set_pipeline(instanced_cone_pipeline.get());
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, cone_mesh_buffer.get());    // Unit cone mesh
                rp->set_vertex_buffer(1, cone_instance_buffer.get()); // Instance data
                uint32_t vertex_count = static_cast<uint32_t>(unit_cone.vertex_count());
                uint32_t instance_count = static_cast<uint32_t>(cone_instances.size());
                rp->draw(vertex_count, instance_count, 0, 0);
            }

            // 1c. Draw instanced spheres (if available)
            if (instanced_sphere_available && !sphere_instances.empty()) {
                rp->set_pipeline(instanced_sphere_pipeline.get());
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, sphere_mesh_buffer.get());    // Unit sphere mesh
                rp->set_vertex_buffer(1, sphere_instance_buffer.get()); // Instance data
                uint32_t vertex_count = static_cast<uint32_t>(unit_sphere.vertex_count());
                uint32_t instance_count = static_cast<uint32_t>(sphere_instances.size());
                rp->draw(vertex_count, instance_count, 0, 0);
            }

            // 2. When NOT using WBOIT: draw transparent with simple alpha blend
            //    When using WBOIT: skip here, will be done in WBOIT pass
            if (!will_use_wboit && !transparent_verts.empty()) {
                rp->set_pipeline(transparent_pipeline.get());
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, transparent_buffer.get());
                uint32_t trans_count = static_cast<uint32_t>(transparent_verts.size());
                rp->draw(trans_count, 1, 0, 0);
            }

            // 3. When NOT using WBOIT: draw lines here (end of pass 1)
            //    When using WBOIT: skip here, will draw after composite pass
            if (!will_use_wboit) {
                // Draw axes
                rp->set_pipeline(line_pipeline.get());
                rp->push_constants(vp.m, sizeof(math::mat4));
                rp->set_vertex_buffer(0, axis_buffer.get());
                rp->draw(static_cast<uint32_t>(axis_verts.size()), 1, 0, 0);

                // Draw edge lines
                if (!line_verts.empty()) {
                    rp->set_pipeline(line_pipeline.get());
                    rp->push_constants(vp.m, sizeof(math::mat4));
                    rp->set_vertex_buffer(0, line_buffer.get());
                    uint32_t line_count = static_cast<uint32_t>(line_verts.size());
                    rp->draw(line_count, 1, 0, 0);
                }
            }

            rp->end();
        }

        // ========== PASS 2: WBOIT transparent pass ==========
        bool use_wboit = wboit_available && !transparent_verts.empty() && oit_accum_texture && oit_reveal_texture;
        if (use_wboit) {
            // Clear and render transparent geometry to OIT textures
            gal::RenderPassColorAttachment oit_attachments[2];

            // Accum texture - clear to (0,0,0,0)
            oit_attachments[0].texture = oit_accum_texture.get();
            oit_attachments[0].load_op = gal::LoadOp::Clear;
            oit_attachments[0].store_op = gal::StoreOp::Store;
            oit_attachments[0].clear_color[0] = 0.0f;
            oit_attachments[0].clear_color[1] = 0.0f;
            oit_attachments[0].clear_color[2] = 0.0f;
            oit_attachments[0].clear_color[3] = 0.0f;

            // Reveal texture - clear to 1.0 (fully transparent = background shows through)
            oit_attachments[1].texture = oit_reveal_texture.get();
            oit_attachments[1].load_op = gal::LoadOp::Clear;
            oit_attachments[1].store_op = gal::StoreOp::Store;
            oit_attachments[1].clear_color[0] = 1.0f;
            oit_attachments[1].clear_color[1] = 0.0f;
            oit_attachments[1].clear_color[2] = 0.0f;
            oit_attachments[1].clear_color[3] = 0.0f;

            gal::RenderPassBeginInfo oit_rp_info;
            oit_rp_info.pipeline = wboit_pipeline.get();
            oit_rp_info.color_attachments = oit_attachments;
            oit_rp_info.color_attachment_count = 2;

            auto oit_rp = encoder->begin_render_pass(oit_rp_info);
            if (oit_rp) {
                oit_rp->set_viewport(0, 0, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f);
                oit_rp->set_scissor(0, 0, w, h);

                // Draw transparent triangles
                oit_rp->set_pipeline(wboit_pipeline.get());
                oit_rp->push_constants(vp.m, sizeof(math::mat4));
                oit_rp->set_vertex_buffer(0, transparent_buffer.get());
                uint32_t trans_count = static_cast<uint32_t>(transparent_verts.size());
                oit_rp->draw(trans_count, 1, 0, 0);

                oit_rp->end();
            }

            // ========== PASS 3: Composite OIT onto swapchain ==========
            gal::RenderPassColorAttachment composite_att;
            composite_att.texture = tex;
            composite_att.load_op = gal::LoadOp::Load;  // Preserve opaque content
            composite_att.store_op = gal::StoreOp::Store;

            gal::RenderPassBeginInfo composite_rp_info;
            composite_rp_info.pipeline = composite_pipeline.get();
            composite_rp_info.color_attachments = &composite_att;
            composite_rp_info.color_attachment_count = 1;

            auto comp_rp = encoder->begin_render_pass(composite_rp_info);
            if (comp_rp) {
                comp_rp->set_viewport(0, 0, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f);
                comp_rp->set_scissor(0, 0, w, h);

                comp_rp->set_pipeline(composite_pipeline.get());
                comp_rp->set_bind_group(0, composite_bind_group.get());
                comp_rp->draw(3, 1, 0, 0);  // Fullscreen triangle

                comp_rp->end();
            }

            // ========== PASS 4: Lines over composited result ==========
            gal::RenderPassColorAttachment lines_att;
            lines_att.texture = tex;
            lines_att.load_op = gal::LoadOp::Load;  // Preserve composited content
            lines_att.store_op = gal::StoreOp::Store;

            gal::RenderPassBeginInfo lines_rp_info;
            lines_rp_info.pipeline = line_pipeline.get();
            lines_rp_info.color_attachments = &lines_att;
            lines_rp_info.color_attachment_count = 1;

            auto lines_rp = encoder->begin_render_pass(lines_rp_info);
            if (lines_rp) {
                lines_rp->set_viewport(0, 0, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f);
                lines_rp->set_scissor(0, 0, w, h);

                // Draw axes
                lines_rp->set_pipeline(line_pipeline.get());
                lines_rp->push_constants(vp.m, sizeof(math::mat4));
                lines_rp->set_vertex_buffer(0, axis_buffer.get());
                lines_rp->draw(static_cast<uint32_t>(axis_verts.size()), 1, 0, 0);

                // Draw edge lines
                if (!line_verts.empty()) {
                    lines_rp->set_pipeline(line_pipeline.get());
                    lines_rp->push_constants(vp.m, sizeof(math::mat4));
                    lines_rp->set_vertex_buffer(0, line_buffer.get());
                    uint32_t line_count = static_cast<uint32_t>(line_verts.size());
                    lines_rp->draw(line_count, 1, 0, 0);
                }

                lines_rp->end();
            }
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
    device->wait_idle();

    // Cleanup
    // OIT resources
    composite_bind_group.reset();
    composite_bind_group_layout.reset();
    oit_sampler.reset();
    oit_reveal_texture.reset();
    oit_accum_texture.reset();
    composite_pipeline.reset();
    wboit_line_pipeline.reset();
    wboit_pipeline.reset();
    composite_fragment_shader.reset();
    composite_vertex_shader.reset();
    wboit_fragment_shader.reset();
    wboit_vertex_shader.reset();

    // Instanced cone resources
    cone_instance_buffer.reset();
    cone_mesh_buffer.reset();
    instanced_cone_pipeline.reset();
    instance_cone_vertex_shader.reset();

    // Instanced sphere resources
    sphere_instance_buffer.reset();
    sphere_mesh_buffer.reset();
    instanced_sphere_pipeline.reset();
    instance_sphere_vertex_shader.reset();

    // Main resources
    line_buffer.reset();
    transparent_buffer.reset();
    triangle_buffer.reset();
    axis_buffer.reset();
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
