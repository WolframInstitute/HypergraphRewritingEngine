// Native application entry point for the Hypergraph Visualization
// Phase 3: 3D camera, vertex buffer rendering, and GPU layout

#include <platform/window.hpp>
#include <gal/gal.hpp>
#include <gal/vulkan/vk_loader.hpp>
#include <camera/camera.hpp>
#include <math/types.hpp>
#include <layout/layout_engine.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>

using namespace viz;

// Vulkan surface creation helper (defined in vk_swapchain.cpp)
namespace viz::gal {
    VkSurfaceKHR create_xcb_surface(VkInstance instance, void* connection, void* window);
    VkSurfaceKHR create_win32_surface(VkInstance instance, void* hinstance, void* hwnd);
    VkInstance get_vk_instance(Device* device);
}

// Load SPIR-V shader from file
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

// Vertex structure for 3D colored geometry
struct Vertex {
    float x, y, z;       // Position
    float r, g, b, a;    // Color
};

// Create a simple cube mesh
std::vector<Vertex> create_cube(float size = 1.0f) {
    float s = size * 0.5f;
    std::vector<Vertex> vertices;

    // Front face (red)
    vertices.push_back({-s, -s,  s,  1.0f, 0.2f, 0.2f, 1.0f});
    vertices.push_back({ s, -s,  s,  1.0f, 0.2f, 0.2f, 1.0f});
    vertices.push_back({ s,  s,  s,  1.0f, 0.2f, 0.2f, 1.0f});
    vertices.push_back({-s, -s,  s,  1.0f, 0.2f, 0.2f, 1.0f});
    vertices.push_back({ s,  s,  s,  1.0f, 0.2f, 0.2f, 1.0f});
    vertices.push_back({-s,  s,  s,  1.0f, 0.2f, 0.2f, 1.0f});

    // Back face (cyan)
    vertices.push_back({ s, -s, -s,  0.2f, 1.0f, 1.0f, 1.0f});
    vertices.push_back({-s, -s, -s,  0.2f, 1.0f, 1.0f, 1.0f});
    vertices.push_back({-s,  s, -s,  0.2f, 1.0f, 1.0f, 1.0f});
    vertices.push_back({ s, -s, -s,  0.2f, 1.0f, 1.0f, 1.0f});
    vertices.push_back({-s,  s, -s,  0.2f, 1.0f, 1.0f, 1.0f});
    vertices.push_back({ s,  s, -s,  0.2f, 1.0f, 1.0f, 1.0f});

    // Top face (green)
    vertices.push_back({-s,  s,  s,  0.2f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({ s,  s,  s,  0.2f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({ s,  s, -s,  0.2f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({-s,  s,  s,  0.2f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({ s,  s, -s,  0.2f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({-s,  s, -s,  0.2f, 1.0f, 0.2f, 1.0f});

    // Bottom face (magenta)
    vertices.push_back({-s, -s, -s,  1.0f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({ s, -s, -s,  1.0f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({ s, -s,  s,  1.0f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({-s, -s, -s,  1.0f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({ s, -s,  s,  1.0f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({-s, -s,  s,  1.0f, 0.2f, 1.0f, 1.0f});

    // Right face (yellow)
    vertices.push_back({ s, -s,  s,  1.0f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({ s, -s, -s,  1.0f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({ s,  s, -s,  1.0f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({ s, -s,  s,  1.0f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({ s,  s, -s,  1.0f, 1.0f, 0.2f, 1.0f});
    vertices.push_back({ s,  s,  s,  1.0f, 1.0f, 0.2f, 1.0f});

    // Left face (blue)
    vertices.push_back({-s, -s, -s,  0.2f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({-s, -s,  s,  0.2f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({-s,  s,  s,  0.2f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({-s, -s, -s,  0.2f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({-s,  s,  s,  0.2f, 0.2f, 1.0f, 1.0f});
    vertices.push_back({-s,  s, -s,  0.2f, 0.2f, 1.0f, 1.0f});

    return vertices;
}

// Create coordinate axis lines
std::vector<Vertex> create_axes(float length = 5.0f) {
    std::vector<Vertex> vertices;

    // X axis (red)
    vertices.push_back({0, 0, 0,  1, 0, 0, 1});
    vertices.push_back({length, 0, 0,  1, 0, 0, 1});

    // Y axis (green)
    vertices.push_back({0, 0, 0,  0, 1, 0, 1});
    vertices.push_back({0, length, 0,  0, 1, 0, 1});

    // Z axis (blue)
    vertices.push_back({0, 0, 0,  0, 0, 1, 1});
    vertices.push_back({0, 0, length,  0, 0, 1, 1});

    return vertices;
}

// Create a UV sphere
std::vector<Vertex> create_sphere(const math::vec3& center, float radius,
                                   const math::vec4& color, int segments = 16, int rings = 12) {
    std::vector<Vertex> vertices;

    for (int ring = 0; ring < rings; ++ring) {
        float theta1 = math::PI * ring / rings;
        float theta2 = math::PI * (ring + 1) / rings;

        for (int seg = 0; seg < segments; ++seg) {
            float phi1 = math::TAU * seg / segments;
            float phi2 = math::TAU * (seg + 1) / segments;

            // Four corners of this quad
            math::vec3 p1(std::sin(theta1) * std::cos(phi1),
                          std::cos(theta1),
                          std::sin(theta1) * std::sin(phi1));
            math::vec3 p2(std::sin(theta1) * std::cos(phi2),
                          std::cos(theta1),
                          std::sin(theta1) * std::sin(phi2));
            math::vec3 p3(std::sin(theta2) * std::cos(phi2),
                          std::cos(theta2),
                          std::sin(theta2) * std::sin(phi2));
            math::vec3 p4(std::sin(theta2) * std::cos(phi1),
                          std::cos(theta2),
                          std::sin(theta2) * std::sin(phi1));

            p1 = center + p1 * radius;
            p2 = center + p2 * radius;
            p3 = center + p3 * radius;
            p4 = center + p4 * radius;

            // Two triangles
            vertices.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
            vertices.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
            vertices.push_back({p3.x, p3.y, p3.z, color.x, color.y, color.z, color.w});

            vertices.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
            vertices.push_back({p3.x, p3.y, p3.z, color.x, color.y, color.z, color.w});
            vertices.push_back({p4.x, p4.y, p4.z, color.x, color.y, color.z, color.w});
        }
    }

    return vertices;
}

// Create edges as lines between nodes
std::vector<Vertex> create_edges(const std::vector<math::vec3>& positions,
                                  const std::vector<std::pair<int, int>>& edges,
                                  const math::vec4& color) {
    std::vector<Vertex> vertices;

    for (const auto& [from, to] : edges) {
        if (from < positions.size() && to < positions.size()) {
            const auto& p1 = positions[from];
            const auto& p2 = positions[to];
            vertices.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
            vertices.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
        }
    }

    return vertices;
}

// Interactive graph with live layout
struct InteractiveGraph {
    layout::LayoutGraph layout_graph;
    std::vector<math::vec4> node_colors;
    std::vector<std::pair<int, int>> edges;
    std::unique_ptr<layout::ILayoutEngine> engine;
    layout::LayoutParams params;
    bool layout_running = true;
    uint32_t total_iterations = 0;

    void init_sample_graph() {
        // Random initial positions
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

        // Colors by generation depth
        std::vector<math::vec4> colors = {
            {1.0f, 1.0f, 1.0f, 1.0f},  // Root: white
            {1.0f, 0.4f, 0.4f, 1.0f}, {0.4f, 1.0f, 0.4f, 1.0f}, {0.4f, 0.4f, 1.0f, 1.0f},  // Gen 1
            {1.0f, 0.6f, 0.6f, 1.0f}, {1.0f, 0.6f, 0.6f, 1.0f},  // Gen 2 (red branch)
            {0.6f, 1.0f, 0.6f, 1.0f}, {0.6f, 1.0f, 0.6f, 1.0f},  // Gen 2 (green branch)
            {0.6f, 0.6f, 1.0f, 1.0f}, {0.6f, 0.6f, 1.0f, 1.0f},  // Gen 2 (blue branch)
            {1.0f, 0.8f, 0.8f, 1.0f}, {1.0f, 0.8f, 0.8f, 1.0f},  // Gen 3
        };

        // Add vertices with random positions
        for (size_t i = 0; i < colors.size(); i++) {
            layout_graph.add_vertex(dist(gen), dist(gen), dist(gen), 1.0f, false);
            node_colors.push_back(colors[i]);
        }

        // Tree edges
        edges = {{0,1}, {0,2}, {0,3}, {1,4}, {1,5}, {2,6}, {2,7}, {3,8}, {3,9}, {4,10}, {4,11}};
        for (const auto& [s, d] : edges) {
            layout_graph.add_edge(s, d, 2.0f, 1.0f);
        }

        // Setup layout parameters
        params.spring_constant = 0.3f;
        params.repulsion_constant = 3.0f;
        params.damping = 0.9f;
        params.max_displacement = 0.5f;
        params.convergence_threshold = 0.005f;

        // Create engine
        engine = layout::create_layout_engine(layout::LayoutBackend::Auto);
        if (engine) {
            std::cout << "Layout backend: " << layout::backend_name(engine->get_backend()) << std::endl;
            engine->upload_graph(layout_graph);
        }
    }

    // Run one layout iteration, return true if still running
    bool step() {
        if (!engine || !layout_running) return false;

        auto result = engine->iterate(params);
        total_iterations++;

        if (result.converged || total_iterations >= 1000) {
            layout_running = false;
            std::cout << "Layout settled: " << total_iterations << " iterations" << std::endl;
        }

        // Download updated positions
        engine->download_positions(layout_graph);
        return layout_running;
    }

    // Restart layout with new random positions
    void restart() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

        for (uint32_t i = 0; i < layout_graph.vertex_count(); i++) {
            layout_graph.positions_x[i] = dist(gen);
            layout_graph.positions_y[i] = dist(gen);
            layout_graph.positions_z[i] = dist(gen);
        }

        if (engine) {
            engine->upload_graph(layout_graph);
        }
        layout_running = true;
        total_iterations = 0;
        std::cout << "Layout restarted" << std::endl;
    }

    // Get positions for rendering
    std::vector<math::vec3> get_positions() const {
        std::vector<math::vec3> pos;
        pos.reserve(layout_graph.vertex_count());
        for (uint32_t i = 0; i < layout_graph.vertex_count(); i++) {
            pos.push_back({layout_graph.positions_x[i],
                          layout_graph.positions_y[i],
                          layout_graph.positions_z[i]});
        }
        return pos;
    }
};

// Regenerate node geometry from positions
void update_node_geometry(std::vector<Vertex>& vertices,
                          const std::vector<math::vec3>& positions,
                          const std::vector<math::vec4>& colors,
                          float radius) {
    vertices.clear();
    for (size_t i = 0; i < positions.size(); i++) {
        auto sphere = create_sphere(positions[i], radius, colors[i], 12, 8);
        vertices.insert(vertices.end(), sphere.begin(), sphere.end());
    }
}

// Regenerate edge geometry from positions
void update_edge_geometry(std::vector<Vertex>& vertices,
                          const std::vector<math::vec3>& positions,
                          const std::vector<std::pair<int, int>>& edges,
                          const math::vec4& color) {
    vertices.clear();
    for (const auto& [from, to] : edges) {
        if (from < positions.size() && to < positions.size()) {
            const auto& p1 = positions[from];
            const auto& p2 = positions[to];
            vertices.push_back({p1.x, p1.y, p1.z, color.x, color.y, color.z, color.w});
            vertices.push_back({p2.x, p2.y, p2.z, color.x, color.y, color.z, color.w});
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Hypergraph Visualization - Phase 3 (Interactive Layout)" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Left-drag: Orbit camera" << std::endl;
    std::cout << "  Scroll: Zoom in/out" << std::endl;
    std::cout << "  R: Restart layout" << std::endl;
    std::cout << "  Space: Pause/resume layout" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;
    std::cout << std::endl;

    // Create window
    platform::WindowDesc window_desc;
    window_desc.title = "Hypergraph Viz - 3D Camera Test";
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

    // Create device
    gal::DeviceDesc device_desc;
    device_desc.app_name = "HypergraphViz";
    device_desc.app_version = 1;
#ifdef NDEBUG
    device_desc.enable_validation = false;
#else
    device_desc.enable_validation = true;
#endif

    auto device = gal::Device::create(device_desc);
    if (!device) {
        std::cerr << "Failed to create device" << std::endl;
        gal::shutdown();
        return 1;
    }

    const auto& info = device->get_info();
    std::cout << "Device: " << info.device_name << std::endl;

    // Create Vulkan surface
    VkInstance vk_instance = gal::get_vk_instance(device.get());
    VkSurfaceKHR surface = VK_NULL_HANDLE;

#if defined(VIZ_PLATFORM_LINUX)
    surface = gal::create_xcb_surface(
        vk_instance,
        window->get_native_display(),
        window->get_native_window()
    );
#elif defined(VIZ_PLATFORM_WINDOWS)
    surface = gal::create_win32_surface(
        vk_instance,
        GetModuleHandle(nullptr),
        window->get_native_window()
    );
#endif

    if (surface == VK_NULL_HANDLE) {
        std::cerr << "Failed to create Vulkan surface" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // Create swapchain
    auto swapchain = device->create_swapchain(
        reinterpret_cast<gal::Handle>(surface),
        window->get_width(),
        window->get_height()
    );

    if (!swapchain) {
        std::cerr << "Failed to create swapchain" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // Load shaders
    auto vert_spirv = load_spirv("../shaders/spirv/basic3d.vert.spv");
    auto frag_spirv = load_spirv("../shaders/spirv/basic3d.frag.spv");

    if (vert_spirv.empty() || frag_spirv.empty()) {
        std::cerr << "Failed to load shaders" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
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

    // Create vertex buffer layout
    gal::VertexAttribute attribs[] = {
        {0, gal::Format::RGB32_FLOAT, 0},                          // position
        {1, gal::Format::RGBA32_FLOAT, sizeof(float) * 3}          // color
    };

    gal::VertexBufferLayout vertex_layout;
    vertex_layout.stride = sizeof(Vertex);
    vertex_layout.step_mode = gal::VertexStepMode::Vertex;
    vertex_layout.attributes = attribs;
    vertex_layout.attribute_count = 2;

    // Create render pipeline for triangles
    gal::Format color_format = swapchain->get_format();
    gal::BlendState blend_state;
    blend_state.write_mask = 0xF;

    gal::RenderPipelineDesc pipeline_desc;
    pipeline_desc.vertex_shader = vertex_shader.get();
    pipeline_desc.fragment_shader = fragment_shader.get();
    pipeline_desc.vertex_layouts = &vertex_layout;
    pipeline_desc.vertex_layout_count = 1;
    pipeline_desc.topology = gal::PrimitiveTopology::TriangleList;
    pipeline_desc.rasterizer.cull_mode = gal::CullMode::Back;
    pipeline_desc.depth_stencil.depth_test_enable = false;
    pipeline_desc.blend_states = &blend_state;
    pipeline_desc.blend_state_count = 1;
    pipeline_desc.color_formats = &color_format;
    pipeline_desc.color_format_count = 1;
    pipeline_desc.push_constant_size = sizeof(math::mat4);

    auto triangle_pipeline = device->create_render_pipeline(pipeline_desc);

    // Create render pipeline for lines
    pipeline_desc.topology = gal::PrimitiveTopology::LineList;
    pipeline_desc.rasterizer.cull_mode = gal::CullMode::None;

    auto line_pipeline = device->create_render_pipeline(pipeline_desc);

    if (!triangle_pipeline || !line_pipeline) {
        std::cerr << "Failed to create render pipelines" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    // Create geometry
    auto axis_vertices = create_axes(5.0f);

    // Create interactive graph with live layout
    InteractiveGraph graph;
    graph.init_sample_graph();

    // Initial geometry
    float node_radius = 0.15f;
    math::vec4 edge_color(0.6f, 0.6f, 0.6f, 1.0f);

    std::vector<Vertex> node_vertices;
    std::vector<Vertex> edge_vertices;
    auto positions = graph.get_positions();
    update_node_geometry(node_vertices, positions, graph.node_colors, node_radius);
    update_edge_geometry(edge_vertices, positions, graph.edges, edge_color);

    // Create vertex buffers
    gal::BufferDesc axis_buffer_desc;
    axis_buffer_desc.size = axis_vertices.size() * sizeof(Vertex);
    axis_buffer_desc.usage = gal::BufferUsage::Vertex;
    axis_buffer_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
    axis_buffer_desc.initial_data = axis_vertices.data();

    auto axis_buffer = device->create_buffer(axis_buffer_desc);

    // Node and edge buffers are CPU_TO_GPU for easy updates during layout
    gal::BufferDesc node_buffer_desc;
    node_buffer_desc.size = node_vertices.size() * sizeof(Vertex);
    node_buffer_desc.usage = gal::BufferUsage::Vertex;
    node_buffer_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
    node_buffer_desc.initial_data = node_vertices.data();

    auto node_buffer = device->create_buffer(node_buffer_desc);

    gal::BufferDesc edge_buffer_desc;
    edge_buffer_desc.size = std::max(edge_vertices.size() * sizeof(Vertex), size_t(1024));  // Min size
    edge_buffer_desc.usage = gal::BufferUsage::Vertex;
    edge_buffer_desc.memory = gal::MemoryLocation::CPU_TO_GPU;
    edge_buffer_desc.initial_data = edge_vertices.empty() ? nullptr : edge_vertices.data();

    auto edge_buffer = device->create_buffer(edge_buffer_desc);

    if (!axis_buffer || !node_buffer || !edge_buffer) {
        std::cerr << "Failed to create vertex buffers" << std::endl;
        device.reset();
        gal::shutdown();
        return 1;
    }

    std::cout << "Created graph: " << graph.layout_graph.vertex_count() << " nodes, "
              << graph.edges.size() << " edges" << std::endl;
    std::cout << "Initial geometry: " << node_vertices.size() << " node verts, "
              << edge_vertices.size() << " edge verts" << std::endl;

    // Create camera
    camera::PerspectiveCamera cam;
    cam.set_perspective(60.0f, static_cast<float>(window->get_width()) / window->get_height(),
                        0.1f, 1000.0f);
    cam.set_target(math::vec3(0, 0, 0));
    cam.set_distance(8.0f);
    cam.orbit(0.5f, 0.3f);  // Initial orientation

    // Camera controller
    camera::CameraController controller(&cam);

    // Create synchronization objects
    auto image_available_semaphore = device->create_semaphore();
    auto render_finished_semaphore = device->create_semaphore();
    auto in_flight_fence = device->create_fence(true);

    // Input state
    bool left_mouse_down = false;
    float last_mouse_x = 0, last_mouse_y = 0;

    // Set up window callbacks
    bool should_resize = false;
    uint32_t new_width = 0, new_height = 0;

    platform::WindowCallbacks callbacks;
    callbacks.on_resize = [&](uint32_t w, uint32_t h) {
        should_resize = true;
        new_width = w;
        new_height = h;
        cam.set_aspect_ratio(static_cast<float>(w) / h);
    };

    callbacks.on_key = [&](platform::KeyCode key, bool pressed, platform::Modifiers mods) {
        if (pressed) {
            if (key == platform::KeyCode::Escape) {
                window->request_close();
            } else if (key == platform::KeyCode::R) {
                graph.restart();
            } else if (key == platform::KeyCode::Space) {
                graph.layout_running = !graph.layout_running;
                std::cout << "Layout " << (graph.layout_running ? "resumed" : "paused") << std::endl;
            }
        }
    };

    callbacks.on_mouse_button = [&](platform::MouseButton button, bool pressed, int x, int y, platform::Modifiers mods) {
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

    callbacks.on_close = [&]() {
        std::cout << "Close requested" << std::endl;
    };

    window->set_callbacks(callbacks);

    std::cout << "\nStarting render loop..." << std::endl;

    // Render loop
    uint64_t frame_count = 0;
    auto last_fps_time = std::chrono::high_resolution_clock::now();

    while (window->is_open()) {
        window->poll_events();

        // Handle resize
        if (should_resize && new_width > 0 && new_height > 0) {
            device->wait_idle();
            swapchain->resize(new_width, new_height);
            should_resize = false;
        }

        if (window->is_minimized()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Wait for previous frame
        in_flight_fence->wait();
        in_flight_fence->reset();

        // Update layout (step the simulation)
        if (graph.step()) {
            // Layout changed - update geometry
            positions = graph.get_positions();
            update_node_geometry(node_vertices, positions, graph.node_colors, node_radius);
            update_edge_geometry(edge_vertices, positions, graph.edges, edge_color);

            // Update GPU buffers
            node_buffer->write(node_vertices.data(), node_vertices.size() * sizeof(Vertex));
            if (!edge_vertices.empty()) {
                edge_buffer->write(edge_vertices.data(), edge_vertices.size() * sizeof(Vertex));
            }
        }

        // Acquire swapchain image
        auto acquire_result = swapchain->acquire_next_image(
            image_available_semaphore.get(), nullptr);

        if (!acquire_result.success) {
            device->wait_idle();
            swapchain->resize(window->get_width(), window->get_height());
            continue;
        }

        auto* swapchain_texture = swapchain->get_texture(acquire_result.image_index);
        uint32_t width = swapchain_texture->get_size().width;
        uint32_t height = swapchain_texture->get_size().height;

        // Get view-projection matrix
        math::mat4 view_proj = cam.get_view_projection_matrix();

        // Record commands
        auto encoder = device->create_command_encoder();

        gal::RenderPassColorAttachment color_attachment;
        color_attachment.texture = swapchain_texture;
        color_attachment.load_op = gal::LoadOp::Clear;
        color_attachment.store_op = gal::StoreOp::Store;
        color_attachment.clear_color[0] = 0.05f;
        color_attachment.clear_color[1] = 0.05f;
        color_attachment.clear_color[2] = 0.08f;
        color_attachment.clear_color[3] = 1.0f;

        gal::RenderPassBeginInfo rp_info;
        rp_info.pipeline = triangle_pipeline.get();
        rp_info.color_attachments = &color_attachment;
        rp_info.color_attachment_count = 1;

        auto render_pass = encoder->begin_render_pass(rp_info);
        if (render_pass) {
            render_pass->set_viewport(0, 0, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
            render_pass->set_scissor(0, 0, width, height);

            // Draw axes (lines)
            render_pass->set_pipeline(line_pipeline.get());
            render_pass->push_constants(view_proj.m, sizeof(math::mat4));
            render_pass->set_vertex_buffer(0, axis_buffer.get());
            render_pass->draw(static_cast<uint32_t>(axis_vertices.size()), 1, 0, 0);

            // Draw edges (lines)
            render_pass->set_vertex_buffer(0, edge_buffer.get());
            render_pass->draw(static_cast<uint32_t>(edge_vertices.size()), 1, 0, 0);

            // Draw nodes (spheres - triangles)
            render_pass->set_pipeline(triangle_pipeline.get());
            render_pass->push_constants(view_proj.m, sizeof(math::mat4));
            render_pass->set_vertex_buffer(0, node_buffer.get());
            render_pass->draw(static_cast<uint32_t>(node_vertices.size()), 1, 0, 0);

            render_pass->end();
        }

        auto cmd = encoder->finish();

        // Submit
        device->submit(cmd.get(),
                       image_available_semaphore.get(),
                       render_finished_semaphore.get(),
                       in_flight_fence.get());

        // Present
        if (!swapchain->present(render_finished_semaphore.get())) {
            device->wait_idle();
            swapchain->resize(window->get_width(), window->get_height());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));

        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - last_fps_time).count();
        if (elapsed >= 2.0) {
            std::cout << "FPS: " << static_cast<int>(frame_count / elapsed) << std::endl;
            frame_count = 0;
            last_fps_time = current_time;
        }
    }

    std::cout << "\nShutting down..." << std::endl;
    device->wait_idle();

    // Cleanup
    node_buffer.reset();
    edge_buffer.reset();
    axis_buffer.reset();
    triangle_pipeline.reset();
    line_pipeline.reset();
    vertex_shader.reset();
    fragment_shader.reset();
    in_flight_fence.reset();
    render_finished_semaphore.reset();
    image_available_semaphore.reset();
    swapchain.reset();

    if (surface != VK_NULL_HANDLE) {
        gal::vk::vkDestroySurfaceKHR(vk_instance, surface, nullptr);
    }

    device.reset();
    gal::shutdown();
    window.reset();

    std::cout << "Cleanup complete" << std::endl;
    return 0;
}
