#pragma once

#include <gal/types.hpp>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace viz::gal {

// Forward declarations for descriptor structures
struct BufferDesc;
struct TextureDesc;
struct SamplerDesc;
struct ShaderDesc;
struct RenderPipelineDesc;
struct ComputePipelineDesc;
struct BindGroupLayoutDesc;
struct BindGroupDesc;
struct RenderPassDesc;

// Device creation info
struct DeviceDesc {
    Backend backend = Backend::Auto;
    bool enable_validation = false;
    bool enable_debug_names = false;
    const char* app_name = "HypergraphViz";
    uint32_t app_version = 1;

    // Optional: preferred physical device (by name substring match)
    const char* preferred_device = nullptr;

    // Surface handle for presentation (platform-specific)
    Handle surface_handle = nullptr;
};

// Device capabilities/limits
struct DeviceLimits {
    uint32_t max_texture_dimension_1d;
    uint32_t max_texture_dimension_2d;
    uint32_t max_texture_dimension_3d;
    uint32_t max_texture_array_layers;
    uint32_t max_bind_groups;
    uint32_t max_bindings_per_group;
    uint32_t max_uniform_buffer_binding_size;
    uint32_t max_storage_buffer_binding_size;
    uint32_t max_vertex_buffers;
    uint32_t max_vertex_attributes;
    uint32_t max_vertex_buffer_stride;
    uint32_t max_push_constant_size;
    uint32_t max_compute_workgroup_size_x;
    uint32_t max_compute_workgroup_size_y;
    uint32_t max_compute_workgroup_size_z;
    uint32_t max_compute_workgroups_per_dimension;
    uint64_t max_buffer_size;
    uint32_t min_uniform_buffer_offset_alignment;
    uint32_t min_storage_buffer_offset_alignment;
    float max_sampler_anisotropy;
    uint32_t max_color_attachments;
    uint32_t max_samples;  // Max MSAA samples
};

// Device info
struct DeviceInfo {
    std::string device_name;
    std::string vendor_name;
    std::string driver_version;
    std::string api_version;
    bool is_discrete_gpu;
    bool is_integrated_gpu;
    uint64_t dedicated_video_memory;
    DeviceLimits limits;
};

// Abstract device interface
class Device {
public:
    virtual ~Device() = default;

    // Factory method
    static std::unique_ptr<Device> create(const DeviceDesc& desc);

    // Device info
    virtual Backend get_backend() const = 0;
    virtual const DeviceInfo& get_info() const = 0;

    // Resource creation
    virtual std::unique_ptr<Buffer> create_buffer(const BufferDesc& desc) = 0;
    virtual std::unique_ptr<Texture> create_texture(const TextureDesc& desc) = 0;
    virtual std::unique_ptr<Sampler> create_sampler(const SamplerDesc& desc) = 0;
    virtual std::unique_ptr<Shader> create_shader(const ShaderDesc& desc) = 0;
    virtual std::unique_ptr<RenderPipeline> create_render_pipeline(const RenderPipelineDesc& desc) = 0;
    virtual std::unique_ptr<ComputePipeline> create_compute_pipeline(const ComputePipelineDesc& desc) = 0;
    virtual std::unique_ptr<BindGroupLayout> create_bind_group_layout(const BindGroupLayoutDesc& desc) = 0;
    virtual std::unique_ptr<BindGroup> create_bind_group(const BindGroupDesc& desc) = 0;

    // Swapchain
    virtual std::unique_ptr<Swapchain> create_swapchain(Handle surface, uint32_t width, uint32_t height) = 0;

    // Command submission
    virtual std::unique_ptr<CommandEncoder> create_command_encoder() = 0;
    virtual void submit(CommandBuffer* cmd, Fence* signal_fence = nullptr) = 0;
    virtual void submit(CommandBuffer* cmd,
                        Semaphore* wait_semaphore,
                        Semaphore* signal_semaphore,
                        Fence* signal_fence = nullptr) = 0;
    virtual void submit_and_wait(CommandBuffer* cmd) = 0;

    // Synchronization
    virtual std::unique_ptr<Fence> create_fence(bool signaled = false) = 0;
    virtual std::unique_ptr<Semaphore> create_semaphore() = 0;
    virtual void wait_idle() = 0;

    // Debug naming (optional, no-op if not supported)
    virtual void set_debug_name(Buffer* resource, const char* name) {}
    virtual void set_debug_name(Texture* resource, const char* name) {}
    virtual void set_debug_name(RenderPipeline* resource, const char* name) {}
    virtual void set_debug_name(ComputePipeline* resource, const char* name) {}
};

// Buffer descriptor
struct BufferDesc {
    size_t size = 0;
    BufferUsage usage = BufferUsage::None;
    MemoryLocation memory = MemoryLocation::GPU_ONLY;
    const void* initial_data = nullptr;  // Optional initial data
    const char* debug_name = nullptr;
};

// Abstract buffer interface
class Buffer {
public:
    virtual ~Buffer() = default;

    virtual size_t get_size() const = 0;
    virtual BufferUsage get_usage() const = 0;
    virtual MemoryLocation get_memory_location() const = 0;

    // Map for CPU access (only valid for CPU_TO_GPU or GPU_TO_CPU)
    virtual void* map() = 0;
    virtual void unmap() = 0;

    // Write data (convenience, handles mapping internally)
    virtual void write(const void* data, size_t size, size_t offset = 0) = 0;

    // Read data (only for GPU_TO_CPU buffers)
    virtual void read(void* data, size_t size, size_t offset = 0) = 0;

    // Get native handle
    virtual Handle get_native_handle() const = 0;
};

// Texture descriptor
struct TextureDesc {
    Extent3D size = {1, 1, 1};
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;
    Format format = Format::RGBA8_UNORM;
    TextureDimension dimension = TextureDimension::Tex2D;
    TextureUsage usage = TextureUsage::Sampled;
    uint32_t sample_count = 1;  // For MSAA
    const void* initial_data = nullptr;
    const char* debug_name = nullptr;
};

// Abstract texture interface
class Texture {
public:
    virtual ~Texture() = default;

    virtual Extent3D get_size() const = 0;
    virtual Format get_format() const = 0;
    virtual TextureDimension get_dimension() const = 0;
    virtual TextureUsage get_usage() const = 0;
    virtual uint32_t get_mip_levels() const = 0;
    virtual uint32_t get_array_layers() const = 0;
    virtual uint32_t get_sample_count() const = 0;

    virtual Handle get_native_handle() const = 0;
    virtual Handle get_native_view() const = 0;  // Image view / texture view
};

// Sampler descriptor
struct SamplerDesc {
    Filter mag_filter = Filter::Linear;
    Filter min_filter = Filter::Linear;
    MipmapMode mipmap_mode = MipmapMode::Linear;
    AddressMode address_u = AddressMode::Repeat;
    AddressMode address_v = AddressMode::Repeat;
    AddressMode address_w = AddressMode::Repeat;
    float mip_lod_bias = 0.0f;
    bool anisotropy_enable = false;
    float max_anisotropy = 1.0f;
    bool compare_enable = false;
    CompareFunc compare_op = CompareFunc::Never;
    float min_lod = 0.0f;
    float max_lod = 1000.0f;
};

// Abstract sampler interface
class Sampler {
public:
    virtual ~Sampler() = default;
    virtual Handle get_native_handle() const = 0;
};

// Shader descriptor
struct ShaderDesc {
    ShaderStage stage = ShaderStage::Vertex;
    const uint32_t* spirv_code = nullptr;  // SPIR-V bytecode
    size_t spirv_size = 0;                  // Size in bytes
    const char* entry_point = "main";
    const char* debug_name = nullptr;
};

// Abstract shader interface
class Shader {
public:
    virtual ~Shader() = default;
    virtual ShaderStage get_stage() const = 0;
    virtual Handle get_native_handle() const = 0;
};

// Render pipeline descriptor
struct RenderPipelineDesc {
    Shader* vertex_shader = nullptr;
    Shader* fragment_shader = nullptr;

    // Vertex input
    const VertexBufferLayout* vertex_layouts = nullptr;
    uint32_t vertex_layout_count = 0;

    // Primitive assembly
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;

    // Rasterization
    RasterizerState rasterizer;

    // Multisample
    MultisampleState multisample;

    // Depth/stencil
    DepthStencilState depth_stencil;

    // Color attachments
    const BlendState* blend_states = nullptr;
    uint32_t blend_state_count = 0;

    // Render pass format (for compatibility)
    const Format* color_formats = nullptr;
    uint32_t color_format_count = 0;
    Format depth_format = Format::Undefined;

    // If true, color attachments will be used as shader inputs after rendering
    // (uses SHADER_READ_ONLY_OPTIMAL layout instead of PRESENT_SRC_KHR)
    bool render_to_texture = false;

    // Pipeline layout
    const BindGroupLayout* const* bind_group_layouts = nullptr;
    uint32_t bind_group_layout_count = 0;
    uint32_t push_constant_size = 0;

    const char* debug_name = nullptr;
};

// Abstract render pipeline
class RenderPipeline {
public:
    virtual ~RenderPipeline() = default;
    virtual Handle get_native_handle() const = 0;
    virtual Handle get_native_layout() const = 0;
};

// Compute pipeline descriptor
struct ComputePipelineDesc {
    Shader* compute_shader = nullptr;

    const BindGroupLayout* const* bind_group_layouts = nullptr;
    uint32_t bind_group_layout_count = 0;
    uint32_t push_constant_size = 0;

    const char* debug_name = nullptr;
};

// Abstract compute pipeline
class ComputePipeline {
public:
    virtual ~ComputePipeline() = default;
    virtual Handle get_native_handle() const = 0;
    virtual Handle get_native_layout() const = 0;
};

// Bind group layout descriptor
struct BindGroupLayoutDesc {
    const BindGroupLayoutEntry* entries = nullptr;
    uint32_t entry_count = 0;
    const char* debug_name = nullptr;
};

// Abstract bind group layout
class BindGroupLayout {
public:
    virtual ~BindGroupLayout() = default;
    virtual Handle get_native_handle() const = 0;
};

// Bind group entry (for creating bind groups)
struct BindGroupEntry {
    uint32_t binding = 0;

    // One of these should be set
    Buffer* buffer = nullptr;
    size_t buffer_offset = 0;
    size_t buffer_size = 0;  // 0 = whole buffer

    Texture* texture = nullptr;
    Sampler* sampler = nullptr;
};

// Bind group descriptor
struct BindGroupDesc {
    BindGroupLayout* layout = nullptr;
    const BindGroupEntry* entries = nullptr;
    uint32_t entry_count = 0;
    const char* debug_name = nullptr;
};

// Abstract bind group
class BindGroup {
public:
    virtual ~BindGroup() = default;
    virtual Handle get_native_handle() const = 0;
};

// Fence for CPU-GPU synchronization
class Fence {
public:
    virtual ~Fence() = default;
    virtual void wait(uint64_t timeout_ns = UINT64_MAX) = 0;
    virtual bool is_signaled() const = 0;
    virtual void reset() = 0;
    virtual Handle get_native_handle() const = 0;
};

// Semaphore for GPU-GPU synchronization
class Semaphore {
public:
    virtual ~Semaphore() = default;
    virtual Handle get_native_handle() const = 0;
};

} // namespace viz::gal
