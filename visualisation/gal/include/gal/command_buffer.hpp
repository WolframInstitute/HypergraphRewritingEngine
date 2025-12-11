#pragma once

#include <gal/types.hpp>
#include <gal/device.hpp>

namespace viz::gal {

// Color attachment for render pass
struct RenderPassColorAttachment {
    Texture* texture = nullptr;
    Texture* resolve_texture = nullptr;  // For MSAA resolve
    LoadOp load_op = LoadOp::Clear;
    StoreOp store_op = StoreOp::Store;
    float clear_color[4] = {0, 0, 0, 1};
};

// Depth/stencil attachment for render pass
struct RenderPassDepthAttachment {
    Texture* texture = nullptr;
    LoadOp depth_load_op = LoadOp::Clear;
    StoreOp depth_store_op = StoreOp::Store;
    LoadOp stencil_load_op = LoadOp::DontCare;
    StoreOp stencil_store_op = StoreOp::DontCare;
    float clear_depth = 1.0f;
    uint32_t clear_stencil = 0;
};

// Render pass begin info
struct RenderPassBeginInfo {
    RenderPipeline* pipeline = nullptr;  // Pipeline provides render pass
    const RenderPassColorAttachment* color_attachments = nullptr;
    uint32_t color_attachment_count = 0;
    RenderPassDepthAttachment depth_attachment;
};

// Buffer copy info for texture operations
struct BufferTextureCopy {
    size_t buffer_offset = 0;
    uint32_t buffer_row_length = 0;
    uint32_t buffer_image_height = 0;
    uint32_t texture_mip_level = 0;
    uint32_t texture_array_layer = 0;
    Offset3D texture_offset = {0, 0, 0};
    Extent3D copy_size = {1, 1, 1};
};

// Texture-to-texture copy info
struct TextureCopy {
    uint32_t src_mip_level = 0;
    uint32_t src_array_layer = 0;
    Offset3D src_offset = {0, 0, 0};
    uint32_t dst_mip_level = 0;
    uint32_t dst_array_layer = 0;
    Offset3D dst_offset = {0, 0, 0};
    Extent3D copy_size = {1, 1, 1};
};

// Abstract render pass encoder
class RenderPassEncoder {
public:
    virtual ~RenderPassEncoder() = default;

    // Pipeline binding
    virtual void set_pipeline(RenderPipeline* pipeline) = 0;

    // Bind groups (descriptor sets)
    virtual void set_bind_group(uint32_t index, BindGroup* group) = 0;

    // Vertex/index buffers
    virtual void set_vertex_buffer(uint32_t slot, Buffer* buffer, size_t offset = 0) = 0;
    virtual void set_index_buffer(Buffer* buffer, IndexFormat format, size_t offset = 0) = 0;

    // Dynamic state
    virtual void set_viewport(float x, float y, float width, float height,
                               float min_depth, float max_depth) = 0;
    virtual void set_scissor(int32_t x, int32_t y, uint32_t width, uint32_t height) = 0;
    virtual void set_blend_constant(float r, float g, float b, float a) = 0;
    virtual void set_stencil_reference(uint32_t reference) = 0;

    // Push constants
    virtual void push_constants(const void* data, size_t size, size_t offset = 0) = 0;

    // Draw commands
    virtual void draw(uint32_t vertex_count, uint32_t instance_count = 1,
                      uint32_t first_vertex = 0, uint32_t first_instance = 0) = 0;
    virtual void draw_indexed(uint32_t index_count, uint32_t instance_count = 1,
                               uint32_t first_index = 0, int32_t vertex_offset = 0,
                               uint32_t first_instance = 0) = 0;
    virtual void draw_indirect(Buffer* buffer, size_t offset, uint32_t draw_count) = 0;
    virtual void draw_indexed_indirect(Buffer* buffer, size_t offset, uint32_t draw_count) = 0;

    // End the render pass
    virtual void end() = 0;
};

// Abstract compute pass encoder
class ComputePassEncoder {
public:
    virtual ~ComputePassEncoder() = default;

    virtual void set_pipeline(ComputePipeline* pipeline) = 0;
    virtual void set_bind_group(uint32_t index, BindGroup* group) = 0;
    virtual void push_constants(const void* data, size_t size, size_t offset = 0) = 0;

    virtual void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) = 0;
    virtual void dispatch_indirect(Buffer* buffer, size_t offset) = 0;

    virtual void end() = 0;
};

// Command encoder - records commands into a command buffer
class CommandEncoder {
public:
    virtual ~CommandEncoder() = default;

    // Begin render/compute passes
    virtual std::unique_ptr<RenderPassEncoder> begin_render_pass(const RenderPassBeginInfo& info) = 0;
    virtual std::unique_ptr<ComputePassEncoder> begin_compute_pass() = 0;

    // Buffer copy operations
    virtual void copy_buffer_to_buffer(Buffer* src, size_t src_offset,
                                        Buffer* dst, size_t dst_offset,
                                        size_t size) = 0;

    virtual void copy_buffer_to_texture(Buffer* src, Texture* dst,
                                         const BufferTextureCopy& copy) = 0;

    virtual void copy_texture_to_buffer(Texture* src, Buffer* dst,
                                         const BufferTextureCopy& copy) = 0;

    virtual void copy_texture_to_texture(Texture* src, Texture* dst,
                                          const TextureCopy& copy) = 0;

    // Finish recording and get the command buffer
    virtual std::unique_ptr<CommandBuffer> finish() = 0;
};

// Recorded command buffer - ready for submission
class CommandBuffer {
public:
    virtual ~CommandBuffer() = default;
    virtual Handle get_native_handle() const = 0;
};

} // namespace viz::gal
