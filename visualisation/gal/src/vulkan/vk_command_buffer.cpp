#include "vk_internal.hpp"

#include <iostream>
#include <vector>
#include <unordered_map>

namespace viz::gal {

// Forward declarations
class VulkanRenderPipeline;
class VulkanComputePipeline;
class VulkanBuffer;
class VulkanBindGroup;

// Get render pass from pipeline (for compatibility check only)
VkRenderPass get_vk_render_pass(RenderPipeline* pipeline);

// Helper to convert GAL LoadOp to Vulkan
static VkAttachmentLoadOp to_vk_load_op(LoadOp op) {
    switch (op) {
        case LoadOp::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
        case LoadOp::Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
        case LoadOp::DontCare: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        default: return VK_ATTACHMENT_LOAD_OP_CLEAR;
    }
}

// Helper to convert GAL StoreOp to Vulkan
static VkAttachmentStoreOp to_vk_store_op(StoreOp op) {
    switch (op) {
        case StoreOp::Store: return VK_ATTACHMENT_STORE_OP_STORE;
        case StoreOp::DontCare: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
        default: return VK_ATTACHMENT_STORE_OP_STORE;
    }
}

// Helper to convert GAL Format to Vulkan
static VkFormat to_vk_format_cmd(Format format) {
    switch (format) {
        case Format::R8_UNORM: return VK_FORMAT_R8_UNORM;
        case Format::R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
        case Format::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case Format::BGRA8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case Format::BGRA8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        case Format::RGBA16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case Format::RGBA32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case Format::D16_UNORM: return VK_FORMAT_D16_UNORM;
        case Format::D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
        case Format::D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
        case Format::D32_FLOAT_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

// Render pass key for caching
struct RenderPassKey {
    std::vector<VkFormat> color_formats;
    std::vector<VkAttachmentLoadOp> color_load_ops;
    std::vector<VkAttachmentStoreOp> color_store_ops;
    std::vector<bool> is_swapchain;  // Track if attachment goes to present
    std::vector<bool> has_resolve;   // Track if color attachment has resolve target
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkAttachmentLoadOp depth_load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp depth_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    uint32_t samples = 1;

    bool operator==(const RenderPassKey& other) const {
        return color_formats == other.color_formats &&
               color_load_ops == other.color_load_ops &&
               color_store_ops == other.color_store_ops &&
               is_swapchain == other.is_swapchain &&
               has_resolve == other.has_resolve &&
               depth_format == other.depth_format &&
               depth_load_op == other.depth_load_op &&
               depth_store_op == other.depth_store_op &&
               samples == other.samples;
    }
};

struct RenderPassKeyHash {
    size_t operator()(const RenderPassKey& key) const {
        size_t hash = 0;
        for (auto fmt : key.color_formats) hash ^= std::hash<int>()(fmt) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        for (auto op : key.color_load_ops) hash ^= std::hash<int>()(op) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        for (auto op : key.color_store_ops) hash ^= std::hash<int>()(op) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        for (auto sw : key.is_swapchain) hash ^= std::hash<bool>()(sw) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        for (auto res : key.has_resolve) hash ^= std::hash<bool>()(res) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>()(key.depth_format) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>()(key.depth_load_op) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>()(key.depth_store_op) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>()(key.samples) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

// Global render pass cache (per device in a real implementation)
static std::unordered_map<RenderPassKey, VkRenderPass, RenderPassKeyHash> g_render_pass_cache;
static VkDevice g_cache_device = VK_NULL_HANDLE;

// Called from device destruction to clean up cached render passes
void cleanup_render_pass_cache(VkDevice device) {
    if (g_cache_device == device) {
        for (auto& [key, rp] : g_render_pass_cache) {
            vk::vkDestroyRenderPass(device, rp, nullptr);
        }
        g_render_pass_cache.clear();
        g_cache_device = VK_NULL_HANDLE;
    }
}

class VulkanCommandBuffer : public CommandBuffer {
public:
    VulkanCommandBuffer(VkDevice device, VkCommandBuffer cmd, std::vector<VkFramebuffer>&& framebuffers)
        : device_(device), cmd_(cmd), framebuffers_(std::move(framebuffers)) {}

    ~VulkanCommandBuffer() override {
        // Framebuffers are destroyed when command buffer is destroyed
        // (which should happen after GPU execution completes via fence wait)
        for (auto fb : framebuffers_) {
            vk::vkDestroyFramebuffer(device_, fb, nullptr);
        }
    }

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(cmd_); }

    VkCommandBuffer get_command_buffer() const { return cmd_; }

private:
    VkDevice device_;
    VkCommandBuffer cmd_;
    std::vector<VkFramebuffer> framebuffers_;  // Owned framebuffers - destroyed when command buffer is done
};

class VulkanRenderPassEncoder : public RenderPassEncoder {
public:
    VulkanRenderPassEncoder(VkCommandBuffer cmd, VkRenderPass render_pass,
                            VkFramebuffer framebuffer, uint32_t width, uint32_t height)
        : cmd_(cmd), render_pass_(render_pass), framebuffer_(framebuffer),
          width_(width), height_(height) {}

    void set_pipeline(RenderPipeline* pipeline) override;
    void set_bind_group(uint32_t index, BindGroup* bind_group) override;
    void set_vertex_buffer(uint32_t slot, Buffer* buffer, size_t offset = 0) override;
    void set_index_buffer(Buffer* buffer, IndexFormat format, size_t offset = 0) override;
    void set_viewport(float x, float y, float width, float height, float min_depth, float max_depth) override;
    void set_scissor(int32_t x, int32_t y, uint32_t width, uint32_t height) override;
    void set_blend_constant(float r, float g, float b, float a) override;
    void set_stencil_reference(uint32_t reference) override;

    void push_constants(const void* data, size_t size, size_t offset = 0) override;

    void draw(uint32_t vertex_count, uint32_t instance_count = 1,
              uint32_t first_vertex = 0, uint32_t first_instance = 0) override;
    void draw_indexed(uint32_t index_count, uint32_t instance_count = 1,
                      uint32_t first_index = 0, int32_t vertex_offset = 0,
                      uint32_t first_instance = 0) override;
    void draw_indirect(Buffer* buffer, size_t offset, uint32_t draw_count) override;
    void draw_indexed_indirect(Buffer* buffer, size_t offset, uint32_t draw_count) override;

    void end() override;

private:
    VkCommandBuffer cmd_;
    VkRenderPass render_pass_;
    VkFramebuffer framebuffer_;
    uint32_t width_;
    uint32_t height_;
    VkPipelineLayout current_layout_ = VK_NULL_HANDLE;
};

class VulkanComputePassEncoder : public ComputePassEncoder {
public:
    VulkanComputePassEncoder(VkCommandBuffer cmd) : cmd_(cmd) {}

    void set_pipeline(ComputePipeline* pipeline) override;
    void set_bind_group(uint32_t index, BindGroup* bind_group) override;
    void push_constants(const void* data, size_t size, size_t offset = 0) override;

    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override;
    void dispatch_indirect(Buffer* buffer, size_t offset) override;

    void end() override;

private:
    VkCommandBuffer cmd_;
    VkPipelineLayout current_layout_ = VK_NULL_HANDLE;
};

class VulkanCommandEncoder : public CommandEncoder {
public:
    VulkanCommandEncoder(VkDevice device, VkCommandPool pool)
        : device_(device), pool_(pool) {}

    ~VulkanCommandEncoder() override {
        // Framebuffers are transferred to the command buffer in finish()
        // Only clean up if finish() was never called (error path)
        for (auto fb : framebuffers_) {
            vk::vkDestroyFramebuffer(device_, fb, nullptr);
        }
    }

    bool initialize();

    std::unique_ptr<RenderPassEncoder> begin_render_pass(const RenderPassBeginInfo& info) override;
    std::unique_ptr<ComputePassEncoder> begin_compute_pass() override;

    void copy_buffer_to_buffer(Buffer* src, size_t src_offset,
                               Buffer* dst, size_t dst_offset, size_t size) override;
    void copy_buffer_to_texture(Buffer* src, Texture* dst, const BufferTextureCopy& copy) override;
    void copy_texture_to_buffer(Texture* src, Buffer* dst, const BufferTextureCopy& copy) override;
    void copy_texture_to_texture(Texture* src, Texture* dst, const TextureCopy& copy) override;

    std::unique_ptr<CommandBuffer> finish() override;

    VkCommandBuffer get_command_buffer() const { return cmd_; }

private:
    VkDevice device_;
    VkCommandPool pool_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;
};

bool VulkanCommandEncoder::initialize() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vk::vkAllocateCommandBuffers(device_, &alloc_info, &cmd_) != VK_SUCCESS) {
        std::cerr << "Failed to allocate command buffer" << std::endl;
        return false;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vk::vkBeginCommandBuffer(cmd_, &begin_info) != VK_SUCCESS) {
        std::cerr << "Failed to begin command buffer" << std::endl;
        return false;
    }

    return true;
}

// Helper to create or get cached render pass
static VkRenderPass get_or_create_render_pass(VkDevice device, const RenderPassBeginInfo& info) {
    // Build render pass key from attachment info
    RenderPassKey key;

    for (uint32_t i = 0; i < info.color_attachment_count; ++i) {
        auto* tex = info.color_attachments[i].texture;
        VkFormat fmt = to_vk_format_cmd(tex->get_format());
        key.color_formats.push_back(fmt);
        key.color_load_ops.push_back(to_vk_load_op(info.color_attachments[i].load_op));
        key.color_store_ops.push_back(to_vk_store_op(info.color_attachments[i].store_op));

        // Check if resolve texture is present (MSAA resolve)
        bool has_resolve = info.color_attachments[i].resolve_texture != nullptr;
        key.has_resolve.push_back(has_resolve);

        // Check if this is a swapchain image (usage includes RenderTarget but not Sampled typically)
        // For now, assume BGRA8_SRGB format means swapchain
        // If resolving, the resolve target determines the final layout
        auto* final_tex = has_resolve ? info.color_attachments[i].resolve_texture : tex;
        bool is_swapchain = (final_tex->get_format() == Format::BGRA8_SRGB || final_tex->get_format() == Format::BGRA8_UNORM);
        key.is_swapchain.push_back(is_swapchain);
        key.samples = tex->get_sample_count();
    }

    if (info.depth_attachment.texture) {
        key.depth_format = to_vk_format_cmd(info.depth_attachment.texture->get_format());
        key.depth_load_op = to_vk_load_op(info.depth_attachment.depth_load_op);
        key.depth_store_op = to_vk_store_op(info.depth_attachment.depth_store_op);
    }

    // Check cache
    if (g_cache_device != device) {
        // Device changed, clear cache
        for (auto& [k, rp] : g_render_pass_cache) {
            vk::vkDestroyRenderPass(g_cache_device, rp, nullptr);
        }
        g_render_pass_cache.clear();
        g_cache_device = device;
    }

    auto it = g_render_pass_cache.find(key);
    if (it != g_render_pass_cache.end()) {
        return it->second;
    }

    // Create new render pass
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> color_refs;
    std::vector<VkAttachmentReference> resolve_refs;
    bool has_any_resolve = false;

    for (uint32_t i = 0; i < info.color_attachment_count; ++i) {
        bool has_resolve = key.has_resolve[i];
        has_any_resolve = has_any_resolve || has_resolve;

        VkAttachmentDescription attachment{};
        attachment.format = key.color_formats[i];
        attachment.samples = static_cast<VkSampleCountFlagBits>(key.samples);
        attachment.loadOp = key.color_load_ops[i];
        // If resolving, MSAA attachment doesn't need to be stored
        attachment.storeOp = has_resolve ? VK_ATTACHMENT_STORE_OP_DONT_CARE : key.color_store_ops[i];
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        // Initial layout depends on loadOp and whether this is swapchain:
        // - LOAD + swapchain: must be PRESENT_SRC_KHR (comes from presentation)
        // - LOAD + non-swapchain: must be SHADER_READ_ONLY_OPTIMAL (was sampled) or COLOR_ATTACHMENT_OPTIMAL
        // - CLEAR/DONT_CARE: can be UNDEFINED (content discarded)
        if (key.color_load_ops[i] == VK_ATTACHMENT_LOAD_OP_LOAD) {
            // Swapchain images come from present, non-swapchain textures from shader sampling
            attachment.initialLayout = key.is_swapchain[i] ?
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        } else {
            attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // For MSAA with resolve, MSAA attachment stays as COLOR_ATTACHMENT_OPTIMAL
        // The resolve attachment gets the final layout
        if (has_resolve) {
            attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        } else if (key.is_swapchain[i]) {
            attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        } else {
            attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        attachments.push_back(attachment);

        VkAttachmentReference ref{};
        ref.attachment = static_cast<uint32_t>(attachments.size() - 1);
        ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_refs.push_back(ref);
    }

    // Add resolve attachments after all color attachments
    for (uint32_t i = 0; i < info.color_attachment_count; ++i) {
        if (key.has_resolve[i]) {
            // Resolve attachment (single-sample)
            VkAttachmentDescription resolve_attachment{};
            resolve_attachment.format = key.color_formats[i];  // Same format as MSAA
            resolve_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
            resolve_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;  // Will be overwritten by resolve
            resolve_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            resolve_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            resolve_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            resolve_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            resolve_attachment.finalLayout = key.is_swapchain[i] ?
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentReference resolve_ref{};
            resolve_ref.attachment = static_cast<uint32_t>(attachments.size());
            resolve_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            resolve_refs.push_back(resolve_ref);

            attachments.push_back(resolve_attachment);
        } else {
            // No resolve for this attachment
            VkAttachmentReference no_resolve_ref{};
            no_resolve_ref.attachment = VK_ATTACHMENT_UNUSED;
            no_resolve_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            resolve_refs.push_back(no_resolve_ref);
        }
    }

    // Depth attachment
    VkAttachmentReference depth_ref{};
    bool has_depth = key.depth_format != VK_FORMAT_UNDEFINED;
    if (has_depth) {
        VkAttachmentDescription depth_attachment{};
        depth_attachment.format = key.depth_format;
        depth_attachment.samples = static_cast<VkSampleCountFlagBits>(key.samples);
        depth_attachment.loadOp = key.depth_load_op;
        depth_attachment.storeOp = key.depth_store_op;
        depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        if (key.depth_load_op == VK_ATTACHMENT_LOAD_OP_LOAD) {
            depth_attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        } else {
            depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
        depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        depth_ref.attachment = static_cast<uint32_t>(attachments.size());
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depth_attachment);
    }

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(color_refs.size());
    subpass.pColorAttachments = color_refs.data();
    subpass.pResolveAttachments = has_any_resolve ? resolve_refs.data() : nullptr;
    subpass.pDepthStencilAttachment = has_depth ? &depth_ref : nullptr;

    // Subpass dependencies for proper synchronization
    std::vector<VkSubpassDependency> dependencies;

    // Dependency for render pass start (external -> subpass 0)
    VkSubpassDependency dep1{};
    dep1.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep1.dstSubpass = 0;
    dep1.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep1.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep1.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies.push_back(dep1);

    // Dependency for render pass end (subpass 0 -> external)
    // This ensures layout transitions and writes complete before subsequent shader reads
    VkSubpassDependency dep2{};
    dep2.srcSubpass = 0;
    dep2.dstSubpass = VK_SUBPASS_EXTERNAL;
    dep2.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep2.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep2.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dep2.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    dependencies.push_back(dep2);

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = static_cast<uint32_t>(dependencies.size());
    render_pass_info.pDependencies = dependencies.data();

    VkRenderPass render_pass;
    if (vk::vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
        std::cerr << "Failed to create dynamic render pass" << std::endl;
        return VK_NULL_HANDLE;
    }

    // Cache it
    g_render_pass_cache[key] = render_pass;
    return render_pass;
}

std::unique_ptr<RenderPassEncoder> VulkanCommandEncoder::begin_render_pass(const RenderPassBeginInfo& info) {
    if (!info.pipeline) {
        std::cerr << "Render pass requires a pipeline" << std::endl;
        return nullptr;
    }

    // Get dimensions from first color attachment
    uint32_t width = 0, height = 0;
    if (info.color_attachment_count > 0 && info.color_attachments[0].texture) {
        auto size = info.color_attachments[0].texture->get_size();
        width = size.width;
        height = size.height;
    }

    // Create render pass dynamically based on actual attachment loadOp/storeOp
    VkRenderPass render_pass = get_or_create_render_pass(device_, info);
    if (render_pass == VK_NULL_HANDLE) {
        std::cerr << "Failed to get/create render pass" << std::endl;
        return nullptr;
    }

    // Create framebuffer
    // Attachment order must match render pass: color attachments, then resolve attachments, then depth
    std::vector<VkImageView> attachments;
    for (uint32_t i = 0; i < info.color_attachment_count; ++i) {
        auto* tex = info.color_attachments[i].texture;
        attachments.push_back(reinterpret_cast<VkImageView>(tex->get_native_view()));
    }
    // Add resolve attachments (only if present)
    for (uint32_t i = 0; i < info.color_attachment_count; ++i) {
        auto* resolve_tex = info.color_attachments[i].resolve_texture;
        if (resolve_tex) {
            attachments.push_back(reinterpret_cast<VkImageView>(resolve_tex->get_native_view()));
        }
    }
    if (info.depth_attachment.texture) {
        attachments.push_back(reinterpret_cast<VkImageView>(info.depth_attachment.texture->get_native_view()));
    }

    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    fb_info.pAttachments = attachments.data();
    fb_info.width = width;
    fb_info.height = height;
    fb_info.layers = 1;

    VkFramebuffer framebuffer;
    if (vk::vkCreateFramebuffer(device_, &fb_info, nullptr, &framebuffer) != VK_SUCCESS) {
        std::cerr << "Failed to create framebuffer" << std::endl;
        return nullptr;
    }
    framebuffers_.push_back(framebuffer);

    // Begin render pass
    // Only include clear values for attachments that actually use CLEAR loadOp
    // Otherwise validation warns about "ClearValueWithoutLoadOpClear"
    std::vector<VkClearValue> clear_values;
    bool has_any_clear = false;

    for (uint32_t i = 0; i < info.color_attachment_count; ++i) {
        VkClearValue clear{};
        if (info.color_attachments[i].load_op == LoadOp::Clear) {
            clear.color = {{info.color_attachments[i].clear_color[0],
                            info.color_attachments[i].clear_color[1],
                            info.color_attachments[i].clear_color[2],
                            info.color_attachments[i].clear_color[3]}};
            has_any_clear = true;
        }
        clear_values.push_back(clear);
    }
    // Add placeholder clear values for resolve attachments (they use DONT_CARE but VK wants the array aligned)
    for (uint32_t i = 0; i < info.color_attachment_count; ++i) {
        if (info.color_attachments[i].resolve_texture) {
            VkClearValue clear{};
            clear.color = {{0, 0, 0, 0}};
            clear_values.push_back(clear);
        }
    }
    if (info.depth_attachment.texture) {
        VkClearValue clear{};
        if (info.depth_attachment.depth_load_op == LoadOp::Clear) {
            clear.depthStencil = {info.depth_attachment.clear_depth, info.depth_attachment.clear_stencil};
            has_any_clear = true;
        }
        clear_values.push_back(clear);
    }

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea.offset = {0, 0};
    rp_begin.renderArea.extent = {width, height};
    // Only provide clear values if at least one attachment uses CLEAR loadOp
    rp_begin.clearValueCount = has_any_clear ? static_cast<uint32_t>(clear_values.size()) : 0;
    rp_begin.pClearValues = has_any_clear ? clear_values.data() : nullptr;

    vk::vkCmdBeginRenderPass(cmd_, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    return std::make_unique<VulkanRenderPassEncoder>(cmd_, render_pass, framebuffer, width, height);
}

std::unique_ptr<ComputePassEncoder> VulkanCommandEncoder::begin_compute_pass() {
    return std::make_unique<VulkanComputePassEncoder>(cmd_);
}

void VulkanCommandEncoder::copy_buffer_to_buffer(Buffer* src, size_t src_offset,
                                                  Buffer* dst, size_t dst_offset, size_t size) {
    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size;

    vk::vkCmdCopyBuffer(cmd_,
        reinterpret_cast<VkBuffer>(src->get_native_handle()),
        reinterpret_cast<VkBuffer>(dst->get_native_handle()),
        1, &region);
}

void VulkanCommandEncoder::copy_buffer_to_texture(Buffer* src, Texture* dst, const BufferTextureCopy& copy) {
    VkBufferImageCopy region{};
    region.bufferOffset = copy.buffer_offset;
    region.bufferRowLength = copy.buffer_row_length;
    region.bufferImageHeight = copy.buffer_image_height;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = copy.texture_mip_level;
    region.imageSubresource.baseArrayLayer = copy.texture_array_layer;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {static_cast<int32_t>(copy.texture_offset.x),
                          static_cast<int32_t>(copy.texture_offset.y),
                          static_cast<int32_t>(copy.texture_offset.z)};
    region.imageExtent = {copy.copy_size.width, copy.copy_size.height, copy.copy_size.depth};

    vk::vkCmdCopyBufferToImage(cmd_,
        reinterpret_cast<VkBuffer>(src->get_native_handle()),
        reinterpret_cast<VkImage>(dst->get_native_handle()),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region);
}

void VulkanCommandEncoder::copy_texture_to_buffer(Texture* src, Buffer* dst, const BufferTextureCopy& copy) {
    VkBufferImageCopy region{};
    region.bufferOffset = copy.buffer_offset;
    region.bufferRowLength = copy.buffer_row_length;
    region.bufferImageHeight = copy.buffer_image_height;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = copy.texture_mip_level;
    region.imageSubresource.baseArrayLayer = copy.texture_array_layer;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {static_cast<int32_t>(copy.texture_offset.x),
                          static_cast<int32_t>(copy.texture_offset.y),
                          static_cast<int32_t>(copy.texture_offset.z)};
    region.imageExtent = {copy.copy_size.width, copy.copy_size.height, copy.copy_size.depth};

    vk::vkCmdCopyImageToBuffer(cmd_,
        reinterpret_cast<VkImage>(src->get_native_handle()),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        reinterpret_cast<VkBuffer>(dst->get_native_handle()),
        1, &region);
}

void VulkanCommandEncoder::copy_texture_to_texture(Texture* src, Texture* dst, const TextureCopy& copy) {
    VkImageCopy region{};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.mipLevel = copy.src_mip_level;
    region.srcSubresource.baseArrayLayer = copy.src_array_layer;
    region.srcSubresource.layerCount = 1;
    region.srcOffset = {static_cast<int32_t>(copy.src_offset.x),
                        static_cast<int32_t>(copy.src_offset.y),
                        static_cast<int32_t>(copy.src_offset.z)};
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.mipLevel = copy.dst_mip_level;
    region.dstSubresource.baseArrayLayer = copy.dst_array_layer;
    region.dstSubresource.layerCount = 1;
    region.dstOffset = {static_cast<int32_t>(copy.dst_offset.x),
                        static_cast<int32_t>(copy.dst_offset.y),
                        static_cast<int32_t>(copy.dst_offset.z)};
    region.extent = {copy.copy_size.width, copy.copy_size.height, copy.copy_size.depth};

    vk::vkCmdCopyImage(cmd_,
        reinterpret_cast<VkImage>(src->get_native_handle()),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        reinterpret_cast<VkImage>(dst->get_native_handle()),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region);
}

std::unique_ptr<CommandBuffer> VulkanCommandEncoder::finish() {
    if (vk::vkEndCommandBuffer(cmd_) != VK_SUCCESS) {
        std::cerr << "Failed to end command buffer" << std::endl;
        return nullptr;
    }

    // Transfer framebuffer ownership to the command buffer
    // They will be destroyed when the command buffer is destroyed (after fence wait)
    return std::make_unique<VulkanCommandBuffer>(device_, cmd_, std::move(framebuffers_));
}

// Render pass encoder implementation
void VulkanRenderPassEncoder::set_pipeline(RenderPipeline* pipeline) {
    VkPipeline vk_pipeline = reinterpret_cast<VkPipeline>(pipeline->get_native_handle());
    current_layout_ = reinterpret_cast<VkPipelineLayout>(pipeline->get_native_layout());
    vk::vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);
}

void VulkanRenderPassEncoder::set_bind_group(uint32_t index, BindGroup* bind_group) {
    VkDescriptorSet set = reinterpret_cast<VkDescriptorSet>(bind_group->get_native_handle());
    vk::vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
        current_layout_, index, 1, &set, 0, nullptr);
}

void VulkanRenderPassEncoder::set_vertex_buffer(uint32_t slot, Buffer* buffer, size_t offset) {
    VkBuffer vk_buffer = reinterpret_cast<VkBuffer>(buffer->get_native_handle());
    VkDeviceSize vk_offset = offset;
    vk::vkCmdBindVertexBuffers(cmd_, slot, 1, &vk_buffer, &vk_offset);
}

void VulkanRenderPassEncoder::set_index_buffer(Buffer* buffer, IndexFormat format, size_t offset) {
    VkBuffer vk_buffer = reinterpret_cast<VkBuffer>(buffer->get_native_handle());
    VkIndexType index_type = (format == IndexFormat::Uint16) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    vk::vkCmdBindIndexBuffer(cmd_, vk_buffer, offset, index_type);
}

void VulkanRenderPassEncoder::set_viewport(float x, float y, float width, float height, float min_depth, float max_depth) {
    VkViewport viewport{};
    viewport.x = x;
    viewport.y = y;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = min_depth;
    viewport.maxDepth = max_depth;
    vk::vkCmdSetViewport(cmd_, 0, 1, &viewport);
}

void VulkanRenderPassEncoder::set_scissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    VkRect2D scissor{};
    scissor.offset = {x, y};
    scissor.extent = {width, height};
    vk::vkCmdSetScissor(cmd_, 0, 1, &scissor);
}

void VulkanRenderPassEncoder::set_blend_constant(float r, float g, float b, float a) {
    float constants[4] = {r, g, b, a};
    vk::vkCmdSetBlendConstants(cmd_, constants);
}

void VulkanRenderPassEncoder::set_stencil_reference(uint32_t reference) {
    vk::vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, reference);
}

void VulkanRenderPassEncoder::push_constants(const void* data, size_t size, size_t offset) {
    vk::vkCmdPushConstants(cmd_, current_layout_,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        static_cast<uint32_t>(offset), static_cast<uint32_t>(size), data);
}

void VulkanRenderPassEncoder::draw(uint32_t vertex_count, uint32_t instance_count,
                                    uint32_t first_vertex, uint32_t first_instance) {
    vk::vkCmdDraw(cmd_, vertex_count, instance_count, first_vertex, first_instance);
}

void VulkanRenderPassEncoder::draw_indexed(uint32_t index_count, uint32_t instance_count,
                                            uint32_t first_index, int32_t vertex_offset,
                                            uint32_t first_instance) {
    vk::vkCmdDrawIndexed(cmd_, index_count, instance_count, first_index, vertex_offset, first_instance);
}

void VulkanRenderPassEncoder::draw_indirect(Buffer* buffer, size_t offset, uint32_t draw_count) {
    VkBuffer vk_buffer = reinterpret_cast<VkBuffer>(buffer->get_native_handle());
    vk::vkCmdDrawIndirect(cmd_, vk_buffer, offset, draw_count, sizeof(VkDrawIndirectCommand));
}

void VulkanRenderPassEncoder::draw_indexed_indirect(Buffer* buffer, size_t offset, uint32_t draw_count) {
    VkBuffer vk_buffer = reinterpret_cast<VkBuffer>(buffer->get_native_handle());
    vk::vkCmdDrawIndexedIndirect(cmd_, vk_buffer, offset, draw_count, sizeof(VkDrawIndexedIndirectCommand));
}

void VulkanRenderPassEncoder::end() {
    vk::vkCmdEndRenderPass(cmd_);
}

// Compute pass encoder implementation
void VulkanComputePassEncoder::set_pipeline(ComputePipeline* pipeline) {
    VkPipeline vk_pipeline = reinterpret_cast<VkPipeline>(pipeline->get_native_handle());
    current_layout_ = reinterpret_cast<VkPipelineLayout>(pipeline->get_native_layout());
    vk::vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);
}

void VulkanComputePassEncoder::set_bind_group(uint32_t index, BindGroup* bind_group) {
    VkDescriptorSet set = reinterpret_cast<VkDescriptorSet>(bind_group->get_native_handle());
    vk::vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
        current_layout_, index, 1, &set, 0, nullptr);
}

void VulkanComputePassEncoder::push_constants(const void* data, size_t size, size_t offset) {
    vk::vkCmdPushConstants(cmd_, current_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
        static_cast<uint32_t>(offset), static_cast<uint32_t>(size), data);
}

void VulkanComputePassEncoder::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    vk::vkCmdDispatch(cmd_, x, y, z);
}

void VulkanComputePassEncoder::dispatch_indirect(Buffer* buffer, size_t offset) {
    VkBuffer vk_buffer = reinterpret_cast<VkBuffer>(buffer->get_native_handle());
    vk::vkCmdDispatchIndirect(cmd_, vk_buffer, offset);
}

void VulkanComputePassEncoder::end() {
    // Compute passes don't have an explicit end in Vulkan
}

// Factory function used by VulkanDevice
std::unique_ptr<CommandEncoder> create_vulkan_command_encoder(VkDevice device, VkCommandPool pool) {
    auto encoder = std::make_unique<VulkanCommandEncoder>(device, pool);
    if (encoder->initialize()) {
        return encoder;
    }
    return nullptr;
}

} // namespace viz::gal
