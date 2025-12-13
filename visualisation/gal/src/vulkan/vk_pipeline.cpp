#include "vk_internal.hpp"

#include <array>
#include <iostream>
#include <vector>
#include <cstring>

namespace viz::gal {

// Shader module implementation
class VulkanShader : public Shader {
public:
    VulkanShader(VkDevice device) : device_(device) {}
    ~VulkanShader() override {
        if (module_ != VK_NULL_HANDLE) {
            vk::vkDestroyShaderModule(device_, module_, nullptr);
        }
    }

    bool initialize(const ShaderDesc& desc);

    ShaderStage get_stage() const override { return stage_; }
    Handle get_native_handle() const override { return reinterpret_cast<Handle>(module_); }

    VkShaderModule get_module() const { return module_; }
    const char* get_entry_point() const { return entry_point_.c_str(); }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
    ShaderStage stage_ = ShaderStage::Vertex;
    std::string entry_point_ = "main";
};

bool VulkanShader::initialize(const ShaderDesc& desc) {
    stage_ = desc.stage;
    entry_point_ = desc.entry_point ? desc.entry_point : "main";

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = desc.spirv_size;
    create_info.pCode = desc.spirv_code;

    if (vk::vkCreateShaderModule(device_, &create_info, nullptr, &module_) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module" << std::endl;
        return false;
    }

    return true;
}

// VulkanBindGroupLayout is defined in vk_internal.hpp

static VkDescriptorType to_vk_descriptor_type(BindingType type) {
    switch (type) {
        case BindingType::UniformBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case BindingType::StorageBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case BindingType::SampledTexture: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case BindingType::StorageTexture: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case BindingType::Sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
        case BindingType::CombinedTextureSampler: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        default: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    }
}

static VkShaderStageFlags to_vk_shader_stage_flags(ShaderStage stage) {
    VkShaderStageFlags flags = 0;
    if (static_cast<uint32_t>(stage) & static_cast<uint32_t>(ShaderStage::Vertex))
        flags |= VK_SHADER_STAGE_VERTEX_BIT;
    if (static_cast<uint32_t>(stage) & static_cast<uint32_t>(ShaderStage::Fragment))
        flags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    if (static_cast<uint32_t>(stage) & static_cast<uint32_t>(ShaderStage::Compute))
        flags |= VK_SHADER_STAGE_COMPUTE_BIT;
    return flags ? flags : VK_SHADER_STAGE_ALL;
}

bool VulkanBindGroupLayout::initialize(const BindGroupLayoutDesc& desc) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(desc.entry_count);

    for (uint32_t i = 0; i < desc.entry_count; ++i) {
        const auto& entry = desc.entries[i];
        auto& binding = bindings[i];

        binding.binding = entry.binding;
        binding.descriptorType = to_vk_descriptor_type(entry.type);
        binding.descriptorCount = entry.count;
        binding.stageFlags = to_vk_shader_stage_flags(entry.visibility);
        binding.pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vk::vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout" << std::endl;
        return false;
    }

    return true;
}

// Render pipeline implementation
class VulkanRenderPipeline : public RenderPipeline {
public:
    VulkanRenderPipeline(VkDevice device, VkPipelineCache cache = VK_NULL_HANDLE)
        : device_(device), pipeline_cache_(cache) {}
    ~VulkanRenderPipeline() override {
        if (pipeline_ != VK_NULL_HANDLE) {
            vk::vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        if (layout_ != VK_NULL_HANDLE) {
            vk::vkDestroyPipelineLayout(device_, layout_, nullptr);
        }
        if (render_pass_ != VK_NULL_HANDLE) {
            vk::vkDestroyRenderPass(device_, render_pass_, nullptr);
        }
    }

    bool initialize(const RenderPipelineDesc& desc);

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(pipeline_); }
    Handle get_native_layout() const override { return reinterpret_cast<Handle>(layout_); }

    VkPipeline get_pipeline() const { return pipeline_; }
    VkPipelineLayout get_layout() const { return layout_; }
    VkRenderPass get_render_pass() const { return render_pass_; }

private:
    bool create_render_pass(const RenderPipelineDesc& desc);
    bool create_pipeline_layout(const RenderPipelineDesc& desc);
    bool create_pipeline(const RenderPipelineDesc& desc);

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
};

static VkFormat to_vk_format(Format format);  // Forward declaration

static VkFormat to_vk_format(Format format) {
    switch (format) {
        case Format::R8_UNORM: return VK_FORMAT_R8_UNORM;
        case Format::R8_SNORM: return VK_FORMAT_R8_SNORM;
        case Format::R8_UINT: return VK_FORMAT_R8_UINT;
        case Format::R8_SINT: return VK_FORMAT_R8_SINT;
        case Format::R16_UINT: return VK_FORMAT_R16_UINT;
        case Format::R16_SINT: return VK_FORMAT_R16_SINT;
        case Format::R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
        case Format::RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
        case Format::RG8_SNORM: return VK_FORMAT_R8G8_SNORM;
        case Format::RG8_UINT: return VK_FORMAT_R8G8_UINT;
        case Format::RG8_SINT: return VK_FORMAT_R8G8_SINT;
        case Format::R32_UINT: return VK_FORMAT_R32_UINT;
        case Format::R32_SINT: return VK_FORMAT_R32_SINT;
        case Format::R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case Format::RG16_UINT: return VK_FORMAT_R16G16_UINT;
        case Format::RG16_SINT: return VK_FORMAT_R16G16_SINT;
        case Format::RG16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
        case Format::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
        case Format::RGBA8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
        case Format::RGBA8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
        case Format::RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case Format::BGRA8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case Format::BGRA8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        case Format::RG32_UINT: return VK_FORMAT_R32G32_UINT;
        case Format::RG32_SINT: return VK_FORMAT_R32G32_SINT;
        case Format::RG32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case Format::RGB32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
        case Format::RGBA16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
        case Format::RGBA16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
        case Format::RGBA16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case Format::RGBA32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
        case Format::RGBA32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
        case Format::RGBA32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case Format::D16_UNORM: return VK_FORMAT_D16_UNORM;
        case Format::D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
        case Format::D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
        case Format::D32_FLOAT_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

static VkPrimitiveTopology to_vk_topology(PrimitiveTopology topology) {
    switch (topology) {
        case PrimitiveTopology::PointList: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case PrimitiveTopology::LineList: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case PrimitiveTopology::LineStrip: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case PrimitiveTopology::TriangleList: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case PrimitiveTopology::TriangleStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

static VkPolygonMode to_vk_polygon_mode(PolygonMode mode) {
    switch (mode) {
        case PolygonMode::Fill: return VK_POLYGON_MODE_FILL;
        case PolygonMode::Line: return VK_POLYGON_MODE_LINE;
        case PolygonMode::Point: return VK_POLYGON_MODE_POINT;
        default: return VK_POLYGON_MODE_FILL;
    }
}

static VkCullModeFlags to_vk_cull_mode(CullMode mode) {
    switch (mode) {
        case CullMode::None: return VK_CULL_MODE_NONE;
        case CullMode::Front: return VK_CULL_MODE_FRONT_BIT;
        case CullMode::Back: return VK_CULL_MODE_BACK_BIT;
        default: return VK_CULL_MODE_NONE;
    }
}

static VkFrontFace to_vk_front_face(FrontFace face) {
    switch (face) {
        case FrontFace::CCW: return VK_FRONT_FACE_COUNTER_CLOCKWISE;
        case FrontFace::CW: return VK_FRONT_FACE_CLOCKWISE;
        default: return VK_FRONT_FACE_COUNTER_CLOCKWISE;
    }
}

static VkCompareOp to_vk_compare_op(CompareFunc func) {
    switch (func) {
        case CompareFunc::Never: return VK_COMPARE_OP_NEVER;
        case CompareFunc::Less: return VK_COMPARE_OP_LESS;
        case CompareFunc::Equal: return VK_COMPARE_OP_EQUAL;
        case CompareFunc::LessEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case CompareFunc::Greater: return VK_COMPARE_OP_GREATER;
        case CompareFunc::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
        case CompareFunc::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case CompareFunc::Always: return VK_COMPARE_OP_ALWAYS;
        default: return VK_COMPARE_OP_ALWAYS;
    }
}

static VkBlendFactor to_vk_blend_factor(BlendFactor factor) {
    switch (factor) {
        case BlendFactor::Zero: return VK_BLEND_FACTOR_ZERO;
        case BlendFactor::One: return VK_BLEND_FACTOR_ONE;
        case BlendFactor::SrcColor: return VK_BLEND_FACTOR_SRC_COLOR;
        case BlendFactor::OneMinusSrcColor: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case BlendFactor::DstColor: return VK_BLEND_FACTOR_DST_COLOR;
        case BlendFactor::OneMinusDstColor: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case BlendFactor::SrcAlpha: return VK_BLEND_FACTOR_SRC_ALPHA;
        case BlendFactor::OneMinusSrcAlpha: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case BlendFactor::DstAlpha: return VK_BLEND_FACTOR_DST_ALPHA;
        case BlendFactor::OneMinusDstAlpha: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case BlendFactor::ConstantColor: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case BlendFactor::OneMinusConstantColor: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case BlendFactor::SrcAlphaSaturate: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

static VkBlendOp to_vk_blend_op(BlendOp op) {
    switch (op) {
        case BlendOp::Add: return VK_BLEND_OP_ADD;
        case BlendOp::Subtract: return VK_BLEND_OP_SUBTRACT;
        case BlendOp::ReverseSubtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case BlendOp::Min: return VK_BLEND_OP_MIN;
        case BlendOp::Max: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

static VkVertexInputRate to_vk_input_rate(VertexStepMode mode) {
    switch (mode) {
        case VertexStepMode::Vertex: return VK_VERTEX_INPUT_RATE_VERTEX;
        case VertexStepMode::Instance: return VK_VERTEX_INPUT_RATE_INSTANCE;
        default: return VK_VERTEX_INPUT_RATE_VERTEX;
    }
}

bool VulkanRenderPipeline::create_render_pass(const RenderPipelineDesc& desc) {
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> color_refs;

    // Color attachments
    for (uint32_t i = 0; i < desc.color_format_count; ++i) {
        VkAttachmentDescription attachment{};
        attachment.format = to_vk_format(desc.color_formats[i]);
        attachment.samples = static_cast<VkSampleCountFlagBits>(desc.multisample.count);
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        // Use SHADER_READ_ONLY for render-to-texture, PRESENT_SRC for swapchain
        attachment.finalLayout = desc.render_to_texture ?
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        attachments.push_back(attachment);

        VkAttachmentReference ref{};
        ref.attachment = i;
        ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_refs.push_back(ref);
    }

    // Depth attachment
    VkAttachmentReference depth_ref{};
    bool has_depth = desc.depth_format != Format::Undefined;
    if (has_depth) {
        VkAttachmentDescription depth_attachment{};
        depth_attachment.format = to_vk_format(desc.depth_format);
        depth_attachment.samples = static_cast<VkSampleCountFlagBits>(desc.multisample.count);
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        depth_ref.attachment = static_cast<uint32_t>(attachments.size());
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depth_attachment);
    }

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(color_refs.size());
    subpass.pColorAttachments = color_refs.data();
    subpass.pDepthStencilAttachment = has_depth ? &depth_ref : nullptr;

    // Subpass dependencies must match what's in vk_command_buffer.cpp for render pass compatibility
    std::array<VkSubpassDependency, 2> dependencies{};

    // Dependency 1: External -> subpass 0 (for render pass begin)
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // Dependency 2: subpass 0 -> External (for render pass end)
    // Ensures layout transitions and writes complete before subsequent shader reads
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = static_cast<uint32_t>(dependencies.size());
    render_pass_info.pDependencies = dependencies.data();

    if (vk::vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS) {
        std::cerr << "Failed to create render pass" << std::endl;
        return false;
    }

    return true;
}

bool VulkanRenderPipeline::create_pipeline_layout(const RenderPipelineDesc& desc) {
    std::vector<VkDescriptorSetLayout> set_layouts;
    for (uint32_t i = 0; i < desc.bind_group_layout_count; ++i) {
        auto* layout = static_cast<VulkanBindGroupLayout*>(const_cast<BindGroupLayout*>(desc.bind_group_layouts[i]));
        set_layouts.push_back(layout->get_layout());
    }

    VkPushConstantRange push_constant{};
    push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push_constant.offset = 0;
    push_constant.size = desc.push_constant_size;

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
    layout_info.pSetLayouts = set_layouts.data();
    layout_info.pushConstantRangeCount = desc.push_constant_size > 0 ? 1 : 0;
    layout_info.pPushConstantRanges = desc.push_constant_size > 0 ? &push_constant : nullptr;

    if (vk::vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        return false;
    }

    return true;
}

bool VulkanRenderPipeline::create_pipeline(const RenderPipelineDesc& desc) {
    // Shader stages
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;

    if (desc.vertex_shader) {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = static_cast<VulkanShader*>(desc.vertex_shader)->get_module();
        stage.pName = static_cast<VulkanShader*>(desc.vertex_shader)->get_entry_point();
        shader_stages.push_back(stage);
    }

    if (desc.fragment_shader) {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage.module = static_cast<VulkanShader*>(desc.fragment_shader)->get_module();
        stage.pName = static_cast<VulkanShader*>(desc.fragment_shader)->get_entry_point();
        shader_stages.push_back(stage);
    }

    // Vertex input
    std::vector<VkVertexInputBindingDescription> binding_descs;
    std::vector<VkVertexInputAttributeDescription> attrib_descs;

    for (uint32_t i = 0; i < desc.vertex_layout_count; ++i) {
        const auto& layout = desc.vertex_layouts[i];

        VkVertexInputBindingDescription binding{};
        binding.binding = i;
        binding.stride = layout.stride;
        binding.inputRate = to_vk_input_rate(layout.step_mode);
        binding_descs.push_back(binding);

        for (uint32_t j = 0; j < layout.attribute_count; ++j) {
            const auto& attr = layout.attributes[j];
            VkVertexInputAttributeDescription attrib{};
            attrib.location = attr.shader_location;
            attrib.binding = i;
            attrib.format = to_vk_format(attr.format);
            attrib.offset = attr.offset;
            attrib_descs.push_back(attrib);
        }
    }

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = static_cast<uint32_t>(binding_descs.size());
    vertex_input.pVertexBindingDescriptions = binding_descs.data();
    vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrib_descs.size());
    vertex_input.pVertexAttributeDescriptions = attrib_descs.data();

    // Input assembly
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = to_vk_topology(desc.topology);
    input_assembly.primitiveRestartEnable = VK_FALSE;

    // Viewport/scissor - dynamic state
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = desc.rasterizer.depth_clamp_enable;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = to_vk_polygon_mode(desc.rasterizer.polygon_mode);
    rasterizer.lineWidth = desc.rasterizer.line_width;
    rasterizer.cullMode = to_vk_cull_mode(desc.rasterizer.cull_mode);
    rasterizer.frontFace = to_vk_front_face(desc.rasterizer.front_face);
    rasterizer.depthBiasEnable = desc.rasterizer.depth_bias_enable;
    rasterizer.depthBiasConstantFactor = desc.rasterizer.depth_bias_constant;
    rasterizer.depthBiasSlopeFactor = desc.rasterizer.depth_bias_slope;
    rasterizer.depthBiasClamp = desc.rasterizer.depth_bias_clamp;

    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = desc.multisample.sample_shading_enable;
    multisampling.rasterizationSamples = static_cast<VkSampleCountFlagBits>(desc.multisample.count);
    multisampling.minSampleShading = desc.multisample.min_sample_shading;
    multisampling.pSampleMask = desc.multisample.sample_mask ? &desc.multisample.sample_mask : nullptr;
    multisampling.alphaToCoverageEnable = desc.multisample.alpha_to_coverage_enable;
    multisampling.alphaToOneEnable = VK_FALSE;

    // Depth/stencil
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = desc.depth_stencil.depth_test_enable;
    depth_stencil.depthWriteEnable = desc.depth_stencil.depth_write_enable;
    depth_stencil.depthCompareOp = to_vk_compare_op(desc.depth_stencil.depth_compare);
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = desc.depth_stencil.stencil_test_enable;

    // Color blending
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(desc.blend_state_count);
    for (uint32_t i = 0; i < desc.blend_state_count; ++i) {
        const auto& blend = desc.blend_states[i];
        auto& attachment = blend_attachments[i];

        attachment.blendEnable = blend.blend_enable;
        attachment.srcColorBlendFactor = to_vk_blend_factor(blend.src_color);
        attachment.dstColorBlendFactor = to_vk_blend_factor(blend.dst_color);
        attachment.colorBlendOp = to_vk_blend_op(blend.color_op);
        attachment.srcAlphaBlendFactor = to_vk_blend_factor(blend.src_alpha);
        attachment.dstAlphaBlendFactor = to_vk_blend_factor(blend.dst_alpha);
        attachment.alphaBlendOp = to_vk_blend_op(blend.alpha_op);
        attachment.colorWriteMask = blend.write_mask;
    }

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.attachmentCount = static_cast<uint32_t>(blend_attachments.size());
    color_blending.pAttachments = blend_attachments.data();

    // Dynamic state
    std::vector<VkDynamicState> dynamic_states = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
    dynamic_state.pDynamicStates = dynamic_states.data();

    // Create pipeline
    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = layout_;
    pipeline_info.renderPass = render_pass_;
    pipeline_info.subpass = 0;

    if (vk::vkCreateGraphicsPipelines(device_, pipeline_cache_, 1, &pipeline_info, nullptr, &pipeline_) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline" << std::endl;
        return false;
    }

    return true;
}

bool VulkanRenderPipeline::initialize(const RenderPipelineDesc& desc) {
    if (!create_render_pass(desc)) return false;
    if (!create_pipeline_layout(desc)) return false;
    if (!create_pipeline(desc)) return false;
    return true;
}

// Compute pipeline implementation
class VulkanComputePipeline : public ComputePipeline {
public:
    VulkanComputePipeline(VkDevice device) : device_(device) {}
    ~VulkanComputePipeline() override {
        if (pipeline_ != VK_NULL_HANDLE) {
            vk::vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        if (layout_ != VK_NULL_HANDLE) {
            vk::vkDestroyPipelineLayout(device_, layout_, nullptr);
        }
    }

    bool initialize(const ComputePipelineDesc& desc);

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(pipeline_); }
    Handle get_native_layout() const override { return reinterpret_cast<Handle>(layout_); }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

bool VulkanComputePipeline::initialize(const ComputePipelineDesc& desc) {
    // Create pipeline layout
    std::vector<VkDescriptorSetLayout> set_layouts;
    for (uint32_t i = 0; i < desc.bind_group_layout_count; ++i) {
        auto* layout = static_cast<VulkanBindGroupLayout*>(const_cast<BindGroupLayout*>(desc.bind_group_layouts[i]));
        set_layouts.push_back(layout->get_layout());
    }

    VkPushConstantRange push_constant{};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = desc.push_constant_size;

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
    layout_info.pSetLayouts = set_layouts.data();
    layout_info.pushConstantRangeCount = desc.push_constant_size > 0 ? 1 : 0;
    layout_info.pPushConstantRanges = desc.push_constant_size > 0 ? &push_constant : nullptr;

    if (vk::vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline layout" << std::endl;
        return false;
    }

    // Create pipeline
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = static_cast<VulkanShader*>(desc.compute_shader)->get_module();
    stage.pName = static_cast<VulkanShader*>(desc.compute_shader)->get_entry_point();

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = layout_;

    if (vk::vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        return false;
    }

    return true;
}

// Factory functions used by VulkanDevice
std::unique_ptr<Shader> create_vulkan_shader(VkDevice device, const ShaderDesc& desc) {
    auto shader = std::make_unique<VulkanShader>(device);
    if (shader->initialize(desc)) {
        return shader;
    }
    return nullptr;
}

std::unique_ptr<BindGroupLayout> create_vulkan_bind_group_layout(VkDevice device, const BindGroupLayoutDesc& desc) {
    auto layout = std::make_unique<VulkanBindGroupLayout>(device);
    if (layout->initialize(desc)) {
        return layout;
    }
    return nullptr;
}

std::unique_ptr<RenderPipeline> create_vulkan_render_pipeline(VkDevice device, VkPipelineCache cache, const RenderPipelineDesc& desc) {
    auto pipeline = std::make_unique<VulkanRenderPipeline>(device, cache);
    if (pipeline->initialize(desc)) {
        return pipeline;
    }
    return nullptr;
}

std::unique_ptr<ComputePipeline> create_vulkan_compute_pipeline(VkDevice device, const ComputePipelineDesc& desc) {
    auto pipeline = std::make_unique<VulkanComputePipeline>(device);
    if (pipeline->initialize(desc)) {
        return pipeline;
    }
    return nullptr;
}

// Helper function to get render pass from pipeline (used by vk_command_buffer.cpp)
VkRenderPass get_vk_render_pass(RenderPipeline* pipeline) {
    return static_cast<VulkanRenderPipeline*>(pipeline)->get_render_pass();
}

} // namespace viz::gal
