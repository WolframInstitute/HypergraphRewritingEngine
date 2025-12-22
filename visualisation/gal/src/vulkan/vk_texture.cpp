// Vulkan texture, sampler, and bind group implementation

#include "vk_internal.hpp"
#include <gal/types.hpp>  // for format_size
#include <iostream>
#include <cstring>  // for memcpy

namespace viz::gal {

// Format conversion helper
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

static VkImageUsageFlags to_vk_image_usage(TextureUsage usage) {
    VkImageUsageFlags flags = 0;
    if (has_flag(usage, TextureUsage::Sampled)) flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (has_flag(usage, TextureUsage::Storage)) flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (has_flag(usage, TextureUsage::RenderTarget)) flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (has_flag(usage, TextureUsage::DepthStencil)) flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (has_flag(usage, TextureUsage::TransferSrc)) flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (has_flag(usage, TextureUsage::TransferDst)) flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return flags;
}

static VkImageType to_vk_image_type(TextureDimension dim) {
    switch (dim) {
        case TextureDimension::Tex1D: return VK_IMAGE_TYPE_1D;
        case TextureDimension::Tex2D: return VK_IMAGE_TYPE_2D;
        case TextureDimension::Tex3D: return VK_IMAGE_TYPE_3D;
        case TextureDimension::Cube: return VK_IMAGE_TYPE_2D;
        default: return VK_IMAGE_TYPE_2D;
    }
}

static VkImageViewType to_vk_image_view_type(TextureDimension dim, uint32_t array_layers) {
    switch (dim) {
        case TextureDimension::Tex1D:
            return array_layers > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
        case TextureDimension::Tex2D:
            return array_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        case TextureDimension::Tex3D:
            return VK_IMAGE_VIEW_TYPE_3D;
        case TextureDimension::Cube:
            return array_layers > 6 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
        default:
            return VK_IMAGE_VIEW_TYPE_2D;
    }
}

static VkImageAspectFlags get_aspect_flags(Format format) {
    if (is_depth_format(format)) {
        if (has_stencil(format)) {
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

static uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter,
                                  VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vk::vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

std::unique_ptr<Texture> create_vulkan_texture(VkDevice device, VkPhysicalDevice physical_device,
                                               const TextureDesc& desc,
                                               VkCommandPool cmd_pool,
                                               VkQueue graphics_queue) {
    VkFormat vk_format = to_vk_format(desc.format);
    if (vk_format == VK_FORMAT_UNDEFINED) {
        std::cerr << "Invalid texture format" << std::endl;
        return nullptr;
    }

    // Create image
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = to_vk_image_type(desc.dimension);
    image_info.format = vk_format;
    image_info.extent.width = desc.size.width;
    image_info.extent.height = desc.size.height;
    image_info.extent.depth = desc.size.depth;
    image_info.mipLevels = desc.mip_levels;
    image_info.arrayLayers = desc.array_layers;
    image_info.samples = static_cast<VkSampleCountFlagBits>(desc.sample_count);
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = to_vk_image_usage(desc.usage);

    // Add transfer dst if we have initial data to upload
    if (desc.initial_data && cmd_pool != VK_NULL_HANDLE && graphics_queue != VK_NULL_HANDLE) {
        image_info.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (desc.dimension == TextureDimension::Cube) {
        image_info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }

    VkImage image;
    if (vk::vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan image" << std::endl;
        return nullptr;
    }

    // Allocate memory
    VkMemoryRequirements mem_reqs;
    vk::vkGetImageMemoryRequirements(device, image, &mem_reqs);

    uint32_t mem_type = find_memory_type(physical_device, mem_reqs.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type == UINT32_MAX) {
        std::cerr << "Failed to find suitable memory type for texture" << std::endl;
        vk::vkDestroyImage(device, image, nullptr);
        return nullptr;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type;

    // Add memory priority extension info if supported (reduces validation warnings on NVIDIA)
    // Higher priority (0.8) for render targets/depth buffers, normal (0.5) for regular textures
    VkMemoryPriorityAllocateInfoEXT priority_info{};
    priority_info.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT;
    bool is_attachment = has_flag(desc.usage, TextureUsage::RenderTarget) ||
                         has_flag(desc.usage, TextureUsage::DepthStencil);
    priority_info.priority = is_attachment ? 0.8f : 0.5f;
    alloc_info.pNext = &priority_info;

    VkDeviceMemory memory;
    if (vk::vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate texture memory" << std::endl;
        vk::vkDestroyImage(device, image, nullptr);
        return nullptr;
    }

    if (vk::vkBindImageMemory(device, image, memory, 0) != VK_SUCCESS) {
        std::cerr << "Failed to bind texture memory" << std::endl;
        vk::vkFreeMemory(device, memory, nullptr);
        vk::vkDestroyImage(device, image, nullptr);
        return nullptr;
    }

    // Create image view
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = to_vk_image_view_type(desc.dimension, desc.array_layers);
    view_info.format = vk_format;
    view_info.subresourceRange.aspectMask = get_aspect_flags(desc.format);
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = desc.mip_levels;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = desc.array_layers;

    VkImageView view;
    if (vk::vkCreateImageView(device, &view_info, nullptr, &view) != VK_SUCCESS) {
        std::cerr << "Failed to create texture image view" << std::endl;
        vk::vkFreeMemory(device, memory, nullptr);
        vk::vkDestroyImage(device, image, nullptr);
        return nullptr;
    }

    // Upload initial data using staging buffer if provided
    if (desc.initial_data && cmd_pool != VK_NULL_HANDLE && graphics_queue != VK_NULL_HANDLE) {
        // Calculate data size
        VkDeviceSize data_size = desc.size.width * desc.size.height * desc.size.depth *
                                 desc.array_layers * format_size(desc.format);

        // Create staging buffer
        VkBufferCreateInfo staging_buffer_info{};
        staging_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_buffer_info.size = data_size;
        staging_buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        staging_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer staging_buffer;
        if (vk::vkCreateBuffer(device, &staging_buffer_info, nullptr, &staging_buffer) != VK_SUCCESS) {
            std::cerr << "Failed to create staging buffer for texture upload" << std::endl;
            // Continue without initial data - texture is still valid
        } else {
            VkMemoryRequirements staging_mem_reqs;
            vk::vkGetBufferMemoryRequirements(device, staging_buffer, &staging_mem_reqs);

            uint32_t staging_mem_type = find_memory_type(physical_device, staging_mem_reqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            if (staging_mem_type == UINT32_MAX) {
                std::cerr << "Failed to find host-visible memory for staging buffer" << std::endl;
                vk::vkDestroyBuffer(device, staging_buffer, nullptr);
            } else {
                VkMemoryAllocateInfo staging_alloc_info{};
                staging_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                staging_alloc_info.allocationSize = staging_mem_reqs.size;
                staging_alloc_info.memoryTypeIndex = staging_mem_type;

                VkDeviceMemory staging_memory;
                if (vk::vkAllocateMemory(device, &staging_alloc_info, nullptr, &staging_memory) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate staging memory" << std::endl;
                    vk::vkDestroyBuffer(device, staging_buffer, nullptr);
                } else {
                    vk::vkBindBufferMemory(device, staging_buffer, staging_memory, 0);

                    // Copy data to staging buffer
                    void* mapped;
                    vk::vkMapMemory(device, staging_memory, 0, data_size, 0, &mapped);
                    memcpy(mapped, desc.initial_data, data_size);
                    vk::vkUnmapMemory(device, staging_memory);

                    // Create one-time command buffer
                    VkCommandBufferAllocateInfo cmd_alloc_info{};
                    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                    cmd_alloc_info.commandPool = cmd_pool;
                    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                    cmd_alloc_info.commandBufferCount = 1;

                    VkCommandBuffer cmd;
                    vk::vkAllocateCommandBuffers(device, &cmd_alloc_info, &cmd);

                    VkCommandBufferBeginInfo begin_info{};
                    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    vk::vkBeginCommandBuffer(cmd, &begin_info);

                    // Transition image to transfer-dst layout
                    VkImageMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.image = image;
                    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    barrier.subresourceRange.baseMipLevel = 0;
                    barrier.subresourceRange.levelCount = desc.mip_levels;
                    barrier.subresourceRange.baseArrayLayer = 0;
                    barrier.subresourceRange.layerCount = desc.array_layers;
                    barrier.srcAccessMask = 0;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                    vk::vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &barrier);

                    // Copy buffer to image
                    VkBufferImageCopy region{};
                    region.bufferOffset = 0;
                    region.bufferRowLength = 0;
                    region.bufferImageHeight = 0;
                    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    region.imageSubresource.mipLevel = 0;
                    region.imageSubresource.baseArrayLayer = 0;
                    region.imageSubresource.layerCount = desc.array_layers;
                    region.imageOffset = {0, 0, 0};
                    region.imageExtent = {desc.size.width, desc.size.height, desc.size.depth};

                    vk::vkCmdCopyBufferToImage(cmd, staging_buffer, image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                    // Transition image to shader-read layout
                    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                    vk::vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &barrier);

                    vk::vkEndCommandBuffer(cmd);

                    // Submit and wait
                    VkSubmitInfo submit_info{};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &cmd;

                    vk::vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
                    vk::vkQueueWaitIdle(graphics_queue);

                    // Cleanup
                    vk::vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
                    vk::vkDestroyBuffer(device, staging_buffer, nullptr);
                    vk::vkFreeMemory(device, staging_memory, nullptr);
                }
            }
        }
    }

    return std::make_unique<VulkanTexture>(device, image, memory, view, desc);
}

static VkFilter to_vk_filter(Filter filter) {
    switch (filter) {
        case Filter::Nearest: return VK_FILTER_NEAREST;
        case Filter::Linear: return VK_FILTER_LINEAR;
        default: return VK_FILTER_LINEAR;
    }
}

static VkSamplerMipmapMode to_vk_mipmap_mode(MipmapMode mode) {
    switch (mode) {
        case MipmapMode::Nearest: return VK_SAMPLER_MIPMAP_MODE_NEAREST;
        case MipmapMode::Linear: return VK_SAMPLER_MIPMAP_MODE_LINEAR;
        default: return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

static VkSamplerAddressMode to_vk_address_mode(AddressMode mode) {
    switch (mode) {
        case AddressMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case AddressMode::MirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case AddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case AddressMode::ClampToBorder: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
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
        default: return VK_COMPARE_OP_NEVER;
    }
}

std::unique_ptr<Sampler> create_vulkan_sampler(VkDevice device, const SamplerDesc& desc) {
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = to_vk_filter(desc.mag_filter);
    sampler_info.minFilter = to_vk_filter(desc.min_filter);
    sampler_info.mipmapMode = to_vk_mipmap_mode(desc.mipmap_mode);
    sampler_info.addressModeU = to_vk_address_mode(desc.address_u);
    sampler_info.addressModeV = to_vk_address_mode(desc.address_v);
    sampler_info.addressModeW = to_vk_address_mode(desc.address_w);
    sampler_info.mipLodBias = desc.mip_lod_bias;
    sampler_info.anisotropyEnable = desc.anisotropy_enable ? VK_TRUE : VK_FALSE;
    sampler_info.maxAnisotropy = desc.max_anisotropy;
    sampler_info.compareEnable = desc.compare_enable ? VK_TRUE : VK_FALSE;
    sampler_info.compareOp = to_vk_compare_op(desc.compare_op);
    sampler_info.minLod = desc.min_lod;
    sampler_info.maxLod = desc.max_lod;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    VkSampler sampler;
    if (vk::vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan sampler" << std::endl;
        return nullptr;
    }

    return std::make_unique<VulkanSampler>(device, sampler);
}

// VulkanBindGroupLayout is defined in vk_internal.hpp

std::unique_ptr<BindGroup> create_vulkan_bind_group(VkDevice device, const BindGroupDesc& desc) {
    if (!desc.layout) {
        std::cerr << "Bind group layout is null" << std::endl;
        return nullptr;
    }

    auto* layout = static_cast<VulkanBindGroupLayout*>(desc.layout);
    VkDescriptorSetLayout vk_layout = layout->get_layout();

    // Count descriptor types needed for pool
    std::vector<VkDescriptorPoolSize> pool_sizes;
    uint32_t combined_sampler_count = 0;
    uint32_t uniform_buffer_count = 0;
    uint32_t storage_buffer_count = 0;
    uint32_t sampled_image_count = 0;
    uint32_t sampler_count = 0;
    uint32_t storage_image_count = 0;

    for (uint32_t i = 0; i < desc.entry_count; ++i) {
        const auto& entry = desc.entries[i];
        if (entry.texture && entry.sampler) {
            combined_sampler_count++;
        } else if (entry.texture) {
            sampled_image_count++;
        } else if (entry.sampler) {
            sampler_count++;
        } else if (entry.buffer) {
            // Assume uniform for now - could check usage
            uniform_buffer_count++;
        }
    }

    if (combined_sampler_count > 0) {
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, combined_sampler_count});
    }
    if (uniform_buffer_count > 0) {
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniform_buffer_count});
    }
    if (storage_buffer_count > 0) {
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storage_buffer_count});
    }
    if (sampled_image_count > 0) {
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampled_image_count});
    }
    if (sampler_count > 0) {
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, sampler_count});
    }

    if (pool_sizes.empty()) {
        // Add a dummy pool size to avoid Vulkan errors
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1});
    }

    // Create descriptor pool
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();

    VkDescriptorPool pool;
    if (vk::vkCreateDescriptorPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        return nullptr;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &vk_layout;

    VkDescriptorSet set;
    if (vk::vkAllocateDescriptorSets(device, &alloc_info, &set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set" << std::endl;
        vk::vkDestroyDescriptorPool(device, pool, nullptr);
        return nullptr;
    }

    // Update descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> image_infos;
    std::vector<VkDescriptorBufferInfo> buffer_infos;

    image_infos.reserve(desc.entry_count);
    buffer_infos.reserve(desc.entry_count);

    for (uint32_t i = 0; i < desc.entry_count; ++i) {
        const auto& entry = desc.entries[i];

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = entry.binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;

        if (entry.texture && entry.sampler) {
            auto* tex = static_cast<VulkanTexture*>(entry.texture);
            auto* samp = static_cast<VulkanSampler*>(entry.sampler);

            VkDescriptorImageInfo img_info{};
            img_info.sampler = samp->get_sampler();
            img_info.imageView = tex->get_view();
            img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_infos.push_back(img_info);

            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &image_infos.back();
        } else if (entry.buffer) {
            VkDescriptorBufferInfo buf_info{};
            buf_info.buffer = reinterpret_cast<VkBuffer>(entry.buffer->get_native_handle());
            buf_info.offset = entry.buffer_offset;
            buf_info.range = entry.buffer_size > 0 ? entry.buffer_size : VK_WHOLE_SIZE;
            buffer_infos.push_back(buf_info);

            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &buffer_infos.back();
        } else {
            continue;  // Skip invalid entries
        }

        writes.push_back(write);
    }

    if (!writes.empty()) {
        vk::vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    return std::make_unique<VulkanBindGroup>(device, pool, set);
}

} // namespace viz::gal
