#include "vk_internal.hpp"

#include <cstring>
#include <iostream>

namespace viz::gal {

// Forward declarations from vk_device.cpp
class VulkanDevice;
uint32_t find_memory_type(VkPhysicalDevice physical_device,
                          const VkPhysicalDeviceMemoryProperties& mem_props,
                          uint32_t type_filter, VkMemoryPropertyFlags properties);

class VulkanBuffer : public Buffer {
public:
    VulkanBuffer(VkDevice device, VkPhysicalDevice physical_device,
                 const VkPhysicalDeviceMemoryProperties& mem_props)
        : device_(device), physical_device_(physical_device), mem_props_(mem_props) {}

    ~VulkanBuffer() override {
        destroy();
    }

    bool initialize(const BufferDesc& desc);
    void destroy();

    // Buffer interface
    size_t get_size() const override { return size_; }
    BufferUsage get_usage() const override { return usage_; }
    MemoryLocation get_memory_location() const override { return memory_location_; }

    void* map() override;
    void unmap() override;

    void write(const void* data, size_t size, size_t offset = 0) override;
    void read(void* data, size_t size, size_t offset = 0) override;

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(buffer_); }

    VkBuffer get_buffer() const { return buffer_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props_{};

    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;

    size_t size_ = 0;
    BufferUsage usage_ = BufferUsage::None;
    MemoryLocation memory_location_ = MemoryLocation::GPU_ONLY;

    void* mapped_ptr_ = nullptr;
    bool persistently_mapped_ = false;
};

static VkBufferUsageFlags to_vk_buffer_usage(BufferUsage usage) {
    VkBufferUsageFlags flags = 0;
    if (has_flag(usage, BufferUsage::Vertex)) flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (has_flag(usage, BufferUsage::Index)) flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (has_flag(usage, BufferUsage::Uniform)) flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (has_flag(usage, BufferUsage::Storage)) flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (has_flag(usage, BufferUsage::Indirect)) flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (has_flag(usage, BufferUsage::TransferSrc)) flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (has_flag(usage, BufferUsage::TransferDst)) flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return flags;
}

static VkMemoryPropertyFlags to_vk_memory_properties(MemoryLocation location) {
    switch (location) {
        case MemoryLocation::GPU_ONLY:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case MemoryLocation::CPU_TO_GPU:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryLocation::GPU_TO_CPU:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        default:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
}

uint32_t find_memory_type(VkPhysicalDevice physical_device,
                          const VkPhysicalDeviceMemoryProperties& mem_props,
                          uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    // Fallback: find any compatible memory type
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (type_filter & (1 << i)) {
            return i;
        }
    }

    return UINT32_MAX;
}

bool VulkanBuffer::initialize(const BufferDesc& desc) {
    size_ = desc.size;
    usage_ = desc.usage;
    memory_location_ = desc.memory;

    // Create buffer
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size_;
    buffer_info.usage = to_vk_buffer_usage(usage_);
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Add transfer dst if we have initial data and it's GPU-only
    if (desc.initial_data && memory_location_ == MemoryLocation::GPU_ONLY) {
        buffer_info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    if (vk::vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer" << std::endl;
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements mem_requirements;
    vk::vkGetBufferMemoryRequirements(device_, buffer_, &mem_requirements);

    // Allocate memory
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;

    VkMemoryPropertyFlags mem_properties = to_vk_memory_properties(memory_location_);
    alloc_info.memoryTypeIndex = find_memory_type(
        physical_device_, mem_props_,
        mem_requirements.memoryTypeBits, mem_properties);

    if (alloc_info.memoryTypeIndex == UINT32_MAX) {
        std::cerr << "Failed to find suitable memory type" << std::endl;
        vk::vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        return false;
    }

    if (vk::vkAllocateMemory(device_, &alloc_info, nullptr, &memory_) != VK_SUCCESS) {
        std::cerr << "Failed to allocate buffer memory" << std::endl;
        vk::vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        return false;
    }

    // Bind memory to buffer
    vk::vkBindBufferMemory(device_, buffer_, memory_, 0);

    // For CPU-accessible buffers, persistently map
    if (memory_location_ != MemoryLocation::GPU_ONLY) {
        if (vk::vkMapMemory(device_, memory_, 0, size_, 0, &mapped_ptr_) == VK_SUCCESS) {
            persistently_mapped_ = true;
        }
    }

    // Upload initial data
    if (desc.initial_data) {
        if (memory_location_ != MemoryLocation::GPU_ONLY) {
            // Direct copy for host-visible memory
            write(desc.initial_data, size_, 0);
        } else {
            // Would need staging buffer - for now, require CPU_TO_GPU for initial data
            std::cerr << "Warning: Initial data for GPU_ONLY buffer requires staging (not implemented)" << std::endl;
        }
    }

    return true;
}

void VulkanBuffer::destroy() {
    if (mapped_ptr_ && !persistently_mapped_) {
        vk::vkUnmapMemory(device_, memory_);
        mapped_ptr_ = nullptr;
    }

    if (memory_ != VK_NULL_HANDLE) {
        vk::vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }

    if (buffer_ != VK_NULL_HANDLE) {
        vk::vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
}

void* VulkanBuffer::map() {
    if (persistently_mapped_) {
        return mapped_ptr_;
    }

    if (memory_location_ == MemoryLocation::GPU_ONLY) {
        std::cerr << "Cannot map GPU_ONLY buffer" << std::endl;
        return nullptr;
    }

    if (vk::vkMapMemory(device_, memory_, 0, size_, 0, &mapped_ptr_) != VK_SUCCESS) {
        std::cerr << "Failed to map buffer memory" << std::endl;
        return nullptr;
    }

    return mapped_ptr_;
}

void VulkanBuffer::unmap() {
    if (persistently_mapped_) {
        return; // Keep mapped
    }

    if (mapped_ptr_) {
        vk::vkUnmapMemory(device_, memory_);
        mapped_ptr_ = nullptr;
    }
}

void VulkanBuffer::write(const void* data, size_t size, size_t offset) {
    if (memory_location_ == MemoryLocation::GPU_ONLY) {
        std::cerr << "Cannot write directly to GPU_ONLY buffer" << std::endl;
        return;
    }

    void* ptr = map();
    if (ptr) {
        memcpy(static_cast<uint8_t*>(ptr) + offset, data, size);
        if (!persistently_mapped_) {
            unmap();
        }
    }
}

void VulkanBuffer::read(void* data, size_t size, size_t offset) {
    if (memory_location_ == MemoryLocation::GPU_ONLY) {
        std::cerr << "Cannot read directly from GPU_ONLY buffer" << std::endl;
        return;
    }

    void* ptr = map();
    if (ptr) {
        memcpy(data, static_cast<uint8_t*>(ptr) + offset, size);
        if (!persistently_mapped_) {
            unmap();
        }
    }
}

// Factory function used by VulkanDevice
std::unique_ptr<Buffer> create_vulkan_buffer(VkDevice device, VkPhysicalDevice physical_device,
                                              const VkPhysicalDeviceMemoryProperties& mem_props,
                                              const BufferDesc& desc) {
    auto buffer = std::make_unique<VulkanBuffer>(device, physical_device, mem_props);
    if (buffer->initialize(desc)) {
        return buffer;
    }
    return nullptr;
}

} // namespace viz::gal
