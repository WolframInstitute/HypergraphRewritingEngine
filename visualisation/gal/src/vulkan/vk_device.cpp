#include "vk_internal.hpp"

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <fstream>
#include <chrono>

namespace viz::gal {

// Forward declarations for types not in vk_internal.hpp
class VulkanBuffer;
class VulkanSampler;
class VulkanShader;
class VulkanRenderPipeline;
class VulkanComputePipeline;
class VulkanBindGroupLayout;
class VulkanBindGroup;
class VulkanCommandEncoder;
class VulkanCommandBuffer;

// Factory functions from other implementation files
std::unique_ptr<Buffer> create_vulkan_buffer(VkDevice device, VkPhysicalDevice physical_device,
                                              const VkPhysicalDeviceMemoryProperties& mem_props,
                                              const BufferDesc& desc);

// Cleanup function for render pass cache (defined in vk_command_buffer.cpp)
void cleanup_render_pass_cache(VkDevice device);

std::unique_ptr<Shader> create_vulkan_shader(VkDevice device, const ShaderDesc& desc);
std::unique_ptr<BindGroupLayout> create_vulkan_bind_group_layout(VkDevice device, const BindGroupLayoutDesc& desc);
std::unique_ptr<RenderPipeline> create_vulkan_render_pipeline(VkDevice device, VkPipelineCache cache, const RenderPipelineDesc& desc);
std::unique_ptr<ComputePipeline> create_vulkan_compute_pipeline(VkDevice device, const ComputePipelineDesc& desc);

std::unique_ptr<CommandEncoder> create_vulkan_command_encoder(VkDevice device, VkCommandPool pool);

std::unique_ptr<Texture> create_vulkan_texture(VkDevice device, VkPhysicalDevice physical_device,
                                               const TextureDesc& desc,
                                               VkCommandPool cmd_pool,
                                               VkQueue graphics_queue);
std::unique_ptr<Sampler> create_vulkan_sampler(VkDevice device, const SamplerDesc& desc);
std::unique_ptr<BindGroup> create_vulkan_bind_group(VkDevice device, const BindGroupDesc& desc);

// Vulkan format conversion
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

// Debug callback
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* user_data
) {
    const char* severity_str = "INFO";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) severity_str = "ERROR";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) severity_str = "WARNING";

    std::cerr << "[Vulkan " << severity_str << "] " << data->pMessage << std::endl;

    return VK_FALSE;
}


// Vulkan Device Implementation
class VulkanDevice : public Device {
public:
    VulkanDevice() = default;
    ~VulkanDevice() override { destroy(); }

    bool initialize(const DeviceDesc& desc);
    void destroy();

    // Device interface
    Backend get_backend() const override { return Backend::Vulkan; }
    const DeviceInfo& get_info() const override { return info_; }

    std::unique_ptr<Buffer> create_buffer(const BufferDesc& desc) override;
    std::unique_ptr<Texture> create_texture(const TextureDesc& desc) override;
    std::unique_ptr<Sampler> create_sampler(const SamplerDesc& desc) override;
    std::unique_ptr<Shader> create_shader(const ShaderDesc& desc) override;
    std::unique_ptr<RenderPipeline> create_render_pipeline(const RenderPipelineDesc& desc) override;
    std::unique_ptr<ComputePipeline> create_compute_pipeline(const ComputePipelineDesc& desc) override;
    std::unique_ptr<BindGroupLayout> create_bind_group_layout(const BindGroupLayoutDesc& desc) override;
    std::unique_ptr<BindGroup> create_bind_group(const BindGroupDesc& desc) override;

    std::unique_ptr<Swapchain> create_swapchain(Handle surface, uint32_t width, uint32_t height) override;
    std::unique_ptr<CommandEncoder> create_command_encoder() override;

    void submit(CommandBuffer* cmd, Fence* signal_fence = nullptr) override;
    void submit(CommandBuffer* cmd, Semaphore* wait_semaphore, Semaphore* signal_semaphore,
                Fence* signal_fence = nullptr) override;
    void submit_and_wait(CommandBuffer* cmd) override;

    std::unique_ptr<Fence> create_fence(bool signaled = false) override;
    std::unique_ptr<Semaphore> create_semaphore() override;
    void wait_idle() override;

    void set_debug_name(Buffer* resource, const char* name) override;
    void set_debug_name(Texture* resource, const char* name) override;

    // Internal accessors
    VkInstance get_instance() const { return instance_; }
    VkPhysicalDevice get_physical_device() const { return physical_device_; }
    VkDevice get_device() const { return device_; }
    VkQueue get_graphics_queue() const { return graphics_queue_; }
    uint32_t get_graphics_queue_family() const { return graphics_queue_family_; }
    VkCommandPool get_command_pool() const { return command_pool_; }
    VkPipelineCache get_pipeline_cache() const { return pipeline_cache_; }

    // Memory allocation helper
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

private:
    bool create_instance(const DeviceDesc& desc);
    bool select_physical_device(const DeviceDesc& desc);
    bool create_logical_device(const DeviceDesc& desc);
    void query_device_info();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkQueue transfer_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_ = UINT32_MAX;
    uint32_t compute_queue_family_ = UINT32_MAX;
    uint32_t transfer_queue_family_ = UINT32_MAX;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    bool pipeline_cache_loaded_from_disk_ = false;

    VkPhysicalDeviceMemoryProperties memory_properties_{};
    DeviceInfo info_;
    bool validation_enabled_ = false;
    bool has_memory_priority_ = false;
    bool has_pageable_memory_ = false;

    // Pipeline cache file operations
    bool load_pipeline_cache();
    void save_pipeline_cache();
    static constexpr const char* PIPELINE_CACHE_FILENAME = "pipeline_cache.bin";
};

bool VulkanDevice::initialize(const DeviceDesc& desc) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (!vk::load_vulkan_library()) {
        std::cerr << "Failed to load Vulkan library" << std::endl;
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  [Device] load_vulkan_library: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;

    if (!create_instance(desc)) {
        return false;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "  [Device] create_instance: "
              << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;

    // Instance functions loaded in create_instance()

    if (!select_physical_device(desc)) {
        return false;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "  [Device] select_physical_device: "
              << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms" << std::endl;

    if (!create_logical_device(desc)) {
        return false;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "  [Device] create_logical_device: "
              << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms" << std::endl;

    if (!vk::load_device_functions(device_)) {
        std::cerr << "Failed to load Vulkan device functions" << std::endl;
        return false;
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "  [Device] load_device_functions: "
              << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms" << std::endl;

    // Get queues
    vk::vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
    if (compute_queue_family_ != UINT32_MAX) {
        vk::vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
    }
    if (transfer_queue_family_ != UINT32_MAX) {
        vk::vkGetDeviceQueue(device_, transfer_queue_family_, 0, &transfer_queue_);
    }

    // Create command pool for the graphics queue
    // Note: We use TRANSIENT_BIT since command buffers are short-lived (one-time submit)
    // We don't use RESET_COMMAND_BUFFER_BIT - instead we recreate command buffers each frame
    // This follows best practices for avoiding per-command-buffer reset overhead
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = graphics_queue_family_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    if (vk::vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool" << std::endl;
        return false;
    }

    // Get memory properties
    vk::vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

    // Query device info
    query_device_info();

    // Create pipeline cache (loads from disk if available)
    load_pipeline_cache();

    return true;
}

void VulkanDevice::destroy() {
    if (device_) {
        vk::vkDeviceWaitIdle(device_);

        // Save and destroy pipeline cache
        save_pipeline_cache();
        if (pipeline_cache_) {
            vk::vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
            pipeline_cache_ = VK_NULL_HANDLE;
        }

        // Clean up cached render passes before destroying device
        cleanup_render_pass_cache(device_);

        if (command_pool_) {
            vk::vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
        }

        vk::vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (debug_messenger_ && vk::vkDestroyDebugUtilsMessengerEXT) {
        vk::vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
        debug_messenger_ = VK_NULL_HANDLE;
    }

    if (instance_) {
        vk::vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

bool VulkanDevice::load_pipeline_cache() {
    std::vector<uint8_t> cache_data;

    // Try to load existing cache from disk
    std::ifstream file(PIPELINE_CACHE_FILENAME, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        auto size = file.tellg();
        if (size > 0) {
            cache_data.resize(static_cast<size_t>(size));
            file.seekg(0, std::ios::beg);
            file.read(reinterpret_cast<char*>(cache_data.data()), size);
            std::cout << "[Pipeline Cache] Loaded " << size << " bytes from disk" << std::endl;
            pipeline_cache_loaded_from_disk_ = true;
        }
        file.close();
    }

    VkPipelineCacheCreateInfo cache_info{};
    cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cache_info.initialDataSize = cache_data.size();
    cache_info.pInitialData = cache_data.empty() ? nullptr : cache_data.data();

    if (vk::vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        std::cerr << "[Pipeline Cache] Failed to create pipeline cache" << std::endl;
        return false;
    }

    if (cache_data.empty()) {
        std::cout << "[Pipeline Cache] Created new empty cache" << std::endl;
    }

    return true;
}

void VulkanDevice::save_pipeline_cache() {
    if (!pipeline_cache_) return;

    // Skip saving if we loaded from disk (cache hasn't changed)
    if (pipeline_cache_loaded_from_disk_) {
        return;
    }

    // Get cache data size
    size_t cache_size = 0;
    if (vk::vkGetPipelineCacheData(device_, pipeline_cache_, &cache_size, nullptr) != VK_SUCCESS) {
        std::cerr << "[Pipeline Cache] Failed to get cache size" << std::endl;
        return;
    }

    if (cache_size == 0) {
        std::cout << "[Pipeline Cache] Cache is empty, nothing to save" << std::endl;
        return;
    }

    // Get cache data
    std::vector<uint8_t> cache_data(cache_size);
    if (vk::vkGetPipelineCacheData(device_, pipeline_cache_, &cache_size, cache_data.data()) != VK_SUCCESS) {
        std::cerr << "[Pipeline Cache] Failed to get cache data" << std::endl;
        return;
    }

    // Write to disk
    std::ofstream file(PIPELINE_CACHE_FILENAME, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(cache_data.data()), cache_size);
        file.close();
        std::cout << "[Pipeline Cache] Saved " << cache_size << " bytes to disk" << std::endl;
    } else {
        std::cerr << "[Pipeline Cache] Failed to open file for writing" << std::endl;
    }
}

bool VulkanDevice::create_instance(const DeviceDesc& desc) {
    auto ci_start = std::chrono::high_resolution_clock::now();

    // Check for validation layers
    std::vector<const char*> layers;
    if (desc.enable_validation) {
        uint32_t layer_count;
        vk::vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> available_layers(layer_count);
        vk::vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

        for (const auto& layer : available_layers) {
            if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                layers.push_back("VK_LAYER_KHRONOS_validation");
                validation_enabled_ = true;
                break;
            }
        }
    }
    auto ci_layers = std::chrono::high_resolution_clock::now();
    std::cout << "    [Instance] layer enumeration: "
              << std::chrono::duration<double, std::milli>(ci_layers - ci_start).count()
              << " ms (validation=" << (desc.enable_validation ? "requested" : "disabled")
              << ", found=" << (validation_enabled_ ? "yes" : "no") << ")" << std::endl;

    // Required extensions
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

#if defined(VIZ_PLATFORM_LINUX)
    extensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VIZ_PLATFORM_WINDOWS)
    extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VIZ_PLATFORM_MACOS)
    extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    if (validation_enabled_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // App info
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = desc.app_name;
    app_info.applicationVersion = desc.app_version;
    app_info.pEngineName = "HypergraphViz";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    // Instance create info
    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.data();

#if defined(VIZ_PLATFORM_MACOS)
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    // Debug messenger create info (for instance creation debugging)
    VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
    if (validation_enabled_) {
        debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_create_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_create_info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_create_info.pfnUserCallback = debug_callback;

        create_info.pNext = &debug_create_info;
    }

    auto ci_before = std::chrono::high_resolution_clock::now();
    VkResult result = vk::vkCreateInstance(&create_info, nullptr, &instance_);
    auto ci_after = std::chrono::high_resolution_clock::now();
    std::cout << "    [Instance] vkCreateInstance: "
              << std::chrono::duration<double, std::milli>(ci_after - ci_before).count() << " ms" << std::endl;
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance: " << result << std::endl;
        return false;
    }

    // Load instance functions
    if (!vk::load_instance_functions(instance_)) {
        std::cerr << "Failed to load Vulkan instance functions" << std::endl;
        return false;
    }

    // Create debug messenger after instance
    if (validation_enabled_ && vk::vkCreateDebugUtilsMessengerEXT) {
        VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
        messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        messenger_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messenger_info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messenger_info.pfnUserCallback = debug_callback;

        if (vk::vkCreateDebugUtilsMessengerEXT(instance_, &messenger_info, nullptr, &debug_messenger_) != VK_SUCCESS) {
            std::cerr << "Warning: Failed to create debug messenger" << std::endl;
        } else {
            std::cout << "Vulkan validation layers enabled" << std::endl;
        }
    }

    return true;
}

bool VulkanDevice::select_physical_device(const DeviceDesc& desc) {
    uint32_t device_count = 0;
    vk::vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

    if (device_count == 0) {
        std::cerr << "No Vulkan-capable GPUs found" << std::endl;
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vk::vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    // Score devices and pick the best one
    int best_score = -1;
    VkPhysicalDevice best_device = VK_NULL_HANDLE;

    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties props;
        vk::vkGetPhysicalDeviceProperties(device, &props);

        // Check for required extensions
        uint32_t ext_count;
        vk::vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vk::vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, exts.data());

        bool has_swapchain = false;
        for (const auto& ext : exts) {
            if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                has_swapchain = true;
                break;
            }
        }

        if (!has_swapchain) continue;

        // Check for graphics queue
        uint32_t queue_family_count = 0;
        vk::vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vk::vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

        bool has_graphics = false;
        for (const auto& qf : queue_families) {
            if (qf.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                has_graphics = true;
                break;
            }
        }

        if (!has_graphics) continue;

        // Score the device
        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            score += 100;
        }

        // Prefer devices matching the preferred name
        if (desc.preferred_device && strstr(props.deviceName, desc.preferred_device)) {
            score += 10000;
        }

        if (score > best_score) {
            best_score = score;
            best_device = device;
        }
    }

    if (best_device == VK_NULL_HANDLE) {
        std::cerr << "No suitable GPU found" << std::endl;
        return false;
    }

    physical_device_ = best_device;

    // Find queue families
    uint32_t queue_family_count = 0;
    vk::vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vk::vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; ++i) {
        const auto& qf = queue_families[i];

        if ((qf.queueFlags & VK_QUEUE_GRAPHICS_BIT) && graphics_queue_family_ == UINT32_MAX) {
            graphics_queue_family_ = i;
        }

        // Prefer dedicated compute queue
        if ((qf.queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            compute_queue_family_ = i;
        }

        // Prefer dedicated transfer queue
        if ((qf.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(qf.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !(qf.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            transfer_queue_family_ = i;
        }
    }

    // Fallback: use graphics queue for compute/transfer if no dedicated queues
    if (compute_queue_family_ == UINT32_MAX) compute_queue_family_ = graphics_queue_family_;
    if (transfer_queue_family_ == UINT32_MAX) transfer_queue_family_ = graphics_queue_family_;

    return true;
}

bool VulkanDevice::create_logical_device(const DeviceDesc& desc) {
    // Queue create infos
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::vector<uint32_t> unique_families;

    unique_families.push_back(graphics_queue_family_);
    if (compute_queue_family_ != graphics_queue_family_) {
        unique_families.push_back(compute_queue_family_);
    }
    if (transfer_queue_family_ != graphics_queue_family_ &&
        transfer_queue_family_ != compute_queue_family_) {
        unique_families.push_back(transfer_queue_family_);
    }

    float queue_priority = 1.0f;
    for (uint32_t family : unique_families) {
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_info);
    }

    // Query available device features first (required by best practices)
    VkPhysicalDeviceFeatures available_features{};
    vk::vkGetPhysicalDeviceFeatures(physical_device_, &available_features);

    // Enable only the features we need (and that are available)
    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy = available_features.samplerAnisotropy;
    features.fillModeNonSolid = available_features.fillModeNonSolid;  // For wireframe
    features.wideLines = available_features.wideLines;                 // For line width > 1
    features.independentBlend = available_features.independentBlend;  // For WBOIT (different blend states per attachment)

    // Extensions
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    // Enumerate available device extensions
    uint32_t ext_count = 0;
    vk::vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available_extensions(ext_count);
    vk::vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, available_extensions.data());

    // Check for optional extensions
    bool has_portability_subset = false;
    bool has_memory_priority = false;
    bool has_pageable_memory = false;

    for (const auto& ext : available_extensions) {
        if (strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0) {
            has_portability_subset = true;
        } else if (strcmp(ext.extensionName, "VK_EXT_memory_priority") == 0) {
            has_memory_priority = true;
        } else if (strcmp(ext.extensionName, "VK_EXT_pageable_device_local_memory") == 0) {
            has_pageable_memory = true;
        }
    }

    // VK_KHR_portability_subset MUST be enabled if supported
    if (has_portability_subset) {
        extensions.push_back("VK_KHR_portability_subset");
    }

    // VK_EXT_memory_priority is required for VK_EXT_pageable_device_local_memory
    if (has_memory_priority) {
        extensions.push_back("VK_EXT_memory_priority");
    }
    if (has_pageable_memory && has_memory_priority) {
        extensions.push_back("VK_EXT_pageable_device_local_memory");
    }

    // Build feature chain for optional features
    VkPhysicalDeviceMemoryPriorityFeaturesEXT memory_priority_features{};
    memory_priority_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
    memory_priority_features.memoryPriority = VK_TRUE;

    VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageable_memory_features{};
    pageable_memory_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT;
    pageable_memory_features.pageableDeviceLocalMemory = VK_TRUE;

    // Create device
    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();
    create_info.pEnabledFeatures = &features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    // Chain feature structs
    void** next_ptr = const_cast<void**>(&create_info.pNext);
    if (has_memory_priority) {
        *next_ptr = &memory_priority_features;
        next_ptr = &memory_priority_features.pNext;
    }
    if (has_pageable_memory && has_memory_priority) {
        *next_ptr = &pageable_memory_features;
        next_ptr = &pageable_memory_features.pNext;
    }

    VkResult result = vk::vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan logical device: " << result << std::endl;
        return false;
    }

    // Store feature availability for later use
    has_memory_priority_ = has_memory_priority;
    has_pageable_memory_ = has_pageable_memory && has_memory_priority;

    return true;
}

void VulkanDevice::query_device_info() {
    VkPhysicalDeviceProperties props;
    vk::vkGetPhysicalDeviceProperties(physical_device_, &props);

    info_.device_name = props.deviceName;
    info_.is_discrete_gpu = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    info_.is_integrated_gpu = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);

    info_.api_version = std::to_string(VK_VERSION_MAJOR(props.apiVersion)) + "." +
                        std::to_string(VK_VERSION_MINOR(props.apiVersion)) + "." +
                        std::to_string(VK_VERSION_PATCH(props.apiVersion));

    info_.driver_version = std::to_string(VK_VERSION_MAJOR(props.driverVersion)) + "." +
                           std::to_string(VK_VERSION_MINOR(props.driverVersion)) + "." +
                           std::to_string(VK_VERSION_PATCH(props.driverVersion));

    // Vendor name from vendor ID
    switch (props.vendorID) {
        case 0x1002: info_.vendor_name = "AMD"; break;
        case 0x10DE: info_.vendor_name = "NVIDIA"; break;
        case 0x8086: info_.vendor_name = "Intel"; break;
        case 0x13B5: info_.vendor_name = "ARM"; break;
        case 0x5143: info_.vendor_name = "Qualcomm"; break;
        default: info_.vendor_name = "Unknown"; break;
    }

    // Limits
    info_.limits.max_texture_dimension_1d = props.limits.maxImageDimension1D;
    info_.limits.max_texture_dimension_2d = props.limits.maxImageDimension2D;
    info_.limits.max_texture_dimension_3d = props.limits.maxImageDimension3D;
    info_.limits.max_texture_array_layers = props.limits.maxImageArrayLayers;
    info_.limits.max_bind_groups = 4;  // Vulkan has descriptor sets, typically 4
    info_.limits.max_uniform_buffer_binding_size = props.limits.maxUniformBufferRange;
    info_.limits.max_storage_buffer_binding_size = props.limits.maxStorageBufferRange;
    info_.limits.max_vertex_buffers = props.limits.maxVertexInputBindings;
    info_.limits.max_vertex_attributes = props.limits.maxVertexInputAttributes;
    info_.limits.max_vertex_buffer_stride = props.limits.maxVertexInputBindingStride;
    info_.limits.max_push_constant_size = props.limits.maxPushConstantsSize;
    info_.limits.max_compute_workgroup_size_x = props.limits.maxComputeWorkGroupSize[0];
    info_.limits.max_compute_workgroup_size_y = props.limits.maxComputeWorkGroupSize[1];
    info_.limits.max_compute_workgroup_size_z = props.limits.maxComputeWorkGroupSize[2];
    info_.limits.max_compute_workgroups_per_dimension = props.limits.maxComputeWorkGroupCount[0];
    info_.limits.min_uniform_buffer_offset_alignment = static_cast<uint32_t>(props.limits.minUniformBufferOffsetAlignment);
    info_.limits.min_storage_buffer_offset_alignment = static_cast<uint32_t>(props.limits.minStorageBufferOffsetAlignment);
    info_.limits.max_sampler_anisotropy = props.limits.maxSamplerAnisotropy;
    info_.limits.max_color_attachments = props.limits.maxColorAttachments;

    // Max MSAA samples (intersection of color and depth sample counts)
    VkSampleCountFlags sample_counts = props.limits.framebufferColorSampleCounts
                                     & props.limits.framebufferDepthSampleCounts;
    // Find highest bit set (max supported sample count)
    if (sample_counts & VK_SAMPLE_COUNT_64_BIT) info_.limits.max_samples = 64;
    else if (sample_counts & VK_SAMPLE_COUNT_32_BIT) info_.limits.max_samples = 32;
    else if (sample_counts & VK_SAMPLE_COUNT_16_BIT) info_.limits.max_samples = 16;
    else if (sample_counts & VK_SAMPLE_COUNT_8_BIT) info_.limits.max_samples = 8;
    else if (sample_counts & VK_SAMPLE_COUNT_4_BIT) info_.limits.max_samples = 4;
    else if (sample_counts & VK_SAMPLE_COUNT_2_BIT) info_.limits.max_samples = 2;
    else info_.limits.max_samples = 1;

    // Memory info
    VkPhysicalDeviceMemoryProperties mem_props;
    vk::vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            info_.dedicated_video_memory = mem_props.memoryHeaps[i].size;
            break;
        }
    }
}

uint32_t VulkanDevice::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

void VulkanDevice::wait_idle() {
    vk::vkDeviceWaitIdle(device_);
}

// Accessor functions for other modules
VkInstance get_vk_instance(Device* device) {
    return static_cast<VulkanDevice*>(device)->get_instance();
}

VkPhysicalDevice get_vk_physical_device(Device* device) {
    return static_cast<VulkanDevice*>(device)->get_physical_device();
}

VkDevice get_vk_device(Device* device) {
    return static_cast<VulkanDevice*>(device)->get_device();
}

VkQueue get_vk_graphics_queue(Device* device) {
    return static_cast<VulkanDevice*>(device)->get_graphics_queue();
}

uint32_t get_vk_graphics_queue_family(Device* device) {
    return static_cast<VulkanDevice*>(device)->get_graphics_queue_family();
}

VkCommandPool get_vk_command_pool(Device* device) {
    return static_cast<VulkanDevice*>(device)->get_command_pool();
}

// Resource creation implementations
std::unique_ptr<Buffer> VulkanDevice::create_buffer(const BufferDesc& desc) {
    return create_vulkan_buffer(device_, physical_device_, memory_properties_, desc);
}

std::unique_ptr<Texture> VulkanDevice::create_texture(const TextureDesc& desc) {
    return create_vulkan_texture(device_, physical_device_, desc, command_pool_, graphics_queue_);
}

std::unique_ptr<Sampler> VulkanDevice::create_sampler(const SamplerDesc& desc) {
    return create_vulkan_sampler(device_, desc);
}

std::unique_ptr<Shader> VulkanDevice::create_shader(const ShaderDesc& desc) {
    return create_vulkan_shader(device_, desc);
}

std::unique_ptr<RenderPipeline> VulkanDevice::create_render_pipeline(const RenderPipelineDesc& desc) {
    return create_vulkan_render_pipeline(device_, pipeline_cache_, desc);
}

std::unique_ptr<ComputePipeline> VulkanDevice::create_compute_pipeline(const ComputePipelineDesc& desc) {
    return create_vulkan_compute_pipeline(device_, desc);
}

std::unique_ptr<BindGroupLayout> VulkanDevice::create_bind_group_layout(const BindGroupLayoutDesc& desc) {
    return create_vulkan_bind_group_layout(device_, desc);
}

std::unique_ptr<BindGroup> VulkanDevice::create_bind_group(const BindGroupDesc& desc) {
    return create_vulkan_bind_group(device_, desc);
}

std::unique_ptr<Swapchain> VulkanDevice::create_swapchain(Handle surface, uint32_t width, uint32_t height) {
    VkSurfaceKHR vk_surface = reinterpret_cast<VkSurfaceKHR>(surface);

    auto swapchain = std::make_unique<VulkanSwapchain>(
        device_, physical_device_, vk_surface, graphics_queue_family_);

    if (!swapchain->initialize(width, height, PresentMode::Fifo)) {
        std::cerr << "Failed to initialize swapchain" << std::endl;
        return nullptr;
    }

    return swapchain;
}

std::unique_ptr<CommandEncoder> VulkanDevice::create_command_encoder() {
    return create_vulkan_command_encoder(device_, command_pool_);
}

void VulkanDevice::submit(CommandBuffer* cmd, Fence* signal_fence) {
    VkCommandBuffer vk_cmd = reinterpret_cast<VkCommandBuffer>(cmd->get_native_handle());

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &vk_cmd;

    VkFence fence = VK_NULL_HANDLE;
    if (signal_fence) {
        fence = reinterpret_cast<VkFence>(signal_fence->get_native_handle());
    }

    vk::vkQueueSubmit(graphics_queue_, 1, &submit_info, fence);
}

void VulkanDevice::submit(CommandBuffer* cmd, Semaphore* wait_semaphore,
                          Semaphore* signal_semaphore, Fence* signal_fence) {
    VkCommandBuffer vk_cmd = reinterpret_cast<VkCommandBuffer>(cmd->get_native_handle());

    VkSemaphore wait_sem = VK_NULL_HANDLE;
    VkSemaphore signal_sem = VK_NULL_HANDLE;
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    if (wait_semaphore) {
        wait_sem = reinterpret_cast<VkSemaphore>(wait_semaphore->get_native_handle());
    }
    if (signal_semaphore) {
        signal_sem = reinterpret_cast<VkSemaphore>(signal_semaphore->get_native_handle());
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = wait_sem ? 1 : 0;
    submit_info.pWaitSemaphores = wait_sem ? &wait_sem : nullptr;
    submit_info.pWaitDstStageMask = wait_sem ? &wait_stage : nullptr;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &vk_cmd;
    submit_info.signalSemaphoreCount = signal_sem ? 1 : 0;
    submit_info.pSignalSemaphores = signal_sem ? &signal_sem : nullptr;

    VkFence fence = VK_NULL_HANDLE;
    if (signal_fence) {
        fence = reinterpret_cast<VkFence>(signal_fence->get_native_handle());
    }

    vk::vkQueueSubmit(graphics_queue_, 1, &submit_info, fence);
}

void VulkanDevice::submit_and_wait(CommandBuffer* cmd) {
    VkCommandBuffer vk_cmd = reinterpret_cast<VkCommandBuffer>(cmd->get_native_handle());

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &vk_cmd;

    vk::vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
    vk::vkQueueWaitIdle(graphics_queue_);
}

std::unique_ptr<Fence> VulkanDevice::create_fence(bool signaled) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) {
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    }

    VkFence fence;
    if (vk::vkCreateFence(device_, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        std::cerr << "Failed to create fence" << std::endl;
        return nullptr;
    }

    return std::make_unique<VulkanFence>(device_, fence);
}

std::unique_ptr<Semaphore> VulkanDevice::create_semaphore() {
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore semaphore;
    if (vk::vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS) {
        std::cerr << "Failed to create semaphore" << std::endl;
        return nullptr;
    }

    return std::make_unique<VulkanSemaphore>(device_, semaphore);
}

void VulkanDevice::set_debug_name(Buffer* resource, const char* name) {
    // TODO: Implement with VK_EXT_debug_utils
}

void VulkanDevice::set_debug_name(Texture* resource, const char* name) {
    // TODO: Implement with VK_EXT_debug_utils
}

// Device factory implementation
std::unique_ptr<Device> Device::create(const DeviceDesc& desc) {
    Backend backend = desc.backend;
    if (backend == Backend::Auto) {
#ifdef VIZ_VULKAN
        backend = Backend::Vulkan;
#elif defined(VIZ_WEBGPU)
        backend = Backend::WebGPU;
#else
        return nullptr;
#endif
    }

    if (backend == Backend::Vulkan) {
#ifdef VIZ_VULKAN
        auto device = std::make_unique<VulkanDevice>();
        if (device->initialize(desc)) {
            return device;
        }
#endif
    }

    // TODO: WebGPU backend

    return nullptr;
}

// GAL initialization
static Backend s_active_backend = Backend::Auto;

bool initialize(Backend preferred_backend) {
    if (preferred_backend == Backend::Auto) {
#ifdef VIZ_VULKAN
        preferred_backend = Backend::Vulkan;
#elif defined(VIZ_WEBGPU)
        preferred_backend = Backend::WebGPU;
#else
        return false;
#endif
    }

    if (preferred_backend == Backend::Vulkan) {
#ifdef VIZ_VULKAN
        if (vk::load_vulkan_library()) {
            s_active_backend = Backend::Vulkan;
            return true;
        }
#endif
    }

    return false;
}

void shutdown() {
    if (s_active_backend == Backend::Vulkan) {
        vk::unload_vulkan_library();
    }
    s_active_backend = Backend::Auto;
}

bool is_backend_available(Backend backend) {
    if (backend == Backend::Vulkan) {
#ifdef VIZ_VULKAN
        return vk::is_vulkan_available();
#else
        return false;
#endif
    }
    return false;
}

Backend get_active_backend() {
    return s_active_backend;
}

} // namespace viz::gal
