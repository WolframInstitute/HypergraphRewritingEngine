#pragma once

// Internal header for Vulkan implementation classes
// Not part of the public API

#include <gal/gal.hpp>
#include <gal/vulkan/vk_loader.hpp>

#include <vector>
#include <memory>

namespace viz::gal {

// Forward declare VulkanDevice
class VulkanDevice;

// Accessor functions for cross-module use
VkInstance get_vk_instance(Device* device);
VkPhysicalDevice get_vk_physical_device(Device* device);
VkDevice get_vk_device(Device* device);
VkQueue get_vk_graphics_queue(Device* device);
uint32_t get_vk_graphics_queue_family(Device* device);
VkCommandPool get_vk_command_pool(Device* device);

// Vulkan Fence implementation
class VulkanFence : public Fence {
public:
    VulkanFence(VkDevice device, VkFence fence) : device_(device), fence_(fence) {}
    ~VulkanFence() override {
        if (fence_ != VK_NULL_HANDLE) {
            vk::vkDestroyFence(device_, fence_, nullptr);
        }
    }

    void wait(uint64_t timeout_ns = UINT64_MAX) override {
        vk::vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeout_ns);
    }

    void reset() override {
        vk::vkResetFences(device_, 1, &fence_);
    }

    bool is_signaled() const override {
        return vk::vkGetFenceStatus(device_, fence_) == VK_SUCCESS;
    }

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(fence_); }

    VkFence get_handle() const { return fence_; }

private:
    VkDevice device_;
    VkFence fence_;
};

// Vulkan Semaphore implementation
class VulkanSemaphore : public Semaphore {
public:
    VulkanSemaphore(VkDevice device, VkSemaphore semaphore) : device_(device), semaphore_(semaphore) {}
    ~VulkanSemaphore() override {
        if (semaphore_ != VK_NULL_HANDLE) {
            vk::vkDestroySemaphore(device_, semaphore_, nullptr);
        }
    }

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(semaphore_); }

    VkSemaphore get_handle() const { return semaphore_; }

private:
    VkDevice device_;
    VkSemaphore semaphore_;
};

// Vulkan Texture implementation for swapchain images
class VulkanSwapchainTexture : public Texture {
public:
    VulkanSwapchainTexture(VkDevice device, VkImage image, VkImageView view,
                           uint32_t width, uint32_t height, Format format)
        : device_(device), image_(image), view_(view),
          width_(width), height_(height), format_(format) {}

    ~VulkanSwapchainTexture() override {
        if (view_ != VK_NULL_HANDLE) {
            vk::vkDestroyImageView(device_, view_, nullptr);
        }
        // Don't destroy image - owned by swapchain
    }

    Extent3D get_size() const override { return {width_, height_, 1}; }
    Format get_format() const override { return format_; }
    TextureDimension get_dimension() const override { return TextureDimension::Tex2D; }
    TextureUsage get_usage() const override { return TextureUsage::RenderTarget; }
    uint32_t get_mip_levels() const override { return 1; }
    uint32_t get_array_layers() const override { return 1; }
    uint32_t get_sample_count() const override { return 1; }

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(image_); }
    Handle get_native_view() const override { return reinterpret_cast<Handle>(view_); }

    VkImage get_image() const { return image_; }
    VkImageView get_view() const { return view_; }

private:
    VkDevice device_;
    VkImage image_;
    VkImageView view_;
    uint32_t width_;
    uint32_t height_;
    Format format_;
};

// Vulkan Swapchain implementation
class VulkanSwapchain : public Swapchain {
public:
    VulkanSwapchain(VkDevice device, VkPhysicalDevice physical_device,
                    VkSurfaceKHR surface, uint32_t graphics_queue_family)
        : device_(device), physical_device_(physical_device),
          surface_(surface), graphics_queue_family_(graphics_queue_family) {}

    ~VulkanSwapchain() override { destroy(); }

    bool initialize(uint32_t width, uint32_t height, PresentMode mode);
    void destroy();

    // Swapchain interface
    uint32_t get_width() const override { return width_; }
    uint32_t get_height() const override { return height_; }
    Format get_format() const override { return format_; }
    uint32_t get_image_count() const override { return static_cast<uint32_t>(textures_.size()); }
    PresentMode get_present_mode() const override { return present_mode_; }

    Texture* get_texture(uint32_t index) override {
        if (index < textures_.size()) {
            return textures_[index].get();
        }
        return nullptr;
    }

    uint32_t get_current_index() const override { return current_image_index_; }

    AcquireResult acquire_next_image(Semaphore* signal_semaphore = nullptr,
                                      Fence* signal_fence = nullptr,
                                      uint64_t timeout_ns = UINT64_MAX) override;
    bool present(Semaphore* wait_semaphore = nullptr) override;
    bool resize(uint32_t width, uint32_t height) override;

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(swapchain_); }

    VkSwapchainKHR get_handle() const { return swapchain_; }

private:
    bool create_swapchain(uint32_t width, uint32_t height, PresentMode mode);
    bool create_image_views();
    VkSurfaceFormatKHR choose_surface_format();
    VkPresentModeKHR choose_present_mode(PresentMode mode);
    VkExtent2D choose_extent(uint32_t width, uint32_t height);

    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkSurfaceKHR surface_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_;

    std::vector<VkImage> images_;
    std::vector<std::unique_ptr<VulkanSwapchainTexture>> textures_;

    Format format_ = Format::BGRA8_SRGB;
    VkFormat vk_format_ = VK_FORMAT_B8G8R8A8_SRGB;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t current_image_index_ = 0;
    PresentMode present_mode_ = PresentMode::Fifo;
};

// Vulkan Bind Group Layout implementation
class VulkanBindGroupLayout : public BindGroupLayout {
public:
    VulkanBindGroupLayout(VkDevice device) : device_(device) {}
    ~VulkanBindGroupLayout() override {
        if (layout_ != VK_NULL_HANDLE) {
            vk::vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        }
    }

    bool initialize(const BindGroupLayoutDesc& desc);

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(layout_); }
    VkDescriptorSetLayout get_layout() const { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
};

// Vulkan Texture implementation (user-created textures)
class VulkanTexture : public Texture {
public:
    VulkanTexture(VkDevice device, VkImage image, VkDeviceMemory memory, VkImageView view,
                  const TextureDesc& desc)
        : device_(device), image_(image), memory_(memory), view_(view), desc_(desc) {}

    ~VulkanTexture() override {
        if (view_ != VK_NULL_HANDLE) {
            vk::vkDestroyImageView(device_, view_, nullptr);
        }
        if (image_ != VK_NULL_HANDLE) {
            vk::vkDestroyImage(device_, image_, nullptr);
        }
        if (memory_ != VK_NULL_HANDLE) {
            vk::vkFreeMemory(device_, memory_, nullptr);
        }
    }

    Extent3D get_size() const override { return desc_.size; }
    Format get_format() const override { return desc_.format; }
    TextureDimension get_dimension() const override { return desc_.dimension; }
    TextureUsage get_usage() const override { return desc_.usage; }
    uint32_t get_mip_levels() const override { return desc_.mip_levels; }
    uint32_t get_array_layers() const override { return desc_.array_layers; }
    uint32_t get_sample_count() const override { return desc_.sample_count; }

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(image_); }
    Handle get_native_view() const override { return reinterpret_cast<Handle>(view_); }

    VkImage get_image() const { return image_; }
    VkImageView get_view() const { return view_; }

private:
    VkDevice device_;
    VkImage image_;
    VkDeviceMemory memory_;
    VkImageView view_;
    TextureDesc desc_;
};

// Vulkan Sampler implementation
class VulkanSampler : public Sampler {
public:
    VulkanSampler(VkDevice device, VkSampler sampler) : device_(device), sampler_(sampler) {}

    ~VulkanSampler() override {
        if (sampler_ != VK_NULL_HANDLE) {
            vk::vkDestroySampler(device_, sampler_, nullptr);
        }
    }

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(sampler_); }
    VkSampler get_sampler() const { return sampler_; }

private:
    VkDevice device_;
    VkSampler sampler_;
};

// Vulkan BindGroup implementation
class VulkanBindGroup : public BindGroup {
public:
    VulkanBindGroup(VkDevice device, VkDescriptorPool pool, VkDescriptorSet set)
        : device_(device), pool_(pool), set_(set) {}

    ~VulkanBindGroup() override {
        // Descriptor set is freed with the pool
        if (pool_ != VK_NULL_HANDLE) {
            vk::vkDestroyDescriptorPool(device_, pool_, nullptr);
        }
    }

    Handle get_native_handle() const override { return reinterpret_cast<Handle>(set_); }
    VkDescriptorSet get_set() const { return set_; }

private:
    VkDevice device_;
    VkDescriptorPool pool_;  // Each bind group owns its pool for simplicity
    VkDescriptorSet set_;
};

// Factory functions
std::unique_ptr<Texture> create_vulkan_texture(VkDevice device, VkPhysicalDevice physical_device,
                                               const TextureDesc& desc);
std::unique_ptr<Sampler> create_vulkan_sampler(VkDevice device, const SamplerDesc& desc);
std::unique_ptr<BindGroup> create_vulkan_bind_group(VkDevice device, const BindGroupDesc& desc);

// Surface creation helpers
VkSurfaceKHR create_xcb_surface(VkInstance instance, void* connection, void* window);
VkSurfaceKHR create_win32_surface(VkInstance instance, void* hinstance, void* hwnd);

} // namespace viz::gal
