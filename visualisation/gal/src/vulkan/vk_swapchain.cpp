#include "vk_internal.hpp"

#include <algorithm>
#include <iostream>
#include <cassert>

namespace viz::gal {

bool VulkanSwapchain::initialize(uint32_t width, uint32_t height, PresentMode mode) {
    // Get the present queue (usually same as graphics)
    vk::vkGetDeviceQueue(device_, graphics_queue_family_, 0, &present_queue_);

    if (!create_swapchain(width, height, mode)) {
        return false;
    }

    if (!create_image_views()) {
        return false;
    }

    return true;
}

void VulkanSwapchain::destroy() {
    if (device_ != VK_NULL_HANDLE) {
        vk::vkDeviceWaitIdle(device_);
    }

    textures_.clear();
    images_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
        vk::vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

bool VulkanSwapchain::create_swapchain(uint32_t width, uint32_t height, PresentMode mode) {
    // Query surface capabilities
    VkSurfaceCapabilitiesKHR capabilities;
    vk::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &capabilities);

    VkSurfaceFormatKHR surface_format = choose_surface_format();
    VkPresentModeKHR present_mode = choose_present_mode(mode);
    VkExtent2D extent = choose_extent(width, height);

    // Choose image count (prefer triple buffering)
    uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface_;
    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;

    if (vk::vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_) != VK_SUCCESS) {
        std::cerr << "Failed to create swapchain" << std::endl;
        return false;
    }

    // Get swapchain images
    vk::vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    images_.resize(image_count);
    vk::vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, images_.data());

    vk_format_ = surface_format.format;
    width_ = extent.width;
    height_ = extent.height;

    // Convert VkFormat to our Format
    switch (vk_format_) {
        case VK_FORMAT_B8G8R8A8_SRGB: format_ = Format::BGRA8_SRGB; break;
        case VK_FORMAT_B8G8R8A8_UNORM: format_ = Format::BGRA8_UNORM; break;
        case VK_FORMAT_R8G8B8A8_SRGB: format_ = Format::RGBA8_SRGB; break;
        case VK_FORMAT_R8G8B8A8_UNORM: format_ = Format::RGBA8_UNORM; break;
        default: format_ = Format::BGRA8_SRGB; break;
    }

    return true;
}

bool VulkanSwapchain::create_image_views() {
    textures_.clear();
    textures_.reserve(images_.size());

    for (size_t i = 0; i < images_.size(); ++i) {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = images_[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = vk_format_;
        view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        VkImageView view;
        if (vk::vkCreateImageView(device_, &view_info, nullptr, &view) != VK_SUCCESS) {
            std::cerr << "Failed to create swapchain image view" << std::endl;
            return false;
        }

        textures_.push_back(std::make_unique<VulkanSwapchainTexture>(
            device_, images_[i], view, width_, height_, format_));
    }

    return true;
}

VkSurfaceFormatKHR VulkanSwapchain::choose_surface_format() {
    uint32_t format_count;
    vk::vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    vk::vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, formats.data());

    // Prefer UNORM B8G8R8A8 with SRGB color space
    // This stores colors directly without conversion, and the display
    // compositor handles the final gamma presentation
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    // Try RGBA UNORM as fallback
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_R8G8B8A8_UNORM &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    // Fallback to SRGB if UNORM not available
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    // Fallback to first available
    return formats[0];
}

VkPresentModeKHR VulkanSwapchain::choose_present_mode(PresentMode mode) {
    uint32_t mode_count;
    vk::vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &mode_count, nullptr);
    std::vector<VkPresentModeKHR> modes(mode_count);
    vk::vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &mode_count, modes.data());

    VkPresentModeKHR desired;
    switch (mode) {
        case PresentMode::Immediate:
            desired = VK_PRESENT_MODE_IMMEDIATE_KHR;
            break;
        case PresentMode::Mailbox:
            desired = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        case PresentMode::FifoRelaxed:
            desired = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
            break;
        case PresentMode::Fifo:
        default:
            desired = VK_PRESENT_MODE_FIFO_KHR;
            break;
    }

    for (const auto& available : modes) {
        if (available == desired) {
            present_mode_ = mode;
            return desired;
        }
    }

    // FIFO is always available
    present_mode_ = PresentMode::Fifo;
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapchain::choose_extent(uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR capabilities;
    vk::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &capabilities);

    // Note: On Windows with DPI awareness, currentExtent may not match our desired size.
    // We prefer using the explicitly passed dimensions (which come from GetClientRect
    // in physical pixels) rather than relying on currentExtent.
    // Only use currentExtent as a fallback if our values are invalid (0x0).
    VkExtent2D extent;
    if (width == 0 || height == 0) {
        // Fallback to currentExtent if we don't have valid dimensions
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        }
        extent = {1, 1};  // Minimum valid extent
    } else {
        extent = {width, height};
    }

    // Clamp to supported range
    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return extent;
}

AcquireResult VulkanSwapchain::acquire_next_image(Semaphore* signal_semaphore, Fence* signal_fence, uint64_t timeout_ns) {
    VkSemaphore vk_semaphore = VK_NULL_HANDLE;
    VkFence vk_fence = VK_NULL_HANDLE;

    if (signal_semaphore) {
        vk_semaphore = static_cast<VulkanSemaphore*>(signal_semaphore)->get_handle();
    }
    if (signal_fence) {
        vk_fence = static_cast<VulkanFence*>(signal_fence)->get_handle();
    }

    VkResult result = vk::vkAcquireNextImageKHR(
        device_, swapchain_, timeout_ns, vk_semaphore, vk_fence, &current_image_index_);

    AcquireResult acquire_result;
    acquire_result.image_index = current_image_index_;

    switch (result) {
        case VK_SUCCESS:
            acquire_result.success = true;
            acquire_result.suboptimal = false;
            acquire_result.out_of_date = false;
            break;
        case VK_SUBOPTIMAL_KHR:
            acquire_result.success = true;
            acquire_result.suboptimal = true;
            acquire_result.out_of_date = false;
            break;
        case VK_ERROR_OUT_OF_DATE_KHR:
            acquire_result.success = false;
            acquire_result.suboptimal = false;
            acquire_result.out_of_date = true;
            break;
        default:
            acquire_result.success = false;
            acquire_result.suboptimal = false;
            acquire_result.out_of_date = false;
            break;
    }

    return acquire_result;
}

bool VulkanSwapchain::present(Semaphore* wait_semaphore) {
    VkSemaphore vk_wait_semaphore = VK_NULL_HANDLE;
    if (wait_semaphore) {
        vk_wait_semaphore = static_cast<VulkanSemaphore*>(wait_semaphore)->get_handle();
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = (vk_wait_semaphore != VK_NULL_HANDLE) ? 1 : 0;
    present_info.pWaitSemaphores = &vk_wait_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_;
    present_info.pImageIndices = &current_image_index_;

    VkResult result = vk::vkQueuePresentKHR(present_queue_, &present_info);
    return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
}

bool VulkanSwapchain::resize(uint32_t width, uint32_t height) {
    vk::vkDeviceWaitIdle(device_);
    destroy();
    if (!create_swapchain(width, height, present_mode_)) {
        return false;
    }
    if (!create_image_views()) {
        return false;
    }
    return true;
}

// Surface creation helpers
VkSurfaceKHR create_xcb_surface(VkInstance instance, void* connection, void* window) {
#if defined(VIZ_PLATFORM_LINUX)
    VkXcbSurfaceCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
    create_info.connection = static_cast<xcb_connection_t*>(connection);
    create_info.window = static_cast<xcb_window_t>(reinterpret_cast<uintptr_t>(window));

    VkSurfaceKHR surface;
    if (vk::vkCreateXcbSurfaceKHR(instance, &create_info, nullptr, &surface) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return surface;
#else
    return VK_NULL_HANDLE;
#endif
}

VkSurfaceKHR create_win32_surface(VkInstance instance, void* hinstance, void* hwnd) {
#if defined(VIZ_PLATFORM_WINDOWS)
    VkWin32SurfaceCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    create_info.hinstance = static_cast<HINSTANCE>(hinstance);
    create_info.hwnd = static_cast<HWND>(hwnd);

    VkSurfaceKHR surface;
    if (vk::vkCreateWin32SurfaceKHR(instance, &create_info, nullptr, &surface) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return surface;
#else
    return VK_NULL_HANDLE;
#endif
}

} // namespace viz::gal
