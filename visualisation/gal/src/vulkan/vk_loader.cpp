#include <gal/vulkan/vk_loader.hpp>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #define VK_LIBRARY_NAME "vulkan-1.dll"
#elif defined(__APPLE__)
    #include <dlfcn.h>
    // MoltenVK paths - try multiple locations
    #define VK_LIBRARY_NAME "libvulkan.dylib"
    #define VK_LIBRARY_NAME_ALT "libMoltenVK.dylib"
#else
    // Linux
    #include <dlfcn.h>
    #define VK_LIBRARY_NAME "libvulkan.so.1"
    #define VK_LIBRARY_NAME_ALT "libvulkan.so"
#endif

namespace viz::gal::vk {

// Library handle
static void* s_vulkan_library = nullptr;

// Define all function pointers as nullptr initially

// Instance-level functions
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
PFN_vkCreateInstance vkCreateInstance = nullptr;
PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = nullptr;
PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties = nullptr;
PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;

// Instance functions
PFN_vkDestroyInstance vkDestroyInstance = nullptr;
PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2 = nullptr;
PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = nullptr;
PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2 = nullptr;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties = nullptr;
PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
PFN_vkCreateDevice vkCreateDevice = nullptr;
PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;

// Surface functions
PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR = nullptr;
PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR = nullptr;
PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR = nullptr;
PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR = nullptr;

#ifdef VK_USE_PLATFORM_XCB_KHR
PFN_vkCreateXcbSurfaceKHR vkCreateXcbSurfaceKHR = nullptr;
PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR vkGetPhysicalDeviceXcbPresentationSupportKHR = nullptr;
#endif

#ifdef VK_USE_PLATFORM_XLIB_KHR
PFN_vkCreateXlibSurfaceKHR vkCreateXlibSurfaceKHR = nullptr;
PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR vkGetPhysicalDeviceXlibPresentationSupportKHR = nullptr;
#endif

#ifdef VK_USE_PLATFORM_WIN32_KHR
PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR = nullptr;
PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR vkGetPhysicalDeviceWin32PresentationSupportKHR = nullptr;
#endif

#ifdef VK_USE_PLATFORM_METAL_EXT
PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT = nullptr;
#endif

// Debug utils
PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT = nullptr;

// Device functions
PFN_vkDestroyDevice vkDestroyDevice = nullptr;
PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
PFN_vkQueueSubmit vkQueueSubmit = nullptr;
PFN_vkQueuePresentKHR vkQueuePresentKHR = nullptr;

// Memory
PFN_vkAllocateMemory vkAllocateMemory = nullptr;
PFN_vkFreeMemory vkFreeMemory = nullptr;
PFN_vkMapMemory vkMapMemory = nullptr;
PFN_vkUnmapMemory vkUnmapMemory = nullptr;
PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges = nullptr;
PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges = nullptr;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements = nullptr;
PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
PFN_vkBindImageMemory vkBindImageMemory = nullptr;

// Buffers
PFN_vkCreateBuffer vkCreateBuffer = nullptr;
PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;

// Images
PFN_vkCreateImage vkCreateImage = nullptr;
PFN_vkDestroyImage vkDestroyImage = nullptr;
PFN_vkCreateImageView vkCreateImageView = nullptr;
PFN_vkDestroyImageView vkDestroyImageView = nullptr;

// Samplers
PFN_vkCreateSampler vkCreateSampler = nullptr;
PFN_vkDestroySampler vkDestroySampler = nullptr;

// Shaders
PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;

// Pipelines
PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines = nullptr;
PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
PFN_vkCreatePipelineCache vkCreatePipelineCache = nullptr;
PFN_vkDestroyPipelineCache vkDestroyPipelineCache = nullptr;
PFN_vkGetPipelineCacheData vkGetPipelineCacheData = nullptr;

// Descriptor sets
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
PFN_vkResetDescriptorPool vkResetDescriptorPool = nullptr;
PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
PFN_vkFreeDescriptorSets vkFreeDescriptorSets = nullptr;
PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;

// Render passes
PFN_vkCreateRenderPass vkCreateRenderPass = nullptr;
PFN_vkDestroyRenderPass vkDestroyRenderPass = nullptr;
PFN_vkCreateFramebuffer vkCreateFramebuffer = nullptr;
PFN_vkDestroyFramebuffer vkDestroyFramebuffer = nullptr;

// Command buffers
PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
PFN_vkResetCommandPool vkResetCommandPool = nullptr;
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
PFN_vkResetCommandBuffer vkResetCommandBuffer = nullptr;

// Command buffer commands
PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
PFN_vkCmdBindVertexBuffers vkCmdBindVertexBuffers = nullptr;
PFN_vkCmdBindIndexBuffer vkCmdBindIndexBuffer = nullptr;
PFN_vkCmdSetViewport vkCmdSetViewport = nullptr;
PFN_vkCmdSetScissor vkCmdSetScissor = nullptr;
PFN_vkCmdSetLineWidth vkCmdSetLineWidth = nullptr;
PFN_vkCmdSetDepthBias vkCmdSetDepthBias = nullptr;
PFN_vkCmdSetBlendConstants vkCmdSetBlendConstants = nullptr;
PFN_vkCmdSetStencilReference vkCmdSetStencilReference = nullptr;
PFN_vkCmdDraw vkCmdDraw = nullptr;
PFN_vkCmdDrawIndexed vkCmdDrawIndexed = nullptr;
PFN_vkCmdDrawIndirect vkCmdDrawIndirect = nullptr;
PFN_vkCmdDrawIndexedIndirect vkCmdDrawIndexedIndirect = nullptr;
PFN_vkCmdDispatch vkCmdDispatch = nullptr;
PFN_vkCmdDispatchIndirect vkCmdDispatchIndirect = nullptr;
PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr;
PFN_vkCmdCopyImage vkCmdCopyImage = nullptr;
PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage = nullptr;
PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer = nullptr;
PFN_vkCmdBlitImage vkCmdBlitImage = nullptr;
PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
PFN_vkCmdBeginRenderPass vkCmdBeginRenderPass = nullptr;
PFN_vkCmdEndRenderPass vkCmdEndRenderPass = nullptr;
PFN_vkCmdNextSubpass vkCmdNextSubpass = nullptr;
PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
PFN_vkCmdClearColorImage vkCmdClearColorImage = nullptr;
PFN_vkCmdClearDepthStencilImage vkCmdClearDepthStencilImage = nullptr;
PFN_vkCmdResolveImage vkCmdResolveImage = nullptr;

// Synchronization
PFN_vkCreateFence vkCreateFence = nullptr;
PFN_vkDestroyFence vkDestroyFence = nullptr;
PFN_vkResetFences vkResetFences = nullptr;
PFN_vkGetFenceStatus vkGetFenceStatus = nullptr;
PFN_vkWaitForFences vkWaitForFences = nullptr;
PFN_vkCreateSemaphore vkCreateSemaphore = nullptr;
PFN_vkDestroySemaphore vkDestroySemaphore = nullptr;

// Swapchain
PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR = nullptr;
PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR = nullptr;
PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR = nullptr;
PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR = nullptr;


// Platform-specific library loading
#if defined(_WIN32)

static void* load_library(const char* name) {
    return LoadLibraryA(name);
}

static void unload_library(void* handle) {
    if (handle) FreeLibrary((HMODULE)handle);
}

static void* get_proc_address(void* handle, const char* name) {
    return (void*)GetProcAddress((HMODULE)handle, name);
}

#else

static void* load_library(const char* name) {
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

static void unload_library(void* handle) {
    if (handle) dlclose(handle);
}

static void* get_proc_address(void* handle, const char* name) {
    return dlsym(handle, name);
}

#endif


bool load_vulkan_library() {
    if (s_vulkan_library) {
        return true;  // Already loaded
    }

    // Try primary library name
    s_vulkan_library = load_library(VK_LIBRARY_NAME);

#ifdef VK_LIBRARY_NAME_ALT
    // Try alternative name if primary failed
    if (!s_vulkan_library) {
        s_vulkan_library = load_library(VK_LIBRARY_NAME_ALT);
    }
#endif

    if (!s_vulkan_library) {
        return false;
    }

    // Load vkGetInstanceProcAddr first - it's the entry point for everything else
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)get_proc_address(
        s_vulkan_library, "vkGetInstanceProcAddr");

    if (!vkGetInstanceProcAddr) {
        unload_vulkan_library();
        return false;
    }

    // Load global functions (don't require an instance)
    vkCreateInstance = (PFN_vkCreateInstance)
        vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
    vkEnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceExtensionProperties");
    vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceLayerProperties");
    vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");

    return vkCreateInstance != nullptr;
}

void unload_vulkan_library() {
    if (s_vulkan_library) {
        unload_library(s_vulkan_library);
        s_vulkan_library = nullptr;
    }

    // Reset all function pointers
    vkGetInstanceProcAddr = nullptr;
    vkCreateInstance = nullptr;
    // ... could reset all others, but they'll be null anyway after library unload
}

bool load_instance_functions(VkInstance instance) {
    if (!instance || !vkGetInstanceProcAddr) {
        return false;
    }

#define LOAD_INSTANCE_FUNC(name) \
    name = (PFN_##name)vkGetInstanceProcAddr(instance, #name)

    // Core instance functions
    LOAD_INSTANCE_FUNC(vkDestroyInstance);
    LOAD_INSTANCE_FUNC(vkEnumeratePhysicalDevices);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceProperties);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceProperties2);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceFeatures);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceFeatures2);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceFormatProperties);
    LOAD_INSTANCE_FUNC(vkEnumerateDeviceExtensionProperties);
    LOAD_INSTANCE_FUNC(vkCreateDevice);
    LOAD_INSTANCE_FUNC(vkGetDeviceProcAddr);

    // Surface functions (VK_KHR_surface)
    LOAD_INSTANCE_FUNC(vkDestroySurfaceKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceSurfaceSupportKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceSurfaceFormatsKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceSurfacePresentModesKHR);

    // Platform-specific surface creation
#ifdef VK_USE_PLATFORM_XCB_KHR
    LOAD_INSTANCE_FUNC(vkCreateXcbSurfaceKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceXcbPresentationSupportKHR);
#endif

#ifdef VK_USE_PLATFORM_XLIB_KHR
    LOAD_INSTANCE_FUNC(vkCreateXlibSurfaceKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceXlibPresentationSupportKHR);
#endif

#ifdef VK_USE_PLATFORM_WIN32_KHR
    LOAD_INSTANCE_FUNC(vkCreateWin32SurfaceKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceWin32PresentationSupportKHR);
#endif

#ifdef VK_USE_PLATFORM_METAL_EXT
    LOAD_INSTANCE_FUNC(vkCreateMetalSurfaceEXT);
#endif

    // Debug utils (optional extension)
    LOAD_INSTANCE_FUNC(vkCreateDebugUtilsMessengerEXT);
    LOAD_INSTANCE_FUNC(vkDestroyDebugUtilsMessengerEXT);
    LOAD_INSTANCE_FUNC(vkSetDebugUtilsObjectNameEXT);

#undef LOAD_INSTANCE_FUNC

    // Verify essential functions loaded
    return vkDestroyInstance && vkEnumeratePhysicalDevices && vkCreateDevice;
}

bool load_device_functions(VkDevice device) {
    if (!device || !vkGetDeviceProcAddr) {
        return false;
    }

#define LOAD_DEVICE_FUNC(name) \
    name = (PFN_##name)vkGetDeviceProcAddr(device, #name)

    // Core device functions
    LOAD_DEVICE_FUNC(vkDestroyDevice);
    LOAD_DEVICE_FUNC(vkGetDeviceQueue);
    LOAD_DEVICE_FUNC(vkDeviceWaitIdle);
    LOAD_DEVICE_FUNC(vkQueueWaitIdle);
    LOAD_DEVICE_FUNC(vkQueueSubmit);
    LOAD_DEVICE_FUNC(vkQueuePresentKHR);

    // Memory
    LOAD_DEVICE_FUNC(vkAllocateMemory);
    LOAD_DEVICE_FUNC(vkFreeMemory);
    LOAD_DEVICE_FUNC(vkMapMemory);
    LOAD_DEVICE_FUNC(vkUnmapMemory);
    LOAD_DEVICE_FUNC(vkFlushMappedMemoryRanges);
    LOAD_DEVICE_FUNC(vkInvalidateMappedMemoryRanges);
    LOAD_DEVICE_FUNC(vkGetBufferMemoryRequirements);
    LOAD_DEVICE_FUNC(vkGetImageMemoryRequirements);
    LOAD_DEVICE_FUNC(vkBindBufferMemory);
    LOAD_DEVICE_FUNC(vkBindImageMemory);

    // Buffers
    LOAD_DEVICE_FUNC(vkCreateBuffer);
    LOAD_DEVICE_FUNC(vkDestroyBuffer);

    // Images
    LOAD_DEVICE_FUNC(vkCreateImage);
    LOAD_DEVICE_FUNC(vkDestroyImage);
    LOAD_DEVICE_FUNC(vkCreateImageView);
    LOAD_DEVICE_FUNC(vkDestroyImageView);

    // Samplers
    LOAD_DEVICE_FUNC(vkCreateSampler);
    LOAD_DEVICE_FUNC(vkDestroySampler);

    // Shaders
    LOAD_DEVICE_FUNC(vkCreateShaderModule);
    LOAD_DEVICE_FUNC(vkDestroyShaderModule);

    // Pipelines
    LOAD_DEVICE_FUNC(vkCreatePipelineLayout);
    LOAD_DEVICE_FUNC(vkDestroyPipelineLayout);
    LOAD_DEVICE_FUNC(vkCreateGraphicsPipelines);
    LOAD_DEVICE_FUNC(vkCreateComputePipelines);
    LOAD_DEVICE_FUNC(vkDestroyPipeline);
    LOAD_DEVICE_FUNC(vkCreatePipelineCache);
    LOAD_DEVICE_FUNC(vkDestroyPipelineCache);
    LOAD_DEVICE_FUNC(vkGetPipelineCacheData);

    // Descriptor sets
    LOAD_DEVICE_FUNC(vkCreateDescriptorSetLayout);
    LOAD_DEVICE_FUNC(vkDestroyDescriptorSetLayout);
    LOAD_DEVICE_FUNC(vkCreateDescriptorPool);
    LOAD_DEVICE_FUNC(vkDestroyDescriptorPool);
    LOAD_DEVICE_FUNC(vkResetDescriptorPool);
    LOAD_DEVICE_FUNC(vkAllocateDescriptorSets);
    LOAD_DEVICE_FUNC(vkFreeDescriptorSets);
    LOAD_DEVICE_FUNC(vkUpdateDescriptorSets);

    // Render passes
    LOAD_DEVICE_FUNC(vkCreateRenderPass);
    LOAD_DEVICE_FUNC(vkDestroyRenderPass);
    LOAD_DEVICE_FUNC(vkCreateFramebuffer);
    LOAD_DEVICE_FUNC(vkDestroyFramebuffer);

    // Command buffers
    LOAD_DEVICE_FUNC(vkCreateCommandPool);
    LOAD_DEVICE_FUNC(vkDestroyCommandPool);
    LOAD_DEVICE_FUNC(vkResetCommandPool);
    LOAD_DEVICE_FUNC(vkAllocateCommandBuffers);
    LOAD_DEVICE_FUNC(vkFreeCommandBuffers);
    LOAD_DEVICE_FUNC(vkBeginCommandBuffer);
    LOAD_DEVICE_FUNC(vkEndCommandBuffer);
    LOAD_DEVICE_FUNC(vkResetCommandBuffer);

    // Command buffer commands
    LOAD_DEVICE_FUNC(vkCmdBindPipeline);
    LOAD_DEVICE_FUNC(vkCmdBindDescriptorSets);
    LOAD_DEVICE_FUNC(vkCmdBindVertexBuffers);
    LOAD_DEVICE_FUNC(vkCmdBindIndexBuffer);
    LOAD_DEVICE_FUNC(vkCmdSetViewport);
    LOAD_DEVICE_FUNC(vkCmdSetScissor);
    LOAD_DEVICE_FUNC(vkCmdSetLineWidth);
    LOAD_DEVICE_FUNC(vkCmdSetDepthBias);
    LOAD_DEVICE_FUNC(vkCmdSetBlendConstants);
    LOAD_DEVICE_FUNC(vkCmdSetStencilReference);
    LOAD_DEVICE_FUNC(vkCmdDraw);
    LOAD_DEVICE_FUNC(vkCmdDrawIndexed);
    LOAD_DEVICE_FUNC(vkCmdDrawIndirect);
    LOAD_DEVICE_FUNC(vkCmdDrawIndexedIndirect);
    LOAD_DEVICE_FUNC(vkCmdDispatch);
    LOAD_DEVICE_FUNC(vkCmdDispatchIndirect);
    LOAD_DEVICE_FUNC(vkCmdCopyBuffer);
    LOAD_DEVICE_FUNC(vkCmdCopyImage);
    LOAD_DEVICE_FUNC(vkCmdCopyBufferToImage);
    LOAD_DEVICE_FUNC(vkCmdCopyImageToBuffer);
    LOAD_DEVICE_FUNC(vkCmdBlitImage);
    LOAD_DEVICE_FUNC(vkCmdPipelineBarrier);
    LOAD_DEVICE_FUNC(vkCmdBeginRenderPass);
    LOAD_DEVICE_FUNC(vkCmdEndRenderPass);
    LOAD_DEVICE_FUNC(vkCmdNextSubpass);
    LOAD_DEVICE_FUNC(vkCmdPushConstants);
    LOAD_DEVICE_FUNC(vkCmdClearColorImage);
    LOAD_DEVICE_FUNC(vkCmdClearDepthStencilImage);
    LOAD_DEVICE_FUNC(vkCmdResolveImage);

    // Synchronization
    LOAD_DEVICE_FUNC(vkCreateFence);
    LOAD_DEVICE_FUNC(vkDestroyFence);
    LOAD_DEVICE_FUNC(vkResetFences);
    LOAD_DEVICE_FUNC(vkGetFenceStatus);
    LOAD_DEVICE_FUNC(vkWaitForFences);
    LOAD_DEVICE_FUNC(vkCreateSemaphore);
    LOAD_DEVICE_FUNC(vkDestroySemaphore);

    // Swapchain (VK_KHR_swapchain)
    LOAD_DEVICE_FUNC(vkCreateSwapchainKHR);
    LOAD_DEVICE_FUNC(vkDestroySwapchainKHR);
    LOAD_DEVICE_FUNC(vkGetSwapchainImagesKHR);
    LOAD_DEVICE_FUNC(vkAcquireNextImageKHR);

#undef LOAD_DEVICE_FUNC

    // Verify essential functions loaded
    return vkDestroyDevice && vkGetDeviceQueue && vkCreateCommandPool;
}

bool is_vulkan_available() {
    if (s_vulkan_library) {
        return true;
    }

    if (load_vulkan_library()) {
        return true;
    }

    return false;
}

void* get_vulkan_library_handle() {
    return s_vulkan_library;
}

} // namespace viz::gal::vk
