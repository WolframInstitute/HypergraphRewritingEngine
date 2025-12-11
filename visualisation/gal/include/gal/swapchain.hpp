#pragma once

#include <gal/types.hpp>
#include <gal/device.hpp>

namespace viz::gal {

// Present mode
enum class PresentMode : uint8_t {
    Immediate,     // No vsync, may tear
    Mailbox,       // Triple buffering, no tearing, low latency
    Fifo,          // Vsync, no tearing
    FifoRelaxed,   // Vsync but may tear if late
};

// Swapchain descriptor
struct SwapchainDesc {
    Handle surface = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    Format format = Format::BGRA8_UNORM;  // Typically BGRA or RGBA
    uint32_t image_count = 3;              // Triple buffering
    PresentMode present_mode = PresentMode::Mailbox;
    bool srgb = false;
};

// Result of acquiring an image
struct AcquireResult {
    bool success = false;
    bool suboptimal = false;  // Swapchain should be recreated
    bool out_of_date = false; // Swapchain must be recreated
    uint32_t image_index = 0;
};

// Abstract swapchain interface
class Swapchain {
public:
    virtual ~Swapchain() = default;

    // Get swapchain properties
    virtual uint32_t get_width() const = 0;
    virtual uint32_t get_height() const = 0;
    virtual Format get_format() const = 0;
    virtual uint32_t get_image_count() const = 0;
    virtual PresentMode get_present_mode() const = 0;

    // Get swapchain textures (for creating framebuffers)
    virtual Texture* get_texture(uint32_t index) = 0;
    virtual uint32_t get_current_index() const = 0;

    // Acquire next image for rendering
    // Returns the image index, or -1 if swapchain needs recreation
    virtual AcquireResult acquire_next_image(Semaphore* signal_semaphore = nullptr,
                                              Fence* signal_fence = nullptr,
                                              uint64_t timeout_ns = UINT64_MAX) = 0;

    // Present the current image
    // wait_semaphore: semaphore to wait on before presenting (e.g., render complete)
    virtual bool present(Semaphore* wait_semaphore = nullptr) = 0;

    // Resize swapchain (e.g., after window resize)
    virtual bool resize(uint32_t width, uint32_t height) = 0;

    // Get native handle
    virtual Handle get_native_handle() const = 0;
};

} // namespace viz::gal
