#pragma once

// Graphics Abstraction Layer
// Unified interface over Vulkan (native) and WebGPU (web)

#include <gal/types.hpp>
#include <gal/device.hpp>
#include <gal/command_buffer.hpp>
#include <gal/swapchain.hpp>

namespace viz::gal {

// Initialize the graphics abstraction layer
// Must be called before any other GAL functions
bool initialize(Backend preferred_backend = Backend::Auto);

// Shutdown and cleanup
void shutdown();

// Check if a backend is available
bool is_backend_available(Backend backend);

// Get the active backend
Backend get_active_backend();

} // namespace viz::gal
