cmake_minimum_required(VERSION 3.14)
project(HypergraphVisualization VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(VIZ_VULKAN "Enable Vulkan backend" ON)
option(VIZ_WEBGPU "Enable WebGPU backend" OFF)
option(VIZ_CUDA_LAYOUT "Enable CUDA graph layout (NVIDIA only)" OFF)
option(VIZ_BUILD_FULL "Build all libraries (later phases)" OFF)

# Platform detection
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(VIZ_PLATFORM_LINUX ON)
    add_compile_definitions(VIZ_PLATFORM_LINUX=1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(VIZ_PLATFORM_WINDOWS ON)
    add_compile_definitions(VIZ_PLATFORM_WINDOWS=1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(VIZ_PLATFORM_MACOS ON)
    add_compile_definitions(VIZ_PLATFORM_MACOS=1)
elseif(EMSCRIPTEN)
    set(VIZ_PLATFORM_WEB ON)
    add_compile_definitions(VIZ_PLATFORM_WEB=1)
    set(VIZ_VULKAN OFF)
    set(VIZ_WEBGPU ON)
endif()

# Graphics Abstraction Layer (GAL)
add_library(viz_gal STATIC
    gal/src/vulkan/vk_loader.cpp
)

target_include_directories(viz_gal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/gal/include
)

if(VIZ_VULKAN)
    # Check for Vulkan headers (we dynamically load the library at runtime, but need headers for types)
    # For cross-compilation, prefer VULKAN_SDK or explicit path to avoid picking up host headers
    if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_INCLUDE_DIR $ENV{VULKAN_SDK}/Include)
    else()
        find_path(VULKAN_INCLUDE_DIR vulkan/vulkan.h
            PATHS
                /usr/include
                /usr/local/include
            NO_CMAKE_FIND_ROOT_PATH  # For cross-compilation, still find host headers
        )
    endif()

    if(NOT VULKAN_INCLUDE_DIR OR NOT EXISTS ${VULKAN_INCLUDE_DIR}/vulkan/vulkan.h)
        message(FATAL_ERROR
            "Vulkan headers not found. Please install vulkan development headers:\n"
            "  Ubuntu/Debian: sudo apt install libvulkan-dev\n"
            "  Fedora: sudo dnf install vulkan-headers\n"
            "  Arch: sudo pacman -S vulkan-headers\n"
            "Or set VULKAN_SDK environment variable to Vulkan SDK path.\n"
            "Note: We only need headers - the library is loaded dynamically at runtime.")
    else()
        message(STATUS "Vulkan headers found: ${VULKAN_INCLUDE_DIR}")
    endif()

    target_compile_definitions(viz_gal PUBLIC VIZ_VULKAN=1)

    # For cross-compilation: Vulkan headers are platform-agnostic (just type definitions)
    # but they include <stdint.h> which can pick up the wrong system headers.
    # We use -idirafter to add vulkan headers AFTER compiler's standard includes,
    # ensuring <stdint.h> is found in MinGW sysroot, not /usr/include.
    # This must be PUBLIC so executables linking to viz_gal also get the headers.
    if(CMAKE_CROSSCOMPILING)
        target_compile_options(viz_gal PUBLIC "-idirafter${VULKAN_INCLUDE_DIR}")
    else()
        target_include_directories(viz_gal SYSTEM PUBLIC ${VULKAN_INCLUDE_DIR})
    endif()
    target_sources(viz_gal PRIVATE
        gal/src/vulkan/vk_device.cpp
        gal/src/vulkan/vk_buffer.cpp
        gal/src/vulkan/vk_texture.cpp
        gal/src/vulkan/vk_pipeline.cpp
        gal/src/vulkan/vk_swapchain.cpp
        gal/src/vulkan/vk_command_buffer.cpp
    )

    # Linux: Link dl for dynamic loading
    if(VIZ_PLATFORM_LINUX)
        target_link_libraries(viz_gal PRIVATE dl)
    endif()
endif()

if(VIZ_WEBGPU)
    target_compile_definitions(viz_gal PUBLIC VIZ_WEBGPU=1)
    # WebGPU sources will be added in a later phase
endif()

# Platform library (windowing/input)
add_library(viz_platform STATIC)

target_include_directories(viz_platform PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/platform/include
)

if(VIZ_PLATFORM_LINUX)
    target_sources(viz_platform PRIVATE
        platform/src/linux/window_xcb.cpp
    )
    # Find XCB libraries
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(XCB REQUIRED xcb xcb-keysyms xcb-icccm)
    target_include_directories(viz_platform PRIVATE ${XCB_INCLUDE_DIRS})
    target_link_libraries(viz_platform PRIVATE ${XCB_LIBRARIES})
elseif(VIZ_PLATFORM_WINDOWS)
    target_sources(viz_platform PRIVATE
        platform/src/windows/window_win32.cpp
    )
elseif(VIZ_PLATFORM_WEB)
    # Emscripten sources will be added in a later phase
endif()

target_link_libraries(viz_platform PUBLIC viz_gal)

# Math library (header-only for now)
add_library(viz_math INTERFACE)
target_include_directories(viz_math INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/math/include
)

# Camera library
add_library(viz_camera STATIC
    camera/src/camera.cpp
)
target_include_directories(viz_camera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/camera/include
)
target_link_libraries(viz_camera PUBLIC viz_math)

# Layout library (basic version without full dependencies)
add_library(viz_layout STATIC
    layout/src/layout_manager.cpp
)
target_include_directories(viz_layout PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/layout/include
)

# Scene library (header-only for now - data structures and renderers)
add_library(viz_scene INTERFACE)
target_include_directories(viz_scene INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/include
)
target_link_libraries(viz_scene INTERFACE viz_math)

# Events library (header-only for visualization event streaming)
add_library(viz_events INTERFACE)
target_include_directories(viz_events INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/events/include
)

# CUDA layout backend (optional)
if(VIZ_CUDA_LAYOUT)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_sources(viz_layout PRIVATE
        layout/cuda/barnes_hut_kernels.cu
        layout/cuda/cuda_layout_engine.cu
    )
    target_compile_definitions(viz_layout PUBLIC VIZ_HAS_CUDA=1)
    target_link_libraries(viz_layout PUBLIC CUDA::cudart)

    # CUDA include directories for the kernels
    target_include_directories(viz_layout PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/layout/cuda
    )
endif()

# Phase 2+ executable: 3D camera, basic rendering, and layout
if(VIZ_PLATFORM_LINUX OR VIZ_PLATFORM_WINDOWS)
    add_executable(hypergraph_viz
        app/native_main.cpp
    )
    target_link_libraries(hypergraph_viz PRIVATE viz_platform viz_gal viz_camera viz_math viz_layout)

    # Hypergraph rendering test application
    add_executable(hypergraph_test
        app/hypergraph_test.cpp
    )
    target_link_libraries(hypergraph_test PRIVATE viz_platform viz_gal viz_camera viz_math viz_layout viz_scene)

    # Evolution visualisation application - links to hypergraph engine
    # Only built when hypergraph library is available (built from top-level CMake)
    if(TARGET hypergraph)
        add_executable(evolution_viz
            app/evolution_visualisation.cpp
        )
        target_link_libraries(evolution_viz PRIVATE
            viz_platform
            viz_gal
            viz_camera
            viz_math
            viz_layout
            viz_scene
            viz_events
            hypergraph
            job_system
        )
        target_compile_definitions(evolution_viz PRIVATE HYPERGRAPH_ENABLE_VISUALIZATION=1)
        message(STATUS "Building evolution_viz with hypergraph engine integration")
    else()
        message(STATUS "Skipping evolution_viz (hypergraph target not available)")
    endif()

    # Windows: Static link C++ runtime to avoid DLL dependencies
    if(VIZ_PLATFORM_WINDOWS)
        target_link_options(hypergraph_viz PRIVATE -static-libgcc -static-libstdc++ -static -pthread)
        target_link_options(hypergraph_test PRIVATE -static-libgcc -static-libstdc++ -static -pthread)
        if(TARGET evolution_viz)
            target_link_options(evolution_viz PRIVATE -static-libgcc -static-libstdc++ -static -pthread)
        endif()
    endif()
endif()

# Additional libraries and executables for later phases
if(VIZ_BUILD_FULL)
    # Camera library
    add_library(viz_camera STATIC
        camera/src/camera.cpp
        camera/src/controller.cpp
    )
    target_include_directories(viz_camera PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/camera/include
    )
    target_link_libraries(viz_camera PUBLIC viz_math)

    # Geometry library
    add_library(viz_geometry STATIC
        geometry/src/quickhull.cpp
        geometry/src/bubble_mesh.cpp
        geometry/src/arrow_mesh.cpp
    )
    target_include_directories(viz_geometry PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/geometry/include
    )
    target_link_libraries(viz_geometry PUBLIC viz_math)

    # Layout library
    add_library(viz_layout STATIC
        layout/src/spring_electrical.cpp
        layout/src/barnes_hut.cpp
        layout/src/multiway_layout.cpp
        layout/src/layout_manager.cpp
    )
    target_include_directories(viz_layout PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/layout/include
    )
    target_link_libraries(viz_layout PUBLIC viz_gal viz_math)

    if(VIZ_CUDA_LAYOUT)
        enable_language(CUDA)
        target_sources(viz_layout PRIVATE
            cuda/layout_kernels.cu
        )
        target_compile_definitions(viz_layout PUBLIC VIZ_CUDA_LAYOUT=1)
    endif()

    # Events library
    add_library(viz_events STATIC
        events/src/viz_events.cpp
        events/src/event_ring_buffer.cpp
    )
    target_include_directories(viz_events PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/events/include
    )

    # UI library
    add_library(viz_ui STATIC
        ui/src/ui_context.cpp
        ui/src/widgets.cpp
        ui/src/font_atlas.cpp
    )
    target_include_directories(viz_ui PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/include
    )
    target_link_libraries(viz_ui PUBLIC viz_gal viz_math)

    # Scene library
    add_library(viz_scene STATIC
        scene/src/hypergraph_scene.cpp
        scene/src/evolution_observer.cpp
    )
    target_include_directories(viz_scene PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/scene/include
    )
    target_link_libraries(viz_scene PUBLIC
        viz_events
        viz_layout
        viz_geometry
    )

    # Render library
    add_library(viz_render STATIC
        render/src/renderer.cpp
        render/src/passes/opaque_pass.cpp
        render/src/passes/oit_pass.cpp
        render/src/passes/selection_pass.cpp
        render/src/passes/hyperedge_pass.cpp
        render/src/passes/ui_pass.cpp
        render/src/passes/composite_pass.cpp
    )
    target_include_directories(viz_render PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/render/include
    )
    target_link_libraries(viz_render PUBLIC
        viz_gal
        viz_camera
        viz_geometry
        viz_ui
        viz_scene
    )

    # Main visualization library (combines everything)
    add_library(viz_core INTERFACE)
    target_link_libraries(viz_core INTERFACE
        viz_gal
        viz_platform
        viz_camera
        viz_geometry
        viz_layout
        viz_events
        viz_ui
        viz_scene
        viz_render
    )

    # Link full library to main executable
    target_link_libraries(hypergraph_viz PRIVATE viz_core)

    # Link to main hypergraph library if available
    if(TARGET hypergraph)
        target_link_libraries(hypergraph_viz PRIVATE hypergraph)
        target_compile_definitions(hypergraph_viz PRIVATE HYPERGRAPH_ENABLE_VISUALIZATION=1)
    endif()
endif()

# Web executable
if(VIZ_PLATFORM_WEB AND VIZ_BUILD_FULL)
    add_executable(hypergraph_viz_web
        app/web_main.cpp
    )
    target_link_libraries(hypergraph_viz_web PRIVATE viz_core)

    set_target_properties(hypergraph_viz_web PROPERTIES
        SUFFIX ".html"
        LINK_FLAGS "-s USE_WEBGPU=1 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1"
    )
endif()

# Tests
if(BUILD_TESTS AND VIZ_BUILD_FULL)
    add_executable(test_viz_quickhull tests/test_quickhull.cpp)
    target_link_libraries(test_viz_quickhull PRIVATE viz_geometry)

    add_executable(test_viz_ring_buffer tests/test_ring_buffer.cpp)
    target_link_libraries(test_viz_ring_buffer PRIVATE viz_events)
endif()
