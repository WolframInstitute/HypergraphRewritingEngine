cmake_minimum_required(VERSION 3.14)
project(HypergraphVisualization VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(VIZ_VULKAN "Enable Vulkan backend" ON)
option(VIZ_WEBGPU "Enable WebGPU backend" OFF)
option(VIZ_CUDA_LAYOUT "Enable CUDA graph layout (NVIDIA only)" OFF)

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

# Windowing backend detection
#
# The Vulkan GAL and the interactive applications both need the native windowing
# stack (XCB on Linux) - the Vulkan headers pull in <xcb/xcb.h> through
# VK_USE_PLATFORM_XCB_KHR, and the platform layer opens XCB windows. When that
# dependency is absent the windowing translation units cannot compile, but the
# non-windowing library TUs (camera, layout, events, blackhole analysis) still
# build. VIZ_HAS_WINDOWING records whether the windowing backend is available.
set(VIZ_HAS_WINDOWING OFF)

if(VIZ_PLATFORM_LINUX)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(XCB xcb xcb-keysyms xcb-icccm)
    endif()
    if(XCB_FOUND)
        set(VIZ_HAS_WINDOWING ON)
    else()
        message(STATUS
            "XCB development packages not found - skipping the Vulkan GAL, "
            "viz_platform and the interactive viz applications. Non-windowing "
            "library translation units still build. Install libxcb, xcb-keysyms "
            "and xcb-icccm dev packages to enable them.")
    endif()
elseif(VIZ_PLATFORM_WINDOWS)
    set(VIZ_HAS_WINDOWING ON)
endif()

# Graphics Abstraction Layer (GAL) - windowing backend
if(VIZ_HAS_WINDOWING)
add_library(viz_gal STATIC
    gal/src/vulkan/vk_loader.cpp
)

target_include_directories(viz_gal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/gal/include
)

# XCB headers are pulled in by <vulkan/vulkan.h> via VK_USE_PLATFORM_XCB_KHR.
if(VIZ_PLATFORM_LINUX AND XCB_FOUND)
    target_include_directories(viz_gal PUBLIC ${XCB_INCLUDE_DIRS})
endif()

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
    # Use -idirafter (rather than a SYSTEM include) so the path survives: CMake
    # strips SYSTEM include directories that it believes are already implicit
    # compiler search paths, which drops /usr/include for toolchains (e.g. conda)
    # whose implicit set lists it but which do not actually search it at runtime.
    # -idirafter is passed through verbatim and appends after the standard dirs.
    target_compile_options(viz_gal PUBLIC "-idirafter${VULKAN_INCLUDE_DIR}")
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
endif() # VIZ_HAS_WINDOWING (GAL)

# Platform library (windowing/input)
if(VIZ_HAS_WINDOWING)
    add_library(viz_platform STATIC)
    target_include_directories(viz_platform PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/platform/include
    )
    if(VIZ_PLATFORM_LINUX)
        target_sources(viz_platform PRIVATE
            platform/src/linux/window_xcb.cpp
        )
        target_include_directories(viz_platform PRIVATE ${XCB_INCLUDE_DIRS})
        target_link_libraries(viz_platform PRIVATE ${XCB_LIBRARIES})
    elseif(VIZ_PLATFORM_WINDOWS)
        target_sources(viz_platform PRIVATE
            platform/src/windows/window_win32.cpp
        )
    endif()
    target_link_libraries(viz_platform PUBLIC viz_gal)
endif()

# ==============================================================================
# SPIR-V Shader Compilation
# ==============================================================================
# Compile GLSL shaders to SPIR-V using glslc (from Vulkan SDK)

# Find glslc compiler
find_program(GLSLC_EXECUTABLE glslc
    HINTS
        $ENV{VULKAN_SDK}/Bin
        $ENV{VULKAN_SDK}/bin
        /usr/bin
        /usr/local/bin
)

if(GLSLC_EXECUTABLE)
    message(STATUS "Found glslc: ${GLSLC_EXECUTABLE}")

    set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/glsl)
    # Output to build directory, not source tree
    set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders/spirv)

    # List of all GLSL shaders
    set(GLSL_SHADERS
        basic3d.vert
        basic3d.frag
        composite.vert
        composite.frag
        instance_cone.vert
        instance_cylinder.vert
        instance_sphere.vert
        picking_2d.vert
        picking_sphere.vert
        picking_sphere.frag
        text.vert
        text.frag
        triangle.vert
        triangle.frag
        wboit.vert
        wboit.frag
    )

    # Create output directory if it doesn't exist
    file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

    # Generate compile commands for each shader
    set(SPIRV_OUTPUTS)
    foreach(SHADER ${GLSL_SHADERS})
        set(GLSL_SOURCE ${SHADER_SOURCE_DIR}/${SHADER})
        set(SPIRV_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER}.spv)

        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${GLSLC_EXECUTABLE} -o ${SPIRV_OUTPUT} ${GLSL_SOURCE}
            DEPENDS ${GLSL_SOURCE}
            COMMENT "Compiling ${SHADER} -> ${SHADER}.spv"
            VERBATIM
        )

        list(APPEND SPIRV_OUTPUTS ${SPIRV_OUTPUT})
    endforeach()

    # Create a target for all shaders
    add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_OUTPUTS})

else()
    message(STATUS "glslc not found - using precompiled shaders from source tree")

    # Copy precompiled shaders from source tree to build directory
    set(PRECOMPILED_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/spirv)
    set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders/spirv)

    # Create output directory
    file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

    # Copy all .spv files
    file(GLOB PRECOMPILED_SHADERS "${PRECOMPILED_SHADER_DIR}/*.spv")
    foreach(SHADER_FILE ${PRECOMPILED_SHADERS})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
        configure_file(${SHADER_FILE} ${SHADER_OUTPUT_DIR}/${SHADER_NAME} COPYONLY)
    endforeach()

    message(STATUS "Copied precompiled shaders to ${SHADER_OUTPUT_DIR}")
endif()

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

# Events library (viz_events) is created in the top-level CMakeLists.txt
# BEFORE hypergraph, so that hypergraph can link to it for VizEventSink statics.
# We just need to ensure it exists here for local targets to link against.
if(NOT TARGET viz_events)
    # Fallback if building visualisation standalone (not via top-level CMake)
    add_library(viz_events STATIC
        events/src/viz_event_sink.cpp
    )
    target_include_directories(viz_events PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/events/include
    )
endif()

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

# ==============================================================================
# Black hole analysis library (no windowing dependency)
# ==============================================================================
# The renderer is header-only (scene/include/scene/hypergraph_renderer.hpp), so
# the scene/render/ui/geometry object libraries of the earlier modular split do
# not exist. Everything below builds against the header-only renderer plus the
# committed analysis translation units.
if(TARGET hypergraph)
    add_library(viz_blackhole STATIC
        blackhole/src/bh_initial_condition.cpp
        blackhole/src/bh_serialization.cpp
        blackhole/src/hausdorff_analysis.cpp
        blackhole/src/bh_evolution.cpp
        blackhole/src/geodesic_analysis.cpp
        blackhole/src/particle_detection.cpp
        blackhole/src/curvature_analysis.cpp
        blackhole/src/entropy_analysis.cpp
        blackhole/src/branchial_analysis.cpp
        blackhole/src/equilibrium_analysis.cpp
        blackhole/src/branch_alignment.cpp
    )
    target_include_directories(viz_blackhole PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/blackhole/include
    )
    target_compile_definitions(viz_blackhole PUBLIC BLACKHOLE_WITH_LAYOUT=1)
    target_link_libraries(viz_blackhole PUBLIC
        hypergraph
        job_system
        viz_layout
    )

    # Analysis tool for .bhdata files (no windowing needed)
    add_executable(analyze_bhdata
        tools/analyze_bhdata.cpp
    )
    target_link_libraries(analyze_bhdata PRIVATE viz_blackhole)
else()
    message(STATUS "Skipping viz_blackhole (hypergraph target not available)")
endif()

# ==============================================================================
# Interactive applications (require the windowing backend)
# ==============================================================================
if(VIZ_HAS_WINDOWING)
    add_executable(hypergraph_viz
        app/native_main.cpp
    )
    target_link_libraries(hypergraph_viz PRIVATE viz_platform viz_gal viz_camera viz_math viz_layout)
    if(TARGET compile_shaders)
        add_dependencies(hypergraph_viz compile_shaders)
    endif()

    # Hypergraph rendering test application
    add_executable(hypergraph_test
        app/hypergraph_test.cpp
    )
    target_link_libraries(hypergraph_test PRIVATE viz_platform viz_gal viz_camera viz_math viz_layout viz_scene)
    if(TARGET compile_shaders)
        add_dependencies(hypergraph_test compile_shaders)
    endif()

    # Evolution visualisation application - links to hypergraph engine
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
        if(TARGET compile_shaders)
            add_dependencies(evolution_viz compile_shaders)
        endif()
        message(STATUS "Building evolution_viz with hypergraph engine integration")
    endif()

    # Black hole visualisation application
    if(TARGET viz_blackhole)
        add_executable(blackhole_viz
            app/blackhole_viz.cpp
        )
        target_link_libraries(blackhole_viz PRIVATE
            viz_platform
            viz_gal
            viz_camera
            viz_math
            viz_events
            viz_blackhole
        )
        target_compile_definitions(blackhole_viz PRIVATE HYPERGRAPH_ENABLE_VISUALIZATION=1)
        if(TARGET compile_shaders)
            add_dependencies(blackhole_viz compile_shaders)
        endif()
        message(STATUS "Building blackhole_viz with hypergraph engine integration")
    endif()

    # Windows: Static link C++ runtime to avoid DLL dependencies
    if(VIZ_PLATFORM_WINDOWS)
        target_link_options(hypergraph_viz PRIVATE -static-libgcc -static-libstdc++ -static -pthread)
        target_link_options(hypergraph_test PRIVATE -static-libgcc -static-libstdc++ -static -pthread)
        if(TARGET evolution_viz)
            target_link_options(evolution_viz PRIVATE -static-libgcc -static-libstdc++ -static -pthread)
        endif()
        if(TARGET blackhole_viz)
            target_link_options(blackhole_viz PRIVATE -static-libgcc -static-libstdc++ -static -pthread)
        endif()
    endif()
else()
    message(STATUS "Skipping interactive viz applications (no windowing backend)")
endif()
