# Find the Wolfram Language using our custom finder module
# This handles Windows, Linux, macOS, and WSL automatically
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Allow manual override via cache variable
set(WOLFRAM_LANGUAGE_INSTALL_DIR "" CACHE PATH "Path to Wolfram Language installation directory")

find_package(WolframLanguage REQUIRED)

if(NOT WolframLanguage_FOUND)
    message(FATAL_ERROR "Wolfram Language not found. Set WOLFRAM_LANGUAGE_INSTALL_DIR or WOLFRAM_LANGUAGE_HOME environment variable.")
    return()
endif()

# Source files - relative to LibraryResources directory
set(SOURCES hypergraph_ffi.cpp paclet_support.cpp)

# Create the library
add_library(HypergraphRewriting SHARED ${SOURCES})

# The build stamp (common/include/hgcommon/build_stamp.hpp). All three shipped artifacts compile
# hypergraph_ffi.cpp, which defines it; each names itself so one scan over a platform directory
# can tell the library from the two executables.
target_compile_definitions(HypergraphRewriting PRIVATE
    HG_BUILD_VARIANT="paclet-library")

# =============================================================================
# Standalone process-isolation binary (WXF over stdin/stdout).
# Reuses the exact marshaling of hypergraph_ffi.cpp via run_rewriting_core, but
# compiled with HG_STANDALONE_BINARY so no Wolfram SDK is pulled in. This is the
# deployment path that supersedes the LibraryLink DLL: abort = process kill,
# crashes are isolated from the notebook. See docs/ARCHITECTURE.md.
add_executable(hg_evolve
    hg_evolve_main.cpp
    hypergraph_ffi.cpp
    paclet_support.cpp
)
target_compile_definitions(hg_evolve PRIVATE HG_STANDALONE_BINARY
    HG_BUILD_VARIANT="hg_evolve")
target_include_directories(hg_evolve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/hypergraph/include
    ${CMAKE_SOURCE_DIR}/job_system/include
    ${CMAKE_SOURCE_DIR}/lockfree_deque/include
)
target_link_libraries(hg_evolve hypergraph wxf)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # The LibraryLink helper functions compile out under HG_STANDALONE_BINARY,
    # leaving some static helpers unused in this translation unit.
    target_compile_options(hg_evolve PRIVATE -O3 -Wno-unused-function)
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(hg_evolve PROPERTIES POSITION_INDEPENDENT_CODE ON)
    find_package(Threads REQUIRED)
    target_link_libraries(hg_evolve Threads::Threads)
    # Fold the C++ runtime into the process binary so it runs on a target whose
    # libstdc++ is older than the build host's. glibc stays dynamic (it is always
    # present, and a fully static glibc breaks NSS/dlopen). macOS ships libc++ in the
    # base system, so there is nothing to bundle there.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(hg_evolve PRIVATE -static-libgcc -static-libstdc++)
    endif()
elseif(CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    # clang -> lld-link (e.g. Windows-ARM64). Two things it can't do the ordinary way:
    #  1. The hypergraph/wxf static libs are GNU-ar archives; lld-link cannot pull objects from them
    #     on demand (their symbol index isn't in the MSVC format), so every symbol reports as
    #     undefined. Force-include them with /WHOLEARCHIVE exactly as the HypergraphRewriting DLL
    #     does above.
    #  2. A target_link_libraries name for winsock becomes a malformed `-lws2_32.lib`; pass the
    #     import lib straight to the linker instead.
    # job_system.lib is under the same GNU-ar constraint: without /WHOLEARCHIVE its members are
    # never pulled and every symbol it owns (park_if_equal, JobSlotPool, CapacityExhausted)
    # reports undefined while the archive sits on the very same link line.
    target_link_options(hg_evolve PRIVATE
        "LINKER:/WHOLEARCHIVE:${CMAKE_BINARY_DIR}/hypergraph.lib"
        "LINKER:/WHOLEARCHIVE:${CMAKE_BINARY_DIR}/wxf.lib"
        "LINKER:/WHOLEARCHIVE:${CMAKE_BINARY_DIR}/job_system.lib"
        "LINKER:ws2_32.lib")  # winsock for --serve-socket
else()
    target_link_libraries(hg_evolve ws2_32)  # winsock for --serve-socket (mingw)
endif()

# GPU-enabled binary: same FFI marshaling, plus the GPU backend. Built only when
# BUILD_GPU produced the hg_gpu target. TargetDevice -> "GPU" in the paclet
# selects this exe; it always runs hg_gpu::evolve and marshals the result
# through the same WXF path as the CPU binary. See docs/ARCHITECTURE.md.
if(TARGET hg_gpu)
    add_executable(hg_evolve_gpu
        hg_evolve_main.cpp
        hypergraph_ffi.cpp
        hg_gpu_backend.cpp
        paclet_support.cpp
    )
    target_compile_definitions(hg_evolve_gpu PRIVATE HG_STANDALONE_BINARY HG_GPU_BACKEND
        HG_BUILD_VARIANT="hg_evolve_gpu")
    target_include_directories(hg_evolve_gpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/hypergraph/include
        ${CMAKE_SOURCE_DIR}/job_system/include
        ${CMAKE_SOURCE_DIR}/lockfree_deque/include
            ${CMAKE_SOURCE_DIR}/gpu/include
        ${CMAKE_SOURCE_DIR}/common/include
    )
    target_link_libraries(hg_evolve_gpu hg_gpu hypergraph wxf)
    # Static CUDA runtime: link libcudart_static so the binary needs no cudart shared library.
    # (The NVIDIA driver, nvcuda.dll / libcuda.so, remains a runtime dependency -- it is the
    # installed GPU driver and cannot be statically linked; it is present wherever a GPU is.)
    set_property(TARGET hg_evolve_gpu PROPERTY CUDA_RUNTIME_LIBRARY Static)
    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::cudart_static)
        target_link_libraries(hg_evolve_gpu CUDA::cudart_static)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(hg_evolve_gpu PRIVATE -O3 -Wno-unused-function)
    endif()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set_target_properties(hg_evolve_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
        find_package(Threads REQUIRED)
        target_link_libraries(hg_evolve_gpu Threads::Threads)
    else()
        target_link_libraries(hg_evolve_gpu ws2_32)  # winsock for --serve-socket
    endif()
endif()

# Link against our existing targets
# Note: Wolfram Language libraries are NOT needed at link time for LibraryLink
# LibraryLink is header-only - the Wolfram Language loads our library and calls our exports
# to avoid requiring BUILD_VISUALIZATION=ON
target_link_libraries(HypergraphRewriting
    hypergraph
    wxf
)

# Include directories
target_include_directories(HypergraphRewriting PRIVATE
    ${WolframLanguage_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/hypergraph/include
    ${CMAKE_SOURCE_DIR}/job_system/include
    ${CMAKE_SOURCE_DIR}/lockfree_deque/include
)

# Progress reporting and abort run over the process-isolation transport: the engine
# executes in a standalone hg_evolve process, and the paclet exchanges WXF with it over a
# NUL-safe socket (--serve-socket), falling back to one-shot RunProcess stdio. So the paclet
# library links no WSTP SDK.

# Set target properties based on target platform
# Windows doesn't use -fPIC; only set for Unix-like systems
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(HypergraphRewriting PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Set library naming based on target system
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(HypergraphRewriting PROPERTIES
        PREFIX ""
        SUFFIX ".dll"
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(HypergraphRewriting PROPERTIES
        PREFIX "lib"
        SUFFIX ".dylib"
    )
else()  # Linux and others
    set_target_properties(HypergraphRewriting PROPERTIES
        PREFIX "lib"
        SUFFIX ".so"
    )
endif()

# For Windows, the .def file is the export list. --export-all-symbols is not added beside it:
# with it, mingw's ld refuses the DLL over the comparator structs local to ir_canonical_hash
# ("cannot export ...: symbol wrong type"), and the LibraryLink entry points the paclet loads
# are the .def's.
if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Check if using MinGW (GNU toolchain) or Clang/MSVC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
        # MinGW uses GNU-style linker flags
        set_target_properties(HypergraphRewriting PROPERTIES
            LINK_FLAGS "-Wl,--out-implib,HypergraphRewriting.lib -Wl,${CMAKE_CURRENT_SOURCE_DIR}/HypergraphRewriting.def"
        )
    else()
        # Clang with MSVC target (e.g., ARM64) uses lld-link with MSVC-style flags
        # Use /WHOLEARCHIVE to force inclusion of all symbols from static libraries
        #
        # job_system IS NAMED HERE and the other two are not enough. hypergraph links it PUBLIC,
        # so on every other configuration its objects arrive transitively; on this one the
        # generated link line carries wxf.lib and hypergraph.lib and nothing else, and the DLL
        # fails with eight undefined symbols that are exactly job_system's four sources --
        # park.cpp (park_if_equal, unpark_all, unpark_one), job_pool.cpp (JobSlotPool::allocate,
        # ::deallocate), affinity.cpp (cache_domains_of, pin_this_thread_to_cpu) and
        # capacity.cpp (CapacityExhausted). The archive is present and correct (coff-arm64, the
        # symbols defined) -- it is simply never handed to the linker, so naming it is the fix.
        set_target_properties(HypergraphRewriting PROPERTIES
            LINK_FLAGS "-Xlinker /DEF:${CMAKE_CURRENT_SOURCE_DIR}/HypergraphRewriting.def -Xlinker /WHOLEARCHIVE:${CMAKE_BINARY_DIR}/wxf.lib -Xlinker /WHOLEARCHIVE:${CMAKE_BINARY_DIR}/hypergraph.lib -Xlinker /WHOLEARCHIVE:${CMAKE_BINARY_DIR}/job_system.lib"
        )
    endif()
endif()

# Platform-specific linker settings
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS-specific flags
    target_link_options(HypergraphRewriting PRIVATE
        "LINKER:-framework,Foundation"
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Windows-specific settings
    target_compile_definitions(HypergraphRewriting PRIVATE _WIN32)

    # MinGW linker options
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_link_options(HypergraphRewriting PRIVATE
            "LINKER:--out-implib,${CMAKE_CURRENT_BINARY_DIR}/HypergraphRewriting.lib"
        )
    endif()
endif()

# Find and link pthreads (needed on Unix-like systems)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    find_package(Threads REQUIRED)
    target_link_libraries(HypergraphRewriting Threads::Threads)
endif()


# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(HypergraphRewriting PRIVATE
        -Wall -Wextra -O3
    )

    # Add -fPIC for non-Windows platforms (Windows doesn't use it)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_compile_options(HypergraphRewriting PRIVATE -fPIC)
    endif()

    # Enable pthread for non-Windows platforms
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_compile_options(HypergraphRewriting PRIVATE -pthread)
    endif()

    # Enable MS extensions for Windows DLL exports
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_compile_options(HypergraphRewriting PRIVATE -fms-extensions)
    endif()
endif()

# Static linking for cross-compiled Windows binaries to avoid runtime dependencies
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(HypergraphRewriting PRIVATE
        -static-libgcc -static-libstdc++
    )
endif()

# The fallback DLL is self-contained on Linux for the same reason as the process
# binary: fold in libstdc++/libgcc so it loads under a kernel with a different
# libstdc++ than the build host. (glibc dynamic; macOS libc++ is in the base system.)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(HypergraphRewriting PRIVATE
        -static-libgcc -static-libstdc++
    )
endif()

# Determine platform directory based on CMAKE_SYSTEM_NAME (set by toolchain)
if(NOT DEFINED PLATFORM_DIR)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        # macOS. A native build picks the slice via CMAKE_OSX_ARCHITECTURES (e.g. an x86_64
        # build on Apple silicon); a cross build keys off the target processor.
        if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
            set(PLATFORM_DIR "MacOSX-ARM64")
        elseif(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
            set(PLATFORM_DIR "MacOSX-x86-64")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(PLATFORM_DIR "MacOSX-ARM64")
        else()
            set(PLATFORM_DIR "MacOSX-x86-64")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # Windows
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM|aarch64")
            set(PLATFORM_DIR "Windows-ARM64")
        else()
            set(PLATFORM_DIR "Windows-x86-64")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Linux
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(PLATFORM_DIR "Linux-ARM64")
        else()
            set(PLATFORM_DIR "Linux-x86-64")
        endif()
    else()
        message(WARNING "Unknown platform: ${CMAKE_SYSTEM_NAME}")
        set(PLATFORM_DIR "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()

# Set output directory to paclet LibraryResources platform directory
set_target_properties(HypergraphRewriting PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/paclet/LibraryResources/${PLATFORM_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/paclet/LibraryResources/${PLATFORM_DIR}"
)

# The standalone binary ships alongside the DLL in the same platform directory.
set_target_properties(hg_evolve PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/paclet/LibraryResources/${PLATFORM_DIR}"
)
if(TARGET hg_evolve_gpu)
    set_target_properties(hg_evolve_gpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/paclet/LibraryResources/${PLATFORM_DIR}"
    )
endif()

# Create platform directory if it doesn't exist
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/paclet/LibraryResources/${PLATFORM_DIR}")

# Custom target to create paclet archive
# Requires wolframscript to be available (found by FindWolframLanguage.cmake)
if(WolframLanguage_WOLFRAMSCRIPT)
    # Paclet distribution directory (contains only distribution files, no build sources)
    set(PACLET_DIST_DIR "${CMAKE_SOURCE_DIR}/paclet")

    # Paclet name (must match PacletInfo.wl) - "/" becomes "__" in archive filename
    set(PACLET_NAME "WolframInstitute/HypergraphRewriteEngine")
    string(REPLACE "/" "__" PACLET_ARCHIVE_NAME "${PACLET_NAME}")

    # Create paclet_archive directory at project root
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/paclet_archive")

    # PacletBuild performs both documentation building and paclet archive creation
    # WSL requires special handling due to Java file locking issues with Lucene indices
    if(IS_WSL)
        # WSL workaround: Copy to Windows filesystem to avoid Java file locking errors
        # PacletBuild uses Java/Lucene for documentation indices which can't lock files on WSL filesystem
        add_custom_target(create_paclet_archive
            # Clean up temp directories
            COMMAND ${CMAKE_COMMAND} -E rm -rf /mnt/c/Temp/paclet_build
            COMMAND ${CMAKE_COMMAND} -E rm -rf /mnt/c/Temp/paclet_output
            # Copy paclet source to Windows temp directory
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${PACLET_DIST_DIR}" /mnt/c/Temp/paclet_build
            # Run PacletBuild on Windows filesystem (builds documentation + creates archive)
            COMMAND "${WolframLanguage_WOLFRAMSCRIPT}" "${CMAKE_CURRENT_SOURCE_DIR}/build_paclet.wls" "C:\\Temp\\paclet_build" "C:\\Temp\\paclet_output"
            # Copy resulting .paclet file to paclet_archive (only .paclet files)
            COMMAND ${CMAKE_COMMAND} -E copy_if_different /mnt/c/Temp/paclet_output/${PACLET_ARCHIVE_NAME}-${PROJECT_VERSION}.paclet "${CMAKE_SOURCE_DIR}/paclet_archive/"
            DEPENDS HypergraphRewriting
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Building paclet archive with documentation (WSL mode)"
            VERBATIM
        )
        message(STATUS "Paclet archive target enabled (WSL mode - will use Windows temp directory)")
    else()
        # Native Windows/Linux/macOS: Run PacletBuild directly
        add_custom_target(create_paclet_archive
            COMMAND "${WolframLanguage_WOLFRAMSCRIPT}" "${CMAKE_CURRENT_SOURCE_DIR}/build_paclet.wls" "${PACLET_DIST_DIR}" "${CMAKE_SOURCE_DIR}/paclet_archive"
            DEPENDS HypergraphRewriting
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Building paclet archive with documentation"
            VERBATIM
        )
        message(STATUS "Paclet archive target enabled (native mode)")
    endif()
else()
    message(STATUS "Paclet archive target disabled (wolframscript not found)")
endif()

# Add convenience target to build just the paclet library
add_custom_target(paclet
    DEPENDS HypergraphRewriting
    COMMENT "Building Wolfram Language paclet library"
)

message(STATUS "Paclet targets:")
message(STATUS "  paclet - Build the Wolfram Language paclet library")
message(STATUS "  create_paclet_archive - Build documentation and create paclet archive (requires wolframscript)")

# Print paclet-specific configuration
message(STATUS "Paclet Build Configuration:")
message(STATUS "  Platform: ${PLATFORM_DIR}")
message(STATUS "  Output: ${CMAKE_SOURCE_DIR}/paclet/LibraryResources/${PLATFORM_DIR}")

# =============================================================================
# gpu_ffi_tests - the SAME serialization gates, answered by the DEVICE
# =============================================================================
# hg_gpu_backend.cpp shipped in hg_evolve_gpu and in no test target, so nothing executed it:
# its reply-shape gating -- which counts a job returns -- was checked by compiling it and by
# its CPU twin, and by nothing that runs. The GPU binary routes EVERY job to run_gpu_job, so
# the same test file compiled with HG_GPU_BACKEND asks the DEVICE the questions all_tests asks
# the host. One test file, two devices, which is the rule the shared graph marshaller already
# follows one level down.
#
# It lives here rather than in testing/ because testing/ is added before gpu/, so TARGET hg_gpu
# does not exist yet there -- the same reason hg_evolve_gpu is declared in this file.
if(TARGET hg_gpu AND BUILD_TESTS)
    add_executable(gpu_ffi_tests
        ${CMAKE_SOURCE_DIR}/testing/main.cpp
        tests/test_wxf_serialization.cpp
        hypergraph_ffi.cpp
        hg_gpu_backend.cpp
        paclet_support.cpp
    )
    target_compile_definitions(gpu_ffi_tests PRIVATE HG_STANDALONE_BINARY HG_GPU_BACKEND
        HG_BUILD_VARIANT="gpu_ffi_tests"
        HG_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(gpu_ffi_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/hypergraph/include
        ${CMAKE_SOURCE_DIR}/job_system/include
        ${CMAKE_SOURCE_DIR}/lockfree_deque/include
        ${CMAKE_SOURCE_DIR}/gpu/include
        ${CMAKE_SOURCE_DIR}/common/include
        ${CMAKE_SOURCE_DIR}/wxf
        ${CMAKE_SOURCE_DIR}/testing
        ${CMAKE_SOURCE_DIR}
    )
    target_link_libraries(gpu_ffi_tests PRIVATE hypergraph hg_gpu wxf GTest::gtest GTest::gtest_main)
endif()
