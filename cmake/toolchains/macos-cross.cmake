# macOS Cross-Compilation Toolchain
# For cross-compiling to macOS from Linux/Windows
# Supports: OSXCross (Clang), native Clang
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-cross.cmake \
#         -DCMAKE_SYSTEM_PROCESSOR=x86_64|arm64 \
#         -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
#         ..
#
# Environment variables:
#   CC, CXX - Override compiler detection
#   OSXCROSS_ROOT - Path to OSXCross installation
#   TARGET_ARCH - Set target architecture (x86_64, arm64)

set(CMAKE_SYSTEM_NAME Darwin)

# Target architecture (user-configurable)
if(NOT CMAKE_SYSTEM_PROCESSOR)
    if(DEFINED ENV{TARGET_ARCH})
        set(CMAKE_SYSTEM_PROCESSOR $ENV{TARGET_ARCH})
    else()
        set(CMAKE_SYSTEM_PROCESSOR x86_64)
    endif()
endif()

# Normalize architecture name for macOS
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_SYSTEM_PROCESSOR "arm64")
endif()

# macOS deployment target (minimum macOS version)
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS version")
endif()

# Helper function to find macOS compiler
function(find_macos_compiler)
    # Check environment variables first
    if(DEFINED ENV{CC} AND DEFINED ENV{CXX})
        set(FOUND_CC $ENV{CC} PARENT_SCOPE)
        set(FOUND_CXX $ENV{CXX} PARENT_SCOPE)
        set(COMPILER_TYPE "env" PARENT_SCOPE)
        return()
    endif()

    # Try OSXCross first (for cross-compilation from Linux/Windows)
    if(DEFINED ENV{OSXCROSS_ROOT})
        set(OSXCROSS_ROOT $ENV{OSXCROSS_ROOT})
    else()
        set(OSXCROSS_ROOT "/opt/osxcross")
    endif()

    # Determine OSXCross target triplet
    # OSXCross uses format: {arch}-apple-darwin{version}
    # We'll try to find any available target
    file(GLOB OSXCROSS_TARGETS "${OSXCROSS_ROOT}/target/bin/${CMAKE_SYSTEM_PROCESSOR}-apple-darwin*-clang")

    if(OSXCROSS_TARGETS)
        list(GET OSXCROSS_TARGETS 0 OSXCROSS_CLANG)
        string(REGEX REPLACE "-clang$" "" OSXCROSS_PREFIX "${OSXCROSS_CLANG}")

        set(FOUND_CC "${OSXCROSS_PREFIX}-clang" PARENT_SCOPE)
        set(FOUND_CXX "${OSXCROSS_PREFIX}-clang++" PARENT_SCOPE)
        set(COMPILER_TYPE "osxcross" PARENT_SCOPE)
        set(OSXCROSS_PREFIX ${OSXCROSS_PREFIX} PARENT_SCOPE)
        return()
    endif()

    # Try native Clang (might be on macOS host or have macOS SDK)
    find_program(CLANG_CC clang)
    find_program(CLANG_CXX clang++)

    if(CLANG_CC AND CLANG_CXX)
        set(FOUND_CC ${CLANG_CC} PARENT_SCOPE)
        set(FOUND_CXX ${CLANG_CXX} PARENT_SCOPE)
        set(COMPILER_TYPE "clang" PARENT_SCOPE)
        return()
    endif()

    # Not found
    set(COMPILER_TYPE "notfound" PARENT_SCOPE)
endfunction()

# Find compiler
find_macos_compiler()

if(COMPILER_TYPE STREQUAL "notfound")
    message(FATAL_ERROR
        "No macOS cross-compiler found\n"
        "Install options:\n"
        "  OSXCross (recommended for Linux/Windows):\n"
        "    1. Clone: git clone https://github.com/tpoechtrager/osxcross\n"
        "    2. Download macOS SDK (see OSXCross README)\n"
        "    3. Build: cd osxcross && ./build.sh\n"
        "    4. Set OSXCROSS_ROOT=/path/to/osxcross\n"
        "  Native macOS: Use Xcode or Command Line Tools\n"
        "\n"
        "Or set CC/CXX environment variables to your cross-compiler"
    )
endif()

# Set compilers
set(CMAKE_C_COMPILER ${FOUND_CC})
set(CMAKE_CXX_COMPILER ${FOUND_CXX})

# Compiler-specific setup
if(COMPILER_TYPE STREQUAL "osxcross")
    # OSXCross provides its own tools
    set(CMAKE_AR "${OSXCROSS_PREFIX}-ar" CACHE FILEPATH "Archiver")
    set(CMAKE_RANLIB "${OSXCROSS_PREFIX}-ranlib" CACHE FILEPATH "Ranlib")
    set(CMAKE_INSTALL_NAME_TOOL "${OSXCROSS_PREFIX}-install_name_tool" CACHE FILEPATH "install_name_tool")

    # Use OSXCross linker
    set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_C_COMPILER> <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

    # OSXCross provides its own sysroot
    string(REGEX REPLACE "target/bin/[^/]*$" "target/SDK/" OSXCROSS_SDK_DIR "${OSXCROSS_PREFIX}")
    file(GLOB OSXCROSS_SDKS "${OSXCROSS_SDK_DIR}MacOSX*.sdk")
    if(OSXCROSS_SDKS)
        list(GET OSXCROSS_SDKS 0 CMAKE_OSX_SYSROOT)
        message(STATUS "Using OSXCross SDK: ${CMAKE_OSX_SYSROOT}")
    endif()

    set(CMAKE_FIND_ROOT_PATH "${OSXCROSS_SDK_DIR}")
elseif(COMPILER_TYPE STREQUAL "clang")
    # Native Clang needs explicit target
    set(CLANG_TARGET "${CMAKE_SYSTEM_PROCESSOR}-apple-darwin")
    set(CMAKE_C_COMPILER_TARGET ${CLANG_TARGET})
    set(CMAKE_CXX_COMPILER_TARGET ${CLANG_TARGET})
    set(CMAKE_C_FLAGS_INIT "--target=${CLANG_TARGET} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    set(CMAKE_CXX_FLAGS_INIT "--target=${CLANG_TARGET} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")

    # Try to find archiver tools
    find_program(CMAKE_AR ar llvm-ar)
    find_program(CMAKE_RANLIB ranlib llvm-ranlib)
endif()

# Adjust the default behavior of FIND_XXX() commands
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# macOS library naming
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")

# Force dylib suffix for shared libraries
set(CMAKE_SHARED_MODULE_SUFFIX ".dylib")

# Set architecture
set(CMAKE_OSX_ARCHITECTURES "${CMAKE_SYSTEM_PROCESSOR}" CACHE STRING "Build architecture for macOS")

message(STATUS "macOS Cross-Compilation Toolchain")
message(STATUS "  Target Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Deployment Target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "  Compiler Type: ${COMPILER_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
