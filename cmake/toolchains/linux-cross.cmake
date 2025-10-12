# Linux Cross-Compilation Toolchain
# For cross-compiling to Linux from Windows/macOS
# Supports: GCC, Clang, WSL (hybrid)
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-cross.cmake \
#         -DCMAKE_SYSTEM_PROCESSOR=x86_64|aarch64 \
#         -DLINUX_COMPILER=gcc|clang \
#         ..
#
# Environment variables:
#   CC, CXX - Override compiler detection
#   TARGET_ARCH - Set target architecture (x86_64, aarch64, etc.)

set(CMAKE_SYSTEM_NAME Linux)

# Target architecture (user-configurable)
if(NOT CMAKE_SYSTEM_PROCESSOR)
    if(DEFINED ENV{TARGET_ARCH})
        set(CMAKE_SYSTEM_PROCESSOR $ENV{TARGET_ARCH})
    else()
        set(CMAKE_SYSTEM_PROCESSOR x86_64)
    endif()
endif()

# Compiler preference (gcc, clang)
if(NOT LINUX_COMPILER)
    set(LINUX_COMPILER "auto" CACHE STRING "Linux compiler: auto, gcc, clang")
endif()

# Helper function to find Linux compiler
function(find_linux_compiler)
    # Check environment variables first
    if(DEFINED ENV{CC} AND DEFINED ENV{CXX})
        set(FOUND_CC $ENV{CC} PARENT_SCOPE)
        set(FOUND_CXX $ENV{CXX} PARENT_SCOPE)
        set(COMPILER_TYPE "env" PARENT_SCOPE)
        return()
    endif()

    # Check if we're on WSL (can use native Linux tools)
    if(EXISTS "/proc/sys/fs/binfmt_misc/WSLInterop")
        find_program(WSL_GCC gcc)
        find_program(WSL_GXX g++)

        if(WSL_GCC AND WSL_GXX)
            set(FOUND_CC ${WSL_GCC} PARENT_SCOPE)
            set(FOUND_CXX ${WSL_GXX} PARENT_SCOPE)
            set(COMPILER_TYPE "wsl" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Try explicit cross-compiler
    set(ARCH_PREFIX "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")

    if(LINUX_COMPILER STREQUAL "gcc" OR LINUX_COMPILER STREQUAL "auto")
        # Try GCC cross-compiler
        find_program(CROSS_GCC ${ARCH_PREFIX}-gcc)
        find_program(CROSS_GXX ${ARCH_PREFIX}-g++)

        if(CROSS_GCC AND CROSS_GXX)
            set(FOUND_CC ${CROSS_GCC} PARENT_SCOPE)
            set(FOUND_CXX ${CROSS_GXX} PARENT_SCOPE)
            set(COMPILER_TYPE "gcc-cross" PARENT_SCOPE)
            return()
        endif()

        # Try native GCC (might work if on Linux host)
        find_program(NATIVE_GCC gcc)
        find_program(NATIVE_GXX g++)

        if(NATIVE_GCC AND NATIVE_GXX)
            set(FOUND_CC ${NATIVE_GCC} PARENT_SCOPE)
            set(FOUND_CXX ${NATIVE_GXX} PARENT_SCOPE)
            set(COMPILER_TYPE "gcc-native" PARENT_SCOPE)
            return()
        endif()
    endif()

    if(LINUX_COMPILER STREQUAL "clang" OR LINUX_COMPILER STREQUAL "auto")
        # Try Clang
        find_program(CLANG_CC clang)
        find_program(CLANG_CXX clang++)

        if(CLANG_CC AND CLANG_CXX)
            set(FOUND_CC ${CLANG_CC} PARENT_SCOPE)
            set(FOUND_CXX ${CLANG_CXX} PARENT_SCOPE)
            set(COMPILER_TYPE "clang" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Not found
    set(COMPILER_TYPE "notfound" PARENT_SCOPE)
endfunction()

# Find compiler
find_linux_compiler()

if(COMPILER_TYPE STREQUAL "notfound")
    message(FATAL_ERROR
        "No Linux cross-compiler found\n"
        "Options:\n"
        "  On Windows: Use WSL (Windows Subsystem for Linux)\n"
        "    wsl --install\n"
        "  On macOS/Linux: Install cross-compilation toolchain\n"
        "    Debian/Ubuntu: sudo apt install gcc-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu\n"
        "    Or use Docker/VM for true Linux build\n"
        "  Use native Linux for native builds\n"
        "\n"
        "Or set CC/CXX environment variables to your cross-compiler"
    )
endif()

# Set compilers
set(CMAKE_C_COMPILER ${FOUND_CC})
set(CMAKE_CXX_COMPILER ${FOUND_CXX})

# Compiler-specific setup
if(COMPILER_TYPE STREQUAL "clang")
    # Clang needs explicit target
    set(CLANG_TARGET "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
    set(CMAKE_C_COMPILER_TARGET ${CLANG_TARGET})
    set(CMAKE_CXX_COMPILER_TARGET ${CLANG_TARGET})
    set(CMAKE_C_FLAGS_INIT "--target=${CLANG_TARGET}")
    set(CMAKE_CXX_FLAGS_INIT "--target=${CLANG_TARGET}")
    message(STATUS "Using Clang with target: ${CLANG_TARGET}")
endif()

# Find archiver and ranlib
if(COMPILER_TYPE STREQUAL "gcc-cross")
    find_program(CMAKE_AR ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu-ar)
    find_program(CMAKE_RANLIB ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu-ranlib)
else()
    find_program(CMAKE_AR ar llvm-ar)
    find_program(CMAKE_RANLIB ranlib llvm-ranlib)
endif()

# Adjust the default behavior of FIND_XXX() commands
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Linux library naming
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")

message(STATUS "Linux Cross-Compilation Toolchain")
message(STATUS "  Target Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler Type: ${COMPILER_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
