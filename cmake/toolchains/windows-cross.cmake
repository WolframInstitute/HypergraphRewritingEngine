# Windows Cross-Compilation Toolchain
# For cross-compiling to Windows from Linux/macOS
# Supports: MinGW (GCC), LLVM/Clang, MSVC (clang-cl)
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/windows-cross.cmake \
#         -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
#         -DWINDOWS_COMPILER=mingw|clang|msvc \
#         ..
#
# Environment variables:
#   CC, CXX - Override compiler detection
#   TARGET_ARCH - Set target architecture (x86_64, aarch64, etc.)

set(CMAKE_SYSTEM_NAME Windows)

# Target architecture (user-configurable)
if(NOT CMAKE_SYSTEM_PROCESSOR)
    if(DEFINED ENV{TARGET_ARCH})
        set(CMAKE_SYSTEM_PROCESSOR $ENV{TARGET_ARCH})
    else()
        set(CMAKE_SYSTEM_PROCESSOR x86_64)
    endif()
endif()

# Compiler preference (mingw, clang, msvc)
if(NOT WINDOWS_COMPILER)
    set(WINDOWS_COMPILER "auto" CACHE STRING "Windows compiler: auto, mingw, clang, msvc")
endif()

# Helper function to find compilers
function(find_windows_compiler)
    # Check environment variables first
    if(DEFINED ENV{CC} AND DEFINED ENV{CXX})
        set(FOUND_CC $ENV{CC} PARENT_SCOPE)
        set(FOUND_CXX $ENV{CXX} PARENT_SCOPE)
        set(COMPILER_TYPE "env" PARENT_SCOPE)
        return()
    endif()

    # Auto-detect or use specified compiler
    set(ARCH_PREFIX "${CMAKE_SYSTEM_PROCESSOR}-w64-mingw32")

    if(WINDOWS_COMPILER STREQUAL "mingw" OR WINDOWS_COMPILER STREQUAL "auto")
        # Try MinGW-w64 (GCC-based)
        find_program(MINGW_CC ${ARCH_PREFIX}-gcc)
        find_program(MINGW_CXX ${ARCH_PREFIX}-g++)

        if(MINGW_CC AND MINGW_CXX)
            set(FOUND_CC ${MINGW_CC} PARENT_SCOPE)
            set(FOUND_CXX ${MINGW_CXX} PARENT_SCOPE)
            set(COMPILER_TYPE "mingw" PARENT_SCOPE)
            return()
        endif()
    endif()

    if(WINDOWS_COMPILER STREQUAL "clang" OR WINDOWS_COMPILER STREQUAL "auto")
        # Try Clang with MinGW target
        find_program(CLANG_CC clang)
        find_program(CLANG_CXX clang++)

        if(CLANG_CC AND CLANG_CXX)
            set(FOUND_CC ${CLANG_CC} PARENT_SCOPE)
            set(FOUND_CXX ${CLANG_CXX} PARENT_SCOPE)
            set(COMPILER_TYPE "clang" PARENT_SCOPE)
            # Will need --target flag set later
            return()
        endif()
    endif()

    if(WINDOWS_COMPILER STREQUAL "msvc" OR WINDOWS_COMPILER STREQUAL "auto")
        # Try MSVC (clang-cl or cl.exe if available)
        find_program(MSVC_CL clang-cl cl)

        if(MSVC_CL)
            set(FOUND_CC ${MSVC_CL} PARENT_SCOPE)
            set(FOUND_CXX ${MSVC_CL} PARENT_SCOPE)
            set(COMPILER_TYPE "msvc" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Not found
    set(COMPILER_TYPE "notfound" PARENT_SCOPE)
endfunction()

# Find compiler
find_windows_compiler()

if(COMPILER_TYPE STREQUAL "notfound")
    message(FATAL_ERROR
        "No Windows cross-compiler found\n"
        "Install options:\n"
        "  MinGW (Linux/WSL): sudo apt install gcc-mingw-w64-${CMAKE_SYSTEM_PROCESSOR}\n"
        "  MinGW (macOS): brew install mingw-w64\n"
        "  Clang: Use system clang with --target=${CMAKE_SYSTEM_PROCESSOR}-windows-gnu\n"
        "  MSVC: Install Visual Studio Build Tools or clang-cl\n"
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
    set(CLANG_TARGET "${CMAKE_SYSTEM_PROCESSOR}-windows-gnu")
    set(CMAKE_C_COMPILER_TARGET ${CLANG_TARGET})
    set(CMAKE_CXX_COMPILER_TARGET ${CLANG_TARGET})
    set(CMAKE_C_FLAGS_INIT "--target=${CLANG_TARGET}")
    set(CMAKE_CXX_FLAGS_INIT "--target=${CLANG_TARGET}")
    message(STATUS "Using Clang with target: ${CLANG_TARGET}")
elseif(COMPILER_TYPE STREQUAL "msvc")
    # MSVC/clang-cl specific settings
    message(STATUS "Using MSVC/clang-cl")
endif()

# Find archiver and ranlib
if(COMPILER_TYPE STREQUAL "mingw")
    find_program(CMAKE_AR ${CMAKE_SYSTEM_PROCESSOR}-w64-mingw32-ar)
    find_program(CMAKE_RANLIB ${CMAKE_SYSTEM_PROCESSOR}-w64-mingw32-ranlib)
    find_program(CMAKE_RC_COMPILER ${CMAKE_SYSTEM_PROCESSOR}-w64-mingw32-windres)
else()
    find_program(CMAKE_AR ar llvm-ar)
    find_program(CMAKE_RANLIB ranlib llvm-ranlib)
endif()

# Adjust the default behavior of FIND_XXX() commands
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Windows library naming
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")

# Link statically for MinGW/GCC to avoid runtime dependencies
if(COMPILER_TYPE STREQUAL "mingw" OR COMPILER_TYPE STREQUAL "clang")
    set(CMAKE_EXE_LINKER_FLAGS_INIT "-static-libgcc -static-libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS_INIT "-static-libgcc -static-libstdc++")
endif()

message(STATUS "Windows Cross-Compilation Toolchain")
message(STATUS "  Target Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler Type: ${COMPILER_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
