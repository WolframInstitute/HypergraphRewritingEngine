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
        "  MinGW (Linux/WSL): sudo apt install gcc-mingw-w64-${CMAKE_SYSTEM_PROCESSOR} g++-mingw-w64-${CMAKE_SYSTEM_PROCESSOR}\n"
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
    # For ARM64, use MSVC ABI for better compatibility
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        set(CLANG_TARGET "${CMAKE_SYSTEM_PROCESSOR}-pc-windows-msvc")

        # Detect Windows SDK in WSL environment for ARM64 cross-compilation
        if(EXISTS "/mnt/c/Program Files (x86)/Windows Kits")
            # Find the latest Windows SDK version (filter for 10.x.x.x directories only)
            file(GLOB SDK_VERSIONS LIST_DIRECTORIES true "/mnt/c/Program Files (x86)/Windows Kits/10/Lib/10.*")
            list(SORT SDK_VERSIONS)
            list(REVERSE SDK_VERSIONS)
            if(SDK_VERSIONS)
                list(GET SDK_VERSIONS 0 SDK_PATH)
            else()
                set(SDK_PATH "")
            endif()

            if(SDK_PATH AND EXISTS "${SDK_PATH}/um/arm64" AND EXISTS "${SDK_PATH}/ucrt/arm64")
                set(WINSDK_LIB_PATH "${SDK_PATH}")
                message(STATUS "Found Windows SDK for ARM64: ${SDK_PATH}")

                # Find MSVC libraries for ARM64 (search all editions and years)
                if(EXISTS "/mnt/c/Program Files/Microsoft Visual Studio")
                    file(GLOB VS_YEARS LIST_DIRECTORIES true "/mnt/c/Program Files/Microsoft Visual Studio/*")
                    foreach(VS_YEAR ${VS_YEARS})
                        if(IS_DIRECTORY "${VS_YEAR}")
                            # Check all editions: Community, Professional, Enterprise, BuildTools
                            foreach(EDITION Community Professional Enterprise BuildTools)
                                set(MSVC_BASE "${VS_YEAR}/${EDITION}/VC/Tools/MSVC")
                                if(EXISTS "${MSVC_BASE}")
                                    file(GLOB MSVC_VERSIONS LIST_DIRECTORIES true "${MSVC_BASE}/*")
                                    if(MSVC_VERSIONS)
                                        list(SORT MSVC_VERSIONS)
                                        list(REVERSE MSVC_VERSIONS)
                                        list(GET MSVC_VERSIONS 0 MSVC_PATH)

                                        if(EXISTS "${MSVC_PATH}/lib/arm64")
                                            set(MSVC_LIB_PATH "${MSVC_PATH}/lib/arm64")
                                            set(MSVC_INCLUDE_PATH "${MSVC_PATH}/include")
                                            message(STATUS "Found MSVC ARM64 libraries: ${MSVC_LIB_PATH}")
                                            message(STATUS "Found MSVC C++ headers: ${MSVC_INCLUDE_PATH}")
                                            break()
                                        endif()
                                    endif()
                                endif()
                            endforeach()
                            if(DEFINED MSVC_LIB_PATH)
                                break()
                            endif()
                        endif()
                    endforeach()
                endif()

                # Add Windows SDK and MSVC library paths for ARM64 via linker flags
                # lld-link needs /LIBPATH: flags to find libraries
                set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} -Xlinker /LIBPATH:\"${WINSDK_LIB_PATH}/um/arm64\" -Xlinker /LIBPATH:\"${WINSDK_LIB_PATH}/ucrt/arm64\"")
                set(CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT} -Xlinker /LIBPATH:\"${WINSDK_LIB_PATH}/um/arm64\" -Xlinker /LIBPATH:\"${WINSDK_LIB_PATH}/ucrt/arm64\"")

                if(DEFINED MSVC_LIB_PATH)
                    set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} -Xlinker /LIBPATH:\"${MSVC_LIB_PATH}\"")
                    set(CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT} -Xlinker /LIBPATH:\"${MSVC_LIB_PATH}\"")
                else()
                    message(WARNING "MSVC ARM64 libraries not found - linking may fail")
                endif()

                # Add SDK and MSVC include paths
                if(DEFINED MSVC_INCLUDE_PATH)
                    include_directories(SYSTEM "${MSVC_INCLUDE_PATH}")
                endif()

                if(EXISTS "/mnt/c/Program Files (x86)/Windows Kits/10/Include")
                    file(GLOB SDK_INCLUDE_VERSIONS LIST_DIRECTORIES true "/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.*")
                    list(SORT SDK_INCLUDE_VERSIONS)
                    list(REVERSE SDK_INCLUDE_VERSIONS)
                    if(SDK_INCLUDE_VERSIONS)
                        list(GET SDK_INCLUDE_VERSIONS 0 SDK_INCLUDE_PATH)
                        include_directories(SYSTEM
                            "${SDK_INCLUDE_PATH}/ucrt"
                            "${SDK_INCLUDE_PATH}/um"
                            "${SDK_INCLUDE_PATH}/shared"
                        )
                        message(STATUS "Added Windows SDK include paths from: ${SDK_INCLUDE_PATH}")
                    endif()
                endif()
            else()
                message(WARNING "Windows SDK ARM64 libraries not found - build may fail")
            endif()
        endif()
    else()
        set(CLANG_TARGET "${CMAKE_SYSTEM_PROCESSOR}-windows-gnu")
    endif()
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

# Windows doesn't use -fPIC (position independent code)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

# Link statically for MinGW/GCC to avoid runtime dependencies
if(COMPILER_TYPE STREQUAL "mingw")
    set(CMAKE_EXE_LINKER_FLAGS_INIT "-static-libgcc -static-libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS_INIT "-static-libgcc -static-libstdc++")
elseif(COMPILER_TYPE STREQUAL "clang")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        # For ARM64 with MSVC target, use dynamic MSVC runtime (standard for DLLs)
        # No special flags needed - dynamic CRT is the default
        message(STATUS "Using dynamic MSVC runtime (standard for DLLs)")
    else()
        # For x86/x64 with GNU target, use GCC static linking
        set(CMAKE_EXE_LINKER_FLAGS_INIT "-static-libgcc -static-libstdc++")
        set(CMAKE_SHARED_LINKER_FLAGS_INIT "-static-libgcc -static-libstdc++")
    endif()
endif()

message(STATUS "Windows Cross-Compilation Toolchain")
message(STATUS "  Target Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler Type: ${COMPILER_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
