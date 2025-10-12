# FindMathematica.cmake
# Cross-platform Mathematica/Wolfram finder
#
# This module finds Mathematica or Wolfram Engine installations across:
# - Windows (native and via WSL)
# - Linux (native)
# - macOS
#
# Sets the following variables:
#   Mathematica_FOUND - True if Mathematica found
#   Mathematica_INSTALL_DIR - Installation directory
#   Mathematica_INCLUDE_DIRS - Include directories
#   Mathematica_LIBRARIES - Libraries for target platform
#   Mathematica_VERSION - Version string (if detectable)
#   Mathematica_KERNEL - Path to kernel executable
#   Mathematica_WOLFRAMSCRIPT - Path to wolframscript

# Allow manual override
if(MATHEMATICA_INSTALL_DIR AND EXISTS "${MATHEMATICA_INSTALL_DIR}")
    set(Mathematica_INSTALL_DIR "${MATHEMATICA_INSTALL_DIR}")
    message(STATUS "Using user-specified Mathematica: ${MATHEMATICA_INSTALL_DIR}")
else()
    # Build comprehensive search path list based on host OS
    set(SEARCH_PATHS "")

    # Detect host OS (where CMake is running)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        # Native Windows
        file(GLOB WIN_MATHEMATICA "C:/Program Files/Wolfram Research/Mathematica/*/")
        file(GLOB WIN_WOLFRAM "C:/Program Files/Wolfram Research/Wolfram/*/")
        file(GLOB WIN_ENGINE "C:/Program Files/Wolfram Research/WolframEngine/*/")
        list(APPEND SEARCH_PATHS ${WIN_MATHEMATICA} ${WIN_WOLFRAM} ${WIN_ENGINE})

    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        # Linux (native or WSL)

        # Standard Linux paths
        file(GLOB LINUX_MATHEMATICA
            "/usr/local/Wolfram/Mathematica/*/"
            "/opt/Wolfram/Mathematica/*/"
            "$ENV{HOME}/.Mathematica/*/"
        )
        file(GLOB LINUX_ENGINE
            "/usr/local/Wolfram/WolframEngine/*/"
            "/opt/Wolfram/WolframEngine/*/"
            "$ENV{HOME}/.WolframEngine/*/"
        )
        list(APPEND SEARCH_PATHS ${LINUX_MATHEMATICA} ${LINUX_ENGINE})

        # Check if running under WSL
        if(EXISTS "/proc/sys/fs/binfmt_misc/WSLInterop" OR DEFINED ENV{WSL_DISTRO_NAME})
            message(STATUS "WSL environment detected - including Windows paths")

            # WSL can access Windows filesystem
            file(GLOB WSL_MATHEMATICA "/mnt/c/Program Files/Wolfram Research/Mathematica/*/")
            file(GLOB WSL_WOLFRAM "/mnt/c/Program Files/Wolfram Research/Wolfram/*/")
            file(GLOB WSL_ENGINE "/mnt/c/Program Files/Wolfram Research/WolframEngine/*/")

            # Also try other drive letters (D:, E:, etc.)
            file(GLOB WSL_MATHEMATICA_D "/mnt/d/Program Files/Wolfram Research/Mathematica/*/")
            file(GLOB WSL_WOLFRAM_D "/mnt/d/Program Files/Wolfram Research/Wolfram/*/")

            list(APPEND SEARCH_PATHS
                ${WSL_MATHEMATICA} ${WSL_WOLFRAM} ${WSL_ENGINE}
                ${WSL_MATHEMATICA_D} ${WSL_WOLFRAM_D}
            )
        endif()

    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        # macOS
        list(APPEND SEARCH_PATHS
            "/Applications/Mathematica.app/Contents"
            "/Applications/Wolfram Engine.app/Contents"
            "$ENV{HOME}/Applications/Mathematica.app/Contents"
            "$ENV{HOME}/Applications/Wolfram Engine.app/Contents"
        )
    endif()

    # Check environment variables
    if(DEFINED ENV{MATHEMATICA_HOME})
        list(APPEND SEARCH_PATHS "$ENV{MATHEMATICA_HOME}")
    endif()
    if(DEFINED ENV{WOLFRAM_HOME})
        list(APPEND SEARCH_PATHS "$ENV{WOLFRAM_HOME}")
    endif()

    # Find the first valid installation by checking for required files
    foreach(PATH ${SEARCH_PATHS})
        if(EXISTS "${PATH}/SystemFiles/IncludeFiles/C")
            set(Mathematica_INSTALL_DIR "${PATH}")
            message(STATUS "Found Mathematica at: ${PATH}")
            break()
        endif()
    endforeach()
endif()

# Verify we found something
if(NOT Mathematica_INSTALL_DIR OR NOT EXISTS "${Mathematica_INSTALL_DIR}")
    set(Mathematica_FOUND FALSE)
    if(NOT Mathematica_FIND_QUIETLY)
        message(STATUS "Mathematica not found. Searched paths:")
        foreach(PATH ${SEARCH_PATHS})
            message(STATUS "  ${PATH}")
        endforeach()
        message(STATUS "Set MATHEMATICA_INSTALL_DIR to specify manually")
        message(STATUS "Or set MATHEMATICA_HOME environment variable")
    endif()
    return()
endif()

# Set include directories (same for all platforms)
set(Mathematica_INCLUDE_DIRS "${Mathematica_INSTALL_DIR}/SystemFiles/IncludeFiles/C")

if(NOT EXISTS "${Mathematica_INCLUDE_DIRS}")
    set(Mathematica_FOUND FALSE)
    if(NOT Mathematica_FIND_QUIETLY)
        message(WARNING "Mathematica found but include directory missing: ${Mathematica_INCLUDE_DIRS}")
    endif()
    return()
endif()

# Determine library based on TARGET platform (not host platform)
# This is crucial for cross-compilation
set(TARGET_PLATFORM "")
set(LIBRARY_SUBDIR "")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Targeting Windows
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM|aarch64")
        set(LIBRARY_SUBDIR "Windows-ARM64")
    else()
        set(LIBRARY_SUBDIR "Windows-x86-64")
    endif()
    set(LIBRARY_NAME "WolframRTL.lib")

elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # Targeting macOS
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(LIBRARY_SUBDIR "MacOSX-ARM64")
    else()
        set(LIBRARY_SUBDIR "MacOSX-x86-64")
    endif()
    set(LIBRARY_NAME "libWolframRTL.dylib")

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Targeting Linux
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(LIBRARY_SUBDIR "Linux-ARM64")
    else()
        set(LIBRARY_SUBDIR "Linux-x86-64")
    endif()
    set(LIBRARY_NAME "libWolframRTL.so")

else()
    set(Mathematica_FOUND FALSE)
    message(WARNING "Unknown target platform: ${CMAKE_SYSTEM_NAME}")
    return()
endif()

# Construct library path
set(Mathematica_LIBRARIES "${Mathematica_INSTALL_DIR}/SystemFiles/Libraries/${LIBRARY_SUBDIR}/${LIBRARY_NAME}")

# Check if library exists (for reference, but not required for LibraryLink)
if(NOT EXISTS "${Mathematica_LIBRARIES}")
    message(STATUS "Note: ${LIBRARY_SUBDIR} WolframRTL library not found")
    message(STATUS "      This is OK - LibraryLink only needs headers at compile time")
    message(STATUS "      Mathematica will load the compiled library at runtime")

    # Clear the library variable since it doesn't exist
    set(Mathematica_LIBRARIES "")
endif()

# Try to detect version from directory name
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" Mathematica_VERSION "${Mathematica_INSTALL_DIR}")

# Find kernel and wolframscript executables
set(KERNEL_NAMES "MathKernel" "WolframKernel" "math" "wolfram")
set(SCRIPT_NAMES "wolframscript" "WolframScript")

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    list(TRANSFORM KERNEL_NAMES APPEND ".exe")
    list(TRANSFORM SCRIPT_NAMES APPEND ".exe")
endif()

# Search for kernel
find_program(Mathematica_KERNEL
    NAMES ${KERNEL_NAMES}
    PATHS "${Mathematica_INSTALL_DIR}"
    PATH_SUFFIXES "MacOS" "Executables" "."
    NO_DEFAULT_PATH
)

# Search for wolframscript
find_program(Mathematica_WOLFRAMSCRIPT
    NAMES ${SCRIPT_NAMES}
    PATHS "${Mathematica_INSTALL_DIR}"
    PATH_SUFFIXES "MacOS" "Executables" "." "bin"
    NO_DEFAULT_PATH
)

# On WSL, also check for Windows executables
if(EXISTS "/proc/sys/fs/binfmt_misc/WSLInterop" OR DEFINED ENV{WSL_DISTRO_NAME})
    if(NOT Mathematica_WOLFRAMSCRIPT)
        find_program(Mathematica_WOLFRAMSCRIPT
            NAMES "wolframscript.exe"
            PATHS "${Mathematica_INSTALL_DIR}"
            PATH_SUFFIXES "."
            NO_DEFAULT_PATH
        )
    endif()
endif()

# Success!
set(Mathematica_FOUND TRUE)

# Report findings
if(NOT Mathematica_FIND_QUIETLY)
    message(STATUS "Mathematica Configuration:")
    message(STATUS "  Installation: ${Mathematica_INSTALL_DIR}")
    message(STATUS "  Version: ${Mathematica_VERSION}")
    message(STATUS "  Include: ${Mathematica_INCLUDE_DIRS}")
    message(STATUS "  Library: ${Mathematica_LIBRARIES}")
    message(STATUS "  Target Platform: ${LIBRARY_SUBDIR}")
    if(Mathematica_KERNEL)
        message(STATUS "  Kernel: ${Mathematica_KERNEL}")
    endif()
    if(Mathematica_WOLFRAMSCRIPT)
        message(STATUS "  WolframScript: ${Mathematica_WOLFRAMSCRIPT}")
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(
    Mathematica_INSTALL_DIR
    Mathematica_INCLUDE_DIRS
    Mathematica_LIBRARIES
    Mathematica_KERNEL
    Mathematica_WOLFRAMSCRIPT
)
