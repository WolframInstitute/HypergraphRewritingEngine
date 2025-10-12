# Find Mathematica using our custom finder module
# This handles Windows, Linux, macOS, and WSL automatically
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Allow manual override via cache variable
set(MATHEMATICA_INSTALL_DIR "" CACHE PATH "Path to Mathematica installation directory")

find_package(Mathematica REQUIRED)

if(NOT Mathematica_FOUND)
    message(FATAL_ERROR "Mathematica not found. Set MATHEMATICA_INSTALL_DIR or MATHEMATICA_HOME environment variable.")
    return()
endif()

# Source files - relative to LibraryResources directory
set(SOURCES
    hypergraph_ffi.cpp
)

# Create the library
add_library(HypergraphRewriting SHARED ${SOURCES})

# Link against our existing targets
# Note: Mathematica libraries are NOT needed at link time for LibraryLink
# LibraryLink is header-only - Mathematica loads our library and calls our exports
target_link_libraries(HypergraphRewriting
    hypergraph
    wxf
)

# Include directories
target_include_directories(HypergraphRewriting PRIVATE
    ${Mathematica_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/hypergraph/include
    ${CMAKE_SOURCE_DIR}/job_system/include
    ${CMAKE_SOURCE_DIR}/lockfree_deque/include
)

# Set target properties based on target platform
set_target_properties(HypergraphRewriting PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

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

# For Windows, use the .def file to ensure proper exports
if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(HypergraphRewriting PROPERTIES
        LINK_FLAGS "-Wl,--out-implib,HypergraphRewriting.lib -Wl,--export-all-symbols -Wl,${CMAKE_CURRENT_SOURCE_DIR}/HypergraphRewriting.def"
    )
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
        -Wall -Wextra -O3 -fPIC
    )

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

# Determine platform directory based on CMAKE_SYSTEM_NAME (set by toolchain)
if(NOT DEFINED PLATFORM_DIR)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        # macOS
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
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
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${PLATFORM_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${PLATFORM_DIR}"
)

# Create platform directory if it doesn't exist
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${PLATFORM_DIR}")

# Custom target to create paclet archive
# Requires wolframscript to be available (found by FindMathematica.cmake)
if(Mathematica_WOLFRAMSCRIPT)
    # Get the paclet directory (parent of LibraryResources)
    get_filename_component(PACLET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

    # Create paclet_archive directory at project root
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/paclet_archive")

    add_custom_target(create_paclet_archive
        COMMAND "${Mathematica_WOLFRAMSCRIPT}" -code "CreatePacletArchive[\"${PACLET_DIR}\", \"${CMAKE_SOURCE_DIR}/paclet_archive\"]"
        DEPENDS HypergraphRewriting
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Creating Mathematica paclet archive in paclet_archive/"
        VERBATIM
    )

    message(STATUS "Paclet archive target enabled (wolframscript available)")
else()
    message(STATUS "Paclet archive target disabled (wolframscript not found)")
endif()

# Print paclet-specific configuration
message(STATUS "Paclet Configuration:")
message(STATUS "  Platform: ${PLATFORM_DIR}")
message(STATUS "  Output: ${CMAKE_CURRENT_SOURCE_DIR}/${PLATFORM_DIR}")