# Allow manual specification of Mathematica installation
set(MATHEMATICA_INSTALL_DIR "" CACHE PATH "Path to Mathematica installation directory")

# Find Mathematica
find_package(Mathematica QUIET)

if(NOT Mathematica_FOUND)
    message(STATUS "Mathematica not found via find_package. Trying manual configuration...")
    
    # Check if user provided path
    if(MATHEMATICA_INSTALL_DIR AND EXISTS "${MATHEMATICA_INSTALL_DIR}")
        message(STATUS "Using user-specified Mathematica path: ${MATHEMATICA_INSTALL_DIR}")
    else()
        # Try to find Mathematica installation automatically
        set(SEARCH_PATHS "")
        
        if(APPLE)
            list(APPEND SEARCH_PATHS "/Applications/Mathematica.app/Contents")
        elseif(WIN32)
            file(GLOB WIN_PATHS "C:/Program Files/Wolfram Research/Mathematica/*/")
            list(APPEND SEARCH_PATHS ${WIN_PATHS})
        else() # Linux or WSL
            # Standard Linux paths
            file(GLOB LINUX_PATHS "/usr/local/Wolfram/Mathematica/*/")
            list(APPEND SEARCH_PATHS ${LINUX_PATHS})
            
            # WSL2 paths - try to access Windows Wolfram installation (newer versions)
            file(GLOB WSL_WOLFRAM_PATHS "/mnt/c/Program Files/Wolfram Research/Wolfram/*/")
            list(APPEND SEARCH_PATHS ${WSL_WOLFRAM_PATHS})
            
            # WSL2 paths - try to access Windows Mathematica installation (older versions)
            file(GLOB WSL_MATH_PATHS "/mnt/c/Program Files/Wolfram Research/Mathematica/*/")
            list(APPEND SEARCH_PATHS ${WSL_MATH_PATHS})
            
            # Additional common Linux paths
            file(GLOB OPT_PATHS "/opt/Wolfram/Mathematica/*/")
            list(APPEND SEARCH_PATHS ${OPT_PATHS})
            
            # Home directory installation
            file(GLOB HOME_PATHS "$ENV{HOME}/Wolfram/Mathematica/*/")
            list(APPEND SEARCH_PATHS ${HOME_PATHS})
        endif()
        
        # Find the first valid installation
        foreach(PATH ${SEARCH_PATHS})
            if(EXISTS "${PATH}/SystemFiles/IncludeFiles/C")
                set(MATHEMATICA_INSTALL_DIR "${PATH}")
                break()
            endif()
        endforeach()
    endif()
    
    if(MATHEMATICA_INSTALL_DIR AND EXISTS "${MATHEMATICA_INSTALL_DIR}")
        set(Mathematica_INCLUDE_DIRS "${MATHEMATICA_INSTALL_DIR}/SystemFiles/IncludeFiles/C")
        
        # Determine target platform libraries
        if(APPLE)
            set(Mathematica_LIBRARIES "${MATHEMATICA_INSTALL_DIR}/SystemFiles/Libraries/MacOSX-x86-64/libWolframRTL.dylib")
        elseif(WIN32 OR CMAKE_CROSSCOMPILING)
            # For Windows or cross-compilation, use Windows libraries
            set(Mathematica_LIBRARIES "${MATHEMATICA_INSTALL_DIR}/SystemFiles/Libraries/Windows-x86-64/WolframRTL.lib")
        else()
            # For Linux, but check if we're in WSL and targeting Windows
            if(DEFINED ENV{WSL_DISTRO_NAME})
                message(STATUS "Detected WSL environment")
                # Check if we have Windows libraries available (WSL accessing Windows Wolfram)
                if(EXISTS "${MATHEMATICA_INSTALL_DIR}/SystemFiles/Libraries/Windows-x86-64/WolframRTL.lib")
                    message(STATUS "Windows Wolfram libraries found - using Windows libraries for cross-compilation")
                    set(Mathematica_LIBRARIES "${MATHEMATICA_INSTALL_DIR}/SystemFiles/Libraries/Windows-x86-64/WolframRTL.lib")
                    set(PLATFORM_DIR "Windows-x86-64")
                else()
                    message(STATUS "Using Linux libraries")
                    set(Mathematica_LIBRARIES "${MATHEMATICA_INSTALL_DIR}/SystemFiles/Libraries/Linux-x86-64/libWolframRTL.so")
                endif()
            else()
                set(Mathematica_LIBRARIES "${MATHEMATICA_INSTALL_DIR}/SystemFiles/Libraries/Linux-x86-64/libWolframRTL.so")
            endif()
        endif()
        
        # Verify the include directory exists
        if(EXISTS "${Mathematica_INCLUDE_DIRS}")
            set(Mathematica_FOUND TRUE)
            message(STATUS "Found Mathematica at: ${MATHEMATICA_INSTALL_DIR}")
            message(STATUS "Include directory: ${Mathematica_INCLUDE_DIRS}")
            message(STATUS "Libraries: ${Mathematica_LIBRARIES}")
        else()
            message(WARNING "Mathematica installation found but include directory missing: ${Mathematica_INCLUDE_DIRS}")
        endif()
    else()
        message(WARNING "Could not find Mathematica installation.")
        message(STATUS "Searched paths:")
        foreach(PATH ${SEARCH_PATHS})
            message(STATUS "  ${PATH}")
        endforeach()
        message(STATUS "To specify manually, use: -DMATHEMATICA_INSTALL_DIR=/path/to/mathematica")
        message(STATUS "For WSL2 with Windows Mathematica, try: -DMATHEMATICA_INSTALL_DIR=\"/mnt/c/Program Files/Wolfram Research/Mathematica/13.3\"")
        return()
    endif()
endif()

# Source files - relative to LibraryResources directory
set(SOURCES
    hypergraph_ffi.cpp
)

# Create the library
add_library(HypergraphRewriting SHARED ${SOURCES})

# Link against our existing targets and Mathematica libraries
target_link_libraries(HypergraphRewriting 
    hypergraph
    ${Mathematica_LIBRARIES}
)

# Include directories
target_include_directories(HypergraphRewriting PRIVATE
    ${Mathematica_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/hypergraph/include
    ${CMAKE_SOURCE_DIR}/job_system/include
    ${CMAKE_SOURCE_DIR}/lockfree_deque/include
)

# Set target properties - Force Windows DLL for WSL cross-compilation
set_target_properties(HypergraphRewriting PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    POSITION_INDEPENDENT_CODE ON
)

# For Windows, use the .def file to ensure proper exports
if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(HypergraphRewriting PROPERTIES
        LINK_FLAGS "-Wl,--out-implib,HypergraphRewriting.lib -Wl,--export-all-symbols -Wl,${CMAKE_CURRENT_SOURCE_DIR}/HypergraphRewriting.def"
    )
endif()

# Platform-specific settings
if(APPLE)
    set_target_properties(HypergraphRewriting PROPERTIES
        SUFFIX ".dylib"
        LINK_FLAGS "-framework Foundation"
    )
elseif(WIN32 OR DEFINED ENV{WSL_DISTRO_NAME})
    # For Windows or WSL targeting Windows
    set_target_properties(HypergraphRewriting PROPERTIES
        SUFFIX ".dll"
    )
    # Define _WIN32 only when actually cross-compiling to Windows
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_compile_definitions(HypergraphRewriting PRIVATE _WIN32)
    endif()
    
    # Auto-detect and warn about MinGW requirement for WSL
    if(DEFINED ENV{WSL_DISTRO_NAME} AND NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
        message(WARNING "WSL detected but not cross-compiling to Windows. For proper Windows DLL:")
        message(WARNING "  Use: CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ cmake .. -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_SYSTEM_PROCESSOR=x86_64")
        message(WARNING "  Install MinGW if needed: sudo apt install gcc-mingw-w64-x86-64")
    endif()
    
    # Set up proper Windows cross-compilation when targeting Windows
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # Use MinGW for Windows cross-compilation if available
        if(UNIX AND NOT APPLE)
            find_program(MINGW_CC x86_64-w64-mingw32-gcc)
            find_program(MINGW_CXX x86_64-w64-mingw32-g++)
            if(MINGW_CC AND MINGW_CXX)
                message(STATUS "Using MinGW cross-compiler for Windows DLL")
                target_link_options(HypergraphRewriting PRIVATE
                    "LINKER:--out-implib,${CMAKE_CURRENT_BINARY_DIR}/HypergraphRewriting.lib"
                )
            else()
                message(FATAL_ERROR "MinGW cross-compiler not found. Install with: sudo apt install gcc-mingw-w64-x86-64")
            endif()
        endif()
    endif()
endif()

# Find and link pthreads on Unix systems
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(HypergraphRewriting Threads::Threads)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(HypergraphRewriting PRIVATE
        -Wall -Wextra -O3 -fPIC
        -pthread
        -fms-extensions  # Enable __declspec for Windows DLL exports
    )
    # For Windows cross-compilation, link statically to avoid runtime dependencies
    if(WIN32 OR DEFINED ENV{WSL_DISTRO_NAME})
        target_link_options(HypergraphRewriting PRIVATE
            -static-libgcc -static-libstdc++
        )
    endif()
endif()

# Determine platform directory (may be overridden above for WSL)
if(NOT DEFINED PLATFORM_DIR)
    if(APPLE)
        set(PLATFORM_DIR "MacOSX-x86-64")
    elseif(WIN32 OR DEFINED ENV{WSL_DISTRO_NAME})
        # Force Windows target for WSL cross-compilation
        set(PLATFORM_DIR "Windows-x86-64")
    else()
        set(PLATFORM_DIR "Linux-x86-64")
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
add_custom_target(paclet_archive
    COMMAND ${CMAKE_COMMAND} -E echo "Creating paclet archive..."
    DEPENDS HypergraphRewriting
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Creating Mathematica paclet archive"
)

# Print configuration information
message(STATUS "Paclet: Mathematica include directories: ${Mathematica_INCLUDE_DIRS}")
message(STATUS "Paclet: Mathematica libraries: ${Mathematica_LIBRARIES}")
message(STATUS "Paclet: Platform directory: ${PLATFORM_DIR}")
message(STATUS "Paclet: Output directory: ${CMAKE_CURRENT_SOURCE_DIR}/${PLATFORM_DIR}")