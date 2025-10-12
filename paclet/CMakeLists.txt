# Mathematica Paclet CMakeLists.txt
# This file is included from the main project CMakeLists.txt

# Include the LibraryResources CMakeLists.txt which contains the actual build logic
add_subdirectory(LibraryResources)

# Add convenience targets for paclet development
if(TARGET HypergraphRewriting)
    # Target to build just the paclet library
    add_custom_target(paclet
        DEPENDS HypergraphRewriting
        COMMENT "Building Mathematica paclet library"
    )

    message(STATUS "Paclet targets added:")
    message(STATUS "  paclet - Build the Mathematica paclet library")
    message(STATUS "  create_paclet_archive - Create paclet archive (requires wolframscript)")
endif()