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
    
    # Target to build and install paclet in Mathematica (requires Mathematica to be running)
    add_custom_target(install_paclet_mathematica
        COMMAND ${CMAKE_COMMAND} -E echo "To install the paclet in Mathematica, run:"
        COMMAND ${CMAKE_COMMAND} -E echo "PacletInstall[\"${CMAKE_CURRENT_SOURCE_DIR}\"]"
        DEPENDS HypergraphRewriting
        COMMENT "Instructions for installing paclet in Mathematica"
    )
    
    message(STATUS "Paclet targets added:")
    message(STATUS "  paclet - Build the Mathematica paclet library")
    message(STATUS "  install_paclet_mathematica - Show instructions for installing in Mathematica")
    message(STATUS "  paclet_archive - Create paclet archive")
endif()