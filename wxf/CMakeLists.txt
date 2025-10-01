cmake_minimum_required(VERSION 3.14)

# WXF (Wolfram Exchange Format) Library
project(WXF VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create WXF static library
add_library(wxf STATIC wxf.cpp)

# Set include directories
target_include_directories(wxf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Require C++17 features
target_compile_features(wxf PUBLIC cxx_std_17)

# Optional: Install targets
install(TARGETS wxf
    EXPORT WXFTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(FILES wxf.hpp DESTINATION include)