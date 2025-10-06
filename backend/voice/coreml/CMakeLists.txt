cmake_minimum_required(VERSION 3.15)
project(CoreMLVoiceEngine)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Objective-C++ compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -x objective-c++ -fobjc-arc")

# macOS specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment version")
endif()

# Find required frameworks
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREML_FRAMEWORK CoreML REQUIRED)
find_library(AUDIO_TOOLBOX_FRAMEWORK AudioToolbox REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Source files
set(SOURCES
    voice_engine.mm
)

# Header files
set(HEADERS
    voice_engine.hpp
)

# Create shared library
add_library(voice_engine SHARED ${SOURCES} ${HEADERS})

# Link frameworks
target_link_libraries(voice_engine
    ${FOUNDATION_FRAMEWORK}
    ${COREML_FRAMEWORK}
    ${AUDIO_TOOLBOX_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

# Set output name
set_target_properties(voice_engine PROPERTIES
    OUTPUT_NAME "voice_engine"
    PREFIX "lib"
    SUFFIX ".dylib"
)

# Include directories
target_include_directories(voice_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags
target_compile_options(voice_engine PRIVATE
    -Wall
    -Wextra
    -O3
    -ffast-math
)

# Export C symbols for ctypes
set_target_properties(voice_engine PROPERTIES
    C_VISIBILITY_PRESET default
    CXX_VISIBILITY_PRESET default
)

# Installation
install(TARGETS voice_engine
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
    ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)

# Custom target to build and install
add_custom_target(install_lib
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install
    DEPENDS voice_engine
)

# Print configuration
message(STATUS "CoreML Voice Engine Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  macOS Deployment Target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output: libvoice_engine.dylib")
