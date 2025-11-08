cmake_minimum_required(VERSION 3.12)
project(jarvis_fast_capture)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Use pip-installed pybind11
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(pybind11_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
else()
    message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
endif()

# macOS specific settings
if(APPLE)
    # Find required frameworks
    find_library(COREGRAPHICS_LIBRARY CoreGraphics)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    find_library(IMAGEIO_LIBRARY ImageIO)
    find_library(APPSERVICES_LIBRARY ApplicationServices)
    find_library(FOUNDATION_LIBRARY Foundation)
    find_library(APPKIT_LIBRARY AppKit)
    find_library(SCREENCAPTUREKIT_LIBRARY ScreenCaptureKit)
    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    find_library(COREVIDEO_LIBRARY CoreVideo)
    find_library(COREMEDIA_LIBRARY CoreMedia)
    find_library(UNIFORMTYPEIDENTIFIERS_LIBRARY UniformTypeIdentifiers)

    set(PLATFORM_LIBRARIES
        ${COREGRAPHICS_LIBRARY}
        ${COREFOUNDATION_LIBRARY}
        ${IMAGEIO_LIBRARY}
        ${APPSERVICES_LIBRARY}
        ${FOUNDATION_LIBRARY}
        ${APPKIT_LIBRARY}
        ${SCREENCAPTUREKIT_LIBRARY}
        ${METAL_LIBRARY}
        ${METALKIT_LIBRARY}
        ${COREVIDEO_LIBRARY}
        ${COREMEDIA_LIBRARY}
        ${UNIFORMTYPEIDENTIFIERS_LIBRARY}
    )

    # Require macOS 12.3+ for ScreenCaptureKit
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.3" CACHE STRING "Minimum macOS deployment version")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files - use .mm for Objective-C++
set(SOURCES
    src/fast_capture.mm
    src/python_bindings.cpp
)

# Create the Python module
pybind11_add_module(fast_capture ${SOURCES})

# Set source file properties for Objective-C++ (only for .mm files, not during linking)
if(APPLE)
    set_source_files_properties(src/fast_capture.mm PROPERTIES
        COMPILE_FLAGS "-x objective-c++"
    )
endif()

# Set properties
set_target_properties(fast_capture PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Link libraries
target_link_libraries(fast_capture PRIVATE ${PLATFORM_LIBRARIES})

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(APPLE)
        # macOS optimizations
        target_compile_options(fast_capture PRIVATE 
            -O3 
            -march=native 
            -mtune=native
            -ffast-math
            -funroll-loops
            -ftree-vectorize
        )
    endif()
endif()

# Install target
install(TARGETS fast_capture
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../
)